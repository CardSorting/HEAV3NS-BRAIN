//! Text and tool conversation serialization for the V4.1 checkpoint protocol.
use crate::model_error as error;
use crate::v41_prompt::{ThinkingMode, BOS, EOS};
use crate::v41_tools::{encode_calls, ToolCall};
use singularity_core::SingularityError;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Invocation {
    pub id: String,
    pub call: ToolCall,
}
#[derive(Debug, Clone)]
pub enum Message {
    System(String),
    User(String),
    Assistant {
        content: String,
        reasoning: String,
        calls: Vec<Invocation>,
        wo_eos: bool,
    },
    Tool {
        call_id: String,
        content: String,
    },
}
#[derive(Clone)]
enum Block {
    Text(String),
    Tool { id: String, content: String },
}
#[derive(Clone)]
enum Normalized {
    System(String),
    User(Vec<Block>),
    Assistant {
        content: String,
        reasoning: String,
        calls: Vec<Invocation>,
        wo_eos: bool,
    },
}

/// Serialize text and tool messages. Tool results are merged into user blocks
/// and ordered to match the preceding assistant's tool-call IDs.
pub fn encode(
    messages: &[Message],
    mode: ThinkingMode,
    effort: u8,
) -> Result<String, SingularityError> {
    encode_inner(messages, mode, effort, false)
}

/// Tool definitions are checkpoint-format function objects (not OpenAI wrappers)
/// and are appended to the first system message. Their presence retains thinking.
pub fn encode_with_tools(
    messages: &[Message],
    mode: ThinkingMode,
    effort: u8,
    tools: &[serde_json::Value],
) -> Result<String, SingularityError> {
    if tools.is_empty() {
        return encode(messages, mode, effort);
    }
    if tools
        .iter()
        .any(|tool| !tool.is_object() || !tool.get("name").is_some_and(|v| v.is_string()))
    {
        return Err(error("Invalid tool schema"));
    }
    let mut messages = messages.to_vec();
    let first = messages
        .first_mut()
        .ok_or_else(|| error("Missing system message for tools"))?;
    let Message::System(content) = first else {
        return Err(error("Tool schemas require a first system message"));
    };
    let schemas = tools
        .iter()
        .map(crate::v41_tools::json_python)
        .collect::<Vec<_>>()
        .join("\n");
    content.push_str(&format!("\n\n{}", render_tools(&schemas)));
    encode_inner(&messages, mode, effort, true)
}

fn render_tools(schemas: &str) -> String {
    format!(
        r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#
    )
}

fn encode_inner(
    messages: &[Message],
    mode: ThinkingMode,
    effort: u8,
    preserve_reasoning: bool,
) -> Result<String, SingularityError> {
    if !(1..=100).contains(&effort) {
        return Err(error("V4.1 reasoning effort must be between 1 and 100"));
    }
    let mut normalized: Vec<Normalized> = Vec::new();
    for message in messages {
        match message {
            Message::System(content) => normalized.push(Normalized::System(content.clone())),
            Message::Assistant {
                content,
                reasoning,
                calls,
                wo_eos,
            } => normalized.push(Normalized::Assistant {
                content: content.clone(),
                reasoning: reasoning.clone(),
                calls: calls.clone(),
                wo_eos: *wo_eos,
            }),
            Message::User(content) => {
                let block = Block::Text(content.clone());
                if let Some(Normalized::User(blocks)) = normalized.last_mut() {
                    blocks.push(block)
                } else {
                    normalized.push(Normalized::User(vec![block]))
                }
            }
            Message::Tool { call_id, content } => {
                let block = Block::Tool {
                    id: call_id.clone(),
                    content: content.clone(),
                };
                if let Some(Normalized::User(blocks)) = normalized.last_mut() {
                    blocks.push(block)
                } else {
                    normalized.push(Normalized::User(vec![block]))
                }
            }
        }
    }
    let mut call_order = HashMap::new();
    for message in &mut normalized {
        match message {
            Normalized::Assistant { calls, .. } if !calls.is_empty() => {
                call_order.clear();
                for (i, c) in calls.iter().enumerate() {
                    if !c.id.is_empty() {
                        call_order.insert(c.id.clone(), i);
                    }
                }
            }
            Normalized::User(blocks) if !call_order.is_empty() => {
                let mut results: Vec<_> = blocks
                    .iter()
                    .filter(|b| matches!(b, Block::Tool { .. }))
                    .cloned()
                    .collect();
                results.sort_by_key(|b| {
                    if let Block::Tool { id, .. } = b {
                        *call_order.get(id).unwrap_or(&0)
                    } else {
                        0
                    }
                });
                let mut sorted = results.into_iter();
                for block in blocks {
                    if matches!(block, Block::Tool { .. }) {
                        *block = sorted.next().unwrap();
                    }
                }
            }
            _ => {}
        }
    }
    let last_user = normalized
        .iter()
        .enumerate()
        .rfind(|(i, m)| match m {
            Normalized::User(blocks) => blocks.iter().any(|b| matches!(b, Block::Text(_))),
            Normalized::System(_) => *i > 0,
            _ => false,
        })
        .map(|(i, _)| i);
    let mut output = BOS.to_owned();
    for (i, message) in normalized.iter().enumerate() {
        if i == 0
            && (matches!(mode, ThinkingMode::Thinking) || matches!(message, Normalized::System(_)))
        {
            output.push_str("<｜System｜>");
        }
        if i == 0 && matches!(mode, ThinkingMode::Thinking) {
            output.push_str(&format!("Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"));
        }
        match message {
            Normalized::System(content) => {
                if i > 0 {
                    output.push_str("<｜System｜>");
                }
                output.push_str(content);
            }
            Normalized::User(blocks) => {
                output.push_str("<｜User｜>");
                let parts: Vec<String> = blocks
                    .iter()
                    .map(|b| match b {
                        Block::Text(text) => text.clone(),
                        Block::Tool { content, .. } => {
                            format!("<tool_result>{content}</tool_result>")
                        }
                    })
                    .collect();
                output.push_str(&parts.join("\n\n"));
            }
            Normalized::Assistant {
                content,
                reasoning,
                calls,
                wo_eos,
            } => {
                if matches!(mode, ThinkingMode::Thinking)
                    && (preserve_reasoning || last_user.is_none_or(|idx| i > idx))
                {
                    output.push_str(reasoning);
                    output.push_str("</think>");
                }
                output.push_str(content);
                output.push_str(&encode_calls(
                    &calls.iter().map(|c| c.call.clone()).collect::<Vec<_>>(),
                )?);
                if !wo_eos {
                    output.push_str(EOS);
                }
            }
        }
        let next = normalized
            .get(i + 1)
            .is_none_or(|m| matches!(m, Normalized::Assistant { .. }));
        if next
            && (matches!(message, Normalized::User(_))
                || (i > 0 && matches!(message, Normalized::System(_))))
        {
            output.push_str("<｜Assistant｜>");
            if matches!(mode, ThinkingMode::Thinking)
                && (preserve_reasoning || last_user.is_none_or(|idx| i >= idx))
            {
                output.push_str("<think>");
            } else {
                output.push_str("</think>");
            }
        }
    }
    if output.contains("<｜deepseek_image｜>") {
        return Err(error("Image special tokens are unsupported"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::Value;
    #[derive(Deserialize)]
    struct Vector {
        name: String,
        mode: ThinkingMode,
        messages: Vec<Value>,
        prompt: String,
    }
    fn convert(message: &Value) -> Message {
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        match message["role"].as_str().unwrap() {
            "system" => Message::System(content),
            "user" => Message::User(content),
            "tool" => Message::Tool {
                call_id: message["tool_call_id"].as_str().unwrap().into(),
                content,
            },
            "assistant" => Message::Assistant {
                content,
                reasoning: message
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .into(),
                wo_eos: false,
                calls: message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .map(|call| Invocation {
                                id: call["id"].as_str().unwrap().into(),
                                call: ToolCall {
                                    name: call["function"]["name"].as_str().unwrap().into(),
                                    namespace: None,
                                    arguments: serde_json::from_str(
                                        call["function"]["arguments"].as_str().unwrap(),
                                    )
                                    .unwrap(),
                                },
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            },
            _ => panic!("unsupported test role"),
        }
    }
    #[test]
    fn tool_conversations_match_checkpoint_encoder() {
        let vectors: Vec<Vector> = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_conversation_vectors.json"
        ))
        .unwrap();
        for v in vectors {
            let messages: Vec<_> = v.messages.iter().map(convert).collect();
            assert_eq!(
                encode(&messages, v.mode, 75).unwrap(),
                v.prompt,
                "{}",
                v.name
            );
        }
    }

    #[test]
    fn schema_and_reasoning_retention_match_checkpoint() {
        let v: Value =
            serde_json::from_str(include_str!("../tests/fixtures/v41_schema_vectors.json"))
                .unwrap();
        let messages = vec![
            Message::System("Use tools".into()),
            Message::User("first".into()),
            Message::Assistant {
                content: "answer".into(),
                reasoning: "remember me".into(),
                calls: vec![],
                wo_eos: false,
            },
            Message::User("again".into()),
        ];
        assert_eq!(
            encode_with_tools(
                &messages,
                ThinkingMode::Thinking,
                75,
                &[v["function"].clone()]
            )
            .unwrap(),
            v["prompt"].as_str().unwrap()
        );
        assert!(encode_with_tools(
            &messages[1..],
            ThinkingMode::Thinking,
            75,
            &[v["function"].clone()]
        )
        .is_err());
    }
}
