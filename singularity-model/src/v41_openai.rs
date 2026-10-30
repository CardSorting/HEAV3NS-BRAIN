//! OpenAI-shaped text/tool request adaptation for the V4.1 protocol.
//! This module formats data only; it does not execute tools or serve requests.
use crate::{
    v41_conversation::{self, Invocation, Message},
    v41_prompt::ThinkingMode,
    v41_tools::{self, ToolCall},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use singularity_core::SingularityError;
fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MessageInput {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<CallInput>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallInput {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
    #[serde(default)]
    pub namespace: Option<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub namespace: Option<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: Map<String, Value>,
    #[serde(default)]
    pub namespace: Option<Value>,
}

fn arguments(text: &str) -> Map<String, Value> {
    let mut value = Value::String(text.to_owned());
    for _ in 0..2 {
        if let Value::String(s) = &value {
            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                value = parsed;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    match value {
        Value::Object(m) => m,
        _ => {
            let mut m = Map::new();
            m.insert("arguments".into(), Value::String(text.to_owned()));
            m
        }
    }
}
fn invocations(calls: &[CallInput]) -> Result<Vec<Invocation>, SingularityError> {
    calls
        .iter()
        .map(|c| {
            if c.kind != "function" || c.id.is_empty() {
                return Err(error("Invalid function call"));
            }
            let namespace = c
                .namespace
                .as_ref()
                .or(c.function.namespace.as_ref())
                .cloned();
            Ok(Invocation {
                id: c.id.clone(),
                call: ToolCall {
                    name: c.function.name.clone(),
                    namespace,
                    arguments: arguments(&c.function.arguments),
                },
            })
        })
        .collect()
}
fn normalize_tool(tool: &ToolDefinition) -> Result<Value, SingularityError> {
    if tool.kind != "function" {
        return Err(error("Only function tools are supported"));
    }
    let mut function = tool.function.clone();
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| error("Tool definition missing name"))?
        .to_owned();
    let namespace = tool
        .namespace
        .as_ref()
        .or_else(|| function.get("namespace"));
    let (namespace, description) = match namespace {
        None => (None, None),
        Some(Value::String(s)) => (Some(s.as_str()), None),
        Some(Value::Object(m)) => (
            Some(
                m.get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| error("Invalid tool namespace"))?,
            ),
            m.get("description").and_then(Value::as_str),
        ),
        _ => return Err(error("Invalid tool namespace")),
    };
    let qualified = if let Some(ns) = namespace {
        if ns.is_empty() || ns.contains("::") {
            return Err(error("Invalid tool namespace"));
        }
        if let Some((existing, _)) = name.split_once("::") {
            if existing != ns {
                return Err(error("Conflicting tool namespaces"));
            }
            name.clone()
        } else {
            format!("{ns}::{name}")
        }
    } else {
        name.clone()
    };
    if qualified.split("::").count() > 2 || qualified.split("::").any(str::is_empty) {
        return Err(error("Invalid tool name"));
    }
    let description = description.map(str::to_owned);
    function.remove("namespace");
    function.insert("name".into(), Value::String(qualified));
    if let Some(prefix) = description {
        let original = function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        function.insert(
            "description".into(),
            Value::String(format!("{prefix}\n{original}")),
        );
    }
    Ok(Value::Object(function))
}
/// Convert OpenAI-shaped messages and function definitions to a checkpoint prompt.
/// Unsupported structured content is rejected during deserialization.
pub fn encode(
    messages: &[MessageInput],
    tools: &[ToolDefinition],
    mode: ThinkingMode,
    effort: u8,
) -> Result<String, SingularityError> {
    let mut converted = Vec::with_capacity(messages.len());
    for msg in messages {
        let content = msg.content.clone().unwrap_or_default();
        match msg.role.as_str() {
            "system"
                if msg.tool_calls.is_empty()
                    && msg.tool_call_id.is_none()
                    && msg.reasoning_content.is_none() =>
            {
                converted.push(Message::System(content))
            }
            "user"
                if msg.tool_calls.is_empty()
                    && msg.tool_call_id.is_none()
                    && msg.reasoning_content.is_none() =>
            {
                converted.push(Message::User(content))
            }
            "assistant" if msg.tool_call_id.is_none() => converted.push(Message::Assistant {
                content,
                reasoning: msg.reasoning_content.clone().unwrap_or_default(),
                calls: invocations(&msg.tool_calls)?,
                wo_eos: false,
            }),
            "tool" if msg.tool_calls.is_empty() && msg.reasoning_content.is_none() => converted
                .push(Message::Tool {
                    call_id: msg
                        .tool_call_id
                        .clone()
                        .filter(|id| !id.is_empty())
                        .ok_or_else(|| error("Tool result missing call ID"))?,
                    content,
                }),
            _ => return Err(error("Unsupported role or message fields")),
        }
    }
    let schemas = tools
        .iter()
        .map(normalize_tool)
        .collect::<Result<Vec<_>, _>>()?;
    if !schemas.is_empty() && !matches!(converted.first(), Some(Message::System(_))) {
        converted.insert(0, Message::System(String::new()));
    }
    v41_conversation::encode_with_tools(&converted, mode, effort, &schemas)
}
#[derive(Debug, Serialize)]
pub struct OutputFunction {
    pub name: String,
    pub arguments: String,
}
#[derive(Debug, Serialize)]
pub struct OutputCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: OutputFunction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct AssistantOutput {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub reasoning_content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OutputCall>,
}
/// Decode a complete generation; the caller assigns stable, unique call IDs.
pub fn decode(
    text: &str,
    mode: ThinkingMode,
    ids: &[String],
) -> Result<AssistantOutput, SingularityError> {
    let completion = v41_tools::parse_completion(text, mode)?;
    if ids.len() != completion.calls.len() || ids.iter().any(String::is_empty) {
        return Err(error("Tool call IDs do not match completion"));
    }
    let mut seen = std::collections::HashSet::new();
    if ids.iter().any(|id| !seen.insert(id)) {
        return Err(error("Duplicate tool call ID"));
    }
    let tool_calls = completion
        .calls
        .into_iter()
        .zip(ids)
        .map(|(call, id)| OutputCall {
            id: id.clone(),
            kind: "function",
            function: OutputFunction {
                name: call.name,
                arguments: serde_json::to_string(&call.arguments).unwrap(),
            },
            namespace: call.namespace,
        })
        .collect();
    Ok(AssistantOutput {
        role: "assistant",
        content: completion.content,
        reasoning_content: completion.reasoning,
        tool_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Deserialize)]
    struct Vector {
        name: String,
        mode: ThinkingMode,
        messages: Vec<MessageInput>,
        prompt: String,
    }
    #[test]
    fn openai_messages_match_checkpoint_vectors() {
        let vectors: Vec<Vector> = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_conversation_vectors.json"
        ))
        .unwrap();
        for v in vectors {
            assert_eq!(
                encode(&v.messages, &[], v.mode, 75).unwrap(),
                v.prompt,
                "{}",
                v.name
            );
        }
    }
    #[test]
    fn schema_adapter_and_completion_roundtrip() {
        let v: Value =
            serde_json::from_str(include_str!("../tests/fixtures/v41_schema_vectors.json"))
                .unwrap();
        let messages:Vec<MessageInput>=serde_json::from_value(serde_json::json!([{"role":"system","content":"Use tools"},{"role":"user","content":"first"},{"role":"assistant","content":"answer","reasoning_content":"remember me"},{"role":"user","content":"again"}])).unwrap();
        let tools: Vec<ToolDefinition> = serde_json::from_value(
            serde_json::json!([{"type":"function","function":v["function"]}]),
        )
        .unwrap();
        assert_eq!(
            encode(&messages, &tools, ThinkingMode::Thinking, 75).unwrap(),
            v["prompt"].as_str().unwrap()
        );
        let without_system = encode(&messages[1..], &tools, ThinkingMode::Thinking, 75).unwrap();
        assert!(without_system.contains("## Tools"));
        assert!(without_system.contains("<｜User｜>first"));
        let vectors: Value =
            serde_json::from_str(include_str!("../tests/fixtures/v41_tool_vectors.json")).unwrap();
        let text = vectors[1]["completion"].as_str().unwrap();
        assert!(decode(text, ThinkingMode::Thinking, &["only-one".into()]).is_err());
        let output = decode(text, ThinkingMode::Thinking, &["a".into(), "b".into()]).unwrap();
        assert_eq!(output.tool_calls[0].namespace.as_deref(), Some("math"));
        assert_eq!(output.tool_calls[0].function.name, "sum");
        assert_eq!(
            serde_json::from_str::<Value>(&output.tool_calls[0].function.arguments).unwrap(),
            serde_json::json!({"items":[1,2],"strict":true})
        );
        assert!(decode(text, ThinkingMode::Thinking, &["a".into(), "a".into()]).is_err());
    }
    #[test]
    fn unsupported_message_fields_are_rejected() {
        assert!(serde_json::from_value::<MessageInput>(
            serde_json::json!({"role":"user","content":[{"type":"image_url"}]})
        )
        .is_err());
        assert!(serde_json::from_value::<MessageInput>(
            serde_json::json!({"role":"user","content":"hello","unexpected":true})
        )
        .is_err());
    }
}
