//! DeepSeek V4.1 DSML tool call serialization and completion parsing.
//! Based on checkpoint encoding/encoding.py, MIT, pinned revision 1bae0e1088993b9b82de08f59de5be5a34e42170.
use crate::v41_prompt::{ThinkingMode, BOS, EOS};
use serde_json::{Map, Value};
use singularity_core::SingularityError;
const PREFIX: &str = "｜DSML｜";
const CALLS: &str = "\n\n<｜DSML｜ calls>\n";
const CALLS_END: &str = "</｜DSML｜ calls>";
const INVOKE: &str = "<｜DSML｜ invoke name=\"";
const INVOKE_END: &str = "</｜DSML｜ invoke>";
const PARAM: &str = "<｜DSML｜ parameter name=\"";
const PARAM_END: &str = "</｜DSML｜ parameter>";
use crate::model_error as error;

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub namespace: Option<String>,
    pub arguments: Map<String, Value>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub content: String,
    pub reasoning: String,
    pub calls: Vec<ToolCall>,
}

fn split_name(
    name: &str,
    explicit: Option<&str>,
) -> Result<(Option<String>, String), SingularityError> {
    let (namespace, name) = match name.split_once("::") {
        Some((namespace, name)) => {
            if explicit.is_some_and(|v| v != namespace) {
                return Err(error("Conflicting tool namespaces"));
            }
            (Some(namespace), name)
        }
        None => (explicit, name),
    };
    if name.is_empty()
        || name.contains("::")
        || namespace.is_some_and(|n| n.is_empty() || n.contains("::"))
        || name.contains(['\"', '\n', '<', '>'])
        || namespace.is_some_and(|n| n.contains(['\"', '\n', '<', '>']))
    {
        return Err(error("Invalid tool name"));
    }
    Ok((namespace.map(str::to_owned), name.to_owned()))
}
pub(crate) fn json_python(value: &Value) -> String {
    match value {
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(json_python).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), json_python(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => serde_json::to_string(value).unwrap(),
    }
}
/// Encode calls following assistant text, without the EOS token.
pub fn encode_calls(calls: &[ToolCall]) -> Result<String, SingularityError> {
    if calls.is_empty() {
        return Ok(String::new());
    }
    let mut result = CALLS.to_owned();
    for (i, call) in calls.iter().enumerate() {
        let (namespace, name) = split_name(&call.name, call.namespace.as_deref())?;
        let qualified = namespace.map_or(name.clone(), |ns| format!("{ns}::{name}"));
        if i > 0 {
            result.push('\n');
        }
        result.push_str(&format!("{INVOKE}{qualified}\">\n"));
        for (j, (key, value)) in call.arguments.iter().enumerate() {
            if key.contains(['\"', '\n', '<', '>']) {
                return Err(error("Invalid tool argument name"));
            }
            let (kind, text) = match value {
                Value::String(s) => ("true", s.clone()),
                _ => ("false", json_python(value)),
            };
            if text.contains(PARAM_END) {
                return Err(error("Tool argument contains DSML closing delimiter"));
            }
            if j > 0 {
                result.push('\n');
            }
            result.push_str(&format!(
                "{PARAM}{key}\" string=\"{kind}\">{text}{PARAM_END}"
            ));
        }
        result.push_str(&format!("\n{INVOKE_END}"));
    }
    result.push_str(&format!("\n{CALLS_END}"));
    Ok(result)
}

fn take_until<'a>(input: &mut &'a str, delimiter: &str) -> Result<&'a str, SingularityError> {
    let (part, rest) = input
        .split_once(delimiter)
        .ok_or_else(|| error("Incomplete DSML completion"))?;
    *input = rest;
    Ok(part)
}
fn consume(input: &mut &str, prefix: &str) -> Result<(), SingularityError> {
    *input = input
        .strip_prefix(prefix)
        .ok_or_else(|| error("Malformed DSML completion"))?;
    Ok(())
}
/// Parse a complete assistant continuation, including EOS. The generation header
/// already supplied the opening thinking token; pass only newly generated text.
pub fn parse_completion(text: &str, mode: ThinkingMode) -> Result<Completion, SingularityError> {
    let mut remaining = text;
    let reasoning = if matches!(mode, ThinkingMode::Thinking) {
        take_until(&mut remaining, "</think>")?.to_owned()
    } else {
        String::new()
    };
    let eos = remaining
        .find(EOS)
        .ok_or_else(|| error("Missing completion EOS"))?;
    let call_start = remaining.find(CALLS);
    let content_end = call_start.unwrap_or(eos);
    if content_end > eos {
        return Err(error("Unexpected text after EOS"));
    }
    let content = remaining[..content_end].to_owned();
    remaining = &remaining[content_end..];
    let mut calls = Vec::new();
    if call_start.is_some_and(|p| p < eos) {
        consume(&mut remaining, CALLS)?;
        loop {
            consume(&mut remaining, INVOKE)?;
            let qualified = take_until(&mut remaining, "\">\n")?;
            let (namespace, name) = split_name(qualified, None)?;
            let mut arguments = Map::new();
            while remaining.starts_with(PARAM) {
                consume(&mut remaining, PARAM)?;
                let key = take_until(&mut remaining, "\" string=\"")?;
                if key.contains(['\"', '\n', '<', '>']) {
                    return Err(error("Invalid tool argument name"));
                }
                let kind = take_until(&mut remaining, "\">")?;
                let body = take_until(&mut remaining, PARAM_END)?;
                let value = match kind {
                    "true" => Value::String(body.to_owned()),
                    "false" => serde_json::from_str(body)
                        .map_err(|_| error("Invalid JSON tool argument"))?,
                    _ => return Err(error("Invalid tool argument type")),
                };
                if arguments.insert(key.to_owned(), value).is_some() {
                    return Err(error("Duplicate tool argument"));
                }
                consume(&mut remaining, "\n")?;
            }
            // An empty argument list emits an empty line before the closing tag.
            if arguments.is_empty() {
                consume(&mut remaining, "\n")?;
            }
            consume(&mut remaining, INVOKE_END)?;
            calls.push(ToolCall {
                name,
                namespace,
                arguments,
            });
            consume(&mut remaining, "\n")?;
            if remaining.starts_with(CALLS_END) {
                consume(&mut remaining, CALLS_END)?;
                break;
            }
        }
    }
    consume(&mut remaining, EOS)?;
    if !remaining.is_empty()
        || [BOS, EOS, "<think>", "</think>", PREFIX]
            .iter()
            .any(|token| content.contains(token) || reasoning.contains(token))
    {
        return Err(error("Unexpected completion content"));
    }
    Ok(Completion {
        content,
        reasoning,
        calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct InputCall {
        name: String,
        arguments: Map<String, Value>,
    }
    #[derive(Deserialize)]
    struct Vector {
        name: String,
        mode: ThinkingMode,
        reasoning: String,
        content: String,
        calls: Vec<InputCall>,
        encoded_calls: String,
        completion: String,
    }
    #[test]
    fn checkpoint_vectors_encode_and_parse() {
        let vectors: Vec<Vector> =
            serde_json::from_str(include_str!("../tests/fixtures/v41_tool_vectors.json")).unwrap();
        for v in vectors {
            let calls: Vec<_> = v
                .calls
                .iter()
                .map(|c| ToolCall {
                    name: c.name.clone(),
                    namespace: None,
                    arguments: c.arguments.clone(),
                })
                .collect();
            assert_eq!(encode_calls(&calls).unwrap(), v.encoded_calls, "{}", v.name);
            let parsed = parse_completion(&v.completion, v.mode).unwrap();
            assert_eq!(parsed.content, v.content);
            assert_eq!(parsed.reasoning, v.reasoning);
            assert_eq!(parsed.calls.len(), calls.len());
            for (actual, input) in parsed.calls.iter().zip(calls) {
                let (ns, name) = split_name(&input.name, None).unwrap();
                assert_eq!(
                    (&actual.namespace, &actual.name, &actual.arguments),
                    (&ns, &name, &input.arguments)
                );
            }
        }
    }
    #[test]
    fn rejects_malformed_and_ambiguous_calls() {
        assert!(parse_completion("partial", ThinkingMode::Chat).is_err());
        let duplicate=format!("{CALLS}{INVOKE}f\">\n{PARAM}x\" string=\"false\">1{PARAM_END}\n{PARAM}x\" string=\"true\">y{PARAM_END}\n{INVOKE_END}\n{CALLS_END}{EOS}");
        assert!(parse_completion(&duplicate, ThinkingMode::Chat).is_err());
        let bad_json = duplicate.replace("false\">1", "false\">not-json");
        assert!(parse_completion(&bad_json, ThinkingMode::Chat).is_err());
        let mut args = Map::new();
        args.insert("x".into(), Value::String(PARAM_END.into()));
        assert!(encode_calls(&[ToolCall {
            name: "f".into(),
            namespace: None,
            arguments: args
        }])
        .is_err());
    }
}
