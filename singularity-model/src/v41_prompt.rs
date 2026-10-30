//! DeepSeek V4.1 text-only prompt serialization.
//! Format derived from the checkpoint's MIT-licensed encoding/encoding.py at
//! revision 1bae0e1088993b9b82de08f59de5be5a34e42170. Tool calls, structured
//! output, task tokens and image blocks are not accepted by this API yet.
use serde::Deserialize;
use singularity_core::SingularityError;

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
const SYSTEM: &str = "<｜System｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const IMAGE: &str = "<｜deepseek_image｜>";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextMessage {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub reasoning_content: String,
    #[serde(default)]
    pub wo_eos: bool,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

/// Encode the text-only conversation using the reference's default
/// `drop_thinking=true`, with an explicit numeric reasoning budget.
pub fn encode_text(
    messages: &[TextMessage],
    mode: ThinkingMode,
    effort: u8,
) -> Result<String, SingularityError> {
    if !(1..=100).contains(&effort) {
        return Err(error("V4.1 reasoning effort must be between 1 and 100"));
    }
    if messages
        .iter()
        .any(|m| m.content.contains(IMAGE) || m.reasoning_content.contains(IMAGE))
    {
        return Err(error("Image special tokens must not be supplied as text"));
    }
    let last_user = messages
        .iter()
        .enumerate()
        .rfind(|(i, m)| matches!(m.role, Role::User) || (*i > 0 && matches!(m.role, Role::System)))
        .map(|(i, _)| i);
    let mut prompt = BOS.to_owned();
    for (i, msg) in messages.iter().enumerate() {
        if i == 0 && (matches!(mode, ThinkingMode::Thinking) || matches!(msg.role, Role::System)) {
            prompt.push_str(SYSTEM);
        }
        if i == 0 && matches!(mode, ThinkingMode::Thinking) {
            prompt.push_str(&format!("Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"));
        }
        match msg.role {
            Role::System => {
                if i > 0 {
                    prompt.push_str(SYSTEM);
                }
                prompt.push_str(&msg.content);
            }
            Role::User => {
                prompt.push_str(USER);
                prompt.push_str(&msg.content);
            }
            Role::Assistant => {
                if matches!(mode, ThinkingMode::Thinking) && last_user.is_none_or(|idx| i > idx) {
                    prompt.push_str(&msg.reasoning_content);
                    prompt.push_str("</think>");
                }
                prompt.push_str(&msg.content);
                if !msg.wo_eos {
                    prompt.push_str(EOS);
                }
            }
        }
        let next_allows_transition = messages
            .get(i + 1)
            .is_none_or(|m| matches!(m.role, Role::Assistant));
        if next_allows_transition
            && (matches!(msg.role, Role::User) || (i > 0 && matches!(msg.role, Role::System)))
        {
            prompt.push_str(ASSISTANT);
            if matches!(mode, ThinkingMode::Thinking) && last_user.is_none_or(|idx| i >= idx) {
                prompt.push_str("<think>");
            } else {
                prompt.push_str("</think>");
            }
        }
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Deserialize)]
    struct Vector {
        name: String,
        mode: ThinkingMode,
        effort: u8,
        messages: Vec<TextMessage>,
        prompt: String,
        token_ids: Vec<u32>,
    }
    fn vectors() -> Vec<Vector> {
        serde_json::from_str(include_str!("../tests/fixtures/v41_text_vectors.json")).unwrap()
    }
    #[test]
    fn text_matches_checkpoint_reference_vectors() {
        for vector in vectors() {
            assert_eq!(
                encode_text(&vector.messages, vector.mode, vector.effort).unwrap(),
                vector.prompt,
                "{}",
                vector.name
            );
        }
        assert!(encode_text(&[], ThinkingMode::Thinking, 0).is_err());
        assert!(serde_json::from_str::<TextMessage>(r#"{"role":"tool","content":"x"}"#).is_err());
        assert!(
            serde_json::from_str::<TextMessage>(r#"{"role":"user","content":"x","tools":[]}"#)
                .is_err()
        );
    }
    #[test]
    #[ignore = "requires SINGULARITY_V41_TOKENIZER pointing to the pinned tokenizer.json"]
    fn text_token_ids_match_checkpoint_reference() {
        let tokenizer = crate::TokenizerWrapper::from_v41_file(
            std::env::var("SINGULARITY_V41_TOKENIZER").unwrap(),
        )
        .unwrap();
        for vector in vectors() {
            let prompt = encode_text(&vector.messages, vector.mode, vector.effort).unwrap();
            assert_eq!(
                tokenizer.encode(&prompt).unwrap(),
                vector.token_ids,
                "{}",
                vector.name
            );
        }
        assert_eq!((tokenizer.bos_token(), tokenizer.eos_token()), (0, 1));
    }
}
