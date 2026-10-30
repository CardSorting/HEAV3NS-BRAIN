use singularity_core::SingularityError;
use std::path::Path;

pub struct TokenizerWrapper {
    inner: Option<tokenizers::Tokenizer>,
    eos_token: u32,
    bos_token: u32,
}

impl Default for TokenizerWrapper {
    fn default() -> Self {
        Self::new_fallback()
    }
}

impl TokenizerWrapper {
    /// Build the checkpoint's Engram token compression map from this tokenizer.
    pub fn v41_engram_map(&self, expected_size: usize) -> Result<Vec<u32>, SingularityError> {
        let tokenizer = self
            .inner
            .as_ref()
            .ok_or_else(|| SingularityError::ModelError {
                reason: "Engram requires a checkpoint tokenizer".into(),
            })?;
        crate::engram::compressed_vocabulary(tokenizer, expected_size)
    }
    /// Load the V4.1 tokenizer with explicit checkpoint special-token IDs.
    pub fn from_v41_file<P: AsRef<Path>>(path: P) -> Result<Self, SingularityError> {
        let p = path.as_ref();
        let tok =
            tokenizers::Tokenizer::from_file(p).map_err(|e| SingularityError::ModelError {
                reason: format!("Failed to load tokenizer from '{}': {e}", p.display()),
            })?;
        for (token, expected) in [
            ("<｜begin▁of▁sentence｜>", 0),
            ("<｜end▁of▁sentence｜>", 1),
            ("<｜User｜>", 128803),
            ("<｜Assistant｜>", 128804),
            ("<think>", 128821),
            ("</think>", 128822),
        ] {
            if tok.token_to_id(token) != Some(expected) {
                return Err(SingularityError::ModelError {
                    reason: format!("V4.1 tokenizer has an unexpected ID for '{token}'"),
                });
            }
        }
        Ok(Self {
            inner: Some(tok),
            eos_token: 1,
            bos_token: 0,
        })
    }

    pub fn new_fallback() -> Self {
        Self {
            inner: None,
            eos_token: 0,
            bos_token: 1,
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, SingularityError> {
        let p = path.as_ref();
        let tok =
            tokenizers::Tokenizer::from_file(p).map_err(|e| SingularityError::ModelError {
                reason: format!("Failed to load tokenizer from '{}': {e}", p.display()),
            })?;

        let eos_token = tok
            .token_to_id("<|im_end|>")
            .or_else(|| tok.token_to_id("<|endoftext|>"))
            .or_else(|| tok.token_to_id("</s>"))
            .unwrap_or(0);
        let bos_token = tok
            .token_to_id("<|im_start|>")
            .or_else(|| tok.token_to_id("<|startoftext|>"))
            .or_else(|| tok.token_to_id("<s>"))
            .unwrap_or(1);

        Ok(Self {
            inner: Some(tok),
            eos_token,
            bos_token,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, SingularityError> {
        if let Some(tok) = &self.inner {
            let encoding = tok
                .encode(text, false)
                .map_err(|e| SingularityError::ModelError {
                    reason: format!("Failed to encode text: {e}"),
                })?;
            Ok(encoding.get_ids().to_vec())
        } else {
            // Byte-level deterministic fallback
            Ok(text.bytes().map(|b| b as u32).collect())
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, SingularityError> {
        if let Some(tok) = &self.inner {
            let text = tok
                .decode(tokens, false)
                .map_err(|e| SingularityError::ModelError {
                    reason: format!("Failed to decode tokens: {e}"),
                })?;
            Ok(text)
        } else {
            let bytes: Vec<u8> = tokens.iter().map(|&t| (t & 0xFF) as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).to_string())
        }
    }

    pub fn eos_token(&self) -> u32 {
        self.eos_token
    }

    pub fn bos_token(&self) -> u32 {
        self.bos_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_tokenizer_roundtrip() {
        let tok = TokenizerWrapper::new_fallback();
        let input = "Hello, Singularity!";
        let encoded = tok.encode(input).unwrap();
        assert!(!encoded.is_empty());
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }
}
