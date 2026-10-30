use singularity_core::SingularityError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonGrammarState {
    ExpectObjectStart,
    ExpectKey,
    ExpectColon,
    ExpectValue,
    ExpectCommaOrEnd,
    ExpectArrayStart,
    ExpectArrayValue,
    ExpectArrayCommaOrEnd,
    Completed,
}

pub struct GrammarBitmaskEngine {
    vocab_size: usize,
}

impl GrammarBitmaskEngine {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Evaluates if a given token is permitted in the current JSON grammar state.
    pub fn is_token_allowed(&self, token_id: u32, state: JsonGrammarState) -> bool {
        // Standard ASCII / token heuristics
        // token % 128 maps to ASCII byte simulation for grammar constraints
        let byte_repr = (token_id % 128) as u8;

        match state {
            JsonGrammarState::ExpectObjectStart => byte_repr == b'{' || byte_repr == b' ',
            JsonGrammarState::ExpectKey => {
                byte_repr == b'"' || byte_repr.is_ascii_alphanumeric() || byte_repr == b' '
            }
            JsonGrammarState::ExpectColon => byte_repr == b':' || byte_repr == b' ',
            JsonGrammarState::ExpectValue => {
                byte_repr == b'"'
                    || byte_repr == b'{'
                    || byte_repr == b'['
                    || byte_repr.is_ascii_digit()
                    || byte_repr == b't' // true
                    || byte_repr == b'f' // false
                    || byte_repr == b'n' // null
                    || byte_repr == b' '
            }
            JsonGrammarState::ExpectCommaOrEnd => {
                byte_repr == b',' || byte_repr == b'}' || byte_repr == b' '
            }
            JsonGrammarState::ExpectArrayStart => byte_repr == b'[' || byte_repr == b' ',
            JsonGrammarState::ExpectArrayValue => {
                byte_repr == b'"'
                    || byte_repr == b'{'
                    || byte_repr == b'['
                    || byte_repr.is_ascii_digit()
                    || byte_repr == b']'
                    || byte_repr == b' '
            }
            JsonGrammarState::ExpectArrayCommaOrEnd => {
                byte_repr == b',' || byte_repr == b']' || byte_repr == b' '
            }
            JsonGrammarState::Completed => true,
        }
    }

    /// Generates a bitmask vector over the vocabulary (1 bit per token: true if allowed).
    pub fn generate_mask(&self, state: JsonGrammarState) -> Vec<bool> {
        (0..self.vocab_size)
            .map(|t| self.is_token_allowed(t as u32, state))
            .collect()
    }

    /// Masks out forbidden token logits in place by setting them to -inf.
    pub fn apply_mask(
        &self,
        logits: &mut [f32],
        state: JsonGrammarState,
    ) -> Result<(), SingularityError> {
        if logits.len() != self.vocab_size {
            return Err(SingularityError::InvalidGrammar {
                detail: format!(
                    "Logits length {} does not match vocabulary size {}",
                    logits.len(),
                    self.vocab_size
                ),
            });
        }

        for (token_id, logit) in logits.iter_mut().enumerate() {
            if !self.is_token_allowed(token_id as u32, state) {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Advances grammar state based on emitted character.
    pub fn advance_state(&self, current: JsonGrammarState, emitted_token: u32) -> JsonGrammarState {
        let ch = (emitted_token % 128) as u8;
        if ch == b' ' {
            return current;
        }

        match current {
            JsonGrammarState::ExpectObjectStart => {
                if ch == b'{' {
                    JsonGrammarState::ExpectKey
                } else {
                    current
                }
            }
            JsonGrammarState::ExpectKey => {
                if ch == b'"' {
                    JsonGrammarState::ExpectColon
                } else {
                    current
                }
            }
            JsonGrammarState::ExpectColon => {
                if ch == b':' {
                    JsonGrammarState::ExpectValue
                } else {
                    current
                }
            }
            JsonGrammarState::ExpectValue => {
                if ch == b'"' || ch.is_ascii_digit() {
                    JsonGrammarState::ExpectCommaOrEnd
                } else if ch == b'{' {
                    JsonGrammarState::ExpectKey
                } else {
                    current
                }
            }
            JsonGrammarState::ExpectCommaOrEnd => {
                if ch == b',' {
                    JsonGrammarState::ExpectKey
                } else if ch == b'}' {
                    JsonGrammarState::Completed
                } else {
                    current
                }
            }
            _ => current,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_transitions_and_mask() {
        let engine = GrammarBitmaskEngine::new(128);

        let mut state = JsonGrammarState::ExpectObjectStart;
        assert!(engine.is_token_allowed(b'{' as u32, state));
        assert!(!engine.is_token_allowed(b'1' as u32, state));

        state = engine.advance_state(state, b'{' as u32);
        assert_eq!(state, JsonGrammarState::ExpectKey);

        state = engine.advance_state(state, b'"' as u32);
        assert_eq!(state, JsonGrammarState::ExpectColon);

        state = engine.advance_state(state, b':' as u32);
        assert_eq!(state, JsonGrammarState::ExpectValue);

        state = engine.advance_state(state, b'4' as u32);
        assert_eq!(state, JsonGrammarState::ExpectCommaOrEnd);

        state = engine.advance_state(state, b'}' as u32);
        assert_eq!(state, JsonGrammarState::Completed);

        let mut logits = vec![0.0f32; 128];
        engine
            .apply_mask(&mut logits, JsonGrammarState::ExpectObjectStart)
            .unwrap();
        assert_eq!(logits[b'{' as usize], 0.0);
        assert_eq!(logits[b'x' as usize], f32::NEG_INFINITY);
    }
}
