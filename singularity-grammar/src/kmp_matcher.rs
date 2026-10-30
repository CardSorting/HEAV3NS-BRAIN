use crate::bitmask::{GrammarBitmaskEngine, JsonGrammarState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KmpConstraintState {
    OpenVocabulary,
    MatchingTrigger(usize),
    GrammarConstrained,
    MatchingStop(usize),
}

pub struct KmpTriggerMatcher {
    trigger_tokens: Vec<u32>,
    stop_tokens: Vec<u32>,
    trigger_pi: Vec<usize>,
    #[allow(dead_code)]
    stop_pi: Vec<usize>,
    state: KmpConstraintState,
    grammar_state: JsonGrammarState,
    bitmask_engine: GrammarBitmaskEngine,
}

impl KmpTriggerMatcher {
    pub fn new(trigger_tokens: Vec<u32>, stop_tokens: Vec<u32>, vocab_size: usize) -> Self {
        let trigger_pi = compute_kmp_pi(&trigger_tokens);
        let stop_pi = compute_kmp_pi(&stop_tokens);

        Self {
            trigger_tokens,
            stop_tokens,
            trigger_pi,
            stop_pi,
            state: KmpConstraintState::OpenVocabulary,
            grammar_state: JsonGrammarState::ExpectObjectStart,
            bitmask_engine: GrammarBitmaskEngine::new(vocab_size),
        }
    }

    pub fn current_state(&self) -> KmpConstraintState {
        self.state
    }

    pub fn is_constraint_active(&self) -> bool {
        matches!(
            self.state,
            KmpConstraintState::GrammarConstrained | KmpConstraintState::MatchingStop(_)
        )
    }

    /// Feeds the emitted token into the KMP matcher state machine.
    /// Returns true if constraint is now active, false if open vocabulary.
    pub fn feed_token(&mut self, token: u32) -> bool {
        match self.state {
            KmpConstraintState::OpenVocabulary => {
                if !self.trigger_tokens.is_empty() && token == self.trigger_tokens[0] {
                    if self.trigger_tokens.len() == 1 {
                        self.state = KmpConstraintState::GrammarConstrained;
                        self.grammar_state = JsonGrammarState::ExpectObjectStart;
                    } else {
                        self.state = KmpConstraintState::MatchingTrigger(1);
                    }
                }
            }
            KmpConstraintState::MatchingTrigger(idx) => {
                if idx < self.trigger_tokens.len() && token == self.trigger_tokens[idx] {
                    if idx + 1 == self.trigger_tokens.len() {
                        self.state = KmpConstraintState::GrammarConstrained;
                        self.grammar_state = JsonGrammarState::ExpectObjectStart;
                    } else {
                        self.state = KmpConstraintState::MatchingTrigger(idx + 1);
                    }
                } else {
                    // Fallback using pi table
                    let next_idx = self
                        .trigger_pi
                        .get(idx.saturating_sub(1))
                        .copied()
                        .unwrap_or(0);
                    if next_idx > 0 && token == self.trigger_tokens[next_idx] {
                        self.state = KmpConstraintState::MatchingTrigger(next_idx + 1);
                    } else {
                        self.state = KmpConstraintState::OpenVocabulary;
                    }
                }
            }
            KmpConstraintState::GrammarConstrained => {
                // Advance JSON grammar state
                self.grammar_state = self.bitmask_engine.advance_state(self.grammar_state, token);

                // Check if token matches start of stop sequence
                if !self.stop_tokens.is_empty() && token == self.stop_tokens[0] {
                    if self.stop_tokens.len() == 1 {
                        self.state = KmpConstraintState::OpenVocabulary;
                    } else {
                        self.state = KmpConstraintState::MatchingStop(1);
                    }
                }
            }
            KmpConstraintState::MatchingStop(idx) => {
                if idx < self.stop_tokens.len() && token == self.stop_tokens[idx] {
                    if idx + 1 == self.stop_tokens.len() {
                        self.state = KmpConstraintState::OpenVocabulary;
                    } else {
                        self.state = KmpConstraintState::MatchingStop(idx + 1);
                    }
                } else {
                    // Stop matching interrupted, resume grammar constrained
                    self.grammar_state =
                        self.bitmask_engine.advance_state(self.grammar_state, token);
                    self.state = KmpConstraintState::GrammarConstrained;
                }
            }
        }

        self.is_constraint_active()
    }

    /// If constraint is active, generates grammar bitmask to filter candidate logits.
    pub fn filter_logits_if_active(&self, logits: &mut [f32]) {
        if self.is_constraint_active() {
            let _ = self.bitmask_engine.apply_mask(logits, self.grammar_state);
        }
    }
}

fn compute_kmp_pi(pattern: &[u32]) -> Vec<usize> {
    let n = pattern.len();
    let mut pi = vec![0; n];
    let mut j = 0;

    for i in 1..n {
        while j > 0 && pattern[i] != pattern[j] {
            j = pi[j - 1];
        }
        if pattern[i] == pattern[j] {
            j += 1;
        }
        pi[i] = j;
    }

    pi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmp_pi_computation() {
        let pattern = vec![1, 2, 1, 2, 3];
        let pi = compute_kmp_pi(&pattern);
        assert_eq!(pi, vec![0, 0, 1, 2, 0]);
    }

    #[test]
    fn test_kmp_matcher_trigger_and_stop() {
        let mut matcher = KmpTriggerMatcher::new(vec![10, 20], vec![30], 128);
        assert_eq!(matcher.current_state(), KmpConstraintState::OpenVocabulary);

        matcher.feed_token(10);
        assert_eq!(
            matcher.current_state(),
            KmpConstraintState::MatchingTrigger(1)
        );

        matcher.feed_token(20);
        assert_eq!(
            matcher.current_state(),
            KmpConstraintState::GrammarConstrained
        );
        assert!(matcher.is_constraint_active());

        matcher.feed_token(30);
        assert_eq!(matcher.current_state(), KmpConstraintState::OpenVocabulary);
        assert!(!matcher.is_constraint_active());
    }
}
