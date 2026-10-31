use crate::rank_health::CollectiveError;
use singularity_core::{EngineType, SchedulerConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepType {
    Prefill,
    Decode,
    Idle,
}

pub struct DpConsensusEngine {
    config: SchedulerConfig,
    decode_step: usize,
}

impl DpConsensusEngine {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            decode_step: 0,
        }
    }

    /// Evaluates whether the DP group should execute Prefill or Decode on this step.
    /// Implements DeepInfer's two-tier ALL vs ANY consensus logic (csrc/engine/scheduler.hpp#L290-L342).
    ///
    /// local_has_prompts: bool indicating if this rank has waiting prompt requests.
    /// dp_all_reduce_fn: simulated or NCCL all-reduce function over all ranks.
    pub fn decide_step_type<F>(
        &mut self,
        local_has_prompts: bool,
        mut dp_all_reduce_fn: F,
    ) -> StepType
    where
        F: FnMut(i32, &str) -> i32,
    {
        match self.config.engine_type {
            EngineType::Prompt => {
                if local_has_prompts {
                    StepType::Prefill
                } else {
                    StepType::Idle
                }
            }
            EngineType::Completion => StepType::Decode,
            EngineType::Normal => {
                let any_interval = self.config.prefill_any_attempt_interval.max(1);
                let all_interval = self.config.prefill_attempt_interval.max(1);

                let is_any_point =
                    (self.decode_step != 0) && (self.decode_step % any_interval == 0);
                let is_all_point = self.decode_step % all_interval == 0;

                if !is_any_point && !is_all_point {
                    self.decode_step = self.decode_step.wrapping_add(1);
                    return StepType::Decode;
                }

                let want_prefill = if is_any_point {
                    // ANY Point: if any rank wants prefill (min reduce of (want?0:1) == 0)
                    let inverted_vote = if local_has_prompts { 0 } else { 1 };
                    dp_all_reduce_fn(inverted_vote, "MIN") == 0
                } else {
                    // ALL Point: all ranks must want prefill (min reduce == 1)
                    let vote = if local_has_prompts { 1 } else { 0 };
                    dp_all_reduce_fn(vote, "MIN") == 1
                };

                if want_prefill {
                    self.decode_step = 0;
                    StepType::Prefill
                } else {
                    self.decode_step = self.decode_step.wrapping_add(1);
                    StepType::Decode
                }
            }
        }
    }

    /// Like [`Self::decide_step_type`] but the reduce can fail (timeout / dead rank).
    /// On error the engine state is left untouched and the error is returned, so the
    /// caller fails fast instead of acting on a fabricated vote.
    pub fn decide_step_type_checked<F>(
        &mut self,
        local_has_prompts: bool,
        mut dp_all_reduce_fn: F,
    ) -> Result<StepType, CollectiveError>
    where
        F: FnMut(i32, &str) -> Result<i32, CollectiveError>,
    {
        let saved = self.decode_step;
        let mut failure: Option<CollectiveError> = None;
        let step = self.decide_step_type(local_has_prompts, |v, op| {
            if failure.is_some() {
                return v;
            }
            match dp_all_reduce_fn(v, op) {
                Ok(x) => x,
                Err(e) => {
                    failure = Some(e);
                    v
                }
            }
        });
        match failure {
            Some(e) => {
                self.decode_step = saved;
                Err(e)
            }
            None => Ok(step),
        }
    }

    /// Two-phase validation: if all ranks scheduled 0 tokens during prefill,
    /// switch step type immediately to Decode to avoid infinite zero-work loops.
    pub fn validate_two_phase<F>(
        &mut self,
        scheduled_tokens: usize,
        mut dp_all_reduce_fn: F,
    ) -> StepType
    where
        F: FnMut(i32, &str) -> i32,
    {
        let vote = if scheduled_tokens > 0 { 1 } else { 0 };
        let max_scheduled = dp_all_reduce_fn(vote, "MAX");
        if max_scheduled == 0 {
            // No rank actually scheduled tokens (e.g. KV cache full across cluster)
            self.decode_step = self.decode_step.wrapping_add(1);
            StepType::Decode
        } else {
            StepType::Prefill
        }
    }

    /// Like [`Self::validate_two_phase`] but the reduce can fail; state untouched on error.
    pub fn validate_two_phase_checked<F>(
        &mut self,
        scheduled_tokens: usize,
        mut dp_all_reduce_fn: F,
    ) -> Result<StepType, CollectiveError>
    where
        F: FnMut(i32, &str) -> Result<i32, CollectiveError>,
    {
        let vote = if scheduled_tokens > 0 { 1 } else { 0 };
        let max_scheduled = dp_all_reduce_fn(vote, "MAX")?;
        if max_scheduled == 0 {
            self.decode_step = self.decode_step.wrapping_add(1);
            Ok(StepType::Decode)
        } else {
            Ok(StepType::Prefill)
        }
    }

    pub fn current_decode_step(&self) -> usize {
        self.decode_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp_consensus_handles_zero_interval_without_panic() {
        let config = SchedulerConfig {
            prefill_attempt_interval: 0,
            prefill_any_attempt_interval: 0,
            ..Default::default()
        };
        let mut consensus = DpConsensusEngine::new(config);
        // Should not panic with divide by zero!
        let step = consensus.decide_step_type(false, |_vote, _op| 0);
        assert_eq!(step, StepType::Decode);
    }

    #[test]
    fn checked_decide_propagates_collective_failure_and_keeps_state() {
        let config = SchedulerConfig {
            prefill_attempt_interval: 1,
            prefill_any_attempt_interval: 1,
            ..Default::default()
        };
        let mut consensus = DpConsensusEngine::new(config);
        let err = consensus
            .decide_step_type_checked(true, |_v, _op| {
                Err(CollectiveError::Timeout(std::time::Duration::from_secs(1)))
            })
            .unwrap_err();
        assert!(matches!(err, CollectiveError::Timeout(_)));
        assert_eq!(consensus.current_decode_step(), 0);
        let ok = consensus.decide_step_type_checked(true, |v, _op| Ok(v));
        assert!(ok.is_ok());
    }
}
