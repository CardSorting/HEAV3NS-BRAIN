use serde::{Deserialize, Serialize};
use std::time::Instant;

pub type SequenceId = u64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    StopToken(u32),
    LengthLimit,
    ClientCancelled,
    Preempted,
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Preempted,
    Finished(FinishReason),
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub grammar_constraint: Option<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 256,
            stop_tokens: vec![0, 1, 2], // Standard EOS tokens
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            grammar_constraint: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: SequenceId,
    pub prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub status: SequenceStatus,
    pub sampling_params: SamplingParams,
    pub allocated_block_ids: Vec<usize>,
    pub num_cached_tokens: usize,
    pub created_at: Instant,
    pub first_token_time: Option<Instant>,
    pub ttft_ns: Option<u64>,
    pub itl_records_ns: Vec<u64>,
    pub last_token_time: Option<Instant>,
}

impl Sequence {
    pub fn new(
        id: SequenceId,
        prompt: String,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
    ) -> Self {
        Self {
            id,
            prompt,
            prompt_tokens,
            output_tokens: Vec::new(),
            status: SequenceStatus::Waiting,
            sampling_params: params,
            allocated_block_ids: Vec::new(),
            num_cached_tokens: 0,
            created_at: Instant::now(),
            first_token_time: None,
            ttft_ns: None,
            itl_records_ns: Vec::new(),
            last_token_time: None,
        }
    }

    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    #[inline]
    pub fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    #[inline]
    pub fn output_len(&self) -> usize {
        self.output_tokens.len()
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        matches!(
            self.status,
            SequenceStatus::Finished(_) | SequenceStatus::Cancelled
        )
    }

    pub fn append_token(&mut self, token: u32) {
        let now = Instant::now();
        if self.output_tokens.is_empty() {
            self.first_token_time = Some(now);
            self.ttft_ns = Some(now.duration_since(self.created_at).as_nanos() as u64);
        } else if let Some(last) = self.last_token_time {
            let itl = now.duration_since(last).as_nanos() as u64;
            self.itl_records_ns.push(itl);
        }
        self.last_token_time = Some(now);
        self.output_tokens.push(token);

        if self.sampling_params.stop_tokens.contains(&token) {
            self.status = SequenceStatus::Finished(FinishReason::StopToken(token));
        } else if self.output_tokens.len() >= self.sampling_params.max_tokens {
            self.status = SequenceStatus::Finished(FinishReason::LengthLimit);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenBlock {
    pub block_id: usize,
    pub physical_index: usize,
    pub ref_count: usize,
    pub tokens: Vec<u32>,
    pub is_pinned: bool,
    pub is_prefix_cached: bool,
    pub generation: u64,
}

impl TokenBlock {
    pub fn new(block_id: usize, physical_index: usize) -> Self {
        Self {
            block_id,
            physical_index,
            ref_count: 0,
            tokens: Vec::new(),
            is_pinned: false,
            is_prefix_cached: false,
            generation: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ref_count = 0;
        self.tokens.clear();
        self.is_pinned = false;
        self.is_prefix_cached = false;
        self.generation = self.generation.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_lifecycle_and_ttft() {
        let params = SamplingParams {
            max_tokens: 3,
            stop_tokens: vec![99],
            ..Default::default()
        };
        let mut seq = Sequence::new(1, "prompt".to_string(), vec![10, 20], params);
        assert_eq!(seq.prompt_len(), 2);
        assert_eq!(seq.output_len(), 0);
        assert_eq!(seq.total_tokens(), 2);
        assert!(!seq.is_finished());

        // First token sets TTFT
        seq.append_token(50);
        assert_eq!(seq.output_len(), 1);
        assert!(seq.ttft_ns.is_some());
        assert!(!seq.is_finished());

        // Second token records ITL
        seq.append_token(51);
        assert_eq!(seq.output_len(), 2);
        assert_eq!(seq.itl_records_ns.len(), 1);
        assert!(!seq.is_finished());

        // Third token hits length limit
        seq.append_token(52);
        assert_eq!(seq.output_len(), 3);
        assert!(seq.is_finished());
        assert_eq!(
            seq.status,
            SequenceStatus::Finished(FinishReason::LengthLimit)
        );
    }

    #[test]
    fn test_sequence_stop_token() {
        let params = SamplingParams {
            max_tokens: 10,
            stop_tokens: vec![99],
            ..Default::default()
        };
        let mut seq = Sequence::new(2, "prompt".to_string(), vec![10], params);
        seq.append_token(99);
        assert!(seq.is_finished());
        assert_eq!(
            seq.status,
            SequenceStatus::Finished(FinishReason::StopToken(99))
        );
    }
}
