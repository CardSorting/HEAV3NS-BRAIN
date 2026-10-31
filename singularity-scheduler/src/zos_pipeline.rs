use crate::dp_consensus::{DpConsensusEngine, StepType};
use crate::lcfs::schedule_decode_lcfs;
use crate::mtp_estimator::{
    DynamicMtpDraftAllocator, ForwardRuntimeEstimator, SamplerRuntimeEstimator,
};
use parking_lot::Mutex;
use singularity_cache::{PagedBlockManager, RadixTree};
use singularity_core::{SchedulerConfig, Sequence, SequenceId, SequenceStatus, SingularityError};
use singularity_cuda::{GpuBackend, GpuEvent, GpuStream};
use singularity_model::{KvCache, Sampler, TransformerConfig, TransformerModel};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

pub struct DeferredRelease {
    pub unpin_blocks: Vec<usize>,
    pub completed_seq_ids: Vec<SequenceId>,
}

impl Default for DeferredRelease {
    fn default() -> Self {
        Self::new()
    }
}

impl DeferredRelease {
    pub fn new() -> Self {
        Self {
            unpin_blocks: Vec::new(),
            completed_seq_ids: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.unpin_blocks.is_empty() && self.completed_seq_ids.is_empty()
    }

    pub fn execute(&mut self, block_manager: &PagedBlockManager) {
        for &bid in &self.unpin_blocks {
            let _ = block_manager.unpin_block(bid);
        }
        self.unpin_blocks.clear();
        self.completed_seq_ids.clear();
    }
}

#[derive(Debug, Clone)]
pub enum SequenceOutputEvent {
    Token(u32),
    Finished(Box<Sequence>),
    Cancelled,
}

pub struct StepOutput {
    pub step_type: StepType,
    pub scheduled_seq_ids: Vec<SequenceId>,
    pub newly_finished_seqs: Vec<Sequence>,
    pub generated_tokens: HashMap<SequenceId, Vec<u32>>,
    pub preempted_seq_ids: Vec<SequenceId>,
    pub step_duration_ns: u64,
}

pub struct ZosScheduler {
    pub config: SchedulerConfig,
    pub block_manager: Arc<PagedBlockManager>,
    pub radix_tree: Arc<RadixTree>,
    #[allow(dead_code)]
    pub gpu_backend: Arc<dyn GpuBackend>,
    #[allow(dead_code)]
    pub compute_stream: Arc<GpuStream>,
    #[allow(dead_code)]
    pub sample_end_event: Arc<GpuEvent>,
    pub model: Arc<TransformerModel>,
    kv_caches: Mutex<HashMap<SequenceId, KvCache>>,

    waiting_queue: Mutex<VecDeque<Sequence>>,
    running_sequences: Mutex<Vec<Sequence>>,
    preempted_queue: Mutex<VecDeque<Sequence>>,

    dp_consensus: Mutex<DpConsensusEngine>,
    #[allow(dead_code)]
    pub forward_estimator: Mutex<ForwardRuntimeEstimator>,
    #[allow(dead_code)]
    pub sampler_estimator: Mutex<SamplerRuntimeEstimator>,
    pub draft_allocator: DynamicMtpDraftAllocator,
    pub deferred_release: Mutex<DeferredRelease>,

    listeners: Mutex<HashMap<SequenceId, tokio::sync::mpsc::UnboundedSender<SequenceOutputEvent>>>,
    step_lock: Mutex<()>,
}

impl ZosScheduler {
    pub fn new(
        config: SchedulerConfig,
        block_manager: Arc<PagedBlockManager>,
        radix_tree: Arc<RadixTree>,
        gpu_backend: Arc<dyn GpuBackend>,
    ) -> Result<Self, SingularityError> {
        let default_model = Arc::new(TransformerModel::new_random(TransformerConfig::tiny()));
        Self::with_model(
            config,
            block_manager,
            radix_tree,
            gpu_backend,
            default_model,
        )
    }

    pub fn with_model(
        config: SchedulerConfig,
        block_manager: Arc<PagedBlockManager>,
        radix_tree: Arc<RadixTree>,
        gpu_backend: Arc<dyn GpuBackend>,
        model: Arc<TransformerModel>,
    ) -> Result<Self, SingularityError> {
        let compute_stream = gpu_backend.create_stream("zos_compute_stream")?;
        let sample_end_event = gpu_backend.create_event()?;

        let dp_consensus = Mutex::new(DpConsensusEngine::new(config.clone()));
        let forward_estimator = Mutex::new(ForwardRuntimeEstimator::default_estimator());
        let sampler_estimator = Mutex::new(SamplerRuntimeEstimator::default_estimator());
        let draft_allocator = DynamicMtpDraftAllocator::default_allocator();

        Ok(Self {
            config,
            block_manager,
            radix_tree,
            gpu_backend,
            compute_stream,
            sample_end_event,
            model,
            kv_caches: Mutex::new(HashMap::new()),
            waiting_queue: Mutex::new(VecDeque::new()),
            running_sequences: Mutex::new(Vec::new()),
            preempted_queue: Mutex::new(VecDeque::new()),
            dp_consensus,
            forward_estimator,
            sampler_estimator,
            draft_allocator,
            deferred_release: Mutex::new(DeferredRelease::new()),
            listeners: Mutex::new(HashMap::new()),
            step_lock: Mutex::new(()),
        })
    }

    pub fn subscribe(
        &self,
        seq_id: SequenceId,
    ) -> tokio::sync::mpsc::UnboundedReceiver<SequenceOutputEvent> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.listeners.lock().insert(seq_id, tx);
        rx
    }

    pub fn has_work(&self) -> bool {
        self.waiting_count() > 0 || self.running_count() > 0 || self.preempted_count() > 0
    }

    pub fn step_cooperative(&self) -> Result<Option<StepOutput>, SingularityError> {
        let _guard = match self.step_lock.try_lock() {
            Some(g) => g,
            None => return Ok(None),
        };
        if self.has_work() {
            let out = self.step()?;
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    pub fn submit_sequence(&self, mut seq: Sequence) -> Result<(), SingularityError> {
        // Check Radix prefix cache for prompt prefix reuse
        let (matched_blocks, matched_tokens) = self.radix_tree.match_prefix(&seq.prompt_tokens);
        if matched_tokens > 0 {
            for &bid in &matched_blocks {
                self.block_manager.pin_block(bid)?;
            }
            seq.allocated_block_ids = matched_blocks;
            seq.num_cached_tokens = matched_tokens;
        }

        self.waiting_queue.lock().push_back(seq);
        Ok(())
    }

    pub fn cancel_sequence(&self, seq_id: SequenceId) -> bool {
        if let Some(tx) = self.listeners.lock().remove(&seq_id) {
            let _ = tx.send(SequenceOutputEvent::Cancelled);
        }
        self.kv_caches.lock().remove(&seq_id);

        // Check waiting queue
        {
            let mut waiting = self.waiting_queue.lock();
            if let Some(pos) = waiting.iter().position(|s| s.id == seq_id) {
                let seq = waiting.remove(pos).unwrap();
                let _ = self.block_manager.free_blocks(&seq.allocated_block_ids);
                return true;
            }
        }
        // Check running sequences
        {
            let mut running = self.running_sequences.lock();
            if let Some(pos) = running.iter().position(|s| s.id == seq_id) {
                let mut seq = running.remove(pos);
                seq.status = SequenceStatus::Cancelled;
                let _ = self.block_manager.free_blocks(&seq.allocated_block_ids);
                return true;
            }
        }
        // Check preempted queue
        {
            let mut preempted = self.preempted_queue.lock();
            if let Some(pos) = preempted.iter().position(|s| s.id == seq_id) {
                let seq = preempted.remove(pos).unwrap();
                let _ = self.block_manager.free_blocks(&seq.allocated_block_ids);
                return true;
            }
        }
        false
    }

    /// Primary ZOS execution step:
    /// Coordinates DP consensus, prefill / decode scheduling, LCFS preemption,
    /// mock/GPU stream execution, and deferred resource cleanup.
    pub fn step(&self) -> Result<StepOutput, SingularityError> {
        let step_start = Instant::now();

        // 1. Pre-Sampler Sync & Deferred Resource Release from Step N-1
        {
            let mut deferred = self.deferred_release.lock();
            if !deferred.is_empty() {
                deferred.execute(&self.block_manager);
            }
        }

        // 2. DP Consensus Voting: Prefill vs Decode
        let has_waiting =
            !self.waiting_queue.lock().is_empty() || !self.preempted_queue.lock().is_empty();
        let step_type = {
            let mut dp = self.dp_consensus.lock();
            dp.decide_step_type(has_waiting, |val, _op| val) // In single-node/standalone, local vote is unanimous
        };

        let mut newly_finished = Vec::new();
        let mut generated = HashMap::new();
        let mut scheduled_seq_ids = Vec::new();
        let mut preempted_seq_ids = Vec::new();

        match step_type {
            StepType::Prefill => {
                // Schedule prompts from preempted first, then waiting queue
                let mut scheduled_prefill = Vec::new();
                let mut current_chunk_tokens = 0;

                // Priority 1: Resume preempted requests
                {
                    let mut preempted = self.preempted_queue.lock();
                    while let Some(mut seq) = preempted.pop_front() {
                        let remaining_tokens =
                            seq.total_tokens().saturating_sub(seq.num_cached_tokens);
                        if current_chunk_tokens + remaining_tokens <= self.config.prefill_chunk_size
                        {
                            let needed_blocks =
                                seq.total_tokens().div_ceil(self.block_manager.block_size());
                            if self.block_manager.can_allocate(needed_blocks) {
                                if !seq.allocated_block_ids.is_empty() {
                                    let _ =
                                        self.block_manager.free_blocks(&seq.allocated_block_ids);
                                    seq.allocated_block_ids.clear();
                                }
                                let blocks = self.block_manager.allocate_blocks(needed_blocks)?;
                                seq.allocated_block_ids = blocks;
                                current_chunk_tokens += remaining_tokens;
                                scheduled_prefill.push(seq);
                            } else {
                                preempted.push_front(seq);
                                break;
                            }
                        } else {
                            preempted.push_front(seq);
                            break;
                        }
                    }
                }

                // Priority 2: New waiting prompts
                {
                    let mut waiting = self.waiting_queue.lock();
                    while let Some(mut seq) = waiting.pop_front() {
                        let remaining_tokens =
                            seq.prompt_len().saturating_sub(seq.num_cached_tokens);
                        if current_chunk_tokens + remaining_tokens <= self.config.prefill_chunk_size
                            && scheduled_prefill.len() < self.config.max_batch_size
                        {
                            let total_needed_blocks =
                                seq.prompt_len().div_ceil(self.block_manager.block_size());
                            let additional_blocks =
                                total_needed_blocks.saturating_sub(seq.allocated_block_ids.len());

                            if self.block_manager.can_allocate(additional_blocks) {
                                if additional_blocks > 0 {
                                    let new_blocks =
                                        self.block_manager.allocate_blocks(additional_blocks)?;
                                    seq.allocated_block_ids.extend_from_slice(&new_blocks);
                                }
                                current_chunk_tokens += remaining_tokens;

                                // Cache prompt prefix in Radix Tree for future requests
                                let _ = self.radix_tree.insert_prefix(
                                    &seq.prompt_tokens,
                                    &seq.allocated_block_ids,
                                    &self.block_manager,
                                );

                                scheduled_prefill.push(seq);
                            } else {
                                waiting.push_front(seq);
                                break;
                            }
                        } else {
                            waiting.push_front(seq);
                            break;
                        }
                    }
                }

                if scheduled_prefill.is_empty() {
                    // Two-phase consensus: 0 tokens scheduled in prefill step.
                    // If running sequences exist, switch to Decode immediately to avoid starvation/infinite prefill spin loops.
                    let fallback_step = self
                        .dp_consensus
                        .lock()
                        .validate_two_phase(0, |val, _op| val);
                    if fallback_step == StepType::Decode
                        && !self.running_sequences.lock().is_empty()
                    {
                        self.execute_decode(
                            &mut scheduled_seq_ids,
                            &mut preempted_seq_ids,
                            &mut generated,
                            &mut newly_finished,
                        )?;
                    }
                } else {
                    // Execute prefill and move to running queue
                    for mut seq in scheduled_prefill {
                        seq.status = SequenceStatus::Running;
                        let mut cache = KvCache::new(&self.model.config);
                        // Rebuild KV cache across all prompt and output tokens (if resumed after preemption)
                        let all_tokens: Vec<u32> = seq
                            .prompt_tokens
                            .iter()
                            .chain(seq.output_tokens.iter())
                            .copied()
                            .collect();
                        let prompt = if all_tokens.is_empty() {
                            vec![1]
                        } else {
                            all_tokens
                        };
                        let logits = self.model.forward(&prompt, 0, &mut cache)?;
                        let first_token =
                            Sampler::sample(&logits, &seq.output_tokens, &seq.sampling_params);
                        self.kv_caches.lock().insert(seq.id, cache);

                        seq.append_token(first_token);
                        generated.insert(seq.id, vec![first_token]);
                        scheduled_seq_ids.push(seq.id);

                        // Sync tokens to device-backed physical block manager if enabled
                        if self.block_manager.device_buffer().is_some()
                            && !seq.allocated_block_ids.is_empty()
                        {
                            let block_size = self.block_manager.block_size();
                            for (idx, &tok) in prompt.iter().enumerate() {
                                let block_idx_in_seq = idx / block_size;
                                if block_idx_in_seq < seq.allocated_block_ids.len() {
                                    let physical_block_id =
                                        seq.allocated_block_ids[block_idx_in_seq];
                                    let offset_in_block = (idx % block_size) * 4;
                                    let token_bytes = tok.to_le_bytes();
                                    let _ = self.block_manager.write_block_data(
                                        physical_block_id,
                                        offset_in_block,
                                        &token_bytes,
                                    );
                                }
                            }
                            let token_idx = seq.total_tokens().saturating_sub(1);
                            let block_idx_in_seq = token_idx / block_size;
                            if block_idx_in_seq < seq.allocated_block_ids.len() {
                                let physical_block_id = seq.allocated_block_ids[block_idx_in_seq];
                                let offset_in_block = (token_idx % block_size) * 4;
                                let token_bytes = first_token.to_le_bytes();
                                let _ = self.block_manager.write_block_data(
                                    physical_block_id,
                                    offset_in_block,
                                    &token_bytes,
                                );
                            }
                        }

                        if seq.is_finished() {
                            newly_finished.push(seq.clone());
                            let mut deferred = self.deferred_release.lock();
                            deferred
                                .unpin_blocks
                                .extend_from_slice(&seq.allocated_block_ids);
                            deferred.completed_seq_ids.push(seq.id);
                            self.kv_caches.lock().remove(&seq.id);
                        } else {
                            self.running_sequences.lock().push(seq);
                        }
                    }
                }
            }

            StepType::Decode => {
                self.execute_decode(
                    &mut scheduled_seq_ids,
                    &mut preempted_seq_ids,
                    &mut generated,
                    &mut newly_finished,
                )?;
            }

            StepType::Idle => {
                // Standby step
            }
        }

        let step_duration = step_start.elapsed().as_nanos() as u64;

        // Fan out generated tokens and completion events to subscribers
        {
            let mut listeners = self.listeners.lock();
            for (&sid, toks) in &generated {
                if let Some(tx) = listeners.get(&sid) {
                    for &t in toks {
                        let _ = tx.send(SequenceOutputEvent::Token(t));
                    }
                }
            }
            for fin in &newly_finished {
                if let Some(tx) = listeners.remove(&fin.id) {
                    let _ = tx.send(SequenceOutputEvent::Finished(Box::new(fin.clone())));
                }
            }
        }

        Ok(StepOutput {
            step_type,
            scheduled_seq_ids,
            newly_finished_seqs: newly_finished,
            generated_tokens: generated,
            preempted_seq_ids,
            step_duration_ns: step_duration,
        })
    }

    fn execute_decode(
        &self,
        scheduled_seq_ids: &mut Vec<SequenceId>,
        preempted_seq_ids: &mut Vec<SequenceId>,
        generated: &mut HashMap<SequenceId, Vec<u32>>,
        newly_finished: &mut Vec<Sequence>,
    ) -> Result<(), SingularityError> {
        let running = {
            let mut guard = self.running_sequences.lock();
            std::mem::take(&mut *guard)
        };

        let lcfs_res =
            schedule_decode_lcfs(running, self.config.max_batch_size, &self.block_manager)?;

        for victim in lcfs_res.preempted {
            preempted_seq_ids.push(victim.id);
            let mut victim_seq = victim;
            victim_seq.status = SequenceStatus::Preempted;
            // Evict preempted sequence cache from host memory
            self.kv_caches.lock().remove(&victim_seq.id);
            self.preempted_queue.lock().push_back(victim_seq);
        }

        // Sequences not scheduled purely due to max_batch_size limit remain running
        for unalloc in lcfs_res.unallocated {
            self.running_sequences.lock().push(unalloc);
        }

        let mut remaining_running = Vec::new();
        for mut seq in lcfs_res.scheduled {
            scheduled_seq_ids.push(seq.id);

            // Dynamic MTP draft prediction
            let confidence = 0.85; // Simulated confidence
            let draft_tokens = if self.config.enable_mtp {
                self.draft_allocator.allocate_draft_tokens(confidence)
            } else {
                1
            };

            let mut tokens_for_seq = Vec::new();
            let mut cache = {
                let mut caches = self.kv_caches.lock();
                caches
                    .remove(&seq.id)
                    .unwrap_or_else(|| KvCache::new(&self.model.config))
            };

            for _ in 0..draft_tokens {
                let last_tok = seq
                    .output_tokens
                    .last()
                    .copied()
                    .or_else(|| seq.prompt_tokens.last().copied())
                    .unwrap_or(1);
                let pos = seq.total_tokens().saturating_sub(1);
                let logits = self.model.forward_token(last_tok, pos, &mut cache)?;
                let tok = Sampler::sample(&logits, &seq.output_tokens, &seq.sampling_params);

                seq.append_token(tok);
                tokens_for_seq.push(tok);

                // Sync token to device-backed physical block manager if enabled
                if self.block_manager.device_buffer().is_some()
                    && !seq.allocated_block_ids.is_empty()
                {
                    let token_idx = seq.total_tokens().saturating_sub(1);
                    let block_size = self.block_manager.block_size();
                    let block_idx_in_seq = token_idx / block_size;
                    if block_idx_in_seq < seq.allocated_block_ids.len() {
                        let physical_block_id = seq.allocated_block_ids[block_idx_in_seq];
                        let offset_in_block = (token_idx % block_size) * 4;
                        let token_bytes = tok.to_le_bytes();
                        let _ = self.block_manager.write_block_data(
                            physical_block_id,
                            offset_in_block,
                            &token_bytes,
                        );
                    }
                }

                if seq.is_finished() {
                    break;
                }
            }

            generated.insert(seq.id, tokens_for_seq);

            if seq.is_finished() {
                newly_finished.push(seq.clone());
                let mut deferred = self.deferred_release.lock();
                deferred
                    .unpin_blocks
                    .extend_from_slice(&seq.allocated_block_ids);
                deferred.completed_seq_ids.push(seq.id);
            } else {
                self.kv_caches.lock().insert(seq.id, cache);
                remaining_running.push(seq);
            }
        }

        self.running_sequences.lock().extend(remaining_running);
        Ok(())
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.lock().len()
    }

    pub fn running_count(&self) -> usize {
        self.running_sequences.lock().len()
    }

    pub fn preempted_count(&self) -> usize {
        self.preempted_queue.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use singularity_core::SamplingParams;
    use singularity_cuda::MockGpuBackend;

    #[tokio::test]
    async fn test_zos_scheduler_subscription_and_event_delivery() {
        let config = SchedulerConfig::default();
        let block_manager = Arc::new(PagedBlockManager::new(64, 16));
        let radix = Arc::new(RadixTree::new(100));
        let gpu = Arc::new(MockGpuBackend::new_b200());
        let scheduler = ZosScheduler::new(config, block_manager, radix, gpu).unwrap();

        let seq_id = 999;
        let mut rx = scheduler.subscribe(seq_id);

        let params = SamplingParams {
            max_tokens: 2, // Finish after 2 tokens
            ..Default::default()
        };
        let seq = Sequence::new(seq_id, "hello".to_string(), vec![1, 2, 3], params);
        scheduler.submit_sequence(seq).unwrap();

        // Step 1: Prefill (generates 1 token)
        let _ = scheduler.step().unwrap();
        let event1 = rx.recv().await.unwrap();
        match event1 {
            SequenceOutputEvent::Token(tok) => assert_eq!(tok, 158),
            _ => panic!("Expected token event"),
        }

        // Step 2: Decode (generates 2nd token and finishes)
        let _ = scheduler.step().unwrap();
        // May receive token then Finished
        let mut got_finished = false;
        while let Some(ev) = rx.recv().await {
            match ev {
                SequenceOutputEvent::Token(_) => {}
                SequenceOutputEvent::Finished(fin_seq) => {
                    assert_eq!(fin_seq.id, seq_id);
                    got_finished = true;
                    break;
                }
                SequenceOutputEvent::Cancelled => panic!("Unexpected cancel"),
            }
        }
        assert!(got_finished);
    }

    #[test]
    fn test_prefill_starvation_two_phase_fallback_to_decode() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            prefill_attempt_interval: 1,
            prefill_any_attempt_interval: 1,
            ..Default::default()
        };
        // Total 2 blocks of size 4
        let block_manager = Arc::new(PagedBlockManager::new(2, 4));
        let radix = Arc::new(RadixTree::new(100));
        let gpu = Arc::new(MockGpuBackend::new_b200());
        let scheduler = ZosScheduler::new(config, block_manager.clone(), radix, gpu).unwrap();

        // Sequence 1: 1 prompt token, max_tokens: 2
        let seq1 = Sequence::new(
            100,
            "seq1".to_string(),
            vec![1],
            SamplingParams {
                max_tokens: 2,
                ..Default::default()
            },
        );
        scheduler.submit_sequence(seq1).unwrap();
        // Prefill seq1 (uses block 0, has capacity for 4 tokens)
        let _ = scheduler.step().unwrap();
        assert_eq!(block_manager.free_block_count(), 1);

        // Sequence 2: 5 prompt tokens with no shared prefix -> needs 2 blocks (5 / 4 = 2)
        // But only 1 block is free, and 0 prefix matched!
        let seq2 = Sequence::new(
            200,
            "seq2".to_string(),
            vec![20, 21, 22, 23, 24],
            SamplingParams {
                max_tokens: 2,
                ..Default::default()
            },
        );
        scheduler.submit_sequence(seq2).unwrap();

        // Step 2: Prefill attempts to schedule seq2, but seq2 cannot allocate 2 blocks!
        // Two-phase fallback immediately routes to Decode so seq1 is NOT starved!
        let out2 = scheduler.step().unwrap();
        assert!(
            out2.scheduled_seq_ids.contains(&100),
            "Running sequence 100 must decode despite waiting prompt failing prefill"
        );
        assert!(out2.newly_finished_seqs.iter().any(|s| s.id == 100));
    }
}
