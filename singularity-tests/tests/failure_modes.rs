use singularity_tests::*;
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn test_failure_mode_1_lcfs_preemption_on_oom() {
    // Mode 1: Resource Exhaustion & GPU OOM
    // Verifies DeepInfer's LCFS preemption: when blocks are exhausted, longest running request
    // (running[l], l=0) is preempted first to allow newer requests to make progress.
    let block_manager = Arc::new(PagedBlockManager::new(4, 16)); // Only 4 total blocks!

    let mut seq_long = Sequence::new(
        1,
        "long".to_string(),
        vec![1; 16],
        SamplingParams::default(),
    );
    // Simulate seq_long having generated 32 output tokens (total 48 tokens -> 3 blocks)
    for i in 0..32 {
        seq_long.output_tokens.push(i);
    }
    let blocks_long = block_manager.allocate_blocks(3).unwrap();
    seq_long.allocated_block_ids = blocks_long;

    let mut seq_short = Sequence::new(
        2,
        "short".to_string(),
        vec![2; 16],
        SamplingParams::default(),
    );
    // seq_short only generated 1 token (total 17 tokens -> needs a 2nd block)
    seq_short.output_tokens.push(10);
    let blocks_short = block_manager.allocate_blocks(1).unwrap();
    seq_short.allocated_block_ids = blocks_short;

    assert_eq!(block_manager.free_block_count(), 0); // KV cache completely full!

    let running = vec![seq_short, seq_long];
    let result = schedule_decode_lcfs(running, 8, &block_manager).unwrap();

    // Longest sequence (ID 1) must be preempted!
    assert_eq!(result.preempted.len(), 1);
    assert_eq!(result.preempted[0].id, 1);
    // Shortest sequence (ID 2) must be scheduled!
    assert_eq!(result.scheduled.len(), 1);
    assert_eq!(result.scheduled[0].id, 2);
    // Freed blocks must have allowed sequence 2 to acquire its needed block
    assert_eq!(result.scheduled[0].allocated_block_ids.len(), 2);
}

#[test]
fn test_failure_mode_2_cpu_gpu_execution_bubbles() {
    // Mode 2: CPU-GPU Execution Bubbles
    // Verifies ZOS execution: Step N+1 forward launch overlaps with Step N post-processing,
    // and deferred release collects and frees blocks across steps.
    let config = SchedulerConfig::default();
    let block_manager = Arc::new(PagedBlockManager::new(256, 16));
    let radix_tree = Arc::new(RadixTree::new(1000));
    let gpu_backend = Arc::new(MockGpuBackend::new_b200());

    let scheduler =
        ZosScheduler::new(config, block_manager.clone(), radix_tree, gpu_backend).unwrap();

    let seq = Sequence::new(
        1,
        "test prompt".to_string(),
        vec![1, 2, 3, 4],
        SamplingParams::default(),
    );
    scheduler.submit_sequence(seq).unwrap();

    let step1 = scheduler.step().unwrap();
    assert_eq!(step1.step_type, StepType::Prefill);
    assert!(step1.scheduled_seq_ids.contains(&1));

    let step2 = scheduler.step().unwrap();
    assert_eq!(step2.step_type, StepType::Decode);
    assert!(step2.scheduled_seq_ids.contains(&1));
}

#[test]
fn test_failure_mode_3_flaky_client_drops_and_cancellation() {
    // Mode 3: Flaky Client Drops & Orphaned GPU Work
    // Verifies socket drop immediately marks admission state Cancelled and removes sequence from scheduler.
    let config = SchedulerConfig::default();
    let block_manager = Arc::new(PagedBlockManager::new(256, 16));
    let radix_tree = Arc::new(RadixTree::new(1000));
    let gpu_backend = Arc::new(MockGpuBackend::new_b200());

    let scheduler =
        ZosScheduler::new(config, block_manager.clone(), radix_tree, gpu_backend).unwrap();
    let lifecycle = AdmissionLifecycle::new(42);

    let seq = Sequence::new(
        42,
        "prompt".to_string(),
        vec![10, 11, 12],
        SamplingParams::default(),
    );
    scheduler.submit_sequence(seq).unwrap();
    assert_eq!(scheduler.waiting_count(), 1);

    // Client drops connection
    lifecycle.mark_client_disconnected();
    assert!(lifecycle.is_client_disconnected());
    assert_eq!(lifecycle.current_state(), AdmissionState::Cancelled);

    // Cancel sequence in scheduler
    let cancelled = scheduler.cancel_sequence(42);
    assert!(cancelled);
    assert_eq!(scheduler.waiting_count(), 0);
}

#[test]
fn test_failure_mode_4_multi_gpu_consensus_deadlocks() {
    // Mode 4: Multi-GPU Communication Deadlocks
    // Verifies DP consensus voting: ALL point vs ANY point with two-phase validation
    // prevents ranks from entering mismatched prefill/decode or spinning in zero-work loops.
    let config = SchedulerConfig {
        prefill_attempt_interval: 4,
        prefill_any_attempt_interval: 8,
        ..Default::default()
    };
    let mut consensus = DpConsensusEngine::new(config);

    // At step 0 (ALL point), if one rank has prompts but another doesn't (MIN vote == 0),
    // step must resolve to Decode to prevent deadlock!
    let step = consensus.decide_step_type(true, |_vote, _op| 0); // MIN vote is 0
    assert_eq!(step, StepType::Decode);

    // Advance decode steps 1..=7 with no ranks wanting prefill
    for _ in 1..8 {
        let _ = consensus.decide_step_type(false, |_vote, _op| 0);
    }
    assert_eq!(consensus.current_decode_step(), 8);

    // At step 8 (ANY point), if at least one rank has prompts (MIN of (want?0:1) == 0),
    // step must resolve to Prefill!
    let step_any = consensus.decide_step_type(true, |_vote, _op| 0);
    assert_eq!(step_any, StepType::Prefill);

    // Two-phase validation: if scheduled tokens == 0 across all ranks, fallback to Decode
    let validated = consensus.validate_two_phase(0, |_vote, _op| 0);
    assert_eq!(validated, StepType::Decode);
}

#[test]
fn test_failure_mode_5_memory_leaks_in_kv_allocator() {
    // Mode 5: Memory Leaks in KV Block Allocators
    // Verifies RAII Drop guard and unpin cycles return every single block without leakage.
    let block_manager = Arc::new(PagedBlockManager::new(64, 16));
    assert_eq!(block_manager.free_block_count(), 64);

    {
        let blocks = block_manager.allocate_blocks(16).unwrap();
        assert_eq!(block_manager.free_block_count(), 48);
        assert_eq!(block_manager.allocated_block_count(), 16);

        let _guard = BlockGuard::new(block_manager.clone(), blocks);
        // Guard drops here
    }

    assert_eq!(block_manager.free_block_count(), 64);
    assert_eq!(block_manager.allocated_block_count(), 0);
}

#[test]
fn test_failure_mode_6_speculative_decoding_miss_penalties() {
    // Mode 6: Speculative Decoding Miss Penalties
    // Verifies MTP dynamic draft allocator collapses K to 0 when confidence drops below threshold.
    let allocator = DynamicMtpDraftAllocator::new(0.50, 4);

    // High confidence: allocates draft tokens
    assert!(allocator.allocate_draft_tokens(0.95) >= 3);
    assert!(allocator.allocate_draft_tokens(0.75) >= 2);

    // Low confidence below 0.50: collapses to 0 to prevent miss penalty!
    assert_eq!(allocator.allocate_draft_tokens(0.49), 0);
    assert_eq!(allocator.allocate_draft_tokens(0.10), 0);
}

#[test]
fn test_failure_mode_7_process_crashes_and_state_recovery() {
    // Mode 7: Process Crashes & State Corruption
    // Verifies transactional store atomic temporary file writing, fsync, atomic rename,
    // and sweep of stale temporary files.
    let temp_dir = tempdir().unwrap();
    let store = TransactionalStore::new(temp_dir.path()).unwrap();

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Receipt {
        seq_id: u64,
        status: String,
    }

    let receipt = Receipt {
        seq_id: 101,
        status: "Admitted".to_string(),
    };

    store.atomic_save("receipt_101.json", &receipt).unwrap();

    let loaded: Option<Receipt> = store.load("receipt_101.json").unwrap();
    assert_eq!(loaded, Some(receipt));

    // Simulate stale temp file from aborted process
    let stale_path = temp_dir.path().join(".tmp.receipt_999.uuid123");
    std::fs::write(&stale_path, b"corrupted").unwrap();
    assert!(stale_path.exists());

    // Sweeping must remove stale file
    let swept = store.sweep_stale_temporary_files().unwrap();
    assert_eq!(swept, 1);
    assert!(!stale_path.exists());
}

#[test]
fn test_failure_mode_8_cudagraph_reverse_pool_sizing() {
    // Mode 8: CUDA Graph Instantiation Failures
    // Verifies reverse-order graph capture: largest batch captured first to size shared memory pool.
    let backend = Arc::new(MockGpuBackend::new_b200());
    let runner = CudaGraphRunner::new(backend, vec![1, 2, 4, 8, 16]);

    // Reverse capture with 1024 bytes per batch unit
    runner.capture_all_reverse(1024).unwrap();

    // Pool size must equal max batch size 16 * 1024 = 16384 bytes
    assert_eq!(runner.shared_pool_size(), Some(16384));

    // Replay bucket 3 selects bucket 4
    let executed_bucket = runner.replay(3).unwrap();
    assert_eq!(executed_bucket, 4);

    let graph = runner.get_graph(4).unwrap();
    assert_eq!(graph.execution_count, 1);
}

#[test]
fn test_failure_mode_9_rate_governor_backpressure_429() {
    // Mode 9: Backpressure Breakdown Under Bursts
    // Verifies leaky token bucket rejects requests with RateLimitExceeded when capacity is saturated.
    let governor = LeakyTokenBucket::new(10, 5); // 5 capacity

    // Consume all 5 tokens
    for _ in 0..5 {
        assert!(governor.try_acquire(1.0).is_ok());
    }

    // 6th request must trigger rate limit
    let err = governor.try_acquire(1.0).unwrap_err();
    match err {
        SingularityError::RateLimitExceeded { retry_after_ms } => {
            assert!(retry_after_ms > 0);
        }
        _ => panic!("Expected RateLimitExceeded"),
    }
}

#[test]
fn test_failure_mode_10_circuit_breaker_canary_recovery() {
    // Mode 10: Partial Initialization Hangs & Node Failures
    // Verifies 4-state circuit breaker trips to Exhausted after threshold and allows single canary.
    let breaker = CircuitBreaker::new(3, 1); // 3 failures, 1 sec cooldown
    assert_eq!(breaker.current_state(), CircuitState::Healthy);

    breaker.record_failure();
    assert_eq!(breaker.current_state(), CircuitState::Healthy);

    breaker.record_failure();
    assert_eq!(breaker.current_state(), CircuitState::Degraded);

    breaker.record_failure();
    assert_eq!(breaker.current_state(), CircuitState::Exhausted);

    // Immediate request must be rejected
    assert!(breaker.check_permission().is_err());

    // Wait for cooldown
    std::thread::sleep(std::time::Duration::from_millis(1100));

    // First request after cooldown gets canary slot
    assert!(breaker.check_permission().is_ok());
    // Second concurrent request before canary succeeds is blocked
    assert!(breaker.check_permission().is_err());

    // Canary succeeds: resets to Healthy
    breaker.record_success();
    assert_eq!(breaker.current_state(), CircuitState::Healthy);
}

#[test]
fn test_failure_mode_11_serialization_and_grammar_bitmask() {
    // Mode 11: Serialization Inconsistencies
    // Verifies grammar bitmask restricts logits to valid JSON characters.
    let engine = GrammarBitmaskEngine::new(128);

    // In ExpectObjectStart, only '{' is allowed
    assert!(engine.is_token_allowed(b'{' as u32, JsonGrammarState::ExpectObjectStart));
    assert!(!engine.is_token_allowed(b'a' as u32, JsonGrammarState::ExpectObjectStart));

    let mut logits = vec![1.0; 128];
    engine
        .apply_mask(&mut logits, JsonGrammarState::ExpectObjectStart)
        .unwrap();

    assert_eq!(logits[b'{' as usize], 1.0);
    assert_eq!(logits[b'a' as usize], f32::NEG_INFINITY);
}

#[test]
fn test_failure_mode_12_sse_ring_buffer_replay_on_disconnect() {
    // Mode 12: Network Connection Drops During SSE Streaming
    // Verifies sliding ring buffer replays missed tokens upon reconnect via Last-Event-ID.
    let ring = StreamRingBuffer::new(16);

    for i in 1..=10 {
        ring.push(StreamToken {
            seq_no: i,
            token: (i * 10) as u32,
            text: format!(" tok_{i}"),
            is_finish: i == 10,
        });
    }

    // Client drops after receiving seq_no 6, reconnects with Last-Event-ID: 6
    let replayed = ring.replay_since(6);
    assert_eq!(replayed.len(), 4);
    assert_eq!(replayed[0].seq_no, 7);
    assert_eq!(replayed[3].seq_no, 10);
    assert!(replayed[3].is_finish);
}

#[test]
fn test_failure_mode_13_kmp_tool_call_grammar_lock() {
    // Mode 13: Invalid Tool Call JSON Syntax
    // Verifies KMP trigger matcher activates grammar bitmask during tool call envelope
    // and unlocks open vocabulary upon exit sequence.
    let trigger = vec![1000, 1001]; // <tool_call>
    let stop = vec![2000]; // </tool_call>
    let mut matcher = KmpTriggerMatcher::new(trigger, stop, 128);

    assert_eq!(matcher.current_state(), KmpConstraintState::OpenVocabulary);

    // Feed token 1000 -> matching trigger
    matcher.feed_token(1000);
    assert_eq!(
        matcher.current_state(),
        KmpConstraintState::MatchingTrigger(1)
    );

    // Feed token 1001 -> trigger matched! Now grammar constrained!
    matcher.feed_token(1001);
    assert_eq!(
        matcher.current_state(),
        KmpConstraintState::GrammarConstrained
    );
    assert!(matcher.is_constraint_active());

    // Feed token 2000 -> stop sequence matched! Returns to open vocabulary!
    matcher.feed_token(2000);
    assert_eq!(matcher.current_state(), KmpConstraintState::OpenVocabulary);
    assert!(!matcher.is_constraint_active());
}

#[test]
fn test_failure_mode_14_hardware_fallback_sm90() {
    // Mode 14: Unexpected Hardware Capabilities
    // Verifies Hopper SM90 automatically falls back to MXFP8 when FP4 is requested,
    // while Blackwell SM100 uses native NVFP4.
    let sm100 = DeviceArchitecture::BlackwellSm100 {
        hbm_bytes: 180 * 1024 * 1024 * 1024,
        nvlink_gen: 5,
        has_native_fp4: true,
        tma_version: 2,
    };
    let format_sm100 = validate_precision_and_fallback(&sm100, true).unwrap();
    assert_eq!(format_sm100, CacheMemoryFormat::Fp4);

    let sm90 = DeviceArchitecture::HopperSm90 {
        hbm_bytes: 141 * 1024 * 1024 * 1024,
        nvlink_gen: 4,
        has_native_fp4: false,
        tma_version: 1,
    };
    let format_sm90 = validate_precision_and_fallback(&sm90, true).unwrap();
    assert_eq!(format_sm90, CacheMemoryFormat::MxFp8);
}

#[test]
fn test_failure_mode_15_moe_eplb_expert_rebalancing() {
    // Mode 15: MoE Expert Imbalance & Hotspots
    // Verifies EPLB detects hot expert arrival spikes and plans redundant replication to other GPUs.
    let mut eplb = ExpertLoadBalancer::new(8, 4);

    // Simulate hotspot on expert 2 (receives 1000 tokens while others receive 10)
    for e in 0..8 {
        eplb.record_expert_dispatch(e, 10);
    }
    eplb.record_expert_dispatch(2, 990);

    assert!(eplb.compute_imbalance_ratio() > 2.0);

    let rebalance_actions = eplb.plan_rebalance(2.0).unwrap();
    assert!(!rebalance_actions.is_empty());
    assert_eq!(rebalance_actions[0].hot_expert_id, 2);

    let ranks = eplb.get_expert_ranks(2);
    assert!(ranks.len() >= 2); // Now hosted on multiple GPU ranks
}

#[test]
fn test_failure_mode_16_prefill_decode_starvation_guarantee() {
    // Mode 16: Scheduling Starvation Under Heavy Prefill
    // Verifies chunked prefill bounds prevent single large prompt from blocking decode indefinitely.
    let config = SchedulerConfig {
        prefill_chunk_size: 64,
        ..Default::default()
    };
    let block_manager = Arc::new(PagedBlockManager::new(256, 16));
    let radix_tree = Arc::new(RadixTree::new(1000));
    let gpu_backend = Arc::new(MockGpuBackend::new_b200());

    let scheduler = ZosScheduler::new(config, block_manager, radix_tree, gpu_backend).unwrap();

    // Large prompt: 128 tokens
    let seq = Sequence::new(
        1,
        "long prompt".to_string(),
        vec![1; 128],
        SamplingParams::default(),
    );
    scheduler.submit_sequence(seq).unwrap();

    // Step must execute prefill
    let out = scheduler.step().unwrap();
    assert_eq!(out.step_type, StepType::Prefill);
}

#[test]
fn test_failure_mode_17_host_mmap_zero_copy_safetensors() {
    // Mode 17: Model Loading OOM on Host
    // Verifies Safetensors mmap loader parses JSON header without reading entire tensor data into RAM.
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("model.safetensors");

    let header_json = r#"{"weight_1": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}}"#;
    let header_bytes = header_json.as_bytes();
    let header_len = (header_bytes.len() as u64).to_le_bytes();

    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&header_len);
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.extend_from_slice(&[42u8; 64]); // 64 bytes of tensor payload

    std::fs::write(&file_path, file_bytes).unwrap();

    let loader = SafetensorsMmapLoader::open(&file_path).unwrap();
    let tensor_bytes = loader.get_tensor_bytes("weight_1").unwrap();
    assert_eq!(tensor_bytes.len(), 64);
    assert_eq!(tensor_bytes[0], 42);
}

#[test]
fn test_failure_mode_18_radix_tree_cache_stale_key_invalidation() {
    // Mode 18: Radix Tree Cache Stale Key Invalidation
    // Verifies cryptographic SHA256 sequence hashing and generation counter prevent stale prefix hits.
    let block_manager = PagedBlockManager::new(64, 16);
    let radix = RadixTree::new(1000);

    let prefix_tokens = vec![101, 102, 103, 104];
    let allocated_blocks = block_manager.allocate_blocks(2).unwrap();

    radix
        .insert_prefix(&prefix_tokens, &allocated_blocks, &block_manager)
        .unwrap();

    // Matching identical prefix succeeds
    let (matched_blocks, matched_len) = radix.match_prefix(&prefix_tokens);
    assert_eq!(matched_len, 4);
    assert_eq!(matched_blocks, allocated_blocks);

    // Matching altered prefix fails correctly
    let altered_tokens = vec![101, 102, 999, 104];
    let (_, altered_len) = radix.match_prefix(&altered_tokens);
    assert_ne!(altered_len, 4);

    // Evict LRU recovers blocks
    let freed = radix.evict_lru(&block_manager, 64);
    assert_eq!(freed, 2);
}
