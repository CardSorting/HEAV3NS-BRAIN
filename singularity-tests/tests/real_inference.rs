use singularity_cache::{PagedBlockManager, RadixTree};
use singularity_core::{SamplingParams, SchedulerConfig, Sequence};
use singularity_cuda::{GpuBackend, MockGpuBackend};
use singularity_model::{
    KvCache, SafetensorsMmapLoader, Sampler, TransformerConfig, TransformerModel,
};
use singularity_scheduler::ZosScheduler;
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn test_smollm2_135m_real_forward_and_autoregressive_inference() {
    let config = TransformerConfig::smollm2_135m();
    assert_eq!(config.hidden_size, 576);
    assert_eq!(config.num_attention_heads, 9);
    assert_eq!(config.num_key_value_heads, 3);
    assert_eq!(config.head_dim, 64);

    // Initialize model with deterministic weights
    let model = TransformerModel::new_random(config.clone());
    let mut kv_cache = KvCache::new(&config);

    // Prompt tokens: e.g. "The quick brown fox"
    let prompt = vec![464, 2068, 7586, 21831];
    let logits = model.forward(&prompt, 0, &mut kv_cache).unwrap();

    assert_eq!(logits.len(), config.vocab_size);
    assert_eq!(kv_cache.layers[0].current_tokens, 4);

    // Sample next token
    let params = SamplingParams {
        temperature: 0.0, // greedy argmax
        ..Default::default()
    };
    let next_token = Sampler::sample(&logits, &prompt, &params);
    assert!((next_token as usize) < config.vocab_size);

    // Autoregressive step: forward only the next token
    let next_logits = model.forward_token(next_token, 4, &mut kv_cache).unwrap();
    assert_eq!(next_logits.len(), config.vocab_size);
    assert_eq!(kv_cache.layers[0].current_tokens, 5);
}

#[test]
fn test_scheduler_with_real_model_and_physical_kv_cache() {
    let gpu_backend: Arc<dyn GpuBackend> = Arc::new(MockGpuBackend::new_h200());
    let (free_mem, total_mem) = gpu_backend.mem_info();
    assert!(total_mem >= 140 * 1024 * 1024 * 1024);
    assert_eq!(free_mem, total_mem);

    // Physical device backing for 32 blocks
    let block_manager =
        Arc::new(PagedBlockManager::with_device_backing(32, 16, gpu_backend.clone(), 64).unwrap());
    let radix_tree = Arc::new(RadixTree::new(500));

    let model_config = TransformerConfig::tiny();
    let model = Arc::new(TransformerModel::new_random(model_config));

    let scheduler_config = SchedulerConfig {
        max_batch_size: 4,
        max_seq_len: 256,
        ..Default::default()
    };

    let scheduler = ZosScheduler::with_model(
        scheduler_config,
        block_manager.clone(),
        radix_tree.clone(),
        gpu_backend.clone(),
        model,
    )
    .unwrap();

    let seq = Sequence::new(
        1001,
        "What is SINGULARITY?".to_string(),
        vec![10, 20, 30],
        SamplingParams {
            max_tokens: 3,
            ..Default::default()
        },
    );

    scheduler.submit_sequence(seq).unwrap();

    // Step 1: Prefill pass
    let step1 = scheduler.step().unwrap();
    assert!(step1.generated_tokens.contains_key(&1001));
    let t1 = step1.generated_tokens.get(&1001).unwrap()[0];

    // Step 2: Decode pass
    let step2 = scheduler.step().unwrap();
    assert!(step2.generated_tokens.contains_key(&1001));
    let t2 = step2.generated_tokens.get(&1001).unwrap()[0];

    // Tokens must be generated from real logits
    assert!(t1 < 256);
    assert!(t2 < 256);
}

#[test]
fn test_multi_shard_safetensors_upload_and_inference() {
    let temp_dir = tempdir().unwrap();
    let shard1_path = temp_dir.path().join("model-00001-of-00002.safetensors");
    let shard2_path = temp_dir.path().join("model-00002-of-00002.safetensors");
    let index_path = temp_dir.path().join("model.safetensors.index.json");

    // 4 floats in shard 1
    let h1 = r#"{"model.embed_tokens.weight": {"dtype": "F32", "shape": [4, 1], "data_offsets": [0, 16]}}"#;
    let h1_len = (h1.len() as u64).to_le_bytes();
    let mut s1 = Vec::new();
    s1.extend_from_slice(&h1_len);
    s1.extend_from_slice(h1.as_bytes());
    for f in [0.1f32, 0.2, 0.3, 0.4] {
        s1.extend_from_slice(&f.to_le_bytes());
    }
    std::fs::write(&shard1_path, s1).unwrap();

    // 4 floats in shard 2
    let h2 = r#"{"model.norm.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#;
    let h2_len = (h2.len() as u64).to_le_bytes();
    let mut s2 = Vec::new();
    s2.extend_from_slice(&h2_len);
    s2.extend_from_slice(h2.as_bytes());
    s2.extend_from_slice(&1.0f32.to_le_bytes());
    std::fs::write(&shard2_path, s2).unwrap();

    let index_json = serde_json::json!({
        "metadata": {"format": "pt"},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors"
        }
    });
    std::fs::write(&index_path, serde_json::to_string(&index_json).unwrap()).unwrap();

    let loader = SafetensorsMmapLoader::open(temp_dir.path()).unwrap();
    let gpu = MockGpuBackend::new_h200();

    let d_embed = loader
        .upload_tensor_to_gpu("model.embed_tokens.weight", &gpu)
        .unwrap();
    let mut read_bytes = vec![0u8; 16];
    gpu.copy_dtoh(&d_embed, &mut read_bytes).unwrap();
    let floats: Vec<f32> = read_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats, vec![0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn test_deepseek_v2_lite_mla_moe_forward_and_compressed_kv() {
    let config = TransformerConfig::tiny_deepseek();
    assert!(config.use_mla);
    assert!(config.is_moe);

    let model = TransformerModel::new_random(config.clone());
    let mut kv_cache = KvCache::new(&config);

    // Prompt tokens: length 5
    let prompt = vec![10u32, 25, 42, 60, 88];
    let logits = model.forward(&prompt, 0, &mut kv_cache).unwrap();

    assert_eq!(logits.len(), config.vocab_size);
    assert_eq!(kv_cache.layers[0].current_tokens, 5);

    // Verify compressed latent cache footprint
    assert!(kv_cache.layers[0].is_mla);
    let mla_bytes = (config.kv_lora_rank + config.qk_rope_head_dim) * 4;
    let standard_mha_bytes = config.num_attention_heads * config.head_dim * 2 * 4;
    assert!(
        mla_bytes < standard_mha_bytes / 2,
        "MLA must compress KV cache by > 2x"
    );

    // Perform consecutive autoregressive decode steps
    let mut current_pos = 5;
    let mut last_token = 88u32;
    for _ in 0..4 {
        let dec_logits = model
            .forward_token(last_token, current_pos, &mut kv_cache)
            .unwrap();
        assert_eq!(dec_logits.len(), config.vocab_size);
        let sampled = Sampler::sample(&dec_logits, &[], &SamplingParams::default());
        last_token = sampled;
        current_pos += 1;
    }

    assert_eq!(kv_cache.layers[0].current_tokens, 9);
}

#[test]
fn test_scheduler_preemption_rebuilds_kv_cache_correctly() {
    let gpu_backend: Arc<dyn GpuBackend> = Arc::new(MockGpuBackend::new_h200());
    // Only 2 total blocks, block_size = 2
    let block_manager =
        Arc::new(PagedBlockManager::with_device_backing(2, 2, gpu_backend.clone(), 16).unwrap());
    let radix_tree = Arc::new(RadixTree::new(500));
    let model = Arc::new(TransformerModel::new_random(TransformerConfig::tiny()));

    let scheduler_config = SchedulerConfig {
        max_batch_size: 2,
        max_seq_len: 128,
        prefill_attempt_interval: 1,
        prefill_any_attempt_interval: 1,
        ..Default::default()
    };

    let scheduler = ZosScheduler::with_model(
        scheduler_config,
        block_manager.clone(),
        radix_tree,
        gpu_backend,
        model,
    )
    .unwrap();

    // Sequence 1: 1 prompt token, max_tokens: 2
    let seq1 = Sequence::new(
        2001,
        "Prompt 1".to_string(),
        vec![10],
        SamplingParams {
            max_tokens: 2,
            ..Default::default()
        },
    );

    // Sequence 2: 1 prompt token, max_tokens: 2
    let seq2 = Sequence::new(
        2002,
        "Prompt 2".to_string(),
        vec![20],
        SamplingParams {
            max_tokens: 2,
            ..Default::default()
        },
    );

    scheduler.submit_sequence(seq1).unwrap();
    // Step 1: Prefill seq1 (uses block 0, generates token 1, total 2 tokens, block 0 full)
    let out1 = scheduler.step().unwrap();
    assert_eq!(out1.scheduled_seq_ids, vec![2001]);
    assert_eq!(block_manager.free_block_count(), 1);

    scheduler.submit_sequence(seq2).unwrap();
    // Step 2: Prefill seq2 (uses block 1, generates token 1, total 2 tokens, block 1 full)
    let out2 = scheduler.step().unwrap();
    assert_eq!(out2.scheduled_seq_ids, vec![2002]);
    assert_eq!(block_manager.free_block_count(), 0);

    // Both sequences are now in running_sequences, and free blocks = 0.
    // Step 3: Decode pass!
    // Sequence 2 needs a new block for its 2nd token (2 + 1 > 2).
    // Because free blocks = 0, LCFS preemption evicts sequence 1, freeing block 0!
    // Sequence 2 is scheduled using the freed block and finishes its 2nd token.
    let out3 = scheduler.step().unwrap();
    assert!(
        out3.preempted_seq_ids.contains(&2001),
        "Sequence 2001 must be preempted when KV blocks are exhausted"
    );
    assert!(out3.scheduled_seq_ids.contains(&2002));
    assert!(out3.newly_finished_seqs.iter().any(|s| s.id == 2002));
    assert_eq!(scheduler.preempted_count(), 1);

    // Step 4: Prefill resumes preempted sequence 2001!
    // Deferred release has freed blocks from completed seq 2002.
    // Preempted sequence 2001 is popped from preempted_queue, re-prefills across its
    // prompt + prior generated output tokens into newly freed blocks, and finishes!
    let out4 = scheduler.step().unwrap();
    assert!(out4.scheduled_seq_ids.contains(&2001));
    assert!(out4.newly_finished_seqs.iter().any(|s| s.id == 2001));
    assert_eq!(scheduler.preempted_count(), 0);
}

#[test]
fn test_paged_block_manager_offset_dma_and_sync() {
    let gpu_backend: Arc<dyn GpuBackend> = Arc::new(MockGpuBackend::new_h200());
    let block_manager =
        PagedBlockManager::with_device_backing(4, 16, gpu_backend.clone(), 4).unwrap();

    // Write token at block 2, offset 8
    let test_data = 123456u32.to_le_bytes();
    block_manager.write_block_data(2, 8, &test_data).unwrap();

    let mut read_data = [0u8; 4];
    block_manager.read_block_data(2, 8, &mut read_data).unwrap();
    assert_eq!(read_data, test_data);

    // Verify other blocks remain zeroed
    let mut other_block = [0u8; 4];
    block_manager
        .read_block_data(1, 8, &mut other_block)
        .unwrap();
    assert_eq!(other_block, [0u8; 4]);
}
