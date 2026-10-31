# ADR-004: ZOS, LCFS, FlashMLA-4, and DSpark Runtime Integration Triad

- **Status**: Accepted
- **Date**: 2026-10-31
- **Components**: `singularity-cuda`, `singularity-cache`, `singularity-scheduler`, `singularity-model`

---

## 1. Context

DeepSeek-V4.1-Flash-1017 combines 40 transformer layers (`40` MoE layers with `n_routed_experts = 384`, `num_experts_per_tok = 6`, `n_shared_experts = 1`, and `sqrtsoftplus` routing), Multi-head Latent Attention (`q_lora_rank = 2048`, `head_dim = 512`, `qk_rope_head_dim = 64`, `sliding_window = 512`), conditional Engram memory (`engram_layer_ids = [1, 14]`), and a built-in lightweight speculative drafter (`DSpark`, `num_nextn_predict_layers = 4`) that consumes hidden states from `dspark_target_layer_ids = [37, 38, 39]`.

Running this architecture at low latency and high throughput requires tightly coordinated scheduling, memory management, and speculative verification.

## 2. Decision

1. **Zero-Overhead Scheduler (`ZOS`, `singularity-scheduler`)**:
   - Interleaves chunked prefill (`prefill_chunk_size`) and active decode steps in a single cooperative loop (`step_cooperative`), preventing long prompts from starving ongoing token streams.
   - Employs Last-Come-First-Served (`LCFS`) preemption on KV block exhaustion so younger sequences with minimal invested compute yield blocks before older sequences near completion.
2. **Paged Block Manager + BLAKE3 Content-Addressed Prefix Cache (`singularity-cache`)**:
   - Manages physical KV blocks via an $O(1)$ bitset allocator (`leading_zeros` / `trailing_zeros`) and caches shared system/tool prefixes in a radix tree keyed by BLAKE3 token-chunk hashes (`PinnedNodeGuard` prevents eviction of active prefixes).
3. **FlashMLA-4 Latent KV Compression (`singularity-model::v41_attention`)**:
   - Stores only the compressed latent KV vector (`head_dim = 512`) across shared window/compressed caches instead of full per-head uncompressed KV tensors (`64 × 512 × 2 = 65,536` elements per token per layer), reducing KV cache memory footprint by over **92%**.
4. **DSpark Speculative Verification (`singularity-model::v41_dspark`)**:
   - Uses `TextModel::prefill_token_with_targets` on prompt tokens `0..P-1` to populate the KV cache and extract target hidden states without computing the `129,280 × 5,120` output vocabulary head (`1.32 GiB` saved per prompt token), then drafts and verifies speculative continuations with rollback-safe `ModelCache` / `DraftModel` snapshots.

## 3. Impact on the Optimal Runtime Experience (ORE)

- **Performance**: Over `92%` KV cache compression enables 8× higher concurrent sequence capacity in HBM3e.
- **Speed**: DSpark prefill head-elision + speculative verification cuts both TTFT and TPOT without altering greedy output distributions.
- **Reliability**: Rollback-safe cache transactions (`ModelCache` + `DraftModel`) guarantee state consistency if any layer or draft step fails.

