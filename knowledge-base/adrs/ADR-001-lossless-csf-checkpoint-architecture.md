# ADR-001: Native Lossless CSF Checkpoint Architecture (`lil-mxfp4-csf-checkpoint/1`)

- **Status**: Accepted & Deployed in Production (`im-R1TdH6JxBrlEvExHAIqaph`)
- **Date**: 2026-10-31
- **Components**: [`singularity-model/src/safetensors.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/safetensors.rs), [`deploy/modal/model_storage.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/model_storage.py), [`deploy/modal/config.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/config.py)

---

## 1. Context

The canonical `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`1bae0e1088993b9b82de08f59de5be5a34e42170`) checkpoint comprises 48 SafeTensors shards containing `96,085` logical tensors (`510,286,023,000` bytes, `~510.29 GB`), of which `46,080` tensors (`40` MoE layers × `384` routed experts × `3` matrices `w1`/`w2`/`w3`) are MXFP4 (`E2M1`) expert weight matrices paired with 8-bit `E8M0` block scales (`.scale`).

The `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (`90d665c19e1a9ff2097cc787515e370d86b616f0`) repository encodes those `46,080` MXFP4 scale matrices using the **`lil-mxfp4-csf-checkpoint/1`** schema and **`row-base-offset1-u24-exceptions/1`** codec:
- Each physical shard stores `.scale.mxfp4_csf_fixed` (`U8`, 16-row blocks of 1-byte row bases followed by 1-byte 8-column bitmasks) and `.scale.mxfp4_csf_exceptions` (`U32` little-endian `flat_index | (true_scale_u8 << 24)`), resulting in `142,165` raw SafeTensors header entries across the 48 shards (`96,085` logical tensors).
- Total checkpoint footprint shrinks by **`14.65 GB`** (`510,286,023,000` bytes / `510.29 GB` $\rightarrow$ `495,632,175,852` bytes / `495.63 GB`) with **100% bit-exact lossless reconstruction** verified against `model.safetensors.index.json` (`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`) and per-shard SHA-256 receipts (`receipts/model-000XX-of-00048.safetensors.json`).
- Furthermore, the CSF repository organizes files under `metadata/`, `tensors/`, and `receipts/` subdirectories rather than a flat directory.

## 2. Considered Alternatives

1. **Offline Full Shard Rewriting Before Serving**:
   - Reconstruct and rewrite all 48 shards (`510.29 GB`) to uncompressed `.safetensors` files on the Modal Volume during hydration.
   - *Rejected*: Doubles Modal Volume storage consumption (`~1.0 TB`), adds 20+ minutes of disk I/O during hydration, and forfeits the `14.65 GB` page-cache and I/O reduction at container startup.
2. **Native On-Demand CSF Decode in `SafetensorsMmapLoader` + Zero-Copy Symlink Normalization**:
   - Keep the 48 CSF shards untouched on disk, link `metadata/` and `tensors/` into the root checkpoint directory via zero-copy relative symlinks (`prepare_csf_checkpoint_directory`), and decode `.scale` tensors on demand inside `SafetensorsMmapLoader` with `OnceLock<Arc<[u8]>>` caching.

## 3. Decision

We adopted **Alternative 2**:
- [`singularity-model/src/safetensors.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/safetensors.rs) natively detects `<name>.mxfp4_csf_fixed` + `<name>.mxfp4_csf_exceptions` pairs when parsing SafeTensors headers, synthesizes the logical `<name>` (`F8_E8M0`, `[rows, cols_div_32]`) entry in `TensorLocation`, and reconstructs the exact E8M0 scale bytes on first access via `decode_mxfp4_csf_scale`.
- [`deploy/modal/model_storage.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/model_storage.py) provides `prepare_csf_checkpoint_directory`, `decode_mxfp4_csf_scale`, and an optional offline `reconstruct_csf_shard` utility that verifies SHA-256 digests against `receipts/*.json`.
- [`singularity-server/src/v41_engine.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/v41_engine.rs) routes both `deepseek-ai/DeepSeek-V4.1-Flash-1017` and `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` model identifiers seamlessly.

## 4. Impact on the Optimal Runtime Experience (ORE)

- **Performance**: Saves `14.65 GB` of network/volume I/O and host page-cache pressure while keeping memory-mapped weight access zero-copy.
- **Speed**: Lazy `OnceLock` scale materialization ensures only accessed layers/experts decode their CSF scale blocks during bounded slice serving.
- **Reliability**: Bit-exact SHA-256 parity with the original `deepseek-ai/DeepSeek-V4.1-Flash-1017` E8M0 scales across all `46,080` compressed scale tensors (`96,085` logical tensors total).

## 5. Verification

- Rust unit test: `singularity-model::safetensors::tests::reconstructs_lossless_csf_mxfp4_scales`
- Python unit test: `deploy/modal/test_modal_pipeline.py::ModalPipelineTests::test_csf_codec_reconstruct_and_checkpoint_preparation`

