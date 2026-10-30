//! Single-token CPU indexer reference for compressed V4.1 attention.
//! Shared index keys and candidate masks are owned by the caller.
use crate::{
    v41_block::RmsNorm,
    v41_compressor::Bf16Matrix,
    v41_config::TextConfig,
    v41_quant::{fp4_activation_dequant, CheckpointMatrix},
    v41_rope,
    v41_sparse::{candidate_blocks, compressed_topk, index_scores_bf16},
    SafetensorsMmapLoader,
};
use half::bf16;
use singularity_core::SingularityError;
fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

pub struct Indexer<'a> {
    q: CheckpointMatrix<'a>,
    weights: Bf16Matrix<'a>,
    key: Option<Bf16Matrix<'a>>,
    key_norm: Option<RmsNorm>,
    heads: usize,
    dim: usize,
    top_k: usize,
    ratio: usize,
    candidate_source: bool,
    uses_candidates: bool,
    block_size: usize,
    top_blocks: usize,
    frequencies: Vec<f32>,
}
pub struct Selection {
    pub indices: Vec<Option<usize>>,
    pub candidates: Option<Vec<bool>>,
    pub scores: Vec<f32>,
}
impl<'a> Indexer<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        config: &TextConfig,
        layer: usize,
    ) -> Result<Self, SingularityError> {
        if layer >= config.num_hidden_layers || !config.index_source_layer_ids.contains(&layer) {
            return Err(error("Layer is not an index source"));
        }
        let ratio = *config
            .compress_ratios
            .get(layer)
            .ok_or_else(|| error("Missing compression ratio"))?;
        let heads = config.index_n_heads;
        let dim = config.index_head_dim;
        let rows = heads
            .checked_mul(dim)
            .ok_or_else(|| error("Indexer shape overflow"))?;
        if ratio == 0 || heads == 0 || dim == 0 || dim % 32 != 0 || config.index_topk == 0 {
            return Err(error("Invalid indexer configuration"));
        }
        let prefix = format!("layers.{layer}.attn.indexer");
        let q = CheckpointMatrix::load(
            loader,
            &format!("{prefix}.wq_b.weight"),
            &format!("{prefix}.wq_b.scale"),
        )?;
        if !matches!(q, CheckpointMatrix::Dense { .. }) || q.shape() != (rows, config.q_lora_rank) {
            return Err(error("Invalid index query projection"));
        }
        let owns_key = config.kv_source_layer_ids.contains(&layer);
        let key = if owns_key {
            Some(Bf16Matrix::load(
                loader,
                &format!("{prefix}.wk.weight"),
                dim,
                config.head_dim,
            )?)
        } else {
            None
        };
        let key_norm = if owns_key {
            Some(RmsNorm::load(
                loader,
                &format!("{prefix}.k_norm.weight"),
                dim,
                config.rms_norm_eps,
            )?)
        } else {
            None
        };
        Ok(Self {
            q,
            weights: Bf16Matrix::load(
                loader,
                &format!("{prefix}.weights_proj.weight"),
                heads,
                config.hidden_size,
            )?,
            key,
            key_norm,
            heads,
            dim,
            top_k: config.index_topk,
            ratio,
            candidate_source: config.candidate_source_layer_id == Some(layer),
            uses_candidates: config
                .candidate_source_layer_id
                .is_some_and(|source| source < layer),
            block_size: config.candidate_block_size,
            top_blocks: config.candidate_topk_blocks,
            frequencies: v41_rope::frequencies_for_layer(config, layer)?,
        })
    }
    /// Produce a key for a just-completed compressed group before its KV is rotated.
    pub fn key(&self, latent: &[f32], group_start: usize) -> Result<Vec<f32>, SingularityError> {
        if group_start % self.ratio != 0 {
            return Err(error("Invalid compressed group position"));
        }
        let projection = self
            .key
            .as_ref()
            .ok_or_else(|| error("Indexer does not own keys"))?
            .project(latent, true)?;
        let normalized = self
            .key_norm
            .as_ref()
            .ok_or_else(|| error("Missing index key norm"))?
            .forward(&projection, true)?;
        fp4_activation_dequant(
            &v41_rope::rotate(&normalized, &self.frequencies, group_start, false, true)?,
            false,
        )
    }
    /// Select positions from the caller's shared keys, which must contain exactly
    /// the groups reachable at `position`. Offset locates them after window KV.
    pub fn select(
        &self,
        input: &[f32],
        qr: &[f32],
        keys: &[f32],
        position: usize,
        offset: usize,
        candidates: Option<&[bool]>,
    ) -> Result<Selection, SingularityError> {
        let reachable = position
            .checked_add(1)
            .ok_or_else(|| error("Indexer position overflow"))?
            / self.ratio;
        if keys.len()
            != reachable
                .checked_mul(self.dim)
                .ok_or_else(|| error("Indexer key size overflow"))?
        {
            return Err(error("Shared index keys do not match position"));
        }
        if self.uses_candidates && candidates.is_none() {
            return Err(error("Missing shared candidates"));
        }
        let query = self.q.matvec_quantized(qr)?;
        let mut quantized = Vec::with_capacity(query.len());
        for head in query.chunks_exact(self.dim) {
            quantized.extend(fp4_activation_dequant(
                &v41_rope::rotate(head, &self.frequencies, position, false, true)?,
                false,
            )?);
        }
        let scale = ((self.dim * self.heads) as f32).sqrt().recip();
        let weights: Vec<f32> = self
            .weights
            .project(input, true)?
            .into_iter()
            .map(|w| bf16::from_f32(w * scale).to_f32())
            .collect();
        let scores = index_scores_bf16(&quantized, keys, &weights, self.dim)?;
        let new_candidates = if self.candidate_source {
            Some(candidate_blocks(
                &scores,
                reachable,
                self.block_size,
                self.top_blocks,
            )?)
        } else {
            None
        };
        let selected = compressed_topk(
            &scores,
            reachable,
            if self.uses_candidates {
                candidates
            } else {
                None
            },
            self.top_k,
            offset,
        )?;
        Ok(Selection {
            indices: selected,
            candidates: new_candidates,
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires SINGULARITY_V41_INDEXER_PROJ, _SCALE and _QUERY checkpoint ranges"]
    fn real_checkpoint_indexer_matches_pytorch() {
        use sha2::{Digest, Sha256};
        #[derive(serde::Deserialize)]
        struct Expected {
            sha256: Vec<String>,
            keys: Vec<Vec<f32>>,
            scores: Vec<f32>,
            selected: Vec<usize>,
        }
        let expected: Expected = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_indexer_real_vectors.json"
        ))
        .unwrap();
        let data: Vec<Vec<u8>> = [
            "SINGULARITY_V41_INDEXER_PROJ",
            "SINGULARITY_V41_INDEXER_SCALE",
            "SINGULARITY_V41_INDEXER_QUERY",
        ]
        .iter()
        .map(|name| std::fs::read(std::env::var(name).unwrap()).unwrap())
        .collect();
        for (data, digest) in data.iter().zip(&expected.sha256) {
            assert_eq!(&format!("{:x}", Sha256::digest(data)), digest);
        }
        let tensors = [
            ("k_norm.weight", "BF16", vec![128], &data[0][..256]),
            (
                "weights_proj.weight",
                "BF16",
                vec![32, 5120],
                &data[0][256..327936],
            ),
            ("wk.weight", "BF16", vec![128, 512], &data[0][327936..]),
            ("wq_b.scale", "F8_E8M0", vec![128, 40], data[1].as_slice()),
            (
                "wq_b.weight",
                "F8_E4M3",
                vec![4096, 1280],
                data[2].as_slice(),
            ),
        ];
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend(bytes);
            header.insert(format!("layers.2.attn.indexer.{name}"),serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, file).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let mut config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        config.text_config.index_topk = 2;
        let indexer = Indexer::load(&loader, &config.text_config, 2).unwrap();
        let mut keys = Vec::new();
        for t in 0..4 {
            let latent: Vec<f32> = (0..512)
                .map(|i| (((i * 17 + t * 19) % 113) as i32 - 56) as f32 / 64.0)
                .collect();
            let key = indexer.key(&latent, 2 * t).unwrap();
            assert_eq!(key.len(), expected.keys[t].len());
            let err: f64 = key
                .iter()
                .zip(&expected.keys[t])
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            let norm: f64 = expected.keys[t].iter().map(|x| (*x as f64).powi(2)).sum();
            let rms = (err / norm).sqrt();
            eprintln!("key {t} relative RMS: {rms:.8}");
            assert!(rms < 0.01);
            keys.extend(key);
        }
        let qr: Vec<f32> = (0..1280)
            .map(|i| (((i * 7) % 103) as i32 - 51) as f32 / 64.0)
            .collect();
        let input: Vec<f32> = (0..5120)
            .map(|i| (((i * 11) % 89) as i32 - 44) as f32 / 64.0)
            .collect();
        let selection = indexer.select(&input, &qr, &keys, 7, 128, None).unwrap();
        assert_eq!(selection.scores.len(), expected.scores.len());
        let err: f64 = selection
            .scores
            .iter()
            .zip(&expected.scores)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum();
        let norm: f64 = expected.scores.iter().map(|x| (*x as f64).powi(2)).sum();
        let rms = (err / norm).sqrt();
        eprintln!("score relative RMS: {rms:.8}");
        assert!(rms < 0.02);
        assert_eq!(
            selection.indices,
            expected
                .selected
                .iter()
                .map(|x| Some(x + 128))
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn checkpoint_indexer_builds_keys_and_selects_causal_groups() {
        let mut cfg: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut cfg.text_config;
        c.hidden_size = 32;
        c.head_dim = 32;
        c.q_lora_rank = 32;
        c.index_n_heads = 1;
        c.index_head_dim = 32;
        c.index_topk = 1;
        c.qk_rope_head_dim = 2;
        c.candidate_source_layer_id = Some(2);
        c.candidate_block_size = 1;
        c.candidate_topk_blocks = 1;
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut add = |name: &str, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = data.len();
            data.extend(bytes);
            header.insert(
                format!("layers.2.attn.indexer.{name}"),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
            );
        };
        add("wq_b.weight", "F8_E4M3", vec![32, 32], vec![0x20; 1024]);
        add("wq_b.scale", "F8_E8M0", vec![1, 1], vec![127]);
        add(
            "weights_proj.weight",
            "BF16",
            vec![1, 32],
            vec![0x80, 0x3f].repeat(32),
        );
        add(
            "wk.weight",
            "BF16",
            vec![32, 32],
            vec![0x80, 0x3f].repeat(1024),
        );
        add(
            "k_norm.weight",
            "BF16",
            vec![32],
            vec![0x80, 0x3f].repeat(32),
        );
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let indexer = Indexer::load(&loader, c, 2).unwrap();
        let input = vec![1.0; 32];
        let key = indexer.key(&input, 0).unwrap();
        assert!(indexer.key(&input, 1).is_err());
        assert!(indexer.select(&input, &input, &key, 0, 2, None).is_err());
        let result = indexer.select(&input, &input, &key, 1, 2, None).unwrap();
        assert_eq!(result.indices, vec![Some(2)]);
        assert_eq!(result.candidates, Some(vec![true]));
    }
}
