//! Single-token V4.1 attention CPU reference.
use crate::{
    v41_block::RmsNorm,
    v41_quant::{e4m3fn, quantize_activation_fp8, ue8m0, CheckpointMatrix},
    v41_rope,
    v41_sparse::{sparse_attention_kernel_reference, WindowCache},
};
use half::bf16;
use singularity_core::SingularityError;

/// Checkpoint-backed dense projections for an uncompressed backbone layer.
pub struct LoadedWindowAttention<'a> {
    pub attention: WindowAttention,
    q_a: CheckpointMatrix<'a>,
    q_b: CheckpointMatrix<'a>,
    kv: CheckpointMatrix<'a>,
    output_b: CheckpointMatrix<'a>,
    hidden: usize,
    ratio: usize,
}

impl<'a> LoadedWindowAttention<'a> {
    pub fn load(
        loader: &'a crate::SafetensorsMmapLoader,
        config: &crate::v41_config::TextConfig,
        layer: usize,
    ) -> Result<Self, SingularityError> {
        if layer >= config.num_hidden_layers {
            return Err(error("Attention layer out of range"));
        }
        let ratio = *config
            .compress_ratios
            .get(layer)
            .ok_or_else(|| error("Missing compression ratio"))?;
        Self::load_prefix(
            loader,
            config,
            &format!("layers.{layer}.attn"),
            layer,
            ratio,
        )
    }

    pub fn load_draft(
        loader: &'a crate::SafetensorsMmapLoader,
        config: &crate::v41_config::TextConfig,
        stage: usize,
    ) -> Result<Self, SingularityError> {
        if stage >= config.num_nextn_predict_layers {
            return Err(error("Draft stage out of range"));
        }
        let layer = config
            .num_hidden_layers
            .checked_add(stage)
            .ok_or_else(|| error("Draft layer overflow"))?;
        if config.compress_ratios.get(layer) != Some(&0) {
            return Err(error("Draft attention must be uncompressed"));
        }
        Self::load_prefix(loader, config, &format!("mtp.{stage}.attn"), layer, 0)
    }

    fn load_prefix(
        loader: &'a crate::SafetensorsMmapLoader,
        config: &crate::v41_config::TextConfig,
        prefix: &str,
        layer: usize,
        ratio: usize,
    ) -> Result<Self, SingularityError> {
        let heads = config.num_attention_heads;
        let dim = config.head_dim;
        let groups = config.o_groups;
        let rank = config.o_lora_rank;
        let total = heads
            .checked_mul(dim)
            .ok_or_else(|| error("Attention shape overflow"))?;
        if heads == 0
            || dim == 0
            || groups == 0
            || rank == 0
            || heads % groups != 0
            || dim % 32 != 0
            || (total / groups) % 32 != 0
        {
            return Err(error("Invalid attention configuration"));
        }
        let matrix = |name: &str,
                      shape: (usize, usize)|
         -> Result<CheckpointMatrix<'a>, SingularityError> {
            let p = format!("{prefix}.{name}");
            let m = CheckpointMatrix::load(loader, &format!("{p}.weight"), &format!("{p}.scale"))?;
            if m.shape() != shape || !matches!(m, CheckpointMatrix::Dense { .. }) {
                return Err(error(&format!("Invalid attention matrix {p}")));
            }
            Ok(m)
        };
        let q_a = matrix("wq_a", (config.q_lora_rank, config.hidden_size))?;
        let q_b = matrix("wq_b", (total, config.q_lora_rank))?;
        let kv = matrix("wkv", (dim, config.hidden_size))?;
        let output_b = matrix(
            "wo_b",
            (
                config.hidden_size,
                groups
                    .checked_mul(rank)
                    .ok_or_else(|| error("Attention shape overflow"))?,
            ),
        )?;
        let output_a = matrix("wo_a", (groups * rank, total / groups))?;
        let output_a = match output_a {
            CheckpointMatrix::Dense {
                weights,
                scales,
                rows,
                cols,
            } => {
                let blocks = cols.div_ceil(32);
                (0..rows)
                    .flat_map(|row| {
                        (0..cols).map(move |col| {
                            bf16::from_f32(
                                e4m3fn(weights[row * cols + col])
                                    * ue8m0(scales[(row / 32) * blocks + col / 32]),
                            )
                            .to_f32()
                        })
                    })
                    .collect()
            }
            _ => unreachable!(),
        };
        let sink_name = format!("{prefix}.attn_sink");
        let info = loader
            .get_tensor_info(&sink_name)
            .ok_or_else(|| error("Missing attention sink"))?;
        if info.dtype != "F32" || info.shape != [heads] {
            return Err(error("Invalid attention sink"));
        }
        Ok(Self {
            attention: WindowAttention {
                heads,
                head_dim: dim,
                groups,
                output_rank: rank,
                frequencies: v41_rope::frequencies_for_layer(config, layer)?,
                q_norm: RmsNorm::load(
                    loader,
                    &format!("{prefix}.q_norm.weight"),
                    config.q_lora_rank,
                    config.rms_norm_eps,
                )?,
                kv_norm: RmsNorm::load(
                    loader,
                    &format!("{prefix}.kv_norm.weight"),
                    dim,
                    config.rms_norm_eps,
                )?,
                sink: loader.get_tensor_f32(&sink_name)?,
                output_a,
                cache: WindowCache::new(config.sliding_window, dim)?,
            },
            q_a,
            q_b,
            kv,
            output_b,
            hidden: config.hidden_size,
            ratio,
        })
    }

    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, SingularityError> {
        if self.ratio != 0 {
            return Err(error("Compressed attention requires shared KV"));
        }
        self.forward_with_compressed(input, |_, _, _| Ok((Vec::new(), Vec::new())))
    }

    fn projected_kv(&self, input: &[f32], position: usize) -> Result<Vec<f32>, SingularityError> {
        if input.len() != self.hidden {
            return Err(error("Invalid DSpark attention input"));
        }
        let kv = self
            .attention
            .kv_norm
            .forward(&self.kv.matvec_quantized(input)?, true)?;
        quantize_dequantize(&v41_rope::rotate(
            &kv,
            &self.attention.frequencies,
            position,
            false,
            true,
        )?)
    }

    /// Seed or extend the main-token cache without executing draft attention.
    pub fn seed_draft_main(&mut self, input: &[f32]) -> Result<(), SingularityError> {
        if self.ratio != 0 {
            return Err(error("DSpark requires uncompressed attention"));
        }
        let position = self.attention.cache.position();
        let kv = self.projected_kv(input, position)?;
        self.attention.cache.append(&kv)?;
        Ok(())
    }

    /// One main-token update plus a whole bidirectional DSpark draft block.
    /// Draft keys are temporary; only the main-token cache is committed.
    pub fn forward_draft(
        &mut self,
        main: &[f32],
        draft: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, SingularityError> {
        let position = self.attention.cache.position();
        if self.ratio != 0 || position == 0 || draft.is_empty() {
            return Err(error(
                "DSpark requires a seeded main cache and nonempty draft block",
            ));
        }
        let start = position
            .checked_add(1)
            .ok_or_else(|| error("DSpark position overflow"))?;
        let mut next = self.attention.cache.clone();
        next.append(&self.projected_kv(main, position)?)?;
        let offset = next.data().len() / self.attention.head_dim;
        let mut combined = next.data().to_vec();
        for (i, x) in draft.iter().enumerate() {
            combined.extend(
                self.projected_kv(
                    x,
                    start
                        .checked_add(i)
                        .ok_or_else(|| error("DSpark position overflow"))?,
                )?,
            );
        }
        let mut indices: Vec<Option<usize>> = (0..offset.min(start)).map(Some).collect();
        indices.extend((0..draft.len()).map(|i| Some(offset + i)));
        let mut outputs = Vec::with_capacity(draft.len());
        for (i, input) in draft.iter().enumerate() {
            let draft_pos = start
                .checked_add(i)
                .ok_or_else(|| error("DSpark position overflow"))?;
            let qr = self
                .attention
                .q_norm
                .forward(&self.q_a.matvec_quantized(input)?, true)?;
            let query = self.q_b.matvec_quantized(&qr)?;
            let mut q = Vec::with_capacity(query.len());
            for head in query.chunks_exact(self.attention.head_dim) {
                q.extend(v41_rope::rotate(
                    head,
                    &self.attention.frequencies,
                    draft_pos,
                    false,
                    true,
                )?);
            }
            let attended = sparse_attention_kernel_reference(
                &q,
                &combined,
                &self.attention.sink,
                &indices,
                self.attention.head_dim,
                (self.attention.head_dim as f32).sqrt().recip(),
            )?;
            let mut rotated = Vec::with_capacity(attended.len());
            for head in attended.chunks_exact(self.attention.head_dim) {
                rotated.extend(v41_rope::rotate(
                    head,
                    &self.attention.frequencies,
                    draft_pos,
                    true,
                    true,
                )?);
            }
            let group_dim = self.attention.heads * self.attention.head_dim / self.attention.groups;
            let mut low_rank =
                Vec::with_capacity(self.attention.groups * self.attention.output_rank);
            for (group, values) in rotated.chunks_exact(group_dim).enumerate() {
                let values = quantize_dequantize(values)?;
                for row in 0..self.attention.output_rank {
                    let offset = (group * self.attention.output_rank + row) * group_dim;
                    let result = self.attention.output_a[offset..offset + group_dim]
                        .iter()
                        .zip(&values)
                        .map(|(w, x)| w * x)
                        .sum::<f32>();
                    low_rank.push(bf16::from_f32(result).to_f32());
                }
            }
            outputs.push(self.output_b.matvec_quantized(&low_rank)?);
        }
        self.attention.cache = next;
        Ok(outputs)
    }

    /// `compressed` returns latent KV and indices relative to the combined
    /// window and compressed arrays, after receiving this layer's normalized query.
    pub fn forward_with_compressed<F>(
        &mut self,
        input: &[f32],
        compressed: F,
    ) -> Result<Vec<f32>, SingularityError>
    where
        F: FnOnce(&[f32], usize, usize) -> Result<(Vec<f32>, Vec<Option<usize>>), SingularityError>,
    {
        if input.len() != self.hidden {
            return Err(error("Invalid attention input shape"));
        }
        self.attention.forward_with_compressed(
            input,
            |x| self.q_a.matvec_quantized(x),
            |x| self.q_b.matvec_quantized(x),
            |x| self.kv.matvec_quantized(x),
            |x| self.output_b.matvec_quantized(x),
            compressed,
        )
    }
}

fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

/// Emulate the reference's in-place BF16 -> block FP8 -> BF16 activation quantization.
fn quantize_dequantize(input: &[f32]) -> Result<Vec<f32>, SingularityError> {
    let q = quantize_activation_fp8(input)?;
    Ok(q.values
        .iter()
        .enumerate()
        .map(|(i, &v)| bf16::from_f32(e4m3fn(v) * ue8m0(q.scales[i / 32])).to_f32())
        .collect())
}

pub struct WindowAttention {
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub output_rank: usize,
    pub frequencies: Vec<f32>,
    pub q_norm: RmsNorm,
    pub kv_norm: RmsNorm,
    pub sink: Vec<f32>,
    /// Dequantized BF16 weights [groups, output_rank, heads * head_dim / groups].
    pub output_a: Vec<f32>,
    pub cache: WindowCache,
}

impl WindowAttention {
    /// Projections must reproduce checkpoint linear operations and BF16 output boundaries.
    /// The cache advances only if all projections and attention succeed.
    pub fn forward<Qa, Qb, Kv, Ob>(
        &mut self,
        x: &[f32],
        q_a: Qa,
        q_b: Qb,
        kv_proj: Kv,
        output_b: Ob,
    ) -> Result<Vec<f32>, SingularityError>
    where
        Qa: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
        Qb: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
        Kv: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
        Ob: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
    {
        self.forward_with_compressed(x, q_a, q_b, kv_proj, output_b, |_, _, _| {
            Ok((Vec::new(), Vec::new()))
        })
    }

    pub fn forward_with_compressed<Qa, Qb, Kv, Ob, C>(
        &mut self,
        x: &[f32],
        q_a: Qa,
        q_b: Qb,
        kv_proj: Kv,
        output_b: Ob,
        compressed: C,
    ) -> Result<Vec<f32>, SingularityError>
    where
        Qa: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
        Qb: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
        Kv: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
        Ob: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
        C: FnOnce(&[f32], usize, usize) -> Result<(Vec<f32>, Vec<Option<usize>>), SingularityError>,
    {
        let total = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| error("Attention shape overflow"))?;
        if self.heads == 0
            || self.head_dim == 0
            || self.groups == 0
            || self.output_rank == 0
            || self.heads % self.groups != 0
            || self.sink.len() != self.heads
            || total % self.groups != 0
            || self.output_a.len()
                != total
                    .checked_mul(self.output_rank)
                    .ok_or_else(|| error("Attention shape overflow"))?
        {
            return Err(error("Invalid window attention dimensions"));
        }
        let position = self.cache.position();
        let qr = self.q_norm.forward(&q_a(x)?, true)?;
        let q = q_b(&qr)?;
        if q.len() != total {
            return Err(error("Invalid query projection shape"));
        }
        let mut rotated_q = Vec::with_capacity(total);
        for head in q.chunks_exact(self.head_dim) {
            rotated_q.extend(v41_rope::rotate(
                head,
                &self.frequencies,
                position,
                false,
                true,
            )?);
        }
        let kv = self.kv_norm.forward(&kv_proj(x)?, true)?;
        if kv.len() != self.head_dim {
            return Err(error("Invalid KV projection shape"));
        }
        let kv = quantize_dequantize(&v41_rope::rotate(
            &kv,
            &self.frequencies,
            position,
            false,
            true,
        )?)?;
        let mut next_cache = self.cache.clone();
        let mut indices = next_cache.append(&kv)?;
        let offset = next_cache.data().len() / self.head_dim;
        let (extra_kv, extra_indices) = compressed(&qr, position, offset)?;
        if extra_kv.len() % self.head_dim != 0
            || extra_indices
                .iter()
                .flatten()
                .any(|&i| i < offset || i >= offset + extra_kv.len() / self.head_dim)
        {
            return Err(error("Invalid compressed attention data"));
        }
        let mut combined = next_cache.data().to_vec();
        combined.extend(extra_kv);
        indices.extend(extra_indices);
        let attended = sparse_attention_kernel_reference(
            &rotated_q,
            &combined,
            &self.sink,
            &indices,
            self.head_dim,
            (self.head_dim as f32).sqrt().recip(),
        )?;
        let mut output = Vec::with_capacity(total);
        for head in attended.chunks_exact(self.head_dim) {
            output.extend(v41_rope::rotate(
                head,
                &self.frequencies,
                position,
                true,
                true,
            )?);
        }
        let group_dim = total / self.groups;
        let mut low_rank = Vec::with_capacity(self.groups * self.output_rank);
        for (group, values) in output.chunks_exact(group_dim).enumerate() {
            let values = quantize_dequantize(values)?;
            for row in 0..self.output_rank {
                let offset = (group * self.output_rank + row) * group_dim;
                let value = self.output_a[offset..offset + group_dim]
                    .iter()
                    .zip(&values)
                    .map(|(w, x)| w * x)
                    .sum::<f32>();
                low_rank.push(bf16::from_f32(value).to_f32());
            }
        }
        let result = output_b(&low_rank)?;
        if result.len() != x.len() {
            return Err(error("Invalid output projection shape"));
        }
        self.cache = next_cache;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_loader_executes_and_rejects_compressed_layers() {
        let mut config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut config.text_config;
        c.hidden_size = 32;
        c.num_attention_heads = 1;
        c.head_dim = 32;
        c.q_lora_rank = 32;
        c.o_lora_rank = 32;
        c.o_groups = 1;
        c.sliding_window = 2;
        c.qk_rope_head_dim = 2;
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut add = |name: String, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = payload.len();
            payload.extend(bytes);
            header.insert(name, serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}));
        };
        for name in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
            add(
                format!("layers.0.attn.{name}.weight"),
                "F8_E4M3",
                vec![32, 32],
                vec![0x20; 1024],
            );
            add(
                format!("mtp.0.attn.{name}.weight"),
                "F8_E4M3",
                vec![32, 32],
                vec![0x20; 1024],
            );
            add(
                format!("mtp.0.attn.{name}.scale"),
                "F8_E8M0",
                vec![1, 1],
                vec![127],
            );
            add(
                format!("layers.0.attn.{name}.scale"),
                "F8_E8M0",
                vec![1, 1],
                vec![127],
            );
        }
        for name in ["q_norm", "kv_norm"] {
            add(
                format!("layers.0.attn.{name}.weight"),
                "BF16",
                vec![32],
                vec![0x80, 0x3f].repeat(32),
            );
            add(
                format!("mtp.0.attn.{name}.weight"),
                "BF16",
                vec![32],
                vec![0x80, 0x3f].repeat(32),
            );
        }
        add(
            "layers.0.attn.attn_sink".into(),
            "F32",
            vec![1],
            0.0f32.to_le_bytes().to_vec(),
        );
        add("mtp.0.attn.attn_sink".into(), "F32", vec![1], vec![0; 4]);
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, file).unwrap();
        let loader = crate::SafetensorsMmapLoader::open(path).unwrap();
        assert!(LoadedWindowAttention::load(&loader, c, 2).is_err());
        let mut attention = LoadedWindowAttention::load(&loader, c, 0).unwrap();
        let output = attention.forward(&vec![1.0; 32]).unwrap();
        assert_eq!(output.len(), 32);
        assert!(output.iter().all(|x| x.is_finite() && *x > 0.0));
        assert_eq!(attention.attention.cache.position(), 1);

        let mut draft = LoadedWindowAttention::load_draft(&loader, c, 0).unwrap();
        draft.attention.sink[0] = 22.0;
        assert!(draft.forward_draft(&vec![1.; 32], &[vec![1.; 32]]).is_err());
        draft.seed_draft_main(&vec![1.; 32]).unwrap();
        let old = draft.attention.cache.clone();
        assert!(draft.forward_draft(&vec![1.; 32], &[vec![1.; 31]]).is_err());
        assert_eq!(draft.attention.cache.position(), 1);
        let result = draft
            .forward_draft(&vec![1.; 32], &[vec![1.; 32], vec![0.; 32]])
            .unwrap();
        assert_eq!(result.len(), 2);
        #[derive(serde::Deserialize)]
        struct Reference {
            outputs: Vec<Vec<f32>>,
        }
        let reference: Reference = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_dspark_attention_vectors.json"
        ))
        .unwrap();
        for (actual, expected) in result.iter().zip(&reference.outputs) {
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() <= 0.01 * b.abs().max(1e-5), "{a} != {b}");
            }
        }
        assert_eq!(draft.attention.cache.position(), 2);
        draft.attention.cache = old;
        let different = draft
            .forward_draft(&vec![1.; 32], &[vec![1.; 32], vec![1.; 32]])
            .unwrap();
        assert_ne!(result[0], different[0]); // first query can attend to the later draft key
    }
    #[test]
    fn window_attention_advances_and_rolls_back_cache() {
        let mut a = WindowAttention {
            heads: 1,
            head_dim: 32,
            groups: 1,
            output_rank: 1,
            frequencies: vec![1.0],
            q_norm: RmsNorm {
                weight: vec![1.0; 32],
                epsilon: 1e-6,
            },
            kv_norm: RmsNorm {
                weight: vec![1.0; 32],
                epsilon: 1e-6,
            },
            sink: vec![0.0],
            output_a: vec![1.0; 32],
            cache: WindowCache::new(2, 32).unwrap(),
        };
        let x = vec![1.0; 32];
        let copy = |v: &[f32]| Ok(v.to_vec());
        let result = a.forward(&x, copy, copy, copy, |_| Err(error("projection failed")));
        assert!(result.is_err());
        assert_eq!(a.cache.position(), 0);
        for pos in 1..=3 {
            let y = a
                .forward(&x, copy, copy, copy, |v| Ok(vec![v[0]; 32]))
                .unwrap();
            assert_eq!(a.cache.position(), pos);
            assert!(y.iter().all(|v| v.is_finite() && *v > 0.0));
        }
    }
}
