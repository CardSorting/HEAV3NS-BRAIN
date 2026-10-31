//! CPU reference coordination for compressed attention across backbone layers.
//! One SharedAttention belongs to one sequence; layers must run in order per token.
use crate::model_error as error;
use crate::{
    v41_attention::LoadedWindowAttention, v41_compressor::Compressor, v41_config::TextConfig,
    v41_indexer::Indexer, v41_quant::fp4_activation_dequant, v41_rope, SafetensorsMmapLoader,
};
use singularity_core::SingularityError;
use std::collections::HashMap;

#[derive(Clone, Default)]
pub struct SharedAttention {
    kv: HashMap<usize, Vec<f32>>,
    keys: HashMap<usize, Vec<f32>>,
    position: Option<usize>,
    last_layer: Option<usize>,
    selection: Option<Vec<Option<usize>>>,
    candidates: Option<Vec<bool>>,
}
impl SharedAttention {
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

pub struct CompressedAttention<'a> {
    pub attention: LoadedWindowAttention<'a>,
    compressor: Option<Compressor<'a>>,
    indexer: Option<Indexer<'a>>,
    layer: usize,
    kv_source: usize,
    ratio: usize,
    dim: usize,
    frequencies: Vec<f32>,
}
impl<'a> CompressedAttention<'a> {
    pub fn reset(&mut self) -> Result<(), SingularityError> {
        self.attention.attention.cache.clear();
        if let Some(compressor) = &mut self.compressor {
            compressor.reset()?;
        }
        Ok(())
    }
    pub(crate) fn checkpoint(&self) -> (crate::v41_sparse::WindowCache, Option<Compressor<'a>>) {
        (
            self.attention.attention.cache.clone(),
            self.compressor.clone(),
        )
    }
    pub(crate) fn restore(
        &mut self,
        state: (crate::v41_sparse::WindowCache, Option<Compressor<'a>>),
    ) {
        self.attention.attention.cache = state.0;
        self.compressor = state.1;
    }
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        config: &TextConfig,
        layer: usize,
    ) -> Result<Self, SingularityError> {
        let ratio = *config
            .compress_ratios
            .get(layer)
            .ok_or_else(|| error("Missing layer compression ratio"))?;
        if layer >= config.num_hidden_layers || ratio == 0 {
            return Err(error("Layer does not use compressed attention"));
        }
        let source =
            crate::v41_config::V41Config::source_for_layer(&config.kv_source_layer_ids, layer)
                .ok_or_else(|| error("Missing compressed KV source"))?;
        if config.compress_ratios.get(source) != Some(&ratio)
            || crate::v41_config::V41Config::source_for_layer(&config.index_source_layer_ids, layer)
                .is_none()
        {
            return Err(error("Invalid compressed source topology"));
        }
        Ok(Self {
            attention: LoadedWindowAttention::load(loader, config, layer)?,
            compressor: if layer == source {
                Some(Compressor::load(loader, config, layer)?)
            } else {
                None
            },
            indexer: if config.index_source_layer_ids.contains(&layer) {
                Some(Indexer::load(loader, config, layer)?)
            } else {
                None
            },
            layer,
            kv_source: source,
            ratio,
            dim: config.head_dim,
            frequencies: v41_rope::frequencies_for_layer(config, layer)?,
        })
    }
    /// Advance one token. On error, shared state, compressor and window cache stay unchanged.
    pub fn forward(
        &mut self,
        input: &[f32],
        shared: &mut SharedAttention,
    ) -> Result<Vec<f32>, SingularityError> {
        let mut next = shared.clone();
        let mut compressor = self.compressor.clone();
        let layer = self.layer;
        let source = self.kv_source;
        let ratio = self.ratio;
        let dim = self.dim;
        let indexer = &self.indexer;
        let freqs = &self.frequencies;
        let output = self
            .attention
            .forward_with_compressed(input, |qr, pos, offset| {
                match next.position {
                    None if pos == 0 => {
                        next.position = Some(pos);
                    }
                    Some(previous) if pos == previous => {}
                    Some(previous) if previous.checked_add(1) == Some(pos) => {
                        next.position = Some(pos);
                        next.last_layer = None;
                        next.selection = None;
                        next.candidates = None;
                    }
                    _ => return Err(error("Shared attention position is out of sequence")),
                }
                if next.last_layer.is_some_and(|last| layer <= last) {
                    return Err(error("Compressed layers must execute in order"));
                }
                if let Some(compressor) = &mut compressor {
                    if compressor.position() != pos {
                        return Err(error("Compressor position mismatch"));
                    }
                    if let Some(latent) = compressor.push(input)? {
                        let start = pos + 1 - ratio;
                        let index = indexer
                            .as_ref()
                            .ok_or_else(|| error("KV source requires indexer"))?;
                        let key = index.key(&latent, start)?;
                        next.keys.entry(source).or_default().extend(key);
                        let rotated = v41_rope::rotate(&latent, freqs, start, false, true)?;
                        next.kv
                            .entry(source)
                            .or_default()
                            .extend(fp4_activation_dequant(&rotated, true)?);
                    }
                }
                let kv = next.kv.get(&source).cloned().unwrap_or_default();
                let keys = next.keys.get(&source).map(Vec::as_slice).unwrap_or(&[]);
                let reachable = (pos + 1) / ratio;
                if kv.len()
                    != reachable
                        .checked_mul(dim)
                        .ok_or_else(|| error("Compressed KV size overflow"))?
                {
                    return Err(error("Compressed KV count mismatch"));
                }
                if let Some(indexer) = indexer {
                    let selection =
                        indexer.select(input, qr, keys, pos, offset, next.candidates.as_deref())?;
                    next.selection = Some(selection.indices);
                    if let Some(candidates) = selection.candidates {
                        next.candidates = Some(candidates);
                    }
                }
                let selected = next
                    .selection
                    .clone()
                    .ok_or_else(|| error("Missing shared index selection"))?;
                next.last_layer = Some(layer);
                Ok((kv, selected))
            })?;
        self.compressor = compressor;
        *shared = next;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_and_follower_share_completed_groups_and_reject_bad_order() {
        let mut cfg: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut cfg.text_config;
        c.hidden_size = 32;
        c.head_dim = 32;
        c.num_attention_heads = 1;
        c.q_lora_rank = 32;
        c.o_lora_rank = 32;
        c.o_groups = 1;
        c.sliding_window = 2;
        c.qk_rope_head_dim = 2;
        c.index_n_heads = 1;
        c.index_head_dim = 32;
        c.index_topk = 1;
        c.candidate_source_layer_id = None;
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut add = |name: String, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = data.len();
            data.extend(bytes);
            header.insert(
                name,
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
            );
        };
        for layer in [2, 3] {
            for matrix in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
                add(
                    format!("layers.{layer}.attn.{matrix}.weight"),
                    "F8_E4M3",
                    vec![32, 32],
                    vec![0x20; 1024],
                );
                add(
                    format!("layers.{layer}.attn.{matrix}.scale"),
                    "F8_E8M0",
                    vec![1, 1],
                    vec![127],
                );
            }
            for norm in ["q_norm", "kv_norm"] {
                add(
                    format!("layers.{layer}.attn.{norm}.weight"),
                    "BF16",
                    vec![32],
                    vec![0x80, 0x3f].repeat(32),
                );
            }
            add(
                format!("layers.{layer}.attn.attn_sink"),
                "F32",
                vec![1],
                vec![0; 4],
            );
        }
        for matrix in ["wkv", "wgate"] {
            add(
                format!("layers.2.attn.compressor.{matrix}.weight"),
                "BF16",
                vec![32, 32],
                vec![0x80, 0x3f].repeat(1024),
            );
        }
        add(
            "layers.2.attn.compressor.norm.weight".into(),
            "BF16",
            vec![32],
            vec![0x80, 0x3f].repeat(32),
        );
        add(
            "layers.2.attn.indexer.wq_b.weight".into(),
            "F8_E4M3",
            vec![32, 32],
            vec![0x20; 1024],
        );
        add(
            "layers.2.attn.indexer.wq_b.scale".into(),
            "F8_E8M0",
            vec![1, 1],
            vec![127],
        );
        add(
            "layers.2.attn.indexer.weights_proj.weight".into(),
            "BF16",
            vec![1, 32],
            vec![0x80, 0x3f].repeat(32),
        );
        add(
            "layers.2.attn.indexer.wk.weight".into(),
            "BF16",
            vec![32, 32],
            vec![0x80, 0x3f].repeat(1024),
        );
        add(
            "layers.2.attn.indexer.k_norm.weight".into(),
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
        let mut source = CompressedAttention::load(&loader, c, 2).unwrap();
        let mut follower = CompressedAttention::load(&loader, c, 3).unwrap();
        let mut shared = SharedAttention::default();
        let x = vec![1.; 32];
        assert!(follower.forward(&x, &mut shared).is_err());
        assert_eq!(follower.attention.attention.cache.position(), 0);
        for pos in 0..4 {
            assert!(source
                .forward(&x, &mut shared)
                .unwrap()
                .iter()
                .all(|x| x.is_finite()));
            assert!(follower.forward(&x, &mut shared).is_ok());
            assert_eq!(shared.kv.get(&2).map_or(0, Vec::len), ((pos + 1) / 2) * 32);
        }
        assert_eq!(source.attention.attention.cache.position(), 4);
        assert_eq!(follower.attention.attention.cache.position(), 4);
    }
}
