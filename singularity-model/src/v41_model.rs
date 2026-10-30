//! Single-sequence text-only V4.1 CPU reference runner. Not a production backend.
use crate::{
    engram::Flash1017HashState,
    v41_attention::LoadedWindowAttention,
    v41_block::Block,
    v41_compressed_attention::{CompressedAttention, SharedAttention},
    v41_compressor::Compressor,
    v41_config::TextConfig,
    v41_engram::Engram,
    v41_io::TokenIo,
    v41_moe::LoadedMoe,
    v41_sparse::WindowCache,
    SafetensorsMmapLoader,
};
use singularity_core::SingularityError;
use std::sync::Arc;
fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

enum Attention<'a> {
    Window(LoadedWindowAttention<'a>),
    Compressed(CompressedAttention<'a>),
}
enum Snapshot<'a> {
    Window(WindowCache),
    Compressed((WindowCache, Option<Compressor<'a>>)),
}
struct Layer<'a> {
    block: Block,
    attention: Attention<'a>,
    moe: LoadedMoe<'a>,
    engram: Option<(usize, Engram<'a>)>,
}

pub struct TextModel<'a> {
    io: TokenIo<'a>,
    layers: Vec<Layer<'a>>,
    hashes: Option<Flash1017HashState>,
    shared: SharedAttention,
    position: usize,
    target_layers: Vec<usize>,
}
pub struct ForwardOutput {
    pub logits: Vec<f32>,
    pub target_hidden: Vec<f32>,
}
impl<'a> TextModel<'a> {
    /// `token_map` must be produced from the pinned checkpoint tokenizer.
    /// Loading all experts and Engram tables requires the complete checkpoint.
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        config: &TextConfig,
        token_map: Option<Arc<[u32]>>,
        pad_token_id: u32,
    ) -> Result<Self, SingularityError> {
        if config
            .dspark_target_layer_ids
            .iter()
            .any(|&layer| layer >= config.num_hidden_layers)
            || config
                .dspark_target_layer_ids
                .windows(2)
                .any(|ids| ids[0] >= ids[1])
        {
            return Err(error("Invalid DSpark target layers"));
        }
        let hashes = if config.engram_layer_ids.is_empty() {
            None
        } else {
            Some(Flash1017HashState::new(
                config,
                token_map.ok_or_else(|| error("Missing Engram tokenizer map"))?,
                pad_token_id,
            )?)
        };
        let io = TokenIo::load(loader, config)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let ratio = *config
                .compress_ratios
                .get(layer)
                .ok_or_else(|| error("Missing compression ratio"))?;
            let attention = if ratio == 0 {
                Attention::Window(LoadedWindowAttention::load(loader, config, layer)?)
            } else {
                Attention::Compressed(CompressedAttention::load(loader, config, layer)?)
            };
            let engram = match config.engram_layer_ids.iter().position(|&id| id == layer) {
                Some(index) => Some((
                    index,
                    Engram::load(loader, &format!("layers.{layer}.engram"), config, layer)?,
                )),
                None => None,
            };
            layers.push(Layer {
                block: Block::load(loader, layer, config)?,
                attention,
                moe: LoadedMoe::load(loader, layer, config)?,
                engram,
            });
        }
        Ok(Self {
            io,
            layers,
            hashes,
            shared: SharedAttention::default(),
            position: 0,
            target_layers: config.dspark_target_layer_ids.clone(),
        })
    }
    pub fn position(&self) -> usize {
        self.position
    }
    /// Prefill a fresh sequence and greedily generate at most `max_new_tokens`.
    /// The returned tokens include EOS if it is selected.
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos: u32,
    ) -> Result<Vec<u32>, SingularityError> {
        if self.position != 0 || prompt.is_empty() {
            return Err(error(
                "Greedy generation requires a fresh sequence and nonempty prompt",
            ));
        }
        let mut logits = Vec::new();
        for &token in prompt {
            logits = self.forward_token(token)?;
        }
        let mut generated = Vec::with_capacity(max_new_tokens);
        for i in 0..max_new_tokens {
            if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
                return Err(error("Invalid generation logits"));
            }
            let index = logits
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(id, _)| id)
                .ok_or_else(|| error("Empty logits"))?;
            let token = u32::try_from(index).map_err(|_| error("Token index exceeds u32"))?;
            generated.push(token);
            if token == eos || i + 1 == max_new_tokens {
                break;
            }
            logits = self.forward_token(token)?;
        }
        Ok(generated)
    }
    pub fn reset(&mut self) -> Result<(), SingularityError> {
        for layer in &mut self.layers {
            match &mut layer.attention {
                Attention::Window(a) => a.attention.cache.clear(),
                Attention::Compressed(a) => a.reset()?,
            }
        }
        if let Some(hashes) = &mut self.hashes {
            hashes.clear();
        }
        self.shared.clear();
        self.position = 0;
        Ok(())
    }
    /// Run one text token and return vocabulary logits. On error all state rolls back.
    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>, SingularityError> {
        Ok(self.forward_token_with_targets(token)?.logits)
    }

    /// Also return concatenated target-layer residual means before attention,
    /// as consumed by the checkpoint's DSpark input projection.
    pub fn forward_token_with_targets(
        &mut self,
        token: u32,
    ) -> Result<ForwardOutput, SingularityError> {
        let next_position = self
            .position
            .checked_add(1)
            .ok_or_else(|| error("Sequence position overflow"))?;
        let snapshots: Vec<_> = self
            .layers
            .iter()
            .map(|layer| match &layer.attention {
                Attention::Window(a) => Snapshot::Window(a.attention.cache.clone()),
                Attention::Compressed(a) => Snapshot::Compressed(a.checkpoint()),
            })
            .collect();
        let old_shared = self.shared.clone();
        let old_hashes = self.hashes.clone();
        let result = (|| {
            let mut state = self.io.embed(token as usize)?;
            let mut target_hidden = Vec::new();
            let hashes = match &mut self.hashes {
                Some(h) => Some(h.push(Some(token))?),
                None => None,
            };
            for (index, layer) in self.layers.iter_mut().enumerate() {
                if let Some((index, engram)) = &layer.engram {
                    let ids = hashes
                        .as_ref()
                        .and_then(|h| h.get(*index))
                        .ok_or_else(|| error("Missing Engram hashes"))?;
                    state.residual = engram.forward(&state.residual, ids, false)?;
                }
                if self.target_layers.binary_search(&index).is_ok() {
                    let streams = state.next_pre.len();
                    if streams == 0 || state.residual.len() % streams != 0 {
                        return Err(error("Invalid target hidden dimensions"));
                    }
                    let hidden = state.residual.len() / streams;
                    for j in 0..hidden {
                        let mean = (0..streams)
                            .map(|s| state.residual[s * hidden + j])
                            .sum::<f32>()
                            / streams as f32;
                        target_hidden.push(half::bf16::from_f32(mean).to_f32());
                    }
                }
                match &mut layer.attention {
                    Attention::Window(attention) => layer
                        .block
                        .forward_window(&mut state, attention, &layer.moe, false)?,
                    Attention::Compressed(attention) => layer.block.forward_compressed(
                        &mut state,
                        attention,
                        &mut self.shared,
                        &layer.moe,
                        false,
                    )?,
                }
            }
            Ok(ForwardOutput {
                logits: self.io.logits(&state)?,
                target_hidden,
            })
        })();
        match result {
            Ok(output) => {
                self.position = next_position;
                Ok(output)
            }
            Err(err) => {
                for (layer, snapshot) in self.layers.iter_mut().zip(snapshots) {
                    match (&mut layer.attention, snapshot) {
                        (Attention::Window(a), Snapshot::Window(cache)) => {
                            a.attention.cache = cache
                        }
                        (Attention::Compressed(a), Snapshot::Compressed(checkpoint)) => {
                            a.restore(checkpoint)
                        }
                        _ => unreachable!(),
                    }
                }
                self.shared = old_shared;
                self.hashes = old_hashes;
                Err(err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_checkpoint_runs_multiple_tokens_and_reset() {
        let mut cfg: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut cfg.text_config;
        c.vocab_size = 2;
        c.hidden_size = 32;
        c.num_hidden_layers = 1;
        c.compress_ratios = vec![0];
        c.engram_layer_ids.clear();
        c.hc_mult = 2;
        c.num_attention_heads = 1;
        c.head_dim = 32;
        c.q_lora_rank = 32;
        c.o_lora_rank = 32;
        c.o_groups = 1;
        c.sliding_window = 2;
        c.qk_rope_head_dim = 2;
        c.n_routed_experts = 1;
        c.num_experts_per_tok = 1;
        c.moe_intermediate_size = 32;
        c.dspark_target_layer_ids = vec![0];
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
        for name in ["embed.weight", "head.weight"] {
            add(
                name.into(),
                "BF16",
                vec![2, 32],
                vec![0x80, 0x3f].repeat(64),
            );
        }
        add(
            "norm.weight".into(),
            "BF16",
            vec![32],
            vec![0x80, 0x3f].repeat(32),
        );
        for hc in ["hc_attn", "hc_ffn"] {
            add(
                format!("layers.0.{hc}_fn"),
                "F32",
                vec![8, 64],
                vec![0; 2048],
            );
            add(format!("layers.0.{hc}_base"), "F32", vec![8], vec![0; 32]);
            add(format!("layers.0.{hc}_scale"), "F32", vec![3], vec![0; 12]);
        }
        for norm in ["attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"] {
            add(
                format!("layers.0.{norm}.weight"),
                "BF16",
                vec![32],
                vec![0x80, 0x3f].repeat(32),
            );
        }
        for name in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
            add(
                format!("layers.0.attn.{name}.weight"),
                "F8_E4M3",
                vec![32, 32],
                vec![0x20; 1024],
            );
            add(
                format!("layers.0.attn.{name}.scale"),
                "F8_E8M0",
                vec![1, 1],
                vec![127],
            );
        }
        add("layers.0.attn.attn_sink".into(), "F32", vec![1], vec![0; 4]);
        add(
            "layers.0.ffn.gate.weight".into(),
            "BF16",
            vec![1, 32],
            vec![0x80, 0x3f].repeat(32),
        );
        add("layers.0.ffn.gate.bias".into(), "F32", vec![1], vec![0; 4]);
        for (name, dtype, cols, value, scale_rows) in [
            ("experts.0", "I8", 16, 0x11, 32),
            ("shared_experts", "F8_E4M3", 32, 0x20, 1),
        ] {
            for mat in ["w1", "w2", "w3"] {
                add(
                    format!("layers.0.ffn.{name}.{mat}.weight"),
                    dtype,
                    vec![32, cols],
                    vec![value; 32 * cols],
                );
                add(
                    format!("layers.0.ffn.{name}.{mat}.scale"),
                    "F8_E8M0",
                    vec![scale_rows, 1],
                    vec![127; scale_rows],
                );
            }
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let mut model = TextModel::load(&loader, c, None, 0).unwrap();
        let first_output = model.forward_token_with_targets(0).unwrap();
        assert_eq!(first_output.target_hidden, vec![1.; 32]);
        let first = first_output.logits;
        assert_eq!(first.len(), 2);
        assert!(first.iter().all(|v| v.is_finite()));
        assert!(model.forward_token(2).is_err());
        assert_eq!(model.position(), 1);
        assert!(model.forward_token(1).is_ok());
        assert_eq!(model.position(), 2);
        model.reset().unwrap();
        assert_eq!(model.position(), 0);
        assert_eq!(model.forward_token(0).unwrap(), first);
        model.reset().unwrap();
        assert_eq!(model.generate_greedy(&[0], 3, 1).unwrap(), vec![0, 0, 0]);
        assert!(model.generate_greedy(&[0], 1, 1).is_err());
        model.reset().unwrap();
        assert!(model.generate_greedy(&[], 1, 1).is_err());
    }
}
