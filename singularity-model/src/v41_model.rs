//! Single-sequence V4.1 CPU reference runner (text and multimodal). Not a production GPU backend.
use crate::model_error as error;
use crate::{
    engram::Flash1017HashState,
    v41_attention::LoadedWindowAttention,
    v41_block::Block,
    v41_compressed_attention::{CompressedAttention, SharedAttention},
    v41_compressor::Compressor,
    v41_config::{TextConfig, V41Config, VisionConfig},
    v41_engram::Engram,
    v41_hyper::HcState,
    v41_io::TokenIo,
    v41_moe::LoadedMoe,
    v41_sparse::WindowCache,
    v41_vision::{expand_image_tokens, prepare_image_patches, Aligner, TokenType},
    v41_vit::VisionEncoder,
    SafetensorsMmapLoader,
};
use singularity_core::SingularityError;
use std::sync::Arc;

#[allow(clippy::large_enum_variant)]
enum Attention<'a> {
    Window(LoadedWindowAttention<'a>),
    Compressed(CompressedAttention<'a>),
}
#[allow(clippy::large_enum_variant)]
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
    pub fn hidden_size(&self) -> usize {
        self.io.hidden_size()
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
        for &token in &prompt[..prompt.len() - 1] {
            self.prefill_token(token)?;
        }
        let mut logits = self.forward_token(prompt[prompt.len() - 1])?;
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
    /// Advance the KV cache and state for one prompt token without computing vocabulary logits.
    pub fn prefill_token(&mut self, token: u32) -> Result<(), SingularityError> {
        let state = self.io.embed(token as usize)?;
        self.forward_state_with_targets_impl(state, Some(token), false, false)
            .map(|_| ())
    }
    /// Advance the KV cache and return target-layer residual means without computing vocabulary logits.
    pub fn prefill_token_with_targets(&mut self, token: u32) -> Result<Vec<f32>, SingularityError> {
        let state = self.io.embed(token as usize)?;
        Ok(self
            .forward_state_with_targets_impl(state, Some(token), false, false)?
            .target_hidden)
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
        let state = self.io.embed(token as usize)?;
        self.forward_state_with_targets(state, Some(token), false)
    }

    /// Run a custom embedding token (such as an aligned vision token) through the backbone.
    /// When `image_token` is true, Engram gating is zeroed and MoE routing uses `bias_vl`.
    pub fn forward_embedding_with_targets(
        &mut self,
        embedding: &[f32],
        hash_token: Option<u32>,
        image_token: bool,
    ) -> Result<ForwardOutput, SingularityError> {
        let state = self.io.embed_custom(embedding)?;
        self.forward_state_with_targets(state, hash_token, image_token)
    }

    fn forward_state_with_targets(
        &mut self,
        state: HcState,
        hash_token: Option<u32>,
        image_token: bool,
    ) -> Result<ForwardOutput, SingularityError> {
        self.forward_state_with_targets_impl(state, hash_token, image_token, true)
    }

    fn forward_state_with_targets_impl(
        &mut self,
        mut state: HcState,
        hash_token: Option<u32>,
        image_token: bool,
        compute_logits: bool,
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
            let mut target_hidden = Vec::new();
            let hashes = match &mut self.hashes {
                Some(h) => Some(h.push(hash_token)?),
                None => None,
            };
            for (index, layer) in self.layers.iter_mut().enumerate() {
                if let Some((index, engram)) = &layer.engram {
                    let ids = hashes
                        .as_ref()
                        .and_then(|h| h.get(*index))
                        .ok_or_else(|| error("Missing Engram hashes"))?;
                    state.residual = engram.forward(&state.residual, ids, image_token)?;
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
                    Attention::Window(attention) => layer.block.forward_window(
                        &mut state,
                        attention,
                        &layer.moe,
                        image_token,
                    )?,
                    Attention::Compressed(attention) => layer.block.forward_compressed(
                        &mut state,
                        attention,
                        &mut self.shared,
                        &layer.moe,
                        image_token,
                    )?,
                }
            }
            let logits = if compute_logits {
                self.io.logits(&state)?
            } else {
                Vec::new()
            };
            Ok(ForwardOutput {
                logits,
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

/// Raw RGB8 image input (`width * height * 3` row-major HWC bytes) for V4.1 multimodal execution.
#[derive(Debug, Clone)]
pub struct MultimodalImageInput {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Combined V4.1 multimodal runner integrating VisionEncoder, Aligner, image token expansion, and TextModel.
pub struct MultimodalModel<'a> {
    pub text: TextModel<'a>,
    pub vision: VisionEncoder<'a>,
    pub aligner: Aligner<'a>,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
}

impl<'a> MultimodalModel<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        config: &V41Config,
        token_map: Option<Arc<[u32]>>,
        pad_token_id: u32,
    ) -> Result<Self, SingularityError> {
        config.validate()?;
        Ok(Self {
            text: TextModel::load(loader, &config.text_config, token_map, pad_token_id)?,
            vision: VisionEncoder::load(loader, &config.vision_config)?,
            aligner: Aligner::load(loader, &config.vision_config, &config.text_config)?,
            vision_config: config.vision_config.clone(),
            image_token_id: config.image_token_id,
        })
    }

    pub fn position(&self) -> usize {
        self.text.position()
    }

    pub fn reset(&mut self) -> Result<(), SingularityError> {
        self.text.reset()
    }

    /// Encode all `images` via `VisionEncoder` + `Aligner`, expand `image_token_id` placeholders in `prompt`,
    /// and advance the backbone across the expanded sequence.
    pub fn prefill(
        &mut self,
        prompt: &[u32],
        images: &[MultimodalImageInput],
    ) -> Result<ForwardOutput, SingularityError> {
        if prompt.is_empty() {
            return Err(error("Multimodal prompt cannot be empty"));
        }
        let hidden = self.text.hidden_size();
        let mut grids = Vec::with_capacity(images.len());
        let mut spans = Vec::with_capacity(images.len());
        for img in images {
            let (grid, patches) =
                prepare_image_patches(&img.rgb, img.width, img.height, &self.vision_config)?;
            let vit_features = self
                .vision
                .forward(&patches, grid.vit_height, grid.vit_width)?;
            let span = self.aligner.span(grid, &vit_features)?;
            grids.push(grid);
            spans.push(span);
        }
        let expanded = expand_image_tokens(prompt, self.image_token_id, &grids)?;
        let mut span_cursors: Vec<usize> = vec![0; spans.len()];
        let mut current_image_idx = 0usize;
        let mut last_output = ForwardOutput {
            logits: Vec::new(),
            target_hidden: Vec::new(),
        };
        for (&token, &kind) in expanded.tokens.iter().zip(&expanded.types) {
            match kind {
                TokenType::Text => {
                    last_output = self.text.forward_token_with_targets(token)?;
                }
                TokenType::ImageStart
                | TokenType::Image
                | TokenType::ImageNewline
                | TokenType::ImageEnd => {
                    if kind == TokenType::ImageStart
                        && span_cursors.get(current_image_idx).copied().unwrap_or(0) > 0
                    {
                        current_image_idx += 1;
                    }
                    let span = spans
                        .get(current_image_idx)
                        .ok_or_else(|| error("Image span index out of bounds"))?;
                    let offset = span_cursors[current_image_idx];
                    let end = offset
                        .checked_add(hidden)
                        .ok_or_else(|| error("Image span offset overflow"))?;
                    if end > span.len() {
                        return Err(error("Image span embedding exhausted early"));
                    }
                    span_cursors[current_image_idx] = end;
                    last_output =
                        self.text
                            .forward_embedding_with_targets(&span[offset..end], None, true)?;
                }
            }
        }
        for (span, consumed) in spans.iter().zip(span_cursors) {
            if span.len() != consumed {
                return Err(error("Unconsumed image span embeddings"));
            }
        }
        Ok(last_output)
    }

    /// Prefill a fresh multimodal prompt and greedily generate up to `max_new_tokens`.
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        images: &[MultimodalImageInput],
        max_new_tokens: usize,
        eos: u32,
    ) -> Result<Vec<u32>, SingularityError> {
        if self.position() != 0 || prompt.is_empty() {
            return Err(error(
                "Greedy multimodal generation requires a fresh sequence and nonempty prompt",
            ));
        }
        let mut logits = self.prefill(prompt, images)?.logits;
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
            logits = self.text.forward_token(token)?;
        }
        Ok(generated)
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

    #[test]
    fn multimodal_checkpoint_runs_image_and_text_tokens() {
        use half::bf16;
        let mut cfg: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        cfg.image_token_id = 1;
        let c = &mut cfg.text_config;
        c.vocab_size = 2;
        c.hidden_size = 32;
        c.num_hidden_layers = 1;
        c.num_nextn_predict_layers = 0;
        c.compress_ratios = vec![0];
        c.kv_source_layer_ids.clear();
        c.index_source_layer_ids.clear();
        c.candidate_source_layer_id = None;
        c.engram_layer_ids.clear();
        c.hc_mult = 2;
        c.num_attention_heads = 1;
        c.head_dim = 32;
        c.q_lora_rank = 32;
        c.o_lora_rank = 32;
        c.o_groups = 1;
        c.sliding_window = 16;
        c.qk_rope_head_dim = 2;
        c.n_routed_experts = 1;
        c.num_experts_per_tok = 1;
        c.moe_intermediate_size = 32;
        c.dspark_target_layer_ids = vec![0];
        let v = &mut cfg.vision_config;
        v.num_hidden_layers = 1;
        v.hidden_size = 4;
        v.num_attention_heads = 1;
        v.intermediate_size = 4;
        v.patch_size = 2;
        v.downsample_ratio = 2;
        v.min_pixels = 4;
        v.max_image_tokens = 16;

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
        let bf16_ones =
            |count: usize| -> Vec<u8> { bf16::from_f32(0.125).to_le_bytes().repeat(count) };
        let bf16_zeros = |count: usize| -> Vec<u8> { vec![0; count * 2] };

        for name in ["embed.weight", "head.weight"] {
            add(name.into(), "BF16", vec![2, 32], bf16_ones(64));
        }
        add("norm.weight".into(), "BF16", vec![32], bf16_ones(32));
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
                bf16_ones(32),
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
            bf16_ones(32),
        );
        add("layers.0.ffn.gate.bias".into(), "F32", vec![1], vec![0; 4]);
        add(
            "layers.0.ffn.gate.bias_vl".into(),
            "F32",
            vec![1],
            vec![0; 4],
        );
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
        // Vision encoder tensors (hidden_size=4, intermediate_size=4, patch_size=2 -> patch_dim=12)
        add(
            "vision.patch_embed.proj.weight".into(),
            "BF16",
            vec![4, 12],
            bf16_ones(48),
        );
        add(
            "vision.patch_embed.proj.bias".into(),
            "BF16",
            vec![4],
            bf16_zeros(4),
        );
        add("vision.norm.weight".into(), "BF16", vec![4], bf16_ones(4));
        add(
            "vision.blocks.0.norm1.weight".into(),
            "BF16",
            vec![4],
            bf16_ones(4),
        );
        add(
            "vision.blocks.0.attn.wqkv.weight".into(),
            "BF16",
            vec![12, 4],
            bf16_ones(48),
        );
        add(
            "vision.blocks.0.attn.wqkv.bias".into(),
            "BF16",
            vec![12],
            bf16_zeros(12),
        );
        add(
            "vision.blocks.0.attn.wo.weight".into(),
            "BF16",
            vec![4, 4],
            bf16_ones(16),
        );
        add(
            "vision.blocks.0.attn.wo.bias".into(),
            "BF16",
            vec![4],
            bf16_zeros(4),
        );
        add(
            "vision.blocks.0.norm2.weight".into(),
            "BF16",
            vec![4],
            bf16_ones(4),
        );
        add(
            "vision.blocks.0.mlp.w1.weight".into(),
            "BF16",
            vec![8, 4],
            bf16_ones(32),
        );
        add(
            "vision.blocks.0.mlp.w2.weight".into(),
            "BF16",
            vec![4, 4],
            bf16_ones(16),
        );
        // Aligner tensors (vision_dim=4, ratio=2 -> input=16, text_hidden=32)
        add(
            "aligner.w1.weight".into(),
            "BF16",
            vec![32, 16],
            bf16_ones(512),
        );
        add("aligner.w1.bias".into(), "BF16", vec![32], bf16_zeros(32));
        add(
            "aligner.w2.weight".into(),
            "BF16",
            vec![32, 32],
            bf16_ones(1024),
        );
        add("aligner.w2.bias".into(), "BF16", vec![32], bf16_zeros(32));
        for marker in ["image_start", "image_newline", "image_end"] {
            add(marker.into(), "BF16", vec![32], bf16_ones(32));
        }

        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();

        let mut mm = MultimodalModel::load(&loader, &cfg, None, 0).unwrap();
        let img = MultimodalImageInput {
            width: 2,
            height: 2,
            rgb: vec![128; 2 * 2 * 3],
        };
        // Prompt: [text(0), image(1), text(0)]
        // 2x2 image with patch_size=2, downsample_ratio=2 -> vit=1x1, llm=1x1 -> 4 image tokens (Start, Image, Newline, End)
        // Total prefill tokens = 1 + 4 + 1 = 6
        let generated = mm.generate_greedy(&[0, 1, 0], &[img], 2, 99).unwrap();
        assert_eq!(generated.len(), 2);
        // 6 prefill tokens + 1 generated token forwarded for second step = 7 positions
        assert_eq!(mm.position(), 7);
        mm.reset().unwrap();
        assert_eq!(mm.position(), 0);
    }
}
