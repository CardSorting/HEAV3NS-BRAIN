//! Checkpoint configuration and topology inspection, not a V4.1 execution backend.
//! Field semantics follow the checkpoint config and DeepInfer's HF config loader.
use crate::SafetensorsMmapLoader;
use serde::Deserialize;
use singularity_core::SingularityError;
use std::path::Path;

fn default_eos_token_id() -> u32 {
    1
}

fn default_pad_token_id() -> u32 {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct V41Config {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub dtype: String,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,
    pub image_token_id: u32,
    pub quantization_config: QuantizationConfig,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub expert_dtype: String,
    pub activation_scheme: String,
    pub weight_block_size: Vec<usize>,
    pub scale_fmt: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub sliding_window: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub num_hidden_layers: usize,
    pub num_nextn_predict_layers: usize,
    pub compress_ratios: Vec<usize>,
    pub kv_source_layer_ids: Vec<usize>,
    pub index_source_layer_ids: Vec<usize>,
    pub candidate_source_layer_id: Option<usize>,
    pub engram_layer_ids: Vec<usize>,
    pub engram_num_embeddings: Vec<u64>,
    pub engram_max_ngram_size: usize,
    pub engram_vocab_size: u64,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub engram_compressed_vocab_size: usize,
    #[serde(default = "default_pad_token_id")]
    pub engram_pad_token_id: u32,
    pub hc_mult: usize,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: usize,
    pub rms_norm_eps: f32,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub swiglu_limit: f32,
    pub qk_rope_head_dim: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: u32,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_num_experts_per_tok: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub rope_theta: f32,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointInventorySummary {
    pub verified_tensors: usize,
    pub backbone_layers: usize,
    pub draft_stages: usize,
    pub vision_layers: usize,
}

impl V41Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SingularityError> {
        let p = path.as_ref();
        let resolved = if p.is_dir() {
            let direct = p.join("config.json");
            if direct.exists() {
                direct
            } else {
                let meta = p.join("metadata").join("config.json");
                if meta.exists() {
                    meta
                } else {
                    direct
                }
            }
        } else if !p.exists() {
            if let (Some(parent), Some(name)) = (p.parent(), p.file_name()) {
                let meta = parent.join("metadata").join(name);
                if meta.exists() {
                    meta
                } else {
                    p.to_path_buf()
                }
            } else {
                p.to_path_buf()
            }
        } else {
            p.to_path_buf()
        };
        let config: Self = serde_json::from_reader(std::fs::File::open(resolved)?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), SingularityError> {
        let fail = |reason: &str| SingularityError::ModelError {
            reason: reason.into(),
        };
        if self.model_type != "deepseek_v41"
            || self.architectures.as_slice() != ["DeepseekV41ForCausalLM"]
        {
            return Err(fail(
                "Not a supported DeepSeek V4.1 checkpoint configuration",
            ));
        }
        let q = &self.quantization_config;
        if self.dtype != "bfloat16"
            || q.quant_method != "fp8"
            || q.expert_dtype != "fp4"
            || q.activation_scheme != "dynamic"
            || q.weight_block_size != [32, 32]
            || q.scale_fmt != "ue8m0"
        {
            return Err(fail("Unsupported V4.1 quantization configuration"));
        }
        let t = &self.text_config;
        let expected = t
            .num_hidden_layers
            .checked_add(t.num_nextn_predict_layers)
            .ok_or_else(|| fail("Layer count overflow"))?;
        if t.hidden_size == 0
            || t.num_hidden_layers == 0
            || t.hc_mult == 0
            || t.compress_ratios.len() != expected
        {
            return Err(fail("Invalid V4.1 layer topology"));
        }
        for ids in [
            &t.kv_source_layer_ids,
            &t.index_source_layer_ids,
            &t.engram_layer_ids,
            &t.dspark_target_layer_ids,
        ] {
            if ids.iter().any(|&i| i >= t.num_hidden_layers) || ids.windows(2).any(|w| w[0] >= w[1])
            {
                return Err(fail("Invalid V4.1 layer IDs"));
            }
        }
        if t.candidate_source_layer_id
            .is_some_and(|i| i >= t.num_hidden_layers)
        {
            return Err(fail("Invalid V4.1 candidate source layer"));
        }
        if t.num_nextn_predict_layers > 0
            && (t.dspark_block_size == 0
                || t.dspark_markov_rank == 0
                || t.dspark_target_layer_ids.is_empty()
                || t.dspark_noise_token_id as usize >= t.vocab_size
                || t.dspark_n_routed_experts == 0
                || t.dspark_num_experts_per_tok == 0
                || t.dspark_num_experts_per_tok > t.dspark_n_routed_experts
                || t.compress_ratios[t.num_hidden_layers..]
                    .iter()
                    .any(|&ratio| ratio != 0))
        {
            return Err(fail("Invalid DSpark topology"));
        }
        let vision = &self.vision_config;
        if vision.num_hidden_layers > 0
            && (vision.hidden_size == 0
                || vision.num_attention_heads == 0
                || vision.hidden_size % vision.num_attention_heads != 0
                || (vision.hidden_size / vision.num_attention_heads) % 4 != 0
                || vision.intermediate_size == 0
                || vision.patch_size == 0
                || vision.downsample_ratio == 0
                || vision.max_image_tokens < 4
                || vision.max_wh_ratio == Some(0)
                || !vision.rope_theta.is_finite()
                || vision.rope_theta <= 1.0
                || self.image_token_id as usize >= t.vocab_size)
        {
            return Err(fail("Invalid V4.1 vision configuration"));
        }
        Ok(())
    }

    /// Nearest source no later than `layer`, or None before the first source.
    pub fn source_for_layer(sources: &[usize], layer: usize) -> Option<usize> {
        sources
            .iter()
            .copied()
            .filter(|&source| source <= layer)
            .max()
    }

    /// Visit every required V4.1 checkpoint tensor `(name, expected_dtype)` for the
    /// configured backbone, compressor, indexer, Engram, DSpark, and vision layers.
    pub fn visit_required_tensors(
        &self,
        mut visitor: impl FnMut(&str, &'static str) -> Result<(), SingularityError>,
    ) -> Result<CheckpointInventorySummary, SingularityError> {
        self.validate()?;
        let mut count = 0usize;
        let mut require =
            |name: &str, expected_dtype: &'static str| -> Result<(), SingularityError> {
                visitor(name, expected_dtype)?;
                count += 1;
                Ok(())
            };

        require("embed.weight", "BF16")?;
        require("head.weight", "BF16")?;
        require("norm.weight", "BF16")?;

        let t = &self.text_config;
        let has_vision = self.vision_config.num_hidden_layers > 0;
        for layer in 0..t.num_hidden_layers {
            for hc in ["hc_attn", "hc_ffn"] {
                require(&format!("layers.{layer}.{hc}_fn"), "F32")?;
                require(&format!("layers.{layer}.{hc}_base"), "F32")?;
                require(&format!("layers.{layer}.{hc}_scale"), "F32")?;
            }
            for norm in ["attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"] {
                require(&format!("layers.{layer}.{norm}.weight"), "BF16")?;
            }
            for mat in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
                require(&format!("layers.{layer}.attn.{mat}.weight"), "F8_E4M3")?;
                require(&format!("layers.{layer}.attn.{mat}.scale"), "F8_E8M0")?;
            }
            require(&format!("layers.{layer}.attn.attn_sink"), "F32")?;

            let ratio = t.compress_ratios[layer];
            if ratio > 0 {
                if t.kv_source_layer_ids.contains(&layer) {
                    require(
                        &format!("layers.{layer}.attn.compressor.wkv.weight"),
                        "BF16",
                    )?;
                    require(
                        &format!("layers.{layer}.attn.compressor.norm.weight"),
                        "BF16",
                    )?;
                    if ratio > 1 {
                        require(
                            &format!("layers.{layer}.attn.compressor.wgate.weight"),
                            "BF16",
                        )?;
                    }
                }
                if t.index_source_layer_ids.contains(&layer) {
                    require(
                        &format!("layers.{layer}.attn.indexer.wq_b.weight"),
                        "F8_E4M3",
                    )?;
                    require(
                        &format!("layers.{layer}.attn.indexer.wq_b.scale"),
                        "F8_E8M0",
                    )?;
                    require(
                        &format!("layers.{layer}.attn.indexer.weights_proj.weight"),
                        "BF16",
                    )?;
                    if t.kv_source_layer_ids.contains(&layer) {
                        require(&format!("layers.{layer}.attn.indexer.wk.weight"), "BF16")?;
                        require(
                            &format!("layers.{layer}.attn.indexer.k_norm.weight"),
                            "BF16",
                        )?;
                    }
                }
            }

            require(&format!("layers.{layer}.ffn.gate.weight"), "BF16")?;
            require(&format!("layers.{layer}.ffn.gate.bias"), "F32")?;
            if has_vision {
                require(&format!("layers.{layer}.ffn.gate.bias_vl"), "F32")?;
            }
            for mat in ["w1", "w2", "w3"] {
                require(
                    &format!("layers.{layer}.ffn.shared_experts.{mat}.weight"),
                    "F8_E4M3",
                )?;
                require(
                    &format!("layers.{layer}.ffn.shared_experts.{mat}.scale"),
                    "F8_E8M0",
                )?;
            }
            for expert in 0..t.n_routed_experts {
                for mat in ["w1", "w2", "w3"] {
                    require(
                        &format!("layers.{layer}.ffn.experts.{expert}.{mat}.weight"),
                        "I8",
                    )?;
                    require(
                        &format!("layers.{layer}.ffn.experts.{expert}.{mat}.scale"),
                        "F8_E8M0",
                    )?;
                }
            }

            if t.engram_layer_ids.contains(&layer) {
                for mat in ["embed", "wkv"] {
                    require(&format!("layers.{layer}.engram.{mat}.weight"), "F8_E4M3")?;
                    require(&format!("layers.{layer}.engram.{mat}.scale"), "F8_E8M0")?;
                }
                require(&format!("layers.{layer}.engram.q_weight"), "BF16")?;
                require(&format!("layers.{layer}.engram.k_weight"), "BF16")?;
            }
        }

        for stage in 0..t.num_nextn_predict_layers {
            if stage == 0 {
                require("mtp.0.main_proj.weight", "F8_E4M3")?;
                require("mtp.0.main_proj.scale", "F8_E8M0")?;
                require("mtp.0.main_norm.weight", "BF16")?;
            }
            if stage + 1 == t.num_nextn_predict_layers {
                require(&format!("mtp.{stage}.norm.weight"), "BF16")?;
                require(&format!("mtp.{stage}.markov_head.embed.weight"), "BF16")?;
                require(&format!("mtp.{stage}.markov_head.head.weight"), "BF16")?;
                require(&format!("mtp.{stage}.confidence_head.proj.weight"), "BF16")?;
            }
            for hc in ["hc_attn", "hc_ffn"] {
                require(&format!("mtp.{stage}.{hc}_fn"), "F32")?;
                require(&format!("mtp.{stage}.{hc}_base"), "F32")?;
                require(&format!("mtp.{stage}.{hc}_scale"), "F32")?;
            }
            for norm in ["attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"] {
                require(&format!("mtp.{stage}.{norm}.weight"), "BF16")?;
            }
            for mat in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
                require(&format!("mtp.{stage}.attn.{mat}.weight"), "F8_E4M3")?;
                require(&format!("mtp.{stage}.attn.{mat}.scale"), "F8_E8M0")?;
            }
            require(&format!("mtp.{stage}.attn.attn_sink"), "F32")?;
            require(&format!("mtp.{stage}.ffn.gate.weight"), "BF16")?;
            require(&format!("mtp.{stage}.ffn.gate.bias"), "F32")?;
            if has_vision {
                require(&format!("mtp.{stage}.ffn.gate.bias_vl"), "F32")?;
            }
            for mat in ["w1", "w2", "w3"] {
                require(
                    &format!("mtp.{stage}.ffn.shared_experts.{mat}.weight"),
                    "F8_E4M3",
                )?;
                require(
                    &format!("mtp.{stage}.ffn.shared_experts.{mat}.scale"),
                    "F8_E8M0",
                )?;
            }
            for expert in 0..t.dspark_n_routed_experts {
                for mat in ["w1", "w2", "w3"] {
                    require(
                        &format!("mtp.{stage}.ffn.experts.{expert}.{mat}.weight"),
                        "I8",
                    )?;
                    require(
                        &format!("mtp.{stage}.ffn.experts.{expert}.{mat}.scale"),
                        "F8_E8M0",
                    )?;
                }
            }
        }

        let v = &self.vision_config;
        if v.num_hidden_layers > 0 {
            require("vision.patch_embed.proj.weight", "BF16")?;
            require("vision.patch_embed.proj.bias", "BF16")?;
            require("vision.norm.weight", "BF16")?;
            for layer in 0..v.num_hidden_layers {
                let p = format!("vision.blocks.{layer}");
                require(&format!("{p}.norm1.weight"), "BF16")?;
                require(&format!("{p}.attn.wqkv.weight"), "BF16")?;
                require(&format!("{p}.attn.wqkv.bias"), "BF16")?;
                require(&format!("{p}.attn.wo.weight"), "BF16")?;
                require(&format!("{p}.attn.wo.bias"), "BF16")?;
                require(&format!("{p}.norm2.weight"), "BF16")?;
                require(&format!("{p}.mlp.w1.weight"), "BF16")?;
                require(&format!("{p}.mlp.w2.weight"), "BF16")?;
            }
            for sub in ["aligner.w1", "aligner.w2"] {
                require(&format!("{sub}.weight"), "BF16")?;
                require(&format!("{sub}.bias"), "BF16")?;
            }
            for marker in ["image_start", "image_newline", "image_end"] {
                require(marker, "BF16")?;
            }
        }

        Ok(CheckpointInventorySummary {
            verified_tensors: count,
            backbone_layers: t.num_hidden_layers,
            draft_stages: t.num_nextn_predict_layers,
            vision_layers: v.num_hidden_layers,
        })
    }

    /// Verify that a loaded `SafetensorsMmapLoader` contains all required V4.1 tensors
    /// with expected dtypes for the configured backbone, Engram, DSpark, and vision layers.
    pub fn validate_checkpoint_inventory(
        &self,
        loader: &SafetensorsMmapLoader,
    ) -> Result<CheckpointInventorySummary, SingularityError> {
        self.visit_required_tensors(|name, expected_dtype| {
            let info =
                loader
                    .get_tensor_info(name)
                    .ok_or_else(|| SingularityError::ModelError {
                        reason: format!("Missing required V4.1 checkpoint tensor: {name}"),
                    })?;
            if info.dtype != expected_dtype {
                return Err(SingularityError::ModelError {
                    reason: format!(
                        "Tensor {name} has dtype {} (expected {expected_dtype})",
                        info.dtype
                    ),
                });
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_checkpoint_config_and_shared_topology() {
        let mut config: V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.bos_token_id, 0);
        assert_eq!(config.eos_token_id, 1);
        assert_eq!(config.pad_token_id, 2);
        assert_eq!(config.text_config.engram_pad_token_id, 2);
        assert_eq!(
            (
                config.text_config.num_hidden_layers,
                config.text_config.num_nextn_predict_layers
            ),
            (40, 3)
        );
        assert_eq!(config.text_config.compress_ratios.len(), 43);
        assert_eq!(
            V41Config::source_for_layer(&config.text_config.kv_source_layer_ids, 19),
            Some(14)
        );
        assert_eq!(
            V41Config::source_for_layer(&config.text_config.kv_source_layer_ids, 1),
            None
        );
        // Verify the full pinned V4.1 config visits the exact 96,085 tensors of the 48-shard index
        let summary = config.visit_required_tensors(|_, _| Ok(())).unwrap();
        assert_eq!(summary.verified_tensors, 96_085);
        assert_eq!(summary.backbone_layers, 40);
        assert_eq!(summary.draft_stages, 3);
        assert_eq!(summary.vision_layers, 32);

        config.text_config.compress_ratios.pop();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_checkpoint_inventory_detects_missing_and_valid_tensors() {
        let mut cfg: V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        cfg.image_token_id = 1;
        cfg.text_config.vocab_size = 2;
        cfg.text_config.hidden_size = 32;
        cfg.text_config.num_hidden_layers = 3;
        cfg.text_config.num_nextn_predict_layers = 2;
        cfg.text_config.compress_ratios = vec![0, 2, 1, 0, 0];
        cfg.text_config.kv_source_layer_ids = vec![1];
        cfg.text_config.index_source_layer_ids = vec![1, 2];
        cfg.text_config.candidate_source_layer_id = Some(1);
        cfg.text_config.engram_layer_ids = vec![0];
        cfg.text_config.dspark_target_layer_ids = vec![0, 2];
        cfg.text_config.n_routed_experts = 1;
        cfg.text_config.num_experts_per_tok = 1;
        cfg.text_config.dspark_n_routed_experts = 1;
        cfg.text_config.dspark_num_experts_per_tok = 1;
        cfg.text_config.dspark_noise_token_id = 0;
        cfg.vision_config.num_hidden_layers = 1;

        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut add = |name: &str, dtype: &str| {
            let start = data.len();
            let elem_bytes = match dtype {
                "F32" => 4,
                "BF16" => 2,
                _ => 1,
            };
            data.extend(std::iter::repeat_n(0u8, elem_bytes));
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype":dtype,"shape":[1],"data_offsets":[start,data.len()]}),
            );
        };
        cfg.visit_required_tensors(|name, dtype| {
            add(name, dtype);
            Ok(())
        })
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let h_bytes = serde_json::to_vec(&header).unwrap();
        let mut bytes = (h_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend(h_bytes);
        bytes.extend(&data);
        std::fs::write(&path, bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(&path).unwrap();

        let summary = cfg.validate_checkpoint_inventory(&loader).unwrap();
        assert_eq!(summary.backbone_layers, 3);
        assert_eq!(summary.draft_stages, 2);
        assert_eq!(summary.vision_layers, 1);
        // Verify key compressor, indexer, engram, and final-stage MTP tensors are present
        assert!(header.contains_key("layers.1.attn.compressor.wkv.weight"));
        assert!(header.contains_key("layers.1.attn.compressor.wgate.weight"));
        assert!(header.contains_key("layers.1.attn.indexer.wk.weight"));
        assert!(!header.contains_key("layers.2.attn.indexer.wk.weight"));
        assert!(header.contains_key("layers.2.attn.indexer.wq_b.weight"));
        assert!(header.contains_key("layers.0.engram.embed.weight"));
        assert!(header.contains_key("layers.0.engram.wkv.weight"));
        assert!(header.contains_key("layers.0.engram.q_weight"));
        assert!(!header.contains_key("mtp.0.markov_head.head.weight"));
        assert!(header.contains_key("mtp.1.markov_head.head.weight"));

        // Removing a required tensor should fail inventory validation
        header.clear();
        data.clear();
        let mut add_without_wgate = |name: &str, dtype: &str| {
            if name == "layers.1.attn.compressor.wgate.weight" {
                return;
            }
            let start = data.len();
            let elem_bytes = match dtype {
                "F32" => 4,
                "BF16" => 2,
                _ => 1,
            };
            data.extend(std::iter::repeat_n(0u8, elem_bytes));
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype":dtype,"shape":[1],"data_offsets":[start,data.len()]}),
            );
        };
        cfg.visit_required_tensors(|name, dtype| {
            add_without_wgate(name, dtype);
            Ok(())
        })
        .unwrap();
        let bad_path = dir.path().join("model_missing.safetensors");
        let h_bytes = serde_json::to_vec(&header).unwrap();
        let mut bytes = (h_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend(h_bytes);
        bytes.extend(&data);
        std::fs::write(&bad_path, bytes).unwrap();
        let bad_loader = SafetensorsMmapLoader::open(&bad_path).unwrap();
        assert!(cfg.validate_checkpoint_inventory(&bad_loader).is_err());
    }
}
