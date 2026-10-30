//! Checkpoint configuration and topology inspection, not a V4.1 execution backend.
//! Field semantics follow the checkpoint config and DeepInfer's HF config loader.
use serde::Deserialize;
use singularity_core::SingularityError;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct V41Config {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub dtype: String,
    pub image_token_id: u32,
    pub quantization_config: QuantizationConfig,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

#[derive(Debug, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub expert_dtype: String,
    pub activation_scheme: String,
    pub weight_block_size: Vec<usize>,
    pub scale_fmt: String,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Deserialize)]
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

impl V41Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SingularityError> {
        let config: Self = serde_json::from_reader(std::fs::File::open(path)?)?;
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
        config.text_config.compress_ratios.pop();
        assert!(config.validate().is_err());
    }
}
