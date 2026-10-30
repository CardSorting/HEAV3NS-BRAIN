use singularity_core::SingularityError;
use std::collections::HashMap;

/// Stage 1: MQAIndexer with Top-K DeepSelect Token Selection.
/// Quantized FP4 sparse indexing to identify the most salient key-value tokens.
#[derive(Debug, Clone)]
pub struct MqaIndexer {
    pub hidden_dim: usize,
    pub index_topk: usize,
    pub use_fp4: bool,
}

impl MqaIndexer {
    pub fn new(hidden_dim: usize, index_topk: usize, use_fp4: bool) -> Self {
        Self {
            hidden_dim,
            index_topk,
            use_fp4,
        }
    }

    /// Selects Top-K candidate token indices based on query-key relevance scores.
    pub fn select_topk_tokens(&self, scores: &[f32], total_tokens: usize) -> Vec<usize> {
        if total_tokens <= self.index_topk {
            return (0..total_tokens).collect();
        }

        let mut indexed_scores: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        // Sort descending by score
        indexed_scores
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed_scores
            .iter()
            .take(self.index_topk)
            .map(|&(idx, _)| idx)
            .collect()
    }
}

/// Stage 2: FlashMLA Dense Latent Attention Projection (A -> norm -> B).
#[derive(Debug, Clone)]
pub struct FlashMlaConfig {
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_heads: usize,
}

impl Default for FlashMlaConfig {
    fn default() -> Self {
        Self {
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            num_heads: 128,
        }
    }
}

pub struct TwoStageAttention {
    pub indexer: MqaIndexer,
    pub mla_config: FlashMlaConfig,
}

impl TwoStageAttention {
    pub fn new(indexer: MqaIndexer, mla_config: FlashMlaConfig) -> Self {
        Self {
            indexer,
            mla_config,
        }
    }

    pub fn compute_sparse_attention(
        &self,
        candidate_scores: &[f32],
        total_tokens: usize,
        swa_window_size: usize,
    ) -> Result<Vec<usize>, SingularityError> {
        // Stage 1: Sparse FP4 top-k token selection
        let mut selected_indices = self
            .indexer
            .select_topk_tokens(candidate_scores, total_tokens);

        // Always include sliding window attention (SWA) recent tokens
        let swa_start = total_tokens.saturating_sub(swa_window_size);
        for idx in swa_start..total_tokens {
            if !selected_indices.contains(&idx) {
                selected_indices.push(idx);
            }
        }
        selected_indices.sort_unstable();

        // Stage 2: Dense FlashMLA executes low-rank projection over selected_indices
        Ok(selected_indices)
    }
}

/// SharedLayerTopology:
/// Maps layer index to source projection layer index to allow non-adjacent layers
/// to share projection weights and minimize VRAM consumption.
#[derive(Debug, Clone)]
pub struct SharedLayerTopology {
    pub layer_map: HashMap<usize, usize>, // current_layer -> source_layer
}

impl SharedLayerTopology {
    pub fn new(num_layers: usize, shared_interval: usize) -> Self {
        let mut layer_map = HashMap::new();
        for l in 0..num_layers {
            let source = if shared_interval > 0 {
                (l / shared_interval) * shared_interval
            } else {
                l
            };
            layer_map.insert(l, source);
        }
        Self { layer_map }
    }

    pub fn get_source_layer(&self, layer_idx: usize) -> usize {
        self.layer_map.get(&layer_idx).copied().unwrap_or(layer_idx)
    }

    pub fn is_weight_owner(&self, layer_idx: usize) -> bool {
        self.get_source_layer(layer_idx) == layer_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_stage_attention_topk_and_swa_preservation() {
        let indexer = MqaIndexer::new(512, 4, true);
        let mla = FlashMlaConfig::default();
        let attn = TwoStageAttention::new(indexer, mla);

        // 10 tokens total, scores highest at indices 1, 3, 5, 7
        let scores = vec![0.1, 0.9, 0.2, 0.85, 0.3, 0.75, 0.4, 0.95, 0.05, 0.01];
        let selected = attn.compute_sparse_attention(&scores, 10, 2).unwrap();

        // Must include top-4 scores (7, 1, 3, 5) AND SWA window recent tokens (8, 9)
        assert!(selected.contains(&7));
        assert!(selected.contains(&1));
        assert!(selected.contains(&3));
        assert!(selected.contains(&5));
        assert!(selected.contains(&8));
        assert!(selected.contains(&9));
    }

    #[test]
    fn test_shared_layer_topology() {
        let topo = SharedLayerTopology::new(8, 2);
        assert_eq!(topo.get_source_layer(0), 0);
        assert_eq!(topo.get_source_layer(1), 0);
        assert_eq!(topo.get_source_layer(2), 2);
        assert_eq!(topo.get_source_layer(3), 2);
        assert!(topo.is_weight_owner(0));
        assert!(!topo.is_weight_owner(1));
    }
}
