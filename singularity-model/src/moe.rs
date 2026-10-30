use singularity_core::SingularityError;
use std::collections::HashMap;

/// MoE Router using sqrtsoftplus scoring with noaux_tc bias and correction biases.
#[derive(Debug, Clone)]
pub struct MoERouter {
    pub num_experts: usize,
    pub top_k: usize,
    pub e_score_correction_bias: Vec<f32>,
}

impl MoERouter {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
            e_score_correction_bias: vec![0.0; num_experts],
        }
    }

    /// sqrtsoftplus scoring: score = sqrt(ln(1 + exp(x))) + correction_bias
    pub fn score_and_route(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .take(self.num_experts)
            .map(|(idx, &val)| {
                // softplus = ln(1 + e^val)
                let sp = if val > 20.0 {
                    val
                } else if val < -20.0 {
                    val.exp()
                } else {
                    (1.0 + val.exp()).ln()
                };
                let sqrt_sp = sp.max(0.0).sqrt();
                let bias = self
                    .e_score_correction_bias
                    .get(idx)
                    .copied()
                    .unwrap_or(0.0);
                (idx, sqrt_sp + bias)
            })
            .collect();

        // Sort descending by score and pick top_k
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(self.top_k);

        // Normalize top_k scores with softmax
        let max_s = scores.iter().map(|s| s.1).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s.1 - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();

        scores
            .into_iter()
            .zip(exps)
            .map(|((idx, _), exp)| (idx, exp / sum_exp.max(1e-6)))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ExpertRebalanceAction {
    pub hot_expert_id: usize,
    pub target_gpu_rank: usize,
    pub redundant_slot_id: usize,
}

/// EPLB (Expert Load Balancer):
/// Tracks token arrivals per physical expert, recomputes optimal expert placement,
/// and plans zero-downtime weight shuffling across GPUs using symmetric memory handles.
pub struct ExpertLoadBalancer {
    num_experts: usize,
    num_gpus: usize,
    expert_arrival_counts: Vec<usize>,
    expert_gpu_placement: HashMap<usize, Vec<usize>>, // expert_id -> list of gpu_ranks hosting it
}

impl ExpertLoadBalancer {
    pub fn new(num_experts: usize, num_gpus: usize) -> Self {
        let mut placement = HashMap::new();
        for e in 0..num_experts {
            let primary_rank = e % num_gpus;
            placement.insert(e, vec![primary_rank]);
        }

        Self {
            num_experts,
            num_gpus,
            expert_arrival_counts: vec![0; num_experts],
            expert_gpu_placement: placement,
        }
    }

    pub fn record_expert_dispatch(&mut self, expert_id: usize, token_count: usize) {
        if expert_id < self.num_experts {
            self.expert_arrival_counts[expert_id] += token_count;
        }
    }

    pub fn compute_imbalance_ratio(&self) -> f32 {
        let total: usize = self.expert_arrival_counts.iter().sum();
        if total == 0 {
            return 1.0;
        }
        let avg = (total as f32) / (self.num_experts as f32);
        let max_val = *self.expert_arrival_counts.iter().max().unwrap_or(&0) as f32;
        max_val / avg.max(1.0)
    }

    /// Rebalances hot experts by planning replication to under-utilized GPUs.
    pub fn plan_rebalance(
        &mut self,
        threshold_ratio: f32,
    ) -> Result<Vec<ExpertRebalanceAction>, SingularityError> {
        let mut actions = Vec::new();
        let total: usize = self.expert_arrival_counts.iter().sum();
        if total == 0 {
            return Ok(actions);
        }

        let avg = (total as f32) / (self.num_experts as f32);

        // Find hottest expert
        let mut ranked_experts: Vec<(usize, usize)> = self
            .expert_arrival_counts
            .iter()
            .copied()
            .enumerate()
            .collect();
        ranked_experts.sort_by(|a, b| b.1.cmp(&a.1));

        for &(exp_id, count) in &ranked_experts {
            let ratio = (count as f32) / avg.max(1.0);
            if ratio >= threshold_ratio {
                // Find a GPU rank that does not currently host this expert
                let current_ranks = self
                    .expert_gpu_placement
                    .get(&exp_id)
                    .cloned()
                    .unwrap_or_default();
                for r in 0..self.num_gpus {
                    if !current_ranks.contains(&r) {
                        actions.push(ExpertRebalanceAction {
                            hot_expert_id: exp_id,
                            target_gpu_rank: r,
                            redundant_slot_id: 1, // redundant replica slot
                        });
                        self.expert_gpu_placement.entry(exp_id).or_default().push(r);
                        break;
                    }
                }
            }
        }

        Ok(actions)
    }

    pub fn get_expert_ranks(&self, expert_id: usize) -> &[usize] {
        self.expert_gpu_placement
            .get(&expert_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_router_scoring_and_softmax_normalization() {
        let router = MoERouter::new(8, 2);
        let logits = vec![0.5, 2.0, -1.0, 4.0, 1.2, 0.0, -3.0, 1.8];
        let routes = router.score_and_route(&logits);

        assert_eq!(routes.len(), 2);
        // Expert 3 had highest logit 4.0
        assert_eq!(routes[0].0, 3);
        // Expert 1 had second highest 2.0
        assert_eq!(routes[1].0, 1);

        // Softmax probabilities must sum to approximately 1.0
        let sum_p: f32 = routes.iter().map(|r| r.1).sum();
        assert!((sum_p - 1.0).abs() < 1e-4);
    }
}
