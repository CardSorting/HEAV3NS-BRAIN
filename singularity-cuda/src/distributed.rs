//! Multi-GPU (8x NVIDIA B200) distributed placement planner and collective coordination
//! for `deepseek-ai/DeepSeek-V4.1-Flash-1017` (510.3 GB checkpoint across 1.44 TB HBM3e).

use serde::{Deserialize, Serialize};
use singularity_core::SingularityError;

/// Pinned byte footprint of the 48-shard `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint.
pub const V41_PINNED_CHECKPOINT_BYTES: u64 = 510_286_023_000;

/// Hardware specification for an 8x NVIDIA B200 node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct B200ClusterSpec {
    pub world_size: usize,
    pub hbm_per_gpu_bytes: u64,
    pub nvlink_bandwidth_gbps: u32,
    pub compute_capability: (u32, u32),
    pub safety_headroom_bytes: u64,
}

impl Default for B200ClusterSpec {
    fn default() -> Self {
        Self {
            world_size: 8,
            // 180 GiB HBM3e per NVIDIA B200 GPU
            hbm_per_gpu_bytes: 180 * 1024 * 1024 * 1024,
            nvlink_bandwidth_gbps: 1800,
            compute_capability: (10, 0),
            // Reserve at least 12 GiB per GPU for CUDA context, NCCL buffers, and fragmentation
            safety_headroom_bytes: 12 * 1024 * 1024 * 1024,
        }
    }
}

/// Architectural parameters governing V4.1-Flash-1017 distributed sharding and memory footprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V41ClusterTopology {
    pub num_hidden_layers: usize,
    pub num_nextn_predict_layers: usize,
    pub n_routed_experts: usize,
    pub dspark_n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub engram_layer_ids: Vec<usize>,
    pub engram_rows_per_layer: u64,
    pub engram_dim: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub vocab_size: usize,
    pub max_context_len: usize,
    pub sliding_window: usize,
}

impl Default for V41ClusterTopology {
    fn default() -> Self {
        Self {
            num_hidden_layers: 40,
            num_nextn_predict_layers: 3,
            n_routed_experts: 384,
            dspark_n_routed_experts: 128,
            num_experts_per_tok: 6,
            engram_layer_ids: vec![1, 14],
            engram_rows_per_layer: 384_016_682,
            engram_dim: 256,
            hidden_size: 5120,
            moe_intermediate_size: 2304,
            vocab_size: 129_280,
            max_context_len: 131_072,
            sliding_window: 128,
        }
    }
}

/// Per-GPU placement and HBM allocation plan for a single rank in the 8x B200 topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankPlacementPlan {
    pub rank: usize,
    pub device_ordinal: usize,
    /// `[start_expert, end_expert)` on each of the 40 backbone layers (`EP = world_size`).
    pub backbone_expert_range: (usize, usize),
    /// `[start_expert, end_expert)` on each of the 3 DSpark draft stages (`EP = world_size`).
    pub dspark_expert_range: (usize, usize),
    /// `[start_row, end_row)` in each of the Engram layers (`384,016,682` rows sharded across ranks).
    pub engram_row_range: (u64, u64),
    /// Replicated/TP dense weights (embeddings, head, attention, shared experts, vision/aligner, norms).
    pub dense_replicated_bytes: u64,
    /// Sharded NVFP4 routed experts + UE8M0 scales assigned to this rank.
    pub routed_expert_bytes: u64,
    /// Sharded FP8 Engram tables + UE8M0 scales + projections assigned to this rank.
    pub engram_shard_bytes: u64,
    /// Allocated KV cache + compressed KV pool + indexer key cache on this rank.
    pub kv_cache_budget_bytes: u64,

    /// Transient workspace for activations, expert all-to-all buffers, and CUDA graphs.
    pub activation_workspace_bytes: u64,
    /// Total HBM bytes allocated on this rank.
    pub total_allocated_bytes: u64,
    /// Remaining unallocated HBM bytes on this rank.
    pub headroom_bytes: u64,
}

/// Full 8-GPU cluster placement plan for `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterPlacementPlan {
    pub spec: B200ClusterSpec,
    pub topology: V41ClusterTopology,
    pub max_active_sequences: usize,
    pub total_checkpoint_bytes: u64,
    pub ranks: Vec<RankPlacementPlan>,
}

impl ClusterPlacementPlan {
    /// Compute and validate the deterministic multi-GPU placement plan for V4.1-Flash-1017.
    pub fn plan_v41_cluster(
        spec: B200ClusterSpec,
        topology: V41ClusterTopology,
        max_active_sequences: usize,
    ) -> Result<Self, SingularityError> {
        if spec.world_size == 0
            || max_active_sequences == 0
            || topology.num_hidden_layers == 0
            || topology.n_routed_experts == 0
            || !topology.n_routed_experts.is_multiple_of(spec.world_size)
            || (topology.num_nextn_predict_layers > 0
                && (topology.dspark_n_routed_experts == 0
                    || !topology
                        .dspark_n_routed_experts
                        .is_multiple_of(spec.world_size)))
        {
            return Err(SingularityError::ModelError {
                reason: "Invalid cluster world_size or expert divisibility for V4.1 placement"
                    .into(),
            });
        }
        if spec.compute_capability.0 < 10 {
            return Err(SingularityError::UnsupportedHardware {
                major: spec.compute_capability.0 as i32,
                minor: spec.compute_capability.1 as i32,
                reason: format!(
                    "V4.1 B200 cluster requires SM 10.0+ (Blackwell), got SM {}.{}",
                    spec.compute_capability.0, spec.compute_capability.1
                ),
            });
        }

        let world = spec.world_size as u64;
        let experts_per_rank = topology.n_routed_experts / spec.world_size;
        let dspark_experts_per_rank = if topology.num_nextn_predict_layers > 0 {
            topology.dspark_n_routed_experts / spec.world_size
        } else {
            0
        };

        // Per-expert byte size: 3 matrices (w1, w2, w3) of [5120, 2304] packed FP4 (0.5 B/elem)
        // + UE8M0 32-element block scales (1 B per 32 elements) = 18,800,640 B/expert.
        let elems_per_expert = 3u64
            .saturating_mul(topology.hidden_size as u64)
            .saturating_mul(topology.moe_intermediate_size as u64);
        let bytes_per_expert = (elems_per_expert / 2) + (elems_per_expert / 32);
        let total_routed_layers = (topology.num_hidden_layers as u64)
            * (topology.n_routed_experts as u64)
            + (topology.num_nextn_predict_layers as u64)
                * (topology.dspark_n_routed_experts as u64);
        let total_routed_expert_bytes = total_routed_layers.saturating_mul(bytes_per_expert);
        let routed_expert_bytes_per_rank = total_routed_expert_bytes.div_ceil(world);

        // Engram table bytes: 2 layers (1, 14) * ~384,016,682 rows * 256 cols (FP8 + 1/32 UE8M0 scale) + projections
        let engram_layers = topology.engram_layer_ids.len() as u64;
        let engram_table_elems_per_layer = topology
            .engram_rows_per_layer
            .saturating_mul(topology.engram_dim as u64);
        let engram_bytes_per_layer =
            engram_table_elems_per_layer + (engram_table_elems_per_layer / 32) + 162_201_600;
        let total_engram_bytes = engram_layers.saturating_mul(engram_bytes_per_layer);
        let engram_shard_bytes_per_rank = total_engram_bytes.div_ceil(world);

        // Remaining dense/attention/vision/shared-expert weights from the 510.3 GB checkpoint:
        // Replicated embeddings/norms/gates + tensor-parallel attention/shared-expert shards.
        let remaining_checkpoint_bytes = V41_PINNED_CHECKPOINT_BYTES
            .saturating_sub(total_routed_expert_bytes.saturating_add(total_engram_bytes));
        // Embeddings + LM head (~2.65 GB) + vision/aligner (~1.0 GB) replicated, rest TP-sharded:
        let replicated_base_bytes: u64 = 4_500_000_000;
        let dense_replicated_bytes =
            replicated_base_bytes.saturating_add(remaining_checkpoint_bytes.div_ceil(world));

        // KV cache budget per sequence across sliding-window caches + shared compressed KV sources + indexers

        let kv_bytes_per_seq: u64 = 663_000_000;
        let kv_cache_budget_bytes = (max_active_sequences as u64).saturating_mul(kv_bytes_per_seq);
        let activation_workspace_bytes: u64 = 4 * 1024 * 1024 * 1024; // 4 GiB

        let rows_per_rank = topology.engram_rows_per_layer.div_ceil(world);
        let mut ranks = Vec::with_capacity(spec.world_size);

        for rank in 0..spec.world_size {
            let start_row = (rank as u64).saturating_mul(rows_per_rank);
            let end_row = ((rank + 1) as u64)
                .saturating_mul(rows_per_rank)
                .min(topology.engram_rows_per_layer);
            let total_allocated = dense_replicated_bytes
                .saturating_add(routed_expert_bytes_per_rank)
                .saturating_add(engram_shard_bytes_per_rank)
                .saturating_add(kv_cache_budget_bytes)
                .saturating_add(activation_workspace_bytes);

            let max_usable = spec
                .hbm_per_gpu_bytes
                .saturating_sub(spec.safety_headroom_bytes);
            if total_allocated > max_usable {
                return Err(SingularityError::OutOfMemory {
                    requested_bytes: total_allocated as usize,
                    available_bytes: max_usable as usize,
                });
            }

            ranks.push(RankPlacementPlan {
                rank,
                device_ordinal: rank,
                backbone_expert_range: (rank * experts_per_rank, (rank + 1) * experts_per_rank),
                dspark_expert_range: (
                    rank * dspark_experts_per_rank,
                    (rank + 1) * dspark_experts_per_rank,
                ),
                engram_row_range: (start_row, end_row),
                dense_replicated_bytes,
                routed_expert_bytes: routed_expert_bytes_per_rank,
                engram_shard_bytes: engram_shard_bytes_per_rank,
                kv_cache_budget_bytes,
                activation_workspace_bytes,
                total_allocated_bytes: total_allocated,
                headroom_bytes: spec.hbm_per_gpu_bytes.saturating_sub(total_allocated),
            });
        }

        Ok(Self {
            spec,
            topology,
            max_active_sequences,
            total_checkpoint_bytes: V41_PINNED_CHECKPOINT_BYTES,
            ranks,
        })
    }

    /// Map a global backbone expert ID `[0, n_routed_experts)` to `(owning_rank, local_expert_index)`.
    pub fn route_backbone_expert(
        &self,
        expert_id: usize,
    ) -> Result<(usize, usize), SingularityError> {
        if expert_id >= self.topology.n_routed_experts {
            return Err(SingularityError::ModelError {
                reason: format!(
                    "Backbone expert ID {expert_id} exceeds {}",
                    self.topology.n_routed_experts
                ),
            });
        }
        let per_rank = self.topology.n_routed_experts / self.spec.world_size;
        Ok((expert_id / per_rank, expert_id % per_rank))
    }

    /// Map a global Engram row ID `[0, engram_rows_per_layer)` to `(owning_rank, local_row_index)`.
    pub fn route_engram_row(&self, row_id: u64) -> Result<(usize, u64), SingularityError> {
        if row_id >= self.topology.engram_rows_per_layer {
            return Err(SingularityError::ModelError {
                reason: format!(
                    "Engram row ID {row_id} exceeds {}",
                    self.topology.engram_rows_per_layer
                ),
            });
        }
        for r in &self.ranks {
            if row_id >= r.engram_row_range.0 && row_id < r.engram_row_range.1 {
                return Ok((r.rank, row_id - r.engram_row_range.0));
            }
        }
        Err(SingularityError::ModelError {
            reason: format!("Unmapped Engram row ID {row_id}"),
        })
    }
}

/// Token dispatch packet for EP=8 all-to-all expert execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertDispatchPacket {
    pub token_index: usize,
    pub global_expert_id: usize,
    pub target_rank: usize,
    pub local_expert_id: usize,
    pub routing_weight: f32,
}

/// Deterministic collective coordinator for tensor-parallel all-reduce and EP=8 expert dispatch/combine.
pub struct CollectiveCoordinator {
    plan: ClusterPlacementPlan,
}

impl CollectiveCoordinator {
    pub fn new(plan: ClusterPlacementPlan) -> Self {
        Self { plan }
    }

    pub fn plan(&self) -> &ClusterPlacementPlan {
        &self.plan
    }

    /// Deterministic ring-ordered FP32 all-reduce sum across `world_size` rank buffers.
    pub fn all_reduce_sum(&self, rank_buffers: &[Vec<f32>]) -> Result<Vec<f32>, SingularityError> {
        if rank_buffers.len() != self.plan.spec.world_size || rank_buffers.is_empty() {
            return Err(SingularityError::HardwareError {
                detail: format!(
                    "Expected {} rank buffers for all_reduce_sum, got {}",
                    self.plan.spec.world_size,
                    rank_buffers.len()
                ),
            });
        }
        let len = rank_buffers[0].len();
        if rank_buffers.iter().any(|b| b.len() != len) {
            return Err(SingularityError::HardwareError {
                detail: "Mismatched buffer lengths across ranks in all_reduce_sum".into(),
            });
        }
        let mut reduced = vec![0.0f32; len];
        for buf in rank_buffers {
            for (dst, &val) in reduced.iter_mut().zip(buf) {
                if !val.is_finite() {
                    return Err(SingularityError::HardwareError {
                        detail: "Non-finite value encountered in all_reduce_sum".into(),
                    });
                }
                *dst += val;
            }
        }
        Ok(reduced)
    }

    /// Partition routed `(expert_id, weight)` pairs per token into per-rank dispatch queues.
    pub fn dispatch_tokens_to_experts(
        &self,
        token_routes: &[Vec<(usize, f32)>],
    ) -> Result<Vec<Vec<ExpertDispatchPacket>>, SingularityError> {
        let mut per_rank = vec![Vec::new(); self.plan.spec.world_size];
        for (token_index, routes) in token_routes.iter().enumerate() {
            for &(global_expert_id, routing_weight) in routes {
                if !routing_weight.is_finite() {
                    return Err(SingularityError::HardwareError {
                        detail: "Non-finite expert routing weight".into(),
                    });
                }
                let (target_rank, local_expert_id) =
                    self.plan.route_backbone_expert(global_expert_id)?;
                per_rank[target_rank].push(ExpertDispatchPacket {
                    token_index,
                    global_expert_id,
                    target_rank,
                    local_expert_id,
                    routing_weight,
                });
            }
        }
        Ok(per_rank)
    }

    /// Combine expert outputs from all ranks back into `[num_tokens][hidden_size]` in deterministic rank order.
    pub fn combine_expert_outputs(
        &self,
        num_tokens: usize,
        hidden_size: usize,
        rank_outputs: &[Vec<(ExpertDispatchPacket, Vec<f32>)>],
    ) -> Result<Vec<Vec<f32>>, SingularityError> {
        if rank_outputs.len() != self.plan.spec.world_size || hidden_size == 0 {
            return Err(SingularityError::HardwareError {
                detail: "Invalid rank count or hidden_size in combine_expert_outputs".into(),
            });
        }
        let mut combined = vec![vec![0.0f32; hidden_size]; num_tokens];
        for rank_items in rank_outputs {
            for (packet, output) in rank_items {
                if packet.token_index >= num_tokens || output.len() != hidden_size {
                    return Err(SingularityError::HardwareError {
                        detail: "Invalid token index or expert output dimension".into(),
                    });
                }
                let dst = &mut combined[packet.token_index];
                for (d, &val) in dst.iter_mut().zip(output) {
                    *d += val * packet.routing_weight;
                }
            }
        }
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_8x_b200_cluster_fits_v41_checkpoint_with_headroom() {
        let spec = B200ClusterSpec::default();
        let topology = V41ClusterTopology::default();
        let plan = ClusterPlacementPlan::plan_v41_cluster(spec.clone(), topology, 32).unwrap();
        assert_eq!(plan.ranks.len(), 8);
        for (i, r) in plan.ranks.iter().enumerate() {
            assert_eq!(r.rank, i);
            assert_eq!(r.backbone_expert_range, (i * 48, (i + 1) * 48));
            assert_eq!(r.dspark_expert_range, (i * 16, (i + 1) * 16));
            assert!(r.total_allocated_bytes < spec.hbm_per_gpu_bytes);
            assert!(r.headroom_bytes >= spec.safety_headroom_bytes);
        }
        assert_eq!(plan.route_backbone_expert(0).unwrap(), (0, 0));
        assert_eq!(plan.route_backbone_expert(47).unwrap(), (0, 47));
        assert_eq!(plan.route_backbone_expert(48).unwrap(), (1, 0));
        assert_eq!(plan.route_backbone_expert(383).unwrap(), (7, 47));
        assert!(plan.route_backbone_expert(384).is_err());

        let (rank_last, _) = plan.route_engram_row(384_016_681).unwrap();
        assert_eq!(rank_last, 7);
        assert!(plan.route_engram_row(384_016_682).is_err());
    }

    #[test]
    fn rejects_non_blackwell_or_excessive_kv_budget() {
        let bad_sm = B200ClusterSpec {
            compute_capability: (9, 0),
            ..Default::default()
        };
        assert!(
            ClusterPlacementPlan::plan_v41_cluster(bad_sm, V41ClusterTopology::default(), 8)
                .is_err()
        );

        // Requesting 10,000 max-context sequences per rank exceeds 180 GiB HBM3e
        assert!(ClusterPlacementPlan::plan_v41_cluster(
            B200ClusterSpec::default(),
            V41ClusterTopology::default(),
            10_000
        )
        .is_err());
    }

    #[test]
    fn collective_coordinator_dispatches_combines_and_reduces() {
        let plan = ClusterPlacementPlan::plan_v41_cluster(
            B200ClusterSpec::default(),
            V41ClusterTopology::default(),
            8,
        )
        .unwrap();
        let coord = CollectiveCoordinator::new(plan);

        let reduced = coord
            .all_reduce_sum(&(0..8).map(|r| vec![r as f32, 1.0]).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(reduced, vec![28.0, 8.0]);

        let routes = vec![vec![(0, 0.25), (48, 0.75)], vec![(383, 1.0)]];
        let dispatched = coord.dispatch_tokens_to_experts(&routes).unwrap();
        assert_eq!(dispatched[0].len(), 1);
        assert_eq!(dispatched[1].len(), 1);
        assert_eq!(dispatched[7].len(), 1);

        let rank_outputs: Vec<Vec<(ExpertDispatchPacket, Vec<f32>)>> = dispatched
            .into_iter()
            .map(|packets| {
                packets
                    .into_iter()
                    .map(|p| {
                        let v = vec![(p.target_rank + 1) as f32; 4];
                        (p, v)
                    })
                    .collect()
            })
            .collect();
        let combined = coord.combine_expert_outputs(2, 4, &rank_outputs).unwrap();
        // Token 0 routed to rank 0 (val 1.0 * 0.25) and rank 1 (val 2.0 * 0.75) => 1.75
        assert_eq!(combined[0], vec![1.75; 4]);
        // Token 1 routed to rank 7 (val 8.0 * 1.0) => 8.0
        assert_eq!(combined[1], vec![8.0; 4]);
    }
}
