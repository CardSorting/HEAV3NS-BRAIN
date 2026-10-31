use singularity_cache::PagedBlockManager;
use singularity_core::{Sequence, SingularityError};
use std::sync::Arc;
use tracing::info;

pub struct LcfsSchedulerResult {
    pub scheduled: Vec<Sequence>,
    pub preempted: Vec<Sequence>,
    pub unallocated: Vec<Sequence>,
}

/// LCFS (Last-Come-First-Served) Decode Prioritization Engine.
/// Matches the verified code reality of DeepInfer (csrc/engine/scheduler.hpp#L622-L646):
/// 1. Sorts running sequences descending by output length: `running[0]` has the longest output, `running[N-1]` the shortest.
/// 2. Iterates from r-1 (shortest output first) to schedule newest requests.
/// 3. If KV cache block allocation fails, preempts `running[l]` (`l = 0`, the longest running request).
pub fn schedule_decode_lcfs(
    mut running: Vec<Sequence>,
    max_batch_size: usize,
    block_manager: &Arc<PagedBlockManager>,
) -> Result<LcfsSchedulerResult, SingularityError> {
    if running.is_empty() {
        return Ok(LcfsSchedulerResult {
            scheduled: Vec::new(),
            preempted: Vec::new(),
            unallocated: Vec::new(),
        });
    }

    // Sort descending by output length: running[0] is longest, running[last] is shortest
    running.sort_by_key(|b| std::cmp::Reverse(b.output_len()));

    let mut l = 0;
    let mut r = running.len();
    let mut scheduled = Vec::new();
    let mut preempted = Vec::new();

    while l < r && scheduled.len() < max_batch_size {
        // Need at least 1 block if sequence needs to grow beyond its current block capacity
        let seq = &running[r - 1];
        let current_tokens = seq.total_tokens();
        let allocated_capacity = seq.allocated_block_ids.len() * block_manager.block_size();
        let needs_block = current_tokens + 1 > allocated_capacity;

        if needs_block {
            // Check if we need to preempt victim sequence
            while !block_manager.can_allocate(1) && r - l > 1 {
                // Victim is running[l] — the longest output sequence!
                let mut victim = running[l].clone();
                info!(
                    "LCFS Preemption: Evicting sequence ID {} (output_len={}) to reclaim blocks for sequence ID {} (output_len={})",
                    victim.id, victim.output_len(), seq.id, seq.output_len()
                );
                // Free victim's allocated blocks back to manager
                let _ = block_manager.free_blocks(&victim.allocated_block_ids);
                victim.allocated_block_ids.clear();
                victim.num_cached_tokens = 0;
                victim.status = singularity_core::SequenceStatus::Preempted;
                preempted.push(victim);
                l += 1;
            }

            if block_manager.can_allocate(1) {
                // Allocate block for the scheduled sequence
                let new_block = block_manager.allocate_single()?;
                let mut scheduled_seq = running[r - 1].clone();
                scheduled_seq.allocated_block_ids.push(new_block);
                scheduled.push(scheduled_seq);
            } else {
                // Cannot allocate even after preempting all available victims:
                // Preempt this sequence as well to avoid deadlocking memory
                let mut victim = running[r - 1].clone();
                info!(
                    "LCFS Preemption: Evicting sequence ID {} (output_len={}) due to lack of free blocks",
                    victim.id, victim.output_len()
                );
                let _ = block_manager.free_blocks(&victim.allocated_block_ids);
                victim.allocated_block_ids.clear();
                victim.num_cached_tokens = 0;
                victim.status = singularity_core::SequenceStatus::Preempted;
                preempted.push(victim);
            }
        } else {
            // Already has enough block capacity for the next token
            scheduled.push(running[r - 1].clone());
        }

        r -= 1;
    }

    // Any remaining sequences between l and r were neither scheduled nor preempted
    let unallocated = if l < r {
        running[l..r].to_vec()
    } else {
        Vec::new()
    };

    Ok(LcfsSchedulerResult {
        scheduled,
        preempted,
        unallocated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use singularity_core::{SamplingParams, Sequence};

    #[test]
    fn test_lcfs_preemption_updates_status_and_clears_blocks() {
        let block_manager = Arc::new(PagedBlockManager::new(2, 16));
        let mut s1 = Sequence::new(
            1,
            "long".to_string(),
            vec![1; 16],
            SamplingParams::default(),
        );
        s1.output_tokens = vec![10; 32];
        s1.allocated_block_ids = block_manager.allocate_blocks(2).unwrap();

        let mut s2 = Sequence::new(
            2,
            "short".to_string(),
            vec![2; 16],
            SamplingParams::default(),
        );
        s2.output_tokens = vec![20; 1]; // needs block

        let res = schedule_decode_lcfs(vec![s2, s1], 4, &block_manager).unwrap();
        assert_eq!(res.preempted.len(), 1);
        assert_eq!(res.preempted[0].id, 1);
        assert_eq!(
            res.preempted[0].status,
            singularity_core::SequenceStatus::Preempted
        );
        assert!(res.preempted[0].allocated_block_ids.is_empty());
    }
}
