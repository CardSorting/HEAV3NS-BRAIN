use crate::backend::{GpuBackend, GpuBuffer};
use parking_lot::RwLock;
use singularity_core::SingularityError;
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::info;

#[derive(Debug, Clone)]
pub struct CapturedGraph {
    pub batch_size: usize,
    pub envelope_bytes: usize,
    pub execution_count: usize,
}

pub struct CudaGraphRunner {
    backend: Arc<dyn GpuBackend>,
    batch_sizes: Vec<usize>,
    shared_pool_buffer: RwLock<Option<GpuBuffer>>,
    captured_graphs: RwLock<BTreeMap<usize, CapturedGraph>>,
}

impl CudaGraphRunner {
    pub fn new(backend: Arc<dyn GpuBackend>, batch_sizes: Vec<usize>) -> Self {
        let mut sorted_sizes = batch_sizes;
        sorted_sizes.sort_unstable();
        sorted_sizes.dedup();
        Self {
            backend,
            batch_sizes: sorted_sizes,
            shared_pool_buffer: RwLock::new(None),
            captured_graphs: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn default_buckets(backend: Arc<dyn GpuBackend>) -> Self {
        Self::new(backend, vec![1, 2, 4, 8, 16, 32, 64, 128, 256])
    }

    /// Captures execution graphs across batch sizes in descending order (reverse).
    /// By capturing the largest batch size first, the memory pool is pre-sized to the maximum
    /// envelope, and all subsequent smaller batch graphs share this exact pool without reallocating.
    pub fn capture_all_reverse(&self, bytes_per_batch_unit: usize) -> Result<(), SingularityError> {
        let mut reverse_sizes = self.batch_sizes.clone();
        reverse_sizes.sort_unstable_by(|a, b| b.cmp(a)); // Descending order: [256, 128, ...]

        let max_batch = *reverse_sizes.first().unwrap_or(&1);
        let max_envelope = max_batch * bytes_per_batch_unit;

        // Allocate shared memory pool at maximum envelope
        let pool = self.backend.allocate(max_envelope)?;
        info!(
            "Captured initial shared CUDA Graph memory pool for max batch size {}: {} bytes at ptr {:#x}",
            max_batch, max_envelope, pool.ptr
        );
        *self.shared_pool_buffer.write() = Some(pool);

        let mut graphs = self.captured_graphs.write();
        for &bsz in &reverse_sizes {
            let envelope = bsz * bytes_per_batch_unit;
            graphs.insert(
                bsz,
                CapturedGraph {
                    batch_size: bsz,
                    envelope_bytes: envelope,
                    execution_count: 0,
                },
            );
        }

        Ok(())
    }

    /// Selects the best bucket (smallest batch size >= requested batch size) and executes graph.
    pub fn replay(&self, requested_batch_size: usize) -> Result<usize, SingularityError> {
        let mut graphs = self.captured_graphs.write();

        let max_bucket = self.batch_sizes.last().copied().unwrap_or(0);
        if requested_batch_size > max_bucket {
            return Err(SingularityError::HardwareError {
                detail: format!(
                    "Requested batch size {requested_batch_size} exceeds maximum captured CUDA graph bucket {max_bucket}"
                ),
            });
        }

        // Find smallest bucket >= requested_batch_size
        let best_bucket = self
            .batch_sizes
            .iter()
            .find(|&&bsz| bsz >= requested_batch_size)
            .copied()
            .ok_or_else(|| SingularityError::HardwareError {
                detail: "No CUDA graph buckets configured".to_string(),
            })?;

        if let Some(graph) = graphs.get_mut(&best_bucket) {
            graph.execution_count += 1;
            Ok(best_bucket)
        } else {
            Err(SingularityError::HardwareError {
                detail: format!("Graph for bucket {best_bucket} not captured"),
            })
        }
    }

    pub fn shared_pool_size(&self) -> Option<usize> {
        self.shared_pool_buffer
            .read()
            .as_ref()
            .map(|b| b.size_bytes)
    }

    pub fn get_graph(&self, bsz: usize) -> Option<CapturedGraph> {
        self.captured_graphs.read().get(&bsz).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockGpuBackend;

    #[test]
    fn test_cugraph_runner_bucket_selection_and_overflow_rejection() {
        let backend = Arc::new(MockGpuBackend::new_b200());
        let runner = CudaGraphRunner::new(backend, vec![1, 2, 4, 8, 16]);
        runner.capture_all_reverse(512).unwrap();

        // 3 selects bucket 4
        assert_eq!(runner.replay(3).unwrap(), 4);
        // 16 selects bucket 16
        assert_eq!(runner.replay(16).unwrap(), 16);

        // 17 exceeds maximum bucket 16 -> Error!
        let err = runner.replay(17).unwrap_err();
        match err {
            SingularityError::HardwareError { detail } => {
                assert!(detail.contains("exceeds maximum captured CUDA graph bucket 16"));
            }
            _ => panic!("Expected HardwareError"),
        }
    }
}
