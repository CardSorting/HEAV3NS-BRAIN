use parking_lot::Mutex;
use singularity_core::{SingularityError, TokenBlock};
use singularity_cuda::{GpuBackend, GpuBuffer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct PagedBlockManager {
    total_blocks: usize,
    block_size: usize,
    free_blocks: Mutex<Vec<usize>>,
    allocated_count: AtomicUsize,
    allocated_bitmap: Mutex<Vec<u64>>,
    blocks: Vec<Mutex<TokenBlock>>,
    device_buffer: Option<GpuBuffer>,
    gpu_backend: Option<Arc<dyn GpuBackend>>,
    bytes_per_block: usize,
}

impl PagedBlockManager {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        let mut free = Vec::with_capacity(total_blocks);
        let mut blocks = Vec::with_capacity(total_blocks);
        let num_words = total_blocks.div_ceil(64);

        for i in 0..total_blocks {
            free.push(i);
            blocks.push(Mutex::new(TokenBlock::new(i, i)));
        }

        Self {
            total_blocks,
            block_size,
            free_blocks: Mutex::new(free),
            allocated_count: AtomicUsize::new(0),
            allocated_bitmap: Mutex::new(vec![0u64; num_words]),
            blocks,
            device_buffer: None,
            gpu_backend: None,
            bytes_per_block: 0,
        }
    }

    pub fn with_device_backing(
        total_blocks: usize,
        block_size: usize,
        backend: Arc<dyn GpuBackend>,
        bytes_per_token: usize,
    ) -> Result<Self, SingularityError> {
        let bytes_per_block = block_size * bytes_per_token;
        let total_bytes = total_blocks * bytes_per_block;
        let device_buffer = backend.allocate(total_bytes)?;

        let mut free = Vec::with_capacity(total_blocks);
        let mut blocks = Vec::with_capacity(total_blocks);
        let num_words = total_blocks.div_ceil(64);

        for i in 0..total_blocks {
            free.push(i);
            blocks.push(Mutex::new(TokenBlock::new(i, i)));
        }

        Ok(Self {
            total_blocks,
            block_size,
            free_blocks: Mutex::new(free),
            allocated_count: AtomicUsize::new(0),
            allocated_bitmap: Mutex::new(vec![0u64; num_words]),
            blocks,
            device_buffer: Some(device_buffer),
            gpu_backend: Some(backend),
            bytes_per_block,
        })
    }

    pub fn device_buffer(&self) -> Option<&GpuBuffer> {
        self.device_buffer.as_ref()
    }

    pub fn bytes_per_block(&self) -> usize {
        self.bytes_per_block
    }

    pub fn write_block_data(
        &self,
        block_id: usize,
        offset_in_block: usize,
        data: &[u8],
    ) -> Result<(), SingularityError> {
        let backend = self
            .gpu_backend
            .as_ref()
            .ok_or_else(|| SingularityError::CacheError {
                detail: "No GPU backend attached to PagedBlockManager".to_string(),
            })?;
        let device_buffer =
            self.device_buffer
                .as_ref()
                .ok_or_else(|| SingularityError::CacheError {
                    detail: "No device buffer allocated for PagedBlockManager".to_string(),
                })?;

        if block_id >= self.total_blocks {
            return Err(SingularityError::CacheError {
                detail: format!("Block ID {block_id} out of bounds"),
            });
        }

        let global_offset = block_id * self.bytes_per_block + offset_in_block;
        if global_offset + data.len() > device_buffer.size_bytes {
            return Err(SingularityError::CacheError {
                detail: "Write exceeds device buffer capacity".to_string(),
            });
        }

        backend.copy_htod_offset(device_buffer, global_offset, data)?;
        Ok(())
    }

    pub fn read_block_data(
        &self,
        block_id: usize,
        offset_in_block: usize,
        out: &mut [u8],
    ) -> Result<(), SingularityError> {
        let backend = self
            .gpu_backend
            .as_ref()
            .ok_or_else(|| SingularityError::CacheError {
                detail: "No GPU backend attached to PagedBlockManager".to_string(),
            })?;
        let device_buffer =
            self.device_buffer
                .as_ref()
                .ok_or_else(|| SingularityError::CacheError {
                    detail: "No device buffer allocated for PagedBlockManager".to_string(),
                })?;

        if block_id >= self.total_blocks {
            return Err(SingularityError::CacheError {
                detail: format!("Block ID {block_id} out of bounds"),
            });
        }

        let global_offset = block_id * self.bytes_per_block + offset_in_block;
        if global_offset + out.len() > device_buffer.size_bytes {
            return Err(SingularityError::CacheError {
                detail: "Read exceeds device buffer capacity".to_string(),
            });
        }

        backend.copy_dtoh_offset(device_buffer, global_offset, out)?;
        Ok(())
    }

    #[inline]
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[inline]
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.lock().len()
    }

    #[inline]
    pub fn allocated_block_count(&self) -> usize {
        self.allocated_count.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn can_allocate(&self, needed: usize) -> bool {
        self.free_block_count() >= needed
    }

    pub fn is_block_allocated(&self, block_id: usize) -> bool {
        if block_id >= self.total_blocks {
            return false;
        }
        let word_idx = block_id / 64;
        let bit_idx = block_id % 64;
        let bitmap = self.allocated_bitmap.lock();
        if word_idx < bitmap.len() {
            (bitmap[word_idx] & (1u64 << bit_idx)) != 0
        } else {
            false
        }
    }

    pub fn is_block_pinned(&self, block_id: usize) -> bool {
        if block_id >= self.total_blocks {
            return false;
        }
        let block = self.blocks[block_id].lock();
        block.is_pinned
    }

    pub fn allocate_single(&self) -> Result<usize, SingularityError> {
        let mut free = self.free_blocks.lock();
        if let Some(block_id) = free.pop() {
            {
                let mut bitmap = self.allocated_bitmap.lock();
                let word_idx = block_id / 64;
                let bit_idx = block_id % 64;
                bitmap[word_idx] |= 1u64 << bit_idx;
            }
            self.allocated_count.fetch_add(1, Ordering::SeqCst);
            let mut block = self.blocks[block_id].lock();
            block.ref_count = 1;
            block.is_pinned = false;
            block.tokens.clear();
            Ok(block_id)
        } else {
            Err(SingularityError::BlockAllocationFailed { needed: 1, free: 0 })
        }
    }

    pub fn allocate_blocks(&self, count: usize) -> Result<Vec<usize>, SingularityError> {
        let mut free = self.free_blocks.lock();
        if free.len() < count {
            return Err(SingularityError::BlockAllocationFailed {
                needed: count,
                free: free.len(),
            });
        }

        let mut allocated = Vec::with_capacity(count);
        {
            let mut bitmap = self.allocated_bitmap.lock();
            for _ in 0..count {
                let id = free.pop().unwrap();
                let word_idx = id / 64;
                let bit_idx = id % 64;
                bitmap[word_idx] |= 1u64 << bit_idx;

                let mut block = self.blocks[id].lock();
                block.ref_count = 1;
                block.is_pinned = false;
                block.tokens.clear();
                allocated.push(id);
            }
        }

        self.allocated_count.fetch_add(count, Ordering::SeqCst);
        Ok(allocated)
    }

    pub fn pin_block(&self, block_id: usize) -> Result<(), SingularityError> {
        if block_id >= self.total_blocks {
            return Err(SingularityError::CacheError {
                detail: format!("Invalid block_id {block_id} out of range"),
            });
        }
        let mut block = self.blocks[block_id].lock();
        block.ref_count += 1;
        block.is_pinned = true;
        Ok(())
    }

    pub fn unpin_block(&self, block_id: usize) -> Result<bool, SingularityError> {
        if block_id >= self.total_blocks {
            return Err(SingularityError::CacheError {
                detail: format!("Invalid block_id {block_id} out of range"),
            });
        }
        let mut block = self.blocks[block_id].lock();
        if block.ref_count > 0 {
            block.ref_count -= 1;
            block.is_pinned = false;
            if block.ref_count == 0 {
                // If it's not prefix cached, free it
                if !block.is_prefix_cached {
                    drop(block);
                    self.free_block(block_id)?;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    pub fn free_block(&self, block_id: usize) -> Result<(), SingularityError> {
        if block_id >= self.total_blocks {
            return Err(SingularityError::CacheError {
                detail: format!("Invalid block_id {block_id} out of range"),
            });
        }

        {
            let mut bitmap = self.allocated_bitmap.lock();
            let word_idx = block_id / 64;
            let bit_idx = block_id % 64;
            if (bitmap[word_idx] & (1u64 << bit_idx)) == 0 {
                return Err(SingularityError::CacheError {
                    detail: format!("Attempted double free on unallocated block_id {block_id}"),
                });
            }
            bitmap[word_idx] &= !(1u64 << bit_idx);
        }

        {
            let mut block = self.blocks[block_id].lock();
            block.reset();
        }

        let mut free = self.free_blocks.lock();
        free.push(block_id);
        self.allocated_count.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    pub fn free_blocks(&self, block_ids: &[usize]) -> Result<(), SingularityError> {
        for &id in block_ids {
            self.free_block(id)?;
        }
        Ok(())
    }

    pub fn set_block_tokens(
        &self,
        block_id: usize,
        tokens: &[u32],
    ) -> Result<(), SingularityError> {
        if block_id >= self.total_blocks {
            return Err(SingularityError::CacheError {
                detail: format!("Invalid block_id {block_id}"),
            });
        }
        let mut block = self.blocks[block_id].lock();
        block.tokens = tokens.to_vec();
        Ok(())
    }

    pub fn mark_prefix_cached(&self, block_id: usize, cached: bool) {
        if block_id < self.total_blocks {
            let mut block = self.blocks[block_id].lock();
            block.is_prefix_cached = cached;
        }
    }
}

impl Drop for PagedBlockManager {
    fn drop(&mut self) {
        if let (Some(buf), Some(backend)) = (&self.device_buffer, &self.gpu_backend) {
            let _ = backend.free(buf);
        }
    }
}

pub struct BlockGuard {
    block_manager: Arc<PagedBlockManager>,
    pub block_ids: Vec<usize>,
}

impl BlockGuard {
    pub fn new(block_manager: Arc<PagedBlockManager>, block_ids: Vec<usize>) -> Self {
        Self {
            block_manager,
            block_ids,
        }
    }
}

impl Drop for BlockGuard {
    fn drop(&mut self) {
        for &id in &self.block_ids {
            let _ = self.block_manager.unpin_block(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use singularity_cuda::MockGpuBackend;

    #[test]
    fn test_bitset_allocation_and_double_free_rejection() {
        let mgr = PagedBlockManager::new(32, 16);
        let b1 = mgr.allocate_single().unwrap();
        assert!(mgr.is_block_allocated(b1));

        mgr.pin_block(b1).unwrap();
        assert!(mgr.is_block_pinned(b1));

        mgr.unpin_block(b1).unwrap();
        assert!(!mgr.is_block_pinned(b1));

        // Freeing once succeeds
        mgr.free_block(b1).unwrap();
        assert!(!mgr.is_block_allocated(b1));

        // Freeing again rejects double free!
        let err = mgr.free_block(b1).unwrap_err();
        match err {
            SingularityError::CacheError { detail } => {
                assert!(detail.contains("double free"));
            }
            _ => panic!("Expected CacheError with double free"),
        }
    }

    #[test]
    fn test_paged_block_manager_device_backing() {
        let backend = Arc::new(MockGpuBackend::new_h200());
        let mgr = PagedBlockManager::with_device_backing(8, 16, backend.clone(), 64).unwrap();

        assert!(mgr.device_buffer().is_some());
        assert_eq!(mgr.bytes_per_block(), 16 * 64);

        let data = vec![42u8; 128];
        mgr.write_block_data(2, 0, &data).unwrap();

        let mut read = vec![0u8; 128];
        mgr.read_block_data(2, 0, &mut read).unwrap();
        assert_eq!(read, data);
    }
}
