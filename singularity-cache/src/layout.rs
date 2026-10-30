use singularity_core::{CacheMemoryFormat, SingularityError};

pub const FP4_BYTES_PER_TOKEN: usize = 288;
pub const MXFP8_BYTES_PER_TOKEN: usize = 528;
pub const MEMORY_BUS_ALIGNMENT: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalCacheStride {
    pub format: CacheMemoryFormat,
    pub bytes_per_token: usize,
    pub block_size: usize,
    pub block_bytes: usize,
}

impl PhysicalCacheStride {
    pub fn new(format: CacheMemoryFormat, block_size: usize) -> Self {
        let bytes_per_token = match format {
            CacheMemoryFormat::Fp4 => FP4_BYTES_PER_TOKEN,
            CacheMemoryFormat::MxFp8 => MXFP8_BYTES_PER_TOKEN,
        };
        // Align block bytes to 32-byte memory bus boundary
        let raw_block_bytes = bytes_per_token * block_size;
        let block_bytes =
            (raw_block_bytes + MEMORY_BUS_ALIGNMENT - 1) & !(MEMORY_BUS_ALIGNMENT - 1);

        Self {
            format,
            bytes_per_token,
            block_size,
            block_bytes,
        }
    }

    /// Token-interleaved offset calculation:
    /// In DeepInfer V4.1 format, data and scales are interleaved per token to preserve 32-byte
    /// alignment and burst read efficiency.
    #[inline]
    pub fn token_byte_offset(&self, block_index: usize, token_within_block: usize) -> usize {
        block_index * self.block_bytes + token_within_block * self.bytes_per_token
    }

    #[inline]
    pub fn scale_byte_offset(&self, block_index: usize, token_within_block: usize) -> usize {
        let base = self.token_byte_offset(block_index, token_within_block);
        match self.format {
            CacheMemoryFormat::Fp4 => base + 256, // 256 bytes data + 32 bytes E4M3 scales
            CacheMemoryFormat::MxFp8 => base + 512, // 512 bytes data + 16 bytes UE8M0 scales
        }
    }
}

/// SWA (Sliding Window Attention) Ring Buffer with backtracking boundary protection.
#[derive(Debug, Clone)]
pub struct SwaRingBuffer {
    window_size: usize,
    head: usize,
    count: usize,
    buffer: Vec<u32>,
}

impl SwaRingBuffer {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            head: 0,
            count: 0,
            buffer: vec![0; window_size],
        }
    }

    pub fn push(&mut self, token: u32) {
        self.buffer[self.head] = token;
        self.head = (self.head + 1) % self.window_size;
        if self.count < self.window_size {
            self.count += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get_recent_tokens(&self) -> Vec<u32> {
        let mut result = Vec::with_capacity(self.count);
        if self.count < self.window_size {
            result.extend_from_slice(&self.buffer[0..self.count]);
        } else {
            // Read from head to end, then 0 to head
            result.extend_from_slice(&self.buffer[self.head..self.window_size]);
            result.extend_from_slice(&self.buffer[0..self.head]);
        }
        result
    }
}

/// Disaggregated StateBlock Partitioning:
/// Explicitly splits memory into:
/// - Stored Region: SWA window required for persistence and Prefill Engine (PE) -> Completion Engine (CE) transfer.
/// - Dropped Region: YOCO consumer SWA and compression states, discarded and recomputed on the fly.
#[derive(Debug, Clone)]
pub struct StateBlockPartition {
    pub stored_region_blocks: Vec<usize>,
    pub dropped_region_blocks: Vec<usize>,
    pub transfer_bytes: usize,
}

impl StateBlockPartition {
    pub fn partition_for_disaggregation(
        all_blocks: &[usize],
        swa_window_size: usize,
        block_size: usize,
        stride: &PhysicalCacheStride,
    ) -> Result<Self, SingularityError> {
        let needed_swa_blocks = swa_window_size.div_ceil(block_size);
        let stored_count = needed_swa_blocks.min(all_blocks.len());

        let stored_region_blocks = all_blocks[all_blocks.len() - stored_count..].to_vec();
        let dropped_region_blocks = all_blocks[0..all_blocks.len() - stored_count].to_vec();

        let transfer_bytes = stored_region_blocks.len() * stride.block_bytes;

        Ok(Self {
            stored_region_blocks,
            dropped_region_blocks,
            transfer_bytes,
        })
    }
}
