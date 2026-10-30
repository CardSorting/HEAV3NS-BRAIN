use crate::backend::{GpuBackend, GpuBuffer, GpuEvent, GpuStream};
use crate::hardware::DeviceArchitecture;
use parking_lot::RwLock;
use singularity_core::SingularityError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct MockGpuBackend {
    arch: DeviceArchitecture,
    total_bytes: usize,
    allocated_bytes: AtomicUsize,
    high_watermark: AtomicUsize,
    buffers: RwLock<HashMap<u64, GpuBuffer>>,
    memory: RwLock<HashMap<u64, Vec<u8>>>,
    next_virtual_addr: AtomicUsize,
}

impl MockGpuBackend {
    pub fn new(arch: DeviceArchitecture) -> Self {
        let total_bytes = arch.hbm_bytes();
        Self {
            arch,
            total_bytes,
            allocated_bytes: AtomicUsize::new(0),
            high_watermark: AtomicUsize::new(0),
            buffers: RwLock::new(HashMap::new()),
            memory: RwLock::new(HashMap::new()),
            next_virtual_addr: AtomicUsize::new(0x7f00_0000_0000),
        }
    }

    pub fn new_b200() -> Self {
        Self::new(DeviceArchitecture::BlackwellSm100 {
            hbm_bytes: 180 * 1024 * 1024 * 1024,
            nvlink_gen: 5,
            has_native_fp4: true,
            tma_version: 2,
        })
    }

    pub fn new_h200() -> Self {
        Self::new(DeviceArchitecture::HopperSm90 {
            hbm_bytes: 141 * 1024 * 1024 * 1024,
            nvlink_gen: 4,
            has_native_fp4: false,
            tma_version: 1,
        })
    }

    pub fn high_watermark_bytes(&self) -> usize {
        self.high_watermark.load(Ordering::Relaxed)
    }

    pub fn active_allocations(&self) -> usize {
        self.buffers.read().len()
    }
}

impl GpuBackend for MockGpuBackend {
    fn allocate(&self, size_bytes: usize) -> Result<GpuBuffer, SingularityError> {
        // Enforce 32-byte alignment to match Blackwell/Hopper memory bus requirements
        let aligned_size =
            size_bytes
                .checked_add(31)
                .ok_or_else(|| SingularityError::HardwareError {
                    detail: "Mock allocation size overflow".into(),
                })?
                & !31;
        if aligned_size == 0 {
            return Err(SingularityError::HardwareError {
                detail: "Mock allocation size must be nonzero".into(),
            });
        }

        let mut current = self.allocated_bytes.load(Ordering::Relaxed);
        loop {
            if current
                .checked_add(aligned_size)
                .is_none_or(|total| total > self.total_bytes)
            {
                return Err(SingularityError::OutOfMemory {
                    requested_bytes: aligned_size,
                    available_bytes: self.total_bytes.saturating_sub(current),
                });
            }
            match self.allocated_bytes.compare_exchange_weak(
                current,
                current + aligned_size,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }

        let new_allocated = current + aligned_size;
        let mut hw = self.high_watermark.load(Ordering::Relaxed);
        while new_allocated > hw {
            match self.high_watermark.compare_exchange_weak(
                hw,
                new_allocated,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => hw = actual,
            }
        }

        let ptr = self
            .next_virtual_addr
            .fetch_add(aligned_size, Ordering::SeqCst);
        let buffer = GpuBuffer::new(aligned_size, ptr, false);

        self.buffers.write().insert(buffer.id, buffer.clone());
        self.memory
            .write()
            .insert(buffer.id, vec![0u8; aligned_size]);
        Ok(buffer)
    }

    fn free(&self, buffer: &GpuBuffer) -> Result<(), SingularityError> {
        let removed = self.buffers.write().remove(&buffer.id);
        self.memory.write().remove(&buffer.id);
        if let Some(buf) = removed {
            self.allocated_bytes
                .fetch_sub(buf.size_bytes, Ordering::SeqCst);
            Ok(())
        } else {
            Err(SingularityError::HardwareError {
                detail: format!(
                    "Attempted to double free or free unknown GPU buffer ID {}",
                    buffer.id
                ),
            })
        }
    }

    fn copy_htod_offset(
        &self,
        dst: &GpuBuffer,
        dst_offset: usize,
        src_bytes: &[u8],
    ) -> Result<(), SingularityError> {
        let mut mem_guard = self.memory.write();
        let target = mem_guard
            .get_mut(&dst.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: format!("Buffer {} not allocated in mock memory", dst.id),
            })?;
        if dst_offset
            .checked_add(src_bytes.len())
            .is_none_or(|end| end > target.len())
        {
            return Err(SingularityError::HardwareError {
                detail: format!(
                    "Host data offset {} + size {} exceeds buffer capacity {}",
                    dst_offset,
                    src_bytes.len(),
                    target.len()
                ),
            });
        }
        target[dst_offset..dst_offset + src_bytes.len()].copy_from_slice(src_bytes);
        Ok(())
    }

    fn copy_dtoh_offset(
        &self,
        src: &GpuBuffer,
        src_offset: usize,
        dst_bytes: &mut [u8],
    ) -> Result<(), SingularityError> {
        let mem_guard = self.memory.read();
        let source = mem_guard
            .get(&src.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: format!("Buffer {} not allocated in mock memory", src.id),
            })?;
        if src_offset
            .checked_add(dst_bytes.len())
            .is_none_or(|end| end > source.len())
        {
            return Err(SingularityError::HardwareError {
                detail: format!(
                    "Source offset {} + dst size {} exceeds buffer size {}",
                    src_offset,
                    dst_bytes.len(),
                    source.len()
                ),
            });
        }
        dst_bytes.copy_from_slice(&source[src_offset..src_offset + dst_bytes.len()]);
        Ok(())
    }

    fn gemm_f32(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &GpuBuffer,
        lda: usize,
        b: &GpuBuffer,
        ldb: usize,
        beta: f32,
        c: &GpuBuffer,
        ldc: usize,
    ) -> Result<(), SingularityError> {
        let a_vec: Vec<f32>;
        let b_vec: Vec<f32>;
        {
            let mem = self.memory.read();
            let a_bytes = mem
                .get(&a.id)
                .ok_or_else(|| SingularityError::HardwareError {
                    detail: format!("Buffer A {} not found", a.id),
                })?;
            let b_bytes = mem
                .get(&b.id)
                .ok_or_else(|| SingularityError::HardwareError {
                    detail: format!("Buffer B {} not found", b.id),
                })?;
            a_vec = a_bytes
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes(ch.try_into().unwrap()))
                .collect();
            b_vec = b_bytes
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes(ch.try_into().unwrap()))
                .collect();
        }

        let mut mem = self.memory.write();
        let c_bytes = mem
            .get_mut(&c.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: format!("Buffer C {} not found", c.id),
            })?;
        crate::backend::validate_gemm(
            trans_a,
            trans_b,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            a_vec.len() * 4,
            b_vec.len() * 4,
            c_bytes.len(),
        )?;
        let mut c_vec: Vec<f32> = c_bytes
            .chunks_exact(4)
            .map(|ch| f32::from_le_bytes(ch.try_into().unwrap()))
            .collect();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let a_val = if trans_a {
                        let idx = p * lda + i;
                        if idx < a_vec.len() {
                            a_vec[idx]
                        } else {
                            0.0
                        }
                    } else {
                        let idx = i * lda + p;
                        if idx < a_vec.len() {
                            a_vec[idx]
                        } else {
                            0.0
                        }
                    };
                    let b_val = if trans_b {
                        let idx = j * ldb + p;
                        if idx < b_vec.len() {
                            b_vec[idx]
                        } else {
                            0.0
                        }
                    } else {
                        let idx = p * ldb + j;
                        if idx < b_vec.len() {
                            b_vec[idx]
                        } else {
                            0.0
                        }
                    };
                    sum += a_val * b_val;
                }
                let c_idx = i * ldc + j;
                if c_idx < c_vec.len() {
                    let prev = if beta == 0.0 { 0.0 } else { c_vec[c_idx] };
                    c_vec[c_idx] = alpha * sum + beta * prev;
                }
            }
        }

        for (idx, val) in c_vec.iter().enumerate() {
            let start = idx * 4;
            let end = start + 4;
            if end <= c_bytes.len() {
                c_bytes[start..end].copy_from_slice(&val.to_le_bytes());
            }
        }

        Ok(())
    }

    fn create_stream(&self, name: &str) -> Result<Arc<GpuStream>, SingularityError> {
        Ok(Arc::new(GpuStream::new(name)))
    }

    fn create_event(&self) -> Result<Arc<GpuEvent>, SingularityError> {
        Ok(Arc::new(GpuEvent::new()))
    }

    fn destroy_stream(&self, _stream: &GpuStream) -> Result<(), SingularityError> {
        Ok(())
    }
    fn destroy_event(&self, _event: &GpuEvent) -> Result<(), SingularityError> {
        Ok(())
    }

    fn record_event(&self, event: &GpuEvent, _stream: &GpuStream) -> Result<(), SingularityError> {
        event.record();
        Ok(())
    }

    fn wait_event(&self, _stream: &GpuStream, event: &GpuEvent) -> Result<(), SingularityError> {
        if !event.is_recorded() {
            return Err(SingularityError::HardwareError {
                detail: "Mock event has not been recorded".into(),
            });
        }
        Ok(())
    }

    fn synchronize_stream(&self, _stream: &GpuStream) -> Result<(), SingularityError> {
        // Mock synchronization is instantaneous on host CPU
        Ok(())
    }

    fn mem_info(&self) -> (usize, usize) {
        let allocated = self.allocated_bytes.load(Ordering::Relaxed);
        (self.total_bytes.saturating_sub(allocated), self.total_bytes)
    }

    fn architecture(&self) -> &DeviceArchitecture {
        &self.arch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_gpu_backend_allocation_alignment_and_oom() {
        let backend = MockGpuBackend::new(DeviceArchitecture::MockDevice {
            name: "TestGpu".to_string(),
            hbm_bytes: 1024,
            nvlink_gen: 5,
            has_native_fp4: true,
            tma_version: 2,
        });

        // 10 bytes aligns to 32
        let buf = backend.allocate(10).unwrap();
        assert_eq!(buf.size_bytes, 32);

        // Allocating remaining capacity
        let buf2 = backend.allocate(1024 - 32).unwrap();
        assert_eq!(buf2.size_bytes, 1024 - 32);

        // Next allocation must trigger OutOfMemory
        let err = backend.allocate(32).unwrap_err();
        match err {
            SingularityError::OutOfMemory {
                requested_bytes,
                available_bytes,
            } => {
                assert_eq!(requested_bytes, 32);
                assert_eq!(available_bytes, 0);
            }
            _ => panic!("Expected OutOfMemory"),
        }

        // Freeing returns memory
        backend.free(&buf).unwrap();
        let (free, total) = backend.mem_info();
        assert_eq!(free, 32);
        assert_eq!(total, 1024);
    }

    #[test]
    fn test_mock_gpu_backend_htod_dtoh_and_gemm() {
        let backend = MockGpuBackend::new_h200();

        // 2x2 float matrices: 4 floats * 4 bytes = 16 bytes -> 32 aligned
        let buf_a = backend.allocate(16).unwrap();
        let buf_b = backend.allocate(16).unwrap();
        let buf_c = backend.allocate(16).unwrap();

        // A = [[1.0, 2.0], [3.0, 4.0]]
        let a_floats = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut a_bytes = Vec::new();
        for f in a_floats {
            a_bytes.extend_from_slice(&f.to_le_bytes());
        }

        // B = [[5.0, 6.0], [7.0, 8.0]]
        let b_floats = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut b_bytes = Vec::new();
        for f in b_floats {
            b_bytes.extend_from_slice(&f.to_le_bytes());
        }

        // Copy H to D
        backend.copy_htod(&buf_a, &a_bytes).unwrap();
        backend.copy_htod(&buf_b, &b_bytes).unwrap();

        // Verify D to H for buf_a
        let mut read_a = vec![0u8; 16];
        backend.copy_dtoh(&buf_a, &mut read_a).unwrap();
        assert_eq!(read_a, a_bytes);

        // Execute GEMM: C = 1.0 * (A * B) + 0.0 * C
        // A is 2x2 (lda=2), B is 2x2 (ldb=2), C is 2x2 (ldc=2)
        backend
            .gemm_f32(
                false, false, 2, 2, 2, 1.0, &buf_a, 2, &buf_b, 2, 0.0, &buf_c, 2,
            )
            .unwrap();

        // Read C back
        let mut c_bytes = vec![0u8; 16];
        backend.copy_dtoh(&buf_c, &mut c_bytes).unwrap();
        let c_floats: Vec<f32> = c_bytes
            .chunks_exact(4)
            .map(|ch| f32::from_le_bytes(ch.try_into().unwrap()))
            .collect();

        // Expected C = [[19.0, 22.0], [43.0, 50.0]]
        assert_eq!(c_floats, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_mock_gpu_backend_htod_dtoh_offset() {
        let backend = MockGpuBackend::new_h200();
        let buf = backend.allocate(64).unwrap();

        let data1 = vec![1u8, 2, 3, 4];
        let data2 = vec![5u8, 6, 7, 8];

        backend.copy_htod_offset(&buf, 0, &data1).unwrap();
        backend.copy_htod_offset(&buf, 16, &data2).unwrap();

        let mut read1 = vec![0u8; 4];
        let mut read2 = vec![0u8; 4];
        backend.copy_dtoh_offset(&buf, 0, &mut read1).unwrap();
        backend.copy_dtoh_offset(&buf, 16, &mut read2).unwrap();

        assert_eq!(read1, data1);
        assert_eq!(read2, data2);
    }
}
