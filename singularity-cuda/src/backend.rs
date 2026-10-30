use crate::hardware::DeviceArchitecture;
use singularity_core::SingularityError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

static BUFFER_ID_GEN: AtomicU64 = AtomicU64::new(1);
static STREAM_ID_GEN: AtomicU64 = AtomicU64::new(1);
static EVENT_ID_GEN: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct GpuBuffer {
    pub id: u64,
    pub size_bytes: usize,
    pub ptr: usize, // Virtual address pointer
    pub is_pinned: bool,
}

impl GpuBuffer {
    pub fn new(size_bytes: usize, ptr: usize, is_pinned: bool) -> Self {
        Self {
            id: BUFFER_ID_GEN.fetch_add(1, Ordering::Relaxed),
            size_bytes,
            ptr,
            is_pinned,
        }
    }
}

#[derive(Debug)]
pub struct GpuStream {
    pub id: u64,
    pub name: String,
}

impl GpuStream {
    pub fn new(name: &str) -> Self {
        Self {
            id: STREAM_ID_GEN.fetch_add(1, Ordering::Relaxed),
            name: name.to_string(),
        }
    }
}

#[derive(Debug)]
pub struct GpuEvent {
    pub id: u64,
    pub recorded_at: parking_lot::RwLock<Option<Instant>>,
}

impl Default for GpuEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuEvent {
    pub fn new() -> Self {
        Self {
            id: EVENT_ID_GEN.fetch_add(1, Ordering::Relaxed),
            recorded_at: parking_lot::RwLock::new(None),
        }
    }

    pub fn record(&self) {
        let mut guard = self.recorded_at.write();
        *guard = Some(Instant::now());
    }

    pub fn is_recorded(&self) -> bool {
        self.recorded_at.read().is_some()
    }
}

pub trait GpuBackend: Send + Sync {
    fn allocate(&self, size_bytes: usize) -> Result<GpuBuffer, SingularityError>;
    fn free(&self, buffer: &GpuBuffer) -> Result<(), SingularityError>;
    fn copy_htod(&self, dst: &GpuBuffer, src_bytes: &[u8]) -> Result<(), SingularityError> {
        self.copy_htod_offset(dst, 0, src_bytes)
    }
    fn copy_dtoh(&self, src: &GpuBuffer, dst_bytes: &mut [u8]) -> Result<(), SingularityError> {
        self.copy_dtoh_offset(src, 0, dst_bytes)
    }
    fn copy_htod_offset(
        &self,
        dst: &GpuBuffer,
        dst_offset: usize,
        src_bytes: &[u8],
    ) -> Result<(), SingularityError>;
    fn copy_dtoh_offset(
        &self,
        src: &GpuBuffer,
        src_offset: usize,
        dst_bytes: &mut [u8],
    ) -> Result<(), SingularityError>;
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), SingularityError>;
    fn create_stream(&self, name: &str) -> Result<Arc<GpuStream>, SingularityError>;
    fn create_event(&self) -> Result<Arc<GpuEvent>, SingularityError>;
    fn destroy_stream(&self, stream: &GpuStream) -> Result<(), SingularityError>;
    fn destroy_event(&self, event: &GpuEvent) -> Result<(), SingularityError>;
    fn record_event(&self, event: &GpuEvent, stream: &GpuStream) -> Result<(), SingularityError>;
    fn wait_event(&self, stream: &GpuStream, event: &GpuEvent) -> Result<(), SingularityError>;
    fn synchronize_stream(&self, stream: &GpuStream) -> Result<(), SingularityError>;
    fn mem_info(&self) -> (usize, usize); // (free, total)
    fn architecture(&self) -> &DeviceArchitecture;

    fn write_buffer(&self, dst: &GpuBuffer, data: &[u8]) -> Result<(), SingularityError> {
        self.copy_htod(dst, data)
    }

    fn read_buffer(&self, src: &GpuBuffer) -> Result<Vec<u8>, SingularityError> {
        let mut data = vec![0u8; src.size_bytes];
        self.copy_dtoh(src, &mut data)?;
        Ok(data)
    }
}

/// Validate row-major GEMM storage against actual allocations, not caller metadata.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_gemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    a_bytes: usize,
    b_bytes: usize,
    c_bytes: usize,
) -> Result<(), SingularityError> {
    let bad = |detail: &str| SingularityError::HardwareError {
        detail: detail.into(),
    };
    if m == 0
        || n == 0
        || k == 0
        || [m, n, k, lda, ldb, ldc]
            .iter()
            .any(|&v| v > i32::MAX as usize)
    {
        return Err(bad("Invalid GEMM dimensions"));
    }
    let (ar, ac) = if trans_a { (k, m) } else { (m, k) };
    let (br, bc) = if trans_b { (n, k) } else { (k, n) };
    let required = |rows: usize, cols: usize, stride: usize| -> Option<usize> {
        if stride < cols {
            return None;
        }
        (rows - 1)
            .checked_mul(stride)?
            .checked_add(cols)?
            .checked_mul(4)
    };
    for (rows, cols, stride, size) in [
        (ar, ac, lda, a_bytes),
        (br, bc, ldb, b_bytes),
        (m, n, ldc, c_bytes),
    ] {
        if required(rows, cols, stride).is_none_or(|required| required > size) {
            return Err(bad("GEMM matrix stride or buffer size is invalid"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod gemm_validation_tests {
    use super::validate_gemm;
    #[test]
    fn checks_physical_shapes_strides_and_overflow() {
        // A is [2,3], B is [3,4], C is [2,4].
        assert!(validate_gemm(false, false, 2, 4, 3, 3, 4, 4, 24, 48, 32).is_ok());
        // Transposed A physical [3,2], transposed B physical [4,3].
        assert!(validate_gemm(true, true, 2, 4, 3, 2, 3, 4, 24, 48, 32).is_ok());
        assert!(validate_gemm(false, false, 2, 4, 3, 2, 4, 4, 24, 48, 32).is_err());
        assert!(validate_gemm(false, false, 2, 4, 3, 3, 4, 4, 23, 48, 32).is_err());
        assert!(validate_gemm(false, false, 2, 4, 3, 3, 4, 4, 24, 48, 31).is_err());
        assert!(validate_gemm(false, false, usize::MAX, 4, 3, 3, 4, 4, 24, 48, 32).is_err());
        assert!(validate_gemm(false, false, 0, 4, 3, 3, 4, 4, 24, 48, 32).is_err());
    }
}
