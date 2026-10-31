#[cfg(feature = "cuda")]
use {
    crate::backend::{GpuBackend, GpuBuffer, GpuEvent, GpuStream},
    crate::hardware::DeviceArchitecture,
    cudarc::cublas::{CudaBlas, Gemm, GemmConfig},
    cudarc::driver::{
        sys, CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync, LaunchConfig,
    },
    parking_lot::{Mutex, RwLock},
    singularity_core::SingularityError,
    std::collections::HashMap,
    std::sync::Arc,
};

#[cfg(feature = "cuda")]
struct DeviceF32Ptr {
    ptr: sys::CUdeviceptr,
    len: usize,
}

#[cfg(feature = "cuda")]
impl DeviceSlice<f32> for DeviceF32Ptr {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(feature = "cuda")]
impl DevicePtr<f32> for DeviceF32Ptr {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
}

#[cfg(feature = "cuda")]
impl DevicePtrMut<f32> for DeviceF32Ptr {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.ptr
    }
}

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
    arch: DeviceArchitecture,
    allocations: RwLock<HashMap<u64, CudaSlice<u8>>>,
    streams: RwLock<HashMap<u64, usize>>,
    events: RwLock<HashMap<u64, (usize, bool)>>,
    v41_kernel_init: Mutex<()>,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn new(device_id: usize, arch: DeviceArchitecture) -> Result<Self, SingularityError> {
        let device = CudaDevice::new(device_id).map_err(|e| SingularityError::HardwareError {
            detail: format!("Failed to initialize CUDA device {device_id}: {e:?}"),
        })?;
        let blas = Arc::new(CudaBlas::new(device.clone()).map_err(|e| {
            SingularityError::HardwareError {
                detail: format!("Failed to initialize cuBLAS on device {device_id}: {e:?}"),
            }
        })?);

        Ok(Self {
            device,
            blas,
            arch,
            allocations: RwLock::new(HashMap::new()),
            streams: RwLock::new(HashMap::new()),
            events: RwLock::new(HashMap::new()),
            v41_kernel_init: Mutex::new(()),
        })
    }

    /// Correctness-first CUDA V4.1 single-vector GEMM. `input` and `output`
    /// contain BF16 bytes; weights and scales use original checkpoint layouts.
    /// This synchronized implementation is not optimized and has not yet been
    /// validated on B200. It does not implement full model execution.
    #[allow(clippy::too_many_arguments)]
    pub fn v41_quantized_matvec(
        &self,
        weights: &GpuBuffer,
        scales: &GpuBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
        cols: usize,
        expert: bool,
    ) -> Result<(), SingularityError> {
        let fail = |detail: String| SingularityError::HardwareError { detail };
        let layout = crate::v41_layout::V41MatvecLayout::new(rows, cols, expert)?;
        if [weights.id, scales.id, input.id].contains(&output.id) {
            return Err(fail("V4.1 GEMM output cannot alias an input".into()));
        }
        self.device
            .bind_to_thread()
            .map_err(|e| fail(format!("CUDA context binding failed: {e:?}")))?;
        {
            let _init = self.v41_kernel_init.lock();
            if !self.device.has_func("v41_quant", "v41_matvec") {
                let preferred_arch = match self.arch {
                    DeviceArchitecture::BlackwellSm100 { .. } => "compute_100",
                    _ => "compute_90",
                };
                let compile = |arch: &'static str| {
                    let opts = cudarc::nvrtc::CompileOptions {
                        arch: Some(arch),
                        fmad: Some(false),
                        ..Default::default()
                    };
                    cudarc::nvrtc::compile_ptx_with_opts(include_str!("kernels/v41_quant.cu"), opts)
                };
                let ptx = compile(preferred_arch)
                    .or_else(|_| compile("compute_90"))
                    .map_err(|e| fail(format!("NVRTC V4.1 compilation failed: {e:?}")))?;
                self.device
                    .load_ptx(ptx, "v41_quant", &["v41_quantize", "v41_matvec"])
                    .map_err(|e| fail(format!("CUDA V4.1 module loading failed: {e:?}")))?;
            }
        }
        // Exclusive guard keeps the buffers alive and blocks other backend copies
        // or GEMMs from racing these raw pointers until synchronization completes.
        let allocations = self.allocations.write();
        let find =
            |buffer: &GpuBuffer, required: usize| -> Result<sys::CUdeviceptr, SingularityError> {
                let slice = allocations
                    .get(&buffer.id)
                    .ok_or_else(|| fail(format!("Unknown CUDA buffer {}", buffer.id)))?;
                if slice.len() < required {
                    return Err(fail(format!("CUDA buffer {} is too small", buffer.id)));
                }
                Ok(*slice.device_ptr())
            };
        let w = find(weights, layout.weight_bytes)?;
        let s = find(scales, layout.scale_bytes)?;
        let x = find(input, layout.input_bytes)?;
        let y = find(output, layout.output_bytes)?;
        let q = self
            .device
            .alloc_zeros::<u8>(cols)
            .map_err(|e| fail(format!("CUDA activation allocation failed: {e:?}")))?;
        let qs = self
            .device
            .alloc_zeros::<u8>(cols / 32)
            .map_err(|e| fail(format!("CUDA scale allocation failed: {e:?}")))?;
        let status = self
            .device
            .alloc_zeros::<u32>(1)
            .map_err(|e| fail(format!("CUDA status allocation failed: {e:?}")))?;
        let quant = self
            .device
            .get_func("v41_quant", "v41_quantize")
            .ok_or_else(|| fail("CUDA quantizer missing".into()))?;
        let matvec = self
            .device
            .get_func("v41_quant", "v41_matvec")
            .ok_or_else(|| fail("CUDA matvec missing".into()))?;
        let blocks = layout.activation_blocks as u32;
        let launch = |count: u32| LaunchConfig {
            grid_dim: (count.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let launched = unsafe {
            quant
                .launch(
                    launch(blocks),
                    (
                        x,
                        *q.device_ptr(),
                        *qs.device_ptr(),
                        blocks,
                        *status.device_ptr(),
                    ),
                )
                .and_then(|_| {
                    matvec.launch(
                        launch(rows as u32),
                        (
                            w,
                            s,
                            *q.device_ptr(),
                            *qs.device_ptr(),
                            y,
                            rows as u32,
                            cols as u32,
                            u32::from(expert),
                        ),
                    )
                })
        };
        let synchronized = self.device.synchronize();
        launched.map_err(|e| fail(format!("CUDA V4.1 kernel launch failed: {e:?}")))?;
        synchronized.map_err(|e| fail(format!("CUDA V4.1 execution failed: {e:?}")))?;
        let flags = self
            .device
            .dtoh_sync_copy(&status)
            .map_err(|e| fail(format!("CUDA status read failed: {e:?}")))?;
        if flags[0] != 0 {
            return Err(fail(format!(
                "Invalid V4.1 activation input (status {})",
                flags[0]
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaBackend {
    fn drop(&mut self) {
        if self.device.bind_to_thread().is_err() {
            return;
        }
        for (_, (event, _)) in self.events.get_mut().drain() {
            if let Err(e) = unsafe { cudarc::driver::result::event::destroy(event as sys::CUevent) }
            {
                tracing::warn!("CUDA event destruction failed: {e:?}");
            }
        }
        for (_, stream) in self.streams.get_mut().drain() {
            if let Err(e) =
                unsafe { cudarc::driver::result::stream::destroy(stream as sys::CUstream) }
            {
                tracing::warn!("CUDA stream destruction failed: {e:?}");
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl GpuBackend for CudaBackend {
    fn allocate(&self, size_bytes: usize) -> Result<GpuBuffer, SingularityError> {
        let aligned_size =
            size_bytes
                .checked_add(31)
                .ok_or_else(|| SingularityError::HardwareError {
                    detail: "CUDA allocation size overflow".into(),
                })?
                & !31;
        if aligned_size == 0 {
            return Err(SingularityError::HardwareError {
                detail: "CUDA allocation size must be nonzero".into(),
            });
        }
        let slice: CudaSlice<u8> =
            self.device
                .alloc_zeros(aligned_size)
                .map_err(|_e| SingularityError::OutOfMemory {
                    requested_bytes: aligned_size,
                    available_bytes: 0,
                })?;

        // Extract device virtual address
        let ptr = *slice.device_ptr() as usize;
        let buffer = GpuBuffer::new(aligned_size, ptr, false);

        self.allocations.write().insert(buffer.id, slice);
        Ok(buffer)
    }

    fn free(&self, buffer: &GpuBuffer) -> Result<(), SingularityError> {
        let removed = self.allocations.write().remove(&buffer.id);
        if removed.is_some() {
            Ok(())
        } else {
            Err(SingularityError::HardwareError {
                detail: format!("Attempted to free unknown CUDA buffer ID {}", buffer.id),
            })
        }
    }

    fn copy_htod_offset(
        &self,
        dst: &GpuBuffer,
        dst_offset: usize,
        src_bytes: &[u8],
    ) -> Result<(), SingularityError> {
        let guard = self.allocations.read();
        let slice = guard
            .get(&dst.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: format!("CUDA buffer ID {} not found", dst.id),
            })?;

        if dst_offset
            .checked_add(src_bytes.len())
            .is_none_or(|end| end > slice.len())
        {
            return Err(SingularityError::HardwareError {
                detail: format!(
                    "HtoD transfer overflow: offset {} + size {} > capacity {}",
                    dst_offset,
                    src_bytes.len(),
                    dst.size_bytes
                ),
            });
        }

        let dev_ptr = *slice.device_ptr() + dst_offset as sys::CUdeviceptr;
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        unsafe {
            cudarc::driver::result::memcpy_htod_sync(dev_ptr, src_bytes).map_err(|e| {
                SingularityError::HardwareError {
                    detail: format!("CUDA HtoD transfer failed: {e:?}"),
                }
            })?;
        }
        Ok(())
    }

    fn copy_dtoh_offset(
        &self,
        src: &GpuBuffer,
        src_offset: usize,
        dst_bytes: &mut [u8],
    ) -> Result<(), SingularityError> {
        let guard = self.allocations.read();
        let slice = guard
            .get(&src.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: format!("CUDA buffer ID {} not found", src.id),
            })?;

        if src_offset
            .checked_add(dst_bytes.len())
            .is_none_or(|end| end > slice.len())
        {
            return Err(SingularityError::HardwareError {
                detail: format!(
                    "DtoH transfer overflow: offset {} + size {} > capacity {}",
                    src_offset,
                    dst_bytes.len(),
                    src.size_bytes
                ),
            });
        }

        let dev_ptr = *slice.device_ptr() + src_offset as sys::CUdeviceptr;
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(dst_bytes, dev_ptr).map_err(|e| {
                SingularityError::HardwareError {
                    detail: format!("CUDA DtoH transfer failed: {e:?}"),
                }
            })?;
        }
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
        let guard = self.allocations.read();
        let slice_a = guard
            .get(&a.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: format!("Matrix A buffer {} not found", a.id),
            })?;
        let slice_b = guard
            .get(&b.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: format!("Matrix B buffer {} not found", b.id),
            })?;
        let slice_c = guard
            .get(&c.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: format!("Matrix C buffer {} not found", c.id),
            })?;

        if c.id == a.id || c.id == b.id {
            return Err(SingularityError::HardwareError {
                detail: "CUDA GEMM output cannot alias an input".into(),
            });
        }
        crate::backend::validate_gemm(
            trans_a,
            trans_b,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            slice_a.len(),
            slice_b.len(),
            slice_c.len(),
        )?;
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;

        let f32_a = DeviceF32Ptr {
            ptr: *slice_a.device_ptr(),
            len: slice_a.len() / 4,
        };
        let f32_b = DeviceF32Ptr {
            ptr: *slice_b.device_ptr(),
            len: slice_b.len() / 4,
        };
        let mut f32_c = DeviceF32Ptr {
            ptr: *slice_c.device_ptr(),
            len: slice_c.len() / 4,
        };

        // Row-major C = alpha * A * B + beta * C via cuBLAS column-major:
        // C^T = alpha * B^T * A^T + beta * C^T
        let cfg = GemmConfig {
            transa: if trans_b {
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T
            } else {
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N
            },
            transb: if trans_a {
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T
            } else {
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N
            },
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: ldb as i32,
            ldb: lda as i32,
            beta,
            ldc: ldc as i32,
        };

        unsafe {
            self.blas
                .gemm(cfg, &f32_b, &f32_a, &mut f32_c)
                .map_err(|e| SingularityError::HardwareError {
                    detail: format!("cuBLAS GEMM kernel execution failed: {e:?}"),
                })?;
        }

        Ok(())
    }

    fn create_stream(&self, name: &str) -> Result<Arc<GpuStream>, SingularityError> {
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        let handle = cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        )
        .map_err(|e| SingularityError::HardwareError {
            detail: format!("CUDA stream creation failed: {e:?}"),
        })?;
        let stream = Arc::new(GpuStream::new(name));
        self.streams.write().insert(stream.id, handle as usize);
        Ok(stream)
    }

    fn create_event(&self) -> Result<Arc<GpuEvent>, SingularityError> {
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        let handle = cudarc::driver::result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA event creation failed: {e:?}"),
            })?;
        let event = Arc::new(GpuEvent::new());
        self.events
            .write()
            .insert(event.id, (handle as usize, false));
        Ok(event)
    }

    fn destroy_stream(&self, stream: &GpuStream) -> Result<(), SingularityError> {
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        let mut streams = self.streams.write();
        let handle = *streams
            .get(&stream.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: "Unknown CUDA stream".into(),
            })?;
        unsafe { cudarc::driver::result::stream::destroy(handle as sys::CUstream) }.map_err(
            |e| SingularityError::HardwareError {
                detail: format!("CUDA stream destruction failed: {e:?}"),
            },
        )?;
        streams.remove(&stream.id);
        Ok(())
    }

    fn destroy_event(&self, event: &GpuEvent) -> Result<(), SingularityError> {
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        let mut events = self.events.write();
        let &(handle, _) =
            events
                .get(&event.id)
                .ok_or_else(|| SingularityError::HardwareError {
                    detail: "Unknown CUDA event".into(),
                })?;
        unsafe { cudarc::driver::result::event::destroy(handle as sys::CUevent) }.map_err(|e| {
            SingularityError::HardwareError {
                detail: format!("CUDA event destruction failed: {e:?}"),
            }
        })?;
        events.remove(&event.id);
        Ok(())
    }

    fn record_event(&self, event: &GpuEvent, stream: &GpuStream) -> Result<(), SingularityError> {
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        let streams = self.streams.read();
        let stream = *streams
            .get(&stream.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: "Unknown CUDA stream".into(),
            })? as sys::CUstream;
        let mut events = self.events.write();
        let entry = events
            .get_mut(&event.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: "Unknown CUDA event".into(),
            })?;
        unsafe { cudarc::driver::result::event::record(entry.0 as sys::CUevent, stream) }.map_err(
            |e| SingularityError::HardwareError {
                detail: format!("CUDA event record failed: {e:?}"),
            },
        )?;
        entry.1 = true;
        event.record(); // Host submission time, not device completion time.
        Ok(())
    }

    fn wait_event(&self, stream: &GpuStream, event: &GpuEvent) -> Result<(), SingularityError> {
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        let streams = self.streams.read();
        let stream = *streams
            .get(&stream.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: "Unknown CUDA stream".into(),
            })? as sys::CUstream;
        let events = self.events.read();
        let &(event, recorded) =
            events
                .get(&event.id)
                .ok_or_else(|| SingularityError::HardwareError {
                    detail: "Unknown CUDA event".into(),
                })?;
        if !recorded {
            return Err(SingularityError::HardwareError {
                detail: "CUDA event has not been recorded".into(),
            });
        }
        unsafe {
            cudarc::driver::result::stream::wait_event(
                stream,
                event as sys::CUevent,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
        .map_err(|e| SingularityError::HardwareError {
            detail: format!("CUDA stream wait failed: {e:?}"),
        })?;
        Ok(())
    }

    fn synchronize_stream(&self, stream: &GpuStream) -> Result<(), SingularityError> {
        self.device
            .bind_to_thread()
            .map_err(|e| SingularityError::HardwareError {
                detail: format!("CUDA context binding failed: {e:?}"),
            })?;
        let streams = self.streams.read();
        let handle = *streams
            .get(&stream.id)
            .ok_or_else(|| SingularityError::HardwareError {
                detail: "Unknown CUDA stream".into(),
            })? as sys::CUstream;
        unsafe { cudarc::driver::result::stream::synchronize(handle) }.map_err(|e| {
            SingularityError::HardwareError {
                detail: format!("CUDA stream synchronization failed: {e:?}"),
            }
        })?;
        Ok(())
    }

    fn mem_info(&self) -> (usize, usize) {
        if self.device.bind_to_thread().is_err() {
            return (0, 0);
        }
        cudarc::driver::result::mem_get_info().unwrap_or((0, 0))
    }

    fn architecture(&self) -> &DeviceArchitecture {
        &self.arch
    }
}

#[cfg(all(test, feature = "cuda"))]
mod physical_tests {
    use super::*;
    #[test]
    #[ignore = "requires physical CUDA GPU and CUDA driver/cuBLAS libraries"]
    fn physical_cuda_transfers_gemm_and_events() {
        let arch = crate::detect_hardware(0, None).unwrap();
        assert!(matches!(
            arch,
            DeviceArchitecture::BlackwellSm100 { .. } | DeviceArchitecture::HopperSm90 { .. }
        ));
        let backend = CudaBackend::new(0, arch).unwrap();
        let a = backend.allocate(16).unwrap();
        let b = backend.allocate(16).unwrap();
        let c = backend.allocate(16).unwrap();
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        backend.copy_htod(&a, &bytes(&[1., 2., 3., 4.])).unwrap();
        backend.copy_htod(&b, &bytes(&[5., 6., 7., 8.])).unwrap();
        backend
            .gemm_f32(false, false, 2, 2, 2, 1., &a, 2, &b, 2, 0., &c, 2)
            .unwrap();
        let mut out = [0u8; 16];
        backend.copy_dtoh(&c, &mut out).unwrap();
        let result: Vec<f32> = out
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(result, [19., 22., 43., 50.]);
        assert!(backend.copy_htod_offset(&a, usize::MAX, &[0]).is_err());
        assert!(backend
            .gemm_f32(false, false, 2, 2, 2, 1., &a, 1, &b, 2, 0., &c, 2)
            .is_err());
        let s1 = backend.create_stream("producer").unwrap();
        let s2 = backend.create_stream("consumer").unwrap();
        let event = backend.create_event().unwrap();
        assert!(backend.wait_event(&s2, &event).is_err());
        backend.record_event(&event, &s1).unwrap();
        backend.wait_event(&s2, &event).unwrap();
        backend.synchronize_stream(&s2).unwrap();
        backend.destroy_event(&event).unwrap();
        backend.destroy_stream(&s1).unwrap();
        backend.destroy_stream(&s2).unwrap();
        assert!(backend.destroy_event(&event).is_err());
        assert!(backend.synchronize_stream(&s2).is_err());
        backend.free(&a).unwrap();
        backend.free(&b).unwrap();
        backend.free(&c).unwrap();
    }

    #[test]
    #[ignore = "requires physical CUDA GPU, CUDA driver, cuBLAS and NVRTC"]
    fn physical_v41_quantized_matvec() {
        let arch = crate::detect_hardware(0, None).unwrap();
        assert!(matches!(
            arch,
            DeviceArchitecture::BlackwellSm100 { .. } | DeviceArchitecture::HopperSm90 { .. }
        ));
        let backend = CudaBackend::new(0, arch).unwrap();
        let weights = backend.allocate(64).unwrap();
        let scale = backend.allocate(32).unwrap();
        let input = backend.allocate(64).unwrap();
        let output = backend.allocate(4).unwrap();
        let ones: Vec<u8> = (0..32).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
        backend.copy_htod(&input, &ones).unwrap();
        backend.copy_htod(&weights, &[0x38; 64]).unwrap(); // E4M3 1.0
        backend.copy_htod(&scale, &[127]).unwrap();
        backend
            .v41_quantized_matvec(&weights, &scale, &input, &output, 2, 32, false)
            .unwrap();
        let mut bytes = [0u8; 4];
        backend.copy_dtoh(&output, &mut bytes).unwrap();
        assert_eq!(bytes, [0x00, 0x42, 0x00, 0x42]); // BF16 32.0
        backend.copy_htod(&weights, &[0x21; 16]).unwrap(); // FP4 0.5, 1.0
        backend
            .v41_quantized_matvec(&weights, &scale, &input, &output, 1, 32, true)
            .unwrap();
        backend.copy_dtoh(&output, &mut bytes[..2]).unwrap();
        assert_eq!(&bytes[..2], &[0xc0, 0x41]); // BF16 24.0
        assert!(backend
            .v41_quantized_matvec(&weights, &scale, &input, &output, 2, 31, true)
            .is_err());
        assert!(backend
            .v41_quantized_matvec(&weights, &scale, &input, &input, 1, 32, true)
            .is_err());
        let mut invalid = ones.clone();
        invalid[..2].copy_from_slice(&0x7f80u16.to_le_bytes());
        backend.copy_htod(&input, &invalid).unwrap();
        assert!(backend
            .v41_quantized_matvec(&weights, &scale, &input, &output, 1, 32, true)
            .is_err());
        for buffer in [&weights, &scale, &input, &output] {
            backend.free(buffer).unwrap();
        }
    }
}
