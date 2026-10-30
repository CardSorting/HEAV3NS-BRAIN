use singularity_core::{CacheMemoryFormat, SingularityError, TargetArch};
use tracing::warn;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceArchitecture {
    BlackwellSm100 {
        hbm_bytes: usize,
        nvlink_gen: u32,
        has_native_fp4: bool,
        tma_version: u32,
    },
    HopperSm90 {
        hbm_bytes: usize,
        nvlink_gen: u32,
        has_native_fp4: bool,
        tma_version: u32,
    },
    MockDevice {
        name: String,
        hbm_bytes: usize,
        nvlink_gen: u32,
        has_native_fp4: bool,
        tma_version: u32,
    },
    Unsupported {
        major: i32,
        minor: i32,
        reason: String,
    },
}

impl DeviceArchitecture {
    pub fn name(&self) -> &str {
        match self {
            Self::BlackwellSm100 { .. } => "NVIDIA Blackwell (SM100 - B200)",
            Self::HopperSm90 { .. } => "NVIDIA Hopper (SM90 - H200)",
            Self::MockDevice { name, .. } => name.as_str(),
            Self::Unsupported { .. } => "Unsupported Accelerator",
        }
    }

    pub fn hbm_bytes(&self) -> usize {
        match self {
            Self::BlackwellSm100 { hbm_bytes, .. } => *hbm_bytes,
            Self::HopperSm90 { hbm_bytes, .. } => *hbm_bytes,
            Self::MockDevice { hbm_bytes, .. } => *hbm_bytes,
            Self::Unsupported { .. } => 0,
        }
    }

    pub fn has_native_fp4(&self) -> bool {
        match self {
            Self::BlackwellSm100 { has_native_fp4, .. } => *has_native_fp4,
            Self::HopperSm90 { has_native_fp4, .. } => *has_native_fp4,
            Self::MockDevice { has_native_fp4, .. } => *has_native_fp4,
            Self::Unsupported { .. } => false,
        }
    }

    pub fn tma_version(&self) -> u32 {
        match self {
            Self::BlackwellSm100 { tma_version, .. } => *tma_version,
            Self::HopperSm90 { tma_version, .. } => *tma_version,
            Self::MockDevice { tma_version, .. } => *tma_version,
            Self::Unsupported { .. } => 0,
        }
    }

    pub fn nvlink_gen(&self) -> u32 {
        match self {
            Self::BlackwellSm100 { nvlink_gen, .. } => *nvlink_gen,
            Self::HopperSm90 { nvlink_gen, .. } => *nvlink_gen,
            Self::MockDevice { nvlink_gen, .. } => *nvlink_gen,
            Self::Unsupported { .. } => 0,
        }
    }

    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported { .. })
    }
}

pub fn detect_hardware(
    device_id: i32,
    override_arch: Option<TargetArch>,
) -> Result<DeviceArchitecture, SingularityError> {
    if let Some(target) = override_arch {
        match target {
            TargetArch::BlackwellSm100 => {
                return Ok(DeviceArchitecture::BlackwellSm100 {
                    hbm_bytes: 180 * 1024 * 1024 * 1024, // 180 GB HBM3e
                    nvlink_gen: 5,                       // 5th gen NVLink (1800 GB/s)
                    has_native_fp4: true,
                    tma_version: 2,
                });
            }
            TargetArch::HopperSm90 => {
                return Ok(DeviceArchitecture::HopperSm90 {
                    hbm_bytes: 141 * 1024 * 1024 * 1024, // 141 GB HBM3e
                    nvlink_gen: 4,                       // 4th gen NVLink (900 GB/s)
                    has_native_fp4: false,
                    tma_version: 1,
                });
            }
            TargetArch::HostMock => {
                return Ok(DeviceArchitecture::MockDevice {
                    name: format!("Host-Mock-GPU-{device_id}"),
                    hbm_bytes: 180 * 1024 * 1024 * 1024,
                    nvlink_gen: 5,
                    has_native_fp4: true,
                    tma_version: 2,
                });
            }
            TargetArch::AutoDetect => {}
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = cudarc::driver::CudaDevice::new(device_id as usize) {
            let major = dev
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .unwrap_or(9);
            let minor = dev
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .unwrap_or(0);
            let (_, total_mem) =
                cudarc::driver::result::mem_get_info().unwrap_or((0, 141 * 1024 * 1024 * 1024));

            if major >= 10 {
                return Ok(DeviceArchitecture::BlackwellSm100 {
                    hbm_bytes: total_mem,
                    nvlink_gen: 5,
                    has_native_fp4: true,
                    tma_version: 2,
                });
            } else if major == 9 {
                return Ok(DeviceArchitecture::HopperSm90 {
                    hbm_bytes: total_mem,
                    nvlink_gen: 4,
                    has_native_fp4: false,
                    tma_version: 1,
                });
            } else {
                return Ok(DeviceArchitecture::Unsupported {
                    major,
                    minor,
                    reason: format!(
                        "Compute capability {major}.{minor} is below Hopper SM90 baseline"
                    ),
                });
            }
        }
    }

    // When running on non-CUDA development host or when auto-detecting without CUDA
    Ok(DeviceArchitecture::MockDevice {
        name: format!("Host-CPU-Mock (No CUDA Device Detected - Device {device_id})"),
        hbm_bytes: 32 * 1024 * 1024 * 1024,
        nvlink_gen: 0,
        has_native_fp4: false,
        tma_version: 0,
    })
}

pub fn validate_precision_and_fallback(
    arch: &DeviceArchitecture,
    requested_fp4: bool,
) -> Result<CacheMemoryFormat, SingularityError> {
    match arch {
        DeviceArchitecture::BlackwellSm100 { has_native_fp4, .. } => {
            if requested_fp4 && *has_native_fp4 {
                Ok(CacheMemoryFormat::Fp4)
            } else {
                Ok(CacheMemoryFormat::MxFp8)
            }
        }
        DeviceArchitecture::HopperSm90 { has_native_fp4, .. } => {
            if requested_fp4 && !*has_native_fp4 {
                warn!(
                    "FP4 requested on Hopper SM90 lacking native 2nd-gen Tensor Core microscaling; gracefully falling back to MXFP8 (E4M3) dequantization path"
                );
                Ok(CacheMemoryFormat::MxFp8)
            } else if requested_fp4 {
                Ok(CacheMemoryFormat::Fp4)
            } else {
                Ok(CacheMemoryFormat::MxFp8)
            }
        }
        DeviceArchitecture::MockDevice { has_native_fp4, .. } => {
            if requested_fp4 && *has_native_fp4 {
                Ok(CacheMemoryFormat::Fp4)
            } else {
                Ok(CacheMemoryFormat::MxFp8)
            }
        }
        DeviceArchitecture::Unsupported {
            major,
            minor,
            reason,
        } => Err(SingularityError::UnsupportedHardware {
            major: *major,
            minor: *minor,
            reason: reason.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection_and_fallback() {
        let b200 = detect_hardware(0, Some(TargetArch::BlackwellSm100)).unwrap();
        assert!(b200.has_native_fp4());
        assert_eq!(
            validate_precision_and_fallback(&b200, true).unwrap(),
            CacheMemoryFormat::Fp4
        );

        let h200 = detect_hardware(0, Some(TargetArch::HopperSm90)).unwrap();
        assert!(!h200.has_native_fp4());
        assert_eq!(
            validate_precision_and_fallback(&h200, true).unwrap(),
            CacheMemoryFormat::MxFp8
        );

        let mock = detect_hardware(1, Some(TargetArch::HostMock)).unwrap();
        assert!(mock.is_supported());
    }
}
