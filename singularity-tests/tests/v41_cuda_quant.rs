#![cfg(feature = "cuda")]
//! Physical-GPU comparison of CUDA kernels against the independent CPU reference.
use singularity_cuda::{CudaBackend, DeviceArchitecture, GpuBackend};
use singularity_model::v41_quant::CheckpointMatrix;

#[test]
#[ignore = "requires physical SM90/SM100 GPU, CUDA driver, cuBLAS and NVRTC"]
fn cuda_v41_matvec_matches_cpu_reference() {
    let arch = singularity_cuda::detect_hardware(0, None).unwrap();
    assert!(matches!(
        arch,
        DeviceArchitecture::BlackwellSm100 { .. } | DeviceArchitecture::HopperSm90 { .. }
    ));
    let gpu = CudaBackend::new(0, arch).unwrap();
    let rows = 35usize;
    let cols = 64usize;
    let input: Vec<f32> = (0..cols).map(|i| (i as i32 % 17 - 8) as f32 / 8.).collect();
    let input_bytes: Vec<u8> = input
        .iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    for expert in [false, true] {
        let len = rows * if expert { cols / 2 } else { cols };
        // Dense values avoid E4M3FN NaNs. Expert bytes cover all nibbles.
        let weights: Vec<u8> = (0..len)
            .map(|i| {
                if expert {
                    ((i * 19 + 7) % 256) as u8
                } else {
                    ((i * 37 + 13) % 120) as u8
                }
            })
            .collect();
        let scale_count = if expert {
            rows * (cols / 32)
        } else {
            rows.div_ceil(32) * (cols / 32)
        };
        let scales: Vec<u8> = (0..scale_count).map(|i| 125 + (i % 5) as u8).collect();
        let reference = if expert {
            CheckpointMatrix::Expert {
                weights: &weights,
                scales: &scales,
                rows,
                cols,
            }
        } else {
            CheckpointMatrix::Dense {
                weights: &weights,
                scales: &scales,
                rows,
                cols,
            }
        };
        let expected = reference.matvec_quantized(&input).unwrap();
        let w = gpu.allocate(weights.len()).unwrap();
        let s = gpu.allocate(scales.len()).unwrap();
        let x = gpu.allocate(input_bytes.len()).unwrap();
        let y = gpu.allocate(rows * 2).unwrap();
        gpu.copy_htod(&w, &weights).unwrap();
        gpu.copy_htod(&s, &scales).unwrap();
        gpu.copy_htod(&x, &input_bytes).unwrap();
        gpu.v41_quantized_matvec(&w, &s, &x, &y, rows, cols, expert)
            .unwrap();
        let mut out = vec![0u8; rows * 2];
        gpu.copy_dtoh(&y, &mut out).unwrap();
        for (raw, want) in out.chunks_exact(2).zip(expected) {
            let got = f32::from_bits((u16::from_le_bytes(raw.try_into().unwrap()) as u32) << 16);
            assert!(
                (got - want).abs() <= 0.01 * want.abs().max(1.0),
                "expert={expert}: {got} != {want}"
            );
        }
        for buf in [&w, &s, &x, &y] {
            gpu.free(buf).unwrap();
        }
    }
}

#[test]
#[ignore = "requires GPU, NVRTC and pinned expert-309 weight/scale ranges"]
fn cuda_real_expert_w1_matches_cpu_reference() {
    use sha2::{Digest, Sha256};
    let weights = std::fs::read(std::env::var("SINGULARITY_V41_EXPERT_WEIGHT").unwrap()).unwrap();
    let scales = std::fs::read(std::env::var("SINGULARITY_V41_EXPERT_SCALE").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&weights)),
        "456d5f08e85b2630ccb2e0b7b80adcb6545bb7bdc5e8f5ae9a5e361c4fd90a58"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&scales)),
        "3a82b25659af5630781de73b3cf1e3148571ed209f23d6621502252d2bb285dc"
    );
    let arch = singularity_cuda::detect_hardware(0, None).unwrap();
    assert!(matches!(
        arch,
        DeviceArchitecture::BlackwellSm100 { .. } | DeviceArchitecture::HopperSm90 { .. }
    ));
    let gpu = CudaBackend::new(0, arch).unwrap();
    let rows = 2304;
    let cols = 5120;
    let input: Vec<f32> = (0..cols)
        .map(|i| (i as i32 % 97 - 48) as f32 / 64.)
        .collect();
    let input_bytes: Vec<u8> = input
        .iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let matrix = CheckpointMatrix::Expert {
        weights: &weights[..5898240],
        scales: &scales[..368640],
        rows,
        cols,
    };
    let expected = matrix.matvec_quantized(&input).unwrap();
    let w = gpu.allocate(5898240).unwrap();
    let s = gpu.allocate(368640).unwrap();
    let x = gpu.allocate(input_bytes.len()).unwrap();
    let y = gpu.allocate(rows * 2).unwrap();
    gpu.copy_htod(&w, &weights[..5898240]).unwrap();
    gpu.copy_htod(&s, &scales[..368640]).unwrap();
    gpu.copy_htod(&x, &input_bytes).unwrap();
    gpu.v41_quantized_matvec(&w, &s, &x, &y, rows, cols, true)
        .unwrap();
    let mut out = vec![0u8; rows * 2];
    gpu.copy_dtoh(&y, &mut out).unwrap();
    let (mut error, mut norm) = (0.0f64, 0.0f64);
    for (bytes, want) in out.chunks_exact(2).zip(expected) {
        let got = f32::from_bits((u16::from_le_bytes(bytes.try_into().unwrap()) as u32) << 16);
        error += (got as f64 - want as f64).powi(2);
        norm += (want as f64).powi(2);
    }
    let rms = (error / norm).sqrt();
    assert!(rms < 0.01, "relative RMS error {rms}");
    for buf in [&w, &s, &x, &y] {
        gpu.free(buf).unwrap();
    }
}
