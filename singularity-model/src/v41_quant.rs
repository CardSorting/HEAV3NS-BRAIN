//! CPU numerical reference for the V4.1 checkpoint's native weight formats.
//! Layout follows the MIT-licensed checkpoint inference/model.py, kernel.py and
//! convert.py, revision 1bae0e1088993b9b82de08f59de5be5a34e42170.
//! These routines are not optimized GPU kernels and do not quantize activations.
use half::bf16;
use singularity_core::SingularityError;

use crate::model_error as error;

const fn compute_e4m3fn(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 15;
    let mantissa = byte & 7;
    if exponent == 15 && mantissa == 7 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        (mantissa as f32) * (1.0 / 512.0)
    } else {
        let exp_scale = f32::from_bits(((exponent as u32) + 120) << 23);
        (1.0 + (mantissa as f32) * (1.0 / 8.0)) * exp_scale
    };
    sign * magnitude
}

const E4M3FN_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = compute_e4m3fn(i as u8);
        i += 1;
    }
    table
};

const E2M1FN_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const fn compute_ue8m0(byte: u8) -> f32 {
    match byte {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        _ => f32::from_bits((byte as u32) << 23),
    }
}

const UE8M0_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = compute_ue8m0(i as u8);
        i += 1;
    }
    table
};

/// Decode an E4M3FN byte, including subnormals, signed zero and NaN.
#[inline(always)]
pub fn e4m3fn(byte: u8) -> f32 {
    E4M3FN_LUT[byte as usize]
}

/// Decode UE8M0. 255 is the format's NaN encoding; zero denotes 2^-127.
#[inline(always)]
pub fn ue8m0(byte: u8) -> f32 {
    UE8M0_LUT[byte as usize]
}

/// E2M1FN nibble, as stored in the expert checkpoint.
#[inline(always)]
pub fn e2m1fn(nibble: u8) -> f32 {
    E2M1FN_LUT[(nibble & 15) as usize]
}

/// In-place FP4 activation quantization as used by compressed KV and indexer
/// paths. Input and output use BF16 boundaries; scales are per group of 16
/// (E4M3) for compressed KV or 32 (UE8M0) for the indexer.
pub fn fp4_activation_dequant(
    input: &[f32],
    e4m3_scale: bool,
) -> Result<Vec<f32>, SingularityError> {
    let block = if e4m3_scale { 16 } else { 32 };
    if input.is_empty() || input.len() % block != 0 {
        return Err(error("Invalid FP4 activation dimensions"));
    }
    let mut output = Vec::with_capacity(input.len());
    for group in input.chunks_exact(block) {
        let mut stack_buf = [0.0f32; 32];
        let values = &mut stack_buf[..block];
        for (dst, &v) in values.iter_mut().zip(group.iter()) {
            let rounded = bf16::from_f32(v).to_f32();
            if !rounded.is_finite() {
                return Err(error("Non-finite FP4 activation"));
            }
            *dst = rounded;
        }
        let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if e4m3_scale {
            e4m3fn(encode_e4m3fn(amax.max(6.0 * 2.0f32.powi(-9)) / 6.0)?)
        } else {
            let value = amax.max(6.0 * 2.0f32.powi(-126)) * (1.0 / 6.0);
            let bits = value.to_bits();
            let exp = ((bits >> 23) & 255) + u32::from(bits & 0x7fffff != 0);
            if !(1..=254).contains(&exp) {
                return Err(error("FP4 activation scale outside UE8M0 range"));
            }
            ue8m0(exp as u8)
        };
        for &value in values.iter() {
            let normalized = (value.abs() / scale).min(6.0);
            let mut best = 0u8;
            let mut distance = f32::INFINITY;
            for candidate in 0..8u8 {
                let d = (normalized - e2m1fn(candidate)).abs();
                if d < distance || (d == distance && candidate % 2 == 0 && best % 2 != 0) {
                    best = candidate;
                    distance = d;
                }
            }
            let signed = if value.is_sign_negative() {
                -e2m1fn(best)
            } else {
                e2m1fn(best)
            };
            output.push(bf16::from_f32(signed * scale).to_f32());
        }
    }
    Ok(output)
}

/// Round a finite value to E4M3FN, nearest with ties to even, saturating to 448.
pub fn encode_e4m3fn(value: f32) -> Result<u8, SingularityError> {
    if !value.is_finite() {
        return Err(error("Non-finite FP8 input"));
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs().min(448.0);
    let mut lo = 0u8;
    let mut hi = 126u8;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if e4m3fn(mid) < magnitude {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let upper = lo;
    if upper == 0 {
        return Ok(sign);
    }
    let lower = upper - 1;
    let dl = magnitude - e4m3fn(lower);
    let du = e4m3fn(upper) - magnitude;
    let chosen = if dl < du || (dl == du && lower % 2 == 0) {
        lower
    } else {
        upper
    };
    Ok(sign | chosen)
}

/// Quantized BF16 activations and per-32 UE8M0 scales, in checkpoint layout.
pub struct QuantizedActivation {
    pub values: Vec<u8>,
    pub scales: Vec<u8>,
}
pub fn quantize_activation_fp8(input: &[f32]) -> Result<QuantizedActivation, SingularityError> {
    if input.is_empty() || input.len() % 32 != 0 {
        return Err(error(
            "FP8 activation length must be a positive multiple of 32",
        ));
    }
    let mut values = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(input.len() / 32);
    for group in input.chunks_exact(32) {
        let mut rounded = [0.0f32; 32];
        for (dst, &x) in rounded.iter_mut().zip(group.iter()) {
            let r = bf16::from_f32(x).to_f32();
            if !r.is_finite() {
                return Err(error("Non-finite BF16 activation"));
            }
            *dst = r;
        }
        let amax = rounded.iter().map(|x| x.abs()).fold(1e-4f32, f32::max);
        let bits = (amax * (1.0 / 448.0)).to_bits();
        let exponent = ((bits >> 23) & 255) + u32::from(bits & 0x7fffff != 0);
        if !(1..=254).contains(&exponent) {
            return Err(error("FP8 activation scale outside UE8M0 range"));
        }
        let scale = ue8m0(exponent as u8);
        scales.push(exponent as u8);
        for x in rounded {
            values.push(encode_e4m3fn((x / scale).clamp(-448., 448.))?);
        }
    }
    Ok(QuantizedActivation { values, scales })
}

fn dimensions(rows: usize, cols: usize, block: usize) -> Result<(usize, usize), SingularityError> {
    if rows == 0 || cols == 0 || block == 0 {
        return Err(error(
            "Quantized matrix dimensions and block size must be nonzero",
        ));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| error("Quantized matrix size overflow"))?;
    let blocks = cols.div_ceil(block);
    Ok((count, blocks))
}

/// Borrowed view of an original checkpoint matrix and its scales.
pub enum CheckpointMatrix<'a> {
    Dense {
        weights: &'a [u8],
        scales: &'a [u8],
        rows: usize,
        cols: usize,
    },
    Expert {
        weights: &'a [u8],
        scales: &'a [u8],
        rows: usize,
        cols: usize,
    },
    LazyExpert {
        loader: &'a crate::SafetensorsMmapLoader,
        weights: &'a [u8],
        scale_name: String,
        rows: usize,
        cols: usize,
    },
}

impl<'a> CheckpointMatrix<'a> {
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Dense { rows, cols, .. }
            | Self::Expert { rows, cols, .. }
            | Self::LazyExpert { rows, cols, .. } => (*rows, *cols),
        }
    }
    pub fn load(
        loader: &'a crate::SafetensorsMmapLoader,
        weight_name: &str,
        scale_name: &str,
    ) -> Result<Self, SingularityError> {
        let w = loader
            .get_tensor_info(weight_name)
            .ok_or_else(|| error("Missing quantized weight"))?;
        let s = loader
            .get_tensor_info(scale_name)
            .ok_or_else(|| error("Missing quantized scale"))?;
        if w.shape.len() != 2 || s.dtype != "F8_E8M0" || s.shape.len() != 2 {
            return Err(error("Invalid quantized tensor rank or scale dtype"));
        }
        let rows = w.shape[0];
        let stored_cols = w.shape[1];
        let (cols, expected_scale, is_expert) = match w.dtype.as_str() {
            "F8_E4M3" => (
                stored_cols,
                vec![rows.div_ceil(32), stored_cols.div_ceil(32)],
                false,
            ),
            "I8" => {
                let cols = stored_cols
                    .checked_mul(2)
                    .ok_or_else(|| error("FP4 column count overflow"))?;
                if cols % 32 != 0 {
                    return Err(error("FP4 expert columns must be divisible by 32"));
                }
                (cols, vec![rows, cols / 32], true)
            }
            _ => return Err(error("Unsupported quantized checkpoint weight dtype")),
        };
        if rows == 0 || cols == 0 || s.shape != expected_scale {
            return Err(error("Quantized checkpoint scale shape mismatch"));
        }
        let weights = loader.get_tensor_bytes(weight_name)?;
        let expected_weights = rows
            .checked_mul(stored_cols)
            .ok_or_else(|| error("Quantized weight size overflow"))?;
        let expected_scales = s.shape[0]
            .checked_mul(s.shape[1])
            .ok_or_else(|| error("Quantized scale size overflow"))?;
        if weights.len() != expected_weights {
            return Err(error("Quantized tensor payload length mismatch"));
        }
        if is_expert && loader.is_csf_scale(scale_name) {
            return Ok(Self::LazyExpert {
                loader,
                weights,
                scale_name: scale_name.to_string(),
                rows,
                cols,
            });
        }
        let scales = loader.get_tensor_bytes(scale_name)?;
        if scales.len() != expected_scales {
            return Err(error("Quantized tensor payload length mismatch"));
        }
        Ok(if is_expert {
            Self::Expert {
                weights,
                scales,
                rows,
                cols,
            }
        } else {
            Self::Dense {
                weights,
                scales,
                rows,
                cols,
            }
        })
    }

    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>, SingularityError> {
        match self {
            Self::Dense {
                weights,
                scales,
                rows,
                cols,
            } => dense_fp8_matvec(weights, scales, *rows, *cols, x),
            Self::Expert {
                weights,
                scales,
                rows,
                cols,
            } => expert_fp4_matvec(weights, scales, *rows, *cols, x),
            Self::LazyExpert {
                loader,
                weights,
                scale_name,
                rows,
                cols,
            } => {
                let scales = loader.get_tensor_bytes(scale_name)?;
                expert_fp4_matvec(weights, scales, *rows, *cols, x)
            }
        }
    }

    /// Reference block GEMM for one BF16 input vector, producing BF16 values.
    /// Both activation and weight scales are applied per reduction block.
    pub fn matvec_quantized(&self, x: &[f32]) -> Result<Vec<f32>, SingularityError> {
        let (rows, cols) = self.shape();
        if x.len() != cols || cols == 0 || cols % 32 != 0 || rows == 0 {
            return Err(error("Invalid quantized GEMM dimensions"));
        }
        let activation = quantize_activation_fp8(x)?;
        self.matvec_quantized_with_activation(&activation)
    }

    /// Execute block GEMM using a pre-quantized FP8 activation vector (avoiding
    /// redundant re-quantization across `w1`, `w3`, and multiple routed experts).
    pub fn matvec_quantized_with_activation(
        &self,
        activation: &QuantizedActivation,
    ) -> Result<Vec<f32>, SingularityError> {
        let (rows, cols) = self.shape();
        if cols == 0
            || cols % 32 != 0
            || rows == 0
            || activation.values.len() != cols
            || activation.scales.len() != cols / 32
        {
            return Err(error("Invalid quantized GEMM dimensions"));
        }
        let blocks = cols / 32;
        let (weights, scales, expert) = match self {
            Self::Dense {
                weights, scales, ..
            } => (*weights, *scales, false),
            Self::Expert {
                weights, scales, ..
            } => (*weights, *scales, true),
            Self::LazyExpert {
                loader,
                weights,
                scale_name,
                ..
            } => (*weights, loader.get_tensor_bytes(scale_name)?, true),
        };
        let weight_count = rows
            .checked_mul(if expert { cols / 2 } else { cols })
            .ok_or_else(|| error("Weight size overflow"))?;
        let scale_count = (if expert { rows } else { rows.div_ceil(32) })
            .checked_mul(blocks)
            .ok_or_else(|| error("Scale size overflow"))?;
        if weights.len() != weight_count || scales.len() != scale_count {
            return Err(error("Invalid quantized GEMM buffers"));
        }
        let act_values: Vec<f32> = activation.values.iter().map(|&b| e4m3fn(b)).collect();
        let act_scales: Vec<f32> = activation.scales.iter().map(|&b| ue8m0(b)).collect();
        let compute_row = |row: usize| -> f32 {
            let mut result = 0.0f32;
            let scale_row = if expert { row } else { row / 32 };
            let scale_row_slice = &scales[scale_row * blocks..(scale_row + 1) * blocks];
            if expert {
                let half_cols = cols / 2;
                let w_row = &weights[row * half_cols..(row + 1) * half_cols];
                for block in 0..blocks {
                    let mut acc = 0.0f32;
                    let w_block = &w_row[block * 16..(block + 1) * 16];
                    let a_block = &act_values[block * 32..(block + 1) * 32];
                    for (idx, &b) in w_block.iter().enumerate() {
                        acc += a_block[idx * 2] * e2m1fn(b & 15);
                        acc += a_block[idx * 2 + 1] * e2m1fn(b >> 4);
                    }
                    result += acc * act_scales[block] * ue8m0(scale_row_slice[block]);
                }
            } else {
                let w_row = &weights[row * cols..(row + 1) * cols];
                for block in 0..blocks {
                    let mut acc = 0.0f32;
                    let w_block = &w_row[block * 32..(block + 1) * 32];
                    let a_block = &act_values[block * 32..(block + 1) * 32];
                    for k in 0..32 {
                        acc += a_block[k] * e4m3fn(w_block[k]);
                    }
                    result += acc * act_scales[block] * ue8m0(scale_row_slice[block]);
                }
            }
            bf16::from_f32(result).to_f32()
        };
        let output = if rows >= 64 {
            use rayon::prelude::*;
            (0..rows)
                .into_par_iter()
                .with_min_len(64)
                .map(compute_row)
                .collect()
        } else {
            (0..rows).map(compute_row).collect()
        };
        Ok(output)
    }
}

/// Dense FP8 weights [rows, cols], UE8M0 scale [ceil(rows/32), ceil(cols/32)].
/// Computes W * x using unquantized f32 x and f32 accumulation.
pub fn dense_fp8_matvec(
    weights: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, SingularityError> {
    let (count, col_blocks) = dimensions(rows, cols, 32)?;
    let scale_count = rows
        .div_ceil(32)
        .checked_mul(col_blocks)
        .ok_or_else(|| error("FP8 scale size overflow"))?;
    if weights.len() != count || scales.len() != scale_count || x.len() != cols {
        return Err(error("Invalid FP8 matrix, scale, or input length"));
    }
    let compute_row = |row: usize| -> f32 {
        let mut out = 0.0f32;
        let w_row = &weights[row * cols..(row + 1) * cols];
        let s_row = &scales[(row / 32) * col_blocks..(row / 32 + 1) * col_blocks];
        for (col, &input) in x.iter().enumerate() {
            out += e4m3fn(w_row[col]) * ue8m0(s_row[col / 32]) * input;
        }
        out
    };
    let output = if rows >= 64 {
        use rayon::prelude::*;
        (0..rows)
            .into_par_iter()
            .with_min_len(64)
            .map(compute_row)
            .collect()
    } else {
        (0..rows).map(compute_row).collect()
    };
    Ok(output)
}

/// Expert FP4 weights [rows, cols/2], low nibble first; UE8M0 scales
/// [rows, cols/32]. Computes W * x with f32 accumulation.
pub fn expert_fp4_matvec(
    weights: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, SingularityError> {
    let (count, col_blocks) = dimensions(rows, cols, 32)?;
    if cols % 32 != 0 {
        return Err(error("FP4 expert columns must be divisible by 32"));
    }
    let scale_count = rows
        .checked_mul(col_blocks)
        .ok_or_else(|| error("FP4 scale size overflow"))?;
    if weights.len() != count / 2 || scales.len() != scale_count || x.len() != cols {
        return Err(error("Invalid FP4 matrix, scale, or input length"));
    }
    let half_cols = cols / 2;
    let compute_row = |row: usize| -> f32 {
        let mut out = 0.0f32;
        let w_row = &weights[row * half_cols..(row + 1) * half_cols];
        let s_row = &scales[row * col_blocks..(row + 1) * col_blocks];
        for (col, &input) in x.iter().enumerate() {
            let byte = w_row[col / 2];
            let nibble = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
            out += e2m1fn(nibble) * ue8m0(s_row[col / 32]) * input;
        }
        out
    };
    let output = if rows >= 64 {
        use rayon::prelude::*;
        (0..rows)
            .into_par_iter()
            .with_min_len(64)
            .map(compute_row)
            .collect()
    } else {
        (0..rows).map(compute_row).collect()
    };
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_activation_rounding_scales_and_zero_groups() {
        let mut values = vec![0.0; 32];
        values[..8].copy_from_slice(&[6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, -5.0]);
        let out = fp4_activation_dequant(&values, false).unwrap();
        assert_eq!(&out[..8], &[6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, -4.0]);
        assert!(out[8..].iter().all(|x| *x == 0.0));
        let mut kv = vec![0.0; 16];
        kv[0] = 3.0;
        kv[1] = 0.375;
        let result = fp4_activation_dequant(&kv, true).unwrap();
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 0.5);
        assert!(fp4_activation_dequant(&[0.0; 15], true).is_err());
    }
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct MatrixVector {
        rows: usize,
        cols: usize,
        weights: Vec<u8>,
        scales: Vec<u8>,
        x: Vec<f32>,
        y: Vec<f32>,
    }
    #[derive(Deserialize)]
    struct Vectors {
        e4_bits: Vec<u32>,
        e8_bits: Vec<u32>,
        dense: MatrixVector,
        expert: MatrixVector,
    }
    fn vectors() -> Vectors {
        serde_json::from_str(include_str!("../tests/fixtures/v41_quant_vectors.json")).unwrap()
    }
    #[test]
    fn fp8_decoders_match_all_pytorch_byte_values() {
        let v = vectors();
        for byte in 0..=255u8 {
            for (actual, expected) in [
                (e4m3fn(byte), v.e4_bits[byte as usize]),
                (ue8m0(byte), v.e8_bits[byte as usize]),
            ] {
                if f32::from_bits(expected).is_nan() {
                    assert!(actual.is_nan(), "{byte:#x}");
                } else {
                    assert_eq!(actual.to_bits(), expected, "{byte:#x}");
                }
            }
        }
    }
    #[test]
    fn fp8_rounding_and_activation_blocks_match_pytorch() {
        #[derive(Deserialize)]
        struct Rounding {
            input: Vec<f32>,
            values: Vec<u8>,
        }
        #[derive(Deserialize)]
        struct Activation {
            input: Vec<f32>,
            values: Vec<u8>,
            scales: Vec<u8>,
        }
        let r: Rounding =
            serde_json::from_str(include_str!("../tests/fixtures/v41_fp8_rounding.json")).unwrap();
        for (input, expected) in r.input.iter().zip(r.values) {
            assert_eq!(encode_e4m3fn(*input).unwrap(), expected, "{input}");
        }
        let a: Activation = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_activation_vectors.json"
        ))
        .unwrap();
        let actual = quantize_activation_fp8(&a.input).unwrap();
        assert_eq!(actual.values, a.values);
        assert_eq!(actual.scales, a.scales);
        assert!(quantize_activation_fp8(&[0.; 31]).is_err());
        assert!(encode_e4m3fn(f32::NAN).is_err());
    }
    #[test]
    fn native_weight_layouts_match_pytorch_matvec() {
        let v = vectors();
        let d = &v.dense;
        let e = &v.expert;
        let dense = dense_fp8_matvec(&d.weights, &d.scales, d.rows, d.cols, &d.x).unwrap();
        let expert = expert_fp4_matvec(&e.weights, &e.scales, e.rows, e.cols, &e.x).unwrap();
        for (got, expected) in dense.iter().zip(&d.y).chain(expert.iter().zip(&e.y)) {
            assert!(
                (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "{got} != {expected}"
            );
        }
        assert!(dense_fp8_matvec(&d.weights, &d.scales[..3], d.rows, d.cols, &d.x).is_err());
        assert!(expert_fp4_matvec(&e.weights, &e.scales, e.rows, 63, &e.x).is_err());
        assert!(dense_fp8_matvec(&[], &[], usize::MAX, 2, &[]).is_err());
    }

    #[test]
    fn checkpoint_view_loads_packed_weights_and_rejects_bad_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let header = serde_json::json!({
            "layer.weight": {"dtype": "I8", "shape": [1, 16], "data_offsets": [0, 16]},
            "layer.scale": {"dtype": "F8_E8M0", "shape": [1, 1], "data_offsets": [16, 17]},
            "bad.scale": {"dtype": "F8_E8M0", "shape": [1, 1], "data_offsets": [17, 18]},
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0x21; 16]);
        bytes.extend_from_slice(&[127, 127]);
        std::fs::write(&path, bytes).unwrap();
        let loader = crate::SafetensorsMmapLoader::open(path).unwrap();
        let matrix = CheckpointMatrix::load(&loader, "layer.weight", "layer.scale").unwrap();
        assert_eq!(matrix.matvec(&[1.0; 32]).unwrap(), [24.0]);
        assert!(CheckpointMatrix::load(&loader, "layer.weight", "missing").is_err());
        assert!(CheckpointMatrix::load(&loader, "bad.scale", "layer.scale").is_err());
    }
}
