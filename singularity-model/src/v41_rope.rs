//! V4.1 adjacent-pair rotary encoding and YaRN frequencies, CPU reference.
//! Based on checkpoint inference/model.py, MIT license, revision
//! 1bae0e1088993b9b82de08f59de5be5a34e42170.
use half::bf16;
use singularity_core::SingularityError;

use crate::model_error as error;

#[derive(Debug, Clone, Copy)]
pub struct Yarn {
    pub original_context: usize,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

pub fn frequencies_for_layer(
    config: &crate::v41_config::TextConfig,
    layer: usize,
) -> Result<Vec<f32>, SingularityError> {
    let ratio = *config
        .compress_ratios
        .get(layer)
        .ok_or_else(|| error("RoPE layer is outside checkpoint topology"))?;
    let yarn = if ratio == 0 {
        None
    } else {
        match &config.rope_scaling {
            None => None,
            Some(s) if s.rope_type == "yarn" => Some(Yarn {
                original_context: s.original_max_position_embeddings,
                factor: s.factor,
                beta_fast: s.beta_fast,
                beta_slow: s.beta_slow,
            }),
            Some(_) => return Err(error("Unsupported RoPE scaling")),
        }
    };
    frequencies(
        config.qk_rope_head_dim,
        if ratio == 0 {
            config.rope_theta
        } else {
            config.compress_rope_theta
        },
        yarn,
    )
}

/// Precompute per-pair frequencies; use Yarn only for compressed attention.
pub fn frequencies(
    dim: usize,
    base: f32,
    yarn: Option<Yarn>,
) -> Result<Vec<f32>, SingularityError> {
    if dim == 0 || dim % 2 != 0 || !base.is_finite() || base <= 1.0 {
        return Err(error("Invalid RoPE dimensions or base"));
    }
    let mut freqs: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32))
        .collect();
    if let Some(y) = yarn {
        if y.original_context == 0
            || !y.factor.is_finite()
            || y.factor <= 0.0
            || !y.beta_fast.is_finite()
            || y.beta_fast <= 0.0
            || !y.beta_slow.is_finite()
            || y.beta_slow <= 0.0
        {
            return Err(error("Invalid YaRN parameters"));
        }
        let corrected = |rot: f32| {
            dim as f64
                * ((y.original_context as f64) / (rot as f64 * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = corrected(y.beta_fast).floor().max(0.0) as f32;
        let high = corrected(y.beta_slow).ceil().min((dim - 1) as f64) as f32;
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / y.factor * (1.0 - smooth) + *f * smooth;
        }
    }
    Ok(freqs)
}

/// Rotate the final 2*frequencies.len() dimensions, preserving the prefix.
/// Inputs are treated as BF16 when `round_bf16` is enabled; each rotation
/// result is rounded to BF16, as in the checkpoint's in-place operation.
pub fn rotate(
    input: &[f32],
    frequencies: &[f32],
    position: usize,
    inverse: bool,
    round_bf16: bool,
) -> Result<Vec<f32>, SingularityError> {
    let rotary = frequencies
        .len()
        .checked_mul(2)
        .ok_or_else(|| error("RoPE dimension overflow"))?;
    if rotary == 0
        || rotary > input.len()
        || input.iter().chain(frequencies).any(|v| !v.is_finite())
    {
        return Err(error("Invalid RoPE input"));
    }
    let mut out = input.to_vec();
    let start = input.len() - rotary;
    for (i, &freq) in frequencies.iter().enumerate() {
        let angle = (position as f32) * freq;
        let (mut sin, cos) = angle.sin_cos();
        if inverse {
            sin = -sin;
        }
        let (a, b) = (input[start + 2 * i], input[start + 2 * i + 1]);
        let (x, y) = (a * cos - b * sin, a * sin + b * cos);
        out[start + 2 * i] = if round_bf16 {
            bf16::from_f32(x).to_f32()
        } else {
            x
        };
        out[start + 2 * i + 1] = if round_bf16 {
            bf16::from_f32(y).to_f32()
        } else {
            y
        };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct Case {
        yarn: bool,
        base: f32,
        position: usize,
        freqs: Vec<f32>,
        output: Vec<f32>,
    }
    #[derive(Deserialize)]
    struct Vectors {
        input: Vec<f32>,
        cases: Vec<Case>,
    }
    #[test]
    fn rope_and_yarn_match_torch_at_long_positions() {
        let vectors: Vectors =
            serde_json::from_str(include_str!("../tests/fixtures/v41_rope_vectors.json")).unwrap();
        for case in vectors.cases {
            let yarn = case.yarn.then_some(Yarn {
                original_context: 65536,
                factor: 16.,
                beta_fast: 32.,
                beta_slow: 1.,
            });
            let f = frequencies(64, case.base, yarn).unwrap();
            for (a, b) in f.iter().zip(&case.freqs) {
                assert!((a - b).abs() < 1e-7);
            }
            let output = rotate(&vectors.input, &f, case.position, false, true).unwrap();
            assert_eq!(&output[..4], &vectors.input[..4]);
            let max_err = output[4..]
                .iter()
                .zip(&case.output)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 0.01,
                "position {} yarn {} max error {}",
                case.position,
                case.yarn,
                max_err
            );
        }
        assert!(frequencies(63, 10000., None).is_err());
        assert!(rotate(&[1.], &[1.], 0, false, false).is_err());
    }
    #[test]
    fn inverse_rotation_restores_unrounded_vector() {
        let f = frequencies(4, 10000., None).unwrap();
        let x = [1., 2., -1., 0.5];
        let y = rotate(&x, &f, 123, false, false).unwrap();
        let z = rotate(&y, &f, 123, true, false).unwrap();
        for (a, b) in x.iter().zip(z) {
            assert!((a - b).abs() < 1e-6);
        }
        let config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        assert_eq!(
            frequencies_for_layer(&config.text_config, 0).unwrap(),
            frequencies(64, 10000., None).unwrap()
        );
        assert_ne!(
            frequencies_for_layer(&config.text_config, 2).unwrap(),
            frequencies_for_layer(&config.text_config, 0).unwrap()
        );
        assert!(frequencies_for_layer(&config.text_config, 43).is_err());
    }
}
