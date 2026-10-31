//! V4.1 Hyper-Connections CPU reference, per token.
//! Derived from the MIT-licensed checkpoint inference/model.py and kernel.py,
//! revision 1bae0e1088993b9b82de08f59de5be5a34e42170.
//! Sublayer scheduling is shifted: the current pre coefficient comes from
//! the preceding sublayer, while this sublayer's pre is for the next one.
use half::bf16;
use singularity_core::SingularityError;

use crate::model_error as error;

#[derive(Debug)]
pub struct HcWeights {
    pub streams: usize,
    pub hidden: usize,
    /// Row-major [(2 + streams) * streams, streams * hidden].
    pub projection: Vec<f32>,
    pub scale: [f32; 3],
    pub base: Vec<f32>,
    pub norm_epsilon: f32,
    pub sinkhorn_epsilon: f32,
    pub sinkhorn_iterations: usize,
}

#[derive(Debug)]
pub struct HcMix {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    /// Row-major [source stream, destination stream].
    pub residual: Vec<f32>,
}

/// Residual streams and the shifted pre-mix used by the next sublayer.
#[derive(Clone)]
pub struct HcState {
    pub residual: Vec<f32>,
    pub next_pre: Vec<f32>,
}

impl HcState {
    /// Expand the token embedding into streams and select stream zero initially.
    pub fn from_embedding(embedding: &[f32], streams: usize) -> Result<Self, SingularityError> {
        if embedding.is_empty() || streams == 0 || embedding.len().checked_mul(streams).is_none() {
            return Err(error("Invalid HC embedding dimensions"));
        }
        let mut next_pre = vec![0.0; streams];
        next_pre[0] = 1.0;
        Ok(Self {
            residual: embedding.repeat(streams),
            next_pre,
        })
    }

    /// Execute one shifted HC transition. The callback performs the actual
    /// sublayer (including its normalization); state changes only on success.
    pub fn step<F>(
        &mut self,
        weights: &HcWeights,
        round_bf16: bool,
        sublayer: F,
    ) -> Result<(), SingularityError>
    where
        F: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
    {
        let mix = weights.mixes(&self.residual)?;
        let input = hc_pre(&self.residual, &self.next_pre, weights.hidden, round_bf16)?;
        let output = sublayer(&input)?;
        let residual = hc_post(
            &output,
            &self.residual,
            &mix.post,
            &mix.residual,
            round_bf16,
        )?;
        self.residual = residual;
        self.next_pre = mix.pre;
        Ok(())
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl HcWeights {
    pub fn load_for_config(
        loader: &crate::SafetensorsMmapLoader,
        prefix: &str,
        config: &crate::v41_config::TextConfig,
    ) -> Result<Self, SingularityError> {
        Self::load(
            loader,
            prefix,
            config.hc_mult,
            config.hidden_size,
            config.rms_norm_eps,
            config.hc_eps,
            config.hc_sinkhorn_iters,
        )
    }
    /// Load the F32 projection, base and scale tensors from a checkpoint.
    pub fn load(
        loader: &crate::SafetensorsMmapLoader,
        prefix: &str,
        streams: usize,
        hidden: usize,
        norm_epsilon: f32,
        sinkhorn_epsilon: f32,
        sinkhorn_iterations: usize,
    ) -> Result<Self, SingularityError> {
        let outputs = streams
            .checked_add(2)
            .and_then(|x| x.checked_mul(streams))
            .ok_or_else(|| error("HC dimension overflow"))?;
        let inputs = streams
            .checked_mul(hidden)
            .ok_or_else(|| error("HC dimension overflow"))?;
        let load = |suffix: &str, shape: &[usize]| -> Result<Vec<f32>, SingularityError> {
            let name = format!("{prefix}_{suffix}");
            let info = loader
                .get_tensor_info(&name)
                .ok_or_else(|| error(&format!("Missing HC tensor {name}")))?;
            if info.dtype != "F32" || info.shape != shape {
                return Err(error(&format!("Invalid HC tensor {name}")));
            }
            let values = loader.get_tensor_f32(&name)?;
            let expected = shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .ok_or_else(|| error("HC dimension overflow"))?;
            if values.len() != expected {
                return Err(error("Invalid HC tensor payload length"));
            }
            Ok(values)
        };
        let projection = load("fn", &[outputs, inputs])?;
        let base = load("base", &[outputs])?;
        let scale: [f32; 3] = load("scale", &[3])?
            .try_into()
            .map_err(|_| error("Invalid HC scale"))?;
        let result = Self {
            streams,
            hidden,
            projection,
            scale,
            base,
            norm_epsilon,
            sinkhorn_epsilon,
            sinkhorn_iterations,
        };
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> Result<(usize, usize), SingularityError> {
        let n = self
            .streams
            .checked_mul(self.hidden)
            .ok_or_else(|| error("HC dimension overflow"))?;
        let m = self
            .streams
            .checked_add(2)
            .and_then(|x| x.checked_mul(self.streams))
            .ok_or_else(|| error("HC dimension overflow"))?;
        let weight_count = n
            .checked_mul(m)
            .ok_or_else(|| error("HC dimension overflow"))?;
        if n == 0
            || m == 0
            || self.projection.len() != weight_count
            || self.base.len() != m
            || self.sinkhorn_iterations == 0
            || !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.0
            || !self.sinkhorn_epsilon.is_finite()
            || self.sinkhorn_epsilon <= 0.0
        {
            return Err(error("Invalid Hyper-Connections configuration"));
        }
        Ok((n, m))
    }

    /// Compute dynamic coefficients from residual streams [streams, hidden].
    pub fn mixes(&self, residual: &[f32]) -> Result<HcMix, SingularityError> {
        let (n, m) = self.validate()?;
        if residual.len() != n {
            return Err(error("Invalid HC residual shape"));
        }
        let rms = (residual.iter().map(|v| v * v).sum::<f32>() / n as f32 + self.norm_epsilon)
            .sqrt()
            .recip();
        let projected: Vec<f32> = (0..m)
            .map(|i| {
                self.projection[i * n..(i + 1) * n]
                    .iter()
                    .zip(residual)
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
                    * rms
            })
            .collect();
        let h = self.streams;
        let pre = (0..h)
            .map(|i| sigmoid(projected[i] * self.scale[0] + self.base[i]) + self.sinkhorn_epsilon)
            .collect();
        let post = (0..h)
            .map(|i| 2.0 * sigmoid(projected[h + i] * self.scale[1] + self.base[h + i]))
            .collect();
        let mut comb: Vec<f32> = (0..h * h)
            .map(|i| projected[2 * h + i] * self.scale[2] + self.base[2 * h + i])
            .collect();
        for row in comb.chunks_exact_mut(h) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for v in row.iter_mut() {
                *v = (*v - max).exp();
            }
            let sum: f32 = row.iter().sum();
            for v in row.iter_mut() {
                *v = *v / sum + self.sinkhorn_epsilon;
            }
        }
        for iteration in 0..self.sinkhorn_iterations {
            if iteration > 0 {
                for row in comb.chunks_exact_mut(h) {
                    let sum = row.iter().sum::<f32>() + self.sinkhorn_epsilon;
                    for v in row {
                        *v /= sum;
                    }
                }
            }
            for col in 0..h {
                let sum =
                    (0..h).map(|row| comb[row * h + col]).sum::<f32>() + self.sinkhorn_epsilon;
                for row in 0..h {
                    comb[row * h + col] /= sum;
                }
            }
        }
        Ok(HcMix {
            pre,
            post,
            residual: comb,
        })
    }
}

/// Collapse residual streams with the pre-mix from the preceding sublayer.
/// `round_bf16` reproduces the checkpoint's BF16 activation boundary.
pub fn hc_pre(
    residual: &[f32],
    coefficients: &[f32],
    hidden: usize,
    round_bf16: bool,
) -> Result<Vec<f32>, SingularityError> {
    if hidden == 0
        || coefficients.is_empty()
        || coefficients.len().checked_mul(hidden) != Some(residual.len())
    {
        return Err(error("Invalid HC pre dimensions"));
    }
    let mut out = vec![0.0; hidden];
    for (stream, coefficient) in coefficients.iter().enumerate() {
        for (i, value) in out.iter_mut().enumerate() {
            *value += coefficient * residual[stream * hidden + i];
        }
    }
    if round_bf16 {
        for x in &mut out {
            *x = bf16::from_f32(*x).to_f32();
        }
    }
    Ok(out)
}

/// Add sublayer output and mix residual streams. `comb` is indexed `[source, destination]`.
pub fn hc_post(
    output: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    round_bf16: bool,
) -> Result<Vec<f32>, SingularityError> {
    let h = post.len();
    let d = output.len();
    if h == 0
        || d == 0
        || h.checked_mul(d) != Some(residual.len())
        || h.checked_mul(h) != Some(comb.len())
    {
        return Err(error("Invalid HC post dimensions"));
    }
    let mut result = vec![0.0; residual.len()];
    for dst in 0..h {
        for i in 0..d {
            let mixed: f32 = (0..h)
                .map(|src| comb[src * h + dst] * residual[src * d + i])
                .sum();
            let v = post[dst] * output[i] + mixed;
            result[dst * d + i] = if round_bf16 {
                bf16::from_f32(v).to_f32()
            } else {
                v
            };
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use sha2::{Digest, Sha256};

    #[derive(Deserialize)]
    struct Expected {
        pre: Vec<f32>,
        post: Vec<f32>,
        comb: Vec<f32>,
        collapsed: Vec<f32>,
        expanded: Vec<Vec<f32>>,
    }
    #[derive(Deserialize)]
    struct Vectors {
        checkpoint_sha256: String,
        checkpoint: Expected,
        synthetic: Expected,
        two_step_residual: Vec<f32>,
        two_step_pre: Vec<f32>,
    }
    fn vectors() -> Vectors {
        serde_json::from_str(include_str!("../tests/fixtures/v41_hyper_vectors.json")).unwrap()
    }
    fn check(weights: &HcWeights, expected: &Expected) {
        let d = weights.hidden;
        let residual: Vec<_> = (0..d * 4)
            .map(|i| (i as i32 % 97 - 48) as f32 / 64.0)
            .collect();
        let sublayer: Vec<_> = (0..d).map(|i| (i as i32 % 13 - 6) as f32 / 8.0).collect();
        let mixes = weights.mixes(&residual).unwrap();
        for (a, b) in mixes
            .pre
            .iter()
            .zip(&expected.pre)
            .chain(mixes.post.iter().zip(&expected.post))
            .chain(mixes.residual.iter().zip(&expected.comb))
        {
            assert!((a - b).abs() < 2e-5, "{a} != {b}");
        }
        let collapsed = hc_pre(&residual, &[1., 0., 0., 0.], d, true).unwrap();
        let expanded = hc_post(&sublayer, &residual, &mixes.post, &mixes.residual, true).unwrap();
        let indices = [0, 1, d / 2, d - 1];
        for (j, &idx) in indices.iter().enumerate() {
            assert_eq!(collapsed[idx], expected.collapsed[j]);
            for stream in 0..4 {
                assert!(
                    (expanded[stream * d + idx] - expected.expanded[stream][j]).abs() <= 0.008,
                    "stream {stream} idx {idx}"
                );
            }
        }
    }
    #[test]
    fn synthetic_hyper_connection_matches_torch_reference() {
        let weights = HcWeights {
            streams: 4,
            hidden: 8,
            projection: (0..24 * 32).map(|i| (i % 31 - 15) as f32 / 64.).collect(),
            base: (0..24).map(|i| (i % 11 - 5) as f32 / 8.).collect(),
            scale: [0.25, -0.5, 0.75],
            norm_epsilon: 1e-20,
            sinkhorn_epsilon: 1e-6,
            sinkhorn_iterations: 20,
        };
        check(&weights, &vectors().synthetic);
        let embedding: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) / 8.0).collect();
        let mut state = HcState::from_embedding(&embedding, 4).unwrap();
        for _ in 0..2 {
            state
                .step(&weights, true, |x| {
                    Ok(x.iter().map(|v| bf16::from_f32(v * 0.5).to_f32()).collect())
                })
                .unwrap();
        }
        let expected = vectors();
        for (actual, reference) in state.residual.iter().zip(expected.two_step_residual) {
            assert!(
                (actual - reference).abs() <= 0.008,
                "{actual} != {reference}"
            );
        }
        for (actual, reference) in state.next_pre.iter().zip(expected.two_step_pre) {
            assert!((actual - reference).abs() < 2e-5);
        }
        let before = state.residual.clone();
        assert!(state
            .step(&weights, true, |_| Err(error("sublayer failed")))
            .is_err());
        assert_eq!(before, state.residual);
        // Exercise real tensor names, shapes and F32 payloads through the loader.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let header = serde_json::json!({
            "layer.hc_attn_base": {"dtype":"F32", "shape":[24], "data_offsets":[0,96]},
            "layer.hc_attn_fn": {"dtype":"F32", "shape":[24,32], "data_offsets":[96,3168]},
            "layer.hc_attn_scale": {"dtype":"F32", "shape":[3], "data_offsets":[3168,3180]}
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        for value in weights
            .base
            .iter()
            .chain(&weights.projection)
            .chain(&weights.scale)
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let loader = crate::SafetensorsMmapLoader::open(path).unwrap();
        let loaded = HcWeights::load(&loader, "layer.hc_attn", 4, 8, 1e-20, 1e-6, 20).unwrap();
        check(&loaded, &vectors().synthetic);
        assert!(HcWeights::load(&loader, "layer.hc_attn", 4, 9, 1e-20, 1e-6, 20).is_err());
        assert!(hc_post(&[0.], &[0.], &[0.], &[], false).is_err());
        assert!(weights.mixes(&[]).is_err());
    }
    #[test]
    #[ignore = "requires SINGULARITY_V41_HC_ATTN_BYTES from pinned shard 9 range"]
    fn checkpoint_hyper_connection_matches_torch_reference() {
        let v = vectors();
        let raw = std::fs::read(std::env::var("SINGULARITY_V41_HC_ATTN_BYTES").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), v.checkpoint_sha256);
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(data.len(), 24 + 24 * 20480 + 3);
        let weights = HcWeights {
            streams: 4,
            hidden: 5120,
            base: data[..24].to_vec(),
            projection: data[24..24 + 24 * 20480].to_vec(),
            scale: data[data.len() - 3..].try_into().unwrap(),
            norm_epsilon: 1e-20,
            sinkhorn_epsilon: 1e-6,
            sinkhorn_iterations: 20,
        };
        check(&weights, &v.checkpoint);
    }
}
