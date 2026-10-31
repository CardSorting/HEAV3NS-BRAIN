//! CPU reference for the checkpoint's bidirectional vision transformer.
//! Uses BF16 boundaries and FP32 accumulation; not an optimized GPU backend.
use crate::{
    v41_block::RmsNorm, v41_compressor::Bf16Matrix, v41_config::VisionConfig, SafetensorsMmapLoader,
};
use half::bf16;
use singularity_core::SingularityError;
fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}
fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
struct Linear<'a> {
    weight: Bf16Matrix<'a>,
    bias: Option<Vec<f32>>,
}
impl<'a> Linear<'a> {
    fn load(
        loader: &'a SafetensorsMmapLoader,
        name: &str,
        rows: usize,
        cols: usize,
        bias: bool,
    ) -> Result<Self, SingularityError> {
        let weight = Bf16Matrix::load(loader, &format!("{name}.weight"), rows, cols)?;
        let bias = if bias {
            let key = format!("{name}.bias");
            let info = loader
                .get_tensor_info(&key)
                .ok_or_else(|| error(&format!("Missing {key}")))?;
            if info.dtype != "BF16" || info.shape != [rows] {
                return Err(error(&format!("Invalid {key}")));
            }
            Some(loader.get_tensor_f32(&key)?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, SingularityError> {
        let values = self.weight.project(input, false)?;
        Ok(values
            .into_iter()
            .enumerate()
            .map(|(i, v)| bf(v + self.bias.as_ref().map_or(0.0, |b| b[i])))
            .collect())
    }
}
struct Layer<'a> {
    norm1: RmsNorm,
    qkv: Linear<'a>,
    output: Linear<'a>,
    norm2: RmsNorm,
    up: Linear<'a>,
    down: Linear<'a>,
}
impl<'a> Layer<'a> {
    fn load(
        loader: &'a SafetensorsMmapLoader,
        c: &VisionConfig,
        i: usize,
    ) -> Result<Self, SingularityError> {
        let p = format!("vision.blocks.{i}");
        let d = c.hidden_size;
        let mid = c.intermediate_size;
        Ok(Self {
            norm1: RmsNorm::load(loader, &format!("{p}.norm1.weight"), d, 1e-6)?,
            qkv: Linear::load(
                loader,
                &format!("{p}.attn.wqkv"),
                d.checked_mul(3)
                    .ok_or_else(|| error("Vision size overflow"))?,
                d,
                true,
            )?,
            output: Linear::load(loader, &format!("{p}.attn.wo"), d, d, true)?,
            norm2: RmsNorm::load(loader, &format!("{p}.norm2.weight"), d, 1e-6)?,
            up: Linear::load(
                loader,
                &format!("{p}.mlp.w1"),
                mid.checked_mul(2)
                    .ok_or_else(|| error("Vision size overflow"))?,
                d,
                false,
            )?,
            down: Linear::load(loader, &format!("{p}.mlp.w2"), d, mid, false)?,
        })
    }
}
/// Split-half rotary encoding using separate row/column positions.
pub fn rotate_2d(
    input: &[f32],
    row: usize,
    col: usize,
    theta: f32,
) -> Result<Vec<f32>, SingularityError> {
    let dim = input.len();
    if dim == 0
        || dim % 4 != 0
        || !theta.is_finite()
        || theta <= 1.0
        || input.iter().any(|v| !v.is_finite())
    {
        return Err(error("Invalid vision rotary input"));
    }
    let half = dim / 2;
    let quarter = dim / 4;
    let mut out = vec![0.; dim];
    for i in 0..half {
        let position = if i < quarter { row } else { col };
        let local = i % quarter;
        let freq = theta.powf(-((2 * local) as f32) / half as f32);
        let (sin, cos) = ((position as f32) * freq).sin_cos();
        out[i] = bf(input[i] * cos - input[i + half] * sin);
        out[i + half] = bf(input[i + half] * cos + input[i] * sin);
    }
    Ok(out)
}

pub struct VisionEncoder<'a> {
    patch: Linear<'a>,
    layers: Vec<Layer<'a>>,
    norm: RmsNorm,
    dim: usize,
    heads: usize,
    patch_dim: usize,
    theta: f32,
}
impl<'a> VisionEncoder<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        c: &VisionConfig,
    ) -> Result<Self, SingularityError> {
        let d = c.hidden_size;
        let heads = c.num_attention_heads;
        if d == 0
            || heads == 0
            || d % heads != 0
            || (d / heads) % 4 != 0
            || c.num_hidden_layers == 0
            || c.intermediate_size == 0
            || c.patch_size == 0
            || !c.rope_theta.is_finite()
            || c.rope_theta <= 1.0
        {
            return Err(error("Invalid vision encoder config"));
        }
        let pd = c
            .patch_size
            .checked_mul(c.patch_size)
            .and_then(|x| x.checked_mul(3))
            .ok_or_else(|| error("Patch dimensions overflow"))?;
        let patch = Linear::load(loader, "vision.patch_embed.proj", d, pd, true)?;
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load(loader, c, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            patch,
            layers,
            norm: RmsNorm::load(loader, "vision.norm.weight", d, 1e-6)?,
            dim: d,
            heads,
            patch_dim: pd,
            theta: c.rope_theta,
        })
    }
    pub fn forward(
        &self,
        patches: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, SingularityError> {
        let count = height
            .checked_mul(width)
            .ok_or_else(|| error("Vision grid overflow"))?;
        if count == 0 || count.checked_mul(self.patch_dim) != Some(patches.len()) {
            return Err(error("Invalid vision patches"));
        }
        let mut x = Vec::with_capacity(count * self.dim);
        for patch in patches.chunks_exact(self.patch_dim) {
            let input: Vec<f32> = patch.iter().map(|v| bf(*v)).collect();
            x.extend(self.patch.forward(&input)?);
        }
        let head_dim = self.dim / self.heads;
        for layer in &self.layers {
            let mut q = Vec::with_capacity(count * self.dim);
            let mut k = Vec::with_capacity(count * self.dim);
            let mut v = Vec::with_capacity(count * self.dim);
            for (pos, row) in x.chunks_exact(self.dim).enumerate() {
                let norm = layer.norm1.forward(row, true)?;
                let qkv = layer.qkv.forward(&norm)?;
                for head in 0..self.heads {
                    let range = head * head_dim..(head + 1) * head_dim;
                    q.extend(rotate_2d(
                        &qkv[range.clone()],
                        pos / width,
                        pos % width,
                        self.theta,
                    )?);
                    k.extend(rotate_2d(
                        &qkv[self.dim + range.start..self.dim + range.end],
                        pos / width,
                        pos % width,
                        self.theta,
                    )?);
                    v.extend_from_slice(&qkv[2 * self.dim + range.start..2 * self.dim + range.end]);
                }
            }
            let mut attended = vec![0.; count * self.dim];
            let scale = (head_dim as f32).sqrt().recip();
            for pos in 0..count {
                for head in 0..self.heads {
                    let start = pos * self.dim + head * head_dim;
                    let query = &q[start..start + head_dim];
                    let scores: Vec<f32> = (0..count)
                        .map(|j| {
                            let offset = j * self.dim + head * head_dim;
                            query
                                .iter()
                                .zip(&k[offset..offset + head_dim])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                * scale
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f32 = exp.iter().sum();
                    for d in 0..head_dim {
                        let value: f32 = (0..count)
                            .map(|j| exp[j] / sum * v[j * self.dim + head * head_dim + d])
                            .sum();
                        attended[start + d] = bf(value);
                    }
                }
            }
            for pos in 0..count {
                let range = pos * self.dim..(pos + 1) * self.dim;
                let projected = layer.output.forward(&attended[range.clone()])?;
                for (a, b) in x[range.clone()].iter_mut().zip(projected) {
                    *a = bf(*a + b);
                }
                let norm = layer.norm2.forward(&x[range.clone()], true)?;
                let up = layer.up.forward(&norm)?;
                let mid = up.len() / 2;
                let activation: Vec<f32> = (0..mid)
                    .map(|j| bf(bf(up[j] / (1.0 + (-up[j]).exp())) * up[mid + j]))
                    .collect();
                let down = layer.down.forward(&activation)?;
                for (a, b) in x[range].iter_mut().zip(down) {
                    *a = bf(*a + b);
                }
            }
        }
        let mut output = Vec::with_capacity(x.len());
        for row in x.chunks_exact(self.dim) {
            output.extend(self.norm.forward(row, true)?);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    struct TensorData {
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct VitVectors {
        patches: Vec<f32>,
        output: Vec<f32>,
        tensors: BTreeMap<String, TensorData>,
    }

    #[test]
    fn vision_encoder_matches_checkpoint_reference_vectors() {
        let vectors: VitVectors =
            serde_json::from_str(include_str!("../tests/fixtures/v41_vit_vectors.json")).unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, tensor) in &vectors.tensors {
            let start = payload.len();
            for &v in &tensor.values {
                payload.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
            header.insert(
                format!("vision.{name}"),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": tensor.shape,
                    "data_offsets": [start, payload.len()]
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        file_bytes.extend(header_bytes);
        file_bytes.extend(payload);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, file_bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();

        let config = VisionConfig {
            num_hidden_layers: 2,
            hidden_size: 8,
            num_attention_heads: 2,
            intermediate_size: 6,
            patch_size: 2,
            rope_theta: 10000.0,
            downsample_ratio: 2,
            max_image_tokens: 64,
            min_pixels: 16,
            max_wh_ratio: Some(4),
        };
        let encoder = VisionEncoder::load(&loader, &config).unwrap();
        let actual = encoder.forward(&vectors.patches, 2, 2).unwrap();
        assert_eq!(actual.len(), vectors.output.len());
        for (a, b) in actual.iter().zip(&vectors.output) {
            assert!((a - b).abs() <= 0.01 * b.abs().max(0.01), "{a} != {b}");
        }

        // Error and boundary cases
        assert!(encoder.forward(&vectors.patches, 0, 2).is_err());
        assert!(encoder.forward(&vectors.patches[..12], 2, 2).is_err());
        assert!(rotate_2d(&[1.0, 2.0], 0, 0, 10000.0).is_err());
        assert!(rotate_2d(&[1.0, 2.0, 3.0, 4.0], 0, 0, 1.0).is_err());
    }
}
