//! V4.1 Engram lookup and gated residual CPU reference.
//! Based on checkpoint inference/model.py (MIT), revision
//! 1bae0e1088993b9b82de08f59de5be5a34e42170.
//! This is a single-process reference, not distributed GPU placement.
use crate::v41_quant::{e4m3fn, ue8m0, CheckpointMatrix};
use half::bf16;
use singularity_core::SingularityError;

use crate::model_error as error;

pub struct EngramTable<'a> {
    weights: &'a [u8],
    scales: &'a [u8],
    rows: usize,
    dim: usize,
}
impl<'a> EngramTable<'a> {
    pub fn load(
        loader: &'a crate::SafetensorsMmapLoader,
        prefix: &str,
    ) -> Result<Self, SingularityError> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.scale");
        let weight = loader
            .get_tensor_info(&weight_name)
            .ok_or_else(|| error("Missing Engram embedding"))?;
        let scale = loader
            .get_tensor_info(&scale_name)
            .ok_or_else(|| error("Missing Engram embedding scale"))?;
        if weight.dtype != "F8_E4M3"
            || weight.shape.len() != 2
            || scale.dtype != "F8_E8M0"
            || scale.shape.len() != 2
        {
            return Err(error("Invalid Engram embedding types or ranks"));
        }
        let (rows, dim) = (weight.shape[0], weight.shape[1]);
        if rows == 0 || dim == 0 || dim % 32 != 0 || scale.shape != [rows, dim / 32] {
            return Err(error("Invalid Engram embedding shape"));
        }
        let weights = loader.get_tensor_bytes(&weight_name)?;
        let scales = loader.get_tensor_bytes(&scale_name)?;
        if rows.checked_mul(dim) != Some(weights.len())
            || rows.checked_mul(dim / 32) != Some(scales.len())
        {
            return Err(error("Invalid Engram embedding payload size"));
        }
        Ok(Self {
            weights,
            scales,
            rows,
            dim,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Read selected rows in requested order and dequantize to BF16 values.
    pub fn lookup(&self, ids: &[u64]) -> Result<Vec<f32>, SingularityError> {
        let total = ids
            .len()
            .checked_mul(self.dim)
            .ok_or_else(|| error("Engram lookup size overflow"))?;
        let mut out = Vec::with_capacity(total);
        for &id in ids {
            let row = usize::try_from(id).map_err(|_| error("Engram row outside address space"))?;
            if row >= self.rows {
                return Err(error("Engram row out of range"));
            }
            for j in 0..self.dim {
                let value = e4m3fn(self.weights[row * self.dim + j])
                    * ue8m0(self.scales[row * (self.dim / 32) + j / 32]);
                out.push(bf16::from_f32(value).to_f32());
            }
        }
        Ok(out)
    }
}

/// Fused normalized key/stream gate. `projected` contains the per-stream keys
/// followed by one shared value; all tensors are already BF16 represented as f32.
#[allow(clippy::too_many_arguments)]
pub fn gated_residual(
    residual: &[f32],
    projected: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    streams: usize,
    dim: usize,
    epsilon: f32,
    image_token: bool,
) -> Result<Vec<f32>, SingularityError> {
    let n = streams
        .checked_mul(dim)
        .ok_or_else(|| error("Engram dimension overflow"))?;
    let projected_len = n
        .checked_add(dim)
        .ok_or_else(|| error("Engram dimension overflow"))?;
    if n == 0
        || residual.len() != n
        || q_weight.len() != n
        || k_weight.len() != n
        || projected.len() != projected_len
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || residual
            .iter()
            .chain(projected)
            .chain(q_weight)
            .chain(k_weight)
            .any(|v| !v.is_finite())
    {
        return Err(error("Invalid Engram gate inputs"));
    }
    let mut output = vec![0.; n];
    for stream in 0..streams {
        let range = stream * dim..(stream + 1) * dim;
        let h = &residual[range.clone()];
        let key = &projected[range.clone()];
        let hvar = h.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let kvar = key.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let mut dot = 0.0;
        for j in 0..dim {
            dot += h[j] * (q_weight[stream * dim + j] * k_weight[stream * dim + j]) * key[j];
        }
        dot *= (hvar + epsilon).sqrt().recip()
            * (kvar + epsilon).sqrt().recip()
            * (dim as f32).sqrt().recip();
        let signed = dot.abs().max(1e-6).sqrt().copysign(dot);
        let gate = if image_token {
            0.0
        } else {
            1.0 / (1.0 + (-signed).exp())
        };
        for j in 0..dim {
            output[stream * dim + j] = bf16::from_f32(h[j] + gate * projected[n + j]).to_f32();
        }
    }
    Ok(output)
}

pub struct Engram<'a> {
    pub table: EngramTable<'a>,
    pub projection: CheckpointMatrix<'a>,
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub streams: usize,
    pub hidden: usize,
    pub columns: usize,
    pub epsilon: f32,
}
impl<'a> Engram<'a> {
    pub fn load(
        loader: &'a crate::SafetensorsMmapLoader,
        prefix: &str,
        config: &crate::v41_config::TextConfig,
        layer_id: usize,
    ) -> Result<Self, SingularityError> {
        let index = config
            .engram_layer_ids
            .iter()
            .position(|&id| id == layer_id)
            .ok_or_else(|| error("Layer does not own Engram"))?;
        let layout = crate::engram::EngramLayout::from_config(config)?;
        let columns = (config.engram_max_ngram_size - 1)
            .checked_mul(config.engram_n_heads)
            .ok_or_else(|| error("Engram column overflow"))?;
        let table = EngramTable::load(loader, &format!("{prefix}.embed"))?;
        if table.dim() != config.engram_head_dim
            || u64::try_from(table.rows()).ok() != Some(layout.layers[index].rows)
        {
            return Err(error("Engram table differs from checkpoint config"));
        }
        let projection = CheckpointMatrix::load(
            loader,
            &format!("{prefix}.wkv.weight"),
            &format!("{prefix}.wkv.scale"),
        )?;
        let out = config
            .hc_mult
            .checked_add(1)
            .and_then(|v| v.checked_mul(config.hidden_size))
            .ok_or_else(|| error("Engram projection overflow"))?;
        let input = columns
            .checked_mul(table.dim())
            .ok_or_else(|| error("Engram projection overflow"))?;
        if projection.shape() != (out, input)
            || !matches!(projection, CheckpointMatrix::Dense { .. })
        {
            return Err(error("Invalid Engram projection"));
        }
        let get = |suffix: &str| -> Result<Vec<f32>, SingularityError> {
            let name = format!("{prefix}.{suffix}");
            let info = loader
                .get_tensor_info(&name)
                .ok_or_else(|| error("Missing Engram gate weight"))?;
            if info.dtype != "BF16" || info.shape != [config.hc_mult, config.hidden_size] {
                return Err(error("Invalid Engram gate weight"));
            }
            let value = loader.get_tensor_f32(&name)?;
            if value.len()
                != config
                    .hc_mult
                    .checked_mul(config.hidden_size)
                    .unwrap_or(usize::MAX)
            {
                return Err(error("Invalid Engram gate weight size"));
            }
            Ok(value)
        };
        Ok(Self {
            table,
            projection,
            q_weight: get("q_weight")?,
            k_weight: get("k_weight")?,
            streams: config.hc_mult,
            hidden: config.hidden_size,
            columns,
            epsilon: config.rms_norm_eps,
        })
    }
    pub fn forward(
        &self,
        residual: &[f32],
        ids: &[u64],
        image_token: bool,
    ) -> Result<Vec<f32>, SingularityError> {
        if ids.len() != self.columns {
            return Err(error("Invalid Engram hash count"));
        }
        let embeddings = self.table.lookup(ids)?;
        let projected = self.projection.matvec_quantized(&embeddings)?;
        gated_residual(
            residual,
            &projected,
            &self.q_weight,
            &self.k_weight,
            self.streams,
            self.hidden,
            self.epsilon,
            image_token,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    fn checkpoint(path: &std::path::Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        for (name, dtype, shape, data) in tensors {
            let end = offset + data.len();
            header.insert(
                (*name).into(),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,end]}),
            );
            offset = end;
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for (_, _, _, data) in tensors {
            file.write_all(data).unwrap();
        }
    }
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
            .collect()
    }
    #[derive(Deserialize)]
    struct Synthetic {
        output: Vec<f32>,
    }
    #[test]
    fn complete_miniature_engram_matches_torch() {
        let mut config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut config.text_config;
        c.engram_layer_ids = vec![1];
        c.engram_num_embeddings = vec![2];
        c.engram_max_ngram_size = 2;
        c.engram_vocab_size = 2;
        c.engram_n_heads = 1;
        c.engram_head_dim = 32;
        c.hc_mult = 2;
        c.hidden_size = 32;
        let h: Vec<f32> = (0..64).map(|i| (i % 13 - 6) as f32 / 8.).collect();
        let q: Vec<f32> = (0..64).map(|i| (i % 7 - 3) as f32 / 4.).collect();
        let k: Vec<f32> = (0..64).map(|i| (i % 11 - 5) as f32 / 8.).collect();
        let rows: Vec<u8> = (0..64).map(|i| ((i * 7 + 3) % 112) as u8).collect();
        let projection: Vec<u8> = (0..96 * 32).map(|i| ((i * 13 + 5) % 112) as u8).collect();
        let tensors = vec![
            ("layer.embed.weight", "F8_E4M3", vec![2, 32], rows),
            ("layer.embed.scale", "F8_E8M0", vec![2, 1], vec![125, 126]),
            ("layer.wkv.weight", "F8_E4M3", vec![96, 32], projection),
            ("layer.wkv.scale", "F8_E8M0", vec![3, 1], vec![127; 3]),
            ("layer.q_weight", "BF16", vec![2, 32], bf16_bytes(&q)),
            ("layer.k_weight", "BF16", vec![2, 32], bf16_bytes(&k)),
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        checkpoint(&path, &tensors);
        let loader = crate::SafetensorsMmapLoader::open(path).unwrap();
        let layer = Engram::load(&loader, "layer", c, 1).unwrap();
        let output = layer.forward(&h, &[1], false).unwrap();
        let expected: Synthetic =
            serde_json::from_str(include_str!("../tests/fixtures/v41_engram_synthetic.json"))
                .unwrap();
        for (a, b) in output.iter().zip(expected.output) {
            assert!((a - b).abs() <= 0.008 * b.abs().max(1.0), "{a} != {b}");
        }
        assert_eq!(layer.forward(&h, &[1], true).unwrap(), h);
        assert!(layer.forward(&h, &[2], false).is_err());
        assert!(layer.forward(&h, &[], false).is_err());
    }
    #[derive(Deserialize)]
    struct RealVectors {
        weight_sha: String,
        scale_sha: String,
        qk_sha: String,
        embeddings: Vec<f32>,
        gate_output: Vec<f32>,
    }
    #[derive(Deserialize)]
    struct FullRealVectors {
        hashes: Vec<String>,
        output: Vec<f32>,
    }
    #[test]
    #[ignore = "requires pinned shard-48 Engram rows, scales and q/k weights"]
    fn checkpoint_rows_and_gate_match_torch() {
        let expected: RealVectors =
            serde_json::from_str(include_str!("../tests/fixtures/v41_engram_vectors.json"))
                .unwrap();
        let rows = std::fs::read(std::env::var("SINGULARITY_ENGRAM_ROWS").unwrap()).unwrap();
        let scales = std::fs::read(std::env::var("SINGULARITY_ENGRAM_SCALES").unwrap()).unwrap();
        let qk = std::fs::read(std::env::var("SINGULARITY_ENGRAM_QK").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&rows)), expected.weight_sha);
        assert_eq!(format!("{:x}", Sha256::digest(&scales)), expected.scale_sha);
        assert_eq!(format!("{:x}", Sha256::digest(&qk)), expected.qk_sha);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        checkpoint(
            &path,
            &[
                ("embed.weight", "F8_E4M3", vec![4, 256], rows),
                ("embed.scale", "F8_E8M0", vec![4, 8], scales),
            ],
        );
        let loader = crate::SafetensorsMmapLoader::open(path).unwrap();
        let table = EngramTable::load(&loader, "embed").unwrap();
        assert_eq!(table.lookup(&[0, 1, 2, 3]).unwrap(), expected.embeddings);
        let weights: Vec<f32> = qk
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect();
        let h: Vec<f32> = (0..20480).map(|i| (i % 97 - 48) as f32 / 64.).collect();
        let mut projected: Vec<f32> = (0..20480).map(|i| (i % 89 - 44) as f32 / 64.).collect();
        projected.extend((0..5120).map(|i| (i % 37 - 18) as f32 / 32.));
        let out = gated_residual(
            &h,
            &projected,
            &weights[..20480],
            &weights[20480..],
            4,
            5120,
            1e-20,
            false,
        )
        .unwrap();
        assert_eq!(out.len(), expected.gate_output.len());
        let mut error = 0f64;
        let mut norm = 0f64;
        for (a, b) in out.iter().zip(expected.gate_output) {
            error += (*a as f64 - b as f64).powi(2);
            norm += (b as f64).powi(2);
        }
        let relative = (error / norm).sqrt();
        eprintln!("Engram gate relative RMS error: {relative:.8}");
        assert!(relative < 0.005);
    }

    #[test]
    #[ignore = "requires pinned shard-48 Engram rows, q/k and 157 MB projection"]
    fn real_projection_lookup_and_gate_match_torch() {
        let expected: FullRealVectors = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_real_engram_forward.json"
        ))
        .unwrap();
        let names = [
            "SINGULARITY_ENGRAM_ROWS",
            "SINGULARITY_ENGRAM_SCALES",
            "SINGULARITY_ENGRAM_QK",
            "SINGULARITY_ENGRAM_WKV",
            "SINGULARITY_ENGRAM_WKV_SCALES",
        ];
        let mut raw: Vec<Vec<u8>> = names
            .iter()
            .map(|name| std::fs::read(std::env::var(name).unwrap()).unwrap())
            .collect();
        for (bytes, hash) in raw.iter().zip(&expected.hashes) {
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), *hash);
        }
        assert_eq!(
            raw.iter().map(Vec::len).collect::<Vec<_>>(),
            [1024, 32, 81920, 157286400, 153600]
        );
        let qk = std::mem::take(&mut raw[2]);
        let tensors = vec![
            (
                "embed.weight",
                "F8_E4M3",
                vec![4, 256],
                std::mem::take(&mut raw[0]),
            ),
            (
                "embed.scale",
                "F8_E8M0",
                vec![4, 8],
                std::mem::take(&mut raw[1]),
            ),
            (
                "wkv.weight",
                "F8_E4M3",
                vec![25600, 6144],
                std::mem::take(&mut raw[3]),
            ),
            (
                "wkv.scale",
                "F8_E8M0",
                vec![800, 192],
                std::mem::take(&mut raw[4]),
            ),
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("subset.safetensors");
        checkpoint(&path, &tensors);
        drop(tensors);
        let loader = crate::SafetensorsMmapLoader::open(path).unwrap();
        let qk: Vec<f32> = qk
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect();
        let layer = Engram {
            table: EngramTable::load(&loader, "embed").unwrap(),
            projection: CheckpointMatrix::load(&loader, "wkv.weight", "wkv.scale").unwrap(),
            q_weight: qk[..20480].to_vec(),
            k_weight: qk[20480..].to_vec(),
            streams: 4,
            hidden: 5120,
            columns: 24,
            epsilon: 1e-20,
        };
        let h: Vec<f32> = (0..20480).map(|i| (i % 97 - 48) as f32 / 64.).collect();
        let ids: Vec<u64> = (0..24).map(|i| i % 4).collect();
        let output = layer.forward(&h, &ids, false).unwrap();
        assert_eq!(output.len(), expected.output.len());
        let mut error = 0f64;
        let mut norm = 0f64;
        for (a, b) in output.iter().zip(expected.output) {
            error += (*a as f64 - b as f64).powi(2);
            norm += (b as f64).powi(2);
        }
        let relative = (error / norm).sqrt();
        eprintln!("Real Engram forward relative RMS error: {relative:.8}");
        assert!(relative < 0.005);
    }
}
