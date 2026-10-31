//! Incremental CPU reference for a V4.1 compressed-KV source.
//! Output is the normalized latent before rotary encoding and FP4 quantization.
use crate::{
    v41_block::RmsNorm, v41_config::TextConfig, v41_sparse::CompressionPool, SafetensorsMmapLoader,
};
use half::bf16;
use singularity_core::SingularityError;

fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

#[derive(Clone)]
pub(crate) struct Bf16Matrix<'a> {
    data: &'a [u8],
    rows: usize,
    cols: usize,
}
impl<'a> Bf16Matrix<'a> {
    pub(crate) fn row(&self, index: usize) -> Result<Vec<f32>, SingularityError> {
        if index >= self.rows {
            return Err(error("BF16 matrix row out of range"));
        }
        let start = index * self.cols * 2;
        Ok(self.data[start..start + self.cols * 2]
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect())
    }
    pub(crate) fn load(
        loader: &'a SafetensorsMmapLoader,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, SingularityError> {
        let info = loader
            .get_tensor_info(name)
            .ok_or_else(|| error(&format!("Missing compressor tensor {name}")))?;
        if rows == 0 || cols == 0 || info.dtype != "BF16" || info.shape != [rows, cols] {
            return Err(error(&format!("Invalid compressor tensor {name}")));
        }
        Ok(Self {
            data: loader.get_tensor_bytes(name)?,
            rows,
            cols,
        })
    }
    pub(crate) fn project(&self, input: &[f32], round: bool) -> Result<Vec<f32>, SingularityError> {
        if input.len() != self.cols || input.iter().any(|x| !x.is_finite()) {
            return Err(error("Invalid compressor input"));
        }
        let compute = |row: &[u8]| -> f32 {
            let y: f32 = row
                .chunks_exact(2)
                .zip(input)
                .map(|(v, x)| bf16::from_le_bytes([v[0], v[1]]).to_f32() * x)
                .sum();
            if round {
                bf16::from_f32(y).to_f32()
            } else {
                y
            }
        };
        if self.rows >= 64 {
            use rayon::prelude::*;
            Ok(self
                .data
                .par_chunks_exact(self.cols * 2)
                .with_min_len(64)
                .map(compute)
                .collect())
        } else {
            Ok(self.data.chunks_exact(self.cols * 2).map(compute).collect())
        }
    }
}

#[derive(Clone)]
pub struct Compressor<'a> {
    kv: Bf16Matrix<'a>,
    gate: Option<Bf16Matrix<'a>>,
    norm: RmsNorm,
    pool: CompressionPool,
    ratio: usize,
    position: usize,
}
impl<'a> Compressor<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        config: &TextConfig,
        layer: usize,
    ) -> Result<Self, SingularityError> {
        if layer >= config.num_hidden_layers || !config.kv_source_layer_ids.contains(&layer) {
            return Err(error("Layer is not a backbone KV source"));
        }
        let ratio = *config
            .compress_ratios
            .get(layer)
            .ok_or_else(|| error("Missing compression ratio"))?;
        if ratio == 0 {
            return Err(error("KV source must use compression"));
        }
        let prefix = format!("layers.{layer}.attn.compressor");
        let kv = Bf16Matrix::load(
            loader,
            &format!("{prefix}.wkv.weight"),
            config.head_dim,
            config.hidden_size,
        )?;
        let gate = if ratio > 1 {
            Some(Bf16Matrix::load(
                loader,
                &format!("{prefix}.wgate.weight"),
                config.head_dim,
                config.hidden_size,
            )?)
        } else {
            None
        };
        Ok(Self {
            kv,
            gate,
            norm: RmsNorm::load(
                loader,
                &format!("{prefix}.norm.weight"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            pool: CompressionPool::new(ratio, config.head_dim)?,
            ratio,
            position: 0,
        })
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) -> Result<(), SingularityError> {
        self.pool = CompressionPool::new(self.ratio, self.kv.rows)?;
        self.position = 0;
        Ok(())
    }
    /// Consume exactly one sequential token. No state changes on error.
    pub fn push(&mut self, input: &[f32]) -> Result<Option<Vec<f32>>, SingularityError> {
        let next_position = self
            .position
            .checked_add(1)
            .ok_or_else(|| error("Compressor position overflow"))?;
        // The checkpoint input is BF16 before promotion for pooled layers.
        let input: Vec<f32> = input.iter().map(|x| bf16::from_f32(*x).to_f32()).collect();
        let kv = self.kv.project(&input, self.ratio == 1)?;
        let gate = match &self.gate {
            Some(gate) => gate.project(&input, false)?,
            None => Vec::new(),
        };
        let mut pool = self.pool.clone();
        let latent = match pool.push(&kv, &gate)? {
            Some(v) => {
                let v: Vec<f32> = v.into_iter().map(|x| bf16::from_f32(x).to_f32()).collect();
                Some(self.norm.forward(&v, true)?)
            }
            None => None,
        };
        self.pool = pool;
        self.position = next_position;
        Ok(latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires SINGULARITY_V41_COMPRESSOR pointing to pinned layer-2 compressor range"]
    fn real_checkpoint_compressor_matches_pytorch() {
        use sha2::{Digest, Sha256};
        #[derive(serde::Deserialize)]
        struct Expected {
            sha256: String,
            outputs: Vec<Vec<f32>>,
        }
        let expected: Expected = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_compressor_real_vectors.json"
        ))
        .unwrap();
        let bytes = std::fs::read(std::env::var("SINGULARITY_V41_COMPRESSOR").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected.sha256);
        let header = serde_json::json!({
            "layers.2.attn.compressor.norm.weight": {"dtype":"BF16","shape":[512],"data_offsets":[0,1024]},
            "layers.2.attn.compressor.wgate.weight": {"dtype":"BF16","shape":[512,5120],"data_offsets":[1024,5243904]},
            "layers.2.attn.compressor.wkv.weight": {"dtype":"BF16","shape":[512,5120],"data_offsets":[5243904,10486784]}
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(bytes);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, file).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let mut compressor = Compressor::load(&loader, &config.text_config, 2).unwrap();
        for t in 0..4 {
            let input: Vec<f32> = (0..5120)
                .map(|i| (((i * 13 + t * 7) % 97) as i32 - 48) as f32 / 64.0)
                .collect();
            let output = compressor.push(&input).unwrap();
            if t % 2 == 0 {
                assert!(output.is_none());
                continue;
            }
            let output = output.unwrap();
            let reference = &expected.outputs[t / 2];
            assert_eq!(output.len(), reference.len());
            let err: f64 = output
                .iter()
                .zip(reference)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            let norm: f64 = reference.iter().map(|x| (*x as f64).powi(2)).sum();
            let rms = (err / norm).sqrt();
            eprintln!("group {} relative RMS: {rms:.8}", t / 2);
            assert!(rms < 0.01);
        }
    }
    #[test]
    fn checkpoint_pool_waits_for_groups_and_reset_discards_partial() {
        let mut config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut config.text_config;
        c.hidden_size = 2;
        c.head_dim = 2;
        c.rms_norm_eps = 1e-6;
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in [
            (
                "layers.2.attn.compressor.wkv.weight",
                vec![2, 2],
                vec![1., 0., 0., 1.],
            ),
            (
                "layers.2.attn.compressor.wgate.weight",
                vec![2, 2],
                vec![0., 0., 0., 0.],
            ),
            (
                "layers.2.attn.compressor.norm.weight",
                vec![2],
                vec![1., 1.],
            ),
            (
                "layers.20.attn.compressor.wkv.weight",
                vec![2, 2],
                vec![1., 0., 0., 1.],
            ),
            (
                "layers.20.attn.compressor.norm.weight",
                vec![2],
                vec![1., 1.],
            ),
        ] {
            let start = data.len();
            for v in values {
                data.extend(bf16::from_f32(v).to_le_bytes());
            }
            header.insert(
                name.into(),
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,data.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let mut compressor = Compressor::load(&loader, c, 2).unwrap();
        assert!(compressor.push(&[2., 4.]).unwrap().is_none());
        assert!(compressor.push(&[1.]).is_err());
        assert_eq!(compressor.position(), 1);
        let out = compressor.push(&[4., 8.]).unwrap().unwrap();
        assert_eq!(
            out,
            vec![
                bf16::from_f32(3.0 / 22.5f32.sqrt()).to_f32(),
                bf16::from_f32(6.0 / 22.5f32.sqrt()).to_f32()
            ]
        );
        assert!(compressor.push(&[2., 4.]).unwrap().is_none());
        compressor.reset().unwrap();
        assert_eq!(compressor.position(), 0);
        assert!(compressor.push(&[4., 8.]).unwrap().is_none());
        let mut direct = Compressor::load(&loader, c, 20).unwrap();
        assert_eq!(direct.push(&[3., 6.]).unwrap().unwrap(), out);
    }
}
