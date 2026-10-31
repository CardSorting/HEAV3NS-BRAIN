//! Checkpoint-backed token embedding and output head for the V4.1 CPU reference.
use crate::{
    v41_block::RmsNorm,
    v41_config::TextConfig,
    v41_hyper::{hc_pre, HcState},
    SafetensorsMmapLoader,
};
use half::bf16;
use singularity_core::SingularityError;

fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

pub struct TokenIo<'a> {
    embedding: &'a [u8],
    head: &'a [u8],
    norm: RmsNorm,
    vocab: usize,
    hidden: usize,
    streams: usize,
}

impl<'a> TokenIo<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        config: &TextConfig,
    ) -> Result<Self, SingularityError> {
        let vocab = config.vocab_size;
        let hidden = config.hidden_size;
        if vocab == 0
            || hidden == 0
            || config.hc_mult == 0
            || vocab
                .checked_mul(hidden)
                .and_then(|n| n.checked_mul(2))
                .is_none()
        {
            return Err(error("Invalid token IO dimensions"));
        }
        let matrix = |name: &str| -> Result<&'a [u8], SingularityError> {
            let info = loader
                .get_tensor_info(name)
                .ok_or_else(|| error(&format!("Missing tensor {name}")))?;
            if info.dtype != "BF16" || info.shape != [vocab, hidden] {
                return Err(error(&format!("Invalid tensor {name}")));
            }
            loader.get_tensor_bytes(name)
        };
        Ok(Self {
            embedding: matrix("embed.weight")?,
            head: matrix("head.weight")?,
            norm: RmsNorm::load(loader, "norm.weight", hidden, config.rms_norm_eps)?,
            vocab,
            hidden,
            streams: config.hc_mult,
        })
    }

    /// Look up a token and initialize its independent residual streams.
    pub fn embed(&self, token: usize) -> Result<HcState, SingularityError> {
        if token >= self.vocab {
            return Err(error("Token ID outside vocabulary"));
        }
        let row_bytes = self.hidden * 2;
        let row = &self.embedding[token * row_bytes..(token + 1) * row_bytes];
        let values: Vec<f32> = row
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        HcState::from_embedding(&values, self.streams)
    }

    /// Initialize independent residual streams from a precomputed embedding (e.g. aligned vision span).
    pub fn embed_custom(&self, embedding: &[f32]) -> Result<HcState, SingularityError> {
        if embedding.len() != self.hidden {
            return Err(error("Custom embedding dimension mismatch"));
        }
        let values: Vec<f32> = embedding
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect();
        HcState::from_embedding(&values, self.streams)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Collapse the final shifted HC state, normalize in BF16, and compute FP32 logits.
    pub fn logits(&self, state: &HcState) -> Result<Vec<f32>, SingularityError> {
        if state.residual.len() != self.streams * self.hidden
            || state.next_pre.len() != self.streams
        {
            return Err(error("Invalid final HC state"));
        }
        let collapsed = hc_pre(&state.residual, &state.next_pre, self.hidden, true)?;
        let input = self.norm.forward(&collapsed, true)?;
        let row_bytes = self.hidden * 2;
        if self.vocab >= 64 {
            use rayon::prelude::*;
            Ok(self
                .head
                .par_chunks_exact(row_bytes)
                .with_min_len(256)
                .map(|row| {
                    row.chunks_exact(2)
                        .zip(&input)
                        .map(|(b, x)| bf16::from_le_bytes([b[0], b[1]]).to_f32() * x)
                        .sum::<f32>()
                })
                .collect())
        } else {
            Ok(self
                .head
                .chunks_exact(row_bytes)
                .map(|row| {
                    row.chunks_exact(2)
                        .zip(&input)
                        .map(|(b, x)| bf16::from_le_bytes([b[0], b[1]]).to_f32() * x)
                        .sum::<f32>()
                })
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loads_embedding_and_head_and_computes_logits() {
        let mut config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut config.text_config;
        c.vocab_size = 2;
        c.hidden_size = 2;
        c.hc_mult = 2;
        c.rms_norm_eps = 1e-6;
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape, values) in [
            ("embed.weight", vec![2, 2], vec![1., 2., 3., 4.]),
            ("head.weight", vec![2, 2], vec![1., 0., 0., 1.]),
            ("norm.weight", vec![2], vec![1., 1.]),
        ] {
            let start = payload.len();
            for value in values {
                payload.extend_from_slice(&bf16::from_f32(value).to_le_bytes());
            }
            header.insert(name.into(), serde_json::json!({"dtype":"BF16", "shape":shape, "data_offsets":[start,payload.len()]}));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let io = TokenIo::load(&loader, c).unwrap();
        assert!(io.embed(2).is_err());
        let state = io.embed(1).unwrap();
        assert_eq!(state.residual, vec![3., 4., 3., 4.]);
        let logits = io.logits(&state).unwrap();
        assert_eq!(
            logits,
            vec![
                bf16::from_f32(3.0 / 12.5f32.sqrt()).to_f32(),
                bf16::from_f32(4.0 / 12.5f32.sqrt()).to_f32()
            ]
        );
    }
}
