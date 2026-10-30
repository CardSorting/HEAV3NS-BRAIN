//! Checkpoint-specific DSpark input and draft head CPU reference.
//! Draft attention stages and target verification are not implemented here.
use crate::{
    v41_block::RmsNorm,
    v41_compressor::Bf16Matrix,
    v41_config::TextConfig,
    v41_hyper::{hc_pre, HcState},
    v41_io::TokenIo,
    v41_quant::CheckpointMatrix,
    SafetensorsMmapLoader,
};
use singularity_core::SingularityError;
fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

/// One checkpoint DSpark stage. Draft attention sees the entire draft block;
/// each token then independently passes through the stage's routed experts.
pub struct DraftStage<'a> {
    block: crate::v41_block::Block,
    attention: crate::v41_attention::LoadedWindowAttention<'a>,
    moe: crate::v41_moe::LoadedMoe<'a>,
    block_size: usize,
}
impl<'a> DraftStage<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        config: &TextConfig,
        stage: usize,
    ) -> Result<Self, SingularityError> {
        if config.dspark_block_size == 0 {
            return Err(error("Invalid DSpark block size"));
        }
        Ok(Self {
            block: crate::v41_block::Block::load_draft(loader, stage, config)?,
            attention: crate::v41_attention::LoadedWindowAttention::load_draft(
                loader, config, stage,
            )?,
            moe: crate::v41_moe::LoadedMoe::load_draft(loader, stage, config)?,
            block_size: config.dspark_block_size,
        })
    }
    pub fn position(&self) -> usize {
        self.attention.attention.cache.position()
    }
    pub fn reset(&mut self) {
        self.attention.attention.cache.clear();
    }
    pub fn seed_main(&mut self, main: &[f32]) -> Result<(), SingularityError> {
        self.attention.seed_draft_main(main)
    }
    pub fn forward(
        &mut self,
        main: &[f32],
        states: &mut [HcState],
    ) -> Result<(), SingularityError> {
        if states.len() != self.block_size {
            return Err(error("Invalid DSpark draft block size"));
        }
        let mut next = states.to_vec();
        let mut inputs = Vec::with_capacity(next.len());
        for state in &next {
            let input = hc_pre(
                &state.residual,
                &state.next_pre,
                self.block.attention_hc.hidden,
                true,
            )?;
            inputs.push(self.block.attention_norm.forward(&input, true)?);
        }
        let old_cache = self.attention.attention.cache.clone();
        let result = (|| {
            let outputs = self.attention.forward_draft(main, &inputs)?;
            if outputs.len() != next.len() {
                return Err(error("Invalid DSpark attention output"));
            }
            for (state, output) in next.iter_mut().zip(outputs) {
                state.step(&self.block.attention_hc, true, |_| Ok(output))?;
                state.step(&self.block.ffn_hc, true, |input| {
                    let input = self.block.ffn_norm.forward(input, true)?;
                    crate::v41_moe::forward_single_quantized(
                        &self.moe.gate,
                        &self.moe.experts,
                        &self.moe.shared,
                        &input,
                        false,
                    )
                    .map(|(output, _)| output)
                })?;
            }
            Ok(())
        })();
        if let Err(err) = result {
            self.attention.attention.cache = old_cache;
            return Err(err);
        }
        states.clone_from_slice(&next);
        Ok(())
    }
}

pub struct DraftInput<'a> {
    main_projection: CheckpointMatrix<'a>,
    main_norm: RmsNorm,
    block_size: usize,
    noise: u32,
}
impl<'a> DraftInput<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        c: &TextConfig,
    ) -> Result<Self, SingularityError> {
        if c.num_nextn_predict_layers == 0
            || c.dspark_block_size == 0
            || c.dspark_target_layer_ids.is_empty()
            || c.dspark_noise_token_id as usize >= c.vocab_size
        {
            return Err(error("Invalid DSpark input configuration"));
        }
        let input = c
            .hidden_size
            .checked_mul(c.dspark_target_layer_ids.len())
            .ok_or_else(|| error("DSpark dimension overflow"))?;
        let projection =
            CheckpointMatrix::load(loader, "mtp.0.main_proj.weight", "mtp.0.main_proj.scale")?;
        if !matches!(projection, CheckpointMatrix::Dense { .. })
            || projection.shape() != (c.hidden_size, input)
        {
            return Err(error("Invalid DSpark main projection"));
        }
        Ok(Self {
            main_projection: projection,
            main_norm: RmsNorm::load(
                loader,
                "mtp.0.main_norm.weight",
                c.hidden_size,
                c.rms_norm_eps,
            )?,
            block_size: c.dspark_block_size,
            noise: c.dspark_noise_token_id,
        })
    }
    /// Returns normalized target-layer projection and anchor/noise residual states.
    pub fn prepare(
        &self,
        main_hidden: &[f32],
        anchor: u32,
        io: &TokenIo<'_>,
    ) -> Result<(Vec<f32>, Vec<HcState>), SingularityError> {
        let main = self.project_main(main_hidden)?;
        let mut states = Vec::with_capacity(self.block_size);
        for i in 0..self.block_size {
            states.push(io.embed(if i == 0 { anchor } else { self.noise } as usize)?);
        }
        Ok((main, states))
    }
    pub fn project_main(&self, main_hidden: &[f32]) -> Result<Vec<f32>, SingularityError> {
        self.main_norm
            .forward(&self.main_projection.matvec_quantized(main_hidden)?, true)
    }
}

pub struct DraftHead<'a> {
    norm: RmsNorm,
    head: Bf16Matrix<'a>,
    markov_embedding: Bf16Matrix<'a>,
    markov_head: Bf16Matrix<'a>,
    confidence: Bf16Matrix<'a>,
    hidden: usize,
    streams: usize,
    block_size: usize,
}
pub struct DraftOutput {
    pub token_ids: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
    pub confidence: Vec<f32>,
}

/// DSpark draft-only CPU runner. Target verification and cache reconciliation
/// must be supplied by a caller; this API never commits drafts to the backbone.
pub struct DraftModel<'a> {
    input: DraftInput<'a>,
    stages: Vec<DraftStage<'a>>,
    head: DraftHead<'a>,
}

pub struct VerifiedGreedyOutput {
    pub tokens: Vec<u32>,
    pub proposed: usize,
    pub accepted: usize,
}

fn greedy_token(logits: &[f32]) -> Result<u32, SingularityError> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(error("Invalid target logits"));
    }
    let id = logits
        .iter()
        .enumerate()
        .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
        .unwrap()
        .0;
    u32::try_from(id).map_err(|_| error("Target token index overflow"))
}

/// Exact greedy verification, deliberately sequential on the target reference.
/// This provides correctness rather than speculative speedup: every accepted
/// draft token is checked with the target before it is emitted.
pub fn generate_verified_greedy(
    target: &mut crate::v41_model::TextModel<'_>,
    draft: &mut DraftModel<'_>,
    io: &TokenIo<'_>,
    prompt: &[u32],
    max_new_tokens: usize,
    eos: u32,
) -> Result<VerifiedGreedyOutput, SingularityError> {
    if prompt.is_empty() || target.position() != 0 || draft.position() != 0 {
        return Err(error(
            "Verification requires fresh models and a nonempty prompt",
        ));
    }
    let mut current = None;
    for (i, &token) in prompt.iter().enumerate() {
        let output = target.forward_token_with_targets(token)?;
        if i + 1 < prompt.len() {
            draft.seed(&output.target_hidden)?;
        }
        current = Some(output);
    }
    let mut current = current.unwrap();
    let mut result = VerifiedGreedyOutput {
        tokens: Vec::with_capacity(max_new_tokens),
        proposed: 0,
        accepted: 0,
    };
    while result.tokens.len() < max_new_tokens {
        let anchor = greedy_token(&current.logits)?;
        result.tokens.push(anchor);
        if anchor == eos || result.tokens.len() == max_new_tokens {
            break;
        }
        // With a single-token prompt there is no earlier main cache to draft
        // against. Consume one ordinary target step, then speculation can start.
        if draft.position() == 0 {
            draft.seed(&current.target_hidden)?;
            current = target.forward_token_with_targets(anchor)?;
            continue;
        }
        let proposed = draft.draft_greedy(&current.target_hidden, anchor, io)?;
        if proposed.token_ids.first() != Some(&anchor) {
            return Err(error("Draft anchor mismatch"));
        }
        let mut previous = anchor;
        let mut all_accepted = true;
        for &candidate in proposed.token_ids.iter().skip(1) {
            current = target.forward_token_with_targets(previous)?;
            result.proposed += 1;
            if greedy_token(&current.logits)? != candidate {
                all_accepted = false;
                break;
            }
            result.accepted += 1;
            result.tokens.push(candidate);
            if candidate == eos || result.tokens.len() == max_new_tokens {
                return Ok(result);
            }
            draft.seed(&current.target_hidden)?;
            previous = candidate;
        }
        if all_accepted {
            current = target.forward_token_with_targets(previous)?;
        }
    }
    Ok(result)
}
impl<'a> DraftModel<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        config: &TextConfig,
    ) -> Result<Self, SingularityError> {
        let stages = (0..config.num_nextn_predict_layers)
            .map(|i| DraftStage::load(loader, config, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            input: DraftInput::load(loader, config)?,
            stages,
            head: DraftHead::load(loader, config)?,
        })
    }
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }
    pub fn position(&self) -> usize {
        self.stages.first().map_or(0, DraftStage::position)
    }
    /// Prefill a target hidden vector into every stage's main-token cache.
    pub fn seed(&mut self, main_hidden: &[f32]) -> Result<(), SingularityError> {
        let main = self.input.project_main(main_hidden)?;
        let caches: Vec<_> = self
            .stages
            .iter()
            .map(|stage| stage.attention.attention.cache.clone())
            .collect();
        for stage in &mut self.stages {
            if let Err(err) = stage.seed_main(&main) {
                for (stage, cache) in self.stages.iter_mut().zip(caches) {
                    stage.attention.attention.cache = cache;
                }
                return Err(err);
            }
        }
        Ok(())
    }
    pub fn draft_greedy(
        &mut self,
        main_hidden: &[f32],
        anchor: u32,
        io: &TokenIo<'_>,
    ) -> Result<DraftOutput, SingularityError> {
        if self.position() == 0 {
            return Err(error("DSpark must be seeded before drafting"));
        }
        let (main, mut states) = self.input.prepare(main_hidden, anchor, io)?;
        let caches: Vec<_> = self
            .stages
            .iter()
            .map(|stage| stage.attention.attention.cache.clone())
            .collect();
        let result = (|| {
            for stage in &mut self.stages {
                stage.forward(&main, &mut states)?;
            }
            self.head.greedy(&states, anchor)
        })();
        if result.is_err() {
            for (stage, cache) in self.stages.iter_mut().zip(caches) {
                stage.attention.attention.cache = cache;
            }
        }
        result
    }
}
impl<'a> DraftHead<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        c: &TextConfig,
    ) -> Result<Self, SingularityError> {
        if c.num_nextn_predict_layers == 0
            || c.dspark_block_size == 0
            || c.dspark_markov_rank == 0
            || c.hc_mult == 0
        {
            return Err(error("Invalid DSpark head configuration"));
        }
        let prefix = format!("mtp.{}", c.num_nextn_predict_layers - 1);
        let confidence_dim = c
            .hidden_size
            .checked_add(c.dspark_markov_rank)
            .ok_or_else(|| error("DSpark dimension overflow"))?;
        Ok(Self {
            norm: RmsNorm::load(
                loader,
                &format!("{prefix}.norm.weight"),
                c.hidden_size,
                c.rms_norm_eps,
            )?,
            head: Bf16Matrix::load(loader, "head.weight", c.vocab_size, c.hidden_size)?,
            markov_embedding: Bf16Matrix::load(
                loader,
                &format!("{prefix}.markov_head.embed.weight"),
                c.vocab_size,
                c.dspark_markov_rank,
            )?,
            markov_head: Bf16Matrix::load(
                loader,
                &format!("{prefix}.markov_head.head.weight"),
                c.vocab_size,
                c.dspark_markov_rank,
            )?,
            confidence: Bf16Matrix::load(
                loader,
                &format!("{prefix}.confidence_head.proj.weight"),
                1,
                confidence_dim,
            )?,
            hidden: c.hidden_size,
            streams: c.hc_mult,
            block_size: c.dspark_block_size,
        })
    }
    /// Execute the last-stage head greedily. Each Markov embedding uses the
    /// previous output token, beginning with the supplied anchor.
    pub fn greedy(&self, states: &[HcState], anchor: u32) -> Result<DraftOutput, SingularityError> {
        if states.len() != self.block_size {
            return Err(error("Invalid DSpark block size"));
        }
        let mut tokens = vec![anchor];
        let mut logits = Vec::with_capacity(states.len());
        let mut confidence = Vec::with_capacity(states.len());
        for state in states {
            if state.next_pre.len() != self.streams
                || state.residual.len()
                    != self
                        .streams
                        .checked_mul(self.hidden)
                        .ok_or_else(|| error("DSpark dimension overflow"))?
            {
                return Err(error("Invalid DSpark residual shape"));
            }
            let hidden = hc_pre(&state.residual, &state.next_pre, self.hidden, true)?;
            let normalized = self.norm.forward(&hidden, true)?;
            let mut scores = self.head.project(&normalized, false)?;
            let markov = self
                .markov_embedding
                .row(*tokens.last().unwrap() as usize)?;
            for (score, bias) in scores
                .iter_mut()
                .zip(self.markov_head.project(&markov, false)?)
            {
                *score += bias;
            }
            if scores.iter().any(|s| !s.is_finite()) {
                return Err(error("Non-finite DSpark logits"));
            }
            let chosen = scores
                .iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                .ok_or_else(|| error("Empty DSpark logits"))?
                .0;
            tokens.push(u32::try_from(chosen).map_err(|_| error("DSpark token index overflow"))?);
            let mut confidence_input = hidden;
            confidence_input.extend(markov);
            confidence.push(self.confidence.project(&confidence_input, false)?[0]);
            logits.push(scores);
        }
        Ok(DraftOutput {
            token_ids: tokens,
            logits,
            confidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    #[test]
    fn draft_head_matches_pytorch_and_uses_previous_sample() {
        #[derive(serde::Deserialize)]
        struct Expected {
            tokens: Vec<u32>,
            logits: Vec<Vec<f32>>,
            confidence: Vec<f32>,
        }
        let expected: Expected = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_dspark_head_vectors.json"
        ))
        .unwrap();
        let mut config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut config.text_config;
        c.hidden_size = 2;
        c.vocab_size = 3;
        c.hc_mult = 2;
        c.num_nextn_predict_layers = 1;
        c.dspark_markov_rank = 2;
        c.dspark_block_size = 2;
        c.rms_norm_eps = 1e-6;
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape, values) in [
            ("mtp.0.norm.weight", vec![2], vec![1., 1.]),
            ("head.weight", vec![3, 2], vec![1., 0., 0., 1., -1., -1.]),
            (
                "mtp.0.markov_head.embed.weight",
                vec![3, 2],
                vec![1., 0., 0., 1., 1., 1.],
            ),
            (
                "mtp.0.markov_head.head.weight",
                vec![3, 2],
                vec![0., 1., 1., 0., -1., -1.],
            ),
            (
                "mtp.0.confidence_head.proj.weight",
                vec![1, 4],
                vec![1., 2., 3., 4.],
            ),
        ] {
            let start = payload.len();
            for v in values {
                payload.extend(bf16::from_f32(v).to_le_bytes());
            }
            header.insert(name.into(),serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,payload.len()]}));
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(h);
        file.extend(payload);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, file).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let head = DraftHead::load(&loader, c).unwrap();
        let states = vec![
            HcState::from_embedding(&[1., 2.], 2).unwrap(),
            HcState::from_embedding(&[-1., 3.], 2).unwrap(),
        ];
        let result = head.greedy(&states, 0).unwrap();
        assert_eq!(result.token_ids, expected.tokens);
        assert_eq!(result.logits, expected.logits);
        assert_eq!(result.confidence, expected.confidence);
        assert!(head.greedy(&states[..1], 0).is_err());
        assert!(head.greedy(&states, 3).is_err());
    }

    #[test]
    fn draft_input_projects_target_and_places_anchor_and_noise() {
        let mut config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut config.text_config;
        c.hidden_size = 32;
        c.vocab_size = 2;
        c.hc_mult = 2;
        c.num_nextn_predict_layers = 1;
        c.dspark_block_size = 3;
        c.dspark_noise_token_id = 1;
        c.dspark_target_layer_ids = vec![0];
        c.rms_norm_eps = 1e-6;
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        let mut add = |name: &str, dtype: &str, shape: Vec<usize>, data: Vec<u8>| {
            let start = payload.len();
            payload.extend(data);
            header.insert(name.to_owned(),serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}));
        };
        add(
            "mtp.0.main_proj.weight",
            "F8_E4M3",
            vec![32, 32],
            vec![0x20; 1024],
        );
        add("mtp.0.main_proj.scale", "F8_E8M0", vec![1, 1], vec![127]);
        add(
            "mtp.0.main_norm.weight",
            "BF16",
            vec![32],
            vec![0x80, 0x3f].repeat(32),
        );
        add("norm.weight", "BF16", vec![32], vec![0x80, 0x3f].repeat(32));
        let mut embedding = vec![0x80, 0x3f].repeat(32);
        embedding.extend(vec![0x00, 0x40].repeat(32));
        add("embed.weight", "BF16", vec![2, 32], embedding);
        add("head.weight", "BF16", vec![2, 32], vec![0; 128]);
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let io = TokenIo::load(&loader, c).unwrap();
        let input = DraftInput::load(&loader, c).unwrap();
        let (main, states) = input.prepare(&[1.; 32], 0, &io).unwrap();
        assert_eq!(main, vec![1.; 32]);
        assert_eq!(states.len(), 3);
        assert_eq!(states[0].residual, vec![1.; 64]);
        assert_eq!(states[1].residual, vec![2.; 64]);
        assert_eq!(states[2].residual, vec![2.; 64]);
        assert!(input.prepare(&[1.; 31], 0, &io).is_err());
    }
}
