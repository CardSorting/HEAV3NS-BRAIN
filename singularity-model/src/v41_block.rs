//! Single-token V4.1 block orchestration, following the pinned checkpoint reference.
//! Attention and Engram execution remain caller-provided; this is not a full model.
use crate::v41_moe::{self, Expert, Gate};
use crate::{
    safetensors::SafetensorsMmapLoader,
    v41_config::TextConfig,
    v41_hyper::{HcState, HcWeights},
};
use half::bf16;
use singularity_core::SingularityError;

fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub epsilon: f32,
}

impl RmsNorm {
    pub fn load(
        loader: &SafetensorsMmapLoader,
        name: &str,
        hidden: usize,
        epsilon: f32,
    ) -> Result<Self, SingularityError> {
        let info = loader
            .get_tensor_info(name)
            .ok_or_else(|| error(&format!("Missing norm tensor {name}")))?;
        if info.dtype != "BF16"
            || info.shape != [hidden]
            || hidden == 0
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(error(&format!("Invalid norm tensor {name}")));
        }
        Ok(Self {
            weight: loader.get_tensor_f32(name)?,
            epsilon,
        })
    }

    pub fn forward(&self, input: &[f32], round_bf16: bool) -> Result<Vec<f32>, SingularityError> {
        if input.is_empty()
            || input.len() != self.weight.len()
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err(error("Invalid RMSNorm dimensions or epsilon"));
        }
        let inv = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32 + self.epsilon)
            .sqrt()
            .recip();
        Ok(input
            .iter()
            .zip(&self.weight)
            .map(|(x, w)| {
                let value = x * inv * w;
                if round_bf16 {
                    bf16::from_f32(value).to_f32()
                } else {
                    value
                }
            })
            .collect())
    }
}

pub struct Block {
    pub attention_hc: HcWeights,
    pub ffn_hc: HcWeights,
    pub attention_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
}

impl Block {
    pub fn load(
        loader: &SafetensorsMmapLoader,
        layer: usize,
        config: &TextConfig,
    ) -> Result<Self, SingularityError> {
        if layer >= config.num_hidden_layers {
            return Err(error("Block layer out of range"));
        }
        let prefix = format!("layers.{layer}");
        Self::load_prefix(loader, &prefix, config)
    }

    pub fn load_draft(
        loader: &SafetensorsMmapLoader,
        stage: usize,
        config: &TextConfig,
    ) -> Result<Self, SingularityError> {
        if stage >= config.num_nextn_predict_layers {
            return Err(error("Draft stage out of range"));
        }
        Self::load_prefix(loader, &format!("mtp.{stage}"), config)
    }

    fn load_prefix(
        loader: &SafetensorsMmapLoader,
        prefix: &str,
        config: &TextConfig,
    ) -> Result<Self, SingularityError> {
        Ok(Self {
            attention_hc: HcWeights::load_for_config(loader, &format!("{prefix}.hc_attn"), config)?,
            ffn_hc: HcWeights::load_for_config(loader, &format!("{prefix}.hc_ffn"), config)?,
            attention_norm: RmsNorm::load(
                loader,
                &format!("{prefix}.attn_norm.weight"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            ffn_norm: RmsNorm::load(
                loader,
                &format!("{prefix}.ffn_norm.weight"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
        })
    }

    /// State commits only after both sublayers succeed. Callback-owned caches must
    /// separately provide their own rollback if a later operation fails.
    pub fn forward<A, F>(
        &self,
        state: &mut HcState,
        round_bf16: bool,
        attention: A,
        ffn: F,
    ) -> Result<(), SingularityError>
    where
        A: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
        F: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
    {
        let mut next = HcState {
            residual: state.residual.clone(),
            next_pre: state.next_pre.clone(),
        };
        next.step(&self.attention_hc, round_bf16, |input| {
            attention(&self.attention_norm.forward(input, round_bf16)?)
        })?;
        next.step(&self.ffn_hc, round_bf16, |input| {
            ffn(&self.ffn_norm.forward(input, round_bf16)?)
        })?;
        *state = next;
        Ok(())
    }

    /// Execute the checkpoint MoE reference after the caller's attention implementation.
    pub fn forward_with_moe<A>(
        &self,
        state: &mut HcState,
        attention: A,
        gate: &Gate,
        experts: &[Option<Expert<'_>>],
        shared: &Expert<'_>,
        image_token: bool,
    ) -> Result<(), SingularityError>
    where
        A: FnOnce(&[f32]) -> Result<Vec<f32>, SingularityError>,
    {
        self.forward(state, true, attention, |input| {
            v41_moe::forward_single_quantized(gate, experts, shared, input, image_token)
                .map(|(output, _)| output)
        })
    }

    /// Run an uncompressed checkpoint-backed block atomically with respect to
    /// both HC state and its sliding-window cache.
    pub fn forward_window(
        &self,
        state: &mut HcState,
        attention: &mut crate::v41_attention::LoadedWindowAttention<'_>,
        moe: &crate::v41_moe::LoadedMoe<'_>,
        image_token: bool,
    ) -> Result<(), SingularityError> {
        let saved_cache = attention.attention.cache.clone();
        let result = self.forward_with_moe(
            state,
            |input| attention.forward(input),
            &moe.gate,
            &moe.experts,
            &moe.shared,
            image_token,
        );
        if result.is_err() {
            attention.attention.cache = saved_cache;
        }
        result
    }

    pub fn forward_compressed(
        &self,
        state: &mut HcState,
        attention: &mut crate::v41_compressed_attention::CompressedAttention<'_>,
        shared: &mut crate::v41_compressed_attention::SharedAttention,
        moe: &crate::v41_moe::LoadedMoe<'_>,
        image_token: bool,
    ) -> Result<(), SingularityError> {
        let checkpoint = attention.checkpoint();
        let saved_shared = shared.clone();
        let result = self.forward_with_moe(
            state,
            |input| attention.forward(input, shared),
            &moe.gate,
            &moe.experts,
            &moe.shared,
            image_token,
        );
        if result.is_err() {
            attention.restore(checkpoint);
            *shared = saved_shared;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn weights() -> HcWeights {
        HcWeights {
            streams: 2,
            hidden: 2,
            projection: vec![0.0; 32],
            scale: [0.0; 3],
            base: vec![0.0; 8],
            norm_epsilon: 1e-6,
            sinkhorn_epsilon: 1e-6,
            sinkhorn_iterations: 10,
        }
    }
    fn block() -> Block {
        Block {
            attention_hc: weights(),
            ffn_hc: weights(),
            attention_norm: RmsNorm {
                weight: vec![1.0, 2.0],
                epsilon: 1e-6,
            },
            ffn_norm: RmsNorm {
                weight: vec![1.0, 1.0],
                epsilon: 1e-6,
            },
        }
    }
    #[test]
    fn block_applies_norm_and_shifted_transitions_and_rolls_back_on_error() {
        let block = block();
        let mut state = HcState::from_embedding(&[3.0, 4.0], 2).unwrap();
        let original = state.residual.clone();
        let result = block.forward(
            &mut state,
            false,
            |x| {
                assert!((x[0] - 0.848528).abs() < 1e-5);
                assert!((x[1] - 2.262742).abs() < 1e-5);
                Ok(vec![1.0, 2.0])
            },
            |_| Err(error("FFN failed")),
        );
        assert!(result.is_err());
        assert_eq!(state.residual, original);
        assert_eq!(state.next_pre, vec![1.0, 0.0]);
        block
            .forward(
                &mut state,
                false,
                |_| Ok(vec![1.0, 2.0]),
                |x| {
                    assert!(x[0] > 0.0 && x[1] > x[0]);
                    Ok(vec![0.5, 1.0])
                },
            )
            .unwrap();
        assert_ne!(state.residual, original);
        assert_eq!(state.next_pre, vec![0.500001, 0.500001]);
    }

    #[test]
    fn checkpoint_window_block_rolls_back_cache_when_moe_fails() {
        let mut config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let c = &mut config.text_config;
        c.hidden_size = 32;
        c.hc_mult = 2;
        c.num_attention_heads = 1;
        c.head_dim = 32;
        c.q_lora_rank = 32;
        c.o_lora_rank = 32;
        c.o_groups = 1;
        c.sliding_window = 2;
        c.qk_rope_head_dim = 2;
        c.n_routed_experts = 1;
        c.num_experts_per_tok = 1;
        c.moe_intermediate_size = 32;
        c.dspark_n_routed_experts = 1;
        c.dspark_num_experts_per_tok = 1;
        c.dspark_block_size = 2;
        c.num_nextn_predict_layers = 1;
        c.dspark_markov_rank = 32;
        c.dspark_target_layer_ids = vec![0];
        c.dspark_noise_token_id = 1;
        c.vocab_size = 2;
        c.num_hidden_layers = 1;
        c.engram_layer_ids.clear();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut add = |name: String, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = payload.len();
            payload.extend(bytes);
            header.insert(name, serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}));
        };
        for hc in ["hc_attn", "hc_ffn"] {
            add(
                format!("layers.0.{hc}_fn"),
                "F32",
                vec![8, 64],
                vec![0; 8 * 64 * 4],
            );
            add(format!("layers.0.{hc}_base"), "F32", vec![8], vec![0; 32]);
            add(format!("layers.0.{hc}_scale"), "F32", vec![3], vec![0; 12]);
        }
        for name in ["attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"] {
            add(
                format!("layers.0.{name}.weight"),
                "BF16",
                vec![32],
                vec![0x80, 0x3f].repeat(32),
            );
        }
        for name in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
            add(
                format!("layers.0.attn.{name}.weight"),
                "F8_E4M3",
                vec![32, 32],
                vec![0x20; 1024],
            );
            add(
                format!("layers.0.attn.{name}.scale"),
                "F8_E8M0",
                vec![1, 1],
                vec![127],
            );
        }
        add("layers.0.attn.attn_sink".into(), "F32", vec![1], vec![0; 4]);
        add(
            "layers.0.ffn.gate.weight".into(),
            "BF16",
            vec![1, 32],
            vec![0x80, 0x3f].repeat(32),
        );
        add("layers.0.ffn.gate.bias".into(), "F32", vec![1], vec![0; 4]);
        for (name, dtype, cols, data, scale_rows) in [
            ("experts.0", "I8", 16, 0x11, 32),
            ("shared_experts", "F8_E4M3", 32, 0x20, 1),
        ] {
            for matrix in ["w1", "w2", "w3"] {
                add(
                    format!("layers.0.ffn.{name}.{matrix}.weight"),
                    dtype,
                    vec![32, cols],
                    vec![data; 32 * cols],
                );
                add(
                    format!("layers.0.ffn.{name}.{matrix}.scale"),
                    "F8_E8M0",
                    vec![scale_rows, 1],
                    vec![127; scale_rows],
                );
            }
        }
        for (name, mut info) in header.clone() {
            let start = info["data_offsets"][0].as_u64().unwrap() as usize;
            let end = info["data_offsets"][1].as_u64().unwrap() as usize;
            let copy = payload[start..end].to_vec();
            let new_start = payload.len();
            payload.extend(copy);
            info["data_offsets"] = serde_json::json!([new_start, payload.len()]);
            header.insert(name.replacen("layers.0.", "mtp.0.", 1), info);
        }
        let mut add_extra = |name: &str, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = payload.len();
            payload.extend(bytes);
            header.insert(name.to_owned(), serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}));
        };
        for name in [
            "embed.weight",
            "head.weight",
            "mtp.0.markov_head.embed.weight",
            "mtp.0.markov_head.head.weight",
        ] {
            add_extra(name, "BF16", vec![2, 32], vec![0x80, 0x3f].repeat(64));
        }
        for name in ["norm.weight", "mtp.0.main_norm.weight", "mtp.0.norm.weight"] {
            add_extra(name, "BF16", vec![32], vec![0x80, 0x3f].repeat(32));
        }
        add_extra(
            "mtp.0.main_proj.weight",
            "F8_E4M3",
            vec![32, 32],
            vec![0x20; 1024],
        );
        add_extra("mtp.0.main_proj.scale", "F8_E8M0", vec![1, 1], vec![127]);
        add_extra(
            "mtp.0.confidence_head.proj.weight",
            "BF16",
            vec![1, 64],
            vec![0x80, 0x3f].repeat(64),
        );
        // Bias the draft toward token 1 while the shared target head ties and
        // greedily chooses token 0, exercising rejection and cache alignment.
        let offset = header["mtp.0.markov_head.head.weight"]["data_offsets"][0]
            .as_u64()
            .unwrap() as usize;
        for pair in payload[offset..offset + 64].chunks_exact_mut(2) {
            pair.copy_from_slice(&[0x80, 0xbf]);
        }
        for (name, info) in header.clone() {
            if !name.starts_with("mtp.0.") {
                continue;
            }
            let start = info["data_offsets"][0].as_u64().unwrap() as usize;
            let end = info["data_offsets"][1].as_u64().unwrap() as usize;
            for stage in 1..3 {
                let mut copy_info = info.clone();
                let data = payload[start..end].to_vec();
                let new_start = payload.len();
                payload.extend(data);
                copy_info["data_offsets"] = serde_json::json!([new_start, payload.len()]);
                header.insert(
                    name.replacen("mtp.0.", &format!("mtp.{stage}."), 1),
                    copy_info,
                );
            }
        }
        let offset_final = header["mtp.2.markov_head.head.weight"]["data_offsets"][0]
            .as_u64()
            .unwrap() as usize;
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, &bytes).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let block = Block::load(&loader, 0, c).unwrap();
        let mut attention =
            crate::v41_attention::LoadedWindowAttention::load(&loader, c, 0).unwrap();
        let mut moe = crate::v41_moe::LoadedMoe::load(&loader, 0, c).unwrap();
        let mut state = HcState::from_embedding(&vec![1.0; 32], 2).unwrap();
        let original = state.residual.clone();
        let expert = moe.experts[0].take();
        assert!(block
            .forward_window(&mut state, &mut attention, &moe, false)
            .is_err());
        assert_eq!(attention.attention.cache.position(), 0);
        assert_eq!(state.residual, original);
        moe.experts[0] = expert;
        block
            .forward_window(&mut state, &mut attention, &moe, false)
            .unwrap();
        assert_eq!(attention.attention.cache.position(), 1);
        assert_ne!(state.residual, original);
        let mut draft = crate::v41_dspark::DraftStage::load(&loader, c, 0).unwrap();
        draft.seed_main(&[1.0; 32]).unwrap();
        let mut draft_states = vec![
            HcState::from_embedding(&[1.0; 32], 2).unwrap(),
            HcState::from_embedding(&[2.0; 32], 2).unwrap(),
        ];
        let original = draft_states[0].residual.clone();
        assert!(draft.forward(&[1.0; 31], &mut draft_states).is_err());
        assert_eq!(draft.position(), 1);
        assert_eq!(draft_states[0].residual, original);
        draft.forward(&[1.0; 32], &mut draft_states).unwrap();
        assert_eq!(draft.position(), 2);
        assert_ne!(draft_states[0].residual, original);
        let io = crate::v41_io::TokenIo::load(&loader, c).unwrap();
        let mut model = crate::v41_dspark::DraftModel::load(&loader, c).unwrap();
        assert!(model.draft_greedy(&[1.; 32], 0, &io).is_err());
        model.seed(&[1.; 32]).unwrap();
        assert!(model.draft_greedy(&[1.; 31], 0, &io).is_err());
        assert_eq!(model.position(), 1);
        let result = model.draft_greedy(&[1.; 32], 0, &io).unwrap();
        assert_eq!(result.token_ids.len(), 3);
        assert_eq!(result.confidence.len(), 2);
        assert_eq!(model.position(), 2);
        model.reset();
        assert_eq!(model.position(), 0);
        let mut target = crate::v41_model::TextModel::load(&loader, c, None, 0).unwrap();
        let mut ordinary = crate::v41_model::TextModel::load(&loader, c, None, 0).unwrap();
        let mut draft = crate::v41_dspark::DraftModel::load(&loader, c).unwrap();
        let expected = ordinary.generate_greedy(&[0, 0], 7, 99).unwrap();
        let result = crate::v41_dspark::generate_verified_greedy(
            &mut target,
            &mut draft,
            &io,
            &[0, 0],
            7,
            99,
        )
        .unwrap();
        assert_eq!(result.tokens, expected);
        assert!(result.proposed > result.accepted);
        target.reset().unwrap();
        draft.reset();
        let single =
            crate::v41_dspark::generate_verified_greedy(&mut target, &mut draft, &io, &[0], 4, 99)
                .unwrap();
        ordinary.reset().unwrap();
        assert_eq!(
            single.tokens,
            ordinary.generate_greedy(&[0], 4, 99).unwrap()
        );
        let mut equal_bytes = bytes.clone();
        let payload_start = 8 + u64::from_le_bytes(equal_bytes[..8].try_into().unwrap()) as usize;
        for pair in
            equal_bytes[payload_start + offset..payload_start + offset + 64].chunks_exact_mut(2)
        {
            pair.copy_from_slice(&[0x80, 0x3f]);
        }
        for pair in equal_bytes[payload_start + offset_final..payload_start + offset_final + 64]
            .chunks_exact_mut(2)
        {
            pair.copy_from_slice(&[0x80, 0x3f]);
        }
        let equal_path = dir.path().join("equal.safetensors");
        std::fs::write(&equal_path, equal_bytes).unwrap();
        let equal_loader = SafetensorsMmapLoader::open(equal_path).unwrap();
        let equal_io = crate::v41_io::TokenIo::load(&equal_loader, c).unwrap();
        let mut equal_target =
            crate::v41_model::TextModel::load(&equal_loader, c, None, 0).unwrap();
        let mut equal_draft = crate::v41_dspark::DraftModel::load(&equal_loader, c).unwrap();
        let accepted = crate::v41_dspark::generate_verified_greedy(
            &mut equal_target,
            &mut equal_draft,
            &equal_io,
            &[0, 0],
            7,
            99,
        )
        .unwrap();
        assert_eq!(accepted.tokens, expected);
        assert!(accepted.accepted > 0);
        assert_eq!(accepted.accepted, accepted.proposed);
        equal_target.reset().unwrap();
        equal_draft.reset();
        let stopped = crate::v41_dspark::generate_verified_greedy(
            &mut equal_target,
            &mut equal_draft,
            &equal_io,
            &[0, 0],
            7,
            0,
        )
        .unwrap();
        assert_eq!(stopped.tokens, vec![0]);
        assert_eq!(stopped.proposed, 0);
        equal_target.reset().unwrap();
        equal_draft.reset();
        let empty = crate::v41_dspark::generate_verified_greedy(
            &mut equal_target,
            &mut equal_draft,
            &equal_io,
            &[0],
            0,
            0,
        )
        .unwrap();
        assert!(empty.tokens.is_empty());
        c.num_nextn_predict_layers = 3;
        c.compress_ratios[1..4].fill(0);
        let mut multi_target = crate::v41_model::TextModel::load(&loader, c, None, 0).unwrap();
        let mut multi_draft = crate::v41_dspark::DraftModel::load(&loader, c).unwrap();
        let multi = crate::v41_dspark::generate_verified_greedy(
            &mut multi_target,
            &mut multi_draft,
            &io,
            &[0, 0],
            7,
            99,
        )
        .unwrap();
        assert_eq!(multi.tokens, expected);
        assert!(multi.proposed > multi.accepted);
        let mut multi_target =
            crate::v41_model::TextModel::load(&equal_loader, c, None, 0).unwrap();
        let mut multi_draft = crate::v41_dspark::DraftModel::load(&equal_loader, c).unwrap();
        let multi_accepted = crate::v41_dspark::generate_verified_greedy(
            &mut multi_target,
            &mut multi_draft,
            &equal_io,
            &[0, 0],
            7,
            99,
        )
        .unwrap();
        assert_eq!(multi_accepted.tokens, expected);
        assert!(multi_accepted.accepted > 0);
        assert_eq!(multi_accepted.accepted, multi_accepted.proposed);
    }
}
