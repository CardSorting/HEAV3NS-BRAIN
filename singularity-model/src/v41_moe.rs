//! V4.1 CPU reference for expert routing and SwiGLU execution.
//! Derived from the MIT-licensed checkpoint inference/model.py, revision
//! 1bae0e1088993b9b82de08f59de5be5a34e42170. The matrix implementation
//! does not yet emulate activation quantization or run on GPUs.
use crate::v41_quant::CheckpointMatrix;
use half::bf16;
use singularity_core::SingularityError;

fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

#[derive(Debug, Clone)]
pub struct Route {
    pub expert: usize,
    pub weight: f32,
}

/// Select by biased sqrt-softplus scores but weight with unbiased scores.
/// Ties are ordered by ascending expert ID for determinism; PyTorch topk does
/// not guarantee identical tie ordering across backends.
pub fn route_logits(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    normalize: bool,
    scale: f32,
) -> Result<Vec<Route>, SingularityError> {
    if logits.is_empty()
        || logits.len() != bias.len()
        || top_k == 0
        || top_k > logits.len()
        || !scale.is_finite()
        || logits.iter().chain(bias).any(|x| !x.is_finite())
    {
        return Err(error("Invalid V4.1 routing input"));
    }
    let scores: Vec<f32> = logits
        .iter()
        .map(|&x| {
            let softplus = if x > 20.0 { x } else { x.exp().ln_1p() };
            softplus.sqrt()
        })
        .collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        (scores[b] + bias[b])
            .total_cmp(&(scores[a] + bias[a]))
            .then(a.cmp(&b))
    });
    order.truncate(top_k);
    let sum: f32 = order.iter().map(|&i| scores[i]).sum();
    Ok(order
        .into_iter()
        .map(|i| Route {
            expert: i,
            weight: if normalize && top_k > 1 {
                scores[i] / (sum + 1e-20) * scale
            } else {
                scores[i] * scale
            },
        })
        .collect())
}

pub struct Gate {
    pub hidden: usize,
    pub experts: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub image_bias: Option<Vec<f32>>,
    pub top_k: usize,
    pub normalize: bool,
    pub route_scale: f32,
}

impl Gate {
    /// Load checkpoint BF16 gate matrix and FP32 selection biases.
    pub fn load(
        loader: &crate::SafetensorsMmapLoader,
        prefix: &str,
        config: &crate::v41_config::TextConfig,
    ) -> Result<Self, SingularityError> {
        if config.scoring_func != "sqrtsoftplus" || config.topk_method != "noaux_tc" {
            return Err(error("Unsupported V4.1 gate scoring configuration"));
        }
        let n = config.n_routed_experts;
        let d = config.hidden_size;
        let load =
            |suffix: &str, dtype: &str, shape: &[usize]| -> Result<Vec<f32>, SingularityError> {
                let name = format!("{prefix}.{suffix}");
                let info = loader
                    .get_tensor_info(&name)
                    .ok_or_else(|| error(&format!("Missing gate tensor {name}")))?;
                if info.dtype != dtype || info.shape != shape {
                    return Err(error(&format!("Invalid gate tensor {name}")));
                }
                let vals = loader.get_tensor_f32(&name)?;
                let count = shape
                    .iter()
                    .try_fold(1usize, |a, &b| a.checked_mul(b))
                    .ok_or_else(|| error("Gate size overflow"))?;
                if vals.len() != count {
                    return Err(error("Gate payload size mismatch"));
                }
                Ok(vals)
            };
        let weights = load("weight", "BF16", &[n, d])?;
        let bias = load("bias", "F32", &[n])?;
        let image_bias = if loader
            .get_tensor_info(&format!("{prefix}.bias_vl"))
            .is_some()
        {
            Some(load("bias_vl", "F32", &[n])?)
        } else {
            None
        };
        Ok(Self {
            hidden: d,
            experts: n,
            weights,
            bias,
            image_bias,
            top_k: config.num_experts_per_tok,
            normalize: config.norm_topk_prob,
            route_scale: config.routed_scaling_factor,
        })
    }
    pub fn route(&self, input: &[f32], image_token: bool) -> Result<Vec<Route>, SingularityError> {
        if self.hidden == 0
            || self.experts == 0
            || self.hidden.checked_mul(self.experts) != Some(self.weights.len())
            || self.bias.len() != self.experts
            || input.len() != self.hidden
        {
            return Err(error("Invalid gate dimensions"));
        }
        let bias = if image_token {
            self.image_bias
                .as_ref()
                .ok_or_else(|| error("Image gate bias unavailable"))?
        } else {
            &self.bias
        };
        let logits: Vec<f32> = self
            .weights
            .chunks_exact(self.hidden)
            .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
            .collect();
        route_logits(&logits, bias, self.top_k, self.normalize, self.route_scale)
    }
}

/// One quantized checkpoint expert. `w1` is gate, `w3` is up, `w2` is down.
pub struct Expert<'a> {
    pub w1: CheckpointMatrix<'a>,
    pub w2: CheckpointMatrix<'a>,
    pub w3: CheckpointMatrix<'a>,
    pub swiglu_limit: f32,
}
impl<'a> Expert<'a> {
    pub fn load(
        loader: &'a crate::SafetensorsMmapLoader,
        prefix: &str,
        hidden: usize,
        intermediate: usize,
        swiglu_limit: f32,
        routed: bool,
    ) -> Result<Self, SingularityError> {
        let load = |name: &str| {
            CheckpointMatrix::load(
                loader,
                &format!("{prefix}.{name}.weight"),
                &format!("{prefix}.{name}.scale"),
            )
        };
        let w1 = load("w1")?;
        let w2 = load("w2")?;
        let w3 = load("w3")?;
        if w1.shape() != (intermediate, hidden)
            || w3.shape() != (intermediate, hidden)
            || w2.shape() != (hidden, intermediate)
            || [&w1, &w2, &w3].iter().any(|w| {
                !matches!(
                    (routed, w),
                    (true, CheckpointMatrix::Expert { .. })
                        | (false, CheckpointMatrix::Dense { .. })
                )
            })
        {
            return Err(error(
                "Invalid checkpoint expert matrix shapes or precision",
            ));
        }
        Ok(Self {
            w1,
            w2,
            w3,
            swiglu_limit,
        })
    }
    /// CPU reference using unquantized activations, with BF16 boundaries as in
    /// the reference model. Routing weight is applied before the down projection.
    pub fn forward(
        &self,
        input: &[f32],
        routing_weight: Option<f32>,
    ) -> Result<Vec<f32>, SingularityError> {
        self.forward_impl(input, routing_weight, false)
    }

    /// Emulate the reference kernel's block-FP8 activation path on CPU.
    pub fn forward_quantized(
        &self,
        input: &[f32],
        routing_weight: Option<f32>,
    ) -> Result<Vec<f32>, SingularityError> {
        self.forward_impl(input, routing_weight, true)
    }

    fn forward_impl(
        &self,
        input: &[f32],
        routing_weight: Option<f32>,
        quantized: bool,
    ) -> Result<Vec<f32>, SingularityError> {
        if !self.swiglu_limit.is_finite()
            || self.swiglu_limit < 0.0
            || routing_weight.is_some_and(|w| !w.is_finite())
        {
            return Err(error("Invalid expert parameters"));
        }
        let matvec = |w: &CheckpointMatrix<'_>, x: &[f32]| {
            if quantized {
                w.matvec_quantized(x)
            } else {
                w.matvec(x)
            }
        };
        let gate = matvec(&self.w1, input)?;
        let up = matvec(&self.w3, input)?;
        if gate.len() != up.len() {
            return Err(error("Expert intermediate shape mismatch"));
        }
        let hidden: Vec<f32> = gate
            .iter()
            .zip(up)
            .map(|(&g, u)| {
                let mut g = bf16::from_f32(g).to_f32();
                let mut u = bf16::from_f32(u).to_f32();
                if self.swiglu_limit > 0.0 {
                    g = g.min(self.swiglu_limit);
                    u = u.clamp(-self.swiglu_limit, self.swiglu_limit);
                }
                let value = (g / (1.0 + (-g).exp())) * u * routing_weight.unwrap_or(1.0);
                bf16::from_f32(value).to_f32()
            })
            .collect();
        let result = matvec(&self.w2, &hidden)?;
        Ok(result
            .into_iter()
            .map(|x| bf16::from_f32(x).to_f32())
            .collect())
    }
}

/// Single-token local reference. All selected experts must be present; no
/// distributed dispatch or silent fallback is performed.
pub fn forward_single(
    gate: &Gate,
    experts: &[Option<Expert<'_>>],
    shared: &Expert<'_>,
    input: &[f32],
    image_token: bool,
) -> Result<(Vec<f32>, Vec<Route>), SingularityError> {
    forward_single_impl(gate, experts, shared, input, image_token, false)
}

/// Local CPU reference using the checkpoint's FP8 activation blocks.
pub fn forward_single_quantized(
    gate: &Gate,
    experts: &[Option<Expert<'_>>],
    shared: &Expert<'_>,
    input: &[f32],
    image_token: bool,
) -> Result<(Vec<f32>, Vec<Route>), SingularityError> {
    forward_single_impl(gate, experts, shared, input, image_token, true)
}

fn forward_single_impl(
    gate: &Gate,
    experts: &[Option<Expert<'_>>],
    shared: &Expert<'_>,
    input: &[f32],
    image_token: bool,
    quantized: bool,
) -> Result<(Vec<f32>, Vec<Route>), SingularityError> {
    if experts.len() != gate.experts {
        return Err(error("Expert inventory does not match gate"));
    }
    let routes = gate.route(input, image_token)?;
    let mut output = vec![0.0; gate.hidden];
    for route in &routes {
        let expert = experts[route.expert]
            .as_ref()
            .ok_or_else(|| error("Selected expert is not locally available"))?;
        let value = if quantized {
            expert.forward_quantized(input, Some(route.weight))?
        } else {
            expert.forward(input, Some(route.weight))?
        };
        if value.len() != output.len() {
            return Err(error("Expert output shape mismatch"));
        }
        for (sum, v) in output.iter_mut().zip(value) {
            *sum += v;
        }
    }
    let value = if quantized {
        shared.forward_quantized(input, None)?
    } else {
        shared.forward(input, None)?
    };
    if value.len() != output.len() {
        return Err(error("Shared expert output shape mismatch"));
    }
    for (sum, v) in output.iter_mut().zip(value) {
        *sum = bf16::from_f32(*sum + v).to_f32();
    }
    Ok((output, routes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    #[derive(Deserialize)]
    struct RouteExpected {
        ids: Vec<usize>,
        weights: Vec<f32>,
    }
    #[derive(Deserialize)]
    struct Vectors {
        weight_sha256: String,
        bias_sha256: String,
        text: RouteExpected,
        image: RouteExpected,
        expert: Vec<f32>,
        combined: Vec<f32>,
    }
    #[derive(Deserialize)]
    struct RealExpertVectors {
        weights_sha256: String,
        scales_sha256: String,
        route_weight: f32,
        output: Vec<f32>,
        quantized_output: Vec<f32>,
    }
    fn vectors() -> Vectors {
        serde_json::from_str(include_str!("../tests/fixtures/v41_moe_vectors.json")).unwrap()
    }
    #[test]
    fn selection_bias_does_not_change_routing_weight() {
        let routes = route_logits(&[4., 1., 0.], &[0., 5., 0.], 2, true, 1.5).unwrap();
        assert_eq!(routes.iter().map(|r| r.expert).collect::<Vec<_>>(), [1, 0]);
        let a = (1.0f32.exp().ln_1p()).sqrt();
        let b = (4.0f32.exp().ln_1p()).sqrt();
        assert!((routes[0].weight - a / (a + b) * 1.5).abs() < 1e-6);
        assert!(route_logits(&[f32::NAN], &[0.], 1, true, 1.).is_err());
    }
    #[test]
    fn packed_expert_matches_torch_reference() {
        let packed = |seed| {
            (0..512)
                .map(|i| ((i * 19 + seed) % 256) as u8)
                .collect::<Vec<_>>()
        };
        let (w1, w2, w3) = (packed(7), packed(13), packed(23));
        let scales = vec![125u8; 32];
        let matrix = |weights| CheckpointMatrix::Expert {
            weights,
            scales: &scales,
            rows: 32,
            cols: 32,
        };
        let expert = Expert {
            w1: matrix(&w1),
            w2: matrix(&w2),
            w3: matrix(&w3),
            swiglu_limit: 10.0,
        };
        let input: Vec<f32> = (0..32).map(|i| (i % 9 - 4) as f32 / 8.0).collect();
        let result = expert.forward(&input, Some(0.7)).unwrap();
        for (a, b) in result.iter().zip(vectors().expert) {
            assert!((a - b).abs() <= 0.008 * b.abs().max(1.0), "{a} != {b}");
        }
        assert!(expert.forward(&[], None).is_err());
        let create_expert = || Expert {
            w1: matrix(&w1),
            w2: matrix(&w2),
            w3: matrix(&w3),
            swiglu_limit: 10.0,
        };
        let zeros = vec![0u8; 32 * 32];
        let dense_scales = [127u8];
        let dense = || CheckpointMatrix::Dense {
            weights: &zeros,
            scales: &dense_scales,
            rows: 32,
            cols: 32,
        };
        let shared = Expert {
            w1: dense(),
            w2: dense(),
            w3: dense(),
            swiglu_limit: 10.0,
        };
        let gate = Gate {
            hidden: 32,
            experts: 2,
            weights: vec![0.; 64],
            bias: vec![0.1, 0.2],
            image_bias: None,
            top_k: 2,
            normalize: true,
            route_scale: 1.5,
        };
        let experts = [Some(create_expert()), Some(create_expert())];
        let (combined, routes) = forward_single(&gate, &experts, &shared, &input, false).unwrap();
        assert_eq!(routes.iter().map(|r| r.expert).collect::<Vec<_>>(), [1, 0]);
        for (a, b) in combined.iter().zip(vectors().combined) {
            assert!((a - b).abs() <= 0.008 * b.abs().max(1.0));
        }
        let incomplete = [Some(create_expert()), None];
        assert!(forward_single(&gate, &incomplete, &shared, &input, false).is_err());
    }
    #[test]
    #[ignore = "requires pinned layer-6 gate byte ranges"]
    fn checkpoint_gate_matches_torch_reference() {
        let v = vectors();
        let raw = std::fs::read(std::env::var("SINGULARITY_V41_GATE_WEIGHT").unwrap()).unwrap();
        let biases = std::fs::read(std::env::var("SINGULARITY_V41_GATE_BIAS").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), v.weight_sha256);
        assert_eq!(format!("{:x}", Sha256::digest(&biases)), v.bias_sha256);
        let weights = raw
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect();
        let b: Vec<f32> = biases
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let gate = Gate {
            hidden: 5120,
            experts: 384,
            weights,
            bias: b[..384].to_vec(),
            image_bias: Some(b[384..].to_vec()),
            top_k: 6,
            normalize: true,
            route_scale: 1.5,
        };
        let input: Vec<f32> = (0..5120).map(|i| (i % 97 - 48) as f32 / 64.0).collect();
        for (image, expected) in [(false, v.text), (true, v.image)] {
            let actual = gate.route(&input, image).unwrap();
            assert_eq!(
                actual.iter().map(|x| x.expert).collect::<Vec<_>>(),
                expected.ids
            );
            for (route, weight) in actual.iter().zip(expected.weights) {
                assert!((route.weight - weight).abs() < 1e-5);
            }
        }
    }

    #[test]
    #[ignore = "requires pinned layer-6 gate and expert-309 byte ranges"]
    fn routed_real_expert_matches_torch_reference() {
        let expected: RealExpertVectors = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_real_expert_vectors.json"
        ))
        .unwrap();
        let gate_vectors = vectors();
        let gate_raw =
            std::fs::read(std::env::var("SINGULARITY_V41_GATE_WEIGHT").unwrap()).unwrap();
        let gate_bias = std::fs::read(std::env::var("SINGULARITY_V41_GATE_BIAS").unwrap()).unwrap();
        let weight_raw =
            std::fs::read(std::env::var("SINGULARITY_V41_EXPERT_WEIGHT").unwrap()).unwrap();
        let scale_raw =
            std::fs::read(std::env::var("SINGULARITY_V41_EXPERT_SCALE").unwrap()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&gate_raw)),
            gate_vectors.weight_sha256
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&gate_bias)),
            gate_vectors.bias_sha256
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&weight_raw)),
            expected.weights_sha256
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&scale_raw)),
            expected.scales_sha256
        );
        use std::io::Write;
        let n = 5898240;
        let s = 368640;
        assert_eq!(weight_raw.len(), 3 * n);
        assert_eq!(scale_raw.len(), 3 * s);
        let tensors: Vec<(&str, &str, Vec<usize>, &[u8])> = vec![
            ("layer.gate.weight", "BF16", vec![384, 5120], &gate_raw),
            ("layer.gate.bias", "F32", vec![384], &gate_bias[..1536]),
            ("layer.gate.bias_vl", "F32", vec![384], &gate_bias[1536..]),
            (
                "layer.expert.w1.weight",
                "I8",
                vec![2304, 2560],
                &weight_raw[..n],
            ),
            (
                "layer.expert.w2.weight",
                "I8",
                vec![5120, 1152],
                &weight_raw[n..2 * n],
            ),
            (
                "layer.expert.w3.weight",
                "I8",
                vec![2304, 2560],
                &weight_raw[2 * n..],
            ),
            (
                "layer.expert.w1.scale",
                "F8_E8M0",
                vec![2304, 160],
                &scale_raw[..s],
            ),
            (
                "layer.expert.w2.scale",
                "F8_E8M0",
                vec![5120, 72],
                &scale_raw[s..2 * s],
            ),
            (
                "layer.expert.w3.scale",
                "F8_E8M0",
                vec![2304, 160],
                &scale_raw[2 * s..],
            ),
        ];
        let mut offset = 0;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, data) in &tensors {
            let end = offset + data.len();
            header.insert(
                (*name).into(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets":[offset,end]}),
            );
            offset = end;
        }
        let header = serde_json::to_vec(&header).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for (_, _, _, data) in &tensors {
            file.write_all(data).unwrap();
        }
        drop(file);
        let loader = crate::SafetensorsMmapLoader::open(path).unwrap();
        let config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let gate = Gate::load(&loader, "layer.gate", &config.text_config).unwrap();
        let input: Vec<f32> = (0..5120).map(|i| (i % 97 - 48) as f32 / 64.0).collect();
        let route = gate
            .route(&input, false)
            .unwrap()
            .into_iter()
            .find(|r| r.expert == 309)
            .unwrap();
        assert!((route.weight - expected.route_weight).abs() < 1e-5);
        let expert = Expert::load(
            &loader,
            "layer.expert",
            config.text_config.hidden_size,
            config.text_config.moe_intermediate_size,
            config.text_config.swiglu_limit,
            true,
        )
        .unwrap();
        let output = expert.forward(&input, Some(route.weight)).unwrap();
        let (mut squared_error, mut squared_reference) = (0.0f64, 0.0f64);
        for (actual, reference) in output.iter().zip(&expected.output) {
            squared_error += ((*actual as f64) - (*reference as f64)).powi(2);
            squared_reference += (*reference as f64).powi(2);
        }
        assert_eq!(output.len(), expected.output.len());
        let relative_rms = (squared_error / squared_reference).sqrt();
        eprintln!("Expert 309 relative RMS error: {relative_rms:.8}");
        assert!(relative_rms < 0.01, "relative RMS error {relative_rms}");
        let quantized = expert
            .forward_quantized(&input, Some(route.weight))
            .unwrap();
        let (mut squared_error, mut squared_reference) = (0.0f64, 0.0f64);
        for (actual, reference) in quantized.iter().zip(&expected.quantized_output) {
            squared_error += ((*actual as f64) - (*reference as f64)).powi(2);
            squared_reference += (*reference as f64).powi(2);
        }
        assert_eq!(quantized.len(), expected.quantized_output.len());
        let relative_rms = (squared_error / squared_reference).sqrt();
        eprintln!("Quantized expert 309 relative RMS error: {relative_rms:.8}");
        assert!(
            relative_rms < 0.01,
            "quantized relative RMS error {relative_rms}"
        );
    }
}
