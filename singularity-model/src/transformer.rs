use crate::safetensors::SafetensorsMmapLoader;
use serde::{Deserialize, Serialize};
use singularity_core::{SamplingParams, SingularityError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub is_moe: bool,
    pub num_routed_experts: usize,
    pub num_shared_experts: usize,
    pub top_k_experts: usize,
    pub use_mla: bool,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl TransformerConfig {
    pub fn smollm2_135m() -> Self {
        Self {
            vocab_size: 49152,
            hidden_size: 576,
            intermediate_size: 1536,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 100000.0,
            max_position_embeddings: 2048,
            is_moe: false,
            num_routed_experts: 0,
            num_shared_experts: 0,
            top_k_experts: 0,
            use_mla: false,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
        }
    }

    pub fn deepseek_v2_lite() -> Self {
        Self {
            vocab_size: 102400,
            hidden_size: 2048,
            intermediate_size: 1408,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 4096,
            is_moe: true,
            num_routed_experts: 64,
            num_shared_experts: 2,
            top_k_experts: 6,
            use_mla: true,
            q_lora_rank: 0,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
            is_moe: false,
            num_routed_experts: 0,
            num_shared_experts: 0,
            top_k_experts: 0,
            use_mla: false,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
        }
    }

    pub fn tiny_deepseek() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
            is_moe: true,
            num_routed_experts: 4,
            num_shared_experts: 1,
            top_k_experts: 2,
            use_mla: true,
            q_lora_rank: 0,
            kv_lora_rank: 32,
            qk_nope_head_dim: 12,
            qk_rope_head_dim: 4,
            v_head_dim: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerKvCache {
    pub k: Vec<f32>,      // [max_tokens * num_kv_heads * head_dim]
    pub v: Vec<f32>,      // [max_tokens * num_kv_heads * head_dim]
    pub c_kv: Vec<f32>,   // [max_tokens * kv_lora_rank]
    pub k_rope: Vec<f32>, // [max_tokens * qk_rope_head_dim]
    pub current_tokens: usize,
    pub is_mla: bool,
}

impl LayerKvCache {
    pub fn new(max_tokens: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: vec![0.0; max_tokens * num_kv_heads * head_dim],
            v: vec![0.0; max_tokens * num_kv_heads * head_dim],
            c_kv: Vec::new(),
            k_rope: Vec::new(),
            current_tokens: 0,
            is_mla: false,
        }
    }

    pub fn new_mla(max_tokens: usize, kv_lora_rank: usize, qk_rope_head_dim: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            c_kv: vec![0.0; max_tokens * kv_lora_rank],
            k_rope: vec![0.0; max_tokens * qk_rope_head_dim],
            current_tokens: 0,
            is_mla: true,
        }
    }

    pub fn reset(&mut self) {
        self.current_tokens = 0;
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(config: &TransformerConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            if config.use_mla {
                layers.push(LayerKvCache::new_mla(
                    config.max_position_embeddings,
                    config.kv_lora_rank,
                    config.qk_rope_head_dim,
                ));
            } else {
                layers.push(LayerKvCache::new(
                    config.max_position_embeddings,
                    config.num_key_value_heads,
                    config.head_dim,
                ));
            }
        }
        Self {
            layers,
            max_seq_len: config.max_position_embeddings,
        }
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }
}

pub struct AttentionWeights {
    pub q_proj: Vec<f32>, // [num_heads * head_dim, hidden_size]
    pub k_proj: Vec<f32>, // [num_kv_heads * head_dim, hidden_size]
    pub v_proj: Vec<f32>, // [num_kv_heads * head_dim, hidden_size]
    pub o_proj: Vec<f32>, // [hidden_size, num_heads * head_dim]
}

pub struct MlpWeights {
    pub gate_proj: Vec<f32>, // [intermediate_size, hidden_size]
    pub up_proj: Vec<f32>,   // [intermediate_size, hidden_size]
    pub down_proj: Vec<f32>, // [hidden_size, intermediate_size]
}

pub struct MoEWeights {
    pub router_weight: Vec<f32>, // [num_routed_experts, hidden_size]
    pub shared_experts: Vec<MlpWeights>,
    pub routed_experts: Vec<MlpWeights>,
}

pub struct MlaWeights {
    pub w_dq: Option<Vec<f32>>,
    pub w_uq: Vec<f32>,
    pub w_qr: Vec<f32>,
    pub w_dkv: Vec<f32>,
    pub w_kr: Vec<f32>,
    pub w_uk: Vec<f32>,
    pub w_uv: Vec<f32>,
    pub w_o: Vec<f32>,
}

pub struct TransformerLayerWeights {
    pub input_layernorm: Vec<f32>, // [hidden_size]
    pub attention: AttentionWeights,
    pub mla: Option<MlaWeights>,
    pub post_attention_layernorm: Vec<f32>, // [hidden_size]
    pub mlp: Option<MlpWeights>,
    pub moe: Option<MoEWeights>,
}

pub struct TransformerModel {
    pub config: TransformerConfig,
    pub embed_tokens: Vec<f32>, // [vocab_size, hidden_size]
    pub layers: Vec<TransformerLayerWeights>,
    pub norm: Vec<f32>,    // [hidden_size]
    pub lm_head: Vec<f32>, // [vocab_size, hidden_size]
}

impl TransformerModel {
    pub fn new_random(config: TransformerConfig) -> Self {
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;

        let pseudo_random_vec = |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let x = ((i * 1103515245 + 12345) & 0x7fff) as f32 / 32768.0;
                    (x - 0.5) * 2.0 * scale
                })
                .collect()
        };

        let embed_tokens = pseudo_random_vec(vocab * hidden, 0.02);
        let norm = vec![1.0; hidden];
        let lm_head = pseudo_random_vec(vocab * hidden, 0.02);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let input_layernorm = vec![1.0; hidden];
            let post_attention_layernorm = vec![1.0; hidden];

            let attention = AttentionWeights {
                q_proj: pseudo_random_vec(q_dim * hidden, 0.02),
                k_proj: pseudo_random_vec(kv_dim * hidden, 0.02),
                v_proj: pseudo_random_vec(kv_dim * hidden, 0.02),
                o_proj: pseudo_random_vec(hidden * q_dim, 0.02),
            };

            let mla = if config.use_mla {
                let q_in_dim = if config.q_lora_rank > 0 {
                    config.q_lora_rank
                } else {
                    hidden
                };
                let w_dq = if config.q_lora_rank > 0 {
                    Some(pseudo_random_vec(config.q_lora_rank * hidden, 0.02))
                } else {
                    None
                };
                let w_uq = pseudo_random_vec(
                    config.num_attention_heads * config.qk_nope_head_dim * q_in_dim,
                    0.02,
                );
                let w_qr = pseudo_random_vec(
                    config.num_attention_heads * config.qk_rope_head_dim * q_in_dim,
                    0.02,
                );
                let w_dkv = pseudo_random_vec(config.kv_lora_rank * hidden, 0.02);
                let w_kr = pseudo_random_vec(config.qk_rope_head_dim * hidden, 0.02);
                let w_uk = pseudo_random_vec(
                    config.num_attention_heads * config.qk_nope_head_dim * config.kv_lora_rank,
                    0.02,
                );
                let w_uv = pseudo_random_vec(
                    config.num_attention_heads * config.v_head_dim * config.kv_lora_rank,
                    0.02,
                );
                let w_o = pseudo_random_vec(
                    hidden * config.num_attention_heads * config.v_head_dim,
                    0.02,
                );
                Some(MlaWeights {
                    w_dq,
                    w_uq,
                    w_qr,
                    w_dkv,
                    w_kr,
                    w_uk,
                    w_uv,
                    w_o,
                })
            } else {
                None
            };

            let (mlp, moe) = if config.is_moe {
                let mut shared = Vec::with_capacity(config.num_shared_experts);
                for _ in 0..config.num_shared_experts {
                    shared.push(MlpWeights {
                        gate_proj: pseudo_random_vec(intermediate * hidden, 0.02),
                        up_proj: pseudo_random_vec(intermediate * hidden, 0.02),
                        down_proj: pseudo_random_vec(hidden * intermediate, 0.02),
                    });
                }
                let mut routed = Vec::with_capacity(config.num_routed_experts);
                for _ in 0..config.num_routed_experts {
                    routed.push(MlpWeights {
                        gate_proj: pseudo_random_vec(intermediate * hidden, 0.02),
                        up_proj: pseudo_random_vec(intermediate * hidden, 0.02),
                        down_proj: pseudo_random_vec(hidden * intermediate, 0.02),
                    });
                }
                let moe_weights = MoEWeights {
                    router_weight: pseudo_random_vec(config.num_routed_experts * hidden, 0.02),
                    shared_experts: shared,
                    routed_experts: routed,
                };
                (None, Some(moe_weights))
            } else {
                let mlp_weights = MlpWeights {
                    gate_proj: pseudo_random_vec(intermediate * hidden, 0.02),
                    up_proj: pseudo_random_vec(intermediate * hidden, 0.02),
                    down_proj: pseudo_random_vec(hidden * intermediate, 0.02),
                };
                (Some(mlp_weights), None)
            };

            layers.push(TransformerLayerWeights {
                input_layernorm,
                attention,
                mla,
                post_attention_layernorm,
                mlp,
                moe,
            });
        }

        Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }

    pub fn from_loader(
        config: TransformerConfig,
        loader: &SafetensorsMmapLoader,
    ) -> Result<Self, SingularityError> {
        let embed_tokens = loader
            .get_tensor_f32("model.embed_tokens.weight")
            .unwrap_or_else(|_| vec![0.0; config.vocab_size * config.hidden_size]);

        let norm = loader
            .get_tensor_f32("model.norm.weight")
            .unwrap_or_else(|_| vec![1.0; config.hidden_size]);

        let lm_head = loader
            .get_tensor_f32("lm_head.weight")
            .or_else(|_| loader.get_tensor_f32("model.embed_tokens.weight"))
            .unwrap_or_else(|_| vec![0.0; config.vocab_size * config.hidden_size]);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for l in 0..config.num_hidden_layers {
            let input_layernorm = loader
                .get_tensor_f32(&format!("model.layers.{l}.input_layernorm.weight"))
                .unwrap_or_else(|_| vec![1.0; config.hidden_size]);

            let post_attention_layernorm = loader
                .get_tensor_f32(&format!("model.layers.{l}.post_attention_layernorm.weight"))
                .unwrap_or_else(|_| vec![1.0; config.hidden_size]);

            let attention = AttentionWeights {
                q_proj: loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.q_proj.weight"))
                    .unwrap_or_else(|_| {
                        vec![0.0; config.num_attention_heads * config.head_dim * config.hidden_size]
                    }),
                k_proj: loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.k_proj.weight"))
                    .unwrap_or_else(|_| {
                        vec![0.0; config.num_key_value_heads * config.head_dim * config.hidden_size]
                    }),
                v_proj: loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.v_proj.weight"))
                    .unwrap_or_else(|_| {
                        vec![0.0; config.num_key_value_heads * config.head_dim * config.hidden_size]
                    }),
                o_proj: loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.o_proj.weight"))
                    .unwrap_or_else(|_| {
                        vec![0.0; config.hidden_size * config.num_attention_heads * config.head_dim]
                    }),
            };

            let mla = if config.use_mla {
                let q_in_dim = if config.q_lora_rank > 0 {
                    config.q_lora_rank
                } else {
                    config.hidden_size
                };
                let w_dq = if config.q_lora_rank > 0 {
                    loader
                        .get_tensor_f32(&format!("model.layers.{l}.self_attn.q_a_proj.weight"))
                        .ok()
                } else {
                    None
                };
                let w_uq = loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.q_b_proj.weight"))
                    .or_else(|_| {
                        loader.get_tensor_f32(&format!("model.layers.{l}.self_attn.q_proj.weight"))
                    })
                    .unwrap_or_else(|_| {
                        vec![0.0; config.num_attention_heads * config.qk_nope_head_dim * q_in_dim]
                    });
                let w_qr = loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.q_rope_proj.weight"))
                    .unwrap_or_else(|_| {
                        vec![0.0; config.num_attention_heads * config.qk_rope_head_dim * q_in_dim]
                    });
                let w_dkv = loader
                    .get_tensor_f32(&format!(
                        "model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight"
                    ))
                    .unwrap_or_else(|_| vec![0.0; config.kv_lora_rank * config.hidden_size]);
                let w_kr = loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.k_rope_proj.weight"))
                    .unwrap_or_else(|_| vec![0.0; config.qk_rope_head_dim * config.hidden_size]);
                let w_uk = loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.kv_b_proj.weight"))
                    .unwrap_or_else(|_| {
                        vec![
                            0.0;
                            config.num_attention_heads
                                * config.qk_nope_head_dim
                                * config.kv_lora_rank
                        ]
                    });
                let w_uv = loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.v_b_proj.weight"))
                    .unwrap_or_else(|_| {
                        vec![
                            0.0;
                            config.num_attention_heads * config.v_head_dim * config.kv_lora_rank
                        ]
                    });
                let w_o = loader
                    .get_tensor_f32(&format!("model.layers.{l}.self_attn.o_proj.weight"))
                    .unwrap_or_else(|_| {
                        vec![
                            0.0;
                            config.hidden_size * config.num_attention_heads * config.v_head_dim
                        ]
                    });
                Some(MlaWeights {
                    w_dq,
                    w_uq,
                    w_qr,
                    w_dkv,
                    w_kr,
                    w_uk,
                    w_uv,
                    w_o,
                })
            } else {
                None
            };

            let (mlp, moe) = if !config.is_moe {
                (
                    Some(MlpWeights {
                        gate_proj: loader
                            .get_tensor_f32(&format!("model.layers.{l}.mlp.gate_proj.weight"))
                            .unwrap_or_else(|_| {
                                vec![0.0; config.intermediate_size * config.hidden_size]
                            }),
                        up_proj: loader
                            .get_tensor_f32(&format!("model.layers.{l}.mlp.up_proj.weight"))
                            .unwrap_or_else(|_| {
                                vec![0.0; config.intermediate_size * config.hidden_size]
                            }),
                        down_proj: loader
                            .get_tensor_f32(&format!("model.layers.{l}.mlp.down_proj.weight"))
                            .unwrap_or_else(|_| {
                                vec![0.0; config.hidden_size * config.intermediate_size]
                            }),
                    }),
                    None,
                )
            } else {
                let router_weight = loader
                    .get_tensor_f32(&format!("model.layers.{l}.mlp.gate.weight"))
                    .or_else(|_| {
                        loader.get_tensor_f32(&format!("model.layers.{l}.mlp.router.weight"))
                    })
                    .unwrap_or_else(|_| vec![0.0; config.num_routed_experts * config.hidden_size]);

                let mut shared_experts = Vec::with_capacity(config.num_shared_experts);
                for _ in 0..config.num_shared_experts {
                    let gate_proj = loader
                        .get_tensor_f32(&format!(
                            "model.layers.{l}.mlp.shared_experts.gate_proj.weight"
                        ))
                        .unwrap_or_else(|_| {
                            vec![0.0; config.intermediate_size * config.hidden_size]
                        });
                    let up_proj = loader
                        .get_tensor_f32(&format!(
                            "model.layers.{l}.mlp.shared_experts.up_proj.weight"
                        ))
                        .unwrap_or_else(|_| {
                            vec![0.0; config.intermediate_size * config.hidden_size]
                        });
                    let down_proj = loader
                        .get_tensor_f32(&format!(
                            "model.layers.{l}.mlp.shared_experts.down_proj.weight"
                        ))
                        .unwrap_or_else(|_| {
                            vec![0.0; config.hidden_size * config.intermediate_size]
                        });
                    shared_experts.push(MlpWeights {
                        gate_proj,
                        up_proj,
                        down_proj,
                    });
                }

                let mut routed_experts = Vec::with_capacity(config.num_routed_experts);
                for e in 0..config.num_routed_experts {
                    let gate_proj = loader
                        .get_tensor_f32(&format!(
                            "model.layers.{l}.mlp.experts.{e}.gate_proj.weight"
                        ))
                        .unwrap_or_else(|_| {
                            vec![0.0; config.intermediate_size * config.hidden_size]
                        });
                    let up_proj = loader
                        .get_tensor_f32(&format!("model.layers.{l}.mlp.experts.{e}.up_proj.weight"))
                        .unwrap_or_else(|_| {
                            vec![0.0; config.intermediate_size * config.hidden_size]
                        });
                    let down_proj = loader
                        .get_tensor_f32(&format!(
                            "model.layers.{l}.mlp.experts.{e}.down_proj.weight"
                        ))
                        .unwrap_or_else(|_| {
                            vec![0.0; config.hidden_size * config.intermediate_size]
                        });
                    routed_experts.push(MlpWeights {
                        gate_proj,
                        up_proj,
                        down_proj,
                    });
                }

                (
                    None,
                    Some(MoEWeights {
                        router_weight,
                        shared_experts,
                        routed_experts,
                    }),
                )
            };

            layers.push(TransformerLayerWeights {
                input_layernorm,
                attention,
                mla,
                post_attention_layernorm,
                mlp,
                moe,
            });
        }

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Forward pass for a single token at position `pos`.
    /// Updates the KV cache and returns output logits over the vocabulary.
    pub fn forward_token(
        &self,
        token: u32,
        pos: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>, SingularityError> {
        let hidden = self.config.hidden_size;
        let mut x = vec![0.0f32; hidden];

        // 1. Embedding lookup
        let token_idx = (token as usize) % self.config.vocab_size;
        let emb_start = token_idx * hidden;
        let emb_end = emb_start + hidden;
        if emb_end <= self.embed_tokens.len() {
            x.copy_from_slice(&self.embed_tokens[emb_start..emb_end]);
        }

        // 2. Transformer layers
        let mut norm_buf = vec![0.0f32; hidden];
        let mut attn_out = vec![0.0f32; hidden];
        let mut mlp_out = vec![0.0f32; hidden];

        for (l_idx, layer) in self.layers.iter().enumerate() {
            // Pre-norm
            rms_norm(
                &x,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut norm_buf,
            );

            // Self-attention (MLA or GQA)
            if let Some(mla) = &layer.mla {
                forward_mla_attention(
                    &norm_buf,
                    mla,
                    &self.config,
                    pos,
                    &mut kv_cache.layers[l_idx],
                    &mut attn_out,
                );
            } else {
                forward_gqa_attention(
                    &norm_buf,
                    &layer.attention,
                    &self.config,
                    pos,
                    &mut kv_cache.layers[l_idx],
                    &mut attn_out,
                );
            }

            // Residual add
            for i in 0..hidden {
                x[i] += attn_out[i];
            }

            // Post-norm
            rms_norm(
                &x,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut norm_buf,
            );

            // MLP / MoE
            if let Some(mlp) = &layer.mlp {
                forward_swiglu_mlp(&norm_buf, mlp, &self.config, &mut mlp_out);
                for i in 0..hidden {
                    x[i] += mlp_out[i];
                }
            } else if let Some(moe) = &layer.moe {
                forward_moe(&norm_buf, moe, &self.config, &mut mlp_out);
                for i in 0..hidden {
                    x[i] += mlp_out[i];
                }
            }
        }

        // 3. Final RMSNorm
        rms_norm(&x, &self.norm, self.config.rms_norm_eps, &mut norm_buf);

        // 4. LM Head projection to logits
        let mut logits = vec![0.0f32; self.config.vocab_size];
        matvec_mul(
            &self.lm_head,
            &norm_buf,
            self.config.vocab_size,
            hidden,
            &mut logits,
        );

        Ok(logits)
    }

    /// Autoregressive forward over a sequence of prompt tokens.
    /// Returns the logits for the final token.
    pub fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>, SingularityError> {
        if tokens.is_empty() {
            return Err(SingularityError::ModelError {
                reason: "Cannot execute forward pass on empty token slice".to_string(),
            });
        }

        let mut last_logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            last_logits = self.forward_token(tok, pos, kv_cache)?;
        }

        Ok(last_logits)
    }
}

// ---------------------------------------------------------
// Helper Kernel Implementations (CPU reference + Host GEMM)
// ---------------------------------------------------------

#[inline]
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let dim = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = ((sum_sq / dim as f32) + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for (out_val, (&x_val, &w_val)) in out.iter_mut().zip(x.iter().zip(weight.iter())) {
        *out_val = x_val * inv_rms * w_val;
    }
}

#[inline]
pub fn apply_rope(x: &mut [f32], pos: usize, num_heads: usize, head_dim: usize, theta: f32) {
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let head_offset = h * head_dim;
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();

            let x0 = x[head_offset + 2 * i];
            let x1 = x[head_offset + 2 * i + 1];

            x[head_offset + 2 * i] = x0 * cos - x1 * sin;
            x[head_offset + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }
}

/// Computes y = W * x where W has shape [out_dim, in_dim]
#[inline]
pub fn matvec_mul(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize, out: &mut [f32]) {
    for (i, out_val) in out.iter_mut().enumerate().take(out_dim) {
        let mut sum = 0.0f32;
        let row_start = i * in_dim;
        for (j, &x_val) in x.iter().enumerate().take(in_dim) {
            sum += w[row_start + j] * x_val;
        }
        *out_val = sum;
    }
}

fn forward_gqa_attention(
    x: &[f32],
    weights: &AttentionWeights,
    config: &TransformerConfig,
    pos: usize,
    cache: &mut LayerKvCache,
    out: &mut [f32],
) {
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let hidden = config.hidden_size;

    // Q, K, V projections
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    matvec_mul(&weights.q_proj, x, q_dim, hidden, &mut q);
    matvec_mul(&weights.k_proj, x, kv_dim, hidden, &mut k);
    matvec_mul(&weights.v_proj, x, kv_dim, hidden, &mut v);

    // Apply RoPE to Q and K
    apply_rope(
        &mut q,
        pos,
        config.num_attention_heads,
        config.head_dim,
        config.rope_theta,
    );
    apply_rope(
        &mut k,
        pos,
        config.num_key_value_heads,
        config.head_dim,
        config.rope_theta,
    );

    // Store K and V in cache
    let cache_idx = cache.current_tokens;
    if cache_idx < config.max_position_embeddings {
        let offset = cache_idx * kv_dim;
        cache.k[offset..offset + kv_dim].copy_from_slice(&k);
        cache.v[offset..offset + kv_dim].copy_from_slice(&v);
        cache.current_tokens += 1;
    }
    let total_tokens = cache.current_tokens;

    // Scaled Dot-Product Attention with GQA
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let gqa_ratio = config.num_attention_heads / config.num_key_value_heads;
    let mut head_outputs = vec![0.0f32; q_dim];

    for h in 0..config.num_attention_heads {
        let kv_head = h / gqa_ratio;
        let q_head_start = h * config.head_dim;
        let q_slice = &q[q_head_start..q_head_start + config.head_dim];

        // Scores against all cached tokens
        let mut scores = Vec::with_capacity(total_tokens);
        for t in 0..total_tokens {
            let k_offset = t * kv_dim + kv_head * config.head_dim;
            let k_slice = &cache.k[k_offset..k_offset + config.head_dim];

            let dot: f32 = q_slice
                .iter()
                .zip(k_slice.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            scores.push(dot * scale);
        }

        // Softmax over scores
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let inv_sum = 1.0 / sum_exp.max(1e-6);

        // Weighted sum of V
        let out_head_start = h * config.head_dim;
        for (t, exp) in exps.iter().enumerate() {
            let weight = exp * inv_sum;
            let v_offset = t * kv_dim + kv_head * config.head_dim;
            let v_slice = &cache.v[v_offset..v_offset + config.head_dim];

            for d in 0..config.head_dim {
                head_outputs[out_head_start + d] += weight * v_slice[d];
            }
        }
    }

    // Output projection: out = o_proj * head_outputs
    matvec_mul(&weights.o_proj, &head_outputs, hidden, q_dim, out);
}

fn forward_mla_attention(
    x: &[f32],
    weights: &MlaWeights,
    config: &TransformerConfig,
    pos: usize,
    cache: &mut LayerKvCache,
    out: &mut [f32],
) {
    let num_heads = config.num_attention_heads;
    let qk_nope_dim = config.qk_nope_head_dim;
    let qk_rope_dim = config.qk_rope_head_dim;
    let v_dim = config.v_head_dim;
    let kv_lora_rank = config.kv_lora_rank;
    let hidden = config.hidden_size;

    // 1. Query projection
    let q_in = if let Some(w_dq) = &weights.w_dq {
        let mut q_latent = vec![0.0f32; config.q_lora_rank];
        matvec_mul(w_dq, x, config.q_lora_rank, hidden, &mut q_latent);
        q_latent
    } else {
        x.to_vec()
    };
    let q_in_dim = q_in.len();

    let mut q_nope = vec![0.0f32; num_heads * qk_nope_dim];
    let mut q_rope = vec![0.0f32; num_heads * qk_rope_dim];
    matvec_mul(
        &weights.w_uq,
        &q_in,
        num_heads * qk_nope_dim,
        q_in_dim,
        &mut q_nope,
    );
    matvec_mul(
        &weights.w_qr,
        &q_in,
        num_heads * qk_rope_dim,
        q_in_dim,
        &mut q_rope,
    );

    // Apply RoPE to query rope component
    apply_rope(&mut q_rope, pos, num_heads, qk_rope_dim, config.rope_theta);

    // 2. Key-Value down-projection & RoPE key
    let mut c_kv = vec![0.0f32; kv_lora_rank];
    let mut k_rope = vec![0.0f32; qk_rope_dim];
    matvec_mul(&weights.w_dkv, x, kv_lora_rank, hidden, &mut c_kv);
    matvec_mul(&weights.w_kr, x, qk_rope_dim, hidden, &mut k_rope);
    apply_rope(&mut k_rope, pos, 1, qk_rope_dim, config.rope_theta);

    // 3. Store into compressed latent KV cache
    let cache_idx = cache.current_tokens;
    if cache_idx < config.max_position_embeddings {
        let c_offset = cache_idx * kv_lora_rank;
        cache.c_kv[c_offset..c_offset + kv_lora_rank].copy_from_slice(&c_kv);
        let kr_offset = cache_idx * qk_rope_dim;
        cache.k_rope[kr_offset..kr_offset + qk_rope_dim].copy_from_slice(&k_rope);
        cache.current_tokens += 1;
    }
    let total_tokens = cache.current_tokens;

    // 4. Multi-head Latent Attention calculation
    let qk_dim = qk_nope_dim + qk_rope_dim;
    let scale = 1.0 / (qk_dim as f32).sqrt();
    let mut head_outputs = vec![0.0f32; num_heads * v_dim];

    for h in 0..num_heads {
        let q_nope_start = h * qk_nope_dim;
        let q_h_nope = &q_nope[q_nope_start..q_nope_start + qk_nope_dim];
        let q_rope_start = h * qk_rope_dim;
        let q_h_rope = &q_rope[q_rope_start..q_rope_start + qk_rope_dim];

        let w_uk_head =
            &weights.w_uk[h * qk_nope_dim * kv_lora_rank..(h + 1) * qk_nope_dim * kv_lora_rank];
        let w_uv_head = &weights.w_uv[h * v_dim * kv_lora_rank..(h + 1) * v_dim * kv_lora_rank];

        let mut scores = Vec::with_capacity(total_tokens);

        for t in 0..total_tokens {
            let c_offset = t * kv_lora_rank;
            let c_t = &cache.c_kv[c_offset..c_offset + kv_lora_rank];
            let kr_offset = t * qk_rope_dim;
            let kr_t = &cache.k_rope[kr_offset..kr_offset + qk_rope_dim];

            // Compute k_nope_t for head h = w_uk_head * c_t
            let mut k_nope_t = vec![0.0f32; qk_nope_dim];
            matvec_mul(w_uk_head, c_t, qk_nope_dim, kv_lora_rank, &mut k_nope_t);

            // Dot product (q_nope . k_nope) + (q_rope . k_rope)
            let dot_nope: f32 = q_h_nope
                .iter()
                .zip(k_nope_t.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let dot_rope: f32 = q_h_rope.iter().zip(kr_t.iter()).map(|(&a, &b)| a * b).sum();
            scores.push((dot_nope + dot_rope) * scale);
        }

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let inv_sum = 1.0 / sum_exp.max(1e-6);

        // Weighted sum of V
        let out_head_start = h * v_dim;
        let mut v_t = vec![0.0f32; v_dim];
        for (t, exp) in exps.iter().enumerate() {
            let weight = exp * inv_sum;
            let c_offset = t * kv_lora_rank;
            let c_t = &cache.c_kv[c_offset..c_offset + kv_lora_rank];
            matvec_mul(w_uv_head, c_t, v_dim, kv_lora_rank, &mut v_t);

            for d in 0..v_dim {
                head_outputs[out_head_start + d] += weight * v_t[d];
            }
        }
    }

    // 5. Output projection: out = w_o * head_outputs
    matvec_mul(&weights.w_o, &head_outputs, hidden, num_heads * v_dim, out);
}

fn forward_swiglu_mlp(
    x: &[f32],
    weights: &MlpWeights,
    config: &TransformerConfig,
    out: &mut [f32],
) {
    let inter = config.intermediate_size;
    let hidden = config.hidden_size;

    let mut gate = vec![0.0f32; inter];
    let mut up = vec![0.0f32; inter];
    let mut activated = vec![0.0f32; inter];

    matvec_mul(&weights.gate_proj, x, inter, hidden, &mut gate);
    matvec_mul(&weights.up_proj, x, inter, hidden, &mut up);

    // SwiGLU: silu(gate) * up
    for i in 0..inter {
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        activated[i] = silu * up[i];
    }

    matvec_mul(&weights.down_proj, &activated, hidden, inter, out);
}

fn forward_moe(x: &[f32], weights: &MoEWeights, config: &TransformerConfig, out: &mut [f32]) {
    let hidden = config.hidden_size;
    out.fill(0.0);

    // 1. Shared experts
    let mut shared_out = vec![0.0f32; hidden];
    for exp_weights in &weights.shared_experts {
        forward_swiglu_mlp(x, exp_weights, config, &mut shared_out);
        for i in 0..hidden {
            out[i] += shared_out[i];
        }
    }

    // 2. Router top-k scoring
    let mut router_logits = vec![0.0f32; config.num_routed_experts];
    matvec_mul(
        &weights.router_weight,
        x,
        config.num_routed_experts,
        hidden,
        &mut router_logits,
    );

    let router = crate::moe::MoERouter::new(config.num_routed_experts, config.top_k_experts);
    let selected = router.score_and_route(&router_logits);

    // 3. Routed experts
    let mut routed_out = vec![0.0f32; hidden];
    for (exp_id, weight) in selected {
        if exp_id < weights.routed_experts.len() {
            forward_swiglu_mlp(x, &weights.routed_experts[exp_id], config, &mut routed_out);
            for i in 0..hidden {
                out[i] += weight * routed_out[i];
            }
        }
    }
}

// ---------------------------------------------------------
// Sampler Implementation
// ---------------------------------------------------------

pub struct Sampler;

impl Sampler {
    pub fn sample(logits: &[f32], output_tokens: &[u32], params: &SamplingParams) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // Copy and apply penalties
        let mut modified_logits = logits.to_vec();

        if params.presence_penalty != 0.0 || params.frequency_penalty != 0.0 {
            let mut counts = std::collections::HashMap::new();
            for &t in output_tokens {
                *counts.entry(t).or_insert(0usize) += 1;
            }
            for (&tok, &count) in &counts {
                let idx = tok as usize;
                if idx < modified_logits.len() {
                    modified_logits[idx] -=
                        params.presence_penalty + (params.frequency_penalty * count as f32);
                }
            }
        }

        // Greedy argmax if temperature is close to 0
        if params.temperature < 1e-4 {
            let mut best_idx = 0;
            let mut best_logit = f32::NEG_INFINITY;
            for (idx, &val) in modified_logits.iter().enumerate() {
                if val > best_logit {
                    best_logit = val;
                    best_idx = idx;
                }
            }
            return best_idx as u32;
        }

        // Scale by temperature
        let inv_temp = 1.0 / params.temperature;
        for val in &mut modified_logits {
            *val *= inv_temp;
        }

        // Softmax
        let max_logit = modified_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<(usize, f32)> = modified_logits
            .iter()
            .enumerate()
            .map(|(idx, &l)| (idx, (l - max_logit).exp()))
            .collect();

        // Sort descending for top-p (nucleus)
        exps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total_mass: f32 = exps.iter().map(|s| s.1).sum();

        let top_p_threshold = params.top_p * total_mass;
        let mut cumulative = 0.0f32;
        let mut truncated = Vec::new();

        for &(idx, prob) in &exps {
            cumulative += prob;
            truncated.push((idx, prob));
            if cumulative >= top_p_threshold {
                break;
            }
        }

        // Deterministic sampling: choose top candidate
        truncated.first().map(|&(idx, _)| idx as u32).unwrap_or(0)
    }
}

// ---------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_computation() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        rms_norm(&x, &weight, 1e-5, &mut out);

        // mean of squares = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
        // rms = sqrt(7.5) ~= 2.7386
        let expected_inv_rms = 1.0 / (7.5f32).sqrt();
        assert!((out[0] - 1.0 * expected_inv_rms).abs() < 1e-4);
        assert!((out[3] - 4.0 * expected_inv_rms).abs() < 1e-4);
    }

    #[test]
    fn test_rope_rotation() {
        let mut x = vec![1.0, 0.0];
        apply_rope(&mut x, 0, 1, 2, 10000.0);
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_tiny_transformer_forward_pass() {
        let config = TransformerConfig::tiny();
        let model = TransformerModel::new_random(config.clone());
        let mut kv_cache = KvCache::new(&config);

        let prompt = vec![12u32, 45, 99];
        let logits = model.forward(&prompt, 0, &mut kv_cache).unwrap();

        assert_eq!(logits.len(), config.vocab_size);
        // Verify KV cache accumulated 3 tokens
        assert_eq!(kv_cache.layers[0].current_tokens, 3);

        // Sample next token
        let params = SamplingParams::default();
        let sampled = Sampler::sample(&logits, &prompt, &params);
        assert!((sampled as usize) < config.vocab_size);
    }

    #[test]
    fn test_deepseek_moe_forward_pass() {
        let mut config = TransformerConfig::tiny();
        config.is_moe = true;
        config.num_routed_experts = 4;
        config.num_shared_experts = 1;
        config.top_k_experts = 2;

        let model = TransformerModel::new_random(config.clone());
        let mut kv_cache = KvCache::new(&config);

        let prompt = vec![1, 2];
        let logits = model.forward(&prompt, 0, &mut kv_cache).unwrap();
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_deepseek_mla_and_compressed_kv_cache_forward_pass() {
        let config = TransformerConfig::tiny_deepseek();
        assert!(config.use_mla);
        assert!(config.is_moe);

        let model = TransformerModel::new_random(config.clone());
        let mut kv_cache = KvCache::new(&config);

        // Verify that layer 0 uses compressed MLA cache
        assert!(kv_cache.layers[0].is_mla);
        assert_eq!(
            kv_cache.layers[0].c_kv.len(),
            config.max_position_embeddings * config.kv_lora_rank
        );
        assert_eq!(
            kv_cache.layers[0].k_rope.len(),
            config.max_position_embeddings * config.qk_rope_head_dim
        );
        assert_eq!(kv_cache.layers[0].k.len(), 0);

        let prompt = vec![5u32, 12, 19];
        let logits = model.forward(&prompt, 0, &mut kv_cache).unwrap();
        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(kv_cache.layers[0].current_tokens, 3);

        // Autoregressive forward of next token
        let next_logits = model.forward_token(25, 3, &mut kv_cache).unwrap();
        assert_eq!(next_logits.len(), config.vocab_size);
        assert_eq!(kv_cache.layers[0].current_tokens, 4);
    }
}
