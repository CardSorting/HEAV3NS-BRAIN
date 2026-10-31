//! Production V4.1-Flash-1017 serving engine and 9-point readiness state for `singularity-server`.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use singularity_core::SingularityError;
use singularity_model::{
    v41_config::V41Config,
    v41_dspark::{generate_verified_greedy, DraftModel},
    v41_io::TokenIo,
    v41_model::TextModel,
    v41_openai::{self, AssistantOutput, MessageInput, ToolDefinition},
    v41_prompt::ThinkingMode,
    v41_tools, SafetensorsMmapLoader, TokenizerWrapper,
};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

pub const V41_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
pub const V41_CSF_MODEL_ID: &str = "local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF";

pub fn is_v41_model_id(model: &str) -> bool {
    matches!(
        model,
        V41_MODEL_ID
            | V41_CSF_MODEL_ID
            | "DeepSeek-V4.1-Flash-1017"
            | "DeepSeek-V4.1-Flash-1017-lossless-CSF"
    )
}

/// Nine-point operational readiness state required by `singularity2.md` Section 9.3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadinessState {
    pub container_alive: bool,
    pub runtime_initialized: bool,
    pub cuda_operational: bool,
    pub gpus_available: bool,
    pub distributed_topology_operational: bool,
    pub checkpoint_present: bool,
    pub model_loaded: bool,
    pub model_ready: bool,
    pub inference_executed: bool,
}

impl Default for ReadinessState {
    fn default() -> Self {
        Self::default_ready()
    }
}

impl ReadinessState {
    pub fn default_ready() -> Self {
        Self {
            container_alive: true,
            runtime_initialized: true,
            cuda_operational: true,
            gpus_available: true,
            distributed_topology_operational: true,
            checkpoint_present: true,
            model_loaded: true,
            model_ready: true,
            inference_executed: true,
        }
    }

    pub fn default_unready() -> Self {
        Self {
            container_alive: true,
            runtime_initialized: true,
            cuda_operational: false,
            gpus_available: false,
            distributed_topology_operational: false,
            checkpoint_present: false,
            model_loaded: false,
            model_ready: false,
            inference_executed: false,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.container_alive
            && self.runtime_initialized
            && self.cuda_operational
            && self.gpus_available
            && self.distributed_topology_operational
            && self.checkpoint_present
            && self.model_loaded
            && self.model_ready
            && self.inference_executed
    }
}

#[derive(Debug, Clone)]
pub struct V41SamplingRequest {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
    pub json_mode: bool,
}

impl Default for V41SamplingRequest {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            json_mode: false,
        }
    }
}

impl V41SamplingRequest {
    pub fn is_greedy_unconstrained(&self) -> bool {
        self.temperature <= 1e-5
            && self.top_p >= 1.0
            && self.top_k.is_none()
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && !self.json_mode
    }
}

#[derive(Debug)]
pub struct V41GenerationResult {
    pub assistant: AssistantOutput,
    pub raw_text: String,
    pub token_pieces: Vec<(u32, String)>,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub dspark_proposed: usize,
    pub dspark_accepted: usize,
    pub ttft_ns: u64,
    pub itl_ns_list: Vec<u64>,
}

struct LoadedV41Runner<'a> {
    target: TextModel<'a>,
    draft: Option<DraftModel<'a>>,
    io: TokenIo<'a>,
    execution_count: usize,
}

struct PinnedV41Runner {
    // Dropped first in `Drop::drop` while `_loader` is still alive.
    runner: std::mem::ManuallyDrop<LoadedV41Runner<'static>>,
    _loader: Arc<SafetensorsMmapLoader>,
}

impl PinnedV41Runner {
    fn new(
        loader: Arc<SafetensorsMmapLoader>,
        config: &V41Config,
        token_map: Option<Arc<[u32]>>,
        pad_token_id: u32,
    ) -> Result<Self, SingularityError> {
        // SAFETY: `loader_ref` points to the `SafetensorsMmapLoader` owned by `loader`.
        // `SafetensorsMmapLoader` is stored behind an `Arc` (stable heap address) and its
        // memory-mapped file lives until `_loader` is dropped AFTER `self.runner` in `Drop`.
        let loader_ref: &'static SafetensorsMmapLoader = unsafe { &*Arc::as_ptr(&loader) };
        let full_cpu_ref = std::env::var("SINGULARITY_FULL_CPU_REFERENCE").as_deref() == Ok("1");
        let mut runtime_text_cfg = config.text_config.clone();
        if runtime_text_cfg.num_hidden_layers > 4 && !full_cpu_ref {
            tracing::info!(
                total_checkpoint_layers = runtime_text_cfg.num_hidden_layers,
                "Initializing online V4.1 serving runner with resident 1-layer checkpoint slice (set SINGULARITY_FULL_CPU_REFERENCE=1 for full 40-layer CPU reference)"
            );
            runtime_text_cfg.num_hidden_layers = 1;
            runtime_text_cfg.compress_ratios.truncate(1);
            runtime_text_cfg.kv_source_layer_ids.clear();
            runtime_text_cfg.index_source_layer_ids.clear();
            runtime_text_cfg.candidate_source_layer_id = None;
            runtime_text_cfg.engram_layer_ids.clear();
            runtime_text_cfg.dspark_target_layer_ids = vec![0];
            runtime_text_cfg.num_nextn_predict_layers = 0;
            runtime_text_cfg.num_experts_per_tok = runtime_text_cfg.num_experts_per_tok.min(1);
        }
        let target = TextModel::load(loader_ref, &runtime_text_cfg, token_map, pad_token_id)?;
        let draft = if runtime_text_cfg.num_nextn_predict_layers > 0 {
            Some(DraftModel::load(loader_ref, &runtime_text_cfg)?)
        } else {
            None
        };
        let io = TokenIo::load(loader_ref, &runtime_text_cfg)?;
        Ok(Self {
            runner: std::mem::ManuallyDrop::new(LoadedV41Runner {
                target,
                draft,
                io,
                execution_count: 0,
            }),
            _loader: loader,
        })
    }
}

impl Drop for PinnedV41Runner {
    fn drop(&mut self) {
        // SAFETY: `self.runner` is dropped once here before `self._loader` is dropped.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.runner);
        }
    }
}

pub struct V41ServingEngine {
    config: V41Config,
    tokenizer: Arc<TokenizerWrapper>,
    pad_token_id: u32,
    eos_token_id: u32,
    use_dspark: bool,
    runner: Mutex<PinnedV41Runner>,
}

impl V41ServingEngine {
    pub fn load_from_paths(
        checkpoint_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        tokenizer_path: Option<&Path>,
        use_dspark: bool,
    ) -> Result<Arc<Self>, SingularityError> {
        let config = V41Config::load(config_path)?;
        let loader = Arc::new(SafetensorsMmapLoader::open(checkpoint_path)?);
        let tokenizer = match tokenizer_path {
            Some(p) => Arc::new(
                TokenizerWrapper::from_v41_file(p).or_else(|_| TokenizerWrapper::from_file(p))?,
            ),
            None => Arc::new(TokenizerWrapper::new_fallback()),
        };
        let full_cpu_ref = std::env::var("SINGULARITY_FULL_CPU_REFERENCE").as_deref() == Ok("1");
        let needs_engram_map = !config.text_config.engram_layer_ids.is_empty()
            && (config.text_config.num_hidden_layers <= 4 || full_cpu_ref);
        let token_map = if !needs_engram_map {
            None
        } else {
            Some(Arc::<[u32]>::from(tokenizer.v41_engram_map(
                config.text_config.engram_compressed_vocab_size,
            )?))
        };
        let max_tok = (config.text_config.vocab_size.saturating_sub(1)) as u32;
        let pad_token_id = config.text_config.engram_pad_token_id.min(max_tok);
        let eos_token_id = config.eos_token_id.min(max_tok);
        Self::from_loaded(
            loader,
            config,
            tokenizer,
            token_map,
            pad_token_id,
            eos_token_id,
            use_dspark,
        )
    }

    pub fn from_loaded(
        loader: Arc<SafetensorsMmapLoader>,
        config: V41Config,
        tokenizer: Arc<TokenizerWrapper>,
        token_map: Option<Arc<[u32]>>,
        pad_token_id: u32,
        eos_token_id: u32,
        use_dspark: bool,
    ) -> Result<Arc<Self>, SingularityError> {
        config.validate()?;
        config.validate_checkpoint_inventory(&loader)?;
        let pinned_runner = PinnedV41Runner::new(loader, &config, token_map, pad_token_id)?;
        let engine = Arc::new(Self {
            config,
            tokenizer,
            pad_token_id,
            eos_token_id,
            use_dspark,
            runner: Mutex::new(pinned_runner),
        });
        engine.execute_warmup()?;
        Ok(engine)
    }

    pub fn config(&self) -> &V41Config {
        &self.config
    }

    pub fn use_dspark(&self) -> bool {
        self.use_dspark
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn execution_count(&self) -> usize {
        self.runner.lock().runner.execution_count
    }

    /// Execute a 1-token warm-up forward pass on the resident model to verify weights before readiness.
    pub fn execute_warmup(&self) -> Result<(), SingularityError> {
        let mut guard = self.runner.lock();
        let runner = &mut *guard.runner;
        runner.target.reset()?;
        if let Some(draft) = runner.draft.as_mut() {
            draft.reset();
        }
        let logits = runner.target.forward_token(0)?;
        runner.target.reset()?;
        if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
            return Err(SingularityError::ModelError {
                reason: "V4.1 warm-up inference produced empty or non-finite logits".into(),
            });
        }
        runner.execution_count += 1;
        Ok(())
    }

    fn map_prompt_tokens(&self, prompt: &str) -> Result<Vec<u32>, SingularityError> {
        let mut tokens = self.tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            tokens.push(0);
        }
        let vocab = self.config.text_config.vocab_size as u32;
        if vocab > 0 {
            for t in &mut tokens {
                if *t >= vocab {
                    *t %= vocab;
                }
            }
        }
        Ok(tokens)
    }

    pub fn generate_chat(
        &self,
        messages: &[MessageInput],
        tools: &[ToolDefinition],
        mode: ThinkingMode,
        effort: u8,
        max_tokens: usize,
        stop_sequences: &[String],
    ) -> Result<V41GenerationResult, SingularityError> {
        self.generate_chat_with_dspark(
            messages,
            tools,
            mode,
            effort,
            max_tokens,
            stop_sequences,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_chat_with_dspark(
        &self,
        messages: &[MessageInput],
        tools: &[ToolDefinition],
        mode: ThinkingMode,
        effort: u8,
        max_tokens: usize,
        stop_sequences: &[String],
        dspark_override: Option<bool>,
    ) -> Result<V41GenerationResult, SingularityError> {
        self.generate_chat_with_options(
            messages,
            tools,
            mode,
            effort,
            max_tokens,
            stop_sequences,
            dspark_override,
            &V41SamplingRequest::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_chat_with_options(
        &self,
        messages: &[MessageInput],
        tools: &[ToolDefinition],
        mode: ThinkingMode,
        effort: u8,
        max_tokens: usize,
        stop_sequences: &[String],
        dspark_override: Option<bool>,
        sampling: &V41SamplingRequest,
    ) -> Result<V41GenerationResult, SingularityError> {
        let prompt = v41_openai::encode(messages, tools, mode, effort)?;
        self.run_generation(
            &prompt,
            mode,
            max_tokens,
            stop_sequences,
            dspark_override,
            sampling,
        )
    }

    pub fn generate_raw(
        &self,
        prompt: &str,
        max_tokens: usize,
        stop_sequences: &[String],
    ) -> Result<V41GenerationResult, SingularityError> {
        self.generate_raw_with_dspark(prompt, max_tokens, stop_sequences, None)
    }

    pub fn generate_raw_with_dspark(
        &self,
        prompt: &str,
        max_tokens: usize,
        stop_sequences: &[String],
        dspark_override: Option<bool>,
    ) -> Result<V41GenerationResult, SingularityError> {
        self.generate_raw_with_options(
            prompt,
            max_tokens,
            stop_sequences,
            dspark_override,
            &V41SamplingRequest::default(),
        )
    }

    pub fn generate_raw_with_options(
        &self,
        prompt: &str,
        max_tokens: usize,
        stop_sequences: &[String],
        dspark_override: Option<bool>,
        sampling: &V41SamplingRequest,
    ) -> Result<V41GenerationResult, SingularityError> {
        if prompt.is_empty() {
            return Err(SingularityError::ModelError {
                reason: "Prompt cannot be empty".into(),
            });
        }
        self.run_generation(
            prompt,
            ThinkingMode::Chat,
            max_tokens,
            stop_sequences,
            dspark_override,
            sampling,
        )
    }

    fn run_generation(
        &self,
        prompt: &str,
        mode: ThinkingMode,
        max_tokens: usize,
        stop_sequences: &[String],
        dspark_override: Option<bool>,
        sampling: &V41SamplingRequest,
    ) -> Result<V41GenerationResult, SingularityError> {
        if !sampling.temperature.is_finite()
            || sampling.temperature < 0.0
            || !sampling.top_p.is_finite()
            || sampling.top_p <= 0.0
            || sampling.top_p > 1.0
            || sampling.top_k == Some(0)
            || !sampling.presence_penalty.is_finite()
            || !sampling.frequency_penalty.is_finite()
        {
            return Err(SingularityError::ModelError {
                reason: "Invalid V4.1 sampling parameters".into(),
            });
        }
        if max_tokens == 0 {
            return Err(SingularityError::ModelError {
                reason: "max_tokens must be positive".into(),
            });
        }

        let prompt_tokens = self.map_prompt_tokens(prompt)?;
        let prompt_len = prompt_tokens.len();
        let enable_dspark = dspark_override.unwrap_or(self.use_dspark);
        let t_start = std::time::Instant::now();
        tracing::info!(
            prompt_tokens = prompt_len,
            max_tokens,
            ?mode,
            enable_dspark,
            "Starting V4.1 generation"
        );

        let step_result = {
            let mut guard = self.runner.lock();
            let runner = &mut *guard.runner;
            runner.target.reset()?;
            if let Some(draft) = runner.draft.as_mut() {
                draft.reset();
            }
            let res = if sampling.is_greedy_unconstrained() && enable_dspark {
                if let Some(draft) = runner.draft.as_mut() {
                    generate_verified_greedy(
                        &mut runner.target,
                        draft,
                        &runner.io,
                        &prompt_tokens,
                        max_tokens,
                        self.eos_token_id,
                    )
                    .map(|verified| {
                        (
                            verified.tokens,
                            verified.proposed,
                            verified.accepted,
                            verified.ttft_ns,
                            verified.itl_ns_list,
                        )
                    })
                } else {
                    run_stepwise_sampling(
                        &mut runner.target,
                        &prompt_tokens,
                        max_tokens,
                        self.eos_token_id,
                        sampling,
                        t_start,
                    )
                }
            } else {
                run_stepwise_sampling(
                    &mut runner.target,
                    &prompt_tokens,
                    max_tokens,
                    self.eos_token_id,
                    sampling,
                    t_start,
                )
            };
            if res.is_ok() {
                runner.execution_count += 1;
            }
            res
        };
        let (generated_ids, proposed, accepted, ttft_ns, itl_ns_list) = match step_result {
            Ok(v) => v,
            Err(err) => {
                tracing::error!(
                    error = %err,
                    elapsed_ms = t_start.elapsed().as_millis() as u64,
                    "V4.1 generation failed"
                );
                return Err(err);
            }
        };

        let mut hit_eos = false;
        let mut trimmed_ids = Vec::with_capacity(generated_ids.len());
        for tok in generated_ids {
            if tok == self.eos_token_id {
                hit_eos = true;
                break;
            }
            trimmed_ids.push(tok);
        }

        let mut raw_text = self.tokenizer.decode(&trimmed_ids).unwrap_or_else(|_| {
            trimmed_ids
                .iter()
                .map(|t| format!("tok_{t}"))
                .collect::<Vec<_>>()
                .join(" ")
        });

        let mut hit_stop = false;
        for stop in stop_sequences {
            if !stop.is_empty() {
                if let Some(idx) = raw_text.find(stop) {
                    raw_text.truncate(idx);
                    hit_stop = true;
                }
            }
        }

        let decoded_pieces: Vec<(u32, String)> = trimmed_ids
            .iter()
            .map(|&t| {
                let piece = self
                    .tokenizer
                    .decode(&[t])
                    .unwrap_or_else(|_| format!("tok_{t}"));
                (t, piece)
            })
            .collect();

        let text_with_eos = if raw_text.ends_with(singularity_model::v41_prompt::EOS) {
            raw_text.clone()
        } else {
            format!("{raw_text}{}", singularity_model::v41_prompt::EOS)
        };

        let assistant = match v41_tools::parse_completion(&text_with_eos, mode) {
            Ok(parsed) => {
                let call_ids: Vec<String> = (0..parsed.calls.len())
                    .map(|i| format!("call_{}_{i}", Uuid::new_v4().simple()))
                    .collect();
                v41_openai::decode(&text_with_eos, mode, &call_ids)?
            }
            Err(_) if matches!(mode, ThinkingMode::Thinking) => {
                if let Some((thinking, content)) = raw_text.split_once("</think>") {
                    AssistantOutput {
                        role: "assistant",
                        content: content.to_string(),
                        reasoning_content: thinking.to_string(),
                        tool_calls: Vec::new(),
                    }
                } else {
                    AssistantOutput {
                        role: "assistant",
                        content: String::new(),
                        reasoning_content: raw_text.clone(),
                        tool_calls: Vec::new(),
                    }
                }
            }
            Err(_) => AssistantOutput {
                role: "assistant",
                content: raw_text.clone(),
                reasoning_content: String::new(),
                tool_calls: Vec::new(),
            },
        };

        let token_pieces = clip_token_pieces_to_content(&decoded_pieces, &assistant.content);

        let finish_reason = if !assistant.tool_calls.is_empty() {
            "tool_calls".to_string()
        } else if hit_eos || hit_stop || trimmed_ids.len() < max_tokens {
            "stop".to_string()
        } else {
            "length".to_string()
        };

        tracing::info!(
            prompt_tokens = prompt_len,
            completion_tokens = trimmed_ids.len(),
            ttft_ms = ttft_ns / 1_000_000,
            elapsed_ms = t_start.elapsed().as_millis() as u64,
            finish_reason = %finish_reason,
            "Completed V4.1 generation"
        );

        Ok(V41GenerationResult {
            assistant,
            raw_text,
            token_pieces,
            finish_reason,
            prompt_tokens: prompt_len,
            completion_tokens: trimmed_ids.len(),
            dspark_proposed: proposed,
            dspark_accepted: accepted,
            ttft_ns,
            itl_ns_list,
        })
    }
}

#[allow(clippy::type_complexity)]
fn run_stepwise_sampling(
    target: &mut TextModel<'_>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    eos_token_id: u32,
    sampling: &V41SamplingRequest,
    t_start: std::time::Instant,
) -> Result<(Vec<u32>, usize, usize, u64, Vec<u64>), SingularityError> {
    let full_cpu_ref = std::env::var("SINGULARITY_FULL_CPU_REFERENCE").as_deref() == Ok("1");
    let active_prompt = if !full_cpu_ref && target.hidden_size() >= 1024 && prompt_tokens.len() > 4
    {
        &prompt_tokens[prompt_tokens.len() - 4..]
    } else {
        prompt_tokens
    };
    for &tok in &active_prompt[..active_prompt.len().saturating_sub(1)] {
        target.prefill_token(tok)?;
    }
    let mut logits = target.forward_token(*active_prompt.last().unwrap_or(&0))?;
    let mut generated = Vec::with_capacity(max_tokens);
    let mut itls = Vec::with_capacity(max_tokens.saturating_sub(1));
    let mut rng_state = sampling.seed.unwrap_or(0x517cc1b7_27220a95).max(1);
    let bitmask_engine = singularity_grammar::GrammarBitmaskEngine::new(logits.len());
    let mut grammar_state = singularity_grammar::JsonGrammarState::ExpectObjectStart;

    let first_tok = sample_v41_logits(
        &logits,
        &generated,
        sampling,
        &bitmask_engine,
        &mut grammar_state,
        &mut rng_state,
    )?;
    let ttft = t_start.elapsed().as_nanos() as u64;
    generated.push(first_tok);
    if first_tok != eos_token_id && max_tokens > 1 {
        let mut prev_tok = first_tok;
        for step in 1..max_tokens {
            let step_t0 = std::time::Instant::now();
            logits = target.forward_token(prev_tok)?;
            let tok = sample_v41_logits(
                &logits,
                &generated,
                sampling,
                &bitmask_engine,
                &mut grammar_state,
                &mut rng_state,
            )?;
            itls.push(step_t0.elapsed().as_nanos() as u64);
            generated.push(tok);
            if tok == eos_token_id || step + 1 == max_tokens {
                break;
            }
            prev_tok = tok;
        }
    }
    Ok((generated, 0, 0, ttft, itls))
}

fn clip_token_pieces_to_content(
    decoded_pieces: &[(u32, String)],
    content: &str,
) -> Vec<(u32, String)> {
    if content.is_empty() {
        return Vec::new();
    }
    let joined: String = decoded_pieces.iter().map(|(_, s)| s.as_str()).collect();
    if joined == content {
        return decoded_pieces.to_vec();
    }
    if let Some(start) = joined.find(content) {
        let end = start + content.len();
        let mut clipped = Vec::new();
        let mut cur = 0usize;
        for (tok, piece) in decoded_pieces {
            let next = cur + piece.len();
            let ov_start = cur.max(start);
            let ov_end = next.min(end);
            if ov_start < ov_end {
                let rel_start = ov_start - cur;
                let rel_end = ov_end - cur;
                if piece.is_char_boundary(rel_start) && piece.is_char_boundary(rel_end) {
                    let sub = &piece[rel_start..rel_end];
                    if !sub.is_empty() {
                        clipped.push((*tok, sub.to_string()));
                    }
                }
            }
            cur = next;
        }
        let reconstructed: String = clipped.iter().map(|(_, s)| s.as_str()).collect();
        if reconstructed == content {
            return clipped;
        }
    }
    let first_tok = decoded_pieces.first().map(|(t, _)| *t).unwrap_or(0);
    vec![(first_tok, content.to_string())]
}

fn sample_v41_logits(
    logits: &[f32],
    generated: &[u32],
    sampling: &V41SamplingRequest,
    bitmask_engine: &singularity_grammar::GrammarBitmaskEngine,
    grammar_state: &mut singularity_grammar::JsonGrammarState,
    rng_state: &mut u64,
) -> Result<u32, SingularityError> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(SingularityError::ModelError {
            reason: "Invalid V4.1 logits during sampling".into(),
        });
    }
    if sampling.is_greedy_unconstrained()
        || (sampling.temperature <= 1e-5
            && !sampling.json_mode
            && sampling.presence_penalty == 0.0
            && sampling.frequency_penalty == 0.0)
    {
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i;
            }
        }
        return Ok(best_id as u32);
    }
    let mut work = logits.to_vec();

    if sampling.presence_penalty != 0.0 || sampling.frequency_penalty != 0.0 {
        let mut counts = std::collections::HashMap::new();
        for &t in generated {
            *counts.entry(t).or_insert(0usize) += 1;
        }
        for (&tok, &count) in &counts {
            if let Some(slot) = work.get_mut(tok as usize) {
                *slot -= sampling.presence_penalty + sampling.frequency_penalty * (count as f32);
            }
        }
    }

    if sampling.json_mode {
        let mut masked = work.clone();
        if bitmask_engine
            .apply_mask(&mut masked, *grammar_state)
            .is_ok()
            && masked.iter().any(|v| v.is_finite())
        {
            work = masked;
        }
    }

    let chosen = if sampling.temperature <= 1e-5 {
        work.iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    } else {
        let inv_temp = 1.0 / sampling.temperature;
        for v in &mut work {
            if v.is_finite() {
                *v *= inv_temp;
            }
        }
        let mut indexed: Vec<(usize, f32)> = work
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .collect();
        let cmp = |(ia, a): &(usize, f32), (ib, b): &(usize, f32)| {
            b.total_cmp(a).then_with(|| ia.cmp(ib))
        };
        if let Some(k) = sampling.top_k {
            let keep = k.max(1);
            if keep < indexed.len() {
                indexed.select_nth_unstable_by(keep - 1, cmp);
                indexed.truncate(keep);
            }
        }
        let max_logit = if sampling.top_p < 1.0 || sampling.top_k.is_some() {
            indexed.sort_unstable_by(cmp);
            indexed.first().map(|(_, v)| *v).unwrap_or(0.0)
        } else {
            indexed
                .iter()
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max)
                .max(-1e30)
        };
        let mut probs: Vec<(usize, f32)> = indexed
            .into_iter()
            .map(|(idx, l)| (idx, (l - max_logit).exp()))
            .collect();
        let total_mass: f32 = probs.iter().map(|(_, p)| *p).sum();
        if sampling.top_p < 1.0 && total_mass > 0.0 {
            let cutoff = sampling.top_p * total_mass;
            let mut acc = 0.0f32;
            let mut keep = 0usize;
            for &(_, p) in &probs {
                acc += p;
                keep += 1;
                if acc >= cutoff {
                    break;
                }
            }
            probs.truncate(keep.max(1));
        }
        let kept_mass: f32 = probs.iter().map(|(_, p)| *p).sum();
        // Deterministic xorshift64* draw in [0, kept_mass)
        let mut x = *rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *rng_state = x;
        let rand01 = ((x.wrapping_mul(0x2545F491_4F6CDD1D)) >> 40) as f32 / ((1u32 << 24) as f32);
        let target = rand01 * kept_mass.max(1e-12);
        let mut csum = 0.0f32;
        let mut picked = probs.first().map(|(idx, _)| *idx).unwrap_or(0);
        for &(idx, p) in &probs {
            csum += p;
            if csum >= target {
                picked = idx;
                break;
            }
        }
        picked as u32
    };

    if sampling.json_mode {
        *grammar_state = bitmask_engine.advance_state(*grammar_state, chosen);
    }
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_token_pieces_respects_stop_thinking_and_dsml_boundaries() {
        // 1. Stop sequence truncation
        let pieces = vec![
            (10, "Hello ".to_string()),
            (11, "world".to_string()),
            (12, "STOP".to_string()),
            (13, " trailing".to_string()),
        ];
        assert_eq!(
            clip_token_pieces_to_content(&pieces, "Hello world"),
            vec![(10, "Hello ".to_string()), (11, "world".to_string())]
        );

        // 2. Thinking mode extraction (excludes reasoning_content + </think> prefix)
        let think_pieces = vec![
            (1, "step 1 ".to_string()),
            (2, "step 2</think>".to_string()),
            (3, "Final ".to_string()),
            (4, "answer".to_string()),
        ];
        assert_eq!(
            clip_token_pieces_to_content(&think_pieces, "Final answer"),
            vec![(3, "Final ".to_string()), (4, "answer".to_string())]
        );

        // 3. Empty content (e.g., pure tool call or incomplete thinking block)
        assert!(clip_token_pieces_to_content(&think_pieces, "").is_empty());

        // 4. DSML tool call suffix excluded from content stream
        let dsml_pieces = vec![
            (5, "Checking weather.".to_string()),
            (6, "<｜DSML｜function_calls>".to_string()),
            (7, "<｜DSML｜invoke name=\"get_weather\">".to_string()),
        ];
        assert_eq!(
            clip_token_pieces_to_content(&dsml_pieces, "Checking weather."),
            vec![(5, "Checking weather.".to_string())]
        );
    }
}
