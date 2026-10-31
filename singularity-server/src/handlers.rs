use crate::admission::AdmissionLifecycle;
use crate::app::AppState;
use crate::ring_buffer::StreamToken;
use crate::v41_engine::{is_v41_model_id, V41SamplingRequest, V41ServingEngine, V41_MODEL_ID};
use async_stream::stream;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use singularity_core::{SamplingParams, Sequence, SequenceId};
use singularity_model::{
    v41_openai::{CallInput, FunctionCall, MessageInput, ToolDefinition},
    v41_prompt::ThinkingMode,
};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

static SEQ_ID_GEN: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCallFunctionDto {
    pub name: String,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCallDto {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunctionDto,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolDefinitionDto {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: serde_json::Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseFormatDto {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormatDto>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinitionDto>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dspark: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormatDto>,
    pub max_tokens: Option<usize>,
    pub stop: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dspark: Option<bool>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

pub async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cb_state = state.circuit_breaker.current_state();
    let free_blocks = state.block_manager.free_block_count();
    let total_blocks = state.block_manager.total_blocks();
    let readiness = state.readiness();

    Json(serde_json::json!({
        "status": "healthy",
        "ready": readiness.is_ready(),
        "circuit_breaker": format!("{:?}", cb_state),
        "free_blocks": free_blocks,
        "total_blocks": total_blocks,
        "active_sequences": state.scheduler.running_count(),
        "waiting_sequences": state.scheduler.waiting_count(),
        "checks": readiness,
    }))
}

pub async fn ready_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let readiness = state.readiness();
    let cb_ok = state.circuit_breaker.check_permission().is_ok();
    let is_ready = readiness.is_ready() && cb_ok;
    let status_code = if is_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let served_model = if state.v41_engine().is_some() {
        V41_MODEL_ID
    } else {
        "singularity-host-development"
    };
    (
        status_code,
        Json(serde_json::json!({
            "status": if is_ready { "ready" } else { "not_ready" },
            "ready": is_ready,
            "model": served_model,
            "container_alive": readiness.container_alive,
            "runtime_initialized": readiness.runtime_initialized,
            "cuda_operational": readiness.cuda_operational,
            "gpus_available": readiness.gpus_available,
            "distributed_topology_operational": readiness.distributed_topology_operational,
            "checkpoint_present": readiness.checkpoint_present,
            "model_loaded": readiness.model_loaded,
            "model_ready": readiness.model_ready,
            "inference_executed": readiness.inference_executed,
            "checks": readiness,
        })),
    )
}

pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snapshot = state.metrics.snapshot();
    Json(snapshot)
}

pub async fn models_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ModelListResponse>, (StatusCode, String)> {
    state.check_auth(&headers)?;
    let mut data = vec![ModelCard {
        id: "singularity-host-development".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "singularity".to_string(),
    }];
    if state.v41_engine().is_some() {
        data.push(ModelCard {
            id: V41_MODEL_ID.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "deepseek-ai".to_string(),
        });
    }
    let models = ModelListResponse {
        object: "list".to_string(),
        data,
    };
    Ok(Json(models))
}

fn build_v41_sampling_request(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    response_format: Option<&ResponseFormatDto>,
) -> Result<V41SamplingRequest, (StatusCode, String)> {
    let json_mode = match response_format.map(|rf| rf.kind.as_str()) {
        None | Some("text") => false,
        Some("json_object") | Some("json_schema") => true,
        Some(other) => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Unsupported response_format type '{other}'"),
            ))
        }
    };
    Ok(V41SamplingRequest {
        temperature: temperature.unwrap_or(0.0),
        top_p: top_p.unwrap_or(1.0),
        top_k,
        presence_penalty: presence_penalty.unwrap_or(0.0),
        frequency_penalty: frequency_penalty.unwrap_or(0.0),
        seed,
        json_mode,
    })
}

pub async fn completions_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    state.check_auth(&headers)?;
    tracing::info!(
        model = %req.model,
        max_tokens = req.max_tokens.unwrap_or(128),
        "POST /v1/completions received"
    );
    if is_v41_model_id(&req.model) {
        let Some(engine) = state.v41_engine() else {
            tracing::warn!(model = %req.model, "POST /v1/completions rejected: V4.1 engine not loaded");
            return Err((
                StatusCode::NOT_FOUND,
                format!("Model '{}' is not available", req.model),
            ));
        };
        state.circuit_breaker.check_permission().map_err(|e| {
            state.metrics.on_request_rejected();
            (StatusCode::SERVICE_UNAVAILABLE, e.to_string())
        })?;
        state.rate_governor.try_acquire(1.0).map_err(|e| {
            state.metrics.on_request_rejected();
            (StatusCode::TOO_MANY_REQUESTS, e.to_string())
        })?;

        let sampling = build_v41_sampling_request(
            req.temperature,
            req.top_p,
            req.top_k,
            req.presence_penalty,
            req.frequency_penalty,
            req.seed,
            req.response_format.as_ref(),
        )?;
        let max_tokens = req.max_tokens.unwrap_or(128);
        let stops = req.stop.unwrap_or_default();
        let prompt = req.prompt.clone();
        let dspark = req.dspark;
        state.metrics.on_request_admitted(0);
        let task_res = tokio::task::spawn_blocking(move || {
            engine.generate_raw_with_options(&prompt, max_tokens, &stops, dspark, &sampling)
        })
        .await;
        let gen = match task_res {
            Ok(Ok(gen)) => gen,
            Ok(Err(e)) => {
                state.metrics.on_request_aborted();
                tracing::error!(error = %e, "POST /v1/completions failed");
                return Err((StatusCode::BAD_REQUEST, e.to_string()));
            }
            Err(e) => {
                state.metrics.on_request_aborted();
                state.circuit_breaker.record_failure();
                tracing::error!(error = %e, "POST /v1/completions worker task panicked");
                return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
            }
        };
        state
            .metrics
            .total_prompt_tokens
            .fetch_add(gen.prompt_tokens as u64, Ordering::Relaxed);
        state.metrics.record_ttft(gen.ttft_ns);
        for &itl in &gen.itl_ns_list {
            state.metrics.record_itl(itl);
        }
        state.metrics.on_request_completed(gen.completion_tokens);
        state.circuit_breaker.record_success();
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        return Ok(Json(CompletionResponse {
            id: format!("cmpl-{}", Uuid::new_v4()),
            object: "text_completion".to_string(),
            created,
            model: req.model,
            choices: vec![CompletionChoice {
                index: 0,
                text: gen.raw_text,
                finish_reason: Some(gen.finish_reason),
            }],
            usage: Usage {
                prompt_tokens: gen.prompt_tokens,
                completion_tokens: gen.completion_tokens,
                total_tokens: gen.prompt_tokens + gen.completion_tokens,
            },
        }));
    }

    if req.model != "singularity-host-development" {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Model '{}' is not available", req.model),
        ));
    }
    if req.stop.as_ref().is_some_and(|s| !s.is_empty()) {
        return Err((
            StatusCode::BAD_REQUEST,
            "The development model does not support stop strings".into(),
        ));
    }

    let chat_req = ChatCompletionRequest {
        model: req.model.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: req.prompt,
            ..Default::default()
        }],
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        response_format: req.response_format,
        max_tokens: req.max_tokens,
        stream: Some(false),
        stop: None,
        tools: None,
        thinking: None,
        reasoning_effort: None,
        dspark: req.dspark,
    };
    let resp = execute_host_development_chat(state, HeaderMap::new(), chat_req).await?;
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let chat_resp: ChatCompletionResponse = serde_json::from_slice(&bytes)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let text = chat_resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();
    Ok(Json(CompletionResponse {
        id: chat_resp.id.replacen("chatcmpl-", "cmpl-", 1),
        object: "text_completion".to_string(),
        created: chat_resp.created,
        model: chat_resp.model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: Some("stop".to_string()),
        }],
        usage: chat_resp.usage,
    }))
}

pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    state.check_auth(&headers)?;
    tracing::info!(
        model = %req.model,
        messages = req.messages.len(),
        max_tokens = req.max_tokens.unwrap_or(128),
        stream = req.stream.unwrap_or(false),
        "POST /v1/chat/completions received"
    );

    if is_v41_model_id(&req.model) {
        let Some(engine) = state.v41_engine() else {
            tracing::warn!(model = %req.model, "POST /v1/chat/completions rejected: V4.1 engine not loaded");
            return Err((
                StatusCode::NOT_FOUND,
                format!("Model '{}' is not available", req.model),
            ));
        };
        return execute_v41_chat(state, engine, headers, req).await;
    }

    if req.model != "singularity-host-development" {
        tracing::warn!(model = %req.model, "POST /v1/chat/completions rejected: unknown model");
        return Err((
            StatusCode::NOT_FOUND,
            format!("Model '{}' is not available", req.model),
        ));
    }
    execute_host_development_chat(state, headers, req).await
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash.max(1)
}

fn derive_stream_id(headers: &HeaderMap, model: &str, messages: &[ChatMessage]) -> u64 {
    if let Some(raw) = headers.get("x-stream-id").and_then(|h| h.to_str().ok()) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            if let Ok(parsed) = trimmed.parse::<u64>() {
                return parsed.max(1);
            }
            return fnv1a64(trimmed.as_bytes());
        }
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in model.as_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    for msg in messages {
        for &b in msg.role.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        for &b in msg.content.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash.max(1)
}

fn attach_stream_id_header(mut resp: Response, stream_id: u64) -> Response {
    if let Ok(val) = axum::http::HeaderValue::from_str(&stream_id.to_string()) {
        resp.headers_mut().insert("x-stream-id", val);
    }
    resp
}

fn build_replay_sse_response(
    state: Arc<AppState>,
    stream_id: u64,
    last_id: u64,
    model_name: String,
) -> Response {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let replayed = state.ring_buffer.replay_since_for_stream(stream_id, last_id);
    let sse_stream = stream! {
        for tok in replayed {
            let chunk = ChatCompletionChunk {
                id: completion_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: if tok.is_finish { None } else { Some(tok.text) },
                        reasoning_content: None,
                        tool_calls: Vec::new(),
                    },
                    finish_reason: if tok.is_finish { Some("stop".to_string()) } else { None },
                }],
            };
            if let Ok(json) = serde_json::to_string(&chunk) {
                yield Ok::<Event, Infallible>(Event::default().id(tok.seq_no.to_string()).data(json));
            }
        }
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    };
    let resp = Sse::new(sse_stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(state.config.server.sse_keepalive_secs)),
        )
        .into_response();
    attach_stream_id_header(resp, stream_id)
}

async fn execute_v41_chat(
    state: Arc<AppState>,
    engine: Arc<V41ServingEngine>,
    headers: HeaderMap,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    state.circuit_breaker.check_permission().map_err(|e| {
        state.metrics.on_request_rejected();
        (StatusCode::SERVICE_UNAVAILABLE, e.to_string())
    })?;
    state.rate_governor.try_acquire(1.0).map_err(|e| {
        state.metrics.on_request_rejected();
        (StatusCode::TOO_MANY_REQUESTS, e.to_string())
    })?;

    let is_stream = req.stream.unwrap_or(false);
    let stream_id = derive_stream_id(&headers, &req.model, &req.messages);
    let last_event_id: Option<u64> = headers
        .get("last-event-id")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse().ok());

    if is_stream {
        if let Some(last_id) = last_event_id {
            if state.ring_buffer.has_stream(stream_id) {
                return Ok(build_replay_sse_response(
                    state,
                    stream_id,
                    last_id,
                    req.model,
                ));
            }
        }
    }

    let mode = match req.thinking.as_deref().unwrap_or("chat") {
        "chat" | "nothink" | "disabled" | "false" => ThinkingMode::Chat,
        "thinking" | "think" | "enabled" | "true" | "interleaved" | "auto" => {
            ThinkingMode::Thinking
        }
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Unsupported V4.1 thinking mode '{other}'"),
            ))
        }
    };
    let effort = req.reasoning_effort.unwrap_or(50);
    let sampling = build_v41_sampling_request(
        req.temperature,
        req.top_p,
        req.top_k,
        req.presence_penalty,
        req.frequency_penalty,
        req.seed,
        req.response_format.as_ref(),
    )?;

    let messages: Vec<MessageInput> = req
        .messages
        .iter()
        .map(|m| MessageInput {
            role: m.role.clone(),
            content: Some(m.content.clone()),
            reasoning_content: m.reasoning_content.clone(),
            tool_calls: m
                .tool_calls
                .iter()
                .map(|tc| CallInput {
                    id: tc.id.clone(),
                    kind: tc.kind.clone(),
                    function: FunctionCall {
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                        namespace: tc.function.namespace.clone(),
                    },
                    namespace: tc.namespace.clone(),
                })
                .collect(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect();

    let tools: Vec<ToolDefinition> = req
        .tools
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|t| ToolDefinition {
            kind: t.kind.clone(),
            function: t.function.clone(),
            namespace: t.namespace.clone(),
        })
        .collect();

    let max_tokens = req.max_tokens.unwrap_or(128);
    let stops = req.stop.clone().unwrap_or_default();
    let dspark = req.dspark;

    state.metrics.on_request_admitted(0);
    let task_res = tokio::task::spawn_blocking(move || {
        engine.generate_chat_with_options(
            &messages,
            &tools,
            mode,
            effort,
            max_tokens,
            &stops,
            dspark,
            &sampling,
        )
    })
    .await;
    let gen = match task_res {
        Ok(Ok(gen)) => gen,
        Ok(Err(e)) => {
            state.metrics.on_request_aborted();
            tracing::error!(error = %e, "POST /v1/chat/completions failed");
            return Err((StatusCode::BAD_REQUEST, e.to_string()));
        }
        Err(e) => {
            state.metrics.on_request_aborted();
            state.circuit_breaker.record_failure();
            tracing::error!(error = %e, "POST /v1/chat/completions worker task panicked");
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };

    state
        .metrics
        .total_prompt_tokens
        .fetch_add(gen.prompt_tokens as u64, Ordering::Relaxed);
    state.metrics.record_ttft(gen.ttft_ns);
    for &itl in &gen.itl_ns_list {
        state.metrics.record_itl(itl);
    }
    state.metrics.on_request_completed(gen.completion_tokens);
    state.circuit_breaker.record_success();

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let model_name = req.model.clone();

    let tool_calls_dto: Vec<ToolCallDto> = gen
        .assistant
        .tool_calls
        .iter()
        .map(|tc| ToolCallDto {
            id: tc.id.clone(),
            kind: tc.kind.to_string(),
            function: ToolCallFunctionDto {
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
                namespace: None,
            },
            namespace: tc.namespace.clone(),
        })
        .collect();
    let reasoning_opt = if gen.assistant.reasoning_content.is_empty() {
        None
    } else {
        Some(gen.assistant.reasoning_content.clone())
    };

    if is_stream {
        let min_seq_no = last_event_id.unwrap_or(0);
        let finish_reason = gen.finish_reason.clone();
        let pieces = gen.token_pieces;
        let state_for_stream = state.clone();
        let sse_stream = stream! {
            let mut seq_no = 1u64;
            let init_chunk = ChatCompletionChunk {
                id: completion_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".to_string()),
                        content: Some(String::new()),
                        reasoning_content: reasoning_opt.clone(),
                        tool_calls: Vec::new(),
                    },
                    finish_reason: None,
                }],
            };
            if seq_no > min_seq_no {
                if let Ok(json) = serde_json::to_string(&init_chunk) {
                    yield Ok::<Event, Infallible>(Event::default().id(seq_no.to_string()).data(json));
                }
            }
            for (tok, text) in pieces {
                seq_no += 1;
                state_for_stream.ring_buffer.push_for_stream(
                    stream_id,
                    StreamToken {
                        seq_no,
                        token: tok,
                        text: text.clone(),
                        is_finish: false,
                    },
                );
                if seq_no > min_seq_no {
                    let chunk = ChatCompletionChunk {
                        id: completion_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(text),
                                reasoning_content: None,
                                tool_calls: Vec::new(),
                            },
                            finish_reason: None,
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&chunk) {
                        yield Ok(Event::default().id(seq_no.to_string()).data(json));
                    }
                }
            }
            seq_no += 1;
            state_for_stream.ring_buffer.push_for_stream(
                stream_id,
                StreamToken {
                    seq_no,
                    token: 0,
                    text: String::new(),
                    is_finish: true,
                },
            );
            if seq_no > min_seq_no {
                let final_chunk = ChatCompletionChunk {
                    id: completion_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                            reasoning_content: None,
                            tool_calls: tool_calls_dto,
                        },
                        finish_reason: Some(finish_reason),
                    }],
                };
                if let Ok(json) = serde_json::to_string(&final_chunk) {
                    yield Ok(Event::default().id(seq_no.to_string()).data(json));
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        let resp = Sse::new(sse_stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(state.config.server.sse_keepalive_secs)),
            )
            .into_response();
        return Ok(attach_stream_id_header(resp, stream_id));
    }

    let response = ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion".to_string(),
        created,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: gen.assistant.content,
                reasoning_content: reasoning_opt,
                tool_calls: tool_calls_dto,
                tool_call_id: None,
            },
            finish_reason: Some(gen.finish_reason),
        }],
        usage: Usage {
            prompt_tokens: gen.prompt_tokens,
            completion_tokens: gen.completion_tokens,
            total_tokens: gen.prompt_tokens + gen.completion_tokens,
        },
    };
    Ok(Json(response).into_response())
}

async fn execute_host_development_chat(
    state: Arc<AppState>,
    headers: HeaderMap,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    if req.tools.is_some()
        || req.thinking.is_some()
        || req.reasoning_effort.is_some()
        || req.dspark.is_some()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "The development model does not support tools, thinking, or DSpark parameters".into(),
        ));
    }

    if req.messages.iter().any(|message| {
        !matches!(message.role.as_str(), "system" | "user" | "assistant")
            || !message.tool_calls.is_empty()
            || message.tool_call_id.is_some()
            || message.reasoning_content.is_some()
    }) {
        return Err((
            StatusCode::BAD_REQUEST,
            "The development model supports only system, user, and assistant messages".into(),
        ));
    }
    if req.stop.as_ref().is_some_and(|stops| !stops.is_empty()) {
        return Err((
            StatusCode::BAD_REQUEST,
            "The development model does not support stop strings".into(),
        ));
    }
    // 1. Check Circuit Breaker
    state.circuit_breaker.check_permission().map_err(|e| {
        state.metrics.on_request_rejected();
        (StatusCode::SERVICE_UNAVAILABLE, e.to_string())
    })?;

    // 2. Check Rate Governor
    state.rate_governor.try_acquire(1.0).map_err(|e| {
        state.metrics.on_request_rejected();
        (StatusCode::TOO_MANY_REQUESTS, e.to_string())
    })?;

    let is_stream = req.stream.unwrap_or(false);
    let stream_id = derive_stream_id(&headers, &req.model, &req.messages);
    let last_event_id: Option<u64> = headers
        .get("last-event-id")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse().ok());

    if is_stream {
        if let Some(last_id) = last_event_id {
            if state.ring_buffer.has_stream(stream_id) {
                return Ok(build_replay_sse_response(
                    state,
                    stream_id,
                    last_id,
                    req.model,
                ));
            }
        }
    }

    let seq_id = SEQ_ID_GEN.fetch_add(1, Ordering::Relaxed);
    let lifecycle = Arc::new(AdmissionLifecycle::new(seq_id));
    let _ = lifecycle.transition(crate::admission::AdmissionState::Queued);

    // Format prompt text from messages
    let mut prompt_text = String::new();
    for msg in &req.messages {
        prompt_text.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    prompt_text.push_str("assistant: ");

    // Real tokenizer encoding
    let prompt_tokens: Vec<u32> = state
        .tokenizer
        .encode(&prompt_text)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let sampling_params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        max_tokens: req.max_tokens.unwrap_or(128),
        stop_tokens: vec![0, 1, 2],
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        grammar_constraint: None,
    };

    let sequence = Sequence::new(seq_id, prompt_text, prompt_tokens.clone(), sampling_params);
    let prompt_len = sequence.prompt_len();

    let rx = state.scheduler.subscribe(seq_id);

    state
        .scheduler
        .submit_sequence(sequence)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    state.metrics.on_request_admitted(prompt_len);
    let _ = lifecycle.transition(crate::admission::AdmissionState::Admitted);

    if is_stream {
        let stream = create_sse_stream(
            state.clone(),
            seq_id,
            stream_id,
            rx,
            lifecycle,
            req.model,
            last_event_id,
        );
        let resp = Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(state.config.server.sse_keepalive_secs)),
            )
            .into_response();
        Ok(attach_stream_id_header(resp, stream_id))
    } else {
        let response =
            execute_non_streaming(state.clone(), seq_id, rx, lifecycle, req.model, prompt_len)
                .await?;
        Ok(Json(response).into_response())
    }
}

async fn execute_non_streaming(
    state: Arc<AppState>,
    seq_id: SequenceId,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<singularity_scheduler::SequenceOutputEvent>,
    lifecycle: Arc<AdmissionLifecycle>,
    model: String,
    prompt_len: usize,
) -> Result<ChatCompletionResponse, (StatusCode, String)> {
    let _ = lifecycle.transition(crate::admission::AdmissionState::Dispatched);
    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut generated_tokens = Vec::new();
    let mut completed_sequence = None;
    let mut ttft_ns: Option<u64> = None;
    let mut last_token_instant: Option<Instant> = None;
    let mut was_aborted = false;

    let timeout = Duration::from_secs(30);
    let start = Instant::now();

    loop {
        let _ = state.scheduler.step_cooperative();

        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(singularity_scheduler::SequenceOutputEvent::Token(tok)) => {
                        let now = Instant::now();
                        if ttft_ns.is_none() {
                            ttft_ns = Some(start.elapsed().as_nanos() as u64);
                        } else if let Some(prev) = last_token_instant {
                            state.metrics.record_itl(now.duration_since(prev).as_nanos() as u64);
                        }
                        last_token_instant = Some(now);
                        generated_tokens.push(tok);
                    }
                    Some(singularity_scheduler::SequenceOutputEvent::Finished(seq)) => {
                        if ttft_ns.is_none() {
                            ttft_ns = Some(start.elapsed().as_nanos() as u64);
                        }
                        completed_sequence = Some(seq);
                        break;
                    }
                    Some(singularity_scheduler::SequenceOutputEvent::Cancelled) => {
                        was_aborted = true;
                        break;
                    }
                    None => break,
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(2)) => {
                if start.elapsed() > timeout {
                    state.scheduler.cancel_sequence(seq_id);
                    was_aborted = true;
                    break;
                }
            }
        }
    }

    if let Some(ttft) = ttft_ns {
        state.metrics.record_ttft(ttft);
    }
    if was_aborted {
        state.metrics.on_request_aborted();
        let _ = lifecycle.transition(crate::admission::AdmissionState::Cancelled);
        return Err((
            StatusCode::REQUEST_TIMEOUT,
            "Sequence execution cancelled or timed out".to_string(),
        ));
    }

    let completion_len = generated_tokens.len();
    state.metrics.on_request_completed(completion_len);
    state.circuit_breaker.record_success();
    let _ = lifecycle.transition(crate::admission::AdmissionState::Completed);

    let content = if let Some(seq) = completed_sequence {
        state
            .tokenizer
            .decode(&seq.output_tokens)
            .unwrap_or_else(|_| format!("Generated output: {} tokens", seq.output_len()))
    } else {
        state
            .tokenizer
            .decode(&generated_tokens)
            .unwrap_or_else(|_| format!("Generated output: {} tokens", generated_tokens.len()))
    };

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: start_time,
        model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    })
}

fn create_sse_stream(
    state: Arc<AppState>,
    seq_id: SequenceId,
    stream_id: u64,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<singularity_scheduler::SequenceOutputEvent>,
    lifecycle: Arc<AdmissionLifecycle>,
    model: String,
    last_event_id: Option<u64>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    stream! {
        let _ = lifecycle.transition(crate::admission::AdmissionState::Dispatched);
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let mut token_seq_no = 0u64;
        let min_seq_no = last_event_id.unwrap_or(0);

        // Send initial role chunk
        let initial_chunk = ChatCompletionChunk {
            id: completion_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: Some("".to_string()),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
            }],
        };
        token_seq_no += 1;
        if token_seq_no > min_seq_no {
            if let Ok(json) = serde_json::to_string(&initial_chunk) {
                yield Ok(Event::default().id(token_seq_no.to_string()).data(json));
            }
        }

        let mut total_generated = 0;
        let mut ttft_recorded = false;
        let mut last_token_instant: Option<Instant> = None;
        let mut was_aborted = false;
        let timeout = Duration::from_secs(30);
        let start = Instant::now();

        loop {
            let _ = state.scheduler.step_cooperative();

            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Some(singularity_scheduler::SequenceOutputEvent::Token(tok)) => {
                            let now = Instant::now();
                            if !ttft_recorded {
                                state.metrics.record_ttft(start.elapsed().as_nanos() as u64);
                                ttft_recorded = true;
                            } else if let Some(prev) = last_token_instant {
                                state.metrics.record_itl(now.duration_since(prev).as_nanos() as u64);
                            }
                            last_token_instant = Some(now);
                            total_generated += 1;
                            token_seq_no += 1;
                            let text = state
                                .tokenizer
                                .decode(&[tok])
                                .unwrap_or_else(|_| format!(" token_{tok}"));

                            state.ring_buffer.push_for_stream(stream_id, StreamToken {
                                seq_no: token_seq_no,
                                token: tok,
                                text: text.clone(),
                                is_finish: false,
                            });

                            if token_seq_no > min_seq_no {
                                let chunk = ChatCompletionChunk {
                                    id: completion_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: Some(text),
                                            reasoning_content: None,
                                            tool_calls: Vec::new(),
                                        },
                                        finish_reason: None,
                                    }],
                                };

                                if let Ok(json) = serde_json::to_string(&chunk) {
                                    yield Ok(Event::default().id(token_seq_no.to_string()).data(json));
                                }
                            }
                        }
                        Some(singularity_scheduler::SequenceOutputEvent::Finished(_)) => {
                            if !ttft_recorded {
                                state.metrics.record_ttft(start.elapsed().as_nanos() as u64);
                            }
                            token_seq_no += 1;

                            state.ring_buffer.push_for_stream(stream_id, StreamToken {
                                seq_no: token_seq_no,
                                token: 0,
                                text: "".to_string(),
                                is_finish: true,
                            });

                            if token_seq_no > min_seq_no {
                                let final_chunk = ChatCompletionChunk {
                                    id: completion_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: None,
                                            reasoning_content: None,
                                            tool_calls: Vec::new(),
                                        },
                                        finish_reason: Some("stop".to_string()),
                                    }],
                                };

                                if let Ok(json) = serde_json::to_string(&final_chunk) {
                                    yield Ok(Event::default().id(token_seq_no.to_string()).data(json));
                                }
                            }
                            yield Ok(Event::default().data("[DONE]"));
                            break;
                        }
                        Some(singularity_scheduler::SequenceOutputEvent::Cancelled) => {
                            was_aborted = true;
                            break;
                        }
                        None => break,
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(2)) => {
                    if start.elapsed() > timeout {
                        state.scheduler.cancel_sequence(seq_id);
                        was_aborted = true;
                        break;
                    }
                }
            }
        }

        if was_aborted {
            state.metrics.on_request_aborted();
            let _ = lifecycle.transition(crate::admission::AdmissionState::Cancelled);
        } else {
            state.metrics.on_request_completed(total_generated);
            state.circuit_breaker.record_success();
            let _ = lifecycle.transition(crate::admission::AdmissionState::Completed);
        }
    }
}

#[cfg(test)]
mod model_identity_tests {
    use super::*;

    #[tokio::test]
    async fn rejects_unavailable_checkpoint_before_scheduling() {
        let dir = tempfile::tempdir().unwrap();
        let state = AppState::new_with_mock(
            singularity_core::SystemConfig::default(),
            dir.path().to_path_buf(),
        )
        .unwrap();
        let request = ChatCompletionRequest {
            model: "deepseek-ai/DeepSeek-V4.1-Flash-1017".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "hello".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let result =
            chat_completions_handler(State(state.clone()), HeaderMap::new(), Json(request)).await;
        assert!(matches!(result, Err((StatusCode::NOT_FOUND, _))));
        assert_eq!(state.scheduler.waiting_count(), 0);
    }
}
