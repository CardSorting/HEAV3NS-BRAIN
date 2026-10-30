use crate::admission::AdmissionLifecycle;
use crate::app::AppState;
use crate::ring_buffer::StreamToken;
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
use singularity_core::{SamplingParams, Sequence, SequenceId};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

static SEQ_ID_GEN: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
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
    pub role: Option<String>,
    pub content: Option<String>,
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

    Json(serde_json::json!({
        "status": "healthy",
        "circuit_breaker": format!("{:?}", cb_state),
        "free_blocks": free_blocks,
        "total_blocks": total_blocks,
        "active_sequences": state.scheduler.running_count(),
        "waiting_sequences": state.scheduler.waiting_count(),
    }))
}

pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snapshot = state.metrics.snapshot();
    Json(snapshot)
}

pub async fn models_handler() -> impl IntoResponse {
    let models = ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: "singularity-host-development".to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "singularity".to_string(),
        }],
    };
    Json(models)
}

pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    if req.model != "singularity-host-development" {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Model '{}' is not available", req.model),
        ));
    }
    // 1. Check Circuit Breaker
    state
        .circuit_breaker
        .check_permission()
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;

    // 2. Check Rate Governor
    state
        .rate_governor
        .try_acquire(1.0)
        .map_err(|e| (StatusCode::TOO_MANY_REQUESTS, e.to_string()))?;

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

    state.metrics.on_request_admitted(prompt_len);

    let rx = state.scheduler.subscribe(seq_id);

    state
        .scheduler
        .submit_sequence(sequence)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let _ = lifecycle.transition(crate::admission::AdmissionState::Admitted);

    let is_stream = req.stream.unwrap_or(false);

    // Check for Last-Event-ID header for ring-buffer reconnection replay
    let last_event_id: Option<u64> = headers
        .get("last-event-id")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse().ok());

    if is_stream {
        let stream = create_sse_stream(
            state.clone(),
            seq_id,
            rx,
            lifecycle,
            req.model,
            last_event_id,
        );
        Ok(Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(state.config.server.sse_keepalive_secs)),
            )
            .into_response())
    } else {
        let response =
            execute_non_streaming(state.clone(), seq_id, rx, lifecycle, req.model, prompt_len)
                .await?;
        Ok(Json(response).into_response())
    }
}

async fn execute_non_streaming(
    state: Arc<AppState>,
    _seq_id: SequenceId,
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

    let timeout = Duration::from_secs(30);
    let start = Instant::now();

    loop {
        let _ = state.scheduler.step_cooperative();

        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(singularity_scheduler::SequenceOutputEvent::Token(tok)) => {
                        generated_tokens.push(tok);
                    }
                    Some(singularity_scheduler::SequenceOutputEvent::Finished(seq)) => {
                        completed_sequence = Some(seq);
                        break;
                    }
                    Some(singularity_scheduler::SequenceOutputEvent::Cancelled) => {
                        break;
                    }
                    None => break,
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(2)) => {
                if start.elapsed() > timeout {
                    break;
                }
            }
        }
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

        // If client reconnected with Last-Event-ID, replay buffered tokens for this stream first
        if let Some(last_id) = last_event_id {
            let replayed = state.ring_buffer.replay_since_for_stream(seq_id, last_id);
            for tok in replayed {
                let chunk = ChatCompletionChunk {
                    id: completion_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(tok.text.clone()),
                        },
                        finish_reason: if tok.is_finish { Some("stop".to_string()) } else { None },
                    }],
                };
                if let Ok(json) = serde_json::to_string(&chunk) {
                    yield Ok(Event::default().id(tok.seq_no.to_string()).data(json));
                }
            }
        }

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
                },
                finish_reason: None,
            }],
        };
        token_seq_no += 1;
        if let Ok(json) = serde_json::to_string(&initial_chunk) {
            yield Ok(Event::default().id(token_seq_no.to_string()).data(json));
        }

        let mut total_generated = 0;
        let timeout = Duration::from_secs(30);
        let start = Instant::now();

        loop {
            let _ = state.scheduler.step_cooperative();

            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Some(singularity_scheduler::SequenceOutputEvent::Token(tok)) => {
                            total_generated += 1;
                            token_seq_no += 1;
                            let text = state
                                .tokenizer
                                .decode(&[tok])
                                .unwrap_or_else(|_| format!(" token_{tok}"));

                            state.ring_buffer.push_for_stream(seq_id, StreamToken {
                                seq_no: token_seq_no,
                                token: tok,
                                text: text.clone(),
                                is_finish: false,
                            });

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
                                    },
                                    finish_reason: None,
                                }],
                            };

                            if let Ok(json) = serde_json::to_string(&chunk) {
                                yield Ok(Event::default().id(token_seq_no.to_string()).data(json));
                            }
                        }
                        Some(singularity_scheduler::SequenceOutputEvent::Finished(_)) => {
                            token_seq_no += 1;

                            state.ring_buffer.push_for_stream(seq_id, StreamToken {
                                seq_no: token_seq_no,
                                token: 0,
                                text: "".to_string(),
                                is_finish: true,
                            });

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
                                    },
                                    finish_reason: Some("stop".to_string()),
                                }],
                            };

                            if let Ok(json) = serde_json::to_string(&final_chunk) {
                                yield Ok(Event::default().id(token_seq_no.to_string()).data(json));
                            }
                            yield Ok(Event::default().data("[DONE]"));
                            break;
                        }
                        Some(singularity_scheduler::SequenceOutputEvent::Cancelled) => {
                            break;
                        }
                        None => break,
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(2)) => {
                    if start.elapsed() > timeout {
                        break;
                    }
                }
            }
        }

        state.metrics.on_request_completed(total_generated);
        state.circuit_breaker.record_success();
        let _ = lifecycle.transition(crate::admission::AdmissionState::Completed);
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
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
        };
        let result =
            chat_completions_handler(State(state.clone()), HeaderMap::new(), Json(request)).await;
        assert!(matches!(result, Err((StatusCode::NOT_FOUND, _))));
        assert_eq!(state.scheduler.waiting_count(), 0);
    }
}
