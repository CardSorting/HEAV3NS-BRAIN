use axum::http::StatusCode;
use singularity_tests::*;
use tempfile::tempdir;
use tokio::net::TcpListener;

#[tokio::test]
async fn test_e2e_server_endpoints() {
    let temp_dir = tempdir().unwrap();
    let config = SystemConfig::default();
    let state = AppState::new_with_mock(config, temp_dir.path().to_path_buf()).unwrap();
    let app = create_router(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // 1. Test /health
    let health_resp = client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health_resp.status(), StatusCode::OK);
    let health_body: serde_json::Value = health_resp.json().await.unwrap();
    assert_eq!(health_body["status"], "healthy");

    // 2. Test /v1/models
    let models_resp = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(models_resp.status(), StatusCode::OK);
    let models_body: serde_json::Value = models_resp.json().await.unwrap();
    assert_eq!(models_body["object"], "list");
    assert_eq!(models_body["data"][0]["id"], "singularity-host-development");

    let unavailable = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": "hello"}]
    });
    assert_eq!(
        client
            .post(format!("{base_url}/v1/chat/completions"))
            .json(&unavailable)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::NOT_FOUND
    );

    for body in [
        serde_json::json!({"model":"singularity-host-development", "messages":[{"role":"tool","content":"result"}]}),
        serde_json::json!({"model":"singularity-host-development", "messages":[{"role":"user","content":"hello"}], "stop":["end"]}),
    ] {
        assert_eq!(
            client
                .post(format!("{base_url}/v1/chat/completions"))
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    let unsupported = serde_json::json!({"model":"singularity-host-development", "messages":[{"role":"user","content":"hello"}], "tools":[]});
    assert!(client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&unsupported)
        .send()
        .await
        .unwrap()
        .status()
        .is_client_error());

    // 3. Test /v1/chat/completions (non-streaming)
    let req_body = serde_json::json!({
        "model": "singularity-host-development",
        "messages": [
            {"role": "user", "content": "Hello Singularity"}
        ],
        "max_tokens": 8,
        "stream": false
    });

    let chat_resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(chat_resp.status(), StatusCode::OK);
    let chat_json: serde_json::Value = chat_resp.json().await.unwrap();
    assert_eq!(chat_json["object"], "chat.completion");
    assert!(!chat_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .is_empty());

    // 4. Test /v1/chat/completions (streaming SSE)
    let stream_req_body = serde_json::json!({
        "model": "singularity-host-development",
        "messages": [
            {"role": "user", "content": "Stream tokens"}
        ],
        "max_tokens": 4,
        "stream": true
    });

    let sse_resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&stream_req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(sse_resp.status(), StatusCode::OK);
    let host_stream_id = sse_resp
        .headers()
        .get("x-stream-id")
        .and_then(|v| v.to_str().ok())
        .unwrap()
        .to_string();
    let text = sse_resp.text().await.unwrap();
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("[DONE]"));

    // 4b. Test Last-Event-ID reconnect on host-development stream
    let reconnect_resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .header("x-stream-id", &host_stream_id)
        .header("last-event-id", "2")
        .json(&stream_req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(reconnect_resp.status(), StatusCode::OK);
    let reconnect_text = reconnect_resp.text().await.unwrap();
    assert!(!reconnect_text.contains("\nid: 1\n") && !reconnect_text.starts_with("id: 1\n"));
    assert!(!reconnect_text.contains("\nid: 2\n") && !reconnect_text.starts_with("id: 2\n"));
    assert!(reconnect_text.contains("id: 3\n"));
    assert!(reconnect_text.contains("[DONE]"));

    // 5. Test /metrics (including host-development TTFT/ITL recording)
    let metrics_resp = client
        .get(format!("{base_url}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(metrics_resp.status(), StatusCode::OK);
    let metrics_body: serde_json::Value = metrics_resp.json().await.unwrap();
    assert!(metrics_body["total_requests"].as_u64().unwrap() >= 2);
    assert!(metrics_body["avg_ttft_ms"].as_f64().unwrap() > 0.0);
    assert!(metrics_body["optimal_runtime_score"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn test_e2e_v41_serving_readiness_and_auth() {
    let dir = tempfile::tempdir().unwrap();
    let config = SystemConfig::default();
    let state = AppState::new_with_mock(config, dir.path().to_path_buf()).unwrap();

    // Build a miniature V4.1 checkpoint to attach V41ServingEngine
    let mut v41_cfg: singularity_model::v41_config::V41Config = serde_json::from_str(include_str!(
        "../../singularity-model/tests/fixtures/deepseek-v41-flash-1017-config.json"
    ))
    .unwrap();
    v41_cfg.image_token_id = 1;
    let c = &mut v41_cfg.text_config;
    c.vocab_size = 2;
    c.hidden_size = 32;
    c.num_hidden_layers = 1;
    c.num_nextn_predict_layers = 0;
    c.compress_ratios = vec![0];
    c.kv_source_layer_ids.clear();
    c.index_source_layer_ids.clear();
    c.candidate_source_layer_id = None;
    c.engram_layer_ids.clear();
    c.hc_mult = 2;
    c.num_attention_heads = 1;
    c.head_dim = 32;
    c.q_lora_rank = 32;
    c.o_lora_rank = 32;
    c.o_groups = 1;
    c.sliding_window = 16;
    c.qk_rope_head_dim = 2;
    c.n_routed_experts = 1;
    c.num_experts_per_tok = 1;
    c.moe_intermediate_size = 32;
    c.dspark_target_layer_ids = vec![0];
    v41_cfg.vision_config.num_hidden_layers = 0;

    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut add = |name: String, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
        let start = data.len();
        data.extend(bytes);
        header.insert(
            name,
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
        );
    };
    for name in ["embed.weight", "head.weight"] {
        add(name.into(), "BF16", vec![2, 32], [0x80, 0x3f].repeat(64));
    }
    add(
        "norm.weight".into(),
        "BF16",
        vec![32],
        [0x80, 0x3f].repeat(32),
    );
    for hc in ["hc_attn", "hc_ffn"] {
        add(
            format!("layers.0.{hc}_fn"),
            "F32",
            vec![8, 64],
            vec![0; 2048],
        );
        add(format!("layers.0.{hc}_base"), "F32", vec![8], vec![0; 32]);
        add(format!("layers.0.{hc}_scale"), "F32", vec![3], vec![0; 12]);
    }
    for norm in ["attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"] {
        add(
            format!("layers.0.{norm}.weight"),
            "BF16",
            vec![32],
            [0x80, 0x3f].repeat(32),
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
        [0x80, 0x3f].repeat(32),
    );
    add("layers.0.ffn.gate.bias".into(), "F32", vec![1], vec![0; 4]);
    for (name, dtype, cols, value, scale_rows) in [
        ("experts.0", "I8", 16, 0x11, 32),
        ("shared_experts", "F8_E4M3", 32, 0x20, 1),
    ] {
        for mat in ["w1", "w2", "w3"] {
            add(
                format!("layers.0.ffn.{name}.{mat}.weight"),
                dtype,
                vec![32, cols],
                vec![value; 32 * cols],
            );
            add(
                format!("layers.0.ffn.{name}.{mat}.scale"),
                "F8_E8M0",
                vec![scale_rows, 1],
                vec![127; scale_rows],
            );
        }
    }
    let h_bytes = serde_json::to_vec(&header).unwrap();
    let mut bytes = (h_bytes.len() as u64).to_le_bytes().to_vec();
    bytes.extend(h_bytes);
    bytes.extend(data);
    let ckpt_path = dir.path().join("v41_tiny.safetensors");
    std::fs::write(&ckpt_path, bytes).unwrap();
    let loader =
        std::sync::Arc::new(singularity_model::SafetensorsMmapLoader::open(&ckpt_path).unwrap());
    let engine = singularity_server::V41ServingEngine::from_loaded(
        loader,
        v41_cfg,
        std::sync::Arc::new(singularity_model::TokenizerWrapper::new_fallback()),
        None,
        0,
        99,
        false,
    )
    .unwrap();
    assert_eq!(engine.execution_count(), 1); // warm-up pass ran once
    state.set_v41_engine(engine.clone());
    state.set_api_key(Some("secret-token-123".to_string()));

    let app = create_router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // 1. /ready returns 200 when ready, 503 when unready
    let ready_resp = client
        .get(format!("{base_url}/ready"))
        .send()
        .await
        .unwrap();
    assert_eq!(ready_resp.status(), StatusCode::OK);
    let ready_json: serde_json::Value = ready_resp.json().await.unwrap();
    assert_eq!(ready_json["ready"], true);
    assert_eq!(ready_json["model"], "deepseek-ai/DeepSeek-V4.1-Flash-1017");
    assert_eq!(ready_json["inference_executed"], true);

    state.set_readiness(singularity_server::ReadinessState::default_unready());
    let unready_resp = client
        .get(format!("{base_url}/ready"))
        .send()
        .await
        .unwrap();
    assert_eq!(unready_resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    state.set_readiness(singularity_server::ReadinessState::default_ready());

    // 2. Auth enforcement on /v1/models
    let unauth_resp = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(unauth_resp.status(), StatusCode::UNAUTHORIZED);

    let models_resp = client
        .get(format!("{base_url}/v1/models"))
        .bearer_auth("secret-token-123")
        .send()
        .await
        .unwrap();
    assert_eq!(models_resp.status(), StatusCode::OK);
    let models_json: serde_json::Value = models_resp.json().await.unwrap();
    assert_eq!(models_json["data"].as_array().unwrap().len(), 2);
    assert_eq!(
        models_json["data"][1]["id"],
        "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    );

    // 3. V4.1 Chat Completion (non-streaming and streaming)
    let v41_req = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [
            {"role": "system", "content": "You are V4.1."},
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 3,
        "stream": false,
        "dspark": true
    });
    let v41_resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("secret-token-123")
        .json(&v41_req)
        .send()
        .await
        .unwrap();
    assert_eq!(v41_resp.status(), StatusCode::OK);
    let v41_body: serde_json::Value = v41_resp.json().await.unwrap();
    assert_eq!(v41_body["model"], "deepseek-ai/DeepSeek-V4.1-Flash-1017");
    assert_eq!(v41_body["usage"]["completion_tokens"], 3);

    let v41_stream_req = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": "Stream A"}],
        "max_tokens": 3,
        "stream": true,
        "dspark": false
    });
    let v41_sse = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("secret-token-123")
        .json(&v41_stream_req)
        .send()
        .await
        .unwrap();
    assert_eq!(v41_sse.status(), StatusCode::OK);
    let stream_a_id = v41_sse
        .headers()
        .get("x-stream-id")
        .and_then(|v| v.to_str().ok())
        .unwrap()
        .to_string();
    let sse_text = v41_sse.text().await.unwrap();
    assert!(sse_text.contains("[DONE]"));

    // Interleave a second independent stream (Stream B) with a different token count
    let v41_stream_b_req = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": "Stream B"}],
        "max_tokens": 1,
        "stream": true,
        "dspark": false
    });
    let v41_sse_b = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("secret-token-123")
        .json(&v41_stream_b_req)
        .send()
        .await
        .unwrap();
    assert_eq!(v41_sse_b.status(), StatusCode::OK);
    let stream_b_id = v41_sse_b
        .headers()
        .get("x-stream-id")
        .and_then(|v| v.to_str().ok())
        .unwrap()
        .to_string();
    assert_ne!(stream_a_id, stream_b_id);
    let _ = v41_sse_b.text().await.unwrap();

    // Reconnect Stream A from Last-Event-ID: 2 (without x-stream-id header: payload hash must resolve Stream A!)
    let exec_before_reconnect = engine.execution_count();
    let v41_reconnect = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("secret-token-123")
        .header("last-event-id", "2")
        .json(&v41_stream_req)
        .send()
        .await
        .unwrap();
    assert_eq!(v41_reconnect.status(), StatusCode::OK);
    let reconnect_sse = v41_reconnect.text().await.unwrap();
    // Reconnect must NOT re-run the model, must NOT re-emit id 1 or 2, and must replay id 3, 4, 5 + [DONE]
    assert_eq!(engine.execution_count(), exec_before_reconnect);
    assert!(!reconnect_sse.contains("\nid: 1\n") && !reconnect_sse.starts_with("id: 1\n"));
    assert!(!reconnect_sse.contains("\nid: 2\n") && !reconnect_sse.starts_with("id: 2\n"));
    assert!(reconnect_sse.contains("id: 3\n"));
    assert!(reconnect_sse.contains("id: 4\n"));
    assert!(reconnect_sse.contains("id: 5\n"));
    assert!(reconnect_sse.contains("[DONE]"));

    // 4. V4.1 Text Completion (/v1/completions)
    let cmpl_req = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "prompt": "Hello V4.1",
        "max_tokens": 2,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 4,
        "seed": 42,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "response_format": {"type": "json_object"},
        "dspark": false
    });
    let cmpl_resp = client
        .post(format!("{base_url}/v1/completions"))
        .bearer_auth("secret-token-123")
        .json(&cmpl_req)
        .send()
        .await
        .unwrap();
    assert_eq!(cmpl_resp.status(), StatusCode::OK);
    // 4b. V4.1 Chat Completion using CSF model ID alias
    let csf_req = serde_json::json!({
        "model": "local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF",
        "messages": [{"role": "user", "content": "Hello CSF"}],
        "max_tokens": 2,
        "stream": false
    });
    let csf_resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("secret-token-123")
        .json(&csf_req)
        .send()
        .await
        .unwrap();
    assert_eq!(csf_resp.status(), StatusCode::OK);
    let csf_body: serde_json::Value = csf_resp.json().await.unwrap();
    assert_eq!(
        csf_body["model"],
        "local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF"
    );
    // Warm-up (1) + chat (1) + stream_a (1) + stream_b (1) + completion (1) + csf_chat (1) = 6 executions
    assert_eq!(engine.execution_count(), 6);

    // 5. Reject unsupported response_format
    let bad_format_req = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": "Hello"}],
        "response_format": {"type": "xml_object"}
    });
    let bad_format_resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("secret-token-123")
        .json(&bad_format_req)
        .send()
        .await
        .unwrap();
    assert_eq!(bad_format_resp.status(), StatusCode::BAD_REQUEST);

    // 6. Verify /metrics recorded V4.1 TTFT, ITL, and Optimal Runtime Experience (ORE) metrics
    assert!(state.ring_buffer.retained_stream_count() >= 2);
    let metrics_resp = client
        .get(format!("{base_url}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(metrics_resp.status(), StatusCode::OK);
    let metrics_json: serde_json::Value = metrics_resp.json().await.unwrap();
    assert!(metrics_json["completed_requests"].as_u64().unwrap() >= 5);
    assert!(metrics_json["avg_ttft_ms"].as_f64().unwrap() >= 0.0);
    assert!(metrics_json["p95_ttft_ms"].as_f64().unwrap() >= 0.0);
    assert!(metrics_json["completion_success_rate"].as_f64().unwrap() >= 0.99);
    assert!(metrics_json["optimal_runtime_score"].as_f64().unwrap() > 0.0);
}
