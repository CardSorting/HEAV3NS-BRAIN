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
    let text = sse_resp.text().await.unwrap();
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("[DONE]"));

    // 5. Test /metrics
    let metrics_resp = client
        .get(format!("{base_url}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(metrics_resp.status(), StatusCode::OK);
    let metrics_body: serde_json::Value = metrics_resp.json().await.unwrap();
    assert!(metrics_body["total_requests"].as_u64().unwrap() >= 2);
}
