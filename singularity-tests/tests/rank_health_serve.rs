//! Integration tests: `RankHealthMonitor` / bounded collectives wired into the real
//! `ZosScheduler` serve loop (submit -> step_cooperative -> subscribers).
//! A dead or hung rank must be detected, isolated (clients released, KV blocks freed,
//! new work refused) and never deadlock.

use singularity_tests::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const TOTAL_BLOCKS: usize = 64;

fn scheduler_with(
    block_manager: Arc<PagedBlockManager>,
    setup: impl FnOnce(ZosScheduler) -> ZosScheduler,
) -> Arc<ZosScheduler> {
    let config = SchedulerConfig {
        prefill_attempt_interval: 1,
        prefill_any_attempt_interval: 1,
        ..Default::default()
    };
    let radix = Arc::new(RadixTree::new(100));
    let gpu = Arc::new(MockGpuBackend::new_b200());
    let s = ZosScheduler::new(config, block_manager, radix, gpu).unwrap();
    Arc::new(setup(s))
}

fn seq(id: u64, max_tokens: usize) -> Sequence {
    Sequence::new(
        id,
        "p".to_string(),
        vec![1, 2, 3],
        SamplingParams {
            max_tokens,
            ..Default::default()
        },
    )
}

async fn drain_until_terminal(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<SequenceOutputEvent>,
) -> Option<SequenceOutputEvent> {
    loop {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(SequenceOutputEvent::Token(_))) => continue,
            Ok(other) => return other,
            Err(_) => panic!("client deadlocked waiting for an event"),
        }
    }
}

#[tokio::test]
async fn healthy_group_serves_and_idle_time_is_not_a_stall() {
    let bm = Arc::new(PagedBlockManager::new(TOTAL_BLOCKS, 16));
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let monitor = Arc::new(RankHealthMonitor::new(
        2,
        Duration::from_millis(200),
        Instant::now(),
    ));
    let s = scheduler_with(bm.clone(), |s| {
        s.with_rank_health(monitor.clone(), 0, Duration::from_secs(2))
            .with_dp_reduce(Arc::new(move |v, _op| {
                c.fetch_add(1, Ordering::SeqCst);
                v
            }))
    });
    // Idle for longer than the stall timeout: must NOT poison.
    std::thread::sleep(Duration::from_millis(400));
    assert!(s.step_cooperative().unwrap().is_none());
    assert!(s.watchdog_tick().is_none());

    let mut rx = s.subscribe(1);
    s.submit_sequence(seq(1, 3)).unwrap();
    for _ in 0..10 {
        let _ = s.step_cooperative().unwrap();
    }
    assert!(matches!(
        drain_until_terminal(&mut rx).await,
        Some(SequenceOutputEvent::Finished(_))
    ));
    assert!(calls.load(Ordering::SeqCst) > 0, "reduce must be exercised");
    assert!(!s.is_poisoned());
}

#[tokio::test]
async fn hung_collective_is_bounded_isolated_and_not_retried() {
    let bm = Arc::new(PagedBlockManager::new(TOTAL_BLOCKS, 16));
    let monitor = Arc::new(RankHealthMonitor::new(
        2,
        Duration::from_secs(60),
        Instant::now(),
    ));
    let s = scheduler_with(bm.clone(), |s| {
        s.with_rank_health(monitor, 0, Duration::from_millis(100))
            .with_dp_reduce(Arc::new(|v, _op| {
                std::thread::sleep(Duration::from_secs(30)); // hung NCCL all-reduce
                v
            }))
    });
    let mut rx1 = s.subscribe(1);
    let mut rx2 = s.subscribe(2);
    s.submit_sequence(seq(1, 50)).unwrap();
    s.submit_sequence(seq(2, 50)).unwrap();

    let t0 = Instant::now();
    let err = s.step_cooperative().err().expect("must fail");
    assert!(t0.elapsed() < Duration::from_secs(3), "step must not hang");
    assert!(err.to_string().contains("did not complete"), "{err}");

    // Isolation: every client released, blocks freed, work refused, state sticky.
    for rx in [&mut rx1, &mut rx2] {
        assert!(matches!(
            drain_until_terminal(rx).await,
            Some(SequenceOutputEvent::Cancelled)
        ));
    }
    assert_eq!(bm.free_block_count(), TOTAL_BLOCKS, "KV blocks leaked");
    assert!(s.is_poisoned());
    assert!(matches!(s.fatal(), Some(CollectiveError::Timeout(_))));
    assert!(s.submit_sequence(seq(3, 1)).is_err());
    // A retry on the same group must fail fast, not touch the dead communicator.
    let t1 = Instant::now();
    assert!(s.step_cooperative().is_err());
    assert!(t1.elapsed() < Duration::from_millis(50));
}

#[tokio::test]
async fn dead_rank_reported_explicitly_is_detected_even_when_idle() {
    let bm = Arc::new(PagedBlockManager::new(TOTAL_BLOCKS, 16));
    let monitor = Arc::new(RankHealthMonitor::new(
        4,
        Duration::from_secs(60),
        Instant::now(),
    ));
    let s = scheduler_with(bm.clone(), |s| {
        s.with_rank_health(monitor.clone(), 0, Duration::from_secs(1))
    });
    let mut rx = s.subscribe(7);
    s.submit_sequence(seq(7, 50)).unwrap();

    monitor.report_failure(2, "CUDA_ERROR_ILLEGAL_ADDRESS");
    // Independent watchdog sees it without needing the serve loop.
    let fatal = s.watchdog_tick().expect("failure must be detected");
    assert!(matches!(fatal, CollectiveError::RankFailed(ref v) if v[0].0 == 2));
    assert!(matches!(
        drain_until_terminal(&mut rx).await,
        Some(SequenceOutputEvent::Cancelled)
    ));
    assert_eq!(bm.free_block_count(), TOTAL_BLOCKS);
    let e = s.step_cooperative().err().expect("must fail").to_string();
    assert!(e.contains("rank(s) failed"), "{e}");
}

#[tokio::test]
async fn watchdog_releases_clients_while_the_stepping_thread_is_hung() {
    let bm = Arc::new(PagedBlockManager::new(TOTAL_BLOCKS, 16));
    let monitor = Arc::new(RankHealthMonitor::new(
        2,
        Duration::from_millis(300), // stall timeout
        Instant::now(),
    ));
    // Collective deadline far longer than the stall timeout: only the watchdog can save us.
    let s = scheduler_with(bm.clone(), |s| {
        s.with_rank_health(monitor, 0, Duration::from_secs(20))
            .with_dp_reduce(Arc::new(|v, _op| {
                std::thread::sleep(Duration::from_secs(8));
                v
            }))
    });
    let mut rx = s.subscribe(1);
    s.submit_sequence(seq(1, 50)).unwrap();

    let stepper = {
        let s = s.clone();
        std::thread::spawn(move || {
            let _ = s.step_cooperative();
        })
    };
    // Poll the watchdog like the server thread does.
    let t0 = Instant::now();
    let fatal = loop {
        if let Some(e) = s.watchdog_tick() {
            break e;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "watchdog never fired"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(matches!(fatal, CollectiveError::RankStalled { .. }));
    assert!(matches!(
        drain_until_terminal(&mut rx).await,
        Some(SequenceOutputEvent::Cancelled)
    ));
    assert!(s.submit_sequence(seq(2, 1)).is_err());
    drop(stepper); // detached; it is bounded by the 20 s deadline and holds nothing we need
}

#[tokio::test]
async fn external_step_guard_enables_watchdog_stall_detection() {
    let bm = Arc::new(PagedBlockManager::new(TOTAL_BLOCKS, 16));
    let monitor = Arc::new(RankHealthMonitor::new(
        2,
        Duration::from_millis(250),
        Instant::now(),
    ));
    let s = scheduler_with(bm, |s| {
        s.with_rank_health(monitor, 0, Duration::from_secs(10))
    });
    let mut rx = s.subscribe(42);
    s.submit_sequence(seq(42, 10)).unwrap();

    let worker = {
        let s = s.clone();
        std::thread::spawn(move || {
            let _guard = s.enter_external_step().expect("must enter external step");
            std::thread::sleep(Duration::from_millis(600));
        })
    };

    let t0 = Instant::now();
    let fatal = loop {
        if let Some(e) = s.watchdog_tick() {
            break e;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(3),
            "watchdog never detected stalled external step"
        );
        std::thread::sleep(Duration::from_millis(40));
    };
    assert!(matches!(fatal, CollectiveError::RankStalled { .. }));
    assert!(matches!(
        drain_until_terminal(&mut rx).await,
        Some(SequenceOutputEvent::Cancelled)
    ));
    assert!(s.enter_external_step().is_err());
    let _ = worker.join();
}

#[test]
fn submit_sequence_rolls_back_partial_pin_failure() {
    let bm = Arc::new(PagedBlockManager::new(8, 2));
    let radix = Arc::new(RadixTree::new(100));
    let gpu = Arc::new(MockGpuBackend::new_b200());
    let s = ZosScheduler::new(SchedulerConfig::default(), bm.clone(), radix.clone(), gpu).unwrap();

    // Allocate 2 blocks (block_size = 2 -> 4 tokens) and insert into radix tree.
    let blocks = bm.allocate_blocks(2).unwrap();
    assert_eq!(blocks.len(), 2);
    radix
        .insert_prefix(&[10, 11, 12, 13], &blocks, &bm)
        .unwrap();

    // Free `blocks[1]` (ref_count 1 -> 0) so it becomes unallocated,
    // while `blocks[0]` remains allocated (ref_count == 1).
    bm.free_blocks(&[blocks[1]]).unwrap();
    let free_before = bm.free_block_count();

    // Submitting a sequence matching `[10, 11, 12, 13]` will pin `blocks[0]`
    // and then fail on unallocated `blocks[1]`. It must roll back the pin on `blocks[0]`.
    let candidate = Sequence::new(
        99,
        "prefix".to_string(),
        vec![10, 11, 12, 13],
        SamplingParams::default(),
    );
    assert!(s.submit_sequence(candidate).is_err());
    assert!(
        !bm.is_block_pinned(blocks[0]),
        "partial pin on blocks[0] must be rolled back on submit_sequence error"
    );
    assert_eq!(bm.free_block_count(), free_before);
}

#[tokio::test]
async fn poisoned_scheduler_surfaces_503_on_ready_and_chat_without_leaking_listeners() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = SystemConfig::default();
    let state = AppState::new_with_mock(config, temp_dir.path().to_path_buf()).unwrap();
    state
        .scheduler
        .poison_with(CollectiveError::RankFailed(vec![(
            3,
            "NVLink peer lost".to_string(),
        )]));

    let app = create_router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // 1. /ready must return 503 with poisoned=true and distributed_topology_operational=false
    let ready_resp = client
        .get(format!("{base_url}/ready"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        ready_resp.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    let ready_json: serde_json::Value = ready_resp.json().await.unwrap();
    assert_eq!(ready_json["ready"], false);
    assert_eq!(ready_json["poisoned"], true);
    assert_eq!(ready_json["distributed_topology_operational"], false);
    assert!(ready_json["fatal_error"]
        .as_str()
        .unwrap_or_default()
        .contains("NVLink peer lost"));

    // 2. /health must report degraded and poisoned=true
    let health_resp = client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health_resp.status(), axum::http::StatusCode::OK);
    let health_json: serde_json::Value = health_resp.json().await.unwrap();
    assert_eq!(health_json["status"], "degraded");
    assert_eq!(health_json["ready"], false);
    assert_eq!(health_json["poisoned"], true);

    // 3. POST /v1/chat/completions must fail fast with 503 and leave 0 listeners
    let chat_resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "singularity-host-development",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        chat_resp.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(state.scheduler.listener_count(), 0);

    // 4. POST /v1/completions must also fail fast with 503
    let cmpl_resp = client
        .post(format!("{base_url}/v1/completions"))
        .json(&serde_json::json!({
            "model": "singularity-host-development",
            "prompt": "hi"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        cmpl_resp.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );

    server.abort();
}
