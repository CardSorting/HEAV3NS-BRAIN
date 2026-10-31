use crate::circuit_breaker::CircuitBreaker;
use crate::handlers::{
    chat_completions_handler, completions_handler, health_handler, metrics_handler, models_handler,
    ready_handler,
};
use crate::rate_governor::LeakyTokenBucket;
use crate::ring_buffer::StreamRingBuffer;
use crate::store::TransactionalStore;
use crate::v41_engine::{ReadinessState, V41ServingEngine};
use axum::{
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Router,
};
use parking_lot::RwLock;
use singularity_cache::{PagedBlockManager, RadixTree};
use singularity_core::{MetricsCollector, SingularityError, SystemConfig};
use singularity_cuda::{GpuBackend, MockGpuBackend};
use singularity_scheduler::{RankHealthMonitor, ZosScheduler};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Exit code used when the serving group is declared dead (EX_SOFTWARE).
pub const EXIT_GROUP_FAILED: i32 = 70;

/// Max seconds a single scheduler step may run before it is declared hung.
/// `SINGULARITY_STEP_STALL_TIMEOUT_SECS`; 0 disables. Default 600.
fn step_stall_timeout_secs() -> u64 {
    std::env::var("SINGULARITY_STEP_STALL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(600)
}

/// Independent watchdog thread: detects a dead/hung rank (the stepping thread cannot
/// report its own hang), poisons the scheduler so every client is released, then exits
/// the process non-zero after a short grace so the container stops billing instead of
/// serving from a broken group. Holds only a `Weak` ref: it ends when the scheduler drops.
fn spawn_rank_watchdog(scheduler: &Arc<ZosScheduler>) {
    let weak = Arc::downgrade(scheduler);
    let _ = std::thread::Builder::new()
        .name("rank-watchdog".into())
        .spawn(move || loop {
            std::thread::sleep(Duration::from_secs(1));
            let Some(s) = weak.upgrade() else { return };
            if let Some(err) = s.watchdog_tick() {
                drop(s);
                eprintln!("FATAL: {err}; exiting with code {EXIT_GROUP_FAILED}");
                std::thread::sleep(Duration::from_secs(2));
                if weak.strong_count() == 0 {
                    return;
                }
                std::process::exit(EXIT_GROUP_FAILED);
            }
        });
}

pub struct AppState {
    pub config: SystemConfig,
    pub block_manager: Arc<PagedBlockManager>,
    pub radix_tree: Arc<RadixTree>,
    pub gpu_backend: Arc<dyn GpuBackend>,
    pub scheduler: Arc<ZosScheduler>,
    pub tokenizer: Arc<singularity_model::TokenizerWrapper>,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub rate_governor: Arc<LeakyTokenBucket>,
    pub ring_buffer: Arc<StreamRingBuffer>,
    pub store: Arc<TransactionalStore>,
    pub metrics: Arc<MetricsCollector>,
    readiness: RwLock<ReadinessState>,
    v41_engine: RwLock<Option<Arc<V41ServingEngine>>>,
    api_key: RwLock<Option<String>>,
}

impl AppState {
    pub fn new(
        config: SystemConfig,
        storage_dir: PathBuf,
        gpu_backend: Arc<dyn GpuBackend>,
        model: Arc<singularity_model::TransformerModel>,
        tokenizer: Arc<singularity_model::TokenizerWrapper>,
        block_manager: Arc<PagedBlockManager>,
    ) -> Result<Arc<Self>, SingularityError> {
        let radix_tree = Arc::new(RadixTree::new(config.cache.max_prefix_cache_nodes));
        let mut scheduler = ZosScheduler::with_model(
            config.scheduler.clone(),
            block_manager.clone(),
            radix_tree.clone(),
            gpu_backend.clone(),
            model,
        )?;
        let stall_secs = step_stall_timeout_secs();
        if stall_secs > 0 {
            let timeout = Duration::from_secs(stall_secs);
            let monitor = Arc::new(RankHealthMonitor::new(1, timeout, Instant::now()));
            scheduler = scheduler.with_rank_health(monitor, 0, timeout);
        }
        let scheduler = Arc::new(scheduler);
        if stall_secs > 0 {
            spawn_rank_watchdog(&scheduler);
        }

        let circuit_breaker = Arc::new(CircuitBreaker::default_breaker());
        let rate_governor = Arc::new(LeakyTokenBucket::new(
            config.server.rate_limit_rps,
            config.server.burst_capacity,
        ));
        let ring_buffer = Arc::new(StreamRingBuffer::new(config.server.replay_buffer_size));
        let store = Arc::new(TransactionalStore::new(storage_dir)?);
        let metrics = Arc::new(MetricsCollector::new(config.cache.total_blocks));
        let env_api_key = std::env::var("SINGULARITY_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty());

        Ok(Arc::new(Self {
            config,
            block_manager,
            radix_tree,
            gpu_backend,
            scheduler,
            tokenizer,
            circuit_breaker,
            rate_governor,
            ring_buffer,
            store,
            metrics,
            readiness: RwLock::new(ReadinessState::default_ready()),
            v41_engine: RwLock::new(None),
            api_key: RwLock::new(env_api_key),
        }))
    }

    pub fn new_with_mock(
        config: SystemConfig,
        storage_dir: PathBuf,
    ) -> Result<Arc<Self>, SingularityError> {
        let block_manager = Arc::new(PagedBlockManager::new(
            config.cache.total_blocks,
            config.cache.block_size,
        ));
        let gpu_backend: Arc<dyn GpuBackend> = Arc::new(MockGpuBackend::new_b200());
        let model = Arc::new(singularity_model::TransformerModel::new_random(
            singularity_model::TransformerConfig::tiny(),
        ));
        let tokenizer = Arc::new(singularity_model::TokenizerWrapper::new_fallback());

        Self::new(
            config,
            storage_dir,
            gpu_backend,
            model,
            tokenizer,
            block_manager,
        )
    }

    pub fn readiness(&self) -> ReadinessState {
        self.readiness.read().clone()
    }

    pub fn set_readiness(&self, state: ReadinessState) {
        *self.readiness.write() = state;
    }

    pub fn v41_engine(&self) -> Option<Arc<V41ServingEngine>> {
        self.v41_engine.read().clone()
    }

    pub fn set_v41_engine(&self, engine: Arc<V41ServingEngine>) {
        *self.v41_engine.write() = Some(engine);
    }

    pub fn api_key(&self) -> Option<String> {
        self.api_key.read().clone()
    }

    pub fn set_api_key(&self, key: Option<String>) {
        *self.api_key.write() = key.filter(|k| !k.trim().is_empty());
    }

    pub fn check_auth(&self, headers: &HeaderMap) -> Result<(), (StatusCode, String)> {
        let Some(expected) = self.api_key() else {
            return Ok(());
        };
        let provided = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(str::trim);
        if provided == Some(expected.as_str()) {
            Ok(())
        } else {
            Err((
                StatusCode::UNAUTHORIZED,
                "Missing or invalid Bearer API key".to_string(),
            ))
        }
    }
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
