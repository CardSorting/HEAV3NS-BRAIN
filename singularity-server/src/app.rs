use crate::circuit_breaker::CircuitBreaker;
use crate::handlers::{chat_completions_handler, health_handler, metrics_handler, models_handler};
use crate::rate_governor::LeakyTokenBucket;
use crate::ring_buffer::StreamRingBuffer;
use crate::store::TransactionalStore;
use axum::{
    routing::{get, post},
    Router,
};
use singularity_cache::{PagedBlockManager, RadixTree};
use singularity_core::{MetricsCollector, SingularityError, SystemConfig};
use singularity_cuda::{GpuBackend, MockGpuBackend};
use singularity_scheduler::ZosScheduler;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

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
        let scheduler = Arc::new(ZosScheduler::with_model(
            config.scheduler.clone(),
            block_manager.clone(),
            radix_tree.clone(),
            gpu_backend.clone(),
            model,
        )?);

        let circuit_breaker = Arc::new(CircuitBreaker::default_breaker());
        let rate_governor = Arc::new(LeakyTokenBucket::new(
            config.server.rate_limit_rps,
            config.server.burst_capacity,
        ));
        let ring_buffer = Arc::new(StreamRingBuffer::new(config.server.replay_buffer_size));
        let store = Arc::new(TransactionalStore::new(storage_dir)?);
        let metrics = Arc::new(MetricsCollector::new(config.cache.total_blocks));

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
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
