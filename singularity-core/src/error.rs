use thiserror::Error;

#[derive(Error, Debug)]
pub enum SingularityError {
    #[error("Unsupported hardware: major={major}, minor={minor}. Reason: {reason}")]
    UnsupportedHardware {
        major: i32,
        minor: i32,
        reason: String,
    },

    #[error("Out of GPU memory: requested {requested_bytes} bytes, free {available_bytes} bytes")]
    OutOfMemory {
        requested_bytes: usize,
        available_bytes: usize,
    },

    #[error("Block allocation failed: needed {needed} blocks, but only {free} blocks available")]
    BlockAllocationFailed { needed: usize, free: usize },

    #[error("Preemption required for sequence {victim_seq_id}")]
    PreemptionRequired { victim_seq_id: u64 },

    #[error("Circuit breaker tripped ({state}), retry after {retry_after_secs}s")]
    CircuitBreakerTripped {
        state: String,
        retry_after_secs: u64,
    },

    #[error("Rate limit exceeded: queue full, retry after {retry_after_ms}ms")]
    RateLimitExceeded { retry_after_ms: u64 },

    #[error("Client disconnected during execution for sequence {sequence_id}")]
    ClientDisconnected { sequence_id: u64 },

    #[error("Invalid grammar or constraint: {detail}")]
    InvalidGrammar { detail: String },

    #[error("Model load/execution error: {reason}")]
    ModelError { reason: String },

    #[error("Cache error: {detail}")]
    CacheError { detail: String },

    #[error("Scheduler error: {detail}")]
    SchedulerError { detail: String },

    #[error("Hardware execution error: {detail}")]
    HardwareError { detail: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Internal concurrency/lock error: {0}")]
    Internal(String),
}
