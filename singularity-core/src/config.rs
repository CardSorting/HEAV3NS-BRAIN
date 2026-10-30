use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineType {
    Normal,
    Prompt,
    Completion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetArch {
    AutoDetect,
    BlackwellSm100,
    HopperSm90,
    HostMock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheMemoryFormat {
    Fp4,
    MxFp8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub target_arch: TargetArch,
    pub device_id: i32,
    pub memory_fraction: f64,
    pub enable_native_fp4: bool,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            target_arch: TargetArch::AutoDetect,
            device_id: 0,
            memory_fraction: 0.90,
            enable_native_fp4: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub prefill_chunk_size: usize,
    pub dp_size: usize,
    pub dp_rank: usize,
    pub prefill_attempt_interval: usize,     // ALL point interval
    pub prefill_any_attempt_interval: usize, // ANY point interval
    pub engine_type: EngineType,
    pub enable_mtp: bool,
    pub max_draft_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 128,
            max_seq_len: 8192,
            prefill_chunk_size: 512,
            dp_size: 1,
            dp_rank: 0,
            prefill_attempt_interval: 8,
            prefill_any_attempt_interval: 32,
            engine_type: EngineType::Normal,
            enable_mtp: true,
            max_draft_tokens: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub total_blocks: usize,
    pub block_size: usize,
    pub memory_format: CacheMemoryFormat,
    pub swa_window_size: usize,
    pub max_prefix_cache_nodes: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            total_blocks: 1024,
            block_size: 16,
            memory_format: CacheMemoryFormat::Fp4,
            swa_window_size: 1024,
            max_prefix_cache_nodes: 50000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_waiting_requests: usize,
    pub rate_limit_rps: u32,
    pub burst_capacity: u32,
    pub sse_keepalive_secs: u64,
    pub replay_buffer_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_waiting_requests: 1024,
            rate_limit_rps: 500,
            burst_capacity: 1000,
            sse_keepalive_secs: 15,
            replay_buffer_size: 64,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemConfig {
    pub hardware: HardwareConfig,
    pub scheduler: SchedulerConfig,
    pub cache: CacheConfig,
    pub server: ServerConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_config_serde_and_defaults() {
        let default_cfg = SystemConfig::default();
        let json_str = serde_json::to_string(&default_cfg).unwrap();
        let deserialized: SystemConfig = serde_json::from_str(&json_str).unwrap();

        assert_eq!(deserialized.hardware.device_id, 0);
        assert_eq!(deserialized.scheduler.max_batch_size, 128);
        assert_eq!(deserialized.cache.total_blocks, 1024);
        assert_eq!(deserialized.server.port, 8080);
    }
}
