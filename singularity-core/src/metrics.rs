use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub active_requests: usize,
    pub completed_requests: u64,
    pub preempted_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_generated_tokens: u64,
    pub avg_ttft_ms: f64,
    pub avg_itl_ms: f64,
    pub p95_ttft_ms: f64,
    pub p95_itl_ms: f64,
    pub allocated_blocks: usize,
    pub total_blocks: usize,
    pub cache_hit_rate: f64,
}

pub struct MetricsCollector {
    total_requests: AtomicU64,
    active_requests: AtomicUsize,
    completed_requests: AtomicU64,
    preempted_requests: AtomicU64,
    total_prompt_tokens: AtomicU64,
    total_generated_tokens: AtomicU64,
    cache_hits: AtomicU64,
    cache_lookups: AtomicU64,
    ttft_samples_ns: RwLock<Vec<u64>>,
    itl_samples_ns: RwLock<Vec<u64>>,
    total_blocks: AtomicUsize,
    allocated_blocks: AtomicUsize,
}

impl MetricsCollector {
    pub fn new(total_blocks: usize) -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            active_requests: AtomicUsize::new(0),
            completed_requests: AtomicU64::new(0),
            preempted_requests: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_generated_tokens: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_lookups: AtomicU64::new(0),
            ttft_samples_ns: RwLock::new(Vec::with_capacity(10000)),
            itl_samples_ns: RwLock::new(Vec::with_capacity(50000)),
            total_blocks: AtomicUsize::new(total_blocks),
            allocated_blocks: AtomicUsize::new(0),
        }
    }

    pub fn on_request_admitted(&self, prompt_tokens: usize) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
    }

    pub fn on_request_completed(&self, generated_tokens: usize) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.completed_requests.fetch_add(1, Ordering::Relaxed);
        self.total_generated_tokens
            .fetch_add(generated_tokens as u64, Ordering::Relaxed);
    }

    pub fn on_request_preempted(&self) {
        self.preempted_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_ttft(&self, ttft_ns: u64) {
        let mut samples = self.ttft_samples_ns.write();
        if samples.len() >= 10000 {
            samples.remove(0);
        }
        samples.push(ttft_ns);
    }

    pub fn record_itl(&self, itl_ns: u64) {
        let mut samples = self.itl_samples_ns.write();
        if samples.len() >= 50000 {
            samples.remove(0);
        }
        samples.push(itl_ns);
    }

    pub fn record_cache_lookup(&self, hit: bool) {
        self.cache_lookups.fetch_add(1, Ordering::Relaxed);
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn update_blocks(&self, allocated: usize) {
        self.allocated_blocks.store(allocated, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let ttft_guard = self.ttft_samples_ns.read();
        let itl_guard = self.itl_samples_ns.read();

        let (avg_ttft, p95_ttft) = calculate_stats(&ttft_guard);
        let (avg_itl, p95_itl) = calculate_stats(&itl_guard);

        let lookups = self.cache_lookups.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let hit_rate = if lookups > 0 {
            hits as f64 / lookups as f64
        } else {
            0.0
        };

        MetricsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            active_requests: self.active_requests.load(Ordering::Relaxed),
            completed_requests: self.completed_requests.load(Ordering::Relaxed),
            preempted_requests: self.preempted_requests.load(Ordering::Relaxed),
            total_prompt_tokens: self.total_prompt_tokens.load(Ordering::Relaxed),
            total_generated_tokens: self.total_generated_tokens.load(Ordering::Relaxed),
            avg_ttft_ms: avg_ttft / 1_000_000.0,
            avg_itl_ms: avg_itl / 1_000_000.0,
            p95_ttft_ms: p95_ttft / 1_000_000.0,
            p95_itl_ms: p95_itl / 1_000_000.0,
            allocated_blocks: self.allocated_blocks.load(Ordering::Relaxed),
            total_blocks: self.total_blocks.load(Ordering::Relaxed),
            cache_hit_rate: hit_rate,
        }
    }
}

fn calculate_stats(samples: &[u64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    let sum: u64 = sorted.iter().sum();
    let avg = sum as f64 / sorted.len() as f64;

    let p95_idx = ((sorted.len() as f64) * 0.95).min((sorted.len() - 1) as f64) as usize;
    let p95 = sorted[p95_idx] as f64;

    (avg, p95)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_lifecycle_and_stats() {
        let collector = MetricsCollector::new(1024);
        collector.on_request_admitted(100);
        collector.record_ttft(20_000_000); // 20ms
        collector.record_itl(5_000_000); // 5ms
        collector.record_itl(7_000_000); // 7ms
        collector.record_cache_lookup(true);
        collector.record_cache_lookup(false);
        collector.update_blocks(256);
        collector.on_request_completed(10);

        let snap = collector.snapshot();
        assert_eq!(snap.total_requests, 1);
        assert_eq!(snap.active_requests, 0);
        assert_eq!(snap.completed_requests, 1);
        assert_eq!(snap.total_prompt_tokens, 100);
        assert_eq!(snap.total_generated_tokens, 10);
        assert_eq!(snap.allocated_blocks, 256);
        assert_eq!(snap.total_blocks, 1024);
        assert_eq!(snap.cache_hit_rate, 0.5);
        assert!((snap.avg_ttft_ms - 20.0).abs() < 1e-3);
        assert!((snap.avg_itl_ms - 6.0).abs() < 1e-3);
    }
}
