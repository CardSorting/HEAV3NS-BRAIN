use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub active_requests: usize,
    pub completed_requests: u64,
    pub aborted_requests: u64,
    pub rejected_requests: u64,
    pub preempted_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_generated_tokens: u64,
    pub avg_ttft_ms: f64,
    pub avg_itl_ms: f64,
    pub p50_ttft_ms: f64,
    pub p95_ttft_ms: f64,
    pub p99_ttft_ms: f64,
    pub p50_itl_ms: f64,
    pub p95_itl_ms: f64,
    pub p99_itl_ms: f64,
    pub ttft_jitter_ratio: f64,
    pub itl_jitter_ratio: f64,
    pub completion_success_rate: f64,
    pub optimal_runtime_score: f64,
    pub allocated_blocks: usize,
    pub total_blocks: usize,
    pub cache_hit_rate: f64,
}

pub struct MetricsCollector {
    total_requests: AtomicU64,
    active_requests: AtomicUsize,
    completed_requests: AtomicU64,
    aborted_requests: AtomicU64,
    rejected_requests: AtomicU64,
    preempted_requests: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    total_generated_tokens: AtomicU64,
    cache_hits: AtomicU64,
    cache_lookups: AtomicU64,
    ttft_samples_ns: RwLock<VecDeque<u64>>,
    itl_samples_ns: RwLock<VecDeque<u64>>,
    total_blocks: AtomicUsize,
    allocated_blocks: AtomicUsize,
}

impl MetricsCollector {
    pub fn new(total_blocks: usize) -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            active_requests: AtomicUsize::new(0),
            completed_requests: AtomicU64::new(0),
            aborted_requests: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
            preempted_requests: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_generated_tokens: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_lookups: AtomicU64::new(0),
            ttft_samples_ns: RwLock::new(VecDeque::with_capacity(10000)),
            itl_samples_ns: RwLock::new(VecDeque::with_capacity(50000)),
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
        let _ = self
            .active_requests
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_sub(1))
            });
        self.completed_requests.fetch_add(1, Ordering::Relaxed);
        self.total_generated_tokens
            .fetch_add(generated_tokens as u64, Ordering::Relaxed);
    }

    pub fn on_request_aborted(&self) {
        let _ = self
            .active_requests
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_sub(1))
            });
        self.aborted_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn on_request_rejected(&self) {
        self.rejected_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn on_request_preempted(&self) {
        self.preempted_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_ttft(&self, ttft_ns: u64) {
        let mut samples = self.ttft_samples_ns.write();
        if samples.len() >= 10000 {
            samples.pop_front();
        }
        samples.push_back(ttft_ns);
    }

    pub fn record_itl(&self, itl_ns: u64) {
        let mut samples = self.itl_samples_ns.write();
        if samples.len() >= 50000 {
            samples.pop_front();
        }
        samples.push_back(itl_ns);
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

        let (avg_ttft, p50_ttft, p95_ttft, p99_ttft) = calculate_stats(&ttft_guard);
        let (avg_itl, p50_itl, p95_itl, p99_itl) = calculate_stats(&itl_guard);

        let lookups = self.cache_lookups.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let hit_rate = if lookups > 0 {
            hits as f64 / lookups as f64
        } else {
            0.0
        };
        let ttft_jitter_ratio = if p50_ttft > 0.0 {
            p99_ttft / p50_ttft
        } else {
            1.0
        };
        let itl_jitter_ratio = if p50_itl > 0.0 {
            p99_itl / p50_itl
        } else {
            1.0
        };

        let completed = self.completed_requests.load(Ordering::Relaxed);
        let aborted = self.aborted_requests.load(Ordering::Relaxed);
        let rejected = self.rejected_requests.load(Ordering::Relaxed);
        let terminal_total = completed + aborted + rejected;
        let completion_success_rate = if terminal_total > 0 {
            completed as f64 / terminal_total as f64
        } else {
            1.0
        };

        let optimal_runtime_score = compute_optimal_runtime_score(
            completion_success_rate,
            itl_jitter_ratio,
            ttft_jitter_ratio,
        );

        MetricsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            active_requests: self.active_requests.load(Ordering::Relaxed),
            completed_requests: completed,
            aborted_requests: aborted,
            rejected_requests: rejected,
            preempted_requests: self.preempted_requests.load(Ordering::Relaxed),
            total_prompt_tokens: self.total_prompt_tokens.load(Ordering::Relaxed),
            total_generated_tokens: self.total_generated_tokens.load(Ordering::Relaxed),
            avg_ttft_ms: avg_ttft / 1_000_000.0,
            avg_itl_ms: avg_itl / 1_000_000.0,
            p50_ttft_ms: p50_ttft / 1_000_000.0,
            p95_ttft_ms: p95_ttft / 1_000_000.0,
            p99_ttft_ms: p99_ttft / 1_000_000.0,
            p50_itl_ms: p50_itl / 1_000_000.0,
            p95_itl_ms: p95_itl / 1_000_000.0,
            p99_itl_ms: p99_itl / 1_000_000.0,
            ttft_jitter_ratio,
            itl_jitter_ratio,
            completion_success_rate,
            optimal_runtime_score,
            allocated_blocks: self.allocated_blocks.load(Ordering::Relaxed),
            total_blocks: self.total_blocks.load(Ordering::Relaxed),
            cache_hit_rate: hit_rate,
        }
    }
}

/// Computes the composite Optimal Runtime Experience score $\mathcal{S}_{\text{ORE}} \in [0.0, 1.0]$
/// using the damped tail-jitter stability terms shared with `deploy/modal/benchmark.py`.
pub fn compute_optimal_runtime_score(
    completion_success_rate: f64,
    itl_jitter_ratio: f64,
    ttft_jitter_ratio: f64,
) -> f64 {
    let itl_stability = 1.0 / (1.0 + (itl_jitter_ratio - 1.0).max(0.0) * 0.25);
    let ttft_stability = 1.0 / (1.0 + (ttft_jitter_ratio - 1.0).max(0.0) * 0.20);
    (0.40 * completion_success_rate.clamp(0.0, 1.0)
        + 0.35 * itl_stability.clamp(0.0, 1.0)
        + 0.25 * ttft_stability.clamp(0.0, 1.0))
    .clamp(0.0, 1.0)
}

fn interpolate_percentile(sorted: &[u64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let rank = (pct.clamp(0.0, 1.0)) * ((sorted.len() - 1) as f64);
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    let weight = rank - (low as f64);
    (sorted[low] as f64) * (1.0 - weight) + (sorted[high] as f64) * weight
}

fn calculate_stats(samples: &VecDeque<u64>) -> (f64, f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut sorted: Vec<u64> = samples.iter().copied().collect();
    sorted.sort_unstable();

    let sum: u128 = sorted.iter().map(|&v| v as u128).sum();
    let avg = sum as f64 / sorted.len() as f64;

    (
        avg,
        interpolate_percentile(&sorted, 0.50),
        interpolate_percentile(&sorted, 0.95),
        interpolate_percentile(&sorted, 0.99),
    )
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
        // Saturating abort should never underflow active_requests below 0
        collector.on_request_aborted();
        collector.on_request_rejected();

        let snap = collector.snapshot();
        assert_eq!(snap.total_requests, 1);
        assert_eq!(snap.active_requests, 0);
        assert_eq!(snap.completed_requests, 1);
        assert_eq!(snap.aborted_requests, 1);
        assert_eq!(snap.rejected_requests, 1);
        assert_eq!(snap.total_prompt_tokens, 100);
        assert_eq!(snap.total_generated_tokens, 10);
        assert_eq!(snap.allocated_blocks, 256);
        assert_eq!(snap.total_blocks, 1024);
        assert_eq!(snap.cache_hit_rate, 0.5);
        assert!((snap.avg_ttft_ms - 20.0).abs() < 1e-3);
        assert!((snap.p50_ttft_ms - 20.0).abs() < 1e-3);
        assert!((snap.p99_ttft_ms - 20.0).abs() < 1e-3);
        assert!((snap.avg_itl_ms - 6.0).abs() < 1e-3);
        assert!((snap.p50_itl_ms - 6.0).abs() < 1e-3);
        assert!(snap.itl_jitter_ratio >= 1.0);
        assert!(snap.ttft_jitter_ratio >= 1.0);
        assert!((snap.completion_success_rate - (1.0 / 3.0)).abs() < 1e-6);
        assert!(snap.optimal_runtime_score > 0.0 && snap.optimal_runtime_score <= 1.0);
    }

    #[test]
    fn test_optimal_runtime_score_slo_boundary_and_parity() {
        // Ideal zero-jitter, 100% completion -> 1.0
        assert!((compute_optimal_runtime_score(1.0, 1.0, 1.0) - 1.0).abs() < 1e-9);

        // At the documented SLO boundary (J_ITL = 1.35, J_TTFT = 1.50, R_comp = 1.0),
        // S_ORE must satisfy the >= 0.90 "optimal" gate and match benchmark.py (~0.949112).
        let boundary_score = compute_optimal_runtime_score(1.0, 1.35, 1.50);
        assert!(
            boundary_score >= 0.90,
            "SLO boundary must qualify for optimal tier (>= 0.90), got {boundary_score}"
        );
        assert!((boundary_score - 0.9491118077324974).abs() < 1e-6);
    }
}
