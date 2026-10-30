use std::collections::VecDeque;

/// Sliding 9-step median runtime estimator for sampler GPU execution time.
pub struct SamplerRuntimeEstimator {
    window: VecDeque<f64>,
    capacity: usize,
}

impl SamplerRuntimeEstimator {
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn default_estimator() -> Self {
        Self::new(9)
    }

    pub fn record_ms(&mut self, ms: f64) {
        if self.window.len() >= self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(ms);
    }

    pub fn median_ms(&self) -> f64 {
        if self.window.is_empty() {
            return 0.1; // Default 100 microseconds baseline
        }
        let mut sorted: Vec<f64> = self.window.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }
}

/// Dual-multiset median tracker maintaining sliding window to fit forward execution slope:
/// Forward time = baseline + slope * batch_size.
pub struct ForwardRuntimeEstimator {
    samples: VecDeque<(usize, f64)>, // (batch_size, execution_ms)
    capacity: usize,
    baseline_ms: f64,
    slope_ms: f64,
}

impl ForwardRuntimeEstimator {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
            baseline_ms: 0.5,
            slope_ms: 0.02,
        }
    }

    pub fn default_estimator() -> Self {
        Self::new(500)
    }

    pub fn record_step(&mut self, batch_size: usize, ms: f64) {
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back((batch_size, ms));

        // Recompute simple linear regression (least squares) if enough samples
        if self.samples.len() >= 10 {
            let n = self.samples.len() as f64;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_xx = 0.0;

            for &(bsz, ms) in &self.samples {
                let x = bsz as f64;
                sum_x += x;
                sum_y += ms;
                sum_xy += x * ms;
                sum_xx += x * x;
            }

            let denom = n * sum_xx - sum_x * sum_x;
            if denom.abs() > 1e-6 {
                let slope = (n * sum_xy - sum_x * sum_y) / denom;
                let intercept = (sum_y - slope * sum_x) / n;
                self.slope_ms = slope.max(0.001);
                self.baseline_ms = intercept.max(0.1);
            }
        }
    }

    pub fn estimate_ms(&self, batch_size: usize) -> f64 {
        self.baseline_ms + self.slope_ms * (batch_size as f64)
    }
}

/// Dynamic MTP (Multi-Token Prediction) draft token allocator:
/// Allocates draft validation slots (K in [0, 5]) based on confidence score.
/// If confidence is lower than confidence_threshold (e.g. 0.4), draft tokens collapse to 0
/// to avoid wasting compute on speculative decoding miss penalties.
#[derive(Debug, Clone)]
pub struct DynamicMtpDraftAllocator {
    confidence_threshold: f32,
    max_draft_tokens: usize,
}

impl DynamicMtpDraftAllocator {
    pub fn new(confidence_threshold: f32, max_draft_tokens: usize) -> Self {
        Self {
            confidence_threshold,
            max_draft_tokens,
        }
    }

    pub fn default_allocator() -> Self {
        Self::new(0.40, 4)
    }

    pub fn allocate_draft_tokens(&self, confidence_score: f32) -> usize {
        if confidence_score < self.confidence_threshold {
            // Collapse draft tokens to 0 to prevent speculative decoding miss penalty
            0
        } else {
            // Linear scaling between threshold and 1.0 up to max_draft_tokens
            let normalized = (confidence_score - self.confidence_threshold)
                / (1.0 - self.confidence_threshold).max(1e-4);
            let draft = (normalized * (self.max_draft_tokens as f32)).round() as usize;
            draft.clamp(1, self.max_draft_tokens)
        }
    }
}
