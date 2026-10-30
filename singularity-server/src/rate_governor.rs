use parking_lot::Mutex;
use singularity_core::SingularityError;
use std::time::Instant;

pub struct LeakyTokenBucket {
    rate_rps: f64,
    capacity: f64,
    tokens: Mutex<f64>,
    last_update: Mutex<Instant>,
}

impl LeakyTokenBucket {
    pub fn new(rate_rps: u32, capacity: u32) -> Self {
        Self {
            rate_rps: rate_rps as f64,
            capacity: capacity as f64,
            tokens: Mutex::new(capacity as f64),
            last_update: Mutex::new(Instant::now()),
        }
    }

    pub fn try_acquire(&self, needed: f64) -> Result<(), SingularityError> {
        let mut tokens_guard = self.tokens.lock();
        let mut last_update_guard = self.last_update.lock();

        let now = Instant::now();
        let elapsed = now.duration_since(*last_update_guard).as_secs_f64();
        *last_update_guard = now;

        // Refill tokens
        *tokens_guard = (*tokens_guard + elapsed * self.rate_rps).min(self.capacity);

        if *tokens_guard >= needed {
            *tokens_guard -= needed;
            Ok(())
        } else {
            let missing = needed - *tokens_guard;
            let retry_ms = ((missing / self.rate_rps) * 1000.0) as u64;
            Err(SingularityError::RateLimitExceeded {
                retry_after_ms: retry_ms.max(1),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_governor_acquire_and_limit() {
        let bucket = LeakyTokenBucket::new(10, 2);
        assert!(bucket.try_acquire(1.0).is_ok());
        assert!(bucket.try_acquire(1.0).is_ok());
        assert!(bucket.try_acquire(1.0).is_err());
    }
}
