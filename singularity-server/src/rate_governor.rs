use parking_lot::Mutex;
use singularity_core::SingularityError;
use std::time::Instant;

struct BucketState {
    tokens: f64,
    last_update: Instant,
}

pub struct LeakyTokenBucket {
    rate_rps: f64,
    capacity: f64,
    state: Mutex<BucketState>,
}

impl LeakyTokenBucket {
    pub fn new(rate_rps: u32, capacity: u32) -> Self {
        let rate_rps = (rate_rps.max(1)) as f64;
        let capacity = (capacity.max(1)) as f64;
        Self {
            rate_rps,
            capacity,
            state: Mutex::new(BucketState {
                tokens: capacity,
                last_update: Instant::now(),
            }),
        }
    }

    pub fn try_acquire(&self, needed: f64) -> Result<(), SingularityError> {
        if !needed.is_finite() || needed < 0.0 {
            return Err(SingularityError::RateLimitExceeded { retry_after_ms: 1 });
        }
        let mut guard = self.state.lock();

        let now = Instant::now();
        let elapsed = now.duration_since(guard.last_update).as_secs_f64();
        guard.last_update = now;

        // Refill tokens
        guard.tokens = (guard.tokens + elapsed * self.rate_rps).min(self.capacity);

        if guard.tokens >= needed {
            guard.tokens -= needed;
            Ok(())
        } else {
            let missing = needed - guard.tokens;
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
        assert!(bucket.try_acquire(f64::NAN).is_err());
    }
}
