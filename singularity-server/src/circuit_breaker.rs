use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use singularity_core::SingularityError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    Healthy,
    Degraded,
    Exhausted,
    Disabled,
}

pub struct CircuitBreaker {
    state: RwLock<CircuitState>,
    consecutive_failures: AtomicUsize,
    failure_threshold: usize,
    cooldown_duration: Duration,
    tripped_at: RwLock<Option<Instant>>,
    canary_in_flight: AtomicUsize,
    canary_deadline: RwLock<Option<Instant>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, cooldown_secs: u64) -> Self {
        Self {
            state: RwLock::new(CircuitState::Healthy),
            consecutive_failures: AtomicUsize::new(0),
            failure_threshold,
            cooldown_duration: Duration::from_secs(cooldown_secs),
            tripped_at: RwLock::new(None),
            canary_in_flight: AtomicUsize::new(0),
            canary_deadline: RwLock::new(None),
        }
    }

    pub fn default_breaker() -> Self {
        Self::new(5, 10)
    }

    pub fn current_state(&self) -> CircuitState {
        *self.state.read()
    }

    /// Acquires permission to dispatch a request.
    /// In Exhausted state, allows only a single canary probe after cooldown.
    pub fn check_permission(&self) -> Result<(), SingularityError> {
        let state = *self.state.read();

        match state {
            CircuitState::Healthy => Ok(()),
            CircuitState::Degraded => Ok(()),
            CircuitState::Disabled => Err(SingularityError::CircuitBreakerTripped {
                state: "Disabled".to_string(),
                retry_after_secs: 999999,
            }),
            CircuitState::Exhausted => {
                let tripped_at = *self.tripped_at.read();
                if let Some(time) = tripped_at {
                    let elapsed = time.elapsed();
                    if elapsed >= self.cooldown_duration {
                        // Check if previous canary timed out
                        if let Some(deadline) = *self.canary_deadline.read() {
                            if Instant::now() > deadline {
                                self.canary_in_flight.store(0, Ordering::Relaxed);
                            }
                        }

                        // Canary probe opportunity: only 1 in flight
                        if self
                            .canary_in_flight
                            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::Relaxed)
                            .is_ok()
                        {
                            *self.canary_deadline.write() =
                                Some(Instant::now() + Duration::from_secs(5));
                            return Ok(());
                        }
                    }
                    let remaining = self.cooldown_duration.saturating_sub(elapsed).as_secs();
                    return Err(SingularityError::CircuitBreakerTripped {
                        state: "Exhausted".to_string(),
                        retry_after_secs: remaining.max(1),
                    });
                }
                Err(SingularityError::CircuitBreakerTripped {
                    state: "Exhausted".to_string(),
                    retry_after_secs: 10,
                })
            }
        }
    }

    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.canary_in_flight.store(0, Ordering::Relaxed);
        *self.canary_deadline.write() = None;
        let mut state = self.state.write();
        if *state != CircuitState::Healthy {
            *state = CircuitState::Healthy;
        }
    }

    pub fn record_failure(&self) {
        self.canary_in_flight.store(0, Ordering::Relaxed);
        *self.canary_deadline.write() = None;
        let fails = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
        let mut state = self.state.write();

        if fails >= self.failure_threshold {
            *state = CircuitState::Exhausted;
            *self.tripped_at.write() = Some(Instant::now());
        } else if fails > 1 && fails >= self.failure_threshold.div_ceil(2) {
            *state = CircuitState::Degraded;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_transitions() {
        let cb = CircuitBreaker::new(3, 1);
        assert_eq!(cb.current_state(), CircuitState::Healthy);
        assert!(cb.check_permission().is_ok());

        cb.record_failure();
        assert_eq!(cb.current_state(), CircuitState::Healthy);
        assert!(cb.check_permission().is_ok());

        cb.record_failure();
        assert_eq!(cb.current_state(), CircuitState::Degraded);
        assert!(cb.check_permission().is_ok());

        cb.record_failure();
        assert_eq!(cb.current_state(), CircuitState::Exhausted);
        assert!(cb.check_permission().is_err());
    }
}
