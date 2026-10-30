use serde::{Deserialize, Serialize};
use singularity_core::{SequenceId, SingularityError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionState {
    Received,
    Queued,
    Admitted,
    Dispatched,
    Completed,
    Cancelled,
    TimedOut,
    Rejected,
    Failed,
}

pub struct AdmissionLifecycle {
    pub sequence_id: SequenceId,
    pub state: parking_lot::RwLock<AdmissionState>,
    pub client_disconnected: Arc<AtomicBool>,
    pub created_at: Instant,
}

impl AdmissionLifecycle {
    pub fn new(sequence_id: SequenceId) -> Self {
        Self {
            sequence_id,
            state: parking_lot::RwLock::new(AdmissionState::Received),
            client_disconnected: Arc::new(AtomicBool::new(false)),
            created_at: Instant::now(),
        }
    }

    pub fn current_state(&self) -> AdmissionState {
        *self.state.read()
    }

    pub fn transition(&self, new_state: AdmissionState) -> Result<(), SingularityError> {
        let mut guard = self.state.write();
        let current = *guard;

        let valid = matches!(
            (current, new_state),
            (AdmissionState::Received, AdmissionState::Queued)
                | (AdmissionState::Queued, AdmissionState::Admitted)
                | (AdmissionState::Admitted, AdmissionState::Dispatched)
                | (AdmissionState::Dispatched, AdmissionState::Completed)
                | (_, AdmissionState::Cancelled)
                | (_, AdmissionState::TimedOut)
                | (_, AdmissionState::Rejected)
                | (_, AdmissionState::Failed)
        );

        if valid {
            *guard = new_state;
            Ok(())
        } else {
            Err(SingularityError::Internal(format!(
                "Invalid admission transition from {current:?} to {new_state:?}"
            )))
        }
    }

    pub fn mark_client_disconnected(&self) {
        self.client_disconnected.store(true, Ordering::SeqCst);
        let _ = self.transition(AdmissionState::Cancelled);
    }

    pub fn is_client_disconnected(&self) -> bool {
        self.client_disconnected.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admission_lifecycle_transitions() {
        let lifecycle = AdmissionLifecycle::new(100);
        assert_eq!(lifecycle.current_state(), AdmissionState::Received);

        lifecycle.transition(AdmissionState::Queued).unwrap();
        assert_eq!(lifecycle.current_state(), AdmissionState::Queued);

        lifecycle.transition(AdmissionState::Admitted).unwrap();
        assert_eq!(lifecycle.current_state(), AdmissionState::Admitted);

        lifecycle.transition(AdmissionState::Dispatched).unwrap();
        assert_eq!(lifecycle.current_state(), AdmissionState::Dispatched);

        lifecycle.transition(AdmissionState::Completed).unwrap();
        assert_eq!(lifecycle.current_state(), AdmissionState::Completed);

        // Invalid backward transition
        assert!(lifecycle.transition(AdmissionState::Queued).is_err());
    }
}
