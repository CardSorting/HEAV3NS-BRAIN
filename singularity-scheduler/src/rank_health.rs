//! Rank-failure and collective-deadlock detection for the data-parallel group.
//!
//! Scope (be precise): this is the *detection and fail-fast* layer. The workspace's
//! `CollectiveCoordinator` is an in-process deterministic simulation, and the real NCCL
//! boundary is the injected `dp_all_reduce_fn` closure used by [`crate::DpConsensusEngine`].
//! A blocked NCCL call cannot be cancelled from safe Rust, so the contract here is:
//! detect (bounded wait), report which ranks are implicated, and let the caller abort the
//! process so the container exits instead of billing 8 GPUs behind a hung collective.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CollectiveError {
    #[error("collective did not complete within {0:?} (possible deadlock)")]
    Timeout(Duration),
    #[error("rank(s) failed: {0:?}")]
    RankFailed(Vec<(usize, String)>),
    #[error("rank(s) stalled (no progress within {timeout:?}): {ranks:?}")]
    RankStalled {
        ranks: Vec<usize>,
        timeout: Duration,
    },
    #[error("collective worker panicked or was dropped")]
    WorkerLost,
}

#[derive(Debug, Clone)]
struct RankState {
    last_progress: Instant,
    failure: Option<String>,
}

/// Tracks per-rank liveness. Time is passed in explicitly so tests are deterministic.
pub struct RankHealthMonitor {
    stall_timeout: Duration,
    ranks: Mutex<Vec<RankState>>,
}

impl RankHealthMonitor {
    pub fn new(world_size: usize, stall_timeout: Duration, now: Instant) -> Self {
        Self {
            stall_timeout,
            ranks: Mutex::new(vec![
                RankState {
                    last_progress: now,
                    failure: None,
                };
                world_size
            ]),
        }
    }

    pub fn world_size(&self) -> usize {
        self.ranks.lock().len()
    }

    /// A rank completed a step / collective. Ignored for ranks already marked failed
    /// (failure is sticky: a rank is not silently resurrected).
    pub fn report_progress(&self, rank: usize, now: Instant) {
        if let Some(r) = self.ranks.lock().get_mut(rank) {
            if r.failure.is_none() {
                r.last_progress = now;
            }
        }
    }

    /// Refresh every non-failed rank. Called while the group is idle (no work, so no
    /// collective is expected) so that stall time is only measured under load; sticky
    /// explicit failures are unaffected.
    pub fn report_all_progress(&self, now: Instant) {
        for r in self.ranks.lock().iter_mut() {
            if r.failure.is_none() {
                r.last_progress = now;
            }
        }
    }

    /// Explicit failure (CUDA error, process exit, NCCL error code). Sticky.
    pub fn report_failure(&self, rank: usize, reason: impl Into<String>) {
        if let Some(r) = self.ranks.lock().get_mut(rank) {
            if r.failure.is_none() {
                r.failure = Some(reason.into());
            }
        }
    }

    /// Failed ranks take precedence over stalled ranks.
    pub fn check(&self, now: Instant) -> Result<(), CollectiveError> {
        let ranks = self.ranks.lock();
        let failed: Vec<(usize, String)> = ranks
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.failure.clone().map(|f| (i, f)))
            .collect();
        if !failed.is_empty() {
            return Err(CollectiveError::RankFailed(failed));
        }
        let stalled: Vec<usize> = ranks
            .iter()
            .enumerate()
            .filter(|(_, r)| now.saturating_duration_since(r.last_progress) >= self.stall_timeout)
            .map(|(i, _)| i)
            .collect();
        if stalled.is_empty() {
            Ok(())
        } else {
            Err(CollectiveError::RankStalled {
                ranks: stalled,
                timeout: self.stall_timeout,
            })
        }
    }
}

/// Run a (possibly blocking) collective with a hard wait bound.
///
/// On timeout the worker thread is left detached (a blocked NCCL call cannot be
/// cancelled); the caller MUST treat `Timeout` as fatal for the group and abort/exit,
/// never retry on the same communicator.
pub fn run_collective_with_deadline<T, F>(deadline: Duration, f: F) -> Result<T, CollectiveError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    thread::Builder::new()
        .name("collective-deadline".into())
        .spawn(move || {
            let _ = tx.send(f());
        })
        .map_err(|_| CollectiveError::WorkerLost)?;
    match rx.recv_timeout(deadline) {
        Ok(v) => Ok(v),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(CollectiveError::Timeout(deadline)),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(CollectiveError::WorkerLost),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_ranks_pass() {
        let t0 = Instant::now();
        let m = RankHealthMonitor::new(8, Duration::from_secs(30), t0);
        assert_eq!(m.world_size(), 8);
        assert!(m.check(t0 + Duration::from_secs(29)).is_ok());
    }

    #[test]
    fn silent_rank_is_reported_stalled_others_not() {
        let t0 = Instant::now();
        let m = RankHealthMonitor::new(4, Duration::from_secs(30), t0);
        let later = t0 + Duration::from_secs(31);
        for r in [0, 1, 3] {
            m.report_progress(r, later);
        }
        assert_eq!(
            m.check(later),
            Err(CollectiveError::RankStalled {
                ranks: vec![2],
                timeout: Duration::from_secs(30)
            })
        );
    }

    #[test]
    fn explicit_failure_is_sticky_and_wins_over_stall() {
        let t0 = Instant::now();
        let m = RankHealthMonitor::new(2, Duration::from_secs(30), t0);
        m.report_failure(1, "CUDA_ERROR_ILLEGAL_ADDRESS");
        m.report_progress(1, t0 + Duration::from_secs(1)); // must not resurrect
        let err = m.check(t0 + Duration::from_secs(100)).unwrap_err();
        assert_eq!(
            err,
            CollectiveError::RankFailed(vec![(1, "CUDA_ERROR_ILLEGAL_ADDRESS".into())])
        );
    }

    #[test]
    fn out_of_range_rank_reports_are_ignored() {
        let t0 = Instant::now();
        let m = RankHealthMonitor::new(2, Duration::from_secs(1), t0);
        m.report_failure(99, "x");
        m.report_progress(99, t0);
        assert!(m.check(t0).is_ok());
    }

    #[test]
    fn collective_completes_within_deadline() {
        let v = run_collective_with_deadline(Duration::from_secs(2), || 7).unwrap();
        assert_eq!(v, 7);
    }

    #[test]
    fn hung_collective_times_out_instead_of_blocking_forever() {
        let start = Instant::now();
        let err = run_collective_with_deadline(Duration::from_millis(50), || {
            thread::sleep(Duration::from_secs(5));
        })
        .unwrap_err();
        assert_eq!(err, CollectiveError::Timeout(Duration::from_millis(50)));
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn panicking_collective_is_reported_as_worker_lost() {
        let err = run_collective_with_deadline(Duration::from_secs(2), || -> i32 {
            panic!("nccl abort")
        })
        .unwrap_err();
        assert_eq!(err, CollectiveError::WorkerLost);
    }
}
