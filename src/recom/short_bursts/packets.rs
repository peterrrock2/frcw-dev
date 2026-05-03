//! Wire format and channel-dispatch helpers shared between the short-bursts
//! engine and its writer threads. Packet types and dispatch helpers are
//! private to the `short_bursts` module via `pub(super)`.

use crossbeam_channel::Sender;

use crate::partition::Partition;

/// A unit of work sent from the main thread to a worker.
pub(super) struct BurstJobPacket {
    /// The number of accepted steps to sample in this burst.
    pub(super) n_steps: usize,
    /// The full canonical partition for the next burst, when the main thread
    /// has rolled forward to a new global best. `None` means continue from the
    /// worker's current state.
    pub(super) diff: Option<Partition>,
    /// A sentinel used to kill the worker thread.
    pub(super) terminate: bool,
}

/// The result of one burst from a worker.
pub(super) struct BurstResult {
    /// Every accepted step in the burst, in chronological order. Each entry is
    /// `(partition_after_step, score_of_partition)`. Empty when
    /// `collect_trace = false` was passed to the worker.
    pub(super) all_steps: Vec<(Partition, f64)>,
    /// The burst-end best partition and score (the merge candidate). `None` if
    /// the burst produced no accepted steps.
    pub(super) best: Option<(Partition, f64)>,
}

/// A chain-statistics write packet sent from the main thread to the stats
/// writer thread.
pub(super) struct BurstStatsPacket {
    /// Sequential sample number for the writer.
    pub(super) step: u64,
    /// Partition to emit. `None` only for the termination sentinel.
    pub(super) partition: Option<Partition>,
    /// A sentinel used to stop the writer thread.
    pub(super) terminate: bool,
}

/// A score-record write packet sent from the main thread to the score writer
/// thread.
pub(super) struct BurstScorePacket {
    /// Chain step at which this score event occurred.
    pub(super) step: u64,
    /// Objective score at this event.
    pub(super) score: f64,
    /// Best score seen so far.
    pub(super) best_score: f64,
    /// Per-district scores to carry forward from this point on, if the chain
    /// state changed. `None` reuses the writer's previously cached vector.
    pub(super) district_scores: Option<Vec<f64>>,
    /// A sentinel used to stop the writer thread.
    pub(super) terminate: bool,
}

/// Sends a batch of work to a short-bursts worker thread.
pub(super) fn send_burst_batch(
    send: &Sender<BurstJobPacket>,
    diff: Option<Partition>,
    burst_length: usize,
) {
    send.send(BurstJobPacket {
        n_steps: burst_length,
        diff,
        terminate: false,
    })
    .unwrap();
}

/// Stops a short-bursts worker thread.
pub(super) fn terminate_burst_worker(send: &Sender<BurstJobPacket>) {
    send.send(BurstJobPacket {
        n_steps: 0,
        diff: None,
        terminate: true,
    })
    .unwrap();
}
