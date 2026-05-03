//! Shared engine for tilted-run ReCom optimizers.
//!
//! A tilted run is a continuous Markov chain that always accepts plans with
//! better scores and accepts plans with worse scores according to a pluggable
//! [`AcceptanceRule`]. The two acceptance rules currently provided live in
//! sibling modules: `fixed::FixedAcceptance` accepts a worse plan with a fixed
//! probability, and `metropolis::MetropolisAcceptance` accepts a worse plan
//! with probability `exp(beta * delta)` where `delta` is signed by the
//! optimization direction. Unlike short bursts, there is no burst boundary
//! and no resetting to a global best state.
//!
//! Threading follows the same model as `multi_chain` in `run.rs`: worker
//! threads parallelize the tree-drawing step within a single sequential
//! tilted chain. Workers draw trees, score proposals, and decide accept/reject;
//! the main thread interleaves accepted proposals and rejections. Optional
//! output writers run on separate threads fed by bounded channels, so disk I/O
//! and serialization usually do not block proposal generation; the bound only
//! engages if a writer falls persistently behind the chain.
//!
//! See `docs/tilted_runs_spec.md` for full architecture documentation.
use super::super::{
    cut_edge_dist_pair, node_bound, random_split, uniform_dist_pair, RecomParams, RecomProposal,
    RecomVariant,
};
use crate::buffers::{
    graph_connected_buffered, ConnectivityBuffers, SpanningTreeBuffer, SplitBuffer, SubgraphBuffer,
};
use crate::graph::Graph;
use crate::objectives::IncrementalObjective;
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, RegionAwareSampler, SpanningTreeSampler, USTSampler};
use crate::stats::{ScoresWriter, SelfLoopCounts, SelfLoopReason, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

/// Capacity of the bounded channels that feed the stats and score writer
/// threads. Large enough to absorb transient writer stalls; small enough to
/// cap worst-case memory growth if a writer stays persistently behind.
const WRITER_CHANNEL_CAPACITY: usize = 128;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Decides whether to accept a proposal that does not improve the current score.
///
/// Implementors carry the parameters of a particular acceptance rule (for
/// example a fixed probability, or an inverse temperature). The trait is
/// `Copy` so each worker thread holds its own value without coordination.
pub trait AcceptanceRule: Send + Copy {
    /// Returns `true` to accept a worse-or-equal proposal, `false` to reject.
    ///
    /// `current` is the score of the current chain state and `proposed` is the
    /// score of the candidate. `maximize` is `true` when larger scores are
    /// improvements. Implementors may consult `rng` for stochastic decisions.
    /// This method is only called when the proposal is not strictly an
    /// improvement; strict improvements are always accepted by the engine.
    fn accept_worse(
        &self,
        current: f64,
        proposed: f64,
        maximize: bool,
        rng: &mut SmallRng,
    ) -> bool;
}

/// A unit of work sent from the main thread to a worker.
struct TiltedJobPacket {
    /// The accepted proposal to apply, or `None` if no state change.
    diff: Option<RecomProposal>,
    /// The current chain score, used by the worker for accept/reject decisions.
    current_score: f64,
    /// A sentinel used to kill the worker thread.
    terminate: bool,
}

/// The result of one round of work from a worker.
struct TiltedResultPacket {
    /// Number of tilted-criterion rejections (self-loops).
    rejections: usize,
    /// Accepted proposals from this round.
    proposals: Vec<ScoredProposal>,
}

/// A proposal accepted by a worker, plus the score it would produce.
struct ScoredProposal {
    /// Random ID used for deterministic interleaving (same technique as `run.rs`).
    id: u64,
    /// The proposal accepted by a worker.
    proposal: RecomProposal,
    /// The objective score after applying `proposal`.
    score: f64,
}

/// A chain-statistics write packet sent from the chain thread to the writer thread.
struct TiltedStatsPacket {
    /// The accepted proposal step.
    step: u64,
    /// The accepted proposal, or `None` only for the termination sentinel.
    proposal: Option<RecomProposal>,
    /// Tilted rejection counts since the previous accepted proposal.
    counts: SelfLoopCounts,
    /// A sentinel used to stop the writer thread.
    terminate: bool,
}

/// A score write packet sent from the chain thread to the score-writer thread.
struct TiltedScorePacket {
    /// First chain step represented by this packet.
    first_step: u64,
    /// Last chain step represented by this packet.
    last_step: u64,
    /// Objective score for each step in this packet.
    score: f64,
    /// Best score seen so far for each step in this packet.
    best_score: f64,
    /// Per-district scores to carry forward from this point on, if the chain
    /// state changed. `None` on pure-rejection runs, where the writer keeps
    /// reusing its previously cached vector.
    district_scores: Option<Vec<f64>>,
    /// A sentinel used to stop the writer thread.
    terminate: bool,
}

/// Reusable worker-side buffers for one tilted worker.
struct TiltedWorkerBuffers {
    /// Merged two-district subgraph buffer.
    subgraph: SubgraphBuffer,
    /// Spanning tree storage for the merged subgraph.
    spanning_tree: SpanningTreeBuffer,
    /// Random split workspace.
    split: SplitBuffer,
    /// Candidate proposal storage.
    proposal: RecomProposal,
    /// Restores the worker partition after temporary scoring.
    revert: RecomProposal,
    /// Scratch buffers for connectivity checks.
    connectivity: ConnectivityBuffers,
}

impl TiltedWorkerBuffers {
    /// Creates the reusable worker-side buffers.
    ///
    /// # Arguments
    ///
    /// * `graph_nodes` - Number of nodes in the full graph.
    /// * `buf_size` - Capacity for two-district subgraph/proposal buffers.
    /// * `balance_ub` - Soft upper bound used by reversible split buffers.
    ///
    /// # Returns
    ///
    /// A `TiltedWorkerBuffers` value with all reusable worker buffers allocated.
    fn new(graph_nodes: usize, buf_size: usize, balance_ub: u32) -> TiltedWorkerBuffers {
        TiltedWorkerBuffers {
            subgraph: SubgraphBuffer::new(graph_nodes, buf_size),
            spanning_tree: SpanningTreeBuffer::new(buf_size),
            split: SplitBuffer::new(buf_size, balance_ub as usize),
            proposal: RecomProposal::new_buffer(buf_size),
            revert: RecomProposal::new_buffer(buf_size),
            connectivity: ConnectivityBuffers::new(buf_size),
        }
    }
}

/// Main-thread state for the single sequential tilted chain.
struct TiltedMainState {
    /// Total chain steps processed, including accepted proposals and self-loops.
    step: u64,
    /// Canonical chain state.
    partition: Partition,
    /// Objective score of `partition`.
    current_score: f64,
    /// Best score seen so far.
    best_score: f64,
    /// Tilted rejection counts since the last accepted proposal.
    pending_counts: SelfLoopCounts,
}

impl TiltedMainState {
    /// Creates main-thread chain state from the initial partition and score.
    ///
    /// # Arguments
    ///
    /// * `partition` - Initial canonical chain state.
    /// * `current_score` - Objective score of `partition`.
    ///
    /// # Returns
    ///
    /// A `TiltedMainState` initialized at step 0 with the starting plan as the
    /// best-seen plan.
    fn new(partition: Partition, current_score: f64) -> TiltedMainState {
        TiltedMainState {
            step: 0,
            best_score: current_score,
            pending_counts: SelfLoopCounts::default(),
            partition,
            current_score,
        }
    }

    /// Records tilted rejection self-loops.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of consecutive tilted rejections to record.
    /// * `score_send` - Optional channel for per-step objective-score records.
    fn record_rejections(&mut self, count: usize, score_send: Option<&Sender<TiltedScorePacket>>) {
        if count == 0 {
            return;
        }
        let first_step = self.step + 1;
        self.step += count as u64;
        self.pending_counts
            .inc_by(SelfLoopReason::TiltedRejection, count);
        if let Some(send) = score_send {
            send.send(TiltedScorePacket {
                first_step,
                last_step: self.step,
                score: self.current_score,
                best_score: self.best_score,
                district_scores: None,
                terminate: false,
            })
            .unwrap();
        }
    }

    /// Applies an accepted proposal and sends output packets.
    ///
    /// # Arguments
    ///
    /// * `accepted` - Proposal selected by the main-thread interleaving step.
    /// * `maximize` - If true, larger scores are improvements; otherwise smaller scores are.
    /// * `stats_send` - Optional channel for accepted proposal statistics.
    /// * `score_send` - Optional channel for per-step objective-score records.
    fn apply_accepted_proposal(
        &mut self,
        accepted: &ScoredProposal,
        maximize: bool,
        stats_send: Option<&Sender<TiltedStatsPacket>>,
        score_send: Option<&Sender<TiltedScorePacket>>,
    ) {
        self.step += 1;
        self.partition.update(&accepted.proposal);
        self.current_score = accepted.score;

        let is_new_best = if maximize {
            accepted.score > self.best_score
        } else {
            accepted.score < self.best_score
        };
        if is_new_best {
            self.best_score = accepted.score;
        }
        if let Some(send) = stats_send {
            send.send(TiltedStatsPacket {
                step: self.step,
                proposal: Some(accepted.proposal.clone()),
                counts: std::mem::take(&mut self.pending_counts),
                terminate: false,
            })
            .unwrap();
        } else {
            self.pending_counts = SelfLoopCounts::default();
        }
        if let Some(send) = score_send {
            send.send(TiltedScorePacket {
                first_step: self.step,
                last_step: self.step,
                score: self.current_score,
                best_score: self.best_score,
                district_scores: None,
                terminate: false,
            })
            .unwrap();
        }
    }

    /// Sends pending terminal self-loops to the stats-writer thread if one exists.
    ///
    /// # Arguments
    ///
    /// * `stats_send` - Optional channel for accepted proposal statistics.
    fn send_pending_self_loops(&self, stats_send: Option<&Sender<TiltedStatsPacket>>) {
        if self.pending_counts.sum() == 0 {
            return;
        }
        if let Some(send) = stats_send {
            send.send(TiltedStatsPacket {
                step: self.step,
                proposal: None,
                counts: self.pending_counts.clone(),
                terminate: false,
            })
            .unwrap();
        }
    }
}

/// Builds the spanning-tree sampler for the supported variants.
///
/// # Arguments
///
/// * `params` - ReCom parameters containing the variant and optional region weights.
/// * `buf_size` - Capacity used by the sampler's internal buffers.
/// * `rng` - RNG used to seed samplers that need one (UST).
fn make_tilted_sampler(
    params: &RecomParams,
    buf_size: usize,
    rng: &mut SmallRng,
) -> Box<dyn SpanningTreeSampler> {
    match params.variant {
        RecomVariant::DistrictPairsRMST | RecomVariant::CutEdgesRMST => {
            Box::new(RMSTSampler::new(buf_size))
        }
        RecomVariant::DistrictPairsRegionAware | RecomVariant::CutEdgesRegionAware => {
            let region_weights = params
                .region_weights
                .clone()
                .expect("Region weights required for region-aware ReCom.");
            Box::new(RegionAwareSampler::new(buf_size, region_weights))
        }
        RecomVariant::DistrictPairsUST | RecomVariant::CutEdgesUST => {
            Box::new(USTSampler::new(buf_size, rng))
        }
        RecomVariant::Reversible => {
            panic!("Reversible ReCom is not supported by the tilted run optimizer.");
        }
    }
}

/// Draws the next candidate district pair for a tilted proposal. Cut-edge
/// variants pick a pair by sampling a cut edge (which always yields an
/// adjacent pair); district-pair variants sample a pair uniformly and return
/// `None` if it is non-adjacent, signaling the caller to retry.
fn sample_tilted_pair(
    graph: &Graph,
    partition: &mut Partition,
    variant: RecomVariant,
    rng: &mut SmallRng,
) -> Option<(usize, usize)> {
    match variant {
        RecomVariant::CutEdgesRMST
        | RecomVariant::CutEdgesUST
        | RecomVariant::CutEdgesRegionAware => Some(cut_edge_dist_pair(graph, partition, rng)),
        _ => uniform_dist_pair(graph, partition, rng),
    }
}

/// Copies the selected district pair into the reusable subgraph buffer.
///
/// Region-aware variants do not need region attributes copied onto the
/// subgraph: the sampler reads them off the parent graph via `raw_nodes`,
/// and the region-aware cut chooser now does the same via `subgraph_map`.
///
/// # Arguments
///
/// * `graph` - Full graph associated with `partition`.
/// * `partition` - Worker-local partition state.
/// * `buffers` - Worker buffers whose subgraph field is overwritten.
/// * `dist_a` - First selected district label.
/// * `dist_b` - Second selected district label.
fn fill_pair_subgraph(
    graph: &Graph,
    partition: &Partition,
    buffers: &mut TiltedWorkerBuffers,
    dist_a: usize,
    dist_b: usize,
) {
    partition.subgraph(graph, &mut buffers.subgraph, dist_a, dist_b);
}

/// Stores the current state of the two affected districts before scoring a proposal.
///
/// # Arguments
///
/// * `partition` - Worker-local partition before applying `proposal`.
/// * `proposal` - Candidate proposal whose district labels identify affected districts.
/// * `revert` - Proposal buffer populated with the current district state.
fn save_revert(partition: &Partition, proposal: &RecomProposal, revert: &mut RecomProposal) {
    revert.a_label = proposal.a_label;
    revert.b_label = proposal.b_label;
    revert.a_pop = partition.dist_pops[proposal.a_label];
    revert.b_pop = partition.dist_pops[proposal.b_label];
    revert.a_nodes.clear();
    revert
        .a_nodes
        .extend_from_slice(&partition.dist_nodes[proposal.a_label]);
    revert.b_nodes.clear();
    revert
        .b_nodes
        .extend_from_slice(&partition.dist_nodes[proposal.b_label]);
}

/// Starts a chain-statistics writer thread.
///
/// # Arguments
///
/// * `graph` - Writer-owned graph clone.
/// * `partition` - Writer-owned starting partition, updated from accepted proposals.
/// * `writer` - Stats writer receiving accepted proposals and self-loop counts.
/// * `recv` - Channel receiving asynchronous stats write packets.
fn start_tilted_stats_writer(
    graph: &Graph,
    mut partition: Partition,
    writer: &mut dyn StatsWriter,
    recv: Receiver<TiltedStatsPacket>,
) {
    writer.init(graph, &partition).unwrap();
    let mut next = recv.recv().unwrap();
    while !next.terminate {
        if let Some(proposal) = next.proposal {
            partition.update(&proposal);
            writer
                .step(next.step, graph, &partition, &proposal, &next.counts)
                .unwrap();
        } else {
            writer
                .self_loop(next.step, graph, &partition, &next.counts)
                .unwrap();
        }
        next = recv.recv().unwrap();
    }
    writer.close().unwrap();
}
/// Starts a score writer thread.
///
/// # Arguments
///
/// * `writer` - Score writer receiving per-step objective scores.
/// * `initial_score` - Objective score of the starting partition.
/// * `initial_district_scores` - Per-district score vector for the starting
///   partition. An empty vector switches the writer to the legacy three-column
///   `step,score,best_score` output; a non-empty vector emits
///   `step,score,best_score,d_0,...,d_{N-1}` and the writer caches it so
///   pure-rejection packets can replay the last-known district scores.
/// * `recv` - Channel receiving asynchronous score write packets.
fn start_tilted_score_writer(
    writer: &mut ScoresWriter,
    initial_score: f64,
    initial_district_scores: Vec<f64>,
    recv: Receiver<TiltedScorePacket>,
) {
    writer
        .init(initial_score, &initial_district_scores)
        .unwrap();
    let mut last_districts = initial_district_scores;
    let mut next = recv.recv().unwrap();
    while !next.terminate {
        if let Some(new_districts) = next.district_scores.take() {
            last_districts = new_districts;
        }
        for step in next.first_step..=next.last_step {
            writer
                .step(step, next.score, next.best_score, &last_districts)
                .unwrap();
        }
        next = recv.recv().unwrap();
    }
    writer.close().unwrap();
}

/// Sends the next canonical chain state to every worker.
///
/// # Arguments
///
/// * `job_sends` - Worker job channels.
/// * `diff` - Accepted proposal to broadcast, or `None` for an all-self-loop round.
/// * `score` - Current canonical chain score.
fn send_tilted_jobs(
    job_sends: &[Sender<TiltedJobPacket>],
    diff: Option<&RecomProposal>,
    score: f64,
) {
    for job in job_sends.iter() {
        job.send(TiltedJobPacket {
            diff: diff.cloned(),
            current_score: score,
            terminate: false,
        })
        .unwrap();
    }
}

/// Stops all tilted-run worker threads.
///
/// # Arguments
///
/// * `job_sends` - Worker job channels.
fn stop_tilted_workers(job_sends: &[Sender<TiltedJobPacket>]) {
    for job in job_sends.iter() {
        job.send(TiltedJobPacket {
            diff: None,
            current_score: 0.0,
            terminate: true,
        })
        .unwrap();
    }
}

/// Blocks until each worker has returned exactly one round of tilted output.
///
/// # Arguments
///
/// * `result_recv` - Shared result channel from all workers.
/// * `n_threads` - Number of worker packets to collect.
///
/// # Returns
///
/// The total number of tilted rejections and all worker-accepted proposals for
/// this round.
fn collect_tilted_results(
    result_recv: &Receiver<TiltedResultPacket>,
    n_threads: usize,
) -> (usize, Vec<ScoredProposal>) {
    let mut rejections = 0;
    let mut proposals = Vec::<ScoredProposal>::new();

    for _ in 0..n_threads {
        let packet = result_recv.recv().unwrap();
        rejections += packet.rejections;
        proposals.extend(packet.proposals);
    }

    (rejections, proposals)
}

/// Interleaves one batch of worker rejections/proposals into the sequential chain.
///
/// Rejections increment the chain step without changing state. Accepted proposals
/// and score events are sent to asynchronous writer threads.
///
/// # Arguments
///
/// * `state` - Main-thread canonical chain state.
/// * `loops` - Number of worker tilted rejections to interleave as self-loops.
/// * `proposals` - Worker-accepted proposals to interleave with self-loops.
/// * `params` - ReCom parameters containing the target step count.
/// * `rng` - Main-thread RNG used for event interleaving and proposal selection.
/// * `job_sends` - Worker job channels for broadcasting state updates.
/// * `maximize` - If true, larger scores are improvements; otherwise smaller scores are.
/// * `stats_send` - Optional channel for accepted proposal statistics.
/// * `score_send` - Optional channel for per-step objective-score records.
fn interleave_tilted_round(
    state: &mut TiltedMainState,
    mut loops: usize,
    mut proposals: Vec<ScoredProposal>,
    params: &RecomParams,
    rng: &mut SmallRng,
    job_sends: &[Sender<TiltedJobPacket>],
    maximize: bool,
    stats_send: Option<&Sender<TiltedStatsPacket>>,
    score_send: Option<&Sender<TiltedScorePacket>>,
) {
    if proposals.is_empty() {
        let remaining = (params.num_steps - state.step) as usize;
        state.record_rejections(loops.min(remaining), score_send);
        send_tilted_jobs(job_sends, None, state.current_score);
        return;
    }

    proposals.sort_by_key(|proposal| proposal.id);
    let mut total = loops + proposals.len();
    while total > 0 && state.step < params.num_steps {
        let event = rng.random_range(0..total);
        if event < loops {
            state.record_rejections(1, score_send);
            loops -= 1;
            total -= 1;
            continue;
        }

        let accepted = &proposals[rng.random_range(0..proposals.len())];
        state.apply_accepted_proposal(accepted, maximize, stats_send, score_send);
        send_tilted_jobs(job_sends, Some(&accepted.proposal), state.current_score);
        break; // need new round (state changed)
    }
}

/// Runs the main-thread collection/interleaving loop after workers are spawned.
///
/// # Arguments
///
/// * `state` - Main-thread canonical chain state.
/// * `params` - ReCom parameters containing the target step count.
/// * `n_threads` - Number of worker packets to collect per round.
/// * `result_recv` - Shared result channel from all workers.
/// * `job_sends` - Worker job channels.
/// * `rng` - Main-thread RNG used for interleaving.
/// * `maximize` - If true, larger scores are improvements; otherwise smaller scores are.
/// * `stats_send` - Optional channel for accepted proposal statistics.
/// * `score_send` - Optional channel for per-step objective-score records.
/// * `progress_bar` - Optional progress bar tracking total chain steps.
fn run_tilted_main_loop(
    state: &mut TiltedMainState,
    params: &RecomParams,
    n_threads: usize,
    result_recv: &Receiver<TiltedResultPacket>,
    job_sends: &[Sender<TiltedJobPacket>],
    rng: &mut SmallRng,
    maximize: bool,
    stats_send: Option<&Sender<TiltedStatsPacket>>,
    score_send: Option<&Sender<TiltedScorePacket>>,
    progress_bar: Option<&ProgressBar>,
) {
    if params.num_steps > 0 {
        send_tilted_jobs(job_sends, None, state.current_score);
    }

    let progress_chunk = (params.num_steps / 1000 + 1).min(1000);
    let mut last_drawn = state.step;
    while state.step < params.num_steps {
        let (loops, proposals) = collect_tilted_results(result_recv, n_threads);
        interleave_tilted_round(
            state, loops, proposals, params, rng, job_sends, maximize, stats_send, score_send,
        );
        if let Some(progress_bar) = progress_bar {
            if state.step - last_drawn >= progress_chunk || state.step == params.num_steps {
                progress_bar.set_position(state.step);
                last_drawn = state.step;
            }
        }
    }
}

/// Draws one round of tilted output for a closure-scoring worker.
fn draw_tilted_result<R: AcceptanceRule>(
    graph: &Graph,
    partition: &mut Partition,
    params: &RecomParams,
    buffers: &mut TiltedWorkerBuffers,
    st_sampler: &mut Box<dyn SpanningTreeSampler>,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    current_score: f64,
    rule: R,
    maximize: bool,
    rng: &mut SmallRng,
) -> TiltedResultPacket {
    loop {
        let Some((dist_a, dist_b)) = sample_tilted_pair(graph, partition, params.variant, rng)
        else {
            continue;
        };

        fill_pair_subgraph(graph, partition, buffers, dist_a, dist_b);

        if !graph_connected_buffered(&buffers.subgraph.graph, &mut buffers.connectivity) {
            continue;
        }

        st_sampler.random_spanning_tree_with_parent(
            &buffers.subgraph.graph,
            graph,
            &buffers.subgraph.raw_nodes,
            &mut buffers.spanning_tree,
            rng,
        );
        let split = random_split(
            &buffers.subgraph.graph,
            graph,
            rng,
            &buffers.spanning_tree.st,
            dist_a,
            dist_b,
            &mut buffers.split,
            &mut buffers.proposal,
            &buffers.subgraph.raw_nodes,
            params,
        );
        if split.is_err() {
            continue;
        }

        save_revert(partition, &buffers.proposal, &mut buffers.revert);
        partition.update(&buffers.proposal);
        let new_score = obj_fn(graph, partition);
        partition.update(&buffers.revert);

        let is_improvement = if maximize {
            new_score >= current_score
        } else {
            new_score <= current_score
        };
        if is_improvement || rule.accept_worse(current_score, new_score, maximize, rng) {
            return TiltedResultPacket {
                rejections: 0,
                proposals: vec![ScoredProposal {
                    id: rng.random::<u64>(),
                    proposal: buffers.proposal.clone(),
                    score: new_score,
                }],
            };
        }

        return TiltedResultPacket {
            rejections: 1,
            proposals: Vec::new(),
        };
    }
}

/// Starts a closure-scoring tilted worker thread.
fn start_tilted_worker<R: AcceptanceRule>(
    graph: &Graph,
    mut partition: Partition,
    params: RecomParams,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    rule: R,
    maximize: bool,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<TiltedJobPacket>,
    result_send: Sender<TiltedResultPacket>,
) {
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut buffers = TiltedWorkerBuffers::new(n, buf_size, params.balance_ub);
    let mut st_sampler = make_tilted_sampler(&params, buf_size, &mut rng);

    let mut next: TiltedJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        if let Some(diff) = next.diff {
            partition.update(&diff);
        }

        let result = draw_tilted_result(
            graph,
            &mut partition,
            &params,
            &mut buffers,
            &mut st_sampler,
            obj_fn,
            next.current_score,
            rule,
            maximize,
            &mut rng,
        );

        result_send.send(result).unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Runs a tilted-run optimizer with parallel tree drawing.
///
/// Worker threads draw spanning trees in parallel from the same partition state,
/// score each proposal via `obj_fn`, and decide accept/reject with `rule`. The
/// main thread interleaves accepted proposals and rejections (exactly like
/// `multi_chain` in `run.rs`), maintaining a single sequential chain.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The starting partition.
/// * `params` - The chain parameters. `params.num_steps` is the total number
///   of chain steps (accepted + rejected).
/// * `n_threads` - The number of worker threads for parallel tree drawing.
/// * `obj_fn` - The objective function.
/// * `rule` - The acceptance rule applied to non-improving proposals.
/// * `maximize` - If true, higher scores are better. If false, lower scores are better.
/// * `stats_writer` - Optional asynchronous writer for accepted chain proposals
///   and self-loop counts.
/// * `score_writer` - Optional asynchronous writer for per-step objective scores.
/// * `show_progress` - If true, show a progress bar tracking total chain steps.
pub fn multi_tilted_runs_with_writer<R: AcceptanceRule>(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    rule: R,
    maximize: bool,
    stats_writer: Option<&mut dyn StatsWriter>,
    score_writer: Option<&mut ScoresWriter>,
    show_progress: bool,
) -> Result<Partition, String> {
    if n_threads == 0 {
        return Err("n_threads must be at least 1".to_string());
    }

    let node_ub = node_bound(&graph.pops, params.max_pop);

    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads {
        let (s, r): (Sender<TiltedJobPacket>, Receiver<TiltedJobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    let (result_send, result_recv): (Sender<TiltedResultPacket>, Receiver<TiltedResultPacket>) =
        unbounded();

    let current_score = obj_fn(graph, &partition);
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);
    let progress_bar = if show_progress {
        let progress_bar = ProgressBar::with_draw_target(
            Some(params.num_steps),
            indicatif::ProgressDrawTarget::stdout_with_hz(1),
        );
        progress_bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>10}/{len} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        Some(progress_bar)
    } else {
        None
    };

    let scoped_result = scope(|scope| -> Result<Partition, String> {
        let progress_bar_ref = progress_bar.as_ref();
        let stats_send = if let Some(writer) = stats_writer {
            let (send, recv): (Sender<TiltedStatsPacket>, Receiver<TiltedStatsPacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn({
                let partition = partition.clone();
                move |_| start_tilted_stats_writer(graph, partition, writer, recv)
            });
            Some(send)
        } else {
            None
        };
        let score_send = if let Some(writer) = score_writer {
            let (send, recv): (Sender<TiltedScorePacket>, Receiver<TiltedScorePacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope
                .spawn(move |_| start_tilted_score_writer(writer, current_score, Vec::new(), recv));
            Some(send)
        } else {
            None
        };

        for t_idx in 0..n_threads {
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();
            let partition = partition.clone();

            scope.spawn(move |_| {
                start_tilted_worker(
                    graph,
                    partition,
                    params.clone(),
                    obj_fn,
                    rule,
                    maximize,
                    rng_seed,
                    node_ub,
                    job_recv,
                    result_send,
                );
            });
        }

        let mut state = TiltedMainState::new(partition, current_score);
        run_tilted_main_loop(
            &mut state,
            params,
            n_threads,
            &result_recv,
            &job_sends,
            &mut rng,
            maximize,
            stats_send.as_ref(),
            score_send.as_ref(),
            progress_bar_ref,
        );

        state.send_pending_self_loops(stats_send.as_ref());
        if let Some(send) = &stats_send {
            send.send(TiltedStatsPacket {
                step: 0,
                proposal: None,
                counts: SelfLoopCounts::default(),
                terminate: true,
            })
            .unwrap();
        }
        if let Some(send) = &score_send {
            send.send(TiltedScorePacket {
                first_step: 0,
                last_step: 0,
                score: 0.0,
                best_score: 0.0,
                district_scores: None,
                terminate: true,
            })
            .unwrap();
        }
        stop_tilted_workers(&job_sends);
        Ok(state.partition)
    });

    if let Some(progress_bar) = progress_bar {
        progress_bar.set_position(params.num_steps);
        progress_bar.finish_and_clear();
    }

    match scoped_result {
        Ok(inner) => inner,
        Err(_panic) => Err("multi_tilted_runs panicked in a worker thread".to_string()),
    }
}

/// Runs a tilted-run optimizer without writing chain statistics or scores.
///
/// See [`multi_tilted_runs_with_writer`] for the full implementation and argument
/// documentation.
pub fn multi_tilted_runs<R: AcceptanceRule>(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    rule: R,
    maximize: bool,
    show_progress: bool,
) -> Result<Partition, String> {
    multi_tilted_runs_with_writer(
        graph,
        partition,
        params,
        n_threads,
        obj_fn,
        rule,
        maximize,
        None,
        None,
        show_progress,
    )
}

// =====================================================================
// Incremental tilted runner
// =====================================================================
//
// Mirrors the closure-based runner above, but uses an `IncrementalObjective`
// to avoid full-district rescoring per candidate. Workers and the main thread
// each own a cached `ObjectiveState`; worker scoring queries it directly
// without mutating `partition` or `state`. When a diff is received, both the
// worker's partition and its state are updated via `objective.apply_proposal`.

/// Incremental-mode worker buffers (no revert buffer needed).
struct TiltedWorkerBuffersIncremental {
    subgraph: SubgraphBuffer,
    spanning_tree: SpanningTreeBuffer,
    split: SplitBuffer,
    proposal: RecomProposal,
    connectivity: ConnectivityBuffers,
}

impl TiltedWorkerBuffersIncremental {
    fn new(graph_nodes: usize, buf_size: usize, balance_ub: u32) -> Self {
        Self {
            subgraph: SubgraphBuffer::new(graph_nodes, buf_size),
            spanning_tree: SpanningTreeBuffer::new(buf_size),
            split: SplitBuffer::new(buf_size, balance_ub as usize),
            proposal: RecomProposal::new_buffer(buf_size),
            connectivity: ConnectivityBuffers::new(buf_size),
        }
    }
}

/// Main-thread state for the incremental tilted chain.
struct TiltedMainStateIncremental<S: Send + Clone> {
    step: u64,
    partition: Partition,
    objective_state: S,
    current_score: f64,
    best_score: f64,
    pending_counts: SelfLoopCounts,
}

impl<S: Send + Clone> TiltedMainStateIncremental<S> {
    fn new(partition: Partition, objective_state: S, current_score: f64) -> Self {
        Self {
            step: 0,
            best_score: current_score,
            pending_counts: SelfLoopCounts::default(),
            partition,
            objective_state,
            current_score,
        }
    }

    fn record_rejections(&mut self, count: usize, score_send: Option<&Sender<TiltedScorePacket>>) {
        if count == 0 {
            return;
        }
        let first_step = self.step + 1;
        self.step += count as u64;
        self.pending_counts
            .inc_by(SelfLoopReason::TiltedRejection, count);
        if let Some(send) = score_send {
            send.send(TiltedScorePacket {
                first_step,
                last_step: self.step,
                score: self.current_score,
                best_score: self.best_score,
                district_scores: None,
                terminate: false,
            })
            .unwrap();
        }
    }

    fn apply_accepted_proposal<O: IncrementalObjective<State = S>>(
        &mut self,
        graph: &Graph,
        objective: &O,
        accepted: &ScoredProposal,
        maximize: bool,
        stats_send: Option<&Sender<TiltedStatsPacket>>,
        score_send: Option<&Sender<TiltedScorePacket>>,
    ) {
        self.step += 1;
        objective.apply_proposal(graph, &mut self.objective_state, &accepted.proposal);
        self.partition
            .update_with_dist_adj(&accepted.proposal, graph);
        self.current_score = accepted.score;

        let is_new_best = if maximize {
            accepted.score > self.best_score
        } else {
            accepted.score < self.best_score
        };
        if is_new_best {
            self.best_score = accepted.score;
        }
        if let Some(send) = stats_send {
            send.send(TiltedStatsPacket {
                step: self.step,
                proposal: Some(accepted.proposal.clone()),
                counts: std::mem::take(&mut self.pending_counts),
                terminate: false,
            })
            .unwrap();
        } else {
            self.pending_counts = SelfLoopCounts::default();
        }
        if let Some(send) = score_send {
            send.send(TiltedScorePacket {
                first_step: self.step,
                last_step: self.step,
                score: self.current_score,
                best_score: self.best_score,
                district_scores: Some(objective.district_scores(&self.objective_state)),
                terminate: false,
            })
            .unwrap();
        }
    }

    fn send_pending_self_loops(&self, stats_send: Option<&Sender<TiltedStatsPacket>>) {
        if self.pending_counts.sum() == 0 {
            return;
        }
        if let Some(send) = stats_send {
            send.send(TiltedStatsPacket {
                step: self.step,
                proposal: None,
                counts: self.pending_counts.clone(),
                terminate: false,
            })
            .unwrap();
        }
    }
}

/// Fills the worker's subgraph buffer for the selected district pair.
///
/// The incremental objective reads its inputs from the parent graph via cached
/// state, the region-aware sampler reads region weights off the parent graph
/// via `raw_nodes`, and the region-aware cut chooser reads region attributes
/// off the parent graph via `subgraph_map`. The pair subgraph therefore never
/// needs its own copy of node attribute columns.
fn fill_pair_subgraph_incremental(
    graph: &Graph,
    partition: &Partition,
    buffers: &mut TiltedWorkerBuffersIncremental,
    dist_a: usize,
    dist_b: usize,
) {
    partition.subgraph(graph, &mut buffers.subgraph, dist_a, dist_b);
}

fn start_tilted_worker_incremental<O, R>(
    graph: &Graph,
    mut partition: Partition,
    mut objective_state: O::State,
    params: RecomParams,
    objective: O,
    rule: R,
    maximize: bool,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<TiltedJobPacket>,
    result_send: Sender<TiltedResultPacket>,
) where
    O: IncrementalObjective,
    R: AcceptanceRule,
{
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut buffers = TiltedWorkerBuffersIncremental::new(n, buf_size, params.balance_ub);
    let mut st_sampler = make_tilted_sampler(&params, buf_size, &mut rng);

    let mut next: TiltedJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        if let Some(diff) = &next.diff {
            objective.apply_proposal(graph, &mut objective_state, diff);
            partition.update_with_dist_adj(diff, graph);
        }

        let result = draw_tilted_result_incremental(
            graph,
            &mut partition,
            &params,
            &mut buffers,
            &mut st_sampler,
            &objective,
            &objective_state,
            next.current_score,
            rule,
            maximize,
            &mut rng,
        );

        result_send.send(result).unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Draws one tilted result for an incremental worker. Unlike the closure-based
/// worker, this does not need to apply/revert the proposal to score -- it
/// queries the incremental objective directly.
fn draw_tilted_result_incremental<O, R>(
    graph: &Graph,
    partition: &mut Partition,
    params: &RecomParams,
    buffers: &mut TiltedWorkerBuffersIncremental,
    st_sampler: &mut Box<dyn SpanningTreeSampler>,
    objective: &O,
    objective_state: &O::State,
    current_score: f64,
    rule: R,
    maximize: bool,
    rng: &mut SmallRng,
) -> TiltedResultPacket
where
    O: IncrementalObjective,
    R: AcceptanceRule,
{
    loop {
        let Some((dist_a, dist_b)) = sample_tilted_pair(graph, partition, params.variant, rng)
        else {
            continue;
        };

        fill_pair_subgraph_incremental(graph, partition, buffers, dist_a, dist_b);

        if !graph_connected_buffered(&buffers.subgraph.graph, &mut buffers.connectivity) {
            continue;
        }

        st_sampler.random_spanning_tree_with_parent(
            &buffers.subgraph.graph,
            graph,
            &buffers.subgraph.raw_nodes,
            &mut buffers.spanning_tree,
            rng,
        );
        let split = random_split(
            &buffers.subgraph.graph,
            graph,
            rng,
            &buffers.spanning_tree.st,
            dist_a,
            dist_b,
            &mut buffers.split,
            &mut buffers.proposal,
            &buffers.subgraph.raw_nodes,
            params,
        );
        if split.is_err() {
            continue;
        }

        let new_score = objective.score_proposal(graph, objective_state, &buffers.proposal);

        let is_improvement = if maximize {
            new_score >= current_score
        } else {
            new_score <= current_score
        };
        if is_improvement || rule.accept_worse(current_score, new_score, maximize, rng) {
            return TiltedResultPacket {
                rejections: 0,
                proposals: vec![ScoredProposal {
                    id: rng.random::<u64>(),
                    proposal: buffers.proposal.clone(),
                    score: new_score,
                }],
            };
        }

        return TiltedResultPacket {
            rejections: 1,
            proposals: Vec::new(),
        };
    }
}

fn interleave_tilted_round_incremental<O, S>(
    graph: &Graph,
    objective: &O,
    state: &mut TiltedMainStateIncremental<S>,
    mut loops: usize,
    mut proposals: Vec<ScoredProposal>,
    params: &RecomParams,
    rng: &mut SmallRng,
    job_sends: &[Sender<TiltedJobPacket>],
    maximize: bool,
    stats_send: Option<&Sender<TiltedStatsPacket>>,
    score_send: Option<&Sender<TiltedScorePacket>>,
) where
    O: IncrementalObjective<State = S>,
    S: Send + Clone,
{
    if proposals.is_empty() {
        let remaining = (params.num_steps - state.step) as usize;
        state.record_rejections(loops.min(remaining), score_send);
        send_tilted_jobs(job_sends, None, state.current_score);
        return;
    }

    proposals.sort_by_key(|proposal| proposal.id);
    let mut total = loops + proposals.len();
    while total > 0 && state.step < params.num_steps {
        let event = rng.random_range(0..total);
        if event < loops {
            state.record_rejections(1, score_send);
            loops -= 1;
            total -= 1;
            continue;
        }

        let accepted = &proposals[rng.random_range(0..proposals.len())];
        state.apply_accepted_proposal(graph, objective, accepted, maximize, stats_send, score_send);
        send_tilted_jobs(job_sends, Some(&accepted.proposal), state.current_score);
        break;
    }
}

fn run_tilted_main_loop_incremental<O, S>(
    graph: &Graph,
    objective: &O,
    state: &mut TiltedMainStateIncremental<S>,
    params: &RecomParams,
    n_threads: usize,
    result_recv: &Receiver<TiltedResultPacket>,
    job_sends: &[Sender<TiltedJobPacket>],
    rng: &mut SmallRng,
    maximize: bool,
    stats_send: Option<&Sender<TiltedStatsPacket>>,
    score_send: Option<&Sender<TiltedScorePacket>>,
    progress_bar: Option<&ProgressBar>,
) where
    O: IncrementalObjective<State = S>,
    S: Send + Clone,
{
    if params.num_steps > 0 {
        send_tilted_jobs(job_sends, None, state.current_score);
    }

    let progress_chunk = (params.num_steps / 1000 + 1).min(1000);
    let mut last_drawn = state.step;
    while state.step < params.num_steps {
        let (loops, proposals) = collect_tilted_results(result_recv, n_threads);
        interleave_tilted_round_incremental(
            graph, objective, state, loops, proposals, params, rng, job_sends, maximize,
            stats_send, score_send,
        );
        if let Some(progress_bar) = progress_bar {
            if state.step - last_drawn >= progress_chunk || state.step == params.num_steps {
                progress_bar.set_position(state.step);
                last_drawn = state.step;
            }
        }
    }
}

/// Runs a tilted-run optimizer with incremental objective scoring.
///
/// Behaves identically to [`multi_tilted_runs_with_writer`] but scores
/// candidate proposals via [`IncrementalObjective::score_proposal`] instead
/// of rescoring the entire partition on each call. Each worker (and the main
/// thread) owns a clone of the objective's cached state, updated in lockstep
/// with the partition whenever a proposal is accepted.
pub fn multi_tilted_runs_incremental_with_writer<O, R>(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    objective: O,
    rule: R,
    maximize: bool,
    stats_writer: Option<&mut dyn StatsWriter>,
    score_writer: Option<&mut ScoresWriter>,
    show_progress: bool,
) -> Result<Partition, String>
where
    O: IncrementalObjective + Send + Clone + 'static,
    O::State: Send + Clone + 'static,
    R: AcceptanceRule + 'static,
{
    if n_threads == 0 {
        return Err("n_threads must be at least 1".to_string());
    }

    let node_ub = node_bound(&graph.pops, params.max_pop);

    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads {
        let (s, r): (Sender<TiltedJobPacket>, Receiver<TiltedJobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    let (result_send, result_recv): (Sender<TiltedResultPacket>, Receiver<TiltedResultPacket>) =
        unbounded();

    let initial_state = objective.init(graph, &partition);
    let current_score = objective.score_state(&initial_state);
    let initial_district_scores = objective.district_scores(&initial_state);
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);
    let progress_bar = if show_progress {
        let progress_bar = ProgressBar::with_draw_target(
            Some(params.num_steps),
            indicatif::ProgressDrawTarget::stdout_with_hz(1),
        );
        progress_bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>10}/{len} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        Some(progress_bar)
    } else {
        None
    };

    let scoped_result = scope(|scope| -> Result<Partition, String> {
        let progress_bar_ref = progress_bar.as_ref();
        let stats_send = if let Some(writer) = stats_writer {
            let (send, recv): (Sender<TiltedStatsPacket>, Receiver<TiltedStatsPacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn({
                let partition = partition.clone();
                move |_| start_tilted_stats_writer(graph, partition, writer, recv)
            });
            Some(send)
        } else {
            None
        };
        let score_send = if let Some(writer) = score_writer {
            let (send, recv): (Sender<TiltedScorePacket>, Receiver<TiltedScorePacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn(move |_| {
                start_tilted_score_writer(writer, current_score, initial_district_scores, recv)
            });
            Some(send)
        } else {
            None
        };

        for t_idx in 0..n_threads {
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();
            let partition = partition.clone();
            let worker_state = initial_state.clone();
            let worker_obj = objective.clone();

            scope.spawn(move |_| {
                start_tilted_worker_incremental::<O, R>(
                    graph,
                    partition,
                    worker_state,
                    params.clone(),
                    worker_obj,
                    rule,
                    maximize,
                    rng_seed,
                    node_ub,
                    job_recv,
                    result_send,
                );
            });
        }

        let mut state =
            TiltedMainStateIncremental::<O::State>::new(partition, initial_state, current_score);
        run_tilted_main_loop_incremental(
            graph,
            &objective,
            &mut state,
            params,
            n_threads,
            &result_recv,
            &job_sends,
            &mut rng,
            maximize,
            stats_send.as_ref(),
            score_send.as_ref(),
            progress_bar_ref,
        );

        state.send_pending_self_loops(stats_send.as_ref());
        if let Some(send) = &stats_send {
            send.send(TiltedStatsPacket {
                step: 0,
                proposal: None,
                counts: SelfLoopCounts::default(),
                terminate: true,
            })
            .unwrap();
        }
        if let Some(send) = &score_send {
            send.send(TiltedScorePacket {
                first_step: 0,
                last_step: 0,
                score: 0.0,
                best_score: 0.0,
                district_scores: None,
                terminate: true,
            })
            .unwrap();
        }
        stop_tilted_workers(&job_sends);
        Ok(state.partition)
    });

    if let Some(progress_bar) = progress_bar {
        progress_bar.set_position(params.num_steps);
        progress_bar.finish_and_clear();
    }

    match scoped_result {
        Ok(inner) => inner,
        Err(_panic) => Err("multi_tilted_runs_incremental panicked in a worker thread".to_string()),
    }
}

/// Runs the incremental tilted-run optimizer without writing chain statistics
/// or scores.
pub fn multi_tilted_runs_incremental<O, R>(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    objective: O,
    rule: R,
    maximize: bool,
    show_progress: bool,
) -> Result<Partition, String>
where
    O: IncrementalObjective + Send + Clone + 'static,
    O::State: Send + Clone + 'static,
    R: AcceptanceRule + 'static,
{
    multi_tilted_runs_incremental_with_writer(
        graph,
        partition,
        params,
        n_threads,
        objective,
        rule,
        maximize,
        None,
        None,
        show_progress,
    )
}
