//! ReCom-based optimization using tilted runs.
//!
//! A tilted run is a continuous Markov chain that always accepts plans with
//! better scores and accepts plans with worse scores with a fixed probability
//! `accept_worse_prob`. Unlike short bursts, there is no burst boundary and
//! no resetting to a global best state.
//!
//! Threading follows the same model as `multi_chain` in `run.rs`: worker
//! threads parallelize the tree-drawing step within a single sequential
//! tilted chain. Workers draw trees, score proposals, and decide accept/reject;
//! the main thread interleaves accepted proposals and rejections. Optional
//! output writers run on separate threads fed by unbounded channels, so disk I/O
//! and serialization do not block proposal generation.
//!
//! See `docs/tilted_runs_spec.md` for full architecture documentation.
use super::{
    node_bound, random_split, uniform_dist_pair, RecomParams, RecomProposal, RecomVariant,
};
use crate::buffers::{SpanningTreeBuffer, SplitBuffer, SubgraphBuffer};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, RegionAwareSampler, SpanningTreeSampler};
use crate::stats::{ScoresWriter, SelfLoopCounts, SelfLoopReason, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{unbounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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

    /// Sends the current score state to the score-writer thread if one exists.
    ///
    /// # Arguments
    ///
    /// * `score_send` - Optional channel for per-step objective-score records.
    fn send_score(&self, score_send: Option<&Sender<TiltedScorePacket>>) {
        if let Some(send) = score_send {
            send.send(TiltedScorePacket {
                first_step: self.step,
                last_step: self.step,
                score: self.current_score,
                best_score: self.best_score,
                terminate: false,
            })
            .unwrap();
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
                counts: self.pending_counts.clone(),
                terminate: false,
            })
            .unwrap();
        }
        self.pending_counts = SelfLoopCounts::default();
        self.send_score(score_send);
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

/// Returns true iff `graph` has exactly one connected component.
///
/// # Arguments
///
/// * `graph` - Graph to test for connectivity.
///
/// # Returns
///
/// `true` if every node in `graph` is reachable from node 0; otherwise `false`.
fn graph_connected(graph: &Graph) -> bool {
    let n = graph.pops.len();
    if n <= 1 {
        return true;
    }
    let mut visited = vec![false; n];
    let mut stack = Vec::<usize>::with_capacity(n);
    visited[0] = true;
    stack.push(0);
    let mut seen = 1;
    while let Some(node) = stack.pop() {
        for &neighbor in graph.neighbors[node].iter() {
            if !visited[neighbor] {
                visited[neighbor] = true;
                seen += 1;
                stack.push(neighbor);
            }
        }
    }
    seen == n
}

/// Builds the spanning-tree sampler and optional region columns for the supported variants.
///
/// # Arguments
///
/// * `params` - ReCom parameters containing the variant and optional region weights.
/// * `buf_size` - Capacity used by the sampler's internal buffers.
///
/// # Returns
///
/// A boxed spanning-tree sampler and, for region-aware ReCom, the ordered region
/// attribute names to copy into each pair subgraph.
fn make_tilted_sampler(
    params: &RecomParams,
    buf_size: usize,
) -> (Box<dyn SpanningTreeSampler>, Option<Vec<String>>) {
    if params.variant == RecomVariant::DistrictPairsRegionAware {
        let region_weights = params
            .region_weights
            .clone()
            .expect("Region weights required for region-aware ReCom.");
        let region_attrs = region_weights
            .iter()
            .map(|(col, _)| col.to_owned())
            .collect();
        (
            Box::new(RegionAwareSampler::new(buf_size, region_weights)),
            Some(region_attrs),
        )
    } else if params.variant == RecomVariant::DistrictPairsRMST {
        (Box::new(RMSTSampler::new(buf_size)), None)
    } else {
        panic!("ReCom variant not supported by tilted run optimizer.");
    }
}

/// Copies the selected district pair into the reusable subgraph buffer.
///
/// # Arguments
///
/// * `graph` - Full graph associated with `partition`.
/// * `partition` - Worker-local partition state.
/// * `buffers` - Worker buffers whose subgraph field is overwritten.
/// * `region_aware_attrs` - Optional node-attribute columns needed by region-aware sampling.
/// * `dist_a` - First selected district label.
/// * `dist_b` - Second selected district label.
fn fill_pair_subgraph(
    graph: &Graph,
    partition: &Partition,
    buffers: &mut TiltedWorkerBuffers,
    region_aware_attrs: Option<&Vec<String>>,
    dist_a: usize,
    dist_b: usize,
) {
    if let Some(attrs) = region_aware_attrs {
        partition.subgraph_with_attr_subset(
            graph,
            &mut buffers.subgraph,
            attrs.iter(),
            dist_a,
            dist_b,
        );
    } else {
        partition.subgraph_with_attr(graph, &mut buffers.subgraph, dist_a, dist_b);
    }
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

/// Draws one valid proposal attempt for a worker round and returns accept/reject output.
///
/// Invalid non-reversible proposal attempts (non-adjacent district pair, disconnected
/// merged pair, no balanced split) are retried internally and do not count as chain
/// self-loops. A tilted rejection does count as one self-loop.
///
/// # Arguments
///
/// * `graph` - Full graph associated with `partition`.
/// * `partition` - Worker-local partition state, temporarily updated and reverted.
/// * `params` - ReCom parameters used to generate splits.
/// * `buffers` - Reusable worker buffers for subgraphs, splits, proposals, and reverts.
/// * `st_sampler` - Spanning-tree sampler for the selected ReCom variant.
/// * `region_aware_attrs` - Optional node-attribute columns needed by region-aware sampling.
/// * `obj_fn` - Objective function used to score candidate proposals.
/// * `current_score` - Objective score of the canonical chain state.
/// * `accept_worse_prob` - Probability of accepting a non-improving proposal.
/// * `maximize` - If true, larger scores are improvements; otherwise smaller scores are.
/// * `rng` - Worker RNG used for proposal generation and acceptance.
///
/// # Returns
///
/// A `TiltedResultPacket` containing either one accepted scored proposal or one
/// tilted rejection self-loop.
fn draw_tilted_result(
    graph: &Graph,
    partition: &mut Partition,
    params: &RecomParams,
    buffers: &mut TiltedWorkerBuffers,
    st_sampler: &mut Box<dyn SpanningTreeSampler>,
    region_aware_attrs: Option<&Vec<String>>,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    current_score: f64,
    accept_worse_prob: f64,
    maximize: bool,
    rng: &mut SmallRng,
) -> TiltedResultPacket {
    loop {
        let Some((dist_a, dist_b)) = uniform_dist_pair(graph, partition, rng) else {
            continue; // retry (non-reversible)
        };

        fill_pair_subgraph(
            graph,
            partition,
            buffers,
            region_aware_attrs,
            dist_a,
            dist_b,
        );

        if !graph_connected(&buffers.subgraph.graph) {
            continue; // retry (non-reversible)
        }

        st_sampler.random_spanning_tree(&buffers.subgraph.graph, &mut buffers.spanning_tree, rng);
        let split = random_split(
            &buffers.subgraph.graph,
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
            continue; // retry (non-reversible)
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
        if is_improvement || rng.random::<f64>() < accept_worse_prob {
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

/// Starts a tilted-run worker thread.
///
/// Each round, the worker draws one spanning tree, produces a proposal,
/// temporarily applies it to score, always reverts, then decides accept/reject
/// using the tilted criterion. Results are sent back to the main thread.
///
/// # Arguments
///
/// * `graph` - Worker-owned graph clone.
/// * `partition` - Worker-owned partition clone kept in sync from main-thread diffs.
/// * `params` - ReCom parameters.
/// * `obj_fn` - Objective function used to score candidate proposals.
/// * `accept_worse_prob` - Probability of accepting a non-improving proposal.
/// * `maximize` - If true, larger scores are improvements; otherwise smaller scores are.
/// * `rng_seed` - Seed for this worker's RNG.
/// * `buf_size` - Capacity for two-district subgraph/proposal buffers.
/// * `job_recv` - Channel receiving canonical-state updates from the main thread.
/// * `result_send` - Channel sending worker accept/reject results to the main thread.
fn start_tilted_worker(
    graph: Graph,
    mut partition: Partition,
    params: RecomParams,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    accept_worse_prob: f64,
    maximize: bool,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<TiltedJobPacket>,
    result_send: Sender<TiltedResultPacket>,
) {
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut buffers = TiltedWorkerBuffers::new(n, buf_size, params.balance_ub);
    let (mut st_sampler, region_aware_attrs) = make_tilted_sampler(&params, buf_size);

    let mut next: TiltedJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        if let Some(diff) = next.diff {
            partition.update(&diff);
        }

        let result = draw_tilted_result(
            &graph,
            &mut partition,
            &params,
            &mut buffers,
            &mut st_sampler,
            region_aware_attrs.as_ref(),
            obj_fn,
            next.current_score,
            accept_worse_prob,
            maximize,
            &mut rng,
        );

        result_send.send(result).unwrap();
        next = job_recv.recv().unwrap();
    }
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
    graph: Graph,
    mut partition: Partition,
    writer: &mut dyn StatsWriter,
    recv: Receiver<TiltedStatsPacket>,
) {
    writer.init(&graph, &partition).unwrap();
    let mut next = recv.recv().unwrap();
    while !next.terminate {
        if let Some(proposal) = next.proposal {
            partition.update(&proposal);
            writer
                .step(next.step, &graph, &partition, &proposal, &next.counts)
                .unwrap();
        } else {
            writer
                .self_loop(next.step, &graph, &partition, &next.counts)
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
/// * `recv` - Channel receiving asynchronous score write packets.
fn start_tilted_score_writer(
    writer: &mut ScoresWriter,
    initial_score: f64,
    recv: Receiver<TiltedScorePacket>,
) {
    writer.init(initial_score).unwrap();
    let mut next = recv.recv().unwrap();
    while !next.terminate {
        for step in next.first_step..=next.last_step {
            writer.step(step, next.score, next.best_score).unwrap();
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

/// Stops the chain-statistics writer thread.
///
/// # Arguments
///
/// * `send` - Channel sending stats packets to the writer thread.
fn stop_tilted_stats_writer(send: &Sender<TiltedStatsPacket>) {
    send.send(TiltedStatsPacket {
        step: 0,
        proposal: None,
        counts: SelfLoopCounts::default(),
        terminate: true,
    })
    .unwrap();
}

/// Stops the score writer thread.
///
/// # Arguments
///
/// * `send` - Channel sending score packets to the writer thread.
fn stop_tilted_score_writer(send: &Sender<TiltedScorePacket>) {
    send.send(TiltedScorePacket {
        first_step: 0,
        last_step: 0,
        score: 0.0,
        best_score: 0.0,
        terminate: true,
    })
    .unwrap();
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

/// Runs a tilted-run optimizer with parallel tree drawing.
///
/// Worker threads draw spanning trees in parallel from the same partition state.
/// Each worker scores its proposal and decides accept/reject using the tilted
/// criterion. The main thread interleaves accepted proposals and rejections
/// (exactly like `multi_chain` in `run.rs`), maintaining a single sequential chain.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The starting partition.
/// * `params` - The chain parameters. `params.num_steps` is the total number
///   of chain steps (accepted + rejected).
/// * `n_threads` - The number of worker threads for parallel tree drawing.
/// * `obj_fn` - The objective function.
/// * `accept_worse_prob` - Probability of accepting a proposal with a worse score.
/// * `maximize` - If true, higher scores are better. If false, lower scores are better.
/// * `stats_writer` - Optional asynchronous writer for accepted chain proposals
///   and self-loop counts.
/// * `score_writer` - Optional asynchronous writer for per-step objective scores.
/// * `show_progress` - If true, show a progress bar tracking total chain steps.
///
/// # Returns
///
/// `Ok(partition)` containing the terminal chain state, or `Err` if inputs are
/// invalid or a scoped worker panics.
pub fn multi_tilted_runs_with_writer(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    accept_worse_prob: f64,
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
                unbounded();
            scope.spawn({
                let graph = graph.clone();
                let partition = partition.clone();
                move |_| start_tilted_stats_writer(graph, partition, writer, recv)
            });
            Some(send)
        } else {
            None
        };
        let score_send = if let Some(writer) = score_writer {
            let (send, recv): (Sender<TiltedScorePacket>, Receiver<TiltedScorePacket>) =
                unbounded();
            scope.spawn(move |_| start_tilted_score_writer(writer, current_score, recv));
            Some(send)
        } else {
            None
        };

        // Start worker threads.
        for t_idx in 0..n_threads {
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();
            let partition = partition.clone();

            scope.spawn(move |_| {
                start_tilted_worker(
                    graph.clone(),
                    partition,
                    params.clone(),
                    obj_fn,
                    accept_worse_prob,
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
            stop_tilted_stats_writer(send);
        }
        if let Some(send) = &score_send {
            stop_tilted_score_writer(send);
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
pub fn multi_tilted_runs(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    accept_worse_prob: f64,
    maximize: bool,
    show_progress: bool,
) -> Result<Partition, String> {
    multi_tilted_runs_with_writer(
        graph,
        partition,
        params,
        n_threads,
        obj_fn,
        accept_worse_prob,
        maximize,
        None,
        None,
        show_progress,
    )
}
