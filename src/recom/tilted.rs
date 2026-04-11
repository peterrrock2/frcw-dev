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
//! the main thread interleaves accepted proposals and rejections.
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

    /// Writes the current score state if a score writer was provided.
    ///
    /// # Arguments
    ///
    /// * `score_writer` - Optional writer for per-step objective scores.
    fn write_score(&self, score_writer: &mut Option<&mut ScoresWriter>) {
        if let Some(writer) = score_writer.as_deref_mut() {
            writer
                .step(self.step, self.current_score, self.best_score)
                .unwrap();
        }
    }

    /// Records one tilted rejection self-loop.
    ///
    /// # Arguments
    ///
    /// * `score_writer` - Optional writer for per-step objective scores.
    fn record_rejection(&mut self, score_writer: &mut Option<&mut ScoresWriter>) {
        self.step += 1;
        self.pending_counts.inc(SelfLoopReason::TiltedRejection);
        self.write_score(score_writer);
    }

    /// Applies an accepted proposal and writes chain statistics.
    ///
    /// # Arguments
    ///
    /// * `graph` - Full graph associated with the chain.
    /// * `accepted` - Proposal selected by the main-thread interleaving step.
    /// * `maximize` - If true, larger scores are improvements; otherwise smaller scores are.
    /// * `stats_writer` - Optional writer for chain statistics.
    /// * `score_writer` - Optional writer for per-step objective scores.
    fn apply_accepted_proposal(
        &mut self,
        graph: &Graph,
        accepted: &ScoredProposal,
        maximize: bool,
        stats_writer: &mut Option<&mut dyn StatsWriter>,
        score_writer: &mut Option<&mut ScoresWriter>,
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
        if let Some(writer) = stats_writer.as_deref_mut() {
            writer
                .step(
                    self.step,
                    graph,
                    &self.partition,
                    &accepted.proposal,
                    &self.pending_counts,
                )
                .unwrap();
        }
        self.pending_counts = SelfLoopCounts::default();
        self.write_score(score_writer);
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
/// are applied and written through `stats_writer`. Every interleaved event is
/// written through `score_writer`.
///
/// # Arguments
///
/// * `state` - Main-thread canonical chain state.
/// * `loops` - Number of worker tilted rejections to interleave as self-loops.
/// * `proposals` - Worker-accepted proposals to interleave with self-loops.
/// * `graph` - Full graph associated with the chain.
/// * `params` - ReCom parameters containing the target step count.
/// * `rng` - Main-thread RNG used for event interleaving and proposal selection.
/// * `job_sends` - Worker job channels for broadcasting state updates.
/// * `maximize` - If true, larger scores are improvements; otherwise smaller scores are.
/// * `stats_writer` - Optional writer for accepted chain proposals and self-loop counts.
/// * `score_writer` - Optional writer for per-step objective scores.
fn interleave_tilted_round(
    state: &mut TiltedMainState,
    mut loops: usize,
    mut proposals: Vec<ScoredProposal>,
    graph: &Graph,
    params: &RecomParams,
    rng: &mut SmallRng,
    job_sends: &[Sender<TiltedJobPacket>],
    maximize: bool,
    stats_writer: &mut Option<&mut dyn StatsWriter>,
    score_writer: &mut Option<&mut ScoresWriter>,
) {
    if proposals.is_empty() {
        let remaining = (params.num_steps - state.step) as usize;
        for _ in 0..loops.min(remaining) {
            state.record_rejection(score_writer);
        }
        send_tilted_jobs(job_sends, None, state.current_score);
        return;
    }

    proposals.sort_by_key(|proposal| proposal.id);
    let mut total = loops + proposals.len();
    while total > 0 && state.step < params.num_steps {
        let event = rng.random_range(0..total);
        if event < loops {
            state.record_rejection(score_writer);
            loops -= 1;
            total -= 1;
            continue;
        }

        let accepted = &proposals[rng.random_range(0..proposals.len())];
        state.apply_accepted_proposal(graph, accepted, maximize, stats_writer, score_writer);
        send_tilted_jobs(job_sends, Some(&accepted.proposal), state.current_score);
        break; // need new round (state changed)
    }
}

/// Runs the main-thread collection/interleaving loop after workers are spawned.
///
/// # Arguments
///
/// * `state` - Main-thread canonical chain state.
/// * `graph` - Full graph associated with the chain.
/// * `params` - ReCom parameters containing the target step count.
/// * `n_threads` - Number of worker packets to collect per round.
/// * `result_recv` - Shared result channel from all workers.
/// * `job_sends` - Worker job channels.
/// * `rng` - Main-thread RNG used for interleaving.
/// * `maximize` - If true, larger scores are improvements; otherwise smaller scores are.
/// * `stats_writer` - Optional writer for accepted chain proposals and self-loop counts.
/// * `score_writer` - Optional writer for per-step objective scores.
fn run_tilted_main_loop(
    state: &mut TiltedMainState,
    graph: &Graph,
    params: &RecomParams,
    n_threads: usize,
    result_recv: &Receiver<TiltedResultPacket>,
    job_sends: &[Sender<TiltedJobPacket>],
    rng: &mut SmallRng,
    maximize: bool,
    mut stats_writer: Option<&mut dyn StatsWriter>,
    mut score_writer: Option<&mut ScoresWriter>,
) {
    if let Some(writer) = stats_writer.as_deref_mut() {
        writer.init(graph, &state.partition).unwrap();
    }
    if let Some(writer) = score_writer.as_deref_mut() {
        writer.init(state.current_score).unwrap();
    }

    if params.num_steps > 0 {
        send_tilted_jobs(job_sends, None, state.current_score);
    }

    while state.step < params.num_steps {
        let (loops, proposals) = collect_tilted_results(result_recv, n_threads);
        interleave_tilted_round(
            state,
            loops,
            proposals,
            graph,
            params,
            rng,
            job_sends,
            maximize,
            &mut stats_writer,
            &mut score_writer,
        );
    }

    if let Some(writer) = stats_writer.as_deref_mut() {
        writer.close().unwrap();
    }
    if let Some(writer) = score_writer.as_deref_mut() {
        writer.close().unwrap();
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
/// * `stats_writer` - Optional writer for accepted chain proposals and self-loop counts.
/// * `score_writer` - Optional writer for per-step objective scores.
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

    let scoped_result = scope(|scope| -> Result<Partition, String> {
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
            graph,
            params,
            n_threads,
            &result_recv,
            &job_sends,
            &mut rng,
            maximize,
            stats_writer,
            score_writer,
        );

        stop_tilted_workers(&job_sends);
        Ok(state.partition)
    });

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
    _verbose: bool,
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
    )
}
