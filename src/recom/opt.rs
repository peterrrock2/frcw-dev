//! ReCom-based optimization using short bursts.
//!
//! We use the "short bursts" heuristic introduced in Cannon et al. 2020
//! (see "Voting Rights, Markov Chains, and Optimization by Short Bursts",
//!  arXiv: 2011.02288) to optimize arbitrary partition-level objective
//! functions.
use super::{
    cut_edge_dist_pair, node_bound, random_split, uniform_dist_pair, RecomParams, RecomProposal,
    RecomVariant,
};
use crate::buffers::{
    graph_connected_buffered, ConnectivityBuffers, SpanningTreeBuffer, SplitBuffer, SubgraphBuffer,
};
use crate::graph::Graph;
use crate::objectives::{IncrementalObjective, ObjectiveConfig};
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, RegionAwareSampler, SpanningTreeSampler, USTSampler};
use crate::stats::{ScoresWriter, SelfLoopCounts, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{unbounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde_json::json;
use std::collections::HashMap;
pub type ScoreValue = f64;

/// A unit of multithreaded work.
struct OptJobPacket {
    /// The number of steps to sample (*not* the number of unique plans).
    n_steps: usize,
    /// The change in the chain state since the last batch of work.
    /// If no new proposals are accepted, this may be `None`.
    diff: Option<Partition>,
    /// A sentinel used to kill the worker thread.
    terminate: bool,
}

/// The result of a unit of multithreaded work.
struct OptResultPacket {
    /// The best proposal found in a unit of work according to an
    /// objective function.
    best_partition: Option<Partition>,
    /// The score of the best proposal.
    best_score: Option<ScoreValue>,
}

/// Builds the spanning-tree sampler for the supported variants.
fn make_opt_sampler(
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
            panic!("Reversible ReCom is not supported by the short bursts optimizer.");
        }
    }
}

/// Draws the next candidate district pair. Cut-edge variants pick a pair by
/// sampling a cut edge; district-pair variants sample uniformly and return
/// `None` if the pair is non-adjacent.
fn sample_opt_pair(
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

/// Starts a ReCom optimization thread.
/// ReCom optimization threads run short ReCom chains ("short bursts"), which
/// are then aggregated by the main thread.
///
/// Arguments:
/// * `graph` - The graph associated with the chain.
/// * `partition` - The initial state of the chain.
/// * `params` - The chain parameters.
/// * `obj_fn` - The objective function to evaluate proposals against.
/// * `rng_seed` - The RNG seed for the job thread. (This should differ across threads.)
/// * `buf_size` - The buffer size for various chain buffers. This should usually be twice
///   the maximum possible district size (in nodes).
/// * `job_recv` - A Crossbeam channel for receiving batches of work from the main thread.
/// * `result_send` - A Crossbeam channel for sending completed batches to the main thread.
fn start_opt_thread(
    graph: &Graph,
    mut partition: Partition,
    params: RecomParams,
    obj_fn: impl Fn(&Graph, &Partition) -> ScoreValue + Send + Clone + Copy,
    maximize: bool,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<OptJobPacket>,
    result_send: Sender<OptResultPacket>,
) {
    // TODO: consider supporting other ReCom variants.
    // We generally don't (or can't) care about distributional
    // properties, so it would make little sense to support reversible
    // ReCom or the like. RMST sampling is asymptotically more efficient
    // than UST sampling, so we use it as the default for now.
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut subgraph_buf = SubgraphBuffer::new(n, buf_size);
    let mut st_buf = SpanningTreeBuffer::new(buf_size);
    let mut split_buf = SplitBuffer::new(buf_size, params.balance_ub as usize);
    let mut proposal_buf = RecomProposal::new_buffer(buf_size);
    let mut connectivity_buf = ConnectivityBuffers::new(buf_size);
    let mut st_sampler = make_opt_sampler(&params, buf_size, &mut rng);

    let mut next: OptJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        if next.diff.is_some() {
            partition = next.diff.unwrap();
        }

        let mut best_partition: Option<Partition> = None;
        let mut score = obj_fn(graph, &partition);
        let mut best_score: ScoreValue = score;
        let mut step = 0;
        while step < next.n_steps {
            // Sample a ReCom step.
            let dist_pair = sample_opt_pair(graph, &mut partition, params.variant, &mut rng);
            if dist_pair.is_none() {
                continue;
            }
            let (dist_a, dist_b) = dist_pair.unwrap();
            partition.subgraph(graph, &mut subgraph_buf, dist_a, dist_b);
            if !graph_connected_buffered(&subgraph_buf.graph, &mut connectivity_buf) {
                continue;
            }
            st_sampler.random_spanning_tree_with_parent(
                &subgraph_buf.graph,
                graph,
                &subgraph_buf.raw_nodes,
                &mut st_buf,
                &mut rng,
            );
            let split = random_split(
                &subgraph_buf.graph,
                graph,
                &mut rng,
                &st_buf.st,
                dist_a,
                dist_b,
                &mut split_buf,
                &mut proposal_buf,
                &subgraph_buf.raw_nodes,
                &params,
            );
            if split.is_ok() {
                score = obj_fn(graph, &partition);
                partition.update(&proposal_buf);
                if (maximize && score >= best_score) || (!maximize && score <= best_score) {
                    // TODO: reduce allocations by keeping a separate
                    // buffer for the best partition.
                    best_partition = Some(partition.clone());
                    best_score = score;
                }
                step += 1;
            }
        }
        let result = match best_partition {
            Some(partition) => OptResultPacket {
                best_partition: Some(partition),
                best_score: Some(best_score),
            },
            None => OptResultPacket {
                best_partition: None,
                best_score: None,
            },
        };
        result_send.send(result).unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Sends a batch of work to a ReCom optimization thread.
fn next_batch(send: &Sender<OptJobPacket>, diff: Option<Partition>, burst_length: usize) {
    send.send(OptJobPacket {
        n_steps: burst_length,
        diff: diff,
        terminate: false,
    })
    .unwrap();
}

/// Stops a ReCom optimization thread.
fn stop_opt_thread(send: &Sender<OptJobPacket>) {
    send.send(OptJobPacket {
        n_steps: 0,
        diff: None,
        terminate: true,
    })
    .unwrap();
}

/// Runs a multi-threaded ReCom short bursts optimizer.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The partition to start the chain run from (updated in place).
/// * `writer` - The statistics writer.
/// * `params` - The parameters of the ReCom chain runs.
/// * `n_threads` - The number of worker threads (excluding the main thread).
/// * `burst_length` - The number of steps per burst.
pub fn multi_short_bursts(
    graph: &Graph,
    mut partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    obj_fn: impl Fn(&Graph, &Partition) -> ScoreValue + Send + Clone + Copy,
    maximize: bool,
    burst_length: usize,
    verbose: bool,
) -> Result<Partition, String> {
    let mut step = 1;
    let node_ub = node_bound(&graph.pops, params.max_pop);
    let mut job_sends = vec![]; // main thread sends work to job threads
    let mut job_recvs = vec![]; // job threads receive work from main thread
    for _ in 0..n_threads {
        let (s, r): (Sender<OptJobPacket>, Receiver<OptJobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    // All optimization threads send a summary of chain results back to the main thread.
    let (result_send, result_recv): (Sender<OptResultPacket>, Receiver<OptResultPacket>) =
        unbounded();
    let mut score = obj_fn(graph, &partition);

    let scoped_result = scope(|scope| -> Result<Partition, String> {
        // Start optimization threads.
        for t_idx in 0..n_threads {
            // TODO: is this (+ t_idx) a sensible way to seed?
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();
            let worker_partition = partition.clone();

            scope.spawn(move |_| {
                start_opt_thread(
                    graph,
                    worker_partition,
                    params.clone(),
                    obj_fn,
                    maximize,
                    rng_seed,
                    node_ub,
                    job_recv,
                    result_send,
                );
            });
        }

        if params.num_steps > 0 {
            for job in job_sends.iter() {
                next_batch(job, None, burst_length);
            }
        }

        while step < params.num_steps {
            let mut diff = None;
            for _ in 0..n_threads {
                let packet: OptResultPacket = result_recv.recv().unwrap();
                if packet.best_partition.is_some()
                    && ((maximize && packet.best_score.unwrap() >= score)
                        || (!maximize && packet.best_score.unwrap() <= score))
                {
                    partition = packet.best_partition.unwrap();
                    score = packet.best_score.unwrap();
                    diff = Some(partition.clone());
                }
            }
            step += (n_threads * burst_length) as u64;
            if diff.is_some() && verbose {
                println!("{}", json!({
                    "step": step,
                    "score": score,
                    "assignment": partition.assignments.clone().into_iter().enumerate().collect::<HashMap<usize, u32>>()
                }).to_string());
            }
            for job in job_sends.iter() {
                next_batch(job, diff.clone(), burst_length);
            }
        }

        // Terminate worker threads.
        for job in job_sends.iter() {
            stop_opt_thread(job);
        }
        Ok(partition)
    });

    match scoped_result {
        Ok(inner) => inner, // inner: Result<Partition, String>

        // This only happens if some thread panicked.
        Err(_panic) => Err("multi_chain panicked in a worker thread".to_string()),
    }
}

/// Runs a multi-threaded ReCom short bursts optimizer with incremental scoring
/// and optional output writers.
///
/// Workers run short ReCom bursts using the full `ObjectiveConfig::score` path.
/// The main thread maintains per-district cached state via the incremental
/// interface for use by the optional writers. Writers are called synchronously
/// on the main thread: `stats_writer` is called whenever a new global best is
/// found; `scores_writer` is called at each burst boundary.
///
/// Because short bursts workers return only the best partition seen in a burst
/// (not the individual proposals), the stats writer receives a synthetic empty
/// proposal on each call. Writers that require proposal-level data (TSV, JSONL,
/// pcompress) will produce empty/zeroed proposal fields. Prefer the
/// `assignments`, `canonicalized-assignments`, `canonical`, or `ben` writers
/// when using short bursts.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The partition to start the chain run from.
/// * `params` - The parameters of the ReCom chain runs.
/// * `n_threads` - The number of worker threads (excluding the main thread).
/// * `objective` - The incremental objective configuration (`Copy`).
/// * `maximize` - If true, maximize the objective. If false, minimize it.
/// * `burst_length` - The number of accepted steps per short burst per worker.
/// * `stats_writer` - Optional writer called when a new global best is found.
/// * `scores_writer` - Optional writer called at each burst boundary.
/// * `show_progress` - If true, display a progress bar to stdout.
pub fn multi_short_bursts_incremental_with_writer(
    graph: &Graph,
    mut partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    objective: ObjectiveConfig,
    maximize: bool,
    burst_length: usize,
    mut stats_writer: Option<&mut dyn StatsWriter>,
    mut scores_writer: Option<&mut ScoresWriter>,
    show_progress: bool,
) -> Result<Partition, String> {
    let mut step = 1u64;
    let node_ub = node_bound(&graph.pops, params.max_pop);
    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads {
        let (s, r): (Sender<OptJobPacket>, Receiver<OptJobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    let (result_send, result_recv): (Sender<OptResultPacket>, Receiver<OptResultPacket>) =
        unbounded();

    // Build initial incremental state for writer initialization.
    let initial_obj_state = objective.init(graph, &partition);
    let initial_score = objective.score_state(&initial_obj_state);
    if let Some(writer) = stats_writer.as_mut() {
        writer.init(graph, &partition).unwrap();
    }
    if let Some(writer) = scores_writer.as_mut() {
        let ds = objective.district_scores(&initial_obj_state);
        writer.init(initial_score, &ds).unwrap();
    }

    // Worker closure: full-score path (ObjectiveConfig is Copy).
    let obj_fn = move |graph: &Graph, partition: &Partition| -> ScoreValue {
        objective.score(graph, partition)
    };

    let progress_bar = if show_progress {
        let pb = ProgressBar::with_draw_target(
            Some(params.num_steps),
            indicatif::ProgressDrawTarget::stdout_with_hz(1),
        );
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>10}/{len} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        Some(pb)
    } else {
        None
    };

    let scoped_result = scope(|scope| -> Result<Partition, String> {
        for t_idx in 0..n_threads {
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();
            let worker_partition = partition.clone();

            scope.spawn(move |_| {
                start_opt_thread(
                    graph,
                    worker_partition,
                    params.clone(),
                    obj_fn,
                    maximize,
                    rng_seed,
                    node_ub,
                    job_recv,
                    result_send,
                );
            });
        }

        if params.num_steps > 0 {
            for job in job_sends.iter() {
                next_batch(job, None, burst_length);
            }
        }

        let mut score = initial_score;
        let mut best_obj_state = initial_obj_state;

        while step < params.num_steps {
            let mut diff: Option<Partition> = None;
            for _ in 0..n_threads {
                let packet: OptResultPacket = result_recv.recv().unwrap();
                if let Some(new_score) = packet.best_score {
                    if (maximize && new_score >= score) || (!maximize && new_score <= score) {
                        partition = packet.best_partition.unwrap();
                        score = new_score;
                        diff = Some(partition.clone());
                    }
                }
            }
            step += (n_threads * burst_length) as u64;

            // When a new best is found: update incremental state and call stats writer.
            if diff.is_some() {
                best_obj_state = objective.init(graph, &partition);
                if let Some(writer) = stats_writer.as_mut() {
                    // Short bursts workers return partitions, not proposals.
                    // Writers that ignore the proposal (assignments, BEN, canonical)
                    // work correctly. Writers that use proposal fields will see zeros.
                    let dummy = RecomProposal {
                        a_label: 0,
                        b_label: 0,
                        a_pop: 0,
                        b_pop: 0,
                        a_nodes: Vec::new(),
                        b_nodes: Vec::new(),
                    };
                    writer
                        .step(step, graph, &partition, &dummy, &SelfLoopCounts::default())
                        .unwrap();
                }
            }

            // Call scores writer at each burst boundary.
            if let Some(writer) = scores_writer.as_mut() {
                let ds = objective.district_scores(&best_obj_state);
                writer.step(step, score, score, &ds).unwrap();
            }

            if let Some(pb) = progress_bar.as_ref() {
                pb.set_position(step.min(params.num_steps));
            }

            for job in job_sends.iter() {
                next_batch(job, diff.clone(), burst_length);
            }
        }

        for job in job_sends.iter() {
            stop_opt_thread(job);
        }
        Ok(partition)
    });

    if let Some(pb) = progress_bar {
        pb.set_position(params.num_steps);
        pb.finish_and_clear();
    }

    if let Some(writer) = stats_writer {
        writer.close().unwrap();
    }
    if let Some(writer) = scores_writer {
        writer.close().unwrap();
    }

    match scoped_result {
        Ok(inner) => inner,
        Err(_panic) => {
            Err("multi_short_bursts_incremental panicked in a worker thread".to_string())
        }
    }
}
