//! Shared engine for the ReCom short-bursts optimizer (Cannon et al. 2020,
//! arXiv: 2011.02288).
//!
//! Short bursts is a greedy hill-climbing heuristic over an arbitrary
//! partition-level objective. The chain runs in fixed-length "bursts"; within
//! a burst the worker always accepts every valid proposal and tracks the
//! burst-end best partition. After each round of bursts the main thread
//! adopts any new global best as the seed of the next burst across all
//! workers.
//!
//! Threading model mirrors the tilted runner: workers parallelize the
//! tree-drawing and scoring step, the main thread interleaves their results,
//! and optional output writers run on separate threads fed by bounded
//! channels so disk I/O does not block proposal generation.
use super::super::{
    cut_edge_dist_pair, node_bound, random_split, uniform_dist_pair, RecomParams, RecomProposal,
    RecomVariant,
};
use super::packets::{
    send_burst_batch, terminate_burst_worker, BurstJobPacket, BurstResult, BurstScorePacket,
    BurstStatsPacket,
};
use super::writers::{start_burst_score_writer, start_burst_stats_writer};
use crate::buffers::{
    graph_connected_buffered, ConnectivityBuffers, SpanningTreeBuffer, SplitBuffer, SubgraphBuffer,
};
use crate::graph::Graph;
use crate::objectives::{IncrementalObjective, ObjectiveConfig};
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, RegionAwareSampler, SpanningTreeSampler, USTSampler};
use crate::stats::{ScoresWriter, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Capacity of the bounded channels feeding the stats and score writer
/// threads. Matches the tilted runner.
const WRITER_CHANNEL_CAPACITY: usize = 128;

/// Public type alias for the score type used by the short-bursts engine.
pub type ScoreValue = f64;

/// Builds the spanning-tree sampler for the supported variants.
fn make_sampler(
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
fn sample_dist_pair(
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

/// Runs a short-bursts worker thread.
///
/// On each job the worker runs `n_steps` greedy ReCom steps from its current
/// (or freshly-replaced) partition, optionally collecting every accepted step
/// into `BurstResult::all_steps`, and always tracks the burst-end best.
///
/// # Arguments
///
/// * `collect_trace` - When `true`, every accepted step is stored in
///   `BurstResult::all_steps`. When `false`, `all_steps` is empty and only
///   the burst-end best is returned.
fn run_burst_worker(
    graph: &Graph,
    mut partition: Partition,
    params: RecomParams,
    obj_fn: impl Fn(&Graph, &Partition) -> ScoreValue + Send,
    maximize: bool,
    rng_seed: u64,
    buf_size: usize,
    collect_trace: bool,
    job_recv: Receiver<BurstJobPacket>,
    result_send: Sender<BurstResult>,
) {
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut subgraph_buf = SubgraphBuffer::new(n, buf_size);
    let mut st_buf = SpanningTreeBuffer::new(buf_size);
    let mut split_buf = SplitBuffer::new(buf_size, params.balance_ub as usize);
    let mut proposal_buf = RecomProposal::new_buffer(buf_size);
    let mut connectivity_buf = ConnectivityBuffers::new(buf_size);
    let mut st_sampler = make_sampler(&params, buf_size, &mut rng);

    let mut next: BurstJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        if next.diff.is_some() {
            partition = next.diff.unwrap();
        }

        let mut all_steps: Vec<(Partition, ScoreValue)> = Vec::new();
        let mut best: Option<(Partition, ScoreValue)> = None;
        let mut best_score: ScoreValue = obj_fn(graph, &partition);
        let mut step = 0;
        while step < next.n_steps {
            let dist_pair = sample_dist_pair(graph, &mut partition, params.variant, &mut rng);
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
                partition.update(&proposal_buf);
                let score = obj_fn(graph, &partition);
                if collect_trace {
                    all_steps.push((partition.clone(), score));
                }
                if (maximize && score >= best_score) || (!maximize && score <= best_score) {
                    best = Some((partition.clone(), score));
                    best_score = score;
                }
                step += 1;
            }
        }
        result_send
            .send(BurstResult { all_steps, best })
            .unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Runs a multi-threaded ReCom short-bursts optimizer with optional async
/// output writers.
///
/// Workers run short ReCom bursts using the full `ObjectiveConfig::score`
/// path. The main thread maintains per-district cached state via the
/// incremental interface for use by the optional score writer.
///
/// When `write_best_only = false`, the stats writer thread is called for
/// every accepted chain step across all worker bursts; sample numbers are
/// sequential starting at 1. When `write_best_only = true`, the stats writer
/// thread is called only when a new global best is found, and sample numbers
/// count improvements.
///
/// `scores_writer` is always called only when a new global best is found;
/// its step number reflects the total accepted steps at that burst boundary.
///
/// Because short-bursts workers return full partitions (not individual
/// proposals), the stats writer thread receives a synthetic empty proposal on
/// each call. Writers that require proposal-level data (TSV, JSONL, pcompress)
/// will produce empty/zeroed proposal fields; prefer `assignments`,
/// `canonicalized-assignments`, `canonical`, or `ben` writers.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The starting partition.
/// * `params` - The chain parameters of the ReCom chain runs.
/// * `n_threads` - The number of worker threads (excluding the main thread).
/// * `objective` - The incremental objective configuration (`Copy`).
/// * `maximize` - If true, maximize the objective. If false, minimize it.
/// * `burst_length` - The number of accepted steps per burst per worker.
/// * `stats_writer` - Optional asynchronous writer called for every step (or
///   only new bests when `write_best_only = true`).
/// * `scores_writer` - Optional asynchronous writer called whenever a new
///   global best is found.
/// * `show_progress` - If true, display a progress bar to stdout.
/// * `write_best_only` - If true, the stats writer is called only when a new
///   global best is found, and workers skip collecting per-step partitions.
pub fn multi_short_bursts_with_writer(
    graph: &Graph,
    mut partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    objective: ObjectiveConfig,
    maximize: bool,
    burst_length: usize,
    stats_writer: Option<&mut dyn StatsWriter>,
    scores_writer: Option<&mut ScoresWriter>,
    show_progress: bool,
    write_best_only: bool,
) -> Result<Partition, String> {
    let mut step = 1u64;
    let node_ub = node_bound(&graph.pops, params.max_pop);
    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads {
        let (s, r): (Sender<BurstJobPacket>, Receiver<BurstJobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    let (result_send, result_recv): (Sender<BurstResult>, Receiver<BurstResult>) = unbounded();

    // Build initial incremental state for writer initialization.
    let initial_obj_state = objective.init(graph, &partition);
    let initial_score = objective.score_state(&initial_obj_state);
    let initial_district_scores = objective.district_scores(&initial_obj_state);

    // Worker closure: full-score path (ObjectiveConfig is Copy).
    let obj_fn = move |graph: &Graph, partition: &Partition| -> ScoreValue {
        objective.score(graph, partition)
    };
    // Collect per-step partitions only when the stats writer needs every step.
    let collect_trace = !write_best_only;

    // The stats writer emits the seed plan in init(), which counts as the
    // first output record. To keep the total output record count equal to
    // params.num_steps, we run one fewer chain step.
    let effective_steps = params.num_steps.saturating_sub(1);

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
        // Spawn writer threads, if requested.
        let stats_send = if let Some(writer) = stats_writer {
            let (send, recv): (Sender<BurstStatsPacket>, Receiver<BurstStatsPacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn({
                let partition = partition.clone();
                move |_| start_burst_stats_writer(graph, partition, writer, recv)
            });
            Some(send)
        } else {
            None
        };
        let score_send = if let Some(writer) = scores_writer {
            let (send, recv): (Sender<BurstScorePacket>, Receiver<BurstScorePacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn(move |_| {
                start_burst_score_writer(writer, initial_score, initial_district_scores, recv)
            });
            Some(send)
        } else {
            None
        };

        // Spawn worker threads.
        for t_idx in 0..n_threads {
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();
            let worker_partition = partition.clone();

            scope.spawn(move |_| {
                run_burst_worker(
                    graph,
                    worker_partition,
                    params.clone(),
                    obj_fn,
                    maximize,
                    rng_seed,
                    node_ub,
                    collect_trace,
                    job_recv,
                    result_send,
                );
            });
        }

        if effective_steps > 0 {
            for job in job_sends.iter() {
                send_burst_batch(job, None, burst_length);
            }
        }

        let mut score = initial_score;
        let mut best_obj_state = initial_obj_state;
        // Sequential sample number for stats_writer: incremented once per
        // partition emitted to the writer (every step, or every new best).
        let mut writer_step: u64 = 0;

        while step < effective_steps {
            let mut all_packets: Vec<BurstResult> = Vec::with_capacity(n_threads);
            for _ in 0..n_threads {
                all_packets.push(result_recv.recv().unwrap());
            }
            step += (n_threads * burst_length) as u64;

            let mut diff: Option<Partition> = None;

            if write_best_only {
                // Emit only on new global bests.
                for packet in all_packets {
                    if let Some((bp, bs)) = packet.best {
                        if (maximize && bs >= score) || (!maximize && bs <= score) {
                            partition = bp;
                            score = bs;
                            diff = Some(partition.clone());
                            best_obj_state = objective.init(graph, &partition);
                            writer_step += 1;
                            if let Some(send) = stats_send.as_ref() {
                                send.send(BurstStatsPacket {
                                    step: writer_step,
                                    partition: Some(partition.clone()),
                                    terminate: false,
                                })
                                .unwrap();
                            }
                        }
                    }
                }
            } else {
                // Emit every accepted step; cap writes at effective_steps so
                // total records (init seed + per-step) equal params.num_steps.
                for packet in all_packets {
                    for (stepped_partition, stepped_score) in packet.all_steps {
                        if writer_step < effective_steps {
                            writer_step += 1;
                            if let Some(send) = stats_send.as_ref() {
                                send.send(BurstStatsPacket {
                                    step: writer_step,
                                    partition: Some(stepped_partition.clone()),
                                    terminate: false,
                                })
                                .unwrap();
                            }
                        }
                        if (maximize && stepped_score >= score)
                            || (!maximize && stepped_score <= score)
                        {
                            partition = stepped_partition;
                            score = stepped_score;
                            diff = Some(partition.clone());
                            best_obj_state = objective.init(graph, &partition);
                        }
                    }
                }
            }

            // Score writer fires only when a new global best was found this round.
            if diff.is_some() {
                if let Some(send) = score_send.as_ref() {
                    let ds = objective.district_scores(&best_obj_state);
                    send.send(BurstScorePacket {
                        step,
                        score,
                        best_score: score,
                        district_scores: Some(ds),
                        terminate: false,
                    })
                    .unwrap();
                }
            }

            if let Some(pb) = progress_bar.as_ref() {
                pb.set_position((step + 1).min(params.num_steps));
            }

            for job in job_sends.iter() {
                send_burst_batch(job, diff.clone(), burst_length);
            }
        }

        for job in job_sends.iter() {
            terminate_burst_worker(job);
        }

        // Tell the writer threads to drain and exit.
        if let Some(send) = stats_send.as_ref() {
            send.send(BurstStatsPacket {
                step: 0,
                partition: None,
                terminate: true,
            })
            .unwrap();
        }
        if let Some(send) = score_send.as_ref() {
            send.send(BurstScorePacket {
                step: 0,
                score: 0.0,
                best_score: 0.0,
                district_scores: None,
                terminate: true,
            })
            .unwrap();
        }

        Ok(partition)
    });

    if let Some(pb) = progress_bar {
        pb.set_position(params.num_steps);
        pb.finish_and_clear();
    }

    match scoped_result {
        Ok(inner) => inner,
        Err(_panic) => Err("multi_short_bursts panicked in a worker thread".to_string()),
    }
}
