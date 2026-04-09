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
use super::{node_bound, random_split, uniform_dist_pair, RecomParams, RecomProposal, RecomVariant};
use crate::buffers::{SpanningTreeBuffer, SplitBuffer, SubgraphBuffer};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, RegionAwareSampler, SpanningTreeSampler};
use crossbeam::scope;
use crossbeam_channel::{unbounded, Receiver, Sender};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::json;
use std::collections::HashMap;

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
    rejections: u64,
    /// Accepted proposals: `(random_id, proposal, score)`.
    /// The random ID is used for deterministic interleaving (same technique as `run.rs`).
    proposals: Vec<(u64, RecomProposal, f64)>,
}

/// Returns true iff `graph` has exactly one connected component.
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

/// Starts a tilted-run worker thread.
///
/// Each round, the worker draws one spanning tree, produces a proposal,
/// temporarily applies it to score, always reverts, then decides accept/reject
/// using the tilted criterion. Results are sent back to the main thread.
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
    let mut subgraph_buf = SubgraphBuffer::new(n, buf_size);
    let mut st_buf = SpanningTreeBuffer::new(buf_size);
    let mut split_buf = SplitBuffer::new(buf_size, params.balance_ub as usize);
    let mut proposal_buf = RecomProposal::new_buffer(buf_size);
    let mut revert_buf = RecomProposal::new_buffer(buf_size);

    let mut st_sampler: Box<dyn SpanningTreeSampler>;
    let region_aware = params.variant == RecomVariant::DistrictPairsRegionAware;
    let mut region_aware_attrs: Vec<String> = vec![];
    if region_aware {
        st_sampler = Box::new(RegionAwareSampler::new(
            buf_size,
            params
                .region_weights
                .clone()
                .expect("Region weights required for region-aware ReCom."),
        ));
        region_aware_attrs = params
            .region_weights
            .clone()
            .unwrap()
            .iter()
            .map(|(col, _)| col.to_owned())
            .collect();
    } else if params.variant == RecomVariant::DistrictPairsRMST {
        st_sampler = Box::new(RMSTSampler::new(buf_size));
    } else {
        panic!("ReCom variant not supported by tilted run optimizer.");
    }

    let mut next: TiltedJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        // Apply diff from previous round (if any).
        if let Some(diff) = next.diff {
            partition.update(&diff);
        }
        let current_score = next.current_score;

        let mut rejections: u64 = 0;
        let mut proposals: Vec<(u64, RecomProposal, f64)> = Vec::new();

        // Draw one tree, produce one proposal (retry on failures like non-reversible ReCom).
        loop {
            let dist_pair = uniform_dist_pair(&graph, &mut partition, &mut rng);
            if dist_pair.is_none() {
                continue; // retry (non-reversible)
            }
            let (dist_a, dist_b) = dist_pair.unwrap();

            if region_aware {
                partition.subgraph_with_attr_subset(
                    &graph,
                    &mut subgraph_buf,
                    region_aware_attrs.iter(),
                    dist_a,
                    dist_b,
                );
            } else {
                partition.subgraph_with_attr(&graph, &mut subgraph_buf, dist_a, dist_b);
            }

            if !graph_connected(&subgraph_buf.graph) {
                continue; // retry (non-reversible)
            }

            st_sampler.random_spanning_tree(&subgraph_buf.graph, &mut st_buf, &mut rng);
            let split = random_split(
                &subgraph_buf.graph,
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
                // Save revert state for the two affected districts.
                revert_buf.a_label = proposal_buf.a_label;
                revert_buf.b_label = proposal_buf.b_label;
                revert_buf.a_pop = partition.dist_pops[proposal_buf.a_label];
                revert_buf.b_pop = partition.dist_pops[proposal_buf.b_label];
                revert_buf.a_nodes.clear();
                revert_buf
                    .a_nodes
                    .extend_from_slice(&partition.dist_nodes[proposal_buf.a_label]);
                revert_buf.b_nodes.clear();
                revert_buf
                    .b_nodes
                    .extend_from_slice(&partition.dist_nodes[proposal_buf.b_label]);

                // Apply proposal and score.
                partition.update(&proposal_buf);
                let new_score = obj_fn(&graph, &partition);

                // Always revert -- canonical state is managed by the main thread.
                partition.update(&revert_buf);

                // Tilted accept/reject.
                let is_improvement = if maximize {
                    new_score >= current_score
                } else {
                    new_score <= current_score
                };
                if is_improvement || rng.random::<f64>() < accept_worse_prob {
                    proposals.push((rng.random::<u64>(), proposal_buf.clone(), new_score));
                } else {
                    rejections += 1;
                }
                break;
            }
            // else: no valid split, retry (non-reversible)
        }

        result_send
            .send(TiltedResultPacket {
                rejections,
                proposals,
            })
            .unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Sends a job to a tilted-run worker thread.
fn next_tilted_batch(send: &Sender<TiltedJobPacket>, diff: Option<RecomProposal>, score: f64) {
    send.send(TiltedJobPacket {
        diff,
        current_score: score,
        terminate: false,
    })
    .unwrap();
}

/// Stops a tilted-run worker thread.
fn stop_tilted_worker(send: &Sender<TiltedJobPacket>) {
    send.send(TiltedJobPacket {
        diff: None,
        current_score: 0.0,
        terminate: true,
    })
    .unwrap();
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
/// * `verbose` - If true, print a JSONL line to stdout on each global best improvement.
///
/// Returns the terminal partition state of the tilted chain.
pub fn multi_tilted_runs(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    obj_fn: impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
    accept_worse_prob: f64,
    maximize: bool,
    verbose: bool,
) -> Result<Partition, String> {
    if n_threads == 0 {
        return Err("n_threads must be at least 1".to_string());
    }

    let mut step: u64 = 0;
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

    let mut current_score = obj_fn(graph, &partition);
    let mut best_score = current_score;
    let mut best_partition = partition.clone();
    let mut partition = partition;
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

        // Initial dispatch (no diff).
        if params.num_steps > 0 {
            for job in job_sends.iter() {
                next_tilted_batch(job, None, current_score);
            }
        }

        while step < params.num_steps {
            // Collect results from all workers.
            let mut total_rejections: u64 = 0;
            let mut all_proposals: Vec<(u64, RecomProposal, f64)> = Vec::new();
            for _ in 0..n_threads {
                let packet: TiltedResultPacket = result_recv.recv().unwrap();
                total_rejections += packet.rejections;
                all_proposals.extend(packet.proposals);
            }

            let mut loops = total_rejections as usize;
            if !all_proposals.is_empty() {
                // Sort by random ID for deterministic interleaving.
                all_proposals.sort_by(|a, b| a.0.cmp(&b.0));

                let mut total = loops + all_proposals.len();
                while total > 0 && step < params.num_steps {
                    step += 1;
                    let event = rng.random_range(0..total);
                    if event < loops {
                        // Self-loop (rejection): chain stays in place.
                        loops -= 1;
                    } else {
                        // Accepted proposal: apply and broadcast.
                        let idx = rng.random_range(0..all_proposals.len());
                        let (_, ref proposal, new_score) = all_proposals[idx];

                        partition.update(proposal);
                        current_score = new_score;

                        // Track best.
                        let is_new_best = if maximize {
                            new_score > best_score
                        } else {
                            new_score < best_score
                        };
                        if is_new_best {
                            best_score = new_score;
                            best_partition = partition.clone();
                            if verbose {
                                println!(
                                    "{}",
                                    json!({
                                        "step": step,
                                        "score": best_score,
                                        "assignment": best_partition.assignments.clone()
                                            .into_iter()
                                            .enumerate()
                                            .collect::<HashMap<usize, u32>>()
                                    })
                                    .to_string()
                                );
                            }
                        }

                        // Broadcast accepted proposal to all workers.
                        for job in job_sends.iter() {
                            next_tilted_batch(job, Some(proposal.clone()), current_score);
                        }
                        break; // need new round (state changed)
                    }
                    total -= 1;
                }
            } else {
                // All workers failed or rejected -- all self-loops.
                step += total_rejections;
                for job in job_sends.iter() {
                    next_tilted_batch(job, None, current_score);
                }
            }
        }

        // Terminate worker threads.
        for job in job_sends.iter() {
            stop_tilted_worker(job);
        }
        Ok(partition)
    });

    match scoped_result {
        Ok(inner) => inner,
        Err(_panic) => Err("multi_tilted_runs panicked in a worker thread".to_string()),
    }
}
