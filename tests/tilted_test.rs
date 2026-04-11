// Functional tests for tilted run optimization.
use frcw::graph::Graph;
use frcw::objectives::{make_objective_fn, required_node_cols};
use frcw::partition::Partition;
use frcw::recom::tilted::{multi_tilted_runs, multi_tilted_runs_with_writer};
use frcw::recom::RecomProposal;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{ScoresWriter, SelfLoopCounts, StatsWriter};
use std::collections::HashSet;
use std::fs;
use std::io::Result as IOResult;
use std::iter::FromIterator;

use rstest::rstest;
use test_fixtures::fixture_with_attributes;

const RNG_SEED: u64 = 153434375;

// =================================================================================
// == Helpers (same invariant checks as step_test.rs, applied to final partition) ==
// =================================================================================

fn is_connected_subset(graph: &Graph, nodes: &Vec<usize>) -> bool {
    if nodes.is_empty() {
        return true;
    }
    let nodeset = HashSet::<usize>::from_iter(nodes.iter().cloned());
    let mut stack = vec![nodes[0]];
    let mut visited = HashSet::<usize>::from_iter(stack.iter().cloned());
    while let Some(next) = stack.pop() {
        for neighbor in graph.neighbors[next].iter() {
            if nodeset.contains(neighbor) && !visited.contains(neighbor) {
                visited.insert(*neighbor);
                stack.push(*neighbor);
            }
        }
    }
    return visited.len() == nodes.len();
}

fn assert_partition_valid(graph: &Graph, partition: &Partition, min_pop: u32, max_pop: u32) {
    // Node count matches.
    let node_count: usize = partition.dist_nodes.iter().map(|n| n.len()).sum();
    assert_eq!(
        node_count,
        graph.neighbors.len(),
        "Node count mismatch: partition has {}, graph has {}",
        node_count,
        graph.neighbors.len()
    );

    // Total population matches.
    assert_eq!(
        partition.dist_pops.iter().sum::<u32>(),
        graph.total_pop,
        "Total population mismatch"
    );

    // Population bounds.
    for (i, &pop) in partition.dist_pops.iter().enumerate() {
        assert!(
            min_pop <= pop && pop <= max_pop,
            "District {} pop {} outside bounds [{}, {}]",
            i,
            pop,
            min_pop,
            max_pop
        );
    }

    // Population sums match dist_nodes.
    for (i, (nodes, &pop)) in partition
        .dist_nodes
        .iter()
        .zip(partition.dist_pops.iter())
        .enumerate()
    {
        let computed: u32 = nodes.iter().map(|&n| graph.pops[n]).sum();
        assert_eq!(
            computed, pop,
            "District {} pop sum {} != recorded pop {}",
            i, computed, pop
        );
    }

    // Assignments consistent with dist_nodes.
    for (dist, nodes) in partition.dist_nodes.iter().enumerate() {
        for &n in nodes {
            assert_eq!(
                partition.assignments[n] as usize, dist,
                "Node {} assigned to {} but in dist_nodes[{}]",
                n, partition.assignments[n], dist
            );
        }
    }

    // All districts connected.
    for (i, nodes) in partition.dist_nodes.iter().enumerate() {
        assert!(
            is_connected_subset(graph, nodes),
            "District {} is disconnected",
            i
        );
    }
}

/// Simple objective: population of district 0.
/// Useful for testing chain mechanics without depending on the objectives module.
fn dist0_pop_objective(graph: &Graph, partition: &Partition) -> f64 {
    partition.dist_nodes[0]
        .iter()
        .map(|&n| graph.pops[n] as f64)
        .sum()
}

struct CountingStatsWriter {
    init_calls: usize,
    steps: Vec<(u64, usize)>,
}

impl CountingStatsWriter {
    fn new() -> CountingStatsWriter {
        CountingStatsWriter {
            init_calls: 0,
            steps: Vec::new(),
        }
    }
}

impl StatsWriter for CountingStatsWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> IOResult<()> {
        self.init_calls += 1;
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        _partition: &Partition,
        _proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> IOResult<()> {
        self.steps.push((step, counts.sum()));
        Ok(())
    }

    fn close(&mut self) -> IOResult<()> {
        Ok(())
    }
}

// ====================================
// == Partition validity on 6x6 grid ==
// ====================================

#[rstest]
fn test_tilted_partition_valid_grid(
    #[values(1, 4)] n_threads: usize,
    #[values(true, false)] maximize: bool,
    #[values(0.0, 0.5)] accept_worse_prob: f64,
) {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let min_pop: u32 = 5;
    let max_pop: u32 = 7;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let result = multi_tilted_runs(
        &graph,
        partition,
        &params,
        n_threads,
        dist0_pop_objective,
        accept_worse_prob,
        maximize,
        false,
    );
    let final_partition = result.expect("tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}

// ============================================
// == Hill-climbing monotonicity on 6x6 grid ==
// ============================================

#[rstest]
fn test_tilted_hill_climbing_maximize_grid() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let initial_score = dist0_pop_objective(&graph, &partition);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 1000,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        dist0_pop_objective,
        0.0, // pure hill-climbing
        true,
        false,
    )
    .unwrap();
    let final_score = dist0_pop_objective(&graph, &final_partition);
    assert!(
        final_score >= initial_score,
        "Hill-climbing maximize: final {} < initial {}",
        final_score,
        initial_score
    );
}

#[rstest]
fn test_tilted_hill_climbing_minimize_grid() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let initial_score = dist0_pop_objective(&graph, &partition);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 1000,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        dist0_pop_objective,
        0.0, // pure hill-climbing
        false,
        false,
    )
    .unwrap();
    let final_score = dist0_pop_objective(&graph, &final_partition);
    assert!(
        final_score <= initial_score,
        "Hill-climbing minimize: final {} > initial {}",
        final_score,
        initial_score
    );
}

#[test]
fn test_tilted_rejects_zero_threads() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 1,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let err = multi_tilted_runs(
        &graph,
        partition,
        &params,
        0,
        dist0_pop_objective,
        0.0,
        true,
        false,
    )
    .expect_err("n_threads=0 should fail fast");
    assert!(err.contains("n_threads must be at least 1"));
}

#[test]
fn test_tilted_returns_terminal_partition_not_best_seen() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let initial_assignments = partition.assignments.clone();
    let initial_assignments_ref = &initial_assignments;
    let objective = move |_graph: &Graph, p: &Partition| {
        if p.assignments == *initial_assignments_ref {
            1.0
        } else {
            0.0
        }
    };
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 8,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let final_partition =
        multi_tilted_runs(&graph, partition, &params, 1, objective, 1.0, true, false).unwrap();
    assert_ne!(
        final_partition.assignments, initial_assignments,
        "tilted runs should return the terminal chain state, not the best-seen plan"
    );
}

#[test]
fn test_tilted_stats_writer_records_accepted_steps() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 8,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let mut stats_writer = CountingStatsWriter::new();
    multi_tilted_runs_with_writer(
        &graph,
        partition,
        &params,
        1,
        dist0_pop_objective,
        1.0,
        true,
        Some(&mut stats_writer),
        None,
        false,
    )
    .unwrap();
    assert_eq!(stats_writer.init_calls, 1);
    assert_eq!(
        stats_writer.steps,
        (1..=params.num_steps)
            .map(|step| (step, 0))
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_tilted_scores_writer_records_every_step() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 12,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let path = std::env::temp_dir().join(format!(
        "frcw_tilted_scores_{}_{}.csv",
        std::process::id(),
        RNG_SEED
    ));
    let output = Box::new(std::io::BufWriter::new(fs::File::create(&path).unwrap()));
    let mut scores_writer = ScoresWriter::new(output);
    let final_partition = multi_tilted_runs_with_writer(
        &graph,
        partition,
        &params,
        1,
        dist0_pop_objective,
        0.0,
        true,
        None,
        Some(&mut scores_writer),
        false,
    )
    .unwrap();
    let scores = fs::read_to_string(&path).unwrap();
    let lines = scores.lines().collect::<Vec<_>>();
    assert_eq!(lines[0], "step,score,best_score");
    assert_eq!(lines.len(), params.num_steps as usize + 2);

    let mut previous_best = f64::NEG_INFINITY;
    for (idx, line) in lines.iter().enumerate().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].parse::<usize>().unwrap(), idx - 1);
        let best_score = fields[2].parse::<f64>().unwrap();
        assert!(best_score >= previous_best);
        previous_best = best_score;
    }
    let final_score = dist0_pop_objective(&graph, &final_partition);
    let last_fields = lines.last().unwrap().split(',').collect::<Vec<_>>();
    assert_eq!(last_fields[1].parse::<f64>().unwrap(), final_score);

    fs::remove_file(path).unwrap();
}

// ===================================
// == Objective function unit tests ==
// ===================================

#[test]
fn test_election_wins_scoring_tied() {
    // 6x6 grid: every district has a_share sum = 3, b_share sum = 3 (tied).
    // With target "a": 0 wins, best losing share = 3/6 = 0.5.
    // But 0.5 is not > 0.5, so it counts as a loss.
    // Exact ties should map to the largest float below 1.0 so the tiebreaker
    // remains strictly below the next integer win count.
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let score = obj_fn(&graph, &partition);
    let expected = f64::from_bits(1.0f64.to_bits() - 1);
    assert!(
        score.to_bits() == expected.to_bits(),
        "Expected score {:?} for tied districts, got {:?}",
        expected,
        score
    );
}

#[test]
fn test_election_wins_scoring_target_b() {
    // Same tied districts, but target "b" -- should give the same score.
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"b","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let score = obj_fn(&graph, &partition);
    let expected = f64::from_bits(1.0f64.to_bits() - 1);
    assert!(
        score.to_bits() == expected.to_bits(),
        "Expected score {:?} for tied districts with target b, got {:?}",
        expected,
        score
    );
}

#[test]
fn test_required_node_cols_election_wins() {
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"DEM_GOV","votes_b":"REP_GOV"},{"votes_a":"DEM_SEN","votes_b":"REP_SEN"}],"target":"a","aggregation":"mean"}"#;
    let cols = required_node_cols(config);
    assert_eq!(cols, vec!["DEM_GOV", "REP_GOV", "DEM_SEN", "REP_SEN"]);
}

#[test]
fn test_required_node_cols_gingles() {
    let config =
        r#"{"objective":"gingles_partial","threshold":0.5,"min_pop":"BVAP","total_pop":"VAP"}"#;
    let cols = required_node_cols(config);
    assert_eq!(cols, vec!["BVAP", "VAP"]);
}

// ============================================================================
// == Tilted runs on IA counties (99 nodes, 4 districts, real election data) ==
// ============================================================================

#[rstest]
fn test_tilted_iowa_election_wins(
    #[values(1, 4)] n_threads: usize,
    #[values(true, false)] maximize: bool,
) {
    let election_cols = vec!["PRES16D", "PRES16R"];
    let (graph, partition) = fixture_with_attributes("IA", election_cols);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.2;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"PRES16D","votes_b":"PRES16R"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let final_partition = multi_tilted_runs(
        &graph, partition, &params, n_threads, obj_fn, 0.05, maximize, false,
    )
    .expect("IA tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}

#[test]
fn test_tilted_iowa_hill_climbing_maximize() {
    let election_cols = vec!["PRES16D", "PRES16R"];
    let (graph, partition) = fixture_with_attributes("IA", election_cols);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.2;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"PRES16D","votes_b":"PRES16R"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let initial_score = obj_fn(&graph, &partition);
    let final_partition =
        multi_tilted_runs(&graph, partition, &params, 1, obj_fn, 0.0, true, false).unwrap();
    let final_score = obj_fn(&graph, &final_partition);
    assert!(
        final_score >= initial_score,
        "IA hill-climbing maximize: final {} < initial {}",
        final_score,
        initial_score
    );
}

// ================================================================================
// == Tilted runs on VA precincts (2439 nodes, 11 districts, real election data) ==
// ================================================================================

#[rstest]
fn test_tilted_virginia_election_wins(#[values(1, 4)] n_threads: usize) {
    let election_cols = vec!["G18DSEN", "G18RSEN"];
    let (graph, partition) = fixture_with_attributes("VA", election_cols);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.05;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"G18DSEN","votes_b":"G18RSEN"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let final_partition = multi_tilted_runs(
        &graph, partition, &params, n_threads, obj_fn, 0.05, true, false,
    )
    .expect("VA tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}

#[test]
fn test_tilted_virginia_multi_election() {
    // Test with two elections at once.
    let election_cols = vec!["G18DSEN", "G18RSEN", "G16DPRS", "G16RPRS"];
    let (graph, partition) = fixture_with_attributes("VA", election_cols);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.05;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"G18DSEN","votes_b":"G18RSEN"},{"votes_a":"G16DPRS","votes_b":"G16RPRS"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let final_partition =
        multi_tilted_runs(&graph, partition, &params, 2, obj_fn, 0.05, true, false)
            .expect("VA multi-election tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}
