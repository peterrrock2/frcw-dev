// Functional tests for short bursts optimization.
use frcw::graph::Graph;
use frcw::objectives::make_objective;
use frcw::partition::Partition;
use frcw::recom::short_bursts::multi_short_bursts_with_writer;
use frcw::recom::RecomProposal;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{ScoresWriter, SelfLoopCounts, StatsWriter};
use std::io::Result as IOResult;

use test_fixtures::fixture_with_attributes;

const RNG_SEED: u64 = 153434375;

// =================================================================================
// == Test helpers
// =================================================================================

fn make_params(num_steps: u64) -> RecomParams {
    RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    }
}

/// Records every partition and sample number passed to writer.step.
struct RecordingWriter {
    partitions: Vec<Partition>,
    steps: Vec<u64>,
}

impl RecordingWriter {
    fn new() -> Self {
        RecordingWriter {
            partitions: Vec::new(),
            steps: Vec::new(),
        }
    }
}

impl StatsWriter for RecordingWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> IOResult<()> {
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        partition: &Partition,
        _proposal: &RecomProposal,
        _counts: &SelfLoopCounts,
    ) -> IOResult<()> {
        self.partitions.push(partition.clone());
        self.steps.push(step);
        Ok(())
    }

    fn close(&mut self) -> IOResult<()> {
        Ok(())
    }
}

// =================================================================================
// == Stats writer receives every accepted chain step (write_best_only=false).
// =================================================================================

#[test]
fn test_short_bursts_writer_records_every_accepted_step() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let objective = make_objective(config);

    let burst_length: usize = 5;
    let n_threads: usize = 1;
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 50,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };

    let mut writer = RecordingWriter::new();
    multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        n_threads,
        objective,
        true,
        burst_length,
        Some(&mut writer),
        None,
        false,
        false, // write_best_only=false: record every step
    )
    .unwrap();

    // Sample numbers must be strictly sequential starting at 1.
    for (i, &s) in writer.steps.iter().enumerate() {
        assert_eq!(
            s,
            (i + 1) as u64,
            "Sample number at index {} is {}, expected {}",
            i,
            s,
            i + 1
        );
    }

    // Total writer calls must equal num_steps - 1: the seed written by init
    // counts as the first output record, so chain steps fill the remaining slots.
    let expected_steps = params.num_steps - 1;
    assert_eq!(
        writer.steps.len() as u64,
        expected_steps,
        "Expected {} writer calls (num_steps - 1), got {}",
        expected_steps,
        writer.steps.len()
    );
}

// =================================================================================
// == Stats writer receives only new global bests (write_best_only=true).
// =================================================================================

#[test]
fn test_short_bursts_write_best_only_records_only_improvements() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let objective = make_objective(config);
    let initial_score = objective.score(&graph, &partition);

    let params = make_params(500);
    let mut writer = RecordingWriter::new();
    multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        1,
        objective,
        true,
        10,
        Some(&mut writer),
        None,
        false,
        true, // write_best_only=true: record only improvements
    )
    .unwrap();

    // Sample numbers must be sequential (1, 2, 3, ...).
    for (i, &s) in writer.steps.iter().enumerate() {
        assert_eq!(s, (i + 1) as u64, "non-sequential sample number at index {}", i);
    }

    // Every partition recorded must have a score >= the previous recorded score.
    // A violation would mean we wrote a plan that was not actually an improvement.
    let mut prev_score = initial_score;
    for (i, partition) in writer.partitions.iter().enumerate() {
        let score = objective.score(&graph, partition);
        assert!(
            score >= prev_score,
            "write-best-only writer received a non-improving partition at index {}: \
             score {} < previous {}",
            i,
            score,
            prev_score
        );
        prev_score = score;
    }
}

// =================================================================================
// == Cross-validation: write_best_only=true is a strict subset of
// == write_best_only=false, and scores_writer row count matches improvement count.
//
// Both runs use the same RNG seed, so the underlying random walk is identical.
// The write_best_only=false run records every accepted step; write_best_only=true
// records only the steps that were new global bests. Every partition in the
// best-only run must appear (in order) within the all-steps run.
// =================================================================================

#[test]
fn test_short_bursts_write_best_only_cross_validation() {
    use std::fs;

    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let burst_length = 5;
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 200,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
    };

    // --- Run 1: record every step ---
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let mut all_writer = RecordingWriter::new();
    let scores_path = std::env::temp_dir().join(format!(
        "frcw_sb_xval_scores_{}_{}.csv",
        std::process::id(),
        RNG_SEED
    ));
    let scores_out = Box::new(std::io::BufWriter::new(
        fs::File::create(&scores_path).unwrap(),
    ));
    let mut scores_writer = ScoresWriter::new(scores_out);

    multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        1,
        make_objective(config),
        true,
        burst_length,
        Some(&mut all_writer),
        Some(&mut scores_writer),
        false,
        false, // write_best_only=false
    )
    .unwrap();

    // --- Run 2: record only improvements (same seed = same walk) ---
    let (graph2, partition2) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let mut best_writer = RecordingWriter::new();

    multi_short_bursts_with_writer(
        &graph2,
        partition2,
        &params,
        1,
        make_objective(config),
        true,
        burst_length,
        Some(&mut best_writer),
        None,
        false,
        true, // write_best_only=true
    )
    .unwrap();

    // write_best_only=true must produce strictly fewer records: with 200 steps
    // on a small grid, some steps will inevitably be non-improvements.
    assert!(
        best_writer.partitions.len() < all_writer.partitions.len(),
        "write_best_only=true ({} records) should produce fewer records than \
         write_best_only=false ({} records) over {} steps",
        best_writer.partitions.len(),
        all_writer.partitions.len(),
        params.num_steps
    );

    // Every partition from write_best_only=true must appear in the all-steps run.
    // Since both runs use the same seed, they take the same steps in the same
    // order; improvements are a subsequence of all steps.
    let mut search_from = 0;
    for (i, best_p) in best_writer.partitions.iter().enumerate() {
        let found = all_writer.partitions[search_from..]
            .iter()
            .position(|p| p.assignments == best_p.assignments);
        match found {
            Some(offset) => search_from += offset + 1,
            None => panic!(
                "improvement {} (assignments {:?}...) not found as a subsequence in \
                 all-steps run (searched from index {})",
                i,
                &best_p.assignments[..4.min(best_p.assignments.len())],
                search_from
            ),
        }
    }

    // The scores CSV must have exactly one data row per new global best
    // (matching the write_best_only=true writer call count).
    let scores_content = fs::read_to_string(&scores_path).unwrap();
    let scores_lines: Vec<&str> = scores_content.lines().collect();
    // lines[0] = header, lines[1] = init row at step 0, lines[2..] = data rows.
    let data_rows = scores_lines.len().saturating_sub(2);
    assert_eq!(
        data_rows,
        best_writer.partitions.len(),
        "scores CSV has {} data rows but {} new global bests were found",
        data_rows,
        best_writer.partitions.len()
    );

    // best_score column must be strictly non-decreasing.
    let mut prev_best: f64 = f64::NEG_INFINITY;
    for line in scores_lines.iter().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let best = fields[2].parse::<f64>().unwrap();
        assert!(best >= prev_best, "best_score decreased: {} < {}", best, prev_best);
        prev_best = best;
    }

    fs::remove_file(scores_path).unwrap();
}

// =================================================================================
// == Hill-climbing: maximize=true should not return a partition worse than the
// == starting point.
// =================================================================================

#[test]
fn test_short_bursts_hill_climbing_maximize() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let objective = make_objective(config);
    let initial_score = objective.score(&graph, &partition);

    let params = make_params(1000);
    let final_partition = multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        1,
        objective,
        true, // maximize
        10,
        None,
        None,
        false,
        true, // write_best_only: hill-climbing semantics
    )
    .unwrap();

    let final_score = objective.score(&graph, &final_partition);
    assert!(
        final_score >= initial_score,
        "Hill-climbing maximize: final score {} < initial score {}",
        final_score,
        initial_score
    );
}
