// Functional tests for short bursts optimization.
use frcw::graph::Graph;
use frcw::objectives::make_objective;
use frcw::partition::Partition;
use frcw::recom::opt::{multi_short_bursts, multi_short_bursts_incremental_with_writer};
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
// == Bug regression test: scoring must use the post-update partition.
//
// The pre-fix bug computed score(P_i) BEFORE calling partition.update(), then
// stored P_{i+1} paired with score(P_i). On the first step of every burst this
// means score(P_0) >= best_score(P_0) is always true, so P_1 is unconditionally
// stored as "best" no matter how bad it actually is.
//
// This test uses an objective where only the initial partition scores 1.0 and all
// others score 0.0. A correct optimizer keeps returning to the initial partition;
// the buggy optimizer wanders away after the first burst.
// =================================================================================

#[test]
fn test_short_bursts_returns_partition_matching_claimed_best_score() {
    let (graph, partition) = fixture_with_attributes("6x6", vec![]);
    // Box::leak gives us a 'static reference, which is Copy + Send, satisfying
    // multi_short_bursts's Copy bound on the objective closure.
    let initial: &'static [u32] =
        Box::leak(partition.assignments.clone().into_boxed_slice());

    let objective = move |_graph: &Graph, p: &Partition| -> f64 {
        if p.assignments.as_slice() == initial { 1.0 } else { 0.0 }
    };

    let params = make_params(30);
    let final_partition =
        multi_short_bursts(&graph, partition, &params, 1, objective, true, 5, false).unwrap();

    // With correct scoring: every within-burst step has actual score 0.0 < 1.0
    // (the initial best), so no step ever passes the >= check and the optimizer
    // returns the unchanged initial partition.
    //
    // With the pre-fix bug: score(P_0) = 1.0 is computed before the update, then
    // partition advances to P_1. Since 1.0 >= 1.0, P_1 (score 0.0) is stored as
    // the burst best. After the first burst the global partition is P_1. All
    // subsequent bursts do the same random walk and never return to P_0.
    assert_eq!(
        final_partition.assignments.as_slice(),
        initial,
        "Short bursts returned a partition other than the initial, which has the unique \
         maximum score. This indicates the optimizer stored the post-step partition \
         before scoring it (pre-fix scoring bug)."
    );
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
    multi_short_bursts_incremental_with_writer(
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

    // Total writer calls must equal n_rounds * burst_length * n_threads.
    let n_rounds = params.num_steps.div_ceil((n_threads * burst_length) as u64);
    let expected_steps = n_rounds * (n_threads * burst_length) as u64;
    assert_eq!(
        writer.steps.len() as u64,
        expected_steps,
        "Expected {} writer calls (n_rounds={} x burst={}), got {}",
        expected_steps,
        n_rounds,
        burst_length,
        writer.partitions.len()
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
    multi_short_bursts_incremental_with_writer(
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
// == Scores writer is only called when a new global best is found.
// =================================================================================

#[test]
fn test_short_bursts_scores_writer_only_on_new_bests() {
    use std::fs;

    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let objective = make_objective(config);

    let params = make_params(200);
    let path = std::env::temp_dir().join(format!(
        "frcw_sb_scores_{}_{}.csv",
        std::process::id(),
        RNG_SEED
    ));
    let output = Box::new(std::io::BufWriter::new(fs::File::create(&path).unwrap()));
    let mut scores_writer = ScoresWriter::new(output);

    multi_short_bursts_incremental_with_writer(
        &graph,
        partition,
        &params,
        1,
        objective,
        true,
        5,
        None,
        Some(&mut scores_writer),
        false,
        false,
    )
    .unwrap();

    let content = fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    // Must have at least a header and the init row.
    assert!(
        lines.len() >= 2,
        "scores CSV should have at least a header and init row"
    );

    // best_score column must be non-decreasing across all rows.
    let mut prev_best: f64 = f64::NEG_INFINITY;
    for line in lines.iter().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 3,
            "scores CSV row has fewer than 3 fields: {}",
            line
        );
        let best = fields[2].parse::<f64>().unwrap();
        assert!(
            best >= prev_best,
            "best_score decreased: {} < {}",
            best,
            prev_best
        );
        prev_best = best;
    }

    fs::remove_file(path).unwrap();
}
