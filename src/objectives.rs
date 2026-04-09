//! Objective (score) functions for partition optimization.
//!
//! Each objective is configured via a JSON string passed to [`make_objective_fn`].
//! The JSON schema for each objective is documented on its corresponding variant
//! of [`ObjectiveConfig`].
use crate::graph::Graph;
use crate::partition::Partition;
use crate::stats::partition_attr_sums;
use serde_json::Value;

/// Aggregation method for per-district scores.
#[derive(Clone, Copy, Debug)]
pub enum Aggregation {
    Mean,
    Min,
    Sum,
}

/// Largest `f64` strictly smaller than 1.0.
const ONE_BELOW: f64 = f64::from_bits(1.0f64.to_bits() - 1);

impl Aggregation {
    fn from_str(s: &str) -> Aggregation {
        match s {
            "mean" => Aggregation::Mean,
            "min" => Aggregation::Min,
            "sum" => Aggregation::Sum,
            other => panic!(
                "Unknown aggregation '{}'. Use 'mean', 'min', or 'sum'.",
                other
            ),
        }
    }

    fn apply(&self, values: &[f64]) -> f64 {
        match self {
            Aggregation::Mean => values.iter().sum::<f64>() / values.len() as f64,
            Aggregation::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
            Aggregation::Sum => values.iter().sum(),
        }
    }
}

/// A parsed, `Copy`-able representation of an objective function's configuration.
/// The JSON schema for each variant is documented below.
#[derive(Clone, Copy)]
pub enum ObjectiveConfig {
    /// Maximize Gingles opportunity districts with next-partial-district augmentation.
    ///
    /// The score is the number of districts where the minority share exceeds
    /// `threshold`, plus a fractional tiebreaker from the highest sub-threshold
    /// district (so that plans closer to gaining a new opportunity district are
    /// preferred).
    ///
    /// JSON schema:
    /// ```json
    /// {
    ///   "objective": "gingles_partial",
    ///   "threshold": 0.5,
    ///   "min_pop": "BVAP",
    ///   "total_pop": "VAP"
    /// }
    /// ```
    ///
    /// Fields:
    /// - `threshold`: minority share threshold, must be in (0, 1)
    /// - `min_pop`: node attribute column for the minority population (integer-valued)
    /// - `total_pop`: node attribute column for the total population (integer-valued)
    GinglesPartial {
        threshold: f64,
        min_pop_col: &'static str,
        total_pop_col: &'static str,
    },

    /// Maximize (or minimize) the number of districts won by a target party
    /// across a set of elections, with a partial-district tiebreaker.
    ///
    /// For each election, the score is the number of districts where the target
    /// party's vote total exceeds the other party's, plus a fractional tiebreaker
    /// from the closest losing district (so the optimizer has gradient signal
    /// between integer win counts). Per-election scores are aggregated via
    /// `aggregation`.
    ///
    /// JSON schema:
    /// ```json
    /// {
    ///   "objective": "election_wins",
    ///   "elections": [
    ///     {"votes_a": "DEM_GOV_18", "votes_b": "REP_GOV_18"},
    ///     {"votes_a": "DEM_SEN_18", "votes_b": "REP_SEN_18"}
    ///   ],
    ///   "target": "a",
    ///   "aggregation": "mean"
    /// }
    /// ```
    ///
    /// Fields:
    /// - `elections`: array of objects, each with `votes_a` and `votes_b` naming
    ///   node attribute columns containing integer vote counts
    /// - `target`: `"a"` or `"b"` -- which party's wins to count
    /// - `aggregation`: one of `"mean"`, `"min"`, or `"sum"`
    ElectionWins {
        elections: &'static [(&'static str, &'static str)],
        target_a: bool,
        aggregation: Aggregation,
    },

    /// Maximize a Polsby-Popper compactness aggregate across districts.
    ///
    /// The Polsby-Popper score for a district is `4 * pi * area / perimeter^2`.
    /// Per-district scores are aggregated via `aggregation`.
    ///
    /// JSON schema:
    /// ```json
    /// {
    ///   "objective": "polsby_popper",
    ///   "area_col": "area",
    ///   "perim_col": "perim",
    ///   "shared_perim_col": "shared_perim",
    ///   "aggregation": "mean"
    /// }
    /// ```
    ///
    /// Fields:
    /// - `area_col`: node attribute (float-parseable string) giving precinct area
    /// - `perim_col`: node attribute (float-parseable string) giving the total
    ///   perimeter of each precinct (including shared boundaries with neighbors)
    /// - `shared_perim_col`: edge attribute (`graph.edge_attr`) giving the shared
    ///   perimeter between adjacent precincts; must be loaded via `required_edge_cols`
    ///   and passed to `from_networkx`
    /// - `aggregation`: one of `"mean"`, `"min"`, or `"sum"`
    PolsbyPopper {
        area_col: &'static str,
        perim_col: &'static str,
        shared_perim_col: &'static str,
        aggregation: Aggregation,
    },
}

impl ObjectiveConfig {
    /// Evaluates the objective for the given graph and partition.
    pub fn score(&self, graph: &Graph, partition: &Partition) -> f64 {
        match self {
            ObjectiveConfig::GinglesPartial {
                threshold,
                min_pop_col,
                total_pop_col,
            } => {
                let min_pops = partition_attr_sums(graph, partition, min_pop_col);
                let total_pops = partition_attr_sums(graph, partition, total_pop_col);
                let shares: Vec<f64> = min_pops
                    .iter()
                    .zip(total_pops.iter())
                    .map(|(&m, &t)| m as f64 / t as f64)
                    .collect();
                let opportunity_count = shares.iter().filter(|&s| s >= threshold).count();
                let mut sorted_below: Vec<f64> =
                    shares.into_iter().filter(|s| s < threshold).collect();
                sorted_below.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let next_highest = sorted_below.last().copied().unwrap_or(0.0);
                opportunity_count as f64 + (next_highest / threshold)
            }

            ObjectiveConfig::ElectionWins {
                elections,
                target_a,
                aggregation,
            } => {
                let mut election_scores: Vec<f64> = Vec::with_capacity(elections.len());
                for &(col_a, col_b) in elections.iter() {
                    let sums_a = partition_attr_sums(graph, partition, col_a);
                    let sums_b = partition_attr_sums(graph, partition, col_b);

                    let mut wins: usize = 0;
                    let mut best_losing_tiebreak: f64 = 0.0;
                    for (&va, &vb) in sums_a.iter().zip(sums_b.iter()) {
                        let total = va as f64 + vb as f64;
                        if total == 0.0 {
                            continue;
                        }
                        let target_votes = if *target_a { va } else { vb };
                        let other_votes = if *target_a { vb } else { va };
                        if target_votes > other_votes {
                            wins += 1;
                        } else {
                            // Keep ties as losses, but place them strictly below the next
                            // integer win count so the fractional tiebreaker stays in [0, 1).
                            let tiebreak = if target_votes == other_votes {
                                ONE_BELOW
                            } else {
                                (target_votes as f64 / total) / 0.5
                            };
                            if tiebreak > best_losing_tiebreak {
                                best_losing_tiebreak = tiebreak;
                            }
                        }
                    }
                    // Tiebreaker stays in [0, 1), so it sits between integer win counts.
                    election_scores.push(wins as f64 + best_losing_tiebreak);
                }
                aggregation.apply(&election_scores)
            }

            ObjectiveConfig::PolsbyPopper {
                area_col,
                perim_col,
                shared_perim_col,
                aggregation,
            } => {
                let area_vals = graph
                    .attr
                    .get(*area_col)
                    .unwrap_or_else(|| panic!("Missing node attribute '{}'", area_col));
                let perim_vals = graph
                    .attr
                    .get(*perim_col)
                    .unwrap_or_else(|| panic!("Missing node attribute '{}'", perim_col));
                let shared_perim_vals = graph
                    .edge_attr
                    .get(*shared_perim_col)
                    .unwrap_or_else(|| panic!("Missing edge attribute '{}'", shared_perim_col));

                let n_dists = partition.num_dists as usize;
                let mut area_d = vec![0.0f64; n_dists];
                let mut perim_d = vec![0.0f64; n_dists];

                for (dist, nodes) in partition.dist_nodes.iter().enumerate() {
                    for &node in nodes {
                        area_d[dist] += area_vals[node].parse::<f64>().unwrap_or(0.0);
                        perim_d[dist] += perim_vals[node].parse::<f64>().unwrap_or(0.0);
                    }
                }

                // Subtract 2 * shared_perim for edges internal to each district
                // (cancels out the double-counted shared boundaries).
                for (edge_idx, edge) in graph.edges.iter().enumerate() {
                    let d_u = partition.assignments[edge.0] as usize;
                    let d_v = partition.assignments[edge.1] as usize;
                    if d_u == d_v {
                        perim_d[d_u] -= 2.0 * shared_perim_vals[edge_idx];
                    }
                }

                let pp_scores: Vec<f64> = (0..n_dists)
                    .map(|d| {
                        let p = perim_d[d];
                        if p <= 0.0 {
                            0.0
                        } else {
                            4.0 * std::f64::consts::PI * area_d[d] / (p * p)
                        }
                    })
                    .collect();

                aggregation.apply(&pp_scores)
            }
        }
    }
}

/// Returns a `Copy + Clone + Send` closure that scores a partition according
/// to the given JSON objective configuration.
///
/// Dispatches on the `"objective"` field:
/// - `"gingles_partial"` -- see [`ObjectiveConfig::GinglesPartial`]
/// - `"election_wins"` -- see [`ObjectiveConfig::ElectionWins`]
/// - `"polsby_popper"` -- see [`ObjectiveConfig::PolsbyPopper`]
pub fn make_objective_fn(config: &str) -> impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy {
    let data: Value = serde_json::from_str(config).unwrap();
    let obj_type = data["objective"].as_str().unwrap();

    // Use Box::leak so that &'static str column names can be captured by Copy closures.
    let leak_str = |v: &Value, field: &str| -> &'static str {
        &*Box::leak(
            v[field]
                .as_str()
                .unwrap_or_else(|| panic!("Missing field '{}' in objective config", field))
                .to_owned()
                .into_boxed_str(),
        )
    };

    let obj: ObjectiveConfig = match obj_type {
        "gingles_partial" => {
            let threshold = data["threshold"].as_f64().unwrap();
            assert!(
                threshold > 0.0 && threshold < 1.0,
                "'threshold' must be in (0, 1)"
            );
            ObjectiveConfig::GinglesPartial {
                threshold,
                min_pop_col: leak_str(&data, "min_pop"),
                total_pop_col: leak_str(&data, "total_pop"),
            }
        }
        "election_wins" => {
            let elections_arr = data["elections"]
                .as_array()
                .unwrap_or_else(|| panic!("Missing field 'elections' in objective config"));
            let mut pairs: Vec<(&'static str, &'static str)> =
                Vec::with_capacity(elections_arr.len());
            for e in elections_arr {
                let a = leak_str(e, "votes_a");
                let b = leak_str(e, "votes_b");
                pairs.push((a, b));
            }
            let elections: &'static [(&'static str, &'static str)] = Box::leak(pairs.into_boxed_slice());
            let target = data["target"]
                .as_str()
                .unwrap_or_else(|| panic!("Missing field 'target' in objective config"));
            let target_a = match target {
                "a" => true,
                "b" => false,
                other => panic!(
                    "Invalid target '{}'. Use 'a' or 'b'.",
                    other
                ),
            };
            let agg_str = data["aggregation"]
                .as_str()
                .unwrap_or_else(|| panic!("Missing field 'aggregation' in objective config"));
            ObjectiveConfig::ElectionWins {
                elections,
                target_a,
                aggregation: Aggregation::from_str(agg_str),
            }
        }
        "polsby_popper" => {
            let agg_str = data["aggregation"]
                .as_str()
                .unwrap_or_else(|| panic!("Missing field 'aggregation' in objective config"));
            ObjectiveConfig::PolsbyPopper {
                area_col: leak_str(&data, "area_col"),
                perim_col: leak_str(&data, "perim_col"),
                shared_perim_col: leak_str(&data, "shared_perim_col"),
                aggregation: Aggregation::from_str(agg_str),
            }
        }
        other => panic!(
            "Unknown objective '{}'. Supported: 'gingles_partial', 'election_wins', 'polsby_popper'.",
            other
        ),
    };

    move |graph: &Graph, partition: &Partition| -> f64 { obj.score(graph, partition) }
}

/// Returns the node attribute columns required by the given objective config.
///
/// Use this in CLI binaries to ensure these columns are included in the
/// `columns` argument to [`frcw::init::from_networkx`].
pub fn required_node_cols(config: &str) -> Vec<String> {
    let data: Value = serde_json::from_str(config).unwrap();
    match data["objective"].as_str().unwrap() {
        "gingles_partial" => vec![
            data["min_pop"].as_str().unwrap().to_string(),
            data["total_pop"].as_str().unwrap().to_string(),
        ],
        "election_wins" => {
            let mut cols = vec![];
            for e in data["elections"].as_array().unwrap() {
                cols.push(e["votes_a"].as_str().unwrap().to_string());
                cols.push(e["votes_b"].as_str().unwrap().to_string());
            }
            cols
        }
        "polsby_popper" => vec![
            data["area_col"].as_str().unwrap().to_string(),
            data["perim_col"].as_str().unwrap().to_string(),
        ],
        _ => vec![],
    }
}

/// Returns the edge attribute columns required by the given objective config.
///
/// Use this in CLI binaries to determine which columns to pass as `edge_float_cols`
/// to [`frcw::init::from_networkx`].
pub fn required_edge_cols(config: &str) -> Vec<String> {
    let data: Value = serde_json::from_str(config).unwrap();
    match data["objective"].as_str().unwrap() {
        "polsby_popper" => vec![data["shared_perim_col"]
            .as_str()
            .unwrap_or_else(|| panic!("Missing 'shared_perim_col' in polsby_popper config"))
            .to_string()],
        _ => vec![],
    }
}
