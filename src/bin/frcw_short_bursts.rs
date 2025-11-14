//! Short bursts optimization CLI for frcw.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_parser, Arg, Command};
use frcw::config::parse_region_weights_config;
use frcw::graph::Graph;
use frcw::init::from_networkx;
use frcw::partition::Partition;
use frcw::recom::opt::{multi_short_bursts, ScoreValue};
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::partition_attr_sums;
use serde_json::json;
use serde_json::Value;
use sha3::{Digest, Sha3_256};
use std::path::PathBuf;
use std::{fs, io};

fn make_objective_fn(
    objective_config: &str,
) -> impl Fn(&Graph, &Partition) -> ScoreValue + Send + Clone + Copy {
    let data: Value = serde_json::from_str(objective_config).unwrap();
    let obj_type = data["objective"].as_str().unwrap();

    // For now, we only support Gingles optimization with next-partial-district
    // augmentation. see https://github.com/vrdi/shortbursts-gingles/blob/
    // d9fb26ec313cd93ac80171b23095c4f3dfab0422/state_experiments/gingleator.py#L253
    assert!(obj_type == "gingles_partial");
    let threshold = data["threshold"].as_f64().unwrap();
    assert!(threshold > 0.0 && threshold < 1.0);

    // A hack to share strings between threads:
    // https://stackoverflow.com/a/52367953
    let min_pop_col = &*Box::leak(
        data["min_pop"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );
    let total_pop_col = &*Box::leak(
        data["total_pop"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );

    move |graph: &Graph, partition: &Partition| -> ScoreValue {
        let min_pops = partition_attr_sums(graph, partition, min_pop_col);
        let total_pops = partition_attr_sums(graph, partition, total_pop_col);
        let shares: Vec<f64> = min_pops
            .iter()
            .zip(total_pops.iter())
            .map(|(&m, &t)| m as f64 / t as f64)
            .collect();
        let opportunity_count = shares.iter().filter(|&s| s >= &threshold).count();

        // partial ordering on f64:
        // see https://www.reddit.com/r/rust/comments/29kia3/no_ord_for_f32/
        // see https://doc.rust-lang.org/std/vec/struct.Vec.html#method.sort_by
        let mut sorted_below = shares
            .into_iter()
            .filter(|s| s < &threshold)
            .collect::<Vec<f64>>();
        sorted_below.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let next_highest = match sorted_below.last() {
            Some(&v) => v,
            None => 0.0,
        };
        opportunity_count as f64 + (next_highest / threshold)
    }
}

fn main() {
    let cli = Command::new("frcw_short_bursts")
        .version("0.1.3")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("A short bursts optimizer for redistricting")
        .arg(
            Arg::new("graph_json")
                .long("graph-json")
                .required(true)
                .value_parser(value_parser!(String))
                .help("The path of the dual graph (in NetworkX format)."),
        )
        .arg(
            Arg::new("n_steps")
                .long("n-steps")
                .required(true)
                .value_parser(value_parser!(u64))
                .help("The number of proposals to generate."),
        )
        .arg(
            Arg::new("tol")
                .long("tol")
                .required(true)
                .value_parser(value_parser!(f64))
                .help("The relative population tolerance."),
        )
        .arg(
            Arg::new("pop_col")
                .long("pop-col")
                .required(true)
                .value_parser(value_parser!(String))
                .help("The name of the total population column in the graph metadata."),
        )
        .arg(
            Arg::new("assignment_col")
                .long("assignment-col")
                .required(true)
                .value_parser(value_parser!(String))
                .help("The name of the assignment column in the graph metadata."),
        )
        .arg(
            Arg::new("rng_seed")
                .long("rng-seed")
                .required(true)
                .value_parser(value_parser!(u64))
                .help("The seed of the RNG used to draw proposals."),
        )
        .arg(
            Arg::new("n_threads")
                .long("n-threads")
                .required(false)
                .value_parser(value_parser!(usize))
                .default_value("1")
                .help("The number of threads to use."),
        )
        .arg(
            Arg::new("burst_length")
                .long("burst-length")
                .value_parser(value_parser!(usize))
                .required(true)
                .help("The number of accepted steps per short burst."),
        )
        .arg(
            Arg::new("sum_cols")
                .long("sum-cols")
                .value_parser(value_parser!(Option<String>))
                .num_args(1..)
                .default_value(None)
                .help("Additional columns in the graph metadata to sum over districts."),
        )
        .arg(
            Arg::new("objective")
                .long("objective")
                .required(true)
                .help("A JSON-formatted objective function configuration>"),
        )
        .arg(
            Arg::new("region_weights")
                .long("region-weights")
                .value_parser(value_parser!(String))
                .default_value("")
                .help(
                    "Region columns with weights for region-aware ReCom. \
                    Must be entered into the command line using the format:\n\
                    \t'{\"region_col1\": weight1, \"region_col2\": weight2, ...}'",
                ),
        );

    let matches = cli.get_matches();

    let n_steps = *matches
        .get_one::<u64>("n_steps")
        .expect("n_steps is required");
    let n_threads = *matches
        .get_one::<usize>("n_threads")
        .expect("n_threads is required");
    let rng_seed = *matches
        .get_one::<u64>("rng_seed")
        .expect("rng_seed is required");
    let tol = *matches.get_one::<f64>("tol").expect("tol is required");
    let burst_length = *matches
        .get_one::<usize>("burst_length")
        .expect("burst_length is required");

    let graph_path = matches
        .get_one::<String>("graph_json")
        .expect("graph_json is required");
    let graph_path_buf = PathBuf::from(graph_path);
    let graph_json = match fs::canonicalize(&graph_path_buf)
        .map_err(|e| format!("Could not canonicalize path {:?}: {e}", graph_path_buf))
        .expect(format!("Could not create fs buffer from {graph_path}").as_str())
        .into_os_string()
        .into_string()
        .map_err(|_| {
            format!(
                "path for --graph-jaon is not valid UTF-8: {:?}",
                graph_path_buf
            )
        }) {
        Ok(s) => s,
        Err(e) => panic!("{}", e),
    };

    let pop_col = matches
        .get_one::<String>("pop_col")
        .expect("pop_col is required")
        .as_str();
    let assignment_col = matches
        .get_one::<String>("assignment_col")
        .expect("assignment_col is required")
        .as_str();
    let mut sum_cols: Vec<String> = matches
        .get_many::<String>("sum_cols")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();
    let region_weights_raw = (*matches.get_one::<String>("region_weights").unwrap()).as_str();
    let region_weights = parse_region_weights_config(region_weights_raw);
    // Add the keys in the region weights to sum_cols if they are not there already
    // so that the user doesn't have to
    if let Some(weight_pairs_vec) = &region_weights {
        for (key, _) in weight_pairs_vec.iter() {
            if !sum_cols.contains(&key) {
                sum_cols.push(key.clone().to_string());
            }
        }
    }

    let objective_config = matches
        .get_one::<String>("objective")
        .expect("objective is required")
        .as_str();
    let objective_fn = make_objective_fn(objective_config);

    if tol < 0.0 || tol > 1.0 {
        panic!("Parameter error: '--tol' must be between 0 and 1.");
    }

    let (graph, partition) = from_networkx(&graph_json, pop_col, assignment_col, sum_cols).unwrap();
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let params = RecomParams {
        min_pop: ((1.0 - tol) * avg_pop as f64).ceil() as u32,
        max_pop: ((1.0 + tol) * avg_pop as f64).floor() as u32,
        num_steps: n_steps,
        rng_seed: rng_seed,
        balance_ub: 0,
        variant: match region_weights {
            None => RecomVariant::DistrictPairsRMST,
            Some(_) => RecomVariant::DistrictPairsRegionAware,
        },
        region_weights: region_weights.clone(),
    };

    let mut graph_file = fs::File::open(&graph_json).unwrap();
    let mut graph_hasher = Sha3_256::new();
    io::copy(&mut graph_file, &mut graph_hasher).unwrap();
    let graph_hash = format!("{:x}", graph_hasher.finalize());
    let mut meta = json!({
        "assignment_col": assignment_col,
        "tol": tol,
        "pop_col": pop_col,
        "graph_path": graph_json,
        "graph_sha3": graph_hash,
        "rng_seed": rng_seed,
        "num_threads": n_threads,
        "num_steps": n_steps,
        "parallel": true,
        "type": "short_bursts",
        "burst_length": burst_length,
        "graph_json": graph_json,
    });
    if region_weights.is_some() {
        meta.as_object_mut()
            .unwrap()
            .insert("region_weights".to_string(), json!(region_weights));
    }
    println!("{}", json!({ "meta": meta }).to_string());
    let output = multi_short_bursts(
        &graph,
        partition,
        &params,
        n_threads,
        objective_fn,
        burst_length,
        true,
    );

    match output {
        Ok(final_partition) => {
            let final_score = objective_fn(&graph, &final_partition);
            let output = json!({
                "final_score": final_score,
                "final_assignment": final_partition.assignments
            });
            println!("{}", output.to_string());
        }
        Err(e) => {
            eprintln!("Error during optimization: {}", e);
        }
    }
}
