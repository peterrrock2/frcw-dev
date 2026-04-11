//! Tilted run optimization CLI for frcw.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_parser, Arg, ArgAction, Command};
use frcw::config::parse_region_weights_config;
use frcw::init::from_networkx;
use frcw::objectives::{make_objective_fn, required_edge_cols, required_node_cols};
use frcw::recom::tilted::multi_tilted_runs_with_writer;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{
    AssignmentsOnlyWriter, BenWriter, CanonicalWriter, JSONLWriter, PcompressWriter, ScoresWriter,
    StatsWriter, TSVWriter,
};
use serde_json::json;
use sha3::{Digest, Sha3_256};
use std::path::PathBuf;
use std::{fs, io};

fn output_buffer(path: &str) -> Box<dyn io::Write + Send> {
    let path = std::path::Path::new(path);
    if path.exists() {
        panic!("Output file already exists.");
    };
    Box::new(io::BufWriter::new(fs::File::create(path).unwrap()))
}

fn make_stats_writer(
    writer_str: &str,
    output_buffer: Box<dyn io::Write + Send>,
) -> Box<dyn StatsWriter> {
    match writer_str {
        "tsv" => Box::new(TSVWriter::new(output_buffer)),
        "jsonl" => Box::new(JSONLWriter::new(false, false, false, output_buffer)),
        "jsonl-full" => Box::new(JSONLWriter::new(true, false, false, output_buffer)),
        "pcompress" => Box::new(PcompressWriter::new(output_buffer)),
        "assignments" => Box::new(AssignmentsOnlyWriter::new(false, output_buffer)),
        "canonicalized-assignments" => Box::new(AssignmentsOnlyWriter::new(true, output_buffer)),
        "canonical" => Box::new(CanonicalWriter::new(output_buffer)),
        "ben" => Box::new(BenWriter::new(output_buffer)),
        bad => panic!("Parameter error: invalid writer '{}'", bad),
    }
}

fn main() {
    let cli = Command::new("frcw_tilted")
        .version("0.1.3")
        .author("Peter Rock <peter@mggg.org>")
        .about("A tilted run optimizer for redistricting")
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
                .help("The total number of chain steps (accepted + rejected)."),
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
                .help("Number of worker threads for parallel tree drawing."),
        )
        .arg(
            Arg::new("accept_worse_prob")
                .long("accept-worse-prob")
                .required(true)
                .value_parser(value_parser!(f64))
                .help(
                    "Probability of accepting a proposal with a worse score. \
                    Must be in [0, 1]. Use 0.0 for pure hill-climbing, 1.0 for a random walk.",
                ),
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
                .help("A JSON-formatted objective function configuration."),
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
        )
        .arg(
            Arg::new("maximize")
                .long("maximize")
                .value_parser(value_parser!(bool))
                .default_value("true")
                .help("If true, maximize the objective. If false, minimize it."),
        )
        .arg(
            Arg::new("writer")
                .long("writer")
                .value_parser(value_parser!(String))
                .default_value("assignments")
                .help(
                    "Writer for chain records when --output-file is provided.\n\
                    \tassignments (default): TXT output with only assignment vectors\n\
                    \tcanonicalized-assignments: TXT output with canonicalized assignment vectors\n\
                    \tcanonical: Standardized JSONL output with assignment vector and sample number\n\
                    \tjsonl: JSON Lines with basic summary statistics\n\
                    \tjsonl-full: JSON Lines with basic summary statistics and recombined nodes\n\
                    \ttsv: Tab-separated proposal statistics\n\
                    \tpcompress: Compressed binary format for post-processing with pcompress\n\
                    \tben: Compressed binary format for post-processing with BEN",
                ),
        )
        .arg(
            Arg::new("output-file")
                .long("output-file")
                .short('o')
                .help(
                    "Path to write chain records. If omitted, chain records are not written.",
                ),
        )
        .arg(
            Arg::new("scores-output-file")
                .long("scores-output-file")
                .help(
                    "Path to write per-step objective scores as CSV with step, score, and best_score.",
                ),
        )
        .arg(
            Arg::new("show-progress")
                .long("show-progress")
                .action(ArgAction::SetTrue)
                .help("Whether to show a progress bar during execution."),
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
    let accept_worse_prob = *matches
        .get_one::<f64>("accept_worse_prob")
        .expect("accept_worse_prob is required");

    let maximize = *matches
        .get_one::<bool>("maximize")
        .expect("maximize is required");
    let writer_str = matches
        .get_one::<String>("writer")
        .expect("writer has a default value")
        .as_str();
    let show_progress = matches.get_flag("show-progress");

    if tol < 0.0 || tol > 1.0 {
        panic!("Parameter error: '--tol' must be between 0 and 1.");
    }
    if n_threads == 0 {
        panic!("Parameter error: '--n-threads' must be at least 1.");
    }
    if accept_worse_prob < 0.0 || accept_worse_prob > 1.0 {
        panic!("Parameter error: '--accept-worse-prob' must be between 0 and 1.");
    }

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
                "path for --graph-json is not valid UTF-8: {:?}",
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
    // Add region weight keys to sum_cols so the user doesn't have to specify them twice.
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
    let edge_cols = required_edge_cols(objective_config);
    for col in required_node_cols(objective_config) {
        if !sum_cols.contains(&col) {
            sum_cols.push(col);
        }
    }

    let (graph, partition) =
        from_networkx(&graph_json, pop_col, assignment_col, sum_cols, edge_cols).unwrap();
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
        "type": "tilted_run",
        "accept_worse_prob": accept_worse_prob,
        "maximize": maximize,
        "show_progress": show_progress,
        "graph_json": graph_json,
    });
    if let Some(path) = matches.get_one::<String>("output-file") {
        meta.as_object_mut()
            .unwrap()
            .insert("output_file".to_string(), json!(path));
        meta.as_object_mut()
            .unwrap()
            .insert("writer".to_string(), json!(writer_str));
    }
    if let Some(path) = matches.get_one::<String>("scores-output-file") {
        meta.as_object_mut()
            .unwrap()
            .insert("scores_output_file".to_string(), json!(path));
    }
    if region_weights.is_some() {
        meta.as_object_mut()
            .unwrap()
            .insert("region_weights".to_string(), json!(region_weights));
    }
    println!("{}", json!({ "meta": meta }).to_string());

    let mut stats_writer: Option<Box<dyn StatsWriter>> =
        match matches.get_one::<String>("output-file") {
            Some(path) => Some(make_stats_writer(writer_str, output_buffer(path))),
            None => {
                if writer_str != "assignments" {
                    panic!("Parameter error: '--writer' requires '--output-file'.");
                }
                None
            }
        };
    let mut scores_writer: Option<ScoresWriter> = matches
        .get_one::<String>("scores-output-file")
        .map(|path| ScoresWriter::new(output_buffer(path)));

    let output = multi_tilted_runs_with_writer(
        &graph,
        partition,
        &params,
        n_threads,
        objective_fn,
        accept_worse_prob,
        maximize,
        stats_writer
            .as_mut()
            .map(|writer| &mut **writer as &mut dyn StatsWriter),
        scores_writer.as_mut(),
        show_progress,
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
