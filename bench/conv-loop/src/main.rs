mod adapters;
mod assertions;
mod live_runner;
mod mock;
mod reporter;
mod runner;
mod scenario;

use adapters::BenchmarkAdapter;
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "conv-loop-bench", about = "Conversation loop eval harness")]
struct Cli {
    /// Execution mode: "deterministic", "live", or "adapter".
    #[arg(long, default_value = "deterministic")]
    mode: String,

    /// Directory containing scenario TOML files.
    #[arg(long, default_value = "bench/conv-loop/scenarios")]
    scenarios: PathBuf,

    /// Filter scenarios by prefix (e.g. "s01", "l01").
    #[arg(long)]
    filter: Option<String>,

    /// Ollama URL for live/adapter mode.
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Model name for live/adapter mode.
    #[arg(long, default_value = "qwen2.5:7b")]
    model: String,

    /// Benchmark adapter name: "mint" or "tau-bench".
    #[arg(long)]
    adapter: Option<String>,

    /// Path to benchmark data file (JSONL for MINT, JSON for tau-bench).
    #[arg(long)]
    data: Option<PathBuf>,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.mode.as_str() {
        "deterministic" => run_deterministic(&cli),
        "live" => run_live(&cli),
        "adapter" => run_adapter(&cli),
        other => {
            eprintln!("Unknown mode: {other}. Use 'deterministic', 'live', or 'adapter'.");
            std::process::exit(1);
        }
    }
}

fn run_deterministic(cli: &Cli) {
    let scenarios = scenario::load_scenarios(&cli.scenarios, cli.filter.as_deref());
    if scenarios.is_empty() {
        eprintln!("No scenarios found in {:?}", cli.scenarios);
        std::process::exit(1);
    }

    println!(
        "Running {} scenario(s) in deterministic mode\n",
        scenarios.len()
    );

    let mut all_results = Vec::new();

    for s in &scenarios {
        println!("── {} ──", s.name);
        let results = runner::run_scenario(s);
        print_results(&results);
        all_results.push((s.clone(), results));
    }

    println!();
    reporter::print_summary(&all_results);
}

fn run_live(cli: &Cli) {
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    let scenarios = scenario::load_scenarios(&cli.scenarios, cli.filter.as_deref());
    if scenarios.is_empty() {
        eprintln!("No scenarios found in {:?}", cli.scenarios);
        std::process::exit(1);
    }

    let live_config = live_runner::LiveConfig {
        ollama_url: cli.ollama_url.clone(),
        model: cli.model.clone(),
    };

    println!(
        "Running {} scenario(s) in live mode (model: {}, url: {})\n",
        scenarios.len(),
        cli.model,
        cli.ollama_url
    );

    let mut all_results = Vec::new();

    for s in &scenarios {
        println!("── {} ──", s.name);
        let live_result = rt.block_on(live_runner::run_scenario_live(s, &live_config));

        println!(
            "  [{} turns, {} in/{} out tokens, {}ms]",
            live_result.total_turns,
            live_result.total_input_tokens,
            live_result.total_output_tokens,
            live_result.elapsed_ms,
        );

        print_results(&live_result.assertions);
        all_results.push((s.clone(), live_result.assertions));
    }

    println!();
    reporter::print_summary(&all_results);
}

fn run_adapter(cli: &Cli) {
    let adapter_name = cli.adapter.as_deref().unwrap_or_else(|| {
        eprintln!("--adapter required for adapter mode (mint or tau-bench)");
        std::process::exit(1);
    });

    let data_path = cli.data.as_deref().unwrap_or_else(|| {
        eprintln!("--data required for adapter mode");
        std::process::exit(1);
    });

    let adapter: Box<dyn BenchmarkAdapter> = match adapter_name {
        "mint" => Box::new(adapters::mint::MintAdapter),
        "tau-bench" | "tau" => Box::new(adapters::tau_bench::TauBenchAdapter),
        other => {
            eprintln!("Unknown adapter: {other}. Use 'mint' or 'tau-bench'.");
            std::process::exit(1);
        }
    };

    let mut scenarios = adapter.load(data_path).unwrap_or_else(|e| {
        eprintln!("Failed to load {} data: {e}", adapter.name());
        std::process::exit(1);
    });

    // Apply filter if specified.
    if let Some(filter) = &cli.filter {
        scenarios.retain(|s| s.name.to_lowercase().contains(&filter.to_lowercase()));
    }

    if scenarios.is_empty() {
        eprintln!("No scenarios loaded from {data_path:?}");
        std::process::exit(1);
    }

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    let live_config = live_runner::LiveConfig {
        ollama_url: cli.ollama_url.clone(),
        model: cli.model.clone(),
    };

    println!(
        "Running {} {} scenario(s) in adapter mode (model: {}, url: {})\n",
        scenarios.len(),
        adapter.name(),
        cli.model,
        cli.ollama_url
    );

    let mut all_results = Vec::new();

    for s in &scenarios {
        println!("── {} ──", s.name);
        let live_result = rt.block_on(live_runner::run_scenario_live(s, &live_config));

        println!(
            "  [{} turns, {} in/{} out tokens, {}ms]",
            live_result.total_turns,
            live_result.total_input_tokens,
            live_result.total_output_tokens,
            live_result.elapsed_ms,
        );

        print_results(&live_result.assertions);
        all_results.push((s.clone(), live_result.assertions));
    }

    println!();
    reporter::print_summary(&all_results);
}

fn print_results(results: &[assertions::AssertionResult]) {
    for r in results {
        let icon = if r.passed { "✓" } else { "✗" };
        println!("  {icon} {}", r.description);
        if !r.passed {
            if let Some(detail) = &r.detail {
                println!("    {detail}");
            }
        }
    }
}
