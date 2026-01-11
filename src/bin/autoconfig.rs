use clap::{Parser, ValueEnum};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, ValueEnum)]
enum RouterType {
    Standard,
    Hierarchical,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    target_params: String,

    #[arg(short, long, default_value = "dpsn_autoconfig.yaml")]
    output: PathBuf,

    #[arg(short, long)]
    embed_dim: Option<usize>,

    #[arg(short, long, default_value_t = 1)]
    recurrence: usize,

    #[arg(long)]
    tokenizer_path: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = RouterType::Standard)]
    router_type: RouterType,

    #[arg(long, default_value_t = 64)]
    num_clusters: usize,
}

struct ModelParams {
    pool_size: usize,
    embed_dim: usize,
    router_hidden_dim: usize, // Only for Standard
    num_clusters: usize,      // Only for Hierarchical
    vocab_size: usize,
    num_heads: usize,
    k_min: usize,
    k_max: usize,
    router_type: RouterType,
}

impl ModelParams {
    fn calculate_total_params(&self) -> usize {
        let pool_params = self.pool_size * self.embed_dim;
        let embed_params = self.vocab_size * self.embed_dim;
        let output_params = self.embed_dim * self.vocab_size;
        let engine_params = self.embed_dim * self.embed_dim;
        let attn_params = self.embed_dim * self.embed_dim * 4;

        let router_params = match self.router_type {
            RouterType::Standard => {
                self.embed_dim * self.router_hidden_dim * 2
                    + self.router_hidden_dim * self.pool_size
            }
            RouterType::Hierarchical => {
                // complexity (embed * 1) + cluster_scorer (embed * num_clusters) + intra (embed * cluster_size)
                let complexity = self.embed_dim;
                let cluster_scorer = self.embed_dim * self.num_clusters;
                let cluster_size = (self.pool_size + self.num_clusters - 1) / self.num_clusters;
                let intra_scorer = self.embed_dim * cluster_size;
                complexity + cluster_scorer + intra_scorer
            }
        };

        pool_params + router_params + embed_params + output_params + engine_params + attn_params
    }
}

fn parse_target_params(input: &str) -> Result<usize, String> {
    let input = input.trim().to_uppercase();
    let multiplier = if input.ends_with('M') {
        1_000_000.0
    } else if input.ends_with('B') {
        1_000_000_000.0
    } else if input.ends_with('K') {
        1_000.0
    } else {
        1.0
    };

    let numeric_part = input
        .trim_end_matches(|c: char| !c.is_numeric() && c != '.')
        .parse::<f64>()
        .map_err(|_| format!("Invalid number format: {}", input))?;

    Ok((numeric_part * multiplier) as usize)
}

fn optimize_config(
    target: usize,
    fixed_embed_dim: Option<usize>,
    vocab_size: usize,
    router_type: RouterType,
    num_clusters: usize,
) -> ModelParams {
    let tiers = [
        (512, 128, 8),
        (768, 256, 12),
        (1024, 256, 16),
        (1600, 512, 25),
        (2048, 512, 32),
        (2560, 1024, 40),
    ];

    // Filter tiers if embed_dim is fixed
    let candidates: Vec<_> = if let Some(dim) = fixed_embed_dim {
        tiers
            .iter()
            .filter(|(d, _, _)| *d == dim)
            .cloned()
            .collect()
    } else {
        tiers.to_vec()
    };

    if candidates.is_empty() && fixed_embed_dim.is_some() {
        // Fallback for custom dimension
        let dim = fixed_embed_dim.unwrap();
        let hidden = if dim >= 1024 { 512 } else { 256 };
        let heads = (dim / 64).max(1);
        let candidates = vec![(dim, hidden, heads)];
        return solve_for_pool(target, candidates, vocab_size, router_type, num_clusters);
    }

    solve_for_pool(target, candidates, vocab_size, router_type, num_clusters)
}

fn solve_for_pool(
    target: usize,
    candidates: Vec<(usize, usize, usize)>,
    vocab_size: usize,
    router_type: RouterType,
    num_clusters: usize,
) -> ModelParams {
    let mut best_params = ModelParams {
        pool_size: 0,
        embed_dim: 0,
        router_hidden_dim: 0,
        num_clusters: 0,
        vocab_size,
        num_heads: 0,
        k_min: 0,
        k_max: 0,
        router_type: RouterType::Standard,
    };
    let mut best_diff = usize::MAX;

    for (embed_dim, router_hidden, num_heads) in candidates {
        // Calculate constant overhead params (embeddings, attention, engine)
        let embed_params = vocab_size * embed_dim;
        let output_params = embed_dim * vocab_size;
        let engine_params = embed_dim * embed_dim;
        let attn_params = embed_dim * embed_dim * 4;

        // Calculate Pool Size based on Router Type logic
        let (pool_size, router_fixed_cost) = match router_type {
            RouterType::Standard => {
                let router_fixed = embed_dim * router_hidden * 2;
                let constant_overhead =
                    embed_params + output_params + engine_params + attn_params + router_fixed;

                let pool_size = if constant_overhead >= target {
                    1000
                } else {
                    let remaining_budget = target - constant_overhead;
                    let params_per_pool_entry = embed_dim + router_hidden;
                    remaining_budget / params_per_pool_entry
                };
                (pool_size, router_fixed)
            }
            RouterType::Hierarchical => {
                // Router Fixed = Complexity(Embed) + ClusterScorer(Embed*NumClusters)
                let router_fixed = embed_dim + (embed_dim * num_clusters);
                let constant_overhead =
                    embed_params + output_params + engine_params + attn_params + router_fixed;

                let pool_size = if constant_overhead >= target {
                    1000
                } else {
                    let remaining_budget = target - constant_overhead;
                    // Variable Cost = PoolParams(Pool*Embed) + IntraScorer(Embed * Pool/NumClusters)
                    // Cost = Pool * (Embed + Embed/NumClusters)
                    let params_per_pool_entry =
                        embed_dim as f64 + (embed_dim as f64 / num_clusters as f64);
                    (remaining_budget as f64 / params_per_pool_entry) as usize
                };
                (pool_size, router_fixed)
            }
        };

        if pool_size < 100 {
            continue;
        } // Too small to be useful

        let k_max = (pool_size as f64 * 0.05).max(10.0) as usize; // 5% activation heuristic
        let k_min = (k_max as f64 * 0.1).max(1.0) as usize;

        let params = ModelParams {
            pool_size,
            embed_dim,
            router_hidden_dim: router_hidden,
            num_clusters,
            vocab_size,
            num_heads,
            k_min,
            k_max,
            router_type: router_type.clone(),
        };

        let total = params.calculate_total_params();
        let diff = if total > target {
            total - target
        } else {
            target - total
        };

        if diff < best_diff {
            best_diff = diff;
            best_params = params;
        }
    }

    best_params
}

fn main() {
    let args = Args::parse();

    let target = match parse_target_params(&args.target_params) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "Target Parameters: {} (parsed as {})",
        args.target_params, target
    );

    let vocab_size = if let Some(path) = args.tokenizer_path {
        println!("Loading tokenizer from: {:?}", path);
        let tokenizer = Tokenizer::from_file(&path).unwrap_or_else(|e| {
            eprintln!("Error loading tokenizer: {}", e);
            std::process::exit(1);
        });
        let size = tokenizer.get_vocab_size(true);
        println!("Detected Vocab Size: {}", size);
        size
    } else {
        println!("Using default Vocab Size: 50257 (GPT-2)");
        50257
    };

    println!("Optimizing configuration...");

    let params = optimize_config(
        target,
        args.embed_dim,
        vocab_size,
        args.router_type.clone(),
        args.num_clusters,
    );

    if params.pool_size == 0 {
        eprintln!(
            "Error: Could not find a valid configuration for target {} parameters.",
            target
        );
        eprintln!("Try increasing the target size or adjusting embedding dimensions.");
        std::process::exit(1);
    }

    let actual_total = params.calculate_total_params();

    println!("\nOptimal Configuration Found:");
    println!("----------------------------");
    println!("Router Type:     {:?}", params.router_type);
    println!("Pool Size:       {}", params.pool_size);
    println!("Embedding Dim:   {}", params.embed_dim);
    match params.router_type {
        RouterType::Standard => println!("Router Hidden:   {}", params.router_hidden_dim),
        RouterType::Hierarchical => println!("Num Clusters:    {}", params.num_clusters),
    }
    println!("Recurrence:      {}", args.recurrence);
    println!(
        "Active Params:   ~{:.2} M (estimated)",
        (params.k_max * params.embed_dim) as f64 / 1_000_000.0
    );
    println!("----------------------------");
    println!(
        "Total Parameters: {:.2} M",
        actual_total as f64 / 1_000_000.0
    );
    println!(
        "Error Margin:     {:.2}%",
        (actual_total as f64 - target as f64).abs() / target as f64 * 100.0
    );

    // Generate YAML content
    let router_config = match params.router_type {
        RouterType::Standard => format!(
            r#"    type: standard
    k_min: {}
    k_max: {}
    hidden_dim: {}
    exploration_noise: 0.1"#,
            params.k_min, params.k_max, params.router_hidden_dim
        ),
        RouterType::Hierarchical => format!(
            r#"    type: hierarchical
    k_min: {}
    k_max: {}
    num_clusters: {}
    top_clusters: 4
    exploration_noise: 0.1"#,
            params.k_min, params.k_max, params.num_clusters
        ),
    };

    let yaml_content = format!(
        r#"model:
  vocab_size: {}
  embed_dim: {}
  pool_size: {}
  num_heads: {}
  context_length: 1024
  recurrence_steps: {}
  router:
{}

training:
  num_steps: 10000
  num_epochs: null
  batch_size: 8
  learning_rate: 0.0003
  log_interval: 50
  save_interval: 1000
  checkpoint_dir: checkpoints_auto

dataset:
  source: local_file
  data_dir: data
  local_file: "data/tiny_shakespeare.txt"
  huggingface: null
  parquet: null
  tokenizer_path: null
  max_items: null

inference:
  max_tokens: 500
  temperature: 0.8
  default_prompt: "The "

backend:
  backend_type: wgpu

curriculum:
  warmup_steps: 1000
  specialization_steps: 5000
  warmup_epsilon: 1.0
  specialization_epsilon_start: 0.3
  specialization_epsilon_end: 0.05
  maturity_epsilon: 0.01
  balance_weight: 0.1
  efficiency_weight: 0.05
  z_loss_weight: 0.001
  warmup_balance_weight: 0.0
  specialization_balance_weight: 0.1
  maturity_balance_weight: 0.05
  warmup_efficiency_weight: 0.0
  specialization_efficiency_weight: 0.0
  maturity_efficiency_weight: 0.1

device_placement:
  pool: "Cpu"
  router: "Gpu"
  embedding: "Gpu"
  engine: "Gpu"
  head: "Gpu"
"#,
        params.vocab_size,
        params.embed_dim,
        params.pool_size,
        params.num_heads,
        args.recurrence,
        router_config
    );

    let mut file = File::create(&args.output).expect("Unable to create output file");
    file.write_all(yaml_content.as_bytes())
        .expect("Unable to write data");

    println!("\nConfiguration saved to: {:?}", args.output);
}
