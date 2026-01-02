use burn::backend::candle::{Candle, CandleDevice};
use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use dpsn::config::FullConfig;
use dpsn::data::{download_tiny_shakespeare, load_dataset_from_config, CharDataset};
use dpsn::inference::TextGenerator;
use dpsn::model::DPSN;
use dpsn::training::{
    find_latest_checkpoint, load_checkpoint, train_hierarchical, train_with_curriculum,
    CurriculumConfig, TrainingConfig,
};

type NdArrayBackend = NdArray<f32>;
type NdArrayAutodiff = Autodiff<NdArrayBackend>;

type WgpuBackend = Wgpu;
type WgpuAutodiff = Autodiff<WgpuBackend>;

type CandleBackend = Candle;
type CandleAutodiff = Autodiff<CandleBackend>;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
pub enum BackendType {
    Ndarray,
    Wgpu,
    Candle,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Ndarray => write!(f, "ndarray"),
            BackendType::Wgpu => write!(f, "wgpu"),
            BackendType::Candle => write!(f, "candle"),
        }
    }
}

fn parse_backend(s: &str) -> BackendType {
    match s.to_lowercase().as_str() {
        "wgpu" => BackendType::Wgpu,
        "candle" => BackendType::Candle,
        _ => BackendType::Ndarray,
    }
}

fn print_device_info(backend: BackendType) {
    match backend {
        BackendType::Ndarray => {
            println!("Backend: NdArray (CPU)");
            println!("  Accelerator: CPU (pure Rust, no hardware acceleration)");
        }
        BackendType::Wgpu => {
            let device: WgpuDevice = Default::default();
            let accel = match &device {
                WgpuDevice::DiscreteGpu(idx) => format!("Discrete GPU #{}", idx),
                WgpuDevice::IntegratedGpu(idx) => format!("Integrated GPU #{}", idx),
                WgpuDevice::VirtualGpu(idx) => format!("Virtual GPU #{}", idx),
                WgpuDevice::Cpu => "CPU (software rendering)".to_string(),
                WgpuDevice::DefaultDevice => "Default GPU".to_string(),
                _ => "WebGPU Device".to_string(),
            };
            println!("Backend: WGPU (WebGPU)");
            println!("  Accelerator: {}", accel);
            println!("  API: Vulkan/Metal/DX12 (auto-detected)");
        }
        BackendType::Candle => {
            let device: CandleDevice = Default::default();
            let accel = match &device {
                CandleDevice::Cpu => "CPU".to_string(),
                CandleDevice::Cuda(cuda_dev) => format!("CUDA GPU #{}", cuda_dev.index),
                CandleDevice::Metal(metal_dev) => format!("Metal GPU #{}", metal_dev.index),
            };
            println!("Backend: Candle");
            println!("  Accelerator: {}", accel);
        }
    }
    println!();
}

#[derive(Parser)]
#[command(name = "dpsn")]
#[command(about = "Dynamic Parameter Selection Networks - Train and generate text")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(long, short = 'c')]
        config: Option<String>,

        #[arg(long, short, default_value = "ndarray", value_enum)]
        backend: BackendType,

        #[arg(long, default_value = "500")]
        steps: usize,

        #[arg(long, default_value = "32")]
        batch_size: usize,

        #[arg(long, default_value = "0.001")]
        lr: f64,

        #[arg(long, default_value = "20000")]
        pool_size: usize,

        #[arg(long, default_value = "64")]
        embed_dim: usize,

        #[arg(long, default_value = "100")]
        k_min: usize,

        #[arg(long, default_value = "5000")]
        k_max: usize,

        #[arg(long, default_value = "64")]
        context_length: usize,

        #[arg(long, default_value = "data")]
        data_dir: String,
    },
    Generate {
        #[arg(long, short = 'c')]
        config: Option<String>,

        #[arg(long, short, default_value = "ndarray", value_enum)]
        backend: BackendType,

        #[arg(long)]
        checkpoint: Option<String>,

        #[arg(long, default_value = "The ")]
        prompt: String,

        #[arg(long, default_value = "200")]
        max_tokens: usize,

        #[arg(long, default_value = "0.8")]
        temperature: f64,

        #[arg(long, default_value = "data")]
        data_dir: String,
    },
    Demo {
        #[arg(long, short, default_value = "ndarray", value_enum)]
        backend: BackendType,
    },
    Run {
        #[arg(long, short = 'c', required = true)]
        config: String,
    },
    InitConfig {
        #[arg(long, short, default_value = "config.yaml")]
        output: String,

        #[arg(long, default_value = "default")]
        template: String,
    },
    Backends,
}

fn print_backends() {
    println!("=== DPSN - Dynamic Parameter Selection Networks ===\n");
    println!("Supported backends:\n");
    println!("  ndarray  - CPU backend using ndarray (default)");
    println!("             Best for: Development, debugging, CPU-only systems");
    println!("             Speed: Moderate\n");
    println!("  wgpu     - GPU backend using WebGPU");
    println!("             Best for: Cross-platform GPU acceleration");
    println!("             Speed: Fast (Vulkan/Metal/DX12)\n");
    println!("  candle   - GPU backend using Candle");
    println!("             Best for: macOS Metal, Linux/Windows CUDA");
    println!("             Speed: Fast (native Metal/CUDA acceleration)\n");
    println!("Usage:");
    println!("  dpsn train --backend ndarray       # Train with CPU");
    println!("  dpsn train --backend wgpu          # Train with WebGPU");
    println!("  dpsn train --config config.yaml    # Train from config file");
    println!("  dpsn run --config config.yaml      # Run full training from config");
    println!("  dpsn init-config --output my.yaml  # Generate example config\n");
    println!("Commands:");
    println!("  train       - Train a new DPSN model");
    println!("  generate    - Generate text (trains a quick model first)");
    println!("  demo        - Run a quick demonstration");
    println!("  run         - Run training from YAML config file");
    println!("  init-config - Generate example config file");
    println!("  backends    - Show this help message\n");
    println!("Run 'dpsn <command> --help' for more options.");
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        None => {
            print_backends();
        }
        Some(Commands::Backends) => {
            print_backends();
        }
        Some(Commands::InitConfig { output, template }) => {
            init_config(&output, &template);
        }
        Some(Commands::Run { config }) => {
            run_from_config(&config);
        }
        Some(Commands::Train {
            config,
            backend,
            steps,
            batch_size,
            lr,
            pool_size,
            embed_dim,
            k_min,
            k_max,
            context_length,
            data_dir,
        }) => {
            if let Some(config_path) = config {
                run_from_config(&config_path);
            } else {
                print_device_info(backend);
                match backend {
                    BackendType::Ndarray => run_training::<NdArrayAutodiff>(
                        steps,
                        batch_size,
                        lr,
                        pool_size,
                        embed_dim,
                        k_min,
                        k_max,
                        context_length,
                        &data_dir,
                    ),
                    BackendType::Wgpu => run_training::<WgpuAutodiff>(
                        steps,
                        batch_size,
                        lr,
                        pool_size,
                        embed_dim,
                        k_min,
                        k_max,
                        context_length,
                        &data_dir,
                    ),
                    BackendType::Candle => run_training::<CandleAutodiff>(
                        steps,
                        batch_size,
                        lr,
                        pool_size,
                        embed_dim,
                        k_min,
                        k_max,
                        context_length,
                        &data_dir,
                    ),
                }
            }
        }
        Some(Commands::Generate {
            config,
            backend,
            checkpoint,
            prompt,
            max_tokens,
            temperature,
            data_dir,
        }) => {
            if let Some(config_path) = config {
                run_generate_from_config(&config_path, &prompt, checkpoint.as_deref());
            } else {
                print_device_info(backend);
                match backend {
                    BackendType::Ndarray => run_generation::<NdArrayAutodiff>(
                        &prompt,
                        max_tokens,
                        temperature,
                        &data_dir,
                        checkpoint.as_deref(),
                    ),
                    BackendType::Wgpu => run_generation::<WgpuAutodiff>(
                        &prompt,
                        max_tokens,
                        temperature,
                        &data_dir,
                        checkpoint.as_deref(),
                    ),
                    BackendType::Candle => run_generation::<CandleAutodiff>(
                        &prompt,
                        max_tokens,
                        temperature,
                        &data_dir,
                        checkpoint.as_deref(),
                    ),
                }
            }
        }
        Some(Commands::Demo { backend }) => {
            println!("Using backend: {}\n", backend);
            match backend {
                BackendType::Ndarray => run_demo::<NdArrayAutodiff>(),
                BackendType::Wgpu => run_demo::<WgpuAutodiff>(),
                BackendType::Candle => run_demo::<CandleAutodiff>(),
            }
        }
    }
}

fn init_config(output: &str, template: &str) {
    let config = match template {
        "demo" => FullConfig::demo_config(),
        "huggingface" => FullConfig::huggingface_example(),
        _ => FullConfig::default_config(),
    };

    match config.save_to_yaml(output) {
        Ok(_) => {
            println!("Config file created: {}", output);
            println!("\nYou can edit this file and run:");
            println!("  dpsn run --config {}", output);
        }
        Err(e) => {
            eprintln!("Failed to create config file: {}", e);
        }
    }
}

fn run_from_config(config_path: &str) {
    println!("Loading config from: {}\n", config_path);

    let config = match FullConfig::load_from_yaml(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load config: {}", e);
            return;
        }
    };

    let backend = parse_backend(&config.backend.backend_type);
    print_device_info(backend);

    match backend {
        BackendType::Ndarray => run_training_from_config::<NdArrayAutodiff>(&config),
        BackendType::Wgpu => run_training_from_config::<WgpuAutodiff>(&config),
        BackendType::Candle => run_training_from_config::<CandleAutodiff>(&config),
    }
}

fn run_generate_from_config(config_path: &str, prompt: &str, checkpoint: Option<&str>) {
    println!("Loading config from: {}\n", config_path);

    let config = match FullConfig::load_from_yaml(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load config: {}", e);
            return;
        }
    };

    let backend = parse_backend(&config.backend.backend_type);
    print_device_info(backend);

    match backend {
        BackendType::Ndarray => {
            run_generate_with_config::<NdArrayAutodiff>(&config, prompt, checkpoint)
        }
        BackendType::Wgpu => run_generate_with_config::<WgpuAutodiff>(&config, prompt, checkpoint),
        BackendType::Candle => {
            run_generate_with_config::<CandleAutodiff>(&config, prompt, checkpoint)
        }
    }
}

fn run_training_from_config<B: burn::tensor::backend::AutodiffBackend>(config: &FullConfig) {
    println!("=== DPSN Training (from config) ===\n");

    let text = match load_dataset_from_config(&config.dataset) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return;
        }
    };

    let dataset = CharDataset::new(&text, config.model.context_length);

    println!("Dataset loaded:");
    println!("  - Total tokens: {}", dataset.tokens.len());
    println!("  - Vocabulary size: {}", dataset.vocab_size());
    println!("  - Training samples: {}\n", dataset.len());

    let checkpoint_dir = config
        .training
        .checkpoint_dir
        .as_ref()
        .map(|s| PathBuf::from(s));

    let training_config = TrainingConfig::new()
        .with_num_steps(config.training.num_steps)
        .with_batch_size(config.training.batch_size)
        .with_learning_rate(config.training.learning_rate)
        .with_log_interval(config.training.log_interval)
        .with_save_interval(config.training.save_interval)
        .with_checkpoint_dir(checkpoint_dir);

    let curriculum_config = CurriculumConfig::new()
        .with_warmup_steps(config.curriculum.warmup_steps)
        .with_specialization_steps(config.curriculum.specialization_steps)
        .with_warmup_epsilon(config.curriculum.warmup_epsilon)
        .with_specialization_epsilon_start(config.curriculum.specialization_epsilon_start)
        .with_specialization_epsilon_end(config.curriculum.specialization_epsilon_end)
        .with_maturity_epsilon(config.curriculum.maturity_epsilon)
        .with_specialization_balance_weight(config.curriculum.balance_weight)
        .with_maturity_balance_weight(config.curriculum.balance_weight)
        .with_maturity_efficiency_weight(config.curriculum.efficiency_weight)
        .with_z_loss_weight(config.curriculum.z_loss_weight);

    let device = Default::default();

    if config.model.use_hierarchical_router {
        println!("Using HIERARCHICAL router (reduced parameter count)\n");
        let model = train_hierarchical::<B>(
            training_config,
            curriculum_config,
            dataset.vocab_size(),
            config.model.embed_dim,
            config.model.pool_size,
            config.model.k_min,
            config.model.k_max,
            config.model.num_clusters,
            config.model.top_clusters,
            config.model.context_length,
            config.model.exploration_noise,
            &dataset,
            &device,
        );

        println!("\n=== Generation Sample ===\n");
        let generator =
            dpsn::inference::HierarchicalTextGenerator::new(&model, &dataset.tokenizer, device);
        let generated = generator.generate(
            &config.inference.default_prompt,
            config.inference.max_tokens,
            config.inference.temperature,
        );
        println!("{}", generated);
    } else {
        let model = train_with_curriculum::<B>(
            training_config,
            curriculum_config,
            dataset.vocab_size(),
            config.model.embed_dim,
            config.model.pool_size,
            config.model.k_min,
            config.model.k_max,
            config.model.router_hidden_dim,
            config.model.context_length,
            config.model.exploration_noise,
            &dataset,
            &device,
        );

        println!("\n=== Generation Sample ===\n");
        let generator = TextGenerator::new(&model, &dataset.tokenizer, device);
        let generated = generator.generate(
            &config.inference.default_prompt,
            config.inference.max_tokens,
            config.inference.temperature,
        );
        println!("{}", generated);
    }
}

fn run_generate_with_config<B: burn::tensor::backend::AutodiffBackend>(
    config: &FullConfig,
    prompt: &str,
    checkpoint: Option<&str>,
) {
    println!("=== DPSN Text Generation (from config) ===\n");

    let text = match load_dataset_from_config(&config.dataset) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return;
        }
    };

    let dataset = CharDataset::new(&text, config.model.context_length);
    let device: <B::InnerBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let checkpoint_path = checkpoint.map(PathBuf::from).or_else(|| {
        config
            .training
            .checkpoint_dir
            .as_ref()
            .and_then(|dir| find_latest_checkpoint(&PathBuf::from(dir)))
    });

    let model: DPSN<B::InnerBackend> = if let Some(ref ckpt_path) = checkpoint_path {
        println!("Loading model from checkpoint: {:?}\n", ckpt_path);
        let fresh_model: DPSN<B::InnerBackend> = DPSN::new(
            dataset.vocab_size(),
            config.model.embed_dim,
            config.model.pool_size,
            config.model.k_min,
            config.model.k_max,
            config.model.router_hidden_dim,
            config.model.context_length,
            config.model.exploration_noise,
            &device,
        );
        match load_checkpoint(fresh_model, ckpt_path, &device) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load checkpoint: {}. Training new model...", e);
                train_fresh_model::<B>(config, &dataset, &device)
            }
        }
    } else {
        println!("No checkpoint found. Training new model...\n");
        train_fresh_model::<B>(config, &dataset, &device)
    };

    let generator = TextGenerator::new(&model, &dataset.tokenizer, device);
    let generated = generator.generate(
        prompt,
        config.inference.max_tokens,
        config.inference.temperature,
    );

    println!("\n=== Generated Text ===\n");
    println!("{}", generated);
}

fn train_fresh_model<B: burn::tensor::backend::AutodiffBackend>(
    config: &FullConfig,
    dataset: &CharDataset,
    device: &<B::InnerBackend as burn::tensor::backend::Backend>::Device,
) -> DPSN<B::InnerBackend> {
    let training_config = TrainingConfig::new()
        .with_num_steps(config.training.num_steps)
        .with_batch_size(config.training.batch_size)
        .with_learning_rate(config.training.learning_rate);

    let curriculum_config = CurriculumConfig::new()
        .with_warmup_steps(config.curriculum.warmup_steps)
        .with_specialization_steps(config.curriculum.specialization_steps);

    train_with_curriculum::<B>(
        training_config,
        curriculum_config,
        dataset.vocab_size(),
        config.model.embed_dim,
        config.model.pool_size,
        config.model.k_min,
        config.model.k_max,
        config.model.router_hidden_dim,
        config.model.context_length,
        config.model.exploration_noise,
        dataset,
        device,
    )
}

fn run_training<B: burn::tensor::backend::AutodiffBackend>(
    steps: usize,
    batch_size: usize,
    lr: f64,
    pool_size: usize,
    embed_dim: usize,
    k_min: usize,
    k_max: usize,
    context_length: usize,
    data_dir: &str,
) {
    println!("=== DPSN Training ===\n");

    let text = download_tiny_shakespeare(data_dir).expect("Failed to download dataset");
    let dataset = CharDataset::new(&text, context_length);

    println!("Dataset loaded:");
    println!("  - Total tokens: {}", dataset.tokens.len());
    println!("  - Vocabulary size: {}", dataset.vocab_size());
    println!("  - Training samples: {}\n", dataset.len());

    let training_config = TrainingConfig::new()
        .with_num_steps(steps)
        .with_batch_size(batch_size)
        .with_learning_rate(lr);

    let curriculum_config = CurriculumConfig::new()
        .with_warmup_steps(steps / 5)
        .with_specialization_steps(steps * 2 / 5);

    let device = Default::default();

    let model = train_with_curriculum::<B>(
        training_config,
        curriculum_config,
        dataset.vocab_size(),
        embed_dim,
        pool_size,
        k_min,
        k_max,
        128,
        context_length,
        0.1,
        &dataset,
        &device,
    );

    println!("\n=== Generation Sample ===\n");

    let generator = TextGenerator::new(&model, &dataset.tokenizer, device);
    let generated = generator.generate("ROMEO:", 200, 0.8);
    println!("{}", generated);
}

fn run_generation<B: burn::tensor::backend::AutodiffBackend>(
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    data_dir: &str,
    checkpoint: Option<&str>,
) {
    println!("=== DPSN Text Generation ===\n");

    let text = download_tiny_shakespeare(data_dir).expect("Failed to download dataset");
    let dataset = CharDataset::new(&text, 64);
    let device: <B::InnerBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let model: DPSN<B::InnerBackend> = if let Some(ckpt_path) = checkpoint {
        println!("Loading model from checkpoint: {}\n", ckpt_path);
        let fresh_model: DPSN<B::InnerBackend> = DPSN::new(
            dataset.vocab_size(),
            64,
            5000,
            50,
            500,
            64,
            64,
            0.1,
            &device,
        );
        match load_checkpoint(fresh_model, &PathBuf::from(ckpt_path), &device) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load checkpoint: {}. Training new model...", e);
                let training_config = TrainingConfig::new()
                    .with_num_steps(100)
                    .with_batch_size(16);
                let curriculum_config = CurriculumConfig::new()
                    .with_warmup_steps(20)
                    .with_specialization_steps(60);
                train_with_curriculum::<B>(
                    training_config,
                    curriculum_config,
                    dataset.vocab_size(),
                    64,
                    5000,
                    50,
                    500,
                    64,
                    64,
                    0.1,
                    &dataset,
                    &device,
                )
            }
        }
    } else {
        println!("No checkpoint provided. Running quick training first...\n");
        let training_config = TrainingConfig::new()
            .with_num_steps(100)
            .with_batch_size(16);
        let curriculum_config = CurriculumConfig::new()
            .with_warmup_steps(20)
            .with_specialization_steps(60);
        train_with_curriculum::<B>(
            training_config,
            curriculum_config,
            dataset.vocab_size(),
            64,
            5000,
            50,
            500,
            64,
            64,
            0.1,
            &dataset,
            &device,
        )
    };

    let generator = TextGenerator::new(&model, &dataset.tokenizer, device);
    let generated = generator.generate(prompt, max_tokens, temperature);

    println!("\n=== Generated Text ===\n");
    println!("{}", generated);
}

fn run_demo<B: burn::tensor::backend::AutodiffBackend>() {
    println!("=== DPSN Demo Mode ===\n");
    println!("Running a quick demonstration with smaller parameters...\n");

    let text = match download_tiny_shakespeare("data") {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to download dataset: {}", e);
            return;
        }
    };

    let dataset = CharDataset::new(&text, 32);

    println!("Dataset loaded:");
    println!("  - Total tokens: {}", dataset.tokens.len());
    println!("  - Vocabulary size: {}", dataset.vocab_size());
    println!("  - Training samples: {}\n", dataset.len());

    let training_config = TrainingConfig::new()
        .with_num_steps(100)
        .with_batch_size(16)
        .with_log_interval(20);

    let curriculum_config = CurriculumConfig::new()
        .with_warmup_steps(20)
        .with_specialization_steps(60);

    let device = Default::default();

    let model = train_with_curriculum::<B>(
        training_config,
        curriculum_config,
        dataset.vocab_size(),
        32,
        2000,
        20,
        200,
        64,
        32,
        0.1,
        &dataset,
        &device,
    );

    println!("\n=== Quick Generation Test ===\n");

    let generator = TextGenerator::new(&model, &dataset.tokenizer, device);

    let prompts = ["The ", "ROMEO:", "What ", "To be"];

    for prompt in prompts {
        println!("Prompt: \"{}\"", prompt);
        let generated = generator.generate(prompt, 50, 0.9);
        println!("Output: {}\n", generated);
    }

    println!("Demo completed!");
}
