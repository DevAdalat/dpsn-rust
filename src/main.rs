use burn::backend::candle::{Candle, CandleDevice};
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use dpsn::config::{FullConfig, RouterSettings};
use dpsn::data::{download_tiny_shakespeare, load_dataset_from_config, CharDataset};
use dpsn::inference::TextGenerator;
use dpsn::model::config::{HierarchicalRouterConfig, StandardRouterConfig};
use dpsn::model::{DeviceLocation, DPSN};
use dpsn::training::{
    load_checkpoint, train_hierarchical, train_with_curriculum, CurriculumConfig, TrainingConfig,
};

type NdArrayBackend = NdArray<f32>;
type NdArrayAutodiff = Autodiff<NdArrayBackend>;

type WgpuBackend = Wgpu;
type WgpuAutodiff = Autodiff<WgpuBackend>;

type CandleBackend = Candle;
type CandleAutodiff = Autodiff<CandleBackend>;

type CudaBackend = Cuda;
type CudaAutodiff = Autodiff<CudaBackend>;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
pub enum BackendType {
    Ndarray,
    Wgpu,
    Candle,
    Cuda,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Ndarray => write!(f, "ndarray"),
            BackendType::Wgpu => write!(f, "wgpu"),
            BackendType::Candle => write!(f, "candle"),
            BackendType::Cuda => write!(f, "cuda"),
        }
    }
}

fn parse_backend(s: &str) -> BackendType {
    match s.to_lowercase().as_str() {
        "wgpu" => BackendType::Wgpu,
        "candle" => BackendType::Candle,
        "cuda" => BackendType::Cuda,
        _ => BackendType::Ndarray,
    }
}

fn backend_to_device_location(backend: BackendType) -> DeviceLocation {
    match backend {
        BackendType::Ndarray => DeviceLocation::AllCpu,
        BackendType::Wgpu | BackendType::Candle | BackendType::Cuda => DeviceLocation::AllGpu,
    }
}

fn print_device_info(backend: BackendType) {
    match backend {
        BackendType::Ndarray => {
            println!(
                "╔══════════════════════════════════════════════════════════════════════════════╗"
            );
            println!(
                "║                              BACKEND: NdArray (CPU)                          ║"
            );
            println!(
                "╠══════════════════════════════════════════════════════════════════════════════╣"
            );
            println!(
                "║  Accelerator:  CPU (pure Rust, no hardware acceleration)                     ║"
            );
            println!(
                "║  Memory:       System RAM                                                    ║"
            );
            println!(
                "╚══════════════════════════════════════════════════════════════════════════════╝"
            );
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
            println!(
                "╔══════════════════════════════════════════════════════════════════════════════╗"
            );
            println!(
                "║                              BACKEND: WGPU (WebGPU)                          ║"
            );
            println!(
                "╠══════════════════════════════════════════════════════════════════════════════╣"
            );
            println!("║  Accelerator:  {:<60} ║", accel);
            println!(
                "║  API:          Vulkan/Metal/DX12 (auto-detected)                             ║"
            );
            print_gpu_memory_info();
            println!(
                "╚══════════════════════════════════════════════════════════════════════════════╝"
            );
        }
        BackendType::Candle => {
            let device: CandleDevice = Default::default();
            let accel = match &device {
                CandleDevice::Cpu => "CPU".to_string(),
                CandleDevice::Cuda(cuda_dev) => format!("CUDA GPU #{}", cuda_dev.index),
                CandleDevice::Metal(metal_dev) => format!("Metal GPU #{}", metal_dev.index),
            };
            println!(
                "╔══════════════════════════════════════════════════════════════════════════════╗"
            );
            println!(
                "║                              BACKEND: Candle                                 ║"
            );
            println!(
                "╠══════════════════════════════════════════════════════════════════════════════╣"
            );
            println!("║  Accelerator:  {:<60} ║", accel);
            print_gpu_memory_info();
            println!(
                "╚══════════════════════════════════════════════════════════════════════════════╝"
            );
        }
        BackendType::Cuda => {
            let device: CudaDevice = Default::default();
            println!(
                "╔══════════════════════════════════════════════════════════════════════════════╗"
            );
            println!(
                "║                              BACKEND: CUDA (CubeCL)                          ║"
            );
            println!(
                "╠══════════════════════════════════════════════════════════════════════════════╣"
            );
            println!(
                "║  Accelerator:  NVIDIA GPU #{}                                                 ║",
                device.index
            );
            println!(
                "║  API:          CUDA via CubeCL JIT                                           ║"
            );
            print_cuda_memory_info(device.index as i32);
            println!(
                "╚══════════════════════════════════════════════════════════════════════════════╝"
            );
        }
    }
    println!();
}

fn print_gpu_memory_info() {
    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                if let Ok(s) = String::from_utf8(output.stdout) {
                    let parts: Vec<&str> = s.trim().split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 3 {
                        let total = parts[0].parse::<u64>().unwrap_or(0);
                        let free = parts[1].parse::<u64>().unwrap_or(0);
                        let used = parts[2].parse::<u64>().unwrap_or(0);
                        println!("╠────────────────────────────────────────────────────────────────────────────╣");
                        println!("║  GPU Memory:   Total: {:>6} MB | Used: {:>6} MB | Free: {:>6} MB       ║", total, used, free);
                        return;
                    }
                }
            }
        }
    }
    println!("╠────────────────────────────────────────────────────────────────────────────╣");
    println!("║  GPU Memory:   (Unable to query - nvidia-smi not available)               ║");
}

fn print_cuda_memory_info(device_index: i32) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                &format!("--id={}", device_index),
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                if let Ok(s) = String::from_utf8(output.stdout) {
                    let parts: Vec<&str> = s.trim().split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 4 {
                        let name = parts[0];
                        let total = parts[1].parse::<u64>().unwrap_or(0);
                        let free = parts[2].parse::<u64>().unwrap_or(0);
                        let used = parts[3].parse::<u64>().unwrap_or(0);
                        println!("║  GPU Name:     {:<60} ║", name);
                        println!("╠────────────────────────────────────────────────────────────────────────────╣");
                        println!("║  VRAM Total:   {:>6} MB                                                  ║", total);
                        println!("║  VRAM Used:    {:>6} MB                                                  ║", used);
                        println!("║  VRAM Free:    {:>6} MB                                                  ║", free);
                        return;
                    }
                }
            }
        }
    }
    println!("╠────────────────────────────────────────────────────────────────────────────╣");
    println!("║  GPU Memory:   (Unable to query CUDA device memory)                       ║");
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

        #[arg(long)]
        steps: Option<usize>,

        #[arg(long)]
        epochs: Option<usize>,

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

        #[arg(long, default_value = "4")]
        num_heads: usize,

        #[arg(long, default_value = "128")]
        router_hidden_dim: usize,

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
    println!("  cuda     - GPU backend using CubeCL (NVIDIA CUDA)");
    println!("             Best for: NVIDIA GPUs with native CUDA support");
    println!("             Speed: Very Fast (CubeCL JIT compilation)\n");
    println!("Usage:");
    println!("  dpsn train --backend ndarray       # Train with CPU");
    println!("  dpsn train --backend wgpu          # Train with WebGPU");
    println!("  dpsn train --backend cuda          # Train with CUDA (CubeCL)");
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
            epochs,
            batch_size,
            lr,
            pool_size,
            embed_dim,
            k_min,
            k_max,
            num_heads,
            router_hidden_dim,
            context_length,
            data_dir,
        }) => {
            if let Some(config_path) = config {
                run_from_config(&config_path);
            } else {
                print_device_info(backend);
                let device_location = backend_to_device_location(backend);
                match backend {
                    BackendType::Ndarray => run_training::<NdArrayAutodiff>(
                        steps,
                        epochs,
                        batch_size,
                        lr,
                        pool_size,
                        embed_dim,
                        k_min,
                        k_max,
                        router_hidden_dim,
                        num_heads,
                        context_length,
                        &data_dir,
                        device_location,
                    ),
                    BackendType::Wgpu => run_training::<WgpuAutodiff>(
                        steps,
                        epochs,
                        batch_size,
                        lr,
                        pool_size,
                        embed_dim,
                        k_min,
                        k_max,
                        router_hidden_dim,
                        num_heads,
                        context_length,
                        &data_dir,
                        device_location,
                    ),
                    BackendType::Candle => run_training::<CandleAutodiff>(
                        steps,
                        epochs,
                        batch_size,
                        lr,
                        pool_size,
                        embed_dim,
                        k_min,
                        k_max,
                        router_hidden_dim,
                        num_heads,
                        context_length,
                        &data_dir,
                        device_location,
                    ),
                    BackendType::Cuda => run_training::<CudaAutodiff>(
                        steps,
                        epochs,
                        batch_size,
                        lr,
                        pool_size,
                        embed_dim,
                        k_min,
                        k_max,
                        router_hidden_dim,
                        num_heads,
                        context_length,
                        &data_dir,
                        device_location,
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
                let device_location = backend_to_device_location(backend);
                match backend {
                    BackendType::Ndarray => run_generation::<NdArrayAutodiff>(
                        &prompt,
                        max_tokens,
                        temperature,
                        &data_dir,
                        checkpoint.as_deref(),
                        device_location,
                    ),
                    BackendType::Wgpu => run_generation::<WgpuAutodiff>(
                        &prompt,
                        max_tokens,
                        temperature,
                        &data_dir,
                        checkpoint.as_deref(),
                        device_location,
                    ),
                    BackendType::Candle => run_generation::<CandleAutodiff>(
                        &prompt,
                        max_tokens,
                        temperature,
                        &data_dir,
                        checkpoint.as_deref(),
                        device_location,
                    ),
                    BackendType::Cuda => run_generation::<CudaAutodiff>(
                        &prompt,
                        max_tokens,
                        temperature,
                        &data_dir,
                        checkpoint.as_deref(),
                        device_location,
                    ),
                }
            }
        }
        Some(Commands::Demo { backend }) => {
            println!("Using backend: {}\n", backend);
            let device_location = backend_to_device_location(backend);
            match backend {
                BackendType::Ndarray => run_demo::<NdArrayAutodiff>(device_location),
                BackendType::Wgpu => run_demo::<WgpuAutodiff>(device_location),
                BackendType::Candle => run_demo::<CandleAutodiff>(device_location),
                BackendType::Cuda => run_demo::<CudaAutodiff>(device_location),
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

fn resolve_device_location(config: &FullConfig, backend: BackendType) -> DeviceLocation {
    if let Some(placement) = &config.device_placement {
        if placement.is_offloaded() {
            println!("Configuration: Using OFFLOADED architecture (Pool on CPU, Compute on GPU)");
            return DeviceLocation::Offloaded;
        } else if placement.is_all_cpu() {
            println!("Configuration: Using ALL-CPU architecture");
            return DeviceLocation::AllCpu;
        } else if placement.is_all_gpu() {
            println!("Configuration: Using ALL-GPU architecture");
            return DeviceLocation::AllGpu;
        } else {
            println!("Configuration: Mixed placement detected. Defaulting to OFFLOADED for safety (Pool on CPU).");
            if backend == BackendType::Ndarray {
                return DeviceLocation::AllCpu;
            }
            return DeviceLocation::Offloaded;
        }
    }
    backend_to_device_location(backend)
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

    let device_location = resolve_device_location(&config, backend);

    match backend {
        BackendType::Ndarray => {
            run_training_from_config::<NdArrayAutodiff>(&config, device_location)
        }
        BackendType::Wgpu => run_training_from_config::<WgpuAutodiff>(&config, device_location),
        BackendType::Candle => run_training_from_config::<CandleAutodiff>(&config, device_location),
        BackendType::Cuda => run_training_from_config::<CudaAutodiff>(&config, device_location),
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

    let device_location = resolve_device_location(&config, backend);

    match backend {
        BackendType::Ndarray => run_generate_with_config::<NdArrayAutodiff>(
            &config,
            prompt,
            checkpoint,
            device_location,
        ),
        BackendType::Wgpu => {
            run_generate_with_config::<WgpuAutodiff>(&config, prompt, checkpoint, device_location)
        }
        BackendType::Candle => {
            run_generate_with_config::<CandleAutodiff>(&config, prompt, checkpoint, device_location)
        }
        BackendType::Cuda => {
            run_generate_with_config::<CudaAutodiff>(&config, prompt, checkpoint, device_location)
        }
    }
}

fn run_generate_with_config<B: burn::tensor::backend::AutodiffBackend>(
    config: &FullConfig,
    prompt: &str,
    checkpoint: Option<&str>,
    device_location: DeviceLocation,
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

    let model: DPSN<B::InnerBackend> = if let Some(ckpt_path) = checkpoint {
        println!("Loading model from checkpoint: {}\n", ckpt_path);
        let fresh_model: DPSN<B::InnerBackend> = match &config.model.router {
            RouterSettings::Standard(s) => {
                let router_config = StandardRouterConfig {
                    k_min: s.k_min,
                    k_max: s.k_max,
                    hidden_dim: s.hidden_dim,
                    exploration_noise: s.exploration_noise,
                };
                DPSN::new(
                    dataset.vocab_size(),
                    config.model.embed_dim,
                    config.model.pool_size,
                    config.model.num_heads,
                    config.model.context_length,
                    config.model.recurrence_steps,
                    router_config,
                    &device,
                )
            }
            _ => {
                panic!("Only Standard router supported in checkpoing loading for now in this path")
            }
        };
        match load_checkpoint(fresh_model, &PathBuf::from(ckpt_path), &device) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load checkpoint: {}. Training new model...", e);
                // Fallback to training
                let router_config = StandardRouterConfig {
                    k_min: config.model.router.as_standard().unwrap().k_min,
                    k_max: config.model.router.as_standard().unwrap().k_max,
                    hidden_dim: config.model.router.as_standard().unwrap().hidden_dim,
                    exploration_noise: config.model.router.as_standard().unwrap().exploration_noise,
                };
                train_with_curriculum::<B>(
                    TrainingConfig::new()
                        .with_num_steps(config.training.num_steps)
                        .with_batch_size(config.training.batch_size),
                    CurriculumConfig::new()
                        .with_warmup_steps(config.curriculum.warmup_steps)
                        .with_specialization_steps(config.curriculum.specialization_steps),
                    dataset.vocab_size(),
                    config.model.embed_dim,
                    config.model.pool_size,
                    router_config,
                    config.model.num_heads,
                    config.model.context_length,
                    &dataset,
                    &device,
                    device_location,
                )
            }
        }
    } else {
        println!("No checkpoint provided. Running training as per config...\n");
        let router_config = StandardRouterConfig {
            k_min: config.model.router.as_standard().unwrap().k_min,
            k_max: config.model.router.as_standard().unwrap().k_max,
            hidden_dim: config.model.router.as_standard().unwrap().hidden_dim,
            exploration_noise: config.model.router.as_standard().unwrap().exploration_noise,
        };
        train_with_curriculum::<B>(
            TrainingConfig::new()
                .with_num_steps(config.training.num_steps)
                .with_batch_size(config.training.batch_size),
            CurriculumConfig::new()
                .with_warmup_steps(config.curriculum.warmup_steps)
                .with_specialization_steps(config.curriculum.specialization_steps),
            dataset.vocab_size(),
            config.model.embed_dim,
            config.model.pool_size,
            router_config,
            config.model.num_heads,
            config.model.context_length,
            &dataset,
            &device,
            device_location,
        )
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

fn run_training_from_config<B: burn::tensor::backend::AutodiffBackend>(
    config: &FullConfig,
    device_location: DeviceLocation,
) {
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
        .with_num_epochs(config.training.num_epochs)
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

    if let RouterSettings::Hierarchical(h_config) = &config.model.router {
        println!("Using HIERARCHICAL router (reduced parameter count)\n");
        let router_config = HierarchicalRouterConfig {
            k_min: h_config.k_min,
            k_max: h_config.k_max,
            num_clusters: h_config.num_clusters,
            top_clusters: h_config.top_clusters,
            exploration_noise: h_config.exploration_noise,
        };

        let model = train_hierarchical::<B>(
            training_config,
            curriculum_config,
            dataset.vocab_size(),
            config.model.embed_dim,
            config.model.pool_size,
            router_config,
            config.model.num_heads,
            config.model.context_length,
            &dataset,
            &device,
            device_location,
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
        // Standard Router
        let s_config = match &config.model.router {
            RouterSettings::Standard(s) => s,
            _ => unreachable!(),
        };

        let router_config = StandardRouterConfig {
            k_min: s_config.k_min,
            k_max: s_config.k_max,
            hidden_dim: s_config.hidden_dim,
            exploration_noise: s_config.exploration_noise,
        };

        let _model = train_with_curriculum::<B>(
            training_config,
            curriculum_config,
            dataset.vocab_size(),
            config.model.embed_dim,
            config.model.pool_size,
            router_config,
            config.model.num_heads,
            config.model.context_length,
            &dataset,
            &device,
            device_location,
        );
    }
}

fn run_training<B: burn::tensor::backend::AutodiffBackend>(
    steps: Option<usize>,
    epochs: Option<usize>,
    batch_size: usize,
    lr: f64,
    pool_size: usize,
    embed_dim: usize,
    k_min: usize,
    k_max: usize,
    router_hidden_dim: usize,
    num_heads: usize,
    context_length: usize,
    data_dir: &str,
    device_location: DeviceLocation,
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
        .with_num_epochs(epochs)
        .with_batch_size(batch_size)
        .with_learning_rate(lr);

    let steps_estimate = if let Some(e) = epochs {
        e * (dataset.len() / batch_size)
    } else {
        steps.unwrap_or(500)
    };

    let curriculum_config = CurriculumConfig::new()
        .with_warmup_steps(steps_estimate / 5)
        .with_specialization_steps(steps_estimate * 2 / 5);

    let device = Default::default();

    let router_config = StandardRouterConfig {
        k_min,
        k_max,
        hidden_dim: router_hidden_dim,
        exploration_noise: 0.1,
    };

    let model = train_with_curriculum::<B>(
        training_config,
        curriculum_config,
        dataset.vocab_size(),
        embed_dim,
        pool_size,
        router_config,
        num_heads,
        context_length,
        &dataset,
        &device,
        device_location,
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
    device_location: DeviceLocation,
) {
    println!("=== DPSN Text Generation ===\n");

    let text = download_tiny_shakespeare(data_dir).expect("Failed to download dataset");
    let dataset = CharDataset::new(&text, 64);
    let device: <B::InnerBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let model: DPSN<B::InnerBackend> = if let Some(ckpt_path) = checkpoint {
        println!("Loading model from checkpoint: {}\n", ckpt_path);
        let router_config = StandardRouterConfig {
            k_min: 50,
            k_max: 500,
            hidden_dim: 64,
            exploration_noise: 0.1,
        };
        let fresh_model: DPSN<B::InnerBackend> = DPSN::new(
            dataset.vocab_size(),
            64,
            5000,
            4,
            64,
            1,
            router_config,
            &device,
        );
        match load_checkpoint(fresh_model, &PathBuf::from(ckpt_path), &device) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load checkpoint: {}. Training new model...", e);
                let training_config = TrainingConfig::new()
                    .with_num_steps(Some(100))
                    .with_batch_size(16);
                let curriculum_config = CurriculumConfig::new()
                    .with_warmup_steps(20)
                    .with_specialization_steps(60);
                let router_config = StandardRouterConfig {
                    k_min: 50,
                    k_max: 500,
                    hidden_dim: 64,
                    exploration_noise: 0.1,
                };
                train_with_curriculum::<B>(
                    training_config,
                    curriculum_config,
                    dataset.vocab_size(),
                    64,
                    5000,
                    router_config,
                    4, // num_heads default for hardcoded check
                    64,
                    &dataset,
                    &device,
                    device_location,
                )
            }
        }
    } else {
        println!("No checkpoint provided. Running quick training first...\n");
        let training_config = TrainingConfig::new()
            .with_num_steps(Some(100))
            .with_batch_size(16);
        let curriculum_config = CurriculumConfig::new()
            .with_warmup_steps(20)
            .with_specialization_steps(60);
        let router_config = StandardRouterConfig {
            k_min: 50,
            k_max: 500,
            hidden_dim: 64,
            exploration_noise: 0.1,
        };
        train_with_curriculum::<B>(
            training_config,
            curriculum_config,
            dataset.vocab_size(),
            64,
            5000,
            router_config,
            4,
            64,
            &dataset,
            &device,
            device_location,
        )
    };

    let generator = TextGenerator::new(&model, &dataset.tokenizer, device);
    let generated = generator.generate(prompt, max_tokens, temperature);

    println!("\n=== Generated Text ===\n");
    println!("{}", generated);
}

fn run_demo<B: burn::tensor::backend::AutodiffBackend>(device_location: DeviceLocation) {
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
        .with_num_steps(Some(100))
        .with_batch_size(16)
        .with_log_interval(20);

    let curriculum_config = CurriculumConfig::new()
        .with_warmup_steps(20)
        .with_specialization_steps(60);

    let device = Default::default();

    let router_config = StandardRouterConfig {
        k_min: 20,
        k_max: 200,
        hidden_dim: 64,
        exploration_noise: 0.1,
    };

    let model = train_with_curriculum::<B>(
        training_config,
        curriculum_config,
        dataset.vocab_size(),
        32,
        2000,
        router_config,
        4,
        32,
        &dataset,
        &device,
        device_location,
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
