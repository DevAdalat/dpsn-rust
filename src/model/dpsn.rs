use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

use super::execution_engine::{ExecutionEngine, ExecutionEngineConfig};
use super::hierarchical_router::{HierarchicalRouter, HierarchicalRouterConfig};
use super::parameter_pool::{ParameterPool, ParameterPoolConfig};
use super::router::{Router, RouterConfig, RouterOutput, RoutingMode};

#[derive(Module, Debug)]
pub struct DPSN<B: Backend> {
    embedding: Embedding<B>,
    router: Router<B>,
    pool: ParameterPool<B>,
    engine: ExecutionEngine<B>,
    output_head: Linear<B>,
    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    pub embed_dim: usize,
    #[module(skip)]
    pub pool_size: usize,
    #[module(skip)]
    pub k_min: usize,
    #[module(skip)]
    pub k_max: usize,
    #[module(skip)]
    pub router_hidden_dim: usize,
    #[module(skip)]
    pub context_length: usize,
}

pub struct DPSNOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub router_output: RouterOutput<B>,
}

impl<B: Backend> DPSN<B> {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        pool_size: usize,
        k_min: usize,
        k_max: usize,
        router_hidden_dim: usize,
        context_length: usize,
        exploration_noise: f64,
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, embed_dim).init(device);

        let router = RouterConfig {
            embed_dim,
            hidden_dim: router_hidden_dim,
            pool_size,
            k_min,
            k_max,
            noise_scale: exploration_noise,
        }
        .init(device);

        let pool = ParameterPoolConfig {
            pool_size,
            dim: embed_dim,
        }
        .init(device);

        let engine = ExecutionEngineConfig { embed_dim, k_max }.init(device);

        let output_head = LinearConfig::new(embed_dim, vocab_size).init(device);

        DPSN {
            embedding,
            router,
            pool,
            engine,
            output_head,
            vocab_size,
            embed_dim,
            pool_size,
            k_min,
            k_max,
            router_hidden_dim,
            context_length,
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>, training: bool) -> DPSNOutput<B> {
        let mode = if training {
            RoutingMode::Guided {
                epsilon: self.router.noise_scale,
            }
        } else {
            RoutingMode::Deterministic
        };
        self.forward_with_mode(tokens, mode)
    }

    pub fn forward_with_mode(&self, tokens: Tensor<B, 2, Int>, mode: RoutingMode) -> DPSNOutput<B> {
        let [batch_size, seq_len] = tokens.dims();

        let embeddings = self.embedding.forward(tokens);

        let flat_embeddings = embeddings
            .clone()
            .reshape([batch_size * seq_len, self.embed_dim]);

        let router_output = self.router.forward_with_mode(flat_embeddings.clone(), mode);

        let w_active = self.pool.retrieve(router_output.indices.clone());

        let output = self.engine.forward(
            flat_embeddings,
            w_active,
            router_output.selected_scores.clone(),
        );

        let output_reshaped = output.reshape([batch_size, seq_len, self.embed_dim]);

        let logits = self.output_head.forward(output_reshaped);

        DPSNOutput {
            logits,
            router_output,
        }
    }

    pub fn forward_inference(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(tokens, false).logits
    }

    pub fn param_stats(&self) -> ParameterStats {
        let pool_params = self.pool_size * self.embed_dim;
        let router_params = self.embed_dim * self.router_hidden_dim
            + self.router_hidden_dim * self.pool_size
            + self.embed_dim;
        let embed_params = self.vocab_size * self.embed_dim;
        let output_params = self.embed_dim * self.vocab_size;
        let engine_params = self.embed_dim * self.embed_dim;

        ParameterStats {
            pool_params,
            router_params,
            embed_params,
            output_params,
            engine_params,
            total_params: pool_params
                + router_params
                + embed_params
                + output_params
                + engine_params,
            active_params_per_token: self.k_max * self.embed_dim,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParameterStats {
    pub pool_params: usize,
    pub router_params: usize,
    pub embed_params: usize,
    pub output_params: usize,
    pub engine_params: usize,
    pub total_params: usize,
    pub active_params_per_token: usize,
}

impl ParameterStats {
    pub fn pool_memory_bytes(&self, precision: Precision) -> usize {
        self.pool_params * precision.bytes_per_param()
    }

    pub fn router_memory_bytes(&self, precision: Precision) -> usize {
        self.router_params * precision.bytes_per_param()
    }

    pub fn embed_memory_bytes(&self, precision: Precision) -> usize {
        self.embed_params * precision.bytes_per_param()
    }

    pub fn output_memory_bytes(&self, precision: Precision) -> usize {
        self.output_params * precision.bytes_per_param()
    }

    pub fn engine_memory_bytes(&self, precision: Precision) -> usize {
        self.engine_params * precision.bytes_per_param()
    }

    pub fn total_memory_bytes(&self, precision: Precision) -> usize {
        self.total_params * precision.bytes_per_param()
    }

    pub fn active_memory_per_token_bytes(&self, precision: Precision) -> usize {
        self.active_params_per_token * precision.bytes_per_param()
    }

    pub fn optimizer_memory_bytes(&self, precision: Precision) -> usize {
        self.total_params * precision.bytes_per_param() * 2
    }

    pub fn training_memory_bytes(&self, precision: Precision) -> usize {
        self.total_memory_bytes(precision) + self.optimizer_memory_bytes(precision)
    }

    pub fn print_summary(
        &self,
        model_name: &str,
        device_location: DeviceLocation,
        precision: Precision,
    ) {
        let p = precision;

        println!();
        println!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                        {} MODEL SUMMARY                          ║",
            model_name.to_uppercase()
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║  Precision: {:?} ({} bytes/param)                                           ║",
            p,
            p.bytes_per_param()
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║                              PARAMETER COUNT                                 ║"
        );
        println!("╠────────────────────────┬─────────────────┬──────────────────┬───────────────╣");
        println!("║ Component              │ Parameters      │ Memory           │ Location      ║");
        println!("╠────────────────────────┼─────────────────┼──────────────────┼───────────────╣");

        println!(
            "║ Pool                   │ {:>15} │ {:>16} │ {:^13} ║",
            format_params(self.pool_params),
            format_bytes(self.pool_memory_bytes(p)),
            device_location.pool_location()
        );
        println!(
            "║ Router                 │ {:>15} │ {:>16} │ {:^13} ║",
            format_params(self.router_params),
            format_bytes(self.router_memory_bytes(p)),
            device_location.router_location()
        );
        println!(
            "║ Embedding              │ {:>15} │ {:>16} │ {:^13} ║",
            format_params(self.embed_params),
            format_bytes(self.embed_memory_bytes(p)),
            device_location.embed_location()
        );
        println!(
            "║ Output Head            │ {:>15} │ {:>16} │ {:^13} ║",
            format_params(self.output_params),
            format_bytes(self.output_memory_bytes(p)),
            device_location.output_location()
        );
        println!(
            "║ Execution Engine       │ {:>15} │ {:>16} │ {:^13} ║",
            format_params(self.engine_params),
            format_bytes(self.engine_memory_bytes(p)),
            device_location.engine_location()
        );
        println!("╠────────────────────────┼─────────────────┼──────────────────┼───────────────╣");
        println!(
            "║ TOTAL                  │ {:>15} │ {:>16} │               ║",
            format_params(self.total_params),
            format_bytes(self.total_memory_bytes(p))
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║                              MEMORY BREAKDOWN                                ║"
        );
        println!("╠────────────────────────────────────────────────────────────────────────────╣");

        let gpu_mem = device_location.gpu_memory_bytes(self, p);
        let cpu_mem = device_location.cpu_memory_bytes(self, p);

        println!("║  GPU VRAM Required:     {:>52} ║", format_bytes(gpu_mem));
        println!("║  System RAM Required:   {:>52} ║", format_bytes(cpu_mem));
        println!(
            "║  Optimizer State (Adam): {:>51} ║",
            format_bytes(self.optimizer_memory_bytes(p))
        );
        println!(
            "║  Total Training Memory: {:>52} ║",
            format_bytes(self.training_memory_bytes(p))
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║                              INFERENCE STATS                                 ║"
        );
        println!("╠────────────────────────────────────────────────────────────────────────────╣");
        println!(
            "║  Active Params/Token:   {:>52} ║",
            format_params(self.active_params_per_token)
        );
        println!(
            "║  Active Memory/Token:   {:>52} ║",
            format_bytes(self.active_memory_per_token_bytes(p))
        );
        println!(
            "║  Sparsity Ratio:        {:>51.2}% ║",
            100.0 * (1.0 - self.active_params_per_token as f64 / self.pool_params as f64)
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
        println!();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Precision {
    F32,
    F16,
    BF16,
}

impl Precision {
    pub fn bytes_per_param(&self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F16 | Precision::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DeviceLocation {
    AllGpu,
    AllCpu,
    Offloaded,
}

impl DeviceLocation {
    pub fn pool_location(&self) -> &'static str {
        match self {
            DeviceLocation::AllGpu => "GPU",
            DeviceLocation::AllCpu => "CPU",
            DeviceLocation::Offloaded => "CPU (RAM)",
        }
    }

    pub fn router_location(&self) -> &'static str {
        match self {
            DeviceLocation::AllGpu | DeviceLocation::Offloaded => "GPU",
            DeviceLocation::AllCpu => "CPU",
        }
    }

    pub fn embed_location(&self) -> &'static str {
        match self {
            DeviceLocation::AllGpu | DeviceLocation::Offloaded => "GPU",
            DeviceLocation::AllCpu => "CPU",
        }
    }

    pub fn output_location(&self) -> &'static str {
        match self {
            DeviceLocation::AllGpu | DeviceLocation::Offloaded => "GPU",
            DeviceLocation::AllCpu => "CPU",
        }
    }

    pub fn engine_location(&self) -> &'static str {
        match self {
            DeviceLocation::AllGpu | DeviceLocation::Offloaded => "GPU",
            DeviceLocation::AllCpu => "CPU",
        }
    }

    pub fn gpu_memory_bytes(&self, stats: &ParameterStats, p: Precision) -> usize {
        match self {
            DeviceLocation::AllGpu => stats.total_memory_bytes(p),
            DeviceLocation::AllCpu => 0,
            DeviceLocation::Offloaded => {
                stats.router_memory_bytes(p)
                    + stats.embed_memory_bytes(p)
                    + stats.output_memory_bytes(p)
                    + stats.engine_memory_bytes(p)
            }
        }
    }

    pub fn cpu_memory_bytes(&self, stats: &ParameterStats, p: Precision) -> usize {
        match self {
            DeviceLocation::AllGpu => 0,
            DeviceLocation::AllCpu => stats.total_memory_bytes(p),
            DeviceLocation::Offloaded => stats.pool_memory_bytes(p) * 3,
        }
    }
}

fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn format_params(params: usize) -> String {
    const K: usize = 1_000;
    const M: usize = K * 1_000;
    const B: usize = M * 1_000;

    if params >= B {
        format!("{:.2}B", params as f64 / B as f64)
    } else if params >= M {
        format!("{:.2}M", params as f64 / M as f64)
    } else if params >= K {
        format!("{:.2}K", params as f64 / K as f64)
    } else {
        format!("{}", params)
    }
}

#[derive(Module, Debug)]
pub struct HierarchicalDPSN<B: Backend> {
    embedding: Embedding<B>,
    router: HierarchicalRouter<B>,
    pool: ParameterPool<B>,
    engine: ExecutionEngine<B>,
    output_head: Linear<B>,
    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    pub embed_dim: usize,
    #[module(skip)]
    pub pool_size: usize,
    #[module(skip)]
    pub k_min: usize,
    #[module(skip)]
    pub k_max: usize,
    #[module(skip)]
    pub num_clusters: usize,
    #[module(skip)]
    pub context_length: usize,
}

impl<B: Backend> HierarchicalDPSN<B> {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        pool_size: usize,
        k_min: usize,
        k_max: usize,
        num_clusters: usize,
        top_clusters: usize,
        context_length: usize,
        exploration_noise: f64,
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, embed_dim).init(device);

        let router = HierarchicalRouterConfig {
            embed_dim,
            pool_size,
            k_min,
            k_max,
            num_clusters,
            top_clusters,
            noise_scale: exploration_noise,
        }
        .init(device);

        let pool = ParameterPoolConfig {
            pool_size,
            dim: embed_dim,
        }
        .init(device);

        let engine = ExecutionEngineConfig { embed_dim, k_max }.init(device);

        let output_head = LinearConfig::new(embed_dim, vocab_size).init(device);

        HierarchicalDPSN {
            embedding,
            router,
            pool,
            engine,
            output_head,
            vocab_size,
            embed_dim,
            pool_size,
            k_min,
            k_max,
            num_clusters,
            context_length,
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>, training: bool) -> DPSNOutput<B> {
        let mode = if training {
            RoutingMode::Guided {
                epsilon: self.router.noise_scale,
            }
        } else {
            RoutingMode::Deterministic
        };
        self.forward_with_mode(tokens, mode)
    }

    pub fn forward_with_mode(&self, tokens: Tensor<B, 2, Int>, mode: RoutingMode) -> DPSNOutput<B> {
        let [batch_size, seq_len] = tokens.dims();

        let embeddings = self.embedding.forward(tokens);

        let flat_embeddings = embeddings
            .clone()
            .reshape([batch_size * seq_len, self.embed_dim]);

        let hierarchical_output = self.router.forward_with_mode(flat_embeddings.clone(), mode);
        let router_output: RouterOutput<B> = hierarchical_output.into();

        let w_active = self.pool.retrieve(router_output.indices.clone());

        let output = self.engine.forward(
            flat_embeddings,
            w_active,
            router_output.selected_scores.clone(),
        );

        let output_reshaped = output.reshape([batch_size, seq_len, self.embed_dim]);

        let logits = self.output_head.forward(output_reshaped);

        DPSNOutput {
            logits,
            router_output,
        }
    }

    pub fn forward_inference(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(tokens, false).logits
    }

    pub fn param_stats(&self) -> ParameterStats {
        let pool_params = self.pool_size * self.embed_dim;
        let cluster_size = (self.pool_size + self.num_clusters - 1) / self.num_clusters;
        let router_params =
            self.embed_dim * self.num_clusters + self.embed_dim * cluster_size + self.embed_dim;
        let embed_params = self.vocab_size * self.embed_dim;
        let output_params = self.embed_dim * self.vocab_size;
        let engine_params = self.embed_dim * self.embed_dim;

        ParameterStats {
            pool_params,
            router_params,
            embed_params,
            output_params,
            engine_params,
            total_params: pool_params
                + router_params
                + embed_params
                + output_params
                + engine_params,
            active_params_per_token: self.k_max * self.embed_dim,
        }
    }
}
