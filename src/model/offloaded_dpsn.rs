use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;
use std::cell::RefCell;

use super::execution_engine::{ExecutionEngine, ExecutionEngineConfig};
use super::offloaded_pool::{
    compute_pool_gradients, OffloadedParameterPool, OffloadedPoolConfig, PoolGradientTracker,
};
use super::router::{Router, RouterConfig, RouterOutput, RoutingMode};

#[derive(Module, Debug)]
pub struct OffloadedDPSNGpuPart<B: Backend> {
    pub embedding: Embedding<B>,
    pub router: Router<B>,
    pub engine: ExecutionEngine<B>,
    pub output_head: Linear<B>,
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

pub struct OffloadedDPSN<B: Backend, CpuB: Backend> {
    pub gpu_part: OffloadedDPSNGpuPart<B>,
    pub pool: OffloadedParameterPool<B, CpuB>,
    pub grad_tracker: RefCell<PoolGradientTracker<B>>,
    pub gpu_device: B::Device,
}

pub struct OffloadedDPSNOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub router_output: RouterOutput<B>,
    pub w_active: Tensor<B, 3>,
}

#[derive(Debug, Clone)]
pub struct OffloadedDPSNConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub pool_size: usize,
    pub k_min: usize,
    pub k_max: usize,
    pub router_hidden_dim: usize,
    pub context_length: usize,
    pub exploration_noise: f64,
}

impl OffloadedDPSNConfig {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        pool_size: usize,
        k_min: usize,
        k_max: usize,
        router_hidden_dim: usize,
        context_length: usize,
        exploration_noise: f64,
    ) -> Self {
        Self {
            vocab_size,
            embed_dim,
            pool_size,
            k_min,
            k_max,
            router_hidden_dim,
            context_length,
            exploration_noise,
        }
    }

    pub fn init<B: Backend, CpuB: Backend>(
        &self,
        gpu_device: &B::Device,
        cpu_device: &CpuB::Device,
    ) -> OffloadedDPSN<B, CpuB> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(gpu_device);

        let router = RouterConfig {
            embed_dim: self.embed_dim,
            hidden_dim: self.router_hidden_dim,
            pool_size: self.pool_size,
            k_min: self.k_min,
            k_max: self.k_max,
            noise_scale: self.exploration_noise,
        }
        .init(gpu_device);

        let engine = ExecutionEngineConfig {
            embed_dim: self.embed_dim,
            k_max: self.k_max,
        }
        .init(gpu_device);

        let output_head = LinearConfig::new(self.embed_dim, self.vocab_size).init(gpu_device);

        let gpu_part = OffloadedDPSNGpuPart {
            embedding,
            router,
            engine,
            output_head,
            vocab_size: self.vocab_size,
            embed_dim: self.embed_dim,
            pool_size: self.pool_size,
            k_min: self.k_min,
            k_max: self.k_max,
            router_hidden_dim: self.router_hidden_dim,
            context_length: self.context_length,
        };

        let pool = OffloadedPoolConfig::new(self.pool_size, self.embed_dim).init(cpu_device);

        OffloadedDPSN {
            gpu_part,
            pool,
            grad_tracker: RefCell::new(PoolGradientTracker::new()),
            gpu_device: gpu_device.clone(),
        }
    }
}

impl<B: Backend, CpuB: Backend> OffloadedDPSN<B, CpuB> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, training: bool) -> OffloadedDPSNOutput<B> {
        let mode = if training {
            RoutingMode::Guided {
                epsilon: self.gpu_part.router.noise_scale,
            }
        } else {
            RoutingMode::Deterministic
        };
        self.forward_with_mode(tokens, mode)
    }

    pub fn forward_with_mode(
        &self,
        tokens: Tensor<B, 2, Int>,
        mode: RoutingMode,
    ) -> OffloadedDPSNOutput<B> {
        let [batch_size, seq_len] = tokens.dims();
        let embed_dim = self.gpu_part.embed_dim;

        let embeddings = self.gpu_part.embedding.forward(tokens);
        let flat_embeddings = embeddings
            .clone()
            .reshape([batch_size * seq_len, embed_dim]);

        let router_output = self
            .gpu_part
            .router
            .forward_with_mode(flat_embeddings.clone(), mode);

        let w_active = self
            .pool
            .retrieve_to_gpu(router_output.indices.clone(), &self.gpu_device);

        self.grad_tracker.borrow_mut().cache_forward(
            router_output.indices.clone(),
            router_output.selected_scores.clone(),
        );

        let output = self.gpu_part.engine.forward(
            flat_embeddings,
            w_active.clone(),
            router_output.selected_scores.clone(),
        );

        let output_reshaped = output.reshape([batch_size, seq_len, embed_dim]);
        let logits = self.gpu_part.output_head.forward(output_reshaped);

        OffloadedDPSNOutput {
            logits,
            router_output,
            w_active,
        }
    }

    pub fn forward_inference(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(tokens, false).logits
    }

    pub fn param_stats(&self) -> OffloadedParameterStats {
        let pool_params = self.gpu_part.pool_size * self.gpu_part.embed_dim;
        let router_params = self.gpu_part.embed_dim * self.gpu_part.router_hidden_dim
            + self.gpu_part.router_hidden_dim * self.gpu_part.pool_size
            + self.gpu_part.embed_dim;
        let embed_params = self.gpu_part.vocab_size * self.gpu_part.embed_dim;
        let output_params = self.gpu_part.embed_dim * self.gpu_part.vocab_size;
        let engine_params = self.gpu_part.embed_dim * self.gpu_part.embed_dim;

        let gpu_params = router_params + embed_params + output_params + engine_params;

        OffloadedParameterStats {
            pool_params,
            router_params,
            embed_params,
            output_params,
            engine_params,
            total_params: pool_params + gpu_params,
            gpu_params,
            cpu_params: pool_params,
            active_params_per_token: self.gpu_part.k_max * self.gpu_part.embed_dim,
            pool_memory_bytes: self.pool.memory_bytes(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OffloadedParameterStats {
    pub pool_params: usize,
    pub router_params: usize,
    pub embed_params: usize,
    pub output_params: usize,
    pub engine_params: usize,
    pub total_params: usize,
    pub gpu_params: usize,
    pub cpu_params: usize,
    pub active_params_per_token: usize,
    pub pool_memory_bytes: usize,
}

impl OffloadedParameterStats {
    pub fn print_summary(&self) {
        println!("=== Offloaded DPSN Model Statistics ===");
        println!("Total parameters: {}", self.total_params);
        println!(
            "  GPU parameters: {} ({:.2}%)",
            self.gpu_params,
            100.0 * self.gpu_params as f64 / self.total_params as f64
        );
        println!(
            "  CPU parameters (pool): {} ({:.2}%)",
            self.cpu_params,
            100.0 * self.cpu_params as f64 / self.total_params as f64
        );
        println!("Component breakdown:");
        println!("  Pool (CPU): {}", self.pool_params);
        println!("  Router (GPU): {}", self.router_params);
        println!("  Embedding (GPU): {}", self.embed_params);
        println!("  Output head (GPU): {}", self.output_params);
        println!("  Engine (GPU): {}", self.engine_params);
        println!("Active params per token: {}", self.active_params_per_token);
        println!(
            "Pool memory (with Adam state): {:.2} MB",
            self.pool_memory_bytes as f64 / 1024.0 / 1024.0
        );
        println!("=========================================");
    }
}

pub fn compute_w_active_gradients<B: Backend>(
    output_grad: Tensor<B, 2>,
    scores: Tensor<B, 2>,
) -> Tensor<B, 3> {
    compute_pool_gradients(output_grad, scores)
}
