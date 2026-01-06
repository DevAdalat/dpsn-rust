use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

use super::execution_engine::{ExecutionEngine, ExecutionEngineConfig};
use super::hierarchical_router::{HierarchicalRouter, HierarchicalRouterConfig};
use super::parameter_pool::{ParameterPool, ParameterPoolConfig};
use super::router::{Router, RouterConfig, RouterOutput, RoutingMode};

#[derive(Module, Debug)]
pub struct DPSN<B: Backend> {
    embedding: Embedding<B>,
    norm1: LayerNorm<B>,
    attention: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
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
    pub num_heads: usize,
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
        num_heads: usize,
        context_length: usize,
        exploration_noise: f64,
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, embed_dim).init(device);

        let norm1 = LayerNormConfig::new(embed_dim).init(device);
        let attention = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(0.1)
            .init(device);
        let norm2 = LayerNormConfig::new(embed_dim).init(device);

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
            norm1,
            attention,
            norm2,
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
            num_heads,
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

        let x = self.norm1.forward(embeddings.clone());
        let attn_input = MhaInput::self_attn(x);
        let attn_output = self.attention.forward(attn_input);
        let x = embeddings + attn_output.context;

        let flat_embeddings = self
            .norm2
            .forward(x)
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
        let router_params =
            self.embed_dim * self.router_hidden_dim * 2 + self.router_hidden_dim * self.pool_size;
        let embed_params = self.vocab_size * self.embed_dim;
        let output_params = self.embed_dim * self.vocab_size;
        let engine_params = self.embed_dim * self.embed_dim;
        let attn_params = self.embed_dim * self.embed_dim * 4;

        ParameterStats {
            pool_params,
            router_params,
            embed_params,
            output_params,
            engine_params: engine_params + attn_params,
            total_params: pool_params
                + router_params
                + embed_params
                + output_params
                + engine_params
                + attn_params,
            active_params_per_token: self.k_max * self.embed_dim + attn_params,
        }
    }
}

#[derive(Module, Debug)]
pub struct HierarchicalDPSN<B: Backend> {
    embedding: Embedding<B>,
    norm1: LayerNorm<B>,
    attention: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
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
    pub num_heads: usize,
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
        num_heads: usize,
        context_length: usize,
        exploration_noise: f64,
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, embed_dim).init(device);

        let norm1 = LayerNormConfig::new(embed_dim).init(device);
        let attention = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(0.1)
            .init(device);
        let norm2 = LayerNormConfig::new(embed_dim).init(device);

        let router = HierarchicalRouterConfig {
            embed_dim,
            pool_size,
            num_clusters,
            top_clusters,
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

        HierarchicalDPSN {
            embedding,
            norm1,
            attention,
            norm2,
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
            num_heads,
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

        let x = self.norm1.forward(embeddings.clone());
        let attn_input = MhaInput::self_attn(x);
        let attn_output = self.attention.forward(attn_input);
        let x = embeddings + attn_output.context;

        let flat_embeddings = self
            .norm2
            .forward(x)
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
        let attn_params = self.embed_dim * self.embed_dim * 4;

        ParameterStats {
            pool_params,
            router_params,
            embed_params,
            output_params,
            engine_params: engine_params + attn_params,
            total_params: pool_params
                + router_params
                + embed_params
                + output_params
                + engine_params
                + attn_params,
            active_params_per_token: self.k_max * self.embed_dim + attn_params,
        }
    }
}

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
    pub fn print_summary(
        &self,
        model_name: &str,
        device_location: DeviceLocation,
        precision: Precision,
    ) {
        let scale = 1_000_000.0;
        let unit = "M";

        println!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!("║  MODEL SUMMARY: {:<52} ║", model_name);
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║  Parameter Pool:       {:>8.2} {} ({:>5.1}%)                                  ║",
            self.pool_params as f64 / scale,
            unit,
            100.0 * self.pool_params as f64 / self.total_params as f64
        );
        println!(
            "║  Router Network:       {:>8.2} {} ({:>5.1}%)                                  ║",
            self.router_params as f64 / scale,
            unit,
            100.0 * self.router_params as f64 / self.total_params as f64
        );
        println!(
            "║  Embeddings/Heads:     {:>8.2} {} ({:>5.1}%)                                  ║",
            (self.embed_params + self.output_params) as f64 / scale,
            unit,
            100.0 * (self.embed_params + self.output_params) as f64 / self.total_params as f64
        );
        println!(
            "║  Execution Engine:     {:>8.2} {} ({:>5.1}%)                                  ║",
            self.engine_params as f64 / scale,
            unit,
            100.0 * self.engine_params as f64 / self.total_params as f64
        );
        println!("╠────────────────────────────────────────────────────────────────────────────╣");
        println!(
            "║  TOTAL PARAMETERS:     {:>8.2} {}                                         ║",
            self.total_params as f64 / scale,
            unit
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║  Active Params/Token:  {:>8.2} {} ({:>5.2}% sparsity)                         ║",
            self.active_params_per_token as f64 / scale,
            unit,
            100.0 * (1.0 - (self.active_params_per_token as f64 / self.total_params as f64))
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
        println!();
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DeviceLocation {
    AllCpu,
    AllGpu,
    Offloaded,
}

#[derive(Clone, Copy, Debug)]
pub enum Precision {
    F32,
    F16,
}
