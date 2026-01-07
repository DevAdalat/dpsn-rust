use super::config::{HierarchicalRouterConfig as HRouterConfig, StandardRouterConfig};
use super::execution_engine::{ExecutionEngine, ExecutionEngineConfig};
use super::hierarchical_router::{HierarchicalRouter, HierarchicalRouterConfig};
use super::parameter_pool::{ParameterPool, ParameterPoolConfig};
use super::router::{Router, RouterConfig, RouterOutput, RoutingMode};
use super::step_embedding::{StepEmbedding, StepEmbeddingConfig};
use burn::module::Module;
use burn::nn::attention::{
    generate_autoregressive_mask, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig,
};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct DPSN<B: Backend> {
    embedding: Embedding<B>,
    step_embedding: StepEmbedding<B>,
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
    #[module(skip)]
    pub recurrence_steps: usize,
}

#[derive(Clone)]
pub struct DPSNOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub router_outputs: Vec<RouterOutput<B>>,
}

impl<B: Backend> DPSN<B> {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        pool_size: usize,
        num_heads: usize,
        context_length: usize,
        recurrence_steps: usize,
        router_config: StandardRouterConfig,
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, embed_dim).init(device);
        let step_embedding = StepEmbedding::new(
            &StepEmbeddingConfig {
                max_steps: recurrence_steps,
                embed_dim,
            },
            device,
        );

        let norm1 = LayerNormConfig::new(embed_dim).init(device);
        let attention = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(0.1)
            .init(device);
        let norm2 = LayerNormConfig::new(embed_dim).init(device);

        let router = RouterConfig {
            embed_dim,
            hidden_dim: router_config.hidden_dim,
            pool_size,
            k_min: router_config.k_min,
            k_max: router_config.k_max,
            noise_scale: router_config.exploration_noise,
        }
        .init(device);

        let pool = ParameterPoolConfig {
            pool_size,
            dim: embed_dim,
        }
        .init(device);

        let engine = ExecutionEngineConfig {
            embed_dim,
            k_max: router_config.k_max,
        }
        .init(device);

        let output_head = LinearConfig::new(embed_dim, vocab_size).init(device);

        DPSN {
            embedding,
            step_embedding,
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
            k_min: router_config.k_min,
            k_max: router_config.k_max,
            router_hidden_dim: router_config.hidden_dim,
            num_heads,
            context_length,
            recurrence_steps,
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
        let mut x = embeddings.clone();
        let mut router_outputs = Vec::new();

        for t in 0..self.recurrence_steps {
            let step_emb = self.step_embedding.forward(t);
            let step_emb = step_emb.unsqueeze_dim::<3>(0);
            let current_state = x.clone() + step_emb;

            let x_norm1 = self.norm1.forward(current_state);
            let mask = generate_autoregressive_mask(batch_size, seq_len, &x_norm1.device());
            let attn_input = MhaInput::self_attn(x_norm1).mask_attn(mask);
            let attn_output = self.attention.forward(attn_input);
            x = x + attn_output.context; // Residual 1

            // 3. Router & Execution
            let flat_embeddings = self
                .norm2
                .forward(x.clone())
                .reshape([batch_size * seq_len, self.embed_dim]);

            let router_output = self
                .router
                .forward_with_mode(flat_embeddings.clone(), mode.clone());
            let indices = router_output.indices.clone();
            let scores = router_output.selected_scores.clone();

            router_outputs.push(router_output);

            let w_active = self.pool.retrieve(indices);

            let output = self.engine.forward(flat_embeddings, w_active, scores);

            let output_reshaped = output.reshape([batch_size, seq_len, self.embed_dim]);

            x = x + output_reshaped; // Residual 2
        }

        let logits = self.output_head.forward(x);

        DPSNOutput {
            logits,
            router_outputs,
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
    step_embedding: StepEmbedding<B>,
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
    #[module(skip)]
    pub recurrence_steps: usize,
}

impl<B: Backend> HierarchicalDPSN<B> {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        pool_size: usize,
        num_heads: usize,
        context_length: usize,
        recurrence_steps: usize,
        router_config: HRouterConfig,
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, embed_dim).init(device);
        let step_embedding = StepEmbedding::new(
            &StepEmbeddingConfig {
                max_steps: recurrence_steps,
                embed_dim,
            },
            device,
        );

        let norm1 = LayerNormConfig::new(embed_dim).init(device);
        let attention = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(0.1)
            .init(device);
        let norm2 = LayerNormConfig::new(embed_dim).init(device);

        let router = HierarchicalRouterConfig {
            embed_dim,
            pool_size,
            num_clusters: router_config.num_clusters,
            top_clusters: router_config.top_clusters,
            k_min: router_config.k_min,
            k_max: router_config.k_max,
            noise_scale: router_config.exploration_noise,
        }
        .init(device);

        let pool = ParameterPoolConfig {
            pool_size,
            dim: embed_dim,
        }
        .init(device);

        let engine = ExecutionEngineConfig {
            embed_dim,
            k_max: router_config.k_max,
        }
        .init(device);

        let output_head = LinearConfig::new(embed_dim, vocab_size).init(device);

        HierarchicalDPSN {
            embedding,
            step_embedding,
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
            k_min: router_config.k_min,
            k_max: router_config.k_max,
            num_clusters: router_config.num_clusters,
            num_heads,
            context_length,
            recurrence_steps,
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
        let mut x = embeddings.clone();
        let mut router_outputs = Vec::new();

        for t in 0..self.recurrence_steps {
            let step_emb = self.step_embedding.forward(t);
            let step_emb = step_emb.unsqueeze_dim::<3>(0);
            let current_state = x.clone() + step_emb;

            let x_norm1 = self.norm1.forward(current_state);
            let mask = generate_autoregressive_mask(batch_size, seq_len, &x_norm1.device());
            let attn_input = MhaInput::self_attn(x_norm1).mask_attn(mask);
            let attn_output = self.attention.forward(attn_input);
            x = x + attn_output.context; // Residual 1

            // 3. Router & Execution
            let flat_embeddings = self
                .norm2
                .forward(x.clone())
                .reshape([batch_size * seq_len, self.embed_dim]);

            let hierarchical_output = self
                .router
                .forward_with_mode(flat_embeddings.clone(), mode.clone());
            let router_output: RouterOutput<B> = hierarchical_output.into();
            let indices = router_output.indices.clone();
            let scores = router_output.selected_scores.clone();

            router_outputs.push(router_output);

            let w_active = self.pool.retrieve(indices);

            let output = self.engine.forward(flat_embeddings, w_active, scores);

            let output_reshaped = output.reshape([batch_size, seq_len, self.embed_dim]);

            x = x + output_reshaped; // Residual 2
        }

        let logits = self.output_head.forward(x);

        DPSNOutput {
            logits,
            router_outputs,
        }
    }

    pub fn forward_inference(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(tokens, false).logits
    }

    pub fn param_stats(&self) -> ParameterStats {
        let pool_params = self.pool_size * self.embed_dim;
        let router_params = self.router.param_count();
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
    pub fn print_summary(&self, model_name: &str) {
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
