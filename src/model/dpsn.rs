use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

use super::execution_engine::{ExecutionEngine, ExecutionEngineConfig};
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
