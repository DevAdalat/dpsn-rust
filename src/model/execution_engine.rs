use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ExecutionEngine<B: Backend> {
    output_projection: Linear<B>,
    #[module(skip)]
    dim: usize,
}

#[derive(Config, Debug)]
pub struct ExecutionEngineConfig {
    pub embed_dim: usize,
    pub k_max: usize,
}

impl ExecutionEngineConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ExecutionEngine<B> {
        ExecutionEngine {
            output_projection: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            dim: self.embed_dim,
        }
    }
}

impl<B: Backend> ExecutionEngine<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        w_active: Tensor<B, 3>,
        scores: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [_batch_size, _k, _d] = w_active.dims();

        let weights = activation::softmax(scores, 1);

        let weights_expanded = weights.unsqueeze_dim::<3>(2);

        let aggregated: Tensor<B, 2> = (w_active * weights_expanded).sum_dim(1).squeeze();

        let combined = x + aggregated;

        self.output_projection.forward(combined)
    }
}
