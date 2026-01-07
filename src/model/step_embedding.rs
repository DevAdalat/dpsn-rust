use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct StepEmbeddingConfig {
    pub max_steps: usize,
    pub embed_dim: usize,
}

#[derive(Module, Debug)]
pub struct StepEmbedding<B: Backend> {
    embedding: Embedding<B>,
}

impl<B: Backend> StepEmbedding<B> {
    pub fn new(config: &StepEmbeddingConfig, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(config.max_steps, config.embed_dim).init(device);
        Self { embedding }
    }

    pub fn forward(&self, step: usize) -> Tensor<B, 2> {
        let device = self.embedding.devices().first().unwrap().clone();
        let indices =
            Tensor::<B, 1, burn::tensor::Int>::from_ints([step as i32], &device).unsqueeze(); // [1, 1]
        let out = self.embedding.forward(indices); // [1, 1, embed_dim]
        let [_batch, _seq, dim] = out.dims();
        out.reshape([1, dim]) // [1, embed_dim]
    }
}

impl StepEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StepEmbedding<B> {
        StepEmbedding::new(self, device)
    }
}
