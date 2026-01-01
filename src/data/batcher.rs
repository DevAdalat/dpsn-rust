use burn::prelude::*;
use burn::tensor::backend::Backend;

use super::dataset::CharDataset;

pub struct TrainingBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Clone)]
pub struct DPSNBatcher {
    pub context_length: usize,
}

impl DPSNBatcher {
    pub fn new(context_length: usize) -> Self {
        DPSNBatcher { context_length }
    }

    pub fn batch<B: Backend>(
        &self,
        dataset: &CharDataset,
        batch_size: usize,
        device: &B::Device,
    ) -> TrainingBatch<B> {
        let (inputs_vec, targets_vec) = dataset.get_random_batch(batch_size);

        let inputs_flat: Vec<i64> = inputs_vec.iter().flatten().map(|&x| x as i64).collect();

        let targets_flat: Vec<i64> = targets_vec.iter().flatten().map(|&x| x as i64).collect();

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_flat.as_slice(), device)
            .reshape([batch_size, self.context_length]);

        let targets = Tensor::<B, 1, Int>::from_ints(targets_flat.as_slice(), device)
            .reshape([batch_size, self.context_length]);

        TrainingBatch { inputs, targets }
    }
}
