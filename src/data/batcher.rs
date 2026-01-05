use burn::prelude::*;
use burn::tensor::backend::Backend;

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
        inputs_vec: Vec<Vec<usize>>,
        targets_vec: Vec<Vec<usize>>,
        device: &B::Device,
    ) -> TrainingBatch<B> {
        let batch_size = inputs_vec.len();

        let inputs_flat: Vec<i64> = inputs_vec.iter().flatten().map(|&x| x as i64).collect();

        let targets_flat: Vec<i64> = targets_vec.iter().flatten().map(|&x| x as i64).collect();

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_flat.as_slice(), device)
            .reshape([batch_size, self.context_length]);

        let targets = Tensor::<B, 1, Int>::from_ints(targets_flat.as_slice(), device)
            .reshape([batch_size, self.context_length]);

        TrainingBatch { inputs, targets }
    }
}
