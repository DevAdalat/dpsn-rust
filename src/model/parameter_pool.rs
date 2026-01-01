use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ParameterPool<B: Backend> {
    pub pool: Param<Tensor<B, 2>>,
    #[module(skip)]
    pub pool_size: usize,
    #[module(skip)]
    pub dim: usize,
}

#[derive(Config, Debug)]
pub struct ParameterPoolConfig {
    pub pool_size: usize,
    pub dim: usize,
}

impl ParameterPoolConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ParameterPool<B> {
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let pool = initializer.init_with([self.pool_size, self.dim], None, None, device);

        ParameterPool {
            pool,
            pool_size: self.pool_size,
            dim: self.dim,
        }
    }
}

fn make_contiguous<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let shape = tensor.dims();
    tensor.reshape(shape)
}

fn make_contiguous_int<B: Backend, const D: usize>(tensor: Tensor<B, D, Int>) -> Tensor<B, D, Int> {
    let shape = tensor.dims();
    tensor.reshape(shape)
}

impl<B: Backend> ParameterPool<B> {
    pub fn retrieve(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, _k] = indices.dims();
        let pool = self.pool.val();
        let d = self.dim;

        let indices_expanded = indices.clone().unsqueeze_dim::<3>(2).repeat_dim(2, d);
        let indices_expanded = make_contiguous_int(indices_expanded);

        let pool_expanded = pool.clone().unsqueeze_dim::<3>(0).repeat_dim(0, batch_size);
        let pool_expanded = make_contiguous(pool_expanded);

        pool_expanded.gather(1, indices_expanded)
    }
}
