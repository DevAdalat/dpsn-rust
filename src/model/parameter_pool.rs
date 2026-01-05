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
        let [batch_size, k] = indices.dims();
        let pool = self.pool.val();

        let indices_flat = indices.reshape([batch_size * k]);
        let selected = pool.select(0, indices_flat);

        selected.reshape([batch_size, k, self.dim])
    }
}
