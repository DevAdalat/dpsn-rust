//! CPU-Offloaded Parameter Pool
//!
//! This module implements a parameter pool that lives in system RAM (CPU memory)
//! while the rest of the model runs on GPU. This enables training models with
//! billions of pool parameters on consumer GPUs.
//!
//! Architecture:
//! - Pool tensor stays on CPU (NdArray backend)
//! - Router generates indices on GPU
//! - Indices transferred to CPU, gather selected params
//! - Selected params pushed to GPU for forward/backward
//! - Gradients computed on GPU, transferred back to update CPU pool

use burn::prelude::*;
use burn::tensor::backend::Backend;
use burn::tensor::TensorData;
use std::marker::PhantomData;

/// CPU-side parameter pool that offloads memory to system RAM.
///
/// The pool tensor lives on the CPU backend (NdArray) while the main model
/// runs on GPU. Only the selected k parameters are transferred to GPU per forward pass.
#[derive(Debug)]
pub struct OffloadedParameterPool<B: Backend, CpuB: Backend> {
    /// Pool weights stored on CPU - shape: [pool_size, dim]
    pub pool_cpu: Tensor<CpuB, 2>,
    /// Momentum state for Adam optimizer (first moment)
    pub momentum_m: Tensor<CpuB, 2>,
    /// Momentum state for Adam optimizer (second moment)
    pub momentum_v: Tensor<CpuB, 2>,
    /// Adam timestep counter
    pub timestep: usize,
    /// Pool size
    pub pool_size: usize,
    /// Embedding dimension
    pub dim: usize,
    /// CPU device reference
    pub cpu_device: CpuB::Device,
    /// GPU backend marker
    _gpu_marker: PhantomData<B>,
}

#[derive(Debug, Clone)]
pub struct OffloadedPoolConfig {
    pub pool_size: usize,
    pub dim: usize,
}

impl OffloadedPoolConfig {
    pub fn new(pool_size: usize, dim: usize) -> Self {
        Self { pool_size, dim }
    }

    /// Initialize the offloaded pool on CPU
    pub fn init<B: Backend, CpuB: Backend>(
        &self,
        cpu_device: &CpuB::Device,
    ) -> OffloadedParameterPool<B, CpuB> {
        let pool_cpu: Tensor<CpuB, 2> = Tensor::random(
            [self.pool_size, self.dim],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            cpu_device,
        );

        let momentum_m = Tensor::<CpuB, 2>::zeros([self.pool_size, self.dim], cpu_device);
        let momentum_v = Tensor::<CpuB, 2>::zeros([self.pool_size, self.dim], cpu_device);

        OffloadedParameterPool {
            pool_cpu,
            momentum_m,
            momentum_v,
            timestep: 0,
            pool_size: self.pool_size,
            dim: self.dim,
            cpu_device: cpu_device.clone(),
            _gpu_marker: PhantomData,
        }
    }
}

impl<B: Backend, CpuB: Backend> OffloadedParameterPool<B, CpuB> {
    /// Retrieve parameters by indices and transfer to GPU.
    ///
    /// Flow:
    /// 1. Receive indices tensor from GPU (router output)
    /// 2. Transfer indices to CPU
    /// 3. Gather selected parameters from CPU pool
    /// 4. Transfer gathered parameters to GPU
    ///
    /// Returns: [batch_size, k, dim] tensor on GPU device
    pub fn retrieve_to_gpu(
        &self,
        indices_gpu: Tensor<B, 2, Int>,
        gpu_device: &B::Device,
    ) -> Tensor<B, 3> {
        let [batch_size, _k] = indices_gpu.dims();

        // Step 1: Transfer indices from GPU to CPU
        let indices_data = indices_gpu.into_data();
        let indices_cpu = Tensor::<CpuB, 2, Int>::from_data(indices_data, &self.cpu_device);

        // Step 2: Gather on CPU - expand indices for gather operation
        let indices_expanded = indices_cpu.unsqueeze_dim::<3>(2).repeat_dim(2, self.dim);
        let indices_expanded = make_contiguous_int(indices_expanded);

        let pool_expanded = self
            .pool_cpu
            .clone()
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch_size);
        let pool_expanded = make_contiguous(pool_expanded);

        let gathered_cpu: Tensor<CpuB, 3> = pool_expanded.gather(1, indices_expanded);

        // Step 3: Transfer gathered params from CPU to GPU
        let gathered_data = gathered_cpu.into_data();
        Tensor::<B, 3>::from_data(gathered_data, gpu_device)
    }

    /// Update pool parameters with gradients computed on GPU.
    ///
    /// This implements sparse Adam update - only updates the parameters
    /// that were actually used (indexed) in the forward pass.
    ///
    /// Arguments:
    /// - indices_gpu: The indices that were selected [batch_size, k]
    /// - grads_gpu: Gradients for those parameters [batch_size, k, dim]
    /// - lr: Learning rate
    /// - beta1: Adam beta1 (default 0.9)
    /// - beta2: Adam beta2 (default 0.999)
    /// - eps: Adam epsilon (default 1e-8)
    pub fn update_with_gradients(
        &mut self,
        indices_gpu: Tensor<B, 2, Int>,
        grads_gpu: Tensor<B, 3>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    ) {
        self.timestep += 1;

        let [batch_size, k, _dim] = grads_gpu.dims();

        // Transfer gradients from GPU to CPU
        let grads_data = grads_gpu.into_data();
        let grads_cpu = Tensor::<CpuB, 3>::from_data(grads_data, &self.cpu_device);

        // Transfer indices from GPU to CPU
        let indices_data = indices_gpu.into_data();
        let indices_cpu = Tensor::<CpuB, 2, Int>::from_data(indices_data, &self.cpu_device);

        // Aggregate gradients per unique index (sum across batch)
        // First, flatten to process all indices together
        let indices_flat = indices_cpu.clone().reshape([batch_size * k]);
        let grads_flat = grads_cpu.reshape([batch_size * k, self.dim]);

        // Get unique indices and aggregate gradients
        let indices_vec: Vec<i64> = indices_flat.into_data().to_vec().unwrap();

        // Accumulate gradients for each unique index
        let mut grad_accum: std::collections::HashMap<i64, Vec<f32>> =
            std::collections::HashMap::new();
        let mut grad_counts: std::collections::HashMap<i64, usize> =
            std::collections::HashMap::new();

        let grads_data: Vec<f32> = grads_flat.into_data().to_vec().unwrap();

        for (i, &idx) in indices_vec.iter().enumerate() {
            let start = i * self.dim;
            let end = start + self.dim;
            let grad_slice = &grads_data[start..end];

            grad_accum
                .entry(idx)
                .and_modify(|acc| {
                    for (j, &g) in grad_slice.iter().enumerate() {
                        acc[j] += g;
                    }
                })
                .or_insert_with(|| grad_slice.to_vec());

            *grad_counts.entry(idx).or_insert(0) += 1;
        }

        // Apply Adam update for each unique index
        let pool_data: Vec<f32> = self.pool_cpu.clone().into_data().to_vec().unwrap();
        let m_data: Vec<f32> = self.momentum_m.clone().into_data().to_vec().unwrap();
        let v_data: Vec<f32> = self.momentum_v.clone().into_data().to_vec().unwrap();

        let mut new_pool = pool_data.clone();
        let mut new_m = m_data.clone();
        let mut new_v = v_data.clone();

        let bias_correction1 = 1.0 - beta1.powi(self.timestep as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.timestep as i32);

        for (idx, grad_sum) in grad_accum.iter() {
            let count = grad_counts[idx] as f32;
            let param_start = (*idx as usize) * self.dim;

            for d in 0..self.dim {
                let pos = param_start + d;
                let g = grad_sum[d] / count; // Average gradient

                // Adam update
                new_m[pos] = (beta1 as f32) * new_m[pos] + (1.0 - beta1 as f32) * g;
                new_v[pos] = (beta2 as f32) * new_v[pos] + (1.0 - beta2 as f32) * g * g;

                let m_hat = new_m[pos] / bias_correction1 as f32;
                let v_hat = new_v[pos] / bias_correction2 as f32;

                new_pool[pos] -= (lr as f32) * m_hat / (v_hat.sqrt() + eps as f32);
            }
        }

        self.pool_cpu = Tensor::<CpuB, 2>::from_data(
            TensorData::new(new_pool, [self.pool_size, self.dim]),
            &self.cpu_device,
        );

        self.momentum_m = Tensor::<CpuB, 2>::from_data(
            TensorData::new(new_m, [self.pool_size, self.dim]),
            &self.cpu_device,
        );

        self.momentum_v = Tensor::<CpuB, 2>::from_data(
            TensorData::new(new_v, [self.pool_size, self.dim]),
            &self.cpu_device,
        );
    }

    /// Get total memory usage of the pool in bytes (approximate)
    pub fn memory_bytes(&self) -> usize {
        // pool + momentum_m + momentum_v, each f32
        self.pool_size * self.dim * 4 * 3
    }

    /// Get pool size
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
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

/// Gradient tracker for computing pool gradients through autograd
///
/// Since the pool lives on CPU, we need to manually track which parameters
/// were used and compute their gradients from the execution engine output.
#[derive(Debug)]
pub struct PoolGradientTracker<B: Backend> {
    /// Cached indices from forward pass
    pub cached_indices: Option<Tensor<B, 2, Int>>,
    /// Cached scores for gradient scaling
    pub cached_scores: Option<Tensor<B, 2>>,
}

impl<B: Backend> Default for PoolGradientTracker<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> PoolGradientTracker<B> {
    pub fn new() -> Self {
        Self {
            cached_indices: None,
            cached_scores: None,
        }
    }

    pub fn cache_forward(&mut self, indices: Tensor<B, 2, Int>, scores: Tensor<B, 2>) {
        self.cached_indices = Some(indices);
        self.cached_scores = Some(scores);
    }

    pub fn clear(&mut self) {
        self.cached_indices = None;
        self.cached_scores = None;
    }
}

/// Compute gradients for pool parameters given output gradients.
///
/// The execution engine computes: output = sum(w_i * softmax(score_i)) + x
/// So d_loss/d_w_i = d_loss/d_output * softmax(score_i)
pub fn compute_pool_gradients<B: Backend>(
    output_grad: Tensor<B, 2>,
    scores: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let weights = burn::tensor::activation::softmax(scores, 1);
    let [_batch_size, k] = weights.dims();
    let [_, dim] = output_grad.dims();

    // d_w_i = output_grad * weight_i (outer product style, but weight is scalar per k)
    // Shape: [batch_size, k, dim]
    let weights_expanded = weights.unsqueeze_dim::<3>(2).repeat_dim(2, dim);
    let output_grad_expanded = output_grad.unsqueeze_dim::<3>(1).repeat_dim(1, k);

    output_grad_expanded * weights_expanded
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_offloaded_pool_creation() {
        let device = Default::default();
        let config = OffloadedPoolConfig::new(1024, 256);
        let pool: OffloadedParameterPool<TestBackend, TestBackend> = config.init(&device);

        assert_eq!(pool.pool_size(), 1024);
        assert_eq!(pool.dim(), 256);
    }
}
