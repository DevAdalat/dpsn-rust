use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use rand::Rng;

use super::router::{RouterOutput, RoutingMode};

#[derive(Module, Debug)]
pub struct HierarchicalRouter<B: Backend> {
    complexity_layer: Linear<B>,
    cluster_scorer: Linear<B>,
    intra_cluster_scorer: Linear<B>,
    #[module(skip)]
    pub num_clusters: usize,
    #[module(skip)]
    pub cluster_size: usize,
    #[module(skip)]
    pub pool_size: usize,
    #[module(skip)]
    pub k_min: usize,
    #[module(skip)]
    pub k_max: usize,
    #[module(skip)]
    pub top_clusters: usize,
    #[module(skip)]
    pub noise_scale: f64,
}

#[derive(Config, Debug)]
pub struct HierarchicalRouterConfig {
    pub embed_dim: usize,
    pub pool_size: usize,
    pub k_min: usize,
    pub k_max: usize,
    #[config(default = 32)]
    pub num_clusters: usize,
    #[config(default = 4)]
    pub top_clusters: usize,
    #[config(default = 0.1)]
    pub noise_scale: f64,
}

impl HierarchicalRouterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HierarchicalRouter<B> {
        let cluster_size = (self.pool_size + self.num_clusters - 1) / self.num_clusters;

        HierarchicalRouter {
            complexity_layer: LinearConfig::new(self.embed_dim, 1).init(device),
            cluster_scorer: LinearConfig::new(self.embed_dim, self.num_clusters).init(device),
            intra_cluster_scorer: LinearConfig::new(self.embed_dim, cluster_size).init(device),
            num_clusters: self.num_clusters,
            cluster_size,
            pool_size: self.pool_size,
            k_min: self.k_min,
            k_max: self.k_max,
            top_clusters: self.top_clusters,
            noise_scale: self.noise_scale,
        }
    }
}

pub struct HierarchicalRouterOutput<B: Backend> {
    pub complexity: Tensor<B, 2>,
    pub budget: Vec<usize>,
    pub indices: Tensor<B, 2, Int>,
    pub selected_scores: Tensor<B, 2>,
    pub all_scores: Tensor<B, 2>,
    pub routing_probs: Tensor<B, 2>,
}

impl<B: Backend> From<HierarchicalRouterOutput<B>> for RouterOutput<B> {
    fn from(h: HierarchicalRouterOutput<B>) -> Self {
        RouterOutput {
            complexity: h.complexity,
            budget: h.budget,
            indices: h.indices,
            selected_scores: h.selected_scores,
            all_scores: h.all_scores,
            routing_probs: h.routing_probs,
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

impl<B: Backend> HierarchicalRouter<B> {
    pub fn forward(&self, x: Tensor<B, 2>, training: bool) -> HierarchicalRouterOutput<B> {
        let mode = if training {
            RoutingMode::Guided {
                epsilon: self.noise_scale,
            }
        } else {
            RoutingMode::Deterministic
        };
        self.forward_with_mode(x, mode)
    }

    pub fn forward_with_mode(
        &self,
        x: Tensor<B, 2>,
        mode: RoutingMode,
    ) -> HierarchicalRouterOutput<B> {
        let device = x.device();
        let [batch_size, _] = x.dims();

        let complexity = activation::sigmoid(self.complexity_layer.forward(x.clone()));
        let complexity_vals: Vec<f32> = complexity
            .clone()
            .reshape([batch_size])
            .into_data()
            .to_vec()
            .unwrap();

        let budget: Vec<usize> = complexity_vals
            .iter()
            .map(|&c| {
                let c_clamped = c.max(0.01).min(1.0);
                let c_squared = (c_clamped * c_clamped) as f64;
                let k = self.k_min as f64 + (self.k_max - self.k_min) as f64 * c_squared;
                (k.floor() as usize).max(1)
            })
            .collect();

        let cluster_scores = self.cluster_scorer.forward(x.clone());
        let intra_scores = self.intra_cluster_scorer.forward(x);

        let (indices, combined_scores) = match mode {
            RoutingMode::Random => self.random_routing(batch_size, &device),
            RoutingMode::Guided { epsilon } => self.hierarchical_routing(
                batch_size,
                epsilon,
                &device,
                &cluster_scores,
                &intra_scores,
            ),
            RoutingMode::Deterministic => {
                self.hierarchical_routing(batch_size, 0.0, &device, &cluster_scores, &intra_scores)
            }
        };

        let routing_probs = self.compute_routing_probs(&cluster_scores, &intra_scores, &device);
        let selected_scores = self.gather_scores(&combined_scores, &indices);

        HierarchicalRouterOutput {
            complexity,
            budget,
            indices,
            selected_scores,
            all_scores: combined_scores,
            routing_probs,
        }
    }

    fn hierarchical_routing(
        &self,
        batch_size: usize,
        noise_scale: f64,
        device: &B::Device,
        cluster_scores: &Tensor<B, 2>,
        intra_scores: &Tensor<B, 2>,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        let cluster_scores = if noise_scale > 0.0 {
            let noise = Tensor::<B, 2>::random(
                cluster_scores.dims(),
                burn::tensor::Distribution::Normal(0.0, noise_scale),
                device,
            );
            cluster_scores.clone() + noise
        } else {
            cluster_scores.clone()
        };

        let cluster_scores = make_contiguous(cluster_scores);
        let sorted_clusters = cluster_scores.clone().argsort_descending(1);
        let top_cluster_indices = sorted_clusters.slice([0..batch_size, 0..self.top_clusters]);
        let top_cluster_indices = make_contiguous_int(top_cluster_indices);

        let top_cluster_data: Vec<i64> = top_cluster_indices
            .clone()
            .reshape([batch_size * self.top_clusters])
            .into_data()
            .to_vec()
            .unwrap();

        let intra_scores = if noise_scale > 0.0 {
            let noise = Tensor::<B, 2>::random(
                intra_scores.dims(),
                burn::tensor::Distribution::Normal(0.0, noise_scale),
                device,
            );
            intra_scores.clone() + noise
        } else {
            intra_scores.clone()
        };

        let intra_scores = make_contiguous(intra_scores);
        let sorted_intra = intra_scores.clone().argsort_descending(1);

        let params_per_cluster = self.k_max / self.top_clusters;
        let remainder = self.k_max % self.top_clusters;

        let mut all_indices: Vec<i64> = Vec::with_capacity(batch_size * self.k_max);
        let mut all_scores: Vec<f32> = Vec::with_capacity(batch_size * self.pool_size);

        let intra_sorted_data: Vec<i64> = sorted_intra
            .clone()
            .reshape([batch_size * self.cluster_size])
            .into_data()
            .to_vec()
            .unwrap();

        for b in 0..batch_size {
            let mut batch_indices: Vec<i64> = Vec::with_capacity(self.k_max);

            for c in 0..self.top_clusters {
                let cluster_idx = top_cluster_data[b * self.top_clusters + c] as usize;
                let cluster_base = cluster_idx * self.cluster_size;

                let count = if c < remainder {
                    params_per_cluster + 1
                } else {
                    params_per_cluster
                };

                for i in 0..count {
                    if batch_indices.len() >= self.k_max {
                        break;
                    }
                    let intra_idx = intra_sorted_data[b * self.cluster_size + i] as usize;
                    let global_idx = cluster_base + intra_idx;
                    if global_idx < self.pool_size {
                        batch_indices.push(global_idx as i64);
                    }
                }
            }

            while batch_indices.len() < self.k_max {
                batch_indices.push(0);
            }

            all_indices.extend_from_slice(&batch_indices);

            for _ in 0..self.pool_size {
                all_scores.push(0.0);
            }
        }

        let indices = Tensor::<B, 1, Int>::from_ints(all_indices.as_slice(), device)
            .reshape([batch_size, self.k_max]);
        let indices = make_contiguous_int(indices);

        let scores = Tensor::<B, 1>::from_floats(all_scores.as_slice(), device)
            .reshape([batch_size, self.pool_size]);
        let scores = make_contiguous(scores);

        (indices, scores)
    }

    fn random_routing(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        let mut rng = rand::thread_rng();
        let mut indices_vec: Vec<i64> = Vec::with_capacity(batch_size * self.k_max);

        for _ in 0..batch_size {
            let mut sample_indices: Vec<usize> = (0..self.pool_size).collect();
            for i in 0..self.k_max.min(self.pool_size) {
                let j = rng.gen_range(i..self.pool_size);
                sample_indices.swap(i, j);
            }
            for idx in sample_indices.iter().take(self.k_max) {
                indices_vec.push(*idx as i64);
            }
        }

        let indices = Tensor::<B, 1, Int>::from_ints(indices_vec.as_slice(), device)
            .reshape([batch_size, self.k_max]);
        let indices = make_contiguous_int(indices);

        let scores = Tensor::<B, 2>::zeros([batch_size, self.pool_size], device);

        (indices, scores)
    }

    fn compute_routing_probs(
        &self,
        cluster_scores: &Tensor<B, 2>,
        intra_scores: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let [batch_size, _] = cluster_scores.dims();

        let cluster_probs = activation::softmax(cluster_scores.clone(), 1);
        let intra_probs = activation::softmax(intra_scores.clone(), 1);

        let cluster_probs_data: Vec<f32> = cluster_probs
            .clone()
            .reshape([batch_size * self.num_clusters])
            .into_data()
            .to_vec()
            .unwrap();

        let intra_probs_data: Vec<f32> = intra_probs
            .clone()
            .reshape([batch_size * self.cluster_size])
            .into_data()
            .to_vec()
            .unwrap();

        let mut full_probs: Vec<f32> = Vec::with_capacity(batch_size * self.pool_size);

        for b in 0..batch_size {
            for c in 0..self.num_clusters {
                let cluster_prob = cluster_probs_data[b * self.num_clusters + c];
                for i in 0..self.cluster_size {
                    let global_idx = c * self.cluster_size + i;
                    if global_idx < self.pool_size {
                        let intra_prob = intra_probs_data[b * self.cluster_size + i];
                        full_probs.push(cluster_prob * intra_prob);
                    }
                }
            }
            let current_len = full_probs.len() - b * self.pool_size;
            for _ in current_len..self.pool_size {
                full_probs.push(0.0);
            }
        }

        let probs = Tensor::<B, 1>::from_floats(full_probs.as_slice(), device)
            .reshape([batch_size, self.pool_size]);
        make_contiguous(probs)
    }

    fn gather_scores(&self, scores: &Tensor<B, 2>, indices: &Tensor<B, 2, Int>) -> Tensor<B, 2> {
        scores.clone().gather(1, indices.clone())
    }

    pub fn get_effective_k(&self) -> usize {
        self.k_max
    }

    pub fn param_count(&self) -> usize {
        let complexity =
            self.complexity_layer.weight.dims()[0] * self.complexity_layer.weight.dims()[1];
        let cluster = self.cluster_scorer.weight.dims()[0] * self.cluster_scorer.weight.dims()[1];
        let intra =
            self.intra_cluster_scorer.weight.dims()[0] * self.intra_cluster_scorer.weight.dims()[1];
        complexity + cluster + intra
    }
}
