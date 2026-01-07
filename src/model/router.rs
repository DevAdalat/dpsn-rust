use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use rand::Rng;

#[derive(Module, Debug)]
pub struct Router<B: Backend> {
    complexity_layer: Linear<B>,
    relevance_layer1: Linear<B>,
    layer_norm: LayerNorm<B>,
    relevance_layer2: Linear<B>,
    #[module(skip)]
    pub k_min: usize,
    #[module(skip)]
    pub k_max: usize,
    #[module(skip)]
    pub pool_size: usize,
    #[module(skip)]
    pub noise_scale: f64,
}

#[derive(Config, Debug)]
pub struct RouterConfig {
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub pool_size: usize,
    pub k_min: usize,
    pub k_max: usize,
    #[config(default = 0.1)]
    pub noise_scale: f64,
}

impl RouterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Router<B> {
        Router {
            complexity_layer: LinearConfig::new(self.embed_dim, 1).init(device),
            relevance_layer1: LinearConfig::new(self.embed_dim, self.hidden_dim).init(device),
            layer_norm: LayerNormConfig::new(self.hidden_dim).init(device),
            relevance_layer2: LinearConfig::new(self.hidden_dim, self.pool_size).init(device),
            k_min: self.k_min,
            k_max: self.k_max,
            pool_size: self.pool_size,
            noise_scale: self.noise_scale,
        }
    }
}

#[derive(Clone)]
pub struct RouterOutput<B: Backend> {
    pub complexity: Tensor<B, 2>,
    pub budget: Vec<usize>,
    pub indices: Tensor<B, 2, Int>,
    pub selected_scores: Tensor<B, 2>,
    pub all_scores: Tensor<B, 2>,
    pub routing_probs: Tensor<B, 2>,
}

fn make_contiguous<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let shape = tensor.dims();
    tensor.reshape(shape)
}

fn make_contiguous_int<B: Backend, const D: usize>(tensor: Tensor<B, D, Int>) -> Tensor<B, D, Int> {
    let shape = tensor.dims();
    tensor.reshape(shape)
}

#[derive(Debug, Clone, Copy)]
pub enum RoutingMode {
    Random,
    Guided { epsilon: f64 },
    Deterministic,
}

impl<B: Backend> Router<B> {
    pub fn forward(&self, x: Tensor<B, 2>, training: bool) -> RouterOutput<B> {
        let mode = if training {
            RoutingMode::Guided {
                epsilon: self.noise_scale,
            }
        } else {
            RoutingMode::Deterministic
        };
        self.forward_with_mode(x, mode)
    }

    pub fn forward_with_mode(&self, x: Tensor<B, 2>, mode: RoutingMode) -> RouterOutput<B> {
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
                // Clamp complexity to prevent budget=0 which causes NaN
                let c_clamped = c.max(0.01).min(1.0);
                let c_squared = (c_clamped * c_clamped) as f64;
                let k = self.k_min as f64 + (self.k_max - self.k_min) as f64 * c_squared;
                // Ensure minimum budget of 1 to prevent empty tensor operations
                (k.floor() as usize).max(1)
            })
            .collect();

        let hidden = activation::relu(self.relevance_layer1.forward(x));
        let hidden_norm = self.layer_norm.forward(hidden);
        let scores = self.relevance_layer2.forward(hidden_norm);

        let routing_probs = activation::softmax(scores.clone(), 1);

        let (final_indices, final_scores) = match mode {
            RoutingMode::Random => self.random_routing(batch_size, &device, &scores),
            RoutingMode::Guided { epsilon } => {
                self.epsilon_greedy_routing(batch_size, epsilon, &device, &scores)
            }
            RoutingMode::Deterministic => self.deterministic_routing(batch_size, &scores),
        };

        let selected_scores = scores.clone().gather(1, final_indices.clone());

        RouterOutput {
            complexity,
            budget,
            indices: final_indices,
            selected_scores,
            all_scores: final_scores,
            routing_probs,
        }
    }

    fn random_routing(
        &self,
        batch_size: usize,
        device: &B::Device,
        scores: &Tensor<B, 2>,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        let mut rng = rand::thread_rng();
        let mut indices_vec: Vec<i64> = Vec::with_capacity(batch_size * self.k_max);

        for _ in 0..batch_size {
            let mut sample_indices: Vec<usize> = (0..self.pool_size).collect();
            for i in 0..self.k_max {
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

        (indices, scores.clone())
    }

    fn epsilon_greedy_routing(
        &self,
        batch_size: usize,
        epsilon: f64,
        device: &B::Device,
        scores: &Tensor<B, 2>,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < epsilon {
            return self.random_routing(batch_size, device, scores);
        }

        let noise = Tensor::<B, 2>::random(
            scores.dims(),
            burn::tensor::Distribution::Normal(0.0, self.noise_scale),
            device,
        );
        let noisy_scores = scores.clone() + noise;
        let noisy_scores = make_contiguous(noisy_scores);

        let sorted_indices = noisy_scores.clone().argsort_descending(1);
        let top_k_indices = sorted_indices.slice([0..batch_size, 0..self.k_max]);
        let top_k_indices = make_contiguous_int(top_k_indices);

        (top_k_indices, noisy_scores)
    }

    fn deterministic_routing(
        &self,
        batch_size: usize,
        scores: &Tensor<B, 2>,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        let scores = make_contiguous(scores.clone());
        let sorted_indices = scores.clone().argsort_descending(1);
        let top_k_indices = sorted_indices.slice([0..batch_size, 0..self.k_max]);
        let top_k_indices = make_contiguous_int(top_k_indices);

        (top_k_indices, scores)
    }

    pub fn get_effective_k(&self) -> usize {
        self.k_max
    }
}
