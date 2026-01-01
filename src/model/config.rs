use burn::config::Config;

#[derive(Config, Debug)]
pub struct DPSNConfig {
    #[config(default = 65)]
    pub vocab_size: usize,

    #[config(default = 64)]
    pub embed_dim: usize,

    #[config(default = 20000)]
    pub pool_size: usize,

    #[config(default = 100)]
    pub k_min: usize,

    #[config(default = 5000)]
    pub k_max: usize,

    #[config(default = 128)]
    pub router_hidden_dim: usize,

    #[config(default = 64)]
    pub context_length: usize,

    #[config(default = 0.1)]
    pub exploration_noise: f64,
}
