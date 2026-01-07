use burn::config::Config;

#[derive(Config, Debug)]
pub struct DPSNConfig {
    #[config(default = 65)]
    pub vocab_size: usize,

    #[config(default = 64)]
    pub embed_dim: usize,

    #[config(default = 20000)]
    pub pool_size: usize,

    #[config(default = 4)]
    pub num_heads: usize,

    #[config(default = 1)]
    pub recurrence_steps: usize,

    #[config(default = 64)]
    pub context_length: usize,

    pub router: RouterTypeConfig,
}

#[derive(Config, Debug)]
pub enum RouterTypeConfig {
    Standard(StandardRouterConfig),
    Hierarchical(HierarchicalRouterConfig),
}

#[derive(Config, Debug)]
pub struct StandardRouterConfig {
    #[config(default = 100)]
    pub k_min: usize,

    #[config(default = 5000)]
    pub k_max: usize,

    #[config(default = 128)]
    pub hidden_dim: usize,

    #[config(default = 0.1)]
    pub exploration_noise: f64,
}

#[derive(Config, Debug)]
pub struct HierarchicalRouterConfig {
    #[config(default = 100)]
    pub k_min: usize,

    #[config(default = 5000)]
    pub k_max: usize,

    #[config(default = 32)]
    pub num_clusters: usize,

    #[config(default = 4)]
    pub top_clusters: usize,

    #[config(default = 0.1)]
    pub exploration_noise: f64,
}
