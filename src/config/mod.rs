use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub mod device;
pub use device::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullConfig {
    pub model: ModelConfig,
    pub training: TrainingSettings,
    pub dataset: DatasetConfig,
    pub inference: InferenceConfig,
    #[serde(default)]
    pub backend: BackendConfig,
    #[serde(default)]
    pub curriculum: CurriculumSettings,
    #[serde(default)]
    pub device_placement: Option<DevicePlacement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: Option<usize>,
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
    #[serde(default = "default_k_min")]
    pub k_min: usize,
    #[serde(default = "default_k_max")]
    pub k_max: usize,
    #[serde(default = "default_router_hidden_dim")]
    pub router_hidden_dim: usize,
    #[serde(default = "default_context_length")]
    pub context_length: usize,
    #[serde(default = "default_exploration_noise")]
    pub exploration_noise: f64,
    #[serde(default)]
    pub use_hierarchical_router: bool,
    #[serde(default = "default_num_clusters")]
    pub num_clusters: usize,
    #[serde(default = "default_top_clusters")]
    pub top_clusters: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSettings {
    #[serde(default)]
    pub num_steps: Option<usize>,
    #[serde(default)]
    pub num_epochs: Option<usize>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_log_interval")]
    pub log_interval: usize,
    #[serde(default = "default_save_interval")]
    pub save_interval: usize,
    #[serde(default)]
    pub checkpoint_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    #[serde(default = "default_dataset_source")]
    pub source: DatasetSource,
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default)]
    pub huggingface: Option<HuggingFaceConfig>,
    #[serde(default)]
    pub local_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DatasetSource {
    #[default]
    TinyShakespeare,
    HuggingFace,
    LocalFile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    pub repo_id: String,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub split: Option<String>,
    #[serde(default)]
    pub text_column: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_prompt")]
    pub default_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendConfig {
    #[serde(default = "default_backend_type")]
    pub backend_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumSettings {
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: usize,
    #[serde(default = "default_specialization_steps")]
    pub specialization_steps: usize,
    #[serde(default = "default_warmup_epsilon")]
    pub warmup_epsilon: f64,
    #[serde(default = "default_specialization_epsilon_start")]
    pub specialization_epsilon_start: f64,
    #[serde(default = "default_specialization_epsilon_end")]
    pub specialization_epsilon_end: f64,
    #[serde(default = "default_maturity_epsilon")]
    pub maturity_epsilon: f64,
    #[serde(default = "default_balance_weight")]
    pub balance_weight: f64,
    #[serde(default = "default_efficiency_weight")]
    pub efficiency_weight: f64,
    #[serde(default = "default_z_loss_weight")]
    pub z_loss_weight: f64,
}

impl Default for CurriculumSettings {
    fn default() -> Self {
        CurriculumSettings {
            warmup_steps: 100,
            specialization_steps: 400,
            warmup_epsilon: 1.0,
            specialization_epsilon_start: 0.3,
            specialization_epsilon_end: 0.05,
            maturity_epsilon: 0.01,
            balance_weight: 0.1,
            efficiency_weight: 0.1,
            z_loss_weight: 0.001,
        }
    }
}

fn default_vocab_size() -> Option<usize> {
    None
}
fn default_embed_dim() -> usize {
    64
}
fn default_pool_size() -> usize {
    20000
}
fn default_k_min() -> usize {
    100
}
fn default_k_max() -> usize {
    5000
}
fn default_router_hidden_dim() -> usize {
    128
}
fn default_context_length() -> usize {
    64
}
fn default_exploration_noise() -> f64 {
    0.1
}
fn default_num_clusters() -> usize {
    32
}
fn default_top_clusters() -> usize {
    4
}
fn default_num_steps() -> usize {
    500
}
fn default_batch_size() -> usize {
    32
}
fn default_learning_rate() -> f64 {
    0.001
}
fn default_log_interval() -> usize {
    50
}
fn default_save_interval() -> usize {
    100
}
fn default_dataset_source() -> DatasetSource {
    DatasetSource::TinyShakespeare
}
fn default_data_dir() -> String {
    "data".to_string()
}
fn default_max_tokens() -> usize {
    200
}
fn default_temperature() -> f64 {
    0.8
}
fn default_prompt() -> String {
    "The ".to_string()
}
fn default_backend_type() -> String {
    "ndarray".to_string()
}
fn default_warmup_steps() -> usize {
    100
}
fn default_specialization_steps() -> usize {
    400
}
fn default_warmup_epsilon() -> f64 {
    1.0
}
fn default_specialization_epsilon_start() -> f64 {
    0.3
}
fn default_specialization_epsilon_end() -> f64 {
    0.05
}
fn default_maturity_epsilon() -> f64 {
    0.01
}
fn default_balance_weight() -> f64 {
    0.1
}
fn default_efficiency_weight() -> f64 {
    0.1
}
fn default_z_loss_weight() -> f64 {
    0.001
}

impl FullConfig {
    pub fn load_from_yaml<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: FullConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn save_to_yaml<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_yaml::to_string(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn default_config() -> Self {
        FullConfig {
            model: ModelConfig {
                vocab_size: None,
                embed_dim: 64,
                pool_size: 20000,
                k_min: 100,
                k_max: 5000,
                router_hidden_dim: 128,
                context_length: 64,
                exploration_noise: 0.1,
                use_hierarchical_router: false,
                num_clusters: 32,
                top_clusters: 4,
            },
            training: TrainingSettings {
                num_steps: Some(500),
                num_epochs: None,
                batch_size: 32,
                learning_rate: 0.001,
                log_interval: 50,
                save_interval: 100,
                checkpoint_dir: Some("checkpoints".to_string()),
            },
            dataset: DatasetConfig {
                source: DatasetSource::TinyShakespeare,
                data_dir: "data".to_string(),
                huggingface: None,
                local_file: None,
            },
            inference: InferenceConfig {
                max_tokens: 200,
                temperature: 0.8,
                default_prompt: "The ".to_string(),
            },
            backend: BackendConfig {
                backend_type: "ndarray".to_string(),
            },
            curriculum: CurriculumSettings::default(),
            device_placement: None,
        }
    }

    pub fn demo_config() -> Self {
        FullConfig {
            model: ModelConfig {
                vocab_size: None,
                embed_dim: 32,
                pool_size: 2000,
                k_min: 20,
                k_max: 200,
                router_hidden_dim: 64,
                context_length: 32,
                exploration_noise: 0.1,
                use_hierarchical_router: false,
                num_clusters: 32,
                top_clusters: 4,
            },
            training: TrainingSettings {
                num_steps: Some(100),
                num_epochs: None,
                batch_size: 16,
                learning_rate: 0.001,
                log_interval: 20,
                save_interval: 50,
                checkpoint_dir: None,
            },
            dataset: DatasetConfig {
                source: DatasetSource::TinyShakespeare,
                data_dir: "data".to_string(),
                huggingface: None,
                local_file: None,
            },
            inference: InferenceConfig {
                max_tokens: 50,
                temperature: 0.9,
                default_prompt: "ROMEO:".to_string(),
            },
            backend: BackendConfig {
                backend_type: "ndarray".to_string(),
            },
            curriculum: CurriculumSettings {
                warmup_steps: 20,
                specialization_steps: 60,
                ..Default::default()
            },
            device_placement: None,
        }
    }

    pub fn huggingface_example() -> Self {
        FullConfig {
            model: ModelConfig {
                vocab_size: None,
                embed_dim: 64,
                pool_size: 10000,
                k_min: 50,
                k_max: 1000,
                router_hidden_dim: 128,
                context_length: 128,
                exploration_noise: 0.1,
                use_hierarchical_router: true,
                num_clusters: 64,
                top_clusters: 8,
            },
            training: TrainingSettings {
                num_steps: Some(1000),
                num_epochs: None,
                batch_size: 32,
                learning_rate: 0.001,
                log_interval: 50,
                save_interval: 200,
                checkpoint_dir: Some("checkpoints".to_string()),
            },
            dataset: DatasetConfig {
                source: DatasetSource::HuggingFace,
                data_dir: "data".to_string(),
                huggingface: Some(HuggingFaceConfig {
                    repo_id: "roneneldan/TinyStories".to_string(),
                    filename: Some("TinyStoriesV2-GPT4-train.txt".to_string()),
                    split: None,
                    text_column: None,
                    revision: None,
                }),
                local_file: None,
            },
            inference: InferenceConfig {
                max_tokens: 200,
                temperature: 0.8,
                default_prompt: "Once upon a time".to_string(),
            },
            backend: BackendConfig {
                backend_type: "wgpu".to_string(),
            },
            curriculum: CurriculumSettings {
                warmup_steps: 200,
                specialization_steps: 600,
                ..Default::default()
            },
            device_placement: Some(DevicePlacement::new_offloaded()),
        }
    }
}
