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
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_context_length")]
    pub context_length: usize,
    #[serde(default = "default_recurrence_steps")]
    pub recurrence_steps: usize,
    #[serde(default)]
    pub router: RouterSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RouterSettings {
    Standard(StandardRouterSettings),
    Hierarchical(HierarchicalRouterSettings),
}

impl Default for RouterSettings {
    fn default() -> Self {
        RouterSettings::Standard(StandardRouterSettings::default())
    }
}

impl RouterSettings {
    pub fn as_standard(&self) -> Option<&StandardRouterSettings> {
        match self {
            RouterSettings::Standard(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_hierarchical(&self) -> Option<&HierarchicalRouterSettings> {
        match self {
            RouterSettings::Hierarchical(h) => Some(h),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardRouterSettings {
    #[serde(default = "default_k_min")]
    pub k_min: usize,
    #[serde(default = "default_k_max")]
    pub k_max: usize,
    #[serde(default = "default_router_hidden_dim")]
    pub hidden_dim: usize,
    #[serde(default = "default_exploration_noise")]
    pub exploration_noise: f64,
}

impl Default for StandardRouterSettings {
    fn default() -> Self {
        StandardRouterSettings {
            k_min: default_k_min(),
            k_max: default_k_max(),
            hidden_dim: default_router_hidden_dim(),
            exploration_noise: default_exploration_noise(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalRouterSettings {
    #[serde(default = "default_k_min")]
    pub k_min: usize,
    #[serde(default = "default_k_max")]
    pub k_max: usize,
    #[serde(default = "default_num_clusters")]
    pub num_clusters: usize,
    #[serde(default = "default_top_clusters")]
    pub top_clusters: usize,
    #[serde(default = "default_exploration_noise")]
    pub exploration_noise: f64,
}

impl Default for HierarchicalRouterSettings {
    fn default() -> Self {
        HierarchicalRouterSettings {
            k_min: default_k_min(),
            k_max: default_k_max(),
            num_clusters: default_num_clusters(),
            top_clusters: default_top_clusters(),
            exploration_noise: default_exploration_noise(),
        }
    }
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
    #[serde(default)]
    pub parquet: Option<ParquetConfig>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    #[serde(default)]
    pub max_items: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DatasetSource {
    #[default]
    TinyShakespeare,
    HuggingFace,
    LocalFile,
    LocalParquet,
    BurnDataset,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetConfig {
    pub file: String,
    pub column: String,
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
    #[serde(default)]
    pub subset: Option<String>,
    #[serde(default)]
    pub trust_remote_code: Option<bool>,
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
fn default_recurrence_steps() -> usize {
    1
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
fn default_num_heads() -> usize {
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
                num_heads: 4,
                context_length: 64,
                recurrence_steps: 1,
                router: RouterSettings::Standard(StandardRouterSettings::default()),
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
                parquet: None,
                tokenizer_path: None,
                max_items: None,
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
                num_heads: 4,
                context_length: 32,
                recurrence_steps: 1,
                router: RouterSettings::Standard(StandardRouterSettings {
                    k_min: 20,
                    k_max: 200,
                    hidden_dim: 64,
                    exploration_noise: 0.1,
                }),
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
                parquet: None,
                tokenizer_path: None,
                max_items: None,
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
                num_heads: 8,
                context_length: 128,
                recurrence_steps: 4,
                router: RouterSettings::Hierarchical(HierarchicalRouterSettings {
                    k_min: 50,
                    k_max: 1000,
                    num_clusters: 64,
                    top_clusters: 8,
                    exploration_noise: 0.1,
                }),
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
                    subset: None,
                    trust_remote_code: None,
                }),
                local_file: None,
                parquet: None,
                tokenizer_path: None,
                max_items: None,
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
