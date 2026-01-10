use crate::config::{DatasetConfig, DatasetSource, HuggingFaceConfig};
use arrow::array::Array;
use hf_datasets::dataset::StreamingDatasetBuilder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use super::tokenizer::{CharTokenizer, HfTokenizerWrapper, Tokenizer, TokenizerType};

#[derive(Debug, Deserialize, Clone)]
pub struct TextItem {
    #[serde(flatten)]
    pub fields: HashMap<String, serde_json::Value>,
}

const TINY_SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

pub fn download_tiny_shakespeare(data_dir: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file_path = Path::new(data_dir).join("tiny_shakespeare.txt");

    fs::create_dir_all(data_dir)?;

    if file_path.exists() {
        println!("Dataset already exists at {:?}", file_path);
        return Ok(fs::read_to_string(&file_path)?);
    }

    println!("Downloading TinyShakespeare dataset...");
    let response = reqwest::blocking::get(TINY_SHAKESPEARE_URL)?;
    let content = response.text()?;

    fs::write(&file_path, &content)?;
    println!("Downloaded {} bytes to {:?}", content.len(), file_path);

    Ok(content)
}

pub fn download_from_huggingface(
    config: &HuggingFaceConfig,
    data_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    fs::create_dir_all(data_dir)?;

    let filename = config
        .filename
        .clone()
        .unwrap_or_else(|| "dataset.txt".to_string());
    let file_path = Path::new(data_dir).join(&filename);

    if file_path.exists() {
        println!("Dataset already exists at {:?}", file_path);
        return Ok(fs::read_to_string(&file_path)?);
    }

    let url = build_huggingface_url(config);
    println!("Downloading from HuggingFace: {}", url);

    download_with_progress(&url, &file_path)?;

    Ok(fs::read_to_string(&file_path)?)
}

fn build_huggingface_url(config: &HuggingFaceConfig) -> String {
    let revision = config.revision.as_deref().unwrap_or("main");

    if let Some(filename) = &config.filename {
        format!(
            "https://huggingface.co/datasets/{}/resolve/{}/{}",
            config.repo_id, revision, filename
        )
    } else {
        format!(
            "https://huggingface.co/datasets/{}/resolve/{}/data/train.txt",
            config.repo_id, revision
        )
    }
}

fn download_with_progress(url: &str, file_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use indicatif::{ProgressBar, ProgressStyle};

    let client = reqwest::blocking::Client::new();
    let response = client.get(url).send()?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        println!("Downloading (size unknown)...");
        None
    };

    let mut file = File::create(file_path)?;
    let content = response.bytes()?;

    if let Some(pb) = &pb {
        pb.set_position(content.len() as u64);
    }

    file.write_all(&content)?;

    if let Some(pb) = pb {
        pb.finish_with_message("Download complete");
    }

    println!("Downloaded {} bytes to {:?}", content.len(), file_path);

    Ok(())
}

pub fn load_local_file(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    if !Path::new(file_path).exists() {
        return Err(format!("Local file not found: {}", file_path).into());
    }

    println!("Loading local file: {}", file_path);
    let content = fs::read_to_string(file_path)?;
    println!("Loaded {} bytes", content.len());

    Ok(content)
}

pub fn load_from_parquet(
    config: &crate::config::ParquetConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    println!("Loading local parquet file: {}", config.file);
    let file = File::open(&config.file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;

    let mut all_text = String::new();
    let mut processed = 0;

    while let Some(record_batch) = reader.next() {
        let record_batch = record_batch?;
        let schema = record_batch.schema();

        let idx = match schema.index_of(&config.column) {
            Ok(i) => i,
            Err(_) => {
                return Err(format!("Column '{}' not found in parquet file", config.column).into())
            }
        };

        let column = record_batch.column(idx);
        let string_array = column
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or(format!("Column '{}' is not a text column", config.column))?;

        for i in 0..string_array.len() {
            if !string_array.is_null(i) {
                all_text.push_str(string_array.value(i));
                all_text.push('\n');
                processed += 1;
            }
        }
    }

    println!("Loaded {} bytes from {} items", all_text.len(), processed);
    Ok(all_text)
}

pub fn load_from_burn_dataset(
    hf_config: &HuggingFaceConfig,
    max_items: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    println!(
        "Loading dataset from HuggingFace using hf-datasets streaming: {}",
        hf_config.repo_id
    );

    let column_name = hf_config
        .text_column
        .as_deref()
        .ok_or("text_column is required for streaming dataset")?;

    if let Some(subset) = &hf_config.subset {
        println!("Warning: subset '{}' specified but hf-datasets v0.1.0 doesn't support config/subset selection yet", subset);
    }

    let split = hf_config.split.as_deref().unwrap_or("train");
    println!("Dataset split: {}, text column: {}", split, column_name);

    // Block on async runtime to stream dataset
    let runtime = tokio::runtime::Runtime::new()?;
    let all_text = runtime.block_on(async {
        println!(
            "Building streaming dataset for repo: {}, split: {}",
            &hf_config.repo_id, split
        );
        let mut dataset = StreamingDatasetBuilder::new(&hf_config.repo_id)
            .split(split)
            .build()
            .await
            .map_err(|e| format!("Failed to build streaming dataset: {}", e))?;

        println!(
            "Dataset built successfully. File URLs: {:?}",
            dataset.file_urls()
        );

        let mut all_text = String::new();
        let max_limit = max_items.unwrap_or(10000);
        let mut processed = 0;
        let mut skipped = 0;

        println!("Starting to stream examples...");
        while let Some(example_result) = dataset.next().await {
            if processed >= max_limit {
                println!("Reached max_limit of {} items", max_limit);
                break;
            }

            let example =
                example_result.map_err(|e| format!("Failed to fetch dataset item: {}", e))?;

            if processed == 0 {
                println!(
                    "First example fields: {:?}",
                    example.keys().collect::<Vec<_>>()
                );
            }

            if let Some(value) = example.get(column_name) {
                let text = match value {
                    serde_json::Value::String(s) => s.as_str(),
                    _ => {
                        if processed < 5 {
                            println!(
                                "Warning: Column '{}' is not a string. Type: {:?}",
                                column_name, value
                            );
                        }
                        skipped += 1;
                        continue;
                    }
                };
                all_text.push_str(text);
                all_text.push('\n');
                processed += 1;
            } else {
                if processed < 5 {
                    println!(
                        "Warning: Column '{}' not found in example. Available: {:?}",
                        column_name,
                        example.keys().collect::<Vec<_>>()
                    );
                }
                skipped += 1;
            }

            if processed % 1000 == 0 {
                println!("Streamed and processed {} items", processed);
            }
        }

        println!(
            "Streaming complete. Processed: {}, Skipped: {}",
            processed, skipped
        );
        println!(
            "Loaded {} bytes of text data from {} items",
            all_text.len(),
            processed
        );
        Ok::<String, Box<dyn std::error::Error>>(all_text)
    })?;

    Ok(all_text)
}

pub fn load_dataset_from_config(
    config: &DatasetConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    match config.source {
        DatasetSource::TinyShakespeare => download_tiny_shakespeare(&config.data_dir),
        DatasetSource::HuggingFace => {
            let hf_config = config
                .huggingface
                .as_ref()
                .ok_or("HuggingFace config required for huggingface source")?;
            download_from_huggingface(hf_config, &config.data_dir)
        }
        DatasetSource::BurnDataset => {
            let hf_config = config
                .huggingface
                .as_ref()
                .ok_or("HuggingFace config required for burn-dataset source")?;
            load_from_burn_dataset(hf_config, config.max_items)
        }
        DatasetSource::LocalFile => {
            let file_path = config
                .local_file
                .as_ref()
                .ok_or("local_file path required for localfile source")?;
            load_local_file(file_path)
        }
        DatasetSource::LocalParquet => {
            let parquet_config = config
                .parquet
                .as_ref()
                .ok_or("parquet config required for localparquet source")?;
            load_from_parquet(parquet_config)
        }
    }
}

#[derive(Clone)]
pub struct CharDataset {
    pub tokens: Vec<usize>,
    pub context_length: usize,
    pub tokenizer: TokenizerType,
}

impl CharDataset {
    pub fn new(text: &str, context_length: usize) -> Self {
        let tokenizer = TokenizerType::Char(CharTokenizer::from_text(text));
        let tokens = tokenizer.encode(text);

        CharDataset {
            tokens,
            context_length,
            tokenizer,
        }
    }

    pub fn with_tokenizer(
        text: &str,
        context_length: usize,
        tokenizer_path: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = if let Some(path) = tokenizer_path {
            println!("Loading HuggingFace tokenizer from: {}", path);
            let hf_tokenizer = HfTokenizerWrapper::from_file(path)?;
            TokenizerType::HuggingFace(hf_tokenizer)
        } else {
            TokenizerType::Char(CharTokenizer::from_text(text))
        };

        let tokens = tokenizer.encode(text);

        Ok(CharDataset {
            tokens,
            context_length,
            tokenizer,
        })
    }

    pub fn from_config(
        config: &DatasetConfig,
        context_length: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let text = load_dataset_from_config(config)?;
        Self::with_tokenizer(&text, context_length, config.tokenizer_path.as_deref())
    }

    pub fn len(&self) -> usize {
        if self.tokens.len() <= self.context_length {
            0
        } else {
            self.tokens.len() - self.context_length
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_sample(&self, idx: usize) -> Option<(Vec<usize>, Vec<usize>)> {
        if idx >= self.len() {
            return None;
        }

        let input = self.tokens[idx..idx + self.context_length].to_vec();
        let target = self.tokens[idx + 1..idx + self.context_length + 1].to_vec();

        Some((input, target))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    pub fn get_random_batch(&self, batch_size: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut inputs = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = rng.gen_range(0..self.len());
            if let Some((input, target)) = self.get_sample(idx) {
                inputs.push(input);
                targets.push(target);
            }
        }

        (inputs, targets)
    }
}

pub struct DataLoader<'a> {
    dataset: &'a CharDataset,
    indices: Vec<usize>,
    position: usize,
    batch_size: usize,
    shuffle: bool,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a CharDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        DataLoader {
            dataset,
            indices,
            position: 0,
            batch_size,
            shuffle,
        }
    }

    pub fn next_batch(&mut self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        if self.position + self.batch_size > self.indices.len() {
            self.position = 0;
            if self.shuffle {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                self.indices.shuffle(&mut rng);
            }
        }

        let batch_indices = &self.indices[self.position..self.position + self.batch_size];
        self.position += self.batch_size;

        let results: Vec<(Vec<usize>, Vec<usize>)> = batch_indices
            .par_iter()
            .map(|&idx| self.dataset.get_sample(idx))
            .flatten()
            .collect();

        let (inputs, targets): (Vec<Vec<usize>>, Vec<Vec<usize>>) = results.into_iter().unzip();
        (inputs, targets)
    }
}
