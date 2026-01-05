use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use super::tokenizer::CharTokenizer;
use crate::config::{DatasetConfig, DatasetSource, HuggingFaceConfig};

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
        DatasetSource::LocalFile => {
            let file_path = config
                .local_file
                .as_ref()
                .ok_or("local_file path required for localfile source")?;
            load_local_file(file_path)
        }
    }
}

#[derive(Debug, Clone)]
pub struct CharDataset {
    pub tokens: Vec<usize>,
    pub context_length: usize,
    pub tokenizer: CharTokenizer,
}

impl CharDataset {
    pub fn new(text: &str, context_length: usize) -> Self {
        let tokenizer = CharTokenizer::from_text(text);
        let tokens = tokenizer.encode(text);

        CharDataset {
            tokens,
            context_length,
            tokenizer,
        }
    }

    pub fn from_config(
        config: &DatasetConfig,
        context_length: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let text = load_dataset_from_config(config)?;
        Ok(Self::new(&text, context_length))
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
        self.tokenizer.vocab_size
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
        let mut inputs = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            if self.position >= self.indices.len() {
                self.position = 0;
                if self.shuffle {
                    use rand::seq::SliceRandom;
                    let mut rng = rand::thread_rng();
                    self.indices.shuffle(&mut rng);
                }
            }

            let idx = self.indices[self.position];
            if let Some((input, target)) = self.dataset.get_sample(idx) {
                inputs.push(input);
                targets.push(target);
            }
            self.position += 1;
        }

        (inputs, targets)
    }
}
