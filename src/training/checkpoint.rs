use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError};
use burn::tensor::backend::Backend;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::model::dpsn::{HierarchicalDPSN, DPSN};
use crate::model::offloaded_dpsn::OffloadedDPSNGpuPart;

pub type DefaultRecorder = NamedMpkFileRecorder<FullPrecisionSettings>;

pub fn default_recorder() -> DefaultRecorder {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

pub fn save_checkpoint<B: Backend>(
    model: &DPSN<B>,
    checkpoint_dir: &Path,
    step: usize,
) -> Result<PathBuf, RecorderError> {
    fs::create_dir_all(checkpoint_dir).map_err(|e| RecorderError::Unknown(e.to_string()))?;

    let filename = format!("dpsn_step_{}", step);
    let path = checkpoint_dir.join(&filename);

    let recorder = default_recorder();
    model.clone().save_file(&path, &recorder)?;

    println!("Saved checkpoint: {}.mpk", path.display());
    Ok(path)
}

pub fn load_checkpoint<B: Backend>(
    model: DPSN<B>,
    checkpoint_path: &Path,
    device: &B::Device,
) -> Result<DPSN<B>, RecorderError> {
    let recorder = default_recorder();
    let path_without_ext = checkpoint_path.with_extension("");
    model.load_file(path_without_ext, &recorder, device)
}

pub fn save_hierarchical_checkpoint<B: Backend>(
    model: &HierarchicalDPSN<B>,
    checkpoint_dir: &Path,
    step: usize,
) -> Result<PathBuf, RecorderError> {
    fs::create_dir_all(checkpoint_dir).map_err(|e| RecorderError::Unknown(e.to_string()))?;

    let filename = format!("hdpsn_step_{}", step);
    let path = checkpoint_dir.join(&filename);

    let recorder = default_recorder();
    model.clone().save_file(&path, &recorder)?;

    println!("Saved hierarchical checkpoint: {}.mpk", path.display());
    Ok(path)
}

pub fn load_hierarchical_checkpoint<B: Backend>(
    model: HierarchicalDPSN<B>,
    checkpoint_path: &Path,
    device: &B::Device,
) -> Result<HierarchicalDPSN<B>, RecorderError> {
    let recorder = default_recorder();
    let path_without_ext = checkpoint_path.with_extension("");
    model.load_file(path_without_ext, &recorder, device)
}

pub fn save_offloaded_checkpoint<B: Backend>(
    gpu_part: &OffloadedDPSNGpuPart<B>,
    pool_data: &[f32],
    pool_size: usize,
    embed_dim: usize,
    checkpoint_dir: &Path,
    step: usize,
) -> Result<PathBuf, RecorderError> {
    fs::create_dir_all(checkpoint_dir).map_err(|e| RecorderError::Unknown(e.to_string()))?;

    let gpu_filename = format!("offloaded_gpu_step_{}", step);
    let gpu_path = checkpoint_dir.join(&gpu_filename);
    let recorder = default_recorder();
    gpu_part.clone().save_file(&gpu_path, &recorder)?;

    let pool_filename = format!("offloaded_pool_step_{}.bin", step);
    let pool_path = checkpoint_dir.join(&pool_filename);

    let mut file = BufWriter::new(
        File::create(&pool_path).map_err(|e| RecorderError::Unknown(e.to_string()))?,
    );

    file.write_all(&(pool_size as u64).to_le_bytes())
        .map_err(|e| RecorderError::Unknown(e.to_string()))?;
    file.write_all(&(embed_dim as u64).to_le_bytes())
        .map_err(|e| RecorderError::Unknown(e.to_string()))?;

    for &val in pool_data {
        file.write_all(&val.to_le_bytes())
            .map_err(|e| RecorderError::Unknown(e.to_string()))?;
    }
    file.flush()
        .map_err(|e| RecorderError::Unknown(e.to_string()))?;

    println!(
        "Saved offloaded checkpoint: gpu={}.mpk, pool={}",
        gpu_path.display(),
        pool_path.display()
    );
    Ok(gpu_path)
}

pub fn load_offloaded_pool(pool_path: &Path) -> Result<(Vec<f32>, usize, usize), RecorderError> {
    let mut file =
        BufReader::new(File::open(pool_path).map_err(|e| RecorderError::Unknown(e.to_string()))?);

    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)
        .map_err(|e| RecorderError::Unknown(e.to_string()))?;
    let pool_size = u64::from_le_bytes(size_buf) as usize;

    file.read_exact(&mut size_buf)
        .map_err(|e| RecorderError::Unknown(e.to_string()))?;
    let embed_dim = u64::from_le_bytes(size_buf) as usize;

    let mut pool_data = Vec::with_capacity(pool_size * embed_dim);
    let mut val_buf = [0u8; 4];
    for _ in 0..(pool_size * embed_dim) {
        file.read_exact(&mut val_buf)
            .map_err(|e| RecorderError::Unknown(e.to_string()))?;
        pool_data.push(f32::from_le_bytes(val_buf));
    }

    Ok((pool_data, pool_size, embed_dim))
}

pub fn load_offloaded_gpu_part<B: Backend>(
    gpu_part: OffloadedDPSNGpuPart<B>,
    checkpoint_path: &Path,
    device: &B::Device,
) -> Result<OffloadedDPSNGpuPart<B>, RecorderError> {
    let recorder = default_recorder();
    let path_without_ext = checkpoint_path.with_extension("");
    gpu_part.load_file(path_without_ext, &recorder, device)
}

pub fn find_latest_checkpoint(checkpoint_dir: &Path) -> Option<PathBuf> {
    if !checkpoint_dir.exists() {
        return None;
    }

    let mut checkpoints: Vec<(usize, PathBuf)> = fs::read_dir(checkpoint_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let filename = path.file_stem()?.to_str()?;
            if filename.starts_with("dpsn_step_") {
                let step_str = filename.strip_prefix("dpsn_step_")?;
                let step: usize = step_str.parse().ok()?;
                Some((step, path))
            } else {
                None
            }
        })
        .collect();

    checkpoints.sort_by_key(|(step, _)| *step);
    checkpoints.last().map(|(_, path)| path.clone())
}
