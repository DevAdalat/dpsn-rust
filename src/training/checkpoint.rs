use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError};
use burn::tensor::backend::Backend;
use std::fs;
use std::path::{Path, PathBuf};

use crate::model::dpsn::DPSN;

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
