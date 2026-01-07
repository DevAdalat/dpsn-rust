use burn::config::Config;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::path::PathBuf;

const NAN_PATIENCE: usize = 3;

use super::checkpoint::{save_checkpoint, save_hierarchical_checkpoint};
use super::curriculum::{CurriculumConfig, TrainingPhase};
use crate::data::batcher::DPSNBatcher;
use crate::data::dataset::CharDataset;
use crate::data::prefetcher::DataPrefetcher;
use crate::model::config::{HierarchicalRouterConfig, StandardRouterConfig};
use crate::model::dpsn::{DeviceLocation, HierarchicalDPSN, DPSN};
use crate::model::router::{RouterOutput, RoutingMode};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub num_steps: Option<usize>,

    pub num_epochs: Option<usize>,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 1e-3)]
    pub learning_rate: f64,

    #[config(default = 50)]
    pub log_interval: usize,

    #[config(default = 100)]
    pub save_interval: usize,

    pub checkpoint_dir: Option<PathBuf>,

    #[config(default = 1)]
    pub recurrence_steps: usize,
}

pub struct AuxiliaryLosses<B: Backend> {
    pub balance_loss: Tensor<B, 1>,
    pub efficiency_loss: Tensor<B, 1>,
    pub z_loss: Tensor<B, 1>,
}

fn compute_auxiliary_losses<B: Backend>(
    router_outputs: &[RouterOutput<B>],
    k_max: usize,
    pool_size: usize,
) -> AuxiliaryLosses<B> {
    if router_outputs.is_empty() {
        // Should not happen, but safe fallback
        let device = B::Device::default();
        return AuxiliaryLosses {
            balance_loss: Tensor::zeros([1], &device),
            efficiency_loss: Tensor::zeros([1], &device),
            z_loss: Tensor::zeros([1], &device),
        };
    }

    let device = router_outputs[0].indices.device();

    // Accumulate losses across all recurrence steps
    let mut total_balance_loss = Tensor::<B, 1>::zeros([1], &device);
    let mut total_efficiency_loss = Tensor::<B, 1>::zeros([1], &device);
    let mut total_z_loss = Tensor::<B, 1>::zeros([1], &device);

    for output in router_outputs {
        let routing_probs = &output.routing_probs;
        let all_scores = &output.all_scores;
        let budget = &output.budget;

        let mean_probs: Tensor<B, 1> = routing_probs.clone().mean_dim(0).squeeze();
        let balance_loss = mean_probs.powf_scalar(2.0).sum() * (pool_size as f32);
        total_balance_loss = total_balance_loss + balance_loss.reshape([1]);

        let avg_budget: f32 = budget.iter().map(|&b| b as f32).sum::<f32>() / budget.len() as f32;
        let efficiency_ratio = avg_budget / k_max as f32;
        let efficiency_loss = Tensor::<B, 1>::from_floats([efficiency_ratio], &device);
        total_efficiency_loss = total_efficiency_loss + efficiency_loss;

        let log_sum_exp = all_scores.clone().exp().sum_dim(1).log();
        let z_loss = log_sum_exp.powf_scalar(2.0).mean();
        total_z_loss = total_z_loss + z_loss.reshape([1]);
    }

    // Average over steps
    let num_steps = router_outputs.len() as f32;

    AuxiliaryLosses {
        balance_loss: total_balance_loss / num_steps,
        efficiency_loss: total_efficiency_loss / num_steps,
        z_loss: total_z_loss / num_steps,
    }
}

fn get_routing_mode(schedule: &super::curriculum::CurriculumSchedule) -> RoutingMode {
    match schedule.phase {
        TrainingPhase::WarmUp => RoutingMode::Random,
        TrainingPhase::Specialization => RoutingMode::Guided {
            epsilon: schedule.epsilon,
        },
        TrainingPhase::Maturity => RoutingMode::Guided {
            epsilon: schedule.epsilon,
        },
    }
}

fn update_param_histogram<B: Backend>(histogram: &mut [u64], indices: &Tensor<B, 2, Int>) {
    let indices_data: Vec<i32> = indices.clone().into_data().to_vec().unwrap();
    for idx in indices_data {
        if (idx as usize) < histogram.len() {
            histogram[idx as usize] += 1;
        }
    }
}

fn print_routing_statistics(histogram: &[u64], pool_size: usize) {
    let total_selections: u64 = histogram.iter().sum();
    if total_selections == 0 {
        println!("No routing data collected.");
        return;
    }

    let used_params = histogram.iter().filter(|&&c| c > 0).count();
    let avg_per_param = total_selections as f64 / pool_size as f64;

    let mut indexed: Vec<(usize, u64)> = histogram.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║              PARAMETER SELECTION STATISTICS                      ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Pool Size: {:>6} | Used: {:>6} ({:>5.1}%) | Total: {:>12}   ║",
        pool_size,
        used_params,
        100.0 * used_params as f64 / pool_size as f64,
        total_selections
    );
    println!(
        "║ Avg selections per param: {:.1}                                    ║",
        avg_per_param
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");

    println!("║ TOP 15 MOST SELECTED                                             ║");
    println!("╠──────────────┬──────────────┬───────────────────────────────────╣");
    println!("║ Param Index  │    Count     │  % of Total                       ║");
    println!("╠──────────────┼──────────────┼───────────────────────────────────╣");
    for (idx, count) in indexed.iter().take(15) {
        let pct = 100.0 * *count as f64 / total_selections as f64;
        let bar_len = (pct * 2.0).min(30.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "║ {:>10}   │ {:>10}   │ {:>5.2}% {:<26} ║",
            idx, count, pct, bar
        );
    }
    println!("╠══════════════════════════════════════════════════════════════════╣");

    println!("║ TOP 15 LEAST SELECTED (non-zero)                                 ║");
    println!("╠──────────────┬──────────────┬───────────────────────────────────╣");
    let non_zero: Vec<_> = indexed.iter().filter(|(_, c)| *c > 0).collect();
    let least: Vec<_> = non_zero.iter().rev().take(15).collect();
    for (idx, count) in least {
        let pct = 100.0 * *count as f64 / total_selections as f64;
        println!(
            "║ {:>10}   │ {:>10}   │ {:>5.3}%                             ║",
            idx, count, pct
        );
    }
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let zero_count = histogram.iter().filter(|&&c| c == 0).count();
    println!(
        "║ NEVER SELECTED: {} params ({:.1}% of pool)                       ║",
        zero_count,
        100.0 * zero_count as f64 / pool_size as f64
    );

    let top_10_total: u64 = indexed.iter().take(10).map(|(_, c)| c).sum();
    let top_10_pct = 100.0 * top_10_total as f64 / total_selections as f64;
    println!(
        "║ TOP 10 CONCENTRATION: {:.1}% of all selections                    ║",
        top_10_pct
    );

    if top_10_pct > 50.0 {
        println!("║ ⚠️  WARNING: Routing collapse detected! Top 10 params > 50%     ║");
    } else if used_params < pool_size / 2 {
        println!("║ ⚠️  WARNING: Low utilization! <50% of pool used                 ║");
    } else {
        println!("║ ✓  Routing appears healthy                                      ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}

fn print_curriculum_config(curriculum: &CurriculumConfig) {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                          CURRICULUM CONFIGURATION                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Phase 1 - Warm-up:        {:>6} steps  │  Random routing exploration      ║",
        curriculum.warmup_steps
    );
    println!(
        "║  Phase 2 - Specialization: {:>6} steps  │  ε-greedy with decay             ║",
        curriculum.specialization_steps
    );
    println!(
        "║  Phase 3 - Maturity:       {:>6} steps  │  Low ε fine-tuning               ║",
        curriculum
            .total_curriculum_steps()
            .saturating_sub(curriculum.warmup_steps + curriculum.specialization_steps)
            .max(0)
    );
    println!("╠────────────────────────────────────────────────────────────────────────────╣");
    println!(
        "║  Total Curriculum Steps:   {:>6}                                           ║",
        curriculum.total_curriculum_steps()
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    vocab_size: usize,
    embed_dim: usize,
    pool_size: usize,
    router_config: StandardRouterConfig,
    num_heads: usize,
    context_length: usize,
    dataset: &CharDataset,
    device: &B::Device,
    device_location: DeviceLocation,
) -> DPSN<B::InnerBackend> {
    let curriculum = CurriculumConfig::new();
    train_with_curriculum::<B>(
        config,
        curriculum,
        vocab_size,
        embed_dim,
        pool_size,
        router_config,
        num_heads,
        context_length,
        dataset,
        device,
        device_location,
    )
}

pub fn train_with_curriculum<B: AutodiffBackend>(
    config: TrainingConfig,
    curriculum: CurriculumConfig,
    vocab_size: usize,
    embed_dim: usize,
    pool_size: usize,
    router_config: StandardRouterConfig,
    num_heads: usize,
    context_length: usize,

    dataset: &CharDataset,
    device: &B::Device,
    _device_location: DeviceLocation,
) -> DPSN<B::InnerBackend> {
    let mut model: DPSN<B> = DPSN::new(
        vocab_size,
        embed_dim,
        pool_size,
        num_heads,
        context_length,
        config.recurrence_steps,
        router_config.clone(),
        device,
    );

    let mut optimizer = AdamConfig::new().init();
    let batcher = DPSNBatcher::new(context_length);
    let prefetcher = DataPrefetcher::new(dataset.clone(), config.batch_size, true, 4);

    let stats = model.param_stats();
    stats.print_summary("DPSN");

    print_curriculum_config(&curriculum);

    let dataset_len = dataset.len();
    let steps_per_epoch = dataset_len / config.batch_size;

    let total_steps = if let Some(epochs) = config.num_epochs {
        println!(
            "Training for {} epochs ({} steps per epoch)",
            epochs, steps_per_epoch
        );
        epochs * steps_per_epoch
    } else if let Some(steps) = config.num_steps {
        if steps > steps_per_epoch {
            panic!(
                "Error: Requested {} steps, but dataset only has enough data for {} steps (1 epoch).\n\
                 To train for multiple passes over the data, please specify 'num_epochs' instead.",
                steps, steps_per_epoch
            );
        }
        steps
    } else {
        let default_steps = 500;
        if default_steps > steps_per_epoch {
            println!(
                "Defaulting to 1 epoch ({} steps) as 500 steps would exceed data size.",
                steps_per_epoch
            );
            steps_per_epoch
        } else {
            default_steps
        }
    };

    println!("Starting training for {} steps...\n", total_steps);

    let mut running_loss = Tensor::<B, 1>::zeros([1], device);
    let mut running_balance = Tensor::<B, 1>::zeros([1], device);
    let mut running_efficiency = Tensor::<B, 1>::zeros([1], device);
    let mut loss_count = 0;
    let mut last_phase = TrainingPhase::WarmUp;
    let mut nan_count = 0usize;
    let mut param_histogram: Vec<u64> = vec![0; pool_size];

    for step in 0..total_steps {
        let schedule = curriculum.get_schedule(step);

        if schedule.phase != last_phase {
            println!(
                "\n>>> Phase Transition: {} -> {} at step {} <<<\n",
                last_phase,
                schedule.phase,
                step + 1
            );
            last_phase = schedule.phase;
        }

        let (inputs, targets) = prefetcher
            .next()
            .expect("Data prefetcher channel closed unexpectedly");
        let batch = batcher.batch::<B>(inputs, targets, device);

        let routing_mode = get_routing_mode(&schedule);
        let output = model.forward_with_mode(batch.inputs.clone(), routing_mode);

        let [batch_size, seq_len, vocab_size_out] = output.logits.dims();
        let logits_flat = output
            .logits
            .reshape([batch_size * seq_len, vocab_size_out]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let main_loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        let aux_losses =
            compute_auxiliary_losses(&output.router_outputs, router_config.k_max, pool_size);

        let balance_weight = schedule.balance_weight as f32;
        let efficiency_weight = schedule.efficiency_weight as f32;
        let z_loss_weight = schedule.z_loss_weight as f32;

        let total_loss = main_loss.clone()
            + aux_losses.balance_loss.clone() * balance_weight
            + aux_losses.efficiency_loss.clone() * efficiency_weight
            + aux_losses.z_loss.clone() * z_loss_weight;

        // --- OPTIMIZATION: Accumulate on GPU, sync only at log interval ---
        running_loss = running_loss + main_loss.clone().detach();
        running_balance = running_balance + aux_losses.balance_loss.clone().detach();
        running_efficiency = running_efficiency + aux_losses.efficiency_loss.clone().detach();
        loss_count += 1;

        let grads = total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if (step + 1) % config.log_interval == 0 {
            // Move histogram update here to avoid per-step sync
            for router_out in &output.router_outputs {
                update_param_histogram(&mut param_histogram, &router_out.indices);
            }

            // Sync scalars now
            let loss_scalar: f32 = running_loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0)
                / loss_count as f32;

            let balance_scalar: f32 = running_balance
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0)
                / loss_count as f32;

            let efficiency_scalar: f32 = running_efficiency
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0)
                / loss_count as f32;

            // Check for NaN on the averaged loss
            if loss_scalar.is_nan() || loss_scalar.is_infinite() {
                nan_count += 1;
                eprintln!(
                    "WARNING: NaN/Inf average loss at step {} ({}/{}).",
                    step + 1,
                    nan_count,
                    NAN_PATIENCE
                );
                if nan_count >= NAN_PATIENCE {
                    panic!(
                        "FATAL: {} consecutive NaN log intervals. Stopping.",
                        NAN_PATIENCE
                    );
                }
                // We can't easily skip retrospectively, but we reset.
            } else {
                nan_count = 0;
            }

            let avg_budget: f32 = output
                .router_outputs
                .iter()
                .map(|out| {
                    out.budget.iter().map(|&b| b as f32).sum::<f32>() / out.budget.len() as f32
                })
                .sum::<f32>()
                / output.router_outputs.len() as f32;

            println!(
                "Step {}/{} [{}] | Loss: {:.4} | Balance: {:.4} | Eff: {:.2} | Budget: {:.0} | ε: {:.3}",
                step + 1,
                total_steps,
                schedule.phase,
                loss_scalar,
                balance_scalar,
                efficiency_scalar,
                avg_budget,
                schedule.epsilon
            );

            running_loss = Tensor::zeros([1], device);
            running_balance = Tensor::zeros([1], device);
            running_efficiency = Tensor::zeros([1], device);
            loss_count = 0;
        }

        if let Some(ref checkpoint_dir) = config.checkpoint_dir {
            if (step + 1) % config.save_interval == 0 {
                let inner_model = model.clone().valid();
                if let Err(e) = save_checkpoint(&inner_model, checkpoint_dir, step + 1) {
                    eprintln!("Failed to save checkpoint: {}", e);
                }
            }
        }
    }

    println!("\nTraining completed!");
    print_routing_statistics(&param_histogram, pool_size);

    let final_model = model.valid();

    if let Some(ref checkpoint_dir) = config.checkpoint_dir {
        if let Err(e) = save_checkpoint(&final_model, checkpoint_dir, total_steps) {
            eprintln!("Failed to save final checkpoint: {}", e);
        }
    }

    final_model
}

pub fn train_hierarchical<B: AutodiffBackend>(
    config: TrainingConfig,
    curriculum: CurriculumConfig,
    vocab_size: usize,
    embed_dim: usize,
    pool_size: usize,
    router_config: HierarchicalRouterConfig,
    num_heads: usize,
    context_length: usize,
    dataset: &CharDataset,
    device: &B::Device,
    _device_location: DeviceLocation,
) -> HierarchicalDPSN<B::InnerBackend> {
    let mut model: HierarchicalDPSN<B> = HierarchicalDPSN::new(
        vocab_size,
        embed_dim,
        pool_size,
        num_heads,
        context_length,
        config.recurrence_steps,
        router_config.clone(),
        device,
    );
    let mut optimizer = AdamConfig::new().init();
    let batcher = DPSNBatcher::new(context_length);
    let prefetcher = DataPrefetcher::new(dataset.clone(), config.batch_size, true, 4);

    let stats = model.param_stats();
    stats.print_summary("Hierarchical DPSN");

    print_curriculum_config(&curriculum);

    let dataset_len = dataset.len();
    let steps_per_epoch = dataset_len / config.batch_size;

    let total_steps = if let Some(epochs) = config.num_epochs {
        println!(
            "Training for {} epochs ({} steps per epoch)",
            epochs, steps_per_epoch
        );
        epochs * steps_per_epoch
    } else if let Some(steps) = config.num_steps {
        if steps > steps_per_epoch {
            panic!(
                "Error: Requested {} steps, but dataset only has enough data for {} steps (1 epoch).\n\
                 To train for multiple passes over the data, please specify 'num_epochs' instead.",
                steps, steps_per_epoch
            );
        }
        steps
    } else {
        let default_steps = 500;
        if default_steps > steps_per_epoch {
            println!(
                "Defaulting to 1 epoch ({} steps) as 500 steps would exceed data size.",
                steps_per_epoch
            );
            steps_per_epoch
        } else {
            default_steps
        }
    };

    println!(
        "Starting hierarchical training for {} steps...\n",
        total_steps
    );

    let mut running_loss = Tensor::<B, 1>::zeros([1], device);
    let mut running_balance = Tensor::<B, 1>::zeros([1], device);
    let mut running_efficiency = Tensor::<B, 1>::zeros([1], device);
    let mut loss_count = 0;
    let mut last_phase = TrainingPhase::WarmUp;
    let mut nan_count = 0usize;
    let mut param_histogram: Vec<u64> = vec![0; pool_size];

    for step in 0..total_steps {
        let schedule = curriculum.get_schedule(step);

        if schedule.phase != last_phase {
            println!(
                "\n>>> Phase Transition: {} -> {} at step {} <<<\n",
                last_phase,
                schedule.phase,
                step + 1
            );
            last_phase = schedule.phase;
        }

        let (inputs, targets) = prefetcher
            .next()
            .expect("Data prefetcher channel closed unexpectedly");
        let batch = batcher.batch::<B>(inputs, targets, device);

        let routing_mode = get_routing_mode(&schedule);
        let output = model.forward_with_mode(batch.inputs.clone(), routing_mode);

        let [batch_size, seq_len, vocab_size_out] = output.logits.dims();
        let logits_flat = output
            .logits
            .reshape([batch_size * seq_len, vocab_size_out]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let main_loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        let aux_losses =
            compute_auxiliary_losses(&output.router_outputs, router_config.k_max, pool_size);

        let balance_weight = schedule.balance_weight as f32;
        let efficiency_weight = schedule.efficiency_weight as f32;
        let z_loss_weight = schedule.z_loss_weight as f32;

        let total_loss = main_loss.clone()
            + aux_losses.balance_loss.clone() * balance_weight
            + aux_losses.efficiency_loss.clone() * efficiency_weight
            + aux_losses.z_loss.clone() * z_loss_weight;

        // --- OPTIMIZATION: Accumulate on GPU, sync only at log interval ---
        running_loss = running_loss + main_loss.clone().detach();
        running_balance = running_balance + aux_losses.balance_loss.clone().detach();
        running_efficiency = running_efficiency + aux_losses.efficiency_loss.clone().detach();
        loss_count += 1;

        let grads = total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if (step + 1) % config.log_interval == 0 {
            // Move histogram update here to avoid per-step sync
            for router_out in &output.router_outputs {
                update_param_histogram(&mut param_histogram, &router_out.indices);
            }

            // Sync scalars now
            let loss_scalar: f32 = running_loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0)
                / loss_count as f32;

            let balance_scalar: f32 = running_balance
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0)
                / loss_count as f32;

            let efficiency_scalar: f32 = running_efficiency
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0)
                / loss_count as f32;

            // Check for NaN on the averaged loss
            if loss_scalar.is_nan() || loss_scalar.is_infinite() {
                nan_count += 1;
                eprintln!(
                    "WARNING: NaN/Inf average loss at step {} ({}/{}).",
                    step + 1,
                    nan_count,
                    NAN_PATIENCE
                );
                if nan_count >= NAN_PATIENCE {
                    panic!(
                        "FATAL: {} consecutive NaN log intervals. Stopping.",
                        NAN_PATIENCE
                    );
                }
            } else {
                nan_count = 0;
            }

            let avg_budget: f32 = output
                .router_outputs
                .iter()
                .map(|out| {
                    out.budget.iter().map(|&b| b as f32).sum::<f32>() / out.budget.len() as f32
                })
                .sum::<f32>()
                / output.router_outputs.len() as f32;

            println!(
                "Step {}/{} [{}] | Loss: {:.4} | Balance: {:.4} | Eff: {:.2} | Budget: {:.0} | ε: {:.3}",
                step + 1,
                total_steps,
                schedule.phase,
                loss_scalar,
                balance_scalar,
                efficiency_scalar,
                avg_budget,
                schedule.epsilon
            );

            running_loss = Tensor::zeros([1], device);
            running_balance = Tensor::zeros([1], device);
            running_efficiency = Tensor::zeros([1], device);
            loss_count = 0;
        }

        if let Some(ref checkpoint_dir) = config.checkpoint_dir {
            if (step + 1) % config.save_interval == 0 {
                let inner_model = model.clone().valid();
                if let Err(e) = save_hierarchical_checkpoint(&inner_model, checkpoint_dir, step + 1)
                {
                    eprintln!("Failed to save checkpoint: {}", e);
                }
            }
        }
    }

    println!("\nHierarchical training completed!");
    print_routing_statistics(&param_histogram, pool_size);

    let final_model = model.valid();

    if let Some(ref checkpoint_dir) = config.checkpoint_dir {
        if let Err(e) = save_hierarchical_checkpoint(&final_model, checkpoint_dir, total_steps) {
            eprintln!("Failed to save final checkpoint: {}", e);
        }
    }

    final_model
}
