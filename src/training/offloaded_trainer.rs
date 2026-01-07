use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use super::checkpoint::save_offloaded_checkpoint;
use super::curriculum::{CurriculumConfig, CurriculumSchedule, TrainingPhase};
use super::trainer::TrainingConfig;
use crate::data::batcher::DPSNBatcher;
use crate::data::dataset::{CharDataset, DataLoader};
use crate::model::offloaded_dpsn::{
    compute_w_active_gradients, OffloadedDPSN, OffloadedDPSNConfig, OffloadedDPSNGpuPart,
};
use crate::model::router::{RouterOutput, RoutingMode};

const NAN_PATIENCE: usize = 3;

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

    let num_steps = router_outputs.len() as f32;

    AuxiliaryLosses {
        balance_loss: total_balance_loss / num_steps,
        efficiency_loss: total_efficiency_loss / num_steps,
        z_loss: total_z_loss / num_steps,
    }
}

fn get_routing_mode(schedule: &CurriculumSchedule) -> RoutingMode {
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
    println!("║         OFFLOADED PARAMETER SELECTION STATISTICS                ║");
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

    println!("║ TOP 10 MOST SELECTED                                             ║");
    println!("╠──────────────┬──────────────┬───────────────────────────────────╣");
    for (idx, count) in indexed.iter().take(10) {
        let pct = 100.0 * *count as f64 / total_selections as f64;
        let bar_len = (pct * 2.0).min(30.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "║ {:>10}   │ {:>10}   │ {:>5.2}% {:<26} ║",
            idx, count, pct, bar
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}

pub fn train_offloaded<B: AutodiffBackend, CpuB: Backend>(
    config: TrainingConfig,
    curriculum: CurriculumConfig,
    model_config: OffloadedDPSNConfig,
    dataset: &CharDataset,
    gpu_device: &B::Device,
    cpu_device: &CpuB::Device,
) -> (OffloadedDPSNGpuPart<B::InnerBackend>, Vec<f32>)
where
    B::InnerBackend: Backend,
{
    let mut model: OffloadedDPSN<B, CpuB> = model_config.init(gpu_device, cpu_device);
    let mut gpu_optimizer = AdamConfig::new().init();
    let batcher = DPSNBatcher::new(model_config.context_length);
    let mut dataloader = DataLoader::new(dataset, config.batch_size, true);

    let stats = model.param_stats();
    stats.print_summary();

    println!("\n=== Curriculum Configuration ===");
    println!("Warm-up steps: {}", curriculum.warmup_steps);
    println!("Specialization steps: {}", curriculum.specialization_steps);
    println!(
        "Total curriculum steps: {}",
        curriculum.total_curriculum_steps()
    );
    println!("================================\n");

    // --- Steps / Epochs Logic ---
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

    println!("Starting OFFLOADED training for {} steps...\n", total_steps);

    println!("================================\n");

    // --- Steps / Epochs Logic ---
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
        // Default to 500 steps if nothing specified, but cap at 1 epoch
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

    println!("Starting OFFLOADED training for {} steps...\n", total_steps);
    println!("Pool on CPU: {} params", stats.cpu_params);
    println!("Model on GPU: {} params\n", stats.gpu_params);

    let mut running_loss = 0.0f32;
    let mut running_balance = 0.0f32;
    let mut running_efficiency = 0.0f32;
    let mut loss_count = 0;
    let mut last_phase = TrainingPhase::WarmUp;
    let mut nan_count = 0usize;
    let mut param_histogram: Vec<u64> = vec![0; model_config.pool_size];
    let mut pool_data: Vec<f32>;

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

        let (inputs, targets) = dataloader.next_batch();
        let batch = batcher.batch::<B>(inputs, targets, gpu_device);
        let routing_mode = get_routing_mode(&schedule);

        let output = model.forward_with_mode(batch.inputs.clone(), routing_mode);

        update_param_histogram(&mut param_histogram, &output.router_outputs[0].indices);

        let [batch_size, seq_len, vocab_size_out] = output.logits.dims();
        let logits_flat = output
            .logits
            .reshape([batch_size * seq_len, vocab_size_out]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let main_loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        let aux_losses = compute_auxiliary_losses(
            &output.router_outputs,
            model_config.k_max,
            model_config.pool_size,
        );

        let balance_weight = schedule.balance_weight as f32;
        let efficiency_weight = schedule.efficiency_weight as f32;
        let z_loss_weight = schedule.z_loss_weight as f32;

        let total_loss = main_loss.clone()
            + aux_losses.balance_loss.clone() * balance_weight
            + aux_losses.efficiency_loss.clone() * efficiency_weight
            + aux_losses.z_loss.clone() * z_loss_weight;

        let loss_scalar: f32 = main_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .first()
            .copied()
            .unwrap_or(0.0);

        let balance_scalar: f32 = aux_losses
            .balance_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .first()
            .copied()
            .unwrap_or(0.0);

        let efficiency_scalar: f32 = aux_losses
            .efficiency_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .first()
            .copied()
            .unwrap_or(0.0);

        running_loss += loss_scalar;
        running_balance += balance_scalar;
        running_efficiency += efficiency_scalar;
        loss_count += 1;

        if loss_scalar.is_nan() || loss_scalar.is_infinite() {
            nan_count += 1;
            eprintln!(
                "WARNING: NaN/Inf loss at step {} ({}/{}). Skipping.",
                step + 1,
                nan_count,
                NAN_PATIENCE
            );
            if nan_count >= NAN_PATIENCE {
                eprintln!(
                    "FATAL: {} consecutive NaN losses. Stopping training.",
                    NAN_PATIENCE
                );
                break;
            }
            continue;
        }
        nan_count = 0;

        let grads = total_loss.backward();

        let gpu_grads = GradientsParams::from_grads(grads, &model.gpu_part);
        model.gpu_part = gpu_optimizer.step(config.learning_rate, model.gpu_part, gpu_grads);

        let flat_batch_size = batch_size * seq_len;
        let output_grad =
            Tensor::<B, 2>::ones([flat_batch_size, model_config.embed_dim], gpu_device);

        let tracker = model.grad_tracker.borrow();
        if let Some(ref scores) = tracker.cached_scores {
            // TODO: Update gradient computation for recurrence.
            // Currently, this only supports the last step if cached_scores stores one step.
            // If we want to train properly with recurrence, we need to accumulate gradients across steps.
            // This requires major changes to PoolGradientTracker or manual backprop loop.
            // For now, let's assume single step or just warn.
            eprintln!("Warning: Offloaded training with recurrence and gradient tracking is not fully implemented yet.");
            let w_grads = compute_w_active_gradients(output_grad, scores.clone());

            drop(tracker);

            let cached_indices = model.grad_tracker.borrow().cached_indices.clone().unwrap();

            model.pool.update_with_gradients(
                cached_indices,
                w_grads,
                config.learning_rate,
                0.9,
                0.999,
                1e-8,
            );

            model.grad_tracker.borrow_mut().clear();
        }

        if (step + 1) % config.log_interval == 0 {
            let avg_loss = running_loss / loss_count as f32;
            let avg_balance = running_balance / loss_count as f32;
            let avg_efficiency = running_efficiency / loss_count as f32;

            let avg_budget: f32 = output
                .router_outputs
                .iter()
                .map(|out| {
                    out.budget.iter().map(|&b| b as f32).sum::<f32>() / out.budget.len() as f32
                })
                .sum::<f32>()
                / output.router_outputs.len() as f32;

            println!(
                "Step {}/{} [{}] | Loss: {:.4} | Bal: {:.4} | Eff: {:.2} | k: {:.0} | ε: {:.3}",
                step + 1,
                total_steps,
                schedule.phase,
                avg_loss,
                avg_balance,
                avg_efficiency,
                avg_budget,
                schedule.epsilon
            );

            running_loss = 0.0;
            running_balance = 0.0;
            running_efficiency = 0.0;
            loss_count = 0;
        }

        if let Some(ref checkpoint_dir) = config.checkpoint_dir {
            if (step + 1) % config.save_interval == 0 {
                let inner_gpu_part = model.gpu_part.clone().valid();
                pool_data = model.pool.pool_cpu.clone().into_data().to_vec().unwrap();
                if let Err(e) = save_offloaded_checkpoint(
                    &inner_gpu_part,
                    &pool_data,
                    model_config.pool_size,
                    model_config.embed_dim,
                    checkpoint_dir,
                    step + 1,
                ) {
                    eprintln!("Failed to save checkpoint: {}", e);
                }
            }
        }
    }

    println!("\nOffloaded training completed!");
    print_routing_statistics(&param_histogram, model_config.pool_size);

    let final_gpu_part = model.gpu_part.valid();
    pool_data = model.pool.pool_cpu.clone().into_data().to_vec().unwrap();

    if let Some(ref checkpoint_dir) = config.checkpoint_dir {
        if let Err(e) = save_offloaded_checkpoint(
            &final_gpu_part,
            &pool_data,
            model_config.pool_size,
            model_config.embed_dim,
            checkpoint_dir,
            total_steps,
        ) {
            eprintln!("Failed to save final checkpoint: {}", e);
        }
    }

    (final_gpu_part, pool_data)
}
