use burn::config::Config;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::path::PathBuf;

use super::checkpoint::save_checkpoint;
use super::curriculum::{CurriculumConfig, TrainingPhase};
use crate::data::batcher::DPSNBatcher;
use crate::data::dataset::CharDataset;
use crate::model::dpsn::DPSN;
use crate::model::router::RoutingMode;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 500)]
    pub num_steps: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 1e-3)]
    pub learning_rate: f64,

    #[config(default = 50)]
    pub log_interval: usize,

    #[config(default = 100)]
    pub save_interval: usize,

    pub checkpoint_dir: Option<PathBuf>,
}

pub struct AuxiliaryLosses<B: Backend> {
    pub balance_loss: Tensor<B, 1>,
    pub efficiency_loss: Tensor<B, 1>,
    pub z_loss: Tensor<B, 1>,
}

fn compute_auxiliary_losses<B: Backend>(
    routing_probs: Tensor<B, 2>,
    all_scores: Tensor<B, 2>,
    budget: &[usize],
    k_max: usize,
    pool_size: usize,
) -> AuxiliaryLosses<B> {
    let device = routing_probs.device();

    let mean_probs: Tensor<B, 1> = routing_probs.clone().mean_dim(0).squeeze(0);
    let balance_loss = mean_probs.powf_scalar(2.0).sum() * (pool_size as f32);

    let avg_budget: f32 = budget.iter().map(|&b| b as f32).sum::<f32>() / budget.len() as f32;
    let efficiency_ratio = avg_budget / k_max as f32;
    let efficiency_loss = Tensor::<B, 1>::from_floats([efficiency_ratio], &device);

    let log_sum_exp = all_scores.clone().exp().sum_dim(1).log();
    let z_loss = log_sum_exp.powf_scalar(2.0).mean();
    let z_loss = z_loss.reshape([1]);

    AuxiliaryLosses {
        balance_loss: balance_loss.reshape([1]),
        efficiency_loss,
        z_loss,
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

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    vocab_size: usize,
    embed_dim: usize,
    pool_size: usize,
    k_min: usize,
    k_max: usize,
    router_hidden_dim: usize,
    context_length: usize,
    exploration_noise: f64,
    dataset: &CharDataset,
    device: &B::Device,
) -> DPSN<B::InnerBackend> {
    let curriculum = CurriculumConfig::new();
    train_with_curriculum::<B>(
        config,
        curriculum,
        vocab_size,
        embed_dim,
        pool_size,
        k_min,
        k_max,
        router_hidden_dim,
        context_length,
        exploration_noise,
        dataset,
        device,
    )
}

pub fn train_with_curriculum<B: AutodiffBackend>(
    config: TrainingConfig,
    curriculum: CurriculumConfig,
    vocab_size: usize,
    embed_dim: usize,
    pool_size: usize,
    k_min: usize,
    k_max: usize,
    router_hidden_dim: usize,
    context_length: usize,
    exploration_noise: f64,
    dataset: &CharDataset,
    device: &B::Device,
) -> DPSN<B::InnerBackend> {
    let mut model: DPSN<B> = DPSN::new(
        vocab_size,
        embed_dim,
        pool_size,
        k_min,
        k_max,
        router_hidden_dim,
        context_length,
        exploration_noise,
        device,
    );
    let mut optimizer = AdamConfig::new().init();
    let batcher = DPSNBatcher::new(context_length);

    let stats = model.param_stats();
    println!("=== DPSN Model Statistics ===");
    println!("Total parameters: {}", stats.total_params);
    println!("Pool parameters: {}", stats.pool_params);
    println!("Router parameters: {}", stats.router_params);
    println!("Embedding parameters: {}", stats.embed_params);
    println!("Output parameters: {}", stats.output_params);
    println!("Active params per token: {}", stats.active_params_per_token);
    println!("=============================\n");

    println!("=== Curriculum Configuration ===");
    println!("Warm-up steps: {}", curriculum.warmup_steps);
    println!("Specialization steps: {}", curriculum.specialization_steps);
    println!(
        "Total curriculum steps: {}",
        curriculum.total_curriculum_steps()
    );
    println!("================================\n");

    println!("Starting training for {} steps...\n", config.num_steps);

    let mut running_loss = 0.0f32;
    let mut running_balance = 0.0f32;
    let mut running_efficiency = 0.0f32;
    let mut loss_count = 0;
    let mut last_phase = TrainingPhase::WarmUp;

    for step in 0..config.num_steps {
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

        let batch = batcher.batch::<B>(dataset, config.batch_size, device);

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

        let aux_losses = compute_auxiliary_losses(
            output.router_output.routing_probs.clone(),
            output.router_output.all_scores.clone(),
            &output.router_output.budget,
            k_max,
            pool_size,
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

        let grads = total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if (step + 1) % config.log_interval == 0 {
            let avg_loss = running_loss / loss_count as f32;
            let avg_balance = running_balance / loss_count as f32;
            let avg_efficiency = running_efficiency / loss_count as f32;

            let avg_budget: f32 = output
                .router_output
                .budget
                .iter()
                .map(|&b| b as f32)
                .sum::<f32>()
                / output.router_output.budget.len() as f32;

            println!(
                "Step {}/{} [{}] | Loss: {:.4} | Balance: {:.4} | Eff: {:.2} | Budget: {:.0} | ε: {:.3}",
                step + 1,
                config.num_steps,
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
                let inner_model = model.clone().valid();
                if let Err(e) = save_checkpoint(&inner_model, checkpoint_dir, step + 1) {
                    eprintln!("Failed to save checkpoint: {}", e);
                }
            }
        }
    }

    println!("\nTraining completed!");

    let final_model = model.valid();

    if let Some(ref checkpoint_dir) = config.checkpoint_dir {
        if let Err(e) = save_checkpoint(&final_model, checkpoint_dir, config.num_steps) {
            eprintln!("Failed to save final checkpoint: {}", e);
        }
    }

    final_model
}
