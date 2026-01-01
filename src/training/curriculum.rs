use burn::config::Config;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingPhase {
    WarmUp,
    Specialization,
    Maturity,
}

impl TrainingPhase {
    pub fn from_step(step: usize, config: &CurriculumConfig) -> Self {
        if step < config.warmup_steps {
            TrainingPhase::WarmUp
        } else if step < config.warmup_steps + config.specialization_steps {
            TrainingPhase::Specialization
        } else {
            TrainingPhase::Maturity
        }
    }
}

#[derive(Config, Debug)]
pub struct CurriculumConfig {
    #[config(default = 100)]
    pub warmup_steps: usize,

    #[config(default = 400)]
    pub specialization_steps: usize,

    #[config(default = 1.0)]
    pub warmup_epsilon: f64,

    #[config(default = 0.3)]
    pub specialization_epsilon_start: f64,

    #[config(default = 0.05)]
    pub specialization_epsilon_end: f64,

    #[config(default = 0.01)]
    pub maturity_epsilon: f64,

    #[config(default = 0.0)]
    pub warmup_balance_weight: f64,

    #[config(default = 0.1)]
    pub specialization_balance_weight: f64,

    #[config(default = 0.05)]
    pub maturity_balance_weight: f64,

    #[config(default = 0.0)]
    pub warmup_efficiency_weight: f64,

    #[config(default = 0.0)]
    pub specialization_efficiency_weight: f64,

    #[config(default = 0.1)]
    pub maturity_efficiency_weight: f64,

    #[config(default = 0.001)]
    pub z_loss_weight: f64,
}

pub struct CurriculumSchedule {
    pub phase: TrainingPhase,
    pub epsilon: f64,
    pub balance_weight: f64,
    pub efficiency_weight: f64,
    pub z_loss_weight: f64,
}

impl CurriculumConfig {
    pub fn get_schedule(&self, step: usize) -> CurriculumSchedule {
        let phase = TrainingPhase::from_step(step, self);

        match phase {
            TrainingPhase::WarmUp => CurriculumSchedule {
                phase,
                epsilon: self.warmup_epsilon,
                balance_weight: self.warmup_balance_weight,
                efficiency_weight: self.warmup_efficiency_weight,
                z_loss_weight: self.z_loss_weight,
            },
            TrainingPhase::Specialization => {
                let spec_step = step - self.warmup_steps;
                let progress = spec_step as f64 / self.specialization_steps as f64;
                let epsilon = self.specialization_epsilon_start
                    + (self.specialization_epsilon_end - self.specialization_epsilon_start)
                        * progress;

                CurriculumSchedule {
                    phase,
                    epsilon,
                    balance_weight: self.specialization_balance_weight,
                    efficiency_weight: self.specialization_efficiency_weight,
                    z_loss_weight: self.z_loss_weight,
                }
            }
            TrainingPhase::Maturity => CurriculumSchedule {
                phase,
                epsilon: self.maturity_epsilon,
                balance_weight: self.maturity_balance_weight,
                efficiency_weight: self.maturity_efficiency_weight,
                z_loss_weight: self.z_loss_weight,
            },
        }
    }

    pub fn total_curriculum_steps(&self) -> usize {
        self.warmup_steps + self.specialization_steps
    }
}

impl std::fmt::Display for TrainingPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingPhase::WarmUp => write!(f, "WarmUp"),
            TrainingPhase::Specialization => write!(f, "Specialization"),
            TrainingPhase::Maturity => write!(f, "Maturity"),
        }
    }
}
