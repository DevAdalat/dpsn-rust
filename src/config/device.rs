use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum DeviceType {
    #[default]
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePlacement {
    /// Device for the large Parameter Pool
    pub pool: DeviceType,
    /// Device for the Router (Selection Logic)
    pub router: DeviceType,
    /// Device for the Token Embeddings
    pub embedding: DeviceType,
    /// Device for the Execution Engine (Attention/Computation)
    pub engine: DeviceType,
    /// Device for the Output Head (Projection)
    pub head: DeviceType,
}

impl Default for DevicePlacement {
    fn default() -> Self {
        Self {
            pool: DeviceType::Gpu,
            router: DeviceType::Gpu,
            embedding: DeviceType::Gpu,
            engine: DeviceType::Gpu,
            head: DeviceType::Gpu,
        }
    }
}

impl DevicePlacement {
    pub fn new_offloaded() -> Self {
        Self {
            pool: DeviceType::Cpu,
            router: DeviceType::Gpu,
            embedding: DeviceType::Gpu,
            engine: DeviceType::Gpu,
            head: DeviceType::Gpu,
        }
    }

    pub fn new_all_cpu() -> Self {
        Self {
            pool: DeviceType::Cpu,
            router: DeviceType::Cpu,
            embedding: DeviceType::Cpu,
            engine: DeviceType::Cpu,
            head: DeviceType::Cpu,
        }
    }

    pub fn is_offloaded(&self) -> bool {
        self.pool == DeviceType::Cpu && self.router == DeviceType::Gpu
    }

    pub fn is_all_cpu(&self) -> bool {
        self.pool == DeviceType::Cpu
            && self.router == DeviceType::Cpu
            && self.embedding == DeviceType::Cpu
            && self.engine == DeviceType::Cpu
            && self.head == DeviceType::Cpu
    }

    pub fn is_all_gpu(&self) -> bool {
        self.pool == DeviceType::Gpu
            && self.router == DeviceType::Gpu
            && self.embedding == DeviceType::Gpu
            && self.engine == DeviceType::Gpu
            && self.head == DeviceType::Gpu
    }
}
