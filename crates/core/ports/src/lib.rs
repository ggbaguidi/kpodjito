//! Ports define inbound and outbound boundaries for the application.

use kpodjito_core_domain::entities::ModelId;
use kpodjito_core_error::Result;

pub trait ModelRepository {
    fn save(&mut self, model_id: &ModelId) -> Result<()>;
    fn load(&self, model_id: &ModelId) -> Result<bool>;
}

pub trait Clock {
    fn now_millis(&self) -> u64;
}