//! Application use cases for train, infer, and evaluate.

use kpodjito_core_domain::{commands::Initialize, entities::ModelId};
use kpodjito_core_error::Result;
use kpodjito_core_ports::ModelRepository;

pub struct BootstrapUseCase;

impl BootstrapUseCase {
    pub fn execute(repository: &mut impl ModelRepository) -> Result<()> {
        let _command = Initialize;
        repository.save(&ModelId("bootstrap".to_string()))
    }
}