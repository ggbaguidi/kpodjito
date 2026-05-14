//! Application use cases for train, infer, and evaluate.

use kpodjito_core_domain::{commands::Initialize, entities::ModelId};
use kpodjito_core_data_processing::{BatchScanReport, DataTypeDetector, Detection};
use kpodjito_core_error::Result;
use kpodjito_core_ports::ModelRepository;

pub struct BootstrapUseCase;

impl BootstrapUseCase {
    pub fn execute(repository: &mut impl ModelRepository) -> Result<()> {
        let _command = Initialize;
        repository.save(&ModelId("bootstrap".to_string()))
    }
}

pub struct DetectDataTypeUseCase;

impl DetectDataTypeUseCase {
    pub fn execute(path: Option<&str>, bytes: &[u8]) -> Detection {
        DataTypeDetector::detect(path, bytes)
    }
}

pub struct ScanDatasetUseCase;

impl ScanDatasetUseCase {
    pub fn execute(root: &str) -> Result<BatchScanReport> {
        DataTypeDetector::scan_folder(root)
    }
}