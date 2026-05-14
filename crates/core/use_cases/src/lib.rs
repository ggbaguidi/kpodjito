//! Application use cases for train, infer, and evaluate.

use kpodjito_core_domain::{commands::Initialize, entities::ModelId};
use kpodjito_core_data_processing::{
    BatchScanReport, DataTypeDetector, Detection, LoadConfig, TensorDataset, TrainingDataLoader,
};
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

pub struct LoadTrainingDatasetUseCase;

impl LoadTrainingDatasetUseCase {
    pub fn execute(path: &str, config: &LoadConfig) -> Result<TensorDataset> {
        TrainingDataLoader::load_from_path(path, config)
    }
}

pub struct SplitDatasetUseCase;

impl SplitDatasetUseCase {
    pub fn execute(dataset: &TensorDataset, valid_ratio: f32) -> Result<(TensorDataset, TensorDataset)> {
        TrainingDataLoader::train_valid_split(dataset, valid_ratio)
    }
}