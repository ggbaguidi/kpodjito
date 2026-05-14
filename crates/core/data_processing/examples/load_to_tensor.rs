use kpodjito_core_data_processing::{LoadConfig, Normalization, TrainingDataLoader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Demo tabular dataset for binary classification.
    let csv = "age,income,label\n25,32000,0\n30,45000,1\n22,28000,0\n40,70000,1\n35,54000,1\n";

    let config = LoadConfig {
        delimiter: ',',
        has_header: true,
        target_column: Some(2),
        normalization: Normalization::Standard,
    };

    let dataset = TrainingDataLoader::load_from_text(csv, &config)?;

    println!("=== Load + Preprocess To Tensor ===");
    println!("features shape: {:?}", dataset.features.shape().dims());
    println!("targets shape: {:?}", dataset.targets.as_ref().map(|t| t.shape().dims()));
    println!("feature names: {:?}", dataset.feature_names);
    println!("target name: {:?}", dataset.target_name);
    println!("first feature row: {:?}", &dataset.features.data()[..2]);

    let (train, valid) = TrainingDataLoader::train_valid_split(&dataset, 0.4)?;
    println!("\ntrain shape: {:?}", train.features.shape().dims());
    println!("valid shape: {:?}", valid.features.shape().dims());

    Ok(())
}
