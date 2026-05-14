use kpodjito_core_data_processing::{DataTypeDetector, ScanOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).cloned().unwrap_or_else(|| ".".to_string());
    let include_hidden = args.iter().any(|arg| arg == "--include-hidden");

    let options = ScanOptions {
        include_hidden,
        ..ScanOptions::default()
    };

    let report = DataTypeDetector::scan_folder_with_options(&root, &options)?;

    println!("=== Dataset Batch Scan ===");
    println!("root: {}", report.root);
    println!("files scanned: {}", report.files_scanned);
    println!("files failed: {}", report.files_failed);

    if let Some(conf) = &report.overall_confidence {
        println!(
            "overall confidence: min={:.3} mean={:.3} max={:.3}",
            conf.min, conf.mean, conf.max
        );
    }

    println!("\n--- Modality Distribution ---");
    for stat in &report.modality_stats {
        println!(
            "{:?}: count={} ratio={:.2}% confidence[min={:.3}, mean={:.3}, max={:.3}]",
            stat.data_type,
            stat.count,
            stat.ratio * 100.0,
            stat.confidence.min,
            stat.confidence.mean,
            stat.confidence.max,
        );
    }

    println!("\n--- Per-file ---");
    for item in &report.per_file {
        println!(
            "{} [{} bytes] => {:?}/{:?} conf={:.3} ({})",
            item.path,
            item.size_bytes,
            item.detection.data_type,
            item.detection.format,
            item.detection.confidence,
            item.detection.reason
        );
    }

    if !report.failures.is_empty() {
        println!("\n--- Failures ---");
        for failure in &report.failures {
            println!("{} => {}", failure.path, failure.error);
        }
    }

    Ok(())
}
