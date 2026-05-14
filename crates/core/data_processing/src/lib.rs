use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use kpodjito_core_tensor::Tensor;
pub use kpodjito_core_error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Normalization {
    None,
    Standard,
    MinMax,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadConfig {
    pub delimiter: char,
    pub has_header: bool,
    pub target_column: Option<usize>,
    pub normalization: Normalization,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            has_header: true,
            target_column: None,
            normalization: Normalization::None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorDataset {
    pub features: Tensor<f32>,
    pub targets: Option<Tensor<f32>>,
    pub feature_names: Vec<String>,
    pub target_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    Text,
    Image,
    Sound,
    Graph,
    Unknown,
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataFormat {
    PlainText,
    Json,
    Csv,
    Png,
    Jpeg,
    Gif,
    Bmp,
    Webp,
    Tiff,
    Wav,
    Mp3,
    Flac,
    Ogg,
    Midi,
    Dot,
    GraphMl,
    Gml,
    EdgeList,
    AdjList,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub data_type: DataType,
    pub format: DataFormat,
    pub confidence: f32,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FileScanResult {
    pub path: String,
    pub size_bytes: u64,
    pub detection: Detection,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FailedFile {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModalityStats {
    pub data_type: DataType,
    pub count: usize,
    pub ratio: f32,
    pub confidence: ConfidenceStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchScanReport {
    pub root: String,
    pub files_scanned: usize,
    pub files_failed: usize,
    pub per_file: Vec<FileScanResult>,
    pub failures: Vec<FailedFile>,
    pub modality_stats: Vec<ModalityStats>,
    pub overall_confidence: Option<ConfidenceStats>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanOptions {
    pub max_bytes_per_file: usize,
    pub include_hidden: bool,
    pub excluded_dir_names: Vec<String>,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            max_bytes_per_file: 64 * 1024,
            include_hidden: false,
            excluded_dir_names: vec![".git".to_string(), "target".to_string()],
        }
    }
}

pub struct DataTypeDetector;

pub struct TrainingDataLoader;

impl DataTypeDetector {
    pub fn detect(path: Option<&str>, bytes: &[u8]) -> Detection {
        if bytes.is_empty() {
            if let Some(path_value) = path {
                if let Some(ext_detection) = detect_from_extension(path_value) {
                    return ext_detection;
                }
            }
            return Detection {
                data_type: DataType::Empty,
                format: DataFormat::Unknown,
                confidence: 1.0,
                reason: "empty payload".to_string(),
            };
        }

        if let Some(detection) = detect_image_from_signature(bytes) {
            return detection;
        }

        if let Some(detection) = detect_sound_from_signature(bytes) {
            return detection;
        }

        if let Some(text) = decode_text_if_likely(bytes) {
            if let Some(graph_detection) = detect_graph_from_text(&text) {
                return graph_detection;
            }
            if let Some(text_detection) = detect_text_format(&text) {
                return text_detection;
            }
            return Detection {
                data_type: DataType::Text,
                format: DataFormat::PlainText,
                confidence: 0.85,
                reason: "utf8 and mostly printable text".to_string(),
            };
        }

        if let Some(path_value) = path {
            if let Some(ext_detection) = detect_from_extension(path_value) {
                return ext_detection;
            }
        }

        Detection {
            data_type: DataType::Unknown,
            format: DataFormat::Unknown,
            confidence: 0.2,
            reason: "no known signatures or patterns matched".to_string(),
        }
    }

    pub fn detect_from_text(text: &str) -> Detection {
        if text.trim().is_empty() {
            return Detection {
                data_type: DataType::Empty,
                format: DataFormat::PlainText,
                confidence: 1.0,
                reason: "empty text".to_string(),
            };
        }

        if let Some(graph_detection) = detect_graph_from_text(text) {
            return graph_detection;
        }

        if let Some(text_detection) = detect_text_format(text) {
            return text_detection;
        }

        Detection {
            data_type: DataType::Text,
            format: DataFormat::PlainText,
            confidence: 0.9,
            reason: "plain text content".to_string(),
        }
    }

    pub fn scan_folder(root: &str) -> Result<BatchScanReport> {
        Self::scan_folder_with_options(root, &ScanOptions::default())
    }

    pub fn scan_folder_with_options(root: &str, options: &ScanOptions) -> Result<BatchScanReport> {
        let root_path = Path::new(root);
        if !root_path.exists() {
            return Err(Error::InvalidDataFormat {
                details: format!("scan root does not exist: {root}"),
            });
        }
        if !root_path.is_dir() {
            return Err(Error::InvalidDataFormat {
                details: format!("scan root is not a folder: {root}"),
            });
        }

        let mut file_paths = Vec::new();
        collect_files_recursive(root_path, &mut file_paths, options)?;
        file_paths.sort();

        let mut per_file = Vec::with_capacity(file_paths.len());
        let mut failures = Vec::new();

        for path in file_paths {
            match scan_one_file(&path, options.max_bytes_per_file) {
                Ok(result) => per_file.push(result),
                Err(error) => failures.push(FailedFile {
                    path: path.to_string_lossy().to_string(),
                    error,
                }),
            }
        }

        let modality_stats = build_modality_stats(&per_file);
        let overall_confidence = confidence_stats_from(per_file.iter().map(|item| item.detection.confidence));

        Ok(BatchScanReport {
            root: root.to_string(),
            files_scanned: per_file.len(),
            files_failed: failures.len(),
            per_file,
            failures,
            modality_stats,
            overall_confidence,
        })
    }
}

impl TrainingDataLoader {
    pub fn load_from_path(path: &str, config: &LoadConfig) -> Result<TensorDataset> {
        let bytes = fs::read(path).map_err(|error| Error::InvalidDataFormat {
            details: format!("cannot read dataset path {path}: {error}"),
        })?;
        Self::load_from_bytes(Some(path), &bytes, config)
    }

    pub fn load_from_bytes(path: Option<&str>, bytes: &[u8], config: &LoadConfig) -> Result<TensorDataset> {
        let detected = DataTypeDetector::detect(path, bytes);
        match detected.data_type {
            DataType::Text => {
                let text = std::str::from_utf8(bytes).map_err(|error| Error::InvalidDataFormat {
                    details: format!("dataset is not utf8 text: {error}"),
                })?;
                Self::load_from_text(text, config)
            }
            DataType::Empty => Err(Error::InvalidDataFormat {
                details: "dataset payload is empty".to_string(),
            }),
            _ => Err(Error::UnsupportedDataType {
                details: format!(
                    "currently only text/tabular datasets are supported for tensor loading, detected {:?}",
                    detected.data_type
                ),
            }),
        }
    }

    pub fn load_from_text(text: &str, config: &LoadConfig) -> Result<TensorDataset> {
        let parsed = parse_csv_numeric(text, config)?;
        let features = Tensor::new(vec![parsed.rows, parsed.feature_cols], parsed.features)?;
        let targets = if let Some(target_data) = parsed.targets {
            Some(Tensor::new(vec![parsed.rows, 1], target_data)?)
        } else {
            None
        };

        Ok(TensorDataset {
            features,
            targets,
            feature_names: parsed.feature_names,
            target_name: parsed.target_name,
        })
    }

    pub fn train_valid_split(
        dataset: &TensorDataset,
        valid_ratio: f32,
    ) -> Result<(TensorDataset, TensorDataset)> {
        if !(0.0..1.0).contains(&valid_ratio) {
            return Err(Error::InvalidDataFormat {
                details: format!("valid_ratio must be in [0,1), got {valid_ratio}"),
            });
        }

        let rows = dataset.features.shape().dims()[0];
        let cols = dataset.features.shape().dims()[1];
        if rows < 2 {
            return Err(Error::InvalidDataFormat {
                details: "at least 2 rows are required to split train/validation".to_string(),
            });
        }

        let valid_rows = ((rows as f32) * valid_ratio).round() as usize;
        let valid_rows = valid_rows.min(rows - 1);
        let train_rows = rows - valid_rows;

        let feature_data = dataset.features.data();
        let train_feature_data = feature_data[..train_rows * cols].to_vec();
        let valid_feature_data = feature_data[train_rows * cols..].to_vec();

        let train_features = Tensor::new(vec![train_rows, cols], train_feature_data)?;
        let valid_features = Tensor::new(vec![valid_rows, cols], valid_feature_data)?;

        let (train_targets, valid_targets) = if let Some(targets) = &dataset.targets {
            let target_data = targets.data();
            let train_target_data = target_data[..train_rows].to_vec();
            let valid_target_data = target_data[train_rows..].to_vec();
            (
                Some(Tensor::new(vec![train_rows, 1], train_target_data)?),
                Some(Tensor::new(vec![valid_rows, 1], valid_target_data)?),
            )
        } else {
            (None, None)
        };

        let train = TensorDataset {
            features: train_features,
            targets: train_targets,
            feature_names: dataset.feature_names.clone(),
            target_name: dataset.target_name.clone(),
        };
        let valid = TensorDataset {
            features: valid_features,
            targets: valid_targets,
            feature_names: dataset.feature_names.clone(),
            target_name: dataset.target_name.clone(),
        };

        Ok((train, valid))
    }
}

#[derive(Debug, Clone)]
struct ParsedTabular {
    rows: usize,
    feature_cols: usize,
    features: Vec<f32>,
    targets: Option<Vec<f32>>,
    feature_names: Vec<String>,
    target_name: Option<String>,
}

fn parse_csv_numeric(text: &str, config: &LoadConfig) -> Result<ParsedTabular> {
    let all_lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    if all_lines.is_empty() {
        return Err(Error::InvalidDataFormat {
            details: "dataset has no non-empty rows".to_string(),
        });
    }

    let (header_opt, data_lines) = if config.has_header {
        let header = split_row(all_lines[0], config.delimiter);
        (Some(header), &all_lines[1..])
    } else {
        (None, &all_lines[..])
    };

    if data_lines.is_empty() {
        return Err(Error::InvalidDataFormat {
            details: "dataset has header but no data rows".to_string(),
        });
    }

    let first_fields = split_row(data_lines[0], config.delimiter);
    let total_cols = first_fields.len();
    if total_cols == 0 {
        return Err(Error::InvalidDataFormat {
            details: "dataset first row has zero columns".to_string(),
        });
    }

    if let Some(target_idx) = config.target_column {
        if target_idx >= total_cols {
            return Err(Error::InvalidDataFormat {
                details: format!(
                    "target_column {target_idx} out of bounds for {total_cols} columns"
                ),
            });
        }
        if total_cols < 2 {
            return Err(Error::InvalidDataFormat {
                details: "target_column requires at least 2 columns".to_string(),
            });
        }
    }

    let feature_cols = if config.target_column.is_some() {
        total_cols - 1
    } else {
        total_cols
    };

    let mut features = Vec::with_capacity(data_lines.len() * feature_cols);
    let mut targets = config.target_column.map(|_| Vec::with_capacity(data_lines.len()));

    for (row_index, line) in data_lines.iter().enumerate() {
        let fields = split_row(line, config.delimiter);
        if fields.len() != total_cols {
            return Err(Error::InvalidDataFormat {
                details: format!(
                    "row {} has {} columns, expected {}",
                    row_index + 1,
                    fields.len(),
                    total_cols
                ),
            });
        }

        for (col_index, raw) in fields.iter().enumerate() {
            let value = raw.parse::<f32>().map_err(|error| Error::InvalidDataFormat {
                details: format!(
                    "cannot parse numeric value at row {}, col {}: '{}' ({error})",
                    row_index + 1,
                    col_index + 1,
                    raw
                ),
            })?;

            if Some(col_index) == config.target_column {
                if let Some(target_vec) = &mut targets {
                    target_vec.push(value);
                }
            } else {
                features.push(value);
            }
        }
    }

    apply_normalization(&mut features, data_lines.len(), feature_cols, &config.normalization);

    let header_names = header_opt.map(|parts| {
        parts
            .into_iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
    });
    let (feature_names, target_name) = derive_column_names(header_names, total_cols, config.target_column);

    Ok(ParsedTabular {
        rows: data_lines.len(),
        feature_cols,
        features,
        targets,
        feature_names,
        target_name,
    })
}

fn split_row(line: &str, delimiter: char) -> Vec<String> {
    line.split(delimiter)
        .map(|field| field.trim().trim_matches('"').to_string())
        .collect()
}

fn derive_column_names(
    header: Option<Vec<String>>,
    total_cols: usize,
    target_column: Option<usize>,
) -> (Vec<String>, Option<String>) {
    let default_names: Vec<String> = (0..total_cols).map(|idx| format!("col_{idx}")).collect();
    let names = header.unwrap_or(default_names);

    if let Some(target_idx) = target_column {
        let mut feature_names = Vec::with_capacity(total_cols - 1);
        let mut target_name = None;
        for (idx, name) in names.into_iter().enumerate() {
            if idx == target_idx {
                target_name = Some(name);
            } else {
                feature_names.push(name);
            }
        }
        (feature_names, target_name)
    } else {
        (names, None)
    }
}

fn apply_normalization(values: &mut [f32], rows: usize, cols: usize, normalization: &Normalization) {
    match normalization {
        Normalization::None => {}
        Normalization::Standard => normalize_standard(values, rows, cols),
        Normalization::MinMax => normalize_minmax(values, rows, cols),
    }
}

fn normalize_standard(values: &mut [f32], rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    for col in 0..cols {
        let mean = (0..rows).map(|r| values[r * cols + col]).sum::<f32>() / rows as f32;
        let variance = (0..rows)
            .map(|r| {
                let diff = values[r * cols + col] - mean;
                diff * diff
            })
            .sum::<f32>()
            / rows as f32;
        let stddev = variance.sqrt();
        if stddev <= f32::EPSILON {
            continue;
        }
        for row in 0..rows {
            let idx = row * cols + col;
            values[idx] = (values[idx] - mean) / stddev;
        }
    }
}

fn normalize_minmax(values: &mut [f32], rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    for col in 0..cols {
        let mut min_v = values[col];
        let mut max_v = values[col];
        for row in 1..rows {
            let v = values[row * cols + col];
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }

        let range = max_v - min_v;
        if range <= f32::EPSILON {
            continue;
        }
        for row in 0..rows {
            let idx = row * cols + col;
            values[idx] = (values[idx] - min_v) / range;
        }
    }
}

fn collect_files_recursive(root: &Path, output: &mut Vec<PathBuf>, options: &ScanOptions) -> Result<()> {
    let entries = fs::read_dir(root).map_err(|error| Error::InvalidDataFormat {
        details: format!("cannot read directory {}: {error}", root.display()),
    })?;

    for entry_result in entries {
        let entry = entry_result.map_err(|error| Error::InvalidDataFormat {
            details: format!("cannot read entry in {}: {error}", root.display()),
        })?;
        let path = entry.path();
        let metadata = entry.metadata().map_err(|error| Error::InvalidDataFormat {
            details: format!("cannot read metadata {}: {error}", path.display()),
        })?;

        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();

        if !options.include_hidden && file_name.starts_with('.') {
            continue;
        }

        if metadata.is_dir() {
            if options
                .excluded_dir_names
                .iter()
                .any(|excluded| excluded == file_name)
            {
                continue;
            }
            collect_files_recursive(&path, output, options)?;
        } else if metadata.is_file() {
            output.push(path);
        }
    }

    Ok(())
}

fn scan_one_file(path: &Path, max_bytes_per_file: usize) -> std::result::Result<FileScanResult, String> {
    let metadata = fs::metadata(path).map_err(|error| format!("cannot stat file: {error}"))?;
    let size_bytes = metadata.len();

    let mut file = fs::File::open(path).map_err(|error| format!("cannot open file: {error}"))?;
    let mut sample = Vec::new();
    let mut limited = file.by_ref().take(max_bytes_per_file as u64);
    limited
        .read_to_end(&mut sample)
        .map_err(|error| format!("cannot read file sample: {error}"))?;

    let detection = DataTypeDetector::detect(path.to_str(), &sample);
    Ok(FileScanResult {
        path: path.to_string_lossy().to_string(),
        size_bytes,
        detection,
    })
}

fn build_modality_stats(per_file: &[FileScanResult]) -> Vec<ModalityStats> {
    if per_file.is_empty() {
        return Vec::new();
    }

    let mut grouped: BTreeMap<DataType, Vec<f32>> = BTreeMap::new();
    for item in per_file {
        grouped
            .entry(item.detection.data_type.clone())
            .or_default()
            .push(item.detection.confidence);
    }

    let total = per_file.len() as f32;
    grouped
        .into_iter()
        .map(|(data_type, confidences)| {
            let count = confidences.len();
            let confidence = confidence_stats_from(confidences.into_iter()).unwrap_or(ConfidenceStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
            });
            ModalityStats {
                data_type,
                count,
                ratio: count as f32 / total,
                confidence,
            }
        })
        .collect()
}

fn confidence_stats_from<I>(values: I) -> Option<ConfidenceStats>
where
    I: IntoIterator<Item = f32>,
{
    let mut iter = values.into_iter();
    let first = iter.next()?;
    let mut min = first;
    let mut max = first;
    let mut sum = first;
    let mut count = 1usize;

    for value in iter {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
        sum += value;
        count += 1;
    }

    Some(ConfidenceStats {
        min,
        max,
        mean: sum / count as f32,
    })
}

fn detect_image_from_signature(bytes: &[u8]) -> Option<Detection> {
    if bytes.len() >= 8 && bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(found(DataType::Image, DataFormat::Png, 1.0, "png signature"));
    }
    if bytes.len() >= 3 && bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(found(DataType::Image, DataFormat::Jpeg, 1.0, "jpeg signature"));
    }
    if bytes.len() >= 6 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return Some(found(DataType::Image, DataFormat::Gif, 1.0, "gif signature"));
    }
    if bytes.len() >= 2 && bytes.starts_with(b"BM") {
        return Some(found(DataType::Image, DataFormat::Bmp, 0.98, "bmp signature"));
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some(found(DataType::Image, DataFormat::Webp, 1.0, "webp signature"));
    }
    if bytes.len() >= 4
        && (bytes.starts_with(&[b'I', b'I', 0x2A, 0x00])
            || bytes.starts_with(&[b'M', b'M', 0x00, 0x2A]))
    {
        return Some(found(DataType::Image, DataFormat::Tiff, 0.98, "tiff signature"));
    }
    None
}

fn detect_sound_from_signature(bytes: &[u8]) -> Option<Detection> {
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        return Some(found(DataType::Sound, DataFormat::Wav, 1.0, "wav signature"));
    }
    if bytes.len() >= 3 && bytes.starts_with(b"ID3") {
        return Some(found(DataType::Sound, DataFormat::Mp3, 0.98, "mp3 id3 header"));
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return Some(found(DataType::Sound, DataFormat::Mp3, 0.9, "mp3 frame sync"));
    }
    if bytes.len() >= 4 && bytes.starts_with(b"fLaC") {
        return Some(found(DataType::Sound, DataFormat::Flac, 1.0, "flac signature"));
    }
    if bytes.len() >= 4 && bytes.starts_with(b"OggS") {
        return Some(found(DataType::Sound, DataFormat::Ogg, 0.98, "ogg signature"));
    }
    if bytes.len() >= 4 && bytes.starts_with(b"MThd") {
        return Some(found(DataType::Sound, DataFormat::Midi, 0.98, "midi signature"));
    }
    None
}

fn detect_text_format(text: &str) -> Option<Detection> {
    let trimmed = text.trim();

    if looks_like_json(trimmed) {
        return Some(found(DataType::Text, DataFormat::Json, 0.9, "json-like structure"));
    }

    if looks_like_csv(trimmed) {
        return Some(found(DataType::Text, DataFormat::Csv, 0.85, "csv-like rows"));
    }

    None
}

fn detect_graph_from_text(text: &str) -> Option<Detection> {
    let lower = text.to_ascii_lowercase();
    let trimmed = lower.trim();

    if trimmed.contains("<graphml") || (trimmed.contains("<node") && trimmed.contains("<edge")) {
        return Some(found(DataType::Graph, DataFormat::GraphMl, 0.95, "graphml tags"));
    }

    if trimmed.contains("graph [") && trimmed.contains("node [") {
        return Some(found(DataType::Graph, DataFormat::Gml, 0.9, "gml pattern"));
    }

    if (trimmed.contains("digraph") || trimmed.contains("graph "))
        && trimmed.contains('{')
        && (trimmed.contains("->") || trimmed.contains("--"))
    {
        return Some(found(DataType::Graph, DataFormat::Dot, 0.92, "dot/graphviz syntax"));
    }

    if (trimmed.contains("\"nodes\"") && trimmed.contains("\"edges\""))
        || (trimmed.contains("'nodes'") && trimmed.contains("'edges'"))
    {
        return Some(found(
            DataType::Graph,
            DataFormat::Unknown,
            0.8,
            "graph-like json keys",
        ));
    }

    if looks_like_edge_list(trimmed) {
        return Some(found(DataType::Graph, DataFormat::EdgeList, 0.78, "edge-list rows"));
    }

    if looks_like_adj_list(trimmed) {
        return Some(found(DataType::Graph, DataFormat::AdjList, 0.75, "adjacency-list rows"));
    }

    None
}

fn detect_from_extension(path: &str) -> Option<Detection> {
    let ext = path.rsplit('.').next()?.to_ascii_lowercase();
    match ext.as_str() {
        "txt" | "md" | "rst" | "log" => {
            Some(found(DataType::Text, DataFormat::PlainText, 0.55, "text extension"))
        }
        "json" => Some(found(DataType::Text, DataFormat::Json, 0.55, "json extension")),
        "csv" | "tsv" => Some(found(DataType::Text, DataFormat::Csv, 0.55, "tabular extension")),
        "png" => Some(found(DataType::Image, DataFormat::Png, 0.6, "image extension")),
        "jpg" | "jpeg" => Some(found(DataType::Image, DataFormat::Jpeg, 0.6, "image extension")),
        "gif" => Some(found(DataType::Image, DataFormat::Gif, 0.6, "image extension")),
        "bmp" => Some(found(DataType::Image, DataFormat::Bmp, 0.6, "image extension")),
        "webp" => Some(found(DataType::Image, DataFormat::Webp, 0.6, "image extension")),
        "tif" | "tiff" => Some(found(DataType::Image, DataFormat::Tiff, 0.6, "image extension")),
        "wav" => Some(found(DataType::Sound, DataFormat::Wav, 0.6, "sound extension")),
        "mp3" => Some(found(DataType::Sound, DataFormat::Mp3, 0.6, "sound extension")),
        "flac" => Some(found(DataType::Sound, DataFormat::Flac, 0.6, "sound extension")),
        "ogg" => Some(found(DataType::Sound, DataFormat::Ogg, 0.6, "sound extension")),
        "mid" | "midi" => Some(found(DataType::Sound, DataFormat::Midi, 0.6, "sound extension")),
        "dot" | "gv" => Some(found(DataType::Graph, DataFormat::Dot, 0.65, "graph extension")),
        "graphml" => Some(found(DataType::Graph, DataFormat::GraphMl, 0.65, "graph extension")),
        "gml" => Some(found(DataType::Graph, DataFormat::Gml, 0.65, "graph extension")),
        "edgelist" => Some(found(
            DataType::Graph,
            DataFormat::EdgeList,
            0.65,
            "graph extension",
        )),
        "adjlist" => Some(found(
            DataType::Graph,
            DataFormat::AdjList,
            0.65,
            "graph extension",
        )),
        _ => None,
    }
}

fn decode_text_if_likely(bytes: &[u8]) -> Option<String> {
    let text = String::from_utf8(bytes.to_vec()).ok()?;
    if text.trim().is_empty() {
        return Some(text);
    }

    let printable = text
        .chars()
        .filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
        .count();
    let ratio = printable as f32 / text.chars().count() as f32;
    if ratio >= 0.85 {
        Some(text)
    } else {
        None
    }
}

fn looks_like_json(text: &str) -> bool {
    (text.starts_with('{') && text.ends_with('}')) || (text.starts_with('[') && text.ends_with(']'))
}

fn looks_like_csv(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).take(6).collect();
    if lines.len() < 2 {
        return false;
    }

    let first_count = lines[0].split(',').count();
    if first_count < 2 {
        return false;
    }

    lines.iter().all(|line| line.split(',').count() == first_count)
}

fn looks_like_edge_list(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).take(12).collect();
    if lines.len() < 3 {
        return false;
    }

    let mut edge_like = 0;
    for line in lines {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if (tokens.len() == 2 || tokens.len() == 3)
            && tokens
                .iter()
                .all(|token| token.chars().all(is_edge_token_char))
        {
            edge_like += 1;
        }
    }

    edge_like >= 3
}

fn is_edge_token_char(value: char) -> bool {
    value.is_ascii_alphanumeric() || value == '_' || value == '-' || value == '.'
}

fn looks_like_adj_list(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).take(12).collect();
    if lines.len() < 3 {
        return false;
    }

    let mut matches = 0;
    for line in lines {
        if line.contains(':') {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() == 2
                && !parts[0].trim().is_empty()
                && !parts[1].trim().is_empty()
                && parts[0].trim().chars().all(is_edge_token_char)
                && parts[1]
                    .split_whitespace()
                    .all(|token| token.chars().all(is_edge_token_char))
            {
                matches += 1;
            }
        }
    }
    matches >= 3
}

fn found(data_type: DataType, format: DataFormat, confidence: f32, reason: &str) -> Detection {
    Detection {
        data_type,
        format,
        confidence,
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn detects_plain_text() {
        let input = b"hello world\nthis is text";
        let out = DataTypeDetector::detect(None, input);
        assert_eq!(out.data_type, DataType::Text);
        assert_eq!(out.format, DataFormat::PlainText);
    }

    #[test]
    fn detects_json_text() {
        let input = b"{\"name\":\"kpodjito\",\"ok\":true}";
        let out = DataTypeDetector::detect(None, input);
        assert_eq!(out.data_type, DataType::Text);
        assert_eq!(out.format, DataFormat::Json);
    }

    #[test]
    fn detects_png() {
        let input = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        let out = DataTypeDetector::detect(None, &input);
        assert_eq!(out.data_type, DataType::Image);
        assert_eq!(out.format, DataFormat::Png);
    }

    #[test]
    fn detects_wav() {
        let input = [b'R', b'I', b'F', b'F', 0, 0, 0, 0, b'W', b'A', b'V', b'E', 0, 0];
        let out = DataTypeDetector::detect(None, &input);
        assert_eq!(out.data_type, DataType::Sound);
        assert_eq!(out.format, DataFormat::Wav);
    }

    #[test]
    fn detects_graphviz_dot() {
        let input = b"digraph G { A -> B; B -> C; }";
        let out = DataTypeDetector::detect(None, input);
        assert_eq!(out.data_type, DataType::Graph);
        assert_eq!(out.format, DataFormat::Dot);
    }

    #[test]
    fn detects_graphml() {
        let input = b"<graphml><graph><node id=\"n1\"/><edge source=\"n1\" target=\"n2\"/></graph></graphml>";
        let out = DataTypeDetector::detect(None, input);
        assert_eq!(out.data_type, DataType::Graph);
        assert_eq!(out.format, DataFormat::GraphMl);
    }

    #[test]
    fn uses_extension_when_content_is_empty() {
        let out = DataTypeDetector::detect(Some("dataset.graphml"), &[]);
        assert_eq!(out.data_type, DataType::Graph);
        assert_eq!(out.format, DataFormat::GraphMl);
    }

    #[test]
    fn returns_unknown_for_unmatched_binary() {
        let input = [0x80, 0x81, 0xA0, 0x00, 0xFF, 0x10, 0x22, 0x01];
        let out = DataTypeDetector::detect(None, &input);
        assert_eq!(out.data_type, DataType::Unknown);
    }

    #[test]
    fn scan_folder_reports_distribution_and_confidence() {
        let root = unique_test_dir("batch_scan");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(root.join("nested")).unwrap();

        fs::write(root.join("notes.txt"), b"hello scanner").unwrap();
        fs::write(
            root.join("img.png"),
            [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00],
        )
        .unwrap();
        fs::write(
            root.join("nested").join("audio.wav"),
            [b'R', b'I', b'F', b'F', 0, 0, 0, 0, b'W', b'A', b'V', b'E', 0, 0],
        )
        .unwrap();
        fs::write(
            root.join("nested").join("graph.dot"),
            b"digraph G { A -> B; }",
        )
        .unwrap();

        let report = DataTypeDetector::scan_folder(root.to_str().unwrap()).unwrap();
        assert_eq!(report.files_scanned, 4);
        assert_eq!(report.files_failed, 0);
        assert_eq!(report.per_file.len(), 4);
        assert!(report.overall_confidence.is_some());

        let mut counts = BTreeMap::<DataType, usize>::new();
        for stat in &report.modality_stats {
            counts.insert(stat.data_type.clone(), stat.count);
        }
        assert_eq!(counts.get(&DataType::Text), Some(&1));
        assert_eq!(counts.get(&DataType::Image), Some(&1));
        assert_eq!(counts.get(&DataType::Sound), Some(&1));
        assert_eq!(counts.get(&DataType::Graph), Some(&1));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scan_folder_rejects_missing_path() {
        let bad_path = "/tmp/kpodjito-no-such-folder-for-scan";
        let result = DataTypeDetector::scan_folder(bad_path);
        assert!(result.is_err());
    }

    #[test]
    fn scan_folder_skips_hidden_and_excluded_by_default() {
        let root = unique_test_dir("batch_scan_hidden");
        fs::create_dir_all(root.join(".hidden")).unwrap();
        fs::create_dir_all(root.join("target")).unwrap();

        fs::write(root.join("visible.txt"), b"visible").unwrap();
        fs::write(root.join(".hidden").join("secret.txt"), b"hidden").unwrap();
        fs::write(root.join("target").join("artifact.bin"), [0_u8, 1_u8]).unwrap();

        let report = DataTypeDetector::scan_folder(root.to_str().unwrap()).unwrap();
        assert_eq!(report.files_scanned, 1);
        assert!(report
            .per_file
            .iter()
            .all(|entry| entry.path.ends_with("visible.txt")));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_csv_into_feature_and_target_tensors() {
        let csv = "f1,f2,target\n1.0,10.0,0\n2.0,20.0,1\n3.0,30.0,0\n";
        let config = LoadConfig {
            target_column: Some(2),
            ..LoadConfig::default()
        };

        let dataset = TrainingDataLoader::load_from_text(csv, &config).unwrap();
        assert_eq!(dataset.features.shape().dims(), &[3, 2]);
        assert_eq!(dataset.targets.as_ref().unwrap().shape().dims(), &[3, 1]);
        assert_eq!(dataset.feature_names, vec!["f1".to_string(), "f2".to_string()]);
        assert_eq!(dataset.target_name, Some("target".to_string()));
        assert_eq!(dataset.features.data(), &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        assert_eq!(dataset.targets.as_ref().unwrap().data(), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn applies_standard_normalization() {
        let csv = "a,b\n1.0,10.0\n2.0,20.0\n3.0,30.0\n";
        let config = LoadConfig {
            target_column: None,
            normalization: Normalization::Standard,
            ..LoadConfig::default()
        };

        let dataset = TrainingDataLoader::load_from_text(csv, &config).unwrap();
        let data = dataset.features.data();
        // First column should be approximately [-1.2247, 0, 1.2247]
        assert!((data[0] + 1.2247449).abs() < 1e-4);
        assert!(data[2].abs() < 1e-6);
        assert!((data[4] - 1.2247449).abs() < 1e-4);
    }

    #[test]
    fn splits_train_and_validation_sets() {
        let csv = "x,y,target\n1,10,0\n2,20,1\n3,30,0\n4,40,1\n5,50,0\n";
        let config = LoadConfig {
            target_column: Some(2),
            ..LoadConfig::default()
        };

        let dataset = TrainingDataLoader::load_from_text(csv, &config).unwrap();
        let (train, valid) = TrainingDataLoader::train_valid_split(&dataset, 0.4).unwrap();

        assert_eq!(train.features.shape().dims(), &[3, 2]);
        assert_eq!(valid.features.shape().dims(), &[2, 2]);
        assert_eq!(train.targets.as_ref().unwrap().shape().dims(), &[3, 1]);
        assert_eq!(valid.targets.as_ref().unwrap().shape().dims(), &[2, 1]);
    }

    #[test]
    fn rejects_non_numeric_csv_values() {
        let csv = "x,y\n1.0,cat\n";
        let config = LoadConfig::default();
        let result = TrainingDataLoader::load_from_text(csv, &config);
        assert!(result.is_err());
    }

    fn unique_test_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kpodjito-{prefix}-{nanos}"))
    }
}
