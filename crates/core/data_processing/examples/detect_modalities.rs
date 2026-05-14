use kpodjito_core_data_processing::DataTypeDetector;

fn main() {
    let samples: [(&str, Vec<u8>); 4] = [
        ("notes.txt", b"A short text payload for NLP tasks.".to_vec()),
        (
            "image.png",
            vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x01],
        ),
        (
            "speech.wav",
            vec![b'R', b'I', b'F', b'F', 0, 0, 0, 0, b'W', b'A', b'V', b'E'],
        ),
        (
            "network.dot",
            b"digraph G { user -> item; item -> category; }".to_vec(),
        ),
    ];

    println!("=== Automatic Data Type Detection ===");
    for (path, payload) in samples {
        let detection = DataTypeDetector::detect(Some(path), &payload);
        println!(
            "{path}: {:?} / {:?} (confidence {:.2}) - {}",
            detection.data_type, detection.format, detection.confidence, detection.reason
        );
    }
}
