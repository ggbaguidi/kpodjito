use kpodjito_core_math::{dot, l2_norm, mean, softmax, sum};

fn main() {
    println!("=== Core Math Operations ===");
    println!();

    // Dot product
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let d = dot(&a, &b).expect("dot failed");
    println!("dot([1,2,3], [4,5,6]) = {}", d);

    // Sum and mean
    let vals = [1.0_f32, 2.0, 3.0, 4.0];
    let s = sum(&vals);
    let m = mean(&vals).expect("mean failed");
    println!("sum([1,2,3,4]) = {}", s);
    println!("mean([1,2,3,4]) = {}", m);

    // L2 norm
    let v = [3.0_f32, 4.0];
    let norm = l2_norm(&v).expect("norm failed");
    println!("l2_norm([3,4]) = {} (expected 5.0)", norm);

    // Softmax
    println!();
    let logits = [1.0_f64, 2.0, 3.0];
    let probs = softmax(&logits).expect("softmax failed");
    println!("softmax([1,2,3]) = {:.4?}", probs);
    let total: f64 = probs.iter().sum();
    println!("sum of probabilities = {:.10} (should be 1.0)", total);

    println!();
    println!("All math operations completed successfully!");
}
