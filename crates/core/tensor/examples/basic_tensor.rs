use kpodjito_core_tensor::Tensor;

fn main() {
    println!("=== Tensor Operations ===");
    println!();

    // Elementwise operations
    println!("--- Elementwise Operations ---");
    let a = Tensor::new(vec![2], vec![1.0_f32, 2.0]).expect("tensor a");
    let b = Tensor::new(vec![2], vec![3.0_f32, 4.0]).expect("tensor b");
    let c = a.add(&b).expect("add");
    println!("a = {:?}", a);
    println!("b = {:?}", b);
    println!("a + b = {:?}", c);
    println!();

    // Matrix multiplication
    println!("--- Matrix Multiplication ---");
    let m = Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).expect("m");
    let n = Tensor::new(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).expect("n");
    let p = m.matmul(&n).expect("matmul");
    println!("M = {:?}", m);
    println!("N = {:?}", n);
    println!("M @ N = {:?}", p);
    println!();

    // Transpose
    println!("--- Transpose ---");
    let mt = m.transpose().expect("transpose");
    println!("M^T = {:?}", mt);
    println!();

    // Diagonal and trace
    println!("--- Diagonal and Trace ---");
    let diag = m.diagonal().expect("diagonal");
    println!("diag(M) = {:?}", diag);
    let tr = m.trace().expect("trace");
    println!("trace(M) = {}", tr);
    println!();

    // Determinant and inverse
    println!("--- Determinant and Inverse ---");
    let det = m.determinant().expect("determinant");
    println!("det(M) = {}", det);
    let inv = m.inverse().expect("inverse");
    println!("M^-1 = {:?}", inv);
    // verify M @ M^-1 ≈ I
    let prod = m.matmul(&inv).expect("matmul");
    println!("M @ M^-1 = {:?} (should be ~identity)", prod);
    println!();

    // Broadcasting
    println!("--- Broadcasting ---");
    let x = Tensor::new(vec![2, 1], vec![1.0_f32, 2.0]).expect("x");
    let y = Tensor::new(vec![1, 3], vec![10.0_f32, 20.0, 30.0]).expect("y");
    let z = x.add_broadcast(&y).expect("broadcast add");
    println!("x (shape [2,1]) = {:?}", x);
    println!("y (shape [1,3]) = {:?}", y);
    println!("x + y (shape [2,3]) = {:?}", z);
    println!();

    // Power iteration (dominant eigenvalue/eigenvector)
    println!("--- Power Iteration (Dominant Eigenvalue) ---");
    let eig_matrix = Tensor::new(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 1.0]).expect("eig_matrix");
    let (lambda, eigvec) = eig_matrix.power_iteration(50, 1e-9).expect("power iteration");
    println!("Matrix = {:?}", eig_matrix);
    println!("Dominant eigenvalue λ = {}", lambda);
    println!("Corresponding eigenvector = {:?}", eigvec);
    println!();

    // Reductions
    println!("--- Reductions ---");
    let mean_a = a.mean().expect("mean");
    let sum_a = a.sum();
    println!("a = {:?}", a);
    println!("mean(a) = {}", mean_a);
    println!("sum(a) = {}", sum_a);
    println!();

    println!("All tensor operations completed successfully!");
}
