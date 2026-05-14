use kpodjito_core_autograd::Variable;
use kpodjito_core_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Phase 2: Automatic Differentiation Demo ===\n");

    // Example 1: Simple scalar computation
    println!("Example 1: Forward and Backward Pass");
    println!("--------------------------------------");
    let x = Variable::new(Tensor::new(vec![1], vec![2.0_f32])?);
    let y = Variable::new(Tensor::new(vec![1], vec![3.0_f32])?);
    
    // f(x,y) = x*y + x
    let xy = x.mul(&y)?;
    let f = xy.add(&x)?;
    
    println!("x = {}", x.tensor().data()[0]);
    println!("y = {}", y.tensor().data()[0]);
    println!("f = x*y + x = {}", f.tensor().data()[0]);
    
    // Backward pass
    f.backward()?;
    println!("\nGradients via backpropagation:");
    println!("df/dx = y + 1 = {}", x.grad().unwrap().data()[0]);
    println!("df/dy = x = {}", y.grad().unwrap().data()[0]);
    
    // Example 2: Matrix multiplication with gradient
    println!("\n\nExample 2: Matrix Operations");
    println!("-----------------------------");
    let a = Variable::new(
        Tensor::new(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0])?
    );
    let b = Variable::new(
        Tensor::new(vec![2, 2], vec![5.0_f32, 6.0, 7.0, 8.0])?
    );
    
    println!("A =");
    print_matrix(a.tensor().data(), 2, 2);
    println!("\nB =");
    print_matrix(b.tensor().data(), 2, 2);
    
    let c = a.matmul(&b)?;
    println!("\nC = A @ B =");
    print_matrix(c.tensor().data(), 2, 2);
    
    let loss = c.sum();
    loss.backward()?;
    
    println!("\nAfter backward pass (dL/dA):");
    let grad_a = a.grad().unwrap();
    print_matrix(grad_a.data(), 2, 2);
    
    // Example 3: Chained operations with mean
    println!("\n\nExample 3: Chain Rule with Mean");
    println!("--------------------------------");
    let x = Variable::new(
        Tensor::new(vec![4], vec![1.0_f32, 2.0, 3.0, 4.0])?
    );
    
    let y = x.mul_scalar(2.0);
    let z = y.mean()?;
    
    println!("x = {:?}", x.tensor().data());
    println!("y = 2*x = {:?}", y.tensor().data());
    println!("z = mean(y) = {}", z.tensor().data()[0]);
    
    z.backward()?;
    println!("\ndz/dx = 2/n = {:?}", x.grad().unwrap().data());
    
    // Example 4: Broadcasting in gradients
    println!("\n\nExample 4: Vector-wise Operations");
    println!("----------------------------------");
    let a = Variable::new(
        Tensor::new(vec![2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?
    );
    let b = Variable::new(
        Tensor::new(vec![2, 3], vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0])?
    );
    
    println!("A shape: {:?}", a.tensor().shape().dims());
    println!("B shape: {:?}", b.tensor().shape().dims());
    
    let c = a.add(&b)?;
    println!("C = A + B");
    println!("C shape: {:?}", c.tensor().shape().dims());
    println!("C =");
    print_matrix(c.tensor().data(), 2, 3);
    
    let loss = c.sum();
    loss.backward()?;
    
    println!("\nGradient of B:");
    let grad_b = b.grad().unwrap();
    println!("grad_b shape: {:?}", grad_b.shape().dims());
    println!("grad_b = {:?}", grad_b.data());
    
    // Example 5: Gradient checking (numerical vs analytical)
    println!("\n\nExample 5: Numerical Gradient Checking");
    println!("--------------------------------------");
    numerical_gradient_check()?;
    
    println!("\n=== All Examples Completed Successfully ===");
    Ok(())
}

fn print_matrix(data: &[f32], rows: usize, cols: usize) {
    for i in 0..rows {
        print!("[");
        for j in 0..cols {
            print!("{:6.2}", data[i * cols + j]);
            if j < cols - 1 {
                print!(", ");
            }
        }
        println!(" ]");
    }
}

fn numerical_gradient_check() -> Result<(), Box<dyn std::error::Error>> {
    let h = 1e-4_f32;
    
    // f(x) = x^2 + 2*x, df/dx = 2*x + 2
    let x_val = 3.0_f32;
    
    // Analytical gradient: df/dx = 2*x + 2 = 2*3 + 2 = 8
    let x = Variable::new(Tensor::new(vec![1], vec![x_val])?);
    
    // f(x) = x*x + 2*x
    let xx = x.mul(&x)?;
    let f = xx.add(&x.mul_scalar(2.0))?;
    f.backward()?;
    let analytical_grad = x.grad().unwrap().data()[0];
    
    // Numerical gradient: (f(x+h) - f(x-h)) / 2h
    let f_plus_val = (x_val + h).powi(2) + 2.0 * (x_val + h);
    let f_minus_val = (x_val - h).powi(2) + 2.0 * (x_val - h);
    let numerical_grad = (f_plus_val - f_minus_val) / (2.0 * h);
    
    println!("x = {}", x_val);
    println!("f(x) = x^2 + 2*x = {}", f.tensor().data()[0]);
    println!("Analytical df/dx = {}", analytical_grad);
    println!("Numerical df/dx = {}", numerical_grad);
    
    let error = (analytical_grad - numerical_grad).abs();
    println!("Gradient error = {}", error);
    println!("✓ Gradient check passed (error < 1e-2): {}", error < 1e-2);
    
    Ok(())
}
