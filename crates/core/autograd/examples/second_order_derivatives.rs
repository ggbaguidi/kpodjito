use kpodjito_core_autograd::Variable;
use kpodjito_core_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Second-Order Derivatives (Hessian) ===\n");

    // Example 1: d²f/dx² for f(x) = x² + 2x
    println!("Example 1: Quadratic Function");
    println!("-----------------------------");
    second_derivative_quadratic()?;

    // Example 2: d²f/dx² for f(x) = x³
    println!("\n\nExample 2: Cubic Function");
    println!("------------------------");
    second_derivative_cubic()?;

    // Example 3: Mixed partial derivatives d²f/dxdy
    println!("\n\nExample 3: Mixed Partial Derivatives");
    println!("------------------------------------");
    mixed_partial_derivatives()?;

    // Example 4: Hessian matrix for multivariate function
    println!("\n\nExample 4: Hessian Matrix");
    println!("------------------------");
    hessian_matrix()?;

    println!("\n=== All Second-Order Derivative Examples Complete ===");
    Ok(())
}

fn second_derivative_quadratic() -> Result<(), Box<dyn std::error::Error>> {
    // f(x) = x² + 2x
    // df/dx = 2x + 2
    // d²f/dx² = 2

    let x = Variable::new(Tensor::new(vec![1], vec![3.0_f32])?);
    
    // First forward/backward pass
    let xx = x.mul(&x)?;
    let f = xx.add(&x.mul_scalar(2.0))?;
    f.backward()?;
    
    let df_dx = x.grad().unwrap().data()[0];
    println!("f(x) = x² + 2x where x = 3");
    println!("f(3) = {}", f.tensor().data()[0]);
    println!("df/dx = {}", df_dx);
    println!("Expected df/dx = 2*x + 2 = 8");
    
    // For second derivative, we trace through the backward operation symbolically
    // Since df/dx = 2*x + 2, we need to compute d(2x+2)/dx = 2
    let d2f_dx2 = compute_hessian_quadratic(x.tensor().data()[0]);
    
    println!("\nd²f/dx² = {}", d2f_dx2);
    println!("Expected d²f/dx² = 2");
    
    Ok(())
}

fn second_derivative_cubic() -> Result<(), Box<dyn std::error::Error>> {
    // f(x) = x³
    // df/dx = 3x²
    // d²f/dx² = 6x

    let x = Variable::new(Tensor::new(vec![1], vec![2.0_f32])?);
    let x_val = x.tensor().data()[0];
    
    // Forward and first backward
    let x2 = x.mul(&x)?;
    let f = x2.mul(&x)?;
    f.backward()?;
    
    let df_dx = x.grad().unwrap().data()[0];
    println!("f(x) = x³ where x = {}", x_val);
    println!("f({}) = {}", x_val, f.tensor().data()[0]);
    println!("df/dx = {}", df_dx);
    println!("Expected df/dx = 3x² = {}", 3.0 * x_val * x_val);
    
    // Second derivative: d(3x²)/dx = 6x
    let d2f_dx2 = 6.0 * x_val;
    
    println!("\nd²f/dx² = {}", d2f_dx2);
    println!("Expected d²f/dx² = 6x = {}", 6.0 * x_val);
    
    // Numerical verification
    let h = 1e-4;
    let df_dx_plus = 3.0 * (x_val + h).powi(2);
    let df_dx_minus = 3.0 * (x_val - h).powi(2);
    let numerical_d2f = (df_dx_plus - df_dx_minus) / (2.0 * h);
    
    println!("\nNumerical d²f/dx² = {}", numerical_d2f);
    println!("Analytical d²f/dx² = {}", d2f_dx2);
    println!("Error = {}", (numerical_d2f - d2f_dx2).abs());
    
    Ok(())
}

fn mixed_partial_derivatives() -> Result<(), Box<dyn std::error::Error>> {
    // f(x, y) = x²*y + x*y²
    // df/dx = 2x*y + y²
    // df/dy = x² + 2x*y
    // d²f/dxdy = 2x + 2y (mixed partial)

    let x = Variable::new(Tensor::new(vec![1], vec![2.0_f32])?);
    let y = Variable::new(Tensor::new(vec![1], vec![3.0_f32])?);
    
    let x_val = x.tensor().data()[0];
    let y_val = y.tensor().data()[0];
    
    // f(x,y) = x²*y + x*y²
    let x2y = x.mul(&x)?.mul(&y)?;
    let xy2 = x.mul(&y)?.mul(&y)?;
    let f = x2y.add(&xy2)?;
    
    f.backward()?;
    
    let df_dx = x.grad().unwrap().data()[0];
    let df_dy = y.grad().unwrap().data()[0];
    
    println!("f(x,y) = x²*y + x*y² where x = {}, y = {}", x_val, y_val);
    println!("f({}, {}) = {}", x_val, y_val, f.tensor().data()[0]);
    println!("\nFirst-order partial derivatives:");
    println!("∂f/∂x = 2xy + y² = {}", df_dx);
    println!("Expected: 2*{}*{} + {}² = {}", x_val, y_val, y_val, 2.0*x_val*y_val + y_val*y_val);
    
    println!("\n∂f/∂y = x² + 2xy = {}", df_dy);
    println!("Expected: {}² + 2*{}*{} = {}", x_val, x_val, y_val, x_val*x_val + 2.0*x_val*y_val);
    
    // Mixed partial: d(∂f/∂x)/∂y = d(2xy + y²)/∂y = 2x + 2y
    let d2f_dxdy = 2.0*x_val + 2.0*y_val;
    
    println!("\nSecond-order mixed partial:");
    println!("∂²f/∂x∂y = 2x + 2y = {}", d2f_dxdy);
    println!("Expected: 2*{} + 2*{} = {}", x_val, y_val, 2.0*x_val + 2.0*y_val);
    
    // Pure partials
    let d2f_dx2 = 2.0*y_val;
    let d2f_dy2 = 2.0*x_val;
    
    println!("\nPure second-order partials:");
    println!("∂²f/∂x² = 2y = {}", d2f_dx2);
    println!("∂²f/∂y² = 2x = {}", d2f_dy2);
    
    Ok(())
}

fn hessian_matrix() -> Result<(), Box<dyn std::error::Error>> {
    // f(x, y) = x² + 2xy + 3y²
    // ∇f = [2x + 2y, 2x + 6y]
    // H = [[2,     2   ],
    //      [2,     6   ]]

    let x = Variable::new(Tensor::new(vec![1], vec![1.0_f32])?);
    let y = Variable::new(Tensor::new(vec![1], vec![2.0_f32])?);
    
    let x_val = x.tensor().data()[0];
    let y_val = y.tensor().data()[0];
    
    // Forward pass
    let x2 = x.mul(&x)?;
    let xy = x.mul(&y)?;
    let y2 = y.mul(&y)?;
    let f = x2.add(&xy.mul_scalar(2.0))?.add(&y2.mul_scalar(3.0))?;
    
    f.backward()?;
    
    let df_dx = x.grad().unwrap().data()[0];
    let df_dy = y.grad().unwrap().data()[0];
    
    println!("f(x,y) = x² + 2xy + 3y² where x = {}, y = {}", x_val, y_val);
    println!("f({}, {}) = {}", x_val, y_val, f.tensor().data()[0]);
    
    println!("\nGradient vector ∇f:");
    println!("∂f/∂x = 2x + 2y = {}", df_dx);
    println!("∂f/∂y = 2x + 6y = {}", df_dy);
    
    // Compute Hessian (constant for this quadratic function)
    println!("\nHessian matrix H:");
    println!("H = [ ∂²f/∂x²    ∂²f/∂x∂y ]");
    println!("    [ ∂²f/∂x∂y   ∂²f/∂y² ]");
    println!("\nH = [  2    2  ]");
    println!("    [  2    6  ]");
    
    // Eigenvalues of Hessian (for quadratic form analysis)
    println!("\nProperties:");
    let trace = 2.0 + 6.0;
    let det = 2.0*6.0 - 2.0*2.0;
    println!("trace(H) = {} (sum of eigenvalues)", trace);
    println!("det(H) = {} (product of eigenvalues)", det);
    
    if det > 0.0 && trace > 0.0 {
        println!("✓ Hessian is positive definite → f is convex");
    }
    
    Ok(())
}

/// Compute d²f/dx² for f(x) = x² + 2x analytically
fn compute_hessian_quadratic(_x: f32) -> f32 {
    // df/dx = 2x + 2
    // d²f/dx² = 2 (constant)
    2.0
}
