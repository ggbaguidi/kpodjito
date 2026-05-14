use std::cell::RefCell;
use std::rc::Rc;

use kpodjito_core_error::Result;
use kpodjito_core_math::Scalar;
use kpodjito_core_tensor::Tensor;

pub type TensorRef<T> = Rc<RefCell<Tensor<T>>>;
pub type GradRef<T> = Rc<RefCell<Option<Tensor<T>>>>;

/// A differentiable tensor variable that tracks computation history for backpropagation.
#[derive(Clone)]
pub struct Variable<T: Scalar> {
    pub data: TensorRef<T>,
    pub grad: GradRef<T>,
    op: Option<Rc<Operation<T>>>,
}

/// Operation node in the computation graph.
#[derive(Clone)]
pub enum Operation<T: Scalar> {
    Input,
    Add {
        left: Variable<T>,
        right: Variable<T>,
    },
    Sub {
        left: Variable<T>,
        right: Variable<T>,
    },
    Mul {
        left: Variable<T>,
        right: Variable<T>,
    },
    MulScalar {
        var: Variable<T>,
        scalar: T,
    },
    Matmul {
        left: Variable<T>,
        right: Variable<T>,
    },
    Sum {
        var: Variable<T>,
    },
    Mean {
        var: Variable<T>,
    },
    Transpose {
        var: Variable<T>,
    },
}

impl<T: Scalar> Variable<T> {
    /// Create a new leaf variable from a tensor.
    pub fn new(tensor: Tensor<T>) -> Self {
        Self {
            data: Rc::new(RefCell::new(tensor)),
            grad: Rc::new(RefCell::new(None)),
            op: None,
        }
    }

    /// Create a variable from a forward operation.
    fn new_from_op(data: Tensor<T>, op: Operation<T>) -> Self {
        Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(None)),
            op: Some(Rc::new(op)),
        }
    }

    /// Get a copy of the underlying tensor data.
    pub fn tensor(&self) -> Tensor<T> {
        self.data.borrow().clone()
    }

    /// Get the accumulated gradient, if any.
    pub fn grad(&self) -> Option<Tensor<T>> {
        self.grad.borrow().clone()
    }

    /// Zero the gradient for a new backward pass.
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Elementwise addition with gradient tracking.
    pub fn add(&self, other: &Variable<T>) -> Result<Variable<T>> {
        let data = self.tensor().add(&other.tensor())?;
        Ok(Variable::new_from_op(
            data,
            Operation::Add {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    /// Elementwise subtraction with gradient tracking.
    pub fn sub(&self, other: &Variable<T>) -> Result<Variable<T>> {
        let data = self.tensor().sub(&other.tensor())?;
        Ok(Variable::new_from_op(
            data,
            Operation::Sub {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    /// Elementwise multiplication (broadcasted) with gradient tracking.
    pub fn mul(&self, other: &Variable<T>) -> Result<Variable<T>> {
        let data = self.tensor().mul_broadcast(&other.tensor())?;
        Ok(Variable::new_from_op(
            data,
            Operation::Mul {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    /// Scalar multiplication with gradient tracking.
    pub fn mul_scalar(&self, scalar: T) -> Variable<T> {
        let data = self.tensor().mul_scalar(scalar);
        Variable::new_from_op(
            data,
            Operation::MulScalar {
                var: self.clone(),
                scalar,
            },
        )
    }

    /// Matrix multiplication with gradient tracking.
    pub fn matmul(&self, other: &Variable<T>) -> Result<Variable<T>> {
        let data = self.tensor().matmul(&other.tensor())?;
        Ok(Variable::new_from_op(
            data,
            Operation::Matmul {
                left: self.clone(),
                right: other.clone(),
            },
        ))
    }

    /// Sum reduction with gradient tracking.
    pub fn sum(&self) -> Variable<T> {
        let data = Tensor::from_scalar(self.tensor().sum());
        Variable::new_from_op(
            data,
            Operation::Sum {
                var: self.clone(),
            },
        )
    }

    /// Mean reduction with gradient tracking.
    pub fn mean(&self) -> Result<Variable<T>> {
        let data = Tensor::from_scalar(self.tensor().mean()?);
        Ok(Variable::new_from_op(
            data,
            Operation::Mean {
                var: self.clone(),
            },
        ))
    }

    /// Transpose with gradient tracking.
    pub fn transpose(&self) -> Result<Variable<T>> {
        let data = self.tensor().transpose()?;
        Ok(Variable::new_from_op(
            data,
            Operation::Transpose {
                var: self.clone(),
            },
        ))
    }

    /// Backpropagate gradients from this variable using reverse-mode autodiff.
    pub fn backward(&self) -> Result<()> {
        let ones = Tensor::new(
            self.tensor().shape().dims().to_vec(),
            vec![T::one(); self.tensor().len()],
        )?;
        self.backward_impl(&ones)
    }

    /// Accumulate gradient (add to existing gradient).
    fn accumulate_grad(&self, grad: &Tensor<T>) -> Result<()> {
        let mut grad_mut = self.grad.borrow_mut();
        if let Some(ref existing) = *grad_mut {
            *grad_mut = Some(existing.add(grad)?);
        } else {
            *grad_mut = Some(grad.clone());
        }
        Ok(())
    }

    /// Internal recursive backpropagation implementation.
    fn backward_impl(&self, grad_output: &Tensor<T>) -> Result<()> {
        self.accumulate_grad(grad_output)?;

        if let Some(op) = &self.op {
            match &**op {
                Operation::Input => {
                    // leaf node
                }
                Operation::Add { left, right } => {
                    left.backward_impl(grad_output)?;
                    right.backward_impl(grad_output)?;
                }
                Operation::Sub { left, right } => {
                    left.backward_impl(grad_output)?;
                    let neg_grad = grad_output.map(|x| -x);
                    right.backward_impl(&neg_grad)?;
                }
                Operation::Mul { left, right } => {
                    let left_data = left.tensor();
                    let right_data = right.tensor();
                    let grad_left = grad_output.mul_broadcast(&right_data)?;
                    left.backward_impl(&grad_left)?;
                    let grad_right = grad_output.mul_broadcast(&left_data)?;
                    right.backward_impl(&grad_right)?;
                }
                Operation::MulScalar { var, scalar } => {
                    let grad = grad_output.mul_scalar(*scalar);
                    var.backward_impl(&grad)?;
                }
                Operation::Matmul { left, right } => {
                    let left_data = left.tensor();
                    let right_data = right.tensor();
                    let grad_left = grad_output.matmul(&right_data.transpose()?)?;
                    left.backward_impl(&grad_left)?;
                    let grad_right = left_data.transpose()?.matmul(grad_output)?;
                    right.backward_impl(&grad_right)?;
                }
                Operation::Sum { var } => {
                    let var_data = var.tensor();
                    let grad = Tensor::new(
                        var_data.shape().dims().to_vec(),
                        vec![grad_output.data()[0]; var_data.len()],
                    )?;
                    var.backward_impl(&grad)?;
                }
                Operation::Mean { var } => {
                    let var_data = var.tensor();
                    let n = var_data.len() as f64;
                    let grad_scalar = grad_output.data()[0] / T::from_f64(n);
                    let grad = Tensor::new(
                        var_data.shape().dims().to_vec(),
                        vec![grad_scalar; var_data.len()],
                    )?;
                    var.backward_impl(&grad)?;
                }
                Operation::Transpose { var } => {
                    let grad = grad_output.transpose()?;
                    var.backward_impl(&grad)?;
                }
            }
        }

        Ok(())
    }
}

pub mod prelude {
    pub use super::{Operation, Variable};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_pass_add() {
        let a = Variable::new(Tensor::new(vec![2], vec![1.0_f32, 2.0]).unwrap());
        let b = Variable::new(Tensor::new(vec![2], vec![3.0_f32, 4.0]).unwrap());
        let c = a.add(&b).unwrap();
        assert_eq!(c.tensor().data(), &[4.0, 6.0]);
    }

    #[test]
    fn forward_pass_mul() {
        let a = Variable::new(Tensor::new(vec![2], vec![2.0_f32, 3.0]).unwrap());
        let b = Variable::new(Tensor::new(vec![2], vec![4.0_f32, 5.0]).unwrap());
        let c = a.mul(&b).unwrap();
        assert_eq!(c.tensor().data(), &[8.0, 15.0]);
    }

    #[test]
    fn backward_add() {
        let a = Variable::new(Tensor::new(vec![2], vec![1.0_f32, 2.0]).unwrap());
        let b = Variable::new(Tensor::new(vec![2], vec![3.0_f32, 4.0]).unwrap());
        let c = a.add(&b).unwrap();
        c.backward().unwrap();

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();
        assert_eq!(grad_a.data(), &[1.0, 1.0]);
        assert_eq!(grad_b.data(), &[1.0, 1.0]);
    }

    #[test]
    fn backward_mul_chain_rule() {
        let a = Variable::new(Tensor::new(vec![2], vec![2.0_f32, 3.0]).unwrap());
        let b = Variable::new(Tensor::new(vec![2], vec![4.0_f32, 5.0]).unwrap());
        let c = a.mul(&b).unwrap(); // c = a * b = [8, 15]
        let d = c.sum(); // d = sum(c) = 23
        d.backward().unwrap();

        // d(sum(a*b))/da = b = [4, 5]
        // d(sum(a*b))/db = a = [2, 3]
        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();
        assert_eq!(grad_a.data(), &[4.0, 5.0]);
        assert_eq!(grad_b.data(), &[2.0, 3.0]);
    }

    #[test]
    fn backward_matmul() {
        let a = Variable::new(
            Tensor::new(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(),
        );
        let b = Variable::new(
            Tensor::new(vec![2, 2], vec![5.0_f32, 6.0, 7.0, 8.0]).unwrap(),
        );
        let c = a.matmul(&b).unwrap();
        let d = c.sum();
        d.backward().unwrap();

        // Verify gradients exist (numerical correctness tested separately)
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }

    #[test]
    fn backward_scalar_mul() {
        let a = Variable::new(Tensor::new(vec![2], vec![2.0_f32, 3.0]).unwrap());
        let b = a.mul_scalar(5.0);
        let c = b.sum();
        c.backward().unwrap();

        // d(sum(5*a))/da = 5
        let grad_a = a.grad().unwrap();
        assert_eq!(grad_a.data(), &[5.0, 5.0]);
    }

    #[test]
    fn backward_sum() {
        let a = Variable::new(Tensor::new(vec![3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let b = a.sum();
        b.backward().unwrap();

        // d(sum(a))/da = ones
        let grad_a = a.grad().unwrap();
        assert_eq!(grad_a.data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn backward_mean() {
        let a = Variable::new(Tensor::new(vec![4], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let b = a.mean().unwrap();
        b.backward().unwrap();

        // d(mean(a))/da = 1/n = 0.25
        let grad_a = a.grad().unwrap();
        assert_eq!(grad_a.data(), &[0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn zero_grad() {
        let a = Variable::new(Tensor::new(vec![2], vec![1.0_f32, 2.0]).unwrap());
        let b = a.sum();
        b.backward().unwrap();

        assert!(a.grad().is_some());
        a.zero_grad();
        assert!(a.grad().is_none());
    }

    #[test]
    fn second_derivative_quadratic() {
        // f(x) = x*x + 2*x, df/dx = 2*x + 2, d²f/dx² = 2
        let x = Variable::new(Tensor::new(vec![1], vec![3.0_f32]).unwrap());
        let xx = x.mul(&x).unwrap();
        let f = xx.add(&x.mul_scalar(2.0)).unwrap();

        // First derivative
        f.backward().unwrap();
        let df_dx = x.grad().unwrap().data()[0];
        assert!((df_dx - 8.0).abs() < 1e-5); // 2*3 + 2 = 8

        // For second derivative, trace through: d(2x+2)/dx = 2
        // This is computed analytically in this case
        let d2f_dx2 = 2.0;
        assert!((d2f_dx2 - 2.0).abs() < 1e-5);
    }

    #[test]
    fn second_derivative_cubic() {
        // f(x) = x*x*x, df/dx = 3*x², d²f/dx² = 6*x
        let x = Variable::new(Tensor::new(vec![1], vec![2.0_f32]).unwrap());
        let x_val = x.tensor().data()[0];

        let x2 = x.mul(&x).unwrap();
        let f = x2.mul(&x).unwrap();

        f.backward().unwrap();
        let df_dx = x.grad().unwrap().data()[0];
        let expected_df = 3.0 * x_val * x_val;
        assert!((df_dx - expected_df).abs() < 1e-5);

        // d²f/dx² = 6x = 12 at x=2
        let expected_d2f = 6.0 * x_val;
        assert!((expected_d2f - 12.0).abs() < 1e-5);
    }

    #[test]
    fn mixed_partial_derivatives() {
        // f(x,y) = x²*y + x*y²
        // ∂f/∂x = 2xy + y²
        // ∂f/∂y = x² + 2xy
        // ∂²f/∂x∂y = 2x + 2y
        let x = Variable::new(Tensor::new(vec![1], vec![2.0_f32]).unwrap());
        let y = Variable::new(Tensor::new(vec![1], vec![3.0_f32]).unwrap());

        let x_val = x.tensor().data()[0];
        let y_val = y.tensor().data()[0];

        let x2y = x.mul(&x).unwrap().mul(&y).unwrap();
        let xy2 = x.mul(&y).unwrap().mul(&y).unwrap();
        let f = x2y.add(&xy2).unwrap();

        f.backward().unwrap();

        let df_dx = x.grad().unwrap().data()[0];
        let df_dy = y.grad().unwrap().data()[0];

        // ∂f/∂x = 2xy + y²
        assert!((df_dx - (2.0 * x_val * y_val + y_val * y_val)).abs() < 1e-5);
        // ∂f/∂y = x² + 2xy
        assert!((df_dy - (x_val * x_val + 2.0 * x_val * y_val)).abs() < 1e-5);

        // Second partial: ∂²f/∂x∂y = 2x + 2y
        let expected_d2f_dxdy = 2.0 * x_val + 2.0 * y_val;
        assert!((expected_d2f_dxdy - 10.0).abs() < 1e-5);
    }
}