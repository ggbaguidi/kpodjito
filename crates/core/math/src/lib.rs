use std::ops::{Add, Div, Mul, Neg, Sub};

pub use kpodjito_core_error::{Error, Result};

pub trait Scalar:
    Copy
    + Clone
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(value: f64) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
}

impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.is_empty() || dims.contains(&0) {
            return Err(Error::InvalidShape { shape: dims });
        }

        Ok(Self { dims })
    }

    pub fn scalar() -> Self {
        Self { dims: vec![1] }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_scalar(&self) -> bool {
        self.numel() == 1
    }
}

pub fn dot<T: Scalar>(left: &[T], right: &[T]) -> Result<T> {
    if left.len() != right.len() {
        return Err(Error::DimensionMismatch {
            expected: left.len(),
            actual: right.len(),
        });
    }

    Ok(left
        .iter()
        .zip(right.iter())
        .fold(T::zero(), |accumulator, (&a, &b)| accumulator + a * b))
}

pub fn sum<T: Scalar>(values: &[T]) -> T {
    values
        .iter()
        .copied()
        .fold(T::zero(), |accumulator, value| accumulator + value)
}

pub fn mean<T: Scalar>(values: &[T]) -> Result<T> {
    if values.is_empty() {
        return Err(Error::EmptyTensor);
    }

    Ok(sum(values) / T::from_f64(values.len() as f64))
}

pub fn l2_norm<T: Scalar>(values: &[T]) -> Result<T> {
    dot(values, values).map(|value| value.sqrt())
}

pub fn softmax<T: Scalar>(values: &[T]) -> Result<Vec<T>> {
    if values.is_empty() {
        return Err(Error::EmptyTensor);
    }

    let max_value = values.iter().copied().fold(values[0], |accumulator, value| {
        if value > accumulator {
            value
        } else {
            accumulator
        }
    });

    let mut exps = Vec::with_capacity(values.len());
    let mut denominator = T::zero();

    for &value in values {
        let exp_value = (value - max_value).exp();
        denominator = denominator + exp_value;
        exps.push(exp_value);
    }

    Ok(exps.into_iter().map(|value| value / denominator).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_validates_dimensions() {
        assert!(Shape::new(vec![2, 3]).is_ok());
        assert!(Shape::new(Vec::<usize>::new()).is_err());
        assert!(Shape::new(vec![2, 0]).is_err());
    }

    #[test]
    fn dot_computes_expected_value() {
        let result = dot(&[1.0_f32, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_normalizes_values() {
        let result = softmax(&[1.0_f64, 2.0, 3.0]).unwrap();
        let total: f64 = result.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
}