use std::fmt;

use kpodjito_core_error::{Error, Result};
use kpodjito_core_math::{dot, mean as math_mean, sum as math_sum, Scalar, Shape};

#[derive(Clone, PartialEq)]
pub struct Tensor<T: Scalar> {
    shape: Shape,
    data: Vec<T>,
}

impl<T: Scalar> fmt::Debug for Tensor<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape.dims())
            .field("data", &self.data)
            .finish()
    }
}

impl<T: Scalar> Tensor<T> {
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<T>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if shape.numel() != data.len() {
            return Err(Error::DimensionMismatch {
                expected: shape.numel(),
                actual: data.len(),
            });
        }

        Ok(Self { shape, data })
    }

    pub fn from_scalar(value: T) -> Self {
        Self {
            shape: Shape::scalar(),
            data: vec![value],
        }
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        Ok(Self {
            data: vec![T::zero(); shape.numel()],
            shape,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Result<T> {
        self.data
            .get(index)
            .copied()
            .ok_or(Error::InvalidIndex { index, len: self.data.len() })
    }

    pub fn map<F>(&self, mut function: F) -> Self
    where
        F: FnMut(T) -> T,
    {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().copied().map(&mut function).collect(),
        }
    }

    pub fn zip_map<F>(&self, other: &Self, mut function: F) -> Result<Self>
    where
        F: FnMut(T, T) -> T,
    {
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                left: self.shape.dims().to_vec(),
                right: other.shape.dims().to_vec(),
            });
        }

        Ok(Self {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .copied()
                .zip(other.data.iter().copied())
                .map(|(left, right)| function(left, right))
                .collect(),
        })
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.zip_map(other, |left, right| left + right)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.zip_map(other, |left, right| left - right)
    }

    pub fn mul_scalar(&self, scalar: T) -> Self {
        self.map(|value| value * scalar)
    }

    pub fn sum(&self) -> T {
        math_sum(&self.data)
    }

    pub fn mean(&self) -> Result<T> {
        math_mean(&self.data)
    }

    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if shape.numel() != self.len() {
            return Err(Error::DimensionMismatch {
                expected: self.len(),
                actual: shape.numel(),
            });
        }

        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let left_dims = self.shape.dims();
        let right_dims = other.shape.dims();

        if left_dims.len() != 2 || right_dims.len() != 2 {
            return Err(Error::DimensionMismatch {
                expected: 2,
                actual: left_dims.len().max(right_dims.len()),
            });
        }

        let rows = left_dims[0];
        let shared = left_dims[1];
        let columns = right_dims[1];

        if shared != right_dims[0] {
            return Err(Error::DimensionMismatch {
                expected: shared,
                actual: right_dims[0],
            });
        }

        let mut result = Vec::with_capacity(rows * columns);

        for row in 0..rows {
            for column in 0..columns {
                let left_row = &self.data[row * shared..(row + 1) * shared];
                let mut right_column = Vec::with_capacity(shared);

                for index in 0..shared {
                    right_column.push(other.data[index * columns + column]);
                }

                result.push(dot(left_row, &right_column)?);
            }
        }

        Tensor::new(vec![rows, columns], result)
    }

    // --- Broadcasting helpers ---
    fn strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len()).rev() {
            if i + 1 < shape.len() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        strides
    }

    fn coords_from_flat(mut idx: usize, dims: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; dims.len()];
        for i in (0..dims.len()).rev() {
            let d = dims[i];
            coords[i] = idx % d;
            idx /= d;
        }
        coords
    }

    fn flat_index_from_coords(coords: &[usize], strides: &[usize]) -> usize {
        coords.iter().zip(strides.iter()).map(|(&c, &s)| c * s).sum()
    }

    fn broadcast_shape(a: &Shape, b: &Shape) -> Result<Shape> {
        let ad = a.dims();
        let bd = b.dims();
        let mut out = Vec::new();
        let al = ad.len();
        let bl = bd.len();
        let ml = al.max(bl);
        for i in 0..ml {
            let a_dim = if i >= ml - al { Some(ad[i - (ml - al)]) } else { None };
            let b_dim = if i >= ml - bl { Some(bd[i - (ml - bl)]) } else { None };

            let dim = match (a_dim, b_dim) {
                (Some(x), Some(y)) => {
                    if x == y { x }
                    else if x == 1 { y }
                    else if y == 1 { x }
                    else { return Err(Error::ShapeMismatch { left: ad.to_vec(), right: bd.to_vec() }); }
                }
                (Some(x), None) => x,
                (None, Some(y)) => y,
                (None, None) => unreachable!(),
            };

            out.push(dim);
        }

        Shape::new(out)
    }

    fn broadcasted_element<F>(&self, other: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(T, T) -> T,
    {
        let out_shape = Self::broadcast_shape(&self.shape, &other.shape)?;
        let out_dims = out_shape.dims();
        let out_len = out_shape.numel();

        let a_dims = self.shape.dims();
        let b_dims = other.shape.dims();
        let a_strides = Self::strides(a_dims);
        let b_strides = Self::strides(b_dims);

        let mut out = Vec::with_capacity(out_len);
        for idx in 0..out_len {
            let coords = Self::coords_from_flat(idx, out_dims);

            // map to a coords (right-align)
            let mut a_coords = vec![0; a_dims.len()];
            let out_rank = out_dims.len();
            let a_rank = a_dims.len();
            for j in 0..out_rank {
                let coord = coords[j];
                let a_pos_opt = if a_rank > out_rank {
                    Some(j + (a_rank - out_rank))
                } else if a_rank < out_rank {
                    if j < out_rank - a_rank { None } else { Some(j - (out_rank - a_rank)) }
                } else { Some(j) };

                if let Some(a_pos) = a_pos_opt {
                    let a_dim = a_dims[a_pos];
                    a_coords[a_pos] = if a_dim == 1 { 0 } else { coord };
                }
            }

            // map to b coords (right-align)
            let mut b_coords = vec![0; b_dims.len()];
            let b_rank = b_dims.len();
            for j in 0..out_rank {
                let coord = coords[j];
                let b_pos_opt = if b_rank > out_rank {
                    Some(j + (b_rank - out_rank))
                } else if b_rank < out_rank {
                    if j < out_rank - b_rank { None } else { Some(j - (out_rank - b_rank)) }
                } else { Some(j) };

                if let Some(b_pos) = b_pos_opt {
                    let b_dim = b_dims[b_pos];
                    b_coords[b_pos] = if b_dim == 1 { 0 } else { coord };
                }
            }

            let a_idx = Self::flat_index_from_coords(&a_coords, &a_strides);
            let b_idx = Self::flat_index_from_coords(&b_coords, &b_strides);
            let aval = self.data[a_idx];
            let bval = other.data[b_idx];
            out.push(f(aval, bval));
        }

        Tensor::new(out_dims.to_vec(), out)
    }

    pub fn add_broadcast(&self, other: &Self) -> Result<Self> {
        self.broadcasted_element(other, |a, b| a + b)
    }

    pub fn sub_broadcast(&self, other: &Self) -> Result<Self> {
        self.broadcasted_element(other, |a, b| a - b)
    }

    pub fn mul_broadcast(&self, other: &Self) -> Result<Self> {
        self.broadcasted_element(other, |a, b| a * b)
    }

    pub fn div_broadcast(&self, other: &Self) -> Result<Self> {
        self.broadcasted_element(other, |a, b| a / b)
    }

    pub fn transpose(&self) -> Result<Self> {
        let dims = self.shape.dims();
        if dims.len() != 2 {
            return Err(Error::DimensionMismatch { expected: 2, actual: dims.len() });
        }

        let rows = dims[0];
        let cols = dims[1];
        let mut out = Vec::with_capacity(self.len());
        for c in 0..cols {
            for r in 0..rows {
                out.push(self.data[r * cols + c]);
            }
        }

        Tensor::new(vec![cols, rows], out)
    }

    pub fn diagonal(&self) -> Result<Self> {
        let dims = self.shape.dims();
        if dims.len() != 2 {
            return Err(Error::DimensionMismatch { expected: 2, actual: dims.len() });
        }

        if dims[0] != dims[1] {
            return Err(Error::DimensionMismatch { expected: dims[0], actual: dims[1] });
        }

        let n = dims[0];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.data[i * n + i]);
        }

        Tensor::new(vec![n], out)
    }

    pub fn diag_from_vector(v: &Self) -> Result<Self> {
        let dims = v.shape.dims();
        if dims.len() != 1 {
            return Err(Error::DimensionMismatch { expected: 1, actual: dims.len() });
        }

        let n = dims[0];
        let mut out = vec![T::zero(); n * n];
        for i in 0..n {
            out[i * n + i] = v.data[i];
        }

        Tensor::new(vec![n, n], out)
    }

    pub fn trace(&self) -> Result<T> {
        let diag = self.diagonal()?;
        Ok(diag.sum())
    }

    fn matvec(&self, v: &Self) -> Result<Self> {
        let left_dims = self.shape.dims();
        let v_dims = v.shape.dims();

        if left_dims.len() != 2 || v_dims.len() != 1 {
            return Err(Error::DimensionMismatch { expected: 2, actual: left_dims.len().max(v_dims.len()) });
        }

        let rows = left_dims[0];
        let shared = left_dims[1];
        if shared != v_dims[0] {
            return Err(Error::DimensionMismatch { expected: shared, actual: v_dims[0] });
        }

        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = &self.data[r * shared..(r + 1) * shared];
            out.push(dot(row, &v.data)?);
        }

        Tensor::new(vec![rows], out)
    }

    pub fn determinant(&self) -> Result<T> {
        let dims = self.shape.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(Error::DimensionMismatch { expected: 2, actual: dims.len() });
        }

        let n = dims[0];
        // Make a copy as f64 ops are not available; operate in-place on T
        let mut m: Vec<T> = self.data.clone();
        let mut det = T::one();

        for i in 0..n {
            // partial pivot
            let mut pivot = i;
            let mut max = m[i * n + i].abs();
            for r in (i + 1)..n {
                let val = m[r * n + i].abs();
                if val > max {
                    max = val;
                    pivot = r;
                }
            }

            if pivot != i {
                // swap rows
                for c in 0..n {
                    m.swap(i * n + c, pivot * n + c);
                }
                det = det * (T::from_f64(-1.0));
            }

            let pivot_val = m[i * n + i];
            if pivot_val == T::zero() {
                return Ok(T::zero());
            }

            det = det * pivot_val;

            // normalize row
            for r in (i + 1)..n {
                let factor = m[r * n + i] / pivot_val;
                for c in (i + 1)..n {
                    m[r * n + c] = m[r * n + c] - factor * m[i * n + c];
                }
            }

        }

        Ok(det)
    }

    pub fn inverse(&self) -> Result<Self> {
        let dims = self.shape.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(Error::DimensionMismatch { expected: 2, actual: dims.len() });
        }
        let n = dims[0];
        // create augmented matrix [A | I]
        let mut aug = Vec::with_capacity(n * n * 2);
        for r in 0..n {
            for c in 0..n {
                aug.push(self.data[r * n + c]);
            }
            for c in 0..n {
                aug.push(if r == c { T::one() } else { T::zero() });
            }
        }

        // Gauss-Jordan with partial pivoting
        for i in 0..n {
            // pivot
            let mut pivot = i;
            let mut max = aug[i * (2 * n) + i].abs();
            for r in (i + 1)..n {
                let val = aug[r * (2 * n) + i].abs();
                if val > max {
                    max = val;
                    pivot = r;
                }
            }

            if pivot != i {
                for c in 0..(2 * n) {
                    aug.swap(i * (2 * n) + c, pivot * (2 * n) + c);
                }
            }

            let piv = aug[i * (2 * n) + i];
            if piv == T::zero() {
                return Err(Error::DimensionMismatch { expected: 1, actual: 0 });
            }

            // normalize pivot row
            for c in 0..(2 * n) {
                aug[i * (2 * n) + c] = aug[i * (2 * n) + c] / piv;
            }

            // eliminate other rows
            for r in 0..n {
                if r == i { continue }
                let factor = aug[r * (2 * n) + i];
                for c in 0..(2 * n) {
                    let tmp = aug[i * (2 * n) + c] * factor;
                    aug[r * (2 * n) + c] = aug[r * (2 * n) + c] - tmp;
                }
            }
        }

        // extract inverse
        let mut inv = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                inv.push(aug[r * (2 * n) + (n + c)]);
            }
        }

        Tensor::new(vec![n, n], inv)
    }

    pub fn power_iteration(&self, max_iters: usize, tol: f64) -> Result<(T, Self)> {
        let dims = self.shape.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(Error::DimensionMismatch { expected: 2, actual: dims.len() });
        }

        let n = dims[0];
        // initial vector of ones
        let mut b = Tensor::new(vec![n], vec![T::one(); n])?;
        let mut lambda = T::zero();

        for _ in 0..max_iters {
            // matvec
            let y = self.matvec(&b)?;
            let norm = kpodjito_core_math::l2_norm(y.data())?;
            if norm == T::zero() {
                return Err(Error::EmptyTensor);
            }
            let b_next = y.mul_scalar(T::one() / norm);

            // Rayleigh quotient
            let ay = self.matvec(&b_next)?;
            let num = dot(b_next.data(), ay.data())?;
            let denom = dot(b_next.data(), b_next.data())?;
            let lambda_next = num / denom;

            let diff = (lambda_next - lambda).abs();
            if diff < T::from_f64(tol) {
                return Ok((lambda_next, b_next));
            }

            lambda = lambda_next;
            b = b_next;
        }

        Ok((lambda, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_adds_elementwise() {
        let left = Tensor::new(vec![2], vec![1.0_f32, 2.0]).unwrap();
        let right = Tensor::new(vec![2], vec![3.0_f32, 4.0]).unwrap();

        let result = left.add(&right).unwrap();
        assert_eq!(result.data(), &[4.0, 6.0]);
    }

    #[test]
    fn tensor_matmul_multiplies_matrices() {
        let left = Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let right = Tensor::new(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();

        let result = left.matmul(&right).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn tensor_mean_uses_math_layer() {
        let tensor = Tensor::new(vec![4], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        assert!((tensor.mean().unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn determinant_and_inverse_edge_cases() {
        // singular matrix
        let sing = Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();
        assert!((sing.determinant().unwrap() - 0.0).abs() < 1e-12);
        assert!(sing.inverse().is_err());

        // identity inverse
        let id = Tensor::new(vec![3, 3], vec![1.0_f64,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]).unwrap();
        let inv = id.inverse().unwrap();
        assert_eq!(inv.data(), id.data());
    }

    #[test]
    fn power_iteration_dominant_eigenpair() {
        // matrix with dominant eigenvalue 3
        let a = Tensor::new(vec![2,2], vec![3.0_f64, 0.0, 0.0, 1.0]).unwrap();
        let (lambda, vec) = a.power_iteration(100, 1e-9).unwrap();
        assert!((lambda - 3.0).abs() < 1e-6);
        // eigenvector should emphasize first component
        assert!(vec.data()[0].abs() > vec.data()[1].abs());
    }

    #[test]
    fn broadcasting_add_sub_cases() {
        let a = Tensor::new(vec![2,1], vec![1.0_f32, 2.0]).unwrap();
        let b = Tensor::new(vec![1,3], vec![10.0_f32, 20.0, 30.0]).unwrap();
        let c = a.add_broadcast(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2,3]);
        assert_eq!(c.data(), &[11.0,21.0,31.0,12.0,22.0,32.0]);

        // scalar broadcasting
        let s = Tensor::new(vec![1], vec![5.0_f32]).unwrap();
        let d = a.mul_broadcast(&s).unwrap();
        assert_eq!(d.data(), &[5.0,10.0]);
    }
}