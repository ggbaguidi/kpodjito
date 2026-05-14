use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    EmptyTensor,
    InvalidShape { shape: Vec<usize> },
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    DimensionMismatch { expected: usize, actual: usize },
    InvalidIndex { index: usize, len: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTensor => write!(f, "tensor is empty"),
            Self::InvalidShape { shape } => write!(f, "invalid shape: {shape:?}"),
            Self::ShapeMismatch { left, right } => {
                write!(f, "shape mismatch: left={left:?}, right={right:?}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidIndex { index, len } => {
                write!(f, "invalid index {index} for length {len}")
            }
        }
    }
}

impl std::error::Error for Error {}