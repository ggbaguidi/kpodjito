use kpodjito_core_error::{Error, Result};

pub trait RandomSource {
    fn next_u64(&mut self) -> u64;

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64) / ((1u64 << 53) as f64)
    }

    fn fill_bytes(&mut self, output: &mut [u8]) {
        for chunk in output.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            let len = chunk.len();
            chunk.copy_from_slice(&bytes[..len]);
        }
    }
}

#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn seeded(seed: u64) -> Result<Self> {
        if seed == 0 {
            return Err(Error::InvalidIndex { index: 0, len: 1 });
        }

        Ok(Self { state: seed })
    }
}

impl RandomSource for XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_rng_produces_values() {
        let mut rng = XorShift64::seeded(1).unwrap();
        assert_ne!(rng.next_u64(), 0);
    }
}