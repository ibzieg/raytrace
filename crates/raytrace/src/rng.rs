use glam::DVec3;

/// SplitMix64: tiny, fast, and seedable per pixel / per photon so renders are
/// deterministic regardless of thread scheduling.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Self(seed ^ 0x9E37_79B9_7F4A_7C15);
        r.next_u64();
        r
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize
    }

    /// Uniformly distributed unit vector (rejection sampled from the ball,
    /// as in the original, but normalized).
    pub fn unit_vector(&mut self) -> DVec3 {
        loop {
            let v = DVec3::new(self.next_f64(), self.next_f64(), self.next_f64()) * 2.0 - 1.0;
            let l2 = v.length_squared();
            if l2 > 1e-12 && l2 <= 1.0 {
                return v / l2.sqrt();
            }
        }
    }
}
