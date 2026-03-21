use rand::Rng;
use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    // --- Creation ---

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    pub fn randn(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(size);

        // Box-Muller transform, generate pairs
        let mut i = 0;
        while i < size {
            let u1: f32 = rng.r#gen::<f32>().max(1e-10); // avoid log(0)
            let u2: f32 = rng.r#gen::<f32>();
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * PI * u2).cos();
            let z1 = mag * (2.0 * PI * u2).sin();
            data.push(z0);
            if i + 1 < size {
                data.push(z1);
            }
            i += 2;
        }
        data.truncate(size);

        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn from_data(shape: &[usize], data: Vec<f32>) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(data.len(), size, "data length must match shape product");
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    // --- Element-wise operations (return new tensors) ---

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shapes must match for add");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shapes must match for sub");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shapes must match for mul");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn scale(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    // --- In-place mutation (for weight updates) ---

    pub fn add_inplace(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "shapes must match for add_inplace");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    pub fn scale_inplace(&mut self, scalar: f32) {
        for x in self.data.iter_mut() {
            *x *= scalar;
        }
    }

    // --- MatMul: 2D only, (M x K) @ (K x N) -> (M x N) ---

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(other.shape[0], k, "inner dimensions must match for matmul");
        let n = other.shape[1];

        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }

        Tensor {
            data,
            shape: vec![m, n],
        }
    }

    // --- Softmax along last dimension ---

    pub fn softmax(&self) -> Tensor {
        assert!(
            self.shape.len() >= 1,
            "softmax requires at least 1 dimension"
        );
        let last_dim = *self.shape.last().unwrap();
        let num_rows = self.data.len() / last_dim;
        let mut data = vec![0.0f32; self.data.len()];

        for row in 0..num_rows {
            let offset = row * last_dim;
            let slice = &self.data[offset..offset + last_dim];

            // Numerical stability: subtract max
            let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();

            for (i, e) in exps.iter().enumerate() {
                data[offset + i] = e / sum;
            }
        }

        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    // --- Layer Norm along last dimension with learnable gamma/beta ---

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let last_dim = *self.shape.last().unwrap();
        assert_eq!(gamma.data.len(), last_dim, "gamma must match last dim");
        assert_eq!(beta.data.len(), last_dim, "beta must match last dim");

        let num_rows = self.data.len() / last_dim;
        let mut data = vec![0.0f32; self.data.len()];

        for row in 0..num_rows {
            let offset = row * last_dim;
            let slice = &self.data[offset..offset + last_dim];

            let mean: f32 = slice.iter().sum::<f32>() / last_dim as f32;
            let var: f32 =
                slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / last_dim as f32;
            let std_inv = 1.0 / (var + eps).sqrt();

            for i in 0..last_dim {
                let normalized = (slice[i] - mean) * std_inv;
                data[offset + i] = gamma.data[i] * normalized + beta.data[i];
            }
        }

        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    // --- GELU activation ---

    pub fn gelu(&self) -> Tensor {
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi = (2.0f32 / PI).sqrt();
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    // --- Row indexing (embedding lookup) ---
    // self is 2D (vocab_size x embed_dim), indices is a list of row indices.
    // Returns (len(indices) x embed_dim).

    pub fn row_index(&self, indices: &[usize]) -> Tensor {
        assert_eq!(self.shape.len(), 2, "row_index requires a 2D tensor");
        let embed_dim = self.shape[1];
        let num_indices = indices.len();
        let mut data = Vec::with_capacity(num_indices * embed_dim);

        for &idx in indices {
            assert!(idx < self.shape[0], "row index {} out of bounds", idx);
            let start = idx * embed_dim;
            data.extend_from_slice(&self.data[start..start + embed_dim]);
        }

        Tensor {
            data,
            shape: vec![num_indices, embed_dim],
        }
    }

    // --- Transpose 2D ---

    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "transpose requires a 2D tensor");
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0f32; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Tensor {
            data,
            shape: vec![cols, rows],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![0.0; 6]);
    }

    #[test]
    fn test_randn_shape() {
        let t = Tensor::randn(&[4, 5]);
        assert_eq!(t.shape, vec![4, 5]);
        assert_eq!(t.data.len(), 20);
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_data(&[2], vec![1.0, 2.0]);
        let b = Tensor::from_data(&[2], vec![3.0, 4.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from_data(&[2], vec![5.0, 3.0]);
        let b = Tensor::from_data(&[2], vec![1.0, 2.0]);
        let c = a.sub(&b);
        assert_eq!(c.data, vec![4.0, 1.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_data(&[2], vec![2.0, 3.0]);
        let b = Tensor::from_data(&[2], vec![4.0, 5.0]);
        let c = a.mul(&b);
        assert_eq!(c.data, vec![8.0, 15.0]);
    }

    #[test]
    fn test_scale() {
        let a = Tensor::from_data(&[3], vec![1.0, 2.0, 3.0]);
        let b = a.scale(2.0);
        assert_eq!(b.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_matmul() {
        // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        let a = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_data(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_data(&[1, 3], vec![1.0, 2.0, 3.0]);
        let s = t.softmax();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Values should be monotonically increasing
        assert!(s.data[0] < s.data[1]);
        assert!(s.data[1] < s.data[2]);
    }

    #[test]
    fn test_layer_norm() {
        let t = Tensor::from_data(&[1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let gamma = Tensor::from_data(&[4], vec![1.0; 4]);
        let beta = Tensor::from_data(&[4], vec![0.0; 4]);
        let normed = t.layer_norm(&gamma, &beta, 1e-5);
        // Mean of normalized should be ~0
        let mean: f32 = normed.data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_data(&[3], vec![-1.0, 0.0, 1.0]);
        let g = t.gelu();
        // GELU(0) = 0
        assert!(g.data[1].abs() < 1e-5);
        // GELU(1) ~ 0.8413
        assert!((g.data[2] - 0.8413).abs() < 0.01);
        // GELU(-1) ~ -0.1587
        assert!((g.data[0] - (-0.1587)).abs() < 0.01);
    }

    #[test]
    fn test_row_index() {
        let t = Tensor::from_data(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let selected = t.row_index(&[0, 2]);
        assert_eq!(selected.shape, vec![2, 2]);
        assert_eq!(selected.data, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_data(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tr = t.transpose();
        assert_eq!(tr.shape, vec![3, 2]);
        assert_eq!(tr.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_inplace_ops() {
        let mut a = Tensor::from_data(&[2], vec![1.0, 2.0]);
        let b = Tensor::from_data(&[2], vec![3.0, 4.0]);
        a.add_inplace(&b);
        assert_eq!(a.data, vec![4.0, 6.0]);
        a.scale_inplace(0.5);
        assert_eq!(a.data, vec![2.0, 3.0]);
    }
}
