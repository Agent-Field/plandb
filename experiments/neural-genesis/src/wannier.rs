//! Wannier Basis Weight Factorization for GPT.
//!
//! Physics inspiration: In solid-state physics, the same electronic structure
//! can be represented as Bloch waves (delocalized) or Wannier functions (localized),
//! related by a unitary transformation. Here we factorize weight matrices as:
//!
//!   W = U @ S @ V^T
//!
//! where U, V are orthogonal (built from sweep-based Givens rotations) and S is
//! a banded sparse matrix. This finds the natural basis where the weight is localized.
//!
//! Key distinction from SVD: SVD gives dense U, V and diagonal S (low-rank).
//! Wannier gives orthogonal U, V and BANDED S (sparse in the right basis).

use crate::model::{
    adam_step, add_vecs, gelu_backward, layer_norm, layer_norm_backward, matmul,
    matmul_backward_both, randn_vec, scale_vec, Config,
};
use rand::Rng;

// ---------------------------------------------------------------------------
// Givens rotation helpers (sweep-based for parameter efficiency)
// ---------------------------------------------------------------------------

/// Apply a single Givens rotation G(i, j, theta) to rows i and j of a matrix.
/// Matrix is row-major with shape (rows, cols).
fn apply_givens_rotation(
    matrix: &mut [f32],
    _rows: usize,
    cols: usize,
    i: usize,
    j: usize,
    theta: f32,
) {
    let c = theta.cos();
    let s = theta.sin();
    for col in 0..cols {
        let a = matrix[i * cols + col];
        let b = matrix[j * cols + col];
        matrix[i * cols + col] = c * a - s * b;
        matrix[j * cols + col] = s * a + c * b;
    }
}

/// Build an orthogonal matrix from sweep-based Givens angles.
/// Instead of all m(m-1)/2 rotations, use `n_sweeps` sweeps of nearest-neighbor
/// rotations: G(0,1), G(1,2), ..., G(m-2,m-1). Each sweep has (m-1) angles.
/// Total angles = n_sweeps * (m-1).
fn build_orthogonal_sweep(angles: &[f32], size: usize, n_sweeps: usize) -> Vec<f32> {
    let mut matrix = vec![0.0f32; size * size];
    // Identity
    for i in 0..size {
        matrix[i * size + i] = 1.0;
    }
    let angles_per_sweep = size - 1;
    for sweep in 0..n_sweeps {
        let base = sweep * angles_per_sweep;
        for k in 0..angles_per_sweep {
            let i = k;
            let j = k + 1;
            apply_givens_rotation(&mut matrix, size, size, i, j, angles[base + k]);
        }
    }
    matrix
}

// ---------------------------------------------------------------------------
// Banded sparse matrix
// ---------------------------------------------------------------------------

/// Expand banded values into a dense matrix.
/// For row r, the non-zero columns are [center - bandwidth/2 .. center + bandwidth/2)
/// where center = r * cols / rows (scaled diagonal).
/// `values` has length rows * bandwidth.
fn banded_to_dense(
    values: &[f32],
    rows: usize,
    cols: usize,
    bandwidth: usize,
) -> Vec<f32> {
    let mut dense = vec![0.0f32; rows * cols];
    let half_bw = bandwidth / 2;
    for r in 0..rows {
        // Center column for this row (scaled diagonal)
        let center = if rows == cols {
            r
        } else {
            (r as f64 * cols as f64 / rows as f64).round() as usize
        };
        for b in 0..bandwidth {
            let col_signed = center as isize - half_bw as isize + b as isize;
            if col_signed >= 0 && (col_signed as usize) < cols {
                let col = col_signed as usize;
                dense[r * cols + col] = values[r * bandwidth + b];
            }
        }
    }
    dense
}

/// Project a dense gradient matrix back to banded storage.
/// Returns gradient for the banded values only.
fn dense_to_banded_grad(
    d_dense: &[f32],
    rows: usize,
    cols: usize,
    bandwidth: usize,
) -> Vec<f32> {
    let mut d_values = vec![0.0f32; rows * bandwidth];
    let half_bw = bandwidth / 2;
    for r in 0..rows {
        let center = if rows == cols {
            r
        } else {
            (r as f64 * cols as f64 / rows as f64).round() as usize
        };
        for b in 0..bandwidth {
            let col_signed = center as isize - half_bw as isize + b as isize;
            if col_signed >= 0 && (col_signed as usize) < cols {
                let col = col_signed as usize;
                d_values[r * bandwidth + b] = d_dense[r * cols + col];
            }
        }
    }
    d_values
}

// ---------------------------------------------------------------------------
// Matrix utilities
// ---------------------------------------------------------------------------

/// Transpose a row-major matrix (r, c) -> (c, r).
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

/// Compute effective weight: W = U @ S_dense @ V^T
fn compute_effective_weight(
    u_angles: &[f32],
    s_values: &[f32],
    v_angles: &[f32],
    m: usize,
    n: usize,
    bandwidth: usize,
    n_sweeps: usize,
) -> Vec<f32> {
    let u = build_orthogonal_sweep(u_angles, m, n_sweeps); // m x m
    let s_dense = banded_to_dense(s_values, m, n, bandwidth); // m x n
    let v = build_orthogonal_sweep(v_angles, n, n_sweeps); // n x n
    let us = matmul(&u, &s_dense, m, m, n); // m x n
    let v_t = transpose(&v, n, n); // n x n
    matmul(&us, &v_t, m, n, n) // m x n
}

// ---------------------------------------------------------------------------
// Number of Givens angles for sweep-based rotations
// ---------------------------------------------------------------------------

fn num_angles(size: usize, n_sweeps: usize) -> usize {
    n_sweeps * (size - 1)
}

// ---------------------------------------------------------------------------
// WannierGPT struct
// ---------------------------------------------------------------------------

/// Wannier-factorized GPT model.
/// Large weight matrices (QKV, attn_proj, FF1, FF2) are stored as U @ S @ V^T
/// with sweep-based Givens rotations for U, V and banded S.
pub struct WannierGPT {
    // Standard embeddings (not factorized)
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,

    // Per-layer Wannier factors for QKV weight: (n_embd, 3*n_embd)
    pub qkv_u_angles: Vec<Vec<f32>>,
    pub qkv_v_angles: Vec<Vec<f32>>,
    pub qkv_s_values: Vec<Vec<f32>>,

    // Per-layer Wannier factors for attn projection: (n_embd, n_embd)
    pub proj_u_angles: Vec<Vec<f32>>,
    pub proj_v_angles: Vec<Vec<f32>>,
    pub proj_s_values: Vec<Vec<f32>>,

    // Per-layer Wannier factors for FF1: (n_embd, 4*n_embd)
    pub ff1_u_angles: Vec<Vec<f32>>,
    pub ff1_v_angles: Vec<Vec<f32>>,
    pub ff1_s_values: Vec<Vec<f32>>,

    // Per-layer Wannier factors for FF2: (4*n_embd, n_embd)
    pub ff2_u_angles: Vec<Vec<f32>>,
    pub ff2_v_angles: Vec<Vec<f32>>,
    pub ff2_s_values: Vec<Vec<f32>>,

    // Layer norms and biases (standard, small)
    pub ln1_gamma: Vec<Vec<f32>>,
    pub ln1_beta: Vec<Vec<f32>>,
    pub ln2_gamma: Vec<Vec<f32>>,
    pub ln2_beta: Vec<Vec<f32>>,
    pub ff_b1: Vec<Vec<f32>>,
    pub ff_b2: Vec<Vec<f32>>,

    pub ln_f_gamma: Vec<f32>,
    pub ln_f_beta: Vec<f32>,
    pub lm_head: Vec<f32>,

    pub config: Config,
    pub bandwidth: usize,
    pub n_sweeps: usize,

    // Adam optimizer state
    pub w_m: Vec<Vec<f32>>,
    pub w_v: Vec<Vec<f32>>,
    pub w_t: usize,
}

/// Gradient storage mirroring WannierGPT parameters.
pub struct WannierGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,

    pub qkv_u_angles: Vec<Vec<f32>>,
    pub qkv_v_angles: Vec<Vec<f32>>,
    pub qkv_s_values: Vec<Vec<f32>>,

    pub proj_u_angles: Vec<Vec<f32>>,
    pub proj_v_angles: Vec<Vec<f32>>,
    pub proj_s_values: Vec<Vec<f32>>,

    pub ff1_u_angles: Vec<Vec<f32>>,
    pub ff1_v_angles: Vec<Vec<f32>>,
    pub ff1_s_values: Vec<Vec<f32>>,

    pub ff2_u_angles: Vec<Vec<f32>>,
    pub ff2_v_angles: Vec<Vec<f32>>,
    pub ff2_s_values: Vec<Vec<f32>>,

    pub ln1_gamma: Vec<Vec<f32>>,
    pub ln1_beta: Vec<Vec<f32>>,
    pub ln2_gamma: Vec<Vec<f32>>,
    pub ln2_beta: Vec<Vec<f32>>,
    pub ff_b1: Vec<Vec<f32>>,
    pub ff_b2: Vec<Vec<f32>>,

    pub ln_f_gamma: Vec<f32>,
    pub ln_f_beta: Vec<f32>,
    pub lm_head: Vec<f32>,
}

/// Forward cache for backward pass.
#[derive(Default)]
struct WannierLayerCache {
    x_input: Vec<f32>,
    // Materialized effective weights (needed for backward)
    eff_qkv_w: Vec<f32>,
    eff_attn_proj: Vec<f32>,
    eff_ff_w1: Vec<f32>,
    eff_ff_w2: Vec<f32>,
    ln1_out: Vec<f32>,
    #[allow(dead_code)]
    ln1_mean: Vec<f32>,
    #[allow(dead_code)]
    ln1_rstd: Vec<f32>,
    qkv: Vec<f32>,
    attn_weights: Vec<f32>,
    attn_out_pre_proj: Vec<f32>,
    x_after_attn_residual: Vec<f32>,
    ln2_out: Vec<f32>,
    #[allow(dead_code)]
    ln2_mean: Vec<f32>,
    #[allow(dead_code)]
    ln2_rstd: Vec<f32>,
    ff_pre_gelu: Vec<f32>,
    ff_post_gelu: Vec<f32>,
}

#[allow(dead_code)]
struct WannierForwardCache {
    tokens: Vec<usize>,
    x_after_emb: Vec<f32>,
    layer_caches: Vec<WannierLayerCache>,
    x_before_final_ln: Vec<f32>,
    x_after_final_ln: Vec<f32>,
}

// ---------------------------------------------------------------------------
// WannierGradients impl
// ---------------------------------------------------------------------------

impl WannierGradients {
    pub fn zero_like(cfg: &Config, bandwidth: usize, n_sweeps: usize) -> Self {
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;
        let nl = cfg.n_layer;

        let qkv_u_n = num_angles(e, n_sweeps);
        let qkv_v_n = num_angles(3 * e, n_sweeps);
        let qkv_s_n = e * bandwidth;

        let proj_u_n = num_angles(e, n_sweeps);
        let proj_v_n = num_angles(e, n_sweeps);
        let proj_s_n = e * bandwidth;

        let ff1_u_n = num_angles(e, n_sweeps);
        let ff1_v_n = num_angles(inner, n_sweeps);
        let ff1_s_n = e * bandwidth;

        let ff2_u_n = num_angles(inner, n_sweeps);
        let ff2_v_n = num_angles(e, n_sweeps);
        let ff2_s_n = inner * bandwidth;

        Self {
            token_emb: vec![0.0; v * e],
            pos_emb: vec![0.0; cfg.block_size * e],

            qkv_u_angles: vec![vec![0.0; qkv_u_n]; nl],
            qkv_v_angles: vec![vec![0.0; qkv_v_n]; nl],
            qkv_s_values: vec![vec![0.0; qkv_s_n]; nl],

            proj_u_angles: vec![vec![0.0; proj_u_n]; nl],
            proj_v_angles: vec![vec![0.0; proj_v_n]; nl],
            proj_s_values: vec![vec![0.0; proj_s_n]; nl],

            ff1_u_angles: vec![vec![0.0; ff1_u_n]; nl],
            ff1_v_angles: vec![vec![0.0; ff1_v_n]; nl],
            ff1_s_values: vec![vec![0.0; ff1_s_n]; nl],

            ff2_u_angles: vec![vec![0.0; ff2_u_n]; nl],
            ff2_v_angles: vec![vec![0.0; ff2_v_n]; nl],
            ff2_s_values: vec![vec![0.0; ff2_s_n]; nl],

            ln1_gamma: vec![vec![0.0; e]; nl],
            ln1_beta: vec![vec![0.0; e]; nl],
            ln2_gamma: vec![vec![0.0; e]; nl],
            ln2_beta: vec![vec![0.0; e]; nl],
            ff_b1: vec![vec![0.0; inner]; nl],
            ff_b2: vec![vec![0.0; e]; nl],

            ln_f_gamma: vec![0.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: vec![0.0; e * v],
        }
    }

    pub fn accumulate(&mut self, other: &WannierGradients) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        for l in 0..self.ln1_gamma.len() {
            add_vecs(&mut self.qkv_u_angles[l], &other.qkv_u_angles[l]);
            add_vecs(&mut self.qkv_v_angles[l], &other.qkv_v_angles[l]);
            add_vecs(&mut self.qkv_s_values[l], &other.qkv_s_values[l]);
            add_vecs(&mut self.proj_u_angles[l], &other.proj_u_angles[l]);
            add_vecs(&mut self.proj_v_angles[l], &other.proj_v_angles[l]);
            add_vecs(&mut self.proj_s_values[l], &other.proj_s_values[l]);
            add_vecs(&mut self.ff1_u_angles[l], &other.ff1_u_angles[l]);
            add_vecs(&mut self.ff1_v_angles[l], &other.ff1_v_angles[l]);
            add_vecs(&mut self.ff1_s_values[l], &other.ff1_s_values[l]);
            add_vecs(&mut self.ff2_u_angles[l], &other.ff2_u_angles[l]);
            add_vecs(&mut self.ff2_v_angles[l], &other.ff2_v_angles[l]);
            add_vecs(&mut self.ff2_s_values[l], &other.ff2_s_values[l]);
            add_vecs(&mut self.ln1_gamma[l], &other.ln1_gamma[l]);
            add_vecs(&mut self.ln1_beta[l], &other.ln1_beta[l]);
            add_vecs(&mut self.ln2_gamma[l], &other.ln2_gamma[l]);
            add_vecs(&mut self.ln2_beta[l], &other.ln2_beta[l]);
            add_vecs(&mut self.ff_b1[l], &other.ff_b1[l]);
            add_vecs(&mut self.ff_b2[l], &other.ff_b2[l]);
        }
        add_vecs(&mut self.ln_f_gamma, &other.ln_f_gamma);
        add_vecs(&mut self.ln_f_beta, &other.ln_f_beta);
        add_vecs(&mut self.lm_head, &other.lm_head);
    }

    pub fn scale(&mut self, factor: f32) {
        scale_vec(&mut self.token_emb, factor);
        scale_vec(&mut self.pos_emb, factor);
        for l in 0..self.ln1_gamma.len() {
            scale_vec(&mut self.qkv_u_angles[l], factor);
            scale_vec(&mut self.qkv_v_angles[l], factor);
            scale_vec(&mut self.qkv_s_values[l], factor);
            scale_vec(&mut self.proj_u_angles[l], factor);
            scale_vec(&mut self.proj_v_angles[l], factor);
            scale_vec(&mut self.proj_s_values[l], factor);
            scale_vec(&mut self.ff1_u_angles[l], factor);
            scale_vec(&mut self.ff1_v_angles[l], factor);
            scale_vec(&mut self.ff1_s_values[l], factor);
            scale_vec(&mut self.ff2_u_angles[l], factor);
            scale_vec(&mut self.ff2_v_angles[l], factor);
            scale_vec(&mut self.ff2_s_values[l], factor);
            scale_vec(&mut self.ln1_gamma[l], factor);
            scale_vec(&mut self.ln1_beta[l], factor);
            scale_vec(&mut self.ln2_gamma[l], factor);
            scale_vec(&mut self.ln2_beta[l], factor);
            scale_vec(&mut self.ff_b1[l], factor);
            scale_vec(&mut self.ff_b2[l], factor);
        }
        scale_vec(&mut self.ln_f_gamma, factor);
        scale_vec(&mut self.ln_f_beta, factor);
        scale_vec(&mut self.lm_head, factor);
    }
}

// ---------------------------------------------------------------------------
// WannierGPT impl
// ---------------------------------------------------------------------------

impl WannierGPT {
    pub fn new(config: Config, bandwidth: usize, n_sweeps: usize) -> Self {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let bs = config.block_size;
        let inner = 4 * e;

        let emb_scale = 0.02;
        let angle_scale = 0.01; // small initial rotations
        let s_scale = (0.02 / (nl as f32).sqrt()).max(0.001);

        let mut qkv_u_angles = Vec::new();
        let mut qkv_v_angles = Vec::new();
        let mut qkv_s_values = Vec::new();
        let mut proj_u_angles = Vec::new();
        let mut proj_v_angles = Vec::new();
        let mut proj_s_values = Vec::new();
        let mut ff1_u_angles = Vec::new();
        let mut ff1_v_angles = Vec::new();
        let mut ff1_s_values = Vec::new();
        let mut ff2_u_angles = Vec::new();
        let mut ff2_v_angles = Vec::new();
        let mut ff2_s_values = Vec::new();

        let mut ln1_gamma = Vec::new();
        let mut ln1_beta_v = Vec::new();
        let mut ln2_gamma = Vec::new();
        let mut ln2_beta_v = Vec::new();
        let mut ff_b1 = Vec::new();
        let mut ff_b2 = Vec::new();

        for _ in 0..nl {
            // QKV: (e, 3e)
            qkv_u_angles.push(randn_vec(num_angles(e, n_sweeps), angle_scale));
            qkv_v_angles.push(randn_vec(num_angles(3 * e, n_sweeps), angle_scale));
            qkv_s_values.push(randn_vec(e * bandwidth, s_scale));

            // Attn proj: (e, e)
            proj_u_angles.push(randn_vec(num_angles(e, n_sweeps), angle_scale));
            proj_v_angles.push(randn_vec(num_angles(e, n_sweeps), angle_scale));
            proj_s_values.push(randn_vec(e * bandwidth, s_scale));

            // FF1: (e, inner)
            ff1_u_angles.push(randn_vec(num_angles(e, n_sweeps), angle_scale));
            ff1_v_angles.push(randn_vec(num_angles(inner, n_sweeps), angle_scale));
            ff1_s_values.push(randn_vec(e * bandwidth, s_scale));

            // FF2: (inner, e)
            ff2_u_angles.push(randn_vec(num_angles(inner, n_sweeps), angle_scale));
            ff2_v_angles.push(randn_vec(num_angles(e, n_sweeps), angle_scale));
            ff2_s_values.push(randn_vec(inner * bandwidth, s_scale));

            ln1_gamma.push(vec![1.0; e]);
            ln1_beta_v.push(vec![0.0; e]);
            ln2_gamma.push(vec![1.0; e]);
            ln2_beta_v.push(vec![0.0; e]);
            ff_b1.push(vec![0.0; inner]);
            ff_b2.push(vec![0.0; e]);
        }

        let mut model = WannierGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            qkv_u_angles,
            qkv_v_angles,
            qkv_s_values,
            proj_u_angles,
            proj_v_angles,
            proj_s_values,
            ff1_u_angles,
            ff1_v_angles,
            ff1_s_values,
            ff2_u_angles,
            ff2_v_angles,
            ff2_s_values,
            ln1_gamma,
            ln1_beta: ln1_beta_v,
            ln2_gamma,
            ln2_beta: ln2_beta_v,
            ff_b1,
            ff_b2,
            ln_f_gamma: vec![1.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: randn_vec(e * v, emb_scale),
            config,
            bandwidth,
            n_sweeps,
            w_m: Vec::new(),
            w_v: Vec::new(),
            w_t: 0,
        };

        let param_sizes = model.param_sizes();
        model.w_m = param_sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.w_v = param_sizes.iter().map(|&s| vec![0.0; s]).collect();

        model
    }

    /// Count total trainable parameters.
    pub fn count_params(&self) -> usize {
        self.param_sizes().iter().sum()
    }

    fn param_sizes(&self) -> Vec<usize> {
        let nl = self.config.n_layer;
        let mut sizes = Vec::new();
        sizes.push(self.token_emb.len());
        sizes.push(self.pos_emb.len());
        for l in 0..nl {
            sizes.push(self.qkv_u_angles[l].len());
            sizes.push(self.qkv_v_angles[l].len());
            sizes.push(self.qkv_s_values[l].len());
            sizes.push(self.proj_u_angles[l].len());
            sizes.push(self.proj_v_angles[l].len());
            sizes.push(self.proj_s_values[l].len());
            sizes.push(self.ff1_u_angles[l].len());
            sizes.push(self.ff1_v_angles[l].len());
            sizes.push(self.ff1_s_values[l].len());
            sizes.push(self.ff2_u_angles[l].len());
            sizes.push(self.ff2_v_angles[l].len());
            sizes.push(self.ff2_s_values[l].len());
            sizes.push(self.ln1_gamma[l].len());
            sizes.push(self.ln1_beta[l].len());
            sizes.push(self.ln2_gamma[l].len());
            sizes.push(self.ln2_beta[l].len());
            sizes.push(self.ff_b1[l].len());
            sizes.push(self.ff_b2[l].len());
        }
        sizes.push(self.ln_f_gamma.len());
        sizes.push(self.ln_f_beta.len());
        sizes.push(self.lm_head.len());
        sizes
    }

    /// Forward pass with cache for backward.
    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, WannierForwardCache) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = e / nh;

        // Embedding lookup
        let mut x = vec![0.0f32; t * e];
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                x[i * e + j] = self.token_emb[tok * e + j] + self.pos_emb[i * e + j];
            }
        }

        let mut cache = WannierForwardCache {
            tokens: tokens.to_vec(),
            x_after_emb: x.clone(),
            layer_caches: Vec::new(),
            x_before_final_ln: Vec::new(),
            x_after_final_ln: Vec::new(),
        };

        let inner = 4 * e;

        for l in 0..cfg.n_layer {
            let mut lc = WannierLayerCache::default();
            lc.x_input = x.clone();

            // Materialize effective weights from Wannier factors
            let eff_qkv_w = compute_effective_weight(
                &self.qkv_u_angles[l],
                &self.qkv_s_values[l],
                &self.qkv_v_angles[l],
                e, 3 * e, self.bandwidth, self.n_sweeps,
            );
            let eff_attn_proj = compute_effective_weight(
                &self.proj_u_angles[l],
                &self.proj_s_values[l],
                &self.proj_v_angles[l],
                e, e, self.bandwidth, self.n_sweeps,
            );
            let eff_ff_w1 = compute_effective_weight(
                &self.ff1_u_angles[l],
                &self.ff1_s_values[l],
                &self.ff1_v_angles[l],
                e, inner, self.bandwidth, self.n_sweeps,
            );
            let eff_ff_w2 = compute_effective_weight(
                &self.ff2_u_angles[l],
                &self.ff2_s_values[l],
                &self.ff2_v_angles[l],
                inner, e, self.bandwidth, self.n_sweeps,
            );

            // Layer norm 1
            let (ln1_out, ln1_mean, ln1_rstd) =
                layer_norm(&x, &self.ln1_gamma[l], &self.ln1_beta[l], t, e);
            lc.ln1_out = ln1_out.clone();
            lc.ln1_mean = ln1_mean;
            lc.ln1_rstd = ln1_rstd;

            // QKV projection
            let qkv = matmul(&ln1_out, &eff_qkv_w, t, e, 3 * e);
            lc.qkv = qkv.clone();

            // Multi-head attention
            let mut attn_out = vec![0.0f32; t * e];
            let mut all_attn_weights = vec![0.0f32; nh * t * t];

            for h in 0..nh {
                let scale = 1.0 / (hs as f32).sqrt();
                for i in 0..t {
                    for j in 0..t {
                        if j > i {
                            all_attn_weights[h * t * t + i * t + j] = f32::NEG_INFINITY;
                        } else {
                            let mut dot = 0.0f32;
                            for k in 0..hs {
                                let qi = qkv[i * 3 * e + h * hs + k];
                                let kj = qkv[j * 3 * e + e + h * hs + k];
                                dot += qi * kj;
                            }
                            all_attn_weights[h * t * t + i * t + j] = dot * scale;
                        }
                    }
                }

                // Softmax per row
                for i in 0..t {
                    let offset = h * t * t + i * t;
                    let max_val = all_attn_weights[offset..offset + t]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for j in 0..t {
                        let exp_val = (all_attn_weights[offset + j] - max_val).exp();
                        all_attn_weights[offset + j] = exp_val;
                        sum += exp_val;
                    }
                    for j in 0..t {
                        all_attn_weights[offset + j] /= sum;
                    }
                }

                // Weighted sum of V
                for i in 0..t {
                    for k in 0..hs {
                        let mut sum = 0.0f32;
                        for j in 0..t {
                            let w = all_attn_weights[h * t * t + i * t + j];
                            let vj = qkv[j * 3 * e + 2 * e + h * hs + k];
                            sum += w * vj;
                        }
                        attn_out[i * e + h * hs + k] = sum;
                    }
                }
            }
            lc.attn_weights = all_attn_weights;
            lc.attn_out_pre_proj = attn_out.clone();

            // Output projection
            let proj_out = matmul(&attn_out, &eff_attn_proj, t, e, e);
            for i in 0..t * e {
                x[i] += proj_out[i];
            }
            lc.x_after_attn_residual = x.clone();

            // Layer norm 2
            let (ln2_out, ln2_mean, ln2_rstd) =
                layer_norm(&x, &self.ln2_gamma[l], &self.ln2_beta[l], t, e);
            lc.ln2_out = ln2_out.clone();
            lc.ln2_mean = ln2_mean;
            lc.ln2_rstd = ln2_rstd;

            // Feed-forward
            let mut ff_hidden = matmul(&ln2_out, &eff_ff_w1, t, e, inner);
            for i in 0..t {
                for j in 0..inner {
                    ff_hidden[i * inner + j] += self.ff_b1[l][j];
                }
            }
            lc.ff_pre_gelu = ff_hidden.clone();

            // GELU
            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            for val in ff_hidden.iter_mut() {
                let x3 = *val * *val * *val;
                let inner_val = sqrt_2_over_pi * (*val + 0.044715 * x3);
                *val = 0.5 * *val * (1.0 + inner_val.tanh());
            }
            lc.ff_post_gelu = ff_hidden.clone();

            let mut ff_out = matmul(&ff_hidden, &eff_ff_w2, t, inner, e);
            for i in 0..t {
                for j in 0..e {
                    ff_out[i * e + j] += self.ff_b2[l][j];
                }
            }
            for i in 0..t * e {
                x[i] += ff_out[i];
            }

            lc.eff_qkv_w = eff_qkv_w;
            lc.eff_attn_proj = eff_attn_proj;
            lc.eff_ff_w1 = eff_ff_w1;
            lc.eff_ff_w2 = eff_ff_w2;

            cache.layer_caches.push(lc);
        }

        cache.x_before_final_ln = x.clone();
        let (ln_out, _, _) = layer_norm(&x, &self.ln_f_gamma, &self.ln_f_beta, t, e);
        cache.x_after_final_ln = ln_out.clone();
        let logits = matmul(&ln_out, &self.lm_head, t, e, v);

        (logits, cache)
    }

    /// Forward + backward returning loss and gradients.
    ///
    /// Strategy: materialize W = U @ S @ V^T, run standard forward/backward to get dW,
    /// then project: dS = U^T @ dW @ V (since U, V orthogonal).
    /// For Givens angles: use numerical gradient (few angles, cheap).
    pub fn forward_backward(
        &self,
        tokens: &[usize],
        targets: &[usize],
    ) -> (f32, WannierGradients) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;

        let (logits, cache) = self.forward_with_cache(tokens);

        // Cross-entropy loss
        let mut probs = vec![0.0f32; t * v];
        let mut loss = 0.0f32;
        for i in 0..t {
            let offset = i * v;
            let max_val = logits[offset..offset + v]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v {
                probs[offset + j] = (logits[offset + j] - max_val).exp();
                sum += probs[offset + j];
            }
            for j in 0..v {
                probs[offset + j] /= sum;
            }
            loss -= probs[offset + targets[i]].max(1e-10).ln();
        }
        loss /= t as f32;

        let mut d_logits = probs;
        for i in 0..t {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v {
                d_logits[i * v + j] /= t as f32;
            }
        }

        let mut grads = WannierGradients::zero_like(cfg, self.bandwidth, self.n_sweeps);

        // Backward through LM head
        let d_ln_out = matmul_backward_both(
            &cache.x_after_final_ln,
            &self.lm_head,
            &d_logits,
            t, e, v,
            &mut grads.lm_head,
        );

        // Backward through final layer norm
        let mut dx = layer_norm_backward(
            &cache.x_before_final_ln,
            &d_ln_out,
            &self.ln_f_gamma,
            t, e,
            &mut grads.ln_f_gamma,
            &mut grads.ln_f_beta,
        );

        // Backward through transformer blocks (reverse order)
        for l in (0..cfg.n_layer).rev() {
            let lc = &cache.layer_caches[l];

            // --- FF backward ---
            let d_ff_out = dx.clone();
            for i in 0..t {
                for j in 0..e {
                    grads.ff_b2[l][j] += d_ff_out[i * e + j];
                }
            }

            let mut d_eff_ff_w2 = vec![0.0f32; inner * e];
            let d_ff_hidden = matmul_backward_both(
                &lc.ff_post_gelu, &lc.eff_ff_w2, &d_ff_out,
                t, inner, e, &mut d_eff_ff_w2,
            );

            // Project dW back to Wannier factors for FF2: (inner, e)
            self.wannier_backward_weight(
                &d_eff_ff_w2, l, inner, e,
                &self.ff2_u_angles[l], &self.ff2_v_angles[l],
                &self.ff2_s_values[l],
                &mut grads.ff2_s_values[l],
                &mut grads.ff2_u_angles[l],
                &mut grads.ff2_v_angles[l],
            );

            let d_ff_pre_gelu = gelu_backward(&lc.ff_pre_gelu, &d_ff_hidden);
            for i in 0..t {
                for j in 0..inner {
                    grads.ff_b1[l][j] += d_ff_pre_gelu[i * inner + j];
                }
            }

            let mut d_eff_ff_w1 = vec![0.0f32; e * inner];
            let d_ln2_out = matmul_backward_both(
                &lc.ln2_out, &lc.eff_ff_w1, &d_ff_pre_gelu,
                t, e, inner, &mut d_eff_ff_w1,
            );

            // Project dW back to Wannier factors for FF1: (e, inner)
            self.wannier_backward_weight(
                &d_eff_ff_w1, l, e, inner,
                &self.ff1_u_angles[l], &self.ff1_v_angles[l],
                &self.ff1_s_values[l],
                &mut grads.ff1_s_values[l],
                &mut grads.ff1_u_angles[l],
                &mut grads.ff1_v_angles[l],
            );

            let d_from_ln2 = layer_norm_backward(
                &lc.x_after_attn_residual, &d_ln2_out, &self.ln2_gamma[l],
                t, e, &mut grads.ln2_gamma[l], &mut grads.ln2_beta[l],
            );

            let mut dx_mid = dx;
            for i in 0..t * e {
                dx_mid[i] += d_from_ln2[i];
            }

            // --- Attention backward ---
            let d_proj_out = dx_mid.clone();

            let mut d_eff_attn_proj = vec![0.0f32; e * e];
            let d_attn_out = matmul_backward_both(
                &lc.attn_out_pre_proj, &lc.eff_attn_proj, &d_proj_out,
                t, e, e, &mut d_eff_attn_proj,
            );

            // Project dW back to Wannier factors for attn_proj: (e, e)
            self.wannier_backward_weight(
                &d_eff_attn_proj, l, e, e,
                &self.proj_u_angles[l], &self.proj_v_angles[l],
                &self.proj_s_values[l],
                &mut grads.proj_s_values[l],
                &mut grads.proj_u_angles[l],
                &mut grads.proj_v_angles[l],
            );

            // Multi-head attention backward
            let nh = cfg.n_head;
            let hs = e / nh;
            let mut d_qkv = vec![0.0f32; t * 3 * e];

            for h in 0..nh {
                // dV from weighted sum
                for i in 0..t {
                    for k in 0..hs {
                        let d_out = d_attn_out[i * e + h * hs + k];
                        for j in 0..t {
                            let w = lc.attn_weights[h * t * t + i * t + j];
                            d_qkv[j * 3 * e + 2 * e + h * hs + k] += w * d_out;
                        }
                    }
                }

                // d_attn_score
                let mut d_attn_score = vec![0.0f32; t * t];
                for i in 0..t {
                    for j in 0..t {
                        let mut dw = 0.0f32;
                        for k in 0..hs {
                            let vj = lc.qkv[j * 3 * e + 2 * e + h * hs + k];
                            dw += vj * d_attn_out[i * e + h * hs + k];
                        }
                        d_attn_score[i * t + j] = dw;
                    }
                }

                // Softmax backward
                for i in 0..t {
                    let mut dot = 0.0f32;
                    for j in 0..t {
                        dot += d_attn_score[i * t + j]
                            * lc.attn_weights[h * t * t + i * t + j];
                    }
                    for j in 0..t {
                        let w = lc.attn_weights[h * t * t + i * t + j];
                        d_attn_score[i * t + j] = w * (d_attn_score[i * t + j] - dot);
                    }
                }

                let scale = 1.0 / (hs as f32).sqrt();
                for val in d_attn_score.iter_mut() {
                    *val *= scale;
                }

                // dQ, dK
                for i in 0..t {
                    for j in 0..=i {
                        let ds = d_attn_score[i * t + j];
                        if ds.abs() > 1e-12 {
                            for k in 0..hs {
                                let qi = lc.qkv[i * 3 * e + h * hs + k];
                                let kj = lc.qkv[j * 3 * e + e + h * hs + k];
                                d_qkv[i * 3 * e + h * hs + k] += ds * kj;
                                d_qkv[j * 3 * e + e + h * hs + k] += ds * qi;
                            }
                        }
                    }
                }
            }

            // Backward through QKV projection
            let mut d_eff_qkv_w = vec![0.0f32; e * 3 * e];
            let d_ln1_out = matmul_backward_both(
                &lc.ln1_out, &lc.eff_qkv_w, &d_qkv,
                t, e, 3 * e, &mut d_eff_qkv_w,
            );

            // Project dW back to Wannier factors for QKV: (e, 3e)
            self.wannier_backward_weight(
                &d_eff_qkv_w, l, e, 3 * e,
                &self.qkv_u_angles[l], &self.qkv_v_angles[l],
                &self.qkv_s_values[l],
                &mut grads.qkv_s_values[l],
                &mut grads.qkv_u_angles[l],
                &mut grads.qkv_v_angles[l],
            );

            // Backward through layer norm 1
            let d_from_attn = layer_norm_backward(
                &lc.x_input, &d_ln1_out, &self.ln1_gamma[l],
                t, e, &mut grads.ln1_gamma[l], &mut grads.ln1_beta[l],
            );

            dx = dx_mid;
            for i in 0..t * e {
                dx[i] += d_from_attn[i];
            }
        }

        // Embedding gradients
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                grads.token_emb[tok * e + j] += dx[i * e + j];
                grads.pos_emb[i * e + j] += dx[i * e + j];
            }
        }

        (loss, grads)
    }

    /// Project gradient dW back through Wannier factorization W = U @ S @ V^T.
    ///
    /// For S (banded): dS_dense = U^T @ dW @ V, then extract banded entries.
    /// For Givens angles: numerical gradient via finite differences.
    fn wannier_backward_weight(
        &self,
        d_w: &[f32],
        _layer: usize,
        m: usize,
        n: usize,
        u_angles: &[f32],
        v_angles: &[f32],
        s_values: &[f32],
        d_s_values: &mut Vec<f32>,
        d_u_angles: &mut Vec<f32>,
        d_v_angles: &mut Vec<f32>,
    ) {
        let bw = self.bandwidth;
        let ns = self.n_sweeps;

        // --- Analytical gradient for S ---
        // dS_dense = U^T @ dW @ V
        let u = build_orthogonal_sweep(u_angles, m, ns);
        let v = build_orthogonal_sweep(v_angles, n, ns);
        let u_t = transpose(&u, m, m);
        let ut_dw = matmul(&u_t, d_w, m, m, n); // m x n
        let ds_dense = matmul(&ut_dw, &v, m, n, n); // m x n
        let ds_banded = dense_to_banded_grad(&ds_dense, m, n, bw);
        add_vecs(d_s_values, &ds_banded);

        // --- Numerical gradient for Givens angles ---
        // Perturb each angle by eps, recompute W, measure directional derivative.
        let eps = 1e-3f32;

        let s_dense = banded_to_dense(s_values, m, n, bw);
        let v_t = transpose(&v, n, n);
        for i in 0..u_angles.len() {
            let mut angles_plus = u_angles.to_vec();
            angles_plus[i] += eps;
            let u_plus = build_orthogonal_sweep(&angles_plus, m, ns);
            let us_plus = matmul(&u_plus, &s_dense, m, m, n);
            let w_plus = matmul(&us_plus, &v_t, m, n, n);

            let mut angles_minus = u_angles.to_vec();
            angles_minus[i] -= eps;
            let u_minus = build_orthogonal_sweep(&angles_minus, m, ns);
            let us_minus = matmul(&u_minus, &s_dense, m, m, n);
            let w_minus = matmul(&us_minus, &v_t, m, n, n);

            let mut grad = 0.0f32;
            for k in 0..m * n {
                grad += d_w[k] * (w_plus[k] - w_minus[k]);
            }
            d_u_angles[i] += grad / (2.0 * eps);
        }

        // Gradient for V angles
        let us = matmul(&u, &s_dense, m, m, n);
        for i in 0..v_angles.len() {
            let mut angles_plus = v_angles.to_vec();
            angles_plus[i] += eps;
            let v_plus = build_orthogonal_sweep(&angles_plus, n, ns);
            let v_t_plus = transpose(&v_plus, n, n);
            let w_plus = matmul(&us, &v_t_plus, m, n, n);

            let mut angles_minus = v_angles.to_vec();
            angles_minus[i] -= eps;
            let v_minus = build_orthogonal_sweep(&angles_minus, n, ns);
            let v_t_minus = transpose(&v_minus, n, n);
            let w_minus = matmul(&us, &v_t_minus, m, n, n);

            let mut grad = 0.0f32;
            for k in 0..m * n {
                grad += d_w[k] * (w_plus[k] - w_minus[k]);
            }
            d_v_angles[i] += grad / (2.0 * eps);
        }
    }

    /// Forward pass for inference (no cache).
    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        self.forward_with_cache(tokens).0
    }

    /// Generate text autoregressively.
    pub fn generate(&self, start_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut tokens = start_tokens.to_vec();
        let mut rng = rand::thread_rng();
        let v = self.config.vocab_size;

        for _ in 0..max_new_tokens {
            let start = if tokens.len() > self.config.block_size {
                tokens.len() - self.config.block_size
            } else {
                0
            };
            let context = &tokens[start..];
            let logits = self.forward(context);
            let t = context.len();

            let last_offset = (t - 1) * v;
            let last_logits = &logits[last_offset..last_offset + v];

            let temperature = 0.8f32;
            let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; v];
            let mut sum = 0.0f32;
            for i in 0..v {
                probs[i] = ((last_logits[i] - max_val) / temperature).exp();
                sum += probs[i];
            }
            for p in probs.iter_mut() {
                *p /= sum;
            }

            let r: f32 = rng.r#gen();
            let mut cumsum = 0.0f32;
            let mut next_token = 0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    next_token = i;
                    break;
                }
            }
            tokens.push(next_token);
        }

        tokens
    }

    /// Apply gradients with Adam optimizer.
    pub fn apply_gradients(&mut self, grads: &WannierGradients, lr: f32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        self.w_t += 1;
        let t_f = self.w_t as f32;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);

        let mut idx = 0usize;

        macro_rules! adam_update {
            ($param:expr, $grad:expr) => {{
                adam_step(
                    $param, $grad, &mut self.w_m[idx], &mut self.w_v[idx],
                    lr, beta1, beta2, eps, bc1, bc2,
                );
                idx += 1;
                let _ = idx;
            }};
        }

        adam_update!(&mut self.token_emb, &grads.token_emb);
        adam_update!(&mut self.pos_emb, &grads.pos_emb);

        for l in 0..self.config.n_layer {
            adam_update!(&mut self.qkv_u_angles[l], &grads.qkv_u_angles[l]);
            adam_update!(&mut self.qkv_v_angles[l], &grads.qkv_v_angles[l]);
            adam_update!(&mut self.qkv_s_values[l], &grads.qkv_s_values[l]);
            adam_update!(&mut self.proj_u_angles[l], &grads.proj_u_angles[l]);
            adam_update!(&mut self.proj_v_angles[l], &grads.proj_v_angles[l]);
            adam_update!(&mut self.proj_s_values[l], &grads.proj_s_values[l]);
            adam_update!(&mut self.ff1_u_angles[l], &grads.ff1_u_angles[l]);
            adam_update!(&mut self.ff1_v_angles[l], &grads.ff1_v_angles[l]);
            adam_update!(&mut self.ff1_s_values[l], &grads.ff1_s_values[l]);
            adam_update!(&mut self.ff2_u_angles[l], &grads.ff2_u_angles[l]);
            adam_update!(&mut self.ff2_v_angles[l], &grads.ff2_v_angles[l]);
            adam_update!(&mut self.ff2_s_values[l], &grads.ff2_s_values[l]);
            adam_update!(&mut self.ln1_gamma[l], &grads.ln1_gamma[l]);
            adam_update!(&mut self.ln1_beta[l], &grads.ln1_beta[l]);
            adam_update!(&mut self.ln2_gamma[l], &grads.ln2_gamma[l]);
            adam_update!(&mut self.ln2_beta[l], &grads.ln2_beta[l]);
            adam_update!(&mut self.ff_b1[l], &grads.ff_b1[l]);
            adam_update!(&mut self.ff_b2[l], &grads.ff_b2[l]);
        }

        adam_update!(&mut self.ln_f_gamma, &grads.ln_f_gamma);
        adam_update!(&mut self.ln_f_beta, &grads.ln_f_beta);
        adam_update!(&mut self.lm_head, &grads.lm_head);
    }
}
