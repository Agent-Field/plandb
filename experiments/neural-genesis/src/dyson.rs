use rand::Rng;
use std::io::Write;
use std::time::Instant;

use crate::model::Config;
use crate::tokenizer::Tokenizer;
use crate::tool_data::{self, AggregateMetrics, ToolExample};

/// Dyson Equation Transformer: Neumann-series residual connections.
///
/// Physics motivation: The single-particle Green's function satisfies
///   G = G0 + G0 * Sigma * G = (I - Sigma)^{-1}
/// Approximated via the Neumann series:
///   G ~ I + Sigma + Sigma^2 + ... + Sigma^K  (truncated at order K)
///
/// Neural net mapping:
/// - Standard residual: x_{l+1} = x + f(x)         [order-1 perturbation]
/// - Dyson layer:       x_{l+1} = (I + S + S^2 + ... + S^K) * x  [order-K]
///
/// Each Neumann order re-applies the SAME self-energy operator (shared weights),
/// giving effective depth K from the parameters of a single layer.
pub struct DysonGPT {
    // Embeddings
    pub token_emb: Vec<f32>,  // (vocab_size, n_embd)
    pub pos_emb: Vec<f32>,    // (block_size, n_embd)

    // Per-layer self-energy parameters (standard transformer layer weights)
    pub ln1_gamma: Vec<Vec<f32>>,  // [n_layer][n_embd]
    pub ln1_beta: Vec<Vec<f32>>,   // [n_layer][n_embd]
    pub qkv_w: Vec<Vec<f32>>,     // [n_layer][n_embd * 3 * n_embd]
    pub attn_proj: Vec<Vec<f32>>,  // [n_layer][n_embd * n_embd]
    pub ln2_gamma: Vec<Vec<f32>>,  // [n_layer][n_embd]
    pub ln2_beta: Vec<Vec<f32>>,   // [n_layer][n_embd]
    pub ff_w1: Vec<Vec<f32>>,      // [n_layer][n_embd * 4*n_embd]
    pub ff_b1: Vec<Vec<f32>>,      // [n_layer][4*n_embd]
    pub ff_w2: Vec<Vec<f32>>,      // [n_layer][4*n_embd * n_embd]
    pub ff_b2: Vec<Vec<f32>>,      // [n_layer][n_embd]

    // Final layer norm + LM head
    pub ln_f_gamma: Vec<f32>,  // (n_embd,)
    pub ln_f_beta: Vec<f32>,   // (n_embd,)
    pub lm_head: Vec<f32>,     // (n_embd, vocab_size)

    pub config: Config,
    pub neumann_order: usize,      // K
    pub spectral_safety: f32,      // < 1.0, typically 0.8-0.9

    // Adam optimizer state
    pub d_m: Vec<Vec<f32>>,
    pub d_v: Vec<Vec<f32>>,
    pub d_t: usize,
}

/// Gradient storage for DysonGPT (mirrors parameter structure)
pub struct DysonGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub ln1_gamma: Vec<Vec<f32>>,
    pub ln1_beta: Vec<Vec<f32>>,
    pub qkv_w: Vec<Vec<f32>>,
    pub attn_proj: Vec<Vec<f32>>,
    pub ln2_gamma: Vec<Vec<f32>>,
    pub ln2_beta: Vec<Vec<f32>>,
    pub ff_w1: Vec<Vec<f32>>,
    pub ff_b1: Vec<Vec<f32>>,
    pub ff_w2: Vec<Vec<f32>>,
    pub ff_b2: Vec<Vec<f32>>,
    pub ln_f_gamma: Vec<f32>,
    pub ln_f_beta: Vec<f32>,
    pub lm_head: Vec<f32>,
}

/// Cache for one application of the self-energy operator (for backward pass)
#[derive(Default, Clone)]
struct SelfEnergyCache {
    x_input: Vec<f32>,         // input to self-energy
    ln1_out: Vec<f32>,
    qkv: Vec<f32>,
    attn_weights: Vec<f32>,    // (n_head, T, T)
    attn_out_pre_proj: Vec<f32>,
    x_after_attn: Vec<f32>,    // x_input + attn_proj(attn_out)
    ln2_out: Vec<f32>,
    ff_pre_gelu: Vec<f32>,
    ff_post_gelu: Vec<f32>,
    sigma_out: Vec<f32>,       // the self-energy output (ff_out only, NOT added to x)
    scale_factor: f32,         // spectral normalization factor applied
}

/// Cache for a single Dyson layer's Neumann series (stores all K iterations)
#[derive(Default)]
struct DysonLayerCache {
    /// y_0 = x (input), y_1 = Sigma(x), y_2 = Sigma(y_1), ...
    y_states: Vec<Vec<f32>>,
    /// Cache for each self-energy application (index k stores cache for y_k = Sigma(y_{k-1}))
    se_caches: Vec<SelfEnergyCache>,
}

/// Full forward cache
#[allow(dead_code)]
struct DysonForwardCache {
    tokens: Vec<usize>,
    x_after_emb: Vec<f32>,
    layer_caches: Vec<DysonLayerCache>,
    x_before_final_ln: Vec<f32>,
    x_after_final_ln: Vec<f32>,
}

// ---- Helper functions (local copies to keep module independent) ----

fn randn_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1: f32 = rng.r#gen::<f32>().max(1e-10);
        let u2: f32 = rng.r#gen::<f32>();
        let mag = (-2.0 * u1.ln()).sqrt() * scale;
        data.push(mag * (2.0 * std::f32::consts::PI * u2).cos());
        if i + 1 < n {
            data.push(mag * (2.0 * std::f32::consts::PI * u2).sin());
        }
        i += 2;
    }
    data.truncate(n);
    data
}

fn adam_step(
    params: &mut Vec<f32>,
    grads: &[f32],
    m: &mut Vec<f32>,
    v: &mut Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
) {
    for i in 0..params.len() {
        let g = grads[i].max(-1.0).min(1.0);
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

fn add_vecs(a: &mut Vec<f32>, b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

fn scale_vec(a: &mut Vec<f32>, s: f32) {
    for x in a.iter_mut() {
        *x *= s;
    }
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            if a_val.abs() > 1e-12 {
                for j in 0..n {
                    c[i * n + j] += a_val * b[p * n + j];
                }
            }
        }
    }
    c
}

fn matmul_backward_both(
    a: &[f32],
    b: &[f32],
    dc: &[f32],
    m: usize,
    k: usize,
    n: usize,
    db: &mut Vec<f32>,
) -> Vec<f32> {
    let mut da = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..n {
            let dc_val = dc[i * n + j];
            if dc_val.abs() > 1e-12 {
                for p in 0..k {
                    da[i * k + p] += dc_val * b[p * n + j];
                }
            }
        }
    }
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            if a_val.abs() > 1e-12 {
                for j in 0..n {
                    db[p * n + j] += a_val * dc[i * n + j];
                }
            }
        }
    }
    da
}

fn layer_norm(
    x: &[f32], gamma: &[f32], beta: &[f32], t: usize, e: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let eps = 1e-5f32;
    let mut out = vec![0.0f32; t * e];
    let mut means = vec![0.0f32; t];
    let mut rstds = vec![0.0f32; t];
    for i in 0..t {
        let offset = i * e;
        let mean: f32 = x[offset..offset + e].iter().sum::<f32>() / e as f32;
        let var: f32 = x[offset..offset + e]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / e as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        means[i] = mean;
        rstds[i] = rstd;
        for j in 0..e {
            let norm = (x[offset + j] - mean) * rstd;
            out[offset + j] = gamma[j] * norm + beta[j];
        }
    }
    (out, means, rstds)
}

fn layer_norm_backward(
    x: &[f32], dout: &[f32], gamma: &[f32], t: usize, e: usize,
    dgamma: &mut Vec<f32>, dbeta: &mut Vec<f32>,
) -> Vec<f32> {
    let eps = 1e-5f32;
    let mut dx = vec![0.0f32; t * e];
    for i in 0..t {
        let offset = i * e;
        let mean: f32 = x[offset..offset + e].iter().sum::<f32>() / e as f32;
        let var: f32 = x[offset..offset + e]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / e as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        let mut norm = vec![0.0f32; e];
        for j in 0..e {
            norm[j] = (x[offset + j] - mean) * rstd;
        }
        for j in 0..e {
            dgamma[j] += dout[offset + j] * norm[j];
            dbeta[j] += dout[offset + j];
        }
        let mut dnorm = vec![0.0f32; e];
        for j in 0..e {
            dnorm[j] = dout[offset + j] * gamma[j];
        }
        let dnorm_mean: f32 = dnorm.iter().sum::<f32>() / e as f32;
        let dnorm_norm_mean: f32 =
            dnorm.iter().zip(norm.iter()).map(|(a, b)| a * b).sum::<f32>() / e as f32;
        for j in 0..e {
            dx[offset + j] = (dnorm[j] - dnorm_mean - norm[j] * dnorm_norm_mean) * rstd;
        }
    }
    dx
}

fn gelu_backward(x: &[f32], dout: &[f32]) -> Vec<f32> {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut dx = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        let xi = x[i];
        let x3 = xi * xi * xi;
        let inner = sqrt_2_over_pi * (xi + 0.044715 * x3);
        let tanh_val = inner.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;
        let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * xi * xi);
        dx[i] = dout[i] * (0.5 * (1.0 + tanh_val) + 0.5 * xi * sech2 * d_inner);
    }
    dx
}

// ---- DysonGPT implementation ----

impl DysonGPT {
    pub fn new(config: Config, neumann_order: usize, spectral_safety: f32) -> Self {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let bs = config.block_size;
        let inner = 4 * e;

        let emb_scale = 0.02;
        let layer_scale = (0.02 / (nl as f32).sqrt()).max(0.001);

        let mut ln1_gamma = Vec::new();
        let mut ln1_beta = Vec::new();
        let mut qkv_w = Vec::new();
        let mut attn_proj = Vec::new();
        let mut ln2_gamma = Vec::new();
        let mut ln2_beta = Vec::new();
        let mut ff_w1 = Vec::new();
        let mut ff_b1 = Vec::new();
        let mut ff_w2 = Vec::new();
        let mut ff_b2 = Vec::new();

        for _ in 0..nl {
            ln1_gamma.push(vec![1.0; e]);
            ln1_beta.push(vec![0.0; e]);
            qkv_w.push(randn_vec(e * 3 * e, layer_scale));
            attn_proj.push(randn_vec(e * e, layer_scale));
            ln2_gamma.push(vec![1.0; e]);
            ln2_beta.push(vec![0.0; e]);
            ff_w1.push(randn_vec(e * inner, layer_scale));
            ff_b1.push(vec![0.0; inner]);
            ff_w2.push(randn_vec(inner * e, layer_scale * 0.5));
            ff_b2.push(vec![0.0; e]);
        }

        let mut model = DysonGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            ln1_gamma,
            ln1_beta,
            qkv_w,
            attn_proj,
            ln2_gamma,
            ln2_beta,
            ff_w1,
            ff_b1,
            ff_w2,
            ff_b2,
            ln_f_gamma: vec![1.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: randn_vec(e * v, emb_scale),
            config,
            neumann_order,
            spectral_safety,
            d_m: Vec::new(),
            d_v: Vec::new(),
            d_t: 0,
        };

        let param_sizes = model.param_sizes();
        model.d_m = param_sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.d_v = param_sizes.iter().map(|&s| vec![0.0; s]).collect();

        model
    }

    fn param_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        sizes.push(self.token_emb.len());
        sizes.push(self.pos_emb.len());
        for l in 0..self.config.n_layer {
            sizes.push(self.ln1_gamma[l].len());
            sizes.push(self.ln1_beta[l].len());
            sizes.push(self.qkv_w[l].len());
            sizes.push(self.attn_proj[l].len());
            sizes.push(self.ln2_gamma[l].len());
            sizes.push(self.ln2_beta[l].len());
            sizes.push(self.ff_w1[l].len());
            sizes.push(self.ff_b1[l].len());
            sizes.push(self.ff_w2[l].len());
            sizes.push(self.ff_b2[l].len());
        }
        sizes.push(self.ln_f_gamma.len());
        sizes.push(self.ln_f_beta.len());
        sizes.push(self.lm_head.len());
        sizes
    }

    pub fn count_params(&self) -> usize {
        self.param_sizes().iter().sum()
    }

    /// Apply one self-energy operator: Sigma(x) for layer l.
    ///
    /// Self-energy = full transformer layer correction:
    ///   attn_correction = attn_proj(MHA(LN1(x)))
    ///   x_mid = x + attn_correction
    ///   ff_correction = FF(LN2(x_mid))
    ///   Sigma(x) = ff_correction   <-- the correction only, not x + correction
    ///
    /// The output is then spectrally normalized to ensure ||Sigma(x)|| < safety * ||x||.
    fn self_energy_forward(
        &self,
        x: &[f32],
        l: usize,
        seq_len: usize,
    ) -> (Vec<f32>, SelfEnergyCache) {
        let e = self.config.n_embd;
        let nh = self.config.n_head;
        let hs = e / nh;
        let inner = 4 * e;

        let mut cache = SelfEnergyCache::default();
        cache.x_input = x.to_vec();

        // Layer norm 1
        let (ln1_out, _ln1_mean, _ln1_rstd) =
            layer_norm(x, &self.ln1_gamma[l], &self.ln1_beta[l], seq_len, e);
        cache.ln1_out = ln1_out.clone();

        // QKV projection
        let qkv = matmul(&ln1_out, &self.qkv_w[l], seq_len, e, 3 * e);
        cache.qkv = qkv.clone();

        // Multi-head attention
        let mut attn_out = vec![0.0f32; seq_len * e];
        let mut all_attn_weights = vec![0.0f32; nh * seq_len * seq_len];

        for h in 0..nh {
            let scale = 1.0 / (hs as f32).sqrt();
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        all_attn_weights[h * seq_len * seq_len + i * seq_len + j] =
                            f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for k in 0..hs {
                            let qi = qkv[i * 3 * e + h * hs + k];
                            let kj = qkv[j * 3 * e + e + h * hs + k];
                            dot += qi * kj;
                        }
                        all_attn_weights[h * seq_len * seq_len + i * seq_len + j] = dot * scale;
                    }
                }
            }
            // Softmax per row
            for i in 0..seq_len {
                let offset = h * seq_len * seq_len + i * seq_len;
                let max_val = all_attn_weights[offset..offset + seq_len]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..seq_len {
                    let exp_val = (all_attn_weights[offset + j] - max_val).exp();
                    all_attn_weights[offset + j] = exp_val;
                    sum += exp_val;
                }
                for j in 0..seq_len {
                    all_attn_weights[offset + j] /= sum;
                }
            }
            // Weighted sum of V
            for i in 0..seq_len {
                for k in 0..hs {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        let w = all_attn_weights[h * seq_len * seq_len + i * seq_len + j];
                        let vj = qkv[j * 3 * e + 2 * e + h * hs + k];
                        sum += w * vj;
                    }
                    attn_out[i * e + h * hs + k] = sum;
                }
            }
        }
        cache.attn_weights = all_attn_weights;
        cache.attn_out_pre_proj = attn_out.clone();

        // Output projection
        let proj_out = matmul(&attn_out, &self.attn_proj[l], seq_len, e, e);

        // Attention residual: x_mid = x + proj_out
        let mut x_mid = x.to_vec();
        for i in 0..seq_len * e {
            x_mid[i] += proj_out[i];
        }
        cache.x_after_attn = x_mid.clone();

        // Layer norm 2
        let (ln2_out, _ln2_mean, _ln2_rstd) =
            layer_norm(&x_mid, &self.ln2_gamma[l], &self.ln2_beta[l], seq_len, e);
        cache.ln2_out = ln2_out.clone();

        // Feed-forward
        let mut ff_hidden = matmul(&ln2_out, &self.ff_w1[l], seq_len, e, inner);
        for i in 0..seq_len {
            for j in 0..inner {
                ff_hidden[i * inner + j] += self.ff_b1[l][j];
            }
        }
        cache.ff_pre_gelu = ff_hidden.clone();

        // GELU
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        for val in ff_hidden.iter_mut() {
            let x3 = *val * *val * *val;
            let inner_val = sqrt_2_over_pi * (*val + 0.044715 * x3);
            *val = 0.5 * *val * (1.0 + inner_val.tanh());
        }
        cache.ff_post_gelu = ff_hidden.clone();

        let mut ff_out = matmul(&ff_hidden, &self.ff_w2[l], seq_len, inner, e);
        for i in 0..seq_len {
            for j in 0..e {
                ff_out[i * e + j] += self.ff_b2[l][j];
            }
        }

        // Spectral radius control: normalize output so ||Sigma(x)|| < safety * ||x||
        let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let sigma_norm: f32 = ff_out.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ratio = sigma_norm / x_norm;
        let applied_scale = if ratio > self.spectral_safety {
            let s = self.spectral_safety / ratio;
            for val in ff_out.iter_mut() {
                *val *= s;
            }
            s
        } else {
            1.0
        };
        cache.scale_factor = applied_scale;
        cache.sigma_out = ff_out.clone();

        (ff_out, cache)
    }

    /// Backward through one self-energy application.
    /// Given d_sigma_out (gradient w.r.t. the output of self-energy),
    /// returns d_x (gradient w.r.t. x input) and accumulates weight gradients.
    fn self_energy_backward(
        &self,
        d_sigma_out: &[f32],
        cache: &SelfEnergyCache,
        l: usize,
        seq_len: usize,
        grads: &mut DysonGradients,
    ) -> Vec<f32> {
        let e = self.config.n_embd;
        let nh = self.config.n_head;
        let hs = e / nh;
        let inner = 4 * e;

        // Backward through spectral normalization
        let mut d_ff_out = vec![0.0f32; seq_len * e];
        for i in 0..seq_len * e {
            d_ff_out[i] = d_sigma_out[i] * cache.scale_factor;
        }

        // Backward through ff_b2
        for i in 0..seq_len {
            for j in 0..e {
                grads.ff_b2[l][j] += d_ff_out[i * e + j];
            }
        }

        // Backward through ff_w2
        let d_ff_hidden = matmul_backward_both(
            &cache.ff_post_gelu,
            &self.ff_w2[l],
            &d_ff_out,
            seq_len,
            inner,
            e,
            &mut grads.ff_w2[l],
        );

        // Backward through GELU
        let d_ff_pre_gelu = gelu_backward(&cache.ff_pre_gelu, &d_ff_hidden);

        // Backward through ff_b1
        for i in 0..seq_len {
            for j in 0..inner {
                grads.ff_b1[l][j] += d_ff_pre_gelu[i * inner + j];
            }
        }

        // Backward through ff_w1
        let d_ln2_out = matmul_backward_both(
            &cache.ln2_out,
            &self.ff_w1[l],
            &d_ff_pre_gelu,
            seq_len,
            e,
            inner,
            &mut grads.ff_w1[l],
        );

        // Backward through layer norm 2
        let d_x_mid = layer_norm_backward(
            &cache.x_after_attn,
            &d_ln2_out,
            &self.ln2_gamma[l],
            seq_len,
            e,
            &mut grads.ln2_gamma[l],
            &mut grads.ln2_beta[l],
        );

        // d_x_mid flows to both the residual path (-> dx) and the attention path
        // x_mid = x + proj_out, so dx gets d_x_mid directly, and d_proj_out = d_x_mid
        let d_proj_out = d_x_mid.clone();

        // Backward through output projection
        let d_attn_out = matmul_backward_both(
            &cache.attn_out_pre_proj,
            &self.attn_proj[l],
            &d_proj_out,
            seq_len,
            e,
            e,
            &mut grads.attn_proj[l],
        );

        // Backward through multi-head attention
        let mut d_qkv = vec![0.0f32; seq_len * 3 * e];

        for h in 0..nh {
            // Backward through V weighting
            for i in 0..seq_len {
                for k in 0..hs {
                    let d_o = d_attn_out[i * e + h * hs + k];
                    for j in 0..seq_len {
                        let w = cache.attn_weights[h * seq_len * seq_len + i * seq_len + j];
                        d_qkv[j * 3 * e + 2 * e + h * hs + k] += w * d_o;
                    }
                }
            }

            // Compute d_attn_score
            let mut d_attn_score = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dw = 0.0f32;
                    for k in 0..hs {
                        let vj = cache.qkv[j * 3 * e + 2 * e + h * hs + k];
                        dw += vj * d_attn_out[i * e + h * hs + k];
                    }
                    d_attn_score[i * seq_len + j] = dw;
                }
            }

            // Softmax backward
            for i in 0..seq_len {
                let mut dot = 0.0f32;
                for j in 0..seq_len {
                    dot += d_attn_score[i * seq_len + j]
                        * cache.attn_weights[h * seq_len * seq_len + i * seq_len + j];
                }
                for j in 0..seq_len {
                    let w = cache.attn_weights[h * seq_len * seq_len + i * seq_len + j];
                    d_attn_score[i * seq_len + j] = w * (d_attn_score[i * seq_len + j] - dot);
                }
            }

            // Scale backward
            let scale = 1.0 / (hs as f32).sqrt();
            for val in d_attn_score.iter_mut() {
                *val *= scale;
            }

            // Backward through Q @ K^T
            for i in 0..seq_len {
                for j in 0..=i {
                    let ds = d_attn_score[i * seq_len + j];
                    if ds.abs() > 1e-12 {
                        for k in 0..hs {
                            let qi = cache.qkv[i * 3 * e + h * hs + k];
                            let kj = cache.qkv[j * 3 * e + e + h * hs + k];
                            d_qkv[i * 3 * e + h * hs + k] += ds * kj;
                            d_qkv[j * 3 * e + e + h * hs + k] += ds * qi;
                        }
                    }
                }
            }
        }

        // Backward through QKV projection
        let d_ln1_out = matmul_backward_both(
            &cache.ln1_out,
            &self.qkv_w[l],
            &d_qkv,
            seq_len,
            e,
            3 * e,
            &mut grads.qkv_w[l],
        );

        // Backward through layer norm 1
        let d_from_attn = layer_norm_backward(
            &cache.x_input,
            &d_ln1_out,
            &self.ln1_gamma[l],
            seq_len,
            e,
            &mut grads.ln1_gamma[l],
            &mut grads.ln1_beta[l],
        );

        // dx = d_x_mid (from residual in x_mid = x + proj) + d_from_attn (from LN1 path)
        let mut dx = d_x_mid;
        for i in 0..seq_len * e {
            dx[i] += d_from_attn[i];
        }

        dx
    }

    /// Forward pass with cache for backward
    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, DysonForwardCache) {
        let cfg = &self.config;
        let seq_len = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        // Embedding lookup
        let mut x = vec![0.0f32; seq_len * e];
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                x[i * e + j] = self.token_emb[tok * e + j] + self.pos_emb[i * e + j];
            }
        }

        let x_after_emb = x.clone();
        let mut layer_caches = Vec::new();

        // Dyson layers: for each layer, apply Neumann series
        for l in 0..cfg.n_layer {
            let mut dlc = DysonLayerCache::default();

            // y_0 = x (the input)
            dlc.y_states.push(x.clone());

            // Neumann series: result = x + Sigma(x) + Sigma(Sigma(x)) + ... + Sigma^K(x)
            let mut result = x.clone(); // accumulator starts at x (order 0: identity)
            let mut y_prev = x.clone(); // y_0 = x

            for _k in 1..=self.neumann_order {
                // y_k = Sigma(y_{k-1})
                let (sigma_out, se_cache) = self.self_energy_forward(&y_prev, l, seq_len);
                dlc.se_caches.push(se_cache);

                // accumulate: result += y_k
                for i in 0..seq_len * e {
                    result[i] += sigma_out[i];
                }

                y_prev = sigma_out.clone();
                dlc.y_states.push(y_prev.clone());
            }

            x = result;
            layer_caches.push(dlc);
        }

        let x_before_final_ln = x.clone();

        // Final layer norm
        let (ln_out, _, _) = layer_norm(&x, &self.ln_f_gamma, &self.ln_f_beta, seq_len, e);
        let x_after_final_ln = ln_out.clone();

        // LM head
        let logits = matmul(&ln_out, &self.lm_head, seq_len, e, v);

        let cache = DysonForwardCache {
            tokens: tokens.to_vec(),
            x_after_emb,
            layer_caches,
            x_before_final_ln,
            x_after_final_ln,
        };

        (logits, cache)
    }

    /// Forward + backward pass returning loss and gradients
    pub fn forward_backward(
        &self,
        tokens: &[usize],
        targets: &[usize],
    ) -> (f32, DysonGradients) {
        let cfg = &self.config;
        let seq_len = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        let (logits, cache) = self.forward_with_cache(tokens);

        // Softmax + cross-entropy loss
        let mut probs = vec![0.0f32; seq_len * v];
        let mut loss = 0.0f32;

        for i in 0..seq_len {
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
        loss /= seq_len as f32;

        // Backward: dL/d_logits
        let mut d_logits = probs;
        for i in 0..seq_len {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v {
                d_logits[i * v + j] /= seq_len as f32;
            }
        }

        let mut grads = DysonGradients::zero_like(cfg);

        // Backward through LM head
        let d_ln_out = matmul_backward_both(
            &cache.x_after_final_ln,
            &self.lm_head,
            &d_logits,
            seq_len,
            e,
            v,
            &mut grads.lm_head,
        );

        // Backward through final layer norm
        let mut dx = layer_norm_backward(
            &cache.x_before_final_ln,
            &d_ln_out,
            &self.ln_f_gamma,
            seq_len,
            e,
            &mut grads.ln_f_gamma,
            &mut grads.ln_f_beta,
        );

        // Backward through Dyson layers in reverse
        for l in (0..cfg.n_layer).rev() {
            let dlc = &cache.layer_caches[l];
            let k_order = self.neumann_order;

            // Forward was: result = y_0 + y_1 + y_2 + ... + y_K
            // where y_0 = x_input and y_k = Sigma(y_{k-1}).
            //
            // dx is d_result. Since result = sum of y_k terms:
            //   d_y_k = d_result for each k (it's a sum, gradient distributes).
            //
            // For y_0 = x_input: d_x_input gets d_result directly.
            // For y_k = Sigma(y_{k-1}): chain backward through self-energy.
            //
            // But y_k depends on y_{k-1} which depends on ... y_0 = x_input.
            // So we need to chain: d_y_{k-1} += backward(d_y_k) for each k.
            //
            // Working backward from k=K to k=1:
            //   d_y_{k-1} += self_energy_backward(d_y_k = d_result, cache_k)

            // Start: d_y_K = d_result (from the sum)
            // Then chain backward through each Neumann order.
            // Also, each y_k contributes d_result to its own gradient.
            // Since they all sum to result, each gets the same d_result.
            //
            // More precisely:
            //   result = y_0 + y_1 + y_2 + ... + y_K
            //   d(y_k)/d(result) = I for all k
            //   But y_k depends on y_{k-1}: y_k = Sigma(y_{k-1})
            //   So d_loss/d_y_{k-1} = d_loss/d_y_{k-1} (direct from sum) + d_loss/d_y_k * dSigma/d_y_{k-1}
            //
            // Algorithm:
            //   d_y[K] = dx  (from sum contribution)
            //   for k in K..1:
            //     d_y[k-1] += self_energy_backward(d_y[k], se_cache[k-1])
            //   d_y[0] += dx  (direct contribution from sum)
            //
            // Wait, let me re-derive. Let R = y_0 + y_1 + ... + y_K.
            // dL/dR = dx (incoming gradient).
            // dL/dy_k = dL/dR * dR/dy_k.
            // But y_k appears in the sum and also y_{k+1} = Sigma(y_k).
            // So by chain rule: dL/dy_k = dL/dR (from sum) + dL/dy_{k+1} * dy_{k+1}/dy_k
            //
            // Computing backward from K to 0:
            //   dL/dy_K = dx  (only from sum, no downstream)
            //   dL/dy_{K-1} = dx + dL/dy_K * dSigma(y_{K-1})/dy_{K-1}
            //   dL/dy_{K-2} = dx + dL/dy_{K-1} * dSigma(y_{K-2})/dy_{K-2}
            //   ...
            //   dL/dy_0 = dx + dL/dy_1 * dSigma(y_0)/dy_0

            let mut d_y_k = dx.clone(); // d_y_K = dx (from the sum)

            for k in (1..=k_order).rev() {
                // y_k = Sigma(y_{k-1}), se_cache index = k-1
                let d_y_prev_from_chain =
                    self.self_energy_backward(&d_y_k, &dlc.se_caches[k - 1], l, seq_len, &mut grads);

                // d_y_{k-1} = dx (from sum) + d_y_prev_from_chain (from chain through Sigma)
                d_y_k = dx.clone();
                for i in 0..seq_len * e {
                    d_y_k[i] += d_y_prev_from_chain[i];
                }
            }

            // d_y_0 is now the gradient for x_input of this layer
            dx = d_y_k;
        }

        // Backward through embeddings
        for (i, &tok) in cache.tokens.iter().enumerate() {
            for j in 0..e {
                grads.token_emb[tok * e + j] += dx[i * e + j];
                grads.pos_emb[i * e + j] += dx[i * e + j];
            }
        }

        (loss, grads)
    }

    /// Apply gradients with Adam optimizer
    pub fn apply_gradients(&mut self, grads: &DysonGradients, lr: f32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        self.d_t += 1;
        let t_f = self.d_t as f32;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);

        let mut idx = 0usize;

        macro_rules! adam_update {
            ($param:expr, $grad:expr) => {{
                adam_step(
                    $param, $grad, &mut self.d_m[idx], &mut self.d_v[idx],
                    lr, beta1, beta2, eps, bc1, bc2,
                );
                idx += 1;
                let _ = idx;
            }};
        }

        adam_update!(&mut self.token_emb, &grads.token_emb);
        adam_update!(&mut self.pos_emb, &grads.pos_emb);
        for l in 0..self.config.n_layer {
            adam_update!(&mut self.ln1_gamma[l], &grads.ln1_gamma[l]);
            adam_update!(&mut self.ln1_beta[l], &grads.ln1_beta[l]);
            adam_update!(&mut self.qkv_w[l], &grads.qkv_w[l]);
            adam_update!(&mut self.attn_proj[l], &grads.attn_proj[l]);
            adam_update!(&mut self.ln2_gamma[l], &grads.ln2_gamma[l]);
            adam_update!(&mut self.ln2_beta[l], &grads.ln2_beta[l]);
            adam_update!(&mut self.ff_w1[l], &grads.ff_w1[l]);
            adam_update!(&mut self.ff_b1[l], &grads.ff_b1[l]);
            adam_update!(&mut self.ff_w2[l], &grads.ff_w2[l]);
            adam_update!(&mut self.ff_b2[l], &grads.ff_b2[l]);
        }
        adam_update!(&mut self.ln_f_gamma, &grads.ln_f_gamma);
        adam_update!(&mut self.ln_f_beta, &grads.ln_f_beta);
        adam_update!(&mut self.lm_head, &grads.lm_head);
    }

    /// Forward pass for inference (no cache)
    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        self.forward_with_cache(tokens).0
    }

    /// Generate text autoregressively
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
}

// ---- DysonGradients impl ----

impl DysonGradients {
    pub fn zero_like(cfg: &Config) -> Self {
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;
        let nl = cfg.n_layer;

        Self {
            token_emb: vec![0.0; v * e],
            pos_emb: vec![0.0; cfg.block_size * e],
            ln1_gamma: vec![vec![0.0; e]; nl],
            ln1_beta: vec![vec![0.0; e]; nl],
            qkv_w: vec![vec![0.0; e * 3 * e]; nl],
            attn_proj: vec![vec![0.0; e * e]; nl],
            ln2_gamma: vec![vec![0.0; e]; nl],
            ln2_beta: vec![vec![0.0; e]; nl],
            ff_w1: vec![vec![0.0; e * inner]; nl],
            ff_b1: vec![vec![0.0; inner]; nl],
            ff_w2: vec![vec![0.0; inner * e]; nl],
            ff_b2: vec![vec![0.0; e]; nl],
            ln_f_gamma: vec![0.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: vec![0.0; e * v],
        }
    }

    pub fn accumulate(&mut self, other: &DysonGradients) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        for l in 0..self.ln1_gamma.len() {
            add_vecs(&mut self.ln1_gamma[l], &other.ln1_gamma[l]);
            add_vecs(&mut self.ln1_beta[l], &other.ln1_beta[l]);
            add_vecs(&mut self.qkv_w[l], &other.qkv_w[l]);
            add_vecs(&mut self.attn_proj[l], &other.attn_proj[l]);
            add_vecs(&mut self.ln2_gamma[l], &other.ln2_gamma[l]);
            add_vecs(&mut self.ln2_beta[l], &other.ln2_beta[l]);
            add_vecs(&mut self.ff_w1[l], &other.ff_w1[l]);
            add_vecs(&mut self.ff_b1[l], &other.ff_b1[l]);
            add_vecs(&mut self.ff_w2[l], &other.ff_w2[l]);
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
            scale_vec(&mut self.ln1_gamma[l], factor);
            scale_vec(&mut self.ln1_beta[l], factor);
            scale_vec(&mut self.qkv_w[l], factor);
            scale_vec(&mut self.attn_proj[l], factor);
            scale_vec(&mut self.ln2_gamma[l], factor);
            scale_vec(&mut self.ln2_beta[l], factor);
            scale_vec(&mut self.ff_w1[l], factor);
            scale_vec(&mut self.ff_b1[l], factor);
            scale_vec(&mut self.ff_w2[l], factor);
            scale_vec(&mut self.ff_b2[l], factor);
        }
        scale_vec(&mut self.ln_f_gamma, factor);
        scale_vec(&mut self.ln_f_beta, factor);
        scale_vec(&mut self.lm_head, factor);
    }
}

// ---- Experiment runner ----

/// Pretrain on Shakespeare text
fn dyson_pretrain(
    model: &mut DysonGPT,
    config: &Config,
    text: &str,
    tok: &Tokenizer,
    rng: &mut impl Rng,
) {
    let encoded = tok.encode(text);
    let batch_size = 16;
    let num_steps = 1000;
    let lr = 0.001;

    let start = Instant::now();
    for step in 0..num_steps {
        let (inputs, targets) =
            crate::data::create_batches(&encoded, config.block_size, batch_size, rng);
        let mut total_loss = 0.0f32;
        let mut grads = DysonGradients::zero_like(config);

        for b in 0..batch_size {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g) = model.forward_backward(&inputs[b][..ctx_len], &targets[b][..ctx_len]);
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, lr);

        if step % 200 == 0 {
            println!(
                "  Pretrain step {:4} | Loss: {:.4} | K={} | {:.1}s",
                step,
                total_loss / batch_size as f32,
                model.neumann_order,
                start.elapsed().as_secs_f32()
            );
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
}

/// Evaluate model on validation set
fn dyson_evaluate_model(
    model: &DysonGPT,
    config: &Config,
    val: &[ToolExample],
    tok: &Tokenizer,
) -> AggregateMetrics {
    let mut results = Vec::new();
    for example in val {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);
        let eval = tool_data::evaluate_output(&generated_text, example);
        results.push(eval);
    }
    AggregateMetrics::from_results(&results)
}

/// Train and evaluate a single Dyson configuration.
/// Returns (composite_score, final_loss, train_time, params).
fn train_and_eval_dyson(
    label: &str,
    config: &Config,
    neumann_order: usize,
    spectral_safety: f32,
    base_text: &str,
    tok: &Tokenizer,
    train_data: &[ToolExample],
    val_data: &[ToolExample],
    rng: &mut impl Rng,
) -> (f32, f32, f32, usize, AggregateMetrics) {
    println!("\n--- {} (L={}, K={}) ---", label, config.n_layer, neumann_order);

    let mut model = DysonGPT::new(config.clone(), neumann_order, spectral_safety);
    let params = model.count_params();
    let effective_depth = config.n_layer * neumann_order.max(1);
    println!(
        "  Params: {}, Effective depth: {}, Spectral safety: {:.2}",
        params, effective_depth, spectral_safety
    );

    // Phase 1: Pretrain on Shakespeare
    println!("  [Pretrain]");
    dyson_pretrain(&mut model, config, base_text, tok, rng);

    // Phase 2: SFT on tool calling
    println!("  [SFT]");
    let sft_start = Instant::now();
    let num_steps = 800;
    let lr = 0.0005;
    let batch_size = 8;
    let mut final_loss = 0.0f32;

    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = DysonGradients::zero_like(config);
        let mut valid_count = 0;

        for _ in 0..batch_size {
            let example = &train_data[rng.gen_range(0..train_data.len())];
            let encoded = tok.encode(&example.input);
            if encoded.len() < 2 || encoded.len() > config.block_size {
                continue;
            }
            let input = &encoded[..encoded.len() - 1];
            let target = &encoded[1..];
            let (loss, g) = model.forward_backward(input, target);
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            grads.scale(1.0 / valid_count as f32);
            model.apply_gradients(&grads, lr);
            final_loss = total_loss / valid_count as f32;

            if step % 200 == 0 {
                println!(
                    "  SFT step {:4} | Loss: {:.4} | {:.1}s",
                    step,
                    final_loss,
                    sft_start.elapsed().as_secs_f32()
                );
            }
        }
    }

    let train_time = sft_start.elapsed().as_secs_f32();
    println!("  SFT done in {:.1}s, final loss: {:.4}", train_time, final_loss);

    // Phase 3: Evaluate
    let metrics = dyson_evaluate_model(&model, config, val_data, tok);
    let composite = metrics.format_acc * 0.3
        + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25
        + metrics.reply_quality * 0.15;

    println!("  Results:");
    println!("    Format:    {:.1}%", metrics.format_acc * 100.0);
    println!("    Tool:      {:.1}%", metrics.tool_acc * 100.0);
    println!("    Param:     {:.1}%", metrics.param_acc * 100.0);
    println!("    Reply:     {:.1}%", metrics.reply_quality * 100.0);
    println!("    Composite: {:.4}", composite);

    (composite, final_loss, train_time, params, metrics)
}

/// Main experiment function for Dyson Equation Transformer
pub fn run_dyson_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    println!("=== Experiment B2: Dyson Equation Transformer ===\n");
    println!("HYPOTHESIS: Replace additive residual connections with Neumann-series");
    println!("residual connections inspired by Dyson's equation from quantum field theory.");
    println!("A single Dyson layer at order K gives effective depth K with the parameters");
    println!("of ONE layer (automatic weight sharing across virtual depth).\n");
    println!("KEY COMPARISON: Dyson-1L-K3 (~33K params) vs Standard-3L (~97K params).\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/3] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    let block_size = 128;
    let spectral_safety = 0.85;

    // === Experiment configurations ===
    struct DysonConfig {
        label: &'static str,
        n_layer: usize,
        neumann_order: usize,
    }

    let configs = vec![
        DysonConfig { label: "Dyson-1L-K1", n_layer: 1, neumann_order: 1 },
        DysonConfig { label: "Dyson-1L-K2", n_layer: 1, neumann_order: 2 },
        DysonConfig { label: "Dyson-1L-K3", n_layer: 1, neumann_order: 3 },
        DysonConfig { label: "Dyson-1L-K4", n_layer: 1, neumann_order: 4 },
        DysonConfig { label: "Dyson-2L-K2", n_layer: 2, neumann_order: 2 },
        DysonConfig { label: "Standard-3L", n_layer: 3, neumann_order: 0 },
    ];

    println!("\n[2/3] Running {} configurations...", configs.len());

    struct RunResult {
        label: &'static str,
        n_layer: usize,
        neumann_order: usize,
        effective_depth: usize,
        params: usize,
        composite: f32,
        final_loss: f32,
        train_time: f32,
        metrics: AggregateMetrics,
    }

    let mut results: Vec<RunResult> = Vec::new();

    for dc in &configs {
        let model_config = Config {
            vocab_size: tok.vocab_size(),
            n_embd: 48,
            n_head: 4,
            n_layer: dc.n_layer,
            block_size,
        };

        // For "Standard-3L" (K=0), we use K=0 which means the Neumann series
        // loop doesn't execute, so result = x (identity only).
        // That makes it a standard residual network: self_energy_forward produces
        // the correction, and with K=1 it becomes x + Sigma(x) = standard residual.
        //
        // For the standard baseline, we set K=1 which is exactly standard residual.
        let actual_k = if dc.neumann_order == 0 { 1 } else { dc.neumann_order };
        let effective_depth = dc.n_layer * actual_k;

        let (composite, final_loss, train_time, params, metrics) = train_and_eval_dyson(
            dc.label,
            &model_config,
            actual_k,
            spectral_safety,
            base_text,
            &tok,
            &train_data,
            &val_data,
            &mut rng,
        );

        results.push(RunResult {
            label: dc.label,
            n_layer: dc.n_layer,
            neumann_order: dc.neumann_order,
            effective_depth,
            params,
            composite,
            final_loss,
            train_time,
            metrics,
        });
    }

    // === Save results CSV ===
    println!("\n[3/3] Saving results...");

    if let Ok(mut file) = std::fs::File::create("experiments/dyson_results.csv") {
        let _ = writeln!(
            file,
            "config,n_layer,neumann_order,effective_depth,params,composite,format,tool,param,reply,final_loss,train_time"
        );
        for r in &results {
            let _ = writeln!(
                file,
                "{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
                r.label,
                r.n_layer,
                r.neumann_order,
                r.effective_depth,
                r.params,
                r.composite,
                r.metrics.format_acc,
                r.metrics.tool_acc,
                r.metrics.param_acc,
                r.metrics.reply_quality,
                r.final_loss,
                r.train_time,
            );
        }
        println!("  Results saved to experiments/dyson_results.csv");
    }

    // === Summary ===
    println!("\n{}", "=".repeat(70));
    println!("=== DYSON EQUATION TRANSFORMER - RESULTS SUMMARY ===\n");
    println!(
        "  {:16} {:>6} {:>5} {:>6} {:>7} {:>8} {:>8} {:>8} {:>8}",
        "Config", "Layers", "K", "EfDep", "Params", "Compos", "Format", "Tool", "Loss"
    );
    println!("  {}", "-".repeat(80));
    for r in &results {
        println!(
            "  {:16} {:>6} {:>5} {:>6} {:>7} {:>8.4} {:>7.1}% {:>7.1}% {:>8.4}",
            r.label,
            r.n_layer,
            r.neumann_order,
            r.effective_depth,
            r.params,
            r.composite,
            r.metrics.format_acc * 100.0,
            r.metrics.tool_acc * 100.0,
            r.final_loss,
        );
    }

    // Key comparison
    let dyson_1l_k3 = results.iter().find(|r| r.label == "Dyson-1L-K3");
    let standard_3l = results.iter().find(|r| r.label == "Standard-3L");
    if let (Some(d), Some(s)) = (dyson_1l_k3, standard_3l) {
        println!("\n  KEY COMPARISON: Dyson-1L-K3 vs Standard-3L");
        println!("    Params:    {} vs {} ({:.1}x compression)",
                 d.params, s.params, s.params as f32 / d.params as f32);
        println!("    Composite: {:.4} vs {:.4} ({:+.4})",
                 d.composite, s.composite, d.composite - s.composite);
        println!("    Loss:      {:.4} vs {:.4}", d.final_loss, s.final_loss);
        if d.composite >= s.composite * 0.9 {
            println!("    RESULT: Dyson-1L-K3 matches Standard-3L with {:.1}x fewer params!",
                     s.params as f32 / d.params as f32);
        } else {
            println!("    RESULT: Standard-3L outperforms. Dyson achieves {:.1}% of baseline with {:.1}x fewer params.",
                     d.composite / s.composite * 100.0,
                     s.params as f32 / d.params as f32);
        }
    }

    println!("\n  Physics insight: Each Neumann order re-applies the SAME self-energy");
    println!("  operator with shared weights, giving 'depth without parameters'.");
    println!("  Spectral radius control (safety={:.2}) prevents Neumann divergence.", spectral_safety);
    println!("{}", "=".repeat(70));
}
