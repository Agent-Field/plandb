use rand::Rng;
use std::io::Write;

use crate::model::Config;

/// Hamiltonian GPT: Symplectic residual connections via leapfrog integration.
///
/// Maps transformer concepts to Hamiltonian mechanics:
/// - Position q = token representations (main hidden state)
/// - Momentum p = auxiliary state vector (same dim as q)
/// - V(q) = attention layer ("potential energy" from token interactions)
/// - T(p) = feed-forward layer ("kinetic energy" from momentum processing)
///
/// Each layer applies a leapfrog (Stormer-Verlet) step:
///   p_half = p - (h/2) * V(q)
///   q_new  = q + h * T(p_half)
///   p_new  = p_half - (h/2) * V(q_new)
///
/// Key property: symplectic integration conserves energy, preventing
/// gradient explosion/vanishing.
pub struct HamiltonianGPT {
    // Embeddings
    pub token_emb: Vec<f32>,  // (vocab_size, n_embd)
    pub pos_emb: Vec<f32>,    // (block_size, n_embd)

    // Momentum initialization projection: q -> p
    pub w_p_init: Vec<f32>,   // (n_embd, n_embd)

    // Per-layer parameters for V(q) = attention
    pub ln_v_gamma: Vec<Vec<f32>>,  // [n_layer][n_embd]
    pub ln_v_beta: Vec<Vec<f32>>,   // [n_layer][n_embd]
    pub qkv_w: Vec<Vec<f32>>,      // [n_layer][n_embd * 3 * n_embd]
    pub attn_proj: Vec<Vec<f32>>,   // [n_layer][n_embd * n_embd]

    // Per-layer parameters for T(p) = feed-forward
    pub ln_t_gamma: Vec<Vec<f32>>,  // [n_layer][n_embd]
    pub ln_t_beta: Vec<Vec<f32>>,   // [n_layer][n_embd]
    pub ff_w1: Vec<Vec<f32>>,       // [n_layer][n_embd * 4*n_embd]
    pub ff_b1: Vec<Vec<f32>>,       // [n_layer][4*n_embd]
    pub ff_w2: Vec<Vec<f32>>,       // [n_layer][4*n_embd * n_embd]
    pub ff_b2: Vec<Vec<f32>>,       // [n_layer][n_embd]

    // Per-layer step size h (learned)
    pub step_sizes: Vec<f32>,  // [n_layer]

    // Final layer norm + head
    pub ln_f_gamma: Vec<f32>,  // (n_embd,)
    pub ln_f_beta: Vec<f32>,   // (n_embd,)
    pub lm_head: Vec<f32>,     // (n_embd, vocab_size)

    pub config: Config,

    // Adam optimizer state
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub t: usize,
}

/// Gradient storage for HamiltonianGPT
pub struct HamiltonianGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub w_p_init: Vec<f32>,
    pub ln_v_gamma: Vec<Vec<f32>>,
    pub ln_v_beta: Vec<Vec<f32>>,
    pub qkv_w: Vec<Vec<f32>>,
    pub attn_proj: Vec<Vec<f32>>,
    pub ln_t_gamma: Vec<Vec<f32>>,
    pub ln_t_beta: Vec<Vec<f32>>,
    pub ff_w1: Vec<Vec<f32>>,
    pub ff_b1: Vec<Vec<f32>>,
    pub ff_w2: Vec<Vec<f32>>,
    pub ff_b2: Vec<Vec<f32>>,
    pub step_sizes: Vec<f32>,
    pub ln_f_gamma: Vec<f32>,
    pub ln_f_beta: Vec<f32>,
    pub lm_head: Vec<f32>,
}

/// Per-layer energy diagnostics from forward pass
#[derive(Clone, Default)]
pub struct EnergyDiagnostics {
    pub per_layer_q_norm: Vec<f32>,
    pub per_layer_p_norm: Vec<f32>,
    pub per_layer_total_energy: Vec<f32>,
}

// Forward cache structures
#[derive(Default)]
struct HLayerCache {
    // Inputs to the layer
    q_input: Vec<f32>,
    p_input: Vec<f32>,

    // First half-step: V(q) for p update
    ln_v1_out: Vec<f32>,
    ln_v1_mean: Vec<f32>,
    ln_v1_rstd: Vec<f32>,
    qkv1: Vec<f32>,
    attn_weights1: Vec<f32>,
    attn_out1: Vec<f32>,
    v1_out: Vec<f32>,  // after projection

    // Full-step: T(p_half) for q update
    p_half: Vec<f32>,
    ln_t_out: Vec<f32>,
    ln_t_mean: Vec<f32>,
    ln_t_rstd: Vec<f32>,
    ff_pre_gelu: Vec<f32>,
    ff_post_gelu: Vec<f32>,
    t_out: Vec<f32>,

    // Second half-step: V(q_new) for p update
    q_new: Vec<f32>,
    ln_v2_out: Vec<f32>,
    ln_v2_mean: Vec<f32>,
    ln_v2_rstd: Vec<f32>,
    qkv2: Vec<f32>,
    attn_weights2: Vec<f32>,
    attn_out2: Vec<f32>,
    v2_out: Vec<f32>,
}

#[allow(dead_code)]
struct HForwardCache {
    tokens: Vec<usize>,
    x_after_emb: Vec<f32>,
    p_after_init: Vec<f32>,
    tanh_input: Vec<f32>, // pre-tanh values for backward
    layer_caches: Vec<HLayerCache>,
    q_before_final_ln: Vec<f32>,
    q_after_final_ln: Vec<f32>,
}

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

impl HamiltonianGPT {
    pub fn new(config: Config) -> Self {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let bs = config.block_size;
        let inner = 4 * e;

        let emb_scale = 0.02;
        let layer_scale = (0.02 / (nl as f32).sqrt()).max(0.001);

        let mut ln_v_gamma = Vec::new();
        let mut ln_v_beta = Vec::new();
        let mut qkv_w = Vec::new();
        let mut attn_proj = Vec::new();
        let mut ln_t_gamma = Vec::new();
        let mut ln_t_beta = Vec::new();
        let mut ff_w1 = Vec::new();
        let mut ff_b1 = Vec::new();
        let mut ff_w2 = Vec::new();
        let mut ff_b2 = Vec::new();

        for _ in 0..nl {
            ln_v_gamma.push(vec![1.0; e]);
            ln_v_beta.push(vec![0.0; e]);
            qkv_w.push(randn_vec(e * 3 * e, layer_scale));
            attn_proj.push(randn_vec(e * e, layer_scale));
            ln_t_gamma.push(vec![1.0; e]);
            ln_t_beta.push(vec![0.0; e]);
            ff_w1.push(randn_vec(e * inner, layer_scale));
            ff_b1.push(vec![0.0; inner]);
            ff_w2.push(randn_vec(inner * e, layer_scale * 0.5));
            ff_b2.push(vec![0.0; e]);
        }

        let step_sizes = vec![1.0f32; nl];

        let mut model = HamiltonianGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            w_p_init: randn_vec(e * e, layer_scale),
            ln_v_gamma,
            ln_v_beta,
            qkv_w,
            attn_proj,
            ln_t_gamma,
            ln_t_beta,
            ff_w1,
            ff_b1,
            ff_w2,
            ff_b2,
            step_sizes,
            ln_f_gamma: vec![1.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: randn_vec(e * v, emb_scale),
            config,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        };

        let param_sizes = model.param_sizes();
        model.m = param_sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.v = param_sizes.iter().map(|&s| vec![0.0; s]).collect();

        model
    }

    fn param_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        sizes.push(self.token_emb.len());
        sizes.push(self.pos_emb.len());
        sizes.push(self.w_p_init.len());
        for l in 0..self.config.n_layer {
            sizes.push(self.ln_v_gamma[l].len());
            sizes.push(self.ln_v_beta[l].len());
            sizes.push(self.qkv_w[l].len());
            sizes.push(self.attn_proj[l].len());
            sizes.push(self.ln_t_gamma[l].len());
            sizes.push(self.ln_t_beta[l].len());
            sizes.push(self.ff_w1[l].len());
            sizes.push(self.ff_b1[l].len());
            sizes.push(self.ff_w2[l].len());
            sizes.push(self.ff_b2[l].len());
            sizes.push(1); // step_size
        }
        sizes.push(self.ln_f_gamma.len());
        sizes.push(self.ln_f_beta.len());
        sizes.push(self.lm_head.len());
        sizes
    }

    pub fn count_params(&self) -> usize {
        self.param_sizes().iter().sum()
    }

    /// Compute V(q) = multi-head attention on q (potential energy gradient)
    fn compute_v(
        &self,
        q: &[f32],
        l: usize,
        seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let e = self.config.n_embd;
        let nh = self.config.n_head;
        let hs = e / nh;

        let (ln_out, ln_mean, ln_rstd) =
            layer_norm(q, &self.ln_v_gamma[l], &self.ln_v_beta[l], seq_len, e);

        let qkv = matmul(&ln_out, &self.qkv_w[l], seq_len, e, 3 * e);

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

        // Output projection
        let proj_out = matmul(&attn_out, &self.attn_proj[l], seq_len, e, e);

        (proj_out, ln_out, ln_mean, ln_rstd, qkv, all_attn_weights, attn_out)
    }

    /// Compute T(p) = feed-forward on p (kinetic energy gradient)
    fn compute_t(
        &self,
        p: &[f32],
        l: usize,
        seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let e = self.config.n_embd;
        let inner = 4 * e;

        let (ln_out, ln_mean, ln_rstd) =
            layer_norm(p, &self.ln_t_gamma[l], &self.ln_t_beta[l], seq_len, e);

        let mut ff_hidden = matmul(&ln_out, &self.ff_w1[l], seq_len, e, inner);
        for i in 0..seq_len {
            for j in 0..inner {
                ff_hidden[i * inner + j] += self.ff_b1[l][j];
            }
        }
        let ff_pre_gelu = ff_hidden.clone();

        // GELU
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        for val in ff_hidden.iter_mut() {
            let x3 = *val * *val * *val;
            let inner_val = sqrt_2_over_pi * (*val + 0.044715 * x3);
            *val = 0.5 * *val * (1.0 + inner_val.tanh());
        }
        let ff_post_gelu = ff_hidden.clone();

        let mut ff_out = matmul(&ff_hidden, &self.ff_w2[l], seq_len, inner, e);
        for i in 0..seq_len {
            for j in 0..e {
                ff_out[i * e + j] += self.ff_b2[l][j];
            }
        }

        (ff_out, ln_out, ln_mean, ln_rstd, ff_pre_gelu, ff_post_gelu)
    }

    /// Forward pass with full cache for backward
    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, HForwardCache, EnergyDiagnostics) {
        let cfg = &self.config;
        let seq_len = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        // Embedding lookup: q = token_emb + pos_emb
        let mut q = vec![0.0f32; seq_len * e];
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                q[i * e + j] = self.token_emb[tok * e + j] + self.pos_emb[i * e + j];
            }
        }

        let x_after_emb = q.clone();

        // Initialize momentum: p = tanh(q @ W_p_init)
        let tanh_input = matmul(&q, &self.w_p_init, seq_len, e, e);
        let mut p = tanh_input.clone();
        for val in p.iter_mut() {
            *val = val.tanh();
        }

        let p_after_init = p.clone();

        let mut diagnostics = EnergyDiagnostics::default();
        let mut layer_caches = Vec::new();

        // Leapfrog integration through layers
        for l in 0..cfg.n_layer {
            let mut lc = HLayerCache::default();
            lc.q_input = q.clone();
            lc.p_input = p.clone();

            let h = self.step_sizes[l];
            let h_half = h * 0.5;

            // Step 1: p_half = p - (h/2) * V(q)
            let (v1_out, ln_v1_out, ln_v1_mean, ln_v1_rstd, qkv1, aw1, ao1) =
                self.compute_v(&q, l, seq_len);
            lc.ln_v1_out = ln_v1_out;
            lc.ln_v1_mean = ln_v1_mean;
            lc.ln_v1_rstd = ln_v1_rstd;
            lc.qkv1 = qkv1;
            lc.attn_weights1 = aw1;
            lc.attn_out1 = ao1;
            lc.v1_out = v1_out.clone();

            let mut p_half = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                p_half[i] = p[i] - h_half * v1_out[i];
            }
            lc.p_half = p_half.clone();

            // Step 2: q_new = q + h * T(p_half)
            let (t_out, ln_t_out, ln_t_mean, ln_t_rstd, ff_pre_gelu, ff_post_gelu) =
                self.compute_t(&p_half, l, seq_len);
            lc.ln_t_out = ln_t_out;
            lc.ln_t_mean = ln_t_mean;
            lc.ln_t_rstd = ln_t_rstd;
            lc.ff_pre_gelu = ff_pre_gelu;
            lc.ff_post_gelu = ff_post_gelu;
            lc.t_out = t_out.clone();

            let mut q_new = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                q_new[i] = q[i] + h * t_out[i];
            }
            lc.q_new = q_new.clone();

            // Step 3: p_new = p_half - (h/2) * V(q_new)
            let (v2_out, ln_v2_out, ln_v2_mean, ln_v2_rstd, qkv2, aw2, ao2) =
                self.compute_v(&q_new, l, seq_len);
            lc.ln_v2_out = ln_v2_out;
            lc.ln_v2_mean = ln_v2_mean;
            lc.ln_v2_rstd = ln_v2_rstd;
            lc.qkv2 = qkv2;
            lc.attn_weights2 = aw2;
            lc.attn_out2 = ao2;
            lc.v2_out = v2_out.clone();

            let mut p_new = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                p_new[i] = p_half[i] - h_half * v2_out[i];
            }

            // Track energy: ||q||^2 + ||p||^2
            let q_norm: f32 = q_new.iter().map(|x| x * x).sum::<f32>() / (seq_len * e) as f32;
            let p_norm: f32 = p_new.iter().map(|x| x * x).sum::<f32>() / (seq_len * e) as f32;
            diagnostics.per_layer_q_norm.push(q_norm.sqrt());
            diagnostics.per_layer_p_norm.push(p_norm.sqrt());
            diagnostics.per_layer_total_energy.push(q_norm + p_norm);

            q = q_new;
            p = p_new;

            layer_caches.push(lc);
        }

        let q_before_final_ln = q.clone();

        // Final layer norm on q (discard p)
        let (ln_out, _, _) = layer_norm(&q, &self.ln_f_gamma, &self.ln_f_beta, seq_len, e);
        let q_after_final_ln = ln_out.clone();

        // LM head
        let logits = matmul(&ln_out, &self.lm_head, seq_len, e, v);

        let cache = HForwardCache {
            tokens: tokens.to_vec(),
            x_after_emb,
            p_after_init,
            tanh_input,
            layer_caches,
            q_before_final_ln,
            q_after_final_ln,
        };

        (logits, cache, diagnostics)
    }

    /// Forward + backward pass returning loss, gradients, and energy diagnostics
    pub fn forward_backward(
        &self,
        tokens: &[usize],
        targets: &[usize],
    ) -> (f32, HamiltonianGradients, EnergyDiagnostics) {
        let cfg = &self.config;
        let seq_len = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        let (logits, cache, diagnostics) = self.forward_with_cache(tokens);

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

        let mut grads = HamiltonianGradients::zero_like(cfg);

        // Backward through LM head
        let d_ln_out = matmul_backward_both(
            &cache.q_after_final_ln,
            &self.lm_head,
            &d_logits,
            seq_len,
            e,
            v,
            &mut grads.lm_head,
        );

        // Backward through final layer norm
        let mut dq = layer_norm_backward(
            &cache.q_before_final_ln,
            &d_ln_out,
            &self.ln_f_gamma,
            seq_len,
            e,
            &mut grads.ln_f_gamma,
            &mut grads.ln_f_beta,
        );

        // dp starts at zero (momentum is discarded in output)
        let mut dp = vec![0.0f32; seq_len * e];

        // Backward through layers in reverse
        for l in (0..cfg.n_layer).rev() {
            let lc = &cache.layer_caches[l];
            let _inner = 4 * e;
            let h = self.step_sizes[l];
            let h_half = h * 0.5;

            // === Step 3 backward: p_new = p_half - (h/2) * V(q_new) ===
            // dp -> dp_half, d_v2_out
            // dp_half += dp
            let mut dp_half = dp.clone();
            // d_v2_out = -h/2 * dp
            let mut d_v2_out = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                d_v2_out[i] = -h_half * dp[i];
            }

            // d_h from step 3: sum(-0.5 * v2_out[i] * dp[i])
            let mut d_h: f32 = 0.0;
            for i in 0..seq_len * e {
                d_h += -0.5 * lc.v2_out[i] * dp[i];
            }

            // Backward through V(q_new) for d_v2_out -> dq_new_from_v2
            let dq_from_v2 = self.backward_v(
                &lc.q_new, &d_v2_out, l, seq_len,
                &lc.ln_v2_out, &lc.qkv2, &lc.attn_weights2, &lc.attn_out2,
                &mut grads,
            );

            // dq += dq_from_v2 (dq currently has gradient for q_new)
            for i in 0..seq_len * e {
                dq[i] += dq_from_v2[i];
            }

            // === Step 2 backward: q_new = q + h * T(p_half) ===
            // dq -> dq_input (direct), d_t_out = h * dq
            let mut d_t_out = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                d_t_out[i] = h * dq[i];
            }

            // d_h from step 2: sum(t_out[i] * dq[i])
            for i in 0..seq_len * e {
                d_h += lc.t_out[i] * dq[i];
            }

            // Backward through T(p_half) for d_t_out -> dp_half_from_t
            let dp_from_t = self.backward_t(
                &lc.p_half, &d_t_out, l, seq_len,
                &lc.ln_t_out, &lc.ff_pre_gelu, &lc.ff_post_gelu,
                &mut grads,
            );

            for i in 0..seq_len * e {
                dp_half[i] += dp_from_t[i];
            }

            // dq_input from step 2 = dq (direct path, q_new = q + h*T)
            let dq_input = dq;

            // === Step 1 backward: p_half = p - (h/2) * V(q) ===
            // dp_half -> dp_input, d_v1_out
            let dp_input = dp_half.clone();
            let mut d_v1_out = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                d_v1_out[i] = -h_half * dp_half[i];
            }

            // d_h from step 1: sum(-0.5 * v1_out[i] * dp_half[i])
            for i in 0..seq_len * e {
                d_h += -0.5 * lc.v1_out[i] * dp_half[i];
            }

            // Backward through V(q_input) for d_v1_out -> dq_from_v1
            let dq_from_v1 = self.backward_v(
                &lc.q_input, &d_v1_out, l, seq_len,
                &lc.ln_v1_out, &lc.qkv1, &lc.attn_weights1, &lc.attn_out1,
                &mut grads,
            );

            // Combine: dq for previous layer = dq_input + dq_from_v1
            dq = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                dq[i] = dq_input[i] + dq_from_v1[i];
            }

            // dp for previous layer = dp_input
            dp = dp_input;

            // Gradient for step_size h
            grads.step_sizes[l] += d_h;
        }

        // Backward through momentum initialization: p = tanh(q @ W_p_init)
        // dp -> d_tanh_input -> dq_from_p_init + d_w_p_init
        let mut d_tanh_input = vec![0.0f32; seq_len * e];
        for i in 0..seq_len * e {
            let t_val = cache.tanh_input[i].tanh();
            d_tanh_input[i] = dp[i] * (1.0 - t_val * t_val);
        }

        // d_tanh_input = dq_emb @ W_p_init, so backward through matmul
        let dq_from_init = matmul_backward_both(
            &cache.x_after_emb,
            &self.w_p_init,
            &d_tanh_input,
            seq_len,
            e,
            e,
            &mut grads.w_p_init,
        );

        // Combine dq from layers + dq from momentum init
        for i in 0..seq_len * e {
            dq[i] += dq_from_init[i];
        }

        // Backward through embeddings
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                grads.token_emb[tok * e + j] += dq[i * e + j];
                grads.pos_emb[i * e + j] += dq[i * e + j];
            }
        }

        (loss, grads, diagnostics)
    }

    /// Backward through V(q) = attention, given d_output, returns dq
    fn backward_v(
        &self,
        q_in: &[f32],
        d_out: &[f32],
        l: usize,
        seq_len: usize,
        ln_out: &[f32],
        qkv: &[f32],
        attn_weights: &[f32],
        attn_out_pre_proj: &[f32],
        grads: &mut HamiltonianGradients,
    ) -> Vec<f32> {
        let e = self.config.n_embd;
        let nh = self.config.n_head;
        let hs = e / nh;

        // Backward through output projection
        let d_attn_out = matmul_backward_both(
            attn_out_pre_proj,
            &self.attn_proj[l],
            d_out,
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
                        let w = attn_weights[h * seq_len * seq_len + i * seq_len + j];
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
                        let vj = qkv[j * 3 * e + 2 * e + h * hs + k];
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
                        * attn_weights[h * seq_len * seq_len + i * seq_len + j];
                }
                for j in 0..seq_len {
                    let w = attn_weights[h * seq_len * seq_len + i * seq_len + j];
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
                            let qi = qkv[i * 3 * e + h * hs + k];
                            let kj = qkv[j * 3 * e + e + h * hs + k];
                            d_qkv[i * 3 * e + h * hs + k] += ds * kj;
                            d_qkv[j * 3 * e + e + h * hs + k] += ds * qi;
                        }
                    }
                }
            }
        }

        // Backward through QKV projection
        let d_ln_out = matmul_backward_both(
            ln_out,
            &self.qkv_w[l],
            &d_qkv,
            seq_len,
            e,
            3 * e,
            &mut grads.qkv_w[l],
        );

        // Backward through layer norm
        layer_norm_backward(
            q_in,
            &d_ln_out,
            &self.ln_v_gamma[l],
            seq_len,
            e,
            &mut grads.ln_v_gamma[l],
            &mut grads.ln_v_beta[l],
        )
    }

    /// Backward through T(p) = feed-forward, given d_output, returns dp
    fn backward_t(
        &self,
        p_in: &[f32],
        d_out: &[f32],
        l: usize,
        seq_len: usize,
        ln_out: &[f32],
        ff_pre_gelu: &[f32],
        ff_post_gelu: &[f32],
        grads: &mut HamiltonianGradients,
    ) -> Vec<f32> {
        let e = self.config.n_embd;
        let inner = 4 * e;

        // Backward through ff_b2
        for i in 0..seq_len {
            for j in 0..e {
                grads.ff_b2[l][j] += d_out[i * e + j];
            }
        }

        // Backward through ff_w2
        let d_ff_hidden = matmul_backward_both(
            ff_post_gelu,
            &self.ff_w2[l],
            d_out,
            seq_len,
            inner,
            e,
            &mut grads.ff_w2[l],
        );

        // Backward through GELU
        let d_ff_pre_gelu = gelu_backward(ff_pre_gelu, &d_ff_hidden);

        // Backward through ff_b1
        for i in 0..seq_len {
            for j in 0..inner {
                grads.ff_b1[l][j] += d_ff_pre_gelu[i * inner + j];
            }
        }

        // Backward through ff_w1
        let d_ln_out = matmul_backward_both(
            ln_out,
            &self.ff_w1[l],
            &d_ff_pre_gelu,
            seq_len,
            e,
            inner,
            &mut grads.ff_w1[l],
        );

        // Backward through layer norm
        layer_norm_backward(
            p_in,
            &d_ln_out,
            &self.ln_t_gamma[l],
            seq_len,
            e,
            &mut grads.ln_t_gamma[l],
            &mut grads.ln_t_beta[l],
        )
    }

    /// Apply gradients with Adam optimizer
    pub fn apply_gradients(&mut self, grads: &HamiltonianGradients, lr: f32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        self.t += 1;
        let t_f = self.t as f32;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);

        let mut idx = 0usize;

        macro_rules! adam_update {
            ($param:expr, $grad:expr) => {{
                adam_step(
                    $param, $grad, &mut self.m[idx], &mut self.v[idx],
                    lr, beta1, beta2, eps, bc1, bc2,
                );
                idx += 1;
                let _ = idx;
            }};
        }

        adam_update!(&mut self.token_emb, &grads.token_emb);
        adam_update!(&mut self.pos_emb, &grads.pos_emb);
        adam_update!(&mut self.w_p_init, &grads.w_p_init);
        for l in 0..self.config.n_layer {
            adam_update!(&mut self.ln_v_gamma[l], &grads.ln_v_gamma[l]);
            adam_update!(&mut self.ln_v_beta[l], &grads.ln_v_beta[l]);
            adam_update!(&mut self.qkv_w[l], &grads.qkv_w[l]);
            adam_update!(&mut self.attn_proj[l], &grads.attn_proj[l]);
            adam_update!(&mut self.ln_t_gamma[l], &grads.ln_t_gamma[l]);
            adam_update!(&mut self.ln_t_beta[l], &grads.ln_t_beta[l]);
            adam_update!(&mut self.ff_w1[l], &grads.ff_w1[l]);
            adam_update!(&mut self.ff_b1[l], &grads.ff_b1[l]);
            adam_update!(&mut self.ff_w2[l], &grads.ff_w2[l]);
            adam_update!(&mut self.ff_b2[l], &grads.ff_b2[l]);
            // Step size: single scalar, use Adam directly
            {
                let g = grads.step_sizes[l].max(-1.0).min(1.0);
                self.m[idx][0] = beta1 * self.m[idx][0] + (1.0 - beta1) * g;
                self.v[idx][0] = beta2 * self.v[idx][0] + (1.0 - beta2) * g * g;
                let m_hat = self.m[idx][0] / bc1;
                let v_hat = self.v[idx][0] / bc2;
                self.step_sizes[l] -= lr * m_hat / (v_hat.sqrt() + eps);
                idx += 1;
            }
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

    /// Save weights to binary file
    pub fn save_weights(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        // Use "HGPT" magic for Hamiltonian GPT
        f.write_all(b"HGPT")?;
        f.write_all(&(self.config.vocab_size as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_embd as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_head as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_layer as u32).to_le_bytes())?;
        f.write_all(&(self.config.block_size as u32).to_le_bytes())?;

        let mut all: Vec<f32> = Vec::new();
        all.extend_from_slice(&self.token_emb);
        all.extend_from_slice(&self.pos_emb);
        all.extend_from_slice(&self.w_p_init);
        for l in 0..self.config.n_layer {
            all.extend_from_slice(&self.ln_v_gamma[l]);
            all.extend_from_slice(&self.ln_v_beta[l]);
            all.extend_from_slice(&self.qkv_w[l]);
            all.extend_from_slice(&self.attn_proj[l]);
            all.extend_from_slice(&self.ln_t_gamma[l]);
            all.extend_from_slice(&self.ln_t_beta[l]);
            all.extend_from_slice(&self.ff_w1[l]);
            all.extend_from_slice(&self.ff_b1[l]);
            all.extend_from_slice(&self.ff_w2[l]);
            all.extend_from_slice(&self.ff_b2[l]);
            all.push(self.step_sizes[l]);
        }
        all.extend_from_slice(&self.ln_f_gamma);
        all.extend_from_slice(&self.ln_f_beta);
        all.extend_from_slice(&self.lm_head);

        f.write_all(&(all.len() as u64).to_le_bytes())?;
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(all.as_ptr() as *const u8, all.len() * 4) };
        f.write_all(bytes)?;
        Ok(())
    }

    /// Compute total gradient norm
    pub fn grad_norm(grads: &HamiltonianGradients) -> f32 {
        let mut total = 0.0f32;
        let all_vecs: Vec<&[f32]> = vec![
            &grads.token_emb, &grads.pos_emb, &grads.w_p_init,
            &grads.ln_f_gamma, &grads.ln_f_beta, &grads.lm_head,
        ];
        for v in &all_vecs {
            for x in v.iter() {
                total += x * x;
            }
        }
        for l in 0..grads.ln_v_gamma.len() {
            let layer_vecs: Vec<&[f32]> = vec![
                &grads.ln_v_gamma[l], &grads.ln_v_beta[l],
                &grads.qkv_w[l], &grads.attn_proj[l],
                &grads.ln_t_gamma[l], &grads.ln_t_beta[l],
                &grads.ff_w1[l], &grads.ff_b1[l],
                &grads.ff_w2[l], &grads.ff_b2[l],
            ];
            for v in &layer_vecs {
                for x in v.iter() {
                    total += x * x;
                }
            }
            total += grads.step_sizes[l] * grads.step_sizes[l];
        }
        total.sqrt()
    }

    /// Compute per-layer gradient norms (attention + FF per layer)
    pub fn per_layer_grad_norms(grads: &HamiltonianGradients) -> Vec<f32> {
        let mut norms = Vec::new();
        for l in 0..grads.ln_v_gamma.len() {
            let mut layer_norm_sq = 0.0f32;
            let layer_vecs: Vec<&[f32]> = vec![
                &grads.ln_v_gamma[l], &grads.ln_v_beta[l],
                &grads.qkv_w[l], &grads.attn_proj[l],
                &grads.ln_t_gamma[l], &grads.ln_t_beta[l],
                &grads.ff_w1[l], &grads.ff_b1[l],
                &grads.ff_w2[l], &grads.ff_b2[l],
            ];
            for v in &layer_vecs {
                for x in v.iter() {
                    layer_norm_sq += x * x;
                }
            }
            norms.push(layer_norm_sq.sqrt());
        }
        norms
    }
}

impl HamiltonianGradients {
    pub fn zero_like(cfg: &Config) -> Self {
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;
        let nl = cfg.n_layer;

        Self {
            token_emb: vec![0.0; v * e],
            pos_emb: vec![0.0; cfg.block_size * e],
            w_p_init: vec![0.0; e * e],
            ln_v_gamma: vec![vec![0.0; e]; nl],
            ln_v_beta: vec![vec![0.0; e]; nl],
            qkv_w: vec![vec![0.0; e * 3 * e]; nl],
            attn_proj: vec![vec![0.0; e * e]; nl],
            ln_t_gamma: vec![vec![0.0; e]; nl],
            ln_t_beta: vec![vec![0.0; e]; nl],
            ff_w1: vec![vec![0.0; e * inner]; nl],
            ff_b1: vec![vec![0.0; inner]; nl],
            ff_w2: vec![vec![0.0; inner * e]; nl],
            ff_b2: vec![vec![0.0; e]; nl],
            step_sizes: vec![0.0; nl],
            ln_f_gamma: vec![0.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: vec![0.0; e * v],
        }
    }

    pub fn accumulate(&mut self, other: &HamiltonianGradients) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        add_vecs(&mut self.w_p_init, &other.w_p_init);
        for l in 0..self.ln_v_gamma.len() {
            add_vecs(&mut self.ln_v_gamma[l], &other.ln_v_gamma[l]);
            add_vecs(&mut self.ln_v_beta[l], &other.ln_v_beta[l]);
            add_vecs(&mut self.qkv_w[l], &other.qkv_w[l]);
            add_vecs(&mut self.attn_proj[l], &other.attn_proj[l]);
            add_vecs(&mut self.ln_t_gamma[l], &other.ln_t_gamma[l]);
            add_vecs(&mut self.ln_t_beta[l], &other.ln_t_beta[l]);
            add_vecs(&mut self.ff_w1[l], &other.ff_w1[l]);
            add_vecs(&mut self.ff_b1[l], &other.ff_b1[l]);
            add_vecs(&mut self.ff_w2[l], &other.ff_w2[l]);
            add_vecs(&mut self.ff_b2[l], &other.ff_b2[l]);
            self.step_sizes[l] += other.step_sizes[l];
        }
        add_vecs(&mut self.ln_f_gamma, &other.ln_f_gamma);
        add_vecs(&mut self.ln_f_beta, &other.ln_f_beta);
        add_vecs(&mut self.lm_head, &other.lm_head);
    }

    pub fn scale(&mut self, factor: f32) {
        scale_vec(&mut self.token_emb, factor);
        scale_vec(&mut self.pos_emb, factor);
        scale_vec(&mut self.w_p_init, factor);
        for l in 0..self.ln_v_gamma.len() {
            scale_vec(&mut self.ln_v_gamma[l], factor);
            scale_vec(&mut self.ln_v_beta[l], factor);
            scale_vec(&mut self.qkv_w[l], factor);
            scale_vec(&mut self.attn_proj[l], factor);
            scale_vec(&mut self.ln_t_gamma[l], factor);
            scale_vec(&mut self.ln_t_beta[l], factor);
            scale_vec(&mut self.ff_w1[l], factor);
            scale_vec(&mut self.ff_b1[l], factor);
            scale_vec(&mut self.ff_w2[l], factor);
            scale_vec(&mut self.ff_b2[l], factor);
            self.step_sizes[l] *= factor;
        }
        scale_vec(&mut self.ln_f_gamma, factor);
        scale_vec(&mut self.ln_f_beta, factor);
        scale_vec(&mut self.lm_head, factor);
    }
}

// ---- Helper functions (duplicated from model.rs to keep modules independent) ----

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
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    t: usize,
    e: usize,
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
    x: &[f32],
    dout: &[f32],
    gamma: &[f32],
    t: usize,
    e: usize,
    dgamma: &mut Vec<f32>,
    dbeta: &mut Vec<f32>,
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
