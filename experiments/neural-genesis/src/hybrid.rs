use rand::Rng;
use std::io::Write;

use crate::model::Config;

/// Number of shared weight matrices (same as RG)
const NUM_SHARED: usize = 4; // qkv_w, attn_proj, ff_w1, ff_w2

/// HybridGPT: Combines RG Weight Sharing with Hamiltonian Residual Flow.
///
/// From RG: shared weight matrices with per-layer alpha/beta scaling
///   => parameter efficiency (57% fewer params)
///
/// From Hamiltonian: dual state (q, p) with leapfrog integrator
///   => gradient stability (symplectic energy conservation)
///
/// Combined: shared weights are used for BOTH V(q) and T(p) computations,
/// with per-layer alpha/beta providing scale differentiation.
pub struct HybridGPT {
    // Embeddings (not shared)
    pub token_emb: Vec<f32>,  // (vocab_size, n_embd)
    pub pos_emb: Vec<f32>,    // (block_size, n_embd)

    // Momentum initialization projection: q -> p
    pub w_p_init: Vec<f32>,   // (n_embd, n_embd)

    // Shared weight matrices (ONE copy, used across all layers)
    pub qkv_w_shared: Vec<f32>,      // (n_embd, 3*n_embd)
    pub attn_proj_shared: Vec<f32>,   // (n_embd, n_embd)
    pub ff_w1_shared: Vec<f32>,       // (n_embd, 4*n_embd)
    pub ff_w2_shared: Vec<f32>,       // (4*n_embd, n_embd)

    // Per-layer RG scale factors for V(q) attention path
    pub rg_alpha_v: Vec<[f32; NUM_SHARED]>,
    pub rg_beta_v: Vec<[f32; NUM_SHARED]>,

    // Per-layer RG scale factors for T(p) feed-forward path
    pub rg_alpha_t: Vec<[f32; NUM_SHARED]>,
    pub rg_beta_t: Vec<[f32; NUM_SHARED]>,

    // Per-layer layer norm params for V(q) path
    pub ln_v_gamma: Vec<Vec<f32>>,
    pub ln_v_beta: Vec<Vec<f32>>,

    // Per-layer layer norm params for T(p) path
    pub ln_t_gamma: Vec<Vec<f32>>,
    pub ln_t_beta: Vec<Vec<f32>>,

    // Per-layer biases (kept separate, small)
    pub ff_b1: Vec<Vec<f32>>,
    pub ff_b2: Vec<Vec<f32>>,

    // Per-layer step sizes for leapfrog (learned)
    pub step_sizes: Vec<f32>,

    // Final layer norm + head
    pub ln_f_gamma: Vec<f32>,
    pub ln_f_beta: Vec<f32>,
    pub lm_head: Vec<f32>,

    pub config: Config,

    // Adam optimizer state
    pub adam_m: Vec<Vec<f32>>,
    pub adam_v: Vec<Vec<f32>>,
    pub adam_t: usize,
}

/// Gradient storage for HybridGPT
pub struct HybridGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub w_p_init: Vec<f32>,
    pub qkv_w_shared: Vec<f32>,
    pub attn_proj_shared: Vec<f32>,
    pub ff_w1_shared: Vec<f32>,
    pub ff_w2_shared: Vec<f32>,
    pub rg_alpha_v: Vec<[f32; NUM_SHARED]>,
    pub rg_beta_v: Vec<[f32; NUM_SHARED]>,
    pub rg_alpha_t: Vec<[f32; NUM_SHARED]>,
    pub rg_beta_t: Vec<[f32; NUM_SHARED]>,
    pub ln_v_gamma: Vec<Vec<f32>>,
    pub ln_v_beta: Vec<Vec<f32>>,
    pub ln_t_gamma: Vec<Vec<f32>>,
    pub ln_t_beta: Vec<Vec<f32>>,
    pub ff_b1: Vec<Vec<f32>>,
    pub ff_b2: Vec<Vec<f32>>,
    pub step_sizes: Vec<f32>,
    pub ln_f_gamma: Vec<f32>,
    pub ln_f_beta: Vec<f32>,
    pub lm_head: Vec<f32>,
}

/// Per-layer energy diagnostics
#[derive(Clone, Default)]
pub struct HybridDiagnostics {
    pub per_layer_q_norm: Vec<f32>,
    pub per_layer_p_norm: Vec<f32>,
    pub per_layer_total_energy: Vec<f32>,
}

// Forward cache structures
#[derive(Default)]
struct HybridLayerCache {
    q_input: Vec<f32>,
    p_input: Vec<f32>,

    // V(q) computation - first half-step
    eff_qkv_w_v1: Vec<f32>,
    eff_attn_proj_v1: Vec<f32>,
    ln_v1_out: Vec<f32>,
    ln_v1_mean: Vec<f32>,
    ln_v1_rstd: Vec<f32>,
    qkv1: Vec<f32>,
    attn_weights1: Vec<f32>,
    attn_out1: Vec<f32>,
    v1_out: Vec<f32>,

    // T(p_half) computation
    eff_ff_w1_t: Vec<f32>,
    eff_ff_w2_t: Vec<f32>,
    p_half: Vec<f32>,
    ln_t_out: Vec<f32>,
    ln_t_mean: Vec<f32>,
    ln_t_rstd: Vec<f32>,
    ff_pre_gelu: Vec<f32>,
    ff_post_gelu: Vec<f32>,
    t_out: Vec<f32>,

    // V(q_new) computation - second half-step
    eff_qkv_w_v2: Vec<f32>,
    eff_attn_proj_v2: Vec<f32>,
    q_new: Vec<f32>,
    ln_v2_out: Vec<f32>,
    ln_v2_mean: Vec<f32>,
    ln_v2_rstd: Vec<f32>,
    qkv2: Vec<f32>,
    attn_weights2: Vec<f32>,
    attn_out2: Vec<f32>,
    v2_out: Vec<f32>,
}

struct HybridForwardCache {
    #[allow(dead_code)]
    tokens: Vec<usize>,
    x_after_emb: Vec<f32>,
    #[allow(dead_code)]
    p_after_init: Vec<f32>,
    tanh_input: Vec<f32>,
    layer_caches: Vec<HybridLayerCache>,
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

impl HybridGPT {
    pub fn new(config: Config) -> Self {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let bs = config.block_size;
        let inner = 4 * e;

        let emb_scale = 0.02;
        let layer_scale = (0.02 / (nl as f32).sqrt()).max(0.001);

        let mut ln_v_gamma = Vec::new();
        let mut ln_v_beta_v = Vec::new();
        let mut ln_t_gamma = Vec::new();
        let mut ln_t_beta_v = Vec::new();
        let mut ff_b1 = Vec::new();
        let mut ff_b2 = Vec::new();

        for _ in 0..nl {
            ln_v_gamma.push(vec![1.0; e]);
            ln_v_beta_v.push(vec![0.0; e]);
            ln_t_gamma.push(vec![1.0; e]);
            ln_t_beta_v.push(vec![0.0; e]);
            ff_b1.push(vec![0.0; inner]);
            ff_b2.push(vec![0.0; e]);
        }

        let rg_alpha_v = vec![[1.0f32; NUM_SHARED]; nl];
        let rg_beta_v = vec![[0.0f32; NUM_SHARED]; nl];
        let rg_alpha_t = vec![[1.0f32; NUM_SHARED]; nl];
        let rg_beta_t = vec![[0.0f32; NUM_SHARED]; nl];
        let step_sizes = vec![1.0f32; nl];

        let mut model = HybridGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            w_p_init: randn_vec(e * e, layer_scale),
            qkv_w_shared: randn_vec(e * 3 * e, layer_scale),
            attn_proj_shared: randn_vec(e * e, layer_scale),
            ff_w1_shared: randn_vec(e * inner, layer_scale),
            ff_w2_shared: randn_vec(inner * e, layer_scale * 0.5),
            rg_alpha_v,
            rg_beta_v,
            rg_alpha_t,
            rg_beta_t,
            ln_v_gamma,
            ln_v_beta: ln_v_beta_v,
            ln_t_gamma,
            ln_t_beta: ln_t_beta_v,
            ff_b1,
            ff_b2,
            step_sizes,
            ln_f_gamma: vec![1.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: randn_vec(e * v, emb_scale),
            config,
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            adam_t: 0,
        };

        let param_sizes = model.param_sizes();
        model.adam_m = param_sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.adam_v = param_sizes.iter().map(|&s| vec![0.0; s]).collect();

        model
    }

    fn param_sizes(&self) -> Vec<usize> {
        let nl = self.config.n_layer;
        let mut sizes = Vec::new();
        sizes.push(self.token_emb.len());
        sizes.push(self.pos_emb.len());
        sizes.push(self.w_p_init.len());
        sizes.push(self.qkv_w_shared.len());
        sizes.push(self.attn_proj_shared.len());
        sizes.push(self.ff_w1_shared.len());
        sizes.push(self.ff_w2_shared.len());
        for _ in 0..nl {
            sizes.push(NUM_SHARED); // alpha_v
            sizes.push(NUM_SHARED); // beta_v
            sizes.push(NUM_SHARED); // alpha_t
            sizes.push(NUM_SHARED); // beta_t
        }
        for l in 0..nl {
            sizes.push(self.ln_v_gamma[l].len());
            sizes.push(self.ln_v_beta[l].len());
            sizes.push(self.ln_t_gamma[l].len());
            sizes.push(self.ln_t_beta[l].len());
            sizes.push(self.ff_b1[l].len());
            sizes.push(self.ff_b2[l].len());
            sizes.push(1); // step_size
        }
        sizes.push(self.ln_f_gamma.len());
        sizes.push(self.ln_f_beta.len());
        sizes.push(self.lm_head.len());
        sizes
    }

    pub fn count_params(&self) -> usize {
        let e = self.config.n_embd;
        let v = self.config.vocab_size;
        let nl = self.config.n_layer;
        let inner = 4 * e;
        let bs = self.config.block_size;

        let emb = v * e + bs * e;
        let p_init = e * e;
        let shared = e * 3 * e + e * e + e * inner + inner * e;
        let rg_scalars = nl * NUM_SHARED * 4; // alpha_v, beta_v, alpha_t, beta_t
        let per_layer = nl * (e + e + e + e + inner + e + 1); // ln_v, ln_t, ff_b1, ff_b2, step_size
        let head = e + e + e * v;

        emb + p_init + shared + rg_scalars + per_layer + head
    }

    /// Compute V(q) = attention using shared weights with RG scaling
    fn compute_v(
        &self,
        q: &[f32],
        l: usize,
        seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let e = self.config.n_embd;
        let nh = self.config.n_head;
        let hs = e / nh;

        // Effective weights: W_eff = alpha * W_shared + beta
        let eff_qkv_w: Vec<f32> = self.qkv_w_shared.iter()
            .map(|&w| self.rg_alpha_v[l][0] * w + self.rg_beta_v[l][0]).collect();
        let eff_attn_proj: Vec<f32> = self.attn_proj_shared.iter()
            .map(|&w| self.rg_alpha_v[l][1] * w + self.rg_beta_v[l][1]).collect();

        let (ln_out, ln_mean, ln_rstd) =
            layer_norm(q, &self.ln_v_gamma[l], &self.ln_v_beta[l], seq_len, e);

        let qkv = matmul(&ln_out, &eff_qkv_w, seq_len, e, 3 * e);

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

            for i in 0..seq_len {
                let offset = h * seq_len * seq_len + i * seq_len;
                let max_val = all_attn_weights[offset..offset + seq_len]
                    .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

        let proj_out = matmul(&attn_out, &eff_attn_proj, seq_len, e, e);

        (proj_out, ln_out, ln_mean, ln_rstd, qkv, all_attn_weights, attn_out, eff_qkv_w, eff_attn_proj)
    }

    /// Compute T(p) = feed-forward using shared weights with RG scaling
    fn compute_t(
        &self,
        p: &[f32],
        l: usize,
        seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let e = self.config.n_embd;
        let inner = 4 * e;

        // Effective weights for T path
        let eff_ff_w1: Vec<f32> = self.ff_w1_shared.iter()
            .map(|&w| self.rg_alpha_t[l][2] * w + self.rg_beta_t[l][2]).collect();
        let eff_ff_w2: Vec<f32> = self.ff_w2_shared.iter()
            .map(|&w| self.rg_alpha_t[l][3] * w + self.rg_beta_t[l][3]).collect();

        let (ln_out, ln_mean, ln_rstd) =
            layer_norm(p, &self.ln_t_gamma[l], &self.ln_t_beta[l], seq_len, e);

        let mut ff_hidden = matmul(&ln_out, &eff_ff_w1, seq_len, e, inner);
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

        let mut ff_out = matmul(&ff_hidden, &eff_ff_w2, seq_len, inner, e);
        for i in 0..seq_len {
            for j in 0..e {
                ff_out[i * e + j] += self.ff_b2[l][j];
            }
        }

        (ff_out, ln_out, ln_mean, ln_rstd, ff_pre_gelu, ff_post_gelu, eff_ff_w1, eff_ff_w2)
    }

    /// Forward pass with full cache for backward
    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, HybridForwardCache, HybridDiagnostics) {
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
        let mut diagnostics = HybridDiagnostics::default();
        let mut layer_caches = Vec::new();

        // Leapfrog integration through layers
        for l in 0..cfg.n_layer {
            let mut lc = HybridLayerCache::default();
            lc.q_input = q.clone();
            lc.p_input = p.clone();

            let h = self.step_sizes[l];
            let h_half = h * 0.5;

            // Step 1: p_half = p - (h/2) * V(q)
            let (v1_out, ln_v1_out, ln_v1_mean, ln_v1_rstd, qkv1, aw1, ao1, eff_qkv1, eff_ap1) =
                self.compute_v(&q, l, seq_len);
            lc.eff_qkv_w_v1 = eff_qkv1;
            lc.eff_attn_proj_v1 = eff_ap1;
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
            let (t_out, ln_t_out, ln_t_mean, ln_t_rstd, ff_pre_gelu, ff_post_gelu, eff_fw1, eff_fw2) =
                self.compute_t(&p_half, l, seq_len);
            lc.eff_ff_w1_t = eff_fw1;
            lc.eff_ff_w2_t = eff_fw2;
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
            let (v2_out, ln_v2_out, ln_v2_mean, ln_v2_rstd, qkv2, aw2, ao2, eff_qkv2, eff_ap2) =
                self.compute_v(&q_new, l, seq_len);
            lc.eff_qkv_w_v2 = eff_qkv2;
            lc.eff_attn_proj_v2 = eff_ap2;
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

            // Track energy
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
        let (ln_out, _, _) = layer_norm(&q, &self.ln_f_gamma, &self.ln_f_beta, seq_len, e);
        let q_after_final_ln = ln_out.clone();
        let logits = matmul(&ln_out, &self.lm_head, seq_len, e, v);

        let cache = HybridForwardCache {
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

    /// Forward + backward returning loss, gradients, and diagnostics
    pub fn forward_backward(
        &self,
        tokens: &[usize],
        targets: &[usize],
    ) -> (f32, HybridGradients, HybridDiagnostics) {
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
                .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

        let mut d_logits = probs;
        for i in 0..seq_len {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v {
                d_logits[i * v + j] /= seq_len as f32;
            }
        }

        let mut grads = HybridGradients::zero_like(cfg);

        // Backward through LM head
        let d_ln_out = matmul_backward_both(
            &cache.q_after_final_ln, &self.lm_head, &d_logits,
            seq_len, e, v, &mut grads.lm_head,
        );

        // Backward through final layer norm
        let mut dq = layer_norm_backward(
            &cache.q_before_final_ln, &d_ln_out, &self.ln_f_gamma,
            seq_len, e, &mut grads.ln_f_gamma, &mut grads.ln_f_beta,
        );

        let mut dp = vec![0.0f32; seq_len * e];

        // Backward through layers in reverse
        for l in (0..cfg.n_layer).rev() {
            let lc = &cache.layer_caches[l];
            let h = self.step_sizes[l];
            let h_half = h * 0.5;

            // === Step 3 backward: p_new = p_half - (h/2) * V(q_new) ===
            let mut dp_half = dp.clone();
            let mut d_v2_out = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                d_v2_out[i] = -h_half * dp[i];
            }

            let mut d_h: f32 = 0.0;
            for i in 0..seq_len * e {
                d_h += -0.5 * lc.v2_out[i] * dp[i];
            }

            // Backward through V(q_new) with RG
            let dq_from_v2 = self.backward_v_rg(
                &lc.q_new, &d_v2_out, l, seq_len,
                &lc.ln_v2_out, &lc.qkv2, &lc.attn_weights2, &lc.attn_out2,
                &lc.eff_qkv_w_v2, &lc.eff_attn_proj_v2,
                &mut grads,
            );

            for i in 0..seq_len * e {
                dq[i] += dq_from_v2[i];
            }

            // === Step 2 backward: q_new = q + h * T(p_half) ===
            let mut d_t_out = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                d_t_out[i] = h * dq[i];
            }

            for i in 0..seq_len * e {
                d_h += lc.t_out[i] * dq[i];
            }

            let dp_from_t = self.backward_t_rg(
                &lc.p_half, &d_t_out, l, seq_len,
                &lc.ln_t_out, &lc.ff_pre_gelu, &lc.ff_post_gelu,
                &lc.eff_ff_w1_t, &lc.eff_ff_w2_t,
                &mut grads,
            );

            for i in 0..seq_len * e {
                dp_half[i] += dp_from_t[i];
            }

            let dq_input = dq;

            // === Step 1 backward: p_half = p - (h/2) * V(q) ===
            let dp_input = dp_half.clone();
            let mut d_v1_out = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                d_v1_out[i] = -h_half * dp_half[i];
            }

            for i in 0..seq_len * e {
                d_h += -0.5 * lc.v1_out[i] * dp_half[i];
            }

            let dq_from_v1 = self.backward_v_rg(
                &lc.q_input, &d_v1_out, l, seq_len,
                &lc.ln_v1_out, &lc.qkv1, &lc.attn_weights1, &lc.attn_out1,
                &lc.eff_qkv_w_v1, &lc.eff_attn_proj_v1,
                &mut grads,
            );

            dq = vec![0.0f32; seq_len * e];
            for i in 0..seq_len * e {
                dq[i] = dq_input[i] + dq_from_v1[i];
            }

            dp = dp_input;
            grads.step_sizes[l] += d_h;
        }

        // Backward through momentum initialization
        let mut d_tanh_input = vec![0.0f32; seq_len * e];
        for i in 0..seq_len * e {
            let t_val = cache.tanh_input[i].tanh();
            d_tanh_input[i] = dp[i] * (1.0 - t_val * t_val);
        }

        let dq_from_init = matmul_backward_both(
            &cache.x_after_emb, &self.w_p_init, &d_tanh_input,
            seq_len, e, e, &mut grads.w_p_init,
        );

        for i in 0..seq_len * e {
            dq[i] += dq_from_init[i];
        }

        // Embedding gradients
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                grads.token_emb[tok * e + j] += dq[i * e + j];
                grads.pos_emb[i * e + j] += dq[i * e + j];
            }
        }

        (loss, grads, diagnostics)
    }

    /// Backward through V(q) with RG weight sharing
    fn backward_v_rg(
        &self,
        q_in: &[f32],
        d_out: &[f32],
        l: usize,
        seq_len: usize,
        ln_out: &[f32],
        qkv: &[f32],
        attn_weights: &[f32],
        attn_out_pre_proj: &[f32],
        eff_qkv_w: &[f32],
        eff_attn_proj: &[f32],
        grads: &mut HybridGradients,
    ) -> Vec<f32> {
        let e = self.config.n_embd;
        let nh = self.config.n_head;
        let hs = e / nh;

        // Backward through output projection (using effective weights)
        let mut d_eff_attn_proj = vec![0.0f32; e * e];
        let d_attn_out = matmul_backward_both(
            attn_out_pre_proj, eff_attn_proj, d_out,
            seq_len, e, e, &mut d_eff_attn_proj,
        );

        // RG backward for attn_proj (index 1)
        let alpha_1 = self.rg_alpha_v[l][1];
        for i in 0..d_eff_attn_proj.len() {
            grads.rg_alpha_v[l][1] += d_eff_attn_proj[i] * self.attn_proj_shared[i];
            grads.rg_beta_v[l][1] += d_eff_attn_proj[i];
            grads.attn_proj_shared[i] += alpha_1 * d_eff_attn_proj[i];
        }

        // Multi-head attention backward
        let mut d_qkv = vec![0.0f32; seq_len * 3 * e];

        for h in 0..nh {
            for i in 0..seq_len {
                for k in 0..hs {
                    let d_o = d_attn_out[i * e + h * hs + k];
                    for j in 0..seq_len {
                        let w = attn_weights[h * seq_len * seq_len + i * seq_len + j];
                        d_qkv[j * 3 * e + 2 * e + h * hs + k] += w * d_o;
                    }
                }
            }

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

            let scale = 1.0 / (hs as f32).sqrt();
            for val in d_attn_score.iter_mut() {
                *val *= scale;
            }

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

        // Backward through QKV projection (using effective weights)
        let mut d_eff_qkv_w = vec![0.0f32; e * 3 * e];
        let d_ln_out = matmul_backward_both(
            ln_out, eff_qkv_w, &d_qkv,
            seq_len, e, 3 * e, &mut d_eff_qkv_w,
        );

        // RG backward for qkv_w (index 0)
        let alpha_0 = self.rg_alpha_v[l][0];
        for i in 0..d_eff_qkv_w.len() {
            grads.rg_alpha_v[l][0] += d_eff_qkv_w[i] * self.qkv_w_shared[i];
            grads.rg_beta_v[l][0] += d_eff_qkv_w[i];
            grads.qkv_w_shared[i] += alpha_0 * d_eff_qkv_w[i];
        }

        // Backward through layer norm
        layer_norm_backward(
            q_in, &d_ln_out, &self.ln_v_gamma[l],
            seq_len, e, &mut grads.ln_v_gamma[l], &mut grads.ln_v_beta[l],
        )
    }

    /// Backward through T(p) with RG weight sharing
    fn backward_t_rg(
        &self,
        p_in: &[f32],
        d_out: &[f32],
        l: usize,
        seq_len: usize,
        ln_out: &[f32],
        ff_pre_gelu: &[f32],
        ff_post_gelu: &[f32],
        eff_ff_w1: &[f32],
        eff_ff_w2: &[f32],
        grads: &mut HybridGradients,
    ) -> Vec<f32> {
        let e = self.config.n_embd;
        let inner = 4 * e;

        // ff_b2 gradients
        for i in 0..seq_len {
            for j in 0..e {
                grads.ff_b2[l][j] += d_out[i * e + j];
            }
        }

        // Backward through ff_w2
        let mut d_eff_ff_w2 = vec![0.0f32; inner * e];
        let d_ff_hidden = matmul_backward_both(
            ff_post_gelu, eff_ff_w2, d_out,
            seq_len, inner, e, &mut d_eff_ff_w2,
        );

        // RG backward for ff_w2 (index 3)
        let alpha_3 = self.rg_alpha_t[l][3];
        for i in 0..d_eff_ff_w2.len() {
            grads.rg_alpha_t[l][3] += d_eff_ff_w2[i] * self.ff_w2_shared[i];
            grads.rg_beta_t[l][3] += d_eff_ff_w2[i];
            grads.ff_w2_shared[i] += alpha_3 * d_eff_ff_w2[i];
        }

        // GELU backward
        let d_ff_pre_gelu = gelu_backward(ff_pre_gelu, &d_ff_hidden);

        // ff_b1 gradients
        for i in 0..seq_len {
            for j in 0..inner {
                grads.ff_b1[l][j] += d_ff_pre_gelu[i * inner + j];
            }
        }

        // Backward through ff_w1
        let mut d_eff_ff_w1 = vec![0.0f32; e * inner];
        let d_ln_out = matmul_backward_both(
            ln_out, eff_ff_w1, &d_ff_pre_gelu,
            seq_len, e, inner, &mut d_eff_ff_w1,
        );

        // RG backward for ff_w1 (index 2)
        let alpha_2 = self.rg_alpha_t[l][2];
        for i in 0..d_eff_ff_w1.len() {
            grads.rg_alpha_t[l][2] += d_eff_ff_w1[i] * self.ff_w1_shared[i];
            grads.rg_beta_t[l][2] += d_eff_ff_w1[i];
            grads.ff_w1_shared[i] += alpha_2 * d_eff_ff_w1[i];
        }

        // Backward through layer norm
        layer_norm_backward(
            p_in, &d_ln_out, &self.ln_t_gamma[l],
            seq_len, e, &mut grads.ln_t_gamma[l], &mut grads.ln_t_beta[l],
        )
    }

    /// Apply gradients with Adam optimizer
    pub fn apply_gradients(&mut self, grads: &HybridGradients, lr: f32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        self.adam_t += 1;
        let t_f = self.adam_t as f32;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);

        let mut idx = 0usize;

        macro_rules! adam_update {
            ($param:expr, $grad:expr) => {{
                adam_step(
                    $param, $grad, &mut self.adam_m[idx], &mut self.adam_v[idx],
                    lr, beta1, beta2, eps, bc1, bc2,
                );
                idx += 1;
                let _ = idx;
            }};
        }

        adam_update!(&mut self.token_emb, &grads.token_emb);
        adam_update!(&mut self.pos_emb, &grads.pos_emb);
        adam_update!(&mut self.w_p_init, &grads.w_p_init);
        adam_update!(&mut self.qkv_w_shared, &grads.qkv_w_shared);
        adam_update!(&mut self.attn_proj_shared, &grads.attn_proj_shared);
        adam_update!(&mut self.ff_w1_shared, &grads.ff_w1_shared);
        adam_update!(&mut self.ff_w2_shared, &grads.ff_w2_shared);

        for l in 0..self.config.n_layer {
            let mut alpha_v_vec: Vec<f32> = self.rg_alpha_v[l].to_vec();
            let alpha_v_grad: Vec<f32> = grads.rg_alpha_v[l].to_vec();
            adam_step(&mut alpha_v_vec, &alpha_v_grad,
                      &mut self.adam_m[idx], &mut self.adam_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.rg_alpha_v[l].copy_from_slice(&alpha_v_vec);
            idx += 1;

            let mut beta_v_vec: Vec<f32> = self.rg_beta_v[l].to_vec();
            let beta_v_grad: Vec<f32> = grads.rg_beta_v[l].to_vec();
            adam_step(&mut beta_v_vec, &beta_v_grad,
                      &mut self.adam_m[idx], &mut self.adam_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.rg_beta_v[l].copy_from_slice(&beta_v_vec);
            idx += 1;

            let mut alpha_t_vec: Vec<f32> = self.rg_alpha_t[l].to_vec();
            let alpha_t_grad: Vec<f32> = grads.rg_alpha_t[l].to_vec();
            adam_step(&mut alpha_t_vec, &alpha_t_grad,
                      &mut self.adam_m[idx], &mut self.adam_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.rg_alpha_t[l].copy_from_slice(&alpha_t_vec);
            idx += 1;

            let mut beta_t_vec: Vec<f32> = self.rg_beta_t[l].to_vec();
            let beta_t_grad: Vec<f32> = grads.rg_beta_t[l].to_vec();
            adam_step(&mut beta_t_vec, &beta_t_grad,
                      &mut self.adam_m[idx], &mut self.adam_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.rg_beta_t[l].copy_from_slice(&beta_t_vec);
            idx += 1;
        }

        for l in 0..self.config.n_layer {
            adam_update!(&mut self.ln_v_gamma[l], &grads.ln_v_gamma[l]);
            adam_update!(&mut self.ln_v_beta[l], &grads.ln_v_beta[l]);
            adam_update!(&mut self.ln_t_gamma[l], &grads.ln_t_gamma[l]);
            adam_update!(&mut self.ln_t_beta[l], &grads.ln_t_beta[l]);
            adam_update!(&mut self.ff_b1[l], &grads.ff_b1[l]);
            adam_update!(&mut self.ff_b2[l], &grads.ff_b2[l]);
            // Step size scalar
            {
                let g = grads.step_sizes[l].max(-1.0).min(1.0);
                self.adam_m[idx][0] = beta1 * self.adam_m[idx][0] + (1.0 - beta1) * g;
                self.adam_v[idx][0] = beta2 * self.adam_v[idx][0] + (1.0 - beta2) * g * g;
                let m_hat = self.adam_m[idx][0] / bc1;
                let v_hat = self.adam_v[idx][0] / bc2;
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
        f.write_all(b"HYBR")?;
        f.write_all(&(self.config.vocab_size as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_embd as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_head as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_layer as u32).to_le_bytes())?;
        f.write_all(&(self.config.block_size as u32).to_le_bytes())?;

        let mut all: Vec<f32> = Vec::new();
        all.extend_from_slice(&self.token_emb);
        all.extend_from_slice(&self.pos_emb);
        all.extend_from_slice(&self.w_p_init);
        all.extend_from_slice(&self.qkv_w_shared);
        all.extend_from_slice(&self.attn_proj_shared);
        all.extend_from_slice(&self.ff_w1_shared);
        all.extend_from_slice(&self.ff_w2_shared);
        for l in 0..self.config.n_layer {
            all.extend_from_slice(&self.rg_alpha_v[l]);
            all.extend_from_slice(&self.rg_beta_v[l]);
            all.extend_from_slice(&self.rg_alpha_t[l]);
            all.extend_from_slice(&self.rg_beta_t[l]);
            all.extend_from_slice(&self.ln_v_gamma[l]);
            all.extend_from_slice(&self.ln_v_beta[l]);
            all.extend_from_slice(&self.ln_t_gamma[l]);
            all.extend_from_slice(&self.ln_t_beta[l]);
            all.extend_from_slice(&self.ff_b1[l]);
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

    /// Get per-layer alpha values for V path
    pub fn get_alphas_v(&self) -> Vec<[f32; NUM_SHARED]> {
        self.rg_alpha_v.clone()
    }

    /// Get per-layer alpha values for T path
    pub fn get_alphas_t(&self) -> Vec<[f32; NUM_SHARED]> {
        self.rg_alpha_t.clone()
    }

    /// Compute total gradient norm
    pub fn grad_norm(grads: &HybridGradients) -> f32 {
        let mut total = 0.0f32;
        let all_vecs: Vec<&[f32]> = vec![
            &grads.token_emb, &grads.pos_emb, &grads.w_p_init,
            &grads.qkv_w_shared, &grads.attn_proj_shared,
            &grads.ff_w1_shared, &grads.ff_w2_shared,
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
                &grads.ln_t_gamma[l], &grads.ln_t_beta[l],
                &grads.ff_b1[l], &grads.ff_b2[l],
            ];
            for v in &layer_vecs {
                for x in v.iter() {
                    total += x * x;
                }
            }
            for k in 0..NUM_SHARED {
                total += grads.rg_alpha_v[l][k] * grads.rg_alpha_v[l][k];
                total += grads.rg_beta_v[l][k] * grads.rg_beta_v[l][k];
                total += grads.rg_alpha_t[l][k] * grads.rg_alpha_t[l][k];
                total += grads.rg_beta_t[l][k] * grads.rg_beta_t[l][k];
            }
            total += grads.step_sizes[l] * grads.step_sizes[l];
        }
        total.sqrt()
    }
}

impl HybridGradients {
    pub fn zero_like(cfg: &Config) -> Self {
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;
        let nl = cfg.n_layer;

        Self {
            token_emb: vec![0.0; v * e],
            pos_emb: vec![0.0; cfg.block_size * e],
            w_p_init: vec![0.0; e * e],
            qkv_w_shared: vec![0.0; e * 3 * e],
            attn_proj_shared: vec![0.0; e * e],
            ff_w1_shared: vec![0.0; e * inner],
            ff_w2_shared: vec![0.0; inner * e],
            rg_alpha_v: vec![[0.0; NUM_SHARED]; nl],
            rg_beta_v: vec![[0.0; NUM_SHARED]; nl],
            rg_alpha_t: vec![[0.0; NUM_SHARED]; nl],
            rg_beta_t: vec![[0.0; NUM_SHARED]; nl],
            ln_v_gamma: vec![vec![0.0; e]; nl],
            ln_v_beta: vec![vec![0.0; e]; nl],
            ln_t_gamma: vec![vec![0.0; e]; nl],
            ln_t_beta: vec![vec![0.0; e]; nl],
            ff_b1: vec![vec![0.0; inner]; nl],
            ff_b2: vec![vec![0.0; e]; nl],
            step_sizes: vec![0.0; nl],
            ln_f_gamma: vec![0.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: vec![0.0; e * v],
        }
    }

    pub fn accumulate(&mut self, other: &HybridGradients) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        add_vecs(&mut self.w_p_init, &other.w_p_init);
        add_vecs(&mut self.qkv_w_shared, &other.qkv_w_shared);
        add_vecs(&mut self.attn_proj_shared, &other.attn_proj_shared);
        add_vecs(&mut self.ff_w1_shared, &other.ff_w1_shared);
        add_vecs(&mut self.ff_w2_shared, &other.ff_w2_shared);
        for l in 0..self.rg_alpha_v.len() {
            for k in 0..NUM_SHARED {
                self.rg_alpha_v[l][k] += other.rg_alpha_v[l][k];
                self.rg_beta_v[l][k] += other.rg_beta_v[l][k];
                self.rg_alpha_t[l][k] += other.rg_alpha_t[l][k];
                self.rg_beta_t[l][k] += other.rg_beta_t[l][k];
            }
            add_vecs(&mut self.ln_v_gamma[l], &other.ln_v_gamma[l]);
            add_vecs(&mut self.ln_v_beta[l], &other.ln_v_beta[l]);
            add_vecs(&mut self.ln_t_gamma[l], &other.ln_t_gamma[l]);
            add_vecs(&mut self.ln_t_beta[l], &other.ln_t_beta[l]);
            add_vecs(&mut self.ff_b1[l], &other.ff_b1[l]);
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
        scale_vec(&mut self.qkv_w_shared, factor);
        scale_vec(&mut self.attn_proj_shared, factor);
        scale_vec(&mut self.ff_w1_shared, factor);
        scale_vec(&mut self.ff_w2_shared, factor);
        for l in 0..self.rg_alpha_v.len() {
            for k in 0..NUM_SHARED {
                self.rg_alpha_v[l][k] *= factor;
                self.rg_beta_v[l][k] *= factor;
                self.rg_alpha_t[l][k] *= factor;
                self.rg_beta_t[l][k] *= factor;
            }
            scale_vec(&mut self.ln_v_gamma[l], factor);
            scale_vec(&mut self.ln_v_beta[l], factor);
            scale_vec(&mut self.ln_t_gamma[l], factor);
            scale_vec(&mut self.ln_t_beta[l], factor);
            scale_vec(&mut self.ff_b1[l], factor);
            scale_vec(&mut self.ff_b2[l], factor);
            self.step_sizes[l] *= factor;
        }
        scale_vec(&mut self.ln_f_gamma, factor);
        scale_vec(&mut self.ln_f_beta, factor);
        scale_vec(&mut self.lm_head, factor);
    }
}

// ─── Helper functions (duplicated from model.rs / hamiltonian.rs) ───

fn adam_step(
    params: &mut Vec<f32>, grads: &[f32],
    m: &mut Vec<f32>, v: &mut Vec<f32>,
    lr: f32, beta1: f32, beta2: f32, eps: f32, bc1: f32, bc2: f32,
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
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

fn scale_vec(a: &mut Vec<f32>, s: f32) {
    for val in a.iter_mut() {
        *val *= s;
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
    a: &[f32], b: &[f32], d_c: &[f32],
    m: usize, k: usize, n: usize,
    d_b: &mut Vec<f32>,
) -> Vec<f32> {
    let mut d_a = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..n {
            let dc = d_c[i * n + j];
            if dc.abs() > 1e-12 {
                for p in 0..k {
                    d_a[i * k + p] += dc * b[p * n + j];
                    d_b[p * n + j] += dc * a[i * k + p];
                }
            }
        }
    }
    d_a
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
        let var: f32 = x[offset..offset + e].iter()
            .map(|&v| (v - mean) * (v - mean)).sum::<f32>() / e as f32;
        let rstd = 1.0 / (var + eps).sqrt();

        means[i] = mean;
        rstds[i] = rstd;

        for j in 0..e {
            out[offset + j] = gamma[j] * (x[offset + j] - mean) * rstd + beta[j];
        }
    }

    (out, means, rstds)
}

fn layer_norm_backward(
    x: &[f32], d_out: &[f32], gamma: &[f32],
    t: usize, e: usize,
    d_gamma: &mut Vec<f32>, d_beta: &mut Vec<f32>,
) -> Vec<f32> {
    let eps = 1e-5f32;
    let mut dx = vec![0.0f32; t * e];

    for i in 0..t {
        let offset = i * e;
        let mean: f32 = x[offset..offset + e].iter().sum::<f32>() / e as f32;
        let var: f32 = x[offset..offset + e].iter()
            .map(|&v| (v - mean) * (v - mean)).sum::<f32>() / e as f32;
        let rstd = 1.0 / (var + eps).sqrt();

        let mut d_var = 0.0f32;
        let mut d_mean = 0.0f32;

        for j in 0..e {
            let x_hat = (x[offset + j] - mean) * rstd;
            d_gamma[j] += d_out[offset + j] * x_hat;
            d_beta[j] += d_out[offset + j];

            let d_x_hat = d_out[offset + j] * gamma[j];
            d_var += d_x_hat * (x[offset + j] - mean) * -0.5 * rstd * rstd * rstd;
            d_mean += d_x_hat * -rstd;
        }

        d_mean += d_var * -2.0 * x[offset..offset + e].iter()
            .map(|&v| v - mean).sum::<f32>() / e as f32;

        for j in 0..e {
            dx[offset + j] = d_out[offset + j] * gamma[j] * rstd
                + d_var * 2.0 * (x[offset + j] - mean) / e as f32
                + d_mean / e as f32;
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
