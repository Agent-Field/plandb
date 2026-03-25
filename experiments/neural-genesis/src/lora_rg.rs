// ─── EXP8: RG-Informed Architecture Search ───────────────────
//
// Three theory-driven improvements to RG weight sharing, informed by EXP7 scaling findings:
//   EXP8a: LoRA-RG — rank-r per-layer adaptation instead of scalar alpha/beta
//   EXP8b: PreAmpRG — ff_w1 pre-amplified 1.2x, removed from alpha training
//   EXP8c: PerHeadRG — per-head alpha for attention weights

use crate::model::{Config, randn_vec, add_vecs, scale_vec, matmul, matmul_backward_both,
                    layer_norm, layer_norm_backward, gelu_backward, adam_step};
use rand::Rng;
use std::io::Write;

/// Number of shared weight matrices
const NUM_SHARED: usize = 4;

// ═══════════════════════════════════════════════════════════════
// EXP8a: LoRA-RG — Low-Rank Adaptation per layer
// W_eff^(l) = W_shared + U_l @ V_l^T  (rank r per layer)
// ═══════════════════════════════════════════════════════════════

pub struct LoraRGGPT {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,

    // Shared weight matrices (ONE copy)
    pub qkv_w_shared: Vec<f32>,    // e x 3e
    pub attn_proj_shared: Vec<f32>, // e x e
    pub ff_w1_shared: Vec<f32>,    // e x 4e
    pub ff_w2_shared: Vec<f32>,    // 4e x e

    // Per-layer LoRA factors: U_l (d_out x r) and V_l (d_in x r)
    // For each layer and each of 4 matrices
    pub lora_u: Vec<Vec<Vec<f32>>>,  // [layer][matrix_idx] = flattened U (d_out * r)
    pub lora_v: Vec<Vec<Vec<f32>>>,  // [layer][matrix_idx] = flattened V (d_in * r)

    // Per-layer small parameters
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
    pub lora_rank: usize,

    // Adam state
    adam_m: Vec<Vec<f32>>,
    adam_v: Vec<Vec<f32>>,
    adam_t: usize,
}

pub struct LoraRGGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub qkv_w_shared: Vec<f32>,
    pub attn_proj_shared: Vec<f32>,
    pub ff_w1_shared: Vec<f32>,
    pub ff_w2_shared: Vec<f32>,
    pub lora_u: Vec<Vec<Vec<f32>>>,
    pub lora_v: Vec<Vec<Vec<f32>>>,
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

#[derive(Default)]
struct LoraLayerCache {
    x_input: Vec<f32>,
    eff_qkv_w: Vec<f32>,
    eff_attn_proj: Vec<f32>,
    eff_ff_w1: Vec<f32>,
    eff_ff_w2: Vec<f32>,
    ln1_out: Vec<f32>,
    qkv: Vec<f32>,
    attn_weights: Vec<f32>,
    attn_out_pre_proj: Vec<f32>,
    x_after_attn_residual: Vec<f32>,
    ln2_out: Vec<f32>,
    ff_pre_gelu: Vec<f32>,
    ff_post_gelu: Vec<f32>,
}

struct LoraForwardCache {
    tokens: Vec<usize>,
    x_after_emb: Vec<f32>,
    layer_caches: Vec<LoraLayerCache>,
    x_before_final_ln: Vec<f32>,
    x_after_final_ln: Vec<f32>,
}

/// Matrix dimensions for each of the 4 shared weight matrices
fn matrix_dims(e: usize) -> [(usize, usize); NUM_SHARED] {
    let inner = 4 * e;
    [
        (e, 3 * e),   // qkv: input=e, output=3e
        (e, e),       // attn_proj: input=e, output=e
        (e, inner),   // ff_w1: input=e, output=4e
        (inner, e),   // ff_w2: input=4e, output=e
    ]
}

/// Compute W_eff = W_shared + U @ V^T for a single matrix
/// U is (d_out x r), V is (d_in x r), both stored row-major
/// W_shared is (d_in x d_out) stored row-major
fn compute_lora_effective(
    w_shared: &[f32], u: &[f32], v: &[f32],
    d_in: usize, d_out: usize, r: usize,
) -> Vec<f32> {
    // W_eff[i,j] = W_shared[i,j] + sum_k V[i,k] * U[j,k]
    // where W is (d_in x d_out), V is (d_in x r), U is (d_out x r)
    let mut eff = w_shared.to_vec();
    for i in 0..d_in {
        for j in 0..d_out {
            let mut sum = 0.0f32;
            for k in 0..r {
                sum += v[i * r + k] * u[j * r + k];
            }
            eff[i * d_out + j] += sum;
        }
    }
    eff
}

impl LoraRGGradients {
    pub fn zero_like(cfg: &Config, rank: usize) -> Self {
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;
        let nl = cfg.n_layer;
        let dims = matrix_dims(e);

        let mut lora_u = Vec::new();
        let mut lora_v = Vec::new();
        for _ in 0..nl {
            let mut u_layer = Vec::new();
            let mut v_layer = Vec::new();
            for &(d_in, d_out) in &dims {
                u_layer.push(vec![0.0; d_out * rank]);
                v_layer.push(vec![0.0; d_in * rank]);
            }
            lora_u.push(u_layer);
            lora_v.push(v_layer);
        }

        Self {
            token_emb: vec![0.0; v * e],
            pos_emb: vec![0.0; cfg.block_size * e],
            qkv_w_shared: vec![0.0; e * 3 * e],
            attn_proj_shared: vec![0.0; e * e],
            ff_w1_shared: vec![0.0; e * inner],
            ff_w2_shared: vec![0.0; inner * e],
            lora_u,
            lora_v,
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

    pub fn accumulate(&mut self, other: &Self) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        add_vecs(&mut self.qkv_w_shared, &other.qkv_w_shared);
        add_vecs(&mut self.attn_proj_shared, &other.attn_proj_shared);
        add_vecs(&mut self.ff_w1_shared, &other.ff_w1_shared);
        add_vecs(&mut self.ff_w2_shared, &other.ff_w2_shared);
        for l in 0..self.lora_u.len() {
            for k in 0..NUM_SHARED {
                add_vecs(&mut self.lora_u[l][k], &other.lora_u[l][k]);
                add_vecs(&mut self.lora_v[l][k], &other.lora_v[l][k]);
            }
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
        scale_vec(&mut self.qkv_w_shared, factor);
        scale_vec(&mut self.attn_proj_shared, factor);
        scale_vec(&mut self.ff_w1_shared, factor);
        scale_vec(&mut self.ff_w2_shared, factor);
        for l in 0..self.lora_u.len() {
            for k in 0..NUM_SHARED {
                scale_vec(&mut self.lora_u[l][k], factor);
                scale_vec(&mut self.lora_v[l][k], factor);
            }
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

impl LoraRGGPT {
    pub fn new(config: Config, rank: usize) -> Self {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let bs = config.block_size;
        let inner = 4 * e;
        let dims = matrix_dims(e);

        let emb_scale = 0.02;
        let layer_scale = (0.02 / (nl as f32).sqrt()).max(0.001);
        let lora_scale = 0.01 / (rank as f32).sqrt();

        let mut ln1_gamma = Vec::new();
        let mut ln1_beta_v = Vec::new();
        let mut ln2_gamma = Vec::new();
        let mut ln2_beta_v = Vec::new();
        let mut ff_b1 = Vec::new();
        let mut ff_b2 = Vec::new();
        let mut lora_u = Vec::new();
        let mut lora_v = Vec::new();

        for _ in 0..nl {
            ln1_gamma.push(vec![1.0; e]);
            ln1_beta_v.push(vec![0.0; e]);
            ln2_gamma.push(vec![1.0; e]);
            ln2_beta_v.push(vec![0.0; e]);
            ff_b1.push(vec![0.0; inner]);
            ff_b2.push(vec![0.0; e]);

            let mut u_layer = Vec::new();
            let mut v_layer = Vec::new();
            for &(d_in, d_out) in &dims {
                // Initialize U small, V zero (so initial W_eff = W_shared)
                u_layer.push(randn_vec(d_out * rank, lora_scale));
                v_layer.push(vec![0.0; d_in * rank]);
            }
            lora_u.push(u_layer);
            lora_v.push(v_layer);
        }

        let mut model = LoraRGGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            qkv_w_shared: randn_vec(e * 3 * e, layer_scale),
            attn_proj_shared: randn_vec(e * e, layer_scale),
            ff_w1_shared: randn_vec(e * inner, layer_scale),
            ff_w2_shared: randn_vec(inner * e, layer_scale * 0.5),
            lora_u,
            lora_v,
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
            lora_rank: rank,
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            adam_t: 0,
        };

        let sizes = model.param_sizes();
        model.adam_m = sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.adam_v = sizes.iter().map(|&s| vec![0.0; s]).collect();
        model
    }

    pub fn count_params(&self) -> usize {
        let e = self.config.n_embd;
        let v = self.config.vocab_size;
        let nl = self.config.n_layer;
        let inner = 4 * e;
        let bs = self.config.block_size;
        let r = self.lora_rank;
        let dims = matrix_dims(e);

        let emb = v * e + bs * e;
        let shared = e * 3 * e + e * e + e * inner + inner * e;
        // LoRA params: for each layer, each matrix: (d_out*r + d_in*r)
        let lora_per_layer: usize = dims.iter()
            .map(|&(d_in, d_out)| (d_in + d_out) * r)
            .sum();
        let lora_total = nl * lora_per_layer;
        let per_layer = nl * (e + e + e + e + inner + e); // LN + biases
        let head = e + e + e * v;

        emb + shared + lora_total + per_layer + head
    }

    fn param_sizes(&self) -> Vec<usize> {
        let nl = self.config.n_layer;
        let mut sizes = Vec::new();
        sizes.push(self.token_emb.len());
        sizes.push(self.pos_emb.len());
        sizes.push(self.qkv_w_shared.len());
        sizes.push(self.attn_proj_shared.len());
        sizes.push(self.ff_w1_shared.len());
        sizes.push(self.ff_w2_shared.len());
        for l in 0..nl {
            for k in 0..NUM_SHARED {
                sizes.push(self.lora_u[l][k].len());
                sizes.push(self.lora_v[l][k].len());
            }
        }
        for l in 0..nl {
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

    fn get_shared_weight(&self, idx: usize) -> &[f32] {
        match idx {
            0 => &self.qkv_w_shared,
            1 => &self.attn_proj_shared,
            2 => &self.ff_w1_shared,
            3 => &self.ff_w2_shared,
            _ => unreachable!(),
        }
    }

    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, LoraForwardCache) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = e / nh;
        let dims = matrix_dims(e);
        let r = self.lora_rank;

        let mut x = vec![0.0f32; t * e];
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                x[i * e + j] = self.token_emb[tok * e + j] + self.pos_emb[i * e + j];
            }
        }

        let mut cache = LoraForwardCache {
            tokens: tokens.to_vec(),
            x_after_emb: x.clone(),
            layer_caches: Vec::new(),
            x_before_final_ln: Vec::new(),
            x_after_final_ln: Vec::new(),
        };

        for l in 0..cfg.n_layer {
            let mut lc = LoraLayerCache::default();
            lc.x_input = x.clone();

            // Compute effective weights: W_eff = W_shared + U @ V^T
            let eff_qkv_w = compute_lora_effective(
                &self.qkv_w_shared, &self.lora_u[l][0], &self.lora_v[l][0],
                dims[0].0, dims[0].1, r,
            );
            let eff_attn_proj = compute_lora_effective(
                &self.attn_proj_shared, &self.lora_u[l][1], &self.lora_v[l][1],
                dims[1].0, dims[1].1, r,
            );
            let eff_ff_w1 = compute_lora_effective(
                &self.ff_w1_shared, &self.lora_u[l][2], &self.lora_v[l][2],
                dims[2].0, dims[2].1, r,
            );
            let eff_ff_w2 = compute_lora_effective(
                &self.ff_w2_shared, &self.lora_u[l][3], &self.lora_v[l][3],
                dims[3].0, dims[3].1, r,
            );

            // Layer norm 1
            let (ln1_out, _ln1_mean, _ln1_rstd) = layer_norm(
                &x, &self.ln1_gamma[l], &self.ln1_beta[l], t, e,
            );
            lc.ln1_out = ln1_out.clone();

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
                for i in 0..t {
                    let offset = h * t * t + i * t;
                    let max_val = all_attn_weights[offset..offset + t]
                        .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

            let proj_out = matmul(&attn_out, &eff_attn_proj, t, e, e);
            for i in 0..t * e { x[i] += proj_out[i]; }
            lc.x_after_attn_residual = x.clone();

            // Layer norm 2
            let (ln2_out, _ln2_mean, _ln2_rstd) = layer_norm(
                &x, &self.ln2_gamma[l], &self.ln2_beta[l], t, e,
            );
            lc.ln2_out = ln2_out.clone();

            // Feed-forward
            let inner = 4 * e;
            let mut ff_hidden = matmul(&ln2_out, &eff_ff_w1, t, e, inner);
            for i in 0..t {
                for j in 0..inner {
                    ff_hidden[i * inner + j] += self.ff_b1[l][j];
                }
            }
            lc.ff_pre_gelu = ff_hidden.clone();

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
            for i in 0..t * e { x[i] += ff_out[i]; }

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

    pub fn forward_backward(
        &self, tokens: &[usize], targets: &[usize],
    ) -> (f32, LoraRGGradients) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let r = self.lora_rank;
        let dims = matrix_dims(e);

        let (logits, cache) = self.forward_with_cache(tokens);

        // Cross-entropy loss
        let mut probs = vec![0.0f32; t * v];
        let mut loss = 0.0f32;
        for i in 0..t {
            let offset = i * v;
            let max_val = logits[offset..offset + v]
                .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v {
                probs[offset + j] = (logits[offset + j] - max_val).exp();
                sum += probs[offset + j];
            }
            for j in 0..v { probs[offset + j] /= sum; }
            loss -= probs[offset + targets[i]].max(1e-10).ln();
        }
        loss /= t as f32;

        let mut d_logits = probs;
        for i in 0..t {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v { d_logits[i * v + j] /= t as f32; }
        }

        let mut grads = LoraRGGradients::zero_like(cfg, r);

        // Backward through LM head
        let d_ln_out = matmul_backward_both(
            &cache.x_after_final_ln, &self.lm_head, &d_logits,
            t, e, v, &mut grads.lm_head,
        );

        // Backward through final layer norm
        let mut dx = layer_norm_backward(
            &cache.x_before_final_ln, &d_ln_out, &self.ln_f_gamma,
            t, e, &mut grads.ln_f_gamma, &mut grads.ln_f_beta,
        );

        // Backward through transformer blocks
        for l in (0..cfg.n_layer).rev() {
            let lc = &cache.layer_caches[l];
            let inner = 4 * e;

            // FF backward
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

            // LoRA backward for ff_w2 (matrix index 3)
            self.lora_backward(l, 3, &d_eff_ff_w2, &dims, &mut grads);

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

            // LoRA backward for ff_w1 (matrix index 2)
            self.lora_backward(l, 2, &d_eff_ff_w1, &dims, &mut grads);

            let d_from_ln2 = layer_norm_backward(
                &lc.x_after_attn_residual, &d_ln2_out, &self.ln2_gamma[l],
                t, e, &mut grads.ln2_gamma[l], &mut grads.ln2_beta[l],
            );

            let mut dx_mid = dx;
            for i in 0..t * e { dx_mid[i] += d_from_ln2[i]; }

            // Attention backward
            let d_proj_out = dx_mid.clone();

            let mut d_eff_attn_proj = vec![0.0f32; e * e];
            let d_attn_out = matmul_backward_both(
                &lc.attn_out_pre_proj, &lc.eff_attn_proj, &d_proj_out,
                t, e, e, &mut d_eff_attn_proj,
            );

            // LoRA backward for attn_proj (matrix index 1)
            self.lora_backward(l, 1, &d_eff_attn_proj, &dims, &mut grads);

            // Multi-head attention backward
            let nh = cfg.n_head;
            let hs = e / nh;
            let mut d_qkv = vec![0.0f32; t * 3 * e];

            for h in 0..nh {
                for i in 0..t {
                    for k in 0..hs {
                        let d_out = d_attn_out[i * e + h * hs + k];
                        for j in 0..t {
                            let w = lc.attn_weights[h * t * t + i * t + j];
                            d_qkv[j * 3 * e + 2 * e + h * hs + k] += w * d_out;
                        }
                    }
                }

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
                for val in d_attn_score.iter_mut() { *val *= scale; }

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

            // QKV backward
            let mut d_eff_qkv_w = vec![0.0f32; e * 3 * e];
            let d_ln1_out = matmul_backward_both(
                &lc.ln1_out, &lc.eff_qkv_w, &d_qkv,
                t, e, 3 * e, &mut d_eff_qkv_w,
            );

            // LoRA backward for qkv (matrix index 0)
            self.lora_backward(l, 0, &d_eff_qkv_w, &dims, &mut grads);

            let d_from_attn = layer_norm_backward(
                &lc.x_input, &d_ln1_out, &self.ln1_gamma[l],
                t, e, &mut grads.ln1_gamma[l], &mut grads.ln1_beta[l],
            );

            dx = dx_mid;
            for i in 0..t * e { dx[i] += d_from_attn[i]; }
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

    /// Backward pass for LoRA: given dL/dW_eff, compute dL/dW_shared, dL/dU, dL/dV
    fn lora_backward(
        &self, layer: usize, mat_idx: usize,
        d_eff: &[f32], dims: &[(usize, usize); NUM_SHARED],
        grads: &mut LoraRGGradients,
    ) {
        let (d_in, d_out) = dims[mat_idx];
        let r = self.lora_rank;

        // dL/dW_shared += dL/dW_eff (since W_eff = W_shared + U V^T)
        let grad_shared = match mat_idx {
            0 => &mut grads.qkv_w_shared,
            1 => &mut grads.attn_proj_shared,
            2 => &mut grads.ff_w1_shared,
            3 => &mut grads.ff_w2_shared,
            _ => unreachable!(),
        };
        for i in 0..d_eff.len() {
            grad_shared[i] += d_eff[i];
        }

        // dL/dU[j,k] = sum_i d_eff[i,j] * V[i,k]
        // dL/dV[i,k] = sum_j d_eff[i,j] * U[j,k]
        let u = &self.lora_u[layer][mat_idx];
        let v = &self.lora_v[layer][mat_idx];

        for i in 0..d_in {
            for k in 0..r {
                let mut grad_v_ik = 0.0f32;
                for j in 0..d_out {
                    grad_v_ik += d_eff[i * d_out + j] * u[j * r + k];
                }
                grads.lora_v[layer][mat_idx][i * r + k] += grad_v_ik;
            }
        }

        for j in 0..d_out {
            for k in 0..r {
                let mut grad_u_jk = 0.0f32;
                for i in 0..d_in {
                    grad_u_jk += d_eff[i * d_out + j] * v[i * r + k];
                }
                grads.lora_u[layer][mat_idx][j * r + k] += grad_u_jk;
            }
        }
    }

    pub fn apply_gradients(&mut self, grads: &LoraRGGradients, lr: f32) {
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
                adam_step($param, $grad, &mut self.adam_m[idx], &mut self.adam_v[idx],
                          lr, beta1, beta2, eps, bc1, bc2);
                idx += 1;
            }};
        }

        adam_update!(&mut self.token_emb, &grads.token_emb);
        adam_update!(&mut self.pos_emb, &grads.pos_emb);
        adam_update!(&mut self.qkv_w_shared, &grads.qkv_w_shared);
        adam_update!(&mut self.attn_proj_shared, &grads.attn_proj_shared);
        adam_update!(&mut self.ff_w1_shared, &grads.ff_w1_shared);
        adam_update!(&mut self.ff_w2_shared, &grads.ff_w2_shared);

        for l in 0..self.config.n_layer {
            for k in 0..NUM_SHARED {
                adam_update!(&mut self.lora_u[l][k], &grads.lora_u[l][k]);
                adam_update!(&mut self.lora_v[l][k], &grads.lora_v[l][k]);
            }
        }

        for l in 0..self.config.n_layer {
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

    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        self.forward_with_cache(tokens).0
    }

    pub fn generate(&self, start_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut tokens = start_tokens.to_vec();
        let mut rng = rand::thread_rng();
        let v = self.config.vocab_size;

        for _ in 0..max_new_tokens {
            let start = if tokens.len() > self.config.block_size {
                tokens.len() - self.config.block_size
            } else { 0 };
            let context = &tokens[start..];
            let logits = self.forward(context);
            let t = context.len();
            let last_offset = (t - 1) * v;
            let last_logits = &logits[last_offset..last_offset + v];

            let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let temp = 0.8f32;
            let mut probs = vec![0.0f32; v];
            let mut sum = 0.0f32;
            for j in 0..v {
                probs[j] = ((last_logits[j] - max_val) / temp).exp();
                sum += probs[j];
            }
            for j in 0..v { probs[j] /= sum; }

            let mut r_val: f32 = rng.r#gen();
            let mut next_token = 0;
            for j in 0..v {
                r_val -= probs[j];
                if r_val <= 0.0 { next_token = j; break; }
            }
            tokens.push(next_token);
        }
        tokens
    }

    /// Get LoRA norms for analysis
    pub fn get_lora_norms(&self) -> Vec<[f32; NUM_SHARED]> {
        let mut norms = Vec::new();
        for l in 0..self.config.n_layer {
            let mut layer_norms = [0.0f32; NUM_SHARED];
            for k in 0..NUM_SHARED {
                let u_norm: f32 = self.lora_u[l][k].iter().map(|x| x * x).sum::<f32>().sqrt();
                let v_norm: f32 = self.lora_v[l][k].iter().map(|x| x * x).sum::<f32>().sqrt();
                layer_norms[k] = u_norm * v_norm;
            }
            norms.push(layer_norms);
        }
        norms
    }

    pub fn save_weights(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"LRGT")?;
        f.write_all(&(self.config.vocab_size as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_embd as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_head as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_layer as u32).to_le_bytes())?;
        f.write_all(&(self.config.block_size as u32).to_le_bytes())?;
        f.write_all(&(self.lora_rank as u32).to_le_bytes())?;

        let mut all: Vec<f32> = Vec::new();
        all.extend_from_slice(&self.token_emb);
        all.extend_from_slice(&self.pos_emb);
        all.extend_from_slice(&self.qkv_w_shared);
        all.extend_from_slice(&self.attn_proj_shared);
        all.extend_from_slice(&self.ff_w1_shared);
        all.extend_from_slice(&self.ff_w2_shared);
        for l in 0..self.config.n_layer {
            for k in 0..NUM_SHARED {
                all.extend_from_slice(&self.lora_u[l][k]);
                all.extend_from_slice(&self.lora_v[l][k]);
            }
            all.extend_from_slice(&self.ln1_gamma[l]);
            all.extend_from_slice(&self.ln1_beta[l]);
            all.extend_from_slice(&self.ln2_gamma[l]);
            all.extend_from_slice(&self.ln2_beta[l]);
            all.extend_from_slice(&self.ff_b1[l]);
            all.extend_from_slice(&self.ff_b2[l]);
        }
        all.extend_from_slice(&self.ln_f_gamma);
        all.extend_from_slice(&self.ln_f_beta);
        all.extend_from_slice(&self.lm_head);

        f.write_all(&(all.len() as u64).to_le_bytes())?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(all.as_ptr() as *const u8, all.len() * 4)
        };
        f.write_all(bytes)?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════
// EXP8b: PreAmpRG — ff_w1 pre-amplified 1.2x, removed from alpha
// ═══════════════════════════════════════════════════════════════

/// PreAmpRGGPT: like RGGPT but ff_w1 is pre-amplified 1.2x and has no learnable alpha/beta.
/// Only 3 matrices (qkv, attn_proj, ff_w2) have alpha/beta per layer.
pub struct PreAmpRGGPT {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,

    pub qkv_w_shared: Vec<f32>,
    pub attn_proj_shared: Vec<f32>,
    pub ff_w1_shared: Vec<f32>,  // pre-amplified 1.2x
    pub ff_w2_shared: Vec<f32>,

    // Per-layer RG scale factors for 3 matrices only (no ff_w1)
    // Index: 0=qkv, 1=attn_proj, 2=ff_w2
    pub rg_alpha: Vec<[f32; 3]>,
    pub rg_beta: Vec<[f32; 3]>,

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
    adam_m: Vec<Vec<f32>>,
    adam_v: Vec<Vec<f32>>,
    adam_t: usize,
}

pub struct PreAmpRGGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub qkv_w_shared: Vec<f32>,
    pub attn_proj_shared: Vec<f32>,
    pub ff_w1_shared: Vec<f32>,
    pub ff_w2_shared: Vec<f32>,
    pub rg_alpha: Vec<[f32; 3]>,
    pub rg_beta: Vec<[f32; 3]>,
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

#[derive(Default)]
struct PreAmpLayerCache {
    x_input: Vec<f32>,
    eff_qkv_w: Vec<f32>,
    eff_attn_proj: Vec<f32>,
    eff_ff_w2: Vec<f32>,
    ln1_out: Vec<f32>,
    qkv: Vec<f32>,
    attn_weights: Vec<f32>,
    attn_out_pre_proj: Vec<f32>,
    x_after_attn_residual: Vec<f32>,
    ln2_out: Vec<f32>,
    ff_pre_gelu: Vec<f32>,
    ff_post_gelu: Vec<f32>,
}

struct PreAmpForwardCache {
    tokens: Vec<usize>,
    #[allow(dead_code)]
    x_after_emb: Vec<f32>,
    layer_caches: Vec<PreAmpLayerCache>,
    x_before_final_ln: Vec<f32>,
    x_after_final_ln: Vec<f32>,
}

impl PreAmpRGGradients {
    pub fn zero_like(cfg: &Config) -> Self {
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;
        let nl = cfg.n_layer;
        Self {
            token_emb: vec![0.0; v * e],
            pos_emb: vec![0.0; cfg.block_size * e],
            qkv_w_shared: vec![0.0; e * 3 * e],
            attn_proj_shared: vec![0.0; e * e],
            ff_w1_shared: vec![0.0; e * inner],
            ff_w2_shared: vec![0.0; inner * e],
            rg_alpha: vec![[0.0; 3]; nl],
            rg_beta: vec![[0.0; 3]; nl],
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

    pub fn accumulate(&mut self, other: &Self) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        add_vecs(&mut self.qkv_w_shared, &other.qkv_w_shared);
        add_vecs(&mut self.attn_proj_shared, &other.attn_proj_shared);
        add_vecs(&mut self.ff_w1_shared, &other.ff_w1_shared);
        add_vecs(&mut self.ff_w2_shared, &other.ff_w2_shared);
        for l in 0..self.rg_alpha.len() {
            for k in 0..3 {
                self.rg_alpha[l][k] += other.rg_alpha[l][k];
                self.rg_beta[l][k] += other.rg_beta[l][k];
            }
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
        scale_vec(&mut self.qkv_w_shared, factor);
        scale_vec(&mut self.attn_proj_shared, factor);
        scale_vec(&mut self.ff_w1_shared, factor);
        scale_vec(&mut self.ff_w2_shared, factor);
        for l in 0..self.rg_alpha.len() {
            for k in 0..3 {
                self.rg_alpha[l][k] *= factor;
                self.rg_beta[l][k] *= factor;
            }
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

impl PreAmpRGGPT {
    pub fn new(config: Config) -> Self {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let bs = config.block_size;
        let inner = 4 * e;

        let emb_scale = 0.02;
        let layer_scale = (0.02 / (nl as f32).sqrt()).max(0.001);

        let mut ln1_gamma = Vec::new();
        let mut ln1_beta_v = Vec::new();
        let mut ln2_gamma = Vec::new();
        let mut ln2_beta_v = Vec::new();
        let mut ff_b1 = Vec::new();
        let mut ff_b2 = Vec::new();

        for _ in 0..nl {
            ln1_gamma.push(vec![1.0; e]);
            ln1_beta_v.push(vec![0.0; e]);
            ln2_gamma.push(vec![1.0; e]);
            ln2_beta_v.push(vec![0.0; e]);
            ff_b1.push(vec![0.0; inner]);
            ff_b2.push(vec![0.0; e]);
        }

        // Pre-amplify ff_w1 by 1.2x
        let ff_w1_raw = randn_vec(e * inner, layer_scale);
        let ff_w1_amplified: Vec<f32> = ff_w1_raw.iter().map(|w| w * 1.2).collect();

        let mut model = PreAmpRGGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            qkv_w_shared: randn_vec(e * 3 * e, layer_scale),
            attn_proj_shared: randn_vec(e * e, layer_scale),
            ff_w1_shared: ff_w1_amplified,
            ff_w2_shared: randn_vec(inner * e, layer_scale * 0.5),
            rg_alpha: vec![[1.0f32; 3]; nl],
            rg_beta: vec![[0.0f32; 3]; nl],
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
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            adam_t: 0,
        };

        let sizes = model.param_sizes();
        model.adam_m = sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.adam_v = sizes.iter().map(|&s| vec![0.0; s]).collect();
        model
    }

    pub fn count_params(&self) -> usize {
        let e = self.config.n_embd;
        let v = self.config.vocab_size;
        let nl = self.config.n_layer;
        let inner = 4 * e;
        let bs = self.config.block_size;

        let emb = v * e + bs * e;
        let shared = e * 3 * e + e * e + e * inner + inner * e;
        let rg_scalars = nl * 3 * 2; // only 3 matrices have alpha/beta
        let per_layer = nl * (e + e + e + e + inner + e);
        let head = e + e + e * v;

        emb + shared + rg_scalars + per_layer + head
    }

    fn param_sizes(&self) -> Vec<usize> {
        let nl = self.config.n_layer;
        let mut sizes = Vec::new();
        sizes.push(self.token_emb.len());
        sizes.push(self.pos_emb.len());
        sizes.push(self.qkv_w_shared.len());
        sizes.push(self.attn_proj_shared.len());
        sizes.push(self.ff_w1_shared.len());
        sizes.push(self.ff_w2_shared.len());
        for _ in 0..nl {
            sizes.push(3); // alpha
            sizes.push(3); // beta
        }
        for l in 0..nl {
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

    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, PreAmpForwardCache) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = e / nh;

        let mut x = vec![0.0f32; t * e];
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                x[i * e + j] = self.token_emb[tok * e + j] + self.pos_emb[i * e + j];
            }
        }

        let mut cache = PreAmpForwardCache {
            tokens: tokens.to_vec(),
            x_after_emb: x.clone(),
            layer_caches: Vec::new(),
            x_before_final_ln: Vec::new(),
            x_after_final_ln: Vec::new(),
        };

        for l in 0..cfg.n_layer {
            let mut lc = PreAmpLayerCache::default();
            lc.x_input = x.clone();

            // Effective weights: alpha*W+beta for qkv(0), attn_proj(1), ff_w2(2)
            // ff_w1 uses shared directly (pre-amplified)
            let eff_qkv_w: Vec<f32> = self.qkv_w_shared.iter()
                .map(|&w| self.rg_alpha[l][0] * w + self.rg_beta[l][0]).collect();
            let eff_attn_proj: Vec<f32> = self.attn_proj_shared.iter()
                .map(|&w| self.rg_alpha[l][1] * w + self.rg_beta[l][1]).collect();
            let eff_ff_w2: Vec<f32> = self.ff_w2_shared.iter()
                .map(|&w| self.rg_alpha[l][2] * w + self.rg_beta[l][2]).collect();

            // LN1
            let (ln1_out, _, _) = layer_norm(&x, &self.ln1_gamma[l], &self.ln1_beta[l], t, e);
            lc.ln1_out = ln1_out.clone();

            // QKV
            let qkv = matmul(&ln1_out, &eff_qkv_w, t, e, 3 * e);
            lc.qkv = qkv.clone();

            // MHA
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
                                dot += qkv[i * 3 * e + h * hs + k] * qkv[j * 3 * e + e + h * hs + k];
                            }
                            all_attn_weights[h * t * t + i * t + j] = dot * scale;
                        }
                    }
                }
                for i in 0..t {
                    let offset = h * t * t + i * t;
                    let max_val = all_attn_weights[offset..offset + t]
                        .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for j in 0..t {
                        let v = (all_attn_weights[offset + j] - max_val).exp();
                        all_attn_weights[offset + j] = v;
                        sum += v;
                    }
                    for j in 0..t { all_attn_weights[offset + j] /= sum; }
                }
                for i in 0..t {
                    for k in 0..hs {
                        let mut sum = 0.0f32;
                        for j in 0..t {
                            sum += all_attn_weights[h * t * t + i * t + j]
                                * qkv[j * 3 * e + 2 * e + h * hs + k];
                        }
                        attn_out[i * e + h * hs + k] = sum;
                    }
                }
            }
            lc.attn_weights = all_attn_weights;
            lc.attn_out_pre_proj = attn_out.clone();

            let proj_out = matmul(&attn_out, &eff_attn_proj, t, e, e);
            for i in 0..t * e { x[i] += proj_out[i]; }
            lc.x_after_attn_residual = x.clone();

            // LN2
            let (ln2_out, _, _) = layer_norm(&x, &self.ln2_gamma[l], &self.ln2_beta[l], t, e);
            lc.ln2_out = ln2_out.clone();

            // FF with pre-amplified ff_w1 (no alpha/beta)
            let inner = 4 * e;
            let mut ff_hidden = matmul(&ln2_out, &self.ff_w1_shared, t, e, inner);
            for i in 0..t {
                for j in 0..inner { ff_hidden[i * inner + j] += self.ff_b1[l][j]; }
            }
            lc.ff_pre_gelu = ff_hidden.clone();

            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            for val in ff_hidden.iter_mut() {
                let x3 = *val * *val * *val;
                let inner_val = sqrt_2_over_pi * (*val + 0.044715 * x3);
                *val = 0.5 * *val * (1.0 + inner_val.tanh());
            }
            lc.ff_post_gelu = ff_hidden.clone();

            let mut ff_out = matmul(&ff_hidden, &eff_ff_w2, t, inner, e);
            for i in 0..t {
                for j in 0..e { ff_out[i * e + j] += self.ff_b2[l][j]; }
            }
            for i in 0..t * e { x[i] += ff_out[i]; }

            lc.eff_qkv_w = eff_qkv_w;
            lc.eff_attn_proj = eff_attn_proj;
            lc.eff_ff_w2 = eff_ff_w2;

            cache.layer_caches.push(lc);
        }

        cache.x_before_final_ln = x.clone();
        let (ln_out, _, _) = layer_norm(&x, &self.ln_f_gamma, &self.ln_f_beta, t, e);
        cache.x_after_final_ln = ln_out.clone();
        let logits = matmul(&ln_out, &self.lm_head, t, e, v);
        (logits, cache)
    }

    pub fn forward_backward(
        &self, tokens: &[usize], targets: &[usize],
    ) -> (f32, PreAmpRGGradients) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        let (logits, cache) = self.forward_with_cache(tokens);

        // Cross-entropy
        let mut probs = vec![0.0f32; t * v];
        let mut loss = 0.0f32;
        for i in 0..t {
            let offset = i * v;
            let max_val = logits[offset..offset + v]
                .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v {
                probs[offset + j] = (logits[offset + j] - max_val).exp();
                sum += probs[offset + j];
            }
            for j in 0..v { probs[offset + j] /= sum; }
            loss -= probs[offset + targets[i]].max(1e-10).ln();
        }
        loss /= t as f32;

        let mut d_logits = probs;
        for i in 0..t {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v { d_logits[i * v + j] /= t as f32; }
        }

        let mut grads = PreAmpRGGradients::zero_like(cfg);

        let d_ln_out = matmul_backward_both(
            &cache.x_after_final_ln, &self.lm_head, &d_logits,
            t, e, v, &mut grads.lm_head,
        );
        let mut dx = layer_norm_backward(
            &cache.x_before_final_ln, &d_ln_out, &self.ln_f_gamma,
            t, e, &mut grads.ln_f_gamma, &mut grads.ln_f_beta,
        );

        for l in (0..cfg.n_layer).rev() {
            let lc = &cache.layer_caches[l];
            let inner = 4 * e;

            // FF backward
            let d_ff_out = dx.clone();
            for i in 0..t {
                for j in 0..e { grads.ff_b2[l][j] += d_ff_out[i * e + j]; }
            }

            let mut d_eff_ff_w2 = vec![0.0f32; inner * e];
            let d_ff_hidden = matmul_backward_both(
                &lc.ff_post_gelu, &lc.eff_ff_w2, &d_ff_out,
                t, inner, e, &mut d_eff_ff_w2,
            );

            // RG backward for ff_w2 (index 2 in 3-element alpha)
            let alpha_2 = self.rg_alpha[l][2];
            for i in 0..d_eff_ff_w2.len() {
                grads.rg_alpha[l][2] += d_eff_ff_w2[i] * self.ff_w2_shared[i];
                grads.rg_beta[l][2] += d_eff_ff_w2[i];
                grads.ff_w2_shared[i] += alpha_2 * d_eff_ff_w2[i];
            }

            let d_ff_pre_gelu = gelu_backward(&lc.ff_pre_gelu, &d_ff_hidden);
            for i in 0..t {
                for j in 0..inner { grads.ff_b1[l][j] += d_ff_pre_gelu[i * inner + j]; }
            }

            // ff_w1 backward — direct (no alpha/beta)
            let mut d_ff_w1 = vec![0.0f32; e * inner];
            let d_ln2_out = matmul_backward_both(
                &lc.ln2_out, &self.ff_w1_shared, &d_ff_pre_gelu,
                t, e, inner, &mut d_ff_w1,
            );
            add_vecs(&mut grads.ff_w1_shared, &d_ff_w1);

            let d_from_ln2 = layer_norm_backward(
                &lc.x_after_attn_residual, &d_ln2_out, &self.ln2_gamma[l],
                t, e, &mut grads.ln2_gamma[l], &mut grads.ln2_beta[l],
            );

            let mut dx_mid = dx;
            for i in 0..t * e { dx_mid[i] += d_from_ln2[i]; }

            // Attn backward
            let d_proj_out = dx_mid.clone();
            let mut d_eff_attn_proj = vec![0.0f32; e * e];
            let d_attn_out = matmul_backward_both(
                &lc.attn_out_pre_proj, &lc.eff_attn_proj, &d_proj_out,
                t, e, e, &mut d_eff_attn_proj,
            );

            // RG backward for attn_proj (index 1)
            let alpha_1 = self.rg_alpha[l][1];
            for i in 0..d_eff_attn_proj.len() {
                grads.rg_alpha[l][1] += d_eff_attn_proj[i] * self.attn_proj_shared[i];
                grads.rg_beta[l][1] += d_eff_attn_proj[i];
                grads.attn_proj_shared[i] += alpha_1 * d_eff_attn_proj[i];
            }

            // MHA backward
            let nh = cfg.n_head;
            let hs = e / nh;
            let mut d_qkv = vec![0.0f32; t * 3 * e];
            for h in 0..nh {
                for i in 0..t {
                    for k in 0..hs {
                        let d_out = d_attn_out[i * e + h * hs + k];
                        for j in 0..t {
                            d_qkv[j * 3 * e + 2 * e + h * hs + k] +=
                                lc.attn_weights[h * t * t + i * t + j] * d_out;
                        }
                    }
                }
                let mut d_attn_score = vec![0.0f32; t * t];
                for i in 0..t {
                    for j in 0..t {
                        let mut dw = 0.0f32;
                        for k in 0..hs {
                            dw += lc.qkv[j * 3 * e + 2 * e + h * hs + k]
                                * d_attn_out[i * e + h * hs + k];
                        }
                        d_attn_score[i * t + j] = dw;
                    }
                }
                for i in 0..t {
                    let mut dot = 0.0f32;
                    for j in 0..t {
                        dot += d_attn_score[i * t + j] * lc.attn_weights[h * t * t + i * t + j];
                    }
                    for j in 0..t {
                        let w = lc.attn_weights[h * t * t + i * t + j];
                        d_attn_score[i * t + j] = w * (d_attn_score[i * t + j] - dot);
                    }
                }
                let scale = 1.0 / (hs as f32).sqrt();
                for val in d_attn_score.iter_mut() { *val *= scale; }
                for i in 0..t {
                    for j in 0..=i {
                        let ds = d_attn_score[i * t + j];
                        if ds.abs() > 1e-12 {
                            for k in 0..hs {
                                d_qkv[i * 3 * e + h * hs + k] += ds * lc.qkv[j * 3 * e + e + h * hs + k];
                                d_qkv[j * 3 * e + e + h * hs + k] += ds * lc.qkv[i * 3 * e + h * hs + k];
                            }
                        }
                    }
                }
            }

            let mut d_eff_qkv_w = vec![0.0f32; e * 3 * e];
            let d_ln1_out = matmul_backward_both(
                &lc.ln1_out, &lc.eff_qkv_w, &d_qkv,
                t, e, 3 * e, &mut d_eff_qkv_w,
            );

            // RG backward for qkv (index 0)
            let alpha_0 = self.rg_alpha[l][0];
            for i in 0..d_eff_qkv_w.len() {
                grads.rg_alpha[l][0] += d_eff_qkv_w[i] * self.qkv_w_shared[i];
                grads.rg_beta[l][0] += d_eff_qkv_w[i];
                grads.qkv_w_shared[i] += alpha_0 * d_eff_qkv_w[i];
            }

            let d_from_attn = layer_norm_backward(
                &lc.x_input, &d_ln1_out, &self.ln1_gamma[l],
                t, e, &mut grads.ln1_gamma[l], &mut grads.ln1_beta[l],
            );

            dx = dx_mid;
            for i in 0..t * e { dx[i] += d_from_attn[i]; }
        }

        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                grads.token_emb[tok * e + j] += dx[i * e + j];
                grads.pos_emb[i * e + j] += dx[i * e + j];
            }
        }

        (loss, grads)
    }

    pub fn apply_gradients(&mut self, grads: &PreAmpRGGradients, lr: f32) {
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
                adam_step($param, $grad, &mut self.adam_m[idx], &mut self.adam_v[idx],
                          lr, beta1, beta2, eps, bc1, bc2);
                idx += 1;
            }};
        }

        adam_update!(&mut self.token_emb, &grads.token_emb);
        adam_update!(&mut self.pos_emb, &grads.pos_emb);
        adam_update!(&mut self.qkv_w_shared, &grads.qkv_w_shared);
        adam_update!(&mut self.attn_proj_shared, &grads.attn_proj_shared);
        adam_update!(&mut self.ff_w1_shared, &grads.ff_w1_shared);
        adam_update!(&mut self.ff_w2_shared, &grads.ff_w2_shared);

        for l in 0..self.config.n_layer {
            let mut alpha_vec: Vec<f32> = self.rg_alpha[l].to_vec();
            let alpha_grad: Vec<f32> = grads.rg_alpha[l].to_vec();
            adam_step(&mut alpha_vec, &alpha_grad,
                      &mut self.adam_m[idx], &mut self.adam_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.rg_alpha[l].copy_from_slice(&alpha_vec);
            idx += 1;

            let mut beta_vec: Vec<f32> = self.rg_beta[l].to_vec();
            let beta_grad: Vec<f32> = grads.rg_beta[l].to_vec();
            adam_step(&mut beta_vec, &beta_grad,
                      &mut self.adam_m[idx], &mut self.adam_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.rg_beta[l].copy_from_slice(&beta_vec);
            idx += 1;
        }

        for l in 0..self.config.n_layer {
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

    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        self.forward_with_cache(tokens).0
    }

    pub fn generate(&self, start_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut tokens = start_tokens.to_vec();
        let mut rng = rand::thread_rng();
        let v = self.config.vocab_size;
        for _ in 0..max_new_tokens {
            let start = if tokens.len() > self.config.block_size {
                tokens.len() - self.config.block_size
            } else { 0 };
            let context = &tokens[start..];
            let logits = self.forward(context);
            let t = context.len();
            let last_offset = (t - 1) * v;
            let last_logits = &logits[last_offset..last_offset + v];
            let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let temp = 0.8f32;
            let mut probs = vec![0.0f32; v];
            let mut sum = 0.0f32;
            for j in 0..v { probs[j] = ((last_logits[j] - max_val) / temp).exp(); sum += probs[j]; }
            for j in 0..v { probs[j] /= sum; }
            let mut r_val: f32 = rng.r#gen();
            let mut next_token = 0;
            for j in 0..v { r_val -= probs[j]; if r_val <= 0.0 { next_token = j; break; } }
            tokens.push(next_token);
        }
        tokens
    }

    pub fn get_layer_alphas(&self) -> Vec<[f32; 3]> { self.rg_alpha.clone() }
}

// ═══════════════════════════════════════════════════════════════
// EXP8c: PerHeadRG — per-head alpha for attention weights
// ═══════════════════════════════════════════════════════════════

/// PerHeadRGGPT: like RGGPT but attention uses per-head alpha/beta
/// instead of one scalar for the whole QKV matrix.
/// FF matrices still use scalar alpha/beta as in standard RG.
pub struct PerHeadRGGPT {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,

    pub qkv_w_shared: Vec<f32>,
    pub attn_proj_shared: Vec<f32>,
    pub ff_w1_shared: Vec<f32>,
    pub ff_w2_shared: Vec<f32>,

    // Per-layer, per-head alpha/beta for QKV (n_head alphas per layer)
    pub qkv_head_alpha: Vec<Vec<f32>>,  // [layer][head]
    pub qkv_head_beta: Vec<Vec<f32>>,

    // Per-layer, per-head alpha/beta for attn_proj
    pub proj_head_alpha: Vec<Vec<f32>>,
    pub proj_head_beta: Vec<Vec<f32>>,

    // Per-layer scalar alpha/beta for FF (same as standard RG)
    pub ff_alpha: Vec<[f32; 2]>,  // 0=ff_w1, 1=ff_w2
    pub ff_beta: Vec<[f32; 2]>,

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
    adam_m: Vec<Vec<f32>>,
    adam_v: Vec<Vec<f32>>,
    adam_t: usize,
}

pub struct PerHeadRGGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub qkv_w_shared: Vec<f32>,
    pub attn_proj_shared: Vec<f32>,
    pub ff_w1_shared: Vec<f32>,
    pub ff_w2_shared: Vec<f32>,
    pub qkv_head_alpha: Vec<Vec<f32>>,
    pub qkv_head_beta: Vec<Vec<f32>>,
    pub proj_head_alpha: Vec<Vec<f32>>,
    pub proj_head_beta: Vec<Vec<f32>>,
    pub ff_alpha: Vec<[f32; 2]>,
    pub ff_beta: Vec<[f32; 2]>,
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

#[derive(Default)]
struct PerHeadLayerCache {
    x_input: Vec<f32>,
    eff_qkv_w: Vec<f32>,
    eff_attn_proj: Vec<f32>,
    eff_ff_w1: Vec<f32>,
    eff_ff_w2: Vec<f32>,
    ln1_out: Vec<f32>,
    qkv: Vec<f32>,
    attn_weights: Vec<f32>,
    attn_out_pre_proj: Vec<f32>,
    x_after_attn_residual: Vec<f32>,
    ln2_out: Vec<f32>,
    ff_pre_gelu: Vec<f32>,
    ff_post_gelu: Vec<f32>,
}

struct PerHeadForwardCache {
    tokens: Vec<usize>,
    #[allow(dead_code)]
    x_after_emb: Vec<f32>,
    layer_caches: Vec<PerHeadLayerCache>,
    x_before_final_ln: Vec<f32>,
    x_after_final_ln: Vec<f32>,
}

impl PerHeadRGGradients {
    pub fn zero_like(cfg: &Config) -> Self {
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;
        let nl = cfg.n_layer;
        let nh = cfg.n_head;
        Self {
            token_emb: vec![0.0; v * e],
            pos_emb: vec![0.0; cfg.block_size * e],
            qkv_w_shared: vec![0.0; e * 3 * e],
            attn_proj_shared: vec![0.0; e * e],
            ff_w1_shared: vec![0.0; e * inner],
            ff_w2_shared: vec![0.0; inner * e],
            qkv_head_alpha: vec![vec![0.0; nh]; nl],
            qkv_head_beta: vec![vec![0.0; nh]; nl],
            proj_head_alpha: vec![vec![0.0; nh]; nl],
            proj_head_beta: vec![vec![0.0; nh]; nl],
            ff_alpha: vec![[0.0; 2]; nl],
            ff_beta: vec![[0.0; 2]; nl],
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

    pub fn accumulate(&mut self, other: &Self) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        add_vecs(&mut self.qkv_w_shared, &other.qkv_w_shared);
        add_vecs(&mut self.attn_proj_shared, &other.attn_proj_shared);
        add_vecs(&mut self.ff_w1_shared, &other.ff_w1_shared);
        add_vecs(&mut self.ff_w2_shared, &other.ff_w2_shared);
        for l in 0..self.qkv_head_alpha.len() {
            add_vecs(&mut self.qkv_head_alpha[l], &other.qkv_head_alpha[l]);
            add_vecs(&mut self.qkv_head_beta[l], &other.qkv_head_beta[l]);
            add_vecs(&mut self.proj_head_alpha[l], &other.proj_head_alpha[l]);
            add_vecs(&mut self.proj_head_beta[l], &other.proj_head_beta[l]);
            for k in 0..2 {
                self.ff_alpha[l][k] += other.ff_alpha[l][k];
                self.ff_beta[l][k] += other.ff_beta[l][k];
            }
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
        scale_vec(&mut self.qkv_w_shared, factor);
        scale_vec(&mut self.attn_proj_shared, factor);
        scale_vec(&mut self.ff_w1_shared, factor);
        scale_vec(&mut self.ff_w2_shared, factor);
        for l in 0..self.qkv_head_alpha.len() {
            scale_vec(&mut self.qkv_head_alpha[l], factor);
            scale_vec(&mut self.qkv_head_beta[l], factor);
            scale_vec(&mut self.proj_head_alpha[l], factor);
            scale_vec(&mut self.proj_head_beta[l], factor);
            for k in 0..2 {
                self.ff_alpha[l][k] *= factor;
                self.ff_beta[l][k] *= factor;
            }
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

impl PerHeadRGGPT {
    pub fn new(config: Config) -> Self {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let bs = config.block_size;
        let inner = 4 * e;
        let nh = config.n_head;

        let emb_scale = 0.02;
        let layer_scale = (0.02 / (nl as f32).sqrt()).max(0.001);

        let mut ln1_gamma = Vec::new();
        let mut ln1_beta_v = Vec::new();
        let mut ln2_gamma = Vec::new();
        let mut ln2_beta_v = Vec::new();
        let mut ff_b1 = Vec::new();
        let mut ff_b2 = Vec::new();

        for _ in 0..nl {
            ln1_gamma.push(vec![1.0; e]);
            ln1_beta_v.push(vec![0.0; e]);
            ln2_gamma.push(vec![1.0; e]);
            ln2_beta_v.push(vec![0.0; e]);
            ff_b1.push(vec![0.0; inner]);
            ff_b2.push(vec![0.0; e]);
        }

        let mut model = PerHeadRGGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            qkv_w_shared: randn_vec(e * 3 * e, layer_scale),
            attn_proj_shared: randn_vec(e * e, layer_scale),
            ff_w1_shared: randn_vec(e * inner, layer_scale),
            ff_w2_shared: randn_vec(inner * e, layer_scale * 0.5),
            qkv_head_alpha: vec![vec![1.0; nh]; nl],
            qkv_head_beta: vec![vec![0.0; nh]; nl],
            proj_head_alpha: vec![vec![1.0; nh]; nl],
            proj_head_beta: vec![vec![0.0; nh]; nl],
            ff_alpha: vec![[1.0; 2]; nl],
            ff_beta: vec![[0.0; 2]; nl],
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
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            adam_t: 0,
        };

        let sizes = model.param_sizes();
        model.adam_m = sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.adam_v = sizes.iter().map(|&s| vec![0.0; s]).collect();
        model
    }

    pub fn count_params(&self) -> usize {
        let e = self.config.n_embd;
        let v = self.config.vocab_size;
        let nl = self.config.n_layer;
        let inner = 4 * e;
        let bs = self.config.block_size;
        let nh = self.config.n_head;

        let emb = v * e + bs * e;
        let shared = e * 3 * e + e * e + e * inner + inner * e;
        // Per-head: 2*nh for qkv + 2*nh for proj = 4*nh per layer
        // Plus 2*2=4 for FF alpha/beta per layer
        let per_layer_scalars = nl * (4 * nh + 4);
        let per_layer_ln = nl * (e + e + e + e + inner + e);
        let head = e + e + e * v;

        emb + shared + per_layer_scalars + per_layer_ln + head
    }

    fn param_sizes(&self) -> Vec<usize> {
        let nl = self.config.n_layer;
        let nh = self.config.n_head;
        let mut sizes = Vec::new();
        sizes.push(self.token_emb.len());
        sizes.push(self.pos_emb.len());
        sizes.push(self.qkv_w_shared.len());
        sizes.push(self.attn_proj_shared.len());
        sizes.push(self.ff_w1_shared.len());
        sizes.push(self.ff_w2_shared.len());
        for _ in 0..nl {
            sizes.push(nh); // qkv_head_alpha
            sizes.push(nh); // qkv_head_beta
            sizes.push(nh); // proj_head_alpha
            sizes.push(nh); // proj_head_beta
            sizes.push(2);  // ff_alpha
            sizes.push(2);  // ff_beta
        }
        for l in 0..nl {
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

    /// Build effective QKV weight with per-head scaling.
    /// QKV shared is (e x 3e). For each head h, scale the columns belonging to that head.
    fn build_eff_qkv(&self, layer: usize) -> Vec<f32> {
        let e = self.config.n_embd;
        let nh = self.config.n_head;
        let hs = e / nh;
        let mut eff = self.qkv_w_shared.clone();
        // QKV layout: for row i, cols [0..e] = Q, [e..2e] = K, [2e..3e] = V
        // Each section is divided into nh heads of size hs
        for i in 0..e {
            for h in 0..nh {
                let alpha = self.qkv_head_alpha[layer][h];
                let beta = self.qkv_head_beta[layer][h];
                for section in 0..3 {
                    for k in 0..hs {
                        let col = section * e + h * hs + k;
                        let idx = i * 3 * e + col;
                        eff[idx] = alpha * eff[idx] + beta;
                    }
                }
            }
        }
        eff
    }

    /// Build effective attn_proj weight with per-head scaling.
    /// attn_proj is (e x e). Input rows are arranged by head.
    fn build_eff_attn_proj(&self, layer: usize) -> Vec<f32> {
        let e = self.config.n_embd;
        let nh = self.config.n_head;
        let hs = e / nh;
        let mut eff = self.attn_proj_shared.clone();
        // Input dimension is e, arranged as nh heads of hs
        // Scale the rows belonging to each head
        for h in 0..nh {
            let alpha = self.proj_head_alpha[layer][h];
            let beta = self.proj_head_beta[layer][h];
            for row in (h * hs)..((h + 1) * hs) {
                for col in 0..e {
                    let idx = row * e + col;
                    eff[idx] = alpha * eff[idx] + beta;
                }
            }
        }
        eff
    }

    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, PerHeadForwardCache) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = e / nh;

        let mut x = vec![0.0f32; t * e];
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                x[i * e + j] = self.token_emb[tok * e + j] + self.pos_emb[i * e + j];
            }
        }

        let mut cache = PerHeadForwardCache {
            tokens: tokens.to_vec(),
            x_after_emb: x.clone(),
            layer_caches: Vec::new(),
            x_before_final_ln: Vec::new(),
            x_after_final_ln: Vec::new(),
        };

        for l in 0..cfg.n_layer {
            let mut lc = PerHeadLayerCache::default();
            lc.x_input = x.clone();

            let eff_qkv_w = self.build_eff_qkv(l);
            let eff_attn_proj = self.build_eff_attn_proj(l);
            let eff_ff_w1: Vec<f32> = self.ff_w1_shared.iter()
                .map(|&w| self.ff_alpha[l][0] * w + self.ff_beta[l][0]).collect();
            let eff_ff_w2: Vec<f32> = self.ff_w2_shared.iter()
                .map(|&w| self.ff_alpha[l][1] * w + self.ff_beta[l][1]).collect();

            let (ln1_out, _, _) = layer_norm(&x, &self.ln1_gamma[l], &self.ln1_beta[l], t, e);
            lc.ln1_out = ln1_out.clone();

            let qkv = matmul(&ln1_out, &eff_qkv_w, t, e, 3 * e);
            lc.qkv = qkv.clone();

            // MHA
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
                                dot += qkv[i * 3 * e + h * hs + k] * qkv[j * 3 * e + e + h * hs + k];
                            }
                            all_attn_weights[h * t * t + i * t + j] = dot * scale;
                        }
                    }
                }
                for i in 0..t {
                    let offset = h * t * t + i * t;
                    let max_val = all_attn_weights[offset..offset + t]
                        .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for j in 0..t {
                        let ev = (all_attn_weights[offset + j] - max_val).exp();
                        all_attn_weights[offset + j] = ev;
                        sum += ev;
                    }
                    for j in 0..t { all_attn_weights[offset + j] /= sum; }
                }
                for i in 0..t {
                    for k in 0..hs {
                        let mut sum = 0.0f32;
                        for j in 0..t {
                            sum += all_attn_weights[h * t * t + i * t + j]
                                * qkv[j * 3 * e + 2 * e + h * hs + k];
                        }
                        attn_out[i * e + h * hs + k] = sum;
                    }
                }
            }
            lc.attn_weights = all_attn_weights;
            lc.attn_out_pre_proj = attn_out.clone();

            let proj_out = matmul(&attn_out, &eff_attn_proj, t, e, e);
            for i in 0..t * e { x[i] += proj_out[i]; }
            lc.x_after_attn_residual = x.clone();

            let (ln2_out, _, _) = layer_norm(&x, &self.ln2_gamma[l], &self.ln2_beta[l], t, e);
            lc.ln2_out = ln2_out.clone();

            let inner = 4 * e;
            let mut ff_hidden = matmul(&ln2_out, &eff_ff_w1, t, e, inner);
            for i in 0..t {
                for j in 0..inner { ff_hidden[i * inner + j] += self.ff_b1[l][j]; }
            }
            lc.ff_pre_gelu = ff_hidden.clone();

            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            for val in ff_hidden.iter_mut() {
                let x3 = *val * *val * *val;
                let inner_val = sqrt_2_over_pi * (*val + 0.044715 * x3);
                *val = 0.5 * *val * (1.0 + inner_val.tanh());
            }
            lc.ff_post_gelu = ff_hidden.clone();

            let mut ff_out = matmul(&ff_hidden, &eff_ff_w2, t, inner, e);
            for i in 0..t {
                for j in 0..e { ff_out[i * e + j] += self.ff_b2[l][j]; }
            }
            for i in 0..t * e { x[i] += ff_out[i]; }

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

    pub fn forward_backward(
        &self, tokens: &[usize], targets: &[usize],
    ) -> (f32, PerHeadRGGradients) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = e / nh;

        let (logits, cache) = self.forward_with_cache(tokens);

        // Cross-entropy
        let mut probs = vec![0.0f32; t * v];
        let mut loss = 0.0f32;
        for i in 0..t {
            let offset = i * v;
            let max_val = logits[offset..offset + v]
                .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v { probs[offset + j] = (logits[offset + j] - max_val).exp(); sum += probs[offset + j]; }
            for j in 0..v { probs[offset + j] /= sum; }
            loss -= probs[offset + targets[i]].max(1e-10).ln();
        }
        loss /= t as f32;

        let mut d_logits = probs;
        for i in 0..t {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v { d_logits[i * v + j] /= t as f32; }
        }

        let mut grads = PerHeadRGGradients::zero_like(cfg);

        let d_ln_out = matmul_backward_both(
            &cache.x_after_final_ln, &self.lm_head, &d_logits,
            t, e, v, &mut grads.lm_head,
        );
        let mut dx = layer_norm_backward(
            &cache.x_before_final_ln, &d_ln_out, &self.ln_f_gamma,
            t, e, &mut grads.ln_f_gamma, &mut grads.ln_f_beta,
        );

        for l in (0..cfg.n_layer).rev() {
            let lc = &cache.layer_caches[l];
            let inner = 4 * e;

            // FF backward
            let d_ff_out = dx.clone();
            for i in 0..t { for j in 0..e { grads.ff_b2[l][j] += d_ff_out[i * e + j]; } }

            let mut d_eff_ff_w2 = vec![0.0f32; inner * e];
            let d_ff_hidden = matmul_backward_both(
                &lc.ff_post_gelu, &lc.eff_ff_w2, &d_ff_out,
                t, inner, e, &mut d_eff_ff_w2,
            );

            // RG backward for ff_w2 (index 1 in ff_alpha)
            let alpha_ff2 = self.ff_alpha[l][1];
            for i in 0..d_eff_ff_w2.len() {
                grads.ff_alpha[l][1] += d_eff_ff_w2[i] * self.ff_w2_shared[i];
                grads.ff_beta[l][1] += d_eff_ff_w2[i];
                grads.ff_w2_shared[i] += alpha_ff2 * d_eff_ff_w2[i];
            }

            let d_ff_pre_gelu = gelu_backward(&lc.ff_pre_gelu, &d_ff_hidden);
            for i in 0..t { for j in 0..inner { grads.ff_b1[l][j] += d_ff_pre_gelu[i * inner + j]; } }

            let mut d_eff_ff_w1 = vec![0.0f32; e * inner];
            let d_ln2_out = matmul_backward_both(
                &lc.ln2_out, &lc.eff_ff_w1, &d_ff_pre_gelu,
                t, e, inner, &mut d_eff_ff_w1,
            );

            // RG backward for ff_w1 (index 0 in ff_alpha)
            let alpha_ff1 = self.ff_alpha[l][0];
            for i in 0..d_eff_ff_w1.len() {
                grads.ff_alpha[l][0] += d_eff_ff_w1[i] * self.ff_w1_shared[i];
                grads.ff_beta[l][0] += d_eff_ff_w1[i];
                grads.ff_w1_shared[i] += alpha_ff1 * d_eff_ff_w1[i];
            }

            let d_from_ln2 = layer_norm_backward(
                &lc.x_after_attn_residual, &d_ln2_out, &self.ln2_gamma[l],
                t, e, &mut grads.ln2_gamma[l], &mut grads.ln2_beta[l],
            );

            let mut dx_mid = dx;
            for i in 0..t * e { dx_mid[i] += d_from_ln2[i]; }

            // Attn backward
            let d_proj_out = dx_mid.clone();
            let mut d_eff_attn_proj = vec![0.0f32; e * e];
            let d_attn_out = matmul_backward_both(
                &lc.attn_out_pre_proj, &lc.eff_attn_proj, &d_proj_out,
                t, e, e, &mut d_eff_attn_proj,
            );

            // Per-head backward for attn_proj
            // d_eff_attn_proj is (e x e). Rows [h*hs..(h+1)*hs] belong to head h.
            for h in 0..nh {
                let alpha = self.proj_head_alpha[l][h];
                for row in (h * hs)..((h + 1) * hs) {
                    for col in 0..e {
                        let idx = row * e + col;
                        grads.proj_head_alpha[l][h] += d_eff_attn_proj[idx] * self.attn_proj_shared[idx];
                        grads.proj_head_beta[l][h] += d_eff_attn_proj[idx];
                        grads.attn_proj_shared[idx] += alpha * d_eff_attn_proj[idx];
                    }
                }
            }

            // MHA backward
            let mut d_qkv = vec![0.0f32; t * 3 * e];
            for h in 0..nh {
                for i in 0..t {
                    for k in 0..hs {
                        let d_out = d_attn_out[i * e + h * hs + k];
                        for j in 0..t {
                            d_qkv[j * 3 * e + 2 * e + h * hs + k] +=
                                lc.attn_weights[h * t * t + i * t + j] * d_out;
                        }
                    }
                }
                let mut d_attn_score = vec![0.0f32; t * t];
                for i in 0..t {
                    for j in 0..t {
                        let mut dw = 0.0f32;
                        for k in 0..hs {
                            dw += lc.qkv[j * 3 * e + 2 * e + h * hs + k]
                                * d_attn_out[i * e + h * hs + k];
                        }
                        d_attn_score[i * t + j] = dw;
                    }
                }
                for i in 0..t {
                    let mut dot = 0.0f32;
                    for j in 0..t {
                        dot += d_attn_score[i * t + j] * lc.attn_weights[h * t * t + i * t + j];
                    }
                    for j in 0..t {
                        let w = lc.attn_weights[h * t * t + i * t + j];
                        d_attn_score[i * t + j] = w * (d_attn_score[i * t + j] - dot);
                    }
                }
                let scale = 1.0 / (hs as f32).sqrt();
                for val in d_attn_score.iter_mut() { *val *= scale; }
                for i in 0..t {
                    for j in 0..=i {
                        let ds = d_attn_score[i * t + j];
                        if ds.abs() > 1e-12 {
                            for k in 0..hs {
                                d_qkv[i * 3 * e + h * hs + k] += ds * lc.qkv[j * 3 * e + e + h * hs + k];
                                d_qkv[j * 3 * e + e + h * hs + k] += ds * lc.qkv[i * 3 * e + h * hs + k];
                            }
                        }
                    }
                }
            }

            // QKV backward
            let mut d_eff_qkv_w = vec![0.0f32; e * 3 * e];
            let d_ln1_out = matmul_backward_both(
                &lc.ln1_out, &lc.eff_qkv_w, &d_qkv,
                t, e, 3 * e, &mut d_eff_qkv_w,
            );

            // Per-head backward for QKV
            for h in 0..nh {
                let alpha = self.qkv_head_alpha[l][h];
                for i in 0..e {
                    for section in 0..3 {
                        for k in 0..hs {
                            let col = section * e + h * hs + k;
                            let idx = i * 3 * e + col;
                            grads.qkv_head_alpha[l][h] += d_eff_qkv_w[idx] * self.qkv_w_shared[idx];
                            grads.qkv_head_beta[l][h] += d_eff_qkv_w[idx];
                            grads.qkv_w_shared[idx] += alpha * d_eff_qkv_w[idx];
                        }
                    }
                }
            }

            let d_from_attn = layer_norm_backward(
                &lc.x_input, &d_ln1_out, &self.ln1_gamma[l],
                t, e, &mut grads.ln1_gamma[l], &mut grads.ln1_beta[l],
            );

            dx = dx_mid;
            for i in 0..t * e { dx[i] += d_from_attn[i]; }
        }

        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                grads.token_emb[tok * e + j] += dx[i * e + j];
                grads.pos_emb[i * e + j] += dx[i * e + j];
            }
        }

        (loss, grads)
    }

    pub fn apply_gradients(&mut self, grads: &PerHeadRGGradients, lr: f32) {
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
                adam_step($param, $grad, &mut self.adam_m[idx], &mut self.adam_v[idx],
                          lr, beta1, beta2, eps, bc1, bc2);
                idx += 1;
            }};
        }

        adam_update!(&mut self.token_emb, &grads.token_emb);
        adam_update!(&mut self.pos_emb, &grads.pos_emb);
        adam_update!(&mut self.qkv_w_shared, &grads.qkv_w_shared);
        adam_update!(&mut self.attn_proj_shared, &grads.attn_proj_shared);
        adam_update!(&mut self.ff_w1_shared, &grads.ff_w1_shared);
        adam_update!(&mut self.ff_w2_shared, &grads.ff_w2_shared);

        for l in 0..self.config.n_layer {
            adam_update!(&mut self.qkv_head_alpha[l], &grads.qkv_head_alpha[l]);
            adam_update!(&mut self.qkv_head_beta[l], &grads.qkv_head_beta[l]);
            adam_update!(&mut self.proj_head_alpha[l], &grads.proj_head_alpha[l]);
            adam_update!(&mut self.proj_head_beta[l], &grads.proj_head_beta[l]);

            let mut ff_a: Vec<f32> = self.ff_alpha[l].to_vec();
            let ff_a_g: Vec<f32> = grads.ff_alpha[l].to_vec();
            adam_step(&mut ff_a, &ff_a_g,
                      &mut self.adam_m[idx], &mut self.adam_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.ff_alpha[l].copy_from_slice(&ff_a);
            idx += 1;

            let mut ff_b: Vec<f32> = self.ff_beta[l].to_vec();
            let ff_b_g: Vec<f32> = grads.ff_beta[l].to_vec();
            adam_step(&mut ff_b, &ff_b_g,
                      &mut self.adam_m[idx], &mut self.adam_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.ff_beta[l].copy_from_slice(&ff_b);
            idx += 1;
        }

        for l in 0..self.config.n_layer {
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

    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        self.forward_with_cache(tokens).0
    }

    pub fn generate(&self, start_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut tokens = start_tokens.to_vec();
        let mut rng = rand::thread_rng();
        let v = self.config.vocab_size;
        for _ in 0..max_new_tokens {
            let start = if tokens.len() > self.config.block_size {
                tokens.len() - self.config.block_size
            } else { 0 };
            let context = &tokens[start..];
            let logits = self.forward(context);
            let t = context.len();
            let last_offset = (t - 1) * v;
            let last_logits = &logits[last_offset..last_offset + v];
            let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let temp = 0.8f32;
            let mut probs = vec![0.0f32; v];
            let mut sum = 0.0f32;
            for j in 0..v { probs[j] = ((last_logits[j] - max_val) / temp).exp(); sum += probs[j]; }
            for j in 0..v { probs[j] /= sum; }
            let mut r_val: f32 = rng.r#gen();
            let mut next_token = 0;
            for j in 0..v { r_val -= probs[j]; if r_val <= 0.0 { next_token = j; break; } }
            tokens.push(next_token);
        }
        tokens
    }

    pub fn get_head_alphas(&self) -> (&Vec<Vec<f32>>, &Vec<Vec<f32>>) {
        (&self.qkv_head_alpha, &self.proj_head_alpha)
    }
}
