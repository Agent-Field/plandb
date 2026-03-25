use rand::Rng;
use rayon::prelude::*;
use std::io::{Read, Write};

/// Configuration for the mini GPT model
#[derive(Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block_size: usize,
}

/// A complete mini GPT with manual backpropagation.
/// All operations store intermediates needed for backward pass.
pub struct GPT {
    // Embeddings
    pub token_emb: Vec<f32>,  // (vocab_size, n_embd)
    pub pos_emb: Vec<f32>,    // (block_size, n_embd)

    // Per-layer parameters
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

    // Final layer norm + head
    pub ln_f_gamma: Vec<f32>,  // (n_embd,)
    pub ln_f_beta: Vec<f32>,   // (n_embd,)
    pub lm_head: Vec<f32>,     // (n_embd, vocab_size)

    pub config: Config,

    // Adam optimizer state
    pub m: Vec<Vec<f32>>,  // first moment
    pub v: Vec<Vec<f32>>,  // second moment
    pub t: usize,          // timestep
}

pub fn randn_vec(n: usize, scale: f32) -> Vec<f32> {
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

impl GPT {
    pub fn new(config: Config) -> Self {
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

        let mut model = GPT {
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
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        };

        // Initialize Adam state
        let param_sizes = model.param_sizes();
        model.m = param_sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.v = param_sizes.iter().map(|&s| vec![0.0; s]).collect();

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

    /// Forward pass returning logits (T, vocab_size) as flat Vec
    /// Also returns all intermediates needed for backward pass
    pub fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, ForwardCache) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = e / nh;

        // Embedding lookup
        let mut x = vec![0.0f32; t * e]; // (T, E)
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                x[i * e + j] = self.token_emb[tok * e + j] + self.pos_emb[i * e + j];
            }
        }

        let mut cache = ForwardCache {
            tokens: tokens.to_vec(),
            x_after_emb: x.clone(),
            layer_caches: Vec::new(),
            x_before_final_ln: Vec::new(),
            x_after_final_ln: Vec::new(),
        };

        // Transformer blocks
        for l in 0..cfg.n_layer {
            let mut lc = LayerCache::default();
            lc.x_input = x.clone();

            // Layer norm 1
            let (ln1_out, ln1_mean, ln1_rstd) = layer_norm(&x, &self.ln1_gamma[l], &self.ln1_beta[l], t, e);
            lc.ln1_out = ln1_out.clone();
            lc.ln1_mean = ln1_mean;
            lc.ln1_rstd = ln1_rstd;

            // QKV projection: (T, E) @ (E, 3E) -> (T, 3E)
            let qkv = matmul(&ln1_out, &self.qkv_w[l], t, e, 3 * e);
            lc.qkv = qkv.clone();

            // Split into Q, K, V and compute multi-head attention
            let mut attn_out = vec![0.0f32; t * e];
            let mut all_attn_weights = vec![0.0f32; nh * t * t];

            for h in 0..nh {
                // Extract Q, K, V for this head
                for i in 0..t {
                    for j in 0..hs {
                        // Q is at offset 0, K at E, V at 2E
                        let q_idx = i * 3 * e + h * hs + j;
                        let k_idx = i * 3 * e + e + h * hs + j;
                        let v_idx = i * 3 * e + 2 * e + h * hs + j;
                        let _ = (qkv[q_idx], qkv[k_idx], qkv[v_idx]);
                    }
                }

                // Compute attention: Q @ K^T / sqrt(hs)
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

            // Output projection: (T, E) @ (E, E) -> (T, E)
            let proj_out = matmul(&attn_out, &self.attn_proj[l], t, e, e);

            // Residual connection
            for i in 0..t * e {
                x[i] += proj_out[i];
            }
            lc.x_after_attn_residual = x.clone();

            // Layer norm 2
            let (ln2_out, ln2_mean, ln2_rstd) = layer_norm(&x, &self.ln2_gamma[l], &self.ln2_beta[l], t, e);
            lc.ln2_out = ln2_out.clone();
            lc.ln2_mean = ln2_mean;
            lc.ln2_rstd = ln2_rstd;

            // Feed-forward: (T, E) @ (E, 4E) + b1 -> GELU -> (T, 4E) @ (4E, E) + b2
            let inner = 4 * e;
            let mut ff_hidden = matmul(&ln2_out, &self.ff_w1[l], t, e, inner);
            // Add bias
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

            let mut ff_out = matmul(&ff_hidden, &self.ff_w2[l], t, inner, e);
            for i in 0..t {
                for j in 0..e {
                    ff_out[i * e + j] += self.ff_b2[l][j];
                }
            }

            // Residual connection
            for i in 0..t * e {
                x[i] += ff_out[i];
            }

            cache.layer_caches.push(lc);
        }

        cache.x_before_final_ln = x.clone();

        // Final layer norm
        let (ln_out, _, _) = layer_norm(&x, &self.ln_f_gamma, &self.ln_f_beta, t, e);
        cache.x_after_final_ln = ln_out.clone();

        // LM head: (T, E) @ (E, V) -> (T, V)
        let logits = matmul(&ln_out, &self.lm_head, t, e, v);

        (logits, cache)
    }

    /// Forward pass + cross entropy loss + backward pass
    /// Returns loss and updates gradients
    pub fn forward_backward(
        &self,
        tokens: &[usize],
        targets: &[usize],
    ) -> (f32, Gradients) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        let (logits, cache) = self.forward_with_cache(tokens);

        // Softmax + cross-entropy loss
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

        // Backward pass: dL/d_logits = probs - one_hot(targets), scaled by 1/T
        let mut d_logits = probs; // start from probs
        for i in 0..t {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v {
                d_logits[i * v + j] /= t as f32;
            }
        }

        let mut grads = Gradients::new(cfg);

        // dL/d_lm_head: ln_out^T @ d_logits -> (E, V)
        // dL/d_ln_out: d_logits @ lm_head^T -> (T, E)
        let d_ln_out = matmul_backward_both(
            &cache.x_after_final_ln,
            &self.lm_head,
            &d_logits,
            t,
            e,
            v,
            &mut grads.lm_head,
        );

        // Backward through final layer norm
        let mut dx = layer_norm_backward(
            &cache.x_before_final_ln,
            &d_ln_out,
            &self.ln_f_gamma,
            t,
            e,
            &mut grads.ln_f_gamma,
            &mut grads.ln_f_beta,
        );

        // Backward through transformer blocks (reverse order)
        for l in (0..cfg.n_layer).rev() {
            let lc = &cache.layer_caches[l];
            let inner = 4 * e;

            // --- FF backward ---
            // Block structure (pre-norm):
            //   x_mid = x_in + attn(ln1(x_in))   [first residual]
            //   x_out = x_mid + ff(ln2(x_mid))    [second residual]
            // dx is d_x_out

            // Second residual backward: dx_mid = dx (direct) + dx through ff path
            // d_ff_out = dx
            let d_ff_out = dx.clone();

            // Backward through ff_b2
            for i in 0..t {
                for j in 0..e {
                    grads.ff_b2[l][j] += d_ff_out[i * e + j];
                }
            }

            // Backward through ff_w2
            let d_ff_hidden = matmul_backward_both(
                &lc.ff_post_gelu,
                &self.ff_w2[l],
                &d_ff_out,
                t,
                inner,
                e,
                &mut grads.ff_w2[l],
            );

            // Backward through GELU
            let d_ff_pre_gelu = gelu_backward(&lc.ff_pre_gelu, &d_ff_hidden);

            // Backward through ff_b1
            for i in 0..t {
                for j in 0..inner {
                    grads.ff_b1[l][j] += d_ff_pre_gelu[i * inner + j];
                }
            }

            // Backward through ff_w1
            let d_ln2_out = matmul_backward_both(
                &lc.ln2_out,
                &self.ff_w1[l],
                &d_ff_pre_gelu,
                t,
                e,
                inner,
                &mut grads.ff_w1[l],
            );

            // Backward through layer norm 2
            let d_from_ln2 = layer_norm_backward(
                &lc.x_after_attn_residual,
                &d_ln2_out,
                &self.ln2_gamma[l],
                t,
                e,
                &mut grads.ln2_gamma[l],
                &mut grads.ln2_beta[l],
            );

            // dx_mid = dx (residual direct) + d_from_ln2 (through ff path)
            let mut dx_mid = dx;
            for i in 0..t * e {
                dx_mid[i] += d_from_ln2[i];
            }

            // --- Attention backward ---
            // First residual backward: dx_in = dx_mid (direct) + dx_mid through attn path
            // d_proj_out = dx_mid
            let d_proj_out = dx_mid.clone();

            // Backward through output projection
            let d_attn_out = matmul_backward_both(
                &lc.attn_out_pre_proj,
                &self.attn_proj[l],
                &d_proj_out,
                t,
                e,
                e,
                &mut grads.attn_proj[l],
            );

            // Backward through multi-head attention
            let nh = cfg.n_head;
            let hs = e / nh;
            let mut d_qkv = vec![0.0f32; t * 3 * e];

            for h in 0..nh {
                // d_attn_out for this head
                // Backward through V weighting: attn_out[i,h*hs+k] = sum_j w[i,j] * V[j,h*hs+k]
                for i in 0..t {
                    for k in 0..hs {
                        let d_out = d_attn_out[i * e + h * hs + k];
                        for j in 0..t {
                            let w = lc.attn_weights[h * t * t + i * t + j];
                            // d_V[j, h*hs+k] += w * d_out
                            d_qkv[j * 3 * e + 2 * e + h * hs + k] += w * d_out;
                            // d_w[i, j] += V[j, h*hs+k] * d_out (for softmax backward)
                        }
                    }
                }

                // Backward through softmax: d_score = attn_weights * (d_w - sum(d_w * attn_weights))
                // First compute d_w (pre-softmax gradient)
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
                        dot += d_attn_score[i * t + j] * lc.attn_weights[h * t * t + i * t + j];
                    }
                    for j in 0..t {
                        let w = lc.attn_weights[h * t * t + i * t + j];
                        d_attn_score[i * t + j] = w * (d_attn_score[i * t + j] - dot);
                    }
                }

                // Scale backward
                let scale = 1.0 / (hs as f32).sqrt();
                for val in d_attn_score.iter_mut() {
                    *val *= scale;
                }

                // Backward through Q @ K^T
                // d_Q[i, k] += sum_j d_score[i,j] * K[j, k]
                // d_K[j, k] += sum_i d_score[i,j] * Q[i, k]
                for i in 0..t {
                    for j in 0..=i {
                        // Only non-masked positions
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
            let d_ln1_out = matmul_backward_both(
                &lc.ln1_out,
                &self.qkv_w[l],
                &d_qkv,
                t,
                e,
                3 * e,
                &mut grads.qkv_w[l],
            );

            // Backward through layer norm 1
            let d_from_attn = layer_norm_backward(
                &lc.x_input,
                &d_ln1_out,
                &self.ln1_gamma[l],
                t,
                e,
                &mut grads.ln1_gamma[l],
                &mut grads.ln1_beta[l],
            );

            // dx_in = dx_mid (residual direct) + d_from_attn (through attn path)
            dx = dx_mid;
            for i in 0..t * e {
                dx[i] += d_from_attn[i];
            }
        }

        // Backward through embeddings
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                grads.token_emb[tok * e + j] += dx[i * e + j];
                grads.pos_emb[i * e + j] += dx[i * e + j];
            }
        }

        (loss, grads)
    }

    /// Apply gradients with Adam optimizer
    pub fn apply_gradients(&mut self, grads: &Gradients, lr: f32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        self.t += 1;
        let t_f = self.t as f32;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);

        let mut idx = 0usize;

        // Helper macro to apply Adam to one parameter
        macro_rules! adam_update {
            ($param:expr, $grad:expr) => {{
                adam_step($param, $grad, &mut self.m[idx], &mut self.v[idx], lr, beta1, beta2, eps, bc1, bc2);
                idx += 1;
                let _ = idx; // suppress last-iteration unused warning
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

            // Get logits for last position
            let last_offset = (t - 1) * v;
            let last_logits = &logits[last_offset..last_offset + v];

            // Temperature sampling
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

            // Sample
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

// ---- Gradient storage ----

pub struct Gradients {
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

impl Gradients {
    pub fn zero_like(cfg: &Config) -> Self {
        Self::new(cfg)
    }

    fn new(cfg: &Config) -> Self {
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

    #[allow(dead_code)]
    fn all_grads_flat(&self) -> Vec<&Vec<f32>> {
        let mut grads = Vec::new();
        grads.push(&self.token_emb);
        grads.push(&self.pos_emb);
        for l in 0..self.ln1_gamma.len() {
            grads.push(&self.ln1_gamma[l]);
            grads.push(&self.ln1_beta[l]);
            grads.push(&self.qkv_w[l]);
            grads.push(&self.attn_proj[l]);
            grads.push(&self.ln2_gamma[l]);
            grads.push(&self.ln2_beta[l]);
            grads.push(&self.ff_w1[l]);
            grads.push(&self.ff_b1[l]);
            grads.push(&self.ff_w2[l]);
            grads.push(&self.ff_b2[l]);
        }
        grads.push(&self.ln_f_gamma);
        grads.push(&self.ln_f_beta);
        grads.push(&self.lm_head);
        grads
    }

    /// Accumulate another gradient into this one
    pub fn accumulate(&mut self, other: &Gradients) {
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

    /// Scale all gradients by a factor
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

// ---- Forward cache for backward pass ----

#[derive(Default)]
pub struct LayerCache {
    pub x_input: Vec<f32>,
    pub ln1_out: Vec<f32>,
    pub ln1_mean: Vec<f32>,
    pub ln1_rstd: Vec<f32>,
    pub qkv: Vec<f32>,
    pub attn_weights: Vec<f32>,     // (n_head, T, T)
    pub attn_out_pre_proj: Vec<f32>,
    pub x_after_attn_residual: Vec<f32>,
    pub ln2_out: Vec<f32>,
    pub ln2_mean: Vec<f32>,
    pub ln2_rstd: Vec<f32>,
    pub ff_pre_gelu: Vec<f32>,
    pub ff_post_gelu: Vec<f32>,
}

#[allow(dead_code)]
pub struct ForwardCache {
    pub tokens: Vec<usize>,
    pub x_after_emb: Vec<f32>,
    pub layer_caches: Vec<LayerCache>,
    pub x_before_final_ln: Vec<f32>,
    pub x_after_final_ln: Vec<f32>,
}

// ---- Helper functions ----

pub fn adam_step(
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

pub fn add_vecs(a: &mut Vec<f32>, b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

pub fn scale_vec(a: &mut Vec<f32>, s: f32) {
    for x in a.iter_mut() {
        *x *= s;
    }
}

/// Matrix multiply: A (m x k) @ B (k x n) -> C (m x n)
/// Parallelized across output rows using rayon.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for p in 0..k {
            let a_val = a[i * k + p];
            if a_val.abs() > 1e-12 {
                for j in 0..n {
                    row[j] += a_val * b[p * n + j];
                }
            }
        }
    });
    c
}

/// Backward through C = A @ B, given dC
/// Returns dA, accumulates into dB
pub fn matmul_backward_both(
    a: &[f32],     // (m, k)
    b: &[f32],     // (k, n)
    dc: &[f32],    // (m, n)
    m: usize,
    k: usize,
    n: usize,
    db: &mut Vec<f32>, // (k, n) gradient accumulator
) -> Vec<f32> {
    // dA = dC @ B^T -> (m, k)
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

    // dB = A^T @ dC -> (k, n)
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

/// Layer norm forward: returns (output, mean, rstd)
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], t: usize, e: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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

/// Layer norm backward
pub fn layer_norm_backward(
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

        // Compute normalized values
        let mut norm = vec![0.0f32; e];
        for j in 0..e {
            norm[j] = (x[offset + j] - mean) * rstd;
        }

        // dgamma and dbeta
        for j in 0..e {
            dgamma[j] += dout[offset + j] * norm[j];
            dbeta[j] += dout[offset + j];
        }

        // dx
        let mut dnorm = vec![0.0f32; e];
        for j in 0..e {
            dnorm[j] = dout[offset + j] * gamma[j];
        }

        let dnorm_mean: f32 = dnorm.iter().sum::<f32>() / e as f32;
        let dnorm_norm_mean: f32 = dnorm.iter().zip(norm.iter()).map(|(a, b)| a * b).sum::<f32>() / e as f32;

        for j in 0..e {
            dx[offset + j] = (dnorm[j] - dnorm_mean - norm[j] * dnorm_norm_mean) * rstd;
        }
    }

    dx
}

/// GELU backward
pub fn gelu_backward(x: &[f32], dout: &[f32]) -> Vec<f32> {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut dx = vec![0.0f32; x.len()];

    for i in 0..x.len() {
        let xi = x[i];
        let x3 = xi * xi * xi;
        let inner = sqrt_2_over_pi * (xi + 0.044715 * x3);
        let tanh_val = inner.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;
        let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * xi * xi);

        // d/dx GELU(x) = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * d_inner
        dx[i] = dout[i] * (0.5 * (1.0 + tanh_val) + 0.5 * xi * sech2 * d_inner);
    }

    dx
}

// ─── Weight serialization ──────────────────────────────────────

impl GPT {
    /// Save model weights to a binary file.
    /// Format: [magic: 4 bytes] [config: 5 x u32] [n_floats: u64] [f32 data...]
    pub fn save_weights(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"MGPT")?;
        f.write_all(&(self.config.vocab_size as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_embd as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_head as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_layer as u32).to_le_bytes())?;
        f.write_all(&(self.config.block_size as u32).to_le_bytes())?;

        let mut all: Vec<f32> = Vec::new();
        all.extend_from_slice(&self.token_emb);
        all.extend_from_slice(&self.pos_emb);
        for l in 0..self.config.n_layer {
            all.extend_from_slice(&self.ln1_gamma[l]);
            all.extend_from_slice(&self.ln1_beta[l]);
            all.extend_from_slice(&self.qkv_w[l]);
            all.extend_from_slice(&self.attn_proj[l]);
            all.extend_from_slice(&self.ln2_gamma[l]);
            all.extend_from_slice(&self.ln2_beta[l]);
            all.extend_from_slice(&self.ff_w1[l]);
            all.extend_from_slice(&self.ff_b1[l]);
            all.extend_from_slice(&self.ff_w2[l]);
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

    /// Load model weights from a binary file.
    pub fn load_weights(path: &str) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"MGPT" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a MGPT weight file"));
        }

        let read_u32 = |f: &mut std::fs::File| -> std::io::Result<u32> {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            Ok(u32::from_le_bytes(buf))
        };

        let vocab_size = read_u32(&mut f)? as usize;
        let n_embd = read_u32(&mut f)? as usize;
        let n_head = read_u32(&mut f)? as usize;
        let n_layer = read_u32(&mut f)? as usize;
        let block_size = read_u32(&mut f)? as usize;
        let config = Config { vocab_size, n_embd, n_head, n_layer, block_size };

        let mut n_buf = [0u8; 8];
        f.read_exact(&mut n_buf)?;
        let n_floats = u64::from_le_bytes(n_buf) as usize;

        let mut raw = vec![0u8; n_floats * 4];
        f.read_exact(&mut raw)?;
        let data: Vec<f32> = raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let mut model = GPT::new(config.clone());
        let mut i = 0;
        let take = |d: &[f32], i: &mut usize, n: usize| -> Vec<f32> {
            let s = d[*i..*i + n].to_vec(); *i += n; s
        };

        model.token_emb = take(&data, &mut i, vocab_size * n_embd);
        model.pos_emb = take(&data, &mut i, block_size * n_embd);
        let inner = 4 * n_embd;
        for l in 0..n_layer {
            model.ln1_gamma[l] = take(&data, &mut i, n_embd);
            model.ln1_beta[l] = take(&data, &mut i, n_embd);
            model.qkv_w[l] = take(&data, &mut i, n_embd * 3 * n_embd);
            model.attn_proj[l] = take(&data, &mut i, n_embd * n_embd);
            model.ln2_gamma[l] = take(&data, &mut i, n_embd);
            model.ln2_beta[l] = take(&data, &mut i, n_embd);
            model.ff_w1[l] = take(&data, &mut i, n_embd * inner);
            model.ff_b1[l] = take(&data, &mut i, inner);
            model.ff_w2[l] = take(&data, &mut i, inner * n_embd);
            model.ff_b2[l] = take(&data, &mut i, n_embd);
        }
        model.ln_f_gamma = take(&data, &mut i, n_embd);
        model.ln_f_beta = take(&data, &mut i, n_embd);
        model.lm_head = take(&data, &mut i, n_embd * vocab_size);
        Ok(model)
    }
}

// ─── Hebbian Inference Memory Bank ─────────────────────────────

/// External memory bank that updates during inference via Hebbian learning.
/// No backprop needed at inference time — the model "learns" from context
/// by writing key-value pairs to memory and reading from it.
pub struct MemoryBank {
    pub m_keys: Vec<f32>,     // (memory_slots x key_dim) — stored flat
    pub m_values: Vec<f32>,   // (memory_slots x value_dim) — stored flat
    pub memory_slots: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub eta: f32,             // Hebbian learning rate
    pub slots_used: usize,    // Track how many slots have been written
}

impl MemoryBank {
    pub fn new(memory_slots: usize, key_dim: usize, value_dim: usize, eta: f32) -> Self {
        // Keys initialized to small random values so cosine similarity is meaningful
        let m_keys = randn_vec(memory_slots * key_dim, 0.01);
        let m_values = vec![0.0; memory_slots * value_dim];
        Self {
            m_keys,
            m_values,
            memory_slots,
            key_dim,
            value_dim,
            eta,
            slots_used: 0,
        }
    }

    /// Clear memory (reset between sequences)
    pub fn clear(&mut self) {
        self.m_keys = randn_vec(self.memory_slots * self.key_dim, 0.01);
        self.m_values = vec![0.0; self.memory_slots * self.value_dim];
        self.slots_used = 0;
    }

    /// Write a key-value pair to the most similar existing slot (Hebbian update)
    pub fn write(&mut self, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.key_dim);
        assert_eq!(value.len(), self.value_dim);

        // Compute similarity: key @ M_keys^T -> (memory_slots,)
        let mut similarities = vec![0.0f32; self.memory_slots];
        for s in 0..self.memory_slots {
            let mut dot = 0.0f32;
            for d in 0..self.key_dim {
                dot += key[d] * self.m_keys[s * self.key_dim + d];
            }
            similarities[s] = dot;
        }

        // Softmax to find best slot
        let max_sim = similarities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sims = vec![0.0f32; self.memory_slots];
        let mut sum = 0.0f32;
        for s in 0..self.memory_slots {
            exp_sims[s] = (similarities[s] - max_sim).exp();
            sum += exp_sims[s];
        }
        for s in 0..self.memory_slots {
            exp_sims[s] /= sum;
        }

        // Find best slot (argmax of softmax)
        let best_slot = exp_sims
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Hebbian update: M_keys[best] += eta * key, M_values[best] += eta * value
        for d in 0..self.key_dim {
            self.m_keys[best_slot * self.key_dim + d] += self.eta * key[d];
        }
        for d in 0..self.value_dim {
            self.m_values[best_slot * self.value_dim + d] += self.eta * value[d];
        }

        // Normalize the updated key to prevent explosion
        let mut norm = 0.0f32;
        for d in 0..self.key_dim {
            let v = self.m_keys[best_slot * self.key_dim + d];
            norm += v * v;
        }
        norm = norm.sqrt().max(1e-8);
        for d in 0..self.key_dim {
            self.m_keys[best_slot * self.key_dim + d] /= norm;
        }

        if best_slot >= self.slots_used {
            self.slots_used = best_slot + 1;
        }
    }

    /// Read from memory: compute attention-weighted sum of values
    /// Returns memory_output of shape (value_dim,)
    pub fn read(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.key_dim);

        // attention_weights = softmax(query @ M_keys^T / sqrt(key_dim))
        let scale = 1.0 / (self.key_dim as f32).sqrt();
        let mut scores = vec![0.0f32; self.memory_slots];
        for s in 0..self.memory_slots {
            let mut dot = 0.0f32;
            for d in 0..self.key_dim {
                dot += query[d] * self.m_keys[s * self.key_dim + d];
            }
            scores[s] = dot * scale;
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = vec![0.0f32; self.memory_slots];
        let mut sum = 0.0f32;
        for s in 0..self.memory_slots {
            weights[s] = (scores[s] - max_score).exp();
            sum += weights[s];
        }
        for s in 0..self.memory_slots {
            weights[s] /= sum;
        }

        // memory_output = weights @ M_values
        let mut output = vec![0.0f32; self.value_dim];
        for s in 0..self.memory_slots {
            for d in 0..self.value_dim {
                output[d] += weights[s] * self.m_values[s * self.value_dim + d];
            }
        }

        output
    }

    /// Compute memory utilization: fraction of slots with non-trivial values
    pub fn utilization(&self) -> f32 {
        let mut active = 0usize;
        for s in 0..self.memory_slots {
            let mut energy = 0.0f32;
            for d in 0..self.value_dim {
                let v = self.m_values[s * self.value_dim + d];
                energy += v * v;
            }
            if energy > 1e-6 {
                active += 1;
            }
        }
        active as f32 / self.memory_slots as f32
    }

    /// Compute key diversity: average pairwise cosine distance
    pub fn key_diversity(&self) -> f32 {
        if self.memory_slots < 2 {
            return 0.0;
        }
        let mut total_dist = 0.0f32;
        let mut count = 0;
        for i in 0..self.memory_slots.min(16) {
            for j in (i + 1)..self.memory_slots.min(16) {
                let mut dot = 0.0f32;
                let mut norm_i = 0.0f32;
                let mut norm_j = 0.0f32;
                for d in 0..self.key_dim {
                    let ki = self.m_keys[i * self.key_dim + d];
                    let kj = self.m_keys[j * self.key_dim + d];
                    dot += ki * kj;
                    norm_i += ki * ki;
                    norm_j += kj * kj;
                }
                let cos = dot / (norm_i.sqrt() * norm_j.sqrt()).max(1e-8);
                total_dist += 1.0 - cos; // cosine distance
                count += 1;
            }
        }
        if count > 0 { total_dist / count as f32 } else { 0.0 }
    }
}

/// HebbianGPT: wraps a base GPT with an external Hebbian memory bank.
/// The W_gate parameter is trained via backprop; the memory updates via
/// Hebbian rule during inference (no gradients needed).
pub struct HebbianGPT {
    pub gpt: GPT,
    pub memory: MemoryBank,
    pub w_gate: Vec<f32>,  // (n_embd,) — sigmoid gate for memory mixing
    // Adam state for w_gate
    pub gate_m: Vec<f32>,
    pub gate_v: Vec<f32>,
}

/// Gradients for HebbianGPT: base GPT grads + W_gate grads
pub struct HebbianGradients {
    pub base: Gradients,
    pub w_gate: Vec<f32>,
}

impl HebbianGPT {
    pub fn new(config: Config) -> Self {
        let e = config.n_embd;
        let gpt = GPT::new(config);
        let memory = MemoryBank::new(32, e, e, 0.1);
        // Initialize gate to small values so sigmoid starts near 0.5
        let w_gate = randn_vec(e, 0.01);
        let gate_m = vec![0.0; e];
        let gate_v = vec![0.0; e];

        Self { gpt, memory, w_gate, gate_m, gate_v }
    }

    /// Forward pass with memory read after the last attention layer.
    /// Memory is read but NOT written during training forward pass.
    /// Returns logits and cache for backward pass.
    fn forward_with_memory_cache(
        &self,
        tokens: &[usize],
        use_memory_read: bool,
    ) -> (Vec<f32>, ForwardCache, Vec<f32>, Vec<f32>) {
        let cfg = &self.gpt.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        // Run the base GPT forward (which stores cache)
        let (logits, cache) = self.gpt.forward_with_cache(tokens);

        if !use_memory_read {
            // Return logits unmodified, with empty gate/memory vecs
            return (logits, cache, vec![], vec![]);
        }

        // Read from memory for each position using hidden state before final LN
        // For efficiency, we read for the LAST position only (most important for generation)
        let last_hidden = &cache.x_before_final_ln[(t - 1) * e..t * e];
        let mem_output = self.memory.read(last_hidden);

        // Compute gate = sigmoid(hidden @ W_gate)
        let mut gate_logit = 0.0f32;
        for d in 0..e {
            gate_logit += last_hidden[d] * self.w_gate[d];
        }
        let gate = 1.0 / (1.0 + (-gate_logit).exp());

        // Mix: modify the logits for the last position
        // We need to re-compute: modified_hidden = hidden + gate * mem_output
        // Then re-apply final LN and LM head for the last position
        let mut modified_hidden = cache.x_before_final_ln.clone();
        for d in 0..e {
            modified_hidden[(t - 1) * e + d] += gate * mem_output[d];
        }

        // Re-apply final layer norm
        let (ln_out, _, _) = layer_norm(
            &modified_hidden,
            &self.gpt.ln_f_gamma,
            &self.gpt.ln_f_beta,
            t,
            e,
        );

        // Re-apply LM head
        let new_logits = matmul(&ln_out, &self.gpt.lm_head, t, e, v);

        // Store gate info for backward
        let gate_info = vec![gate, gate_logit];

        (new_logits, cache, mem_output, gate_info)
    }

    /// Forward + backward for training.
    /// During training, memory is cleared each batch and not used for reading
    /// (or optionally, we can train with memory read to learn W_gate).
    pub fn forward_backward(
        &self,
        tokens: &[usize],
        targets: &[usize],
    ) -> (f32, HebbianGradients) {
        let cfg = &self.gpt.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        // Forward with memory read enabled (so W_gate gets gradients)
        let (logits, cache, mem_output, gate_info) =
            self.forward_with_memory_cache(tokens, true);

        // Compute loss (cross-entropy)
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

        // Use base GPT's forward_backward for the base gradients
        // (this is simpler and correct since memory read is a small perturbation)
        let (_, base_grads) = self.gpt.forward_backward(tokens, targets);

        // Compute W_gate gradient
        // The gate affects the last position's logits through:
        //   modified_hidden[last] = hidden[last] + gate * mem_output
        //   gate = sigmoid(hidden[last] @ W_gate)
        // d_loss/d_W_gate = d_loss/d_gate * d_gate/d_W_gate
        // d_gate/d_W_gate = gate * (1 - gate) * hidden[last]
        let mut d_w_gate = vec![0.0f32; e];
        if !gate_info.is_empty() && !mem_output.is_empty() {
            let gate = gate_info[0];
            let last_hidden = &cache.x_before_final_ln[(t - 1) * e..t * e];

            // d_loss/d_modified_hidden for last position comes from the logit gradient
            // Approximate: use the logit gradient magnitude as signal
            let last_offset = (t - 1) * v;
            let mut d_hidden_from_logits = vec![0.0f32; e];
            for d in 0..e {
                for j in 0..v {
                    let d_logit = probs[last_offset + j]
                        - if j == targets[t - 1] { 1.0 } else { 0.0 };
                    d_hidden_from_logits[d] +=
                        d_logit / t as f32 * self.gpt.lm_head[d * v + j];
                }
            }

            // d_loss/d_gate = sum_d(d_hidden_from_logits[d] * mem_output[d])
            let mut d_gate = 0.0f32;
            for d in 0..e {
                d_gate += d_hidden_from_logits[d] * mem_output[d];
            }

            // d_gate/d_gate_logit = gate * (1 - gate)  (sigmoid derivative)
            let d_gate_logit = d_gate * gate * (1.0 - gate);

            // d_gate_logit/d_W_gate = hidden[last]
            for d in 0..e {
                d_w_gate[d] = d_gate_logit * last_hidden[d];
            }
        }

        (loss, HebbianGradients {
            base: base_grads,
            w_gate: d_w_gate,
        })
    }

    /// Apply gradients with Adam for all parameters including W_gate
    pub fn apply_gradients(&mut self, grads: &HebbianGradients, lr: f32) {
        self.gpt.apply_gradients(&grads.base, lr);

        // Adam update for W_gate
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let t_f = self.gpt.t as f32; // reuse the step counter
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);

        adam_step(
            &mut self.w_gate,
            &grads.w_gate,
            &mut self.gate_m,
            &mut self.gate_v,
            lr,
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
        );
    }

    /// Write to memory after generating a token (Hebbian update)
    /// key = hidden state after layer 1, value = hidden state after final LN
    pub fn write_to_memory(&mut self, tokens: &[usize]) {
        let cfg = &self.gpt.config;
        let t = tokens.len();
        let e = cfg.n_embd;

        if t == 0 {
            return;
        }

        let (_, cache) = self.gpt.forward_with_cache(tokens);

        // Key: hidden state after layer 1 (layer index 1, or 0 if only 1 layer)
        let key_layer = if cfg.n_layer > 1 { 1 } else { 0 };
        let key_start = (t - 1) * e;
        let key: Vec<f32> = if key_layer < cache.layer_caches.len() {
            cache.layer_caches[key_layer].x_after_attn_residual[key_start..key_start + e].to_vec()
        } else {
            cache.x_before_final_ln[key_start..key_start + e].to_vec()
        };

        // Value: hidden state after final layer norm (last position)
        let value = cache.x_after_final_ln[key_start..key_start + e].to_vec();

        self.memory.write(&key, &value);
    }

    /// Clear memory between sequences
    pub fn clear_memory(&mut self) {
        self.memory.clear();
    }

    /// Generate with memory updates after each token
    pub fn generate_with_memory(
        &mut self,
        start_tokens: &[usize],
        max_new_tokens: usize,
    ) -> Vec<usize> {
        let mut tokens = start_tokens.to_vec();
        let mut rng = rand::thread_rng();
        let v = self.gpt.config.vocab_size;

        for _ in 0..max_new_tokens {
            let start = if tokens.len() > self.gpt.config.block_size {
                tokens.len() - self.gpt.config.block_size
            } else {
                0
            };
            let context = &tokens[start..];

            // Forward with memory read
            let (logits, _, _, _) = self.forward_with_memory_cache(context, true);
            let t = context.len();

            // Get logits for last position
            let last_offset = (t - 1) * v;
            let last_logits = &logits[last_offset..last_offset + v];

            // Temperature sampling
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

            // Sample
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

            // Write to memory after each generated token
            self.write_to_memory(&tokens[start..]);
        }

        tokens
    }

    /// Generate without memory (for comparison baseline)
    pub fn generate_without_memory(&self, start_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        self.gpt.generate(start_tokens, max_new_tokens)
    }

    /// Save weights including W_gate
    pub fn save_weights(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        // Magic: "HGPT" for Hebbian GPT
        f.write_all(b"HGPT")?;
        f.write_all(&(self.gpt.config.vocab_size as u32).to_le_bytes())?;
        f.write_all(&(self.gpt.config.n_embd as u32).to_le_bytes())?;
        f.write_all(&(self.gpt.config.n_head as u32).to_le_bytes())?;
        f.write_all(&(self.gpt.config.n_layer as u32).to_le_bytes())?;
        f.write_all(&(self.gpt.config.block_size as u32).to_le_bytes())?;
        f.write_all(&(self.memory.memory_slots as u32).to_le_bytes())?;

        // Collect all floats: base GPT weights + W_gate
        let mut all: Vec<f32> = Vec::new();
        all.extend_from_slice(&self.gpt.token_emb);
        all.extend_from_slice(&self.gpt.pos_emb);
        for l in 0..self.gpt.config.n_layer {
            all.extend_from_slice(&self.gpt.ln1_gamma[l]);
            all.extend_from_slice(&self.gpt.ln1_beta[l]);
            all.extend_from_slice(&self.gpt.qkv_w[l]);
            all.extend_from_slice(&self.gpt.attn_proj[l]);
            all.extend_from_slice(&self.gpt.ln2_gamma[l]);
            all.extend_from_slice(&self.gpt.ln2_beta[l]);
            all.extend_from_slice(&self.gpt.ff_w1[l]);
            all.extend_from_slice(&self.gpt.ff_b1[l]);
            all.extend_from_slice(&self.gpt.ff_w2[l]);
            all.extend_from_slice(&self.gpt.ff_b2[l]);
        }
        all.extend_from_slice(&self.gpt.ln_f_gamma);
        all.extend_from_slice(&self.gpt.ln_f_beta);
        all.extend_from_slice(&self.gpt.lm_head);
        all.extend_from_slice(&self.w_gate);

        f.write_all(&(all.len() as u64).to_le_bytes())?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(all.as_ptr() as *const u8, all.len() * 4)
        };
        f.write_all(bytes)?;
        Ok(())
    }

    /// Count total parameters (base GPT + W_gate)
    pub fn param_count(config: &Config) -> usize {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let inner = 4 * e;
        let mut total = v * e + config.block_size * e;
        for _ in 0..nl {
            total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
        }
        total += e * 2 + e * v;
        total += e; // W_gate
        total
    }
}

// ─── RGGPT: Renormalization Group Weight Sharing ────────────────
//
// Physics inspiration: In the Renormalization Group framework, the same
// fundamental interactions apply at each energy scale, but with different
// coupling constants. Here, all transformer layers share one set of base
// weights, with per-layer scalar scale (alpha) and shift (beta) factors
// acting as "running coupling constants."
//
// Effective weight at layer l: W_l = alpha_l * W_shared + beta_l

/// Number of shared weight matrices (qkv, attn_proj, ff_w1, ff_w2)
const RG_NUM_SHARED: usize = 4;

/// RG-GPT: Transformer with shared weights + per-layer RG scale factors
pub struct RGGPT {
    // Embeddings (not shared)
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,

    // Shared weight matrices (ONE copy)
    pub qkv_w_shared: Vec<f32>,
    pub attn_proj_shared: Vec<f32>,
    pub ff_w1_shared: Vec<f32>,
    pub ff_w2_shared: Vec<f32>,

    // Per-layer RG scale factors
    // [layer][matrix_idx] where idx: 0=qkv, 1=attn_proj, 2=ff_w1, 3=ff_w2
    pub rg_alpha: Vec<[f32; RG_NUM_SHARED]>,
    pub rg_beta: Vec<[f32; RG_NUM_SHARED]>,

    // Per-layer small parameters (kept separate)
    pub ln1_gamma: Vec<Vec<f32>>,
    pub ln1_beta: Vec<Vec<f32>>,
    pub ln2_gamma: Vec<Vec<f32>>,
    pub ln2_beta: Vec<Vec<f32>>,
    pub ff_b1: Vec<Vec<f32>>,
    pub ff_b2: Vec<Vec<f32>>,

    // Final layer norm + head
    pub ln_f_gamma: Vec<f32>,
    pub ln_f_beta: Vec<f32>,
    pub lm_head: Vec<f32>,

    pub config: Config,

    // Adam optimizer state
    pub rg_m: Vec<Vec<f32>>,
    pub rg_v: Vec<Vec<f32>>,
    pub rg_t: usize,
}

/// Gradient storage for RGGPT
pub struct RGGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub qkv_w_shared: Vec<f32>,
    pub attn_proj_shared: Vec<f32>,
    pub ff_w1_shared: Vec<f32>,
    pub ff_w2_shared: Vec<f32>,
    pub rg_alpha: Vec<[f32; RG_NUM_SHARED]>,
    pub rg_beta: Vec<[f32; RG_NUM_SHARED]>,
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

/// Forward cache for RGGPT backward pass
#[allow(dead_code)]
struct RGForwardCache {
    tokens: Vec<usize>,
    x_after_emb: Vec<f32>,
    layer_caches: Vec<RGLayerCache>,
    x_before_final_ln: Vec<f32>,
    x_after_final_ln: Vec<f32>,
}

/// Per-layer cache for RGGPT
#[derive(Default)]
struct RGLayerCache {
    x_input: Vec<f32>,
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

impl RGGradients {
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
            rg_alpha: vec![[0.0; RG_NUM_SHARED]; nl],
            rg_beta: vec![[0.0; RG_NUM_SHARED]; nl],
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

    pub fn accumulate(&mut self, other: &RGGradients) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        add_vecs(&mut self.qkv_w_shared, &other.qkv_w_shared);
        add_vecs(&mut self.attn_proj_shared, &other.attn_proj_shared);
        add_vecs(&mut self.ff_w1_shared, &other.ff_w1_shared);
        add_vecs(&mut self.ff_w2_shared, &other.ff_w2_shared);
        for l in 0..self.rg_alpha.len() {
            for k in 0..RG_NUM_SHARED {
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
            for k in 0..RG_NUM_SHARED {
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

impl RGGPT {
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

        let rg_alpha = vec![[1.0f32; RG_NUM_SHARED]; nl];
        let rg_beta = vec![[0.0f32; RG_NUM_SHARED]; nl];

        let mut model = RGGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            qkv_w_shared: randn_vec(e * 3 * e, layer_scale),
            attn_proj_shared: randn_vec(e * e, layer_scale),
            ff_w1_shared: randn_vec(e * inner, layer_scale),
            ff_w2_shared: randn_vec(inner * e, layer_scale * 0.5),
            rg_alpha,
            rg_beta,
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
            rg_m: Vec::new(),
            rg_v: Vec::new(),
            rg_t: 0,
        };

        let param_sizes = model.param_sizes();
        model.rg_m = param_sizes.iter().map(|&s| vec![0.0; s]).collect();
        model.rg_v = param_sizes.iter().map(|&s| vec![0.0; s]).collect();

        model
    }

    /// Count total trainable parameters
    pub fn count_params(&self) -> usize {
        let e = self.config.n_embd;
        let v = self.config.vocab_size;
        let nl = self.config.n_layer;
        let inner = 4 * e;
        let bs = self.config.block_size;

        let emb = v * e + bs * e;
        let shared = e * 3 * e + e * e + e * inner + inner * e;
        let rg_scalars = nl * RG_NUM_SHARED * 2;
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
            sizes.push(RG_NUM_SHARED); // alpha
            sizes.push(RG_NUM_SHARED); // beta
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

    /// Forward pass with cache for backward
    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, RGForwardCache) {
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

        let mut cache = RGForwardCache {
            tokens: tokens.to_vec(),
            x_after_emb: x.clone(),
            layer_caches: Vec::new(),
            x_before_final_ln: Vec::new(),
            x_after_final_ln: Vec::new(),
        };

        for l in 0..cfg.n_layer {
            let mut lc = RGLayerCache::default();
            lc.x_input = x.clone();

            // Compute effective weights: W_eff = alpha * W_shared + beta
            let eff_qkv_w: Vec<f32> = self.qkv_w_shared.iter()
                .map(|&w| self.rg_alpha[l][0] * w + self.rg_beta[l][0]).collect();
            let eff_attn_proj: Vec<f32> = self.attn_proj_shared.iter()
                .map(|&w| self.rg_alpha[l][1] * w + self.rg_beta[l][1]).collect();
            let eff_ff_w1: Vec<f32> = self.ff_w1_shared.iter()
                .map(|&w| self.rg_alpha[l][2] * w + self.rg_beta[l][2]).collect();
            let eff_ff_w2: Vec<f32> = self.ff_w2_shared.iter()
                .map(|&w| self.rg_alpha[l][3] * w + self.rg_beta[l][3]).collect();

            // Layer norm 1
            let (ln1_out, ln1_mean, ln1_rstd) = layer_norm(
                &x, &self.ln1_gamma[l], &self.ln1_beta[l], t, e,
            );
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
            for i in 0..t * e {
                x[i] += proj_out[i];
            }
            lc.x_after_attn_residual = x.clone();

            // Layer norm 2
            let (ln2_out, ln2_mean, ln2_rstd) = layer_norm(
                &x, &self.ln2_gamma[l], &self.ln2_beta[l], t, e,
            );
            lc.ln2_out = ln2_out.clone();
            lc.ln2_mean = ln2_mean;
            lc.ln2_rstd = ln2_rstd;

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

    /// Forward + backward returning loss and gradients
    pub fn forward_backward(
        &self, tokens: &[usize], targets: &[usize],
    ) -> (f32, RGGradients) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

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

        let mut grads = RGGradients::zero_like(cfg);

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

        // Backward through transformer blocks (reverse order)
        for l in (0..cfg.n_layer).rev() {
            let lc = &cache.layer_caches[l];
            let inner = 4 * e;

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

            // RG backward for ff_w2 (matrix index 3)
            let alpha_3 = self.rg_alpha[l][3];
            for i in 0..d_eff_ff_w2.len() {
                grads.rg_alpha[l][3] += d_eff_ff_w2[i] * self.ff_w2_shared[i];
                grads.rg_beta[l][3] += d_eff_ff_w2[i];
                grads.ff_w2_shared[i] += alpha_3 * d_eff_ff_w2[i];
            }

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

            // RG backward for ff_w1 (matrix index 2)
            let alpha_2 = self.rg_alpha[l][2];
            for i in 0..d_eff_ff_w1.len() {
                grads.rg_alpha[l][2] += d_eff_ff_w1[i] * self.ff_w1_shared[i];
                grads.rg_beta[l][2] += d_eff_ff_w1[i];
                grads.ff_w1_shared[i] += alpha_2 * d_eff_ff_w1[i];
            }

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

            // RG backward for attn_proj (matrix index 1)
            let alpha_1 = self.rg_alpha[l][1];
            for i in 0..d_eff_attn_proj.len() {
                grads.rg_alpha[l][1] += d_eff_attn_proj[i] * self.attn_proj_shared[i];
                grads.rg_beta[l][1] += d_eff_attn_proj[i];
                grads.attn_proj_shared[i] += alpha_1 * d_eff_attn_proj[i];
            }

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
                for val in d_attn_score.iter_mut() {
                    *val *= scale;
                }

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

            // RG backward for qkv_w (matrix index 0)
            let alpha_0 = self.rg_alpha[l][0];
            for i in 0..d_eff_qkv_w.len() {
                grads.rg_alpha[l][0] += d_eff_qkv_w[i] * self.qkv_w_shared[i];
                grads.rg_beta[l][0] += d_eff_qkv_w[i];
                grads.qkv_w_shared[i] += alpha_0 * d_eff_qkv_w[i];
            }

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

    /// Apply gradients with Adam optimizer
    pub fn apply_gradients(&mut self, grads: &RGGradients, lr: f32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        self.rg_t += 1;
        let t_f = self.rg_t as f32;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);

        let mut idx = 0usize;

        macro_rules! adam_update {
            ($param:expr, $grad:expr) => {{
                adam_step($param, $grad, &mut self.rg_m[idx], &mut self.rg_v[idx],
                          lr, beta1, beta2, eps, bc1, bc2);
                idx += 1;
                let _ = idx;
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
                      &mut self.rg_m[idx], &mut self.rg_v[idx],
                      lr, beta1, beta2, eps, bc1, bc2);
            self.rg_alpha[l].copy_from_slice(&alpha_vec);
            idx += 1;

            let mut beta_vec: Vec<f32> = self.rg_beta[l].to_vec();
            let beta_grad: Vec<f32> = grads.rg_beta[l].to_vec();
            adam_step(&mut beta_vec, &beta_grad,
                      &mut self.rg_m[idx], &mut self.rg_v[idx],
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

    /// Forward pass for inference
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
        f.write_all(b"RGPT")?;
        f.write_all(&(self.config.vocab_size as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_embd as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_head as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_layer as u32).to_le_bytes())?;
        f.write_all(&(self.config.block_size as u32).to_le_bytes())?;

        let mut all: Vec<f32> = Vec::new();
        all.extend_from_slice(&self.token_emb);
        all.extend_from_slice(&self.pos_emb);
        all.extend_from_slice(&self.qkv_w_shared);
        all.extend_from_slice(&self.attn_proj_shared);
        all.extend_from_slice(&self.ff_w1_shared);
        all.extend_from_slice(&self.ff_w2_shared);
        for l in 0..self.config.n_layer {
            all.extend_from_slice(&self.rg_alpha[l]);
            all.extend_from_slice(&self.rg_beta[l]);
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

    /// Get per-layer alpha values for analysis
    pub fn get_layer_alphas(&self) -> Vec<[f32; RG_NUM_SHARED]> {
        self.rg_alpha.clone()
    }

    /// Get per-layer beta values for analysis
    pub fn get_layer_betas(&self) -> Vec<[f32; RG_NUM_SHARED]> {
        self.rg_beta.clone()
    }
}
