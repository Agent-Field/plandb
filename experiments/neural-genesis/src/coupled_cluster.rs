//! Coupled Cluster Weight Updates for Neural Networks
//!
//! Physics: In quantum chemistry, Coupled Cluster theory decomposes electron
//! correlation into a systematic hierarchy:
//!   |ψ⟩ = e^{T₁+T₂+T₃+...}|Φ₀⟩
//!   T₁ = singles (single-orbital response) — captures ~80%
//!   T₂ = doubles (pairwise correlation) — CCSD captures ~95%
//!
//! Neural net mapping:
//!   T₁[l] = per-layer gradient using a learned linear surrogate ("critic")
//!           for the downstream error signal. No backprop through other layers.
//!   T₂[l] = exact 2-layer backprop correction, capturing nonlinearity
//!           interactions the linear critic misses.
//!   Full update: ΔW_l = T₁[l] + T₂[l]  (CCSD approximation)
//!
//! Key diagnostic: cosine similarity cos(∇L_CC, ∇L_backprop) measures
//! how well the CC hierarchy approximates the exact gradient.

use crate::data;
use crate::model::*;
use crate::tokenizer::Tokenizer;
use crate::tool_data::{self, ToolExample};
use rand::Rng;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

// ─── Per-Layer Critic (Linear Surrogate) ─────────────────────────────
//
// Physics analogy: In DMRG, the "right environment" encodes the effect
// of all downstream sites. Here, a linear critic h_l(x) ≈ L predicts
// the global loss from layer l's output activations.
//
// h_l(x) = w_l · avg_pool(x) + b_l
// ∂h_l/∂x_{t,e} = w_l[e] / T  (uniform across tokens)

struct LayerCritic {
    weights: Vec<f32>, // [n_embd] — linear projection
    bias: f32,
    // Adam state for critic
    m_w: Vec<f32>,
    m_b: f32,
    v_w: Vec<f32>,
    v_b: f32,
    step: usize,
}

impl LayerCritic {
    fn new(n_embd: usize) -> Self {
        Self {
            weights: vec![0.0; n_embd],
            bias: 0.0,
            m_w: vec![0.0; n_embd],
            m_b: 0.0,
            v_w: vec![0.0; n_embd],
            v_b: 0.0,
            step: 0,
        }
    }

    /// Predict loss from layer activations: h(x) = w · mean(x) + b
    fn predict(&self, x: &[f32], t: usize, e: usize) -> f32 {
        let mut mean = vec![0.0f32; e];
        for i in 0..t {
            for j in 0..e {
                mean[j] += x[i * e + j];
            }
        }
        for j in 0..e {
            mean[j] /= t as f32;
        }
        let mut pred = self.bias;
        for j in 0..e {
            pred += self.weights[j] * mean[j];
        }
        pred
    }

    /// Get the error signal: ∂h/∂x — approximates ∂L/∂x_{l+1}
    /// Returns a (T*E) vector: gradient of critic w.r.t. layer output
    fn error_signal(&self, t: usize, e: usize) -> Vec<f32> {
        let mut signal = vec![0.0f32; t * e];
        let scale = 1.0 / t as f32;
        for i in 0..t {
            for j in 0..e {
                signal[i * e + j] = self.weights[j] * scale;
            }
        }
        signal
    }

    /// Update critic to better predict loss from activations
    fn update(&mut self, x: &[f32], t: usize, e: usize, actual_loss: f32, lr: f32) {
        let pred = self.predict(x, t, e);
        let err = pred - actual_loss;

        // Compute mean activation
        let mut mean = vec![0.0f32; e];
        for i in 0..t {
            for j in 0..e {
                mean[j] += x[i * e + j];
            }
        }
        for j in 0..e {
            mean[j] /= t as f32;
        }

        self.step += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);

        // Adam update for weights
        for j in 0..e {
            let g = err * mean[j];
            self.m_w[j] = beta1 * self.m_w[j] + (1.0 - beta1) * g;
            self.v_w[j] = beta2 * self.v_w[j] + (1.0 - beta2) * g * g;
            let m_hat = self.m_w[j] / bc1;
            let v_hat = self.v_w[j] / bc2;
            self.weights[j] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        // Adam update for bias
        let g = err;
        self.m_b = beta1 * self.m_b + (1.0 - beta1) * g;
        self.v_b = beta2 * self.v_b + (1.0 - beta2) * g * g;
        let m_hat = self.m_b / bc1;
        let v_hat = self.v_b / bc2;
        self.bias -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ─── Coupled Cluster Trainer ─────────────────────────────────────────

struct CCTrainer {
    critics: Vec<LayerCritic>, // one per layer boundary (after each layer)
    output_critic: LayerCritic, // critic for the final output (before LM head)
}

impl CCTrainer {
    fn new(config: &Config) -> Self {
        let mut critics = Vec::new();
        for _ in 0..config.n_layer {
            critics.push(LayerCritic::new(config.n_embd));
        }
        Self {
            output_critic: LayerCritic::new(config.n_embd),
            critics,
        }
    }

    /// Forward pass collecting per-layer activations (layer boundaries)
    /// Returns: logits, loss, and activations[0..n_layer+1]
    /// activations[0] = post-embedding, activations[l+1] = output of layer l
    fn forward_collecting(
        model: &GPT,
        tokens: &[usize],
        targets: &[usize],
    ) -> (Vec<f32>, f32, Vec<Vec<f32>>, ForwardCache) {
        let cfg = &model.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;

        let (logits, cache) = model.forward_with_cache(tokens);

        // Collect layer boundary activations
        let mut activations = Vec::with_capacity(cfg.n_layer + 1);
        activations.push(cache.x_after_emb.clone()); // activation[0]
        for l in 0..cfg.n_layer {
            // Layer l's output = next layer's input (or x_before_final_ln for last)
            if l + 1 < cfg.n_layer {
                activations.push(cache.layer_caches[l + 1].x_input.clone());
            } else {
                activations.push(cache.x_before_final_ln.clone());
            }
        }

        // Compute loss
        let mut loss = 0.0f32;
        for i in 0..t {
            let offset = i * v;
            let max_val = logits[offset..offset + v]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v {
                sum += (logits[offset + j] - max_val).exp();
            }
            let log_sum = max_val + sum.ln();
            loss -= logits[offset + targets[i]] - log_sum;
        }
        loss /= t as f32;

        (logits, loss, activations, cache)
    }

    /// Compute T₁ gradient for layer l (Singles — using critic error signal)
    /// This does ONE layer of backward pass using the critic's error signal
    /// instead of the true backprop signal from downstream.
    fn compute_t1_for_layer(
        model: &GPT,
        l: usize,
        cache: &ForwardCache,
        critic_signal: &[f32], // ∂h/∂x_{l+1} from critic
    ) -> Gradients {
        let cfg = &model.config;
        let t = cache.tokens.len();
        let e = cfg.n_embd;
        let inner = 4 * e;
        let nh = cfg.n_head;
        let hs = e / nh;

        let mut grads = Gradients::zero_like(cfg);
        let lc = &cache.layer_caches[l];

        // dx = critic_signal (this replaces the backprop signal from downstream)
        let dx = critic_signal.to_vec();

        // ── FF backward ──
        let d_ff_out = dx.clone();
        for i in 0..t {
            for j in 0..e {
                grads.ff_b2[l][j] += d_ff_out[i * e + j];
            }
        }
        let d_ff_hidden = matmul_backward_both(
            &lc.ff_post_gelu, &model.ff_w2[l], &d_ff_out, t, inner, e,
            &mut grads.ff_w2[l],
        );
        let d_ff_pre_gelu = gelu_backward(&lc.ff_pre_gelu, &d_ff_hidden);
        for i in 0..t {
            for j in 0..inner {
                grads.ff_b1[l][j] += d_ff_pre_gelu[i * inner + j];
            }
        }
        let d_ln2_out = matmul_backward_both(
            &lc.ln2_out, &model.ff_w1[l], &d_ff_pre_gelu, t, e, inner,
            &mut grads.ff_w1[l],
        );
        let d_from_ln2 = layer_norm_backward(
            &lc.x_after_attn_residual, &d_ln2_out, &model.ln2_gamma[l],
            t, e, &mut grads.ln2_gamma[l], &mut grads.ln2_beta[l],
        );

        // dx_mid = dx (residual) + d_from_ln2 (FF path)
        let mut dx_mid = dx;
        for i in 0..t * e {
            dx_mid[i] += d_from_ln2[i];
        }

        // ── Attention backward ──
        let d_proj_out = dx_mid.clone();
        let d_attn_out = matmul_backward_both(
            &lc.attn_out_pre_proj, &model.attn_proj[l], &d_proj_out,
            t, e, e, &mut grads.attn_proj[l],
        );

        let mut d_qkv = vec![0.0f32; t * 3 * e];
        for h in 0..nh {
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
            // d_V
            for i in 0..t {
                for k in 0..hs {
                    let d_out = d_attn_out[i * e + h * hs + k];
                    for j in 0..t {
                        let w = lc.attn_weights[h * t * t + i * t + j];
                        d_qkv[j * 3 * e + 2 * e + h * hs + k] += w * d_out;
                    }
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
            // d_Q, d_K
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
        let _d_ln1_out = matmul_backward_both(
            &lc.ln1_out, &model.qkv_w[l], &d_qkv, t, e, 3 * e,
            &mut grads.qkv_w[l],
        );
        let _d_from_ln1 = layer_norm_backward(
            &lc.x_input, &_d_ln1_out, &model.ln1_gamma[l],
            t, e, &mut grads.ln1_gamma[l], &mut grads.ln1_beta[l],
        );

        grads
    }

    /// Compute T₂ correction for layer l (Doubles — exact 2-layer backprop)
    /// Uses exact backprop through layers l AND l+1, with critic signal at l+2.
    /// T₂ = (2-layer exact) - T₁
    fn compute_t2_for_layer(
        model: &GPT,
        l: usize,
        cache: &ForwardCache,
        critic_signal_l2: &[f32], // signal at layer l+2 boundary
    ) -> Gradients {
        let cfg = &model.config;
        let t = cache.tokens.len();
        let e = cfg.n_embd;
        let inner = 4 * e;
        let nh = cfg.n_head;
        let hs = e / nh;

        let mut grads = Gradients::zero_like(cfg);

        // First: backprop through layer l+1 to get the error signal at l+1
        let l1 = l + 1;
        let lc1 = &cache.layer_caches[l1];
        let mut dx1 = critic_signal_l2.to_vec();

        // ── Layer l+1 FF backward ──
        let d_ff_out1 = dx1.clone();
        let d_ff_hidden1 = matmul_backward_both(
            &lc1.ff_post_gelu, &model.ff_w2[l1], &d_ff_out1, t, inner, e,
            &mut grads.ff_w2[l1],
        );
        let d_ff_pre_gelu1 = gelu_backward(&lc1.ff_pre_gelu, &d_ff_hidden1);
        let d_ln2_out1 = matmul_backward_both(
            &lc1.ln2_out, &model.ff_w1[l1], &d_ff_pre_gelu1, t, e, inner,
            &mut grads.ff_w1[l1],
        );
        let d_from_ln2_1 = layer_norm_backward(
            &lc1.x_after_attn_residual, &d_ln2_out1, &model.ln2_gamma[l1],
            t, e, &mut grads.ln2_gamma[l1], &mut grads.ln2_beta[l1],
        );
        let mut dx_mid1 = dx1;
        for i in 0..t * e { dx_mid1[i] += d_from_ln2_1[i]; }

        // ── Layer l+1 Attention backward ──
        let d_proj_out1 = dx_mid1.clone();
        let d_attn_out1 = matmul_backward_both(
            &lc1.attn_out_pre_proj, &model.attn_proj[l1], &d_proj_out1,
            t, e, e, &mut grads.attn_proj[l1],
        );
        let mut d_qkv1 = vec![0.0f32; t * 3 * e];
        for h in 0..nh {
            let mut d_score1 = vec![0.0f32; t * t];
            for i in 0..t {
                for j in 0..t {
                    let mut dw = 0.0f32;
                    for k in 0..hs {
                        let vj = lc1.qkv[j * 3 * e + 2 * e + h * hs + k];
                        dw += vj * d_attn_out1[i * e + h * hs + k];
                    }
                    d_score1[i * t + j] = dw;
                }
            }
            for i in 0..t {
                for k in 0..hs {
                    let d_out = d_attn_out1[i * e + h * hs + k];
                    for j in 0..t {
                        let w = lc1.attn_weights[h * t * t + i * t + j];
                        d_qkv1[j * 3 * e + 2 * e + h * hs + k] += w * d_out;
                    }
                }
            }
            for i in 0..t {
                let mut dot = 0.0f32;
                for j in 0..t {
                    dot += d_score1[i * t + j]
                        * lc1.attn_weights[h * t * t + i * t + j];
                }
                for j in 0..t {
                    let w = lc1.attn_weights[h * t * t + i * t + j];
                    d_score1[i * t + j] = w * (d_score1[i * t + j] - dot);
                }
            }
            let scale = 1.0 / (hs as f32).sqrt();
            for val in d_score1.iter_mut() { *val *= scale; }
            for i in 0..t {
                for j in 0..=i {
                    let ds = d_score1[i * t + j];
                    if ds.abs() > 1e-12 {
                        for k in 0..hs {
                            let qi = lc1.qkv[i * 3 * e + h * hs + k];
                            let kj = lc1.qkv[j * 3 * e + e + h * hs + k];
                            d_qkv1[i * 3 * e + h * hs + k] += ds * kj;
                            d_qkv1[j * 3 * e + e + h * hs + k] += ds * qi;
                        }
                    }
                }
            }
        }
        let _d_ln1_out1 = matmul_backward_both(
            &lc1.ln1_out, &model.qkv_w[l1], &d_qkv1, t, e, 3 * e,
            &mut grads.qkv_w[l1],
        );
        let d_from_ln1_1 = layer_norm_backward(
            &lc1.x_input, &_d_ln1_out1, &model.ln1_gamma[l1],
            t, e, &mut grads.ln1_gamma[l1], &mut grads.ln1_beta[l1],
        );

        // Error signal arriving at layer l's output = dx_mid1 (residual) + d_from_ln1_1
        let mut dx_at_l = dx_mid1;
        for i in 0..t * e { dx_at_l[i] += d_from_ln1_1[i]; }

        // Now: backprop through layer l using this EXACT error signal (not critic!)
        // This is the same as compute_t1 but with the exact 2-layer error signal
        let lc = &cache.layer_caches[l];
        let d_ff_out = dx_at_l.clone();
        for i in 0..t {
            for j in 0..e {
                grads.ff_b2[l][j] += d_ff_out[i * e + j];
            }
        }
        let d_ff_hidden = matmul_backward_both(
            &lc.ff_post_gelu, &model.ff_w2[l], &d_ff_out, t, inner, e,
            &mut grads.ff_w2[l],
        );
        let d_ff_pre = gelu_backward(&lc.ff_pre_gelu, &d_ff_hidden);
        for i in 0..t {
            for j in 0..inner {
                grads.ff_b1[l][j] += d_ff_pre[i * inner + j];
            }
        }
        let d_ln2 = matmul_backward_both(
            &lc.ln2_out, &model.ff_w1[l], &d_ff_pre, t, e, inner,
            &mut grads.ff_w1[l],
        );
        let d_from_ln2 = layer_norm_backward(
            &lc.x_after_attn_residual, &d_ln2, &model.ln2_gamma[l],
            t, e, &mut grads.ln2_gamma[l], &mut grads.ln2_beta[l],
        );
        let mut dx_m = dx_at_l;
        for i in 0..t * e { dx_m[i] += d_from_ln2[i]; }

        let d_proj = dx_m.clone();
        let d_attn = matmul_backward_both(
            &lc.attn_out_pre_proj, &model.attn_proj[l], &d_proj,
            t, e, e, &mut grads.attn_proj[l],
        );
        let mut d_qkv = vec![0.0f32; t * 3 * e];
        for h in 0..nh {
            let mut d_sc = vec![0.0f32; t * t];
            for i in 0..t {
                for j in 0..t {
                    let mut dw = 0.0f32;
                    for k in 0..hs {
                        dw += lc.qkv[j * 3 * e + 2 * e + h * hs + k]
                            * d_attn[i * e + h * hs + k];
                    }
                    d_sc[i * t + j] = dw;
                }
            }
            for i in 0..t {
                for k in 0..hs {
                    let d_out = d_attn[i * e + h * hs + k];
                    for j in 0..t {
                        d_qkv[j * 3 * e + 2 * e + h * hs + k] +=
                            lc.attn_weights[h * t * t + i * t + j] * d_out;
                    }
                }
            }
            for i in 0..t {
                let mut dot = 0.0f32;
                for j in 0..t {
                    dot += d_sc[i * t + j] * lc.attn_weights[h * t * t + i * t + j];
                }
                for j in 0..t {
                    let w = lc.attn_weights[h * t * t + i * t + j];
                    d_sc[i * t + j] = w * (d_sc[i * t + j] - dot);
                }
            }
            let scale = 1.0 / (hs as f32).sqrt();
            for v in d_sc.iter_mut() { *v *= scale; }
            for i in 0..t {
                for j in 0..=i {
                    let ds = d_sc[i * t + j];
                    if ds.abs() > 1e-12 {
                        for k in 0..hs {
                            d_qkv[i * 3 * e + h * hs + k] +=
                                ds * lc.qkv[j * 3 * e + e + h * hs + k];
                            d_qkv[j * 3 * e + e + h * hs + k] +=
                                ds * lc.qkv[i * 3 * e + h * hs + k];
                        }
                    }
                }
            }
        }
        let d_ln1 = matmul_backward_both(
            &lc.ln1_out, &model.qkv_w[l], &d_qkv, t, e, 3 * e,
            &mut grads.qkv_w[l],
        );
        let _d_from_ln1 = layer_norm_backward(
            &lc.x_input, &d_ln1, &model.ln1_gamma[l],
            t, e, &mut grads.ln1_gamma[l], &mut grads.ln1_beta[l],
        );

        grads
    }

    /// Compute gradient using CC hierarchy and also return exact backprop for comparison
    fn cc_step(
        &mut self,
        model: &GPT,
        tokens: &[usize],
        targets: &[usize],
        include_t2: bool,
    ) -> (f32, Gradients, Gradients) {
        let cfg = &model.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nl = cfg.n_layer;

        // Forward pass collecting activations
        let (_logits, loss, activations, cache) = Self::forward_collecting(model, tokens, targets);

        // Update all critics with actual loss
        for l in 0..nl {
            self.critics[l].update(&activations[l + 1], t, e, loss, 0.01);
        }
        self.output_critic.update(&cache.x_before_final_ln, t, e, loss, 0.01);

        // ── Compute exact backprop gradient (for comparison) ──
        let (_, exact_grads) = model.forward_backward(tokens, targets);

        // ── Compute CC gradient ──
        let mut cc_grads = Gradients::zero_like(cfg);

        // For the last layer, use exact output gradient (LM head → final LN → last layer)
        // This is the "reference" part — we know the output exactly
        // Compute d_logits
        let mut probs = vec![0.0f32; t * v];
        for i in 0..t {
            let offset = i * v;
            let max_val = _logits[offset..offset + v]
                .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..v {
                probs[offset + j] = (_logits[offset + j] - max_val).exp();
                sum += probs[offset + j];
            }
            for j in 0..v { probs[offset + j] /= sum; }
        }
        let mut d_logits = probs;
        for i in 0..t {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v { d_logits[i * v + j] /= t as f32; }
        }

        // LM head gradient (exact for all methods)
        let d_ln_out = matmul_backward_both(
            &cache.x_after_final_ln, &model.lm_head, &d_logits,
            t, e, v, &mut cc_grads.lm_head,
        );
        // Final LN backward
        let dx_final = layer_norm_backward(
            &cache.x_before_final_ln, &d_ln_out, &model.ln_f_gamma,
            t, e, &mut cc_grads.ln_f_gamma, &mut cc_grads.ln_f_beta,
        );

        // Embedding gradients (exact — just from d_x_after_emb which we'll get from layer 0)
        // We'll accumulate these from the layer backward passes

        // ── T₁: Singles for each layer ──
        // For the last layer (l = nl-1): use exact dx_final as the signal
        // For earlier layers: use critic error signal
        for l in 0..nl {
            let signal = if l == nl - 1 {
                // Last layer gets exact output error
                dx_final.clone()
            } else {
                // Earlier layers get critic error signal
                self.critics[l].error_signal(t, e)
            };

            let t1_grads = Self::compute_t1_for_layer(model, l, &cache, &signal);

            // Accumulate T₁ into cc_grads
            add_vecs(&mut cc_grads.qkv_w[l], &t1_grads.qkv_w[l]);
            add_vecs(&mut cc_grads.attn_proj[l], &t1_grads.attn_proj[l]);
            add_vecs(&mut cc_grads.ff_w1[l], &t1_grads.ff_w1[l]);
            add_vecs(&mut cc_grads.ff_w2[l], &t1_grads.ff_w2[l]);
            add_vecs(&mut cc_grads.ff_b1[l], &t1_grads.ff_b1[l]);
            add_vecs(&mut cc_grads.ff_b2[l], &t1_grads.ff_b2[l]);
            add_vecs(&mut cc_grads.ln1_gamma[l], &t1_grads.ln1_gamma[l]);
            add_vecs(&mut cc_grads.ln1_beta[l], &t1_grads.ln1_beta[l]);
            add_vecs(&mut cc_grads.ln2_gamma[l], &t1_grads.ln2_gamma[l]);
            add_vecs(&mut cc_grads.ln2_beta[l], &t1_grads.ln2_beta[l]);
        }

        // ── T₂: Doubles correction for adjacent pairs ──
        if include_t2 && nl >= 2 {
            for l in 0..(nl - 1) {
                // Signal at l+2 boundary
                let signal_l2 = if l + 1 == nl - 1 {
                    // layer l+1 is the last layer — use exact dx_final
                    dx_final.clone()
                } else {
                    // use critic at l+1
                    self.critics[l + 1].error_signal(t, e)
                };

                let t2_full = Self::compute_t2_for_layer(model, l, &cache, &signal_l2);

                // T₂ correction = (2-layer exact) - T₁ (already accumulated)
                // We add the 2-layer result and subtract what T₁ already contributed
                // For simplicity: just replace T₁ with the better T₂ estimate for layer l
                // T₂_correction = t2_full - t1_for_layer_l
                let t1_signal = self.critics[l].error_signal(t, e);
                let t1_grads = Self::compute_t1_for_layer(model, l, &cache, &t1_signal);

                // Apply correction: cc_grads += (t2_full - t1_grads) for layer l
                sub_add_layer_grads(&mut cc_grads, &t2_full, &t1_grads, l);
            }
        }

        // Token/pos embedding grads — use exact for simplicity
        // (these are shared across all layers and small)
        cc_grads.token_emb = exact_grads.token_emb.clone();
        cc_grads.pos_emb = exact_grads.pos_emb.clone();

        (loss, cc_grads, exact_grads)
    }
}

/// Replace layer l's grads: cc[l] += (new[l] - old[l])
fn sub_add_layer_grads(cc: &mut Gradients, new: &Gradients, old: &Gradients, l: usize) {
    fn apply(dst: &mut [f32], add: &[f32], sub: &[f32]) {
        for i in 0..dst.len() {
            dst[i] += add[i] - sub[i];
        }
    }
    apply(&mut cc.qkv_w[l], &new.qkv_w[l], &old.qkv_w[l]);
    apply(&mut cc.attn_proj[l], &new.attn_proj[l], &old.attn_proj[l]);
    apply(&mut cc.ff_w1[l], &new.ff_w1[l], &old.ff_w1[l]);
    apply(&mut cc.ff_w2[l], &new.ff_w2[l], &old.ff_w2[l]);
    apply(&mut cc.ff_b1[l], &new.ff_b1[l], &old.ff_b1[l]);
    apply(&mut cc.ff_b2[l], &new.ff_b2[l], &old.ff_b2[l]);
    apply(&mut cc.ln1_gamma[l], &new.ln1_gamma[l], &old.ln1_gamma[l]);
    apply(&mut cc.ln1_beta[l], &new.ln1_beta[l], &old.ln1_beta[l]);
    apply(&mut cc.ln2_gamma[l], &new.ln2_gamma[l], &old.ln2_gamma[l]);
    apply(&mut cc.ln2_beta[l], &new.ln2_beta[l], &old.ln2_beta[l]);
}

/// Flatten gradients into a single vector for cosine similarity
fn flatten_grads(g: &Gradients, cfg: &Config) -> Vec<f32> {
    let mut flat = Vec::new();
    for l in 0..cfg.n_layer {
        flat.extend_from_slice(&g.qkv_w[l]);
        flat.extend_from_slice(&g.attn_proj[l]);
        flat.extend_from_slice(&g.ff_w1[l]);
        flat.extend_from_slice(&g.ff_w2[l]);
        flat.extend_from_slice(&g.ff_b1[l]);
        flat.extend_from_slice(&g.ff_b2[l]);
        flat.extend_from_slice(&g.ln1_gamma[l]);
        flat.extend_from_slice(&g.ln1_beta[l]);
        flat.extend_from_slice(&g.ln2_gamma[l]);
        flat.extend_from_slice(&g.ln2_beta[l]);
    }
    flat.extend_from_slice(&g.ln_f_gamma);
    flat.extend_from_slice(&g.ln_f_beta);
    flat.extend_from_slice(&g.lm_head);
    flat
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

fn add_vecs(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

// ─── Experiment Runner ───────────────────────────────────────────────

fn train_with_method(
    config: &Config,
    train_data: &[ToolExample],
    val_data: &[ToolExample],
    tok: &Tokenizer,
    pretrained_weights: &[f32],
    method: &str, // "backprop", "ccs", "ccsd"
    sft_steps: usize,
) -> (f32, f32, f32, Vec<(usize, f32, f32)>) {
    // Create model from pretrained weights
    let mut model = GPT::new(config.clone());
    crate::perturbation::apply_flat_params(&mut model, pretrained_weights);

    let mut rng = rand::thread_rng();
    let batch_size = 8;
    let lr = 0.001;
    let e = config.n_embd;

    let mut cc_trainer = CCTrainer::new(config);
    let mut cos_log: Vec<(usize, f32, f32)> = Vec::new(); // (step, loss, cos_sim)

    let start = Instant::now();

    for step in 0..sft_steps {
        // Sample a batch
        let batch_indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..train_data.len()))
            .collect();

        let mut total_loss = 0.0f32;
        let mut batch_cc_grads = Gradients::zero_like(config);
        let mut avg_cos = 0.0f32;
        let mut n_samples = 0;

        for &idx in &batch_indices {
            let tokens = tok.encode(&train_data[idx].input);
            if tokens.len() < 3 { continue; }
            let ctx_len = tokens.len().min(config.block_size);
            let input_tokens = &tokens[..ctx_len - 1];
            let target_tokens = &tokens[1..ctx_len];

            match method {
                "backprop" => {
                    let (loss, grads) = model.forward_backward(input_tokens, target_tokens);
                    if loss.is_finite() {
                        total_loss += loss;
                        batch_cc_grads.accumulate(&grads);
                        n_samples += 1;
                    }
                }
                "ccs" | "ccsd" => {
                    let include_t2 = method == "ccsd";
                    let (loss, cc_grads, exact_grads) = cc_trainer.cc_step(
                        &model, input_tokens, target_tokens, include_t2,
                    );
                    if loss.is_finite() {
                        total_loss += loss;
                        batch_cc_grads.accumulate(&cc_grads);

                        // Measure gradient quality
                        let cc_flat = flatten_grads(&cc_grads, config);
                        let ex_flat = flatten_grads(&exact_grads, config);
                        let cos = cosine_similarity(&cc_flat, &ex_flat);
                        avg_cos += cos;
                        n_samples += 1;
                    }
                }
                _ => panic!("unknown method"),
            }
        }

        if n_samples == 0 { continue; }
        batch_cc_grads.scale(1.0 / n_samples as f32);
        let avg_loss = total_loss / n_samples as f32;
        avg_cos /= n_samples as f32;

        model.apply_gradients(&batch_cc_grads, lr);

        if step % 100 == 0 || step == sft_steps - 1 {
            let elapsed = start.elapsed().as_secs_f32();
            if method == "backprop" {
                println!("  {} step {:4} | Loss: {:.4} | {:.1}s",
                    method, step, avg_loss, elapsed);
            } else {
                println!("  {} step {:4} | Loss: {:.4} | cos(CC,exact): {:.4} | {:.1}s",
                    method, step, avg_loss, avg_cos, elapsed);
            }
        }
        cos_log.push((step, avg_loss, avg_cos));
    }

    let train_time = start.elapsed().as_secs_f32();

    // Evaluate
    let metrics = evaluate_model(&model, config, val_data, tok);
    let composite = 0.25 * (metrics.0 + metrics.1 + metrics.2 + metrics.3);

    // Average cosine similarity over last 100 steps
    let last_cos: f32 = cos_log.iter().rev().take(100)
        .map(|&(_, _, c)| c).sum::<f32>()
        / cos_log.iter().rev().take(100).count().max(1) as f32;

    (composite, train_time, last_cos, cos_log)
}

fn evaluate_model(
    model: &GPT,
    _config: &Config,
    val_data: &[ToolExample],
    tok: &Tokenizer,
) -> (f32, f32, f32, f32) {
    let mut format_sum = 0.0f32;
    let mut tool_sum = 0.0f32;
    let mut param_sum = 0.0f32;
    let mut reply_sum = 0.0f32;
    let mut count = 0;

    for example in val_data {
        let prompt_tokens = tok.encode(&example.prompt);
        let generated = model.generate(&prompt_tokens, 150);
        let output_text = tok.decode(&generated[prompt_tokens.len()..]);
        let metrics = tool_data::evaluate_output(&output_text, example);
        format_sum += if metrics.format_correct { 1.0 } else { 0.0 };
        tool_sum += if metrics.tool_correct { 1.0 } else { 0.0 };
        param_sum += if metrics.params_correct { 1.0 } else { 0.0 };
        reply_sum += metrics.reply_quality;
        count += 1;
    }

    let n = count as f32;
    (format_sum / n, tool_sum / n, param_sum / n, reply_sum / n)
}

pub fn run_coupled_cluster_experiment() {
    println!("=== Coupled Cluster Weight Updates Experiment ===\n");
    println!("PHYSICS: Coupled Cluster theory decomposes electron correlation");
    println!("into a systematic hierarchy: T₁ (singles) + T₂ (doubles) + ...");
    println!("CCSD (T₁+T₂) captures ~95%% of correlation energy.\n");
    println!("NEURAL NET: Decompose weight gradients into:");
    println!("  T₁ = per-layer gradient using learned linear critic (no backprop)");
    println!("  T₂ = exact 2-layer backprop correction (pairwise nonlinearity)");
    println!("  CCSD = T₁ + T₂\n");
    println!("KEY METRIC: cosine similarity cos(∇L_CC, ∇L_backprop)\n");

    // Generate data
    println!("[1/5] Generating dataset...");
    let mut rng = rand::thread_rng();
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    let base_text = data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());
    println!("  Vocabulary size: {}", tok.vocab_size());

    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size: 48,
    };

    // Pretrain on Shakespeare (shared across all methods)
    println!("\n[2/5] Pretraining on Shakespeare (shared)...");
    let mut model = GPT::new(config.clone());
    let encoded = tok.encode(base_text);
    let pretrain_steps = 1000;
    let batch_size = 16;
    let lr = 0.001;

    for step in 0..pretrain_steps {
        let (inputs, targets) = data::create_batches(&encoded, config.block_size, batch_size, &mut rng);
        let results: Vec<(f32, Gradients)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let ctx = inputs[b].len().min(config.block_size);
                model.forward_backward(&inputs[b][..ctx], &targets[b][..ctx])
            })
            .collect();

        let mut total_loss = 0.0f32;
        let mut grads = Gradients::zero_like(&config);
        for (loss, g) in results {
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, lr);

        if step % 200 == 0 {
            println!("  Pretrain step {:4} | Loss: {:.4}", step, total_loss / batch_size as f32);
        }
    }
    println!("  Pretrain done.");

    // Save pretrained weights
    let pretrained = crate::perturbation::flatten_params(&model);

    let sft_steps = 800;

    // [3] Backprop baseline
    println!("\n[3/5] Training with BACKPROP (baseline)...");
    let (bp_comp, bp_time, _, _) = train_with_method(
        &config, &train_data, &val_data, &tok, &pretrained, "backprop", sft_steps,
    );
    println!("  Backprop composite: {:.4} ({:.1}s)\n", bp_comp, bp_time);

    // [4] CCS (T₁ only)
    println!("[4/5] Training with CCS (T₁ only — critic-based, no backprop for layer grads)...");
    let (ccs_comp, ccs_time, ccs_cos, ccs_log) = train_with_method(
        &config, &train_data, &val_data, &tok, &pretrained, "ccs", sft_steps,
    );
    println!("  CCS composite: {:.4}, avg cos: {:.4} ({:.1}s)\n", ccs_comp, ccs_cos, ccs_time);

    // [5] CCSD (T₁ + T₂)
    println!("[5/5] Training with CCSD (T₁ + T₂ — critic + 2-layer backprop correction)...");
    let (ccsd_comp, ccsd_time, ccsd_cos, ccsd_log) = train_with_method(
        &config, &train_data, &val_data, &tok, &pretrained, "ccsd", sft_steps,
    );
    println!("  CCSD composite: {:.4}, avg cos: {:.4} ({:.1}s)\n", ccsd_comp, ccsd_cos, ccsd_time);

    // Results summary
    println!("======================================================================");
    println!("=== COUPLED CLUSTER RESULTS ===\n");
    println!("  Method      Composite  cos(CC,exact)  Time");
    println!("  -------------------------------------------");
    println!("  Backprop    {:.4}      1.0000         {:.1}s", bp_comp, bp_time);
    println!("  CCS (T₁)   {:.4}      {:.4}         {:.1}s", ccs_comp, ccs_cos, ccs_time);
    println!("  CCSD(T₁+T₂){:.4}      {:.4}         {:.1}s", ccsd_comp, ccsd_cos, ccsd_time);
    println!();
    println!("  CCS quality:  {:.1}% of backprop", ccs_comp / bp_comp.max(0.001) * 100.0);
    println!("  CCSD quality: {:.1}% of backprop", ccsd_comp / bp_comp.max(0.001) * 100.0);
    println!("  T₂ improvement over T₁: {:.1}%",
        (ccsd_comp - ccs_comp) / ccs_comp.max(0.001) * 100.0);
    println!("======================================================================");

    // Write CSV
    let _ = std::fs::create_dir_all("experiments");
    if let Ok(mut f) = std::fs::File::create("experiments/coupled_cluster_results.csv") {
        let _ = writeln!(f, "method,composite,cos_similarity,train_time");
        let _ = writeln!(f, "backprop,{:.4},1.0000,{:.2}", bp_comp, bp_time);
        let _ = writeln!(f, "ccs,{:.4},{:.4},{:.2}", ccs_comp, ccs_cos, ccs_time);
        let _ = writeln!(f, "ccsd,{:.4},{:.4},{:.2}", ccsd_comp, ccsd_cos, ccsd_time);
    }

    // Write cosine similarity log
    if let Ok(mut f) = std::fs::File::create("experiments/cc_cosine_log.csv") {
        let _ = writeln!(f, "method,step,loss,cos_similarity");
        for &(step, loss, cos) in &ccs_log {
            let _ = writeln!(f, "ccs,{},{:.4},{:.4}", step, loss, cos);
        }
        for &(step, loss, cos) in &ccsd_log {
            let _ = writeln!(f, "ccsd,{},{:.4},{:.4}", step, loss, cos);
        }
    }

    println!("\nResults saved to experiments/coupled_cluster_results.csv");
    println!("Cosine log saved to experiments/cc_cosine_log.csv");
}
