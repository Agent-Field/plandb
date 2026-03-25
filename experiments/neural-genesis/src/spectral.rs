//! Spectral Gating Attention — FFT-based O(n log n) attention replacement.
//!
//! Architecture: For each head, replace Q*K^T*V softmax attention with
//! a learned causal convolution implemented via FFT:
//!   1. Project input to head dimension: x_h = x @ W_h (n_embd -> head_size)
//!   2. For each feature f:
//!      a. Extract column signal[i] = x_h[i, f]
//!      b. Pad signal and causal_kernel to padded_len
//!      c. FFT both, element-wise multiply, iFFT
//!      d. Take first T values (guaranteed causal since kernel[k]=0 for k<0)
//!   3. Concatenate heads and project output: @ W_out
//!
//! Causality: The kernel h[0..block_size-1] is a time-domain FIR filter.
//! By construction it is causal: output[t] = sum_{k=0..block_size-1} h[k]*x[t-k].
//! The FFT is just an efficient way to compute this convolution.
//!
//! The kernel is zero-padded to padded_len (= 2 * block_size, next power of 2)
//! to avoid circular wrap-around. This makes the FFT convolution equivalent
//! to a linear (causal) convolution, consistent between training and generation.
//!
//! Pure Rust, no dependencies beyond `rand`.

use rand::Rng;
use std::io::{Read, Write};

use crate::model::Config;

// ─── FFT Implementation (Cooley-Tukey radix-2 DIT) ──────────────────

fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

fn bit_reverse(data_re: &mut [f32], data_im: &mut [f32]) {
    let n = data_re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 { j ^= bit; bit >>= 1; }
        j ^= bit;
        if i < j { data_re.swap(i, j); data_im.swap(i, j); }
    }
}

fn fft_inplace(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    bit_reverse(re, im);
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f32::consts::PI / len as f32;
        for i in (0..n).step_by(len) {
            for k in 0..half {
                let theta = angle * k as f32;
                let wr = theta.cos();
                let wi = theta.sin();
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + half] * wr - im[i + k + half] * wi;
                let v_im = re[i + k + half] * wi + im[i + k + half] * wr;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + half] = u_re - v_re;
                im[i + k + half] = u_im - v_im;
            }
        }
        len <<= 1;
    }
}

fn ifft_inplace(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    for v in im.iter_mut() { *v = -*v; }
    fft_inplace(re, im);
    let inv_n = 1.0 / n as f32;
    for v in re.iter_mut() { *v *= inv_n; }
    for v in im.iter_mut() { *v = -*v * inv_n; }
}

fn fft(signal: &[f32], padded_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut re = vec![0.0f32; padded_len];
    let mut im = vec![0.0f32; padded_len];
    for (i, &v) in signal.iter().enumerate() { re[i] = v; }
    fft_inplace(&mut re, &mut im);
    (re, im)
}

fn ifft_real(re_in: &[f32], im_in: &[f32], orig_len: usize) -> Vec<f32> {
    let mut re = re_in.to_vec();
    let mut im = im_in.to_vec();
    ifft_inplace(&mut re, &mut im);
    re.truncate(orig_len);
    re
}

// ─── Helper: randn ──────────────────────────────────────────────────

fn randn_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1: f32 = rng.r#gen::<f32>().max(1e-10);
        let u2: f32 = rng.r#gen::<f32>();
        let mag = (-2.0 * u1.ln()).sqrt() * scale;
        data.push(mag * (2.0 * std::f32::consts::PI * u2).cos());
        if i + 1 < n { data.push(mag * (2.0 * std::f32::consts::PI * u2).sin()); }
        i += 2;
    }
    data.truncate(n);
    data
}

// ─── SpectralGPT Model ─────────────────────────────────────────────

/// Spectral GPT with causal time-domain kernels.
///
/// Parameters per head per layer:
/// - head_proj_w: (n_embd, head_size) projection
/// - causal_kernel: (head_size, kernel_len) time-domain causal FIR filter
///   kernel_len = block_size (max lookback)
///
/// Forward per feature f:
///   signal = proj[:, f]  (length T)
///   kernel = causal_kernel[f, :]  (length kernel_len)
///   -- zero-pad both to padded_len (2 * block_size, next power of 2)
///   H = FFT(kernel)
///   X = FFT(signal)
///   Y = iFFT(H * X)[:T]  <- causal linear convolution
pub struct SpectralGPT {
    pub token_emb: Vec<f32>,   // (vocab_size, n_embd)
    pub pos_emb: Vec<f32>,     // (block_size, n_embd)

    pub ln1_gamma: Vec<Vec<f32>>,  // [n_layer][n_embd]
    pub ln1_beta: Vec<Vec<f32>>,

    // Spectral attention
    pub head_proj_w: Vec<Vec<Vec<f32>>>,   // [n_layer][n_head][n_embd * head_size]
    pub causal_kernel: Vec<Vec<Vec<f32>>>, // [n_layer][n_head][head_size * kernel_len]
    pub attn_proj: Vec<Vec<f32>>,          // [n_layer][n_embd * n_embd]

    // Feed-forward (same as standard GPT)
    pub ln2_gamma: Vec<Vec<f32>>,
    pub ln2_beta: Vec<Vec<f32>>,
    pub ff_w1: Vec<Vec<f32>>,
    pub ff_b1: Vec<Vec<f32>>,
    pub ff_w2: Vec<Vec<f32>>,
    pub ff_b2: Vec<Vec<f32>>,

    pub ln_f_gamma: Vec<f32>,
    pub ln_f_beta: Vec<f32>,
    pub lm_head: Vec<f32>,

    pub config: Config,
    pub kernel_len: usize,   // = block_size
    pub padded_len: usize,   // = next_power_of_2(2 * block_size) for linear convolution

    // Adam state
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub t_step: usize,
}

impl SpectralGPT {
    pub fn new(config: Config) -> Self {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let bs = config.block_size;
        let nh = config.n_head;
        let hs = e / nh;
        let inner = 4 * e;
        let kernel_len = bs;
        // For linear (non-circular) convolution: pad to at least signal_len + kernel_len - 1
        let padded = next_power_of_2(2 * bs);

        let emb_scale = 0.02;
        let layer_scale = (0.02 / (nl as f32).sqrt()).max(0.001);

        let mut ln1_gamma = Vec::new();
        let mut ln1_beta = Vec::new();
        let mut head_proj_w = Vec::new();
        let mut causal_kernel = Vec::new();
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

            let mut layer_projs = Vec::new();
            let mut layer_kernels = Vec::new();
            for _ in 0..nh {
                layer_projs.push(randn_vec(e * hs, layer_scale));

                // Initialize kernel as delta function: h[0] = 1, h[k>0] = small noise
                // This makes the initial output = input (identity mapping)
                let mut kernel = randn_vec(hs * kernel_len, 0.01);
                // Set h[0] = 1 for each feature (identity)
                for f in 0..hs {
                    kernel[f * kernel_len] = 1.0;
                }
                layer_kernels.push(kernel);
            }
            head_proj_w.push(layer_projs);
            causal_kernel.push(layer_kernels);

            attn_proj.push(randn_vec(e * e, layer_scale));
            ln2_gamma.push(vec![1.0; e]);
            ln2_beta.push(vec![0.0; e]);
            ff_w1.push(randn_vec(e * inner, layer_scale));
            ff_b1.push(vec![0.0; inner]);
            ff_w2.push(randn_vec(inner * e, layer_scale * 0.5));
            ff_b2.push(vec![0.0; e]);
        }

        let mut model = SpectralGPT {
            token_emb: randn_vec(v * e, emb_scale),
            pos_emb: randn_vec(bs * e, emb_scale),
            ln1_gamma, ln1_beta,
            head_proj_w,
            causal_kernel,
            attn_proj,
            ln2_gamma, ln2_beta,
            ff_w1, ff_b1, ff_w2, ff_b2,
            ln_f_gamma: vec![1.0; e],
            ln_f_beta: vec![0.0; e],
            lm_head: randn_vec(e * v, emb_scale),
            config,
            kernel_len,
            padded_len: padded,
            m: Vec::new(),
            v: Vec::new(),
            t_step: 0,
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
        let nh = self.config.n_head;
        for l in 0..self.config.n_layer {
            sizes.push(self.ln1_gamma[l].len());
            sizes.push(self.ln1_beta[l].len());
            for h in 0..nh {
                sizes.push(self.head_proj_w[l][h].len());
                sizes.push(self.causal_kernel[l][h].len());
            }
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

    fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<f32>, SpectralCache) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = e / nh;
        let kl = self.kernel_len;
        let pl = self.padded_len;

        // Embedding
        let mut x = vec![0.0f32; t * e];
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                x[i * e + j] = self.token_emb[tok * e + j] + self.pos_emb[i * e + j];
            }
        }

        let mut cache = SpectralCache {
            tokens: tokens.to_vec(),
            x_after_emb: x.clone(),
            layer_caches: Vec::new(),
            x_before_final_ln: Vec::new(),
            x_after_final_ln: Vec::new(),
        };

        for l in 0..cfg.n_layer {
            let mut lc = LayerCache::default();
            lc.x_input = x.clone();

            // LN1
            let (ln1_out, ln1_mean, ln1_rstd) = layer_norm(
                &x, &self.ln1_gamma[l], &self.ln1_beta[l], t, e,
            );
            lc.ln1_out = ln1_out.clone();
            lc.ln1_mean = ln1_mean;
            lc.ln1_rstd = ln1_rstd;

            // Spectral attention per head
            let mut attn_out = vec![0.0f32; t * e];
            let mut head_caches = Vec::new();

            for h in 0..nh {
                let mut hc = HeadCache::default();

                // Project: (T, E) @ (E, hs) -> (T, hs)
                let proj = matmul(&ln1_out, &self.head_proj_w[l][h], t, e, hs);
                hc.proj = proj.clone();

                let mut spectral_out = vec![0.0f32; t * hs];
                // Per-feature FFT-based causal convolution
                let mut sig_fft_re = vec![0.0f32; hs * pl];
                let mut sig_fft_im = vec![0.0f32; hs * pl];
                let mut kern_fft_re = vec![0.0f32; hs * pl];
                let mut kern_fft_im = vec![0.0f32; hs * pl];

                for f in 0..hs {
                    // Extract signal column
                    let mut signal = vec![0.0f32; t];
                    for i in 0..t { signal[i] = proj[i * hs + f]; }

                    // Extract kernel for this feature
                    let kernel = &self.causal_kernel[l][h][f * kl..(f + 1) * kl];

                    // FFT both (zero-padded to pl)
                    let (sre, sim) = fft(&signal, pl);
                    let (kre, kim) = fft(kernel, pl);

                    // Cache FFTs for backward
                    for k in 0..pl {
                        sig_fft_re[f * pl + k] = sre[k];
                        sig_fft_im[f * pl + k] = sim[k];
                        kern_fft_re[f * pl + k] = kre[k];
                        kern_fft_im[f * pl + k] = kim[k];
                    }

                    // Complex multiply H * X
                    let mut out_re = vec![0.0f32; pl];
                    let mut out_im = vec![0.0f32; pl];
                    for k in 0..pl {
                        out_re[k] = kre[k] * sre[k] - kim[k] * sim[k];
                        out_im[k] = kre[k] * sim[k] + kim[k] * sre[k];
                    }

                    // iFFT, take first T values
                    let time_out = ifft_real(&out_re, &out_im, t);
                    for i in 0..t {
                        spectral_out[i * hs + f] = time_out[i];
                    }
                }

                hc.sig_fft_re = sig_fft_re;
                hc.sig_fft_im = sig_fft_im;
                hc.kern_fft_re = kern_fft_re;
                hc.kern_fft_im = kern_fft_im;
                hc.spectral_out = spectral_out.clone();

                // Place into attn_out
                for i in 0..t {
                    for f in 0..hs {
                        attn_out[i * e + h * hs + f] = spectral_out[i * hs + f];
                    }
                }
                head_caches.push(hc);
            }
            lc.head_caches = head_caches;
            lc.attn_out = attn_out.clone();

            // Output projection
            let proj_out = matmul(&attn_out, &self.attn_proj[l], t, e, e);
            for i in 0..t * e { x[i] += proj_out[i]; }
            lc.x_after_attn = x.clone();

            // LN2
            let (ln2_out, ln2_mean, ln2_rstd) = layer_norm(
                &x, &self.ln2_gamma[l], &self.ln2_beta[l], t, e,
            );
            lc.ln2_out = ln2_out.clone();
            lc.ln2_mean = ln2_mean;
            lc.ln2_rstd = ln2_rstd;

            // FF
            let inner = 4 * e;
            let mut ff_h = matmul(&ln2_out, &self.ff_w1[l], t, e, inner);
            for i in 0..t {
                for j in 0..inner { ff_h[i * inner + j] += self.ff_b1[l][j]; }
            }
            lc.ff_pre_gelu = ff_h.clone();

            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            for val in ff_h.iter_mut() {
                let x3 = *val * *val * *val;
                let iv = sqrt_2_over_pi * (*val + 0.044715 * x3);
                *val = 0.5 * *val * (1.0 + iv.tanh());
            }
            lc.ff_post_gelu = ff_h.clone();

            let mut ff_out = matmul(&ff_h, &self.ff_w2[l], t, inner, e);
            for i in 0..t {
                for j in 0..e { ff_out[i * e + j] += self.ff_b2[l][j]; }
            }
            for i in 0..t * e { x[i] += ff_out[i]; }

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
    ) -> (f32, SpectralGradients) {
        let cfg = &self.config;
        let t = tokens.len();
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = e / nh;
        let kl = self.kernel_len;
        let pl = self.padded_len;

        let (logits, cache) = self.forward_with_cache(tokens);

        // Loss
        let mut probs = vec![0.0f32; t * v];
        let mut loss = 0.0f32;
        for i in 0..t {
            let off = i * v;
            let mx = logits[off..off + v].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0f32;
            for j in 0..v { probs[off + j] = (logits[off + j] - mx).exp(); s += probs[off + j]; }
            for j in 0..v { probs[off + j] /= s; }
            loss -= probs[off + targets[i]].max(1e-10).ln();
        }
        loss /= t as f32;

        let mut d_logits = probs;
        for i in 0..t {
            d_logits[i * v + targets[i]] -= 1.0;
            for j in 0..v { d_logits[i * v + j] /= t as f32; }
        }

        let mut grads = SpectralGradients::new(cfg, kl);

        // Backward lm_head
        let d_ln_out = matmul_backward_both(
            &cache.x_after_final_ln, &self.lm_head, &d_logits,
            t, e, v, &mut grads.lm_head,
        );

        // Backward final LN
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

            let d_ff_h = matmul_backward_both(
                &lc.ff_post_gelu, &self.ff_w2[l], &d_ff_out,
                t, inner, e, &mut grads.ff_w2[l],
            );
            let d_ff_pre = gelu_backward(&lc.ff_pre_gelu, &d_ff_h);
            for i in 0..t { for j in 0..inner { grads.ff_b1[l][j] += d_ff_pre[i * inner + j]; } }

            let d_ln2 = matmul_backward_both(
                &lc.ln2_out, &self.ff_w1[l], &d_ff_pre,
                t, e, inner, &mut grads.ff_w1[l],
            );
            let d_from_ln2 = layer_norm_backward(
                &lc.x_after_attn, &d_ln2, &self.ln2_gamma[l],
                t, e, &mut grads.ln2_gamma[l], &mut grads.ln2_beta[l],
            );

            let mut dx_mid = dx;
            for i in 0..t * e { dx_mid[i] += d_from_ln2[i]; }

            // Spectral attention backward
            let d_proj_out = dx_mid.clone();
            let d_attn_out = matmul_backward_both(
                &lc.attn_out, &self.attn_proj[l], &d_proj_out,
                t, e, e, &mut grads.attn_proj[l],
            );

            let mut d_ln1_out = vec![0.0f32; t * e];

            for h in 0..nh {
                let hc = &lc.head_caches[h];

                // Extract d for this head
                let mut d_head = vec![0.0f32; t * hs];
                for i in 0..t {
                    for f in 0..hs {
                        d_head[i * hs + f] = d_attn_out[i * e + h * hs + f];
                    }
                }

                let mut d_proj = vec![0.0f32; t * hs];

                for f in 0..hs {
                    let mut d_time = vec![0.0f32; t];
                    for i in 0..t { d_time[i] = d_head[i * hs + f]; }

                    // Backward through iFFT:
                    // y = iFFT(Z)[:T] where Z = H * X
                    // dL/dZ = FFT(dL/dy_padded) / N
                    // where dy_padded = [d_time, 0, ..., 0] (length pl)
                    let (d_z_re_raw, d_z_im_raw) = fft(&d_time, pl);
                    let inv_n = 1.0 / pl as f32;
                    let d_z_re: Vec<f32> = d_z_re_raw.iter().map(|v| v * inv_n).collect();
                    let d_z_im: Vec<f32> = d_z_im_raw.iter().map(|v| v * inv_n).collect();

                    // Z = H * X (complex multiply)
                    // dL/dH = dL/dZ * conj(X) component-wise
                    // dL/dX = dL/dZ * conj(H) component-wise (but we need dL/d_kernel, not dL/dX directly)
                    //
                    // For kernel gradient:
                    // H = FFT(kernel), so dL/d_kernel = iFFT(dL/dH) * N, take first kl values
                    // dL/dH_re = d_z_re * X_re + d_z_im * X_im
                    // dL/dH_im = -d_z_re * X_im + d_z_im * X_re

                    let mut d_h_re = vec![0.0f32; pl];
                    let mut d_h_im = vec![0.0f32; pl];
                    let mut d_x_re = vec![0.0f32; pl];
                    let mut d_x_im = vec![0.0f32; pl];

                    for k in 0..pl {
                        let xr = hc.sig_fft_re[f * pl + k];
                        let xi = hc.sig_fft_im[f * pl + k];
                        let hr = hc.kern_fft_re[f * pl + k];
                        let hi = hc.kern_fft_im[f * pl + k];
                        let dzr = d_z_re[k];
                        let dzi = d_z_im[k];

                        // d/dH of (H*X): dZ_re/dH_re = X_re, dZ_re/dH_im = -X_im
                        //                dZ_im/dH_re = X_im, dZ_im/dH_im = X_re
                        d_h_re[k] = dzr * xr + dzi * xi;
                        d_h_im[k] = -dzr * xi + dzi * xr;

                        // d/dX: dZ_re/dX_re = H_re, dZ_re/dX_im = -H_im
                        //       dZ_im/dX_re = H_im, dZ_im/dX_im = H_re
                        d_x_re[k] = dzr * hr + dzi * hi;
                        d_x_im[k] = -dzr * hi + dzi * hr;
                    }

                    // Kernel gradient: iFFT(dL/dH) * N, take first kl values
                    let d_kernel_time = ifft_real(&d_h_re, &d_h_im, pl);
                    // Scale by N because FFT(kernel) -> dL/dH, and
                    // dL/d_kernel = iFFT(dL/dH) needs to undo the FFT scaling
                    for k in 0..kl {
                        grads.causal_kernel[l][h][f * kl + k] += d_kernel_time[k] * pl as f32;
                    }

                    // Signal gradient: iFFT(dL/dX) * N, take first T values
                    let d_signal = ifft_real(&d_x_re, &d_x_im, pl);
                    for i in 0..t {
                        d_proj[i * hs + f] += d_signal[i] * pl as f32;
                    }
                }

                // Backward through head projection
                let d_ln1_h = matmul_backward_both(
                    &lc.ln1_out, &self.head_proj_w[l][h], &d_proj,
                    t, e, hs, &mut grads.head_proj_w[l][h],
                );
                for i in 0..t * e { d_ln1_out[i] += d_ln1_h[i]; }
            }

            // Backward LN1
            let d_from_attn = layer_norm_backward(
                &lc.x_input, &d_ln1_out, &self.ln1_gamma[l],
                t, e, &mut grads.ln1_gamma[l], &mut grads.ln1_beta[l],
            );

            dx = dx_mid;
            for i in 0..t * e { dx[i] += d_from_attn[i]; }
        }

        // Embedding grads
        for (i, &tok) in tokens.iter().enumerate() {
            for j in 0..e {
                grads.token_emb[tok * e + j] += dx[i * e + j];
                grads.pos_emb[i * e + j] += dx[i * e + j];
            }
        }

        (loss, grads)
    }

    pub fn apply_gradients(&mut self, grads: &SpectralGradients, lr: f32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        self.t_step += 1;
        let t_f = self.t_step as f32;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);
        let mut idx = 0usize;

        macro_rules! au {
            ($p:expr, $g:expr) => {{
                adam_step($p, $g, &mut self.m[idx], &mut self.v[idx],
                          lr, beta1, beta2, eps, bc1, bc2);
                idx += 1;
            }};
        }

        au!(&mut self.token_emb, &grads.token_emb);
        au!(&mut self.pos_emb, &grads.pos_emb);
        let nh = self.config.n_head;
        for l in 0..self.config.n_layer {
            au!(&mut self.ln1_gamma[l], &grads.ln1_gamma[l]);
            au!(&mut self.ln1_beta[l], &grads.ln1_beta[l]);
            for h in 0..nh {
                au!(&mut self.head_proj_w[l][h], &grads.head_proj_w[l][h]);
                au!(&mut self.causal_kernel[l][h], &grads.causal_kernel[l][h]);
            }
            au!(&mut self.attn_proj[l], &grads.attn_proj[l]);
            au!(&mut self.ln2_gamma[l], &grads.ln2_gamma[l]);
            au!(&mut self.ln2_beta[l], &grads.ln2_beta[l]);
            au!(&mut self.ff_w1[l], &grads.ff_w1[l]);
            au!(&mut self.ff_b1[l], &grads.ff_b1[l]);
            au!(&mut self.ff_w2[l], &grads.ff_w2[l]);
            au!(&mut self.ff_b2[l], &grads.ff_b2[l]);
        }
        au!(&mut self.ln_f_gamma, &grads.ln_f_gamma);
        au!(&mut self.ln_f_beta, &grads.ln_f_beta);
        au!(&mut self.lm_head, &grads.lm_head);
        let _ = idx;
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
            let last_off = (t - 1) * v;
            let last_logits = &logits[last_off..last_off + v];

            let temperature = 0.7f32;
            let mx = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; v];
            let mut s = 0.0f32;
            for i in 0..v {
                probs[i] = ((last_logits[i] - mx) / temperature).exp();
                s += probs[i];
            }
            for p in probs.iter_mut() { *p /= s; }

            let r: f32 = rng.r#gen();
            let mut cs = 0.0f32;
            let mut nt = 0;
            for (i, &p) in probs.iter().enumerate() {
                cs += p;
                if r < cs { nt = i; break; }
            }
            tokens.push(nt);
        }
        tokens
    }

    pub fn save_weights(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"SGPT")?;
        f.write_all(&(self.config.vocab_size as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_embd as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_head as u32).to_le_bytes())?;
        f.write_all(&(self.config.n_layer as u32).to_le_bytes())?;
        f.write_all(&(self.config.block_size as u32).to_le_bytes())?;

        let mut all: Vec<f32> = Vec::new();
        all.extend_from_slice(&self.token_emb);
        all.extend_from_slice(&self.pos_emb);
        let nh = self.config.n_head;
        for l in 0..self.config.n_layer {
            all.extend_from_slice(&self.ln1_gamma[l]);
            all.extend_from_slice(&self.ln1_beta[l]);
            for h in 0..nh {
                all.extend_from_slice(&self.head_proj_w[l][h]);
                all.extend_from_slice(&self.causal_kernel[l][h]);
            }
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

    #[allow(dead_code)]
    pub fn load_weights(path: &str) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"SGPT" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not SGPT"));
        }
        let ru32 = |f: &mut std::fs::File| -> std::io::Result<u32> {
            let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
        };
        let vs = ru32(&mut f)? as usize;
        let ne = ru32(&mut f)? as usize;
        let nh = ru32(&mut f)? as usize;
        let nl = ru32(&mut f)? as usize;
        let bs = ru32(&mut f)? as usize;
        let cfg = Config { vocab_size: vs, n_embd: ne, n_head: nh, n_layer: nl, block_size: bs };
        let mut nb = [0u8; 8]; f.read_exact(&mut nb)?;
        let nf = u64::from_le_bytes(nb) as usize;
        let mut raw = vec![0u8; nf * 4]; f.read_exact(&mut raw)?;
        let data: Vec<f32> = raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let mut model = SpectralGPT::new(cfg.clone());
        let mut i = 0;
        let take = |d: &[f32], i: &mut usize, n: usize| -> Vec<f32> {
            let s = d[*i..*i + n].to_vec(); *i += n; s
        };
        let hs = ne / nh;
        let kl = model.kernel_len;
        let inner = 4 * ne;
        model.token_emb = take(&data, &mut i, vs * ne);
        model.pos_emb = take(&data, &mut i, bs * ne);
        for l in 0..nl {
            model.ln1_gamma[l] = take(&data, &mut i, ne);
            model.ln1_beta[l] = take(&data, &mut i, ne);
            for h in 0..nh {
                model.head_proj_w[l][h] = take(&data, &mut i, ne * hs);
                model.causal_kernel[l][h] = take(&data, &mut i, hs * kl);
            }
            model.attn_proj[l] = take(&data, &mut i, ne * ne);
            model.ln2_gamma[l] = take(&data, &mut i, ne);
            model.ln2_beta[l] = take(&data, &mut i, ne);
            model.ff_w1[l] = take(&data, &mut i, ne * inner);
            model.ff_b1[l] = take(&data, &mut i, inner);
            model.ff_w2[l] = take(&data, &mut i, inner * ne);
            model.ff_b2[l] = take(&data, &mut i, ne);
        }
        model.ln_f_gamma = take(&data, &mut i, ne);
        model.ln_f_beta = take(&data, &mut i, ne);
        model.lm_head = take(&data, &mut i, ne * vs);
        Ok(model)
    }
}

// ─── Gradients ─────────────────────────────────────────────────────

pub struct SpectralGradients {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub ln1_gamma: Vec<Vec<f32>>,
    pub ln1_beta: Vec<Vec<f32>>,
    pub head_proj_w: Vec<Vec<Vec<f32>>>,
    pub causal_kernel: Vec<Vec<Vec<f32>>>,
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

impl SpectralGradients {
    fn new(cfg: &Config, kl: usize) -> Self {
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        let inner = 4 * e;
        let nl = cfg.n_layer;
        let nh = cfg.n_head;
        let hs = e / nh;
        Self {
            token_emb: vec![0.0; v * e],
            pos_emb: vec![0.0; cfg.block_size * e],
            ln1_gamma: vec![vec![0.0; e]; nl],
            ln1_beta: vec![vec![0.0; e]; nl],
            head_proj_w: (0..nl).map(|_| (0..nh).map(|_| vec![0.0; e * hs]).collect()).collect(),
            causal_kernel: (0..nl).map(|_| (0..nh).map(|_| vec![0.0; hs * kl]).collect()).collect(),
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

    pub fn zero_like(cfg: &Config, kl: usize) -> Self { Self::new(cfg, kl) }

    pub fn accumulate(&mut self, other: &SpectralGradients) {
        add_vecs(&mut self.token_emb, &other.token_emb);
        add_vecs(&mut self.pos_emb, &other.pos_emb);
        let nl = self.ln1_gamma.len();
        let nh = self.head_proj_w[0].len();
        for l in 0..nl {
            add_vecs(&mut self.ln1_gamma[l], &other.ln1_gamma[l]);
            add_vecs(&mut self.ln1_beta[l], &other.ln1_beta[l]);
            for h in 0..nh {
                add_vecs(&mut self.head_proj_w[l][h], &other.head_proj_w[l][h]);
                add_vecs(&mut self.causal_kernel[l][h], &other.causal_kernel[l][h]);
            }
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
        let nl = self.ln1_gamma.len();
        let nh = self.head_proj_w[0].len();
        for l in 0..nl {
            scale_vec(&mut self.ln1_gamma[l], factor);
            scale_vec(&mut self.ln1_beta[l], factor);
            for h in 0..nh {
                scale_vec(&mut self.head_proj_w[l][h], factor);
                scale_vec(&mut self.causal_kernel[l][h], factor);
            }
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

// ─── Cache ─────────────────────────────────────────────────────────

#[derive(Default)]
struct HeadCache {
    proj: Vec<f32>,
    sig_fft_re: Vec<f32>,
    sig_fft_im: Vec<f32>,
    kern_fft_re: Vec<f32>,
    kern_fft_im: Vec<f32>,
    spectral_out: Vec<f32>,
}

#[derive(Default)]
struct LayerCache {
    x_input: Vec<f32>,
    ln1_out: Vec<f32>,
    ln1_mean: Vec<f32>,
    ln1_rstd: Vec<f32>,
    head_caches: Vec<HeadCache>,
    attn_out: Vec<f32>,
    x_after_attn: Vec<f32>,
    ln2_out: Vec<f32>,
    ln2_mean: Vec<f32>,
    ln2_rstd: Vec<f32>,
    ff_pre_gelu: Vec<f32>,
    ff_post_gelu: Vec<f32>,
}

#[allow(dead_code)]
struct SpectralCache {
    tokens: Vec<usize>,
    x_after_emb: Vec<f32>,
    layer_caches: Vec<LayerCache>,
    x_before_final_ln: Vec<f32>,
    x_after_final_ln: Vec<f32>,
}

// ─── Helper functions ──────────────────────────────────────────────

fn adam_step(
    params: &mut Vec<f32>, grads: &[f32],
    m: &mut Vec<f32>, v: &mut Vec<f32>,
    lr: f32, beta1: f32, beta2: f32, eps: f32, bc1: f32, bc2: f32,
) {
    for i in 0..params.len() {
        let g = grads[i].max(-1.0).min(1.0);
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        params[i] -= lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
    }
}

fn add_vecs(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) { *x += y; }
}
fn scale_vec(a: &mut [f32], s: f32) {
    for x in a.iter_mut() { *x *= s; }
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av.abs() > 1e-12 {
                for j in 0..n { c[i * n + j] += av * b[p * n + j]; }
            }
        }
    }
    c
}

fn matmul_backward_both(
    a: &[f32], b: &[f32], dc: &[f32],
    m: usize, k: usize, n: usize, db: &mut [f32],
) -> Vec<f32> {
    let mut da = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..n {
            let dv = dc[i * n + j];
            if dv.abs() > 1e-12 {
                for p in 0..k { da[i * k + p] += dv * b[p * n + j]; }
            }
        }
    }
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av.abs() > 1e-12 {
                for j in 0..n { db[p * n + j] += av * dc[i * n + j]; }
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
        let off = i * e;
        let mean: f32 = x[off..off + e].iter().sum::<f32>() / e as f32;
        let var: f32 = x[off..off + e].iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / e as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        means[i] = mean; rstds[i] = rstd;
        for j in 0..e { out[off + j] = gamma[j] * ((x[off + j] - mean) * rstd) + beta[j]; }
    }
    (out, means, rstds)
}

fn layer_norm_backward(
    x: &[f32], dout: &[f32], gamma: &[f32],
    t: usize, e: usize, dgamma: &mut [f32], dbeta: &mut [f32],
) -> Vec<f32> {
    let eps = 1e-5f32;
    let mut dx = vec![0.0f32; t * e];
    for i in 0..t {
        let off = i * e;
        let mean: f32 = x[off..off + e].iter().sum::<f32>() / e as f32;
        let var: f32 = x[off..off + e].iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / e as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        let mut norm = vec![0.0f32; e];
        for j in 0..e { norm[j] = (x[off + j] - mean) * rstd; }
        for j in 0..e { dgamma[j] += dout[off + j] * norm[j]; dbeta[j] += dout[off + j]; }
        let mut dnorm = vec![0.0f32; e];
        for j in 0..e { dnorm[j] = dout[off + j] * gamma[j]; }
        let dm: f32 = dnorm.iter().sum::<f32>() / e as f32;
        let dnm: f32 = dnorm.iter().zip(norm.iter()).map(|(a, b)| a * b).sum::<f32>() / e as f32;
        for j in 0..e { dx[off + j] = (dnorm[j] - dm - norm[j] * dnm) * rstd; }
    }
    dx
}

fn gelu_backward(x: &[f32], dout: &[f32]) -> Vec<f32> {
    let s2p = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut dx = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        let xi = x[i];
        let inner = s2p * (xi + 0.044715 * xi * xi * xi);
        let th = inner.tanh();
        let di = s2p * (1.0 + 3.0 * 0.044715 * xi * xi);
        dx[i] = dout[i] * (0.5 * (1.0 + th) + 0.5 * xi * (1.0 - th * th) * di);
    }
    dx
}
