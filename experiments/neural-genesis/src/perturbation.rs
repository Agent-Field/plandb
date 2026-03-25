//! Experiment A2: Perturbation Theory Training (Zeroth-Order Gradient Estimation)
//!
//! Physics mapping:
//! - Standard backprop = solving equations exactly
//! - Perturbation theory = approximate the solution by probing around the current state
//! - Random perturbations dW = "test wavefunctions" in variational Monte Carlo
//! - Loss change dL = "energy difference" under perturbation
//! - Gradient estimate = correlation between perturbation direction and energy change
//!
//! Key advantage: embarrassingly parallel — each perturbation is independent.

use crate::data;
use crate::model::{matmul, randn_vec, Config, GPT, Gradients};
use crate::tokenizer::Tokenizer;
use crate::tool_data::{self, AggregateMetrics, ToolExample};
use rand::Rng;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

// ─── Flatten / Unflatten ────────────────────────────────────────

/// Flatten all GPT parameters into a single contiguous vector.
/// Order matches `param_sizes()` / `save_weights()` in model.rs:
///   token_emb, pos_emb, then per-layer (ln1_gamma, ln1_beta, qkv_w,
///   attn_proj, ln2_gamma, ln2_beta, ff_w1, ff_b1, ff_w2, ff_b2),
///   ln_f_gamma, ln_f_beta, lm_head.
pub fn flatten_params(model: &GPT) -> Vec<f32> {
    let mut flat = Vec::new();
    flat.extend_from_slice(&model.token_emb);
    flat.extend_from_slice(&model.pos_emb);
    for l in 0..model.config.n_layer {
        flat.extend_from_slice(&model.ln1_gamma[l]);
        flat.extend_from_slice(&model.ln1_beta[l]);
        flat.extend_from_slice(&model.qkv_w[l]);
        flat.extend_from_slice(&model.attn_proj[l]);
        flat.extend_from_slice(&model.ln2_gamma[l]);
        flat.extend_from_slice(&model.ln2_beta[l]);
        flat.extend_from_slice(&model.ff_w1[l]);
        flat.extend_from_slice(&model.ff_b1[l]);
        flat.extend_from_slice(&model.ff_w2[l]);
        flat.extend_from_slice(&model.ff_b2[l]);
    }
    flat.extend_from_slice(&model.ln_f_gamma);
    flat.extend_from_slice(&model.ln_f_beta);
    flat.extend_from_slice(&model.lm_head);
    flat
}

/// Unflatten a parameter vector back into the GPT model fields.
pub fn apply_flat_params(model: &mut GPT, params: &[f32]) {
    let e = model.config.n_embd;
    let v = model.config.vocab_size;
    let bs = model.config.block_size;
    let inner = 4 * e;
    let mut i = 0;

    let take = |params: &[f32], i: &mut usize, n: usize| -> Vec<f32> {
        let s = params[*i..*i + n].to_vec();
        *i += n;
        s
    };

    model.token_emb = take(params, &mut i, v * e);
    model.pos_emb = take(params, &mut i, bs * e);
    for l in 0..model.config.n_layer {
        model.ln1_gamma[l] = take(params, &mut i, e);
        model.ln1_beta[l] = take(params, &mut i, e);
        model.qkv_w[l] = take(params, &mut i, e * 3 * e);
        model.attn_proj[l] = take(params, &mut i, e * e);
        model.ln2_gamma[l] = take(params, &mut i, e);
        model.ln2_beta[l] = take(params, &mut i, e);
        model.ff_w1[l] = take(params, &mut i, e * inner);
        model.ff_b1[l] = take(params, &mut i, inner);
        model.ff_w2[l] = take(params, &mut i, inner * e);
        model.ff_b2[l] = take(params, &mut i, e);
    }
    model.ln_f_gamma = take(params, &mut i, e);
    model.ln_f_beta = take(params, &mut i, e);
    model.lm_head = take(params, &mut i, e * v);
}

// ─── Forward Loss Only ──────────────────────────────────────────

/// Simplified forward pass: tokens -> logits -> cross-entropy loss.
/// No backward caches stored — just compute the scalar loss.
fn forward_loss_only(model: &GPT, tokens: &[usize], targets: &[usize]) -> f32 {
    let cfg = &model.config;
    let t = tokens.len();
    let e = cfg.n_embd;
    let v = cfg.vocab_size;
    let nh = cfg.n_head;
    let hs = e / nh;

    // Embedding lookup: token_emb + pos_emb
    let mut x = vec![0.0f32; t * e];
    for (i, &tok) in tokens.iter().enumerate() {
        for j in 0..e {
            x[i * e + j] = model.token_emb[tok * e + j] + model.pos_emb[i * e + j];
        }
    }

    // Transformer blocks
    for l in 0..cfg.n_layer {
        // Layer norm 1
        let ln1_out = layer_norm_forward(&x, &model.ln1_gamma[l], &model.ln1_beta[l], t, e);

        // QKV projection: (T, E) @ (E, 3E) -> (T, 3E)
        let qkv = matmul(&ln1_out, &model.qkv_w[l], t, e, 3 * e);

        // Multi-head attention
        let mut attn_out = vec![0.0f32; t * e];
        for h in 0..nh {
            // Compute attention scores: Q @ K^T / sqrt(hs)
            let scale = 1.0 / (hs as f32).sqrt();
            let mut attn_weights = vec![0.0f32; t * t];
            for i in 0..t {
                for j in 0..t {
                    if j > i {
                        attn_weights[i * t + j] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for k in 0..hs {
                            let qi = qkv[i * 3 * e + h * hs + k];
                            let kj = qkv[j * 3 * e + e + h * hs + k];
                            dot += qi * kj;
                        }
                        attn_weights[i * t + j] = dot * scale;
                    }
                }
            }

            // Softmax per row
            for i in 0..t {
                let offset = i * t;
                let max_val = attn_weights[offset..offset + t]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..t {
                    let exp_val = (attn_weights[offset + j] - max_val).exp();
                    attn_weights[offset + j] = exp_val;
                    sum += exp_val;
                }
                if sum > 0.0 {
                    for j in 0..t {
                        attn_weights[offset + j] /= sum;
                    }
                }
            }

            // Weighted sum of V
            for i in 0..t {
                for k in 0..hs {
                    let mut sum = 0.0f32;
                    for j in 0..t {
                        let w = attn_weights[i * t + j];
                        let vj = qkv[j * 3 * e + 2 * e + h * hs + k];
                        sum += w * vj;
                    }
                    attn_out[i * e + h * hs + k] = sum;
                }
            }
        }

        // Output projection + residual
        let proj_out = matmul(&attn_out, &model.attn_proj[l], t, e, e);
        for i in 0..t * e {
            x[i] += proj_out[i];
        }

        // Layer norm 2
        let ln2_out = layer_norm_forward(&x, &model.ln2_gamma[l], &model.ln2_beta[l], t, e);

        // Feed-forward: (T, E) @ (E, 4E) + b1 -> GELU -> (T, 4E) @ (4E, E) + b2
        let inner = 4 * e;
        let mut ff_hidden = matmul(&ln2_out, &model.ff_w1[l], t, e, inner);
        for i in 0..t {
            for j in 0..inner {
                ff_hidden[i * inner + j] += model.ff_b1[l][j];
            }
        }

        // GELU activation
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        for val in ff_hidden.iter_mut() {
            let x3 = *val * *val * *val;
            let inner_val = sqrt_2_over_pi * (*val + 0.044715 * x3);
            *val = 0.5 * *val * (1.0 + inner_val.tanh());
        }

        let mut ff_out = matmul(&ff_hidden, &model.ff_w2[l], t, inner, e);
        for i in 0..t {
            for j in 0..e {
                ff_out[i * e + j] += model.ff_b2[l][j];
            }
        }

        // Residual
        for i in 0..t * e {
            x[i] += ff_out[i];
        }
    }

    // Final layer norm
    let ln_out = layer_norm_forward(&x, &model.ln_f_gamma, &model.ln_f_beta, t, e);

    // LM head: (T, E) @ (E, V) -> (T, V)
    let logits = matmul(&ln_out, &model.lm_head, t, e, v);

    // Cross-entropy loss
    cross_entropy_loss(&logits, targets, t, v)
}

/// Simplified layer norm (forward only, no cache).
fn layer_norm_forward(x: &[f32], gamma: &[f32], beta: &[f32], t: usize, e: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * e];
    let eps = 1e-5f32;

    for i in 0..t {
        let offset = i * e;
        let mean: f32 = x[offset..offset + e].iter().sum::<f32>() / e as f32;
        let var: f32 = x[offset..offset + e]
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>()
            / e as f32;
        let rstd = 1.0 / (var + eps).sqrt();

        for j in 0..e {
            out[offset + j] = (x[offset + j] - mean) * rstd * gamma[j] + beta[j];
        }
    }
    out
}

/// Cross-entropy loss over all positions.
fn cross_entropy_loss(logits: &[f32], targets: &[usize], t: usize, v: usize) -> f32 {
    let mut total_loss = 0.0f32;
    for i in 0..t {
        let offset = i * v;
        let row = &logits[offset..offset + v];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut log_sum_exp = 0.0f32;
        for &val in row {
            log_sum_exp += (val - max_val).exp();
        }
        log_sum_exp = max_val + log_sum_exp.ln();
        total_loss += log_sum_exp - logits[offset + targets[i]];
    }
    total_loss / t as f32
}

// ─── Perturbation Trainer ───────────────────────────────────────

/// Zeroth-order gradient estimator using random perturbations.
struct PerturbationTrainer {
    model: GPT,
    sigma: f32,
    n_perturbations: usize,
    learning_rate: f32,
    use_antithetic: bool,
}

impl PerturbationTrainer {
    fn new(
        model: GPT,
        sigma: f32,
        n_perturbations: usize,
        learning_rate: f32,
        use_antithetic: bool,
    ) -> Self {
        Self {
            model,
            sigma,
            n_perturbations,
            learning_rate,
            use_antithetic,
        }
    }

    /// Estimate gradient via K perturbations (or K/2 antithetic pairs).
    /// Returns the estimated gradient as a flat Vec<f32>.
    fn estimate_gradient(&self, tokens: &[usize], targets: &[usize]) -> Vec<f32> {
        let params = flatten_params(&self.model);
        let n = params.len();
        let sigma = self.sigma;
        let sigma_sq = sigma * sigma;

        if self.use_antithetic {
            // Antithetic sampling: K/2 pairs of (delta, -delta)
            let k_pairs = self.n_perturbations / 2;
            let k_pairs = k_pairs.max(1);

            let pair_results: Vec<(Vec<f32>, f32, f32)> = (0..k_pairs)
                .into_par_iter()
                .map(|_| {
                    let delta = randn_vec(n, sigma);

                    // Positive perturbation: W + delta
                    let mut pos_params = params.clone();
                    for i in 0..n {
                        pos_params[i] += delta[i];
                    }
                    let mut pos_model = GPT::new(self.model.config.clone());
                    apply_flat_params(&mut pos_model, &pos_params);
                    let loss_pos = forward_loss_only(&pos_model, tokens, targets);

                    // Negative perturbation: W - delta
                    let mut neg_params = params.clone();
                    for i in 0..n {
                        neg_params[i] -= delta[i];
                    }
                    let mut neg_model = GPT::new(self.model.config.clone());
                    apply_flat_params(&mut neg_model, &neg_params);
                    let loss_neg = forward_loss_only(&neg_model, tokens, targets);

                    (delta, loss_pos, loss_neg)
                })
                .collect();

            // Accumulate gradient: g_i += (L+ - L-) / (2 * K * sigma^2) * delta_i
            let mut grad = vec![0.0f32; n];
            let k_total = pair_results.len() as f32;
            for (delta, loss_pos, loss_neg) in &pair_results {
                let diff = loss_pos - loss_neg;
                if diff.is_finite() {
                    let scale = diff / (2.0 * k_total * sigma_sq);
                    for i in 0..n {
                        grad[i] += scale * delta[i];
                    }
                }
            }
            grad
        } else {
            // Standard: K independent perturbations
            let base_loss = forward_loss_only(&self.model, tokens, targets);

            let results: Vec<(Vec<f32>, f32)> = (0..self.n_perturbations)
                .into_par_iter()
                .map(|_| {
                    let delta = randn_vec(n, sigma);

                    let mut perturbed_params = params.clone();
                    for i in 0..n {
                        perturbed_params[i] += delta[i];
                    }
                    let mut perturbed_model = GPT::new(self.model.config.clone());
                    apply_flat_params(&mut perturbed_model, &perturbed_params);
                    let loss = forward_loss_only(&perturbed_model, tokens, targets);

                    (delta, loss)
                })
                .collect();

            let mut grad = vec![0.0f32; n];
            let k = results.len() as f32;
            for (delta, loss) in &results {
                let diff = loss - base_loss;
                if diff.is_finite() {
                    let scale = diff / (k * sigma_sq);
                    for i in 0..n {
                        grad[i] += scale * delta[i];
                    }
                }
            }
            grad
        }
    }

    /// One training step: estimate gradient, apply SGD update, return loss.
    fn train_step(&mut self, tokens: &[usize], targets: &[usize]) -> f32 {
        let loss = forward_loss_only(&self.model, tokens, targets);
        let grad = self.estimate_gradient(tokens, targets);

        // SGD update: W -= lr * grad
        let mut params = flatten_params(&self.model);
        for i in 0..params.len() {
            let g = grad[i].max(-1.0).min(1.0); // gradient clipping
            params[i] -= self.learning_rate * g;
        }
        apply_flat_params(&mut self.model, &params);

        loss
    }
}

// ─── Helpers ────────────────────────────────────────────────────

fn count_params(config: &Config) -> usize {
    let e = config.n_embd;
    let v = config.vocab_size;
    let nl = config.n_layer;
    let inner = 4 * e;
    let mut total = v * e + config.block_size * e;
    for _ in 0..nl {
        total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
    }
    total + e * 2 + e * v
}

/// Clone model weights into a Vec<Vec<f32>> for later restoration.
fn clone_model_weights(model: &GPT) -> Vec<Vec<f32>> {
    let mut weights = Vec::new();
    weights.push(model.token_emb.clone());
    weights.push(model.pos_emb.clone());
    for l in 0..model.config.n_layer {
        weights.push(model.ln1_gamma[l].clone());
        weights.push(model.ln1_beta[l].clone());
        weights.push(model.qkv_w[l].clone());
        weights.push(model.attn_proj[l].clone());
        weights.push(model.ln2_gamma[l].clone());
        weights.push(model.ln2_beta[l].clone());
        weights.push(model.ff_w1[l].clone());
        weights.push(model.ff_b1[l].clone());
        weights.push(model.ff_w2[l].clone());
        weights.push(model.ff_b2[l].clone());
    }
    weights.push(model.ln_f_gamma.clone());
    weights.push(model.ln_f_beta.clone());
    weights.push(model.lm_head.clone());
    weights
}

/// Restore model from cloned weights.
fn model_from_weights(config: &Config, weights: &[Vec<f32>]) -> GPT {
    let mut model = GPT::new(config.clone());
    let mut idx = 0;
    model.token_emb = weights[idx].clone(); idx += 1;
    model.pos_emb = weights[idx].clone(); idx += 1;
    for l in 0..config.n_layer {
        model.ln1_gamma[l] = weights[idx].clone(); idx += 1;
        model.ln1_beta[l] = weights[idx].clone(); idx += 1;
        model.qkv_w[l] = weights[idx].clone(); idx += 1;
        model.attn_proj[l] = weights[idx].clone(); idx += 1;
        model.ln2_gamma[l] = weights[idx].clone(); idx += 1;
        model.ln2_beta[l] = weights[idx].clone(); idx += 1;
        model.ff_w1[l] = weights[idx].clone(); idx += 1;
        model.ff_b1[l] = weights[idx].clone(); idx += 1;
        model.ff_w2[l] = weights[idx].clone(); idx += 1;
        model.ff_b2[l] = weights[idx].clone(); idx += 1;
    }
    model.ln_f_gamma = weights[idx].clone(); idx += 1;
    model.ln_f_beta = weights[idx].clone(); idx += 1;
    model.lm_head = weights[idx].clone();
    model
}

/// Pretrain a GPT on Shakespeare text using standard backprop.
fn pretrain(config: &Config, text: &str, tok: &Tokenizer, rng: &mut impl Rng) -> GPT {
    let encoded = tok.encode(text);
    let mut model = GPT::new(config.clone());

    let batch_size = 16;
    let num_steps = 1000;
    let lr = 0.001;

    let start = Instant::now();
    for step in 0..num_steps {
        let (inputs, targets) =
            data::create_batches(&encoded, config.block_size, batch_size, rng);
        let results: Vec<(f32, Gradients)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let ctx_len = inputs[b].len().min(config.block_size);
                model.forward_backward(&inputs[b][..ctx_len], &targets[b][..ctx_len])
            })
            .collect();

        let mut total_loss = 0.0f32;
        let mut grads = Gradients::zero_like(config);
        for (loss, g) in results {
            total_loss += loss;
            grads.accumulate(&g);
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, lr);

        if step % 200 == 0 {
            println!(
                "  Pretrain step {:4} | Loss: {:.4} | {:.1}s",
                step,
                total_loss / batch_size as f32,
                start.elapsed().as_secs_f32()
            );
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
    model
}

/// SFT using standard backprop (baseline for comparison).
fn sft_backprop(
    model: &mut GPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
    num_steps: usize,
) -> Vec<(usize, f32)> {
    let lr = 0.0005;
    let batch_size = 8;
    let start = Instant::now();
    let mut loss_log = Vec::new();

    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = Gradients::zero_like(config);
        let mut valid_count = 0;

        for _ in 0..batch_size {
            let example = &train[rng.gen_range(0..train.len())];
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
            let avg = total_loss / valid_count as f32;
            loss_log.push((step, avg));

            if step % 100 == 0 {
                println!(
                    "  Backprop SFT step {:4} | Loss: {:.4} | {:.1}s",
                    step, avg, start.elapsed().as_secs_f32()
                );
            }
        }
    }
    println!(
        "  Backprop SFT done in {:.1}s",
        start.elapsed().as_secs_f32()
    );
    loss_log
}

/// SFT using perturbation training (zeroth-order).
fn sft_perturbation(
    model: GPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
    num_steps: usize,
    k: usize,
    sigma: f32,
    lr: f32,
    antithetic: bool,
) -> (GPT, Vec<(usize, f32)>) {
    let mut trainer = PerturbationTrainer::new(model, sigma, k, lr, antithetic);
    let start = Instant::now();
    let mut loss_log = Vec::new();

    for step in 0..num_steps {
        // Pick a random training example
        let example = &train[rng.gen_range(0..train.len())];
        let encoded = tok.encode(&example.input);
        if encoded.len() < 2 || encoded.len() > config.block_size {
            continue;
        }
        let tokens = &encoded[..encoded.len() - 1];
        let targets = &encoded[1..];

        let loss = trainer.train_step(tokens, targets);

        if loss.is_finite() {
            loss_log.push((step, loss));
        }

        if step % 100 == 0 {
            let grad_norm = {
                let g = trainer.estimate_gradient(tokens, targets);
                g.iter().map(|x| x * x).sum::<f32>().sqrt()
            };
            println!(
                "  Perturbation step {:4} | Loss: {:.4} | GradNorm: {:.6} | {:.1}s",
                step,
                loss,
                grad_norm,
                start.elapsed().as_secs_f32()
            );
        }
    }
    let total_time = start.elapsed().as_secs_f32();
    println!("  Perturbation SFT done in {:.1}s", total_time);

    (trainer.model, loss_log)
}

/// Evaluate a model on validation data.
fn evaluate_model(
    model: &GPT,
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

/// Compute composite score: 0.3*format + 0.3*tool + 0.2*param + 0.2*reply.
fn composite_score(m: &AggregateMetrics) -> f32 {
    0.3 * m.format_acc + 0.3 * m.tool_acc + 0.2 * m.param_acc + 0.2 * m.reply_quality
}

// ─── Main Experiment ────────────────────────────────────────────

/// Run the full perturbation theory training experiment.
pub fn run_perturbation_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    println!("=== Experiment A2: Perturbation Theory Training ===");
    println!("    Zeroth-order gradient estimation — no backprop needed\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/4] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!(
        "  Train: {} examples, Val: {} examples",
        train_data.len(),
        val_data.len()
    );

    // Build tokenizer
    let base_text = data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    let block_size = 128;
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size,
    };
    let n_params = count_params(&config);
    println!(
        "  Model: {} layers, {} embd, {} heads, {} params\n",
        config.n_layer, config.n_embd, config.n_head, n_params
    );

    // Phase 1: Pretrain on Shakespeare
    println!("[2/4] Pretraining on Shakespeare (backprop, 1000 steps)...");
    let pretrained = pretrain(&config, base_text, &tok, &mut rng);
    let pretrained_weights = clone_model_weights(&pretrained);

    // Phase 2: Backprop baseline SFT
    println!("\n[3/4] Backprop SFT baseline (800 steps)...");
    let mut backprop_model = model_from_weights(&config, &pretrained_weights);
    let backprop_start = Instant::now();
    let backprop_loss_log = sft_backprop(
        &mut backprop_model,
        &config,
        &train_data,
        &tok,
        &mut rng,
        800,
    );
    let backprop_time = backprop_start.elapsed().as_secs_f32();
    let backprop_metrics = evaluate_model(&backprop_model, &config, &val_data, &tok);
    println!("\n  Backprop SFT Results:");
    backprop_metrics.print("Backprop Baseline");

    // Phase 3: Perturbation training configurations
    println!("\n[4/4] Perturbation training experiments...\n");

    struct PertConfig {
        k: usize,
        sigma: f32,
        lr: f32,
        antithetic: bool,
        label: &'static str,
    }

    let configs = vec![
        PertConfig { k: 20,  sigma: 0.01,  lr: 0.001, antithetic: false, label: "K=20, s=0.01" },
        PertConfig { k: 20,  sigma: 0.01,  lr: 0.001, antithetic: true,  label: "K=20, s=0.01, anti" },
        PertConfig { k: 50,  sigma: 0.005, lr: 0.001, antithetic: false, label: "K=50, s=0.005" },
        PertConfig { k: 50,  sigma: 0.005, lr: 0.001, antithetic: true,  label: "K=50, s=0.005, anti" },
        PertConfig { k: 100, sigma: 0.001, lr: 0.001, antithetic: false, label: "K=100, s=0.001" },
        PertConfig { k: 100, sigma: 0.001, lr: 0.001, antithetic: true,  label: "K=100, s=0.001, anti" },
    ];

    let sft_steps = 800;

    // Collect results for CSV
    struct RunResult {
        method: String,
        k: usize,
        sigma: f32,
        antithetic: bool,
        params: usize,
        composite: f32,
        format_acc: f32,
        tool_acc: f32,
        param_acc: f32,
        reply_quality: f32,
        final_loss: f32,
        train_time: f32,
        forward_passes: usize,
    }

    let mut all_results: Vec<RunResult> = Vec::new();

    // Backprop baseline result
    let bp_final_loss = backprop_loss_log
        .last()
        .map(|&(_, l)| l)
        .unwrap_or(f32::NAN);
    let bp_composite = composite_score(&backprop_metrics);
    all_results.push(RunResult {
        method: "backprop".to_string(),
        k: 0,
        sigma: 0.0,
        antithetic: false,
        params: n_params,
        composite: bp_composite,
        format_acc: backprop_metrics.format_acc,
        tool_acc: backprop_metrics.tool_acc,
        param_acc: backprop_metrics.param_acc,
        reply_quality: backprop_metrics.reply_quality,
        final_loss: bp_final_loss,
        train_time: backprop_time,
        forward_passes: sft_steps * 8, // batch_size=8 fwd+bwd ~ 2 passes each
    });

    // Run each perturbation config
    for pc in &configs {
        println!("--- {} ---", pc.label);
        let model = model_from_weights(&config, &pretrained_weights);
        let run_start = Instant::now();
        let (trained_model, loss_log) = sft_perturbation(
            model,
            &config,
            &train_data,
            &tok,
            &mut rng,
            sft_steps,
            pc.k,
            pc.sigma,
            pc.lr,
            pc.antithetic,
        );
        let run_time = run_start.elapsed().as_secs_f32();

        let metrics = evaluate_model(&trained_model, &config, &val_data, &tok);
        println!("  Results:");
        metrics.print(pc.label);
        let comp = composite_score(&metrics);
        println!("  Composite: {:.1}%\n", comp * 100.0);

        let final_loss = loss_log.last().map(|&(_, l)| l).unwrap_or(f32::NAN);
        let fwd_passes = if pc.antithetic {
            sft_steps * pc.k // K/2 pairs * 2 = K forward passes per step
        } else {
            sft_steps * (pc.k + 1) // K perturbations + 1 base
        };

        all_results.push(RunResult {
            method: "perturbation".to_string(),
            k: pc.k,
            sigma: pc.sigma,
            antithetic: pc.antithetic,
            params: n_params,
            composite: comp,
            format_acc: metrics.format_acc,
            tool_acc: metrics.tool_acc,
            param_acc: metrics.param_acc,
            reply_quality: metrics.reply_quality,
            final_loss,
            train_time: run_time,
            forward_passes: fwd_passes,
        });
    }

    // Write CSV
    println!("=== Writing results to experiments/perturbation_results.csv ===\n");
    if let Ok(mut file) = std::fs::File::create("experiments/perturbation_results.csv") {
        let _ = writeln!(
            file,
            "method,K,sigma,antithetic,params,composite,format,tool,param,reply,final_loss,train_time,forward_passes_total"
        );
        for r in &all_results {
            let _ = writeln!(
                file,
                "{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{}",
                r.method,
                r.k,
                r.sigma,
                r.antithetic,
                r.params,
                r.composite,
                r.format_acc,
                r.tool_acc,
                r.param_acc,
                r.reply_quality,
                r.final_loss,
                r.train_time,
                r.forward_passes
            );
        }
        println!("  Results saved.");
    }

    // Print summary table
    println!("\n=== Summary ===\n");
    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Method", "Comp%", "Fmt%", "Tool%", "Param%", "Loss", "Time(s)"
    );
    println!("{}", "-".repeat(90));
    for r in &all_results {
        let label = if r.method == "backprop" {
            "Backprop (baseline)".to_string()
        } else {
            format!(
                "Pert K={} s={} {}",
                r.k,
                r.sigma,
                if r.antithetic { "anti" } else { "" }
            )
        };
        println!(
            "{:<30} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>8.4} {:>10.1}",
            label,
            r.composite * 100.0,
            r.format_acc * 100.0,
            r.tool_acc * 100.0,
            r.param_acc * 100.0,
            r.final_loss,
            r.train_time,
        );
    }
    println!();
    println!("=== Perturbation Experiment Complete ===");
}
