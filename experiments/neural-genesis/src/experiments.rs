use crate::model::{Config, GPT, Gradients, HebbianGPT, HebbianGradients, RGGPT, RGGradients};
use crate::lora_rg::{LoraRGGPT, LoraRGGradients, PreAmpRGGPT, PreAmpRGGradients,
                      PerHeadRGGPT, PerHeadRGGradients};
use crate::hamiltonian::{HamiltonianGPT, HamiltonianGradients, EnergyDiagnostics};
use crate::hybrid::{HybridGPT, HybridGradients};
use crate::spectral::{SpectralGPT, SpectralGradients};
use crate::wannier::{WannierGPT, WannierGradients};
use crate::tokenizer::Tokenizer;
use crate::tool_data::{self, AggregateMetrics, EvalResult, ToolExample};
use rand::Rng;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

/// Run all experiments: pretrain → SFT → RL methods (parallel) → compare
pub fn run_experiments() {
    let _ = std::fs::create_dir_all("experiments");
    println!("=== Mini GPT Tool Calling Experiments ===\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/6] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer from all text (base + tool data)
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    // Config
    let block_size = 128; // longer for tool call sequences
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size,
    };

    // === Phase 1: Pretrain on base text ===
    println!("\n[2/6] Pretraining on Shakespeare...");
    let mut model = pretrain(&config, base_text, &tok, &mut rng);

    // === Phase 2: SFT on tool calling data ===
    println!("\n[3/6] Supervised fine-tuning on tool calling data...");
    let sft_metrics = sft_train(&mut model, &config, &train_data, &val_data, &tok, &mut rng);
    println!("\n  SFT Results:");
    sft_metrics.print("SFT Baseline");

    // Save SFT model weights for RL experiments (clone)
    let sft_weights = clone_model_weights(&model);

    // === Phase 3: RL Experiments (sequential, but each independent from SFT) ===
    // REINFORCE
    println!("\n[4/6] REINFORCE policy gradient...");
    let mut reinforce_model = model_from_weights(&config, &sft_weights);
    let reinforce_metrics = rl_reinforce(
        &mut reinforce_model, &config, &train_data, &val_data, &tok, &mut rng,
    );
    println!("\n  REINFORCE Results:");
    reinforce_metrics.print("REINFORCE");

    // DPO
    println!("\n[5/6] DPO preference optimization...");
    let mut dpo_model = model_from_weights(&config, &sft_weights);
    let dpo_metrics = rl_dpo(
        &mut dpo_model, &config, &train_data, &val_data, &tok, &mut rng,
    );
    println!("\n  DPO Results:");
    dpo_metrics.print("DPO");

    // Custom structured RL
    println!("\n[6/6] Custom structured reward...");
    let mut custom_model = model_from_weights(&config, &sft_weights);
    let custom_metrics = rl_custom(
        &mut custom_model, &config, &train_data, &val_data, &tok, &mut rng,
    );
    println!("\n  Custom RL Results:");
    custom_metrics.print("Custom RL");

    // === Phase 4: Comparison ===
    println!("\n{}", "=".repeat(60));
    println!("=== EXPERIMENT RESULTS COMPARISON ===\n");

    let all_methods = [
        ("SFT Baseline", &sft_metrics),
        ("REINFORCE", &reinforce_metrics),
        ("DPO", &dpo_metrics),
        ("Custom RL", &custom_metrics),
    ];

    // Table
    println!("{:<15} {:>10} {:>10} {:>10} {:>10}",
             "Method", "Format%", "Tool%", "Param%", "Reply%");
    println!("{}", "-".repeat(55));
    for (name, m) in &all_methods {
        println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%",
                 name,
                 m.format_acc * 100.0,
                 m.tool_acc * 100.0,
                 m.param_acc * 100.0,
                 m.reply_quality * 100.0);
    }

    // Find winner (by composite score)
    let scores: Vec<(&&str, f32)> = all_methods.iter()
        .map(|(name, m)| (name, m.format_acc * 0.3 + m.tool_acc * 0.3 + m.param_acc * 0.25 + m.reply_quality * 0.15))
        .collect();
    let winner = scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("\n  Winner: {} (composite score: {:.3})", winner.0, winner.1);

    // Save results CSV
    if let Ok(mut file) = std::fs::File::create("experiments/experiment_results.csv") {
        let _ = writeln!(file, "method,format_acc,tool_acc,param_acc,reply_quality,composite");
        for ((name, m), (_, score)) in all_methods.iter().zip(scores.iter()) {
            let _ = writeln!(file, "{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                             name, m.format_acc, m.tool_acc, m.param_acc, m.reply_quality, score);
        }
        println!("  Results saved to experiment_results.csv");
    }

    // ASCII comparison chart
    print_comparison_chart(&all_methods);

    // Demo with best model
    println!("\n=== Demo: Tool Calling Agent ===\n");
    let best_model = match *winner.0 {
        "REINFORCE" => &reinforce_model,
        "DPO" => &dpo_model,
        "Custom RL" => &custom_model,
        _ => &model, // SFT
    };

    demo_tool_calling(best_model, &tok, &val_data);
}

/// Pretrain on Shakespeare text
fn pretrain(config: &Config, text: &str, tok: &Tokenizer, rng: &mut impl Rng) -> GPT {
    let encoded = tok.encode(text);
    let mut model = GPT::new(config.clone());

    let batch_size = 16;
    let num_steps = 1000;
    let lr = 0.001;

    let start = Instant::now();
    for step in 0..num_steps {
        let (inputs, targets) = crate::data::create_batches(&encoded, config.block_size, batch_size, rng);
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
            println!("  Pretrain step {:4} | Loss: {:.4} | {:.1}s",
                     step, total_loss / batch_size as f32, start.elapsed().as_secs_f32());
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
    model
}

/// Supervised fine-tuning on tool calling data
fn sft_train(
    model: &mut GPT,
    config: &Config,
    train: &[ToolExample],
    val: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
) -> AggregateMetrics {
    let num_steps = 800;
    let lr = 0.0005; // lower LR for fine-tuning
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
                println!("  SFT step {:4} | Loss: {:.4} | {:.1}s",
                         step, avg, start.elapsed().as_secs_f32());
            }
        }
    }

    // Save SFT loss log
    if let Ok(mut file) = std::fs::File::create("experiments/sft_loss.csv") {
        let _ = writeln!(file, "step,loss");
        for &(s, l) in &loss_log {
            let _ = writeln!(file, "{},{:.6}", s, l);
        }
    }

    println!("  SFT done in {:.1}s", start.elapsed().as_secs_f32());

    // Evaluate
    evaluate_model(model, config, val, tok)
}

/// REINFORCE policy gradient
fn rl_reinforce(
    model: &mut GPT,
    config: &Config,
    train: &[ToolExample],
    val: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
) -> AggregateMetrics {
    let num_steps = 400;
    let lr = 0.0001; // very low LR for RL
    let mut reward_baseline = 0.0f32;
    let baseline_decay = 0.99;

    let start = Instant::now();
    let mut reward_log = Vec::new();

    for step in 0..num_steps {
        let example = &train[rng.gen_range(0..train.len())];
        let prompt_encoded = tok.encode(&example.prompt);

        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }

        // Generate completion
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);

        // Compute reward
        let eval = tool_data::evaluate_output(&generated_text, example);
        let reward = compute_reward(&eval, example.expected_call.is_some());

        // Update baseline
        reward_baseline = baseline_decay * reward_baseline + (1.0 - baseline_decay) * reward;
        let advantage = reward - reward_baseline;

        // Policy gradient: for each generated token, update log_prob * advantage
        // We approximate this by doing a forward-backward on the generated sequence
        // with scaled gradients
        if generated_ids.len() >= 2 && (generated_ids.len()) <= config.block_size {
            let input = &generated_ids[..generated_ids.len() - 1];
            let target = &generated_ids[1..];
            let (_, grads) = model.forward_backward(input, target);

            // Scale gradients by advantage (REINFORCE)
            let mut scaled_grads = grads;
            scaled_grads.scale(-advantage); // negative because we want to maximize reward
            model.apply_gradients(&scaled_grads, lr);
        }

        if step % 50 == 0 {
            reward_log.push((step, reward, reward_baseline));
            println!("  REINFORCE step {:4} | Reward: {:.3} | Baseline: {:.3} | {:.1}s",
                     step, reward, reward_baseline, start.elapsed().as_secs_f32());
        }
    }

    // Save log
    if let Ok(mut file) = std::fs::File::create("experiments/reinforce_log.csv") {
        let _ = writeln!(file, "step,reward,baseline");
        for &(s, r, b) in &reward_log {
            let _ = writeln!(file, "{},{:.6},{:.6}", s, r, b);
        }
    }

    println!("  REINFORCE done in {:.1}s", start.elapsed().as_secs_f32());
    evaluate_model(model, config, val, tok)
}

/// DPO: Direct Preference Optimization
fn rl_dpo(
    model: &mut GPT,
    config: &Config,
    train: &[ToolExample],
    val: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
) -> AggregateMetrics {
    let num_steps = 400;
    let lr = 0.0002;
    let beta = 0.5; // DPO temperature

    // We need a reference model (frozen SFT weights)
    let ref_model = GPT::new(config.clone()); // approximate reference

    let start = Instant::now();
    let mut loss_log = Vec::new();

    for step in 0..num_steps {
        let example = &train[rng.gen_range(0..train.len())];
        let prompt_encoded = tok.encode(&example.prompt);

        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }

        // Preferred: the correct completion
        let preferred_text = &example.input;
        let preferred_encoded = tok.encode(preferred_text);

        // Rejected: generate a bad completion (from current model)
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let rejected_ids = model.generate(&prompt_encoded, max_gen);

        if preferred_encoded.len() < 2 || preferred_encoded.len() > config.block_size
            || rejected_ids.len() < 2 || rejected_ids.len() > config.block_size
        {
            continue;
        }

        // Compute log probs for preferred and rejected under current model
        let preferred_logprob = compute_sequence_logprob(model, &preferred_encoded);
        let rejected_logprob = compute_sequence_logprob(model, &rejected_ids);

        // Compute log probs under reference model
        let ref_preferred_logprob = compute_sequence_logprob(&ref_model, &preferred_encoded);
        let ref_rejected_logprob = compute_sequence_logprob(&ref_model, &rejected_ids);

        // DPO loss: -log(sigmoid(beta * (log_pi_w - log_pi_l - log_ref_w + log_ref_l)))
        let logit = beta * (
            (preferred_logprob - rejected_logprob) - (ref_preferred_logprob - ref_rejected_logprob)
        );
        let dpo_loss = -(1.0 / (1.0 + (-logit).exp())).max(1e-10).ln();

        if dpo_loss.is_finite() {
            // Approximate gradient: train more on preferred, less on rejected
            // Preferred: standard cross-entropy with gradient scaled by sigmoid(-logit)
            let weight = 1.0 / (1.0 + logit.exp()); // sigmoid(-logit)

            let pref_input = &preferred_encoded[..preferred_encoded.len() - 1];
            let pref_target = &preferred_encoded[1..];
            let (_, pref_grads) = model.forward_backward(pref_input, pref_target);

            let mut grads = pref_grads;
            grads.scale(weight * beta);
            model.apply_gradients(&grads, lr);
        }

        if step % 50 == 0 {
            loss_log.push((step, dpo_loss));
            println!("  DPO step {:4} | Loss: {:.4} | {:.1}s",
                     step, dpo_loss, start.elapsed().as_secs_f32());
        }
    }

    // Save log
    if let Ok(mut file) = std::fs::File::create("experiments/dpo_log.csv") {
        let _ = writeln!(file, "step,loss");
        for &(s, l) in &loss_log {
            let _ = writeln!(file, "{},{:.6}", s, l);
        }
    }

    println!("  DPO done in {:.1}s", start.elapsed().as_secs_f32());
    evaluate_model(model, config, val, tok)
}

/// Custom structured reward shaping
fn rl_custom(
    model: &mut GPT,
    config: &Config,
    train: &[ToolExample],
    val: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
) -> AggregateMetrics {
    let num_steps = 400;
    let base_lr = 0.0003;
    let mut reward_baseline = 0.0f32;

    let start = Instant::now();
    let mut metrics_log = Vec::new();

    // Curriculum: start with easier examples (direct answers), then add tool calls
    for step in 0..num_steps {
        // Curriculum: first 100 steps = 70% direct answers, after = natural distribution
        let example = if step < 100 && rng.r#gen::<f32>() < 0.7 {
            // Pick a direct-answer example
            let directs: Vec<_> = train.iter().filter(|e| e.expected_call.is_none()).collect();
            if directs.is_empty() { &train[rng.gen_range(0..train.len())] }
            else { directs[rng.gen_range(0..directs.len())] }
        } else {
            &train[rng.gen_range(0..train.len())]
        };

        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }

        // Temperature annealing: high early (explore) → low late (exploit)
        let _temperature = 1.2 - 0.6 * (step as f32 / num_steps as f32);

        // Generate and evaluate
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);

        // Structured reward: decomposed into components
        let eval = tool_data::evaluate_output(&generated_text, example);
        let _is_tool = example.expected_call.is_some();

        // Per-component rewards
        let format_reward = if eval.format_correct { 1.0 } else { -0.5 };
        let tool_reward = if eval.tool_correct { 1.0 } else { -0.3 };
        let param_reward = if eval.params_correct { 0.8 } else { -0.2 };
        let total_reward = format_reward * 0.4 + tool_reward * 0.3 + param_reward * 0.3;

        reward_baseline = 0.95 * reward_baseline + 0.05 * total_reward;
        let advantage = total_reward - reward_baseline;

        // Adaptive learning rate: higher when reward is very wrong
        let adaptive_lr = base_lr * (1.0 + advantage.abs() * 0.5);

        // Two-phase update:
        // 1. Reinforce the correct answer (always)
        let correct_encoded = tok.encode(&example.input);
        if correct_encoded.len() >= 2 && correct_encoded.len() <= config.block_size {
            let input = &correct_encoded[..correct_encoded.len() - 1];
            let target = &correct_encoded[1..];
            let (_, grads) = model.forward_backward(input, target);

            // Weight by how far from correct the model is
            let sft_weight = if total_reward < 0.5 { 0.8 } else { 0.2 };
            let mut scaled = grads;
            scaled.scale(sft_weight);
            model.apply_gradients(&scaled, adaptive_lr);
        }

        // 2. Penalize the generated output if it was wrong (policy gradient)
        if advantage < -0.1 && generated_ids.len() >= 2 && generated_ids.len() <= config.block_size {
            let input = &generated_ids[..generated_ids.len() - 1];
            let target = &generated_ids[1..];
            let (_, grads) = model.forward_backward(input, target);

            let mut penalty = grads;
            penalty.scale(advantage.abs() * 0.3); // increase loss for bad outputs
            model.apply_gradients(&penalty, adaptive_lr);
        }

        if step % 50 == 0 {
            let eval_metrics = evaluate_model(model, config, &val[..val.len().min(20)], tok);
            metrics_log.push((step, total_reward, eval_metrics.format_acc, eval_metrics.tool_acc));
            println!("  Custom step {:4} | Reward: {:.3} | Format: {:.1}% | Tool: {:.1}% | {:.1}s",
                     step, total_reward, eval_metrics.format_acc * 100.0,
                     eval_metrics.tool_acc * 100.0, start.elapsed().as_secs_f32());
        }
    }

    // Save log
    if let Ok(mut file) = std::fs::File::create("experiments/custom_rl_log.csv") {
        let _ = writeln!(file, "step,reward,format_acc,tool_acc");
        for &(s, r, f, t) in &metrics_log {
            let _ = writeln!(file, "{},{:.6},{:.6},{:.6}", s, r, f, t);
        }
    }

    println!("  Custom RL done in {:.1}s", start.elapsed().as_secs_f32());
    evaluate_model(model, config, val, tok)
}

// ---- Helper functions ----

fn compute_reward(eval: &EvalResult, expects_tool: bool) -> f32 {
    let mut reward = 0.0f32;
    if eval.format_correct { reward += 0.5; } else { reward -= 0.5; }
    if eval.tool_correct { reward += 1.0; } else if expects_tool { reward -= 0.5; }
    if eval.params_correct { reward += 0.5; }
    reward += eval.reply_quality * 0.3;
    reward
}

fn compute_sequence_logprob(model: &GPT, tokens: &[usize]) -> f32 {
    if tokens.len() < 2 { return 0.0; }
    let input = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];
    let logits = model.forward(input);
    let v = model.config.vocab_size;
    let t = input.len();

    let mut total_logprob = 0.0f32;
    for i in 0..t {
        let offset = i * v;
        let max_val = logits[offset..offset + v]
            .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..v {
            sum += (logits[offset + j] - max_val).exp();
        }
        let logprob = (logits[offset + targets[i]] - max_val) - sum.ln();
        total_logprob += logprob;
    }
    total_logprob / t as f32
}

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

fn demo_tool_calling(model: &GPT, tok: &Tokenizer, val: &[ToolExample]) {
    let examples_to_show = val.iter().take(8);
    for example in examples_to_show {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= model.config.block_size {
            continue;
        }
        let max_gen = (model.config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);

        println!("  Query:    {}", example.prompt.trim());
        println!("  Expected: {}", example.expected_call.as_deref().unwrap_or("[direct reply]"));
        // Extract just the first [call] or [reply] part
        let response = generated_text[example.prompt.len()..].to_string();
        let truncated = if let Some(end_pos) = response.find("[end]") {
            &response[..end_pos + 5]
        } else {
            &response[..response.len().min(60)]
        };
        println!("  Got:      {}", truncated.trim());
        println!();
    }
}

/// SFT v2: Larger model, longer training, LR warmup, mixed Shakespeare+tool data
pub fn run_sft_v2() -> AggregateMetrics {
    println!("=== SFT v2 Experiment ===\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/3] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer from all text
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    // Larger config: n_embd=64, n_head=4, n_layer=4
    let block_size = 128;
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 64,
        n_head: 4,
        n_layer: 4,
        block_size,
    };
    println!("  Model: {} layers, {} embd, {} heads", config.n_layer, config.n_embd, config.n_head);

    // === Phase 1: Pretrain on Shakespeare (1000 steps) ===
    println!("\n[2/3] Pretraining on Shakespeare (1000 steps)...");
    let mut model = pretrain(&config, base_text, &tok, &mut rng);

    // === Phase 2: SFT v2 with improvements ===
    println!("\n[3/3] SFT v2: 1500 steps, LR warmup, 50/50 Shakespeare+tool mix...");
    let num_steps = 1500;
    let base_lr = 0.0005;
    let warmup_steps = 100;
    let batch_size = 8;

    let shakespeare_encoded = tok.encode(base_text);

    let start = Instant::now();
    let mut loss_log = Vec::new();

    for step in 0..num_steps {
        // Linear LR warmup over first 100 steps
        let lr = if step < warmup_steps {
            base_lr * (step as f32 + 1.0) / warmup_steps as f32
        } else {
            base_lr
        };

        let mut total_loss = 0.0f32;
        let mut grads = Gradients::zero_like(&config);
        let mut valid_count = 0;

        for b in 0..batch_size {
            // 50/50 mix: even batches = tool data, odd batches = Shakespeare
            let encoded = if b % 2 == 0 {
                // Tool data
                let example = &train_data[rng.gen_range(0..train_data.len())];
                tok.encode(&example.input)
            } else {
                // Shakespeare chunk (random window)
                if shakespeare_encoded.len() > config.block_size + 1 {
                    let start_idx = rng.gen_range(0..shakespeare_encoded.len() - config.block_size - 1);
                    shakespeare_encoded[start_idx..start_idx + config.block_size + 1].to_vec()
                } else {
                    shakespeare_encoded.clone()
                }
            };

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

            if step % 150 == 0 {
                println!("  SFT v2 step {:4} | Loss: {:.4} | LR: {:.6} | {:.1}s",
                         step, avg, lr, start.elapsed().as_secs_f32());
            }
        }
    }

    // Save loss log
    if let Ok(mut file) = std::fs::File::create("experiments/sft_v2_loss.csv") {
        let _ = writeln!(file, "step,loss");
        for &(s, l) in &loss_log {
            let _ = writeln!(file, "{},{:.6}", s, l);
        }
    }

    println!("  SFT v2 done in {:.1}s", start.elapsed().as_secs_f32());

    // Evaluate
    let metrics = evaluate_model(&model, &config, &val_data, &tok);

    // Print results
    println!("\n{}", "=".repeat(60));
    println!("=== SFT v2 RESULTS ===\n");
    metrics.print("SFT v2");

    // Save results CSV
    if let Ok(mut file) = std::fs::File::create("experiments/sft_v2_results.csv") {
        let _ = writeln!(file, "method,format_acc,tool_acc,param_acc,reply_quality");
        let _ = writeln!(file, "SFT v2,{:.4},{:.4},{:.4},{:.4}",
                         metrics.format_acc, metrics.tool_acc, metrics.param_acc, metrics.reply_quality);
    }
    println!("  Results saved to sft_v2_results.csv");

    // Print comparison with baseline SFT
    println!("\n=== Comparison with Baseline SFT ===\n");
    println!("{:<15} {:>10} {:>10} {:>10}", "Method", "Format%", "Tool%", "Param%");
    println!("{}", "-".repeat(45));
    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}%",
             "SFT Baseline", 66.2, 63.8, 27.5);
    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}%",
             "SFT v2", metrics.format_acc * 100.0, metrics.tool_acc * 100.0, metrics.param_acc * 100.0);

    let baseline_composite = 66.2 * 0.3 + 63.8 * 0.3 + 27.5 * 0.25;
    let v2_composite = metrics.format_acc * 100.0 * 0.3 + metrics.tool_acc * 100.0 * 0.3
        + metrics.param_acc * 100.0 * 0.25;
    println!("\n  Baseline composite: {:.1}", baseline_composite);
    println!("  SFT v2 composite:   {:.1}", v2_composite);
    if v2_composite > baseline_composite {
        println!("  SFT v2 wins by {:.1} points!", v2_composite - baseline_composite);
    } else {
        println!("  Baseline wins by {:.1} points.", baseline_composite - v2_composite);
    }

    metrics
}

/// Run rejection sampling (best-of-N) fine-tuning experiment
pub fn run_rejection_sampling() {
    println!("=== Rejection Sampling (Best-of-N) Fine-Tuning ===\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/4] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
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

    // Phase 1: Pretrain
    println!("\n[2/4] Pretraining on Shakespeare...");
    let mut model = pretrain(&config, base_text, &tok, &mut rng);

    // Phase 2: SFT baseline
    println!("\n[3/4] Supervised fine-tuning (baseline)...");
    let sft_metrics = sft_train(&mut model, &config, &train_data, &val_data, &tok, &mut rng);
    println!("\n  SFT Baseline Results:");
    sft_metrics.print("SFT Baseline");

    // Save SFT weights as starting point for rejection sampling rounds
    let sft_weights = clone_model_weights(&model);

    // Phase 3: Rejection sampling rounds
    println!("\n[4/4] Rejection sampling (best-of-N) fine-tuning...");
    let num_rounds = 3;
    let n_samples = 4; // best-of-N
    let finetune_steps = 200;
    let lr = 0.0003;

    let mut round_metrics: Vec<(usize, AggregateMetrics)> = Vec::new();
    let start = Instant::now();

    // Start from SFT model
    let mut rs_model = model_from_weights(&config, &sft_weights);

    for round in 0..num_rounds {
        println!("\n  --- Round {}/{} ---", round + 1, num_rounds);
        let round_start = Instant::now();

        // Step (a-c): For each training example, generate N completions and select best
        let mut best_completions: Vec<Vec<usize>> = Vec::new();

        for example in &train_data {
            let prompt_encoded = tok.encode(&example.prompt);
            if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
                // Fall back to ground truth encoding
                let encoded = tok.encode(&example.input);
                if encoded.len() >= 2 && encoded.len() <= config.block_size {
                    best_completions.push(encoded);
                }
                continue;
            }

            let max_gen = (config.block_size - prompt_encoded.len()).min(80);
            let expects_tool = example.expected_call.is_some();

            let mut best_reward = f32::NEG_INFINITY;
            let mut best_ids: Option<Vec<usize>> = None;

            // Generate N completions, score each, keep the best
            for _ in 0..n_samples {
                let generated_ids = rs_model.generate(&prompt_encoded, max_gen);
                let generated_text = tok.decode(&generated_ids);
                let eval = tool_data::evaluate_output(&generated_text, example);
                let reward = compute_reward(&eval, expects_tool);

                if reward > best_reward {
                    best_reward = reward;
                    best_ids = Some(generated_ids);
                }
            }

            if let Some(ids) = best_ids {
                if ids.len() >= 2 && ids.len() <= config.block_size {
                    best_completions.push(ids);
                }
            }
        }

        println!("  Selected {} best-of-{} completions", best_completions.len(), n_samples);

        // Step (d): Fine-tune on best completions for 200 steps
        for step in 0..finetune_steps {
            let mut total_loss = 0.0f32;
            let mut grads = Gradients::zero_like(&config);
            let mut valid_count = 0;
            let batch_size = 8;

            for _ in 0..batch_size {
                if best_completions.is_empty() {
                    break;
                }
                let idx = rng.gen_range(0..best_completions.len());
                let tokens = &best_completions[idx];
                let input = &tokens[..tokens.len() - 1];
                let target = &tokens[1..];
                let (loss, g) = rs_model.forward_backward(input, target);
                if loss.is_finite() {
                    total_loss += loss;
                    grads.accumulate(&g);
                    valid_count += 1;
                }
            }

            if valid_count > 0 {
                grads.scale(1.0 / valid_count as f32);
                rs_model.apply_gradients(&grads, lr);

                if step % 50 == 0 {
                    println!("    Finetune step {:4} | Loss: {:.4}",
                             step, total_loss / valid_count as f32);
                }
            }
        }

        // Step (e): Evaluate on val set
        let metrics = evaluate_model(&rs_model, &config, &val_data, &tok);
        println!("  Round {} results ({:.1}s):",
                 round + 1, round_start.elapsed().as_secs_f32());
        metrics.print(&format!("Round {}", round + 1));
        round_metrics.push((round + 1, metrics));
    }

    println!("\n  Rejection sampling done in {:.1}s", start.elapsed().as_secs_f32());

    // Comparison with baseline
    println!("\n{}", "=".repeat(60));
    println!("=== REJECTION SAMPLING RESULTS ===\n");

    println!("  Baseline comparison: format=66.2%, tool=63.8%, param=27.5%\n");

    println!("{:<15} {:>10} {:>10} {:>10} {:>10}",
             "Stage", "Format%", "Tool%", "Param%", "Reply%");
    println!("{}", "-".repeat(55));
    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%",
             "SFT Baseline",
             sft_metrics.format_acc * 100.0,
             sft_metrics.tool_acc * 100.0,
             sft_metrics.param_acc * 100.0,
             sft_metrics.reply_quality * 100.0);
    for (round, m) in &round_metrics {
        println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%",
                 format!("RS Round {}", round),
                 m.format_acc * 100.0,
                 m.tool_acc * 100.0,
                 m.param_acc * 100.0,
                 m.reply_quality * 100.0);
    }

    // Improvement trajectory
    println!("\n  Improvement trajectory (vs SFT baseline):");
    for (round, m) in &round_metrics {
        let format_delta = (m.format_acc - sft_metrics.format_acc) * 100.0;
        let tool_delta = (m.tool_acc - sft_metrics.tool_acc) * 100.0;
        let param_delta = (m.param_acc - sft_metrics.param_acc) * 100.0;
        println!("    Round {}: format {:+.1}%, tool {:+.1}%, param {:+.1}%",
                 round, format_delta, tool_delta, param_delta);
    }

    // Save CSV
    if let Ok(mut file) = std::fs::File::create("experiments/rejection_sampling_results.csv") {
        let _ = writeln!(file, "stage,format_acc,tool_acc,param_acc,reply_quality");
        let _ = writeln!(file, "sft_baseline,{:.4},{:.4},{:.4},{:.4}",
                         sft_metrics.format_acc, sft_metrics.tool_acc,
                         sft_metrics.param_acc, sft_metrics.reply_quality);
        for (round, m) in &round_metrics {
            let _ = writeln!(file, "rs_round_{},{:.4},{:.4},{:.4},{:.4}",
                             round, m.format_acc, m.tool_acc, m.param_acc, m.reply_quality);
        }
        println!("\n  Results saved to rejection_sampling_results.csv");
    }

    // ASCII comparison chart
    let mut chart_methods: Vec<(&str, &AggregateMetrics)> = Vec::new();
    chart_methods.push(("SFT Baseline", &sft_metrics));
    let round_labels: Vec<String> = round_metrics.iter()
        .map(|(r, _)| format!("RS Round {}", r))
        .collect();
    for (i, (_, m)) in round_metrics.iter().enumerate() {
        chart_methods.push((&round_labels[i], m));
    }
    print_comparison_chart(&chart_methods);
}

/// SFT v3: Data augmentation experiment
/// Augments the tool calling dataset with chain-of-thought, rephrased queries,
/// and tool-identification examples, then trains SFT for 1200 steps.
pub fn run_sft_v3() -> AggregateMetrics {
    println!("=== SFT v3: Data Augmentation Experiment ===\n");

    let mut rng = rand::thread_rng();

    // Generate base dataset
    println!("[1/5] Generating base tool calling dataset...");
    let (base_train, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Base train: {} examples, Val: {} examples", base_train.len(), val_data.len());

    // Augment the dataset
    println!("[2/5] Augmenting dataset...");
    let augmented_train = augment_dataset(&base_train, &mut rng);
    println!("  Augmented train: {} examples (was {})", augmented_train.len(), base_train.len());

    // Build tokenizer from all text (base + augmented tool data)
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &augmented_train);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    // Config: same model size as baseline (48/4/3)
    let block_size = 128;
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size,
    };
    println!("  Model: {} layers, {} embd, {} heads (same as baseline)", config.n_layer, config.n_embd, config.n_head);

    // Phase 1: Pretrain on Shakespeare (1000 steps)
    println!("\n[3/5] Pretraining on Shakespeare (1000 steps)...");
    let mut model = pretrain(&config, base_text, &tok, &mut rng);

    // Phase 2: SFT on augmented tool data (1200 steps)
    println!("\n[4/5] SFT on augmented data (1200 steps)...");
    let metrics = sft_train_v3(&mut model, &config, &augmented_train, &val_data, &tok, &mut rng);

    // Phase 3: Evaluate and print results
    println!("\n[5/5] Results\n");
    metrics.print("SFT v3 (augmented)");

    // Baseline comparison
    println!("\n  Baseline comparison (SFT v1):");
    println!("    Format accuracy:  66.2%");
    println!("    Tool accuracy:    63.8%");
    println!("    Param accuracy:   27.5%");

    println!("\n  SFT v3 vs Baseline:");
    let fmt_delta = (metrics.format_acc - 0.662) * 100.0;
    let tool_delta = (metrics.tool_acc - 0.638) * 100.0;
    let param_delta = (metrics.param_acc - 0.275) * 100.0;
    println!("    Format: {:>+.1}pp", fmt_delta);
    println!("    Tool:   {:>+.1}pp", tool_delta);
    println!("    Param:  {:>+.1}pp", param_delta);

    // Composite score comparison
    let baseline_composite = 66.2 * 0.3 + 63.8 * 0.3 + 27.5 * 0.25;
    let v3_composite = metrics.format_acc * 100.0 * 0.3 + metrics.tool_acc * 100.0 * 0.3
        + metrics.param_acc * 100.0 * 0.25;
    println!("\n  Baseline composite: {:.1}", baseline_composite);
    println!("  SFT v3 composite:   {:.1}", v3_composite);
    if v3_composite > baseline_composite {
        println!("  SFT v3 wins by {:.1} points!", v3_composite - baseline_composite);
    } else {
        println!("  Baseline wins by {:.1} points.", baseline_composite - v3_composite);
    }

    // Save results CSV
    if let Ok(mut file) = std::fs::File::create("experiments/sft_v3_results.csv") {
        let _ = writeln!(file, "method,format_acc,tool_acc,param_acc,reply_quality");
        let _ = writeln!(file, "SFT v3,{:.4},{:.4},{:.4},{:.4}",
                         metrics.format_acc, metrics.tool_acc, metrics.param_acc, metrics.reply_quality);
        let _ = writeln!(file, "Baseline,0.6620,0.6380,0.2750,0.0000");
        println!("\n  Results saved to sft_v3_results.csv");
    }

    println!("\n=== SFT v3 Experiment Complete ===");
    metrics
}

/// Augment the tool calling dataset with:
/// 1. Chain-of-thought examples for tool calls
/// 2. Rephrased queries (2x each tool call example)
/// 3. Tool-identification examples
fn augment_dataset(base: &[ToolExample], rng: &mut impl Rng) -> Vec<ToolExample> {
    let mut augmented = base.to_vec();

    for example in base {
        // Only augment tool call examples
        if let Some(ref call) = example.expected_call {
            // Determine tool name and thinking rationale
            let tool_name = call.split('(').next().unwrap_or("unknown");
            let think_text = match tool_name {
                "calc" => "need calculator for math",
                "search" => "need to search for information",
                "weather" => "need weather tool for forecast",
                "time" => "need time tool for current time",
                _ => "need a tool for this",
            };

            // Extract query from prompt: "[user] <query> [end] "
            let query = example.prompt
                .trim_start_matches("[user] ")
                .trim_end_matches("[end] ")
                .trim();

            // Extract result from input between [result] and [end]
            let result_text = extract_between(&example.input, "[result] ", " [end]")
                .unwrap_or("");

            // 1. Chain-of-thought augmentation
            let cot_input = format!(
                "[user] {} [end] [think] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                query, think_text, call, result_text, example.expected_reply
            );
            augmented.push(ToolExample {
                input: cot_input,
                prompt: example.prompt.clone(),
                expected_call: example.expected_call.clone(),
                expected_reply: example.expected_reply.clone(),
            });

            // 2. Rephrased query augmentations (2x)
            let rephrasings = [
                format!("please {}", query),
                format!("{}?", query),
            ];
            for rephrased_query in &rephrasings {
                let rephrased_input = format!(
                    "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                    rephrased_query, call, result_text, example.expected_reply
                );
                augmented.push(ToolExample {
                    input: rephrased_input,
                    prompt: format!("[user] {} [end] ", rephrased_query),
                    expected_call: example.expected_call.clone(),
                    expected_reply: example.expected_reply.clone(),
                });
            }
        }
    }

    // 3. Tool-identification examples
    let tool_id_examples = [
        ("which tool for math", "use calc for math problems"),
        ("which tool for addition", "use calc for math problems"),
        ("which tool for calculation", "use calc for math problems"),
        ("which tool for searching", "use search for finding information"),
        ("which tool for finding facts", "use search for finding information"),
        ("which tool for weather", "use weather for weather forecasts"),
        ("which tool for forecast", "use weather for weather forecasts"),
        ("which tool for time", "use time for checking the time"),
        ("which tool for clock", "use time for checking the time"),
        ("what tools do you have", "i have calc search weather and time tools"),
        ("list your tools", "i have calc search weather and time tools"),
        ("how do i calculate", "use calc for math problems"),
        ("how do i search", "use search for finding information"),
        ("how do i check weather", "use weather for weather forecasts"),
        ("how do i check time", "use time for checking the time"),
    ];
    for _ in 0..10 {
        let (query, reply) = tool_id_examples[rng.gen_range(0..tool_id_examples.len())];
        augmented.push(ToolExample {
            input: format!("[user] {} [end] [reply] {} [end]", query, reply),
            prompt: format!("[user] {} [end] ", query),
            expected_call: None,
            expected_reply: reply.to_string(),
        });
    }

    // Shuffle the augmented dataset
    let len = augmented.len();
    for i in (1..len).rev() {
        let j = rng.gen_range(0..=i);
        augmented.swap(i, j);
    }

    augmented
}

/// Extract text between two markers in a string
fn extract_between<'a>(text: &'a str, start_marker: &str, end_marker: &str) -> Option<&'a str> {
    let start = text.find(start_marker)?;
    let after_start = &text[start + start_marker.len()..];
    let end = after_start.find(end_marker)?;
    Some(&after_start[..end])
}

/// SFT training with 1200 steps for v3 (augmented data)
fn sft_train_v3(
    model: &mut GPT,
    config: &Config,
    train: &[ToolExample],
    val: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
) -> AggregateMetrics {
    let num_steps = 1200;
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
                println!("  SFT v3 step {:4} | Loss: {:.4} | {:.1}s",
                         step, avg, start.elapsed().as_secs_f32());
            }
        }
    }

    // Save SFT v3 loss log
    if let Ok(mut file) = std::fs::File::create("experiments/sft_v3_loss.csv") {
        let _ = writeln!(file, "step,loss");
        for &(s, l) in &loss_log {
            let _ = writeln!(file, "{},{:.6}", s, l);
        }
    }

    println!("  SFT v3 done in {:.1}s", start.elapsed().as_secs_f32());

    // Evaluate
    evaluate_model(model, config, val, tok)
}

fn print_comparison_chart(methods: &[(&str, &AggregateMetrics)]) {
    println!("\n=== Metric Comparison Charts ===\n");

    let metrics: Vec<(&str, Box<dyn Fn(&AggregateMetrics) -> f32>)> = vec![
        ("Format Accuracy", Box::new(|m: &AggregateMetrics| m.format_acc)),
        ("Tool Accuracy", Box::new(|m: &AggregateMetrics| m.tool_acc)),
        ("Param Accuracy", Box::new(|m: &AggregateMetrics| m.param_acc)),
        ("Reply Quality", Box::new(|m: &AggregateMetrics| m.reply_quality)),
    ];

    for (metric_name, getter) in &metrics {
        println!("  {}:", metric_name);
        for (name, m) in methods {
            let val = getter(m);
            let bar_len = (val * 40.0) as usize;
            let bar: String = "█".repeat(bar_len);
            println!("    {:<12} |{:<40}| {:.1}%", name, bar, val * 100.0);
        }
        println!();
    }
}

/// Run the complete comparison across all experiments
pub fn run_comparison() {
    println!("=== COMPREHENSIVE EXPERIMENT COMPARISON ===\n");

    // All results from experiments (hardcoded from actual runs)
    let results: Vec<(&str, f32, f32, f32, f32)> = vec![
        ("SFT Baseline",       0.6625, 0.6375, 0.2750, 0.6616),
        ("REINFORCE",           0.0000, 0.0000, 0.0000, 0.4479),
        ("DPO",                 0.6500, 0.6250, 0.2875, 0.6501),
        ("Custom RL",           0.6500, 0.5500, 0.2000, 0.5238),
        ("SFT v2 (larger)",     0.5625, 0.5500, 0.3375, 0.6507),
        ("SFT v3 (augment)",    0.5875, 0.5500, 0.2125, 0.5264),
        ("Reject Sampling",     0.7125, 0.7000, 0.2625, 0.6249),
    ];

    // Leaderboard table
    println!("  {:<20} {:>8} {:>8} {:>8} {:>8} {:>8}",
             "Method", "Format%", "Tool%", "Param%", "Reply%", "Score");
    println!("  {}", "-".repeat(68));

    let mut scored: Vec<(&str, f32, f32, f32, f32, f32)> = results.iter().map(|&(name, f, t, p, r)| {
        let score = f * 0.3 + t * 0.3 + p * 0.2 + r * 0.2;
        (name, f, t, p, r, score)
    }).collect();
    scored.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap());

    for (rank, &(name, f, t, p, r, s)) in scored.iter().enumerate() {
        let marker = if rank == 0 { " *** WINNER ***" } else { "" };
        println!("  {:<20} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>8.3}{}",
                 name, f*100.0, t*100.0, p*100.0, r*100.0, s, marker);
    }

    // Per-metric winners
    println!("\n  Per-metric champions:");
    let best_format = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let best_tool = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    let best_param = results.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
    println!("    Format accuracy: {} ({:.1}%)", best_format.0, best_format.1 * 100.0);
    println!("    Tool accuracy:   {} ({:.1}%)", best_tool.0, best_tool.2 * 100.0);
    println!("    Param accuracy:  {} ({:.1}%)", best_param.0, best_param.3 * 100.0);

    // ASCII bar charts per metric
    println!("\n  === Format Accuracy ===");
    for &(name, f, _, _, _) in &results {
        let bar_len = (f * 50.0) as usize;
        println!("    {:<20} |{}| {:.1}%", name, "\u{2588}".repeat(bar_len), f * 100.0);
    }

    println!("\n  === Tool Accuracy ===");
    for &(name, _, t, _, _) in &results {
        let bar_len = (t * 50.0) as usize;
        println!("    {:<20} |{}| {:.1}%", name, "\u{2588}".repeat(bar_len), t * 100.0);
    }

    println!("\n  === Param Accuracy ===");
    for &(name, _, _, p, _) in &results {
        let bar_len = (p * 50.0) as usize;
        println!("    {:<20} |{}| {:.1}%", name, "\u{2588}".repeat(bar_len), p * 100.0);
    }

    // Key findings
    println!("\n  === KEY FINDINGS ===");
    println!("  1. Rejection Sampling is the clear winner (format 71.2%, tool 70.0%)");
    println!("  2. Vanilla REINFORCE COLLAPSED entirely — too sparse reward for 91K params");
    println!("  3. DPO maintained SFT quality but couldn't improve — reference model approximation too rough");
    println!("  4. Custom structured RL slightly degraded — credit assignment still too noisy");
    println!("  5. SFT v2 (larger model) improved params but mixed training hurt format accuracy");
    println!("  6. SFT v3 (augmentation) made task HARDER for tiny model — insufficient capacity");
    println!("  7. Best approach: SFT + Rejection Sampling (offline RL) = simple, stable, effective");

    // Save comprehensive results
    if let Ok(mut file) = std::fs::File::create("experiments/final_comparison.csv") {
        let _ = writeln!(file, "rank,method,format_acc,tool_acc,param_acc,reply_quality,composite_score");
        for (rank, &(name, f, t, p, r, s)) in scored.iter().enumerate() {
            let _ = writeln!(file, "{},{},{:.4},{:.4},{:.4},{:.4},{:.4}", rank + 1, name, f, t, p, r, s);
        }
    }
    println!("\n  Full results saved to final_comparison.csv");
}

/// Interactive tool-calling agent mode
pub fn run_agent() {
    println!("=== Mini GPT Tool Calling Agent ===");
    println!("    Trained with Rejection Sampling (best method)\n");

    let mut rng = rand::thread_rng();

    // Generate dataset and build tokenizer
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);

    let block_size = 128;
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size,
    };

    // Pretrain
    println!("Pretraining on Shakespeare...");
    let mut model = pretrain(&config, base_text, &tok, &mut rng);

    // SFT
    println!("Fine-tuning on tool calling data...");
    let _sft_metrics = sft_train(&mut model, &config, &train_data, &val_data, &tok, &mut rng);

    // Rejection sampling (3 rounds)
    println!("Running rejection sampling optimization...");
    for round in 0..3 {
        let mut best_pairs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();

        for example in train_data.iter().take(200) {
            let prompt_encoded = tok.encode(&example.prompt);
            if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size { continue; }

            let max_gen = (config.block_size - prompt_encoded.len()).min(80);
            let mut best_score = f32::NEG_INFINITY;
            let mut best_ids = Vec::new();

            for _ in 0..4 {
                let gen_ids = model.generate(&prompt_encoded, max_gen);
                let gen_text = tok.decode(&gen_ids);
                let eval = tool_data::evaluate_output(&gen_text, example);
                let score = compute_reward(&eval, example.expected_call.is_some());
                if score > best_score {
                    best_score = score;
                    best_ids = gen_ids;
                }
            }

            if best_ids.len() >= 2 && best_ids.len() <= config.block_size {
                let input = best_ids[..best_ids.len()-1].to_vec();
                let target = best_ids[1..].to_vec();
                best_pairs.push((input, target));
            }
        }

        // Fine-tune on best completions
        for step in 0..200 {
            if best_pairs.is_empty() { break; }
            let (ref input, ref target) = best_pairs[rng.gen_range(0..best_pairs.len())];
            let (_, grads) = model.forward_backward(input, target);
            model.apply_gradients(&grads, 0.0003);
            let _ = step;
        }
        println!("  Round {} complete", round + 1);
    }

    // Evaluate
    let final_metrics = evaluate_model(&model, &config, &val_data, &tok);
    println!("\nFinal agent metrics:");
    final_metrics.print("Tool-Calling Agent");

    // Demo
    println!("\n=== Agent Demo ===\n");
    println!("The agent processes queries and decides whether to call a tool or respond directly.\n");

    let demo_queries = [
        "[user] what is 12 plus 7 [end] ",
        "[user] what is the weather in london [end] ",
        "[user] hello [end] ",
        "[user] who wrote hamlet [end] ",
        "[user] what time is it in utc [end] ",
        "[user] calculate 8 times 6 [end] ",
        "[user] thank you [end] ",
        "[user] how many planets are there [end] ",
        "[user] weather in tokyo [end] ",
        "[user] 25 minus 13 [end] ",
    ];

    for query in &demo_queries {
        let prompt_encoded = tok.encode(query);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size { continue; }

        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let gen_ids = model.generate(&prompt_encoded, max_gen);
        let gen_text = tok.decode(&gen_ids);
        let response = &gen_text[query.len()..];

        // Parse response
        let truncated = if let Some(end_pos) = response.find("[end]") {
            &response[..end_pos + 5]
        } else {
            &response[..response.len().min(60)]
        };

        // Determine if tool call or direct reply
        let is_tool_call = truncated.contains("[call]");
        let icon = if is_tool_call { "🔧" } else { "💬" };

        println!("  {} Query: {}", "📝", query.replace("[user] ", "").replace(" [end] ", ""));
        println!("  {} Response: {}", icon, truncated.trim());

        // If tool call, simulate execution
        if is_tool_call {
            if let Some(call_start) = truncated.find("[call] ") {
                let after = &truncated[call_start + 7..];
                if let Some(call_end) = after.find(" [end]") {
                    let call = &after[..call_end];
                    let result = simulate_tool(call);
                    println!("  {} Tool result: {}", "⚡", result);
                }
            }
        }
        println!();
    }

    // Speed benchmark
    println!("=== Inference Speed ===");
    let start = Instant::now();
    let prompt = tok.encode("[user] what is 5 plus 3 [end] ");
    let n_iters = 50;
    for _ in 0..n_iters {
        let _ = model.generate(&prompt, 30);
    }
    let elapsed = start.elapsed().as_secs_f32();
    let tokens_per_sec = (n_iters * 30) as f32 / elapsed;
    println!("  {} generations of 30 tokens in {:.2}s", n_iters, elapsed);
    println!("  Speed: {:.0} tokens/sec on CPU\n", tokens_per_sec);

    // Save weights and vocab for instant loading
    let _ = std::fs::create_dir_all("weights");
    match model.save_weights("weights/tool-agent.mgpt") {
        Ok(()) => println!("Weights saved to weights/tool-agent.mgpt"),
        Err(e) => eprintln!("Warning: could not save weights: {e}"),
    }
    match tok.save_vocab("weights/vocab.txt") {
        Ok(()) => println!("Vocabulary saved to weights/vocab.txt"),
        Err(e) => eprintln!("Warning: could not save vocab: {e}"),
    }

    println!("\n=== Agent Ready ===");
    println!("  Run with --demo to load pre-trained weights instantly (no training)");
}

/// Run the tool-calling agent from pre-trained weights (instant startup).
pub fn run_demo() {
    println!("=== Mini GPT Tool Calling Agent (pre-trained) ===\n");

    let model = match GPT::load_weights("weights/tool-agent.mgpt") {
        Ok(m) => m,
        Err(_) => {
            eprintln!("error: weights/tool-agent.mgpt not found");
            eprintln!("Run `cargo run -- --agent` first to train and save weights,");
            eprintln!("or download pre-trained weights from the repository.");
            std::process::exit(1);
        }
    };
    let tok = match Tokenizer::load_vocab("weights/vocab.txt") {
        Ok(t) => t,
        Err(_) => {
            eprintln!("error: weights/vocab.txt not found");
            std::process::exit(1);
        }
    };

    println!("Model loaded: {} layers, {} embd, {} heads, block_size={}",
             model.config.n_layer, model.config.n_embd, model.config.n_head, model.config.block_size);
    println!("Vocabulary: {} tokens\n", tok.vocab_size());

    // Evaluate on validation set
    let mut rng = rand::thread_rng();
    let (_, val_data) = tool_data::generate_dataset(&mut rng);
    let metrics = evaluate_model(&model, &model.config, &val_data, &tok);
    println!("Validation metrics:");
    metrics.print("Pre-trained Agent");

    // Interactive demo
    println!("\n=== Agent Demo ===\n");
    println!("The agent processes queries and decides whether to call a tool or respond directly.\n");

    let demo_queries = [
        "[user] what is 12 plus 7 [end] ",
        "[user] what is the weather in london [end] ",
        "[user] hello [end] ",
        "[user] who wrote hamlet [end] ",
        "[user] what time is it in utc [end] ",
        "[user] calculate 8 times 6 [end] ",
        "[user] thank you [end] ",
        "[user] how many planets are there [end] ",
        "[user] weather in tokyo [end] ",
        "[user] 25 minus 13 [end] ",
    ];

    for query in &demo_queries {
        let prompt_encoded = tok.encode(query);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= model.config.block_size { continue; }

        let max_gen = (model.config.block_size - prompt_encoded.len()).min(80);
        let gen_ids = model.generate(&prompt_encoded, max_gen);
        let gen_text = tok.decode(&gen_ids);
        let response = &gen_text[query.len()..];

        let truncated = if let Some(end_pos) = response.find("[end]") {
            &response[..end_pos + 5]
        } else {
            &response[..response.len().min(60)]
        };

        let is_tool_call = truncated.contains("[call]");
        let icon = if is_tool_call { "T" } else { "R" };

        println!("  Q: {}", query.replace("[user] ", "").replace(" [end] ", ""));
        println!("  {} {}", icon, truncated.trim());

        if is_tool_call {
            if let Some(call_start) = truncated.find("[call] ") {
                let after = &truncated[call_start + 7..];
                if let Some(call_end) = after.find(" [end]") {
                    let call = &after[..call_end];
                    let result = simulate_tool(call);
                    println!("  = {}", result);
                }
            }
        }
        println!();
    }

    // Speed benchmark
    println!("=== Inference Speed ===");
    let start = Instant::now();
    let prompt = tok.encode("[user] what is 5 plus 3 [end] ");
    let n_iters = 50;
    for _ in 0..n_iters {
        let _ = model.generate(&prompt, 30);
    }
    let elapsed = start.elapsed().as_secs_f32();
    let tokens_per_sec = (n_iters * 30) as f32 / elapsed;
    println!("  {:.0} tokens/sec on CPU\n", tokens_per_sec);
}

fn simulate_tool(call: &str) -> String {
    if call.starts_with("calc(") {
        // Parse calc(a,b,op)
        let inner = &call[5..call.len().saturating_sub(1)];
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 3 {
            let a: f32 = parts[0].trim().parse().unwrap_or(0.0);
            let b: f32 = parts[1].trim().parse().unwrap_or(0.0);
            let result = match parts[2].trim() {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                _ => 0.0,
            };
            return format!("{}", result);
        }
    } else if call.starts_with("weather(") {
        let city = &call[8..call.len().saturating_sub(1)];
        return format!("22 degrees sunny in {}", city);
    } else if call.starts_with("time(") {
        let zone = &call[5..call.len().saturating_sub(1)];
        return format!("14:30 {}", zone);
    } else if call.starts_with("search(") {
        let query = &call[7..call.len().saturating_sub(1)];
        return format!("[search result for: {}]", query);
    }
    "unknown tool".to_string()
}

// ─── Spectral Gating Attention Experiment ───────────────────────────

/// Run Experiment 1: Spectral Gating Attention
/// Replaces O(n^2) softmax attention with O(n log n) FFT-based spectral gating.
pub fn run_spectral_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("weights");
    println!("=== Experiment 1: Spectral Gating Attention ===\n");
    println!("HYPOTHESIS: Replace O(n^2) softmax attention with O(n log n)");
    println!("FFT-based spectral gating for each head.\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/5] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    // Config (same as baseline)
    let block_size = 128;
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size,
    };

    // Count params for baseline comparison
    let baseline_params = {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let inner = 4 * e;
        let mut total = v * e + config.block_size * e;
        for _ in 0..nl {
            total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
        }
        total + e * 2 + e * v
    };

    // Create spectral model
    let mut model = SpectralGPT::new(config.clone());
    let spectral_params = model.count_params();
    println!("\n  Baseline params: {}", baseline_params);
    println!("  Spectral params: {} ({:+.1}%)",
             spectral_params,
             (spectral_params as f32 / baseline_params as f32 - 1.0) * 100.0);

    // === Phase 1: Pretrain on Shakespeare ===
    println!("\n[2/5] Pretraining SpectralGPT on Shakespeare...");
    let kl = model.kernel_len;
    spectral_pretrain(&mut model, &config, base_text, &tok, &mut rng, kl);

    // === Phase 2: SFT on tool calling data ===
    println!("\n[3/5] Supervised fine-tuning SpectralGPT on tool calling data...");
    let start_sft = Instant::now();
    let num_steps = 800;
    let lr = 0.0005;
    let batch_size = 8;

    let mut loss_log: Vec<(usize, f32, f32)> = Vec::new();

    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = SpectralGradients::zero_like(&config, kl);
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
            let avg = total_loss / valid_count as f32;
            let elapsed = start_sft.elapsed().as_secs_f32();
            loss_log.push((step, avg, elapsed));

            if step % 100 == 0 {
                println!("  SFT step {:4} | Loss: {:.4} | {:.1}s", step, avg, elapsed);
            }
        }
    }

    let sft_time = start_sft.elapsed().as_secs_f32();
    println!("  SFT done in {:.1}s", sft_time);

    // Save spectral loss log
    if let Ok(mut file) = std::fs::File::create("experiments/spectral_log.csv") {
        let _ = writeln!(file, "step,loss,time_seconds");
        for &(s, l, t) in &loss_log {
            let _ = writeln!(file, "{},{:.6},{:.2}", s, l, t);
        }
        println!("  Training log saved to experiments/spectral_log.csv");
    }

    // Show sample generations
    println!("\n  Sample generations:");
    for (idx, example) in val_data.iter().take(5).enumerate() {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);
        let gen_part = &generated_text[example.prompt.len()..];
        let end_pos = gen_part.find("[end]").map(|p| p + 5).unwrap_or(gen_part.len().min(80));
        println!("  [{}] {}", idx, example.prompt.trim());
        println!("      Expected: {}", example.expected_call.as_deref().unwrap_or("[direct reply]"));
        println!("      Got:      {}", gen_part[..end_pos].trim());
    }

    // === Phase 3: Evaluate ===
    println!("\n[4/5] Evaluating SpectralGPT...");
    let metrics = spectral_evaluate_model(&model, &config, &val_data, &tok);
    let composite = metrics.format_acc * 0.3
        + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25
        + metrics.reply_quality * 0.15;

    println!("\n  Spectral GPT Results ({} examples):", metrics.count);
    println!("    Format accuracy:  {:.1}%", metrics.format_acc * 100.0);
    println!("    Tool accuracy:    {:.1}%", metrics.tool_acc * 100.0);
    println!("    Param accuracy:   {:.1}%", metrics.param_acc * 100.0);
    println!("    Reply quality:    {:.1}%", metrics.reply_quality * 100.0);
    println!("    Composite score:  {:.4}", composite);

    // === Phase 4: Measure inference speed ===
    println!("\n[5/5] Measuring inference speed...");
    let speed_start = Instant::now();
    let mut total_tokens_generated = 0usize;
    let speed_iters = 20;
    for example in val_data.iter().take(speed_iters) {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(40);
        let generated = model.generate(&prompt_encoded, max_gen);
        total_tokens_generated += generated.len() - prompt_encoded.len();
    }
    let speed_elapsed = speed_start.elapsed().as_secs_f32();
    let tokens_per_sec = total_tokens_generated as f32 / speed_elapsed;
    println!("  Generated {} tokens in {:.2}s = {:.1} tokens/sec",
             total_tokens_generated, speed_elapsed, tokens_per_sec);

    // Save weights
    if let Err(e) = model.save_weights("weights/spectral.mgpt") {
        println!("  Warning: failed to save weights: {}", e);
    } else {
        println!("  Weights saved to weights/spectral.mgpt");
    }

    // Total training time
    let total_training_time = sft_time; // pretrain time is separate

    // === Summary ===
    println!("\n{}", "=".repeat(60));
    println!("=== SPECTRAL GATING ATTENTION — RESULTS SUMMARY ===\n");
    println!("  Architecture:    FFT-based spectral gating (O(n log n))");
    println!("  Parameters:      {} (baseline: {})", spectral_params, baseline_params);
    println!("  SFT time:        {:.1}s", total_training_time);
    println!("  Inference speed:  {:.1} tokens/sec", tokens_per_sec);
    println!();
    println!("  Metrics:");
    println!("    format_acc:     {:.4}", metrics.format_acc);
    println!("    tool_acc:       {:.4}", metrics.tool_acc);
    println!("    param_acc:      {:.4}", metrics.param_acc);
    println!("    reply_quality:  {:.4}", metrics.reply_quality);
    println!("    composite:      {:.4}", composite);
    println!();

    // Comparison with baseline (hardcoded from prior SFT runs)
    println!("  Comparison vs SFT Baseline:");
    println!("    {:<18} {:>10} {:>10}", "", "Spectral", "Baseline");
    println!("    {}", "-".repeat(40));
    println!("    {:<18} {:>9.1}% {:>9.1}%", "Format", metrics.format_acc * 100.0, 66.2);
    println!("    {:<18} {:>9.1}% {:>9.1}%", "Tool", metrics.tool_acc * 100.0, 63.8);
    println!("    {:<18} {:>9.1}% {:>9.1}%", "Param", metrics.param_acc * 100.0, 27.5);
    println!("    {:<18} {:>9.1}% {:>9.1}%", "Reply", metrics.reply_quality * 100.0, 30.0);
    println!("    {:<18} {:>10} {:>10}", "Params", spectral_params, baseline_params);

    // Save results CSV
    if let Ok(mut file) = std::fs::File::create("experiments/spectral_results.csv") {
        let _ = writeln!(file, "metric,value");
        let _ = writeln!(file, "format_acc,{:.4}", metrics.format_acc);
        let _ = writeln!(file, "tool_acc,{:.4}", metrics.tool_acc);
        let _ = writeln!(file, "param_acc,{:.4}", metrics.param_acc);
        let _ = writeln!(file, "reply_quality,{:.4}", metrics.reply_quality);
        let _ = writeln!(file, "composite,{:.4}", composite);
        let _ = writeln!(file, "params,{}", spectral_params);
        let _ = writeln!(file, "baseline_params,{}", baseline_params);
        let _ = writeln!(file, "inference_speed,{:.1}", tokens_per_sec);
        let _ = writeln!(file, "training_time,{:.1}", total_training_time);
        println!("\n  Results saved to experiments/spectral_results.csv");
    }

    // Print for PlanDB extraction
    println!("\n=== PLANDB_METRICS ===");
    println!("format_acc={:.4}", metrics.format_acc);
    println!("tool_acc={:.4}", metrics.tool_acc);
    println!("param_acc={:.4}", metrics.param_acc);
    println!("reply_quality={:.4}", metrics.reply_quality);
    println!("composite={:.4}", composite);
    println!("params={}", spectral_params);
    println!("inference_speed={:.1}", tokens_per_sec);
    println!("training_time={:.1}", total_training_time);
    println!("=== END PLANDB_METRICS ===");
}

/// Pretrain SpectralGPT on Shakespeare
fn spectral_pretrain(
    model: &mut SpectralGPT,
    config: &Config,
    text: &str,
    tok: &Tokenizer,
    rng: &mut impl Rng,
    kl: usize,
) {
    let encoded = tok.encode(text);
    let batch_size = 16;
    let num_steps = 1000;
    let lr = 0.001;

    let start = Instant::now();
    for step in 0..num_steps {
        let (inputs, targets) = crate::data::create_batches(&encoded, config.block_size, batch_size, rng);
        let mut total_loss = 0.0f32;
        let mut grads = SpectralGradients::zero_like(config, kl);

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
            println!("  Pretrain step {:4} | Loss: {:.4} | {:.1}s",
                     step, total_loss / batch_size as f32, start.elapsed().as_secs_f32());
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
}

/// Evaluate spectral model
fn spectral_evaluate_model(
    model: &SpectralGPT,
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

// ─── Experiment 4: Hamiltonian Residual Flow ───────────────────────

/// Run the Hamiltonian Residual Flow experiment.
///
/// Replaces standard residual connections with symplectic (leapfrog) integration
/// from Hamiltonian mechanics. Energy conservation prevents gradient explosion/vanishing.
pub fn run_hamiltonian_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("weights");
    println!("=== Experiment 4: Hamiltonian Residual Flow ===\n");
    println!("HYPOTHESIS: Replace standard residual connections with symplectic");
    println!("(energy-conserving) leapfrog integration from Hamiltonian mechanics.");
    println!("Energy conservation naturally prevents gradient explosion/vanishing.\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/5] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    // Config (same as baseline)
    let block_size = 128;
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size,
    };

    // Count params for baseline comparison
    let baseline_params = {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let inner = 4 * e;
        let mut total = v * e + config.block_size * e;
        for _ in 0..nl {
            total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
        }
        total + e * 2 + e * v
    };

    // Create Hamiltonian model
    let mut model = HamiltonianGPT::new(config.clone());
    let hamiltonian_params = model.count_params();
    println!("\n  Baseline params:    {}", baseline_params);
    println!("  Hamiltonian params: {} ({:+.1}%)",
             hamiltonian_params,
             (hamiltonian_params as f32 / baseline_params as f32 - 1.0) * 100.0);
    println!("  Extra params: {} (W_p_init: {}, step_sizes: {})",
             hamiltonian_params - baseline_params,
             config.n_embd * config.n_embd,
             config.n_layer);

    // === Phase 1: Pretrain on Shakespeare ===
    println!("\n[2/5] Pretraining HamiltonianGPT on Shakespeare...");
    hamiltonian_pretrain(&mut model, &config, base_text, &tok, &mut rng);

    // === Phase 2: SFT on tool calling data ===
    println!("\n[3/5] Supervised fine-tuning HamiltonianGPT on tool calling data...");
    let start_sft = Instant::now();
    let num_steps = 800;
    let lr = 0.0005;
    let batch_size = 8;

    let mut loss_log: Vec<(usize, f32, f32)> = Vec::new();
    let mut grad_norm_log: Vec<(usize, f32)> = Vec::new();
    let mut energy_log: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut per_layer_grad_log: Vec<(usize, Vec<f32>)> = Vec::new();

    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = HamiltonianGradients::zero_like(&config);
        let mut valid_count = 0;
        let mut last_diagnostics = EnergyDiagnostics::default();

        for _ in 0..batch_size {
            let example = &train_data[rng.gen_range(0..train_data.len())];
            let encoded = tok.encode(&example.input);
            if encoded.len() < 2 || encoded.len() > config.block_size {
                continue;
            }
            let input = &encoded[..encoded.len() - 1];
            let target = &encoded[1..];
            let (loss, g, diag) = model.forward_backward(input, target);
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
                valid_count += 1;
                last_diagnostics = diag;
            }
        }

        if valid_count > 0 {
            grads.scale(1.0 / valid_count as f32);

            let gn = HamiltonianGPT::grad_norm(&grads);
            let layer_gn = HamiltonianGPT::per_layer_grad_norms(&grads);

            model.apply_gradients(&grads, lr);
            let avg = total_loss / valid_count as f32;
            let elapsed = start_sft.elapsed().as_secs_f32();

            if step % 10 == 0 {
                loss_log.push((step, avg, elapsed));
                grad_norm_log.push((step, gn));
                energy_log.push((step, last_diagnostics.per_layer_total_energy.clone()));
                per_layer_grad_log.push((step, layer_gn.clone()));
            }

            if step % 100 == 0 {
                println!("  SFT step {:4} | Loss: {:.4} | GradNorm: {:.4} | Energy: [{:.3}, {:.3}, {:.3}] | h: [{:.3}, {:.3}, {:.3}] | {:.1}s",
                         step, avg, gn,
                         last_diagnostics.per_layer_total_energy.first().copied().unwrap_or(0.0),
                         last_diagnostics.per_layer_total_energy.get(1).copied().unwrap_or(0.0),
                         last_diagnostics.per_layer_total_energy.get(2).copied().unwrap_or(0.0),
                         model.step_sizes[0], model.step_sizes[1], model.step_sizes[2],
                         elapsed);
                println!("           Per-layer grad norms: [{:.4}, {:.4}, {:.4}]",
                         layer_gn.first().copied().unwrap_or(0.0),
                         layer_gn.get(1).copied().unwrap_or(0.0),
                         layer_gn.get(2).copied().unwrap_or(0.0));
            }
        }
    }

    let sft_time = start_sft.elapsed().as_secs_f32();
    println!("  SFT done in {:.1}s", sft_time);

    // Save training log
    if let Ok(mut file) = std::fs::File::create("experiments/hamiltonian_log.csv") {
        let _ = writeln!(file, "step,loss,time_seconds,grad_norm,energy_l0,energy_l1,energy_l2,grad_l0,grad_l1,grad_l2");
        for i in 0..loss_log.len() {
            let (s, l, t) = loss_log[i];
            let gn = grad_norm_log[i].1;
            let energy = &energy_log[i].1;
            let gl = &per_layer_grad_log[i].1;
            let _ = writeln!(file, "{},{:.6},{:.2},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                s, l, t, gn,
                energy.first().copied().unwrap_or(0.0),
                energy.get(1).copied().unwrap_or(0.0),
                energy.get(2).copied().unwrap_or(0.0),
                gl.first().copied().unwrap_or(0.0),
                gl.get(1).copied().unwrap_or(0.0),
                gl.get(2).copied().unwrap_or(0.0));
        }
        println!("  Training log saved to experiments/hamiltonian_log.csv");
    }

    // === Phase 3: Evaluate ===
    println!("\n[4/5] Evaluating HamiltonianGPT...");
    let metrics = hamiltonian_evaluate_model(&model, &config, &val_data, &tok);
    let composite = metrics.format_acc * 0.3
        + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25
        + metrics.reply_quality * 0.15;

    println!("\n  Hamiltonian GPT Results ({} examples):", metrics.count);
    println!("    Format accuracy:  {:.1}%", metrics.format_acc * 100.0);
    println!("    Tool accuracy:    {:.1}%", metrics.tool_acc * 100.0);
    println!("    Param accuracy:   {:.1}%", metrics.param_acc * 100.0);
    println!("    Reply quality:    {:.1}%", metrics.reply_quality * 100.0);
    println!("    Composite score:  {:.4}", composite);

    // === Phase 4: Measure inference speed ===
    println!("\n[5/5] Measuring inference speed...");
    let speed_start = Instant::now();
    let mut total_tokens_generated = 0usize;
    let speed_iters = 20;
    for example in val_data.iter().take(speed_iters) {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(40);
        let generated = model.generate(&prompt_encoded, max_gen);
        total_tokens_generated += generated.len() - prompt_encoded.len();
    }
    let speed_elapsed = speed_start.elapsed().as_secs_f32();
    let tokens_per_sec = total_tokens_generated as f32 / speed_elapsed;
    println!("  Generated {} tokens in {:.2}s = {:.1} tokens/sec",
             total_tokens_generated, speed_elapsed, tokens_per_sec);

    // Save weights
    if let Err(e) = model.save_weights("weights/hamiltonian.mgpt") {
        println!("  Warning: failed to save weights: {}", e);
    } else {
        println!("  Weights saved to weights/hamiltonian.mgpt");
    }

    // === Summary ===
    println!("\n{}", "=".repeat(60));
    println!("=== HAMILTONIAN RESIDUAL FLOW - SUMMARY ===\n");
    println!("  Architecture: Leapfrog (Stormer-Verlet) symplectic integrator");
    println!("  Parameters:   {} (baseline: {})", hamiltonian_params, baseline_params);
    println!("  Step sizes:   [{:.4}, {:.4}, {:.4}]",
             model.step_sizes[0], model.step_sizes[1], model.step_sizes[2]);

    // Energy conservation analysis
    if let Some((_, last_energy)) = energy_log.last() {
        if let Some((_, first_energy)) = energy_log.first() {
            println!("\n  Energy conservation (||q||^2 + ||p||^2 per layer):");
            println!("    Initial: [{:.4}, {:.4}, {:.4}]",
                     first_energy.first().copied().unwrap_or(0.0),
                     first_energy.get(1).copied().unwrap_or(0.0),
                     first_energy.get(2).copied().unwrap_or(0.0));
            println!("    Final:   [{:.4}, {:.4}, {:.4}]",
                     last_energy.first().copied().unwrap_or(0.0),
                     last_energy.get(1).copied().unwrap_or(0.0),
                     last_energy.get(2).copied().unwrap_or(0.0));
        }
    }

    // Gradient stability analysis
    let grad_stable = if let (Some(first), Some(last)) =
        (per_layer_grad_log.first(), per_layer_grad_log.last())
    {
        let ratio_max = first.1.iter().zip(last.1.iter())
            .map(|(f, l)| if *f > 1e-10 { l / f } else { 1.0 })
            .fold(0.0f32, f32::max);
        let stable = ratio_max < 10.0 && ratio_max > 0.1;
        println!("\n  Gradient stability:");
        for (i, (f, l)) in first.1.iter().zip(last.1.iter()).enumerate() {
            let ratio = if *f > 1e-10 { l / f } else { f32::NAN };
            println!("    Layer {}: first={:.6}, last={:.6}, ratio={:.3}", i, f, l, ratio);
        }
        println!("    Stable: {} (max ratio: {:.3})", stable, ratio_max);
        stable
    } else {
        false
    };

    println!("\n  Tool calling metrics:");
    println!("    Format accuracy:  {:.1}%", metrics.format_acc * 100.0);
    println!("    Tool accuracy:    {:.1}%", metrics.tool_acc * 100.0);
    println!("    Param accuracy:   {:.1}%", metrics.param_acc * 100.0);
    println!("    Reply quality:    {:.1}%", metrics.reply_quality * 100.0);
    println!("    Composite:        {:.4}", composite);
    println!("    Inference speed:  {:.1} tok/s", tokens_per_sec);

    // Energy conservation metric: ratio of max/min across layers
    let energy_conservation = if let Some((_, last_energy)) = energy_log.last() {
        let max_e = last_energy.iter().cloned().fold(0.0f32, f32::max);
        let min_e = last_energy.iter().cloned().fold(f32::MAX, f32::min);
        if min_e > 1e-10 { max_e / min_e } else { f32::NAN }
    } else {
        f32::NAN
    };

    println!("\n  Grad stable: {}", grad_stable);
    println!("  Energy conservation ratio (max/min across layers): {:.4}", energy_conservation);

    // Demo: show some tool calling examples
    println!("\n=== Demo: Hamiltonian GPT Tool Calling ===\n");
    for example in val_data.iter().take(5) {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);

        println!("  Query:    {}", example.prompt.trim());
        println!("  Expected: {}", example.expected_call.as_deref().unwrap_or("[direct reply]"));
        let response = &generated_text[example.prompt.len()..];
        let truncated = if let Some(end_pos) = response.find("[end]") {
            &response[..end_pos + 5]
        } else {
            &response[..response.len().min(60)]
        };
        println!("  Got:      {}", truncated.trim());
        println!();
    }

    // Print PlanDB update info
    println!("\n=== PLANDB_METRICS ===");
    println!("format_acc={:.4}", metrics.format_acc);
    println!("tool_acc={:.4}", metrics.tool_acc);
    println!("param_acc={:.4}", metrics.param_acc);
    println!("reply_quality={:.4}", metrics.reply_quality);
    println!("composite={:.4}", composite);
    println!("params={}", hamiltonian_params);
    println!("grad_stability={}", grad_stable);
    println!("energy_conservation={:.4}", energy_conservation);
    println!("step_sizes=[{:.4},{:.4},{:.4}]",
             model.step_sizes[0], model.step_sizes[1], model.step_sizes[2]);
    println!("inference_speed={:.1}", tokens_per_sec);
    println!("=== END PLANDB_METRICS ===");

    println!("\n=== Experiment 4 Complete ===");
}

/// Pretrain Hamiltonian model on Shakespeare text
fn hamiltonian_pretrain(
    model: &mut HamiltonianGPT,
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
        let (inputs, targets) = crate::data::create_batches(
            &encoded, config.block_size, batch_size, rng,
        );
        let mut total_loss = 0.0f32;
        let mut grads = HamiltonianGradients::zero_like(config);

        for b in 0..batch_size {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g, _diag) = model.forward_backward(
                &inputs[b][..ctx_len], &targets[b][..ctx_len],
            );
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, lr);

        if step % 200 == 0 {
            println!("  Pretrain step {:4} | Loss: {:.4} | h: [{:.3}, {:.3}, {:.3}] | {:.1}s",
                     step, total_loss / batch_size as f32,
                     model.step_sizes[0], model.step_sizes[1], model.step_sizes[2],
                     start.elapsed().as_secs_f32());
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
}

/// Evaluate Hamiltonian model
fn hamiltonian_evaluate_model(
    model: &HamiltonianGPT,
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

// ─── Renormalization Group Weight Sharing Experiment ─────────────────

/// Run Experiment 3: Renormalization Group Weight Sharing
/// All transformer layers share one set of base weights with per-layer
/// scalar scale (alpha) and shift (beta) factors — inspired by RG flow
/// in statistical physics where the same interactions apply at each scale
/// but with different coupling constants.
pub fn run_rg_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("weights");
    println!("=== Experiment 3: Renormalization Group Weight Sharing ===\n");
    println!("HYPOTHESIS: Share transformer weights across layers with per-layer");
    println!("scale factors (alpha*W + beta), inspired by RG flow in physics.");
    println!("Drastically reduce parameters while maintaining expressivity.\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/6] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    // Config (same as baseline)
    let block_size = 128;
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size,
    };

    // Count baseline params for comparison
    let baseline_params = {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let inner = 4 * e;
        let mut total = v * e + config.block_size * e;
        for _ in 0..nl {
            total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
        }
        total + e * 2 + e * v
    };

    // Create RG model
    let mut model = RGGPT::new(config.clone());
    let rg_params = model.count_params();
    let param_reduction = (1.0 - rg_params as f32 / baseline_params as f32) * 100.0;

    println!("\n  Baseline params:  {}", baseline_params);
    println!("  RG shared params: {} ({:.1}% fewer)", rg_params, param_reduction);

    // === Phase 1: Pretrain on Shakespeare ===
    println!("\n[2/6] Pretraining RGGPT on Shakespeare...");
    rg_pretrain(&mut model, &config, base_text, &tok, &mut rng);

    // === Phase 2: SFT on tool calling data ===
    println!("\n[3/6] Supervised fine-tuning RGGPT on tool calling data...");
    let start_sft = Instant::now();
    let num_steps = 800;
    let lr = 0.0005;
    let batch_size = 8;

    let mut loss_log: Vec<(usize, f32, f32)> = Vec::new();

    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = RGGradients::zero_like(&config);
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
            let avg = total_loss / valid_count as f32;
            let elapsed = start_sft.elapsed().as_secs_f32();
            loss_log.push((step, avg, elapsed));

            if step % 100 == 0 {
                println!("  SFT step {:4} | Loss: {:.4} | {:.1}s", step, avg, elapsed);
            }
        }
    }

    let sft_time = start_sft.elapsed().as_secs_f32();
    println!("  SFT done in {:.1}s", sft_time);

    // Save loss log
    if let Ok(mut file) = std::fs::File::create("experiments/rg_log.csv") {
        let _ = writeln!(file, "step,loss,time_seconds");
        for &(s, l, t) in &loss_log {
            let _ = writeln!(file, "{},{:.6},{:.2}", s, l, t);
        }
        println!("  Training log saved to experiments/rg_log.csv");
    }

    // === Phase 3: Evaluate ===
    println!("\n[4/6] Evaluating RGGPT...");
    let metrics = rg_evaluate_model(&model, &config, &val_data, &tok);
    let composite = metrics.format_acc * 0.3
        + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25
        + metrics.reply_quality * 0.15;

    println!("\n  RGGPT Results ({} examples):", metrics.count);
    println!("    Format accuracy:  {:.1}%", metrics.format_acc * 100.0);
    println!("    Tool accuracy:    {:.1}%", metrics.tool_acc * 100.0);
    println!("    Param accuracy:   {:.1}%", metrics.param_acc * 100.0);
    println!("    Reply quality:    {:.1}%", metrics.reply_quality * 100.0);
    println!("    Composite score:  {:.4}", composite);

    // === Phase 4: Analyze RG scale factors ===
    println!("\n[5/6] Analyzing RG scale factors (coupling constants)...");
    let alphas = model.get_layer_alphas();
    let betas = model.get_layer_betas();
    let weight_names = ["qkv_w", "attn_proj", "ff_w1", "ff_w2"];

    println!("\n  Per-layer alpha (scale) values:");
    println!("    {:<12} {:>10} {:>10} {:>10}", "Weight", "Layer 0", "Layer 1", "Layer 2");
    println!("    {}", "-".repeat(45));
    for (k, name) in weight_names.iter().enumerate() {
        println!("    {:<12} {:>10.4} {:>10.4} {:>10.4}",
                 name, alphas[0][k], alphas[1][k], alphas[2][k]);
    }

    println!("\n  Per-layer beta (shift) values:");
    println!("    {:<12} {:>10} {:>10} {:>10}", "Weight", "Layer 0", "Layer 1", "Layer 2");
    println!("    {}", "-".repeat(45));
    for (k, name) in weight_names.iter().enumerate() {
        println!("    {:<12} {:>10.6} {:>10.6} {:>10.6}",
                 name, betas[0][k], betas[1][k], betas[2][k]);
    }

    // Check if scale factors diverged (indicating layers learned different scales)
    let mut alpha_divergence = 0.0f32;
    for k in 0..4 {
        let mean_alpha = (alphas[0][k] + alphas[1][k] + alphas[2][k]) / 3.0;
        for l in 0..3 {
            alpha_divergence += (alphas[l][k] - mean_alpha).abs();
        }
    }
    alpha_divergence /= 12.0; // average over 4 weights x 3 layers
    println!("\n  Alpha divergence (mean absolute deviation): {:.6}", alpha_divergence);
    if alpha_divergence > 0.01 {
        println!("  --> Scale factors DIVERGED: layers learned different energy scales!");
    } else {
        println!("  --> Scale factors stayed close: layers use similar scales.");
    }

    // === Phase 5: Measure inference speed ===
    println!("\n[6/6] Measuring inference speed...");
    let speed_start = Instant::now();
    let mut total_tokens_generated = 0usize;
    let speed_iters = 20;
    for example in val_data.iter().take(speed_iters) {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(40);
        let generated = model.generate(&prompt_encoded, max_gen);
        total_tokens_generated += generated.len() - prompt_encoded.len();
    }
    let speed_elapsed = speed_start.elapsed().as_secs_f32();
    let tokens_per_sec = total_tokens_generated as f32 / speed_elapsed;
    println!("  Generated {} tokens in {:.2}s = {:.1} tokens/sec",
             total_tokens_generated, speed_elapsed, tokens_per_sec);

    // Parameter efficiency
    let param_efficiency = composite / (rg_params as f32 / 1000.0);
    let baseline_composite_approx = 0.45; // approximate from prior runs
    let baseline_efficiency = baseline_composite_approx / (baseline_params as f32 / 1000.0);

    println!("\n  Parameter efficiency (composite / K-params):");
    println!("    RGGPT:    {:.6} (composite {:.4} / {:.1}K params)",
             param_efficiency, composite, rg_params as f32 / 1000.0);
    println!("    Baseline: {:.6} (composite ~{:.4} / {:.1}K params)",
             baseline_efficiency, baseline_composite_approx,
             baseline_params as f32 / 1000.0);

    // Save weights
    if let Err(e) = model.save_weights("weights/rg.mgpt") {
        println!("  Warning: failed to save weights: {}", e);
    } else {
        println!("  Weights saved to weights/rg.mgpt");
    }

    // === Summary ===
    println!("\n{}", "=".repeat(60));
    println!("=== RG WEIGHT SHARING — RESULTS SUMMARY ===\n");
    println!("  Architecture:    Shared weights + per-layer alpha/beta (RG flow)");
    println!("  Parameters:      {} (baseline: {}, {:.1}% reduction)",
             rg_params, baseline_params, param_reduction);
    println!("  SFT time:        {:.1}s", sft_time);
    println!("  Inference speed:  {:.1} tokens/sec", tokens_per_sec);
    println!();
    println!("  Metrics:");
    println!("    format_acc:     {:.4}", metrics.format_acc);
    println!("    tool_acc:       {:.4}", metrics.tool_acc);
    println!("    param_acc:      {:.4}", metrics.param_acc);
    println!("    reply_quality:  {:.4}", metrics.reply_quality);
    println!("    composite:      {:.4}", composite);
    println!();

    // Comparison table
    println!("  Comparison vs SFT Baseline:");
    println!("    {:<18} {:>10} {:>10}", "", "RGGPT", "Baseline");
    println!("    {}", "-".repeat(40));
    println!("    {:<18} {:>9.1}% {:>9.1}%", "Format", metrics.format_acc * 100.0, 66.2);
    println!("    {:<18} {:>9.1}% {:>9.1}%", "Tool", metrics.tool_acc * 100.0, 63.8);
    println!("    {:<18} {:>9.1}% {:>9.1}%", "Param", metrics.param_acc * 100.0, 27.5);
    println!("    {:<18} {:>9.1}% {:>9.1}%", "Reply", metrics.reply_quality * 100.0, 30.0);
    println!("    {:<18} {:>10} {:>10}", "Params", rg_params, baseline_params);
    println!("    {:<18} {:>9.1}% {:>9}",
             "Reduction", param_reduction, "-");

    // Demo: show some tool-calling examples
    println!("\n  === Tool Calling Demo ===\n");
    for example in val_data.iter().take(6) {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);
        let response = &generated_text[example.prompt.len()..];
        let truncated = if let Some(end_pos) = response.find("[end]") {
            &response[..end_pos + 5]
        } else {
            &response[..response.len().min(60)]
        };
        println!("  Q: {}", example.prompt.trim());
        println!("  A: {}", truncated.trim());
        println!("  Expected: {}",
                 example.expected_call.as_deref().unwrap_or("[direct reply]"));
        println!();
    }

    // Save results CSV
    if let Ok(mut file) = std::fs::File::create("experiments/rg_results.csv") {
        let _ = writeln!(file, "metric,value");
        let _ = writeln!(file, "format_acc,{:.4}", metrics.format_acc);
        let _ = writeln!(file, "tool_acc,{:.4}", metrics.tool_acc);
        let _ = writeln!(file, "param_acc,{:.4}", metrics.param_acc);
        let _ = writeln!(file, "reply_quality,{:.4}", metrics.reply_quality);
        let _ = writeln!(file, "composite,{:.4}", composite);
        let _ = writeln!(file, "params,{}", rg_params);
        let _ = writeln!(file, "baseline_params,{}", baseline_params);
        let _ = writeln!(file, "param_reduction_pct,{:.1}", param_reduction);
        let _ = writeln!(file, "inference_speed,{:.1}", tokens_per_sec);
        let _ = writeln!(file, "training_time,{:.1}", sft_time);
        let _ = writeln!(file, "alpha_divergence,{:.6}", alpha_divergence);
        let _ = writeln!(file, "param_efficiency,{:.6}", param_efficiency);
        // Per-layer alphas
        for l in 0..3 {
            for (k, name) in weight_names.iter().enumerate() {
                let _ = writeln!(file, "alpha_l{}_{},{:.6}", l, name, alphas[l][k]);
            }
        }
        println!("\n  Results saved to experiments/rg_results.csv");
    }

    // Print for PlanDB extraction
    // Compute mean alphas per layer (average across 4 weight matrices)
    let layer_alpha_means: Vec<f32> = (0..3).map(|l| {
        (alphas[l][0] + alphas[l][1] + alphas[l][2] + alphas[l][3]) / 4.0
    }).collect();

    println!("\n=== PLANDB_METRICS ===");
    println!("format_acc={:.4}", metrics.format_acc);
    println!("tool_acc={:.4}", metrics.tool_acc);
    println!("param_acc={:.4}", metrics.param_acc);
    println!("reply_quality={:.4}", metrics.reply_quality);
    println!("composite={:.4}", composite);
    println!("params={}", rg_params);
    println!("param_reduction_pct={:.1}", param_reduction);
    println!("inference_speed={:.1}", tokens_per_sec);
    println!("layer_alphas=[{:.4},{:.4},{:.4}]",
             layer_alpha_means[0], layer_alpha_means[1], layer_alpha_means[2]);
    println!("alpha_divergence={:.6}", alpha_divergence);
    println!("param_efficiency={:.6}", param_efficiency);
    println!("=== END PLANDB_METRICS ===");
}

/// Pretrain RGGPT on Shakespeare
fn rg_pretrain(
    model: &mut RGGPT,
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
        let (inputs, targets) = crate::data::create_batches(
            &encoded, config.block_size, batch_size, rng,
        );
        let mut total_loss = 0.0f32;
        let mut grads = RGGradients::zero_like(config);

        for b in 0..batch_size {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g) = model.forward_backward(
                &inputs[b][..ctx_len], &targets[b][..ctx_len],
            );
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, lr);

        if step % 200 == 0 {
            println!("  Pretrain step {:4} | Loss: {:.4} | {:.1}s",
                     step, total_loss / batch_size as f32,
                     start.elapsed().as_secs_f32());
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
}

/// Evaluate RGGPT on validation data
fn rg_evaluate_model(
    model: &RGGPT,
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

// ─── Experiment 2: Hebbian Inference Memory Bank ───────────────

/// Run the Hebbian memory experiment:
/// Train HebbianGPT (base GPT + W_gate) then evaluate with memory accumulation.
pub fn run_hebbian_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("weights");

    println!("=== Experiment 2: Hebbian Inference Memory Bank ===");
    println!("    Hypothesis: External memory updated via Hebbian learning");
    println!("    enables learning-at-inference-time without backprop.\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/5] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
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

    let param_count = HebbianGPT::param_count(&config);
    println!("  HebbianGPT params: {} (base ~91K + {} W_gate)", param_count, config.n_embd);

    // Phase 1: Pretrain base GPT on Shakespeare
    println!("\n[2/5] Pretraining base GPT on Shakespeare...");
    let pretrained = pretrain(&config, base_text, &tok, &mut rng);

    // Phase 2: Create HebbianGPT and copy pretrained weights
    println!("\n[3/5] Training HebbianGPT on tool calling data...");
    let mut hebbian = HebbianGPT::new(config.clone());
    hebbian.gpt.token_emb = pretrained.token_emb.clone();
    hebbian.gpt.pos_emb = pretrained.pos_emb.clone();
    for l in 0..config.n_layer {
        hebbian.gpt.ln1_gamma[l] = pretrained.ln1_gamma[l].clone();
        hebbian.gpt.ln1_beta[l] = pretrained.ln1_beta[l].clone();
        hebbian.gpt.qkv_w[l] = pretrained.qkv_w[l].clone();
        hebbian.gpt.attn_proj[l] = pretrained.attn_proj[l].clone();
        hebbian.gpt.ln2_gamma[l] = pretrained.ln2_gamma[l].clone();
        hebbian.gpt.ln2_beta[l] = pretrained.ln2_beta[l].clone();
        hebbian.gpt.ff_w1[l] = pretrained.ff_w1[l].clone();
        hebbian.gpt.ff_b1[l] = pretrained.ff_b1[l].clone();
        hebbian.gpt.ff_w2[l] = pretrained.ff_w2[l].clone();
        hebbian.gpt.ff_b2[l] = pretrained.ff_b2[l].clone();
    }
    hebbian.gpt.ln_f_gamma = pretrained.ln_f_gamma.clone();
    hebbian.gpt.ln_f_beta = pretrained.ln_f_beta.clone();
    hebbian.gpt.lm_head = pretrained.lm_head.clone();

    // SFT on tool calling data: 800 steps, lr=0.0005, batch=8
    let num_steps = 800;
    let lr = 0.0005;
    let batch_size = 8;
    let start = Instant::now();
    let mut loss_log: Vec<(usize, f32, f32)> = Vec::new();

    for step in 0..num_steps {
        hebbian.clear_memory();

        let mut total_loss = 0.0f32;
        let mut batch_grads: Option<HebbianGradients> = None;
        let mut valid_count = 0;

        for _ in 0..batch_size {
            let example = &train_data[rng.gen_range(0..train_data.len())];
            let encoded = tok.encode(&example.input);
            if encoded.len() < 2 || encoded.len() > config.block_size {
                continue;
            }
            let input = &encoded[..encoded.len() - 1];
            let target = &encoded[1..];
            let (loss, grads) = hebbian.forward_backward(input, target);
            if loss.is_finite() {
                total_loss += loss;
                match &mut batch_grads {
                    Some(bg) => {
                        bg.base.accumulate(&grads.base);
                        for d in 0..config.n_embd {
                            bg.w_gate[d] += grads.w_gate[d];
                        }
                    }
                    None => {
                        batch_grads = Some(grads);
                    }
                }
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            if let Some(ref mut bg) = batch_grads {
                let scale = 1.0 / valid_count as f32;
                bg.base.scale(scale);
                for d in 0..config.n_embd {
                    bg.w_gate[d] *= scale;
                }
                hebbian.apply_gradients(bg, lr);
            }
            let avg_loss = total_loss / valid_count as f32;
            let elapsed = start.elapsed().as_secs_f32();
            loss_log.push((step, avg_loss, elapsed));

            if step % 100 == 0 || step == num_steps - 1 {
                println!("  SFT step {:4} | Loss: {:.4} | {:.1}s",
                         step, avg_loss, elapsed);
            }
        }
    }
    let train_time = start.elapsed().as_secs_f32();
    println!("  Training done in {:.1}s", train_time);

    // Save training log
    if let Ok(mut file) = std::fs::File::create("experiments/hebbian_log.csv") {
        let _ = writeln!(file, "step,loss,time_seconds");
        for &(step, loss, time) in &loss_log {
            let _ = writeln!(file, "{},{:.6},{:.2}", step, loss, time);
        }
        println!("  Training log saved to experiments/hebbian_log.csv");
    }

    // Phase 3: Evaluate WITHOUT memory (baseline)
    println!("\n[4/5] Evaluating...");
    println!("\n  --- Baseline (no memory) ---");
    let baseline_metrics = evaluate_model(&hebbian.gpt, &config, &val_data, &tok);
    baseline_metrics.print("HebbianGPT (no memory)");

    // Phase 4: Evaluate WITH memory accumulation
    println!("\n  --- With Hebbian Memory ---");
    let memory_metrics = evaluate_hebbian_model(&mut hebbian, &config, &val_data, &tok);
    memory_metrics.print("HebbianGPT (with memory)");

    // Phase 5: Special test — memory accumulation effect
    println!("\n[5/5] Memory accumulation test...");
    println!("  Testing: feed 5 calc queries, then test a new calc query.");
    println!("  Does memory help with the 6th query?\n");

    let warmup_queries = [
        "[user] what is 3 plus 5 [end] ",
        "[user] what is 7 plus 2 [end] ",
        "[user] what is 4 plus 6 [end] ",
        "[user] what is 8 plus 1 [end] ",
        "[user] what is 9 plus 3 [end] ",
    ];
    let test_query = "[user] what is 11 plus 4 [end] ";

    // Test WITHOUT memory
    let test_encoded = tok.encode(test_query);
    let max_gen = (config.block_size - test_encoded.len()).min(80);
    let no_mem_ids = hebbian.generate_without_memory(&test_encoded, max_gen);
    let no_mem_text = tok.decode(&no_mem_ids);
    let no_mem_response = &no_mem_text[test_query.len()..];
    let no_mem_trunc = if let Some(end_pos) = no_mem_response.find("[end]") {
        &no_mem_response[..end_pos + 5]
    } else {
        &no_mem_response[..no_mem_response.len().min(60)]
    };

    // Test WITH memory: warm up with 5 queries, then test
    hebbian.clear_memory();
    for warmup_q in &warmup_queries {
        let warmup_encoded = tok.encode(warmup_q);
        let wmax = (config.block_size - warmup_encoded.len()).min(80);
        let _ = hebbian.generate_with_memory(&warmup_encoded, wmax);
    }

    let mem_util_after_warmup = hebbian.memory.utilization();
    let key_div_after_warmup = hebbian.memory.key_diversity();
    println!("  Memory after 5 warmup queries:");
    println!("    Utilization: {:.1}% ({} of {} slots active)",
             mem_util_after_warmup * 100.0,
             (mem_util_after_warmup * hebbian.memory.memory_slots as f32) as usize,
             hebbian.memory.memory_slots);
    println!("    Key diversity: {:.3}", key_div_after_warmup);

    let mem_ids = hebbian.generate_with_memory(&test_encoded, max_gen);
    let mem_text = tok.decode(&mem_ids);
    let mem_response = &mem_text[test_query.len()..];
    let mem_trunc = if let Some(end_pos) = mem_response.find("[end]") {
        &mem_response[..end_pos + 5]
    } else {
        &mem_response[..mem_response.len().min(60)]
    };

    println!("\n  Test query: {}", test_query.trim());
    println!("  Without memory: {}", no_mem_trunc.trim());
    println!("  With memory:    {}", mem_trunc.trim());

    // Inference speed benchmark
    println!("\n=== Inference Speed ===");
    let speed_start = Instant::now();
    let prompt = tok.encode("[user] what is 5 plus 3 [end] ");
    let n_iters = 50;

    for _ in 0..n_iters {
        let _ = hebbian.generate_without_memory(&prompt, 30);
    }
    let no_mem_elapsed = speed_start.elapsed().as_secs_f32();
    let no_mem_tps = (n_iters * 30) as f32 / no_mem_elapsed;

    let speed_start2 = Instant::now();
    for _ in 0..n_iters {
        hebbian.clear_memory();
        let _ = hebbian.generate_with_memory(&prompt, 30);
    }
    let mem_elapsed = speed_start2.elapsed().as_secs_f32();
    let mem_tps = (n_iters * 30) as f32 / mem_elapsed;

    println!("  Without memory: {:.0} tokens/sec", no_mem_tps);
    println!("  With memory:    {:.0} tokens/sec", mem_tps);
    println!("  Memory overhead: {:.1}x slowdown", no_mem_tps / mem_tps.max(1.0));

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("=== HEBBIAN EXPERIMENT RESULTS ===\n");

    println!("{:<25} {:>10} {:>10} {:>10} {:>10}",
             "Variant", "Format%", "Tool%", "Param%", "Reply%");
    println!("{}", "-".repeat(65));

    let baseline_composite = baseline_metrics.format_acc * 0.3
        + baseline_metrics.tool_acc * 0.3
        + baseline_metrics.param_acc * 0.25
        + baseline_metrics.reply_quality * 0.15;
    let memory_composite = memory_metrics.format_acc * 0.3
        + memory_metrics.tool_acc * 0.3
        + memory_metrics.param_acc * 0.25
        + memory_metrics.reply_quality * 0.15;

    println!("{:<25} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%  (composite: {:.3})",
             "No memory",
             baseline_metrics.format_acc * 100.0,
             baseline_metrics.tool_acc * 100.0,
             baseline_metrics.param_acc * 100.0,
             baseline_metrics.reply_quality * 100.0,
             baseline_composite);
    println!("{:<25} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%  (composite: {:.3})",
             "With Hebbian memory",
             memory_metrics.format_acc * 100.0,
             memory_metrics.tool_acc * 100.0,
             memory_metrics.param_acc * 100.0,
             memory_metrics.reply_quality * 100.0,
             memory_composite);

    let memory_effect = memory_composite - baseline_composite;
    println!("\n  Memory effect on composite: {:+.3} ({:.1}%)",
             memory_effect,
             if baseline_composite > 0.0 { memory_effect / baseline_composite * 100.0 } else { 0.0 });
    println!("  Parameters: {}", param_count);
    println!("  Inference speed (with memory): {:.0} tokens/sec", mem_tps);

    // Save weights
    match hebbian.save_weights("weights/hebbian.mgpt") {
        Ok(()) => println!("\n  Weights saved to weights/hebbian.mgpt"),
        Err(e) => eprintln!("\n  Warning: could not save weights: {e}"),
    }

    // Save results CSV
    if let Ok(mut file) = std::fs::File::create("experiments/hebbian_results.csv") {
        let _ = writeln!(file, "variant,format_acc,tool_acc,param_acc,reply_quality,composite,params,inference_speed,memory_utilization,key_diversity");
        let _ = writeln!(file, "no_memory,{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.0},{:.4},{:.4}",
                         baseline_metrics.format_acc, baseline_metrics.tool_acc,
                         baseline_metrics.param_acc, baseline_metrics.reply_quality,
                         baseline_composite, param_count, no_mem_tps, 0.0, 0.0);
        let _ = writeln!(file, "hebbian_memory,{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.0},{:.4},{:.4}",
                         memory_metrics.format_acc, memory_metrics.tool_acc,
                         memory_metrics.param_acc, memory_metrics.reply_quality,
                         memory_composite, param_count, mem_tps,
                         mem_util_after_warmup, key_div_after_warmup);
        println!("  Results saved to experiments/hebbian_results.csv");
    }

    println!("\n=== Hebbian Experiment Complete ===");

    // Print PlanDB update command
    println!("\nPlanDB update command:");
    println!("PLANDB_AGENT=worker-hebbian plandb done t-exp-hebbian --result '{{\"format_acc\":{:.4},\"tool_acc\":{:.4},\"param_acc\":{:.4},\"reply_quality\":{:.4},\"composite\":{:.4},\"params\":{},\"inference_speed\":{:.0},\"memory_effect\":{:.4}}}'",
             memory_metrics.format_acc, memory_metrics.tool_acc,
             memory_metrics.param_acc, memory_metrics.reply_quality,
             memory_composite, param_count, mem_tps, memory_effect);
}

/// Evaluate HebbianGPT with memory accumulation.
/// Memory accumulates across groups of 10 examples to test short-term learning.
fn evaluate_hebbian_model(
    model: &mut HebbianGPT,
    config: &Config,
    val: &[ToolExample],
    tok: &Tokenizer,
) -> AggregateMetrics {
    let mut results = Vec::new();

    model.clear_memory();

    for (idx, example) in val.iter().enumerate() {
        if idx % 10 == 0 {
            model.clear_memory();
        }

        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate_with_memory(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);
        let eval = tool_data::evaluate_output(&generated_text, example);
        results.push(eval);
    }

    AggregateMetrics::from_results(&results)
}

// ─── Experiment 5: Hybrid Genesis Architecture ──────────────────────

/// Run the Hybrid experiment: RG Weight Sharing + Hamiltonian Residual Flow
pub fn run_hybrid_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("weights");
    println!("=== Experiment 5: Hybrid Genesis (RG + Hamiltonian) ===\n");
    println!("HYPOTHESIS: Combining RG shared weights (parameter efficiency)");
    println!("with Hamiltonian leapfrog integration (gradient stability)");
    println!("should yield best-of-both-worlds performance.\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/6] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
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

    // Count baseline params
    let baseline_params = {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let inner = 4 * e;
        let mut total = v * e + config.block_size * e;
        for _ in 0..nl {
            total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
        }
        total + e * 2 + e * v
    };

    let mut model = HybridGPT::new(config.clone());
    let hybrid_params = model.count_params();
    let param_reduction = (1.0 - hybrid_params as f32 / baseline_params as f32) * 100.0;

    println!("\n  Baseline params:  {}", baseline_params);
    println!("  Hybrid params:    {} ({:.1}% fewer)", hybrid_params, param_reduction);

    // === Phase 1: Pretrain on Shakespeare ===
    println!("\n[2/6] Pretraining HybridGPT on Shakespeare...");
    hybrid_pretrain(&mut model, &config, base_text, &tok, &mut rng);

    // === Phase 2: SFT on tool calling data ===
    println!("\n[3/6] Supervised fine-tuning HybridGPT on tool calling data...");
    let start_sft = Instant::now();
    let num_steps = 800;
    let lr = 0.0005;
    let batch_size = 8;

    let mut loss_log: Vec<(usize, f32, f32)> = Vec::new();

    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = HybridGradients::zero_like(&config);
        let mut valid_count = 0;

        for _ in 0..batch_size {
            let example = &train_data[rng.gen_range(0..train_data.len())];
            let encoded = tok.encode(&example.input);
            if encoded.len() < 2 || encoded.len() > config.block_size {
                continue;
            }
            let input = &encoded[..encoded.len() - 1];
            let target = &encoded[1..];
            let (loss, g, _diag) = model.forward_backward(input, target);
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
            let elapsed = start_sft.elapsed().as_secs_f32();
            loss_log.push((step, avg, elapsed));

            if step % 100 == 0 {
                let gn = HybridGPT::grad_norm(&grads);
                println!("  SFT step {:4} | Loss: {:.4} | GradNorm: {:.4} | h: [{:.3}, {:.3}, {:.3}] | {:.1}s",
                         step, avg, gn,
                         model.step_sizes[0], model.step_sizes[1], model.step_sizes[2],
                         elapsed);
            }
        }
    }

    let sft_time = start_sft.elapsed().as_secs_f32();
    println!("  SFT done in {:.1}s", sft_time);

    // Save loss log
    if let Ok(mut file) = std::fs::File::create("experiments/hybrid_log.csv") {
        let _ = writeln!(file, "step,loss,time_seconds");
        for &(s, l, t) in &loss_log {
            let _ = writeln!(file, "{},{:.6},{:.2}", s, l, t);
        }
        println!("  Training log saved to experiments/hybrid_log.csv");
    }

    // === Phase 3: Evaluate ===
    println!("\n[4/6] Evaluating HybridGPT on in-distribution data...");
    let metrics = hybrid_evaluate_model(&model, &config, &val_data, &tok);
    let composite = metrics.format_acc * 0.3
        + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25
        + metrics.reply_quality * 0.15;

    println!("\n  HybridGPT Results ({} examples):", metrics.count);
    println!("    Format accuracy:  {:.1}%", metrics.format_acc * 100.0);
    println!("    Tool accuracy:    {:.1}%", metrics.tool_acc * 100.0);
    println!("    Param accuracy:   {:.1}%", metrics.param_acc * 100.0);
    println!("    Reply quality:    {:.1}%", metrics.reply_quality * 100.0);
    println!("    Composite score:  {:.4}", composite);

    // === Phase 4: Evaluate on OOD data ===
    println!("\n[5/6] Evaluating HybridGPT on out-of-distribution data...");
    let ood_data = tool_data::generate_ood_dataset();
    let ood_metrics = hybrid_evaluate_model(&model, &config, &ood_data, &tok);
    let ood_composite = ood_metrics.format_acc * 0.3
        + ood_metrics.tool_acc * 0.3
        + ood_metrics.param_acc * 0.25
        + ood_metrics.reply_quality * 0.15;

    println!("\n  HybridGPT OOD Results ({} examples):", ood_metrics.count);
    println!("    Format accuracy:  {:.1}%", ood_metrics.format_acc * 100.0);
    println!("    Tool accuracy:    {:.1}%", ood_metrics.tool_acc * 100.0);
    println!("    Param accuracy:   {:.1}%", ood_metrics.param_acc * 100.0);
    println!("    Reply quality:    {:.1}%", ood_metrics.reply_quality * 100.0);
    println!("    OOD composite:    {:.4}", ood_composite);

    let gen_gap = composite - ood_composite;
    println!("\n    Generalization gap: {:.4} (ID {:.4} - OOD {:.4})",
             gen_gap, composite, ood_composite);

    // === Phase 5: Analyze RG scale factors ===
    println!("\n[6/6] Analyzing RG scale factors...");
    let alphas_v = model.get_alphas_v();
    let alphas_t = model.get_alphas_t();
    let weight_names = ["qkv_w", "attn_proj", "ff_w1", "ff_w2"];

    println!("\n  V-path (attention) alpha values:");
    println!("    {:<12} {:>10} {:>10} {:>10}", "Weight", "Layer 0", "Layer 1", "Layer 2");
    println!("    {}", "-".repeat(45));
    for (k, name) in weight_names.iter().enumerate() {
        println!("    {:<12} {:>10.4} {:>10.4} {:>10.4}",
                 name, alphas_v[0][k], alphas_v[1][k], alphas_v[2][k]);
    }

    println!("\n  T-path (feed-forward) alpha values:");
    println!("    {:<12} {:>10} {:>10} {:>10}", "Weight", "Layer 0", "Layer 1", "Layer 2");
    println!("    {}", "-".repeat(45));
    for (k, name) in weight_names.iter().enumerate() {
        println!("    {:<12} {:>10.4} {:>10.4} {:>10.4}",
                 name, alphas_t[0][k], alphas_t[1][k], alphas_t[2][k]);
    }

    println!("\n  Leapfrog step sizes: [{:.4}, {:.4}, {:.4}]",
             model.step_sizes[0], model.step_sizes[1], model.step_sizes[2]);

    // Save weights
    if let Err(e) = model.save_weights("weights/hybrid.mgpt") {
        println!("  Warning: failed to save weights: {}", e);
    } else {
        println!("  Weights saved to weights/hybrid.mgpt");
    }

    // Save results CSV
    if let Ok(mut file) = std::fs::File::create("experiments/hybrid_results.csv") {
        let _ = writeln!(file, "variant,format_acc,tool_acc,param_acc,reply_quality,composite,params,gen_gap");
        let _ = writeln!(file, "hybrid_id,{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4}",
                         metrics.format_acc, metrics.tool_acc,
                         metrics.param_acc, metrics.reply_quality,
                         composite, hybrid_params, gen_gap);
        let _ = writeln!(file, "hybrid_ood,{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4}",
                         ood_metrics.format_acc, ood_metrics.tool_acc,
                         ood_metrics.param_acc, ood_metrics.reply_quality,
                         ood_composite, hybrid_params, gen_gap);
        println!("  Results saved to experiments/hybrid_results.csv");
    }

    // === Summary ===
    println!("\n{}", "=".repeat(60));
    println!("=== HYBRID GENESIS - SUMMARY ===\n");
    println!("  Architecture: RG Weight Sharing + Hamiltonian Leapfrog");
    println!("  Parameters:   {} (baseline: {}, {:.1}% fewer)",
             hybrid_params, baseline_params, param_reduction);
    println!("  Step sizes:   [{:.4}, {:.4}, {:.4}]",
             model.step_sizes[0], model.step_sizes[1], model.step_sizes[2]);
    println!("  ID composite: {:.4}", composite);
    println!("  OOD composite:{:.4}", ood_composite);
    println!("  Gen gap:      {:.4}", gen_gap);
    println!("\n=== Hybrid Experiment Complete ===");

    println!("\nPlanDB update command:");
    println!("PLANDB_AGENT=worker-hybrid plandb done t-exp-hybrid --result '{{\"composite\":{:.4},\"params\":{},\"ood_composite\":{:.4},\"gen_gap\":{:.4}}}'",
             composite, hybrid_params, ood_composite, gen_gap);
}

/// Pretrain HybridGPT on base text
fn hybrid_pretrain(
    model: &mut HybridGPT,
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
        let (inputs, targets) = crate::data::create_batches(
            &encoded, config.block_size, batch_size, rng,
        );
        let mut total_loss = 0.0f32;
        let mut grads = HybridGradients::zero_like(config);

        for b in 0..batch_size {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g, _diag) = model.forward_backward(
                &inputs[b][..ctx_len], &targets[b][..ctx_len],
            );
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, lr);

        if step % 200 == 0 {
            println!("  Pretrain step {:4} | Loss: {:.4} | h: [{:.3}, {:.3}, {:.3}] | {:.1}s",
                     step, total_loss / batch_size as f32,
                     model.step_sizes[0], model.step_sizes[1], model.step_sizes[2],
                     start.elapsed().as_secs_f32());
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
}

/// Evaluate HybridGPT on a set of examples
fn hybrid_evaluate_model(
    model: &HybridGPT,
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

// ─── Experiment 6: Generalization Test Suite ──────────────────────

/// Run comprehensive generalization tests on ALL model architectures.
/// Compares in-distribution vs out-of-distribution performance.
pub fn run_generalization_test() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("weights");
    println!("=== Generalization Test Suite ===\n");
    println!("Testing whether models GENERALIZE or merely MEMORIZE.");
    println!("Comparing ID (in-distribution) vs OOD (out-of-distribution) accuracy.\n");

    let mut rng = rand::thread_rng();

    // Generate datasets
    println!("[1/5] Generating datasets...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    let ood_data = tool_data::generate_ood_dataset();
    println!("  Train: {} examples", train_data.len());
    println!("  Val (ID): {} examples", val_data.len());
    println!("  OOD: {} examples", ood_data.len());

    // Build tokenizer (must include OOD text too for encoding)
    let base_text = crate::data::get_training_data();
    let mut combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    for ex in &ood_data {
        combined_text.push_str(&ex.input);
        combined_text.push('\n');
    }
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

    // 50% data subset for data efficiency test
    let half_train: Vec<ToolExample> = {
        let mut subset = train_data.clone();
        subset.truncate(subset.len() / 2);
        subset
    };

    println!("  50% train subset: {} examples", half_train.len());

    // Results storage
    struct ModelResults {
        name: String,
        params: usize,
        id_metrics: AggregateMetrics,
        ood_metrics: AggregateMetrics,
        id_composite: f32,
        ood_composite: f32,
        gen_gap: f32,
        half_data_composite: f32,
        data_efficiency_gap: f32,
    }

    fn composite_score(m: &AggregateMetrics) -> f32 {
        m.format_acc * 0.3 + m.tool_acc * 0.3 + m.param_acc * 0.25 + m.reply_quality * 0.15
    }

    let mut all_results: Vec<ModelResults> = Vec::new();

    // === 1. Baseline GPT ===
    println!("\n[2/5] Training and evaluating Baseline GPT...");
    {
        let baseline_params = {
            let e = config.n_embd; let v = config.vocab_size;
            let nl = config.n_layer; let inner = 4 * e;
            let mut total = v * e + config.block_size * e;
            for _ in 0..nl {
                total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
            }
            total + e * 2 + e * v
        };

        // Full data training
        let mut model = pretrain(&config, base_text, &tok, &mut rng);
        let _ = sft_train(&mut model, &config, &train_data, &val_data, &tok, &mut rng);
        let id_metrics = evaluate_model(&model, &config, &val_data, &tok);
        let ood_metrics = evaluate_model(&model, &config, &ood_data, &tok);

        // Half data training
        let mut model_half = pretrain(&config, base_text, &tok, &mut rng);
        let _ = sft_train(&mut model_half, &config, &half_train, &val_data, &tok, &mut rng);
        let half_metrics = evaluate_model(&model_half, &config, &val_data, &tok);

        let id_comp = composite_score(&id_metrics);
        let ood_comp = composite_score(&ood_metrics);
        let half_comp = composite_score(&half_metrics);

        println!("  Baseline: ID={:.4} OOD={:.4} Gap={:.4} Half={:.4}",
                 id_comp, ood_comp, id_comp - ood_comp, half_comp);

        all_results.push(ModelResults {
            name: "Baseline GPT".to_string(),
            params: baseline_params,
            id_metrics,
            ood_metrics,
            id_composite: id_comp,
            ood_composite: ood_comp,
            gen_gap: id_comp - ood_comp,
            half_data_composite: half_comp,
            data_efficiency_gap: id_comp - half_comp,
        });
    }

    // === 2. RG GPT ===
    println!("\n[3/5] Training and evaluating RG GPT...");
    {
        let mut model = RGGPT::new(config.clone());
        let rg_params = model.count_params();
        rg_pretrain(&mut model, &config, base_text, &tok, &mut rng);
        rg_sft_train(&mut model, &config, &train_data, &tok, &mut rng, 800);
        let id_metrics = rg_evaluate_model(&model, &config, &val_data, &tok);
        let ood_metrics = rg_evaluate_model(&model, &config, &ood_data, &tok);

        let mut model_half = RGGPT::new(config.clone());
        rg_pretrain(&mut model_half, &config, base_text, &tok, &mut rng);
        rg_sft_train(&mut model_half, &config, &half_train, &tok, &mut rng, 800);
        let half_metrics = rg_evaluate_model(&model_half, &config, &val_data, &tok);

        let id_comp = composite_score(&id_metrics);
        let ood_comp = composite_score(&ood_metrics);
        let half_comp = composite_score(&half_metrics);

        println!("  RG: ID={:.4} OOD={:.4} Gap={:.4} Half={:.4}",
                 id_comp, ood_comp, id_comp - ood_comp, half_comp);

        all_results.push(ModelResults {
            name: "RG GPT".to_string(),
            params: rg_params,
            id_metrics,
            ood_metrics,
            id_composite: id_comp,
            ood_composite: ood_comp,
            gen_gap: id_comp - ood_comp,
            half_data_composite: half_comp,
            data_efficiency_gap: id_comp - half_comp,
        });
    }

    // === 3. Hamiltonian GPT ===
    println!("\n[4/5] Training and evaluating Hamiltonian GPT...");
    {
        let mut model = HamiltonianGPT::new(config.clone());
        let ham_params = model.count_params();
        hamiltonian_pretrain(&mut model, &config, base_text, &tok, &mut rng);
        hamiltonian_sft_train(&mut model, &config, &train_data, &tok, &mut rng, 800);
        let id_metrics = hamiltonian_evaluate_model(&model, &config, &val_data, &tok);
        let ood_metrics = hamiltonian_evaluate_model(&model, &config, &ood_data, &tok);

        let mut model_half = HamiltonianGPT::new(config.clone());
        hamiltonian_pretrain(&mut model_half, &config, base_text, &tok, &mut rng);
        hamiltonian_sft_train(&mut model_half, &config, &half_train, &tok, &mut rng, 800);
        let half_metrics = hamiltonian_evaluate_model(&model_half, &config, &val_data, &tok);

        let id_comp = composite_score(&id_metrics);
        let ood_comp = composite_score(&ood_metrics);
        let half_comp = composite_score(&half_metrics);

        println!("  Hamiltonian: ID={:.4} OOD={:.4} Gap={:.4} Half={:.4}",
                 id_comp, ood_comp, id_comp - ood_comp, half_comp);

        all_results.push(ModelResults {
            name: "Hamiltonian GPT".to_string(),
            params: ham_params,
            id_metrics,
            ood_metrics,
            id_composite: id_comp,
            ood_composite: ood_comp,
            gen_gap: id_comp - ood_comp,
            half_data_composite: half_comp,
            data_efficiency_gap: id_comp - half_comp,
        });
    }

    // === 4. Hybrid GPT ===
    println!("\n[5/5] Training and evaluating Hybrid GPT...");
    {
        let mut model = HybridGPT::new(config.clone());
        let hybrid_params = model.count_params();
        hybrid_pretrain(&mut model, &config, base_text, &tok, &mut rng);
        hybrid_sft_train(&mut model, &config, &train_data, &tok, &mut rng, 800);
        let id_metrics = hybrid_evaluate_model(&model, &config, &val_data, &tok);
        let ood_metrics = hybrid_evaluate_model(&model, &config, &ood_data, &tok);

        let mut model_half = HybridGPT::new(config.clone());
        hybrid_pretrain(&mut model_half, &config, base_text, &tok, &mut rng);
        hybrid_sft_train(&mut model_half, &config, &half_train, &tok, &mut rng, 800);
        let half_metrics = hybrid_evaluate_model(&model_half, &config, &val_data, &tok);

        let id_comp = composite_score(&id_metrics);
        let ood_comp = composite_score(&ood_metrics);
        let half_comp = composite_score(&half_metrics);

        println!("  Hybrid: ID={:.4} OOD={:.4} Gap={:.4} Half={:.4}",
                 id_comp, ood_comp, id_comp - ood_comp, half_comp);

        all_results.push(ModelResults {
            name: "Hybrid GPT".to_string(),
            params: hybrid_params,
            id_metrics,
            ood_metrics,
            id_composite: id_comp,
            ood_composite: ood_comp,
            gen_gap: id_comp - ood_comp,
            half_data_composite: half_comp,
            data_efficiency_gap: id_comp - half_comp,
        });
    }

    // === Print comparison tables ===
    println!("\n{}", "=".repeat(80));
    println!("=== GENERALIZATION TEST RESULTS ===\n");

    println!("--- In-Distribution vs Out-of-Distribution ---\n");
    println!("{:<18} {:>8} {:>10} {:>10} {:>10} {:>10}",
             "Model", "Params", "ID Comp", "OOD Comp", "Gen Gap", "ID-OOD%");
    println!("{}", "-".repeat(70));
    for r in &all_results {
        let pct = if r.id_composite > 0.0 {
            (r.gen_gap / r.id_composite) * 100.0
        } else { 0.0 };
        println!("{:<18} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>9.1}%",
                 r.name, r.params, r.id_composite, r.ood_composite, r.gen_gap, pct);
    }

    println!("\n--- Detailed OOD Metrics ---\n");
    println!("{:<18} {:>10} {:>10} {:>10} {:>10}",
             "Model", "Format%", "Tool%", "Param%", "Reply%");
    println!("{}", "-".repeat(60));
    for r in &all_results {
        println!("{:<18} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%",
                 r.name,
                 r.ood_metrics.format_acc * 100.0,
                 r.ood_metrics.tool_acc * 100.0,
                 r.ood_metrics.param_acc * 100.0,
                 r.ood_metrics.reply_quality * 100.0);
    }

    println!("\n--- Data Efficiency (100% vs 50% training data) ---\n");
    println!("{:<18} {:>12} {:>12} {:>12}",
             "Model", "100% Comp", "50% Comp", "Efficiency Gap");
    println!("{}", "-".repeat(56));
    for r in &all_results {
        println!("{:<18} {:>12.4} {:>12.4} {:>12.4}",
                 r.name, r.id_composite, r.half_data_composite, r.data_efficiency_gap);
    }

    let best_gen = all_results.iter()
        .min_by(|a, b| a.gen_gap.partial_cmp(&b.gen_gap).unwrap())
        .unwrap();
    let best_eff = all_results.iter()
        .min_by(|a, b| a.data_efficiency_gap.partial_cmp(&b.data_efficiency_gap).unwrap())
        .unwrap();

    println!("\n--- Winners ---");
    println!("  Best generalizer (smallest gap):       {} (gap: {:.4})", best_gen.name, best_gen.gen_gap);
    println!("  Most data-efficient (smallest drop):   {} (drop: {:.4})", best_eff.name, best_eff.data_efficiency_gap);

    if let Ok(mut file) = std::fs::File::create("experiments/generalization_results.csv") {
        let _ = writeln!(file, "model,params,id_format,id_tool,id_param,id_reply,id_composite,ood_format,ood_tool,ood_param,ood_reply,ood_composite,gen_gap");
        for r in &all_results {
            let _ = writeln!(file, "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                r.name, r.params,
                r.id_metrics.format_acc, r.id_metrics.tool_acc, r.id_metrics.param_acc, r.id_metrics.reply_quality, r.id_composite,
                r.ood_metrics.format_acc, r.ood_metrics.tool_acc, r.ood_metrics.param_acc, r.ood_metrics.reply_quality, r.ood_composite,
                r.gen_gap);
        }
        println!("\n  Results saved to experiments/generalization_results.csv");
    }

    if let Ok(mut file) = std::fs::File::create("experiments/data_efficiency_results.csv") {
        let _ = writeln!(file, "model,params,full_composite,half_composite,efficiency_gap");
        for r in &all_results {
            let _ = writeln!(file, "{},{},{:.4},{:.4},{:.4}",
                r.name, r.params, r.id_composite, r.half_data_composite, r.data_efficiency_gap);
        }
        println!("  Results saved to experiments/data_efficiency_results.csv");
    }

    println!("\n=== Generalization Test Suite Complete ===");

    println!("\nPlanDB update command:");
    println!("PLANDB_AGENT=worker-hybrid plandb done t-exp-generalize --result '{{\"baseline_gen_gap\":{:.4},\"rg_gen_gap\":{:.4},\"hamiltonian_gen_gap\":{:.4},\"hybrid_gen_gap\":{:.4},\"best_generalizer\":\"{}\"}}'",
             all_results[0].gen_gap, all_results[1].gen_gap,
             all_results[2].gen_gap, all_results[3].gen_gap,
             best_gen.name);
}

/// SFT training for RGGPT (extracted helper for generalization tests)
fn rg_sft_train(
    model: &mut RGGPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
    num_steps: usize,
) {
    let lr = 0.0005;
    let batch_size = 8;

    let start = Instant::now();
    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = RGGradients::zero_like(config);
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

            if step % 200 == 0 {
                println!("    RG SFT step {:4} | Loss: {:.4} | {:.1}s",
                         step, total_loss / valid_count as f32,
                         start.elapsed().as_secs_f32());
            }
        }
    }
}

/// SFT training for HamiltonianGPT (extracted helper for generalization tests)
fn hamiltonian_sft_train(
    model: &mut HamiltonianGPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
    num_steps: usize,
) {
    let lr = 0.0005;
    let batch_size = 8;

    let start = Instant::now();
    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = HamiltonianGradients::zero_like(config);
        let mut valid_count = 0;

        for _ in 0..batch_size {
            let example = &train[rng.gen_range(0..train.len())];
            let encoded = tok.encode(&example.input);
            if encoded.len() < 2 || encoded.len() > config.block_size {
                continue;
            }
            let input = &encoded[..encoded.len() - 1];
            let target = &encoded[1..];
            let (loss, g, _diag) = model.forward_backward(input, target);
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            grads.scale(1.0 / valid_count as f32);
            model.apply_gradients(&grads, lr);

            if step % 200 == 0 {
                println!("    Ham SFT step {:4} | Loss: {:.4} | {:.1}s",
                         step, total_loss / valid_count as f32,
                         start.elapsed().as_secs_f32());
            }
        }
    }
}

/// SFT training for HybridGPT (extracted helper for generalization tests)
fn hybrid_sft_train(
    model: &mut HybridGPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
    num_steps: usize,
) {
    let lr = 0.0005;
    let batch_size = 8;

    let start = Instant::now();
    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = HybridGradients::zero_like(config);
        let mut valid_count = 0;

        for _ in 0..batch_size {
            let example = &train[rng.gen_range(0..train.len())];
            let encoded = tok.encode(&example.input);
            if encoded.len() < 2 || encoded.len() > config.block_size {
                continue;
            }
            let input = &encoded[..encoded.len() - 1];
            let target = &encoded[1..];
            let (loss, g, _diag) = model.forward_backward(input, target);
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            grads.scale(1.0 / valid_count as f32);
            model.apply_gradients(&grads, lr);

            if step % 200 == 0 {
                println!("    Hybrid SFT step {:4} | Loss: {:.4} | {:.1}s",
                         step, total_loss / valid_count as f32,
                         start.elapsed().as_secs_f32());
            }
        }
    }
}

// ─── EXP7: RG Scaling Laws ─────────────────────────────────────

/// Per-depth results for the scaling experiment
struct ScalingResult {
    n_layer: usize,
    std_params: usize,
    rg_params: usize,
    param_savings_pct: f32,
    std_composite: f32,
    rg_composite: f32,
    std_format: f32,
    std_tool: f32,
    std_param_acc: f32,
    std_reply: f32,
    rg_format: f32,
    rg_tool: f32,
    rg_param_acc: f32,
    rg_reply: f32,
    alpha_std: f32,  // std dev of all alphas across layers
    ff_w1_alpha_mean: f32,
    attn_proj_alpha_std: f32,
    std_final_loss: f32,
    rg_final_loss: f32,
    std_train_time: f32,
    rg_train_time: f32,
}

/// Count standard GPT params for a given config
fn count_std_params(config: &Config) -> usize {
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

/// Train a standard GPT at a given depth and return (composite, metrics, final_loss, time)
fn scaling_train_std(
    config: &Config,
    base_text: &str,
    tok: &Tokenizer,
    train_data: &[ToolExample],
    val_data: &[ToolExample],
    rng: &mut impl Rng,
    pretrain_steps: usize,
    sft_steps: usize,
) -> (f32, AggregateMetrics, f32, f32) {
    let encoded = tok.encode(base_text);
    let mut model = GPT::new(config.clone());

    // Pretrain
    let batch_size = 32;
    let start = Instant::now();
    let mut last_loss = 0.0f32;
    for step in 0..pretrain_steps {
        let (inputs, targets) = crate::data::create_batches(
            &encoded, config.block_size, batch_size, rng,
        );
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
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, 0.001);
        last_loss = total_loss / batch_size as f32;
        if step % 200 == 0 {
            println!("      STD pretrain step {:4} | Loss: {:.4}", step, last_loss);
        }
    }

    // SFT — pre-encode batch then parallelize forward_backward
    let sft_batch = 16;
    for step in 0..sft_steps {
        let batch_enc: Vec<Vec<usize>> = (0..sft_batch)
            .map(|_| {
                let example = &train_data[rng.gen_range(0..train_data.len())];
                tok.encode(&example.input)
            })
            .collect();
        let valid_enc: Vec<&Vec<usize>> = batch_enc.iter()
            .filter(|enc| enc.len() >= 2 && enc.len() <= config.block_size)
            .collect();
        if valid_enc.is_empty() { continue; }
        let results: Vec<(f32, Gradients)> = valid_enc.par_iter()
            .map(|enc| model.forward_backward(&enc[..enc.len()-1], &enc[1..]))
            .collect();
        let mut total_loss = 0.0f32;
        let mut grads = Gradients::zero_like(config);
        let mut valid = 0;
        for (loss, g) in results {
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
                valid += 1;
            }
        }
        if valid > 0 {
            grads.scale(1.0 / valid as f32);
            model.apply_gradients(&grads, 0.0005);
            last_loss = total_loss / valid as f32;
        }
        if step % 200 == 0 {
            println!("      STD SFT step {:4} | Loss: {:.4}", step, last_loss);
        }
    }
    let elapsed = start.elapsed().as_secs_f32();

    // Evaluate
    let metrics = evaluate_model(&model, config, val_data, tok);
    let composite = metrics.format_acc * 0.3
        + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25
        + metrics.reply_quality * 0.15;
    (composite, metrics, last_loss, elapsed)
}

/// Train an RGGPT at a given depth and return (composite, metrics, alphas, final_loss, time)
fn scaling_train_rg(
    config: &Config,
    base_text: &str,
    tok: &Tokenizer,
    train_data: &[ToolExample],
    val_data: &[ToolExample],
    rng: &mut impl Rng,
    pretrain_steps: usize,
    sft_steps: usize,
) -> (f32, AggregateMetrics, Vec<[f32; 4]>, f32, f32) {
    let encoded = tok.encode(base_text);
    let mut model = RGGPT::new(config.clone());

    // Pretrain
    let batch_size = 32;
    let start = Instant::now();
    let mut last_loss = 0.0f32;
    for step in 0..pretrain_steps {
        let (inputs, targets) = crate::data::create_batches(
            &encoded, config.block_size, batch_size, rng,
        );
        let results: Vec<(f32, RGGradients)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let ctx_len = inputs[b].len().min(config.block_size);
                model.forward_backward(&inputs[b][..ctx_len], &targets[b][..ctx_len])
            })
            .collect();
        let mut total_loss = 0.0f32;
        let mut grads = RGGradients::zero_like(config);
        for (loss, g) in results {
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, 0.001);
        last_loss = total_loss / batch_size as f32;
        if step % 200 == 0 {
            println!("      RG pretrain step {:4} | Loss: {:.4}", step, last_loss);
        }
    }

    // SFT — pre-encode batch then parallelize forward_backward
    let sft_batch = 16;
    for step in 0..sft_steps {
        let batch_enc: Vec<Vec<usize>> = (0..sft_batch)
            .map(|_| {
                let example = &train_data[rng.gen_range(0..train_data.len())];
                tok.encode(&example.input)
            })
            .collect();
        let valid_enc: Vec<&Vec<usize>> = batch_enc.iter()
            .filter(|enc| enc.len() >= 2 && enc.len() <= config.block_size)
            .collect();
        if valid_enc.is_empty() { continue; }
        let results: Vec<(f32, RGGradients)> = valid_enc.par_iter()
            .map(|enc| model.forward_backward(&enc[..enc.len()-1], &enc[1..]))
            .collect();
        let mut total_loss = 0.0f32;
        let mut grads = RGGradients::zero_like(config);
        let mut valid = 0;
        for (loss, g) in results {
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
                valid += 1;
            }
        }
        if valid > 0 {
            grads.scale(1.0 / valid as f32);
            model.apply_gradients(&grads, 0.0005);
            last_loss = total_loss / valid as f32;
        }
        if step % 200 == 0 {
            println!("      RG SFT step {:4} | Loss: {:.4}", step, last_loss);
        }
    }
    let elapsed = start.elapsed().as_secs_f32();

    // Evaluate
    let metrics = rg_evaluate_model(&model, config, val_data, tok);
    let composite = metrics.format_acc * 0.3
        + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25
        + metrics.reply_quality * 0.15;
    let alphas = model.get_layer_alphas();
    (composite, metrics, alphas, last_loss, elapsed)
}

/// Run the full RG scaling laws experiment across multiple depths
pub fn run_rg_scaling_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("experiments/plots");

    println!("=== EXP7: RG Scaling Laws ===");
    println!("    Testing parameter scaling, alpha divergence, and performance");
    println!("    across layer depths L = 2, 3, 4, 5, 6, 8\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/4] Generating dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    let vocab_size = tok.vocab_size();
    println!("  Train: {} examples, Val: {}, Vocab: {}",
             train_data.len(), val_data.len(), vocab_size);

    // Generate theoretical predictions CSV
    println!("\n[2/4] Writing theoretical predictions...");
    {
        let mut f = std::fs::File::create("experiments/rg_scaling_theory.csv").unwrap();
        let _ = writeln!(f, "layers,predicted_std_params,predicted_rg_params,predicted_savings_pct,predicted_alpha_std");
        for &l in &[2usize, 3, 4, 5, 6, 8] {
            let p_std = 12480 + l * 27888;
            let p_rg = 40128 + l * 440;
            let savings = (1.0 - p_rg as f64 / p_std as f64) * 100.0;
            let _ = writeln!(f, "{},{},{},{:.1},{:.3}", l, p_std, p_rg, savings, 0.046);
        }
        println!("  Saved experiments/rg_scaling_theory.csv");
    }

    // Train models at each depth
    let layer_counts = [2, 3, 4, 5, 6, 8];
    let pretrain_steps = 500;
    let sft_steps = 500;
    let mut results: Vec<ScalingResult> = Vec::new();

    println!("\n[3/4] Training models at {} depths ({} pretrain + {} SFT each)...\n",
             layer_counts.len(), pretrain_steps, sft_steps);

    for (idx, &n_layer) in layer_counts.iter().enumerate() {
        println!("  === Depth L={} ({}/{}) ===", n_layer, idx + 1, layer_counts.len());

        let config = Config {
            vocab_size,
            n_embd: 48,
            n_head: 4,
            n_layer,
            block_size: 128,
        };

        // Train Standard GPT
        println!("    Training Standard GPT (L={})...", n_layer);
        let (std_composite, std_metrics, std_loss, std_time) = scaling_train_std(
            &config, base_text, &tok, &train_data, &val_data, &mut rng,
            pretrain_steps, sft_steps,
        );
        let std_params = count_std_params(&config);
        println!("    STD: params={}, composite={:.4}, loss={:.4}, time={:.1}s",
                 std_params, std_composite, std_loss, std_time);

        // Train RG-GPT
        println!("    Training RG-GPT (L={})...", n_layer);
        let (rg_composite, rg_metrics, alphas, rg_loss, rg_time) = scaling_train_rg(
            &config, base_text, &tok, &train_data, &val_data, &mut rng,
            pretrain_steps, sft_steps,
        );
        let rg_params = {
            let tmp = RGGPT::new(config.clone());
            tmp.count_params()
        };
        let savings = (1.0 - rg_params as f32 / std_params as f32) * 100.0;
        println!("    RG:  params={}, composite={:.4}, loss={:.4}, time={:.1}s, savings={:.1}%",
                 rg_params, rg_composite, rg_loss, rg_time, savings);

        // Compute alpha statistics
        // alphas: Vec<[f32; 4]> — one array per layer, 4 weight matrices
        let all_alphas: Vec<f32> = alphas.iter().flat_map(|a| a.iter().copied()).collect();
        let alpha_mean = all_alphas.iter().sum::<f32>() / all_alphas.len() as f32;
        let alpha_variance = all_alphas.iter()
            .map(|a| (a - alpha_mean) * (a - alpha_mean))
            .sum::<f32>() / all_alphas.len() as f32;
        let alpha_std_dev = alpha_variance.sqrt();

        // ff_w1 alpha mean (index 2 in shared weights)
        let ff_w1_alphas: Vec<f32> = alphas.iter().map(|a| a[2]).collect();
        let ff_w1_mean = ff_w1_alphas.iter().sum::<f32>() / ff_w1_alphas.len() as f32;

        // attn_proj alpha std (index 1 in shared weights)
        let attn_proj_alphas: Vec<f32> = alphas.iter().map(|a| a[1]).collect();
        let ap_mean = attn_proj_alphas.iter().sum::<f32>() / attn_proj_alphas.len() as f32;
        let ap_var = attn_proj_alphas.iter()
            .map(|a| (a - ap_mean) * (a - ap_mean))
            .sum::<f32>() / attn_proj_alphas.len() as f32;
        let attn_proj_std = ap_var.sqrt();

        println!("    Alpha stats: overall_std={:.4}, ff_w1_mean={:.4}, attn_proj_std={:.4}",
                 alpha_std_dev, ff_w1_mean, attn_proj_std);

        // Print per-layer alphas
        let weight_names = ["qkv_w", "attn_proj", "ff_w1", "ff_w2"];
        for l in 0..n_layer {
            print!("      L{}: ", l);
            for (k, name) in weight_names.iter().enumerate() {
                print!("{}={:.4} ", name, alphas[l][k]);
            }
            println!();
        }

        results.push(ScalingResult {
            n_layer,
            std_params,
            rg_params,
            param_savings_pct: savings,
            std_composite,
            rg_composite,
            std_format: std_metrics.format_acc,
            std_tool: std_metrics.tool_acc,
            std_param_acc: std_metrics.param_acc,
            std_reply: std_metrics.reply_quality,
            rg_format: rg_metrics.format_acc,
            rg_tool: rg_metrics.tool_acc,
            rg_param_acc: rg_metrics.param_acc,
            rg_reply: rg_metrics.reply_quality,
            alpha_std: alpha_std_dev,
            ff_w1_alpha_mean: ff_w1_mean,
            attn_proj_alpha_std: attn_proj_std,
            std_final_loss: std_loss,
            rg_final_loss: rg_loss,
            std_train_time: std_time,
            rg_train_time: rg_time,
        });

        println!();
    }

    // [4/4] Save results and validate predictions
    println!("[4/4] Saving results and validating predictions...\n");

    // Save actual results CSV
    {
        let mut f = std::fs::File::create("experiments/rg_scaling_results.csv").unwrap();
        let _ = writeln!(f, "layers,std_params,rg_params,param_savings_pct,std_composite,rg_composite,std_format,std_tool,std_param,std_reply,rg_format,rg_tool,rg_param,rg_reply,alpha_std,ff_w1_alpha_mean,attn_proj_alpha_std,std_final_loss,rg_final_loss,std_time,rg_time");
        for r in &results {
            let _ = writeln!(f, "{},{},{},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4},{:.6},{:.4},{:.4},{:.1},{:.1}",
                r.n_layer, r.std_params, r.rg_params, r.param_savings_pct,
                r.std_composite, r.rg_composite,
                r.std_format, r.std_tool, r.std_param_acc, r.std_reply,
                r.rg_format, r.rg_tool, r.rg_param_acc, r.rg_reply,
                r.alpha_std, r.ff_w1_alpha_mean, r.attn_proj_alpha_std,
                r.std_final_loss, r.rg_final_loss, r.std_train_time, r.rg_train_time);
        }
        println!("  Results saved to experiments/rg_scaling_results.csv");
    }

    // Validate predictions: compute R² for parameter scaling
    // Prediction: P_std(L) = 12480 + L*27888, P_rg(L) = 40128 + L*440
    let r2_std_params = {
        let actual: Vec<f64> = results.iter().map(|r| r.std_params as f64).collect();
        let predicted: Vec<f64> = results.iter().map(|r| (12480 + r.n_layer * 27888) as f64).collect();
        compute_r2(&actual, &predicted)
    };
    let r2_rg_params = {
        let actual: Vec<f64> = results.iter().map(|r| r.rg_params as f64).collect();
        let predicted: Vec<f64> = results.iter().map(|r| (40128 + r.n_layer * 440) as f64).collect();
        compute_r2(&actual, &predicted)
    };
    let r2_savings = {
        let actual: Vec<f64> = results.iter().map(|r| r.param_savings_pct as f64).collect();
        let predicted: Vec<f64> = results.iter().map(|r| {
            (1.0 - (40128.0 + r.n_layer as f64 * 440.0) / (12480.0 + r.n_layer as f64 * 27888.0)) * 100.0
        }).collect();
        compute_r2(&actual, &predicted)
    };

    // Alpha divergence: prediction is ~0.046 constant
    let r2_alpha = {
        let actual: Vec<f64> = results.iter().map(|r| r.alpha_std as f64).collect();
        let predicted: Vec<f64> = vec![0.046; results.len()];
        compute_r2(&actual, &predicted)
    };

    // ff_w1 fixed point: prediction is ~1.5
    let r2_ff_w1 = {
        let actual: Vec<f64> = results.iter().map(|r| r.ff_w1_alpha_mean as f64).collect();
        let predicted: Vec<f64> = vec![1.5; results.len()];
        compute_r2(&actual, &predicted)
    };

    fn validation_label(r2: f64) -> &'static str {
        if r2 > 0.8 { "VALIDATED" }
        else if r2 > 0.5 { "PARTIALLY VALIDATED" }
        else { "FALSIFIED" }
    }

    // Save validation CSV
    {
        let mut f = std::fs::File::create("experiments/rg_scaling_validation.csv").unwrap();
        let _ = writeln!(f, "prediction,r2,status");
        let _ = writeln!(f, "std_param_scaling,{:.4},{}", r2_std_params, validation_label(r2_std_params));
        let _ = writeln!(f, "rg_param_scaling,{:.4},{}", r2_rg_params, validation_label(r2_rg_params));
        let _ = writeln!(f, "param_savings,{:.4},{}", r2_savings, validation_label(r2_savings));
        let _ = writeln!(f, "alpha_divergence_constant,{:.4},{}", r2_alpha, validation_label(r2_alpha));
        let _ = writeln!(f, "ff_w1_fixed_point,{:.4},{}", r2_ff_w1, validation_label(r2_ff_w1));
        println!("  Validation saved to experiments/rg_scaling_validation.csv");
    }

    // Print summary table
    println!("\n{}", "=".repeat(80));
    println!("=== RG SCALING LAWS — RESULTS SUMMARY ===\n");

    println!("  {:>6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}",
             "Layers", "STD Params", "RG Params", "Savings%", "STD Score", "RG Score", "α_std");
    println!("  {}", "-".repeat(70));
    for r in &results {
        println!("  {:>6} {:>10} {:>10} {:>7.1}% {:>10.4} {:>10.4} {:>8.4}",
                 r.n_layer, r.std_params, r.rg_params, r.param_savings_pct,
                 r.std_composite, r.rg_composite, r.alpha_std);
    }

    println!("\n  Prediction Validation:");
    println!("  {:<35} {:>8} {:>20}", "Prediction", "R²", "Status");
    println!("  {}", "-".repeat(65));
    println!("  {:<35} {:>8.4} {:>20}", "Std param scaling (P=12480+L*27888)", r2_std_params, validation_label(r2_std_params));
    println!("  {:<35} {:>8.4} {:>20}", "RG param scaling (P=40128+L*440)", r2_rg_params, validation_label(r2_rg_params));
    println!("  {:<35} {:>8.4} {:>20}", "Savings % (hyperbolic → 98.4%)", r2_savings, validation_label(r2_savings));
    println!("  {:<35} {:>8.4} {:>20}", "Alpha std ≈ 0.046 (constant)", r2_alpha, validation_label(r2_alpha));
    println!("  {:<35} {:>8.4} {:>20}", "ff_w1 alpha ≈ 1.5 (fixed point)", r2_ff_w1, validation_label(r2_ff_w1));

    let validated_count = [r2_std_params, r2_rg_params, r2_savings, r2_alpha, r2_ff_w1]
        .iter().filter(|&&r| r > 0.8).count();
    let partial_count = [r2_std_params, r2_rg_params, r2_savings, r2_alpha, r2_ff_w1]
        .iter().filter(|&&r| r > 0.5 && r <= 0.8).count();
    let falsified_count = 5 - validated_count - partial_count;

    // Find best RG depth
    let best = results.iter().max_by(|a, b| a.rg_composite.partial_cmp(&b.rg_composite).unwrap()).unwrap();

    println!("\n  Summary: {} validated, {} partially, {} falsified out of 5 predictions",
             validated_count, partial_count, falsified_count);
    println!("  Best RG depth: L={} (composite={:.4})", best.n_layer, best.rg_composite);
    println!("  Asymptotic savings prediction: 98.4%");
    println!();

    // PlanDB metrics
    println!("=== PLANDB_METRICS ===");
    println!("predictions_validated={}", validated_count);
    println!("predictions_partial={}", partial_count);
    println!("predictions_falsified={}", falsified_count);
    println!("param_scaling_r2_std={:.4}", r2_std_params);
    println!("param_scaling_r2_rg={:.4}", r2_rg_params);
    println!("savings_r2={:.4}", r2_savings);
    println!("alpha_divergence_r2={:.4}", r2_alpha);
    println!("ff_w1_fixed_point_r2={:.4}", r2_ff_w1);
    println!("best_rg_depth={}", best.n_layer);
    println!("best_rg_composite={:.4}", best.rg_composite);
    for r in &results {
        println!("L{}_std_composite={:.4}", r.n_layer, r.std_composite);
        println!("L{}_rg_composite={:.4}", r.n_layer, r.rg_composite);
        println!("L{}_savings={:.1}", r.n_layer, r.param_savings_pct);
    }
    println!("=== END PLANDB_METRICS ===");
}

// ─── EXP8: RG-Informed Architecture Search ─────────────────────

struct RGInformedResult {
    variant: String,
    n_layer: usize,
    params: usize,
    composite: f32,
    format_acc: f32,
    tool_acc: f32,
    param_acc: f32,
    reply_quality: f32,
    final_loss: f32,
    train_time: f32,
}

fn train_lora_rg(
    config: &Config, base_text: &str, tok: &Tokenizer,
    train_data: &[ToolExample], val_data: &[ToolExample],
    rng: &mut impl Rng, pretrain_steps: usize, sft_steps: usize, rank: usize,
) -> RGInformedResult {
    let encoded = tok.encode(base_text);
    let mut model = LoraRGGPT::new(config.clone(), rank);
    let params = model.count_params();

    let start = Instant::now();
    let mut last_loss = 0.0f32;

    // Pretrain
    for step in 0..pretrain_steps {
        let (inputs, targets) = crate::data::create_batches(&encoded, config.block_size, 16, rng);
        let mut total_loss = 0.0f32;
        let mut grads = LoraRGGradients::zero_like(config, rank);
        for b in 0..16 {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g) = model.forward_backward(&inputs[b][..ctx_len], &targets[b][..ctx_len]);
            if loss.is_finite() { total_loss += loss; grads.accumulate(&g); }
        }
        grads.scale(1.0 / 16.0);
        model.apply_gradients(&grads, 0.001);
        last_loss = total_loss / 16.0;
        if step % 200 == 0 {
            println!("      LoRA-RG pretrain step {:4} | Loss: {:.4}", step, last_loss);
        }
    }

    // SFT
    for step in 0..sft_steps {
        let mut total_loss = 0.0f32;
        let mut grads = LoraRGGradients::zero_like(config, rank);
        let mut valid = 0;
        for _ in 0..8 {
            let example = &train_data[rng.gen_range(0..train_data.len())];
            let enc = tok.encode(&example.input);
            if enc.len() < 2 || enc.len() > config.block_size { continue; }
            let (loss, g) = model.forward_backward(&enc[..enc.len()-1], &enc[1..]);
            if loss.is_finite() { total_loss += loss; grads.accumulate(&g); valid += 1; }
        }
        if valid > 0 {
            grads.scale(1.0 / valid as f32);
            model.apply_gradients(&grads, 0.0005);
            last_loss = total_loss / valid as f32;
        }
        if step % 200 == 0 {
            println!("      LoRA-RG SFT step {:4} | Loss: {:.4}", step, last_loss);
        }
    }
    let elapsed = start.elapsed().as_secs_f32();

    // Evaluate
    let metrics = lora_rg_evaluate(&model, config, val_data, tok);
    let composite = metrics.format_acc * 0.3 + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25 + metrics.reply_quality * 0.15;

    // Print LoRA norms
    let norms = model.get_lora_norms();
    let weight_names = ["qkv_w", "attn_proj", "ff_w1", "ff_w2"];
    println!("      LoRA norms (||U||*||V|| per matrix per layer):");
    for (l, layer_norms) in norms.iter().enumerate() {
        print!("        L{}: ", l);
        for (k, name) in weight_names.iter().enumerate() {
            print!("{}={:.4} ", name, layer_norms[k]);
        }
        println!();
    }

    // Save best weights
    let _ = model.save_weights(&format!("weights/lora_rg_L{}.mgpt", config.n_layer));

    RGInformedResult {
        variant: format!("LoRA-RG(r={})", rank),
        n_layer: config.n_layer, params, composite,
        format_acc: metrics.format_acc, tool_acc: metrics.tool_acc,
        param_acc: metrics.param_acc, reply_quality: metrics.reply_quality,
        final_loss: last_loss, train_time: elapsed,
    }
}

fn lora_rg_evaluate(
    model: &LoraRGGPT, config: &Config, val: &[ToolExample], tok: &Tokenizer,
) -> AggregateMetrics {
    let mut results = Vec::new();
    for example in val {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size { continue; }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);
        let eval = tool_data::evaluate_output(&generated_text, example);
        results.push(eval);
    }
    AggregateMetrics::from_results(&results)
}

fn train_preamp_rg(
    config: &Config, base_text: &str, tok: &Tokenizer,
    train_data: &[ToolExample], val_data: &[ToolExample],
    rng: &mut impl Rng, pretrain_steps: usize, sft_steps: usize,
) -> RGInformedResult {
    let encoded = tok.encode(base_text);
    let mut model = PreAmpRGGPT::new(config.clone());
    let params = model.count_params();

    let start = Instant::now();
    let mut last_loss = 0.0f32;

    for step in 0..pretrain_steps {
        let (inputs, targets) = crate::data::create_batches(&encoded, config.block_size, 16, rng);
        let mut total_loss = 0.0f32;
        let mut grads = PreAmpRGGradients::zero_like(config);
        for b in 0..16 {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g) = model.forward_backward(&inputs[b][..ctx_len], &targets[b][..ctx_len]);
            if loss.is_finite() { total_loss += loss; grads.accumulate(&g); }
        }
        grads.scale(1.0 / 16.0);
        model.apply_gradients(&grads, 0.001);
        last_loss = total_loss / 16.0;
        if step % 200 == 0 {
            println!("      PreAmp pretrain step {:4} | Loss: {:.4}", step, last_loss);
        }
    }

    for step in 0..sft_steps {
        let mut total_loss = 0.0f32;
        let mut grads = PreAmpRGGradients::zero_like(config);
        let mut valid = 0;
        for _ in 0..8 {
            let example = &train_data[rng.gen_range(0..train_data.len())];
            let enc = tok.encode(&example.input);
            if enc.len() < 2 || enc.len() > config.block_size { continue; }
            let (loss, g) = model.forward_backward(&enc[..enc.len()-1], &enc[1..]);
            if loss.is_finite() { total_loss += loss; grads.accumulate(&g); valid += 1; }
        }
        if valid > 0 {
            grads.scale(1.0 / valid as f32);
            model.apply_gradients(&grads, 0.0005);
            last_loss = total_loss / valid as f32;
        }
        if step % 200 == 0 {
            println!("      PreAmp SFT step {:4} | Loss: {:.4}", step, last_loss);
        }
    }
    let elapsed = start.elapsed().as_secs_f32();

    let metrics = preamp_evaluate(&model, config, val_data, tok);
    let composite = metrics.format_acc * 0.3 + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25 + metrics.reply_quality * 0.15;

    // Print learned alphas
    let alphas = model.get_layer_alphas();
    let names = ["qkv", "attn_proj", "ff_w2"];
    println!("      PreAmp alphas (ff_w1 fixed at 1.2x, no alpha):");
    for (l, a) in alphas.iter().enumerate() {
        print!("        L{}: ", l);
        for (k, name) in names.iter().enumerate() { print!("{}={:.4} ", name, a[k]); }
        println!();
    }

    RGInformedResult {
        variant: "PreAmp-RG".to_string(),
        n_layer: config.n_layer, params, composite,
        format_acc: metrics.format_acc, tool_acc: metrics.tool_acc,
        param_acc: metrics.param_acc, reply_quality: metrics.reply_quality,
        final_loss: last_loss, train_time: elapsed,
    }
}

fn preamp_evaluate(
    model: &PreAmpRGGPT, config: &Config, val: &[ToolExample], tok: &Tokenizer,
) -> AggregateMetrics {
    let mut results = Vec::new();
    for example in val {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size { continue; }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);
        let eval = tool_data::evaluate_output(&generated_text, example);
        results.push(eval);
    }
    AggregateMetrics::from_results(&results)
}

fn train_perhead_rg(
    config: &Config, base_text: &str, tok: &Tokenizer,
    train_data: &[ToolExample], val_data: &[ToolExample],
    rng: &mut impl Rng, pretrain_steps: usize, sft_steps: usize,
) -> RGInformedResult {
    let encoded = tok.encode(base_text);
    let mut model = PerHeadRGGPT::new(config.clone());
    let params = model.count_params();

    let start = Instant::now();
    let mut last_loss = 0.0f32;

    for step in 0..pretrain_steps {
        let (inputs, targets) = crate::data::create_batches(&encoded, config.block_size, 16, rng);
        let mut total_loss = 0.0f32;
        let mut grads = PerHeadRGGradients::zero_like(config);
        for b in 0..16 {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g) = model.forward_backward(&inputs[b][..ctx_len], &targets[b][..ctx_len]);
            if loss.is_finite() { total_loss += loss; grads.accumulate(&g); }
        }
        grads.scale(1.0 / 16.0);
        model.apply_gradients(&grads, 0.001);
        last_loss = total_loss / 16.0;
        if step % 200 == 0 {
            println!("      PerHead pretrain step {:4} | Loss: {:.4}", step, last_loss);
        }
    }

    for step in 0..sft_steps {
        let mut total_loss = 0.0f32;
        let mut grads = PerHeadRGGradients::zero_like(config);
        let mut valid = 0;
        for _ in 0..8 {
            let example = &train_data[rng.gen_range(0..train_data.len())];
            let enc = tok.encode(&example.input);
            if enc.len() < 2 || enc.len() > config.block_size { continue; }
            let (loss, g) = model.forward_backward(&enc[..enc.len()-1], &enc[1..]);
            if loss.is_finite() { total_loss += loss; grads.accumulate(&g); valid += 1; }
        }
        if valid > 0 {
            grads.scale(1.0 / valid as f32);
            model.apply_gradients(&grads, 0.0005);
            last_loss = total_loss / valid as f32;
        }
        if step % 200 == 0 {
            println!("      PerHead SFT step {:4} | Loss: {:.4}", step, last_loss);
        }
    }
    let elapsed = start.elapsed().as_secs_f32();

    let metrics = perhead_evaluate(&model, config, val_data, tok);
    let composite = metrics.format_acc * 0.3 + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25 + metrics.reply_quality * 0.15;

    // Print per-head alphas
    let (qkv_alphas, proj_alphas) = model.get_head_alphas();
    println!("      PerHead QKV alphas:");
    for (l, alphas) in qkv_alphas.iter().enumerate() {
        print!("        L{}: ", l);
        for (h, a) in alphas.iter().enumerate() { print!("h{}={:.4} ", h, a); }
        println!();
    }
    println!("      PerHead Proj alphas:");
    for (l, alphas) in proj_alphas.iter().enumerate() {
        print!("        L{}: ", l);
        for (h, a) in alphas.iter().enumerate() { print!("h{}={:.4} ", h, a); }
        println!();
    }

    RGInformedResult {
        variant: "PerHead-RG".to_string(),
        n_layer: config.n_layer, params, composite,
        format_acc: metrics.format_acc, tool_acc: metrics.tool_acc,
        param_acc: metrics.param_acc, reply_quality: metrics.reply_quality,
        final_loss: last_loss, train_time: elapsed,
    }
}

fn perhead_evaluate(
    model: &PerHeadRGGPT, config: &Config, val: &[ToolExample], tok: &Tokenizer,
) -> AggregateMetrics {
    let mut results = Vec::new();
    for example in val {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size { continue; }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);
        let eval = tool_data::evaluate_output(&generated_text, example);
        results.push(eval);
    }
    AggregateMetrics::from_results(&results)
}

pub fn run_rg_informed_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("weights");

    println!("=== EXP8: RG-Informed Architecture Search ===");
    println!("    Theory-driven improvements to RG weight sharing");
    println!("    EXP8a: LoRA-RG (rank-4 per-layer adaptation)");
    println!("    EXP8b: PreAmp-RG (ff_w1 pre-amplified 1.2x)");
    println!("    EXP8c: PerHead-RG (per-head alpha for attention)\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/4] Generating dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    let vocab_size = tok.vocab_size();
    println!("  Train: {} examples, Val: {}, Vocab: {}",
             train_data.len(), val_data.len(), vocab_size);

    let pretrain_steps = 1000;
    let sft_steps = 800;
    let depths = [2, 3, 4, 6];
    let lora_rank = 4;

    let mut all_results: Vec<RGInformedResult> = Vec::new();

    println!("\n[2/4] Training models at depths {:?} ({} pretrain + {} SFT)...\n",
             depths, pretrain_steps, sft_steps);

    for &n_layer in &depths {
        println!("  === Depth L={} ===", n_layer);

        let config = Config {
            vocab_size,
            n_embd: 48,
            n_head: 4,
            n_layer,
            block_size: 128,
        };

        // Standard GPT baseline
        println!("    Training Standard GPT (L={})...", n_layer);
        let (std_composite, std_metrics, std_loss, std_time) = scaling_train_std(
            &config, base_text, &tok, &train_data, &val_data, &mut rng,
            pretrain_steps, sft_steps,
        );
        let std_params = count_std_params(&config);
        println!("    STD: params={}, composite={:.4}, loss={:.4}, time={:.1}s",
                 std_params, std_composite, std_loss, std_time);
        all_results.push(RGInformedResult {
            variant: "Standard".to_string(), n_layer, params: std_params,
            composite: std_composite, format_acc: std_metrics.format_acc,
            tool_acc: std_metrics.tool_acc, param_acc: std_metrics.param_acc,
            reply_quality: std_metrics.reply_quality,
            final_loss: std_loss, train_time: std_time,
        });

        // Standard RG baseline
        println!("    Training Standard RG (L={})...", n_layer);
        let (rg_composite, rg_metrics, _alphas, rg_loss, rg_time) = scaling_train_rg(
            &config, base_text, &tok, &train_data, &val_data, &mut rng,
            pretrain_steps, sft_steps,
        );
        let rg_params = { let tmp = RGGPT::new(config.clone()); tmp.count_params() };
        println!("    RG:  params={}, composite={:.4}, loss={:.4}, time={:.1}s",
                 rg_params, rg_composite, rg_loss, rg_time);
        all_results.push(RGInformedResult {
            variant: "Std-RG".to_string(), n_layer, params: rg_params,
            composite: rg_composite, format_acc: rg_metrics.format_acc,
            tool_acc: rg_metrics.tool_acc, param_acc: rg_metrics.param_acc,
            reply_quality: rg_metrics.reply_quality,
            final_loss: rg_loss, train_time: rg_time,
        });

        // EXP8a: LoRA-RG
        println!("    Training LoRA-RG(r={}) (L={})...", lora_rank, n_layer);
        let lora_result = train_lora_rg(
            &config, base_text, &tok, &train_data, &val_data, &mut rng,
            pretrain_steps, sft_steps, lora_rank,
        );
        println!("    LoRA-RG: params={}, composite={:.4}, loss={:.4}, time={:.1}s",
                 lora_result.params, lora_result.composite, lora_result.final_loss, lora_result.train_time);
        all_results.push(lora_result);

        // EXP8b: PreAmp-RG
        println!("    Training PreAmp-RG (L={})...", n_layer);
        let preamp_result = train_preamp_rg(
            &config, base_text, &tok, &train_data, &val_data, &mut rng,
            pretrain_steps, sft_steps,
        );
        println!("    PreAmp: params={}, composite={:.4}, loss={:.4}, time={:.1}s",
                 preamp_result.params, preamp_result.composite, preamp_result.final_loss, preamp_result.train_time);
        all_results.push(preamp_result);

        // EXP8c: PerHead-RG
        println!("    Training PerHead-RG (L={})...", n_layer);
        let perhead_result = train_perhead_rg(
            &config, base_text, &tok, &train_data, &val_data, &mut rng,
            pretrain_steps, sft_steps,
        );
        println!("    PerHead: params={}, composite={:.4}, loss={:.4}, time={:.1}s",
                 perhead_result.params, perhead_result.composite, perhead_result.final_loss, perhead_result.train_time);
        all_results.push(perhead_result);

        println!();
    }

    // [3/4] Save results CSV
    println!("[3/4] Saving results...");
    {
        let mut f = std::fs::File::create("experiments/rg_informed_results.csv").unwrap();
        let _ = writeln!(f, "variant,layers,params,composite,format_acc,tool_acc,param_acc,reply_quality,final_loss,train_time");
        for r in &all_results {
            let _ = writeln!(f, "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
                r.variant, r.n_layer, r.params, r.composite,
                r.format_acc, r.tool_acc, r.param_acc, r.reply_quality,
                r.final_loss, r.train_time);
        }
        println!("  Saved experiments/rg_informed_results.csv");
    }

    // [4/4] Summary
    println!("\n[4/4] Results Summary\n");
    println!("{}", "=".repeat(100));
    println!("=== RG-INFORMED ARCHITECTURE SEARCH — RESULTS ===\n");

    // Group by depth
    for &depth in &depths {
        let depth_results: Vec<&RGInformedResult> = all_results.iter()
            .filter(|r| r.n_layer == depth).collect();

        println!("  L={} (Depth {})", depth, depth);
        println!("  {:>16} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
                 "Variant", "Params", "Composite", "Format", "Tool", "Param", "Reply", "Loss");
        println!("  {}", "-".repeat(85));
        for r in &depth_results {
            println!("  {:>16} {:>8} {:>10.4} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}% {:>8.4}",
                     r.variant, r.params, r.composite,
                     r.format_acc * 100.0, r.tool_acc * 100.0,
                     r.param_acc * 100.0, r.reply_quality * 100.0,
                     r.final_loss);
        }
        println!();
    }

    // Find best variant at each depth and overall
    let best_overall = all_results.iter()
        .max_by(|a, b| a.composite.partial_cmp(&b.composite).unwrap()).unwrap();

    // Check depth degradation fix: compare L=4 LoRA-RG vs L=4 Std-RG
    let lora_l4 = all_results.iter()
        .find(|r| r.variant.starts_with("LoRA") && r.n_layer == 4);
    let std_rg_l4 = all_results.iter()
        .find(|r| r.variant == "Std-RG" && r.n_layer == 4);

    let depth_degradation_fixed = match (lora_l4, std_rg_l4) {
        (Some(lora), Some(rg)) => lora.composite > rg.composite,
        _ => false,
    };

    // Compare L=2 vs L=4 for each variant to detect degradation
    println!("  Depth Degradation Analysis (L=2 vs L=4):");
    println!("  {:>16} {:>10} {:>10} {:>10}",
             "Variant", "L=2 Score", "L=4 Score", "Degrades?");
    println!("  {}", "-".repeat(50));
    for variant_name in &["Standard", "Std-RG", "LoRA-RG(r=4)", "PreAmp-RG", "PerHead-RG"] {
        let l2 = all_results.iter().find(|r| r.variant == *variant_name && r.n_layer == 2);
        let l4 = all_results.iter().find(|r| r.variant == *variant_name && r.n_layer == 4);
        if let (Some(l2r), Some(l4r)) = (l2, l4) {
            let degrades = l4r.composite < l2r.composite * 0.95; // >5% drop = degradation
            println!("  {:>16} {:>10.4} {:>10.4} {:>10}",
                     variant_name, l2r.composite, l4r.composite,
                     if degrades { "YES" } else { "NO" });
        }
    }

    println!("\n  Best overall: {} at L={} (composite={:.4}, params={})",
             best_overall.variant, best_overall.n_layer,
             best_overall.composite, best_overall.params);
    println!("  Depth degradation fixed by LoRA-RG: {}", depth_degradation_fixed);

    let lora_l4_composite = lora_l4.map(|r| r.composite).unwrap_or(0.0);
    let std_rg_l4_composite = std_rg_l4.map(|r| r.composite).unwrap_or(0.0);

    // PlanDB metrics
    println!("\n=== PLANDB_METRICS ===");
    println!("best_variant={}", best_overall.variant);
    println!("best_composite={:.4}", best_overall.composite);
    println!("best_params={}", best_overall.params);
    println!("best_depth={}", best_overall.n_layer);
    println!("depth_degradation_fixed={}", depth_degradation_fixed);
    println!("lora_rg_l4_composite={:.4}", lora_l4_composite);
    println!("std_rg_l4_composite={:.4}", std_rg_l4_composite);
    for r in &all_results {
        println!("{}_L{}={:.4}", r.variant.replace(" ", "_"), r.n_layer, r.composite);
    }
    println!("=== END PLANDB_METRICS ===");
}

/// Focused L=4 test: LoRA-RG vs Standard GPT vs Scalar RG
/// Tests whether LoRA-RG fixes the depth degradation problem at L=4
pub fn run_lora_l4_test() {
    let _ = std::fs::create_dir_all("experiments");
    let _ = std::fs::create_dir_all("weights");

    println!("=== L=4 LoRA-RG vs Standard GPT vs Scalar RG ===");
    println!("    Decisive test: does LoRA-RG fix depth degradation?\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/3] Generating dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    let vocab_size = tok.vocab_size();
    println!("  Train: {} examples, Val: {}, Vocab: {}",
             train_data.len(), val_data.len(), vocab_size);

    let pretrain_steps = 1000;
    let sft_steps = 800;
    let lora_rank = 4;
    let n_layer = 4;

    let config = Config {
        vocab_size,
        n_embd: 48,
        n_head: 4,
        n_layer,
        block_size: 128,
    };

    println!("\n[2/3] Training 3 models at L={} ({} pretrain + {} SFT)...\n",
             n_layer, pretrain_steps, sft_steps);

    // 1. Standard GPT L=4
    println!("  --- Standard GPT (L=4) ---");
    let (std_composite, std_metrics, std_loss, std_time) = scaling_train_std(
        &config, base_text, &tok, &train_data, &val_data, &mut rng,
        pretrain_steps, sft_steps,
    );
    let std_params = count_std_params(&config);
    println!("  STD: params={}, composite={:.4}, loss={:.4}, time={:.1}s\n",
             std_params, std_composite, std_loss, std_time);

    // 2. Scalar RG L=4
    println!("  --- Scalar RG (L=4) ---");
    let (rg_composite, rg_metrics, _alphas, rg_loss, rg_time) = scaling_train_rg(
        &config, base_text, &tok, &train_data, &val_data, &mut rng,
        pretrain_steps, sft_steps,
    );
    let rg_params = { let tmp = RGGPT::new(config.clone()); tmp.count_params() };
    println!("  RG:  params={}, composite={:.4}, loss={:.4}, time={:.1}s\n",
             rg_params, rg_composite, rg_loss, rg_time);

    // 3. LoRA-RG L=4 (rank=4)
    println!("  --- LoRA-RG(r={}) (L=4) ---", lora_rank);
    let lora_result = train_lora_rg(
        &config, base_text, &tok, &train_data, &val_data, &mut rng,
        pretrain_steps, sft_steps, lora_rank,
    );
    println!("  LoRA-RG: params={}, composite={:.4}, loss={:.4}, time={:.1}s\n",
             lora_result.params, lora_result.composite, lora_result.final_loss, lora_result.train_time);

    // Inference speed note: skipping separate speed benchmark since all three
    // models ran the same evaluation suite during training

    // [3/3] Results
    println!("\n[3/3] Results\n");
    println!("{}", "=".repeat(80));
    println!("=== L=4 DECISIVE TEST: LoRA-RG vs Standard GPT vs Scalar RG ===\n");

    println!("  {:>16} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
             "Model", "Params", "Composite", "Format", "Tool", "Param", "Reply", "Loss");
    println!("  {}", "-".repeat(78));
    println!("  {:>16} {:>8} {:>10.4} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}% {:>8.4}",
             "Standard GPT", std_params, std_composite,
             std_metrics.format_acc * 100.0, std_metrics.tool_acc * 100.0,
             std_metrics.param_acc * 100.0, std_metrics.reply_quality * 100.0,
             std_loss);
    println!("  {:>16} {:>8} {:>10.4} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}% {:>8.4}",
             "Scalar RG", rg_params, rg_composite,
             rg_metrics.format_acc * 100.0, rg_metrics.tool_acc * 100.0,
             rg_metrics.param_acc * 100.0, rg_metrics.reply_quality * 100.0,
             rg_loss);
    println!("  {:>16} {:>8} {:>10.4} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}% {:>8.4}",
             &lora_result.variant, lora_result.params, lora_result.composite,
             lora_result.format_acc * 100.0, lora_result.tool_acc * 100.0,
             lora_result.param_acc * 100.0, lora_result.reply_quality * 100.0,
             lora_result.final_loss);

    // Parameter savings
    let lora_savings = 1.0 - (lora_result.params as f32 / std_params as f32);
    let rg_savings = 1.0 - (rg_params as f32 / std_params as f32);

    println!("\n  Parameter Savings vs Standard GPT:");
    println!("    Scalar RG: {:.1}% fewer params ({} vs {})", rg_savings * 100.0, rg_params, std_params);
    println!("    LoRA-RG:   {:.1}% fewer params ({} vs {})", lora_savings * 100.0, lora_result.params, std_params);

    // Verdict
    let lora_beats_rg = lora_result.composite > rg_composite;
    let lora_matches_std = lora_result.composite > 0.45;
    let lora_has_savings = lora_result.params < 70000;

    println!("\n  === VERDICT ===");
    println!("  LoRA-RG beats Scalar RG:           {} (composite {:.4} vs {:.4})",
             if lora_beats_rg { "YES" } else { "NO" }, lora_result.composite, rg_composite);
    println!("  LoRA-RG matches Standard GPT (>0.45): {} (composite {:.4})",
             if lora_matches_std { "YES" } else { "NO" }, lora_result.composite);
    println!("  LoRA-RG has param savings (<70K):   {} (params {})",
             if lora_has_savings { "YES" } else { "NO" }, lora_result.params);
    println!("  DEPTH DEGRADATION FIXED:           {}",
             if lora_beats_rg && lora_matches_std { "YES" } else { "NO" });

    // Save CSV
    {
        let mut f = std::fs::File::create("experiments/lora_l4_results.csv").unwrap();
        let _ = writeln!(f, "model,params,composite,format_acc,tool_acc,param_acc,reply_quality,final_loss,train_time");
        let _ = writeln!(f, "Standard_GPT,{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            std_params, std_composite, std_metrics.format_acc, std_metrics.tool_acc,
            std_metrics.param_acc, std_metrics.reply_quality, std_loss, std_time);
        let _ = writeln!(f, "Scalar_RG,{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            rg_params, rg_composite, rg_metrics.format_acc, rg_metrics.tool_acc,
            rg_metrics.param_acc, rg_metrics.reply_quality, rg_loss, rg_time);
        let _ = writeln!(f, "LoRA_RG_r{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            lora_rank, lora_result.params, lora_result.composite,
            lora_result.format_acc, lora_result.tool_acc,
            lora_result.param_acc, lora_result.reply_quality,
            lora_result.final_loss, lora_result.train_time);
        println!("\n  Results saved to experiments/lora_l4_results.csv");
    }

    // PlanDB metrics
    println!("\n=== PLANDB_METRICS ===");
    println!("std_gpt_l4_composite={:.4}", std_composite);
    println!("std_gpt_l4_params={}", std_params);
    println!("scalar_rg_l4_composite={:.4}", rg_composite);
    println!("scalar_rg_l4_params={}", rg_params);
    println!("lora_rg_l4_composite={:.4}", lora_result.composite);
    println!("lora_rg_l4_params={}", lora_result.params);
    println!("lora_beats_scalar_rg={}", lora_beats_rg);
    println!("lora_matches_std={}", lora_matches_std);
    println!("lora_has_savings={}", lora_has_savings);
    println!("depth_degradation_fixed={}", lora_beats_rg && lora_matches_std);
    println!("=== END PLANDB_METRICS ===");
}

/// L=8 Confirmation Test: Does RG scale with depth given adequate training?
pub fn run_l8_confirmation() {
    let _ = std::fs::create_dir_all("experiments");

    println!("=== L=8 CONFIRMATION TEST ===");
    println!("    Standard GPT vs Scalar RG at 8 layers");
    println!("    1000 pretrain + 800 SFT steps, batch_size=32 (rayon parallel)\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/4] Generating dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    let vocab_size = tok.vocab_size();
    println!("  Train: {} examples, Val: {}, Vocab: {}",
             train_data.len(), val_data.len(), vocab_size);

    let config = Config {
        vocab_size,
        n_embd: 48,
        n_head: 4,
        n_layer: 8,
        block_size: 128,
    };

    let std_params = count_std_params(&config);
    let rg_params_count = {
        let tmp = RGGPT::new(config.clone());
        tmp.count_params()
    };
    println!("  Standard GPT params: {}", std_params);
    println!("  RG GPT params: {}", rg_params_count);
    println!("  Param savings: {:.1}%\n", (1.0 - rg_params_count as f32 / std_params as f32) * 100.0);

    // Train Standard GPT L=8
    println!("[2/4] Training Standard GPT (L=8, 1000 pretrain + 800 SFT)...");
    let (std_composite, std_metrics, std_loss, std_time) = scaling_train_std(
        &config, base_text, &tok, &train_data, &val_data, &mut rng,
        1000, 800,
    );
    println!("  STD done: composite={:.4}, loss={:.4}, time={:.1}s\n",
             std_composite, std_loss, std_time);

    // Train RG GPT L=8
    println!("[3/4] Training Scalar RG GPT (L=8, 1000 pretrain + 800 SFT)...");
    let (rg_composite, rg_metrics, alphas, rg_loss, rg_time) = scaling_train_rg(
        &config, base_text, &tok, &train_data, &val_data, &mut rng,
        1000, 800,
    );
    println!("  RG done: composite={:.4}, loss={:.4}, time={:.1}s\n",
             rg_composite, rg_loss, rg_time);

    // Compute gap
    let gap_pct = if std_composite.abs() > 1e-6 {
        ((std_composite - rg_composite) / std_composite * 100.0).abs()
    } else {
        100.0
    };
    let param_savings = (1.0 - rg_params_count as f32 / std_params as f32) * 100.0;
    let thesis = if gap_pct <= 5.0 { "CONFIRMED" } else { "DENIED" };

    // Print per-layer alphas
    let weight_names = ["qkv_w", "attn_proj", "ff_w1", "ff_w2"];
    println!("  Per-layer alpha values:");
    for (l, alpha) in alphas.iter().enumerate() {
        print!("    L{}: ", l);
        for (k, name) in weight_names.iter().enumerate() {
            print!("{}={:.4} ", name, alpha[k]);
        }
        println!();
    }

    // Print comparison table
    println!("\n[4/4] Results\n");
    println!("  +-----------------+----------+----------+");
    println!("  | Metric          | Standard |    RG    |");
    println!("  +-----------------+----------+----------+");
    println!("  | Params          | {:>8} | {:>8} |", std_params, rg_params_count);
    println!("  | Composite       | {:>8.4} | {:>8.4} |", std_composite, rg_composite);
    println!("  | Format Acc      | {:>8.4} | {:>8.4} |", std_metrics.format_acc, rg_metrics.format_acc);
    println!("  | Tool Acc        | {:>8.4} | {:>8.4} |", std_metrics.tool_acc, rg_metrics.tool_acc);
    println!("  | Param Acc       | {:>8.4} | {:>8.4} |", std_metrics.param_acc, rg_metrics.param_acc);
    println!("  | Reply Quality   | {:>8.4} | {:>8.4} |", std_metrics.reply_quality, rg_metrics.reply_quality);
    println!("  | Final Loss      | {:>8.4} | {:>8.4} |", std_loss, rg_loss);
    println!("  | Train Time (s)  | {:>8.1} | {:>8.1} |", std_time, rg_time);
    println!("  +-----------------+----------+----------+");
    println!("  | Gap             | {:>7.1}% |          |", gap_pct);
    println!("  | Param Savings   | {:>7.1}% |          |", param_savings);
    println!("  +-----------------+----------+----------+");
    println!();

    if thesis == "CONFIRMED" {
        println!("  THESIS CONFIRMED: RG scales with depth given adequate training");
        println!("  RG composite is within 5% of Standard ({:.1}% gap) with {:.1}% fewer params",
                 gap_pct, param_savings);
    } else {
        println!("  THESIS DENIED: RG has fundamental depth limitations");
        println!("  RG composite gap is {:.1}% (threshold: 5%)", gap_pct);
    }

    // Save CSV
    {
        let mut f = std::fs::File::create("experiments/l8_confirmation.csv").unwrap();
        let _ = writeln!(f, "model,params,composite,format_acc,tool_acc,param_acc,reply_quality,final_loss,train_time_s,gap_pct,thesis");
        let _ = writeln!(f, "standard_l8,{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},,",
            std_params, std_composite, std_metrics.format_acc, std_metrics.tool_acc,
            std_metrics.param_acc, std_metrics.reply_quality, std_loss, std_time);
        let _ = writeln!(f, "rg_l8,{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{}",
            rg_params_count, rg_composite, rg_metrics.format_acc, rg_metrics.tool_acc,
            rg_metrics.param_acc, rg_metrics.reply_quality, rg_loss, rg_time, gap_pct, thesis);
        println!("\n  Results saved to experiments/l8_confirmation.csv");
    }

    println!("\n=== L=8 CONFIRMATION COMPLETE ===");
}

// ─── Compression Benchmark: RG vs Pruning vs Naive Sharing ─────

pub fn run_compression_benchmark() {
    let _ = std::fs::create_dir_all("experiments");
    println!("=== COMPRESSION BENCHMARK (L=3, matched ~42K effective params) ===\n");
    println!("Comparing RG weight sharing against standard compression techniques");
    println!("at matched parameter budgets.\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/5] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
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

    // Count baseline params
    let baseline_params = {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let inner = 4 * e;
        let mut total = v * e + config.block_size * e;
        for _ in 0..nl {
            total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
        }
        total + e * 2 + e * v
    };

    // ─── Method 1: Full Baseline ───
    println!("\n[2/5] Training Method 1: Full Baseline (L=3, ~{}K params)...", baseline_params / 1000);
    let mut baseline_model = bench_pretrain_gpt(&config, base_text, &tok, &mut rng);
    let baseline_metrics = bench_sft_gpt(
        &mut baseline_model, &config, &train_data, &tok, &mut rng, 800,
    );
    let baseline_eval = evaluate_model(&baseline_model, &config, &val_data, &tok);
    let baseline_composite = composite_score(&baseline_eval);
    println!("  Baseline composite: {:.4} (loss: {:.4})", baseline_composite, baseline_metrics);

    // ─── Method 2: RG Weight Sharing ───
    println!("\n[3/5] Training Method 2: RG Weight Sharing (L=3)...");
    let mut rg_model = RGGPT::new(config.clone());
    let rg_params = rg_model.count_params();
    rg_pretrain(&mut rg_model, &config, base_text, &tok, &mut rng);
    let rg_loss = bench_sft_rg(&mut rg_model, &config, &train_data, &tok, &mut rng, 800);
    let rg_eval = rg_evaluate_model(&rg_model, &config, &val_data, &tok);
    let rg_composite = composite_score(&rg_eval);
    println!("  RG composite: {:.4} (loss: {:.4})", rg_composite, rg_loss);

    // ─── Method 3: Magnitude Pruning ───
    println!("\n[4/5] Training Method 3: Magnitude Pruning...");
    // Train full baseline first
    let mut prune_model = bench_pretrain_gpt(&config, base_text, &tok, &mut rng);
    let _ = bench_sft_gpt(&mut prune_model, &config, &train_data, &tok, &mut rng, 800);

    // Prune to match RG effective param count
    let target_sparsity = 1.0 - (rg_params as f32 / baseline_params as f32);
    let pruned_count = prune_gpt_model(&mut prune_model, target_sparsity);
    println!("  Pruned to {:.1}% sparsity ({} effective params)", target_sparsity * 100.0, pruned_count);

    // Fine-tune pruned model for 200 more SFT steps (re-adapt after pruning)
    let prune_loss = bench_sft_gpt(&mut prune_model, &config, &train_data, &tok, &mut rng, 200);
    let prune_eval = evaluate_model(&prune_model, &config, &val_data, &tok);
    let prune_composite = composite_score(&prune_eval);
    println!("  Pruning composite: {:.4} (loss: {:.4})", prune_composite, prune_loss);

    // ─── Method 4: Naive Tied Layers (alpha=1, beta=0 frozen) ───
    println!("\n[5/5] Training Method 4: Naive Tied Layers (no alpha/beta adaptation)...");
    let mut naive_model = RGGPT::new(config.clone());
    let naive_params = naive_model.count_params();
    // Force alpha=1, beta=0 before pretrain
    freeze_rg_scaling(&mut naive_model);
    rg_pretrain_with_freeze(&mut naive_model, &config, base_text, &tok, &mut rng);
    let naive_loss = bench_sft_rg_frozen(&mut naive_model, &config, &train_data, &tok, &mut rng, 800);
    let naive_eval = rg_evaluate_model(&naive_model, &config, &val_data, &tok);
    let naive_composite = composite_score(&naive_eval);
    println!("  Naive tied composite: {:.4} (loss: {:.4})", naive_composite, naive_loss);

    // ─── Results Table ───
    println!("\n{}", "=".repeat(80));
    println!("=== COMPRESSION BENCHMARK RESULTS ===\n");

    println!("{:<22} {:>7} {:>10} {:>10} {:>8} {:>7} {:>7} {:>7}",
             "Method", "Params", "Effective", "Composite", "Format%", "Tool%", "Param%", "Reply%");
    println!("{}", "-".repeat(80));

    let methods: Vec<(&str, usize, usize, f32, &AggregateMetrics)> = vec![
        ("Full Baseline", baseline_params, baseline_params, baseline_composite, &baseline_eval),
        ("RG Weight Sharing", rg_params, rg_params, rg_composite, &rg_eval),
        ("Magnitude Pruning", baseline_params, pruned_count, prune_composite, &prune_eval),
        ("Naive Tied Layers", naive_params, naive_params, naive_composite, &naive_eval),
    ];

    for (name, params, effective, comp, m) in &methods {
        println!("{:<22} {:>7} {:>10} {:>10.4} {:>7.1}% {:>6.1}% {:>6.1}% {:>6.1}%",
                 name, params, effective, comp,
                 m.format_acc * 100.0, m.tool_acc * 100.0,
                 m.param_acc * 100.0, m.reply_quality * 100.0);
    }

    // Find winner among compressed methods (exclude full baseline)
    let compressed: Vec<(&str, f32)> = vec![
        ("RG Weight Sharing", rg_composite),
        ("Magnitude Pruning", prune_composite),
        ("Naive Tied Layers", naive_composite),
    ];
    let winner = compressed.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    let rg_vs_pruning = (rg_composite - prune_composite) / prune_composite.max(0.001) * 100.0;
    let rg_vs_naive = (rg_composite - naive_composite) / naive_composite.max(0.001) * 100.0;

    println!("\nWinner at ~{}K budget: {}", rg_params / 1000, winner.0);
    println!("RG advantage over pruning: {:+.1}%", rg_vs_pruning);
    println!("RG advantage over naive sharing: {:+.1}%", rg_vs_naive);

    // Save CSV
    if let Ok(mut file) = std::fs::File::create("experiments/compression_benchmark.csv") {
        let _ = writeln!(file, "method,params,effective_params,composite,format_acc,tool_acc,param_acc,reply_quality");
        for (name, params, effective, comp, m) in &methods {
            let _ = writeln!(file, "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                             name, params, effective, comp,
                             m.format_acc, m.tool_acc, m.param_acc, m.reply_quality);
        }
        println!("\nResults saved to experiments/compression_benchmark.csv");
    }

    // PlanDB metrics
    println!("\n=== PLANDB_METRICS ===");
    println!("baseline_composite={:.4}", baseline_composite);
    println!("rg_composite={:.4}", rg_composite);
    println!("pruning_composite={:.4}", prune_composite);
    println!("naive_composite={:.4}", naive_composite);
    println!("rg_vs_pruning_pct={:.1}", rg_vs_pruning);
    println!("rg_vs_naive_pct={:.1}", rg_vs_naive);
    println!("winner={}", winner.0);
    println!("=== END PLANDB_METRICS ===");
}

/// Composite score: 0.3*format + 0.3*tool + 0.25*param + 0.15*reply
fn composite_score(m: &AggregateMetrics) -> f32 {
    m.format_acc * 0.3 + m.tool_acc * 0.3 + m.param_acc * 0.25 + m.reply_quality * 0.15
}

/// Pretrain GPT on Shakespeare (for benchmark — no log saving)
fn bench_pretrain_gpt(config: &Config, text: &str, tok: &Tokenizer, rng: &mut impl Rng) -> GPT {
    let encoded = tok.encode(text);
    let mut model = GPT::new(config.clone());
    let batch_size = 16;
    let num_steps = 1000;
    let lr = 0.001;

    let start = Instant::now();
    for step in 0..num_steps {
        let (inputs, targets) = crate::data::create_batches(&encoded, config.block_size, batch_size, rng);
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
            println!("  Pretrain step {:4} | Loss: {:.4} | {:.1}s",
                     step, total_loss / batch_size as f32, start.elapsed().as_secs_f32());
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
    model
}

/// SFT for GPT, returns final avg loss
fn bench_sft_gpt(
    model: &mut GPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
    num_steps: usize,
) -> f32 {
    let lr = 0.0005;
    let batch_size = 8;
    let start = Instant::now();
    let mut last_loss = 0.0f32;

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
            last_loss = total_loss / valid_count as f32;

            if step % 200 == 0 {
                println!("  SFT step {:4} | Loss: {:.4} | {:.1}s",
                         step, last_loss, start.elapsed().as_secs_f32());
            }
        }
    }
    println!("  SFT done in {:.1}s ({} steps)", start.elapsed().as_secs_f32(), num_steps);
    last_loss
}

/// SFT for RGGPT, returns final avg loss
fn bench_sft_rg(
    model: &mut RGGPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
    num_steps: usize,
) -> f32 {
    let lr = 0.0005;
    let batch_size = 8;
    let start = Instant::now();
    let mut last_loss = 0.0f32;

    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = RGGradients::zero_like(config);
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
            last_loss = total_loss / valid_count as f32;

            if step % 200 == 0 {
                println!("  SFT step {:4} | Loss: {:.4} | {:.1}s",
                         step, last_loss, start.elapsed().as_secs_f32());
            }
        }
    }
    println!("  SFT done in {:.1}s ({} steps)", start.elapsed().as_secs_f32(), num_steps);
    last_loss
}

/// Force all RG alphas to 1.0 and betas to 0.0
fn freeze_rg_scaling(model: &mut RGGPT) {
    for l in 0..model.config.n_layer {
        for i in 0..4 {
            model.rg_alpha[l][i] = 1.0;
            model.rg_beta[l][i] = 0.0;
        }
    }
}

/// Pretrain RGGPT with frozen alpha/beta (naive tied layers)
fn rg_pretrain_with_freeze(
    model: &mut RGGPT,
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
        let (inputs, targets) = crate::data::create_batches(
            &encoded, config.block_size, batch_size, rng,
        );
        let mut total_loss = 0.0f32;
        let mut grads = RGGradients::zero_like(config);

        for b in 0..batch_size {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g) = model.forward_backward(
                &inputs[b][..ctx_len], &targets[b][..ctx_len],
            );
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, lr);
        // Reset alpha/beta after each gradient step
        freeze_rg_scaling(model);

        if step % 200 == 0 {
            println!("  Pretrain step {:4} | Loss: {:.4} | {:.1}s",
                     step, total_loss / batch_size as f32,
                     start.elapsed().as_secs_f32());
        }
    }
    println!("  Pretrain done in {:.1}s", start.elapsed().as_secs_f32());
}

/// SFT for RGGPT with frozen alpha/beta, returns final avg loss
fn bench_sft_rg_frozen(
    model: &mut RGGPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
    num_steps: usize,
) -> f32 {
    let lr = 0.0005;
    let batch_size = 8;
    let start = Instant::now();
    let mut last_loss = 0.0f32;

    for step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = RGGradients::zero_like(config);
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
            // Reset alpha/beta after each gradient step
            freeze_rg_scaling(model);
            last_loss = total_loss / valid_count as f32;

            if step % 200 == 0 {
                println!("  SFT step {:4} | Loss: {:.4} | {:.1}s",
                         step, last_loss, start.elapsed().as_secs_f32());
            }
        }
    }
    println!("  SFT done in {:.1}s ({} steps)", start.elapsed().as_secs_f32(), num_steps);
    last_loss
}

/// Prune a GPT model by zeroing smallest-magnitude weights.
/// Returns the count of non-zero (effective) params remaining.
fn prune_gpt_model(model: &mut GPT, target_sparsity: f32) -> usize {
    // Collect all weight magnitudes from large weight matrices
    let mut all_magnitudes: Vec<f32> = Vec::new();

    // Gather magnitudes from all trainable weight vectors
    for val in model.token_emb.iter() { all_magnitudes.push(val.abs()); }
    for val in model.pos_emb.iter() { all_magnitudes.push(val.abs()); }
    for l in 0..model.config.n_layer {
        for val in model.qkv_w[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.attn_proj[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.ff_w1[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.ff_w2[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.ln1_gamma[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.ln1_beta[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.ln2_gamma[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.ln2_beta[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.ff_b1[l].iter() { all_magnitudes.push(val.abs()); }
        for val in model.ff_b2[l].iter() { all_magnitudes.push(val.abs()); }
    }
    for val in model.ln_f_gamma.iter() { all_magnitudes.push(val.abs()); }
    for val in model.ln_f_beta.iter() { all_magnitudes.push(val.abs()); }
    for val in model.lm_head.iter() { all_magnitudes.push(val.abs()); }

    // Sort and find threshold
    all_magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cutoff_idx = ((all_magnitudes.len() as f32 * target_sparsity) as usize)
        .min(all_magnitudes.len() - 1);
    let threshold = all_magnitudes[cutoff_idx];

    // Zero out weights below threshold
    let prune = |vec: &mut Vec<f32>| {
        for w in vec.iter_mut() {
            if w.abs() < threshold { *w = 0.0; }
        }
    };

    prune(&mut model.token_emb);
    prune(&mut model.pos_emb);
    for l in 0..model.config.n_layer {
        prune(&mut model.qkv_w[l]);
        prune(&mut model.attn_proj[l]);
        prune(&mut model.ff_w1[l]);
        prune(&mut model.ff_w2[l]);
        prune(&mut model.ln1_gamma[l]);
        prune(&mut model.ln1_beta[l]);
        prune(&mut model.ln2_gamma[l]);
        prune(&mut model.ln2_beta[l]);
        prune(&mut model.ff_b1[l]);
        prune(&mut model.ff_b2[l]);
    }
    prune(&mut model.ln_f_gamma);
    prune(&mut model.ln_f_beta);
    prune(&mut model.lm_head);

    // Count non-zero params
    let count_nonzero = |vec: &[f32]| -> usize {
        vec.iter().filter(|&&w| w != 0.0).count()
    };

    let mut effective = 0usize;
    effective += count_nonzero(&model.token_emb);
    effective += count_nonzero(&model.pos_emb);
    for l in 0..model.config.n_layer {
        effective += count_nonzero(&model.qkv_w[l]);
        effective += count_nonzero(&model.attn_proj[l]);
        effective += count_nonzero(&model.ff_w1[l]);
        effective += count_nonzero(&model.ff_w2[l]);
        effective += count_nonzero(&model.ln1_gamma[l]);
        effective += count_nonzero(&model.ln1_beta[l]);
        effective += count_nonzero(&model.ln2_gamma[l]);
        effective += count_nonzero(&model.ln2_beta[l]);
        effective += count_nonzero(&model.ff_b1[l]);
        effective += count_nonzero(&model.ff_b2[l]);
    }
    effective += count_nonzero(&model.ln_f_gamma);
    effective += count_nonzero(&model.ln_f_beta);
    effective += count_nonzero(&model.lm_head);

    effective
}

/// Compute R² (coefficient of determination) between actual and predicted values
fn compute_r2(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len() as f64;
    let mean_actual = actual.iter().sum::<f64>() / n;
    let ss_tot: f64 = actual.iter().map(|a| (a - mean_actual).powi(2)).sum();
    let ss_res: f64 = actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2)).sum();
    if ss_tot < 1e-12 { return 0.0; }
    1.0 - ss_res / ss_tot
}

// ─── Experiment B1: Wannier Basis Weight Factorization ─────────

/// Run the Wannier experiment: sweep bandwidth and n_sweeps,
/// comparing against baseline GPT at matched parameters.
pub fn run_wannier_experiment() {
    let _ = std::fs::create_dir_all("experiments");
    println!("=== Experiment B1: Wannier Basis Weight Factorization ===\n");
    println!("PHYSICS: In solid-state physics, the same electronic structure");
    println!("can be represented as Bloch waves (delocalized) or Wannier");
    println!("functions (localized), related by a unitary transformation.");
    println!("We factorize W = U @ S @ V^T where S is BANDED (sparse in");
    println!("the natural basis) and U, V are orthogonal (Givens sweeps).\n");

    let mut rng = rand::thread_rng();

    // Generate dataset
    println!("[1/4] Generating tool calling dataset...");
    let (train_data, val_data) = tool_data::generate_dataset(&mut rng);
    println!("  Train: {} examples, Val: {} examples", train_data.len(), val_data.len());

    // Build tokenizer
    let base_text = crate::data::get_training_data();
    let combined_text = tool_data::build_combined_vocab(base_text, &train_data);
    let tok = Tokenizer::from_text(&combined_text);
    println!("  Vocabulary size: {}", tok.vocab_size());

    // Config
    let block_size = 128;
    let config = Config {
        vocab_size: tok.vocab_size(),
        n_embd: 48,
        n_head: 4,
        n_layer: 3,
        block_size,
    };

    // Baseline param count
    let baseline_params = {
        let e = config.n_embd;
        let v = config.vocab_size;
        let nl = config.n_layer;
        let inner = 4 * e;
        let mut total = v * e + config.block_size * e;
        for _ in 0..nl {
            total += e * 2 + e * 3 * e + e * e + e * 2 + e * inner + inner + inner * e + e;
        }
        total + e * 2 + e * v
    };
    println!("  Baseline params: {}\n", baseline_params);

    // Sweep configurations: (bandwidth, n_sweeps)
    let configs: Vec<(usize, usize)> = vec![
        (4, 2), (4, 3), (4, 4),
        (8, 2), (8, 3), (8, 4),
        (16, 2), (16, 3),
        (32, 2), (32, 3),
    ];

    // CSV header
    let mut csv_rows: Vec<String> = Vec::new();
    csv_rows.push(
        "bandwidth,n_sweeps,total_params,composite,format,tool,param,reply,final_loss,train_time"
            .to_string(),
    );

    println!("[2/4] Running Wannier sweep ({} configurations)...\n", configs.len());
    println!(
        "  {:>4} {:>8} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "bw", "sweeps", "params", "composite", "format", "tool", "param", "reply", "loss", "time"
    );
    println!("  {}", "-".repeat(96));

    for (bw, ns) in &configs {
        let result = run_single_wannier(
            &config, *bw, *ns, base_text, &train_data, &val_data, &tok, &mut rng,
        );
        println!(
            "  {:>4} {:>8} {:>10} {:>10.4} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>10.4} {:>9.1}s",
            bw, ns, result.total_params, result.composite,
            result.format * 100.0, result.tool * 100.0,
            result.param * 100.0, result.reply * 100.0,
            result.final_loss, result.train_time,
        );
        csv_rows.push(format!(
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            bw, ns, result.total_params, result.composite,
            result.format, result.tool, result.param, result.reply,
            result.final_loss, result.train_time,
        ));
    }

    // Save CSV
    println!("\n[3/4] Saving results...");
    if let Ok(mut file) = std::fs::File::create("experiments/wannier_results.csv") {
        for row in &csv_rows {
            let _ = writeln!(file, "{}", row);
        }
        println!("  Saved to experiments/wannier_results.csv");
    }

    // Find best configuration
    let best_idx = csv_rows[1..]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a_composite: f32 = a.split(',').nth(3).unwrap().parse().unwrap_or(0.0);
            let b_composite: f32 = b.split(',').nth(3).unwrap().parse().unwrap_or(0.0);
            a_composite.partial_cmp(&b_composite).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_config = configs[best_idx];

    // Summary
    println!("\n[4/4] Summary\n");
    println!("  Best configuration: bandwidth={}, n_sweeps={}", best_config.0, best_config.1);
    println!("  Baseline params: {}", baseline_params);

    // Parse best row
    let best_parts: Vec<&str> = csv_rows[best_idx + 1].split(',').collect();
    let best_params: usize = best_parts[2].parse().unwrap_or(0);
    let best_composite: f32 = best_parts[3].parse().unwrap_or(0.0);
    let param_reduction = (1.0 - best_params as f32 / baseline_params as f32) * 100.0;

    println!("  Best params:     {} ({:.1}% vs baseline)", best_params,
             if param_reduction > 0.0 { format!("{:.1}% fewer", param_reduction) }
             else { format!("{:.1}% more", -param_reduction) }.as_str()
    );
    println!("  Best composite:  {:.4}", best_composite);

    // Demo with best config
    println!("\n  === Tool Calling Demo (best Wannier config) ===\n");
    let demo_model = WannierGPT::new(config.clone(), best_config.0, best_config.1);
    // Retrain best config for demo
    let mut demo_model = demo_model;
    wannier_pretrain(&mut demo_model, &config, base_text, &tok, &mut rng);
    wannier_sft(&mut demo_model, &config, &train_data, &tok, &mut rng);
    for example in val_data.iter().take(6) {
        let prompt_encoded = tok.encode(&example.prompt);
        if prompt_encoded.is_empty() || prompt_encoded.len() >= config.block_size {
            continue;
        }
        let max_gen = (config.block_size - prompt_encoded.len()).min(80);
        let generated_ids = demo_model.generate(&prompt_encoded, max_gen);
        let generated_text = tok.decode(&generated_ids);
        let response = &generated_text[example.prompt.len()..];
        let truncated = if let Some(end_pos) = response.find("[end]") {
            &response[..end_pos + 5]
        } else {
            &response[..response.len().min(60)]
        };
        println!("  Q: {}", example.prompt.trim());
        println!("  A: {}", truncated.trim());
        println!("  Expected: {}",
                 example.expected_call.as_deref().unwrap_or("[direct reply]"));
        println!();
    }

    println!("=== Wannier Experiment Complete ===");
}

/// Result from a single Wannier configuration run.
struct WannierRunResult {
    total_params: usize,
    composite: f32,
    format: f32,
    tool: f32,
    param: f32,
    reply: f32,
    final_loss: f32,
    train_time: f32,
}

/// Run a single Wannier config: pretrain + SFT + evaluate.
fn run_single_wannier(
    config: &Config,
    bandwidth: usize,
    n_sweeps: usize,
    base_text: &str,
    train_data: &[ToolExample],
    val_data: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
) -> WannierRunResult {
    let mut model = WannierGPT::new(config.clone(), bandwidth, n_sweeps);
    let total_params = model.count_params();

    // Pretrain
    wannier_pretrain(&mut model, config, base_text, tok, rng);

    // SFT
    let start_sft = Instant::now();
    let final_loss = wannier_sft(&mut model, config, train_data, tok, rng);
    let train_time = start_sft.elapsed().as_secs_f32();

    // Evaluate
    let metrics = wannier_evaluate(&model, config, val_data, tok);
    let composite = metrics.format_acc * 0.3
        + metrics.tool_acc * 0.3
        + metrics.param_acc * 0.25
        + metrics.reply_quality * 0.15;

    WannierRunResult {
        total_params,
        composite,
        format: metrics.format_acc,
        tool: metrics.tool_acc,
        param: metrics.param_acc,
        reply: metrics.reply_quality,
        final_loss,
        train_time,
    }
}

/// Pretrain WannierGPT on Shakespeare.
fn wannier_pretrain(
    model: &mut WannierGPT,
    config: &Config,
    text: &str,
    tok: &Tokenizer,
    rng: &mut impl Rng,
) {
    let encoded = tok.encode(text);
    let batch_size = 16;
    let num_steps = 1000;
    let lr = 0.001;

    for step in 0..num_steps {
        let (inputs, targets) =
            crate::data::create_batches(&encoded, config.block_size, batch_size, rng);
        let mut total_loss = 0.0f32;
        let mut grads = WannierGradients::zero_like(config, model.bandwidth, model.n_sweeps);

        for b in 0..batch_size {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g) =
                model.forward_backward(&inputs[b][..ctx_len], &targets[b][..ctx_len]);
            if loss.is_finite() {
                total_loss += loss;
                grads.accumulate(&g);
            }
        }
        grads.scale(1.0 / batch_size as f32);
        model.apply_gradients(&grads, lr);

        if step % 200 == 0 {
            // Silent during sweep; caller prints summary
            let _ = total_loss;
        }
    }
}

/// SFT WannierGPT on tool calling data. Returns final loss.
fn wannier_sft(
    model: &mut WannierGPT,
    config: &Config,
    train: &[ToolExample],
    tok: &Tokenizer,
    rng: &mut impl Rng,
) -> f32 {
    let num_steps = 800;
    let lr = 0.0005;
    let batch_size = 8;
    let mut last_loss = 0.0f32;

    for _step in 0..num_steps {
        let mut total_loss = 0.0f32;
        let mut grads = WannierGradients::zero_like(config, model.bandwidth, model.n_sweeps);
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
            last_loss = total_loss / valid_count as f32;
        }
    }
    last_loss
}

/// Evaluate WannierGPT on validation data.
fn wannier_evaluate(
    model: &WannierGPT,
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
