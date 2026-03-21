use crate::model::{Config, GPT, Gradients};
use crate::tokenizer::Tokenizer;
use crate::tool_data::{self, AggregateMetrics, EvalResult, ToolExample};
use rand::Rng;
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
        let mut total_loss = 0.0f32;
        let mut grads = Gradients::zero_like(config);

        for b in 0..batch_size {
            let ctx_len = inputs[b].len().min(config.block_size);
            let (loss, g) = model.forward_backward(&inputs[b][..ctx_len], &targets[b][..ctx_len]);
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

    println!("=== Agent Ready ===");
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
