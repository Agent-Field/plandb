# Mini GPT — From-Scratch Transformer + Automated RL Experimentation

A working GPT-style transformer with tool-calling capabilities, built entirely from scratch in **pure Rust** — no ML framework dependencies, just `rand`. Every operation (forward pass, backpropagation, multi-head attention, layer norm, Adam optimizer) is hand-implemented.

**This entire project — architecture design, implementation, debugging, and a 7-method RL experiment suite — was built autonomously by an AI agent using [PlanDB](https://github.com/Agent-Field/plandb) for task orchestration.** No human wrote any Rust code. The agent designed, built, tested, pivoted, and iterated until the model worked.

## Why This Matters

This isn't just a toy model. It demonstrates **automated AI experimentation** — an agent that:

1. Builds its own ML infrastructure from scratch
2. Designs and runs a systematic experiment comparing 7 training methods
3. **Pivots mid-flight** when experiments fail (REINFORCE collapsed → agent added 3 new methods)
4. Parallelizes independent experiments across workers
5. Produces ranked results with quantitative comparison

The task graph evolved from 6 planned tasks to 20 through mid-flight adaptation — the agent discovered complexity, split tasks, added new experiment branches, and ran parallel workers. This is the kind of autonomous research loop PlanDB is designed to orchestrate.

## Results

### Phase 1: Language Model

Trains a character-level GPT on embedded Shakespeare text. After ~2 minutes of training on CPU, generates coherent English:

```
Seed: "the "
the duke of york and the king of france...
```

### Phase 2: Tool Calling + RL Experiments

The agent extended the base model with tool-calling — the model learns when to call a tool vs answer directly, which tool to use, and how to extract parameters:

```
[user] what is 12 plus 7 [end]
[call] calc(12,7,add) [end]
[result] 19 [end]
[reply] the answer is 19 [end]
```

Four tools: `calc` (arithmetic), `search` (lookups), `weather` (forecasts), `time` (current time).

**7 training methods compared head-to-head:**

| Rank | Method | Format Acc | Tool Acc | Param Acc | Composite |
|------|--------|-----------|----------|-----------|-----------|
| 1 | **Rejection Sampling** | 71.3% | 70.0% | 26.3% | 0.601 |
| 2 | SFT Baseline | 66.3% | 63.8% | 27.5% | 0.577 |
| 3 | DPO | 65.0% | 62.5% | 28.8% | 0.570 |
| 4 | SFT v2 (larger model) | 56.3% | 55.0% | 33.8% | 0.531 |
| 5 | Custom RL | 65.0% | 55.0% | 20.0% | 0.505 |
| 6 | SFT v3 (augmented data) | 58.8% | 55.0% | 21.3% | 0.489 |
| 7 | REINFORCE | 0.0% | 0.0% | 0.0% | 0.090 |

### Key Findings

- **Rejection sampling wins.** Simple offline RL (best-of-4 selection with iterative fine-tuning) beat all online RL methods.
- **REINFORCE catastrophically collapsed.** Policy gradient is too unstable for a 91K-parameter model with sparse rewards.
- **DPO maintained parity** with SFT but couldn't improve — needs a stronger reference model.
- **Bigger model hurt format accuracy.** SFT v2 had better parameter extraction but mixed training diluted the format signal.
- **Data augmentation backfired.** More complexity needs more model capacity to absorb.
- **1,205 tokens/sec on CPU.** Inference is fast enough for interactive use.

## Quick Start

```bash
# Train base language model (generates Shakespeare-like text)
cargo run

# Run the optimized tool-calling agent (pretrain + SFT + rejection sampling → demo)
cargo run -- --agent

# Run full RL experiment suite (SFT → REINFORCE → DPO → Custom RL → compare)
cargo run -- --experiments

# Run individual experiments
cargo run -- --sft-v2          # Larger model SFT
cargo run -- --sft-v3          # Data augmentation SFT
cargo run -- --reject-sample   # Rejection sampling (best method)
cargo run -- --compare         # Compare all saved results
```

### Try the Tool-Calling Agent

The fastest way to see the trained model in action:

```bash
cargo run --release -- --agent
```

This runs the full pipeline (pretrain → SFT → 3 rounds of rejection sampling → evaluation → live demo) in ~3 minutes on a laptop. The model trains from scratch each time — weights are deterministic from the embedded training data, so results are reproducible.

The demo shows the agent processing 10 queries, deciding for each whether to call a tool or respond directly:

```
Query: what is 12 plus 7
→ [call] calc(12,7,add)           ← model chose correct tool + params

Query: hello
→ [reply] hello how can i help    ← model chose direct response (no tool needed)

Query: weather in tokyo
→ [call] weather(tokyo)           ← correct tool selection
```

Pre-computed experiment results are in `experiments/` — you can inspect the CSVs without rerunning.

## Architecture

```
3,769 lines of pure Rust. Single dependency: rand.

src/
├── tensor.rs        # Tensor ops (matmul, softmax, GELU, layer norm)
├── tokenizer.rs     # Character-level tokenizer (encode/decode)
├── data.rs          # Shakespeare training data (embedded) + batching
├── model.rs         # GPT: forward, backward, attention, Adam optimizer
├── tool_data.rs     # Synthetic tool-calling dataset generator + eval
├── experiments.rs   # SFT, REINFORCE, DPO, custom RL, rejection sampling
└── main.rs          # Entry point + ASCII loss chart
```

**Model:** 3 layers, 48-dim embeddings, 4 attention heads, ~91K parameters. Block size 48 (base) / 128 (tool calling).

## How PlanDB Orchestrated This

The agent used PlanDB's compound task graph to autonomously plan, execute, and adapt. The project evolved through 6 distinct phases — each one a response to what the agent discovered in the previous phase.

### Phase 1: Build the Foundation (tasks 1–5)

Linear dependency chain. Agent designed the architecture and built components in dependency order:

```
t-tensor → t-model → t-training
t-tokenizer ↗        ↗
t-data ──────────────╯
```

### Phase 2: Debug & Integrate (task 6, split into 2)

`t-debug` was claimed as a single task but turned out to be complex. Agent used `plandb split` to decompose it into `t-vifa` (increase capacity) and `t-k7ek` (polish output) — **mid-flight decomposition**, the compound graph adapting to discovered complexity.

### Phase 3: Tool Calling Dataset + SFT (tasks 10–11)

After the base model worked, the agent pivoted to tool calling. Designed a synthetic dataset format, implemented SFT. This became the **critical bottleneck** — 6 downstream tasks depended on it.

### Phase 4: RL Experiments — Parallel (tasks 12–14)

Three RL methods launched in parallel from the SFT checkpoint:
- `t-rl-reinforce` → `@main`
- `t-rl-dpo` → `@worker-1`
- `t-rl-custom` → `@worker-2`

**This is PlanDB parallelism in action** — independent tasks claimed by different workers, atomic claiming prevents double-assignment.

### Phase 5: Mid-Flight Pivot (tasks 17–19)

REINFORCE collapsed. Instead of giving up, the agent **added 3 new experiment branches** — SFT v2 (larger model), SFT v3 (data augmentation), and rejection sampling. All three depended on the original SFT and fed into the comparison task.

The task graph grew from 6 to 20 tasks through this adaptive process.

### Phase 6: Compare & Ship (tasks 15–16)

All experiments fed into `t-compare` (fan-in from 6 upstream tasks), which ranked methods and selected the best. `t-final` built the optimized agent binary.

### Final Task Graph

```
t-tensor ──────────────────┐
t-tokenizer ───────────────┤
t-data ────────────────────┼──▶ t-model ──▶ t-training ──▶ t-debug (composite)
                           │                                 ├── t-vifa
                           │                                 └── t-k7ek
t-plot ──▶ t-polish        │
                           │
t-tc-design ──▶ t-sft ─────┼──▶ t-rl-reinforce ──┐
                           ├──▶ t-rl-dpo ─────────┤
                           ├──▶ t-rl-custom ──────┼──▶ t-compare ──▶ t-final
                           ├──▶ t-sft-v2 ─────────┤
                           ├──▶ t-sft-v3 ─────────┤
                           └──▶ t-sft-reject ─────┘
```

### PlanDB Features Used

| Feature | How It Was Used |
|---------|----------------|
| **Custom IDs** | `t-tensor`, `t-model`, `t-sft`, etc. — human-readable task names |
| **Dependency chains** | Linear build pipeline + fan-out from SFT to 6 methods |
| **Recursive decomposition** | `t-debug` split into 2 subtasks mid-flight |
| **Composite auto-completion** | Parent `t-debug` auto-completed when children finished |
| **Parallel workers** | DPO and Custom RL ran on `@worker-1` and `@worker-2` |
| **Mid-flight adaptation** | 14 tasks added after initial 6-task plan |
| **Fan-in dependencies** | `t-compare` waited for all 6 experiment branches |

## Experiment Logs

All training runs produce CSV logs in `experiments/`:

| File | Contents |
|------|----------|
| `training_log.csv` | Base model loss curve (step, loss, time) |
| `sft_loss.csv` | SFT training loss per step |
| `sft_v2_loss.csv` | SFT v2 (larger model) loss |
| `sft_v3_loss.csv` | SFT v3 (augmented data) loss |
| `reinforce_log.csv` | REINFORCE rewards per step |
| `dpo_log.csv` | DPO training log |
| `custom_rl_log.csv` | Custom RL rewards |
| `experiment_results.csv` | 4-method comparison |
| `rejection_sampling_results.csv` | Per-round rejection sampling metrics |
| `final_comparison.csv` | All 7 methods ranked |

## License

Apache License 2.0
