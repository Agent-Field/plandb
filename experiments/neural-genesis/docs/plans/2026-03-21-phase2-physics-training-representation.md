# Phase 2: Physics-Inspired Training & Representation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether electronic-structure-theory concepts can replace backpropagation (Thrust A) or fundamentally improve weight representation (Thrust B) in our from-scratch Rust GPT.

**Architecture:** Four independent experiments: A1 (Kohn-Sham SCF training), A2 (perturbation theory training), B1 (Wannier basis weight factorization), B2 (Dyson equation transformer). Each gets its own source file. All compare against the same baseline GPT on tool-calling task.

**Tech Stack:** Rust (no ML frameworks), rayon for parallelism, from-scratch implementations

---

## Experiment A1: Kohn-Sham SCF Training (`src/kohn_sham.rs`)

### Physics Mapping

| DFT Concept | Neural Net Analogue |
|-------------|-------------------|
| N-body Schrödinger equation | Full backpropagation |
| Kohn-Sham single-particle eqns | Per-layer local optimization |
| Electron density ρ(r) | Activation statistics ρ_l = (mean, var) per layer |
| External potential V_ext | Target output / loss signal |
| Hartree potential V_H | Mean-field effect of other layers' activations |
| Exchange-correlation V_xc | Learned correction for inter-layer correlations |
| SCF iteration | Iterate: forward → local update → forward → ... |
| Density mixing / DIIS | Momentum / exponential moving average of updates |

### Algorithm

```
struct KohnShamTrainer {
    model: GPT,                    // standard GPT architecture
    v_xc: VxcFunctional,          // exchange-correlation approximation
    density_history: Vec<Density>, // for DIIS mixing
    mixing_alpha: f32,             // density mixing parameter
}

fn scf_training_step(&mut self, tokens, targets):
    // 1. Forward pass — collect "density" per layer
    let (logits, activations) = self.model.forward_collecting_density(tokens)
    let loss = cross_entropy(logits, targets)

    // 2. Construct effective potential per layer
    for l in 0..n_layer:
        let v_local = reconstruction_loss(activations[l], activations[l+1])
        let v_xc = self.v_xc.compute(all_densities, loss, l)
        let v_eff = v_local + v_xc

        // 3. Local weight update (minimize E_l, NO chain rule through other layers)
        let local_grad = compute_local_gradient(layer_l, v_eff)
        update_layer_weights(l, local_grad)

    // 4. Check SCF convergence (density change < threshold)
    let density_change = compare_densities(old_density, new_density)
    if density_change > threshold:
        mix_densities()  // DIIS-like mixing for stability
        repeat from step 1
```

### V_xc Variants to Test

1. **LDA (Local Density Approximation):** V_xc_l = α · L · σ(ρ_l) — broadcast global loss scaled by layer density. Simplest possible.
2. **GGA (Gradient-corrected):** V_xc_l = MLP(ρ_all_layers, L, ∇ρ_l) — small 2-layer MLP mapping all densities + loss to per-layer correction.
3. **Target Propagation:** V_xc_l = ||h_l - g_l(h_{l+1})|| where g_l is a learned inverse mapping per layer.

### Files
- Create: `src/kohn_sham.rs` (~700 lines)
- Modify: `src/main.rs` (add module + CLI flag `--exp-kohn-sham`)
- Modify: `src/experiments.rs` (add `run_kohn_sham_experiment()`)
- Output: `experiments/kohn_sham_results.csv`

---

## Experiment A2: Perturbation Theory Training (`src/perturbation.rs`)

### Algorithm

```
fn perturbation_training_step(model, tokens, targets, K, sigma):
    let base_loss = forward_only(model, tokens, targets)
    let all_params = model.flatten_params()  // Vec<f32>, ~42K
    let mut grad_estimate = vec![0.0; all_params.len()]

    // Sample K perturbations (embarrassingly parallel with rayon)
    let perturbations: Vec<(Vec<f32>, f32)> = (0..K).into_par_iter()
        .map(|_| {
            let delta = randn_vec(all_params.len(), sigma)
            let perturbed = add_vecs(&all_params, &delta)
            let loss = forward_with_params(&perturbed, tokens, targets)
            (delta, loss)
        })
        .collect();

    // Estimate gradient via correlation
    for (delta, loss) in &perturbations:
        let scale = (loss - base_loss) / (sigma * sigma)
        for i in 0..grad_estimate.len():
            grad_estimate[i] += scale * delta[i] / K as f32

    // Update (standard SGD or Adam on estimated gradient)
    model.apply_update(&grad_estimate, lr)
```

### Antithetic Sampling (variance reduction)

Use paired perturbations (δ, -δ) to halve variance:
```
grad_i ≈ (L(W+δ) - L(W-δ)) / (2σ²) · δ_i
```

### Files
- Create: `src/perturbation.rs` (~450 lines)
- Modify: `src/main.rs` (add module + CLI flag `--exp-perturbation`)
- Modify: `src/experiments.rs` (add `run_perturbation_experiment()`)
- Output: `experiments/perturbation_results.csv`

---

## Experiment B1: Wannier Basis Weights (`src/wannier.rs`)

### Mathematical Formulation

Standard weight: W ∈ R^{m×n} with mn parameters.

Wannier factorization: **W = U @ S @ V^T** where:
- U ∈ O(m): orthogonal rotation, parameterized as product of m(m-1)/2 Givens rotations
- V ∈ O(n): orthogonal rotation, parameterized as product of n(n-1)/2 Givens rotations
- S ∈ R^{m×n}: SPARSE matrix (banded with bandwidth b)

### Givens Rotation Parameterization

Each Givens rotation G(i,j,θ) rotates in the (i,j) plane by angle θ:
```
U = G(0,1,θ_1) @ G(0,2,θ_2) @ ... @ G(m-2,m-1,θ_{m(m-1)/2})
```
This parameterization:
- Always stays on the orthogonal manifold (no projection needed)
- Has m(m-1)/2 parameters for an m×m rotation
- Gradient flows through θ's naturally

### Sparse Core S

S is block-diagonal or banded:
```
For bandwidth b, S has entries only where |row - col| < b
Non-zero entries: ~b × min(m,n) parameters
```

For our 48→192 FF layer: standard = 9,216 params, Wannier with b=8 → ~48×8 + rotations ≈ 1,500 params

### WannierGPT Structure

```
struct WannierGPT {
    // Standard embeddings
    token_emb, pos_emb,

    // Per-layer Wannier-factored weights
    // QKV: (48, 144) → U(48), S(48×144 banded), V(144)
    qkv_u_angles: Vec<Vec<f32>>,    // [n_layer][48*47/2] Givens angles
    qkv_v_angles: Vec<Vec<f32>>,    // [n_layer][144*143/2] Givens angles
    qkv_s_values: Vec<Vec<f32>>,    // [n_layer][bandwidth * min(48,144)]

    // FF weights similarly factored
    ff1_u_angles, ff1_v_angles, ff1_s_values,
    ff2_u_angles, ff2_v_angles, ff2_s_values,

    // Layer norms, biases, head (kept standard)
    ...
}
```

### Bandwidth Sweep
Test b = 4, 8, 16, 32, 48 (full = standard dense) to find the compression/quality frontier.

### Files
- Create: `src/wannier.rs` (~750 lines)
- Modify: `src/main.rs` (add module + CLI flag `--exp-wannier`)
- Modify: `src/experiments.rs` (add `run_wannier_experiment()`)
- Output: `experiments/wannier_results.csv`

---

## Experiment B2: Dyson Equation Transformer (`src/dyson.rs`)

### Mathematical Formulation

Standard residual: x_{l+1} = x_l + f(x_l)

Dyson propagator: **x_{l+1} = G_l · x_l** where **G_l = (I - Σ_l)^{-1}**

Self-energy Σ_l is computed from x_l via attention:
```
Σ_l(x) = Attn(x) + FF(x)  // standard transformer ops as self-energy
```

### Neumann Series (truncated at order K)

```
G = (I - Σ)^{-1} ≈ I + Σ + Σ² + Σ³ + ... + Σ^K

x_{l+1} = x + Σ(x) + Σ(Σ(x)) + Σ(Σ(Σ(x))) + ...
```

Each order applies the SAME self-energy operator again — this is automatic weight sharing across "virtual depth."

### Spectral Radius Control

For convergence, need ||Σ|| < 1. Enforce via:
```
Σ_normalized = Σ / max(1, ||Σ||_spectral * safety_margin)
```
Approximate spectral radius via power iteration (cheap, ~5 iterations).

### DysonGPT Structure

```
struct DysonGPT {
    token_emb, pos_emb,

    // Per-layer self-energy parameters (same as standard transformer layer)
    ln1_gamma, ln1_beta,
    qkv_w, attn_proj,
    ln2_gamma, ln2_beta,
    ff_w1, ff_b1, ff_w2, ff_b2,

    // Dyson-specific
    neumann_order: usize,     // K: how many perturbative orders
    spectral_safety: f32,     // safety margin for convergence

    ln_f_gamma, ln_f_beta, lm_head,
}
```

### Experiment Matrix

| Config | Layers L | Order K | Effective depth | Params |
|--------|----------|---------|-----------------|--------|
| Standard-3L | 3 | - | 3 | ~97K |
| Dyson-1L-K3 | 1 | 3 | ~3 | ~33K |
| Dyson-1L-K5 | 1 | 5 | ~5 | ~33K |
| Dyson-2L-K2 | 2 | 2 | ~4 | ~61K |
| Dyson-2L-K3 | 2 | 3 | ~6 | ~61K |
| Dyson-3L-K2 | 3 | 2 | ~6 | ~97K |

### Files
- Create: `src/dyson.rs` (~700 lines)
- Modify: `src/main.rs` (add module + CLI flag `--exp-dyson`)
- Modify: `src/experiments.rs` (add `run_dyson_experiment()`)
- Output: `experiments/dyson_results.csv`

---

## Comparison Protocol

All experiments use identical:
- Dataset: 720 train / 80 val tool-calling examples
- Tokenizer: 65-char vocabulary
- Evaluation: format_correct, tool_correct, params_correct, reply_quality → composite
- Training: 1000 pretrain steps (Shakespeare) + 800 SFT steps (tool-calling)
- Baseline: StandardGPT L=3 composite ≈ 0.55

### Success Criteria

| Experiment | "Interesting" threshold | "Publishable" threshold |
|-----------|----------------------|----------------------|
| A1 (KS-SCF) | composite > 0.35 (65% of backprop) | composite > 0.45 with proof of convergence |
| A2 (Perturbation) | composite > 0.30 with K ≤ 100 | composite > 0.40 matching ES literature |
| B1 (Wannier) | beats pruning at matched params | beats RG at matched params |
| B2 (Dyson) | L=1,K=3 matches L=3 standard | L=1,K=3 BEATS L=3 standard |

---

## PlanDB Task IDs

| Task | ID | Dependencies |
|------|-----|-------------|
| Phase 2 parent | t-n1xu | - |
| A1: Kohn-Sham impl | t-9pkt | t-n1xu |
| A2: Perturbation impl | t-5evz | t-n1xu |
| B1: Wannier impl | t-ye6x | t-n1xu |
| B2: Dyson impl | t-vcxd | t-n1xu |
| EXP-A1: Run KS | t-xfdy | t-9pkt |
| EXP-A2: Run Perturbation | t-5gf3 | t-5evz |
| EXP-B1: Run Wannier | t-v70d | t-ye6x |
| EXP-B2: Run Dyson | t-t6lo | t-vcxd |
| Comparison | t-osx8 | t-xfdy, t-5gf3, t-v70d, t-t6lo |
