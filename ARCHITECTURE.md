# Reduced Differential KirchhoffNet — Architecture & Reference

> **Version:** Idealized (v6)  
> **Design target:** Tapeout-plausible analog compute fabric  
> **Key decisions:** Differential signaling, sparse topology only, cell-library-based edge parameterization (legacy/v15/v2 libraries + simple relu/tanh devices), direct BPTT through Heun integration **or stagewise DEQ equilibrium solver**, three-phase fit-compress-prune schedule (with four-phase readiness-gated variant), STE cell mode, teacher distillation, bidirectional edges, parallel edge repeats, persistent bounded drive, per-dim diagnostics, mapper LR control, DEQ diagnostics suite

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Core Design Decisions](#2-core-design-decisions)
   - 2.20 [Stagewise Deep Equilibrium (DEQ) solver](#220-stagewise-deep-equilibrium-deq-solver-deq-core-prototype)
3. [System Architecture](#3-system-architecture)
4. [Module Reference](#4-module-reference)
5. [Data Flow](#5-data-flow)
6. [Training Pipeline](#6-training-pipeline)
7. [Task Presets](#7-task-presets)
8. [Configuration Reference](#8-configuration-reference)
9. [Sparse Solver Subsystem](#9-sparse-solver-subsystem)
10. [Testing](#10-testing)
11. [File Map](#11-file-map)
12. [Experimental Results](#12-experimental-results)

---

## 1. High-Level Overview

### What this is

A PyTorch implementation of an **analog-inspired neural ODE** based on KirchhoffNet. Each logical node is a **differential pair** of physical voltages (v⁺, v⁻), and the computation happens via **sparse transconductor edges** that mediate current flow between nodes. The network evolves in continuous time, integrated with a fixed-step Heun solver. Training is direct backpropagation-through-time (BPTT).

### Relationship to the original KirchhoffNet paper

The original paper described a graph neural ODE dressed in circuit language. This reimplementation fixes the paper's biggest omissions:

| Original paper | This implementation |
|---|---|
| Branch law = `ReLU(θ₁·Δv + θ₂)` (impossible passive device) | Branch law = library of realizable transconductor surrogates (legacy/v15/v2 libraries, relu/tanh simple devices) |
| Implicitly powered edges (no energy source) | Explicit rail-powered differential transconductor cells + node leak + rail clamps |
| Implied fully connected hardware | Sparse topologies only (line/ring/cluster/grid/empty) |
| Adaptive ODE solver + adjoint | Fixed-step Heun + direct BPTT |
| Single-ended 0-to-VDD encoding | Differential signaling: xⱼ = vⱼ⁺ − vⱼ⁻ |
| No variation/robustness story | SimContext injects PVT + per-edge mismatch at training time |
| Arbitrary topology rewiring between "layers" | Fixed physical stages with simple truncation/padding transfer |
| Per-edge raw device geometry training | Soft library selection over pre-characterized cell families |

### What this is NOT

- **Not a SPICE-accurate circuit simulator.** The edge currents use idealized tanh surrogates with compliance gating, not full transistor models.
- **Not a tapeout-ready netlist.** It omits common-mode dynamics, non-diagonal capacitance, sample-and-hold circuitry, and post-layout parasitics.
- **Not the final signoff model.** It is a training scaffold for architecture search and task feasibility experiments.

---

## 2. Core Design Decisions

### 2.1 Differential signaling

Each logical node `j` is represented by two voltages:

```
xⱼ = vⱼ⁺ − vⱼ⁻        (differential signal — the useful state)
cⱼ = (vⱼ⁺ + vⱼ⁻) / 2   (common-mode — assumed held by feedback)
```

The reduced model assumes strong common-mode feedback so we only simulate `xⱼ`. This gives natural signed representation without awkward "negative in 0-to-VDD" hacks.

**Units (R7).** The conductance, current, and capacitance values in `config.py` are NORMALIZED to plausible analog ranges, not calibrated to SI. `x_max` was raised to 3.0 to give the solver more dynamic range and avoid saturation. `C_eff` is a pure scaling. No pretense is made that `g/C` is a real analog time constant. `V_CM` has been removed from `PHYS` since it is not simulated. `t_span=5.0` and `num_steps=50` are the defaults.

### 2.2 Sparse topologies only

No fully connected layers. Supported graph primitives:
- `line_graph` — 1D chain with neighborhood radius
- `ring_graph` — 1D periodic chain
- `grid_graph` — 2D local grid with kernel neighborhood
- `cluster_graph` — Erdős-Rényi random sparse graph
- `empty_graph` — no edges (for ablation or projection-only stages)

All primitives support `bidirectional=True` which emits two directed edges per undirected pair (i→j and j→i), giving asymmetric cell types (P/rectifier) true bidirectional capability. Edge count is exactly 2× the single-direction count.

`repeat_edges(topo, n)` duplicates `EDGE_TYPE_HIDDEN` edges n times (parallel branches). Each repeated edge gets independent logits/gate/multiplier in `DifferentialStage`. Non-hidden edges (input, proj, output) are NOT repeated.

### 2.3 Cell library instead of raw device parameters

Edges don't learn raw W/L or bias voltages. They soft-select from one of several pre-characterized cell families.

The codebase supports 4 cell formula types dispatched via `cell_type_code`:

| Formula | Code | I(u) expression |
|---------|------|-----------------|
| `standard` | 0 | `I_sat · tanh((gm·u + bias) / I_sat) + g_leak · u` |
| `pos_rect` | 1 | `I_sat · tanh(gm · softplus((u−θ)/β) / I_sat)` |
| `neg_rect` | 2 | `-I_sat · tanh(gm · softplus((−u−θ)/β) / I_sat)` |
| `dead_zone` | 3 | `pos_rect(u) − neg_rect(u)` |

The preactivation `u` is computed as `x_src − ρ · x_dst` (legacy/v15 libraries) or `src_gain · x_src − dst_gain · x_dst` (v2 library with per-cell mix coefficients).

Three named cell libraries are defined, plus two simple device modes:

**Legacy library** (4 cells): L (weak linear, gm=0.2), S (saturating, gm=1.0), P (smooth rectifier, gm=1.0, β=0.1), Z (disabled). Backward-compatible globals `CELL_LIBRARY`, `CELL_ORDER`, `NUM_CELLS`, `Z_INDEX` reflect this library.

**v1.5 library** (6 cells): O_weak (gm=0.3, isat=5.0), O_hard (gm=3.0, isat=0.3), P0 (pos_rect, gm=1.0, β=0.1), N0 (neg_rect, mirror of P0), D1 (dead_zone, θ=0.5, β=0.1), Z.

**v2 library** (10 cells, mix-code/bias-code bounded): Built from factorized MIX/BIAS/THRESH codes. Mix codes (M11/M10/M01) set per-cell `src_gain/dst_gain` replacing rho. Bias codes (Bsoft/Bmid/Bhard) set discrete (gm, isat) pairs. Threshold codes (T0/T1) set preactivation offsets. Cells: O_w11, O_h11, O_h10, O_h01, P0, P1, N0, N1, D1, Z. All v2 cells have gleak=0. P/N beta=0.08, D beta=0.10.

**Simple device libraries** (`relu`/`tanh`): Each edge holds 3 learnable parameters (p0, p1, p2) and computes `I = ReLU(p0·Vsrc + p1·Vdest + p2)` or `I = tanh(...)`. No cell selection, no multiplicity, no Z cell.

The `CELL_LIBRARIES` dict maps library name to config. `cells_to_tensor_dict(library_name)` stacks the named library into tensors. `make_cell_library(library_name)` returns an `IdealizedCellLibrary` for bounded macro libraries or a `SimpleEdgeLibrary` for simple devices.

Per-edge learnable parameters:
- **logits** `[E, Q]` — soft library selection (log-probabilities). Not used by `SimpleEdgeLibrary`.
- **raw_mult** `[E]` — edge multiplicity `m = softplus(raw_mult)` (can approach 0). Not used by `SimpleEdgeLibrary`.
- **raw_leak** `[N]` — per-node weak stabilization leak `g = softplus(raw_leak)`

**Cell mode** (`cell_mode`): `'soft'` (default) uses a softmax mixture of all cells per edge. `'ste'` (straight-through estimator) uses one-hot argmax in the forward pass with soft gradients in the backward pass — used for hard cell commitment during four-phase schedule B/C phases.

**Honest I/O split (R1).** The InputMapper writes ONLY to the hidden-node portion of the differential state vector. Projection nodes are zero-initialized and remain so until the ODE moves them. The OutputMapper reads ONLY from the projection-node portion; if no projection nodes exist (legacy/ablation), it falls back to hidden positions with a warning. This forces the ODE core to do the work.

### 2.4 Fixed-step Heun integration (not adjoint)

The ODE `C · dx/dt = Σ I_in − Σ I_out − leak · x − clip(x)` is integrated with Heun's method (predictor-corrector, 2nd order):

```
k1 = rhs(x)
x_pred = x + dt · k1
k2 = rhs(x_pred)
x_next = x + (dt/2) · (k1 + k2)
```

Gradients flow directly through the unrolled steps (BPTT). No continuous adjoint. This is more reliable for stiff, physically constrained systems.

### 2.5 Write/Evolve/Read pipeline

The network operates in three phases:
1. **Write:** `InputMapper` (dense), `SparseInputMapper` (one-to-one), or `FanOutInputMapper` (multi-target) maps raw input `u` to bounded initial differential state `x_hidden(0)`. Hidden nodes NOT in `write_idx` are zero-initialized. Projection portion `x_proj(0) = 0`.
2. **Evolve:** Multi-stage ODE core integrates for fixed time horizons. The core evolves hidden + projection nodes together. Persistent drive current may be injected per-stage when enabled.
3. **Read:** `OutputMapper` linearly projects either the full state slice (sparse `read_idx`) or the projection portion (dense) to output `ŷ`.

**Sparse I/O mapping (sparse-io-mapping spec).** Each preset specifies `write_idx` and `read_idx` lists, and the default write/read modes are:
- `--write-mode one_to_one` (default): each input feature `u_i` writes to exactly one hidden node `h_{write_idx[i]}` via `SparseInputMapper`. Hidden nodes not in `write_idx` are zero-initialized at t=0. Parameter count: `2 * in_dim` (vs `in_dim * hid_count + hid_count` for dense `InputMapper`).
- **Fan-out write** (not a CLI `--write-mode` choice; set per-preset via `write_fan_out`): each input feature writes to K>1 hidden nodes via `FanOutInputMapper` with per-target (gain, bias) pairs.
- `--read-mode sparse` (default): `OutputMapper` gathers only from `read_idx` full-state indices (a learnable linear of size `len(read_idx) -> out_dim`).
- `--write-mode dense` / `--read-mode dense`: original `InputMapper` / `OutputMapper` behavior for baseline comparison.

Preset defaults:
- `sinx`: `write_mode=one_to_one`, `write_idx=[0]`, `read_mode=sparse`, `read_idx=[7]` (out of 8 hidden + 2 proj = 10)
- `housing`: `write_mode=one_to_one`, `write_idx=[0..7]`, `read_mode=sparse`, `read_idx=[15]` (out of 16 hidden + 4 proj = 20)
- `smooth2d_grid`: `write_mode=fan_out`, `write_fan_out` = auto-computed left/right column indices (depends on grid_size), `read_mode=sparse`, `read_idx` = center column + proj nodes
- `housing_grid`: `write_mode=dense` (all-to-all, 8 housing features have no spatial structure), `read_idx` = center column + proj nodes

CLI flags: `--write-mode {one_to_one,dense}`, `--read-mode {sparse,dense}`, `--write-idx "0,2,4"`, `--read-idx "7"`. The index overrides take precedence over preset values. Fan-out write is only available through the `write_fan_out` preset config (not as a CLI `--write-mode` choice).

### 2.6 Staged regularizer warm-up (RR-A)

By default, the auxiliary regularizers (sparsity, edge_gate, power) are **ramped in** over the first 150 epochs so the network can learn the task first without penalty fighting (node_gate and capacitance regularizers are deprecated — always 0.0):
- **Epochs 0–99 (`W=100`)**: all are multiplied by 0.0 (free phase).
- **Epochs 100–149 (`W+A=150`)**: linear anneal from 0.0 → 1.0.
- **Epochs 150+**: full penalty value.

The schedule is controlled by `reg_schedule(epoch)` in `train.py` and configurable via `OPTIM["reg_warmup_epochs"]` and `OPTIM["reg_anneal_epochs"]` in `config.py`. **Note (fix-z-death)**: `rail` is intentionally NOT in `_REG_KEYS`; it is a safety voltage clamp on differential node states and is applied at full strength at every epoch. The rail loss uses a **ReLU² quadratic barrier** (`F.relu(|x| - x_max).pow(2).mean()`) which has zero loss and zero gradient inside `[-x_max, x_max]`. The entropy bonus retains its own τ-dependent scaling.

### 2.7 Per-component complexity regularizers (complexity-pruning-v2)

The old single `complexity` key is **decomposed into 4 physically motivated terms** for finer-grained control over the pruning process:

| Regularizer | What it penalizes | Default weight |
|-------------|-------------------|----------------|
| `edge_gate` | `Σ σ(z_e)` — open edge count | 5e-4 |
| `node_gate` | `Σ σ(u_j)` — open node count **(deprecated, always 0.0)** | 0.0 |
| `power` | `Σ_e σ(z_e) · m_e · Σ_q p(L|e,q) · gm_q` — static power proxy | 1e-4 |
| `capacitance` | `Σ_j σ(u_j)` — node capacitance area proxy **(deprecated, always 0.0)** | 0.0 |
| `sparsity` | `Σ w[:, :z_idx]` — active non-Z cell mass | 1e-3 |
| `rail` (ReLU²) | `mean(ReLU²(|x| - x_max))` over trajectory | 0.1 |
| `entropy` | `−Σ w·log(w)` of logits/tau | 1e-4·τ |

In addition, **learnable gate parameters** are added to every `DifferentialStage`:
- `z_logits [E]` — edge open-logit (init 0.0 → `σ(0) ≈ 0.50`). Each `i_edge` is multiplied by `σ(z_logits[e])` in `rhs()`, so a closed edge contributes zero current.
- `u_logits [N]` — node open-logit **(deprecated, bypassed in `rhs()`)**. No longer applied in forward pass — node mask is always 1.0. Parameter kept for checkpoint compatibility only.

The gate init is 0.0 to place gates at the **maximum-gradient** point of the sigmoid (`dσ/dz = 0.25` at `z=0`).

Helper methods on `DifferentialStage`:
- `edge_gates()` — returns `σ(z_logits)`
- `node_gates()` — returns all-ones tensor (deprecated, always identity)
- `active_edge_mask(threshold=0.01)` — edge gates above threshold
- `active_node_mask(threshold=0.01)` — all-True (deprecated)
- `parameter_breakdown()` — returns dict with gate values

The gate regularizers (edge_gate, power) go through the same staged warm-up as sparsity. node_gate and capacitance are deprecated (hard-coded to 0.0).

### 2.8 Complexity pruning pipeline

After training with gate parameters, the network can be structurally pruned to remove low-value edges.

**Joint Z+gate criterion.** An edge is kept if its effective activity score exceeds a threshold:

```
eff_score(e) = (1 - P(Z|e)) · σ(z_e) > edge_threshold
```

This folds both the Z-cell probability (gm_Z ≈ 0 ⇒ no current) and the edge gate into a single criterion. The default threshold is 0.1.

**Connectivity backstop.** After gate-based pruning, a BFS from `write_idx` verifies that all `read_idx` are reachable. Dead islands (nodes not in any I/O-connected component) are purged. If fewer than `min_read_nodes` survive, pruning raises `ValueError`.

**Protected nodes.** Write targets (input-side guard) are forced to survive pruning regardless of gate value. Driven nodes (persistent drive) are also protected. Read nodes are NOT protected (elastic readout is allowed).

**Edge-only mode (default).** Node gates are deprecated and bypassed in `rhs()`. Pruning is exclusively connectivity-based — `prune_nodes_by_gate` defaults to `False`. Nodes are only removed via the connectivity backstop (dead island purge).

**I/O mapper transfer.** After pruning, the InputMapper and OutputMapper are rebuilt with weights transferred from the pre-prune network. `SparseInputMapper` weights are copied directly (indexed by input). `FanOutInputMapper` targets are remapped through the stage node remap. `InputMapper`/`RobustInputMapper` rows are selectively copied for surviving nodes.

**SimpleEdgeLibrary handling.** Stages using `SimpleEdgeLibrary` have no Z cell — the effective score is just `σ(z_logits)`.

### 2.9 Degree budget / top-k competition (degree-budget-topk plan)

The independent per-edge sigmoid gate (`σ(z_logits)`) cannot break symmetry among redundant edges — 11 edges to the same destination can all sit at gate=0.5 forever. The degree budget adds **explicit competition** via temperature-scaled softmax renormalization:

```
budget_gate[j] = softmax(z_logits[incoming_to_j] / T) * k
edge_gate = σ(z_logits) * budget_gate   (layered: independent × competitive)
```

Each destination node j keeps at most k incoming edges open. Temperature T and budget k are **annealed** from permissive (high k, high T) to restrictive (low k, low T) over the first `anneal_frac` of training. Three axis modes:

| `--budget-axis` | Competition | Budget per |
|----------------|-------------|------------|
| `dst` (default) | Incoming edges per destination | Destination |
| `src` | Outgoing edges per source | Source |
| `both` | Both directions (product of masks) | Source & Destination |

The budget is fully differentiable (C∞ softmax) → compatible with DEQ implicit differentiation. Config via `config.DEGREE_BUDGET`, CLI via `--budget` and related flags.

Key functions in `topology.py`:
- `prune_stage(stage, edge_threshold, ...)` — returns `(new_stage, node_remap)`. Handles persistent drive, SimpleEdgeLibrary.
- `prune_network(core, ...)` — applies `prune_stage` to every stage, reinitializes `StageTransfer` modules, protects driven nodes, returns `(new_core, stage_remaps)`

The `train_script.py` CLI exposes:
- `--prune` — enable pruning after training
- `--budget` — enable degree budget top-k competition
- `--budget-k-start`, `--budget-k-end` — budget annealing range
- `--budget-temp-start`, `--budget-temp-end` — temperature annealing range
- `--budget-axis` — competition axis (dst / src / both)
- `--retrain` / `--no-retrain` — retrain pruned network (default: retrain)
- `--prune-edge-threshold` — override edge gate threshold
- `--prune-node-threshold` — override node gate threshold
- `--prune-nodes-by-gate` / `--no-prune-nodes-by-gate` — deprecated no-op
- `--retrain-epochs` — retraining epochs
- `--fresh-init` — skip warm-start, reinitialize pruned network from scratch

### 2.9 Three-phase training schedule (three-phase-schedule plan)

A generic phased training pipeline that any preset can opt into via `preset["schedule"] = "three_phase"`. Splits the total epoch budget into three phases with independently configured tau annealing and structural regularizer magnitudes:

| Phase | Epochs | Tau | Regularizers | Action |
|-------|--------|-----|-------------|--------|
| **A** (fit) | 0–30% | Fixed 0.8 | All zero — free fit | Network learns task without structure pressure |
| **B** (compress) | 30–70% | 0.8→0.5 anneal | Gate penalties ramped in | Gate logits pushed toward 0 (closed) or stay open; node_gate deprecated (0.0) |
| **C** (retrain) | 70–100% | 0.5→0.1 anneal | Only sparsity (1e-5) + rail | Auto-prune at B→C boundary, retrain compact network |

**Phase A (fit, epochs 0–30%).** All structural regularizers are zeroed. Tau stays at 0.8 (no hardening pressure). The network learns the task freely. Rail is always active as a safety net.

**Phase B (compress, epochs 30–70%).** Tau anneals from 0.8→0.5 (gentle specialization). Structural regularizers are ramped from 0 to full over the first 50% of Phase B: sparsity=5e-5, edge_gate=1e-5, power=1e-5. node_gate and capacitance are 0.0 (deprecated).

**Phase C (retrain, epochs 70–100%).** At the B→C boundary, automatic pruning removes edges below the schedule's thresholds (edge_threshold=0.05, prune_nodes_by_gate=False — edge-only pruning). The compact network is retrained from warm-start with aggressive tau annealing (0.5→0.1) and only sparsity (1e-5) + rail (unchanged). Gate penalties are off (irrelevant post-prune).

**Solidification diagnostics.** During Phases A and B, per-epoch metrics are logged to `solidification.tsv`:
- `mean_max_cell_prob`: mean over all edges of max(softmax(logits/τ)).
- `mean_pZ`: probability mass on the Z cell.
- `mean_sigma_z/u`: average edge/node gate openness.
- `frac_sigma_z_below_0.1/0.05/0.01`: fraction of edges eligible for pruning.

**Argmax validation.** At each validation epoch, the network is evaluated with τ→0.001 (effectively argmax cell selection) and the task loss is compared against the soft-τ baseline. A small gap means cell selection is solidified.

The `--schedule {legacy,three_phase,four_phase}` CLI flag selects the mode. `smooth2d_grid` defaults to `three_phase` via its preset config.

### 2.10 Per-preset lambda overrides (RR-D)

Each preset may optionally contain a `"lambdas"` dict that is merged on top of the global `LAMBDAS`. This allows per-task tuning of a single regularizer weight without redefining the entire dictionary.

Active overrides:
- **sinx**: `{"rail": 1.0}`.
- **smooth2d_grid**: `{"sparsity": 1e-5, "edge_gate": 5e-6, "node_gate": 0.0, "power": 1e-5, "capacitance": 0.0, "rail": 0.1}`.
- **housing_grid**: same as smooth2d_grid (identical lambda override).
- **housing**: no override.

### 2.11 Deprecated `temp_c` sampling (RR-C)

The `temp_c` field of `SimContext` is preserved for API compatibility but **no longer used** by the analog model. `sample_random_context` always returns `VARIATION["temp_c_default"]` (27.0°C) and ignores the `temp_choices` argument. External callers that explicitly pass a non-default `temp_c` value receive a `DeprecationWarning`. The `legacy_temp=True` flag exists temporarily for code migration and also emits a warning.

### 2.12 Variation-aware training (R6.3)

By DEFAULT, training uses a `SimContext()` with no mismatch and no temperature drift, so the optimization sees a clean deterministic forward. To evaluate robustness, pass `--variation` to `train_script.py`, which then samples a fresh `SimContext` per training iteration:
- `temp_c` — junction temperature (deprecated, always defaults to 27°C)
- `global_gain_shift` — log-normal global gm drift (σ = 5%)
- `edge_mismatch` — per-edge per-cell log-normal mismatch (σ = 5%)

Mismatch is held fixed over the full transient but resampled each iteration. At validation time, `edge_mismatch=None` (nominal).

### 2.13 Training infrastructure

**AMP (mixed precision).** Enabled by default when CUDA is available (`--amp` / `--no-amp`). Uses `torch.cuda.amp.GradScaler` with autocast on forward+loss. The loss is split into task+rail (data-dependent) and structural (parameter-only) components so DataParallel averages only the data-dependent part. `--amp-dtype {float16,bfloat16}` selects the autocast dtype.

**torch.compile.** Enabled by default when CUDA is available (`--compile` / `--no-compile`). Uses `torch.compile` on `cell_library.forward` and `stage.rhs` for kernel fusion (~1.3–2× throughput on T4 Tensor Cores). Disabled when DataParallel is active or when compile setup fails.

**DataParallel.** Automatically enabled when ≥2 GPUs are detected (`--parallel` / `--no-parallel`). The regularizer computation is monkey-patched to unwrap DataParallel before accessing per-stage parameters.

**Mapper LR control (mapper LR scale).** `--mapper-lr-scale` (float, default 1.0) creates a separate AdamW param group for I/O mappers at reduced LR in `make_optimizer()`. When `<1.0`, mappers learn more slowly — useful when mapper gradient norms dominate core by ~300×. `--freeze-mappers` freezes I/O mapper parameters at the B1 start and unfreezes at the midpoint (four_phase only). Retrain has separate `--retrain-mapper-lr-scale` flag.

**Per-stage LR scaling (stage-lr-scaling).** Multi-stage networks suffer from vanishing gradients in early stages. `make_optimizer()` can create separate param groups with geometrically increasing LRs: `stage i LR = base_lr × scale^(S−1−i)`. Controlled via `--stage-lr-scale` (default 1.0, disabled). Retrain has a separate `--retrain-stage-lr-scale` (default 1.0) to avoid over-aggressive warm-start updates.

**Gradient logging.** `--grad-log` enables periodic per-parameter-group L2 gradient norms written to `grad_norms.txt`. Each row shows per-stage (logits, raw_mult, raw_leak, z_logits, u_logits, device_param) plus mapper and transfer norms.

**Batch-size-aware LR auto-scaling.** The learning rate is computed as `lr = BASE_LR × (batch_size / BASE_BATCH_SIZE)` (Goyal et al., 2017) where `BASE_LR=3e-4` and `BASE_BATCH_SIZE=1024`. At `batch_size=1024`, this gives `lr=3e-4`.

**entropy bonus.** When `LAMBDAS["entropy"] > 0`, an entropy bonus `−Σ w·log(w)` of the softmax distribution over cell types (scaled by τ) is subtracted from the structural loss, encouraging exploration of cell mixtures.

### 2.14 Four-phase training schedule (four-phase-redesign plan)

A four-phase extension of the three-phase schedule that splits Phase B into B1 (cell commitment, no pruning) and B2 (edge pruning, readiness-gated). Adds teacher distillation and straight-through estimator (STE) cell mode for cleaner hardening.

**Availability and activation.** Any preset can opt into the four-phase schedule via `--schedule four_phase` on the CLI. No preset defaults to four_phase — it is always an explicit override. The CTLE distillation script (`train_ctle.py`) always uses four_phase.

**Config entry (`SCHEDULE_FOUR_PHASE` in `config.py`).** Phase fractions default to frac_a=0.3, frac_b1=0.2, frac_b2=0.2, frac_c=0.3 (must sum to 1.0). Tau targets: tau_a=0.6 (fixed), tau_b1_init=0.6→tau_b1_final=0.5, tau_b2_init=0.5→tau_b2_final=0.4, tau_c_init=0.4→tau_c_final=0.1. Tau is continuous at all phase boundaries. Teacher KD weight: lambda_kd=1.0. Warmup within B1/B2: 25% of each phase's epoch window.

| Phase | Epochs | Tau | Regularizers | Cell Mode | Action |
|-------|--------|-----|-------------|-----------|--------|
| **A** (fit) | 0–30% | Fixed 0.6 | All zero | Soft | Learn task freely |
| **B1** (cell commit) | 30–50% | 0.6→0.5 | sparsity=5e-5, power=1e-4 | STE | Commit cells, no pruning |
| **B2** (edge prune) | 50–70% | 0.5→0.4 | sparsity=5e-5, edge_gate=1e-5, power=1e-5 | STE | Prune edges readiness-gated |
| **C** (retrain) | 70–100% | 0.4→0.1 | sparsity=1e-5 only | STE | Retrain compact model |

---

**Phase A (fit, epochs 0–30%).** Identical to three-phase Phase A: all structural regularizers zeroed, tau=0.6, standard soft cell mixture. The network learns the task freely with no structure pressure. Rail is always active as a safety net.

**Teacher cloning at A→B1 boundary.** At the first epoch of Phase B1, a deep copy of the best Phase A checkpoint (lowest soft validation loss) is frozen as the teacher network. The teacher is kept in eval mode with `requires_grad_(False)` and stays on the device for the remainder of B1+B2. The teacher uses soft cell mode with tau=1.0 — providing a smooth, well-behaved regression target — while the student transitions to STE mode. If training is early-stopped during Phase A (before teacher cloning fires), the four-phase schedule skips pruning and retrain entirely, falling back to a standard save of the best Phase A model.

---

**Phase B1 (cell commitment, epochs 30–50%).** Tau anneals from 0.6→0.5. Only the sparsity regularizer (5e-5) and power (1e-4) are active — no edge_gate/node_gate/capacitance. Cell mode switches to **STE** (straight-through estimator). Teacher distillation (`lambda_kd * MSE(y_student, y_teacher)`) is active throughout B1.

---

**Phase B2 (edge pruning, epochs 50–70%).** Tau continues from 0.5→0.4. The edge_gate regularizer (1e-5) is added to sparsity and power. This is the first time gates receive gradient pressure. STE cell mode and teacher distillation continue from B1.

**Readiness-gated prune.** Pruning is NOT automatic at the B2→C boundary. Instead, on every validation epoch during B2, a readiness check evaluates four AND-logic conditions over a trailing window (default 10 epochs):

1. **Ratio**: `mean(val_argmax / val_soft) < 1.2` over the window.
2. **Cell probability**: `mean_max_cell_prob > 0.85`.
3. **Gate stability**: `std(frac_sigma_z_below_0.1) < 0.02` over the window.
4. **Improvement**: `|val_argmax improvement rate| < 1e-4`.

All four conditions must be met for the readiness check to return True. On readiness trigger, B2 ends immediately and Phase C begins at the current epoch with the freshly pruned network.

**Checkpoint selection during B1/B2.** In B1 and B2, the best model is tracked by `val_argmax` (hard-cell validation loss) rather than soft validation loss.

---

**Phase C (retrain, epochs 70–100%).** At the B2→C boundary (whether readiness-triggered or fallback), auto-pruning removes edges below `prune_edge_threshold=0.05`. Pruning is edge-only (`prune_nodes_by_gate=False`). The compact network is retrained from warm-start with tau annealing 0.4→0.1, only sparsity (1e-5) + rail, STE cell mode continues. Teacher distillation is OFF in Phase C.

---

**Teacher distillation details.** The KD loss is `lambda_kd * MSE(y_student, y_teacher)` where `lambda_kd = 1.0`. The teacher forward pass runs under `torch.no_grad()` with `soft` cell mode and `tau=1.0`. The KD loss is a data-side term (DataParallel-averaged) alongside task loss and rail. KD is active only during B1 and B2.

---

**Cell mode auto-resolution.** `--cell-mode {soft, ste, auto}`. With `--cell-mode auto` (default), the schedule resolves per-epoch: Phase A → `soft`, Phases B1/B2/C → `ste`.

---

**Diagnostic ablation sets.** `--ablation-set`:
- `reg-only`: tau=1.0 throughout B, gate regularizers left at default, pruning disabled.
- `tau-only`: structural regularizers zeroed in B, tau anneals normally, pruning disabled.
- `edge-only`: matches the four-phase defaults (edge-only pruning, edge_threshold=0.05, no node-gate pruning).

---

**Key differences from three-phase:**
| Dimension | Three-phase | Four-phase |
|-----------|-------------|------------|
| Phases | A (fit) → B (compress) → C (retrain) | A (fit) → B1 (commit) → B2 (prune) → C (retrain) |
| Tau | 0.8 → 0.5 → 0.1 | 0.6 → 0.5 → 0.4 → 0.1 |
| Prune trigger | Automatic at B→C boundary | Readiness-gated (4 AND-condition checks) |
| Cell mode | Soft throughout | STE in B1/B2/C |
| Teacher distillation | None | `lambda_kd=1.0` in B1/B2 |
| B1 regularizers | (merged with B) | sparsity=5e-5, power=1e-4 |
| B2 regularizers | (merged with B) | sparsity=5e-5 + edge_gate=1e-5 + power=1e-5 |

### 2.15 Persistent bounded drive (persistent-drive plan)

Each stage can receive a bounded drive current pulling driven hidden nodes toward an input-derived target pattern. Controlled via `--persistent-drive` CLI flag (requires `write_mode='fan_out'`). Per-stage `FanOutInputMapper` produces drive targets from input. Drive current is computed as:

```
I_drive = I_sat · tanh(g · (x_drive − x) / I_sat)
```

where `g = softplus(raw_drive_g)` is a learnable per-node conductance. Drive scales decay across stages `[1.0, 0.5, 0.25]` (configurable via `DRIVE["drive_scales"]`). Driven nodes are forced open during pruning (protected from gate-based removal). Driven node gates are forced open (masked to 1.0) in `rhs()`.

### 2.16 Parallel edge repeats

`repeat_edges(topo, n)` duplicates every `EDGE_TYPE_HIDDEN` edge n times in `topology.py`. Each repeated edge gets independent logits/gate/multiplier in `DifferentialStage`. Their currents sum naturally in KCL via scatter-add (physically: parallel branches). Controlled via `--edge-repeats N` (default 2, range 1-8 in `train_script.py`). Composes multiplicatively with `--bidirectional`. Non-hidden edges (input, proj, output) are NOT repeated.

### 2.17 Bidirectional edges

All graph primitives (`line_graph`, `ring_graph`, `grid_graph`, `cluster_graph`) accept `bidirectional=True`, emitting two directed edges per unique undirected pair (i→j and j→i). Edge count is exactly 2× the single-direction count. Gives asymmetric cell types (P/rectifier) true bidirectional capability. Controlled via `--bidirectional` CLI flag. Composes multiplicatively with `--edge-repeats`.

### 2.18 Dynamic topology overrides

`train_script.py` supports runtime topology overrides via CLI flags:
- `--hidden-family {grid,cluster}`: override the hidden-node graph family.
- `--num-hidden N`: number of hidden nodes (required for cluster family).
- `--num-stages N`: number of ODE stages (divides t_span/num_steps equally).
- `--grid-size N`: override per-problem grid default.
- `--bidirectional`: emit dual edges per pair.
- `--edge-repeats N`: parallel edges per hidden pair (1-8).
- `--cell-library {legacy,v15,v2,relu,tanh}`: select cell library.

The `_make_dynamic_preset()` function in `train_script.py` builds a fresh preset dict that overrides the topology of the named problem while preserving problem-specific fields (num_inputs, loss, out_dim, schedule). `_validate_hidden_family_args()` validates combinations before any expensive setup.

### 2.19 Per-dim diagnostics (CTLE pipeline)

`train_ctle.py` computes per-dimension MSE, R², and target variance on every validation epoch. Stats are logged to `per_dim_stats.txt` (TSV) and summarized in a 4-subplot figure `per_dim_stats.png`. The `worst_dim` (parameter with highest MSE) is printed to console each epoch. The combined plot includes retrain data when pruning is enabled. Refactored into `_plot_per_dim_diagnostics()`.

### 2.20 Stagewise Deep Equilibrium (DEQ) solver (deq-core-prototype)

The DEQ solver replaces the multi-step Heun rollout with a direct fixed-point solve at each stage. Instead of integrating `dx/dt = f(x)` for 50 steps, the DEQ path solves `f(x*) = 0` using the implicit-function theorem (IFT) to backpropagate:

```
Phi(x) = x + dt * rhs(x, ...)    (damped fixed-point map)
x_star = solve_equilibrium(Phi, x0)
```

**Solver backends.** Two backends are available via `deq_solver.solve_equilibrium()`:

| Backend | Description | Gradients |
|---------|-------------|-----------|
| `torchdeq` (default) | Anderson acceleration forward + Anderson IFT backward via `torchdeq.get_deq` | Implicit (no trajectory storage, no gradient chain) |
| `fixed_point_iter` | Naive fixed-point iteration (fallback when torchdeq unavailable) | Via the explicit unrolled chain (same as BPTT — adequate only for sanity checks) |

**Leak floor contractivity.** Under DEQ, a minimum effective leak (`config.DEQ["leak_floor"]` = 0.05) is added per node via `DifferentialStage.set_leak_floor()` so the diagonal damping dominates cross-coupling edges, guaranteeing the fixed-point map Phi(x) is contractive. The `leak_floor` value is captured by the `phi` closure passed to the solver, ensuring the IFT backward re-evaluates with the same leak floor as the forward solve. After the solve, `set_leak_floor(0.0)` restores the default for any subsequent Heun calls on the same stage.

**fp32 solver (AMP safety).** The solver adapter casts `x0` to float32 and wraps the `deq()` call in `torch.autocast(device_type='cuda', enabled=False)`, so both the Anderson forward iteration and the IFT backward linear solve always run in fp32 regardless of the caller's autocast state. The converged `x_star` is cast back to the stage's parameter dtype for downstream AMP consistency.

**Soft-only cell mode.** `forward_equilibrium` raises `ValueError` if `cell_mode != 'soft'`. STE and hard argmax break implicit differentiation because the argmax step function is not differentiable.

**Synthetic trajectory.** The DEQ path builds a 1-timepoint synthetic trajectory `[B, N, 1]` when `store_trajectory=True`, so downstream regularizers (rail, entropy) that expect a time axis work without a special case.

**Input dependence via persistent drive.** Without persistent drive, the DEQ fixed point `x*` depends only on network parameters — the input `u` enters only through the initial condition `x0`. Persistent bounded drive (`I_drive = I_sat · tanh(g · (x_drive − x) / I_sat)`) adds an input-dependent forcing term to `rhs()`, making `x*` genuinely input-dependent. The `--solver deq` CLI emits a warning if `--persistent-drive` is not also set.

**Diagnostic suite.** The `deq_diagnostics.py` module provides three analysis tools:

1. **`gradient_norm_compare()`** — Compares L1 gradient norms of `z_logits` and `logits` under BPTT (short-horizon Heun) vs DEQ on a single batch with a zero-mean squared-loss. The Kimi-note prediction is that DEQ lifts `z_logits` gradients by 2–4 orders of magnitude by replacing the long gradient chain with a single IFT linear solve.

2. **`estimate_jacobian_cond()`** — Estimates `cond(d(rhs)/dx)` at the equilibrium state using `torch.func.jacrev` on a single batch element. Large condition numbers (>100) indicate the leak floor may need to be raised for better contractivity.

3. **`multistart_uniqueness()`** — Solves DEQ from several random initial guesses (`[-1, 0, 1, 5]`) and reports the maximum pairwise `L_inf` distance. Small values (<1e-3) suggest a unique contractive equilibrium; large values indicate multistability.

These can be invoked post-training via the `--run-deq-diagnostics` CLI flag in `train_script.py`.

**CLI flags (train_script.py):**

| Flag | Default | Description |
|------|---------|-------------|
| `--solver {heun,deq}` | `heun` | Stage-level solver. `deq` enforces soft cell mode and leak floor. |
| `--deq-backend {auto,torchdeq,fixed_point_iter}` | `auto` | Solver backend (auto → torchdeq if installed). |
| `--deq-f-max-iter` | 30 | Max forward Anderson iterations. |
| `--deq-f-tol` | 1e-4 | Relative residual tolerance for forward solve. |
| `--deq-b-max-iter` | 20 | Max backward (IFT) iterations. |
| `--deq-step` | 0.1 | Damped step size `dt` for `Phi(x) = x + dt * rhs(x)`. |
| `--leak-floor` | 0.05 | Minimum per-node leak under DEQ (config default). |
| `--run-deq-diagnostics` | False | Run gradient-norm compare, Jacobian cond, multistart uniqueness after training. |

**Speed defaults (as of 2026-06-21).** The DEQ solver config was tuned for ~3–4× speedup: `anderson_max_iter=20`, `backward_max_iter=3`, `anderson_m=5→3`, `anderson_tol=1e-4→1e-3`. These are set in `config.DEQ` and are CLI-overridable.

**DEQ audit fixes (2026-06-22):**
- **C2 (leak_floor threading, HIGH):** `rhs()` now accepts a `leak_floor=None` kwarg. The `phi` closure captures `lf` as a local variable and passes it explicitly to `rhs()`. The IFT backward re-evaluates the same `lf`. The Heun path continues to use `self.leak_floor` (unmodified by DEQ).
- **H1 (autocast, MEDIUM):** `torch.autocast(device_type='cuda', enabled=False)` restored around deq() call in `_solve_torchdeq` (without `no_grad`, preserving `grad_fn`).
- **C1 (drive CLI, HIGH):** `--persistent-drive` flag in `train_script.py`, auto-forces `write_mode=fan_out`, passes `enable_drive` to `build_net_from_preset`. WARNING printed when `--solver deq` without `--persistent-drive`.

---

## 3. System Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           KirchhoffNetWithIO              │
                          │                                          │
   raw input u ──────────►│  InputMapper / SparseInputMapper         │
                          │  / FanOutInputMapper                     │
                          │    xⱼ(0) = x_max · tanh(W·u + b)         │
                          │    or per-feature (gain, bias) pairs     │
                          │                                          │
                          │  KirchhoffNet (ODE core)                 │
                          │  ┌────────────────────────────────────┐  │
                          │  │  Stage 0  ──►  Transfer  ──► ...  │  │
                          │  │  (Heun integration, t_span=5, 50   │  │
                          │  │   steps)                          │  │
                          │  └────────────────────────────────────┘  │
                          │                                          │
                          │  OutputMapper                            │
   prediction ŷ ◄─────────│    ŷ = Linear(x_final)                   │
                          │    or x_final[read_idx] → Linear         │
                          │                                          │
                          └──────────────────────────────────────────┘
```

Readout families (selected by ``--readout``):

- ``linear`` (pre-refactor default): linear OutputMapper reads projection
  (or hidden) nodes; no accumulator tail, no OutputAffine.
- ``temporal`` (default for new runs): ``h*d_out`` OTA mesh
  (one FreeTanh per (hidden, accumulator) pair) + ``OutputAffine`` over
  the accumulator tail.
- ``shared``: one FreeTanh sense per hidden node vs a private learnable
  ``Vref`` rail + a plain dense ``W[d_out x h]`` crossbar. The crossbar
  has no gates and no VCA on the taps; VCA gates the senses only.
- ``shared-x2``: two senses per node (``s=2``) with a +0.5 ``gm`` jitter on
  every odd-indexed sense so each node has one default-init and one
  shifted-init sense + ``W[d_out x 2h]``.

CTLE Phase-A selection (BO loop): the same flag drives
`kn_bayes_opt.py --readout` (study-level, never sampled by Optuna) and
threads through to the dagger harness as `--kn-readout`; the seed trial
(``START_POINTS["ctle"]``) always runs ``temporal`` as a reference anchor.
Implementation: `docs/ctle-phase-a-readout-wiring-plan.md`.

Shared-sense per-stage dynamics:

```
I_sense_k = Isat_k * tanh(gm_k * (A_k * x_j - B_k * Vref + theta_k))
           × σ(readout_sense_z_logits)
           × (VCA gate, optional)
a_dot_i  = Σ_k W[i,k] * I_sense_k - leak_i * a_i - clip(a_i)
Vref     = sigmoid(raw_vref_sense) * x_max   (per-stage scalar)
```

### Inside one DifferentialStage

```
  x_src[j]  ──►  CellLibrary  ──►  i_edge[e]  ──►  scatter-add to dst
  x_dst[j]  ──►  (L/S/P/Z soft)    (μA)           subtract from src
                 × σ(z_logits)      × multiplicity
                 × compliance gate
                                                    ┌──────────────┐
                              ┌─────────────────────►│  acc[dst] += │
                              │                      │  acc[src] -= │
                              │                      └──────────────┘
                                                    ┌──────────────┐
                              ┌─────────────────────►│  -leak·x     │
                              │                      │  -clip(x)    │
                              │                      └──────────────┘
                                                    ┌──────────────┐
                              └─────────────────────►│  dx/dt = Σ/C │
                                                     └──────────────┘

  i_drive (persistent) ──►  acc[dst] += I_drive     (optional)
```

### Node types in SparseTopology (pre-filter)

| Kind | Role | In ODE core? |
|------|------|-------------|
| `input` | Receives write-path initialization from InputMapper | No — filtered out |
| `hidden` | Internal dynamical node | Yes |
| `proj` | Projection node (global coupling, readout tap) | Yes |
| `output` | Readout placeholder for OutputMapper | No — filtered out |

---

## 4. Module Reference

### `config.py`
**Single source of truth for all tunable constants.**

- `CELL_LIBRARY` — legacy L/S/P/Z cell parameters
- `CELL_LIBRARIES` — dict of named libraries: `legacy`, `v15`, `v2`, `relu`, `tanh`
- `CELL_TYPE_STANDARD`, `CELL_TYPE_POS_RECT`, `CELL_TYPE_NEG_RECT`, `CELL_TYPE_DEAD_ZONE`, `CELL_TYPE_OFF` — type identifiers for formula dispatch
- `MIX_CODES`, `BIAS_CODES`, `THRESH_CODES` — v2 library factorized code tables
- `cells_to_tensor_dict(library_name)` — stack a named library into tensors
- `PHYS` — physical constants (x_max=3.0, C_eff=1.0, beta_softness=0.02, clip_current=0.05, clip_softness=0.02)
- `OPTIM` — training hyperparameters (lr auto-computed as `3e-4 * batch_size/1024`, wd=1e-4, epochs=800, batch_size=1024, reg_warmup_epochs=100, reg_anneal_epochs=50, scheduler_T_0=50, CosineAnnealingLR)
- `TAU` — temperature annealing schedule (init=1.0, final=0.1, min=0.15, hardening_epoch_frac=0.1, final_pretrain=0.8 for two-phase pre-prune)
- `LAMBDAS` — regularizer weights (sparsity=1e-3, rail=0.1, edge_gate=5e-4, node_gate=0.0, power=1e-4, capacitance=0.0, entropy=1e-4; node_gate and capacitance deprecated at 0.0)
- `PRUNE` — pruning thresholds (edge_threshold=0.1, node_threshold=0.05, prune_nodes_by_gate=False)
- `SCHEDULE_THREE_PHASE` — three-phase schedule config (frac_a=0.30, frac_b=0.40, frac_c=0.30; tau targets per phase; Phase B/C lambdas; warmup_frac_b)
- `SCHEDULE_FOUR_PHASE` — four-phase schedule config (frac_a=0.3, frac_b1=0.2, frac_b2=0.2, frac_c=0.3; readiness-gated prune, teacher distillation, STE cell mode)
- `SOLVER` — integration defaults (method=heun, t_span=5.0, num_steps=50)
- `DEQ` — Deep Equilibrium solver config (backend=torchdeq, f_solver=anderson, b_solver=anderson, f_max_iter=30, f_tol=1e-4, b_max_iter=20, anderson_m=5, deq_step=0.1, leak_floor=0.05)
- `INIT` — parameter initialization biases (logits_z_bias=0.0; z_logit_init=0.0, u_logit_init=0.0 → σ=0.50, dσ/dz=0.25; raw_mult_init=0.0, raw_leak_init=-3.0, gain_scale=1.0)
- `DRIVE` — persistent bounded source defaults (drive_isat=0.5, raw_drive_g_init=-1.0, drive_scales=[1.0, 0.5, 0.25])
- `VARIATION` — PVT/mismatch defaults (temp_c=27.0, gain_shift_std=0.05, edge_mismatch_std=0.05; temp_c sampling deprecated)
- `PRESETS` — task-specific topology configs (sinx, housing, smooth2d, smooth2d_grid, housing_grid; supports per-preset lambdas, write_mode, schedule flag, write_fan_out)
- `make_smooth2d_grid_preset(grid_size, num_stages, num_proj, bidirectional, edge_repeats)` — dynamic grid preset builder
- `make_housing_grid_preset(grid_size, num_stages, num_proj, bidirectional, edge_repeats)` — dynamic housing grid preset builder

### `cell_library.py`
**IdealizedCellLibrary** — Tanh-surrogate edge cell library with formula dispatch for standard, pos_rect, neg_rect, and dead_zone cell types.

- `forward(x_src, x_dst, logits, raw_mult, x_max, ctx, tau, cell_mode)` → `i_edge [batch, E]`
- Supports `cell_mode='soft'` (softmax mixture) and `cell_mode='ste'` (straight-through estimator)
- Preactivation: legacy/v15 uses `u = x_src - rho * x_dst`; v2 uses per-cell `src_gain/dst_gain` mix
- Injects PVT mismatch multiplicatively on gm: `gm *= exp(edge_mismatch)`
- Injects global gain shift: `gm *= exp(global_gain_shift)`
- Compliance gate: sigmoid-based transition when |x_src| or |x_dst| approaches x_max
- Soft library selection: `weights = softmax(logits / tau)`
- Multiplicity: `m = softplus(raw_mult)`
- `compile_forward(backend)`: wraps `forward` with `torch.compile` for kernel fusion

**SimpleEdgeLibrary** — Single-cell edge device with per-edge learnable parameters (p0, p1, p2). Two modes: `relu` (I = ReLU(p0·Vsrc + p1·Vdest + p2)) and `tanh` (I = tanh(...)). No cell selection, no multiplicity, no Z cell. Compliance gating applied.

**make_cell_library(library_name, num_edges)** — Factory: returns `SimpleEdgeLibrary` for `relu`/`tanh`, `IdealizedCellLibrary` for `legacy`/`v15`/`v2`.

### `topology.py`
**Graph construction, topology management, and pruning.**

Three-layer API:
1. **Primitives:** `line_graph()`, `ring_graph()`, `grid_graph()`, `cluster_graph()`, `empty_graph()` (all support `bidirectional` param)
2. **Connectors:** `connect_bipartite()`, `connect_projection()`
3. **Composer:** `StageTopologyBuilder`, `MultiStageTopology.from_config()`

Key data structures:
- `SparseTopology` — universal sparse graph with src/dst edge lists, node kinds, edge types
- `validate_topology()` — sanity checks (no self-loops, density limits measured on UNIQUE directed pairs, not raw edge count, to support parallel repeats/bidirectional)
- `validate_topology_degrees(write_idx, read_idx)` — hard-error check that write→read is >1 hop
- `repeat_edges(topo, n)` — duplicate `EDGE_TYPE_HIDDEN` edges n times for parallel branches
- `topology_to_stage()` — filters I/O edges, remaps node IDs, builds DifferentialStage (supports persistent drive via `write_idx` param)
- `prune_stage(stage, ...)` — structural pruning with joint Z+gate criterion, connectivity backstop, protected nodes, edge-only mode, SimpleEdgeLibrary handling, persistent drive transfer; returns `(new_stage, node_remap)`
- `prune_network(core, ...)` — applies `prune_stage` to all stages, protects driven nodes, returns `(new_core, stage_remaps)`
- `build_net_from_preset()` / `build_net_from_config()` — factory functions supporting all write/read modes, persistent drive, all per-stage config options (edge_repeats, bidirectional, etc.)

### `differential_stage.py`
**DifferentialStage** — A single ODE stage with sparse COO graph + Heun integration + DEQ equilibrium solver.

Per-node dynamics:
```
C_eff · dxⱼ/dt = Σ_{e: dst(e)=j} I_e − Σ_{e: src(e)=j} I_e + I_driveⱼ − leakⱼ · xⱼ − clip(xⱼ)
```

- `rhs(x, ctx, tau, cell_mode, x_drive, drive_scale, leak_floor)` — computes dx/dt at current state (applies node gates `σ(u_logits)` as all-ones; applies edge gates `σ(z_logits)` to i_edge; accumulates KCL via float32 scatter-add for AMP robustness; adds persistent drive current when enabled; optional `leak_floor` kwarg for DEQ contractivity)
- `forward(x0, ctx, t_span, num_steps, tau, store_trajectory, cell_mode, x_drive, drive_scale, solver, deq_cfg)` — dispatches to Heun (`solver='heun'`, default) or DEQ (`solver='deq'`). The DEQ path calls `forward_equilibrium` and builds a 1-timepoint synthetic trajectory.
- `forward_equilibrium(x0, ctx, tau, cell_mode, x_drive, drive_scale, deq_cfg)` — solves `rhs(x*)=0` via the DEQ solver adapter. Enforces `cell_mode='soft'`, applies `leak_floor`, captures `lf` in the `phi` closure for IFT backward consistency, resets `leak_floor` to 0.0 on return.
- `_effective_leak(num_nodes, leak_floor)` — returns `leak_floor + softplus(raw_leak)` per node. When `leak_floor=0.0` (Heun path) this reproduces the pre-DEQ behaviour exactly.
- `set_leak_floor(leak_floor)` — sets the minimum effective leak for DEQ contractivity.
- `drive_current(x, x_drive, drive_scale)` — bounded tanh drive current at driven nodes
- Parameters: `logits [E, Q]` (None when SimpleEdgeLibrary), `raw_mult [E]` (None when SimpleEdgeLibrary), `raw_leak [N]`, `z_logits [E]`, `u_logits [N]` (deprecated), `raw_drive_g [len(write_idx)]` (when drive enabled)
- Buffers: `src`, `dst` (COO format edge lists)
- `leak_floor`: scalar buffer (default 0.0; set >0 by DEQ path)
- Helper methods: `edge_gates()`, `node_gates()` (deprecated, returns all-ones), `active_edge_mask()`, `active_node_mask()` (deprecated), `parameter_breakdown()`
- `compile_rhs(backend)`: wraps `rhs` with `torch.compile`
- `_is_simple` / `is_simple_device`: True when using `SimpleEdgeLibrary`

### `sim_context.py`
**SimContext** — PVT + mismatch container for one forward pass.

- `SimContext(temp_c, global_gain_shift, edge_mismatch)` — dataclass with `.to(device)` method
- `sample_random_context(num_edges, num_cells, ...)` — factory for training variation injection
- temp_c is deprecated (RR-C): defaults to `VARIATION["temp_c_default"]` (27.0°C), no longer randomized

### `stage_transfer.py`
**StageTransfer** — Width-changing layer between stages. No learnable parameters.

- If `out_nodes < in_nodes`: truncates
- If `out_nodes > in_nodes`: zero-pads
- If equal: identity pass-through

### `io_mapper.py`
**Input/Output mappers** — Write and read phases.

- `InputMapper(in_dim, out_dim)` — `x_hidden(0) = x_max · tanh(Linear(u))`, Xavier init scaled by gain_scale. Used in `--write-mode dense`.
- `RobustInputMapper(in_dim, out_dim)` — adds per-feature learnable log-scale preconditioner. Used in `--write-mode dense` when `use_robust_input=True` (housing preset).
- `SparseInputMapper(in_dim, out_dim, write_idx)` — one-to-one writer. Each input feature `u_i` writes to `h_{write_idx[i]}` via independent `(gain_i, bias_i)` pair; non-write positions are zero. Parameter count = `2 * in_dim`. Used in `--write-mode one_to_one` (default). Raises `ValueError` if `in_dim > out_dim`.
- `FanOutInputMapper(in_dim, out_dim, fan_out_map)` — multi-target writer. Each input feature writes to K>1 hidden nodes via per-target (gain, bias) pairs. Hidden nodes NOT in the union of all targets are zero. Parameter count = `2 * K_total`. Used by `smooth2d_grid` and `ctle_grid` presets (`--write-mode fan_out`).
- `OutputMapper(node_dim, out_dim, read_idx=None)` — `ŷ = Linear(x_read)`, no activation. With `read_idx=None` (dense), reads from projection portion. With `read_idx` (sparse, default), gathers from specified full-state indices.

### `kirchhoff_net.py`
**KirchhoffNet** and **KirchhoffNetWithIO** — Top-level network classes.

- `KirchhoffNet(stages, transfers, stage_times, stage_steps)` — multi-stage ODE/DEQ core
  - Handles per-stage edge_mismatch slicing internally
  - Forward pass supports `cell_mode`, `drive_targets`, `drive_scales`, `solver`, `deq_cfg`
  - Dispatches `solver='heun'` (Heun BPTT) or `solver='deq'` (stagewise equilibrium solve)
  - `parameter_breakdown()` for diagnostics
- `KirchhoffNetWithIO(input_mapper, core, output_mapper, hid_count, proj_count, final_hid_count, final_proj_count, write_idx, read_idx, enable_drive, drive_mappers, drive_scales)` — write/evolve/read pipeline
  - `hid_count` / `proj_count` enforce the honest I/O split
  - `final_hid_count` / `final_proj_count` define the final-stage read_slice
  - `enable_drive` / `drive_mappers` / `drive_scales` implement persistent bounded drive per stage
  - When all `core.stage_times` are 0, forward is identity (mapper-only ablation)
  - `forward(u, ctx, tau, store_trajectory, cell_mode, solver, deq_cfg)` → `(ŷ, trajectories)`
  - Threads `solver` and `deq_cfg` to each stage's `forward()` call

### `deq_solver.py`
**DEQ solver adapter** (deq-core-prototype). Wraps `torchdeq.get_deq` with a small, swappable API.

- `solve_equilibrium(phi, x0, cfg)` → `(x_star, info)` — Solve `x* = phi(x*)` starting from `x0`. Backend auto-selected from `cfg['backend']`: `'torchdeq'` (Anderson acceleration with implicit IFT backward) or `'fixed_point_iter'` (naive iteration fallback).
  - `phi` is the damped fixed-point map `Phi(x) = x + dt * rhs(x)`.
  - Solver runs in fp32; autocast is disabled around the `deq()` call for AMP safety.
  - `info` dict contains `nstep` (iterations used) and `rel_residual` (final relative residual).
- `available_backends()` → `list[str]` — reports which backends are available (torchdeq checked by lazy import).
- Two internal implementations:
  - `_solve_torchdeq()`: Uses `torchdeq.get_deq(f_solver='anderson', b_solver='anderson', ...)`. Casts to fp32, wraps `deq()` in `torch.autocast(enabled=False)`, reassembles `info` dict from torchdeq output.
  - `_solve_fixed_point_iter()`: Simple loop `x ← phi(x)` until `|x_{k+1} - x_k| / |x_k| < tol`. No implicit backward — gradients flow through the unrolled iterations.

### `deq_diagnostics.py`
**DEQ diagnostics suite** — Post-training analysis tools for equilibrium behaviour.

- `gradient_norm_compare(stage, x0, ctx, tau, cell_mode, x_drive, drive_scale, leak_floor, deq_cfg, bptt_t_span, bptt_num_steps)` → `dict` — Compares L1 gradient norms of `z_logits` and `logits` under short-horizon BPTT vs DEQ on a single batch with a zero-mean squared-loss. Returns keys: `z_logits_heun`, `z_logits_deq`, `logits_heun`, `logits_deq`, `z_logits_ratio`, `logits_ratio`.
- `estimate_jacobian_cond(stage, x_star, ctx, tau, cell_mode, x_drive, drive_scale, leak_floor)` → `float` — Estimates `cond(d(rhs)/dx)` at the equilibrium using `torch.func.jacrev` on a single batch element. Returns `float('inf')` if singular.
- `multistart_uniqueness(stage, ctx, tau, cell_mode, x_drive, drive_scale, leak_floor, deq_cfg, starts, batch_shape)` → `dict` — Solves DEQ from several initial guesses and reports `max_pairwise_diff` (L_inf) plus `converged_states`.

### `train.py`
**Loss functions, regularizers, tau annealing, training loop, three-phase and four-phase schedules.**

- `compute_loss(net, x0, target, ctx, task_fn, ..., lambdas, tau, return_parts, amp, amp_dtype, reg_scale, cell_mode, teacher, lambda_kd, teacher_tau, teacher_cell_mode, solver, deq_cfg)` — total = task + rail + KD + reg_scale·(sparsity + edge_gate + node_gate + power + capacitance) − entropy_bonus. Splits into task+rail+KD (data-dependent) and structural (parameter-only) components for DataParallel safety. Supports AMP autocast, cell_mode='ste', teacher distillation. Threads `solver` and `deq_cfg` to the network forward pass.
- `compute_solver_loss(net, b, x_star, A, ctx, ...)` — solver-specific: residual + 0.1·solution + regularizers (preserved on disk; not active in paper v1)
- `tau_for_epoch(epoch, total_epochs, tau_init, tau_final)` — monotonic exponential decay with smooth linear hardening in the last fraction of training
- `reg_schedule(epoch)` — piecewise linear warm-up: [0, W) off, [W, W+A) linear anneal, [W+A, ∞) full
- `apply_reg_schedule(lambdas, epoch)` — returns copy of lambdas with structural terms scaled by reg_schedule
- `phase_boundaries(total_epochs)` / `phase_for_epoch(...)` / `three_phase_tau(...)` / `three_phase_lambdas(...)` — three-phase schedule infrastructure
- `four_phase_boundaries(...)` / `phase_for_epoch_four(...)` / `four_phase_tau(...)` / `four_phase_lambdas(...)` / `four_phase_kd_active(...)` — four-phase schedule infrastructure
- `prune_readiness_check(...)` — readiness-gated prune trigger (AND-logic: ratio, cell prob, gate stability, improvement)
- `compute_solidification_metrics(net, tau)` — returns dict of mean cell prob, P(Z), gate openness fractions
- `validate_argmax(net, val_loader, ...)` — validation with τ→0.001 (argmax cell selection) for solidification diagnostics
- `make_optimizer(net, lr, weight_decay, stage_lr_scale, mapper_lr_scale)` — AdamW with optional per-stage geometric LR scaling AND optional mapper LR scaling. Creates separate param groups for stage/mapper/other params when scale != 1.0.
- `apply_ablation(net, ablation)` — in-place structural ablation: 'none', 'mapper-only', 'empty-graph'
- `train_epoch(net, loader, optimizer, task_fn, ctx_factory, epoch, ...)` — single-epoch loop with AMP support

**Regularizer details:**

| Regularizer | What it penalizes | Weight |
|-------------|-------------------|--------|
| Sparsity | `Σ w[:, :Z_index]` (active non-Z cells) | 1e-3 |
| Rail (ReLU²) | `mean(ReLU²(|x| - x_max))` over trajectory | 0.1 |
| Edge gate | `Σ σ(z_e)` — open edge count | 5e-4 |
| Node gate | `Σ σ(u_j)` — open node count **(deprecated, always 0.0)** | 0.0 |
| Power | `Σ_e σ(z_e) · m_e · Σ_q p(L\|e,q) · gm_q` — static power proxy | 1e-4 |
| Capacitance | `C_eff · Σ_j σ(u_j)` — node capacitance area proxy **(deprecated, always 0.0)** | 0.0 |
| Entropy bonus | `−Σ w·log(w)` of logits/tau (off by default) | 1e-4·τ |

### `train_script.py`
**Main training entry point** — CLI script supporting `--problem {sinx,housing,smooth2d,smooth2d_grid,housing_grid}`.

CLI flags:
- `--problem`, `--output`, `--epochs`, `--lr`, `--device`
- `--cell-library {legacy,v15,v2,relu,tanh}`
- `--hidden-family {grid,cluster}`, `--num-hidden`, `--num-stages`, `--edge-repeats`, `--bidirectional`
- `--grid-size N`
- `--stage-lr-scale`, `--retrain-stage-lr-scale`
- `--mapper-lr-scale`, `--retrain-mapper-lr-scale`, `--freeze-mappers`
- `--amp` / `--no-amp`, `--amp-dtype`, `--compile` / `--no-compile`, `--parallel` / `--no-parallel`
- `--validate-every`, `--early-stop` / `--no-early-stop`, `--patience`, `--min-delta`
- `--ablation {none,mapper-only,empty-graph}`, `--variation`
- `--write-mode {one_to_one,dense}`, `--read-mode {sparse,dense}`
- `--write-idx`, `--read-idx` (comma-separated)
- `--prune`, `--retrain` / `--no-retrain`, `--prune-edge-threshold`, `--prune-node-threshold`
- `--prune-nodes-by-gate` / `--no-prune-nodes-by-gate` (deprecated no-ops)
- `--retrain-epochs`, `--retrain-lr`, `--fresh-init`
- `--scheduler-type {cosine,warm_restarts}`, `--no-scheduler`
- `--grad-log`, `--grad-log-every`
- `--schedule {legacy,three_phase,four_phase}`, `--no-argmax-val`
- `--cell-mode {soft,ste,auto}`
- `--ablation-set {none,reg-only,tau-only,edge-only}`
- `--solver {heun,deq}` — stage-level solver (default heun). `deq` enables DEQ fixed-point solve per stage.
- `--deq-backend {auto,torchdeq,fixed_point_iter}` — DEQ solver backend
- `--deq-f-max-iter`, `--deq-f-tol`, `--deq-b-max-iter` — DEQ solver iteration/tolerance overrides
- `--deq-step` — damped step size dt for Phi(x) = x + dt * rhs(x)
- `--leak-floor` — minimum per-node leak under DEQ
- `--run-deq-diagnostics` — post-training DEQ diagnostics report (gradient-norm compare, Jacobian cond, multistart uniqueness)

Dynamic topology: `_make_dynamic_preset()` builds fresh preset dicts from `--hidden-family`, `--num-hidden`, `--num-stages`, `--edge-repeats`, `--bidirectional`, `--grid-size`. `_validate_hidden_family_args()` validates combinations.

Outputs per run: `loss_history.txt`, `loss_curve.png`, `model.pt`, `config_snapshot.txt`, per-stage graph/selection/trajectory plots, output fit, pipeline diagram, `solidification_metrics.txt` (phased schedules), `grad_norms.txt` (when --grad-log enabled), `prune_summary.txt` (when pruning), `model_pruned.pt`.

### `visualize.py`
**Visualization utilities** (lazy-imports matplotlib/networkx).

- `plot_sparse_topology(topo)` — colored graph with input/hidden/proj/output nodes, optional cell-type coloring via `stage` param
- `plot_stage_graph(stage)` — post-filter ODE core graph
- `plot_multi_stage_topology(multi)` — side-by-side stage graphs
- `plot_trajectories(trajs)` — node voltage vs integration step
- `plot_cell_selection(logits)` — heatmap of P(cell | edge)
- `plot_output_fit(y_pred, y_true)` — scatter + residuals
- `plot_network(net)` — full pipeline visualization

### `gen_network_images.py`
**Standalone image generator** — Runs all non-solver presets and saves visualizations to `network_visualization/`.

### `mlp_benchmark.py`
**MLP baselines** — MLPRegressor(2→H→1) for smooth2d Franke task, benchmark comparison.

### `mlp_benchmark_housing.py`
**Housing MLP baseline** — MLPRegressor(8→H→1) for California Housing task. Matches the `housing_grid` KirchhoffNet in parameter count (~2000) and training hyperparameters (AdamW, Huber loss, CosineAnnealingWarmRestarts). The number of linear layers is configurable via `--num-layers` (default 2, must be >= 2): the architecture is `Linear(in, h) -> Act -> [Linear(h, h) -> Act] x (num_layers-2) -> Linear(h, out)`. Outputs loss history, curve, model, and final metrics in original USD × 100k units.

### `mlp_benchmark_friedman1.py` / `mlp_benchmark_friedman2.py` / `mlp_benchmark_friedman3.py`
**Friedman MLP baselines** — MLPRegressor(10→H→1) for Friedman #1 and MLPRegressor(4→H→1) for Friedman #2 / #3, matching the KirchhoffNet baselines in capacity and training hyperparameters (AdamW, Huber loss, CosineAnnealingWarmRestarts, LHS 20k train + 4k uniform val, σ=1.0, normalized inputs). Same `MLPRegressor` definition as `mlp_benchmark_housing.py` and configurable depth via `--num-layers` (default 2). Outputs share the same artifact set as `mlp_benchmark_housing.py`.

### `mlp_bayes_opt.py`
**Optuna Bayesian hyperparameter search for MLP baselines** — runs N trials of `mlp_benchmark_{housing,friedman1,friedman2,friedman3}.py`, each under a **fixed parameter budget** and **fixed epoch budget**, and minimizes a chosen objective (`best_val` Huber by default). Designed for Kaggle (parallel subprocess execution across Kaggle's CPU cores; single GPU on CUDA sessions).

- **Param budget**: `hidden_dim` is **derived** per trial by solving the MLP param-count formula `(L-2)·h² + (in + L - 1 + out)·h + out = budget` for the nearest integer h, so every trial has the same approximate parameter count. The optimizer explores the depth/width degeneracy at constant params (problem-statement §9b).
- **Epoch budget**: `--epochs` is fixed and `--patience=epochs` so early stopping never curtails the run.
- **Search space**: `num_layers ∈ {2, …, --n-layers-max}`, `lr` log-uniform [1e-4, 1e-2], `weight_decay` log-uniform [1e-6, 1e-2], `batch_size ∈ {512, 1024, 2048, 4096}`, `activation ∈ {relu, tanh}`, `loss` fixed to `huber` (matches KNet, overridable). Default param budgets per dataset: housing 2000, friedman1 6600, friedman2/3 7000.
- **Parallelism**: optuna `n_jobs` ThreadPoolExecutor spawns up to `--n-workers` trial subprocesses concurrently. Default: 1 if CUDA is available (each subprocess grabs the GPU), else `os.cpu_count()`.
- **Trial contract**: each trial invokes `mlp_benchmark_{dataset}.py` via `subprocess.run` (stdout/stderr to `trial_<NNNN>/log.txt`) and parses the resulting `final_metrics.txt` for the objective. Subprocess env forces `PYTHONIOENCODING=utf-8` for cross-platform reproducibility (the Friedman scripts write a `π` character in `config_snapshot.txt`).
- **Outputs** in `outputs/mlp_bayes_opt/<dataset>_budget<P>_e<E>/`: `<study_name>.db` (sqlite, resumable via `--resume`), `best_hyperparams.txt`, `results.csv` (every trial's HPs + metrics + actual param count), `objective_history.png`, and per-trial `trial_<NNNN>/` (log, model, final_metrics, etc., inherited from the benchmark script).

CLI: `python mlp_bayes_opt.py --dataset housing --param-budget 2000 --epochs 800 --n-trials 30 --n-workers 2 --seed 0 [--resume]`.

### `tree_benchmark.py`
**Tree / gradient-boosting benchmark** — a single parameterized benchmark for tree-based classical estimators on the same housing + Friedman 1/2/3 datasets as the MLP and KirchhoffNet baselines. Replaces the previous `xgb_benchmark_housing.py` (only XGBoost for housing) and folds in the full grid:

- **Methods** (select via `--method`): `xgboost`, `lightgbm`, `catboost`, `gradient_boosting` (sklearn), `hist_gradient_boosting` (sklearn), `random_forest`, `extra_trees`.
- **Datasets** (select via `--dataset`): `housing`, `friedman1`, `friedman2`, `friedman3`.

Data pipeline:
- **housing**: reuses `train_script._load_california_housing_data` + `_make_data_split` (seed=42, 80/20) for a byte-identical split to `mlp_benchmark_housing.py`. Tree models are trained on RAW (un-normalized) features AND raw targets (USD × 100k) per the finding in the now-removed `xgb_benchmark_housing.py`: tree models are invariant to monotone feature transforms, and the MLP's min-max+standardized scaling actively hurts tree performance (we measured val RMSE 1.02 with min-max+standardized vs 0.45 with raw+raw on the same split).
- **friedman1/2/3**: reuses `train_script.make_data_friedman{1,2,3}` (20k LHS train + 4k uniform val, seed=42, σ=1.0) for a byte-identical split to `mlp_benchmark_friedman{1,2,3}.py`. Tree models are trained on min-max-normalized [0,1] inputs (monotone per-feature transform — splits are invariant) and raw (denormalized) targets.

Capacity is controlled via standard knobs (`--max-depth`, `--max-leaf-nodes`, `--n-estimators`, `--min-samples-leaf`, `--max-features`, regularization via `--reg-alpha`/`--reg-lambda`). The approximate parameter count (total internal split nodes across all trees, computed from native API dumps for each method) is reported as `approx_node_count` in `final_metrics.txt`, matching the XGBoost benchmark's "node_count" proxy. Models are saved per method (`model.json` for XGBoost, `model.txt` for LightGBM, `model.cbm` for CatBoost, `model.pkl` for sklearn estimators).

SOTA-style defaults (overridable): XGBoost n_estimators=1000, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, gamma=0.1; LightGBM/CatBoost mirror XGBoost; sklearn HistGradientBoosting uses 31 max leaf nodes with early stopping; RandomForest/ExtraTrees use 300 trees with `max_features=1.0`. Huber loss is reported in standardized space (denormalized from raw predictions) for direct comparability with the MLP/KirchhoffNet. Outputs `config_snapshot.txt`, `loss_history.txt`, `loss_curve.png`, `output_fit.png`, `final_metrics.txt`, and a method-specific `model.*` artifact (default dir `./output/{method}_{dataset}`).

### `train_ctle.py`
**CTLE inverse design distillation** — Trains a 3-stage KirchhoffNet student (4 inputs → 7 CTLE logits) via 4-phase knowledge distillation from a pre-trained `RegimeAwareMoE` teacher (loaded from `dagger_student_moe.pt`). Supports both `grid` and `cluster` hidden families, `--q75-input` for 8-dim Q75-scaled features, `--persistent-drive`, `--bidirectional`, `--edge-repeats`, per-dim diagnostics with logging/plotting, readiness-gated pruning, and physical-domain evaluation at the end. Not a standard preset — invoked directly via `python train_ctle.py --teacher-path ...`.

Key components:
- `RegimeAwareMoE` — MoE MLP teacher (trunk + gating + experts)
- `MLPTeacherWrapper` — adapter exposing KirchhoffNet-like forward signature
- `make_ctle_preset(family, ...)` — preset factory supporting grid and cluster families
- `generate_ctle_dataset(...)` — LHS spec sampling + teacher labeling + optional normalization
- `compute_per_dim_stats(...)` / `_plot_per_dim_diagnostics(...)` — per-dim MSE/R²/variance logging/plotting
- `_transfer_input_mapper(...)` / `_transfer_output_mapper(...)` — weight transfer for pruned I/O mappers
- `collect_gradient_norms(...)` / `log_gradient_norms(...)` — gradient norm logging

---

## 5. Data Flow

### Training step (one batch) — DEQ schedule (`--solver deq`)

```
1. Sample SimContext (PVT + mismatch)
       │
2. InputMapper:  u [B, in_dim]  →  x0 [B, N_active₀]  (write phase)
       │
3. Stage 0 DEQ (Anderson-accelerated fixed-point solve):
         │    cfg = {"f_max_iter": 30, "f_tol": 1e-4, "deq_step": 0.1,
         │            "leak_floor": 0.05}
         │    stage.set_leak_floor(leak_floor)
         │    def phi(x):
         │        return x + dt * rhs(x, ctx, tau, cell_mode="soft",
         │                           x_drive, drive_scale, leak_floor=lf)
         │    x_star = solve_equilibrium(phi, x0, cfg)
         │    # x_star in fp32; stage.set_leak_floor(0.0)
         │    traj = x_star.unsqueeze(-1)  (1-timepoint synthetic traj)
         │
       ▼
4. StageTransfer: x [B, N₁] → x [B, N₂]  (truncate or zero-pad)
       │
5. Stage 1 DEQ ... (repeat for all stages)
       │
6. OutputMapper:  x_final [B, N_active_last]  →  ŷ [B, out_dim]
       │
7. Loss = task_loss + rail + reg_scale · Σ λ·regularizer − entropy
         → backward through IFT (implicit, no step-unrolling)
```

Key differences from the Heun path:
- No trajectory stored (1-timepoint synthetic only) → ~4–10× memory reduction on large graphs.
- Backward uses the implicit-function theorem (`∂x*/∂θ = −J⁻¹ · ∂rhs/∂θ`) instead of chain-rule through 50 steps → `z_logits` gradients are 2–4 orders of magnitude larger.
- `cell_mode` is forced to `'soft'` (the DEQ path raises `ValueError` if STE is passed).
- `leak_floor` ensures contractivity of the damped fixed-point map `Phi(x) = x + dt · rhs(x)`.
- The solver runs entirely in fp32 even when AMP is enabled for the rest of the training loop.

### Training step (one batch) — legacy schedule

```
1. Sample SimContext (PVT + mismatch)
       │
2. InputMapper:  u [B, in_dim]  →  x0 [B, N_active₀]  (write phase)
       │
3. Stage 0 Heun (50 steps @ t_span=5):
         │    for step in 1..num_steps:
        │        x_src = x[:, src]; x_dst = x[:, dst]
        │        i_edge = CellLibrary(x_src, x_dst, logits, raw_mult, x_max, ctx, tau,
        │                             cell_mode)
        │        i_edge *= σ(z_logits)                    (edge gate)
        │        acc[dst] += i_edge; acc[src] -= i_edge   (KCL scatter-add in float32)
        │        acc += i_drive                            (persistent drive, optional)
        │        dx/dt = (acc - leak·x - clip(x)) / C_eff
        │        x ← Heun step
        │
       ▼
4. StageTransfer: x [B, N₁] → x [B, N₂]  (truncate or zero-pad)
       │
5. Stage 1 Heun ... (repeat for all stages)
       │
6. OutputMapper:  x_final [B, N_active_last]  →  ŷ [B, out_dim]  (read phase)
       │
7. Compute reg_scale = reg_schedule(epoch)  (RR-A: 0.0 during free phase)
       │
8. Loss = task_loss + rail + λ_kd·KD_loss + reg_scale · Σ λ·regularizer − entropy
         → backward through steps
```

### Training step — three-phase schedule

```
1. Determine phase for current epoch (A/B/C via phase_boundaries).
2. Compute tau via three_phase_tau (phase-dependent annealing).
3. Compute effective_lambdas via three_phase_lambdas (phase-dependent weights,
   including warmup within Phase B).
4. Forward pass (same as legacy).
5. At B→C boundary: auto-prune (remove low-gate edges, transfer I/O mappers).
6. Phase C: retrain compact network with post-prune lambdas.
```

### Training step — four-phase schedule

```
1. Determine phase for epoch (A/B1/B2/C via phase_for_epoch_four).
2. Clone frozen teacher from best Phase A checkpoint (at A→B1 boundary).
3. Compute tau via four_phase_tau (phase-dependent annealing).
4. Compute effective_lambdas via four_phase_lambdas.
5. Resolve cell_mode: Phase A=soft, B1/B2/C=STE (straight-through estimator).
6. Forward pass with teacher distillation in B1/B2:
     L = L_task + lambda_kd * MSE(y_student, y_teacher) + rail + structural
7. In Phase B2: evaluate readiness check every validate epoch.
8. On readiness trigger (or B2→C boundary fallback):
     auto-prune (edge-only, edge_threshold=0.05, no node-gate pruning).
9. Phase C: retrain compact network with STE cell mode.
```

### Training step — CTLE distillation (four-phase with teacher KD)

```
1. Load pre-trained RegimeAwareMoE teacher from checkpoint (frozen, eval mode).
2. Generate synthetic CTLE spec samples (LHS over 4/8 spec dimensions) and label
   them with teacher forward pass → logits (optionally normalized per-dim).
3. Build KirchhoffNet student via make_ctle_preset().
4. Four-phase training (A: free fit, B1: cell commit + KD, B2: edge prune + KD,
   C: retrain compact) using the same four_phase infrastructure from train.py.
5. At the end: evaluate per-dim MSE/R², physical parameter relative error,
   and save both pre-prune and pruned models.
```

### Input/Output edge filtering

`SparseTopology` includes input/output edges for topology visualization and validation. `topology_to_stage()` **removes** them — only hidden and projection edges enter the ODE core. This means:
- Input nodes are placeholders consumed by InputMapper (write path)
- Output nodes are placeholders consumed by OutputMapper (read path)
- The ODE only evolves hidden + projection nodes

---

## 6. Training Pipeline

### Temperature annealing

τ controls the softness of library selection. High τ = soft mixture (exploration), low τ = hard selection (exploitation).

Schedule: monotonic exponential decay with smooth linear hardening in the last 20% of training:

```
τ_base = max(τ_min, τ_init · exp(−epoch / (total_epochs · 0.5)))
τ = τ_base for first 80% of training
τ = linear interpolate from τ_base → τ_final over last 20%
```

Configuration:
- `TAU.init = 1.0`, `TAU.final = 0.1`, `TAU.min = 0.15`
- `TAU.hardening_epoch_frac = 0.1` (10% on each side = 20% transition window)
- `TAU.final_pretrain = 0.8` (two-phase tau: pre-prune annealing stops at 0.8, retrain continues 0.8→0.1)

### Initialization

Initial logits are all zero, with `logits_z_bias=0.0` applied to the Z index. With 4 cells (L/S/P/Z), this gives equal P=0.25 for each cell at init (fix-z-death: was +1.0 bias giving P(Z)≈0.42, which combined with tau annealing 1.0→0.1 amplified to ~99.99% Z by end of training).

Gate logits are initialized at 0.0 so `σ(0) ≈ 0.50` with `dσ/dz ≈ 0.25`. This places gates at the **maximum-gradient** point of the sigmoid, giving the regularizer maximum sensitivity to push gates open or closed.

- `raw_mult`: zeros → multiplicity = softplus(0) = ln(2) ≈ 0.69
- `raw_leak`: −3.0 → leak = softplus(−3) ≈ 0.05
- `raw_drive_g`: −1.0 → drive g = softplus(−1) ≈ 0.27
- InputMapper/SparseInputMapper gain: Xavier uniform scaled by `gain_scale=1.0`

### Gradient handling

- Clip gradients to norm 5.0
- AdamW optimizer with weight decay 1e-4 (lr=3e-4 at default batch_size=1024)
- Two scheduler types:
  - `'cosine'` (default): CosineAnnealingLR with `T_max` based on phase boundaries
  - `'warm_restarts'`: CosineAnnealingWarmRestarts (T_0=50, T_mult=1, eta_min=1e-5)
- SimContext is no_grad (variation doesn't get gradients)
- AMP autocast on forward+loss, GradScaler for grad scaling
- torch.compile on cell_lib and rhs (disabled with DataParallel or when setup fails)

### DEQ solver (--solver deq)

When `--solver deq` is selected, each stage's forward pass calls `forward_equilibrium()` instead of the 50-step Heun rollout:

1. **leak_floor** is set on the stage (default `config.DEQ["leak_floor"]` = 0.05) so `Phi(x) = x + dt · rhs(x)` is contractive.
2. **phi closure** captures the current `leak_floor` value as a local variable and passes it explicitly to `rhs()` — the IFT backward re-evaluates the same `lf`.
3. **solvers** run in fp32 (autocast disabled) regardless of AMP mode.
4. **cell_mode** is forced to `'soft'` — `forward_equilibrium` raises `ValueError` if STE is passed.
5. **Synthetic trajectory** replaces the full time series with a 1-timepoint tensor so downstream regularizers work unchanged.

The DEQ path does **not** change the training loop structure (epochs, phases, LR schedules, pruning, etc.) — only the per-batch forward/backward mechanics differ.

### Mapper LR control

When `--mapper-lr-scale < 1.0`, `make_optimizer()` creates a separate AdamW param group for I/O mapper (input_mapper + output_mapper) parameters at `lr * mapper_lr_scale`. Non-stage, non-mapper parameters receive the base LR. Composition with `--stage-lr-scale` is supported: when both are active, three param groups exist (stage-specific, mapper, other). The mapper group gets `lr * mapper_lr_scale` regardless of stage.

`--freeze-mappers` (four_phase only): freezes I/O mapper `requires_grad` during the first half of the combined B1+B2 duration. Mappers train normally in Phase A, freeze at B1 start, unfreeze at the midpoint. After unfreeze, mappers resume at the `--mapper-lr-scale` rate.

### Per-stage LR scaling (stage-lr-scaling)

Multi-stage networks suffer from **vanishing gradients in early stages**: the gradient norm of stage 0 parameters can be 4–5 orders of magnitude smaller than stage 2 because gradient information must flow backward through multiple ODE solves.

To compensate, `make_optimizer()` can create separate param groups per stage with geometrically increasing learning rates for earlier stages:

```
stage i LR = base_lr × scale^(S − 1 − i)
```

Where `S` = number of stages. Example with 3 stages and `scale=10`:
- stage 0 (earliest, smallest gradient) → `lr × 100`
- stage 1 → `lr × 10`
- stage 2 → `lr × 1`

Non-stage parameters (I/O mappers, StageTransfers) receive the base LR by default. Controlled via `--stage-lr-scale` CLI flag (default `1.0`, which collapses to a single standard param group).

**Retrain uses a separate flag `--retrain-stage-lr-scale` (default `1.0`) to avoid over-aggressive updates during warm-start fine-tuning.**

### Persistent bounded drive

When `--persistent-drive` is passed (requires `write_mode='fan_out'`), each stage receives a `FanOutInputMapper` that produces drive targets from the input. The drive current is a tanh-bounded source:

```
I_drive = I_sat · tanh(g · (x_drive − x) / I_sat)
```

where `I_sat = DRIVE["drive_isat"]` (default 0.5 μA) and `g = softplus(raw_drive_g)` (learnable per-node conductance, init `DRIVE["raw_drive_g_init"] = -1.0`). Drive scales decay across stages `[1.0, 0.5, 0.25]` (configurable via `DRIVE["drive_scales"]`).

Driven nodes are forced to survive pruning. In each stage's `rhs()`, the drive current is added to the KCL accumulation before the leak and clip terms.

---

## 7. Task Presets

### `sinx` — Sine function regression
| | |
|---|---|
| Input | 1D (angle in [−π, π]) |
| Output | 1D (sin(angle)) |
| Architecture | 1 stage: 8 hidden + 2 proj, **line** topology (radius=2) |
| Loss | MSE |
| Train size | 8192 |
| Write/Read | `write_mode=one_to_one`, `write_idx=[0]`, `read_mode=sparse`, `read_idx=[7]` |
| Schedule | `schedule=three_phase` (default; override via `--schedule four_phase`) |
| Special | Lambda override: `{"rail": 1.0}` |

### `housing` — California housing price regression (appendix-only)
| | |
|---|---|
| Input | 8D normalized features (min-max [0,1] per column) |
| Output | 1D (price, standardized) |
| Architecture | 1 stage: 16 hidden + 4 proj, **line** topology (radius=2) |
| Loss | MAE |
| Train size | ~16.5K |
| Write/Read | `write_mode=one_to_one`, `write_idx=[0..7]`, `read_mode=sparse`, `read_idx=[15]` |
| Schedule | `schedule=three_phase` (default; override via `--schedule four_phase`) |
| Special | Uses `RobustInputMapper` (per-feature log-scale preconditioner) |

### `smooth2d` — Franke function 2D regression (line topology)
| | |
|---|---|
| Input | 2D (x, y) in [0, 1]² |
| Output | 1D (Franke function value, zero-mean unit-variance normalized) |
| Architecture | 1 stage: 10 hidden + 2 proj, **line** topology (radius=2) |
| Loss | MSE |
| Train size | 20K (4K val, 4K test, sigma=0.01 noise) |
| Write/Read | `write_mode=one_to_one`, `write_idx=[0,1]`, `read_mode=sparse`, `read_idx=[9]` (hidden readout) |
| Schedule | `schedule=three_phase` (default; override via `--schedule four_phase`) |
| Special | LHS-based sampling for training data. No lambda overrides. |

### `smooth2d_grid` — Franke 2D regression (3-stage grid topology)

| | |
|---|---|
| Input | 2D (x, y) in [0, 1]² |
| Output | 1D (Franke function value, normalized) |
| Architecture | **3 stages**: each 49 hidden (7×7 grid, 8-neighbor kernel_size=3) + 3 proj (all_to_all). StageTransfer identity (52→52). Configurable via `--grid-size N`. |
| Edges/Params | Per stage: 294 hidden edges + 147 proj edges = 441 edges, 2261 params. 3 stages = 6783 core + I/O = ~6800 total (at 7×7 default). Use `--grid-size 4` for 4×4 topology (90 edges/stage, ~1750 total). |
| Integration | t_span=5/3 ≈ 1.667 per stage, 17 steps per stage, dt ≈ 0.098 constant |
| Loss | MSE |
| Train size | 20K (4K val, 4K test, sigma=0.01 noise) |
| Write/Read | `write_mode=fan_out`, `write_fan_out` = left/right column formula (see `make_smooth2d_grid_preset`), `read_mode=sparse`, `read_idx` = center column + proj nodes |
| Schedule | `schedule=three_phase` (default; override via `--schedule four_phase`) |
| Lambda override | `{"sparsity": 1e-5, "edge_gate": 5e-6, "node_gate": 0.0, "power": 1e-5, "capacitance": 0.0, "rail": 0.1}` (legacy path; three_phase uses schedule phase values) |
| Special | Fan-out write spreads each input to grid left/right columns. Proj-only read for grids <5; center column + proj for grids ≥5. Configurable grid size via `make_smooth2d_grid_preset(grid_size)` or `--grid-size N` CLI. With `--schedule four_phase`, checkpoints during B1/B2 track `val_argmax` (hard-cell loss) instead of soft validation loss. Supports `--bidirectional` and `--edge-repeats`. |

#### Stage topology: configurable N×N grid

The hidden nodes are arranged as a square grid of size `N × N` in row-major order (default N=7, configurable via `--grid-size N`). Each node connects to its 8 immediate neighbors (Chebyshev distance ≤ 1). Edge count formulas: `2N(N-1)` unique hidden-to-hidden pairs → `2N(N-1)` directed edges. For N=7: 84 hidden edges.

3 projection nodes are appended after the N² hidden nodes, each connected to ALL N² hidden nodes bidirectionally, yielding N² × 3 projection edges. Total state vector per stage is N² + 3 nodes.

#### Multi-stage design

N_stages independent stages (default 3), each with the same topology but **untied weights**. Total t_span=5 and num_steps=50 are split equally.

#### Fan-out write mapping

Each 2D input (x, y) writes to hidden nodes along the left and right columns of the grid. For grid_size ≥ 5: every other row (rows 0,2,4,...). For grid_size < 5: consecutive rows (0..height-2).

### `ctle_grid` — CTLE inverse design (standalone script)

The `ctle_grid` preset is defined in `train_ctle.py:make_ctle_preset()`.

| | |
|---|---|
| Input | 4D CTLE spec params (power, jitter, height, width), LHS-sampled from empirical ranges; or 8D Q75-scaled features with `--q75-input` |
| Output | 7D CTLE logits (optionally per-dim normalized) |
| Architecture | **3 stages**: each `N²` hidden (N×N grid, kernel_size=3) + 7 proj (all_to_all). StageTransfer identity. Configurable via `--grid-size N` (default 5). Also supports `hidden_family='cluster'` with `--num-hidden`. |
| Loss | MSE on teacher logits |
| Write/Read | Write: fan-out (4 inputs → 4 grid quadrants, 2 rows each) or dense (cluster). Read: center column (N nodes) + 7 proj nodes. |
| Schedule | `schedule=four_phase` (always; this script is KD-only) |
| Special | Teacher distillation from RegimeAwareMoE (`lambda_kd=1.0` in B1/B2). Per-dim diagnostics. Physical-domain evaluation at end (relative error per converted CTLE parameter). Supports `--persistent-drive`, `--bidirectional`, `--edge-repeats`. Standalone script: `python train_ctle.py --teacher-path /path/to/dagger_student_moe.pt`. |

### `housing_grid` — California housing (3-stage grid topology)

| | |
|---|---|
| Input | 8D (min-max scaled to [0, 1] per feature) |
| Output | 1D (price, standardized) |
| Architecture | **3 stages**: each 25 hidden (5×5 grid, kernel_size=3) + 3 proj. StageTransfer identity (28→28). Configurable via `--grid-size N`. |
| Loss | Huber (delta=1.0) |
| Train size | ~13.2K (80/20 split) |
| Write/Read | `write_mode=dense`, `read_idx` = center column + proj nodes |
| Schedule | `schedule=three_phase` (default; override via `--schedule four_phase`) |
| Special | Dense write (all-to-all) since housing features have no spatial structure. Read from center-column hidden nodes + proj (3–8 read positions depending on grid size). Validation logs MAE/RMSE in original units (USD × 100k). Supports `--bidirectional` and `--edge-repeats`. |

### Removed presets (R4.2, R4.3)
- `xor` — removed from active PRESETS (weak analog motivation).
- `solver` — sparse linear system benchmark removed from active PRESETS (scope creep for paper v1). The supporting modules (`sparse_solver_data.py`, `sparse_solver_topology.py`, `sparse_solver_baseline.py`, `sparse_solver_track.py`) are preserved on disk for future work.

---

## 8. Configuration Reference

All constants live in `config.py`. Key groups (R7: units are normalized, not SI):

```python
# Cell library type identifiers for formula dispatch.
CELL_TYPE_STANDARD = "standard"
CELL_TYPE_POS_RECT = "pos_rect"
CELL_TYPE_NEG_RECT = "neg_rect"
CELL_TYPE_DEAD_ZONE = "dead_zone"
CELL_TYPE_OFF = "off"

# Legacy library (4 families: L=linear, S=saturating, P=rectifier, Z=disabled)
CELL_L = {"cell_type": "standard", "gm": 0.2, "isat": 10.0, "rho": 1.0, "gleak": 0.01, ...}
CELL_S = {"cell_type": "standard", "gm": 1.0, "isat": 0.5,  "rho": 1.0, "gleak": 0.01, ...}
CELL_P = {"cell_type": "pos_rect", "gm": 1.0, "isat": 1.0,  "rho": 1.0, "gleak": 0.0, ...}
CELL_Z = {"cell_type": "off",      "gm": 0.0, "isat": 0.0,  "rho": 0.0, "gleak": 0.0, ...}
CELL_ORDER = ["L", "S", "P", "Z"]; Z_INDEX = 3; NUM_CELLS = 4

# Mix codes, bias codes, threshold codes (v2 library)
MIX_CODES = {"M11": {"src_gain": 1.0, "dst_gain": 1.0}, ...}
BIAS_CODES = {"Bsoft": {"gm": 0.25, "isat": 1.50}, ...}
THRESH_CODES = {"T0": 0.00, "T1": 0.35}

# Named library configs
CELL_LIBRARIES = {"legacy": {...}, "v15": {...}, "v2": {...}, "relu": {...}, "tanh": {...}}

# Normalized physical limits (R7: not SI-calibrated; x_max=3.0 for headroom)
PHYS = {"x_max": 3.0, "C_eff": 1.0,
        "beta_softness": 0.02, "clip_current": 0.05, "clip_softness": 0.02}

# Training (RR-A: reg_warmup_epochs=100 for longer free phase;
#              lr auto-scaled: BASE_LR=3e-4 * batch_size/1024)
OPTIM = {"lr": 3e-4, "weight_decay": 1e-4, "grad_clip_norm": 5.0,
         "epochs": 800, "batch_size": 1024,
         "reg_warmup_epochs": 100, "reg_anneal_epochs": 50,
         "scheduler_T_0": 50, "scheduler_T_mult": 1, "scheduler_eta_min": 1e-5}

# Temperature annealing (init=1.0, final=0.1, min=0.15,
#                        final_pretrain=0.8 for two-phase pre-prune)
TAU = {"init": 1.0, "final": 0.1, "min": 0.15,
       "T_0": 80, "hardening_epoch_frac": 0.1, "final_pretrain": 0.8}

# Regularization weights (CP: 4 per-component terms;
#                          rail=0.1 with ReLU² barrier)
LAMBDAS = {"sparsity": 1e-3, "rail": 0.1,
           "edge_gate": 5e-4, "node_gate": 0.0,
           "power": 1e-4, "capacitance": 0.0,
           "entropy": 1e-4}

# Pruning thresholds
PRUNE = {"edge_threshold": 0.1, "node_threshold": 0.05,
         "prune_nodes_by_gate": False}  # DEPRECATED

# Three-phase schedule (fit 30% / compress 40% / retrain 30%)
SCHEDULE_THREE_PHASE = {"frac_a": 0.30, "frac_b": 0.40, "frac_c": 0.30,
                        "tau_a": 0.8, "tau_b_init": 0.8, "tau_b_final": 0.5,
                        "tau_c_init": 0.5, "tau_c_final": 0.1,
                        "warmup_frac_b": 1.0/2.0,
                        "lambdas_b": {"sparsity": 5e-5, "edge_gate": 1e-5,
                                      "node_gate": 0.0, "power": 1e-5,
                                      "capacitance": 0.0},
                        "lambdas_c": {"sparsity": 1e-5},
                        "prune_edge_threshold": 0.05,
                        "prune_node_threshold": 0.05,
                        "prune_nodes_by_gate": False}

# Four-phase schedule
SCHEDULE_FOUR_PHASE = {"frac_a": 0.3, "frac_b1": 0.2, "frac_b2": 0.2, "frac_c": 0.3,
                       "tau_a": 0.6, "tau_b1_init": 0.6, "tau_b1_final": 0.5,
                       "tau_b2_init": 0.5, "tau_b2_final": 0.4,
                       "tau_c_init": 0.4, "tau_c_final": 0.1,
                       "warmup_frac_b1": 0.25, "warmup_frac_b2": 0.25,
                       "lambdas_b1": {"sparsity": 5e-5, "power": 1e-4},
                       "lambdas_b2": {"sparsity": 5e-5, "edge_gate": 1e-5, "power": 1e-5},
                       "lambdas_c": {"sparsity": 1e-5},
                       "lambda_kd": 1.0,
                       "readiness_ratio_max": 1.2, "readiness_prob_min": 0.85,
                       "readiness_stability_max": 0.02, "readiness_improvement_min": 1e-4,
                       "readiness_window": 10,
                       "prune_edge_threshold": 0.05,
                       "prune_node_threshold": 0.05,
                       "prune_nodes_by_gate": False}

# Integration defaults
SOLVER = {"method": "heun", "t_span": 5.0, "num_steps": 50}

# DEQ solver (stagewise fixed-point via torchdeq; leak_floor ensures contractivity)
DEQ = {"backend": "torchdeq", "f_solver": "anderson", "b_solver": "anderson",
       "f_max_iter": 30, "f_tol": 1e-4, "b_max_iter": 20, "anderson_m": 5,
       "deq_step": 0.1, "leak_floor": 0.05}

# Parameter initialization
INIT = {"logits_z_bias": 0.0, "raw_mult_init": 0.0, "raw_leak_init": -3.0,
        "gain_scale": 1.0, "z_logit_init": 0.0, "u_logit_init": 0.0}

# Persistent drive
DRIVE = {"drive_isat": 0.5, "raw_drive_g_init": -1.0, "drive_scales": [1.0, 0.5, 0.25]}

# Degree budget (top-k edge competition; k and temperature annealed over train_frac of epochs)
DEGREE_BUDGET = {"enabled": False, "k_start": 8, "k_end": 2, "temp_start": 1.0,
                 "temp_end": 0.1, "axis": "dst", "anneal_frac": 0.8}

# Variation injection (R6.3: off by default at training time; temp_c deprecated)
VARIATION = {"temp_c_default": 27.0, "temp_c_choices": [0.0, 27.0, 75.0],
             "global_gain_shift_std": 0.05, "edge_mismatch_std": 0.05}
```

---

## 9. Sparse Solver Subsystem (preserved, not in active PRESETS)

The sparse solver benchmark is preserved on disk but **not active in paper v1** (R4.3). Modules:

### `sparse_solver_data.py`
Generates random sparse symmetric positive-definite matrices with controlled condition number via:
1. Random sparse symmetric pattern (scipy.sparse.rand)
2. Diagonal dominance margin (guarantees SPD)
3. Low-rank perturbation to spread eigenvalues to target condition number
4. Bounded random solution `x* = x_max · tanh(randn)`, then `b = A @ x*`

### `sparse_solver_topology.py`
Accumulates matrix sparsity patterns across the training dataset into a fixed union supergraph. Edges appearing in ≥10% of samples become hidden graph edges. Uses `StageTopologyBuilder` to add input/projection edges.

### `sparse_solver_baseline.py`
Digital solvers for comparison against the analog network:
- **Jacobi:** `x_{k+1} = D⁻¹(b − R·x_k)`
- **Conjugate Gradient:** Batched CG with early stopping on convergence
- `compare_against_baselines(net, val_loader, n)` — runs all three methods on the same validation set and prints per-method statistics

### `sparse_solver_track.py`
Convergence diagnostic: captures ODE state snapshots during integration and plots 3-panel figure (residual vs time, solution error vs time, final state magnitude distribution).

---

## 10. Testing

### Smoke test (`test_smoke.py`)

Run with:
```bash
/home/annaik/Documents/ASPDAC_2026/venv/bin/python \
  kirchhoff_redesign/ideal/test_smoke.py
```

Tests cover: config loading, SimContext, topology primitives (line/ring/grid/cluster/empty), StageTransfer, Heun convergence, gradient flow, loss finiteness, sparsity push, tau annealing, round-trip sinx, removed presets, housing preset, I/O filtering, topology validation, visualization, solver subsystem, honest I/O split, no-proj fallback, mapper-only ablation, weighted power/area, active presets stage count, tau monotonic, CLI flags, normalized units, apply_ablation, sparse I/O defaults/validation/select/gradients/CLI, complexity proxy, reg schedule curve, smooth2d_grid sparsity override, tau anneal option/override/backward compat, preset lambda overrides, Z-bias elimination, gate initialization, gate application in rhs, per-component regularizers, prune_stage, parameter transfer, all-removed raises, prune_network, topology degree validation, joint Z+gate pruning, prune_network remap, prune I/O transfer, protected write targets, min_read_nodes guard, smooth2d preset, smooth2d_grid preset, MLP benchmark, FanOutInputMapper, optimizer LR auto-scaling, patience default, scheduler config, tau smooth hardening, retrain warmup bounds, fresh init default, retrain LR CLI, loss history, gradient logging, three-phase schedule (TP-1–9), housing_grid preset/data, stage LR scaling, rail loss ReLU², retrain LR scale, degree budget (BUD-1–10).

**DEQ tests (DEQ-1–15):** 15 DEQ-specific tests at the end of `test_smoke.py`:

| Test | ID | What it verifies |
|------|----|-----------------|
| `test_deq_solver_imports_and_solves` | DEQ-1 | `solve_equilibrium` finds fixed point of contractive linear map |
| `test_deq_solver_run_in_fp32` | DEQ-2 | Solver runs in fp32 regardless of input dtype (AMP safety) |
| `test_deq_equilibrium_rhs_residual_small` | DEQ-3 | At DEQ equilibrium, `max|rhs(x*)| < 1e-2` |
| `test_deq_equilibrium_matches_long_horizon_heun` | DEQ-4 | DEQ x* matches long-horizon Heun rollout (same leak_floor) |
| `test_deq_implicit_backward_gradients_finite` | DEQ-5 | Implicit backward yields finite gradients for all params |
| `test_deq_z_logits_grad_norm_at_least_bptt` | DEQ-6 | DEQ z_logits grad norm within 3 orders of BPTT |
| `test_deq_input_dependence` | DEQ-7 | DEQ with persistent drive produces input-dependent x* |
| `test_deq_ste_mode_rejected` | DEQ-8 | `forward_equilibrium` rejects `cell_mode='ste'` |
| `test_deq_leak_floor_enforced` | DEQ-9 | leak_floor=0.0 matches Heun; leak_floor>0 increases leak |
| `test_deq_solver_kwarg_threads_through_kirchhoff_net` | DEQ-10 | `solver='deq'` threads through KirchhoffNetWithIO |
| `test_deq_heun_regression_unchanged` | DEQ-11 | Heun path unchanged with explicit `solver='heun'` |
| `test_deq_multistart_uniqueness_on_contractive` | DEQ-12 | All starts converge to same x* (contractive map) |
| `test_deq_config_defaults` | DEQ-13 | config.DEQ has expected keys and defaults |
| `test_deq_diagnostics_jacobian_cond` | DEQ-14 | `estimate_jacobian_cond` returns finite cond on tiny stage |
| `test_deq_diagnostics_grad_norm_compare` | DEQ-15 | `gradient_norm_compare` returns finite norms |

**Budget tests (BUD-1–10):** Degree budget / top-k softmax competition tests:

| Test | ID | What it verifies |
|------|----|-----------------|
| `test_budget_gate_basic` | BUD-1 | Budget gate per-destination, sums to ≤ k, values in [0, 1] |
| `test_budget_gate_differentiable` | BUD-2 | Budget gate is differentiable through z_logits |
| `test_budget_k_gte_indegree_noop` | BUD-3 | k ≥ in-degree → budget gate is all-ones |
| `test_budget_temperature_limits` | BUD-4 | T→0 gives argmax (hard); T→∞ gives uniform budget |
| `test_budget_axis_src` | BUD-5 | Per-source competition (outgoing edges) |
| `test_budget_axis_both` | BUD-6 | axis=both equals product of src-mask and dst-mask |
| `test_budget_deq_forward` | BUD-7 | DEQ forward with budget produces finite x* that differs from no-budget |
| `test_budget_annealing_schedule` | BUD-8 | `budget_k_for_epoch` and `budget_temperature_for_epoch` linear anneal |
| `test_budget_disabled_byte_identical` | BUD-9 | rhs() output byte-identical to pre-budget when budget_k=0 |
| `test_budget_simple_edge_library_compat` | BUD-10 | Budget works on SimpleEdgeLibrary stages (logits=None) |

---

## 11. File Map

```
kirchhoff_redesign/ideal/
├── __init__.py                    # Package docstring
├── config.py                      # All tunable constants + presets (legacy/v15/v2 libraries,
│                                  #   three-phase and four-phase schedules, DRIVE config,
│                                  #   DEGREE_BUDGET config)
├── cell_library.py                # IdealizedCellLibrary (formula dispatch: standard/pos_rect/
│                                  #   neg_rect/dead_zone; legacy/v15/v2 libraries; cell_mode)
│                                  #   SimpleEdgeLibrary (relu/tanh), make_cell_library factory
├── topology.py                    # Graph primitives (line/ring/grid/cluster/empty, all with
│                                  #   bidirectional support), repeat_edges, builder, stage
│                                  #   conversion, pruning (joint Z+gate, connectivity backstop,
│                                  #   edge-only mode, SimpleEdgeLibrary, persistent drive)
├── differential_stage.py          # DifferentialStage (COO graph + Heun ODE + DEQ equilibrium
│                                  #   solve + gates + compile + persistent drive + cell_mode
│                                  #   + float32 KCL + leak_floor contractivity + degree budget
│                                  #   top-k competition)
├── sim_context.py                 # SimContext (PVT + mismatch dataclass, temp_c deprecated)
├── stage_transfer.py              # StageTransfer (truncation/zero-padding)
├── io_mapper.py                   # InputMapper, RobustInputMapper, SparseInputMapper,
│                                  #   FanOutInputMapper, OutputMapper
├── kirchhoff_net.py               # KirchhoffNet, KirchhoffNetWithIO
│                                  #   (with persistent drive, cell_mode, solver, deq_cfg)
├── deq_solver.py                  # DEQ solver adapter (torchdeq Anderson + IFT backward /
│                                  #   fixed_point_iter fallback; fp32; AMP-safe autocast)
├── deq_diagnostics.py             # DEQ diagnostics (gradient-norm compare, Jacobian cond,
│                                  #   multistart uniqueness)
├── train.py                       # Loss, regularizers (decomposed CP terms), tau annealing,
│                                  #   three-phase + four-phase schedules, solidification
│                                  #   metrics, argmax validation, readiness-based prune
│                                  #   trigger, teacher distillation, stage-LR scaling,
│                                  #   mapper LR scaling, budget k/temperature annealing,
│                                  #   training loop (solver/deq_cfg threading)
├── train_script.py                # CLI training entry point (5 problems; dynamic topology
│                                  #   overrides; prune/retrain; three-phase + four-phase;
│                                  #   grid-size CLI; ablation-set; cell-mode; cell-library;
│                                  #   bidirectional; edge-repeats; persistent-drive;
│                                  #   mapper-lr-control; freeze-mappers; AMP/compile/DP;
│                                  #   gradient logging; per-dim diagnostics; degree budget
│                                  #   (--budget, --budget-k-*, --budget-temp-*, --budget-axis);
│                                  #   DEQ solver: --solver deq, --deq-backend,
│                                  #   --deq-f-max-iter, --deq-step, --leak-floor,
│                                  #   --run-deq-diagnostics)
├── test_smoke.py                  # Smoke test suite
├── mlp_benchmark.py               # MLPRegressor baseline for smooth2d Franke task
├── mlp_benchmark_housing.py       # MLPRegressor baseline for California Housing task
│                                  #   (--num-layers N for configurable depth; default 2)
├── mlp_benchmark_friedman1.py     # MLPRegressor baseline for Friedman #1 (in_dim=10)
├── mlp_benchmark_friedman2.py     # MLPRegressor baseline for Friedman #2 (in_dim=4)
├── mlp_benchmark_friedman3.py     # MLPRegressor baseline for Friedman #3 (in_dim=4)
├── mlp_bayes_opt.py              # Optuna Bayesian hyperparameter search for the four
│                                  #   mlp_benchmark_*.py baselines under a fixed param
│                                  #   budget (hidden_dim derived from num_layers via
│                                  #   the MLP param-count formula) and fixed epoch
│                                  #   budget (--patience=epochs). Per-trial subprocess
│                                  #   of the benchmark script; optuna n_jobs runs N
│                                  #   trials concurrently for Kaggle; outputs sqlite
│                                  #   study + best_hyperparams.txt + results.csv
│                                  #   + objective_history.png + per-trial artifacts.
├── tree_benchmark.py             # Tree / gradient-boosting benchmark across housing +
│                                  #   Friedman 1/2/3: 7 methods (xgboost, lightgbm,
│                                  #   catboost, sklearn gradient_boosting /
│                                  #   hist_gradient_boosting / random_forest /
│                                  #   extra_trees) via --method x --dataset y;
│                                  #   raw features + raw targets for housing
│                                  #   (tree-invariant scaling), min-max inputs +
│                                  #   raw targets for Friedman; mirrored splits
│                                  #   with mlp_benchmark_*.py for direct comparison.
│                                  #   Replaces the now-removed xgb_benchmark_housing.py.
├── train_ctle.py                  # CTLE inverse design: 4-phase KD from RegimeAwareMoE
│                                  #   teacher; grid + cluster families; per-dim diagnostics;
│                                  #   persistent drive; q75-input; gradient logging
├── visualize.py                   # Matplotlib/networkx visualization utilities
├── gen_network_images.py          # Generate network viz images for all presets
├── sparse_solver_data.py          # Random sparse SPD matrix + dataset generator (preserved)
├── sparse_solver_topology.py      # Union-graph topology builder (preserved)
├── sparse_solver_baseline.py      # Jacobi + CG digital solvers for comparison (preserved)
├── sparse_solver_track.py         # Convergence diagnostic tracker (preserved)
├── implementation_notes_deq_kimi.md   # DEQ design rationale (Kimi notes)
├── implementation_notes_deq_gpt_54.md # DEQ critique and contractivity analysis (GPT-4 notes)
├── ARCHITECTURE.md                # This file
├── results/                       # Training run output directories (generated)
└── network_visualization/         # Generated PNGs from gen_network_images.py (if present)
```

---

## 12. Experimental Results

### 12.1 smooth2d_grid four-phase (5x5, 3-stage) — SUCCESS

**Command:**
```
./venv/bin/python kirchhoff_redesign/ideal/train_script.py \
  --problem smooth2d_grid --grid-size 5 --output result_grid_4_phase_5x5 \
  --schedule four_phase --epochs 800 --lr 3e-4 --batch-size 1024 \
  --stage-lr-scale 1.3 --validate-every 20 --grad-log
```

**Config summary:**
| Field | Value |
|-------|-------|
| Grid size | 5×5 (25 hidden + 3 proj) |
| Stages | 3 |
| Params | ~2000 |
| Teacher | none (direct regression) |
| C_eff | 1.0 |
| Write mode | fan_out |
| Batch size | 1024 |
| LR | 3e-4 |
| Stage LR scale | 1.3 |
| Schedule | four_phase |
| Epochs | 800 |
| Platform | RTX 4090 (kaggle) |

**Outcome:** SUCCESS — val_argmax converged below task loss threshold and readiness-gated prune fired successfully.

**Key metrics:**
| Phase | Epochs | Task loss (train/val) | val_argmax |
|-------|--------|---------------------|------------|
| A | 0–199 | ~0.02 / ~0.03 | ~0.04 |
| B1 | 200–359 | ~0.01 / ~0.02 | ~0.03 |
| B2 | 360–559 | ~0.01 / ~0.02 | ~0.02 |
| C (pruned) | 560–799 | ~0.01 / ~0.02 | ~0.02 |

**Diagnosis:** C_eff=1.0 provided normal solver dynamics. Stage LR scaling (1.3) kept early-stage gradients alive. Fan-out write and center-column + proj readout created clean gradient paths. Readiness gate fired at ~epoch 540 (within B2 window), confirming that cell commitment, gate stability, and convergence all aligned. No Z-hoarding observed — mean_pZ stayed below 0.35 throughout.

### 12.2 CTLE distillation four-phase (4x4, 3-stage, dense write) — FAILED

**Command:**
```
python train_ctle.py --teacher-path /path/to/dagger_student_moe.pt \
  --grid-size 4 --epochs 1000 --lr 12e-4 --batch-size 4096 \
  --write-mode dense --mapper-lr-scale 0.1 --validate-every 20 \
  --stage-lr-scale 1.3
```

Also note that for this experiment t_span was set to 5.0 and num_steps was set to 50.

**Config summary:**
| Field | Value |
|-------|-------|
| Grid size | 4×4 (16 hidden + 7 proj) |
| Stages | 3 |
| Params | 5984 |
| Teacher | MoE 56403 params (~9.4× compression) |
| C_eff | 0.5 |
| Write mode | dense |
| Batch size | 4096 |
| LR | 12e-4 |
| Stage LR scale | 1.3 |
| Mapper LR scale | 0.1 |
| Schedule | four_phase |
| Epochs | 1000 |
| Platform | Kaggle T4x2 |

**Outcome:** FAILED — val plateaued at ~0.70 (never below threshold), readiness-gated prune never fired.

**Key metrics:**
| Phase | Epochs | Best val | Best val_argmax | Notes |
|-------|--------|---------|----------------|-------|
| A | 0–249 | ~0.70 | 3.7 (spike) | Z-hoarding: mean_pZ 0.25 → 0.57 |
| B1 | 250–449 | ~0.70 | ~0.71 | val_argmax recovered slightly with KD |
| B2 | 450–699 | ~0.70 | ~0.71 | Readiness never triggered |
| C (fallback) | 700–999 | ~0.70 | ~0.71 | Retrain from fallback prune, no improvement |

**Diagnosis:**
- C_eff=0.5 accelerated dynamics, amplifying gradient imbalance between early/late stages (confirmed: 800× imbalance epoch 0, 2–6× steady state)
- Batch_size=4096 with lr=12e-4 (Goyal scaling: base 3e-4 × 4096/1024 = 12e-4) was too aggressive — caused val_argmax spike in Phase A
- Dense write path (all-to-all) created 16×16=256 input→hidden connections, drowning out the ODE gradients with mapper gradients
- Z-hoarding: mean_pZ reached 0.69 by B1 end, starving L/S/P edges of probability mass
- Platform: Kaggle T4x2 ~4h wall time insufficient for thorough tuning

**Artifacts:** `result_4_phase_4x4_dense_write_moe_distillation_failed/` contains loss_history, config_snapshot, gradient_norms.txt, solidification_metrics.txt, output_log.txt.

### 12.3 DEQ diagnostics — gradient norm comparison (smooth2d_grid, 5x5, 3-stage)

**Command (post-training):**
```
./venv/bin/python kirchhoff_redesign/ideal/train_script.py \
  --problem smooth2d_grid --grid-size 5 --output result_deq_diag_5x5 \
  --solver heun --schedule three_phase --epochs 10 --batch-size 256 \
  --run-deq-diagnostics
```

**Config summary:**
| Field | Value |
|-------|-------|
| Grid size | 5×5 (25 hidden + 3 proj) |
| Stages | 3 |
| Solver | heun (train); diagnostics compares BPTT vs DEQ post-hoc |
| Diagnostics | gradient_norm_compare, estimate_jacobian_cond, multistart_uniqueness |

**Outcome:** DEQ diagnostics report produced a gradient-norm comparison, Jacobian conditioning estimate, and multistart uniqueness check on each stage. Typical `z_logits_ratio` values ranged from 0.1× to 10× of short-horizon BPTT (confirming DEQ does not catastrophically shrink gate gradients). Jacobian condition numbers were below 50 on all stages with `leak_floor=0.05`. Multistart max pairwise differences were below 1e-3, confirming a unique contractive equilibrium.

### 12.4 DEQ-driven training (smooth2d_grid, 5x5, 3-stage) — TBD

Initial DEQ training runs are in progress. The expectation is that `--solver deq` with `--persistent-drive` will:
- Lift `z_logits` gradients by 2–4 OOM (reducing gate saturation).
- Eliminate the `t_span`/`num_steps` hyperparameter dependency.
- Enable the optimizer to discover sparser topologies without BPTT gradient decay.

### 12.5 Kaggle run notes
- Always verify `--lr` vs Goyal scaling: `lr = base_lr × (batch_size / 1024)`. At batch_size=4096, `lr = 3e-4 × 4 = 12e-4`. Lower to 6e-4 (batch_size=2048) for next attempt.
- C_eff below 1.0 should be avoided unless there is strong evidence the standard dynamics are too slow — the smooth2d SUCCESS used C_eff=1.0.
- Stage LR scaling at 1.3 is effective but may need increase to 1.5 if gradient logs show >10× imbalance in steady state.
- For CTLE distillation, try `--write-mode one_to_one` or fan-out (the ctle_grid default) instead of dense to reduce mapper gradient dominance.
