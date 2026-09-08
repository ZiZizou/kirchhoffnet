# F1 + F2 implementation note — learnable clip sharpness + GLN rails

Status: implemented on `feat/gm-bounds-gln-rails` (plan
`docs/GLN_learned_sharpness.md`). Both features default **off**, so
non-flag runs keep the exact pre-feature behavior (verified: param count
`N0` and epoch-0 forward unchanged, see below).

## F1 — learnable per-stage clip sharpness

Soft rail clip (pre-existing, `differential_stage.py` rhs):

```text
clip(x) = clip_current * (sigma((x - x_max)/s) - sigma((-x - x_max)/s))
```

with defaults `x_max=3.0`, `clip_current=0.05`, `s = clip_softness = 0.02`
(config `PHYS`). With `--learnable-clip-sharpness`, `s` becomes a per-stage
scalar `clip_sharpness_raw` mapped via sigmoid into
`[clip_sharpness_min, clip_sharpness_max] = [1e-3, 0.2]` and
logit-initialized so the mapped value equals `clip_sharpness_init = 0.02`
at startup — epoch-0 forward is bit-identical to the fixed path
(verified: max abs diff `0.0`).

| Flag | Default | Notes |
|---|---|---|
| `--learnable-clip-sharpness` | off | +1 trainable param per stage (`N0 + num_stages`) |
| `--clip-sharpness-init` | `0.02` | logit-inverted so mapped value == init |
| `--clip-sharpness-min` | `1e-3` | lower bound of mapped sharpness |
| `--clip-sharpness-max` | `0.2` | upper bound (must be > min) |

Optimizer: `clip_sharpness_raw` sits in the **dyn** LR group
(`train.py::make_optimizer`, `train_script.py::compute_update_norms`).

F1 gm/isat search was **already implemented** in the repo
(`--gm-min/--gm-max/--isat-min/--isat-max` in `train_script.py`,
log-uniform `[1, 50]` BO dims in `kn_bayes_opt.py`, `gm_max` clamp after
GLN). Build defaults stay `gm_max=isat_max=10.0` so non-BO runs and old
checkpoints do not change. `config.BO_GM_MAX_RANGE` /
`config.BO_ISAT_MAX_RANGE = (1.0, 50.0)` document the search windows.

## F2 — GLN rails (shared, B=4, rank=2, boundary + readout-sense)

```text
rails (shared across stages): z_b = tanh(alpha_b * (a_b . u + c_b)),  b=1..B
  alpha_b = softplus(alpha_raw_b) + 1e-6;  a, c zero-init (z ~ 0 at init)
per family e:  delta_e = (P @ Q)[e, :] . z        # P [E, rank], Q [rank, B]
               gm_e    = clamp(gm0_e * exp(delta_e), gm_min, gm_max)
```

- `gm0` = the cell library's static bounded-sigmoid gm
  (`FreeTanhLibrary.current_gm`).
- **Identity init (load-bearing):** `P = Q = 0` ⇒ `delta = 0` ⇒ `gm = gm0`
  exactly (`exp(0) = 1`), so GLN-on at init == GLN-off forward
  (verified on the Phase-0 arch: max abs diff `0.0`).
- **Families:** `boundary` (boundary OTAs, E = `len(boundary_src)`) and
  `readout` (shared-sense OTAs, E = `readout_senses_per_node ×
  num_hidden`). Core hidden edges and resistive shunts are **never** gated.
  GLN v1 requires `--cell-library tanh_free` (only library exposing
  per-edge `gm_override`) and `readout_mode='shared_sense'` for the
  readout family (build-time validation).
- **Integration:** `z = rails(u)` computed once per stage entry; GLN
  modulates gm *inside* the tanh argument; VCA (when on) multiplies the
  resulting current *outside* — both compose. Frozen paths
  (`freeze_boundary`, `freeze_temporal_read`) fold the modulated gm into
  the precomputed tensor exactly like the dynamic path.
- **Ownership:** one `GLNRails` module registered once on
  `KirchhoffNetWithIO` (`net.gln_rails.*`); stages hold a plain
  (non-registered) reference so params are counted exactly once.

| Flag | Default | Notes |
|---|---|---|
| `--gln-rails` | off | shared rails + per-family edge mix |
| `--gln-B` | `4` | rail count |
| `--gln-rank` | `2` | factorization rank of P@Q |
| `--gln-alpha-init` | `1.0` | pre-softplus rail steepness |
| `--gln-families` | `boundary,readout` | csv; only these two supported |

Optimizer: all GLN params (`a, c, alpha_raw`, per-family `P, Q`) sit in
the **dyn** LR group.

### Param accounting (exact, verified)

`bo_param_sampling._knet_param_count_cached` builds the net with the new
flags; `gln_rails.gln_param_count` gives the analytic closed form
(shared rails `B·(in_dim + 2)` once + `E·rank + rank·B` per family).
`--count-params-only`, the analytic counter, and the BO feasible-arch
lists agree:

| Config (Phase-0 arch: h10 s4 k4 fanout2 shared vca-r2) | params |
|---|---|
| baseline (defaults) | 2,445 |
| + `--learnable-clip-sharpness` | 2,449 |
| + `--gln-rails` (B=4, rank=2) | 2,521 |
| + both | 2,525 |

BO fingerprints: `kn_bayes_opt.py` sampling fingerprint now includes
`gm_max_range`, `isat_max_range`, `clip_sharpness_search`, and `gln_schema`
— old studies refuse resume (intended).

### Scope / known limits

- Core-edge GLN and shunt gating: out of scope (v1), enforced structurally.
- Prune: `prune_stage` hard-errors on shared-sense readout (pre-existing)
  and now also on GLN — GLN-on-shared is **no-prune v1**. `--prune` with
  GLN fails loud instead of silently dropping the modulation.
- GLN is shared/tied across all stages in v1 (no per-stage rails/edge-mix).
- `gm_max` (F1 searchable) is the post-modulation clamp ceiling; `gm0`
  still comes from the bounded sigmoid map into `[gm_min, gm_max]`.

## Dry run

```bash
# Param count (baseline == N0, +clip == N0+S, +gln == N0+analytic)
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe train_script.py --count-params-only \
  --problem friedman2 --hidden-family small_world \
  --cell-library tanh_free --num-hidden 10 --num-stages 4 \
  --small-world-k 4 --edge-repeats 2 --readout shared \
  --boundary-fan-out '{"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}' \
  --freeze-read --leak non-programmable --vca --vca-core \
  --vca-separate-core-bus --interstage-activation residual-relu-tanh \
  --learnable-clip-sharpness --gln-rails --gln-B 4 --gln-rank 2

# BO (fixed study-level flags; gm/isat max already searched by kn_bayes_opt)
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe kn_bayes_opt.py --dataset friedman2 \
  --readout shared --learnable-clip-sharpness \
  --gln-rails --gln-B 4 --gln-rank 2 --gln-families boundary,readout \
  --output <fresh-dir> --param-budget 3000
```

Note: `kn_bayes_opt` trials run train_script with the default three_phase
schedule, which auto-prunes at the B→C boundary; pruning shared-sense
readout is not supported (pre-existing), so BO runs with `--readout
shared` currently end in a loud prune error after training. Run BO with
`--readout temporal` for end-to-end trials, or pass
`--schedule legacy` to train_script for shared-readout training runs.

## Tests

`test_f1_f2.py` — 11 tests: sharpness init identity + param count +
epoch-0 equality + grad flow; GLN W=0 identity + grad flow + param-count
match + clamp bounds + full-net GLN-off/W0 equality + off adds no params.