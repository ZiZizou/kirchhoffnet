"""Optuna-driven Bayesian hyperparameter optimization for MLP baselines.

Tunes the training hyperparameters of ``mlp_benchmark_{dataset}.py`` over
multiple optuna trials, while keeping the **parameter budget** and **epoch
budget** fixed per trial:

- Fixed parameter budget: ``hidden_dim`` is derived from
  ``(num_layers, budget)`` by solving the MLP param-count formula for the
  nearest integer. The optimizer explores the depth/width degeneracy at
  constant param count.
- Fixed epoch budget: ``--epochs`` is passed explicitly and ``--patience``
  defaults to ``epochs`` so early stopping never curtails the run.

Each trial is executed as a subprocess of the corresponding
``mlp_benchmark_{dataset}.py``, so the GPU/CPU isolation of the original
benchmark is preserved and the only artefact contract is the
``final_metrics.txt`` file. Parallelism is provided by optuna's
``n_jobs`` (ThreadPoolExecutor) which concurrently spawns up to ``n_workers``
trial subprocesses - this is the recommended mode on Kaggle (CPU cores or
GPUs). When ``torch.cuda.is_available()`` the default is ``n_workers`` =
number of visible CUDA GPUs, and trials are pinned round-robin to individual
GPUs via per-subprocess ``CUDA_VISIBLE_DEVICES`` so multi-GPU hosts (e.g.
Kaggle 2xT4) use every GPU instead of stacking all workers on GPU 0.

Dataset-in / -out dimensions are hardcoded per dataset (housing=8,
friedman1=10, friedman2/3=4, smooth2d=2, all out_dim=1). Default param
budgets match the problem-statement scales (housing 2000, friedman1 6600,
friedman2/3 7000, smooth2d 1842) and are overridable via ``--param-budget``.

Outputs (in ``--output/<dataset>_budget<P>_e<E>/``):
  - ``<study_name>.db``              optuna sqlite (resumable via ``--resume``)
  - ``best_hyperparams.txt``         best config + metrics + actual param count
  - ``results.csv``                  every trial: HPs + metrics + param_count
  - ``objective_history.png``        trial values + best-so-far curve
  - ``trial_<NNNN>/log.txt``         per-trial subprocess stdout/stderr
  - ``trial_<NNNN>/final_metrics.txt`` (inherited from mlp_benchmark_*.py)

CLI:
    python mlp_bayes_opt.py --dataset housing --param-budget 2000 \\
        --epochs 800 --n-trials 30 --n-workers 2 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.samplers import TPESampler

import bo_param_sampling as bps


def _recover_unfinished_trials(study: optuna.Study) -> tuple[list[dict[str, Any]], set[int]]:
    """Mark abandoned RUNNING trials as FAIL and re-queue their params.

    Mirrors kn_bayes_opt.py:510 for AllianceCan Slurm preemptions. A trial
    stays RUNNING in SQLite if the allocation is killed; study.optimize will
    otherwise never retry it.

    Seed trials carry no suggested params (the seed branch hardcodes its
    config without suggest_* calls), so seed status is tracked separately in
    the returned index set — callers must re-enqueue those with
    ``recovered_seed_trial=True`` so the retry takes the seed branch instead
    of sampling a random arch (parity with kn_bayes_opt).
    """
    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if not running:
        return [], set()
    retry_params: list[dict[str, Any]] = []
    recovered_seed_indices: set[int] = set()
    for trial in running:
        params = dict(trial.params)
        if trial.user_attrs.get("seed_trial") is True:
            recovered_seed_indices.add(len(retry_params))
        retry_params.append(params)
        try:
            study._storage.set_trial_state_values(  # type: ignore[attr-defined]
                trial._trial_id, optuna.trial.TrialState.FAIL, None
            )
        except Exception as exc:
            raise RuntimeError(f"Could not mark abandoned trial {trial.number} as FAIL") from exc
    return retry_params, recovered_seed_indices


DATASETS: dict[str, dict[str, Any]] = {
    "housing": {
        "script": "mlp_benchmark_housing.py",
        "in_dim": 8,
        "out_dim": 1,
        "default_budget": 2000,
    },
    "friedman1": {
        "script": "mlp_benchmark_friedman1.py",
        "in_dim": 10,
        "out_dim": 1,
        "default_budget": 6600,
    },
    "friedman2": {
        "script": "mlp_benchmark_friedman2.py",
        "in_dim": 4,
        "out_dim": 1,
        "default_budget": 7000,
    },
    "friedman3": {
        "script": "mlp_benchmark_friedman3.py",
        "in_dim": 4,
        "out_dim": 1,
        "default_budget": 7000,
    },
    "smooth2d": {
        "script": "mlp_benchmark.py",
        "in_dim": 2,
        "out_dim": 1,
        "default_budget": 1842,
    },
    "ctle": {
        # 4-dim specs -> 7 params, MoE variant (RegimeAwareMoE) via DAgger
        "script": "generative-distillation-improved-dagger-nuance-mlp.py",
        "in_dim": 4,
        "out_dim": 7,
        "default_budget": 6000,
    },
}

BATCH_CHOICES = [512, 1024, 2048, 4096]
ACTIVATION_CHOICES = ["relu", "tanh"]
OBJECTIVE_KEYS = {
    "best_val",
    "best_rmse_orig",
    "best_mse_orig",
    "best_mae_orig",
    "best_mape_orig",
    "final_val",
}


def _penalized_objective(metric: float, actual_params: int,
                         reference_params: int, strength: float) -> float:
    """Scale the metric upward in proportion to the model size."""
    if actual_params < 0:
        # Unknown param count: finite sentinel instead of inf (optuna#3676).
        return metric * (1.0 + strength)
    normalized_params = actual_params / max(1, reference_params)
    normalized_params = actual_params / max(1, reference_params)
    return metric * (1.0 + strength * normalized_params)


def _over_budget_objective(actual_params: int, budget: int, base: float) -> float:
    """Finite graded objective used by KNet for soft-budget violations."""
    return bps.over_budget_objective(actual_params, budget, base)


def moe_param_count(trunk_width: int, trunk_layers: int, num_experts: int,
                    input_dim: int = 4) -> int:
    """Exact trainable parameter count of CTLE's RegimeAwareMoE student.

    The student uses the controlled 4-feature KNet input by default, ``trunk_layers`` affine trunk
    layers, two bias-free 8-to-expert routing heads, and one width-to-7 affine
    head per expert.  Non-trainable output-bound tensors are intentionally not
    included, matching ``sum(p.numel() for p in student.parameters())``.
    """
    return bps.moe_param_count(trunk_width, trunk_layers, num_experts, input_dim)


def max_moe_width(*, trunk_layers: int, num_experts: int, param_limit: int,
                  lower: int, upper: int, input_dim: int = 4) -> int:
    """Largest permitted CTLE MoE width under the shared soft parameter cap."""
    valid = [w for w in range(lower, upper + 1)
             if moe_param_count(w, trunk_layers, num_experts, input_dim) <= param_limit]
    return max(valid, default=-1)


def _param_count(h: int, num_layers: int, in_dim: int, out_dim: int) -> int:
    """Closed-form MLP parameter count.

    params = (L-2)h^2 + (in + L - 1 + out)h + out
    where L = num_layers, h = hidden_dim, in = in_dim, out = out_dim.
    """
    return (num_layers - 2) * h * h + (in_dim + num_layers - 1 + out_dim) * h + out_dim


def derive_hidden_dim(*, num_layers: int, budget: int,
                      in_dim: int, out_dim: int) -> int:
    """Return the largest integer h >= 1 with params(h) <= budget.

    Solves the MLP param-count formula for the exact positive root and
    rounds to nearest integer. If that rounded h exceeds the budget we
    step down until it fits. Returns -1 if no valid h exists (e.g. budget
    too small for the requested num_layers).
    """
    if num_layers < 2:
        raise ValueError(f"num_layers must be >= 2, got {num_layers}")
    a = num_layers - 2
    b = in_dim + num_layers - 1 + out_dim
    c = out_dim - budget
    if a == 0:
        h_exact = (budget - out_dim) / b
        h = max(1, round(h_exact))
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return -1
        h_exact = (-b + math.sqrt(disc)) / (2 * a)
        if h_exact <= 0:
            return -1
        h = max(1, round(h_exact))
    while h > 0 and _param_count(h, num_layers, in_dim, out_dim) > budget:
        h -= 1
    if h < 1:
        return -1
    return h


def _parse_final_metrics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            try:
                out[key.strip()] = float(val.strip())
            except ValueError:
                continue
    return out


def _build_command(
    *,
    python: str,
    script: str,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    activation: str,
    loss: str,
    seed: int,
    output: Path,
    patience: int,
    validate_every: int,
    dataset: str,
    device: str,
) -> list[str]:
    cmd = [
        python,
        script,
        "--hidden-dim", str(hidden_dim),
        "--num-layers", str(num_layers),
        "--epochs", str(epochs),
        "--lr", f"{lr:.6e}",
        "--weight-decay", f"{weight_decay:.6e}",
        "--batch-size", str(batch_size),
        "--activation", activation,
        "--loss", loss,
        "--seed", str(seed),
        "--patience", str(patience),
        "--validate-every", str(validate_every),
        "--device", device,
        "--output", str(output),
    ]
    if dataset.startswith("friedman"):
        cmd += ["--target-noise-std", "1.0"]
    return cmd


def _plot_history(study: optuna.Study, path: Path, *, title: str,
                  objective: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[mlp_bayes_opt] matplotlib not installed; skipping plot")
        return
    vals: list[float] = []
    for t in study.trials:
        if t.value is None or t.value == float("inf"):
            continue
        vals.append(float(t.value))
    if not vals:
        return
    best_so_far: list[float] = []
    cur = float("inf")
    for v in vals:
        cur = min(cur, v)
        best_so_far.append(cur)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(vals, "o", color="C0", alpha=0.5, label="trial")
    ax.plot(best_so_far, "-", color="C3", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel(objective)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--param-budget", type=int, default=None,
                        help="Target parameter count (default: per-dataset).")
    parser.add_argument("--epochs", type=int, default=800,
                        help="Fixed epoch budget (default: 800).")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Optional wall-clock timeout in seconds.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed shared across all trials (default: 0).")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Concurrent trial subprocesses. Default: number of "
                             "visible CUDA GPUs when available, else "
                             "os.cpu_count(). Trials are pinned round-robin to "
                             "GPUs via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device 'auto' | 'cpu' | 'cuda' (default: auto-detect). "
                             "'auto' uses CUDA when available and is forwarded to each "
                             "trial subprocess. An explicit 'cuda' with no GPU falls "
                             "back to CPU.")
    parser.add_argument("--n-layers-max", type=int, default=6,
                        help="Upper bound on num_layers (default: 6).")
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-6)
    parser.add_argument("--wd-max", type=float, default=1e-2)
    parser.add_argument("--objective", default="best_val",
                        choices=sorted(OBJECTIVE_KEYS),
                        help="Metric to minimize (default: best_val).")
    parser.add_argument("--param-penalty", type=float, default=0.25,
                        help="Dimensionless multiplier for the parameter-count "
                             "penalty (default: 0.25; 0 disables it). The BO "
                             "objective is metric * (1 + penalty * params / "
                             "param-budget).")
    parser.add_argument("--param-tolerance", type=float, default=0.10,
                        help="Allowed fractional overage above --param-budget "
                             "before a CTLE trial is rejected (default: 0.10).")
    parser.add_argument("--invalid-param-objective", type=float, default=1e6,
                        help="Base finite objective for CTLE architectures above "
                             "the soft parameter cap (default: 1e6).")
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber",
                        help="Training loss (default: huber, matches KNet).")
    parser.add_argument("--input-preprocessing", choices=["knet", "q75"], default="knet",
                        help="CTLE input representation forwarded to MLP trials (default: knet).")
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=Path("./outputs/mlp_bayes_opt"))
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name. Default: "
                             "<dataset>_budget<P>_e<E>.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study from <run_dir>/"
                        "<study_name>.db")
    parser.add_argument("--ctle-dagger-iterations", type=int, default=4,
                        help="CTLE DAgger iterations per BO trial (default: 4).")
    parser.add_argument("--ctle-epochs-per-iter", type=int, default=100,
                        help="CTLE epochs per DAgger iteration (default: 100).")
    parser.add_argument("--ctle-common-eval-size", type=int, default=1000,
                        help="CTLE common evaluation specs per BO trial (default: 1000).")
    parser.add_argument("--ctle-earlystop-eval-every", type=int, default=5,
                        help="Evaluate CTLE common failure rate every N epochs "
                             "(default: 5; final epoch is always evaluated).")
    # Canonical Phase-A mode (plan canonical-ctle-unify).  When --ctle-phase-a is
    # set, every BO trial runs the static-data, Huber-only Phase-A training
    # mode (no DAgger, no relabeling, no surrogate-through terms).  The
    # validation ZIG failure rate becomes the Optuna value (test reserved for
    # the selected-final refit).  --ctle-objective validation/test selects
    # which log line drives the value.
    parser.add_argument("--ctle-phase-a", action="store_true",
                        help="CTLE BO uses the canonical Phase-A training mode (static data, Huber-only).")
    parser.add_argument("--ctle-phase-a-validity-weight", type=float, default=0.3,
                        help="Phase-A ZIG-validity NLL weight passed to every trial "
                             "as --phase-a-validity-weight (default: 0.3; 0.0 = "
                             "legacy Huber-only). Fixed trial constant, not a "
                             "BO dimension. Must match kn_bayes_opt for comparability.")
    parser.add_argument("--ctle-phase-a-validity-ramp-start", type=int, default=10,
                        help="Phase-A validity ramp start epoch, 1-based "
                             "(default: 10).")
    parser.add_argument("--ctle-phase-a-validity-ramp-epochs", type=int, default=30,
                        help="Phase-A validity linear ramp length in epochs "
                             "(default: 30, i.e. full weight from epoch 40).")
    parser.add_argument("--ctle-canonical-dataset", type=Path, default=None,
                        help="Path to the shared canonical Phase-A .npz; required when "
                             "--ctle-phase-a is set so every trial sees the same labels.")
    parser.add_argument("--ctle-prepare-canonical-dataset", action="store_true",
                        help="If set, build the canonical Phase-A .npz at the supplied path "
                             "from a single trial run, then exit before BO starts.")
    parser.add_argument("--ctle-objective", choices=["validation", "test"], default="validation",
                        help="CTLE BO objective source (default: validation; fixes the test-leakage bug).")
    args = parser.parse_args()

    if args.param_tolerance < 0.0:
        raise ValueError("--param-tolerance must be >= 0")
    if args.invalid_param_objective <= 0.0:
        raise ValueError("--invalid-param-objective must be > 0")
    if (args.ctle_dagger_iterations < 1 or args.ctle_epochs_per_iter < 1
            or args.ctle_common_eval_size < 1 or args.ctle_earlystop_eval_every < 1):
        raise ValueError("CTLE iteration, epoch, evaluation-size, and evaluation-frequency values must be >= 1")

    cfg = DATASETS[args.dataset]
    in_dim: int = cfg["in_dim"]
    out_dim: int = cfg["out_dim"]
    script: str = cfg["script"]
    param_budget: int = (args.param_budget if args.param_budget is not None
                         else cfg["default_budget"])

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"{args.dataset}_budget{param_budget}_e{args.epochs}"
    run_dir.mkdir(parents=True, exist_ok=True)

    study_name = (args.study_name
                  or f"{args.dataset}_budget{param_budget}_e{args.epochs}")
    storage = f"sqlite:///{run_dir / (study_name + '.db')}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("[mlp_bayes_opt] WARNING: --device cuda requested but no CUDA GPU "
                  "detected; falling back to CPU")
            device = "cpu"

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.n_workers is None:
        if device == "cuda" and n_gpus >= 1:
            n_workers = n_gpus
        else:
            n_workers = max(1, os.cpu_count() or 1)
    else:
        n_workers = max(1, args.n_workers)

    patience = args.epochs  # full fixed-budget run, no early stopping

    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    # Sampling fingerprint: changing (budget, tolerance, dataset, n_arches,
    # width_range, layers_range) between runs of the same study.db would crash
    # the second committed trial with "CategoricalDistribution does not
    # support dynamic value space". Record on fresh creation; check on resume.
    sampling_fingerprint = bps.make_sampling_fingerprint({
        "dataset": args.dataset,
        "param_budget": param_budget,
        "param_tolerance": args.param_tolerance,
        "n_layers_max": args.n_layers_max,
        "input_preprocessing": args.input_preprocessing,
        "arch_param_name": "moe_arch_idx" if args.dataset == "ctle" else "mlp_arch_idx",
        "n_arches": (len(moe_feasible) if args.dataset == "ctle" else len(plain_feasible)),
        "ctle_phase_a": bool(args.ctle_phase_a),
        "ctle_objective": args.ctle_objective,
    })
    if args.resume and (run_dir / (study_name + ".db")).exists():
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=sampler)
        bps.check_sampling_fingerprint(study, sampling_fingerprint)
        retry_params, recovered_seed_indices = _recover_unfinished_trials(study)
        if retry_params:
            print(f"[mlp_bayes_opt] recovered {len(retry_params)} unfinished trial(s); they will be retried")
            for retry_index, params in enumerate(retry_params):
                study.enqueue_trial(
                    params,
                    user_attrs={
                        "recovered_trial": True,
                        "recovered_seed_trial": retry_index in recovered_seed_indices,
                    },
                )
    else:
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    sampler=sampler, direction="minimize")
        study.set_user_attr("sampling_fingerprint", sampling_fingerprint)

    # Mirror KNet's queued CTLE baseline, including when --resume is used.
    # This gives both optimizers one deterministic, budget-valid prior.
    if args.dataset == "ctle":
        seed_complete = any(
            t.user_attrs.get("seed_trial") is True
            and t.state == optuna.trial.TrialState.COMPLETE
            for t in study.trials
        )
        seed_waiting = any(
            t.state == optuna.trial.TrialState.WAITING
            and (
                t.user_attrs.get("seed_trial") is True
                or t.user_attrs.get("recovered_seed_trial") is True
            )
            for t in study.trials
        )
        if not seed_complete and not seed_waiting:
            study.enqueue_trial(
                {
                    "moe_trunk_width": 44,
                    "moe_trunk_layers": 3,
                    "moe_num_experts": 3,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 256,
                },
                user_attrs={"seed_trial": True},
            )

    repo_dir = Path(__file__).resolve().parent
    script_path = repo_dir / script
    python_exe = sys.executable
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    print(f"[mlp_bayes_opt] dataset={args.dataset} in_dim={in_dim} out_dim={out_dim} "
          f"budget={param_budget} epochs={args.epochs} n_trials={args.n_trials} "
          f"n_workers={n_workers} objective={args.objective} loss={args.loss} "
          f"device={device}")
    print(f"[mlp_bayes_opt] cuda_available={torch.cuda.is_available()} "
          f"n_gpus={n_gpus} study_name={study_name} storage={storage}")
    print(f"[mlp_bayes_opt] run_dir={run_dir}")
    print(f"[mlp_bayes_opt] script={script_path}")
    if args.dataset == "ctle":
        print(f"[mlp_bayes_opt] CTLE proxy={args.ctle_dagger_iterations}x"
              f"{args.ctle_epochs_per_iter}, eval={args.ctle_common_eval_size} "
              f"every {args.ctle_earlystop_eval_every} epochs, "
              f"param_limit={param_budget * (1.0 + args.param_tolerance):.0f}")

    # Joint feasible-architecture lists, computed once per study. Every
    # suggested architecture satisfies the soft cap by construction; budget
    # misses that survive (preflight mismatch) become finite graded penalties.
    # Same soft cap as CTLE/MoE path: budget*(1+param_tolerance).
    plain_soft_limit = int(param_budget * (1.0 + args.param_tolerance))
    moe_soft_limit = plain_soft_limit
    moe_feasible = bps.moe_feasible_arches(
        soft_limit=moe_soft_limit,
        layers_range=(2, 3),
        experts_range=(2, 4),
        width_range=(32, 64),
        input_dim=4 if args.input_preprocessing == "knet" else 8,
    )
    # Non-CTLE plain-MLP feasible (num_layers, hidden_dim) tuples. Width range
    # is derived analytically per depth (smallest to largest width fitting the
    # soft cap) so uniform categorical sampling yields near-budget trials
    # instead of being dominated by tiny models from a [1,4096] sweep.
    plain_width_ranges = bps.mlp_feasible_width_ranges(
        soft_limit=plain_soft_limit, in_dim=in_dim, out_dim=out_dim,
        layers_range=(2, args.n_layers_max), ln=False,
    )
    plain_feasible = bps.mlp_feasible_arches(
        soft_limit=plain_soft_limit,
        in_dim=in_dim, out_dim=out_dim,
        layers_range=(2, args.n_layers_max),
        width_range=plain_width_ranges,
        ln_options=(False,),
    )
    if args.dataset != "ctle":
        bps.require_feasible(plain_feasible, "plain MLP", plain_soft_limit)
    else:
        bps.require_feasible(moe_feasible, "CTLE MoE", moe_soft_limit)
        # The 44-width seed config must be in the feasible list; fall back to
        # the nearest feasible tuple otherwise so the seed always runs.
        if not any(l == 3 and w == 44 and e == 3 for l, e, w in moe_feasible):
            print(f"[mlp_bayes_opt] WARNING: seed arch (44,3,3) not feasible "
                  f"under soft cap {moe_soft_limit}; first feasible tuple will "
                  "replace it in the seed trial")
    print(f"[mlp_bayes_opt] feasible arches: moe={len(moe_feasible)} "
          f"plain={len(plain_feasible)} (soft_limit={moe_soft_limit}, "
          f"plain widths per layer: {plain_width_ranges})")

    def objective(trial: optuna.Trial) -> float:
        # ── CTLE fast DAgger proxy (4×100, Test 1000) ───────────────────
        if args.dataset == "ctle":
            # Use current defaults as trial 0 seed; otherwise sample MoE + DAgger knobs.
            # Recovered seed retries (re-enqueued after preemption) take the
            # seed branch too — parity with kn_bayes_opt.
            is_seed = (
                trial.user_attrs.get("seed_trial") is True
                or trial.user_attrs.get("recovered_seed_trial") is True
            )
            soft_limit = int(param_budget * (1.0 + args.param_tolerance))
            if is_seed:
                trunk_width = 44
                trunk_layers = 3
                num_experts = 3
                lr = 1e-3
                weight_decay = 1e-4
                batch_size = 256
                # Seed arch must itself satisfy the soft cap; substitute the
                # first feasible tuple otherwise so the seed always runs.
                if bps.moe_param_count(trunk_width, trunk_layers, num_experts,
                                       4 if args.input_preprocessing == "knet" else 8) > soft_limit:
                    trunk_layers, num_experts, trunk_width = moe_feasible[0]
            else:
                # Joint feasible sampling: one fixed categorical over the whole
                # (layers, experts, width) manifold under the soft cap. Replaces
                # max_moe_width + moe_width_fraction + TrialPruned.
                arch_idx = bps.sample_arch_idx(trial, "moe_arch_idx", moe_feasible)
                trunk_layers, num_experts, trunk_width = moe_feasible[arch_idx]
                lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
                weight_decay = trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True)
                batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
            trial_dir = run_dir / f"trial_{trial.number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            log_path = trial_dir / "log.txt"
            expected_params = moe_param_count(
                trunk_width, trunk_layers, num_experts,
                4 if args.input_preprocessing == "knet" else 8,
            )
            trial.set_user_attr("actual_params", expected_params)
            if expected_params > soft_limit:
                trial.set_user_attr("penalized", True)
                trial.set_user_attr("penalty_reason",
                                    f"expected params={expected_params} > soft cap {soft_limit}")
                return _over_budget_objective(
                    expected_params, soft_limit, args.invalid_param_objective)
            # Phase-A mode (plan canonical-ctle-unify): static data, Huber +
            # ramped ZIG-validity NLL, bypass DAgger, shared canonical .npz.
            # Validation failure is the default objective; --ctle-objective
            # test re-enables the test metric for direct comparability with
            # the legacy DAgger logs.
            if args.ctle_phase_a:
                if args.ctle_canonical_dataset is None:
                    raise ValueError(
                        "--ctle-phase-a requires --ctle-canonical-dataset <path> "
                        "so every trial sees the same flow-labelled canonical data."
                    )
                cmd = [
                    python_exe, str(script_path),
                    "--canonical-dataset", str(args.ctle_canonical_dataset),
                    "--phase-a-epochs", str(args.epochs),
                    "--moe-trunk-width", str(trunk_width),
                    "--moe-trunk-layers", str(trunk_layers),
                    "--moe-num-experts", str(num_experts),
                    "--lr", f"{lr:.6e}",
                    "--weight-decay", f"{weight_decay:.6e}",
                    "--batch-size", str(batch_size),
                    "--earlystop-eval-every", str(args.ctle_earlystop_eval_every),
                    "--phase-a-validity-weight", f"{args.ctle_phase_a_validity_weight:.6f}",
                    "--phase-a-validity-ramp-start", str(args.ctle_phase_a_validity_ramp_start),
                    "--phase-a-validity-ramp-epochs", str(args.ctle_phase_a_validity_ramp_epochs),
                    "--output", str(trial_dir),
                    "--device", device,
                    "--seed", str(args.seed),
                    "--input-preprocessing", args.input_preprocessing,
                ]
            else:
                # 4×100 DAgger proxy (legacy path)
                cmd = [
                    python_exe, str(script_path),
                    "--dagger-iterations", str(args.ctle_dagger_iterations),
                    "--epochs-per-iter", str(args.ctle_epochs_per_iter),
                    "--common-eval-size", str(args.ctle_common_eval_size),
                    "--earlystop-eval-every", str(args.ctle_earlystop_eval_every),
                    "--moe-trunk-width", str(trunk_width),
                    "--moe-trunk-layers", str(trunk_layers),
                    "--moe-num-experts", str(num_experts),
                    "--lr", f"{lr:.6e}",
                    "--weight-decay", f"{weight_decay:.6e}",
                    "--batch-size", str(batch_size),
                    "--output", str(trial_dir),
                    "--device", device,
                    "--seed", str(args.seed),
                    "--input-preprocessing", args.input_preprocessing,
                ]
            t0 = time.time()
            sub_env = os.environ.copy()
            sub_env.setdefault("PYTHONIOENCODING", "utf-8")
            sub_env.setdefault("PYTHONUTF8", "1")
            gpu_idx = -1
            if device == "cuda" and n_gpus >= 1:
                gpu_idx = trial.number % n_gpus
                sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            with open(log_path, "w", encoding="utf-8") as logf:
                logf.write(f"$ {' '.join(cmd)}\n")
                logf.flush()
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, cwd=str(repo_dir), creationflags=creationflags, env=sub_env)
            if proc.returncode != 0:
                trial.set_user_attr("penalized", True)
                trial.set_user_attr("penalty_reason",
                                    f"training subprocess failed ({proc.returncode})")
                return _over_budget_objective(soft_limit + 1, soft_limit,
                                              args.invalid_param_objective)
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            # Parse the requested objective.  Phase-A logs the final per-split
            # failure rates in ``phase_a_history.json``; legacy DAgger writes
            # them inline as ``Validation failure rate: X%`` and
            # ``Test failure rate: X%``.
            objective_value = None
            if args.ctle_phase_a:
                import json as _json
                hist_path = trial_dir / "phase_a_history.json"
                if hist_path.is_file():
                    with open(hist_path, encoding="utf-8") as _hf:
                        hist = _json.load(_hf)
                    split = "val" if args.ctle_objective == "validation" else "test"
                    if "final" in hist and split in hist["final"]:
                        objective_value = float(hist["final"][split]["failure_rate"])
            else:
                if args.ctle_objective == "validation":
                    vm = re.search(r"Validation failure rate:\s*([\d\.]+)%", text)
                    if vm:
                        objective_value = float(vm.group(1)) / 100.0
                if objective_value is None:
                    m = re.search(r"Test failure rate:\s*([\d\.]+)%", text)
                    if m:
                        objective_value = float(m.group(1)) / 100.0
            if objective_value is None:
                trial.set_user_attr("penalized", True)
                trial.set_user_attr("penalty_reason",
                                    f"missing {args.ctle_objective} failure rate in log")
                return _over_budget_objective(soft_limit + 1, soft_limit,
                                              args.invalid_param_objective)
            test_rate = objective_value
            # Parse actual param count for --param-budget penalty (mlp dagger logs "Student model: X params")
            pm = re.search(r"Student model:\s*([0-9,]+)\s*params", text)
            if pm is None:
                pm = re.search(r"trainable params:\s*([0-9,]+)", text, re.I)
            param_count = int(pm.group(1).replace(",", "")) if pm else None
            trial.set_user_attr("test_failure_rate", test_rate)
            trial.set_user_attr("moe_trunk_width", trunk_width)
            trial.set_user_attr("moe_num_experts", num_experts)
            if param_count is not None:
                trial.set_user_attr("param_count", param_count)
                trial.set_user_attr("actual_params", param_count)
                if param_count > soft_limit:
                    trial.set_user_attr("penalized", True)
                    trial.set_user_attr("penalty_reason",
                                        f"post-run params={param_count} > soft cap {soft_limit}")
                    return _over_budget_objective(
                        param_count, soft_limit, args.invalid_param_objective)
            # Obey --param-budget via same penalized objective as kn_bayes/mlp non-CTLE
            penalized = _penalized_objective(test_rate, param_count or 0, param_budget, args.param_penalty)
            trial.set_user_attr("raw_test_rate", test_rate)
            trial.set_user_attr("penalized_value", penalized)
            return penalized

        # Joint feasible sampling over the (num_layers, hidden_dim) manifold
        # under the budget. Replaces derive_hidden_dim (always max-W, no
        # width exploration) + TrialPruned for infeasible depths.
        arch_idx = bps.sample_arch_idx(trial, "mlp_arch_idx", plain_feasible)
        num_layers, hidden_dim, _plain_ln = plain_feasible[arch_idx]
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay",
                                           args.wd_min, args.wd_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
        activation = trial.suggest_categorical("activation", ACTIVATION_CHOICES)

        trial_dir = run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "log.txt"

        cmd = _build_command(
            python=python_exe, script=str(script_path),
            hidden_dim=hidden_dim, num_layers=num_layers,
            epochs=args.epochs, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, activation=activation, loss=args.loss,
            seed=args.seed, output=trial_dir, patience=patience,
            validate_every=args.validate_every, dataset=args.dataset,
            device=device,
        )
        t0 = time.time()
        sub_env = os.environ.copy()
        sub_env.setdefault("PYTHONIOENCODING", "utf-8")
        sub_env.setdefault("PYTHONUTF8", "1")
        gpu_idx = -1
        if device == "cuda" and n_gpus >= 1:
            gpu_idx = trial.number % n_gpus
            sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  text=True, cwd=str(repo_dir),
                                  creationflags=creationflags,
                                  env=sub_env)
        elapsed = time.time() - t0

        metrics_path = trial_dir / "final_metrics.txt"
        if proc.returncode != 0 or not metrics_path.exists():
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("subprocess_returncode", proc.returncode)
            trial.set_user_attr("subprocess_seconds", elapsed)
            trial.set_user_attr("penalized", True)
            trial.set_user_attr("penalty_reason",
                                f"training subprocess failed ({proc.returncode})")
            return args.invalid_param_objective

        metrics = _parse_final_metrics(metrics_path)
        if args.objective not in metrics:
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("subprocess_seconds", elapsed)
            trial.set_user_attr("penalized", True)
            trial.set_user_attr("penalty_reason", "missing objective in final_metrics.txt")
            return args.invalid_param_objective

        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("hidden_dim", hidden_dim)
        actual_params = int(metrics.get("param_count", -1))
        trial.set_user_attr("actual_params", actual_params)
        raw_objective = float(metrics[args.objective])
        normalized_params = actual_params / max(1, param_budget)
        objective_value = _penalized_objective(
            raw_objective, actual_params, param_budget, args.param_penalty)
        trial.set_user_attr("raw_objective", raw_objective)
        trial.set_user_attr("normalized_param_count", normalized_params)
        trial.set_user_attr("param_penalty", objective_value - raw_objective)
        trial.set_user_attr("subprocess_seconds", elapsed)
        trial.set_user_attr("gpu", gpu_idx)
        print(f"[mlp_bayes_opt] trial {trial.number:04d} "
              f"L={num_layers} h={hidden_dim} params={int(metrics.get('param_count',-1))} "
              f"lr={lr:.2e} wd={weight_decay:.2e} bs={batch_size} act={activation} "
              f"gpu={gpu_idx} -> {args.objective}={raw_objective:.6f} "
              f"penalized={objective_value:.6f} "
              f"({elapsed:.1f}s)")
        return objective_value

    study.optimize(objective, n_trials=args.n_trials, n_jobs=n_workers,
                   timeout=args.timeout, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"[mlp_bayes_opt] done. {len(completed)}/{len(study.trials)} trials completed.")
    if completed:
        print(f"[mlp_bayes_opt] best_value={study.best_value:.6f}")
        print(f"[mlp_bayes_opt] best_params={study.best_params}")

    with open(run_dir / "best_hyperparams.txt", "w") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"in_dim: {in_dim}\n")
        f.write(f"out_dim: {out_dim}\n")
        f.write(f"param_budget: {param_budget}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"loss: {args.loss}\n")
        f.write(f"objective: {args.objective}\n")
        f.write(f"param_penalty: {args.param_penalty}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"phase_a_validity_weight: {args.ctle_phase_a_validity_weight}\n")
        f.write(f"phase_a_validity_ramp_start: {args.ctle_phase_a_validity_ramp_start}\n")
        f.write(f"phase_a_validity_ramp_epochs: {args.ctle_phase_a_validity_ramp_epochs}\n")
        f.write(f"n_trials: {args.n_trials}\n")
        f.write(f"n_workers: {n_workers}\n")
        f.write(f"device: {device}\n")
        f.write(f"n_gpus: {n_gpus}\n")
        if completed:
            bt = study.best_trial
            f.write(f"best_trial_number: {bt.number}\n")
            f.write(f"best_value: {study.best_value:.6f}\n")
            f.write(f"hidden_dim: {bt.user_attrs.get('hidden_dim')}\n")
            f.write(f"actual_params: {bt.user_attrs.get('actual_params')}\n")
            f.write(f"raw_objective: {bt.user_attrs.get('raw_objective')}\n")
            f.write(f"normalized_param_count: {bt.user_attrs.get('normalized_param_count')}\n")
            f.write(f"subprocess_seconds: {bt.user_attrs.get('subprocess_seconds')}\n")
            f.write(f"gpu: {bt.user_attrs.get('gpu', -1)}\n")
            f.write("params:\n")
            for k, v in bt.params.items():
                f.write(f"  {k}: {v}\n")
            f.write("metrics:\n")
            for k in OBJECTIVE_KEYS | {"param_count", "epochs_run", "elapsed_seconds"}:
                if k in bt.user_attrs:
                    f.write(f"  {k}: {bt.user_attrs[k]}\n")

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial", "state", "hidden_dim", "actual_params", "num_layers",
            "lr", "weight_decay", "batch_size", "activation", "loss",
            "device", "gpu", "objective", "objective_value",
            "best_val", "best_epoch", "best_rmse_orig", "best_mae_orig",
            "best_mape_orig", "final_val", "elapsed_seconds",
            "raw_objective", "normalized_param_count", "param_penalty",
            "penalized", "penalized_value", "penalty_reason",
            "phase_a_validity_weight", "phase_a_validity_ramp_start",
            "phase_a_validity_ramp_epochs",
        ])
        for t in study.trials:
            actual_params = t.user_attrs.get("actual_params", -1)
            hidden_dim = t.user_attrs.get("hidden_dim", -1)
            w.writerow([
                t.number, t.state.name,
                hidden_dim, actual_params,
                t.params.get("num_layers"),
                t.params.get("lr"),
                t.params.get("weight_decay"),
                t.params.get("batch_size"),
                t.params.get("activation"),
                args.loss,
                device,
                t.user_attrs.get("gpu", -1),
                args.objective, t.value if t.value is not None else float("inf"),
                t.user_attrs.get("best_val"),
                t.user_attrs.get("best_epoch"),
                t.user_attrs.get("best_rmse_orig"),
                t.user_attrs.get("best_mae_orig"),
                t.user_attrs.get("best_mape_orig"),
                t.user_attrs.get("final_val"),
                t.user_attrs.get("subprocess_seconds"),
                t.user_attrs.get("raw_objective"),
                t.user_attrs.get("normalized_param_count"),
                t.user_attrs.get("param_penalty"),
                t.user_attrs.get("penalized", False),
                t.user_attrs.get("penalized_value", ""),
                t.user_attrs.get("penalty_reason", ""),
                args.ctle_phase_a_validity_weight,
                args.ctle_phase_a_validity_ramp_start,
                args.ctle_phase_a_validity_ramp_epochs,
            ])

    _plot_history(
        study, run_dir / "objective_history.png",
        title=(f"Optuna @ {args.dataset} "
               f"(budget={param_budget}, epochs={args.epochs})"),
        objective=args.objective,
    )

    print(f"[mlp_bayes_opt] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
