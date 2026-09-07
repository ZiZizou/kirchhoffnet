"""Train a reduced differential KirchhoffNet on a chosen task.

CLI:
    train_script.py --problem {sinx,housing} [--output OUTPUT]
                    [--epochs EPOCHS] [--lr LR]
                    [--ablation {none,mapper-only,empty-graph}]
                    [--variation]

Outputs to --output:
  - loss_history.txt
  - loss_curve.png
  - model.pt
  - config_snapshot.txt
  - stage_*_graph_init.png  (one per stage)
  - stage_*_graph_trained.png
  - cell_selection_trained.png
  - trajectories.png
  - output_fit.png
  - pipeline.png

Regularizer warm-up (RR-A): ``[0, W)`` no penalty, ``[W, W+A)`` linear
anneal, ``[W+A, ∞)`` full penalty (defaults: W=50, A=50, see
``config.OPTIM``). Per-preset ``lambdas`` overrides (RR-D) are merged on
top of the global ``LAMBDAS`` dict before each epoch.
"""

import argparse
import json
import os
import sys
import copy
import time
import warnings
import gc
from pathlib import Path

# retrain-oom-fix/REQ-3: reduce CUDA memory fragmentation by allowing
# the caching allocator to expand segments rather than split. The error
# message at the prune-to-retrain boundary explicitly recommends this.
# Merge with any pre-existing PYTORCH_ALLOC_CONF (e.g. Kaggle's runtime
# may already set one) so this is always active.
_existing_alloc = os.environ.get("PYTORCH_ALLOC_CONF", "")
if "expandable_segments" not in _existing_alloc:
    os.environ["PYTORCH_ALLOC_CONF"] = (
        f"{_existing_alloc},expandable_segments:True" if _existing_alloc
        else "expandable_segments:True"
    )
del _existing_alloc

# retrain-oom-fix/REQ-6: torch.compile recompiles whenever a guard
# fails (e.g. requires_grad mismatch between train/val). The default
# limit is 8, which is hit quickly when validate() and compute_loss()
# alternate each epoch. Raise to 32 so the cache doesn't churn and
# leak compiled-graph memory across the 560+240-epoch training run.
try:
    import torch._dynamo
    torch._dynamo.config.recompile_limit = 32
except Exception:
    pass
try:
    # Separate except for cache_size_limit — it may not exist in older
    # PyTorch, and a failure here should not block recompile_limit.
    torch._dynamo.config.cache_size_limit = 64
except AttributeError:
    pass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math as _math

sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts)

from config import (
    OPTIM,
    PRESETS,
    LAMBDAS,
    SCHEDULE_FOUR_PHASE,
    SOLVER,
    TAU,
    VARIATION,
    DEGREE_BUDGET,
    VCA,
    make_smooth2d_grid_preset,
    make_housing_grid_preset)
from cell_library import make_cell_library, SimpleEdgeLibrary
from topology import build_net_from_preset
from kirchhoff_net import format_parameter_breakdown
from sim_context import SimContext, sample_random_context
from train import (
    compute_loss,
    compute_solver_loss,
    make_optimizer,
    tau_for_epoch,
    reg_schedule,
    phase_boundaries,
    phase_for_epoch,
    three_phase_tau,
    three_phase_lambdas,
    four_phase_boundaries,
    phase_for_epoch_four,
    four_phase_tau,
    four_phase_lambdas,
    four_phase_kd_active,
    budget_frac_for_epoch,
    budget_temperature_for_epoch)
from io_mapper import (
    FanOutInputMapper,
    InputMapper,
    NullInputMapper,
    OutputAffine,
    OutputMapper,
    ProjectedSparseInputMapper,
    ResidualTanhInputMapper,
    ResidualTanhOutputMapper,
    RobustInputMapper,
    SparseInputMapper)

# Monkey-patch _compute_regularizers to handle DataParallel wrapping
# (train.py is read-only on Kaggle; DataParallel strips module internals)
# kirchhoff-noise: also unwraps KirchhoffNetNoiseWrapper before stage access.
import train as _train_module
_orig_compute_regularizers = _train_module._compute_regularizers
# Patch _compute_regularizers to handle DataParallel wrapping

def _dp_safe_compute_regularizers(net, trajs, lambdas):
    # Unwrap DataParallel first (outer-most), then kirchhoff-noise wrapper
    # so regularizer code reaches the base KirchhoffNetWithIO stages.
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    if hasattr(net, "base") and hasattr(net, "_stage_noise_std"):
        net = net.base
    return _orig_compute_regularizers(net, trajs, lambdas)

_train_module._compute_regularizers = _dp_safe_compute_regularizers
from visualize import (
    plot_stage_graph,
    plot_trajectories,
    plot_output_fit,
    plot_network)

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _ensure_dir(path: Path) -> Path:
    if path.exists():
        suffix = time.strftime("%Y%m%d_%H%M%S")
        path = path.with_name(f"{path.name}_{suffix}")
    path.mkdir(parents=True, exist_ok=True)
    return path

def _log_gpu_mem(label: str) -> None:
    """retrain-oom-fix/REQ-4: log CUDA memory state at key transition points.

    Reports PyTorch allocated/reserved, and the device's free/total
    memory when CUDA is available. Pass to print() at any transition
    (after Phase A+B, after DEQ diagnostics, after pruning, etc.) to
    track memory pressure through the prune-to-retrain boundary.
    """
    if not torch.cuda.is_available():
        return
    try:
        free_b, total_b = torch.cuda.mem_get_info()
    except Exception:
        free_b = total_b = 0
    alloc_b = torch.cuda.memory_allocated()
    reserved_b = torch.cuda.memory_reserved()
    def _mb(x): return f"{x / 1024 / 1024:.1f} MiB"
    print(
        f"[mem] {label}: "
        f"alloc={_mb(alloc_b)} reserved={_mb(reserved_b)} "
        f"free={_mb(free_b)} total={_mb(total_b)}"
    )

def _resolve_lambdas(problem: str) -> dict:
    """Build the active lambdas dict for ``problem`` (RR-D).

    Per-preset ``lambdas`` overrides are merged on top of the global
    ``LAMBDAS`` so each preset can tune a single knob (e.g. lower the
    rail penalty on sinx) without redefining the entire dict.
    """
    merged = dict(LAMBDAS)
    preset_lambdas = PRESETS[problem].get("lambdas", {})
    merged.update(preset_lambdas)
    return merged

def _resolve_schedule(problem: str, cli_value: str | None) -> str:
    """Resolve the active schedule mode (three-phase-schedule, four-phase-redesign).

    Precedence: explicit CLI flag > preset['schedule'] > 'legacy'.
    """
    if cli_value is not None:
        return cli_value
    preset_val = PRESETS[problem].get("schedule")
    if preset_val in ("legacy", "three_phase", "four_phase"):
        return preset_val
    return "legacy"

def _build_deq_cfg(args) -> dict:
    """Build the DEQ solver config dict from CLI overrides (deq-core-prototype).

    Returns a dict keyed by the kwargs accepted by :func:`deq_solver.solve_equilibrium`.
    Always includes the config-default DEQ dict, then overlays any explicit CLI
    overrides. ``None``-valued CLI args are ignored (config default applies).
    """
    from config import DEQ
    cfg = dict(DEQ)
    if getattr(args, "deq_backend", None) is not None:
        cfg["backend"] = args.deq_backend
    if getattr(args, "deq_f_max_iter", None) is not None:
        cfg["f_max_iter"] = int(args.deq_f_max_iter)
    if getattr(args, "deq_f_tol", None) is not None:
        cfg["f_tol"] = float(args.deq_f_tol)
    if getattr(args, "deq_b_max_iter", None) is not None:
        cfg["b_max_iter"] = int(args.deq_b_max_iter)
    if getattr(args, "deq_step", None) is not None:
        cfg["deq_step"] = float(args.deq_step)
    if getattr(args, "leak_floor", None) is not None:
        cfg["leak_floor"] = float(args.leak_floor)
    return cfg

def _run_deq_diagnostics_report(net, device, ctx_factory, deq_cfg) -> None:
    """Print a one-shot DEQ diagnostics report for a trained model
    (deq-core-prototype). Compares BPTT vs DEQ gradient norms on a single
    batch, estimates Jacobian conditioning, and runs a multistart uniqueness
    check on each stage.
    """
    import math as _math
    from sim_context import SimContext
    from deq_diagnostics import (
        estimate_jacobian_cond,
        gradient_norm_compare,
        multistart_uniqueness)

    raw = net.module if isinstance(net, torch.nn.DataParallel) else net
    raw.eval()

    print("\n[deq-diagnostics] building a sample batch")
    train_loader = getattr(raw, "_train_loader", None)  # not always present
    # Use a zero-input synthetic batch so we don't depend on data loaders.
    first_stage = raw.core.stages[0] if hasattr(raw, "core") else raw.stages[0]
    num_nodes = first_stage.num_nodes
    num_inputs = max(2, raw.hid_count if hasattr(raw, "hid_count") else num_nodes)
    u = torch.zeros(2, num_inputs, device=device) * 0.1
    ctx = SimContext()
    # Provide a one-shot drive at the write indices if persistent drive is on.
    if getattr(raw, "enable_drive", False):
        try:
            drive_targets = []
            for dm in raw.drive_mappers:
                drive_targets.append(raw._make_full_drive(dm(u)))
            drive_scales = raw.drive_scales
        except Exception:
            drive_targets = None
            drive_scales = None
    else:
        drive_targets = None
        drive_scales = None

    print("[deq-diagnostics] gradient norm comparison (BPTT vs DEQ)")
    try:
        grad_res = gradient_norm_compare(
            first_stage, torch.zeros(2, num_nodes, device=device),
            tau=1.0,
            x_drive=None, drive_scale=0.0,
            leak_floor=float(deq_cfg.get("leak_floor", 0.05)),
            deq_cfg=deq_cfg,
            bptt_t_span=0.3, bptt_num_steps=10)
        for k, v in grad_res.items():
            print(f"  {k}: {v:.3e}")
    except Exception as e:
        print(f"  [grad-norm compare] failed: {e}")

    print("[deq-diagnostics] jacobian cond at equilibrium")
    try:
        x_eq, _ = first_stage.forward_equilibrium(
            torch.zeros(2, num_nodes, device=device), tau=1.0,
            x_drive=None, drive_scale=0.0,
            deq_cfg=deq_cfg)
        cond = estimate_jacobian_cond(
            first_stage, x_eq, tau=1.0,
            x_drive=None, drive_scale=0.0,
            leak_floor=float(deq_cfg.get("leak_floor", 0.05)))
        print(f"  cond(J) = {cond:.3e}" + (" (well-conditioned)" if cond < 100 else " (consider larger leak_floor)"))
    except Exception as e:
        print(f"  [jacobian cond] failed: {e}")

    print("[deq-diagnostics] multistart uniqueness")
    try:
        ms = multistart_uniqueness(
            first_stage, tau=1.0,
            x_drive=None, drive_scale=0.0,
            leak_floor=float(deq_cfg.get("leak_floor", 0.05)),
            deq_cfg=deq_cfg, starts=[-1.0, 0.0, 1.0, 5.0],
            batch_shape=(2, num_nodes))
        print(f"  max pairwise diff = {ms['max_pairwise_diff']:.3e}"
              + (" (single equilibrium)" if ms['max_pairwise_diff'] < 1e-3 else " (multistability suspected)"))
    except Exception as e:
        print(f"  [multistart] failed: {e}")

def _apply_ablation_set(args, schedule_mode: str) -> None:
    """Mutate ``args`` to apply a diagnostic ablation preset in place
    (four-phase-redesign/Phase 1c).

    The three meaningful presets are:

    - ``reg-only`` (A): keep tau=1.0 throughout, turn on gate
      regularizers, disable pruning. Tests whether regularization alone
      commits cells.
    - ``tau-only`` (B): anneal tau normally, all structural regularizers
      off, disable pruning. Tests whether tau annealing alone is enough.
    - ``edge-only`` (C): normal B path but disable node-gate pruning and
      use a lower edge threshold. Tests whether node-gate damage is the
      real culprit.

    Only mutates flags that the user did NOT explicitly set on the CLI
    (we read from ``sys.argv`` to detect explicit overrides). This lets a
    user still override the preset for individual knobs.
    """
    if args.ablation_set in (None, "none"):
        return

    import sys as _sys
    argv_tokens = set(_sys.argv[1:])

    def _set_if_unset(attr: str, value) -> None:
        flag_long = f"--{attr.replace('_', '-')}"
        if flag_long not in argv_tokens and attr not in argv_tokens:
            setattr(args, attr, value)

    if args.ablation_set == "reg-only":
        # A: regularization-only test. Force tau_b_final == tau_b_init == 1.0,
        # leave gate regularizers on (default three_phase behavior), and
        # skip pruning entirely.
        _set_if_unset("prune", False)
        print(
            "[ablation-set=reg-only] tau fixed at 1.0 through B, "
            "gate regularizers active, --prune disabled"
        )
    elif args.ablation_set == "tau-only":
        # B: tau-only test. Disable structural regularizers and skip pruning.
        # The current three_phase path zeros out structural regularizers in
        # Phase A, so the only moving piece here is pruning. We disable it
        # and zero lambdas_b to be explicit.
        _set_if_unset("prune", False)
        # We zero structural lambdas in Phase B by overriding the schedule
        # via a sentinel. The train loop reads the active schedule_cfg so
        # we mutate the schedule dict in place for the run duration
        # (a non-invasive in-memory override).
        from config import SCHEDULE_THREE_PHASE
        for k in ("sparsity", "edge_gate", "node_gate", "power", "capacitance", "tanh_sat"):
            if k in SCHEDULE_THREE_PHASE.get("lambdas_b", {}):
                SCHEDULE_THREE_PHASE["lambdas_b"][k] = 0.0
        from config import SCHEDULE_FOUR_PHASE
        for phase_key in ("lambdas_b1", "lambdas_b2"):
            if phase_key in SCHEDULE_FOUR_PHASE:
                for k in ("sparsity", "edge_gate", "power", "capacitance", "tanh_sat"):
                    if k in SCHEDULE_FOUR_PHASE[phase_key]:
                        SCHEDULE_FOUR_PHASE[phase_key][k] = 0.0
        print(
            "[ablation-set=tau-only] structural regularizers zeroed in B/B1/B2, "
            "tau anneals normally, --prune disabled"
        )
    elif args.ablation_set == "edge-only":
        # C: edge-only pruning test (DEPRECATED: node-gate pruning is
        # permanently disabled; this ablation set is a no-op).
        _set_if_unset("prune_edge_threshold", 0.05)
        _set_if_unset("stage_lr_scale", 1.0)
        _set_if_unset("retrain_stage_lr_scale", 1.0)
        _set_if_unset("mapper_lr_scale", 1.0)
        _set_if_unset("retrain_mapper_lr_scale", 1.0)
        _set_if_unset("struct_lr_scale", 1.0)
        _set_if_unset("dyn_lr_scale", 1.0)
        _set_if_unset("freeze_mappers", False)
        print(
            "[ablation-set=edge-only] node-gate pruning permanently disabled "
            "(deprecate-node-gates), edge threshold 0.05, "
            "stage_lr_scale=1.0, mapper_lr_scale=1.0"
        )

def _validate_hidden_family_args(args) -> None:
    """Validate --hidden-family / --num-hidden / --num-stages / --edge-repeats
    combinations before any expensive setup. Raises ``ValueError`` on bad
    combos so argparse-style early failure is obvious to the user.

    Rules:
      - --hidden-family=small_world requires --num-hidden N (N >= 2)
        and --small-world-k must be even, >= 2, < N
      - --hidden-family=small_world + --grid-size is an error
        (small_world ignores grid; mixing is misleading)
      - --hidden-family=torus uses --grid-size (mirrors grid family)
      - --edge-repeats must be in [1, 8]
      - --num-stages must be >= 1
      - --num-hidden without --hidden-family emits a warning (ignored)
    """
    hf = args.hidden_family
    nh = args.num_hidden
    gs = args.grid_size
    er = args.edge_repeats
    ns = args.num_stages

    if er is not None and (er < 1 or er > 8):
        raise ValueError(
            f"--edge-repeats must be in [1, 8], got {er}"
        )
    if ns is not None and ns < 1:
        raise ValueError(
            f"--num-stages must be >= 1, got {ns}"
        )

    if hf is None:
        if nh is not None:
            print(
                f"[train] WARNING: --num-hidden={nh} is ignored because "
                f"--hidden-family was not specified"
            )
        return

    if hf == "grid":
        if nh is not None:
            print(
                f"[train] note: --num-hidden={nh} is ignored for grid family; "
                f"use --grid-size N (default per problem)"
            )
    elif hf == "small_world":
        if nh is None:
            raise ValueError(
                "--hidden-family=small_world requires --num-hidden N to be set. "
                "Pass an integer (e.g. --num-hidden 25) to specify the "
                "number of hidden nodes."
            )
        if nh < 2:
            raise ValueError(
                f"--num-hidden must be >= 2 for small_world family, got {nh}"
            )
        if gs is not None:
            raise ValueError(
                "--grid-size is not compatible with --hidden-family=small_world "
                "(small_world has no spatial grid; --num-hidden controls size)."
            )
        sw_k = getattr(args, "small_world_k", None)
        if sw_k is not None:
            if sw_k < 2 or sw_k % 2 != 0:
                raise ValueError(
                    f"--small-world-k must be even and >= 2, got {sw_k}"
                )
            if sw_k >= nh:
                raise ValueError(
                    f"--small-world-k must be < --num-hidden (k={sw_k}, num_hidden={nh})"
                )
        sw_p = getattr(args, "small_world_p", None)
        if sw_p is not None and not (0.0 <= sw_p <= 1.0):
            raise ValueError(
                f"--small-world-p must be in [0, 1], got {sw_p}"
            )
    elif hf == "torus":
        if nh is not None:
            print(
                f"[train] note: --num-hidden={nh} is ignored for torus family; "
                f"use --grid-size N (default per problem)"
            )

def _build_grid_write_fan_out(num_inputs: int, grid_size: int | None) -> dict:
    """Build a grid-family write_fan_out map with no duplicate targets.

    Each input gets a unique hidden grid node as a write target. Targets
    are unique across inputs (FanOutInputMapper rejects duplicates).

    To keep write->read >1 hop when the preset's read_idx is the center
    column (e.g. housing_grid, smooth2d_grid), we preferentially place
    write targets in the two end columns (col 0 and col ``grid_size-1``),
    which are at least 2 columns away from the center column when
    ``grid_size >= 5``. For ``grid_size < 5`` the end columns may be
    closer to the center; the ``>1 hop`` validation in
    ``build_net_from_config`` will reject the resulting topology only
    when read targets exist in the conflicting columns. Outer columns
    (those adjacent to center) and finally the center column itself
    are used as fallbacks when the end columns are exhausted, in which
    case the topology may not satisfy ``>1 hop`` and the caller is
    expected to either drop ``--persistent-drive`` or supply an
    explicit ``--read-idx`` matching the allowed topology.
    """
    if grid_size is None or grid_size < 2:
        grid_size = 2
    total_nodes = grid_size * grid_size
    if num_inputs > total_nodes:
        raise ValueError(
            f"Cannot build write_fan_out: num_inputs={num_inputs} exceeds "
            f"grid size {grid_size}x{grid_size}={total_nodes} (need one "
            f"unique target per input)"
        )
    center_col = grid_size // 2
    end_cols = [0, grid_size - 1]
    # Round-robin across the two end columns to maximize horizontal spread.
    col_for_input = [end_cols[i % len(end_cols)] for i in range(num_inputs)]
    # Try, in priority order: assigned end column, the other end column,
    # remaining outer columns, then the center column.
    other_outer_cols = [
        c for c in range(grid_size)
        if c not in end_cols and c != center_col
    ]
    fan_out: dict[int, list[int]] = {}
    used: set[int] = set()
    for i in range(num_inputs):
        col = col_for_input[i]
        cols_to_try = list(dict.fromkeys(
            [col] + [c for c in end_cols if c != col] + other_outer_cols + [center_col]
        ))
        placed = False
        for try_col in cols_to_try:
            for r in range(grid_size):
                node = r * grid_size + try_col
                if node not in used:
                    fan_out[i] = [node]
                    used.add(node)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            raise RuntimeError(
                f"Internal error: no unused grid node found for input {i} "
                f"after exhausting {cols_to_try}"
            )
    return fan_out

def _make_dynamic_preset(
    problem: str,
    hidden_family: str,
    num_hidden: int,
    num_stages: int = 1,
    edge_repeats: int = 2,
    grid_size: int | None = None,
    bidirectional: bool = False,
    write_mode_override: str | None = None,
    read_mode_override: str | None = None,
    small_world_k: int = 4,
    small_world_p: float = 0.3,
    small_world_seed: int | None = None,
    leak_mode: str | None = None,
    leak_constant: float | None = None,
    t_span: float | None = None,
    num_steps: int | None = None) -> dict:
    """Build a fresh preset dict that overrides the topology of the named
    problem. Preserves problem-specific fields (num_inputs, loss, out_dim,
    use_robust_input, schedule, lambdas, tau_anneal, write_idx default).

Args:
        problem: Base problem name (e.g. 'housing', 'sinx').
        hidden_family: 'grid' | 'small_world' | 'torus'.
        num_hidden: Number of hidden nodes. For 'grid' / 'torus', must equal
            grid_size**2 (we recompute grid_size from num_hidden if not
            given). For 'small_world', used directly.
        num_stages: Number of ODE stages (default 1).
        edge_repeats: Parallel edges per hidden pair (default 2).
        grid_size: For 'grid' / 'torus' family; height/width. If None and
            family='grid', defaults to 5 (matches housing_grid default).
            For 'torus', defaults to round(sqrt(num_hidden)) if not given.
        bidirectional: Emit two directed edges per hidden node pair (i->j
            AND j->i). Doubles the hidden edge count and gives asymmetric
            cells (P/rectifier) true bidirectional capability. Composes
            multiplicatively with ``edge_repeats``.
        write_mode_override: If set, use this write_mode instead of
            family default. CLI --write-mode takes precedence.
        read_mode_override: If set, use this read_mode instead of
            family default. CLI --read-mode takes precedence.
        small_world_k: For 'small_world' family; even ring-lattice degree
            (default 4).
        small_world_p: For 'small_world' family; rewiring probability in
            [0, 1] (default 0.3).
        small_world_seed: For 'small_world' family; RNG seed for rewiring
            (default None, inherited from global --seed).

    Returns:
        Fresh dict ready to assign to ``PRESETS[problem]`` before calling
        ``build_net_from_preset(problem)``.
    """
    from config import SOLVER
    base = dict(PRESETS[problem])

    # For friedman presets (friedman1, friedman2, friedman3) preserve original
    # write_mode, read_mode, and write_idx. The base preset already has
    # write_idx=[0,4,8,12] for friedman2/friedman3, so keep it.
    # For small_world override, we still use the preset's write_mode/read_mode
    # unless explicitly overridden by CLI.
    num_inputs = int(base["stages"][0]["num_inputs"])
    n_stages = max(1, int(num_stages))
    er = int(edge_repeats)

    if hidden_family == "small_world":
        num_proj = 0
        eff_num_hidden = int(num_hidden)
        hidden_kwargs = {
            "k": int(small_world_k),
            "p": float(small_world_p),
            "seed": int(small_world_seed) if small_world_seed is not None else 0,
            "bidirectional": bidirectional}
        read_idx = list(range(eff_num_hidden))
    elif hidden_family == "torus":
        if grid_size is None:
            grid_size = max(2, round(int(num_hidden) ** 0.5))
        eff_num_hidden = grid_size * grid_size
        if num_hidden is not None and eff_num_hidden != int(num_hidden):
            print(
                f"[train] note: --num-hidden={num_hidden} rounded to "
                f"grid_size={grid_size} (eff_num_hidden={eff_num_hidden})"
            )
        num_proj = 0
        hidden_kwargs = {
            "height": grid_size,
            "width": grid_size,
            "kernel_size": 3,
            "bidirectional": bidirectional}
        read_idx = list(range(eff_num_hidden))
    elif hidden_family == "grid":
        if grid_size is None:
            grid_size = max(2, round(int(num_hidden) ** 0.5))
        eff_num_hidden = grid_size * grid_size
        if num_hidden is not None and eff_num_hidden != int(num_hidden):
            print(
                f"[train] note: --num-hidden={num_hidden} rounded to "
                f"grid_size={grid_size} (eff_num_hidden={eff_num_hidden})"
            )
        num_proj = 3
        hidden_kwargs = {
            "height": grid_size,
            "width": grid_size,
            "kernel_size": 3,
            "bidirectional": bidirectional}

        if grid_size >= 3:
            center_col = grid_size // 2
            center_nodes = [r * grid_size + center_col for r in range(grid_size)]
            read_idx = center_nodes + list(range(eff_num_hidden, eff_num_hidden + num_proj))
        else:
            read_idx = list(range(eff_num_hidden, eff_num_hidden + num_proj))
    else:
        raise ValueError(f"Unknown hidden_family: {hidden_family!r}")

    # Set write_mode/read_mode: CLI override > base preset > family default
    # For friedman presets, base preset has correct settings (write_mode=sparse_proj,
    # read_mode=dense). For other families, use family default (dense for torus/grid).
    if write_mode_override is not None:
        eff_write_mode = write_mode_override
    elif hidden_family in ("torus", "grid"):
        # These families default to dense write mode (no structure-based mapping)
        eff_write_mode = "dense"
    else:
        # Preserve base preset's write_mode (e.g. friedman2's sparse_proj)
        eff_write_mode = base.get("write_mode", "dense")

    if read_mode_override is not None:
        eff_read_mode = read_mode_override
    else:
        eff_read_mode = base.get("read_mode", "dense")

    stage_cfg = {
        "num_inputs": num_inputs,
        "num_hidden": eff_num_hidden,
        "num_proj": num_proj,
        "num_outputs": 0,
        "hidden_family": hidden_family,
        "hidden_kwargs": hidden_kwargs,
        "edge_repeats": er,
        "input_pattern": "all_to_all",
        "output_pattern": "all_to_all",
        "proj_pattern": "all_to_all",
        "t_span": (t_span if t_span is not None else SOLVER["t_span"]) / n_stages,
        "num_steps": round(
            (num_steps if num_steps is not None else SOLVER["num_steps"]) / n_stages
        )}

    new_preset = dict(base)
    new_preset["stages"] = [stage_cfg] * n_stages
    new_preset["write_mode"] = eff_write_mode
    new_preset["read_mode"] = eff_read_mode
    new_preset["read_idx"] = read_idx

    # Fan-out generation: when write_mode='fan_out', every write target
    # list must be present (build_net_from_config requires it). When the
    # base preset already has write_fan_out (e.g. smooth2d_grid), keep it.
    # Otherwise generate a deterministic fan-out mapping that distributes
    # inputs across the grid without duplicate target nodes (FanOutInputMapper
    # requires each hidden node to be written by at most one input).
    # Works for both grid and torus families since node IDs follow row*grid_size+col.
    if eff_write_mode == "fan_out":
        if "write_fan_out" not in new_preset or new_preset["write_fan_out"] is None:
            new_preset["write_fan_out"] = _build_grid_write_fan_out(
                num_inputs=num_inputs, grid_size=grid_size)
    # Leak mode: forward to build_net_from_config via preset dict.
    # Only set when caller explicitly differs from defaults, so existing
    # checkpoint compatibility is preserved.
    if leak_mode is not None and leak_mode != "programmable":
        new_preset["leak_mode"] = leak_mode
    if leak_constant is not None:
        new_preset["leak_constant"] = leak_constant
    return new_preset


def _log_solidification(log_path, epoch: int, metrics: dict) -> None:
    """Append one row of solidification metrics to ``log_path``.

    Writes header on first call (or if the file is empty). Value ordering
    is deterministic (sorted keys).
    """
    if not isinstance(log_path, Path):
        log_path = Path(log_path)
    sorted_keys = sorted(k for k in metrics.keys())
    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w") as f:
            f.write("epoch\t" + "\t".join(sorted_keys) + "\n")
    with open(log_path, "a") as f:
        row = [str(epoch)]
        for k in sorted_keys:
            v = metrics[k]
            row.append(f"{v:.6e}" if isinstance(v, float) else str(v))
        f.write("\t".join(row) + "\n")

def _save_config_snapshot(out_dir: Path, problem: str, args, lambdas: dict,
                          net=None) -> None:
    snap_path = out_dir / "config_snapshot.txt"
    with open(snap_path, "w") as f:
        f.write(f"problem: {problem}\n")
        f.write(f"readout: {getattr(args, 'readout', 'temporal')}\n")
        f.write(f"freeze_temporal_read: {getattr(args, 'freeze_temporal_read', False)}\n")
        f.write(f"preset: {PRESETS[problem]}\n\n")
        f.write("LAMBDAS (global):\n")
        for k, v in LAMBDAS.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nLAMBDAS (effective for {problem}, preset overrides applied):\n")
        for k, v in lambdas.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nPHYS (effective):\n")
        from config import PHYS
        for k in ("x_max", "C_eff"):
            cli_val = getattr(args, "c_eff" if k == "C_eff" else "x_max", None)
            f.write(f"  {k}: {cli_val if cli_val is not None else PHYS[k]}\n")
        f.write("\nCELL LIBRARY:\n")
        f.write(f"  name: {getattr(args, 'cell_library', None) or 'preset'}\n")
        f.write(f"  gm_min: {getattr(args, 'gm_min', None) or 'config'}\n")
        f.write(f"  gm_max: {getattr(args, 'gm_max', None) or 'config'}\n")
        f.write(f"  isat_min: {getattr(args, 'isat_min', None) or 'config'}\n")
        f.write(f"  isat_max: {getattr(args, 'isat_max', None) or 'config'}\n")
        f.write("\nOPTIM:\n")
        for k, v in OPTIM.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nVARIATION:\n")
        for k, v in VARIATION.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nI/O MAPPING:\n")
        if net is not None:
            from io_mapper import FanOutInputMapper, ProjectedSparseInputMapper, SparseInputMapper, InputMapper
            if isinstance(net.input_mapper, FanOutInputMapper):
                effective_write = "fan_out"
            elif isinstance(net.input_mapper, SparseInputMapper):
                effective_write = "one_to_one"
            elif isinstance(net.input_mapper, ProjectedSparseInputMapper):
                effective_write = "sparse_proj"
            elif isinstance(net.input_mapper, InputMapper):
                effective_write = "dense"
            else:
                effective_write = type(net.input_mapper).__name__
            f.write(f"  write_mode: {effective_write}\n")
            f.write(f"  input_mapper: {type(net.input_mapper).__name__}\n")
            if effective_write == "fan_out":
                f.write(f"  fan_out_map: {net.input_mapper.fan_out_map}\n")
            f.write(f"  write_idx: {list(net.write_idx) if net.write_idx is not None else None}\n")
            f.write(f"  persistent_drive: {args.persistent_drive}\n")
            f.write(f"  drive_mode: {args.drive_mode}\n")
            f.write(f"  output_mapper: {type(net.output_mapper).__name__}\n")
            f.write(f"  read_idx: {list(net.read_idx) if net.read_idx is not None else None}\n")
            f.write(f"  hid_count: {net.hid_count}\n")
            f.write(f"  proj_count: {net.proj_count}\n")
            f.write(f"  stage_lr_scale: {args.stage_lr_scale}\n")
            f.write(f"  mapper_lr_scale: {args.mapper_lr_scale}\n")
            f.write(f"  struct_lr_scale: {args.struct_lr_scale}\n")
            f.write(f"  dyn_lr_scale: {args.dyn_lr_scale}\n")
            f.write(f"  retrain_stage_lr_scale: {args.retrain_stage_lr_scale}\n")
            f.write(f"  retrain_mapper_lr_scale: {args.retrain_mapper_lr_scale}\n")
            f.write(f"  freeze_mappers: {args.freeze_mappers}\n")
            if getattr(net, "skip_linear_enabled", False):
                f.write("\nSKIP-LINEAR CONNECTION (enabled):\n")
                f.write(f"  in_dim: {net.skip_linear_in_dim}\n")
                f.write(f"  out_dim: {net.skip_linear_out_dim}\n")
                f.write(f"  weight shape: {tuple(net.skip_linear.weight.shape)}\n")
                f.write(f"  bias shape: {tuple(net.skip_linear.bias.shape)}\n")
                f.write(
                    f"  l2_lambda (effective): "
                    f"{lambdas.get('skip_linear_l2', 0.0)}\n"
                )
        else:
            f.write(f"  write_mode: {args.write_mode}\n")
            f.write(f"  read_mode: {args.read_mode}\n")
            f.write(f"  write_idx: {args.write_idx}\n")
            f.write(f"  read_idx: {args.read_idx}\n")
            if getattr(args, "skip_linear", False):
                f.write("\nSKIP-LINEAR CONNECTION (enabled):\n")
                f.write(
                    f"  l2_lambda: "
                    f"{lambdas.get('skip_linear_l2', 0.0)}\n"
                )

def _run_noise_diagnostics(
    raw_net,
    val_loader,
    task_fn,
    ctx_factory,
    device,
    args,
    out_dir,
    best_epoch,
    best_val,
    best_metric_name,
    compile_enabled,
    schedule_mode,
    needs_prune):
    """Collect 8 diagnostic data points at the post-training noise-eval site.

    Goal: isolate why `clean_val_mse` in noise_metrics.txt diverges from
    the last validation loss in loss_history.txt (observed for smooth2d_grid
    but not housing_grid). Writes a structured `noise_diagnostics.txt`.

    Diagnostics:
      D1 validate(raw_net) directly — training path
      D2 evaluate_kirchhoff_clean via fresh wrapper — noise path
      D3 tensor-level output comparison on one batch
      D4 checkpoint metadata (best_epoch, best_val, best_metric_name)
      D5 raw_net state_dict hash vs saved model.pt
      D6 device location of model and input tensors
      D7 torch.compile state of cell_lib
    """
    import hashlib
    from analog_noise import NoiseConfig
    from kirchhoff_noise import KirchhoffNetNoiseWrapper, evaluate_kirchhoff_clean

    diag: dict[str, object] = {}
    diag["problem"] = str(args.problem)
    diag["needs_prune"] = bool(needs_prune)
    diag["compile_enabled"] = bool(compile_enabled)

    # D4 — checkpoint metadata
    diag["D4_best_epoch"] = int(best_epoch)
    diag["D4_best_val"] = float(best_val)
    diag["D4_best_metric_name"] = str(best_metric_name)

    # D6 — device location
    try:
        diag["D6_raw_net_device"] = str(next(raw_net.parameters()).device)
    except Exception as e:
        diag["D6_raw_net_device"] = f"ERR: {e}"
    try:
        diag["D6_input_device"] = str(next(iter(val_loader))[0].device)
    except Exception as e:
        diag["D6_input_device"] = f"ERR: {e}"

    # D5 — model state hash vs saved model.pt
    try:
        raw_state = raw_net.state_dict()
        state_keys = sorted(raw_state.keys())
        flat = []
        for k in state_keys:
            flat.append(k)
            flat.append(raw_state[k].detach().cpu().reshape(-1).tolist())
        state_str = repr(flat)
        diag["D5_raw_net_hash"] = hashlib.sha256(state_str.encode()).hexdigest()
        diag["D5_raw_net_keys"] = len(state_keys)

        pt_path = out_dir / "model.pt"
        if pt_path.exists():
            pt_state = torch.load(pt_path, map_location="cpu")
            pt_keys = sorted(pt_state.keys())
            pt_flat = []
            for k in pt_keys:
                pt_flat.append(k)
                pt_flat.append(pt_state[k].detach().cpu().reshape(-1).tolist())
            pt_str = repr(pt_flat)
            diag["D5_model_pt_hash"] = hashlib.sha256(pt_str.encode()).hexdigest()
            diag["D5_model_pt_keys"] = len(pt_keys)
            diag["D5_state_matches_model_pt"] = bool(
                diag["D5_raw_net_hash"] == diag["D5_model_pt_hash"]
            )
        else:
            diag["D5_model_pt_hash"] = "MISSING"
            diag["D5_state_matches_model_pt"] = False
    except Exception as e:
        diag["D5_error"] = f"ERR: {e}"

    # D7 — compile state of cell_lib
    try:
        diag["D7_compile_enabled"] = bool(compile_enabled)
        stages = list(raw_net.core.stages)
        compiled_per_stage = []
        shared_libs = []
        for i, s in enumerate(stages):
            lib = getattr(s, "cell_lib", None)
            is_compiled = lib is not None and hasattr(lib.forward, "__wrapped__")
            compiled_per_stage.append(bool(is_compiled))
        for i in range(len(stages)):
            for j in range(i + 1, len(stages)):
                if stages[i].cell_lib is stages[j].cell_lib:
                    shared_libs.append(f"{i}-{j}")
        diag["D7_cell_lib_compiled_per_stage"] = compiled_per_stage
        diag["D7_shared_cell_lib_pairs"] = shared_libs
    except Exception as e:
        diag["D7_error"] = f"ERR: {e}"

    # D1 — direct clean validation (training path)
    # Ensure raw_net is on `device` before validation. After the prune
    # pipeline (needs_prune=True) raw_net is moved to CPU to free GPU
    # memory (see line ~3646); validation would otherwise fail with
    # "Input type (CUDA) and weight type (CPU) do not match".
    try:
        raw_net.to(device)
        raw_net.eval()
        v1 = validate(
            raw_net, val_loader, task_fn, ctx_factory, device,
            solver="heun")
        diag["D1_validate_raw_net"] = float(v1)
    except Exception as e:
        diag["D1_validate_raw_net"] = f"ERR: {e}"

    # D2 — clean evaluation via wrapper (noise eval path)
    try:
        eval_cfg = NoiseConfig(
            quant_bits=args.quant_bits,
            noise_std=args.noise_std,
            quantize_input=True,
            quantize_output=True,
            quantize_intermediate=True,
            weight_noise=True,
            activation_noise=True,
            mc_trials=args.mc_trials,
            seed=args.noise_seed)
        diag_wrapper = KirchhoffNetNoiseWrapper(
            raw_net, eval_cfg, adc_full_range=args.adc_full_range)
        diag_wrapper.to(device)
        diag_wrapper.eval()
        v2 = evaluate_kirchhoff_clean(
            diag_wrapper, val_loader, task_fn, ctx_factory, device)
        diag["D2_evaluate_kirchhoff_clean"] = float(v2)
    except Exception as e:
        diag["D2_evaluate_kirchhoff_clean"] = f"ERR: {e}"

    # D3 — tensor-level output comparison on one batch
    try:
        u_sample, _ = next(iter(val_loader))
        u_sample = u_sample[:8].to(device)
        ctx_sample = ctx_factory(u_sample.size(0), device=device)
        with torch.no_grad():
            out_raw, _ = raw_net(
                u_sample, ctx=ctx_sample, store_trajectory=False,
                solver="heun")
            out_wrap, _ = diag_wrapper(
                u_sample, ctx=ctx_sample)
        diff = (out_raw - out_wrap).abs()
        diag["D3_max_output_diff"] = float(diff.max().item())
        diag["D3_mean_output_diff"] = float(diff.mean().item())
        diag["D3_raw_out_sample"] = [float(x) for x in out_raw[:3].flatten()[:6].tolist()]
        diag["D3_wrap_out_sample"] = [float(x) for x in out_wrap[:3].flatten()[:6].tolist()]
    except Exception as e:
        diag["D3_error"] = f"ERR: {e}"

    # Write diagnostics
    diag_path = out_dir / "noise_diagnostics.txt"
    with open(diag_path, "w") as f:
        f.write("# Noise Evaluation Diagnostics\n")
        f.write(f"# problem: {args.problem}\n")
        f.write(f"# needs_prune: {needs_prune}\n")
        f.write(f"# compile_enabled: {compile_enabled}\n\n")
        for key, val in diag.items():
            f.write(f"{key}: {val}\n")
    print(f"[diag] wrote {diag_path}")

def _run_noise_evaluation(
    base_net,
    val_loader,
    task_fn,
    ctx_factory,
    device,
    args,
    out_dir: Path,
    label: str,
    metric_name: str = "mse") -> dict:
    """Run kirchhoff-noise MC evaluation on ``base_net`` and write metrics.

    Wraps ``base_net`` in ``KirchhoffNetNoiseWrapper``, runs a clean eval
    plus ``args.mc_trials`` noisy trials, and writes
    ``out_dir / f"noise_metrics_{label}.txt"`` (or
    ``noise_metrics.txt`` when ``label == "main"``).

    Both the clean and noisy passes are evaluated without noise injection.
    Legacy (non-phased) models
    trained purely with soft cells will therefore be evaluated in STE
    mode — flag this in any legacy-result reporting.

    Returns:
        dict with keys ``clean_val`` (float), ``noise_mean``,
        ``noise_std``, ``noise_p50``, ``noise_p90``, ``noise_p95``,
        ``degradation_mean``, ``per_trial_losses``.
    """
    from analog_noise import NoiseConfig
    from kirchhoff_noise import (
        KirchhoffNetNoiseWrapper,
        evaluate_kirchhoff_clean,
        evaluate_kirchhoff_with_noise)

    eval_cfg = NoiseConfig(
        quant_bits=args.quant_bits,
        noise_std=args.noise_std,
        quantize_input=True,
        quantize_output=True,
        quantize_intermediate=True,
        weight_noise=True,
        activation_noise=True,
        mc_trials=args.mc_trials,
        seed=args.noise_seed)
    eval_wrapper = KirchhoffNetNoiseWrapper(
        base_net, eval_cfg, adc_full_range=args.adc_full_range)
    eval_wrapper.to(device)
    eval_wrapper.eval()

    print(
        f"[noise] {label}: running MC noise eval: quant_bits={args.quant_bits} "
        f"noise_std={args.noise_std} trials={args.mc_trials} "
        f"adc_full_range={args.adc_full_range} seed={args.noise_seed}"
    )
    clean_val = evaluate_kirchhoff_clean(
        eval_wrapper, val_loader, task_fn, ctx_factory, device)
    result = evaluate_kirchhoff_with_noise(
        eval_wrapper, val_loader, task_fn, ctx_factory, eval_cfg, device)
    result.clean_loss = clean_val
    degradation = result.mean - clean_val
    print(
        f"[noise] {label}: clean={clean_val:.6f} "
        f"noisy_mean={result.mean:.6f} noisy_std={result.std:.6f} "
        f"p90={result.p90:.6f} p95={result.p95:.6f} "
        f"degradation_mean={degradation:+.6f}"
    )

    suffix = "" if label == "main" else f"_{label}"
    metrics_path = out_dir / f"noise_metrics{suffix}.txt"
    with open(metrics_path, "w") as f:
        f.write(f"quant_bits: {args.quant_bits}\n")
        f.write(f"noise_std: {args.noise_std}\n")
        f.write(f"mc_trials: {args.mc_trials}\n")
        f.write(f"adc_full_range: {args.adc_full_range}\n")
        f.write(f"noise_seed: {args.noise_seed}\n")
        f.write(f"noise_aware_training: {bool(args.noise_aware)}\n")
        f.write(f"label: {label}\n")
        f.write(f"clean_val_{metric_name}: {clean_val:.6f}\n")
        f.write(f"noisy_mean: {result.mean:.6f}\n")
        f.write(f"noisy_std: {result.std:.6f}\n")
        f.write(f"noisy_p50: {result.p50:.6f}\n")
        f.write(f"noisy_p90: {result.p90:.6f}\n")
        f.write(f"noisy_p95: {result.p95:.6f}\n")
        f.write(f"noisy_best: {result.best:.6f}\n")
        f.write(f"noisy_worst: {result.worst:.6f}\n")
        f.write(f"degradation_mean: {degradation:.6f}\n")
        f.write("per_trial_losses:\n")
        for i, l in enumerate(result.losses):
            f.write(f"  trial_{i:03d}: {l:.6f}\n")

    return {
        "clean_val": clean_val,
        "noise_mean": result.mean,
        "noise_std": result.std,
        "noise_p50": result.p50,
        "noise_p90": result.p90,
        "noise_p95": result.p95,
        "degradation_mean": degradation,
        "per_trial_losses": result.losses}

def make_data_sinx(batch_size: int, val_size: int = 1024):
    train_size = 8192
    u_train = (torch.rand(train_size, 1) * 2 * torch.pi) - torch.pi
    y_train = torch.sin(u_train)
    u_val = (torch.rand(val_size, 1) * 2 * torch.pi) - torch.pi
    y_val = torch.sin(u_val)

    train_loader = DataLoader(
        TensorDataset(u_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        TensorDataset(u_val, y_val), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader, F.mse_loss

def _load_california_housing_data() -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Load raw California housing: (X, y_normalized, y_mean, y_std).

    Targets are standardized (subtract mean, divide by std). Features
    are returned *unnormalized* so that each caller can apply its own
    normalization scheme.
    """
    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for the housing preset. "
            "Install with: uv pip install scikit-learn"
        ) from e

    data = fetch_california_housing()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)
    y_mean = float(y.mean().item())
    y_std = float(y.std().clamp(min=1e-6).item())
    y_norm = (y - y_mean) / y_std
    return X, y_norm, y_mean, y_std

def _make_data_split(
    X: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Deterministic 80/20 train/val split + DataLoaders."""
    n = X.shape[0]
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    n_train = int(0.8 * n)
    train_ds = TensorDataset(X[perm[:n_train]], y[perm[:n_train]])
    val_ds = TensorDataset(X[perm[n_train:]], y[perm[n_train:]])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False))

def make_data_housing(batch_size: int):
    """California-housing regression on the line topology.

    Features are min-max scaled to [0, 1] per column. This handles both
    non-negative features (Population) and signed features (Longitude ~-124
    to -114): dividing by column max alone would clamp the negative
    Longitude max to 1e-6 and produce values of ~-1e8 that overflow float16
    under AMP (zeroing all gradients via 0*inf=NaN in the backward pass).
    Targets are standardized. Training loss is Huber (delta=1.0).

    Returns ``(train_loader, val_loader, task_fn, inverse_stats)`` where
    ``inverse_stats`` is ``{"y_mean": ..., "y_std": ...}`` for
    denormalizing validation predictions back to original units.
    """
    X, y_norm, y_mean, y_std = _load_california_housing_data()

    x_min = X.min(dim=0, keepdim=True).values
    x_max = X.max(dim=0, keepdim=True).values
    x_range = (x_max - x_min).clamp(min=1e-6)
    X = (X - x_min) / x_range

    train_loader, val_loader = _make_data_split(X, y_norm, batch_size)

    def task_fn(y_pred, y_target):
        return F.huber_loss(y_pred, y_target, delta=1.0)

    inverse_stats = {"y_mean": y_mean, "y_std": y_std}
    return train_loader, val_loader, task_fn, inverse_stats

def make_data_housing_grid(batch_size: int, huber_delta: float = 1.0):
    """California-housing regression on the 5x5 grid topology.

    Features are min-max scaled to [0, 1] per column (handles both
    non-negative features like ``Population`` and signed features like
    ``Longitude``). Targets are standardized. Training loss is Huber
    (delta=1.0), blending MSE's smooth gradients for small errors with
    L1's robustness to heavy tails in housing prices.

    Returns ``(train_loader, val_loader, task_fn, inverse_stats)`` where
    ``inverse_stats`` is ``{"y_mean": ..., "y_std": ...}`` for
    denormalizing validation predictions back to original units.
    """
    X, y_norm, y_mean, y_std = _load_california_housing_data()

    x_min = X.min(dim=0, keepdim=True).values
    x_max = X.max(dim=0, keepdim=True).values
    x_range = (x_max - x_min).clamp(min=1e-6)
    X = (X - x_min) / x_range

    train_loader, val_loader = _make_data_split(X, y_norm, batch_size)

    def task_fn(y_pred, y_target):
        return F.huber_loss(y_pred, y_target, delta=huber_delta)

    inverse_stats = {"y_mean": y_mean, "y_std": y_std}
    return train_loader, val_loader, task_fn, inverse_stats

def denormalize_targets(y_norm: torch.Tensor, inverse_stats: dict) -> torch.Tensor:
    """Map standardized targets back to original California-housing units."""
    return y_norm * inverse_stats["y_std"] + inverse_stats["y_mean"]


# Friedman synthetic regression tasks (friedman-problems/REQ).
# Canonical Friedman 1991 MARS paper formulas. Constants match the MLP
# benchmark scripts in mlp_benchmark_friedman{1,2,3}.py so train/val
# targets are directly comparable across model families.

_FRIEDMAN1_PI = _math.pi
_FRIEDMAN1_RELEVANT = 5   # only x1..x5 carry signal; x6..x10 are noise
_FRIEDMAN1_IN_DIM = 10
_FRIEDMAN2_IN_DIM = 4
_FRIEDMAN3_IN_DIM = 4
# Per-dim ranges for Friedman #2 and #3 (x1, x2, x3, x4).
_FRIEDMAN2_RANGES = [
    (0.0, 100.0),
    (40.0 * _math.pi, 560.0 * _math.pi),
    (0.0, 1.0),
    (1.0, 11.0),
]
_FRIEDMAN3_RANGES = _FRIEDMAN2_RANGES


def _scale_lhs_to_ranges(u_unit: torch.Tensor, ranges: list[tuple[float, float]]) -> torch.Tensor:
    """Linearly rescale unit-cube LHS samples to per-dim [lo, hi] ranges."""
    lo = torch.tensor([r[0] for r in ranges], dtype=u_unit.dtype, device=u_unit.device)
    hi = torch.tensor([r[1] for r in ranges], dtype=u_unit.dtype, device=u_unit.device)
    return lo + u_unit * (hi - lo)


def _friedman1(x: torch.Tensor) -> torch.Tensor:
    """Friedman #1 deterministic target.

    Args:
        x: Tensor of shape (..., 10). Only the first 5 columns carry signal.
    Returns:
        Tensor of shape (...) with the noise-free target.
    """
    if x.shape[-1] < _FRIEDMAN1_RELEVANT:
        raise ValueError(
            f"_friedman1 requires at least {_FRIEDMAN1_RELEVANT} input columns, "
            f"got {x.shape[-1]}"
        )
    x1, x2, x3, x4, x5 = x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4]
    return (
        10.0 * torch.sin(_FRIEDMAN1_PI * x1 * x2)
        + 20.0 * (x3 - 0.5) ** 2
        + 10.0 * x4
        + 5.0 * x5
    )


def _friedman2(x: torch.Tensor) -> torch.Tensor:
    """Friedman #2 deterministic target: sqrt(x1^2 + (x2*x3 - 1/(x2*x4))^2)."""
    if x.shape[-1] != _FRIEDMAN2_IN_DIM:
        raise ValueError(
            f"_friedman2 requires exactly {_FRIEDMAN2_IN_DIM} input columns, "
            f"got {x.shape[-1]}"
        )
    x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    inner = (x2 * x3) - (1.0 / (x2 * x4))
    return torch.sqrt(x1 ** 2 + inner ** 2)


def _friedman3(x: torch.Tensor) -> torch.Tensor:
    """Friedman #3 deterministic target: atan((x2*x3 - 1/(x2*x4)) / x1)."""
    if x.shape[-1] != _FRIEDMAN3_IN_DIM:
        raise ValueError(
            f"_friedman3 requires exactly {_FRIEDMAN3_IN_DIM} input columns, "
            f"got {x.shape[-1]}"
        )
    x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    inner = (x2 * x3) - (1.0 / (x2 * x4))
    return torch.atan(inner / x1)


def _minmax_normalize_inputs(
    u_train: torch.Tensor, u_val: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-dim min-max normalize features to [0, 1] using train-set stats.

    input-norm-seed/Phase 2: maps each input feature to [0, 1] using min/max
    computed on the *training* set only. Val is normalized with the same
    scaler. Stats are returned for the caller's inverse_stats dict.
    """
    u_min = u_train.amin(dim=0, keepdim=True)
    u_max = u_train.amax(dim=0, keepdim=True)
    u_range = (u_max - u_min).clamp(min=1e-8)
    u_train_n = (u_train - u_min) / u_range
    u_val_n = (u_val - u_min) / u_range
    return u_train_n, u_val_n, u_min, u_range


def make_data_friedman1(batch_size: int, noise_std: float = 1.0, val_size: int = 4000,
                       normalize_inputs: bool = True):
    """Friedman #1 regression on the 5x5 torus topology.

    Returns ``(train_loader, val_loader, task_fn, inverse_stats)``. The
    ``task_fn`` is ``F.huber_loss(o, t, delta=1.0)``. ``inverse_stats``
    contains ``"y_mean"`` and ``"y_std"`` keys for target denormalization,
    plus ``"u_min"`` and ``"u_range"`` keys (only present when
    ``normalize_inputs=True``) for downstream input-distribution analysis.
    Inputs are LHS samples in [0, 1]^10; min-max normalization is a no-op
    for this problem but kept for interface consistency.
    """
    n_train = 20000
    u_train = _lhs_samples(n_train, _FRIEDMAN1_IN_DIM, seed=42)
    y_train = _friedman1(u_train).unsqueeze(1)
    torch.manual_seed(42)
    u_val = torch.rand(val_size, _FRIEDMAN1_IN_DIM)
    y_val = _friedman1(u_val).unsqueeze(1)
    if noise_std > 0:
        y_train = y_train + noise_std * torch.randn_like(y_train)
    if normalize_inputs:
        u_train, u_val, u_min, u_range = _minmax_normalize_inputs(u_train, u_val)
    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-6)
    train_loader = DataLoader(
        TensorDataset(u_train, (y_train - y_mean) / y_std),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(u_val, (y_val - y_mean) / y_std),
        batch_size=batch_size, shuffle=False,
    )
    inverse_stats = {
        "y_mean": float(y_mean.item()),
        "y_std": float(y_std.item()),
    }
    if normalize_inputs:
        inverse_stats["u_min"] = u_min.squeeze(0).tolist()
        inverse_stats["u_range"] = u_range.squeeze(0).tolist()
    return train_loader, val_loader, F.huber_loss, inverse_stats


def make_data_friedman2(batch_size: int, noise_std: float = 1.0, val_size: int = 4000,
                       normalize_inputs: bool = True):
    """Friedman #2 regression on the 4x4 torus topology.

    Inputs are LHS samples scaled to per-dim ranges via ``_scale_lhs_to_ranges``,
    then (when ``normalize_inputs=True``) per-dim min-max normalized to [0, 1]
    from training statistics. Set ``normalize_inputs=False`` to ablate.
    Returns ``(train_loader, val_loader, task_fn, inverse_stats)`` where
    ``inverse_stats`` carries ``"y_mean"`` and ``"y_std"`` for target
    denormalization and (when normalized) ``"u_min"``/``"u_range"`` for
    downstream input-distribution analysis.
    """
    n_train = 20000
    u_train_unit = _lhs_samples(n_train, _FRIEDMAN2_IN_DIM, seed=42)
    u_train = _scale_lhs_to_ranges(u_train_unit, _FRIEDMAN2_RANGES)
    y_train = _friedman2(u_train).unsqueeze(1)
    torch.manual_seed(42)
    u_val_unit = torch.rand(val_size, _FRIEDMAN2_IN_DIM)
    u_val = _scale_lhs_to_ranges(u_val_unit, _FRIEDMAN2_RANGES)
    y_val = _friedman2(u_val).unsqueeze(1)
    if noise_std > 0:
        y_train = y_train + noise_std * torch.randn_like(y_train)
    if normalize_inputs:
        u_train, u_val, u_min, u_range = _minmax_normalize_inputs(u_train, u_val)
    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-6)
    train_loader = DataLoader(
        TensorDataset(u_train, (y_train - y_mean) / y_std),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(u_val, (y_val - y_mean) / y_std),
        batch_size=batch_size, shuffle=False,
    )
    inverse_stats = {
        "y_mean": float(y_mean.item()),
        "y_std": float(y_std.item()),
    }
    if normalize_inputs:
        inverse_stats["u_min"] = u_min.squeeze(0).tolist()
        inverse_stats["u_range"] = u_range.squeeze(0).tolist()
    return train_loader, val_loader, F.huber_loss, inverse_stats


def make_data_friedman3(batch_size: int, noise_std: float = 1.0, val_size: int = 4000,
                       normalize_inputs: bool = True):
    """Friedman #3 regression on the 4x4 torus topology (same shape as #2).

    Inputs are LHS samples scaled to per-dim ranges via ``_scale_lhs_to_ranges``,
    then (when ``normalize_inputs=True``) per-dim min-max normalized to [0, 1]
    from training statistics. Set ``normalize_inputs=False`` to ablate.
    Returns ``(train_loader, val_loader, task_fn, inverse_stats)`` where
    ``inverse_stats`` carries ``"y_mean"`` and ``"y_std"`` for target
    denormalization and (when normalized) ``"u_min"``/``"u_range"`` for
    downstream input-distribution analysis.
    """
    n_train = 20000
    u_train_unit = _lhs_samples(n_train, _FRIEDMAN3_IN_DIM, seed=42)
    u_train = _scale_lhs_to_ranges(u_train_unit, _FRIEDMAN3_RANGES)
    y_train = _friedman3(u_train).unsqueeze(1)
    torch.manual_seed(42)
    u_val_unit = torch.rand(val_size, _FRIEDMAN3_IN_DIM)
    u_val = _scale_lhs_to_ranges(u_val_unit, _FRIEDMAN3_RANGES)
    y_val = _friedman3(u_val).unsqueeze(1)
    if noise_std > 0:
        y_train = y_train + noise_std * torch.randn_like(y_train)
    if normalize_inputs:
        u_train, u_val, u_min, u_range = _minmax_normalize_inputs(u_train, u_val)
    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-6)
    train_loader = DataLoader(
        TensorDataset(u_train, (y_train - y_mean) / y_std),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(u_val, (y_val - y_mean) / y_std),
        batch_size=batch_size, shuffle=False,
    )
    inverse_stats = {
        "y_mean": float(y_mean.item()),
        "y_std": float(y_std.item()),
    }
    if normalize_inputs:
        inverse_stats["u_min"] = u_min.squeeze(0).tolist()
        inverse_stats["u_range"] = u_range.squeeze(0).tolist()
    return train_loader, val_loader, F.huber_loss, inverse_stats


def _franke(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    t1 = -((9 * x1 - 2) ** 2) / 4 - ((9 * x2 - 2) ** 2) / 4
    t2 = -(9 * x1 + 1) ** 2 / 49 - (9 * x2 + 1) / 10
    t3 = -((9 * x1 - 7) ** 2) / 4 - ((9 * x2 - 3) ** 2) / 4
    t4 = -(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2
    return (
        0.75 * t1.exp()
        + 0.75 * t2.exp()
        + 0.5 * t3.exp()
        - 0.2 * t4.exp()
    )

def _lhs_samples(n: int, d: int, seed: int = 42) -> torch.Tensor:
    """Latin Hypercube samples in [0, 1]^d with guaranteed stratification."""
    from scipy.stats.qmc import LatinHypercube
    sampler = LatinHypercube(d=d, seed=seed)
    return torch.from_numpy(sampler.random(n=n)).float()

def make_data_smooth2d(batch_size: int, val_size: int = 4000, huber_delta: float = 1.0):
    # Fixed seed for reproducible train/val splits and noise across runs.
    n_train = 20000
    u_train = _lhs_samples(n_train, 2, seed=42)
    y_train = _franke(u_train[:, 0], u_train[:, 1]).unsqueeze(1)
    torch.manual_seed(42)
    u_val = torch.rand(val_size, 2)
    y_val = _franke(u_val[:, 0], u_val[:, 1]).unsqueeze(1)

    noise_std = 0.01
    y_train = y_train + noise_std * torch.randn_like(y_train)

    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-6)
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    train_loader = DataLoader(
        TensorDataset(u_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(u_val, y_val), batch_size=batch_size, shuffle=False
    )
    inverse_stats = {"y_mean": float(y_mean.item()), "y_std": float(y_std.item())}
    return train_loader, val_loader, lambda o, t: F.huber_loss(o, t, delta=huber_delta), inverse_stats

def make_data(problem: str, batch_size: int, noise_std: float = 0.0,
              normalize_inputs: bool = True):
    """Dispatch to the per-problem data factory.

    ``noise_std`` is forwarded only to the friedman problems (which use
    additive Gaussian target noise); other problems keep their existing
    noise conventions (housing/smooth2d have their own noise logic).

    ``normalize_inputs`` is forwarded only to friedman problems; ignored for
    others. When False, friedman2/3 inputs stay in their raw per-dim ranges
    (no min-max scaling). friedman1 is a no-op either way.
    """
    if problem == "sinx":
        return make_data_sinx(batch_size)
    if problem == "housing":
        return make_data_housing(batch_size)
    if problem == "smooth2d":
        return make_data_smooth2d(batch_size)
    if problem == "smooth2d_grid":
        return make_data_smooth2d(batch_size)
    if problem == "housing_grid":
        return make_data_housing_grid(batch_size)
    if problem == "friedman1":
        return make_data_friedman1(batch_size, noise_std=noise_std,
                                   normalize_inputs=normalize_inputs)
    if problem == "friedman2":
        return make_data_friedman2(batch_size, noise_std=noise_std,
                                   normalize_inputs=normalize_inputs)
    if problem == "friedman3":
        return make_data_friedman3(batch_size, noise_std=noise_std,
                                   normalize_inputs=normalize_inputs)
    raise ValueError(f"Unknown problem: {problem}")

def _unwrap_raw_net(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net

def _deq_batch_stats(raw_net, *, solver: str, deq_cfg: dict | None) -> dict | None:
    """Compute cached DEQ residual stats for the most recent forward pass."""
    core = raw_net.core if hasattr(raw_net, "core") else raw_net
    stage_outputs = getattr(core, "last_stage_outputs", None) or []
    stage_infos = getattr(core, "last_stage_infos", None) or []
    stage_ctxs = getattr(core, "last_stage_ctxs", None) or []
    drive_targets = getattr(core, "last_drive_targets", None) or []
    drive_scales = getattr(core, "last_drive_scales", None) or []
    stages = getattr(core, "stages", None) or []
    if solver != "deq" or not stage_outputs or not stages:
        return None

    abs_residual_means = []
    abs_residual_maxes = []
    nsteps = []
    max_abs_state = 0.0
    for i, stage in enumerate(stages):
        if i >= len(stage_outputs):
            break
        info = stage_infos[i] if i < len(stage_infos) else {}
        if not info:
            continue
        x_stage = stage_outputs[i]
        stage_ctx = stage_ctxs[i] if i < len(stage_ctxs) else None
        x_drive = drive_targets[i] if i < len(drive_targets) else None
        drive_scale = float(drive_scales[i]) if i < len(drive_scales) else 0.0
        tau_i = float(info.get("tau", 1.0))
        
        leak_floor = float(info.get("leak_floor", 0.0))
        deq_step = float(info.get("deq_step", (deq_cfg or {}).get("deq_step", 0.1)))

        with torch.no_grad():
            rhs = stage.rhs(
                x_stage,
                x_drive=x_drive,
                drive_scale=drive_scale,
                leak_floor=leak_floor)
        residual = deq_step * rhs
        abs_residual = residual.abs()
        abs_residual_means.append(float(abs_residual.mean().item()))
        abs_residual_maxes.append(float(abs_residual.max().item()))
        max_abs_state = max(max_abs_state, float(x_stage.abs().max().item()))

        nstep = info.get("nstep", 0)
        if torch.is_tensor(nstep):
            nstep = int(nstep.flatten()[0].item())
        else:
            nstep = int(nstep)
        nsteps.append(nstep)

    if not abs_residual_means:
        return None

    return {
        "residual_mean": float(sum(abs_residual_means) / len(abs_residual_means)),
        "residual_max": float(max(abs_residual_maxes)),
        "nstep_mean": float(sum(nsteps) / len(nsteps)) if nsteps else 0.0,
        "nstep_max": float(max(nsteps)) if nsteps else 0.0,
        "max_abs_state": float(max_abs_state)}

def _deq_gradient_probe(
    net,
    u: torch.Tensor,
    target: torch.Tensor,
    ctx,
    task_fn,
    *,
    solver: str,
    deq_cfg: dict | None) -> dict | None:
    """Run one gradient probe batch and summarize mapper / gate norms."""
    if solver != "deq":
        return None
    raw_net = _unwrap_raw_net(net)
    probe_net = raw_net if isinstance(net, torch.nn.DataParallel) else net
    raw_net.zero_grad(set_to_none=True)
    out, _ = probe_net(
        u,
        ctx=ctx,
        tau=1.0,
        store_trajectory=False,
        solver=solver,
        deq_cfg=deq_cfg)
    loss = task_fn(out, target)
    loss.backward()
    norms = collect_gradient_norms(raw_net)
    batch_stats = _deq_batch_stats(raw_net, solver=solver, deq_cfg=deq_cfg)
    raw_net.zero_grad(set_to_none=True)

    stage_logits_sq = 0.0
    z_logits_sq = 0.0
    for key, val in norms.items():
        if val is None:
            continue
        if key.endswith("_logits") and not key.endswith("_z_logits"):
            stage_logits_sq += float(val) ** 2
        elif key.endswith("_z_logits"):
            z_logits_sq += float(val) ** 2
    mapper_sq = 0.0
    if norms.get("in_mapper") is not None:
        mapper_sq += float(norms["in_mapper"]) ** 2
    if norms.get("out_mapper") is not None:
        mapper_sq += float(norms["out_mapper"]) ** 2

    return {
        "mapper_grad_norm": float(mapper_sq ** 0.5),
        "stage_logits_grad_norm": float(stage_logits_sq ** 0.5),
        "z_logits_grad_norm": float(z_logits_sq ** 0.5),
        "probe_loss": float(loss.item()),
        **(batch_stats or {})}

def _format_deq_summary(metrics: dict | None) -> str:
    if not metrics:
        return ""
    parts = [
        f"res={metrics.get('residual_mean', float('nan')):.2e}/{metrics.get('residual_max', float('nan')):.2e}",
        f"nstep={metrics.get('nstep_mean', float('nan')):.1f}/{metrics.get('nstep_max', float('nan')):.0f}",
        f"|x|_max={metrics.get('max_abs_state', float('nan')):.2e}",
    ]
    if "mapper_grad_norm" in metrics:
        parts.append(
            f"grad={metrics.get('mapper_grad_norm', float('nan')):.2e}/"
            f"{metrics.get('stage_logits_grad_norm', float('nan')):.2e}/"
            f"{metrics.get('z_logits_grad_norm', float('nan')):.2e}"
        )
    return "  DEQ[" + " ".join(parts) + "]"

def _append_deq_validation_row(
    path,
    *,
    epoch: int,
    phase: str,
    split: str,
    train_loss: float,
    val_loss: float,
    metrics: dict | None) -> None:
    if metrics is None:
        return
    new_file = not path.exists()
    with open(path, "a") as f:
        if new_file:
            f.write(
                "epoch\tphase\tsplit\ttrain\tval\tresidual_mean\tresidual_max\t"
                "nstep_mean\tnstep_max\tmax_abs_state\tmapper_grad_norm\t"
                "stage_logits_grad_norm\tz_logits_grad_norm\tprobe_loss\n"
            )
        f.write(
            f"{epoch}\t{phase}\t{split}\t{train_loss:.6f}\t{val_loss:.6f}\t"
            f"{metrics.get('residual_mean', float('nan')):.6e}\t"
            f"{metrics.get('residual_max', float('nan')):.6e}\t"
            f"{metrics.get('nstep_mean', float('nan')):.6f}\t"
            f"{metrics.get('nstep_max', float('nan')):.6f}\t"
            f"{metrics.get('max_abs_state', float('nan')):.6e}\t"
            f"{metrics.get('mapper_grad_norm', float('nan')):.6e}\t"
            f"{metrics.get('stage_logits_grad_norm', float('nan')):.6e}\t"
            f"{metrics.get('z_logits_grad_norm', float('nan')):.6e}\t"
            f"{metrics.get('probe_loss', float('nan')):.6f}\n"
        )

def _append_deq_train_row(
    path,
    *,
    epoch: int,
    phase: str,
    train_loss: float,
    metrics: dict | None) -> None:
    if metrics is None:
        return
    _append_deq_validation_row(
        path,
        epoch=epoch,
        phase=phase,
        split="train",
        train_loss=train_loss,
        val_loss=train_loss,
        metrics=metrics)

def validate(net, val_loader, task_fn, ctx_factory, device,
             solver: str = "heun", deq_cfg: dict | None = None,
             collect_deq_metrics: bool = False):
    net.eval()
    use_cached_deq_stats = collect_deq_metrics and solver == "deq" and not isinstance(net, torch.nn.DataParallel)
    total = 0.0
    n = 0
    deq_sum_residual = 0.0
    deq_max_residual = 0.0
    deq_sum_nstep = 0.0
    deq_max_nstep = 0.0
    deq_max_abs_state = 0.0
    first_batch = None
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            ctx = ctx_factory(u.size(0), device=device)
            out, _ = net(u, store_trajectory=False,
                         solver=solver, deq_cfg=deq_cfg)
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            n += u.size(0)
            if use_cached_deq_stats:
                raw = _unwrap_raw_net(net)
                batch_stats = _deq_batch_stats(raw, solver=solver, deq_cfg=deq_cfg)
                if batch_stats is not None:
                    deq_sum_residual += batch_stats["residual_mean"] * u.size(0)
                    deq_max_residual = max(deq_max_residual, batch_stats["residual_max"])
                    deq_sum_nstep += batch_stats["nstep_mean"] * u.size(0)
                    deq_max_nstep = max(deq_max_nstep, batch_stats["nstep_max"])
                    deq_max_abs_state = max(deq_max_abs_state, batch_stats["max_abs_state"])
            if first_batch is None:
                first_batch = (u.detach(), target.detach(), ctx)
    net.train()
    val_loss = total / max(1, n)
    if not collect_deq_metrics or solver != "deq":
        return val_loss

    probe_metrics = None
    if first_batch is not None:
        probe_metrics = _deq_gradient_probe(
            net,
            first_batch[0],
            first_batch[1],
            first_batch[2],
            task_fn,
            solver=solver,
            deq_cfg=deq_cfg)
    deq_metrics = {
        "residual_mean": deq_sum_residual / max(1, n),
        "residual_max": deq_max_residual,
        "nstep_mean": deq_sum_nstep / max(1, n),
        "nstep_max": deq_max_nstep,
        "max_abs_state": deq_max_abs_state}
    if probe_metrics is not None:
        for key, value in probe_metrics.items():
            if key in ("mapper_grad_norm", "stage_logits_grad_norm", "z_logits_grad_norm", "probe_loss"):
                deq_metrics[key] = value
            elif key not in deq_metrics or deq_metrics[key] == 0.0:
                deq_metrics[key] = value
    return val_loss, deq_metrics

def validate_with_inverse(
    net,
    val_loader,
    task_fn,
    ctx_factory,
    device,
    inverse_stats: dict | None,
    solver: str = "heun",
    deq_cfg: dict | None = None,
    collect_deq_metrics: bool = False) -> dict:
    """Validation that also reports metrics in the original (denormalized) target units.

    Returns a dict with:
      - ``val``: training-space loss (mean of ``task_fn(out, target)``)
      - ``mae_orig``: MAE in original target units (USD x 100k for California housing)
      - ``rmse_orig``: RMSE in original target units
      - ``mse_orig``: MSE in original target units
      - ``mape_orig``: MAPE in original target units (percent). MAPE clips
        ``|y_true|`` at ``1e-8`` to avoid division by zero.
    If ``inverse_stats`` is None, ``mae_orig`` / ``rmse_orig`` / ``mse_orig`` /
    ``mape_orig`` are reported as NaN (no denormalization available).
    """
    net.eval()
    use_cached_deq_stats = collect_deq_metrics and solver == "deq" and not isinstance(net, torch.nn.DataParallel)
    total = 0.0
    n = 0
    se_sum = 0.0
    ae_sum = 0.0
    ape_sum = 0.0
    deq_sum_residual = 0.0
    deq_max_residual = 0.0
    deq_sum_nstep = 0.0
    deq_max_nstep = 0.0
    deq_max_abs_state = 0.0
    first_batch = None
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            ctx = ctx_factory(u.size(0), device=device)
            out, _ = net(u, store_trajectory=False,
                         solver=solver, deq_cfg=deq_cfg)
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            if inverse_stats is not None:
                pred_orig = denormalize_targets(out, inverse_stats)
                targ_orig = denormalize_targets(target, inverse_stats)
                err = pred_orig - targ_orig
                abs_err = err.abs()
                ae_sum += float(abs_err.sum().item())
                se_sum += float((err ** 2).sum().item())
                ape_sum += float((abs_err / targ_orig.abs().clamp(min=1e-8)).sum().item())
            n += u.size(0)
            if use_cached_deq_stats:
                raw = _unwrap_raw_net(net)
                batch_stats = _deq_batch_stats(raw, solver=solver, deq_cfg=deq_cfg)
                if batch_stats is not None:
                    deq_sum_residual += batch_stats["residual_mean"] * u.size(0)
                    deq_max_residual = max(deq_max_residual, batch_stats["residual_max"])
                    deq_sum_nstep += batch_stats["nstep_mean"] * u.size(0)
                    deq_max_nstep = max(deq_max_nstep, batch_stats["nstep_max"])
                    deq_max_abs_state = max(deq_max_abs_state, batch_stats["max_abs_state"])
            if first_batch is None:
                first_batch = (u.detach(), target.detach(), ctx)
    net.train()
    out_dict = {"val": total / max(1, n)}
    if inverse_stats is not None and n > 0:
        out_dict["mae_orig"] = ae_sum / n
        out_dict["rmse_orig"] = (se_sum / n) ** 0.5
        out_dict["mse_orig"] = se_sum / n
        out_dict["mape_orig"] = (ape_sum / n) * 100.0
    else:
        out_dict["mae_orig"] = float("nan")
        out_dict["rmse_orig"] = float("nan")
        out_dict["mse_orig"] = float("nan")
        out_dict["mape_orig"] = float("nan")
    if collect_deq_metrics and solver == "deq":
        probe_metrics = None
        if first_batch is not None:
            probe_metrics = _deq_gradient_probe(
                net,
                first_batch[0],
                first_batch[1],
                first_batch[2],
                task_fn,
                solver=solver,
                deq_cfg=deq_cfg)
        out_dict["deq"] = {
            "residual_mean": deq_sum_residual / max(1, n),
            "residual_max": deq_max_residual,
            "nstep_mean": deq_sum_nstep / max(1, n),
            "nstep_max": deq_max_nstep,
            "max_abs_state": deq_max_abs_state}
        if probe_metrics is not None:
            for key, value in probe_metrics.items():
                if key in ("mapper_grad_norm", "stage_logits_grad_norm", "z_logits_grad_norm", "probe_loss"):
                    out_dict["deq"][key] = value
                elif key not in out_dict["deq"] or out_dict["deq"][key] == 0.0:
                    out_dict["deq"][key] = value
    return out_dict

def collect_predictions(
    net,
    inputs,
    ctx_factory,
    device,
    *,
    solver: str = "heun",
    deq_cfg: dict | None = None) -> torch.Tensor:
    net.eval()
    with torch.no_grad():
        ctx = ctx_factory(inputs.size(0), device=device)
        out, _ = net(
            inputs,
            ctx=ctx,
            store_trajectory=True,
            solver=solver,
            deq_cfg=deq_cfg)
    net.train()
    return out

def apply_ablation(net, ablation: str) -> None:
    """Wrapper around train.apply_ablation with a [ablation=X] log line."""
    from train import apply_ablation as _apply_ablation
    _apply_ablation(net, ablation)
    print(f"[ablation={ablation}] applied to net")

def _parse_int_list(spec: str | None) -> list[int] | None:
    """Parse '0,3,5' -> [0,3,5]. None passes through."""
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]

def collect_gradient_norms(raw_net):
    """Walk ``raw_net`` (assumed unwrapped from DataParallel) and collect
    per-group L2 gradient norms.

    Returns: dict with keys
      stage{i}_raw_leak, stage{i}_z_logits, stage{i}_u_logits
      (one entry per stage)
      stage_transfer
      in_mapper
      out_mapper
    Each value is either a float (the L2 norm) or None if the group has
    no parameter or no gradient.
    """
    stage_sq = {}
    stage_components = ("raw_leak", "z_logits", "u_logits")
    # Per-edge device parameter suffixes: covers SimpleEdgeLibrary.param
    # (I=ReLU/tanh(p0*Vsrc+p1*Vdest+p2)), RealisticTanhLibrary
    # (alpha_raw, bias_raw), RealisticTanhUpgradeLibrary (alpha_raw,
    # gm_raw, isat_raw, bias_raw), FreeTanhLibrary (a_raw, b_raw, s_raw,
    # gm_raw, isat_raw, theta_raw), and AntiParallelFreeTanhLibrary
    # (kappa_raw, gm_raw, isat_raw, theta_raw). All contribute to the same
    # `device_param` gradient-norm metric per stage.
    device_param_suffixes = (
        "param", "alpha_raw", "bias_raw", "gm_raw", "isat_raw",
        "a_raw", "b_raw", "s_raw", "theta_raw", "kappa_raw")
    transfer_sq = 0.0
    transfer_found = False
    in_sq = 0.0
    in_found = False
    out_sq = 0.0
    out_found = False

    # First pass: register all expected keys (even when grads are None),
    # so callers always find them in the dict.
    # F7: match any ``*_cell_lib.*`` device-family suffix, not just the
    # core ``.cell_lib.``, so boundary / output_ode / ref / shared-sense
    # device params also register into ``stage{i}_device_param``.
    for name, p in raw_net.named_parameters():
        if ".stages." in name:
            for comp in stage_components:
                if name.endswith("." + comp):
                    stage_idx = int(name.split(".stages.")[1].split(".")[0])
                    stage_sq.setdefault(f"stage{stage_idx}_{comp}", 0.0)
            if any(
                name.endswith("." + lib_prefix + "." + s)
                for lib_prefix in ("cell_lib", "boundary_cell_lib",
                                   "output_ode_cell_lib", "ref_cell_lib",
                                   "readout_sense_cell_lib")
                for s in device_param_suffixes
            ):
                stage_idx = int(name.split(".stages.")[1].split(".")[0])
                stage_sq.setdefault(f"stage{stage_idx}_device_param", 0.0)

    # Second pass: accumulate squared norms where gradients exist.
    for name, p in raw_net.named_parameters():
        if p.grad is None:
            continue
        gnorm_sq = float(p.grad.norm().item()) ** 2
        if ".stages." in name:
            for comp in stage_components:
                if name.endswith("." + comp):
                    stage_idx = int(name.split(".stages.")[1].split(".")[0])
                    key = f"stage{stage_idx}_{comp}"
                    stage_sq[key] = stage_sq.get(key, 0.0) + gnorm_sq
            # F7: also accumulate non-core cell families (boundary, mesh,
            # ref, shared-sense) into the stage device-param bucket so
            # grad_log reflects every FreeTanh/Realistic family.
            if any(
                name.endswith("." + lib_prefix + "." + s)
                for lib_prefix in ("cell_lib", "boundary_cell_lib",
                                   "output_ode_cell_lib", "ref_cell_lib",
                                   "readout_sense_cell_lib")
                for s in device_param_suffixes
            ):
                stage_idx = int(name.split(".stages.")[1].split(".")[0])
                key = f"stage{stage_idx}_device_param"
                stage_sq[key] = stage_sq.get(key, 0.0) + gnorm_sq
        if "transfers" in name or "stage_transfer" in name:
            transfer_sq += gnorm_sq
            transfer_found = True
        if "input_mapper" in name:
            in_sq += gnorm_sq
            in_found = True
        elif "output_mapper" in name:
            out_sq += gnorm_sq
            out_found = True

    out = {}
    for k, v in stage_sq.items():
        out[k] = v ** 0.5 if v > 0 else None
    out["stage_transfer"] = transfer_sq ** 0.5 if transfer_found else None
    out["in_mapper"] = in_sq ** 0.5 if in_found else None
    out["out_mapper"] = out_sq ** 0.5 if out_found else None
    return out

def compute_update_norms(
    snapshots: dict[str, torch.Tensor],
    net: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Compute per-group param/update/relative norms from saved snapshots.

    Classifies parameters into mapper/struct/dyn/other bins matching the
    logic in ``make_optimizer`` (see :ref:`lr-param-groups`).

    Returns a nested dict: ``{"mapper": {"param_norm": ..., "update_norm": ..., "rel_update": ...},
    "struct": ..., "dyn": ..., "other": ...}``.
    """
    raw = net.module if isinstance(net, torch.nn.DataParallel) else net
    groups: dict[str, dict[str, float]] = {
        "mapper": {"param_sq": 0.0, "update_sq": 0.0},
        "struct": {"param_sq": 0.0, "update_sq": 0.0},
        "dyn": {"param_sq": 0.0, "update_sq": 0.0},
        "other": {"param_sq": 0.0, "update_sq": 0.0}}

    for name, p in raw.named_parameters():
        if name not in snapshots:
            continue
        delta = p.data - snapshots[name]
        if "input_mapper" in name or "output_mapper" in name:
            g = "mapper"
        elif name.endswith(".z_logits"):
            g = "struct"
        elif (
            name.endswith(".vca_W")
            or name.endswith(".vca_W_core")
            or name.endswith(".vca_v_boundary")
            or name.endswith(".vca_v_readout")
            or name.endswith(".vca_v_core")
        ):
            g = "struct"
        elif name.endswith(".raw_leak") or name.endswith(".raw_drive_g"):
            g = "dyn"
        else:
            g = "other"
        groups[g]["param_sq"] += float(p.data.pow(2).sum().item())
        groups[g]["update_sq"] += float(delta.pow(2).sum().item())

    result: dict[str, dict[str, float]] = {}
    for gname, data in groups.items():
        pn = data["param_sq"] ** 0.5
        un = data["update_sq"] ** 0.5
        rel = un / (pn + 1e-12) if pn > 0 else 0.0
        result[gname] = {"param_norm": pn, "update_norm": un, "rel_update": rel}
    return result

def _grad_norm_keys(norms):
    """Deterministic key ordering for gradient norm output (shared by header and data rows)."""
    return sorted(
        [k for k in norms.keys() if k.startswith("stage")]
        + [k for k in ("stage_transfer", "in_mapper", "out_mapper") if k in norms]
    )

def log_gradient_norms(grad_log_path, epoch, raw_net, *, retrain=False, optimizer=None, norms=None):
    """Append one row of per-group L2 gradient norms to ``grad_log_path``.

    On the first call, also writes the header row.

    If an ``optimizer`` with multiple param groups is provided, appends the
    per-group learning rates as additional columns (``lr0``, ``lr1``, ...).
    """
    raw = raw_net.module if isinstance(raw_net, torch.nn.DataParallel) else raw_net
    if norms is None:
        norms = collect_gradient_norms(raw)
    ordered_keys = _grad_norm_keys(norms)

    lr_keys = []
    if optimizer is not None and len(optimizer.param_groups) > 1:
        lr_keys = [f"lr{i}" for i in range(len(optimizer.param_groups))]

    if not grad_log_path.exists():
        cols = ordered_keys + lr_keys
        with open(grad_log_path, "w") as f:
            f.write("epoch\t" + "\t".join(cols) + "\n")

    prefix = "retrain_" if retrain else ""
    row_parts = [f"{prefix}{epoch}"]
    for k in ordered_keys:
        v = norms.get(k)
        row_parts.append(f"{v:.6e}" if v is not None else "-")
    if lr_keys:
        for g in optimizer.param_groups:
            row_parts.append(f"{g['lr']:.6e}")
    with open(grad_log_path, "a") as f:
        f.write("\t".join(row_parts) + "\n")

def log_update_norms(
    path: Path,
    epoch: int,
    update_norms: dict[str, dict[str, float]],
    phase: str = "",
    *,
    retrain: bool = False) -> None:
    """Append one row of per-group param/update/relative norms to ``path``.

    Columns: ``epoch``, ``phase``, then for each group (mapper, struct, dyn, other):
    ``{group}_param``, ``{group}_update``, ``{group}_rel``.
    """
    group_order = ["mapper", "struct", "dyn", "other"]
    if not path.exists():
        cols = ["epoch", "phase"]
        for g in group_order:
            cols += [f"{g}_param", f"{g}_update", f"{g}_rel"]
        with open(path, "w") as f:
            f.write("\t".join(cols) + "\n")
    prefix = "retrain_" if retrain else ""
    parts = [f"{prefix}{epoch}", phase]
    for g in group_order:
        d = update_norms.get(g, {"param_norm": 0.0, "update_norm": 0.0, "rel_update": 0.0})
        parts.append(f"{d['param_norm']:.6e}")
        parts.append(f"{d['update_norm']:.6e}")
        parts.append(f"{d['rel_update']:.6e}")
    with open(path, "a") as f:
        f.write("\t".join(parts) + "\n")

def _vca_stages(net):
    """Return net (unwrapped) and its VCA-core-enabled stages, if any."""
    raw = net.module if isinstance(net, torch.nn.DataParallel) else net
    stages = [s for s in raw.core.stages if getattr(s, "_vca_core_enabled", False)]
    return raw, stages


def log_vca_gates(path: Path, epoch: int, net, val_loader,
                  device, *, phase: str = "", retrain: bool = False,
                  max_batches: int = 8):
    """Append one row of per-epoch VCA gate diagnostics to ``path``.

    For each VCA-enabled stage, on up to ``max_batches`` validation batches:
      * gate distribution (mean/std/min/max over [B, E_core]) — collapse of
        the gate to a constant across samples means the mechanism is unused;
      * input-dependent operating point: std across samples of the per-node
        core-bias magnitude ``|b_j(u)|`` where b_j(u) is the sum of the
        frozen tanh currents folded with the core gate. If the gate is
        constant the std collapses to ~0.

    Stale-cache note: ``_gate_core_cached`` is overwritten by every stage
    forward, so we capture it right after each ``net(u)`` call.
    """
    raw, stages = _vca_stages(net)
    if not stages:
        return
    if not path.exists():
        cols = ["epoch", "phase", "stage", "gate_mean", "gate_std",
                "gate_min", "gate_max", "bj_std"]
        with open(path, "w") as f:
            f.write("\t".join(cols) + "\n")
    prefix = "retrain_" if retrain else ""
    with torch.no_grad():
        seen = 0
        gate_rows = {i: [] for i in range(len(stages))}
        for u, _target in val_loader:
            u = u.to(device)
            net(u, store_trajectory=False)
            for si, s in enumerate(stages):
                g = getattr(s, "_gate_core_cached", None)
                if g is not None:
                    gate_rows[si].append(g)
            seen += 1
            if seen >= max_batches:
                break
    with open(path, "a") as f:
        for si, s in enumerate(stages):
            gs = gate_rows.get(si, [])
            if not gs:
                continue
            G = torch.cat(gs, dim=0)          # [B_total, E_core]
            gmean = float(G.mean().item())
            gstd = float(G.std().item())
            gmin = float(G.min().item())
            gmax = float(G.max().item())
            # Input-dependent operating-point proxy: std over samples of
            # the per-edge gate. If the gate collapsed to a constant the
            # spread goes to ~0 (mechanism unused).
            edge_gstd = G.std(dim=0)          # [E]
            bj = float(edge_gstd.mean().item()) if edge_gstd.numel() > 0 else 0.0
            f.write(
                f"{prefix}{epoch}\t{phase}\t{si}\t{gmean:.6e}\t{gstd:.6e}\t"
                f"{gmin:.6e}\t{gmax:.6e}\t{bj:.6e}\n"
            )

def make_static_ctx_factory():
    """Build a ctx_factory that always returns a default (variation-off) context."""
    def _factory(batch_size_: int, device: torch.device = "cpu", **_):
        return SimContext()
    return _factory

def _add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Populate ``parser`` with the train_script CLI flags.  Split out of
    ``main()`` so smoke tests can introspect the flag surface (PP-5)."""
    parser.add_argument(
        "--problem", choices=["sinx", "housing", "smooth2d", "smooth2d_grid", "housing_grid", "friedman1", "friedman2", "friedman3"], default="sinx",
        help="Task to train (default: sinx)")
    parser.add_argument(
        "--target-noise-std", type=float, default=1.0, dest="target_noise_std",
        help="Additive Gaussian noise std on targets. "
             "Used only by friedman1/friedman2/friedman3 problems (default: 1.0).")
    parser.add_argument(
        "--grid-size", type=int, default=None, dest="grid_size",
        help="Hidden grid height/width for smooth2d_grid and housing_grid. "
             "Per-problem default: smooth2d_grid=7, housing_grid=5. "
             "Explicit --grid-size N overrides either. "
             "Only applies when --problem smooth2d_grid or --problem housing_grid.")
    parser.add_argument(
        "--cell-library", type=str, default=None, dest="cell_library",
        choices=["legacy", "v15", "v2", "relu", "tanh", "tanh_realistic", "tanh_realistic_upgrade", "tanh_free", "tanh_anti"],
        help="Cell library: 'legacy' (L,S,P,Z, default), 'v15' (O_weak,O_hard,P0,N0,D1,Z), "
             "'v2' (mix-code/bias-code bounded library), "
             "'relu' (I=ReLU(p0*Vsrc+p1*Vdest+p2)), 'tanh' (I=tanh(p0*Vsrc+p1*Vdest+p2)), "
             "'tanh_realistic' (I=tanh(A*Vsrc - B*Vdest + C), A,B>0, A+B=1), "
             "'tanh_realistic_upgrade' (I=Isat*tanh(gm*(A*Vsrc - B*Vdest) + C), "
             "bounded gm/Isat per-edge), "
             "'tanh_free' (I=Isat*tanh(gm*(s*(A*Vsrc - B*Vdest) + theta)), "
             "A,B>=0 independent (no A+B=1), s=+/-1 via STE, bounded gm/Isat per-edge), "
             "'tanh_anti' (I=Isat*tanh(gm*relu(kappa*(Vsrc-Vdst)-theta)), "
             "rectified differential OTA slice for antiparallel edge fabrics, "
             "i>=0 and exactly zero at Vsrc=Vdst). "
             "Overrides the preset's cell_library key if present.")
    parser.add_argument(
        "--gm-min", type=float, default=None, dest="gm_min",
        help="Lower rail of the bounded sigmoid gm parameterization for "
             "tanh_realistic_upgrade / tanh_free / tanh_anti cell libraries "
             "(default: config TANH_REALISTIC_GM_MIN=0.01). "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--gm-max", type=float, default=None, dest="gm_max",
        help="Upper rail of the bounded sigmoid gm parameterization "
             "(default: config TANH_REALISTIC_GM_MAX=10.0). Must be > --gm-min. "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--isat-min", type=float, default=None, dest="isat_min",
        help="Lower rail of the bounded sigmoid Isat parameterization "
             "(default: config TANH_REALISTIC_ISAT_MIN=0.01). "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--isat-max", type=float, default=None, dest="isat_max",
        help="Upper rail of the bounded sigmoid Isat parameterization "
             "(default: config TANH_REALISTIC_ISAT_MAX=10.0). Must be > --isat-min. "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--c-eff", type=float, default=None, dest="c_eff",
        help="Effective node capacitance C_eff (default: config PHYS['C_eff']=1.0). "
             "Scales the ODE integration: dx_j/dt = (ΣI - leak - clip) / C_eff. "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--x-max", type=float, default=None, dest="x_max",
        help="Differential rail limit x_max for stage ODE states and the "
             "input/output mappers (default: config PHYS['x_max']=3.0). "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--sparsity-lambda", type=float, default=None, dest="sparsity_lambda",
        help="Override the sparsity regularizer weight (L1 over FreeTanhLibrary "
             "device cell params). Default from LAMBDAS / schedule. "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--entropy-lambda", type=float, default=None, dest="entropy_lambda",
        help="Override the gate-entropy regularizer weight (binary entropy of "
             "the soft edge gates). Default from LAMBDAS / schedule. "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--hidden-family", type=str, default=None, dest="hidden_family",
        choices=["grid", "small_world", "torus"],
        help="Hidden-node topology family (default: from preset). "
             "'grid' uses a 2D grid graph (requires --grid-size). "
             "'small_world' uses a Watts-Strogatz small-world graph "
             "(requires --num-hidden; --small-world-k/p/seed tune it). "
             "'torus' uses a 2D grid with periodic boundary conditions "
             "(uses --grid-size). "
             "When set, dynamically rebuilds the preset's stages config "
             "to use the specified family instead of the hardcoded preset "
             "topology.")
    parser.add_argument(
        "--num-hidden", type=int, default=None, dest="num_hidden",
        help="Number of hidden nodes (default: from preset). "
             "Required when --hidden-family=small_world (must be >= 2). "
             "Ignored for grid family (uses --grid-size).")
    parser.add_argument(
        "--num-stages", type=int, default=None, dest="num_stages",
        help="Number of ODE stages (default: from preset, or 1 if not "
             "specified). Each stage gets an identical topology. "
             "t_span and num_steps are divided evenly across stages.")
    parser.add_argument(
        "--edge-repeats", type=int, default=None, dest="edge_repeats",
        help="Parallel edges per hidden node pair (default: 2, range 1-8). "
             "Each repeated edge gets independent logits/gate/multiplier. "
             "I/O and projection edges are NOT repeated. Composes "
             "multiplicatively with --bidirectional. Set to 1 for the "
             "previous single-edge behavior.")
    parser.add_argument(
        "--bidirectional", dest="bidirectional", action="store_true", default=False,
        help="Emit two directed edges per unique node pair in the hidden graph "
             "(i->j AND j->i). Doubles the hidden edge count and gives "
             "asymmetric cells (P/rectifier) true bidirectional capability. "
             "Default: off (single edge per pair).")
    parser.add_argument(
        "--no-bidirectional", dest="bidirectional", action="store_false",
        help="Disable dual edges per node pair (default).")
    parser.add_argument(
        "--small-world-k", type=int, default=4, dest="small_world_k",
        help="Small-world neighbor count per node in the ring lattice "
             "(default: 4, must be even and <16). "
             "Used only when --hidden-family=small_world.")
    parser.add_argument(
        "--small-world-p", type=float, default=0.3, dest="small_world_p",
        help="Small-world rewiring probability in [0, 1] "
             "(default: 0.3). p=0 recovers a ring lattice, p=1 produces "
             "a random regular graph. Used only when --hidden-family=small_world.")
    parser.add_argument(
        "--small-world-seed", type=int, default=None, dest="small_world_seed",
        help="Small-world rewiring RNG seed (default: from --seed). "
             "If not given, tie to the global --seed for reproducible graphs. "
             "Used only when --hidden-family=small_world.")
    parser.add_argument(
        "--output", type=Path, default=Path("./output"),
        help="Output directory for artifacts (default: ./output)")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Number of epochs (default: {OPTIM['epochs']})")
    parser.add_argument(
        "--count-params-only", action="store_true",
        help="Build the network, print its trainable parameter count, and "
             "exit before loading data or training.")
    parser.add_argument(
        "--lr", type=float, default=None,
        help=f"Learning rate (default: {OPTIM['lr']})")
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        dest="weight_decay",
        help=f"Weight decay (default: {OPTIM['weight_decay']}). "
             "Forwarded to AdamW. kn-bayes-opt uses this as an optimization "
             "dimension.")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        dest="batch_size",
        help=f"Batch size (default: {OPTIM['batch_size']}). "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--t-span", type=float, default=None,
        dest="t_span",
        help=f"Total ODE integration horizon across all stages "
             f"(default: {SOLVER['t_span']}); per-stage t_span = total / num_stages. "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--num-steps", type=int, default=None,
        dest="num_steps",
        help=f"Total Heun steps across all stages "
             f"(default: {SOLVER['num_steps']}); per-stage num_steps = round(total / num_stages). "
             "kn-bayes-opt uses this as an optimization dimension.")
    parser.add_argument(
        "--stage-lr-scale", type=float, default=1.0,
        help="Per-stage geometric LR multiplier (stage-lr-scaling). "
             "When 1.0 (default), all parameters share the same LR. "
             "When >1.0, stage i gets lr * scale^(S-1-i) where S is the "
             "number of stages. Compensates for vanishing gradients in "
             "deep ODE stacks (e.g. scale=10 with 3 stages: stage0=lr*100, "
             "stage1=lr*10, stage2=lr).")
    parser.add_argument(
        "--retrain-stage-lr-scale", type=float, default=1.0,
        help="Per-stage LR scaling for retrain (default: 1.0). "
             "Warm-started pruned networks need gentle fine-tuning, so "
             "this defaults to uniform LR. Set to match --stage-lr-scale "
             "if you want geometric scaling during retrain.")
    parser.add_argument(
        "--mapper-lr-scale", type=float, default=0.1,
        help="LR multiplier for I/O mapper params (input_mapper + output_mapper). "
             "Default 0.1 (slow mapper learning). Use 1.0 to match base LR. "
             "When mapper gradient norms dominate core by ~300x, lowering this "
             "forces more residual error to be explained by the core.")
    parser.add_argument(
        "--retrain-mapper-lr-scale", type=float, default=0.1,
        help="Mapper LR scale for retrain (default: 0.1). "
             "Mirrors --mapper-lr-scale if you want to slow mapper learning "
             "during retrain.")
    parser.add_argument(
        "--encoder-type", type=str, default="linear", dest="encoder_type",
        choices=["linear", "residual_tanh"],
        help="Input encoder architecture. 'linear' (default) uses the "
             "standard dense InputMapper (or sparse variants per --write-mode). "
             "'residual_tanh' adds a residual skip-connection tanh branch "
             "(ResidualTanhEncoder in_dim->hidden->out_dim, followed by "
             "x_max*tanh(...)) for non-linear input expressivity. Only "
             "applied when --write-mode='dense'; sparse modes ignore this flag.")
    parser.add_argument(
        "--decoder-type", type=str, default="linear", dest="decoder_type",
        choices=["linear", "residual_tanh"],
        help="Output decoder architecture. 'linear' (default) uses the "
             "standard OutputMapper. 'residual_tanh' replaces it with a "
             "ResidualTanhOutputMapper (non-linear readout with skip "
             "connection). Mutually exclusive with --nodes-per-target>0 "
             "(grouped readout).")
    parser.add_argument(
        "--encoder-hidden-dim", type=int, default=64, dest="encoder_hidden_dim",
        help="Hidden width of the residual tanh branch in the input encoder "
             "(default: 64). Only used when --encoder-type='residual_tanh'.")
    parser.add_argument(
        "--decoder-hidden-dim", type=int, default=64, dest="decoder_hidden_dim",
        help="Hidden width of the residual tanh branch in the output decoder "
             "(default: 64). Only used when --decoder-type='residual_tanh'.")
    parser.add_argument(
        "--skip-linear", action="store_true", default=False,
        dest="skip_linear",
        help="Add a pure-linear skip connection y = W₁·u + b₁ + f(x), where "
             "W₁ is an nn.Linear from raw input (in_dim) to the output "
             "dimension and f(x) is the KirchhoffNet ODE transformation. "
             "Offloads the linear mixing part from the KirchhoffNet so the "
             "ODE core can specialize on non-linear structure only. Adds "
             "(in_dim * out_dim + out_dim) parameters and is regularized "
             "with an L2 penalty (lambda = config.LAMBDAS['skip_linear_l2']) "
             "so the skip projection is incentivized to be small. Default: "
             "off (no skip connection).")
    parser.add_argument(
        "--device-l2-lambda", type=float, default=0.0,
        dest="device_l2_lambda",
        help="Lambda for the scale-invariant L2 penalty on the 'tanh_free' "
             "device-family parameters: FreeTanhLibrary cell params "
             "(a_raw/b_raw/gm_raw/isat_raw/theta_raw/g_resistive_raw on "
             "core/boundary/output_ode cell libs; s_raw is excluded as a "
             "discrete sign), interstage residual transfer weights "
             "(residual_w1/w2/w3/vth), temporal-readout OutputAffine params "
             "(gain/bias/tanh_gain/relu_gain/relu_threshold), and VCA params "
             "(vca_W[/vca_W_core], vca_v_boundary/vca_v_readout/vca_v_core). "
             "Computed as lambda * mean(p.pow(2)) over the union of penalized "
             "tensors so pressure per parameter is invariant to model size. "
             "Always-on (not multiplied by reg_schedule warm-up). Default: 0.0 "
             "(disabled).")
    parser.add_argument(
        "--struct-lr-scale", type=float, default=2.0,
        help="LR multiplier for structural core params (z_logits, cell logits, "
             "raw_mult). Default 2.0 (modest boost). These combinatorial-ish "
             "parameters often need help in DEQ mode. When != 1.0, uses flat "
             "global groups and ignores --stage-lr-scale.")
    parser.add_argument(
        "--dyn-lr-scale", type=float, default=1.0,
        help="LR multiplier for sensitive dynamical params (raw_leak, "
             "raw_drive_g). Default 1.0 (base LR). These affect the Jacobian "
             "and fixed-point conditioning, so boosting aggressively can "
             "destabilize DEQ solves. When != 1.0, uses flat global groups "
             "and ignores --stage-lr-scale.")
    parser.add_argument(
        "--freeze-mappers", dest="freeze_mappers", action="store_true", default=False,
        help="Freeze mapper requires_grad during the first half of the combined "
             "B1+B2 duration (four_phase schedule only). Mappers train normally "
             "in Phase A, freeze at B1 start, unfreeze at the midpoint. After "
             "unfreeze mappers resume at the --mapper-lr-scale rate. "
             "Ignored for three_phase and legacy schedules.")
    parser.add_argument(
        "--device", default=None,
        help="Device 'cpu' or 'cuda' (default: auto-detect)")
    parser.add_argument(
        "--amp", dest="amp", action="store_true", default=None,
        help="Enable mixed precision (AMP) via torch.cuda.amp (default: on when CUDA)")
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false",
        help="Disable mixed precision")
    parser.add_argument(
        "--compile", dest="compile", action="store_true", default=None,
        help="Enable torch.compile on hot paths (default: on when CUDA)")
    parser.add_argument(
        "--no-compile", dest="compile", action="store_false",
        help="Disable torch.compile")
    parser.add_argument(
        "--parallel", dest="parallel", action="store_true", default=None,
        help="Enable DataParallel across multiple GPUs (default: on when ≥2 GPUs)")
    parser.add_argument(
        "--no-parallel", dest="parallel", action="store_false",
        help="Disable DataParallel")
    parser.add_argument(
        "--validate-every", type=int, default=5,
        help="Validate every N epochs (default: 5). Use 1 for every epoch.")
    parser.add_argument(
        "--early-stop", dest="early_stop", action="store_true", default=True,
        help="Enable early stopping (default: on)")
    parser.add_argument(
        "--no-early-stop", dest="early_stop", action="store_false",
        help="Disable early stopping")
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience in epochs (default: 100). "
             "Stops after N epochs with no val improvement > --min-delta.")
    parser.add_argument(
        "--min-delta", type=float, default=1e-4,
        help="Early stopping min improvement in val loss (default: 1e-4)")
    parser.add_argument(
        "--amp-dtype", choices=["float16", "bfloat16"], default="float16",
        help="Autocast dtype (default: float16; bfloat16 needs Ampere+)")
    parser.add_argument(
        "--ablation", choices=["none", "mapper-only", "empty-graph"], default="none",
        help="Structural ablation to apply (default: none). R2.")
    parser.add_argument(
        "--variation", dest="variation", action="store_true", default=False,
        help="Enable PVT/mismatch injection during training (default: off, R6.3).")
    parser.add_argument(
        "--write-mode", choices=["one_to_one", "dense", "fan_out", "sparse_proj"], default=None,
        help="Input write mapping (default: from preset). 'one_to_one' uses "
             "SparseInputMapper, 'dense' uses InputMapper (nn.Linear), "
             "'fan_out' uses FanOutInputMapper with preset-defined targets, "
             "'sparse_proj' uses ProjectedSparseInputMapper (Linear projection "
             "to len(write_idx) targets; requires --write-idx with len >= in_dim).")
    parser.add_argument(
        "--read-mode", choices=["sparse", "dense"], default=None,
        help="Output read mapping (default: from preset, typically 'sparse'). "
             "'sparse' uses OutputMapper with preset-defined read_idx; "
             "'dense' uses full-projection readout.")
    parser.add_argument(
        "--write-idx", type=str, default=None,
        help="Comma-separated hidden node indices for sparse input write "
             "(overrides preset write_idx). E.g. '0,2,4'.")
    parser.add_argument(
        "--read-idx", type=str, default=None,
        help="Comma-separated full-state indices for sparse output read "
             "(overrides preset read_idx). E.g. '7'.")
    parser.add_argument(
        "--prune", dest="prune", action="store_true", default=False,
        help="Run gate-based pruning after training (CP). Prunes edges and "
             "nodes below the configured thresholds, then either retrains "
             "the compact network or saves it as-is (see --no-retrain).")
    parser.add_argument(
        "--retrain", dest="retrain", action="store_true", default=True,
        help="After pruning, retrain the compact network warm-started from "
             "the surviving pre-prune parameters (default: on). Use "
             "--no-retrain to skip retraining, or --fresh-init to retrain "
             "from random init instead.")
    parser.add_argument(
        "--no-retrain", dest="retrain", action="store_false",
        help="Skip retraining after pruning; only transfer the surviving "
             "parameters from the overcomplete network into the compact one.")
    parser.add_argument(
        "--prune-edge-threshold", type=float, default=None,
        help="Override config.PRUNE['edge_threshold'] for pruning.")
    parser.add_argument(
        "--prune-node-threshold", type=float, default=None,
        help="Override config.PRUNE['node_threshold'] for pruning.")
    parser.add_argument(
        "--prune-nodes-by-gate", dest="prune_nodes_by_gate",
        action="store_true", default=None,
        help="DEPRECATED (deprecate-node-gates): no-op, kept for backward "
             "compat. Node pruning is now connectivity-only regardless of "
             "this flag. Use --no-prune-nodes-by-gate to silence the warning.")
    parser.add_argument(
        "--no-prune-nodes-by-gate", dest="prune_nodes_by_gate",
        action="store_false",
        help="DEPRECATED (deprecate-node-gates): no-op, kept for backward "
             "compat. Nodes are pruned by connectivity only.")
    parser.add_argument(
        "--retrain-epochs", type=int, default=None,
        help="Number of epochs to retrain the compact network (default: "
             "the same value as --epochs, capped at half).")
    parser.add_argument(
        "--retrain-lr", type=float, default=None,
        help="Learning rate for the retrain phase (default: same as --lr).")
    parser.add_argument(
        "--retrain-batch-size", type=int, default=2048,
        help="retrain-oom-fix/REQ-5: batch size for the pruned-network "
             "retrain phase (default: 2048, half the default training "
             "batch size of 4096). Lower this when GPU memory is tight "
             "after Phase A+B to prevent the OOM at the prune-to-retrain "
             "transition. Set to 0 to use the same batch size as "
             "--batch-size. The pruned model is created as a fresh "
             "nn.Module so it duplicates the parameter tensor allocations.")
    parser.add_argument(
        "--fresh-init", dest="fresh_init", action="store_true", default=False,
        help="Re-initialize the pruned network from scratch (skip warm "
             "start from pre-prune parameters). Default: warm-start.")
    parser.add_argument(
        "--no-scheduler", dest="use_scheduler", action="store_false", default=True,
        help="Disable LR scheduler (default: on).")
    parser.add_argument(
        "--scheduler-type", choices=["cosine", "warm_restarts"], default="cosine",
        help="LR scheduler type when --scheduler is enabled (default: 'cosine' — "
             "plain cosine decay over total epochs, no restarts). 'warm_restarts' "
             "uses CosineAnnealingWarmRestarts (legacy behavior).")
    parser.add_argument(
        "--grad-log", dest="grad_log", action="store_true", default=False,
        help="Periodically log per-parameter-group gradient L2 norms to "
             "grad_norms.txt (default: off).")
    parser.add_argument(
        "--grad-log-every", type=int, default=10,
        help="Log gradient norms every N epochs (default: 10). Only used "
             "when --grad-log is enabled.")
    parser.add_argument(
        "--vca-diag", dest="vca_diag", action="store_true", default=False,
        help="Per-epoch log of core-gate distributions and input-dependent "
             "operating-point measures (vca_gates.txt). No-op when no stage "
             "has core VCA enabled (default: off).")
    parser.add_argument(
        "--schedule", choices=["legacy", "three_phase", "four_phase"], default=None,
        help="Training schedule mode (default: from preset['schedule'], "
             "fallback 'legacy'). 'three_phase' implements the phased "
             "fit-compress-prune pipeline (Phase A: fit with no structure "
             "pressure, Phase B: compress via gate penalties, Phase C: "
             "auto-prune + retrain). 'four_phase' adds a cell-commitment "
             "Phase B1 (no pruning), readiness-gated Phase B2 (edge "
             "pruning), and a KD-anchored retrain Phase C. See "
             "spec/four-phase-schedule.md.")

    parser.add_argument(
        "--ablation-set",
        choices=["none", "reg-only", "tau-only", "edge-only"], default="none",
        help="Diagnostic ablation preset (four-phase-redesign/Phase 1c). "
             "Each preset overrides the relevant combination of "
             "tau/regularizer/pruning flags. 'reg-only' keeps tau=1.0 "
             "through B and turns on gate regularizers without pruning. "
             "'tau-only' anneals tau with all structural regularizers off "
             "and no pruning. 'edge-only' is the normal B path but with "
             "node-gate pruning disabled and a lower edge threshold "
             "(matches the four-phase-redesign defaults). 'none' is the "
             "standard schedule behavior with no overrides.")
    parser.add_argument(
        "--cell-mode", choices=["soft", "ste", "auto"], default="auto",
        help="Cell selection mode (). 'soft' "
             "uses a softmax-weighted mixture of cells per edge. 'ste' "
             "uses one cell per edge in the forward pass (argmax) with "
             "straight-through soft gradients in the backward pass. "
             "'auto' uses 'soft' for Phase A and 'ste' for B/C (only "
             "meaningful with --schedule three_phase / four_phase).")

    # --- Deep Equilibrium (DEQ) stagewise solver (deq-core-prototype) ---
    parser.add_argument(
        "--solver", choices=["heun", "deq"], default="heun",
        dest="solver",
        help="Stage solver (default: heun). 'deq' solves each stage's "
             "fixed-point x* s.t. rhs(x*)=0 via torchdeq with implicit "
             "gradients. DEQ enforces --cell-mode soft and applies the "
             "minimum-leak floor (--leak-floor) so the fixed-point map is "
             "contractive. Heun remains available for physical validation.")
    parser.add_argument(
        "--deq-backend", choices=["auto", "torchdeq", "fixed_point_iter"],
        default="auto", dest="deq_backend",
        help="DEQ solver backend (default: auto -> torchdeq if installed).")
    parser.add_argument(
        "--deq-f-max-iter", type=int, default=None, dest="deq_f_max_iter",
        help="Max forward iterations for the DEQ fixed-point solver.")
    parser.add_argument(
        "--deq-f-tol", type=float, default=None, dest="deq_f_tol",
        help="Relative residual tolerance for DEQ convergence.")
    parser.add_argument(
        "--deq-b-max-iter", type=int, default=None, dest="deq_b_max_iter",
        help="Max backward (IFT) iterations.")
    parser.add_argument(
        "--deq-step", type=float, default=None, dest="deq_step",
        help="Damped step size dt for Phi(x)=x+dt*rhs(x).")
    parser.add_argument(
        "--leak-floor", type=float, default=None, dest="leak_floor",
        help="Minimum effective leak per node under DEQ (default: from "
             "config DEQ). Keeps the fixed-point map contractive. Has no "
             "effect on the Heun path beyond the explicit addend.")
    parser.add_argument(
        "--run-deq-diagnostics", action="store_true", default=False,
        dest="run_deq_diagnostics",
        help="Run DEQ diagnostics (gradient-norm compare, Jacobian cond, "
             "multistart uniqueness) once after train/val and print the "
             "report. Useful when prototyping --solver deq.")
    parser.add_argument(
        "--persistent-drive", action="store_true", default=False,
        dest="persistent_drive",
        help="Enable persistent drive current in all stages. The drive "
             "current I_drive = I_sat * tanh(g * (x_drive - x) / I_sat) "
             "makes the fixed point x* input-dependent under DEQ. "
             "Compatible with write_mode='fan_out', 'sparse_proj', and "
             "'one_to_one' (the driven nodes are the write_idx entries). "
             "Has no effect when --solver heun.")
    parser.add_argument(
        "--drive-mode", choices=["fan_out", "projection"], default="fan_out",
        dest="drive_mode",
        help="Drive mapper type when --persistent-drive is active. "
             "'fan_out': per-input scalar (gain,bias) pairs drive each "
             "node independently (no input mixing). 'projection': learned "
             "nn.Linear(in_dim, len(write_idx)) with tanh, mixing all "
             "inputs into every driven node (matches ProjectedSparseInputMapper). "
             "Default: 'fan_out'.")
    parser.add_argument(
        "--write-fan-out", type=str, default=None, dest="write_fan_out",
        help="JSON dict for --write-mode fan_out specifying the input->target "
             "mapping, e.g. '{\"0\": [2, 12], \"1\": [7, 17]}' means input 0 "
             "drives nodes 2 and 12, input 1 drives nodes 7 and 17. Overrides "
             "any preset write_fan_out and any auto-generated fan-out. "
             "Ignored unless --write-mode fan_out.")
    parser.add_argument(
        "--boundary-fan-out", type=str, default=None, dest="boundary_fan_out",
        help="JSON dict specifying sparse OTA edges from fixed-voltage input "
             "boundary terminals into the dynamic fabric, e.g. "
             "'{\"0\": [2, 12], \"1\": [7, 17]}'. The input features become "
             "external voltage terminals that inject current into the listed "
             "target hidden nodes throughout the entire ODE evolution. All "
             "dynamic nodes start at zero (no initial-condition write); "
             "boundary edges inject I_OTA(u_i, x_j) per step, with the "
             "input terminal itself never drained (unlike core OTA edges). "
             "Each boundary edge carries its own OTA cell parameters and "
             "edge gate (compatible with --no-edge-gates).")
    parser.add_argument(
        "--enable-ref-edges", action="store_true", default=False,
        dest="enable_ref_edges",
        help="Enable reference edges (unary nonlinearities via OTA-to-Vref): "
             "every ODE node gets one OTA edge to a global per-stage learnable "
             "reference voltage Vref (scalar, constrained to [0, x_max]). The "
             "reference edge injects I_OTA(Vref, x_j) into node j only (no "
             "source drain — Vref is an ideal voltage source). Implements "
             "programmable thresholding, saturation, soft activation, bias "
             "injection, and nonlinear leak without introducing a heterogeneous "
             "cell type. Each reference edge uses the same OTA cell family as "
             "the core edges and carries its own edge gate (compatible with "
             "--no-edge-gates). Default: disabled.")
    parser.add_argument(
        "--temporal-readout", action="store_true", default=False,
        dest="enable_temporal_readout",
        help="Replace the linear OutputMapper readout with a dynamical "
             "temporal-readout path: append out_dim extra output ODE "
             "accumulator nodes to each stage's ODE state. Hidden nodes "
             "connect all-to-all to each output ODE node via one-way OTA "
             "edges (source read-only, destination writable — the hidden "
             "grid is never drained). The output ODE node voltages are "
             "scaled by a learnable OutputAffine "
             "(gain * V + bias + tanh_gain * tanh(V) + "
             "relu_gain * relu(V - relu_threshold)) layer. "
             "Output ODE nodes receive leak + clip like regular nodes and "
             "are initialized to zero. Requires all stages to have the "
             "same width. Incompatible with --decoder-type residual_tanh "
             "and --grouped-readout. Default: disabled.")
    parser.add_argument(
        "--readout", choices=["linear", "temporal", "shared", "shared-x2"],
        default="temporal", dest="readout",
        help="Readout family. 'temporal' = h*d_out OTA mesh (default): every "
             "hidden node connects to every output-ODE accumulator via its own "
             "FreeTanh cell + edge gate + VCA tap, then OutputAffine reads the "
             "accumulator tail. 'linear' = pre-refactor default: linear "
             "OutputMapper reads from projection nodes (or hidden nodes if no "
             "projection), NO output-ODE accumulator tail, NO OutputAffine. "
             "'shared' = one FreeTanh sense OTA per hidden node driven against "
             "a per-stage learnable Vref rail + a plain dense crossbar "
             "W[d_out x h] mixing the sense currents into the accumulator "
             "tail. 'shared-x2' = two senses per hidden node (different gm "
             "operating points) + W[d_out x 2h]. The crossbar is plain dense "
             "weights (no gates, no VCA); VCA gates the senses only. "
             "Deprecated alias: --temporal-readout selects 'temporal'; "
             "combining it with --readout linear/shared/shared-x2 is an error.")
    parser.add_argument(
        "--vca", action="store_true", default=False,
        dest="vca",
        help="Enable low-rank input-driven VCA (Voltage-Controlled Amplifier) "
             "gating on boundary and readout edges. Each gated edge's tail "
             "current is modulated by vca_e = 2*sigmoid(u^T W v_e), where W "
             "(in_dim x rank) is a shared input projection and v_e (rank) is "
             "a per-edge embedding. Physically: rank global control buses "
             "broadcast from the input terminals, each unfrozen edge taps into "
             "them with a programmable weight. Provides content-dependent "
             "multiplicative cross-node interaction (attention analog). "
             "Compatible with --freeze-read because u is constant per sample "
             "across all Heun/DEQ sub-iterations. Requires at least one of "
             "--boundary-fan-out or a readout tail family "
             "(--readout temporal/shared/shared-x2); in shared-sense mode the "
             "VCA gates the sense bank instead of a temporal mesh. "
             "Default: disabled.")
    parser.add_argument(
        "--vca-rank", type=int, default=None, dest="vca_rank",
        help="VCA projection rank r (default: config.VCA['rank'] = "
             f"{VCA['rank']}). r=1 collapses to a shared input projection "
             "with per-edge scalar gain; larger r gives the optimizer more "
             "axes to express input-edge alignment. Must be >= "
             f"{VCA['min_rank']}.")
    parser.add_argument(
        "--vca-bias", action="store_true", default=None, dest="vca_bias",
        help="Enable learned per-edge affine VCA offsets: gate_e = 2*sigmoid("
             "b_e + (u@W)@v_e). Omitted by default for backward compatibility.")
    parser.add_argument(
        "--vca-core", action="store_true", default=False,
        dest="vca_core",
        help="Enable VCA gating on core (hidden) edges in addition to "
             "boundary/readout edges. Computed once per sample at "
             "stage entry (u is constant per sample across all Heun/DEQ "
             "sub-iterations), so the gate folds into the freeze_read "
             "precompute with zero extra cost in the Heun loop. "
             "Auto-enabled when --vca is on but no gated family exists "
             "(--boundary-fan-out nor a --readout temporal/shared/shared-x2 "
             "family). Default: disabled unless no other gated family exists.")
    parser.add_argument(
        "--vca-gate-shunt", action="store_true", default=False,
        dest="vca_gate_shunt",
        help="Also gate the parallel resistive shunt with the core VCA "
             "gate, making M(u) input-dependent (input-dependent routing "
             "of current flow, not just scaling). Passivity preserved "
             "(gain in (0, 2) is dissipative). Default: disabled (evaluate "
             "last).")
    parser.add_argument(
        "--vca-separate-core-bus", action="store_true", default=False,
        dest="vca_separate_core_bus",
        help="Give the core edge family its own input projection (vca_W_core) "
             "instead of sharing vca_W with boundary/readout edges. "
             "Ablation only; default off (shared bus).")
    parser.add_argument(
        "--leak", choices=["programmable", "non-programmable"],
        default="programmable", dest="leak",
        help="Stage leak mode (default: programmable). 'programmable' "
             "creates a learnable per-node raw_leak parameter. "
             "'non-programmable' uses a fixed scalar leak_constant "
             "for all nodes, saving parameters and eliminating leak "
             "gradients. See also --leak-constant.")
    parser.add_argument(
        "--leak-constant", type=float, default=None, dest="leak_constant",
        help="Fixed leak value when --leak non-programmable (default: "
             "config INIT['leak_constant'] = 0.0486, which matches "
             "softplus(raw_leak_init=-3.0)). Ignored when --leak "
             "programmable.")

    # --- Degree budget / fraction edge competition (degree-budget-topk plan) ---
    parser.add_argument(
        "--budget", action="store_true", default=False,
        dest="budget",
        help="Enable degree-budget edge competition. Each destination "
             "(or source) node keeps a fraction of its incoming edges open "
             "via temperature-scaled softmax renormalization of z_logits. "
             "Replaces the L1 edge_gate pressure with explicit competition.")
    parser.add_argument(
        "--budget-frac-start", type=float, default=None,
        dest="budget_frac_start",
        help="Initial budget fraction per group (permissive, 1.0 = no "
             "restriction). Default: 1.0.")
    parser.add_argument(
        "--budget-frac-end", type=float, default=None,
        dest="budget_frac_end",
        help="Final budget fraction per group (restrictive, 0.0 = disables "
             "budget, 0.75 = keep 75%% of edges). Default: 0.75.")
    parser.add_argument(
        "--budget-temp-start", type=float, default=None,
        dest="budget_temp_start",
        help="Initial softmax temperature (soft). Default: 1.0.")
    parser.add_argument(
        "--budget-temp-end", type=float, default=None,
        dest="budget_temp_end",
        help="Final softmax temperature (sharp, approaches hard top-k_eff). "
             "Default: 0.1.")
    parser.add_argument(
        "--budget-axis", choices=["dst", "src", "both"], default=None,
        dest="budget_axis",
        help="Competition axis: 'dst' (per-destination, default), 'src' "
             "(per-source), or 'both' (multiplicative).")

    # --- Read-only source: decouple read (voltage sense) from write (current injection) ---
    parser.add_argument(
        "--read-only-source", action="store_true", default=False,
        dest="read_only_source",
        help="When set, OTA edges inject current into the destination node "
             "without sourcing equal current from the source node. The source "
             "node voltage is still read to compute the differential current.")

    parser.add_argument(
        "--freeze-read", dest="freeze_read", action="store_true", default=False,
        help="Compute edge currents once from the initial state and hold them "
             "constant across all Heun/DEQ iterations inside every stage. "
             "Disentangles node-voltage reads from same-step writes "
             "(experimental).")
    parser.add_argument(
        "--freeze-boundary", dest="freeze_boundary", action="store_true",
        default=False,
        help="Compute boundary fan-out edge currents (tanh + edge gate + VCA "
             "gate) once from (u, x0) and hold them constant across all "
             "Heun/DEQ iterations inside every stage. The family's resistive "
             "shunt (when present) remains dynamic per-step, mirroring the "
             "--freeze-read behavior for the core graph. Independent of "
             "--freeze-read (combine freely). No-op when --boundary-fan-out "
             "is not set.")
    parser.add_argument(
        "--freeze-temporal-read", dest="freeze_temporal_read",
        action="store_true", default=False,
        help="Compute temporal-readout edge currents (tanh + edge gate + VCA "
             "gate) once from x0 and hold them constant across all Heun/DEQ "
             "iterations inside every stage. The family's resistive shunt "
             "(when present) remains dynamic per-step, mirroring the "
             "--freeze-read behavior for the core graph. Independent of "
             "--freeze-read and --freeze-boundary (combine freely). No-op "
             "when --temporal-readout is not set.")

    # --- Interstage activation: pointwise non-linearity between stages ---
    parser.add_argument(
        "--interstage-activation",
        choices=["none", "relu", "residual", "residual_mixing", "residual-relu-tanh"],
        default="none", dest="interstage_activation",
        help="Pointwise non-linearity applied to the state vector between "
             "stages. 'none' (default) keeps the transfer as a pure "
             "identity/truncate/pad. 'relu' applies ReLU on each node. "
             "'residual' uses a per-node learnable W1*x + W2*tanh(x); "
             "'residual_mixing' adds an additive zero-initialized mixing "
             "term that can mix signals across nodes. Persistently driven "
             "nodes bypass the transform and pass through as identity.")
    parser.add_argument(
        "--interstage-residual-rank", type=int, default=-1,
        dest="interstage_residual_rank",
        help="Rank of the additive mixing term when "
             "--interstage-activation=residual_mixing. -1 (default) or any "
             "value >= out_nodes = full N×N matrix. 0 = pure diagonal "
             "(equivalent to 'residual'). 1..N-1 = low-rank factorization "
             "with N×r and r×N factors. Ignored for other activation modes.")

    # --- kirchhoff-noise: ADC/DAC quant + circuit noise (analog-noise parity) ---
    parser.add_argument(
        "--noise", dest="noise", action="store_true", default=False,
        help="Run Monte Carlo analog-noise evaluation (ADC/DAC quant + "
             "circuit noise) on the trained model and write "
             "noise_metrics.txt. Composes with --variation: variation "
             "perturbs gm/isat via SimContext, noise perturbs state "
             "voltages per stage.")
    parser.add_argument(
        "--noise-aware", dest="noise_aware", action="store_true", default=False,
        help="Train under analog noise so the KirchhoffNet becomes robust "
             "to ADC/DAC quantization and circuit noise. Implies --noise "
             "for final evaluation.")
    parser.add_argument(
        "--quant-bits", type=int, choices=[4, 6], default=4,
        dest="quant_bits",
        help="Bit-width for ADC/DAC quantization (default: 4). Used when "
             "--noise or --noise-aware is set.")
    parser.add_argument(
        "--noise-std", type=float, default=0.05,
        dest="noise_std",
        help="Standard deviation of additive Gaussian circuit noise on "
             "the analog state voltages and on the output mapper (default: "
             "0.05). Used when --noise or --noise-aware is set.")
    parser.add_argument(
        "--mc-trials", type=int, default=20,
        dest="mc_trials",
        help="Number of Monte Carlo trials for noisy evaluation "
             "(default: 20). Used when --noise or --noise-aware is set.")
    parser.add_argument(
        "--adc-full-range", type=float, default=3.0,
        dest="adc_full_range",
        help="Symmetric full-scale range for ADC/DAC quantization "
             "(default: 3.0). Used when --noise or --noise-aware is set.")
    parser.add_argument(
        "--noise-seed", type=int, default=0,
        dest="noise_seed",
        help="Seed for analog-noise sampling (default: 0). Each MC trial "
             "uses noise_seed + trial_idx as its seed. Used when --noise "
             "or --noise-aware is set.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for torch/numpy/python RNGs and cuDNN. "
             "Default: generate random seed at startup. Set to an integer "
             "for fully reproducible runs (note: cuDNN deterministic mode "
             "may slow down training).")
    parser.add_argument(
        "--normalize-inputs", dest="normalize_inputs", action="store_true", default=True,
        help="Per-dim min-max normalize Friedman problem inputs to [0, 1] "
             "(default: on). Use --no-normalize-inputs to disable for "
             "ablation studies.")
    parser.add_argument(
        "--no-normalize-inputs", dest="normalize_inputs", action="store_false",
        help="Disable per-dim input normalization (ablation against the "
             "default normalized mode).")
    parser.add_argument(
        "--tanh-sat-lambda", type=float, default=None,
        dest="tanh_sat_lambda",
        help="Tanh-saturation penalty weight for FreeTanhLibrary edges: "
             "λ * mean_over_edges(tanh(u)^2), where u is the per-edge tanh "
             "argument. Discourages edges from operating in the saturated "
             "region of tanh unless necessary. Set to 0 to disable. "
             "Default: None (use config.LAMBDAS['tanh_sat'], currently 0.0).")
    parser.add_argument(
        "--no-resistive-shunt", action="store_true", default=False,
        dest="no_resistive_shunt",
        help="Disable the parallel resistive shunt in FreeTanhLibrary. "
             "Zeros g_resistive_raw and sets _has_resistive=False on every "
             "stage so the resistive branch is bypassed in rhs() and frozen "
             "with the tanh current in the freeze_read path. The "
             "tanh_sat lambda is unaffected. Default: shunt enabled.")
    parser.add_argument(
        "--no-edge-gates", action="store_true", default=False,
        dest="no_edge_gates",
        help="Remove edge gating: freeze z_logits to +10 (sigmoid~1) so "
             "all edges are permanently on. Disables edge_gate regularizer, "
             "budget gate, and gate-based pruning.")

# ----------------------------------------------------------------
# Pruning helpers (PIT): transferable I/O mapper reconstruction.
# ----------------------------------------------------------------

def _remap_indices(idx_list, remap):
    """Remap a list of compact node ids through a node_remap dict.

    Pruned indices (not present in ``remap``) are silently dropped. This
    is safe because:
      - write targets are protected from pruning (see ``protected_nodes``
        in ``prune_stage``), so write_idx entries always survive.
      - read targets are allowed to be pruned (elastic readout), and the
        surviving entries are remapped; pruned ones are dropped from the
        read_idx list, which the caller handles by rebuilding the
        OutputMapper with fewer input features.

    Returns a list of remapped indices (may be shorter than ``idx_list``).
    """
    out = []
    for i in idx_list:
        if i not in remap:
            continue
        out.append(remap[i])
    return out

def _transfer_input_mapper(raw_mapper, raw_write_idx, stage0_remap,
                            pruned_first_n, in_dim):
    """Return a new InputMapper for the pruned stage with weights
    transferred from ``raw_mapper``.

    For DenseMapper/RobustInputMapper, rows are selectively copied
    for surviving nodes when stage 0's node count changed.

    For SparseInputMapper, ``raw_write_idx`` is remapped through
    ``stage0_remap`` and the per-feature (gain, bias) parameters
    are copied directly (they are indexed by input, not by node).

    For ProjectedSparseInputMapper, ``raw_write_idx`` is remapped through
    ``stage0_remap`` and the ``proj`` linear weights are transferred
    column-by-column (each output column corresponds to one write_idx
    entry in the original ordering).

    For FanOutInputMapper, each target list in ``fan_out_map`` is
    remapped through ``stage0_remap``; per-(input, target)
    (gain, bias) parameters are copied directly.
    """
    if isinstance(raw_mapper, SparseInputMapper):
        if raw_write_idx is None:
            new_write_idx = list(range(min(in_dim, pruned_first_n)))
        else:
            new_write_idx = _remap_indices(raw_write_idx, stage0_remap)
        new_mapper = SparseInputMapper(
            in_dim=in_dim, out_dim=pruned_first_n, write_idx=new_write_idx)
        with torch.no_grad():
            new_mapper.gain.data.copy_(raw_mapper.gain.data)
            new_mapper.bias.data.copy_(raw_mapper.bias.data)
        return new_mapper, new_write_idx

    if isinstance(raw_mapper, ProjectedSparseInputMapper):
        if raw_write_idx is None:
            new_write_idx = list(range(min(in_dim, pruned_first_n)))
        else:
            new_write_idx = _remap_indices(raw_write_idx, stage0_remap)
        new_mapper = ProjectedSparseInputMapper(
            in_dim=raw_mapper.in_dim,
            out_dim=pruned_first_n,
            write_idx=new_write_idx,
            x_max=raw_mapper.x_max,
        )
        with torch.no_grad():
            old_w = raw_mapper.proj.weight.data  # (len(raw_write_idx), in_dim)
            old_b = raw_mapper.proj.bias.data    # (len(raw_write_idx),)
            new_w = new_mapper.proj.weight.data  # (len(new_write_idx), in_dim)
            new_b = new_mapper.proj.bias.data    # (len(new_write_idx),)
            # Iterate raw_write_idx in order; surviving entries get their old
            # row copied to the new row in the same positional order.
            # nn.Linear weight shape: (out_features, in_dim) — rows map to
            # write_idx entries, so we copy rows, not columns.
            new_col = 0
            for old_col, wi in enumerate(raw_write_idx):
                if wi in stage0_remap:
                    new_w[new_col, :].copy_(old_w[old_col, :])
                    new_b[new_col].copy_(old_b[old_col])
                    new_col += 1
        return new_mapper, new_write_idx

    if isinstance(raw_mapper, FanOutInputMapper):
        new_fan_out = {}
        for inp, targets in raw_mapper.fan_out_map.items():
            new_targets = _remap_indices(targets, stage0_remap)
            new_fan_out[inp] = new_targets
        new_mapper = FanOutInputMapper(
            in_dim=raw_mapper.in_dim,
            out_dim=pruned_first_n,
            fan_out_map=new_fan_out,
            x_max=raw_mapper.x_max)
        with torch.no_grad():
            new_mapper.gain.data.copy_(raw_mapper.gain.data)
            new_mapper.bias.data.copy_(raw_mapper.bias.data)
        return new_mapper, None

    if isinstance(raw_mapper, (InputMapper, RobustInputMapper)):
        old_out = raw_mapper.gain.out_features
        surviving_old = sorted(stage0_remap.keys())
        surviving_new = [stage0_remap[o] for o in surviving_old]
        is_identity = (old_out == pruned_first_n and
                       surviving_new == list(range(pruned_first_n)))
        if is_identity:
            return copy.deepcopy(raw_mapper), raw_write_idx
        if isinstance(raw_mapper, RobustInputMapper):
            new_mapper = RobustInputMapper(
                in_dim=in_dim, out_dim=pruned_first_n, x_max=raw_mapper.x_max)
        else:
            new_mapper = InputMapper(
                in_dim=in_dim, out_dim=pruned_first_n, x_max=raw_mapper.x_max)
        with torch.no_grad():
            for old_id, new_id in zip(surviving_old, surviving_new):
                if old_id < old_out:
                    new_mapper.gain.weight.data[new_id].copy_(
                        raw_mapper.gain.weight.data[old_id]
                    )
                    new_mapper.gain.bias.data[new_id].copy_(
                        raw_mapper.gain.bias.data[old_id]
                    )
                else:
                    new_mapper.gain.weight.data[new_id].zero_()
                    new_mapper.gain.bias.data[new_id].zero_()
            if isinstance(raw_mapper, RobustInputMapper):
                new_mapper.log_scale.data.copy_(raw_mapper.log_scale.data)
        return new_mapper, raw_write_idx

    if isinstance(raw_mapper, ResidualTanhInputMapper):
        old_out = raw_mapper.encoder.out_dim
        surviving_old = sorted(stage0_remap.keys())
        surviving_new = [stage0_remap[o] for o in surviving_old]
        is_identity = (old_out == pruned_first_n and
                       surviving_new == list(range(pruned_first_n)))
        if is_identity:
            return copy.deepcopy(raw_mapper), raw_write_idx
        new_mapper = ResidualTanhInputMapper(
            in_dim=in_dim,
            hidden_dim=raw_mapper.hidden_dim,
            out_dim=pruned_first_n,
            x_max=raw_mapper.x_max)
        with torch.no_grad():
            new_lin_w = new_mapper.encoder.W_lin.weight.data
            old_lin_w = raw_mapper.encoder.W_lin.weight.data
            new_lin_b = new_mapper.encoder.W_lin.bias.data
            old_lin_b = raw_mapper.encoder.W_lin.bias.data
            new_w2 = new_mapper.encoder.W_2.weight.data
            old_w2 = raw_mapper.encoder.W_2.weight.data
            for old_id, new_id in zip(surviving_old, surviving_new):
                if old_id < old_out:
                    new_lin_w[new_id].copy_(old_lin_w[old_id])
                    new_lin_b[new_id].copy_(old_lin_b[old_id])
                    new_w2[new_id].copy_(old_w2[old_id])
                else:
                    new_lin_w[new_id].zero_()
                    new_lin_b[new_id].zero_()
                    new_w2[new_id].zero_()
            new_mapper.encoder.W_1.weight.data.copy_(
                raw_mapper.encoder.W_1.weight.data)
            new_mapper.encoder.W_1.bias.data.copy_(
                raw_mapper.encoder.W_1.bias.data)
        return new_mapper, raw_write_idx

    if isinstance(raw_mapper, NullInputMapper):
        # No parameters to transfer — the mapper just returns zeros of
        # shape (batch, out_dim). Recreate with the pruned output dim.
        return NullInputMapper(out_dim=pruned_first_n), None

    raise TypeError(
        f"transfer_input_mapper: unsupported mapper type {type(raw_mapper).__name__}"
    )

def _transfer_output_mapper(raw_mapper, raw_read_idx, last_remap,
                            pruned_last_n, out_dim):
    """Return a new OutputMapper for the pruned stage with weights
    transferred from ``raw_mapper``.

    For sparse read mode, ``raw_read_idx`` is remapped through
    ``last_remap``. If all read nodes survive, position ``i`` in the new
    proj corresponds to old position ``i`` and the proj weight columns
    can be copied directly. If some read nodes were pruned (elastic
    readout), the new mapper has fewer input features; we copy only the
    columns that correspond to surviving read positions, preserving the
    learned readout for those nodes.

    For dense read mode (no read_idx), if last-stage node count
    unchanged, deepcopy. Otherwise, copy ``proj.weight`` columns
    for surviving nodes.
    """
    if isinstance(raw_mapper, OutputMapper):
        if raw_read_idx is not None:
            new_read_idx = _remap_indices(raw_read_idx, last_remap)
            new_mapper = OutputMapper(
                node_dim=pruned_last_n, out_dim=out_dim, read_idx=new_read_idx)
            # Determine which columns of the old weight matrix correspond to
            # surviving read positions (those entries not pruned away).
            surviving_old_positions = [
                i for i, idx in enumerate(raw_read_idx) if idx in last_remap
            ]
            with torch.no_grad():
                new_mapper.proj.weight.data.copy_(
                    raw_mapper.proj.weight.data[:, surviving_old_positions]
                )
                new_mapper.proj.bias.data.copy_(raw_mapper.proj.bias.data)
            return new_mapper, new_read_idx

        old_dim = raw_mapper.proj.in_features
        if old_dim == pruned_last_n:
            return copy.deepcopy(raw_mapper), None
        surviving_old = sorted(last_remap.keys())
        surviving_new = [last_remap[o] for o in surviving_old]
        new_mapper = OutputMapper(node_dim=pruned_last_n, out_dim=out_dim)
        with torch.no_grad():
            new_cols = raw_mapper.proj.weight.data[:, surviving_old]
            new_mapper.proj.weight.data.index_copy_(
                1, torch.tensor(surviving_new, dtype=torch.long), new_cols)
            new_mapper.proj.bias.data.copy_(raw_mapper.proj.bias.data)
        return new_mapper, None

    if isinstance(raw_mapper, ResidualTanhOutputMapper):
        if raw_read_idx is not None:
            new_read_idx = _remap_indices(raw_read_idx, last_remap)
            new_mapper = ResidualTanhOutputMapper(
                in_dim=pruned_last_n,
                hidden_dim=raw_mapper.hidden_dim,
                out_dim=out_dim,
                read_idx=new_read_idx)
            surviving_old_positions = [
                i for i, idx in enumerate(raw_read_idx) if idx in last_remap
            ]
            with torch.no_grad():
                new_mapper.encoder.W_lin.weight.data.copy_(
                    raw_mapper.encoder.W_lin.weight.data[:, surviving_old_positions]
                )
                new_mapper.encoder.W_lin.bias.data.copy_(
                    raw_mapper.encoder.W_lin.bias.data)
                new_mapper.encoder.W_1.weight.data.copy_(
                    raw_mapper.encoder.W_1.weight.data[:, surviving_old_positions]
                )
                new_mapper.encoder.W_2.weight.data.copy_(
                    raw_mapper.encoder.W_2.weight.data)
            return new_mapper, new_read_idx

        old_dim = raw_mapper.encoder.in_dim
        if old_dim == pruned_last_n:
            return copy.deepcopy(raw_mapper), None
        surviving_old = sorted(last_remap.keys())
        surviving_new = [last_remap[o] for o in surviving_old]
        new_mapper = ResidualTanhOutputMapper(
            in_dim=pruned_last_n,
            hidden_dim=raw_mapper.hidden_dim,
            out_dim=out_dim)
        with torch.no_grad():
            new_lin_cols = raw_mapper.encoder.W_lin.weight.data[:, surviving_old]
            new_mapper.encoder.W_lin.weight.data.index_copy_(
                1, torch.tensor(surviving_new, dtype=torch.long), new_lin_cols)
            new_mapper.encoder.W_lin.bias.data.copy_(
                raw_mapper.encoder.W_lin.bias.data)
            new_w1_cols = raw_mapper.encoder.W_1.weight.data[:, surviving_old]
            new_mapper.encoder.W_1.weight.data.index_copy_(
                1, torch.tensor(surviving_new, dtype=torch.long), new_w1_cols)
            new_mapper.encoder.W_2.weight.data.copy_(
                raw_mapper.encoder.W_2.weight.data)
        return new_mapper, None

    if isinstance(raw_mapper, OutputAffine):
        # Learned gain/bias over the output ODE accumulator voltages.
        # Transfer the per-dim gain, bias, tanh_gain, relu_gain, and
        # relu_threshold directly. OutputAffine has no node-dim dependency,
        # so the pruned node count does not change the affine shape.
        new_mapper = OutputAffine(out_dim=out_dim)
        with torch.no_grad():
            new_mapper.gain.data.copy_(raw_mapper.gain.data)
            new_mapper.bias.data.copy_(raw_mapper.bias.data)
            new_mapper.tanh_gain.data.copy_(raw_mapper.tanh_gain.data)
            new_mapper.relu_gain.data.copy_(raw_mapper.relu_gain.data)
            new_mapper.relu_threshold.data.copy_(raw_mapper.relu_threshold.data)
        return new_mapper, None

    raise TypeError(
        f"transfer_output_mapper: unsupported mapper type {type(raw_mapper).__name__}"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Train a reduced differential KirchhoffNet."
    )
    _add_argparse_args(parser)
    args = parser.parse_args()

    # input-norm-seed/Phase 1: reproducible seeding (RNG + cuDNN). The seed
    # is always printed so a random-seed run can be replayed via --seed.
    import random as _random
    if args.seed is None:
        args.seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    print(f"[train] seed={args.seed}")
    # Tie small_world_seed to global seed for graph reproducibility by default.
    # If --small-world-seed is not explicitly set (i.e. is None), inherit from --seed.
    if args.small_world_seed is None:
        args.small_world_seed = args.seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    amp_enabled = args.amp if args.amp is not None else torch.cuda.is_available()
    compile_enabled = args.compile if args.compile is not None else torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parallel_enabled = (
        args.parallel if args.parallel is not None else (n_gpus >= 2)
    )

    epochs = args.epochs if args.epochs is not None else OPTIM["epochs"]
    lr = args.lr if args.lr is not None else OPTIM["lr"]
    weight_decay = (
        args.weight_decay if args.weight_decay is not None
        else float(OPTIM["weight_decay"]))
    batch_size = (
        int(args.batch_size) if args.batch_size is not None
        else int(OPTIM["batch_size"]))

    out_dir = _ensure_dir(args.output.resolve())
    lambdas = _resolve_lambdas(args.problem)
    if args.no_edge_gates:
        lambdas["edge_gate"] = 0.0
        lambdas["sparsity"] = 0.0
        from config import SCHEDULE_THREE_PHASE, SCHEDULE_FOUR_PHASE
        for k in ("sparsity", "edge_gate"):
            if k in SCHEDULE_THREE_PHASE.get("lambdas_b", {}):
                SCHEDULE_THREE_PHASE["lambdas_b"][k] = 0.0
        for phase_key in ("lambdas_b1", "lambdas_b2"):
            if phase_key in SCHEDULE_FOUR_PHASE:
                for k in ("sparsity", "edge_gate"):
                    if k in SCHEDULE_FOUR_PHASE[phase_key]:
                        SCHEDULE_FOUR_PHASE[phase_key][k] = 0.0
    if getattr(args, "tanh_sat_lambda", None) is not None:
        lambdas["tanh_sat"] = float(args.tanh_sat_lambda)
        print(f"[train] tanh_sat lambda overridden via CLI: {lambdas['tanh_sat']}")
    # sparsity/entropy lambda CLI overrides (kn-bayes-opt dims). The schedule
    # functions (three_phase_lambdas / four_phase_lambdas) read sparsity and
    # entropy from the schedule config's lambdas_b/b1/b2/c, NOT from the base
    # lambdas dict. To make the CLI value effective under every schedule, we
    # patch the base lambdas AND the schedule phase targets.
    for _reg_key, _cli_name in (("sparsity", "sparsity_lambda"),
                               ("entropy", "entropy_lambda")):
        if getattr(args, _cli_name, None) is not None:
            _v = float(getattr(args, _cli_name))
            lambdas[_reg_key] = _v
            print(f"[train] {_reg_key} lambda overridden via CLI: {_v}")
            from config import SCHEDULE_THREE_PHASE, SCHEDULE_FOUR_PHASE
            for _phase_key, _map in (
                ("lambdas_b", SCHEDULE_THREE_PHASE),
                ("lambdas_c", SCHEDULE_THREE_PHASE),
            ):
                if _phase_key in _map:
                    _map[_phase_key].setdefault(_reg_key, 0.0)
                    _map[_phase_key][_reg_key] = _v
            for _phase_key in ("lambdas_b1", "lambdas_b2", "lambdas_c"):
                if _phase_key in SCHEDULE_FOUR_PHASE:
                    SCHEDULE_FOUR_PHASE[_phase_key].setdefault(_reg_key, 0.0)
                    SCHEDULE_FOUR_PHASE[_phase_key][_reg_key] = _v
    schedule_mode = _resolve_schedule(args.problem, args.schedule)
    # four-phase-redesign/Phase 1c: apply diagnostic ablation preset
    # overrides BEFORE any other flag is consumed (e.g. before the
    # pruning-threshold resolution below).
    _apply_ablation_set(args, schedule_mode)
    # hidden-family CLI: validate --hidden-family/--num-hidden/
    # --edge-repeats/--num-stages combinations early (cheap, before any
    # expensive setup like data loading or model construction).
    _validate_hidden_family_args(args)
    if schedule_mode == "three_phase":
        a_end, b_end, c_end = phase_boundaries(epochs)
        print(
            f"[train] three_phase schedule: A=[0,{a_end}) B=[{a_end},{b_end}) "
            f"C=[{b_end},{c_end}) (total={epochs})"
        )
    elif schedule_mode == "four_phase":
        fp_a_end, fp_b1_end, fp_b2_end, fp_c_end = four_phase_boundaries(epochs)
        print(
            f"[train] four_phase schedule: A=[0,{fp_a_end}) "
            f"B1=[{fp_a_end},{fp_b1_end}) "
            f"B2=[{fp_b1_end},{fp_b2_end}) "
            f"C=[{fp_b2_end},{fp_c_end}) (total={epochs})"
        )
        if args.freeze_mappers:
            mapper_unfreeze_epoch = fp_a_end + (fp_b2_end - fp_a_end) // 2
            print(
                f"[train] four_phase freeze_mappers: freeze at epoch {fp_a_end}, "
                f"unfreeze at epoch {mapper_unfreeze_epoch}"
            )
        else:
            mapper_unfreeze_epoch = -1
    else:
        mapper_unfreeze_epoch = -1
    if args.freeze_mappers and schedule_mode != "four_phase":
        print(
            f"[train] WARNING: --freeze-mappers ignored for schedule_mode={schedule_mode} "
            f"(only supported for four_phase)"
        )
    # Degree budget / fraction competition (degree-budget-topk plan).
    # Resolve effective config: CLI override > config default.
    budget_enabled = bool(getattr(args, "budget", False)) or bool(
        DEGREE_BUDGET.get("enabled", False)
    )
    budget_frac_start = float(
        args.budget_frac_start if args.budget_frac_start is not None
        else DEGREE_BUDGET["frac_start"]
    )
    budget_frac_end = float(
        args.budget_frac_end if args.budget_frac_end is not None
        else DEGREE_BUDGET["frac_end"]
    )
    budget_temp_start = float(
        args.budget_temp_start if args.budget_temp_start is not None
        else DEGREE_BUDGET["temperature_start"]
    )
    budget_temp_end = float(
        args.budget_temp_end if args.budget_temp_end is not None
        else DEGREE_BUDGET["temperature_end"]
    )
    budget_axis = (
        args.budget_axis if args.budget_axis is not None
        else DEGREE_BUDGET["axis"]
    )
    budget_anneal_frac = float(DEGREE_BUDGET["anneal_frac"])
    if budget_enabled:
        if not (0.0 <= budget_frac_end <= 1.0):
            print(
                f"[train] WARNING: budget_frac_end={budget_frac_end} must be in [0,1], "
                f"clamping to [0,1]"
            )
            budget_frac_end = max(0.0, min(1.0, budget_frac_end))
        if not (0.0 <= budget_frac_start <= 1.0):
            print(
                f"[train] WARNING: budget_frac_start={budget_frac_start} must be in [0,1], "
                f"clamping to [0,1]"
            )
            budget_frac_start = max(0.0, min(1.0, budget_frac_start))
        print(
            f"[train] budget=enabled axis={budget_axis} "
            f"frac: {budget_frac_start:.2f}->{budget_frac_end:.2f} "
            f"temp: {budget_temp_start:.2f}->{budget_temp_end:.2f} "
            f"anneal_frac={budget_anneal_frac}"
        )
        edge_gate_lambda = float(lambdas.get("edge_gate", 0.0))
        if edge_gate_lambda > 0.0:
            print(
                f"[train] WARNING: budget enabled but edge_gate lambda={edge_gate_lambda} > 0. "
                f"The edge_gate regularizer (L1 on σ(z_logits)) may fight the budget. "
                f"Consider setting edge_gate to 0 in the preset's lambdas override."
            )
    if args.no_edge_gates:
        budget_enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device is not None:
        device = args.device
    write_idx_arg = _parse_int_list(args.write_idx)
    read_idx_arg = _parse_int_list(args.read_idx)

    # Resolve cell library: CLI overrides preset, preset overrides default.
    lib_name = "legacy"
    if args.problem in PRESETS:
        lib_name = PRESETS[args.problem].get("cell_library", "legacy")
    if args.cell_library is not None:
        lib_name = args.cell_library
    cell_lib = make_cell_library(
        lib_name,
        gm_min=args.gm_min, gm_max=args.gm_max,
        isat_min=args.isat_min, isat_max=args.isat_max)
    # Per-problem grid-size default (grid7-gate0: smooth2d_grid=7, housing_grid=5).
    # Explicit --grid-size N overrides either.
    if args.problem == "smooth2d_grid":
        resolved_grid_size = args.grid_size if args.grid_size is not None else 7
        PRESETS["smooth2d_grid"] = make_smooth2d_grid_preset(
            grid_size=resolved_grid_size,
            bidirectional=args.bidirectional,
            leak_mode=args.leak,
            leak_constant=args.leak_constant)
    elif args.problem == "housing_grid":
        resolved_grid_size = args.grid_size if args.grid_size is not None else 5
        PRESETS["housing_grid"] = make_housing_grid_preset(
            grid_size=resolved_grid_size,
            bidirectional=args.bidirectional,
            leak_mode=args.leak,
            leak_constant=args.leak_constant)
    else:
        resolved_grid_size = args.grid_size
    args.grid_size = resolved_grid_size
    # hidden-family CLI: when --hidden-family is specified, dynamically
    # rebuild the problem's preset to use the requested family. Done AFTER
    # grid-size resolution so the family branches can read
    # args.grid_size (for error detection).
    if args.hidden_family is not None:
        eff_num_hidden = (
            args.num_hidden if args.num_hidden is not None
            else (resolved_grid_size ** 2 if resolved_grid_size else 5)
        )
        eff_num_stages = args.num_stages if args.num_stages is not None else 1
        eff_edge_repeats = args.edge_repeats if args.edge_repeats is not None else 2
        new_preset = _make_dynamic_preset(
            problem=args.problem,
            hidden_family=args.hidden_family,
            num_hidden=eff_num_hidden,
            num_stages=eff_num_stages,
            edge_repeats=eff_edge_repeats,
            grid_size=resolved_grid_size,
            bidirectional=args.bidirectional,
            write_mode_override=args.write_mode,
            read_mode_override=args.read_mode,
            small_world_k=args.small_world_k,
            small_world_p=args.small_world_p,
            small_world_seed=args.small_world_seed,
            leak_mode=args.leak,
            leak_constant=args.leak_constant,
            t_span=args.t_span,
            num_steps=args.num_steps)
        PRESETS[args.problem] = new_preset
        print(
            f"[train] dynamic preset (hidden_family={args.hidden_family}): "
            f"num_hidden={eff_num_hidden} num_stages={eff_num_stages} "
            f"edge_repeats={eff_edge_repeats} "
            f"bidirectional={args.bidirectional} "
            f"write_mode={new_preset['write_mode']} read_mode={new_preset['read_mode']}"
        )
    # Resolve write_fan_out: explicit CLI override > auto-generate (any family).
    # Done before persistent-drive so it's independent of that flag.
    active_preset = PRESETS.get(args.problem)
    if active_preset is None:
        raise ValueError(f"Unknown problem: {args.problem!r}")
    # Propagate --edge-repeats into every stage of the preset so the CLI flag
    # takes effect even when --hidden-family is NOT specified (the dynamic
    # preset path above is gated on args.hidden_family). Without this, the
    # topology builder would default edge_repeats=1 and silently ignore
    # --edge-repeats for preset-only runs.
    if args.edge_repeats is not None:
        for stage in active_preset.get("stages", []):
            stage["edge_repeats"] = int(args.edge_repeats)
    # Propagate --t-span / --num-steps into every stage of the preset so the
    # CLI overrides apply even when --hidden-family is NOT specified (the
    # dynamic preset path above already wires them). Total-budget semantics:
    # per-stage t_span = args.t_span / num_stages, num_steps = round(total/n).
    # kn-bayes-opt uses these as optimization dimensions.
    stages_list = active_preset.get("stages", [])
    n_stages_eff = len(stages_list) if stages_list else 1
    if n_stages_eff < 1:
        n_stages_eff = 1
    if args.t_span is not None:
        per_stage_t_span = float(args.t_span) / n_stages_eff
        for stage in stages_list:
            stage["t_span"] = per_stage_t_span
    if args.num_steps is not None:
        per_stage_num_steps = round(int(args.num_steps) / n_stages_eff)
        for stage in stages_list:
            stage["num_steps"] = per_stage_num_steps
    if args.write_fan_out is not None:
        try:
            raw = json.loads(args.write_fan_out)
            fan_out_map = {int(k): [int(vv) for vv in v] for k, v in raw.items()}
            # Sanity checks: keys must be >=0, values lists of non-negative ints.
            for k, v in fan_out_map.items():
                if k < 0 or any(x < 0 for x in v):
                    raise ValueError(
                        f"--write-fan-out: indices must be non-negative, got {k}->{v}"
                    )
            num_inputs = int(active_preset["stages"][0]["num_inputs"])
            if any(k >= num_inputs for k in fan_out_map):
                raise ValueError(
                    f"--write-fan-out: input key {max(fan_out_map)} >= num_inputs={num_inputs}"
                )
            # FanOutInputMapper requires each hidden node to be written by at most one input.
            all_targets = [t for tgts in fan_out_map.values() for t in tgts]
            if len(all_targets) != len(set(all_targets)):
                dupes = [t for t in all_targets if all_targets.count(t) > 1]
                raise ValueError(
                    f"--write-fan-out: duplicate target nodes {set(dupes)}; "
                    f"FanOutInputMapper requires each node to be written by at most one input"
                )
            active_preset["write_fan_out"] = fan_out_map
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --write-fan-out JSON: {e}")
    elif args.write_mode == "fan_out" and not active_preset.get("write_fan_out"):
        # Auto-generate fan-out for any family (grid, torus, small_world).
        # _build_grid_write_fan_out works for torus too (same node layout).
        if resolved_grid_size is None:
            raise ValueError(
                f"--write-mode fan_out with no --write-fan-out and no grid-size "
                f"resolved (problem={args.problem!r}). "
                f"Either supply --write-fan-out JSON or specify --hidden-family grid/torus "
                f"with --grid-size N."
            )
        num_inputs = int(active_preset["stages"][0]["num_inputs"])
        active_preset["write_fan_out"] = _build_grid_write_fan_out(
            num_inputs=num_inputs, grid_size=resolved_grid_size)

    # Resolve boundary_fan_out: explicit JSON from --boundary-fan-out.
    # Boundary mode replaces the initial-condition write with continuous
    # OTA edge injection from fixed-voltage input terminals. Mutually
    # exclusive with persistent drive (boundary terminals already inject
    # the signal continuously; adding drive targets on top would be
    # redundant and is rejected).
    boundary_fan_out_parsed = None
    if args.boundary_fan_out is not None:
        try:
            raw = json.loads(args.boundary_fan_out)
            boundary_fan_out_parsed = {int(k): [int(vv) for vv in v] for k, v in raw.items()}
            for k, v in boundary_fan_out_parsed.items():
                if k < 0 or any(x < 0 for x in v):
                    raise ValueError(
                        f"--boundary-fan-out: indices must be non-negative, got {k}->{v}"
                    )
            num_inputs = int(active_preset["stages"][0]["num_inputs"])
            if any(k >= num_inputs for k in boundary_fan_out_parsed):
                raise ValueError(
                    f"--boundary-fan-out: input key {max(boundary_fan_out_parsed)} "
                    f">= num_inputs={num_inputs}"
                )
            all_b_targets = [t for tgts in boundary_fan_out_parsed.values() for t in tgts]
            if len(all_b_targets) != len(set(all_b_targets)):
                dupes = sorted(
                    {t for t in all_b_targets if all_b_targets.count(t) > 1}
                )
                raise ValueError(
                    f"--boundary-fan-out: duplicate target nodes {dupes}"
                )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --boundary-fan-out JSON: {e}")
        if args.persistent_drive:
            raise ValueError(
                "--boundary-fan-out is incompatible with --persistent-drive: "
                "boundary terminals already inject the input signal "
                "continuously; persistent drive would double-inject."
            )

    # Persistent drive: supports fan_out, sparse_proj, and one_to_one modes.
    # write_fan_out is already resolved above; for sparse_proj/one_to_one the
    # driven nodes come from --write-idx (handled by build_net_from_config).
    if args.persistent_drive:
        if args.write_mode is not None and args.write_mode not in (
                "fan_out", "sparse_proj", "one_to_one"):
            raise ValueError(
                "--persistent-drive requires write_mode='fan_out', "
                "'sparse_proj', or 'one_to_one' "
                f"(got --write-mode {args.write_mode!r})"
            )

    # VCA (input-driven attention gating): requires at least one gated
    # edge family (boundary, temporal-readout, or core). With --vca-core
    # (or when no other family exists), the core edge family is gated.
    if args.vca:
        vca_rank_eff = args.vca_rank if args.vca_rank is not None else VCA["rank"]
        if vca_rank_eff < VCA["min_rank"]:
            raise ValueError(
                f"--vca-rank must be >= {VCA['min_rank']}, got {vca_rank_eff}"
            )

    # Readout family resolution (shared-sense-crossbar plan).
    #   --readout linear      = pre-refactor default: linear OutputMapper,
    #                           no accumulator tail, no OutputAffine.
    #   --readout temporal    = legacy h*d_out OTA mesh
    #   --readout shared      = one sense OTA per hidden node + dense crossbar
    #   --readout shared-x2   = two senses per node (different gm operating
    #                           points) + crossbar over 2h senses
    # --temporal-readout (deprecated) is an alias for '--readout temporal';
    # combining it with an explicit linear/shared* choice is contradictory.
    readout_to_mode = {
        "linear": ("ota_mesh", 1),
        "temporal": ("ota_mesh", 1),
        "shared": ("shared_sense", 1),
        "shared-x2": ("shared_sense", 2),
    }
    if args.enable_temporal_readout and args.readout != "temporal":
        raise ValueError(
            f"--temporal-readout is a deprecated alias for '--readout temporal' "
            f"and cannot be combined with --readout {args.readout}."
        )
    readout_mode, readout_senses = readout_to_mode[args.readout]
    # The accumulator tail (``out_dim`` output-ODE nodes appended to every
    # stage + OutputAffine read) is built for the three OTA-tail families
    # (``temporal``, ``shared``, ``shared-x2``) and skipped for ``linear``,
    # which uses the pre-refactor linear OutputMapper over projection nodes.
    # ``readout_mode`` / ``readout_senses`` switch only the edge family inside
    # build_net_from_config; the tail toggle is a separate boolean.
    enable_temporal_readout = (args.readout != "linear")

    net = build_net_from_preset(
        args.problem,
        cell_lib=cell_lib,
        write_mode=args.write_mode,
        read_mode=args.read_mode,
        write_idx=write_idx_arg,
        read_idx=read_idx_arg,
        enable_drive=args.persistent_drive,
        drive_mode=args.drive_mode,
        leak_mode=args.leak,
        leak_constant=args.leak_constant,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        encoder_hidden_dim=args.encoder_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        read_only_source=args.read_only_source,
        freeze_read=args.freeze_read,
        freeze_boundary=args.freeze_boundary,
        freeze_temporal_read=args.freeze_temporal_read,
        interstage_activation=args.interstage_activation,
        interstage_residual_rank=args.interstage_residual_rank,
        enable_skip_linear=args.skip_linear,
        boundary_fan_out=boundary_fan_out_parsed,
        enable_ref_edges=args.enable_ref_edges,
        enable_temporal_readout=enable_temporal_readout,
        readout_mode=readout_mode,
        readout_senses_per_node=readout_senses,
        vca_enabled=args.vca,
        vca_rank=args.vca_rank,
        vca_bias=args.vca_bias,
        vca_core_enabled=args.vca_core,
        vca_gate_shunt=args.vca_gate_shunt,
        vca_separate_core_bus=args.vca_separate_core_bus,
        x_max=args.x_max,
        c_eff=args.c_eff)
    net.to(device)

    if args.no_edge_gates:
        for stage in net.core.stages:
            if hasattr(stage, 'z_logits') and stage.z_logits is not None:
                stage.z_logits.data.fill_(10.0)
                stage.z_logits.requires_grad_(False)
            if getattr(stage, 'budget_enabled', False):
                stage.budget_enabled = False
            if hasattr(stage, 'boundary_z_logits') and stage.boundary_z_logits is not None:
                stage.boundary_z_logits.data.fill_(10.0)
                stage.boundary_z_logits.requires_grad_(False)
            if hasattr(stage, 'ref_z_logits') and stage.ref_z_logits is not None:
                stage.ref_z_logits.data.fill_(10.0)
                stage.ref_z_logits.requires_grad_(False)
            if hasattr(stage, 'output_ode_z_logits') and stage.output_ode_z_logits is not None:
                stage.output_ode_z_logits.data.fill_(10.0)
                stage.output_ode_z_logits.requires_grad_(False)
            if hasattr(stage, 'readout_sense_z_logits') and stage.readout_sense_z_logits is not None:
                stage.readout_sense_z_logits.data.fill_(10.0)
                stage.readout_sense_z_logits.requires_grad_(False)
        print("[train] --no-edge-gates: z_logits (core + boundary + ref + temporal-readout + shared-sense) frozen to +10 (all edges permanently on)")

    # --no-resistive-shunt: bypass the parallel resistive shunt across all
    # stages (no-resistive-shunt). Sets _has_resistive=False so rhs() skips
    # the resistive block, zeros g_resistive_raw (so the static forward()
    # path contributes zero too under --freeze-read), and freezes the
    # parameter so the optimizer ignores it. The tanh_sat lambda is
    # independent of this flag and remains active.
    if args.no_resistive_shunt:
        from cell_library import FreeTanhLibrary as _FreeTanhLibrary
        n_disabled = 0
        for stage in net.core.stages:
            if isinstance(stage.cell_lib, _FreeTanhLibrary) and stage._has_resistive:
                stage._has_resistive = False
                stage.cell_lib.g_resistive_raw.data.zero_()
                stage.cell_lib.g_resistive_raw.requires_grad_(False)
                n_disabled += 1
            # Shared-sense readout family: disable its resistive shunt too
            # (rhs() checks ``stage._sense_has_resistive``; the zeroed +
            # frozen g_resistive_raw also zeroes the static forward path).
            sense_lib = getattr(stage, "readout_sense_cell_lib", None)
            if getattr(stage, "_has_shared_readout", False) and isinstance(
                sense_lib, _FreeTanhLibrary
            ) and stage._sense_has_resistive:
                stage._sense_has_resistive = False
                sense_lib.g_resistive_raw.data.zero_()
                sense_lib.g_resistive_raw.requires_grad_(False)
                n_disabled += 1
        print(f"[train] resistive shunt disabled on {n_disabled} "
              f"stage(s) (--no-resistive-shunt)")
    grid_label = (
        f" {args.grid_size}×{args.grid_size} grid,"
        if args.problem in ("smooth2d_grid", "housing_grid")
        else ""
    )
    in_mapper_name = type(net.input_mapper).__name__
    out_mapper_name = type(net.output_mapper).__name__
    print(
        f"[train] problem={args.problem}{grid_label} epochs={epochs} lr={lr} device={device} "
        f"output={out_dir} amp={amp_enabled} compile={compile_enabled} "
        f"parallel={parallel_enabled} ({n_gpus} GPUs) "
        f"validate_every={args.validate_every} early_stop={args.early_stop} "
        f"ablation={args.ablation} variation={args.variation} "
        f"input_mapper={in_mapper_name} output_mapper={out_mapper_name} "
        f"hid_count={net.hid_count} proj_count={net.proj_count} "
        f"write_idx={list(net.write_idx) if net.write_idx is not None else None} "
        f"read_idx={list(net.read_idx) if net.read_idx is not None else None} "
        f"readout={args.readout} "
        f"freeze_read={args.freeze_read} "
        f"freeze_boundary={args.freeze_boundary} "
        f"freeze_temporal_read={args.freeze_temporal_read}"
    )
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_params:,}")
    if args.count_params_only:
        return
    breakdown = net.parameter_breakdown()
    print(f"[train] param breakdown:\n{format_parameter_breakdown(breakdown)}")
    if getattr(net, "skip_linear_enabled", False):
        skip_l2_lambda = _resolve_lambdas(args.problem).get("skip_linear_l2", 0.0)
        print(
            f"[train] skip_linear ENABLED: "
            f"y = W₁·u + b₁ + f(x), "
            f"shape={net.skip_linear_in_dim}->{net.skip_linear_out_dim}, "
            f"L2 lambda={skip_l2_lambda}"
        )
    if args.device_l2_lambda > 0.0:
        from cell_library import FreeTanhLibrary as _FreeTanhLibrary
        n_tanh_free = sum(
            getattr(s, "cell_lib", None) is not None
            and isinstance(s.cell_lib, _FreeTanhLibrary)
            for s in net.core.stages
        )
        n_boundary = sum(
            getattr(s, "boundary_cell_lib", None) is not None
            and isinstance(s.boundary_cell_lib, _FreeTanhLibrary)
            for s in net.core.stages
        )
        n_temporal = sum(
            getattr(s, "output_ode_cell_lib", None) is not None
            and isinstance(s.output_ode_cell_lib, _FreeTanhLibrary)
            for s in net.core.stages
        )
        n_sense = sum(
            getattr(s, "_has_shared_readout", False)
            and getattr(s, "readout_sense_cell_lib", None) is not None
            and isinstance(s.readout_sense_cell_lib, _FreeTanhLibrary)
            for s in net.core.stages
        )
        print(
            f"[train] device L2 penalty ENABLED: lambda={args.device_l2_lambda:.2e} "
            f"(scale-invariant mean over FreeTanh cell params, "
            f"interstage residual weights, OutputAffine, VCA; s_raw skipped; "
            f"shared-sense params + dense crossbar readout_crossbar_W are "
            f"included as plain L2 targets, not the scale-invariant device "
            f"path). "
            f"tanh_free cell_lib families: core={n_tanh_free}, "
            f"boundary={n_boundary}, temporal-readout={n_temporal}, "
            f"shared-sense={n_sense}"
        )
    if getattr(net, "enable_vca", False):
        vca_rank_eff = net.vca_rank if hasattr(net, "vca_rank") else VCA["rank"]
        n_b_edges = sum(
            len(s.boundary_src) for s in net.core.stages if hasattr(s, "boundary_src")
        )
        # Shared-sense readout: the gated readout family is the sense bank
        # (s*h cells); otherwise the legacy h*d_out temporal mesh. ``linear``
        # mode has no readout tail at all so the gated family is empty.
        n_r_edges = sum(
            (len(s.readout_sense_src) if getattr(s, "_has_shared_readout", False)
             else len(s.output_ode_src))
            for s in net.core.stages
            if hasattr(s, "readout_sense_src") or hasattr(s, "output_ode_src")
        )
        if args.readout == "shared" or args.readout == "shared-x2":
            label_r = "shared-sense"
        elif args.readout == "temporal":
            label_r = "temporal-readout"
        else:
            label_r = "(no readout tail)"
        print(
            f"[train] VCA ENABLED: rank={vca_rank_eff}, "
            f"total gated edges: {n_b_edges} boundary + {n_r_edges} {label_r}"
        )
    # Resolve the actually-applied edge_repeats from the preset (post the
    # CLI-injection step). When multiple stages disagree, fall back to None.
    actual_er_per_stage = [
        stage.get("edge_repeats", 1) for stage in active_preset.get("stages", [])
    ]
    if actual_er_per_stage:
        actual_er = actual_er_per_stage[0] if len(set(actual_er_per_stage)) == 1 else None
    else:
        actual_er = None
    eff_er = actual_er if actual_er is not None else 1
    if args.bidirectional or eff_er > 1 or (
        args.edge_repeats is not None and args.edge_repeats != eff_er
    ):
        mult = (2 if args.bidirectional else 1) * eff_er
        mismatch = (
            args.edge_repeats is not None
            and args.edge_repeats != eff_er
        )
        edges_per_stage = [s.num_edges() for s in net.core.stages]
        suffix = " (CLI flag did NOT match preset)" if mismatch else ""
        if args.bidirectional:
            print(
                f"[train] bidirectional={args.bidirectional} "
                f"edge_repeats={eff_er}: {edges_per_stage} edges per stage "
                f"({mult}× single-edge baseline){suffix}"
            )
        else:
            print(
                f"[train] edge_repeats={eff_er}: {edges_per_stage} edges per stage "
                f"({mult}× single-edge baseline){suffix}"
            )
    if args.solver == "deq" and not args.persistent_drive:
        print(
            "[train] WARNING: --solver deq without --persistent-drive: "
            "the equilibrium x* has no input-dependent forcing term, "
            "so the network can only fit a constant target. "
            "Add --persistent-drive (compatible with write_mode='fan_out', "
            "'sparse_proj', or 'one_to_one') for input-dependent fixed points."
        )

    _save_config_snapshot(out_dir, args.problem, args, lambdas, net=net)

    effective_write_mode = (
        "fan_out" if isinstance(net.input_mapper, FanOutInputMapper)
        else "one_to_one" if isinstance(net.input_mapper, SparseInputMapper)
        else "sparse_proj" if isinstance(net.input_mapper, ProjectedSparseInputMapper)
        else "dense"
    )
    effective_read_mode = "sparse" if net.read_idx is not None else "dense"

    # Friedman problems use --target-noise-std; others ignore it.
    data_out = make_data(
        args.problem, batch_size,
        noise_std=args.target_noise_std,
        normalize_inputs=args.normalize_inputs,
    )
    if len(data_out) == 4:
        train_loader, val_loader, task_fn, inverse_stats = data_out
    else:
        train_loader, val_loader, task_fn = data_out
        inverse_stats = None

    if args.ablation != "none":
        apply_ablation(net, args.ablation)

    if parallel_enabled and n_gpus >= 2:
        compile_enabled = False
        print("[train] disabling compile for DataParallel compatibility (use single GPU for compile)")

    if compile_enabled and isinstance(device, str) and device.startswith("cuda"):
        try:
            cell_lib.compile_forward()
            for stage in net.core.stages:
                if stage.num_edges() > 0:
                    stage.compile_rhs()
            print("[train] torch.compile enabled on cell_lib.forward and stage.rhs")
        except Exception as e:
            print(f"[train] torch.compile setup failed: {e}; continuing without compile")
            compile_enabled = False

    if parallel_enabled and n_gpus >= 2:
        try:
            net = torch.nn.DataParallel(net, device_ids=list(range(n_gpus)))
            print(f"[train] DataParallel enabled on {n_gpus} GPUs")
        except Exception as e:
            print(f"[train] DataParallel setup failed: {e}; continuing single-GPU")

    def _unwrap(m):
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    raw_net = _unwrap(net)

    # kirchhoff-noise: build the noise-aware training wrapper if requested.
    # When --noise-aware is set, we use ``train_wrapper`` for forward passes
    # inside compute_loss (the regularizer monkey-patch above unwraps to base
    # for stage access). When only --noise is set, the wrapper is built later
    # for post-training MC eval.
    train_wrapper: "KirchhoffNetNoiseWrapper | None" = None
    train_noise_cfg: "NoiseConfig | None" = None
    if args.noise_aware:
        from analog_noise import NoiseConfig
        from kirchhoff_noise import KirchhoffNetNoiseWrapper
        train_noise_cfg = NoiseConfig(
            quant_bits=args.quant_bits,
            noise_std=args.noise_std,
            quantize_input=True,
            quantize_output=True,
            quantize_intermediate=True,
            weight_noise=True,
            activation_noise=True,
            seed=args.noise_seed)
        # Wrap the BASE KirchhoffNetWithIO so the wrapper.base is the
        # unwrapped model. The wrapper's parameters() / state_dict()
        # delegate to base, so DataParallel state_dict/parameters of
        # the original net still work. We then mirror ``net``'s
        # DataParallel wrapping so the noise-aware forward still
        # benefits from multi-GPU parallelism. The regularizer monkey
        # patch unwraps both layers to reach the base stages.
        train_wrapper = KirchhoffNetNoiseWrapper(
            raw_net, train_noise_cfg, adc_full_range=args.adc_full_range)
        train_wrapper.to(device)
        if parallel_enabled and n_gpus >= 2:
            train_wrapper = torch.nn.DataParallel(
                train_wrapper, device_ids=list(range(n_gpus)))
            print(f"[noise] DataParallel enabled on {n_gpus} GPUs for train_wrapper")
        print(
            f"[noise] noise-aware training: quant_bits={args.quant_bits} "
            f"noise_std={args.noise_std} adc_full_range={args.adc_full_range} "
            f"seed={args.noise_seed}"
        )

    if args.noise or args.noise_aware:
        print(
            f"[noise] {'noise-aware training + ' if args.noise_aware else ''}"
            f"post-training MC eval enabled: quant_bits={args.quant_bits} "
            f"noise_std={args.noise_std} mc_trials={args.mc_trials} "
            f"adc_full_range={args.adc_full_range} seed={args.noise_seed}"
        )

    if args.variation:
        gss = VARIATION["global_gain_shift_std"]
        ems = VARIATION["edge_mismatch_std"]
        giss = VARIATION.get("global_isat_shift_std", 0.0)
        ims = VARIATION.get("edge_isat_mismatch_std", 0.0)
        print(
            f"[variation] enabled: global_gain_shift_std={gss} "
            f"edge_mismatch_std={ems} global_isat_shift_std={giss} "
            f"edge_isat_mismatch_std={ims}"
        )
        # Capture raw_net by value via default arg so the closure
        # does not break when raw_net is later reassigned (e.g. to
        # None during the free-memory step before building pruned_net).
        _raw_net = raw_net
        def ctx_factory(batch_size_: int, device: torch.device = device, **_):
            total_edges = sum(s.num_edges() for s in _raw_net.core.stages)
            return sample_random_context(
                num_edges=total_edges,
                device=device,
                gain_shift_std=gss,
                mismatch_std=ems,
                global_isat_shift_std=giss,
                isat_mismatch_std=ims)
    else:
        ctx_factory = make_static_ctx_factory()

    optimizer = make_optimizer(
        net, lr=lr,
        weight_decay=weight_decay,
        stage_lr_scale=args.stage_lr_scale,
        mapper_lr_scale=args.mapper_lr_scale,
        struct_lr_scale=args.struct_lr_scale,
        dyn_lr_scale=args.dyn_lr_scale)
    if any(abs(s - 1.0) > 1e-6 for s in [args.stage_lr_scale, args.mapper_lr_scale,
                                          args.struct_lr_scale, args.dyn_lr_scale]):
        lr_strs = [f"{g['lr']:.1e}" for g in optimizer.param_groups]
        print(
            f"[train] scales: stage={args.stage_lr_scale}, mapper={args.mapper_lr_scale}, "
            f"struct={args.struct_lr_scale}, dyn={args.dyn_lr_scale}: "
            f"per-group LRs = {lr_strs}"
        )
    if args.use_scheduler:
        if schedule_mode == "three_phase":
            _, _ab_end, _ = phase_boundaries(epochs)
            sched_tmax = max(1, _ab_end)
        else:
            sched_tmax = max(1, epochs)
        if args.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=sched_tmax,
                eta_min=OPTIM["scheduler_eta_min"])
        else:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=OPTIM["scheduler_T_0"],
                T_mult=OPTIM["scheduler_T_mult"],
                eta_min=OPTIM["scheduler_eta_min"])
    else:
        scheduler = None

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = (
        torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    )

    history = []
    val_history = []

    for i, stage in enumerate(raw_net.core.stages):
        plot_stage_graph(
            stage, save_path=str(out_dir / f"stage{i + 1}_graph_init.png"),
            title=f"{args.problem} — Stage {i + 1} (init)")

    print("[train] starting training loop")
    try:
        from tqdm import tqdm
    except ImportError:
        warnings.warn("tqdm not installed, falling back to plain prints", stacklevel=2)
        tqdm = None

    grad_log_path = out_dir / "grad_norms.txt" if args.grad_log else None
    update_norms_path = out_dir / "update_norms.txt" if args.grad_log else None
    vca_gates_path = out_dir / "vca_gates.txt" if getattr(args, "vca_diag", False) else None
    solid_log_path = out_dir / "solidification_metrics.txt" if schedule_mode in ("three_phase", "four_phase") else None

    # ---------- Determine effective training scope ----------
    # Three-phase: Phase A+B use the overcomplete network (epochs 0..b_end).
    # Four-phase: Phase A+B1+B2 use the overcomplete network (epochs 0..b2_end).
    # Legacy: single phase over all epochs.
    if schedule_mode == "three_phase":
        a_end, b_end, _ = phase_boundaries(epochs)
        ab_total = b_end  # Phase A+B epoch count
        c_total = max(1, epochs - b_end)
        needs_prune = True  # three_phase always prunes at the B→C boundary
    elif schedule_mode == "four_phase":
        fp_a_end, fp_b1_end, fp_b2_end, _ = four_phase_boundaries(epochs)
        a_end = fp_a_end
        b1_end = fp_b1_end
        b2_end = fp_b2_end
        ab_total = b2_end  # Phase A+B1+B2 epoch count
        c_total = max(1, epochs - b2_end)
        needs_prune = False  # four_phase handles pruning inline during B2
    else:
        ab_total = epochs
        c_total = 0
        needs_prune = args.prune

    if args.no_edge_gates:
        needs_prune = False

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    best_metric_name = "val"  # four-phase-redesign/Phase 1b: track which metric
    epochs_without_improve = 0
    stop_training = False

    # four-phase-redesign: teacher distillation state
    teacher_net = None
    teacher_frozen = False
    # Readiness tracking for four_phase B2
    readiness_histories: dict = {}
    readiness_prune_fired = False
    readiness_prune_epoch = -1

    # ---- Phase A + B: overcomplete training ----
    tau_anneal_enabled = PRESETS[args.problem].get("tau_anneal", True)
    if schedule_mode == "legacy" and tau_anneal_enabled:
        tau_kwargs = {}
        if args.prune:
            tau_kwargs["tau_final"] = float(TAU.get("final_pretrain", TAU["final"]))
    if schedule_mode == "three_phase":
        ab_desc = f"train {args.problem} [A+B]"
    elif schedule_mode == "four_phase":
        ab_desc = f"train {args.problem} [A+B1+B2]"
    else:
        ab_desc = f"train {args.problem} [ablation={args.ablation}]"
    ab_iter = (
        tqdm(range(ab_total), desc=ab_desc, unit="epoch")
        if tqdm is not None else range(ab_total)
    )
    # four-phase-redesign: solidification metrics stored for readiness check
    solid_metrics_history: list[dict] = []
    # Validate-only histories (matching cadence of solid_metrics_history)
    # Avoids duplicate entries on non-validate epochs for the readiness check.
    val_v_history: list[float] = []
    val_argmax_history = None
    val_argmax_v_history = None
    # housing_grid: per-epoch dicts of {val, mae_orig, rmse_orig} for
    # logging in original housing-price units (USD x 100k).
    val_orig_history: list[dict] = [] if inverse_stats is not None else None
    # mapper-lr-control: freeze state for --freeze-mappers (four_phase only)
    mappers_frozen = False

    # DEQ (deq-core-prototype): bind the active solver from CLI. Heun
    # remains the default so existing runs are unaffected.
    solver = getattr(args, "solver", "heun")
    deq_cfg = _build_deq_cfg(args)
    deq_train_log_path = out_dir / "deq_training_history.txt" if solver == "deq" else None
    deq_val_log_path = out_dir / "deq_validation_history.txt" if solver == "deq" else None

    train_start_time = time.time()

    for epoch in ab_iter:
        if stop_training:
            break
        net.train()

        # Degree budget / fraction competition (degree-budget-topk plan).
        # Recompute frac and temperature for this epoch and push to all
        # stages. Done once per epoch (global schedule), the per-batch
        # budget gate is recomputed inside rhs() from the stage's current
        # budget_frac / budget_temperature.
        if budget_enabled:
            _b_frac = budget_frac_for_epoch(
                epoch, ab_total, budget_frac_start, budget_frac_end, budget_anneal_frac)
            _b_T = budget_temperature_for_epoch(
                epoch, ab_total, budget_temp_start, budget_temp_end, budget_anneal_frac)
            for _stage in raw_net.core.stages:
                _stage.set_budget_frac(_b_frac, _b_T)
                _stage.budget_axis = budget_axis

        if schedule_mode == "four_phase":
            tau = four_phase_tau(epoch, epochs)
            phase = phase_for_epoch_four(epoch, epochs)
            effective_lambdas = four_phase_lambdas(epoch, epochs, lambdas)
            reg_scale = 1.0  # four_phase_lambdas already includes warmup

            # Teacher cloning at phase transition A -> B1.
            if phase in ("B1", "B2") and not teacher_frozen and best_state is not None:
                # Deep-copy the architecture and load the best Phase A
                # checkpoint into the frozen teacher.
                import copy as _copy
                raw = net.module if isinstance(net, torch.nn.DataParallel) else net
                teacher_net = _copy.deepcopy(raw)
                teacher_net.load_state_dict(best_state)
                teacher_net.requires_grad_(False)
                teacher_net.eval()
                teacher_net.to(device)
                teacher_frozen = True
                print(
                    f"[teacher] cloned best Phase A state (epoch {best_epoch}, "
                    f"{best_metric_name}={best_val:.4f}) as frozen teacher"
                )

            # mapper-lr-control: freeze mappers at A->B1 boundary, unfreeze
            # at the midpoint of the combined B1+B2 duration. Only when
            # --freeze-mappers is set and schedule_mode is four_phase.
            if args.freeze_mappers:
                raw = net.module if isinstance(net, torch.nn.DataParallel) else net
                if epoch == fp_a_end and not mappers_frozen:
                    raw.input_mapper.requires_grad_(False)
                    raw.output_mapper.requires_grad_(False)
                    mappers_frozen = True
                    print(
                        f"[mapper] frozen mappers at epoch {epoch} "
                        f"(unfreeze at {mapper_unfreeze_epoch})"
                    )
                elif epoch == mapper_unfreeze_epoch and mappers_frozen:
                    raw.input_mapper.requires_grad_(True)
                    raw.output_mapper.requires_grad_(True)
                    mappers_frozen = False
                    print(
                        f"[mapper] unfrozen mappers at epoch {epoch} "
                        f"(lr_scale={args.mapper_lr_scale})"
                    )

            # If readiness triggered on a previous validate step, stop here.
            if phase == "B2" and readiness_prune_fired:
                print(
                    f"[four_phase] readiness pruning triggered at epoch {epoch} "
                    f"(B2 cut short, was scheduled for {b2_end})"
                )
                break

        elif schedule_mode == "three_phase":
            tau = three_phase_tau(epoch, epochs)
            phase = phase_for_epoch(epoch, epochs)
            effective_lambdas = three_phase_lambdas(epoch, epochs, lambdas)
            reg_scale = 1.0  # three_phase_lambdas already includes warmup
        else:
            phase = ""
            if tau_anneal_enabled:
                tau = tau_for_epoch(epoch, total_epochs=epochs, **tau_kwargs)
            else:
                tau = float(TAU["init"])
            reg_scale = reg_schedule(epoch)
            effective_lambdas = lambdas

        # : per-epoch cell selection mode.
        # 'auto' uses 'ste' for Phase B/C of phased schedules.
        # DEQ requires soft cell mode (forward_equilibrium enforces it).
        deq_cfg = _build_deq_cfg(args)

        total_loss = 0.0
        n_batches = 0
        should_log_grads = grad_log_path is not None and epoch % args.grad_log_every == 0
        epoch_grad_norms = None
        param_snapshots: dict[str, torch.Tensor] | None = None
        if should_log_grads:
            raw_net_snapshot = net.module if isinstance(net, torch.nn.DataParallel) else net
            param_snapshots = {name: p.data.detach().clone() for name, p in raw_net_snapshot.named_parameters()}
        train_deq_weight = 0
        train_deq_residual_sum = 0.0
        train_deq_residual_max = 0.0
        train_deq_nstep_sum = 0.0
        train_deq_nstep_max = 0.0
        train_deq_abs_state_max = 0.0
        for batch in train_loader:
            ctx = ctx_factory(batch[0].size(0), device=device)
            optimizer.zero_grad()
            u, target = batch
            u = u.to(device)
            target = target.to(device)
            # four-phase-redesign/Phase 3c: pass teacher for KD when active.
            kd_teacher = None
            kd_lambda = 0.0
            if schedule_mode == "four_phase" and teacher_frozen and four_phase_kd_active(epoch, epochs):
                kd_teacher = teacher_net
                kd_lambda = float(SCHEDULE_FOUR_PHASE.get("lambda_kd", 1.0))
            loss_task, loss_structural, _ = compute_loss(
                train_wrapper if train_wrapper is not None else net,
                u, target, ctx, task_fn,
                lambdas=effective_lambdas, return_parts=True,
                amp=amp_enabled, amp_dtype=amp_dtype, reg_scale=reg_scale,
                solver=solver, deq_cfg=deq_cfg,
                teacher=kd_teacher, kd_lambda=kd_lambda,
                device_l2_lambda=args.device_l2_lambda)
            if scaler is not None and scaler._enabled:
                ( scaler.scale(loss_task) + scaler.scale(loss_structural) ).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=OPTIM["grad_clip_norm"])
                if should_log_grads:
                    epoch_grad_norms = collect_gradient_norms(raw_net)
                scaler.step(optimizer)
                scaler.update()
            else:
                (loss_task + loss_structural).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=OPTIM["grad_clip_norm"])
                if should_log_grads:
                    epoch_grad_norms = collect_gradient_norms(raw_net)
                optimizer.step()
            total_loss += float((loss_task + loss_structural).item())
            n_batches += 1
            if solver == "deq" and not isinstance(net, torch.nn.DataParallel):
                batch_stats = _deq_batch_stats(raw_net, solver=solver, deq_cfg=deq_cfg)
                if batch_stats is not None:
                    bs = int(u.size(0))
                    train_deq_weight += bs
                    train_deq_residual_sum += batch_stats["residual_mean"] * bs
                    train_deq_residual_max = max(train_deq_residual_max, batch_stats["residual_max"])
                    train_deq_nstep_sum += batch_stats["nstep_mean"] * bs
                    train_deq_nstep_max = max(train_deq_nstep_max, batch_stats["nstep_max"])
                    train_deq_abs_state_max = max(train_deq_abs_state_max, batch_stats["max_abs_state"])

        avg_train = total_loss / max(1, n_batches)
        train_deq_metrics = None
        if solver == "deq" and train_deq_weight > 0:
            train_deq_metrics = {
                "residual_mean": train_deq_residual_sum / train_deq_weight,
                "residual_max": train_deq_residual_max,
                "nstep_mean": train_deq_nstep_sum / train_deq_weight,
                "nstep_max": train_deq_nstep_max,
                "max_abs_state": train_deq_abs_state_max}
            if deq_train_log_path is not None:
                _append_deq_train_row(
                    deq_train_log_path,
                    epoch=epoch,
                    phase=phase,
                    train_loss=avg_train,
                    metrics=train_deq_metrics)
        do_validate = (epoch % args.validate_every == 0) or (epoch == ab_total - 1)
        if do_validate:
            val_deq_metrics = None
            if inverse_stats is not None:
                if solver == "deq":
                    val_metrics = validate_with_inverse(
                        net, val_loader, task_fn, ctx_factory, device,
                        inverse_stats=inverse_stats,
                        solver=solver, deq_cfg=deq_cfg,
                        collect_deq_metrics=True)
                    val_deq_metrics = val_metrics.pop("deq", None)
                else:
                    val_metrics = validate_with_inverse(
                        net, val_loader, task_fn, ctx_factory, device,
                        inverse_stats=inverse_stats,
                        solver=solver, deq_cfg=deq_cfg)
                val_loss = val_metrics["val"]
            else:
                if solver == "deq":
                    val_loss, val_deq_metrics = validate(
                        net, val_loader, task_fn, ctx_factory, device,
                        solver=solver, deq_cfg=deq_cfg,
                        collect_deq_metrics=True)
                else:
                    val_loss = validate(
                        net, val_loader, task_fn, ctx_factory, device,
                        solver=solver, deq_cfg=deq_cfg)
                val_metrics = None
            val_v_history.append(val_loss)
            if inverse_stats is not None and val_metrics is not None:
                val_orig_history.append(val_metrics)
            if deq_val_log_path is not None and val_deq_metrics is not None:
                _append_deq_validation_row(
                    deq_val_log_path,
                    epoch=epoch,
                    phase=phase,
                    split="val",
                    train_loss=avg_train,
                    val_loss=val_loss,
                    metrics=val_deq_metrics)
            # Argmax validation for phased schedules
            if solid_log_path is not None and phase in ("A", "B", "B1", "B2"):
                _log_solidification(solid_log_path, epoch, {})
                # Store for four_phase readiness check.
                if schedule_mode == "four_phase":
                    solid_metrics_history.append(metrics)

        else:
            val_loss = val_history[-1] if val_history else avg_train
            if inverse_stats is not None and val_orig_history is not None and val_orig_history:
                val_orig_history.append(val_orig_history[-1])

        history.append(avg_train)
        val_history.append(val_loss)

        if train_deq_metrics is not None:
            print(f"[deq-train] epoch {epoch:4d}  train={avg_train:.4f}" + _format_deq_summary(train_deq_metrics))
        if val_deq_metrics is not None:
            print(_format_deq_summary(val_deq_metrics))

        if do_validate:
            sel_metric = float(val_loss)
            sel_name = "val"
            if sel_metric < best_val - args.min_delta:
                best_val = sel_metric
                best_epoch = epoch
                best_metric_name = sel_name
                epochs_without_improve = 0
                raw = net.module if isinstance(net, torch.nn.DataParallel) else net
                best_state = {k: v.detach().clone() for k, v in raw.state_dict().items()}
            else:
                epochs_without_improve += args.validate_every
                if args.early_stop and epochs_without_improve >= args.patience:
                    print(
                        f"[train] early stopping at epoch {epoch}: "
                        f"no {best_metric_name} improvement for {epochs_without_improve} epochs "
                        f"(best {best_metric_name}={best_val:.4f} @ epoch {best_epoch})"
                    )
                    stop_training = True

        if scheduler is not None:
            scheduler.step()

        if should_log_grads:
            raw = net.module if isinstance(net, torch.nn.DataParallel) else net
            log_gradient_norms(grad_log_path, epoch, raw, optimizer=optimizer, norms=epoch_grad_norms)
            if param_snapshots is not None and update_norms_path is not None:
                update_norms = compute_update_norms(param_snapshots, net)
                log_update_norms(update_norms_path, epoch, update_norms, phase=phase)
                param_snapshots = None  # free snapshot memory

        if vca_gates_path is not None and epoch % args.grad_log_every == 0:
            log_vca_gates(vca_gates_path, epoch, net, val_loader, device,
                          phase=phase)

        _lrs = [g["lr"] for g in optimizer.param_groups]
        lr_str = f"{min(_lrs):.1e}..{max(_lrs):.1e}" if len(_lrs) > 1 else f"{_lrs[0]:.2e}"
        phase_tag = f" [{phase}]" if phase else ""
        if tqdm is not None:
            print(
                f"{ab_desc}  train={avg_train:.4e}  val={val_loss:.4e}"
                + f"  tau={tau:.3f}  lr={lr_str}"
                + (f"  reg={reg_scale:.2f}" if schedule_mode == "legacy" else "")
            )
            ab_iter.set_description_str("")
        else:
            print_str = (
                f"  epoch {epoch:4d}{phase_tag}  train={avg_train:.4e}  val={val_loss:.4e}"
                + f"  tau={tau:.3f}  lr={lr_str}"
            )
            print(print_str)

    # ---- DEQ diagnostics on the trained model (deq-core-prototype) ----
    _log_gpu_mem("after_phase_ab")
    if getattr(args, "run_deq_diagnostics", False):
        _run_deq_diagnostics_report(net, device, ctx_factory, deq_cfg)

    # retrain-oom-fix/REQ-4: log GPU memory after DEQ diagnostics so we
    # can see the baseline pressure before pruning duplicates the model.
    _log_gpu_mem("after_deq_diagnostics")

    # ---- End of Phase A+B (or A+B1+B2) ----
    if schedule_mode == "three_phase" and not stop_training:
        a_end, b_end, _ = phase_boundaries(epochs)
        print(
            f"[phase] A+B complete (epoch {b_end}), best {best_metric_name}={best_val:.4f} "
            f"@ epoch {best_epoch}"
        )
    elif schedule_mode == "four_phase":
        if stop_training and not teacher_frozen:
            # Early stopped in Phase A — no pruning, just save.
            print(
                f"[four_phase] early stop in Phase A at epoch {epoch}, "
                f"skipping prune+retrain"
            )
        else:
            if readiness_prune_fired:
                print(
                    f"[four_phase] A+B1+B2 complete, prune triggered at epoch "
                    f"{readiness_prune_epoch}, best {best_metric_name}={best_val:.4f} "
                    f"@ epoch {best_epoch}"
                )
            elif stop_training and teacher_frozen:
                # Early stop during B1/B2 — still prune since compression
                # has already begun.
                print(
                    f"[four_phase] early stop at epoch {epoch} during "
                    f"B1/B2, pruning anyway"
                )
            else:
                print(
                    f"[four_phase] A+B1+B2 complete (fallback prune at epoch {b2_end}), "
                    f"best {best_metric_name}={best_val:.4f} @ epoch {best_epoch}"
                )
                readiness_prune_fired = True
            needs_prune = True

    history_path = out_dir / "loss_history.txt"
    with open(history_path, "w") as f:
        has_orig = val_orig_history is not None and len(val_orig_history) == len(val_history)
        if has_orig:
            f.write("epoch\ttrain\tval\tmae_orig\trmse_orig\tmse_orig\tmape_orig\tphase\n")
            for i, (t, v, m) in enumerate(zip(history, val_history, val_orig_history)):
                if schedule_mode == "three_phase":
                    p = phase_for_epoch(i, epochs)
                elif schedule_mode == "four_phase":
                    p = phase_for_epoch_four(i, epochs)
                else:
                    p = "A"
                f.write(f"{i}\t{t}\t{v}\t{m['mae_orig']:.6f}\t{m['rmse_orig']:.6f}\t{m['mse_orig']:.6f}\t{m['mape_orig']:.6f}\t{p}\n")
        else:
            f.write("epoch\ttrain\tval\tphase\n")
            for i, (t, v) in enumerate(zip(history, val_history)):
                if schedule_mode == "three_phase":
                    p = phase_for_epoch(i, epochs)
                elif schedule_mode == "four_phase":
                    p = phase_for_epoch_four(i, epochs)
                else:
                    p = "A"
                f.write(f"{i}\t{t}\t{v}\t{p}\n")

    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val", color="C3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(
        f"{args.problem} — training loss [ablation={args.ablation}, variation={args.variation}]"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    final_metrics_path = out_dir / "final_metrics.txt"
    elapsed = time.time() - train_start_time
    has_best_orig = (
        val_orig_history is not None
        and len(val_orig_history) > 0
        and 0 <= best_epoch < len(val_orig_history)
        and val_orig_history[best_epoch] is not None
    )
    with open(final_metrics_path, "w") as f:
        f.write(f"problem: {args.problem}\n")
        f.write(f"loss: {PRESETS[args.problem]['loss']}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val: {best_val:.6f}\n")
        if has_best_orig:
            m = val_orig_history[best_epoch]
            f.write(f"best_mse_orig: {m['mse_orig']:.6f}\n")
            f.write(f"best_rmse_orig: {m['rmse_orig']:.6f}\n")
            f.write(f"best_mae_orig: {m['mae_orig']:.6f}\n")
            f.write(f"best_mape_orig: {m['mape_orig']:.6f}\n")
        else:
            f.write("best_mse_orig: nan\n")
            f.write("best_rmse_orig: nan\n")
            f.write("best_mae_orig: nan\n")
            f.write("best_mape_orig: nan\n")
        f.write(f"epochs_run: {len(history)}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")
        f.write(f"param_count: {n_params}\n")

    if best_state is not None:
        raw = net.module if isinstance(net, torch.nn.DataParallel) else net
        raw.load_state_dict(best_state)
        print(
            f"[train] restored best model from epoch {best_epoch} "
            f"(val={best_val:.4f})"
        )
    torch.save(raw_net.state_dict(), out_dir / "model.pt")
    print(f"[train] saved model to {out_dir / 'model.pt'}")

    for i, stage in enumerate(raw_net.core.stages):
        plot_stage_graph(
            stage, save_path=str(out_dir / f"stage{i + 1}_graph_trained.png"),
            title=f"{args.problem} — Stage {i + 1} (trained)")

    val_batch = next(iter(val_loader))
    u_val, y_val = val_batch[0][:64].to(device), val_batch[1][:64].to(device)
    ctx = ctx_factory(u_val.size(0), device=device)
    with torch.no_grad():
        out, trajs = net(
            u_val, store_trajectory=True,
            solver=solver, deq_cfg=deq_cfg)
    if isinstance(trajs, list) and trajs:
        plot_trajectories(
            trajs[0], stage_idx=0,
            save_path=str(out_dir / "trajectories.png"),
            title=f"{args.problem} — Stage 1 trajectories (trained)")

    plot_output_fit(
        out, y_val, loss_name=PRESETS[args.problem]["loss"],
        save_path=str(out_dir / "output_fit.png"),
        title=f"{args.problem} — Output fit (trained)")

    plot_network(
        raw_net, save_path=str(out_dir / "pipeline.png"))

    # ----------------------------------------------------------------
    # kirchhoff-noise: post-training MC noise evaluation. Skip during
    # pruning pipeline because the noise eval reuses val_loader; do it
    # AFTER pruning+retrain completes instead so the noise metrics
    # describe the final deployable network.
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Complexity-regularized pruning pipeline (CP-6).
    # Three-phase schedule: pruning auto-triggers at the B→C boundary.
    # ----------------------------------------------------------------
    if needs_prune:

        from config import PRUNE
        from topology import prune_network
        from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO

        # Import SCHEDULE_THREE_PHASE for schedule-specific edge/node thresholds.
        from config import SCHEDULE_THREE_PHASE

        # For phased schedules, use schedule-specific thresholds;
        # CLI overrides take precedence.
        if args.prune_edge_threshold is None and args.prune_node_threshold is None:
            if schedule_mode == "four_phase":
                _scfg = SCHEDULE_FOUR_PHASE
                edge_thresh = float(_scfg.get("prune_edge_threshold", 0.05))
                node_thresh = float(_scfg.get("prune_node_threshold", 0.05))
            elif schedule_mode == "three_phase":
                _scfg = SCHEDULE_THREE_PHASE
                edge_thresh = float(_scfg.get("prune_edge_threshold", 0.1))
                node_thresh = float(_scfg.get("prune_node_threshold", 0.05))
            else:
                edge_thresh = float(PRUNE["edge_threshold"])
                node_thresh = float(PRUNE["node_threshold"])
        else:
            edge_thresh = args.prune_edge_threshold if args.prune_edge_threshold is not None else float(PRUNE["edge_threshold"])
            node_thresh = args.prune_node_threshold if args.prune_node_threshold is not None else float(PRUNE["node_threshold"])

        # DEPRECATED (deprecate-node-gates): prune_nodes_by_gate is always
        # False; the CLI flag is kept as a deprecated no-op.
        if args.prune_nodes_by_gate is not None:
            import warnings
            warnings.warn(
                "--prune-nodes-by-gate / --no-prune-nodes-by-gate is deprecated "
                "(deprecate-node-gates); node pruning is connectivity-only "
                "regardless of this flag.",
                DeprecationWarning,
                stacklevel=2)
        pnbg = False

        pre_edges = sum(s.num_edges() for s in raw_net.core.stages)
        pre_nodes = sum(s.num_nodes for s in raw_net.core.stages)
        print(
            f"[prune] pre-prune: {pre_edges} edges, {pre_nodes} nodes "
            f"(edge_thresh={edge_thresh}, prune_nodes_by_gate={pnbg})"
        )

        # retrain-oom-fix/REQ-4: log memory state right before pruning.
        _log_gpu_mem("pre_prune")

        pruned_core, stage_remaps = prune_network(
            raw_net.core,
            edge_threshold=edge_thresh,
            node_threshold=node_thresh,
            transfer_params=not args.fresh_init,
            write_idx=list(raw_net.write_idx) if raw_net.write_idx is not None else None,
            read_idx=list(raw_net.read_idx) if raw_net.read_idx is not None else None,
            prune_nodes_by_gate=pnbg)

        post_edges = sum(s.num_edges() for s in pruned_core.stages)
        post_nodes = sum(s.num_nodes for s in pruned_core.stages)
        print(
            f"[prune] post-prune: {post_edges} edges, {post_nodes} nodes "
            f"(removed {pre_edges - post_edges} edges, "
            f"{pre_nodes - post_nodes} nodes)"
        )

        # Rebuild InputMapper/OutputMapper with weights transferred from
        # the pre-prune network.  We treat all surviving nodes as hidden
        # (proj=0) because the hidden/projection split is unknown after
        # remapping.  ``stage_remaps[0]`` maps first-stage compact ids,
        # ``stage_remaps[-1]`` maps last-stage compact ids.
        preset_cfg = PRESETS[args.problem]
        pruned_first_n = pruned_core.stages[0].num_nodes
        pruned_last_n = pruned_core.stages[-1].num_nodes
        in_dim = preset_cfg["stages"][0]["num_inputs"]
        out_dim = preset_cfg.get("out_dim", 1)
        stage0_remap = stage_remaps[0]
        last_remap = stage_remaps[-1]

        raw_write_idx = list(raw_net.write_idx) if raw_net.write_idx is not None else None
        raw_read_idx = list(raw_net.read_idx) if raw_net.read_idx is not None else None
        if effective_read_mode == "sparse" and raw_read_idx is None:
            raw_read_idx = list(read_idx_arg) if read_idx_arg is not None else [0]

        if not args.fresh_init:
            input_mapper_pruned, pruned_write_idx = _transfer_input_mapper(
                raw_net.input_mapper, raw_write_idx, stage0_remap,
                pruned_first_n, in_dim)
            output_mapper_pruned, pruned_read_idx = _transfer_output_mapper(
                raw_net.output_mapper, raw_read_idx, last_remap,
                pruned_last_n, out_dim)
        else:
            if effective_write_mode == "one_to_one":
                if raw_write_idx is None:
                    pruned_write_idx = list(range(min(in_dim, pruned_first_n)))
                else:
                    pruned_write_idx = _remap_indices(raw_write_idx, stage0_remap)
                input_mapper_pruned = SparseInputMapper(
                    in_dim=in_dim, out_dim=pruned_first_n, write_idx=pruned_write_idx)
            elif effective_write_mode == "sparse_proj":
                if raw_write_idx is None:
                    raise ValueError(
                        "write_mode='sparse_proj' requires write_idx; "
                        "cannot synthesize for fresh-init prune retrain"
                    )
                pruned_write_idx = _remap_indices(raw_write_idx, stage0_remap)
                input_mapper_pruned = ProjectedSparseInputMapper(
                    in_dim=in_dim, out_dim=pruned_first_n, write_idx=pruned_write_idx)
            elif effective_write_mode == "fan_out":
                fan_out_map = preset_cfg.get("write_fan_out")
                if fan_out_map is None:
                    raise ValueError(
                        "write_mode='fan_out' requires 'write_fan_out' in config"
                    )
                new_fan_out = {
                    inp: _remap_indices(targets, stage0_remap)
                    for inp, targets in fan_out_map.items()
                }
                input_mapper_pruned = FanOutInputMapper(
                    in_dim=in_dim, out_dim=pruned_first_n, fan_out_map=new_fan_out)
                pruned_write_idx = None
            else:
                MapperCls = (RobustInputMapper
                             if preset_cfg.get("use_robust_input", False)
                             else InputMapper)
                input_mapper_pruned = MapperCls(in_dim=in_dim, out_dim=pruned_first_n)
                pruned_write_idx = None

            if effective_read_mode == "sparse":
                if raw_read_idx is None:
                    pruned_read_idx = [0]
                else:
                    pruned_read_idx = _remap_indices(raw_read_idx, last_remap)
                output_mapper_pruned = OutputMapper(
                    node_dim=pruned_last_n, out_dim=out_dim, read_idx=pruned_read_idx)
            else:
                output_mapper_pruned = OutputMapper(node_dim=pruned_last_n, out_dim=out_dim)
                pruned_read_idx = None

        # retrain-oom-fix/REQ-1: free pre-prune model memory before
        # creating pruned_net. The pruned_net duplicates every
        # parameter tensor via fresh DifferentialStage objects, and
        # even when 0 edges are pruned the new model coexists with
        # raw_net on the GPU. Move raw_net to CPU and drop the
        # original optimizer / scheduler / scaler / teacher; flush
        # the allocator. Keep input_mapper_pruned /
        # output_mapper_pruned / pruned_core alive (they were
        # derived from raw_net and needed for the build below).
        # NOTE: direct = None assignment is required here; exec("del ...")
        # cannot modify enclosing function locals in CPython 3.
        net = None
        optimizer = None
        scheduler = None
        scaler = None
        teacher = None
        teacher_optim = None
        if "raw_net" in locals() and raw_net is not None:
            try:
                raw_net.to("cpu")
            except Exception:
                pass
            # Keep raw_net alive on CPU so the post-training noise evaluation
            # can still use it. GPU memory is freed by .to("cpu") above;
            # setting raw_net = None would break the noise eval call below
            # that passes raw_net to _run_noise_evaluation().
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_gpu_mem("pre_pruned_net_build")

        skip_linear_enabled = bool(getattr(raw_net, "skip_linear_enabled", False))
        skip_in_dim = in_dim
        skip_out_dim = out_dim
        pruned_net = KirchhoffNetWithIO(
            input_mapper_pruned,
            pruned_core,
            output_mapper_pruned,
            hid_count=pruned_first_n,
            proj_count=0,
            final_hid_count=pruned_last_n,
            final_proj_count=0,
            write_idx=pruned_write_idx if effective_write_mode in ("one_to_one", "sparse_proj") else None,
            read_idx=pruned_read_idx if effective_read_mode == "sparse" else None,
            enable_skip_linear=skip_linear_enabled,
            skip_linear_in_dim=skip_in_dim if skip_linear_enabled else None,
            skip_linear_out_dim=skip_out_dim if skip_linear_enabled else None)
        if skip_linear_enabled and getattr(raw_net, "skip_linear", None) is not None:
            with torch.no_grad():
                pruned_net.skip_linear.weight.data.copy_(raw_net.skip_linear.weight.data)
                if pruned_net.skip_linear.bias is not None and raw_net.skip_linear.bias is not None:
                    pruned_net.skip_linear.bias.data.copy_(raw_net.skip_linear.bias.data)
            print(
                f"[prune] copied skip_linear ({skip_in_dim}->{skip_out_dim}) "
                "weights from raw_net to pruned_net"
            )
        pruned_net.to(device)
        # Re-apply --no-resistive-shunt: prune_stage constructs fresh
        # DifferentialStage + FreeTanhLibrary objects (topology.py:1288),
        # which reset _has_resistive=True and g_resistive_raw to a fresh
        # trainable parameter. Without re-applying the disable here, the
        # retrain phase would silently re-enable the resistive shunt.
        if args.no_resistive_shunt:
            from cell_library import FreeTanhLibrary as _FreeTanhLibrary
            for stage in pruned_net.core.stages:
                if isinstance(stage.cell_lib, _FreeTanhLibrary) and stage._has_resistive:
                    stage._has_resistive = False
                    stage.cell_lib.g_resistive_raw.data.zero_()
                    stage.cell_lib.g_resistive_raw.requires_grad_(False)
        n_pruned_params = sum(p.numel() for p in pruned_net.parameters() if p.requires_grad)
        print(f"[prune] retrain trainable params: {n_pruned_params:,}")

        # retrain-oom-fix/REQ-2: the shared cell_lib is the one compiled
        # for raw_net (topology.py:788 sets new_lib=stage.cell_lib). The
        # compiled cell_lib.forward was specialized for raw_net's
        # parameters and tensor shapes; reset it to the uncompiled
        # version so retrain doesn't hit stale cache and so any future
        # recompilation is for pruned_net's actual shapes.
        if compile_enabled and isinstance(device, str) and device.startswith("cuda"):
            try:
                for stage in pruned_net.core.stages:
                    if hasattr(stage, "cell_lib") and not isinstance(
                        stage.cell_lib, SimpleEdgeLibrary
                    ):
                        lib = stage.cell_lib
                        if hasattr(lib.forward, "__wrapped__"):
                            lib.forward = lib.forward.__wrapped__
            except Exception as _e:
                print(f"[prune] warning: failed to reset cell_lib compile: {_e}")

        _log_gpu_mem("post_pruned_net_build")

        # For three_phase, Phase C retrain is always enabled and uses the remainder
        # of the epoch budget. For legacy, respect the --retrain flag.
        retrain_enabled = args.retrain if schedule_mode == "legacy" else True
        best_state_pruned = None  # initialized for noise eval (may be overwritten by retrain)
        if retrain_enabled and pre_edges - post_edges == 0:
            print(
                f"[prune] 0 edges removed — skipping Phase C retrain "
                f"(compact network = original with {post_edges} edges, "
                f"{post_nodes} nodes)"
            )
            retrain_enabled = False
        if retrain_enabled:
            if schedule_mode in ("three_phase", "four_phase"):
                c_epochs = c_total
                c_lr = args.retrain_lr if args.retrain_lr is not None else lr
            else:
                c_epochs = args.retrain_epochs if args.retrain_epochs is not None else max(50, epochs // 2)
                c_lr = args.retrain_lr if args.retrain_lr is not None else lr
            print(
                f"[prune] retraining compact network for {c_epochs} epochs "
                f"(lr={c_lr}, warm_start={not args.fresh_init}, "
                f"scheduler={args.use_scheduler})"
            )
            # retrain-oom-fix/REQ-5: optionally use a smaller batch size
            # for retrain. Build a separate loader (cheap; data is
            # already on CPU) so we don't fight the training loader.
            # Default is 2048 (half the default 4096) to keep retrain
            # memory in budget after Phase A+B. Setting --retrain-batch-size
            # 0 falls back to the training batch_size.
            if args.retrain_batch_size is None or args.retrain_batch_size <= 0:
                retrain_batch_size = batch_size
            else:
                retrain_batch_size = args.retrain_batch_size
            retrain_train_loader = train_loader
            if retrain_batch_size != batch_size:
                from torch.utils.data import DataLoader as _DL
                _ds = train_loader.dataset
                retrain_train_loader = _DL(
                    _ds, batch_size=retrain_batch_size, shuffle=True,
                    num_workers=train_loader.num_workers,
                    pin_memory=train_loader.pin_memory,
                    collate_fn=train_loader.collate_fn,
                    drop_last=False)
                print(
                    f"[prune] retrain batch_size={retrain_batch_size} "
                    f"(overridden from {batch_size})"
                )
            retrain_optimizer = make_optimizer(
                pruned_net, lr=c_lr,
                weight_decay=weight_decay,
                stage_lr_scale=args.retrain_stage_lr_scale,
                mapper_lr_scale=args.retrain_mapper_lr_scale,
                struct_lr_scale=args.struct_lr_scale,
                dyn_lr_scale=args.dyn_lr_scale)
            if max(abs(args.retrain_stage_lr_scale - 1.0), abs(args.retrain_mapper_lr_scale - 1.0),
                   abs(args.struct_lr_scale - 1.0), abs(args.dyn_lr_scale - 1.0)) > 1e-6:
                lr_strs = [f"{g['lr']:.1e}" for g in retrain_optimizer.param_groups]
                print(
                    f"[prune] retrain scales: stage={args.retrain_stage_lr_scale}, "
                    f"mapper={args.retrain_mapper_lr_scale}, "
                    f"struct={args.struct_lr_scale}, dyn={args.dyn_lr_scale}: "
                    f"per-group LRs = {lr_strs}"
                )
            if args.use_scheduler:
                if args.scheduler_type == "cosine":
                    retrain_scheduler = CosineAnnealingLR(
                        retrain_optimizer,
                        T_max=max(1, c_epochs),
                        eta_min=OPTIM["scheduler_eta_min"])
                else:
                    retrain_scheduler = CosineAnnealingWarmRestarts(
                        retrain_optimizer,
                        T_0=OPTIM["scheduler_T_0"],
                        T_mult=OPTIM["scheduler_T_mult"],
                        eta_min=OPTIM["scheduler_eta_min"])
            else:
                retrain_scheduler = None
            retrain_scaler = (
                torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
            )
            retrain_history = []
            retrain_val_history = []
            retrain_orig_history = [] if inverse_stats is not None else None
            retrain_deq_log_path = deq_val_log_path
            best_val_pruned = float("inf")
            best_epoch_pruned = -1
            best_metric_name_c = "val"  # four-phase-redesign/Phase 1b
            ewop = 0
            # retrain-oom-fix/REQ-4: log memory after pruned_net, optimizer,
            # scaler are all set up; this is the snapshot right before the
            # first retrain batch. If the OOM is going to fire, the print
            # below shows how much headroom remained.
            _log_gpu_mem("retrain_epoch_0_start")
            for repoch in range(c_epochs):
                pruned_net.train()
                # Degree budget (degree-budget-topk plan): in Phase C the
                # budget is disabled (frac=0.0) — the compact network is
                # already pruned, so per-edge competition has no work.
                if budget_enabled:
                    for _stage in pruned_net.core.stages:
                        # Disable it for phase C retrain; the compact network is already pruned.
                        _stage.set_budget_frac(0.0, budget_temp_end)
                        _stage.budget_axis = budget_axis
                retrain_deq_weight = 0
                retrain_deq_residual_sum = 0.0
                retrain_deq_residual_max = 0.0
                retrain_deq_nstep_sum = 0.0
                retrain_deq_nstep_max = 0.0
                retrain_deq_abs_state_max = 0.0
                if schedule_mode == "three_phase":
                    global_epoch = b_end + repoch
                    tau_r = three_phase_tau(global_epoch, epochs)
                    effective_c_lambdas = three_phase_lambdas(global_epoch, epochs, lambdas)
                    reg_r = 1.0
                    # Solidification metrics during Phase C.
                    if solid_log_path is not None and repoch % args.validate_every == 0:
                        _log_solidification(solid_log_path, global_epoch, c_metrics)
                elif schedule_mode == "four_phase":
                    global_epoch = b2_end + repoch
                    tau_r = four_phase_tau(global_epoch, epochs)
                    effective_c_lambdas = four_phase_lambdas(global_epoch, epochs, lambdas)
                    reg_r = 1.0
                    if solid_log_path is not None and repoch % args.validate_every == 0:
                        _log_solidification(solid_log_path, global_epoch, c_metrics)
                else:
                    retrain_warmup = (0 if (not args.fresh_init) else max(1, c_epochs // 2))
                    retrain_tau_init = float(TAU.get("final_pretrain", TAU["init"]))
                    retrain_tau_final = float(TAU["final"])
                    tau_r = tau_for_epoch(
                        repoch, total_epochs=c_epochs,
                        tau_init=retrain_tau_init,
                        tau_final=retrain_tau_final)
                    reg_r = reg_schedule(
                        repoch,
                        warmup=retrain_warmup,
                        anneal=max(25, c_epochs // 4))
                    effective_c_lambdas = lambdas
                    tot = 0.0
                    nb = 0
                should_log_retrain_grads = grad_log_path is not None and repoch % args.grad_log_every == 0
                retrain_epoch_grad_norms = None
                retrain_param_snapshots: dict[str, torch.Tensor] | None = None
                if should_log_retrain_grads:
                    retrain_raw = pruned_net.module if isinstance(pruned_net, torch.nn.DataParallel) else pruned_net
                    retrain_param_snapshots = {name: p.data.detach().clone() for name, p in retrain_raw.named_parameters()}
                for batch in retrain_train_loader:
                    ctx = ctx_factory(batch[0].size(0), device=device)
                    retrain_optimizer.zero_grad()
                    u_b, tgt_b = batch
                    u_b = u_b.to(device)
                    tgt_b = tgt_b.to(device)
                    loss_task, loss_structural, _ = compute_loss(
                        pruned_net, u_b, tgt_b, ctx, task_fn,
                        lambdas=effective_c_lambdas_r, return_parts=True,
                        amp=amp_enabled, amp_dtype=amp_dtype, reg_scale=reg_r,
                        device_l2_lambda=args.device_l2_lambda)
                    if retrain_scaler is not None and retrain_scaler._enabled:
                        ( retrain_scaler.scale(loss_task) + retrain_scaler.scale(loss_structural) ).backward()
                        retrain_scaler.unscale_(retrain_optimizer)
                        torch.nn.utils.clip_grad_norm_(pruned_net.parameters(), max_norm=OPTIM["grad_clip_norm"])
                        if should_log_retrain_grads:
                            retrain_epoch_grad_norms = collect_gradient_norms(pruned_net)
                        retrain_scaler.step(retrain_optimizer)
                        retrain_scaler.update()
                    else:
                        (loss_task + loss_structural).backward()
                        torch.nn.utils.clip_grad_norm_(pruned_net.parameters(), max_norm=OPTIM["grad_clip_norm"])
                        if should_log_retrain_grads:
                            retrain_epoch_grad_norms = collect_gradient_norms(pruned_net)
                        retrain_optimizer.step()
                    tot += float((loss_task + loss_structural).item())
                    nb += 1
                    if solver == "deq" and not isinstance(pruned_net, torch.nn.DataParallel):
                        batch_stats = _deq_batch_stats(pruned_net, solver=solver, deq_cfg=deq_cfg)
                        if batch_stats is not None:
                            bs = int(u_b.size(0))
                            retrain_deq_weight += bs
                            retrain_deq_residual_sum += batch_stats["residual_mean"] * bs
                            retrain_deq_residual_max = max(retrain_deq_residual_max, batch_stats["residual_max"])
                            retrain_deq_nstep_sum += batch_stats["nstep_mean"] * bs
                            retrain_deq_nstep_max = max(retrain_deq_nstep_max, batch_stats["nstep_max"])
                            retrain_deq_abs_state_max = max(retrain_deq_abs_state_max, batch_stats["max_abs_state"])
                if retrain_scheduler is not None:
                    retrain_scheduler.step()
                if should_log_retrain_grads:
                    log_gradient_norms(
                        grad_log_path, repoch, pruned_net, retrain=True,
                        optimizer=retrain_optimizer, norms=retrain_epoch_grad_norms)
                    if retrain_param_snapshots is not None and update_norms_path is not None:
                        retrain_update_norms = compute_update_norms(retrain_param_snapshots, pruned_net)
                        log_update_norms(update_norms_path, repoch, retrain_update_norms, phase="C", retrain=True)
                        retrain_param_snapshots = None
                avg = tot / max(1, nb)
                retrain_history.append(avg)
                retrain_deq_metrics = None
                if solver == "deq" and retrain_deq_weight > 0:
                    retrain_deq_metrics = {
                        "residual_mean": retrain_deq_residual_sum / retrain_deq_weight,
                        "residual_max": retrain_deq_residual_max,
                        "nstep_mean": retrain_deq_nstep_sum / retrain_deq_weight,
                        "nstep_max": retrain_deq_nstep_max,
                        "max_abs_state": retrain_deq_abs_state_max}
                    if retrain_deq_log_path is not None:
                        _append_deq_train_row(
                            retrain_deq_log_path,
                            epoch=global_epoch,
                            phase="C",
                            train_loss=avg,
                            metrics=retrain_deq_metrics)
                val_deq_metrics = None
                if repoch % args.validate_every == 0 or repoch == c_epochs - 1:
                    if inverse_stats is not None:
                        if solver == "deq":
                            val_metrics_c = validate_with_inverse(
                                pruned_net, val_loader, task_fn, ctx_factory, device,
                                inverse_stats=inverse_stats,
                                solver=solver, deq_cfg=deq_cfg,
                                collect_deq_metrics=True)
                            val_deq_metrics = val_metrics_c.pop("deq", None)
                        else:
                            val_metrics_c = validate_with_inverse(
                                pruned_net, val_loader, task_fn, ctx_factory, device,
                                inverse_stats=inverse_stats,
                                solver=solver, deq_cfg=deq_cfg)
                        val = val_metrics_c["val"] 
                    else:
                        val_metrics_c = None
                        if solver == "deq":
                            val, val_deq_metrics = validate(
                                pruned_net, val_loader, task_fn, ctx_factory, device,
                                solver=solver, deq_cfg=deq_cfg,
                                collect_deq_metrics=True)
                        else:
                            val = validate(pruned_net, val_loader, task_fn, ctx_factory, device_c)
                    retrain_val_history.append(val)
                    if retrain_orig_history is not None and val_metrics_c is not None:
                        retrain_orig_history.append(val_metrics_c)
                    if retrain_deq_log_path is not None and val_deq_metrics is not None:
                        _append_deq_validation_row(
                            retrain_deq_log_path,
                            epoch=global_epoch,
                            phase="C",
                            split="retrain",
                            train_loss=avg,
                            val_loss=val,
                            metrics=val_deq_metrics)
                    sel_metric_c = float(val)
                    sel_name_c = "val"
                    if sel_metric_c < best_val_pruned - args.min_delta:
                        best_val_pruned = sel_metric_c
                        best_epoch_pruned = repoch
                        best_metric_name_c = sel_name_c
                        ewop = 0
                        best_state_pruned = {k: v.detach().clone() for k, v in pruned_net.state_dict().items()}
                    else:
                        ewop += args.validate_every
                        if args.early_stop and ewop >= args.patience:
                            print(
                                f"[prune] retrain early stop at epoch {repoch} "
                                f"(best {best_metric_name_c}={best_val_pruned:.4f})"
                            )
                            break
                else:
                    retrain_val_history.append(retrain_val_history[-1] if retrain_val_history else avg)
                    if retrain_orig_history is not None:
                        retrain_orig_history.append(
                            retrain_orig_history[-1] if retrain_orig_history
                            else {"val": avg, "mae_orig": float("nan"), "rmse_orig": float("nan")}
                        )
                phase_tag = " [C]" if schedule_mode in ("three_phase", "four_phase") else ""
                print(
                    f"  {'retrain' if schedule_mode == 'legacy' else 'phase-C'} epoch {repoch:4d}{phase_tag}  "
                        f"train={avg:.4e}  "
                    f"val={retrain_val_history[-1]:.4f}  tau={tau_r:.3f}  "
                    f"lr={retrain_optimizer.param_groups[0]['lr']:.2e}"
                    + (_format_deq_summary(retrain_deq_metrics) if retrain_deq_metrics is not None else "")
                    + (_format_deq_summary(val_deq_metrics) if val_deq_metrics is not None else "")
                )

            if best_state_pruned is not None:
                pruned_net.load_state_dict(best_state_pruned)
                pruned_net.to(device)
                print(
                    f"[prune] restored best retrain state "
                    f"(epoch {best_epoch_pruned}, {best_metric_name_c}={best_val_pruned:.4f})"
                )

            if schedule_mode in ("three_phase", "four_phase"):
                phase_c_start = b_end if schedule_mode == "three_phase" else b2_end
                with open(history_path, "r") as f:
                    header = f.readline().strip()
                has_orig = "mae_orig" in header
                with open(history_path, "a") as f:
                    f.write(f"\n# Phase C (prune + retrain): {post_edges}/{pre_edges} edges, {post_nodes}/{pre_nodes} nodes survived\n")
                    if has_orig and retrain_orig_history is not None:
                        for i, (t, v, m) in enumerate(zip(retrain_history, retrain_val_history, retrain_orig_history)):
                            global_ep = phase_c_start + i
                            f.write(f"{global_ep}\t{t}\t{v}\t{m['mae_orig']:.6f}\t{m['rmse_orig']:.6f}\t{m['mse_orig']:.6f}\t{m['mape_orig']:.6f}\tC\n")
                    else:
                        for i, (t, v) in enumerate(zip(retrain_history, retrain_val_history)):
                            global_ep = phase_c_start + i
                            f.write(f"{global_ep}\t{t}\t{v}\tC\n")
            else:
                with open(history_path, "a") as f:
                    f.write(
                        f"\n[prune] pre-prune: {pre_edges} edges, {pre_nodes} nodes "
                        f"(edge_thresh={edge_thresh}, node_thresh={node_thresh})\n"
                    )
                    f.write(
                        f"[prune] post-prune: {post_edges} edges, {post_nodes} nodes "
                        f"(removed {pre_edges - post_edges} edges, "
                        f"{pre_nodes - post_nodes} nodes)\n"
                    )
                    f.write(f"retrain_epoch\ttrain\tval\n")
                    for i, (t, v) in enumerate(zip(retrain_history, retrain_val_history)):
                        f.write(f"{i}\t{t}\t{v}\n")

            # Save compact-network plots.
            for i, stage in enumerate(pruned_core.stages):
                plot_stage_graph(
                    stage, save_path=str(out_dir / f"stage{i + 1}_graph_pruned.png"),
                    title=f"{args.problem} — Stage {i + 1} (pruned, {stage.num_edges()} edges, {stage.num_nodes} nodes)")

            # Pruned output fit.
            with torch.no_grad():
                out_pruned, _ = pruned_net(
                    u_val,
                    ctx=ctx_factory(u_val.size(0), device=device),
                    store_trajectory=False,
                    solver=solver,
                    deq_cfg=deq_cfg)
            plot_output_fit(
                out_pruned, y_val, loss_name=PRESETS[args.problem]["loss"],
                save_path=str(out_dir / "output_fit_pruned.png"),
                title=f"{args.problem} — Output fit (pruned, retrained)")

            # Save pruned model.
            torch.save(pruned_net.state_dict(), out_dir / "model_pruned.pt")
            print(f"[prune] saved pruned model to {out_dir / 'model_pruned.pt'}")
        else:
            # No-retrain mode: pruned model with transferred parameters.
            torch.save(pruned_net.state_dict(), out_dir / "model_pruned.pt")
            print(
                f"[prune] --no-retrain: saved pruned model (with transferred params) "
                f"to {out_dir / 'model_pruned.pt'}"
            )

        # Write a pruning summary file.
        with open(out_dir / "prune_summary.txt", "w") as f:
            f.write(f"edge_threshold: {edge_thresh}\n")
            f.write(f"node_threshold: {node_thresh}\n")
            f.write(f"pre_edges: {pre_edges}\n")
            f.write(f"post_edges: {post_edges}\n")
            f.write(f"pre_nodes: {pre_nodes}\n")
            f.write(f"post_nodes: {post_nodes}\n")
            f.write(f"edges_removed: {pre_edges - post_edges}\n")
            f.write(f"nodes_removed: {pre_nodes - post_nodes}\n")
            f.write(f"retrain: {retrain_enabled}\n")
            f.write(f"fresh_init: {args.fresh_init}\n")
            if retrain_enabled:
                f.write(f"retrain_epochs: {c_epochs}\n")
                f.write(f"retrain_lr: {c_lr}\n")
                f.write(f"best_val_pruned: {best_val_pruned:.6f}\n")
                f.write(f"best_epoch_pruned: {best_epoch_pruned}\n")
                if (retrain_orig_history is not None and len(retrain_orig_history) > 0
                        and 0 <= best_epoch_pruned < len(retrain_orig_history)
                        and retrain_orig_history[best_epoch_pruned] is not None):
                    mp = retrain_orig_history[best_epoch_pruned]
                    f.write(f"best_mse_pruned_orig: {mp['mse_orig']:.6f}\n")
                    f.write(f"best_rmse_pruned_orig: {mp['rmse_orig']:.6f}\n")
                    f.write(f"best_mae_pruned_orig: {mp['mae_orig']:.6f}\n")
                    f.write(f"best_mape_pruned_orig: {mp['mape_orig']:.6f}\n")
                f.write(f"scheduler: {args.use_scheduler}\n")

    # ----------------------------------------------------------------
    # Diagnostic probe: collect D1-D8 right before noise eval so we can
    # isolate the noise-eval clean-loss mismatch vs training val loss.
    # ----------------------------------------------------------------
    if args.noise or args.noise_aware:
        _run_noise_diagnostics(
            raw_net, val_loader, task_fn, ctx_factory,
            device, args, out_dir,
            best_epoch, best_val, best_metric_name,
            compile_enabled, schedule_mode, needs_prune)

    # ----------------------------------------------------------------
    # kirchhoff-noise: final MC noise evaluation on the deployable
    # network. With prune+retrain this is the pruned network; otherwise
    # it's the best-checkpoint pre-prune network. Runs only when
    # --noise or --noise-aware is set.
    # ----------------------------------------------------------------
    if args.noise or args.noise_aware:
        metric_name = PRESETS[args.problem]["loss"]
        if needs_prune:
            # When pruning is on, evaluate both pre-prune (best checkpoint)
            # and post-prune (deployable) networks so the user can compare.
            raw_pruned = pruned_net.module if isinstance(pruned_net, torch.nn.DataParallel) else pruned_net
            if best_state_pruned is not None:
                raw_pruned.load_state_dict(best_state_pruned)
            _run_noise_evaluation(
                raw_net, val_loader, task_fn, ctx_factory,
                device, args, out_dir, "main",
                metric_name=metric_name)
            _run_noise_evaluation(
                raw_pruned, val_loader, task_fn, ctx_factory,
                device, args, out_dir, "pruned",
                metric_name=metric_name)
        else:
            _run_noise_evaluation(
                raw_net, val_loader, task_fn, ctx_factory,
                device, args, out_dir, "main",
                metric_name=metric_name)

    print(f"[train] done — artifacts in {out_dir}")

if __name__ == "__main__":
    main()
