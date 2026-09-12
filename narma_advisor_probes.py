"""Advisor-ordered zero-training probes for the NARMA reservoir plateau.

Implements the four-pillar probe discipline described in the cross-roads
problem statement and ``docs/specs/advisor-probe-suite.md``.  The module
is *side-effect free at import*; only directly-invoked functions build
nets, mutate init, or read checkpoints.

Pillars (called in order, each can be invoked independently):

1. **Baselines calibration** -- delegates to ``narma_experiment.run_baselines``
    via ``run_baselines_calibration``.  The halt/continue decision compares the
    ESN NRMSE to the success band (0.15-0.29 NARMA-10).  Callers should check
    this calibration before interpreting fabric probes.
2. **Frozen-state probe** -- :func:`frozen_state_probe`.  Eval-only fits on
   the existing checkpoint's hidden states: ridge + small MLP on full
    hidden states, ridge on a 5-lag state stack, participation ratio of
    the washed covariance, gate-zero add-on (force ``z_logits`` to a very
    negative value at eval to check if the core edges are doing any work).
3. **Gain-override hook** -- :func:`apply_gain_override`.  Symmetric
   ``gm_raw``/``isat_raw`` overwrite on core + boundary + readout cell
   libraries plus a per-node randomized ``raw_leak`` init, OR a fixed
   ``leak_constant`` (non-programmable mode).  ``cell_library.py`` init
   defaults are NEVER edited; init discipline stays in one place.
4. **E0 sweep** -- :func:`e0_sweep`.  Zero-training 36-point gain x leak x
   drive grid.  Per point reports washout-corrected memory capacity,
   ridge-from-full-states NRMSE, Jacobian eigenvalue instrument, and rail
   fraction; tagged ``PASS``/``FAIL`` against the gates in
   ``advisor-probe-suite``.

Probe outputs land in the leg output dir next to (not inside) training
tables; each row carries config tag plus PASS/FAIL.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Reuse the canonical NARMA experiment pieces (build, run, ridge readouts,
# MC, helpers).  No mutation of ``narma_experiment`` itself.
import narma_experiment as ne  # noqa: E402
from config import INIT as _CONFIG_INIT  # noqa: E402

# Locked grid (advisor-probe-suite, E0 sweep).  With gm in [0.01, 10], gains
# {-5,-2,0,1.5} give approximately {0.077, 1.20, 5.01, 8.18}.  Leak modes
# map to {"slow-fixed", "randomized", 0.15}.  Drive {0.25, 0.5, 1.0} uses
# the existing input_scale path.
E0_GAIN_GRID: tuple[float, ...] = (-5.0, -2.0, 0.0, 1.5)
E0_LEAK_GRID: tuple[Any, ...] = ("slow-fixed", "randomized", 0.15)
E0_DRIVE_GRID: tuple[float, ...] = (0.25, 0.5, 1.0)

# NARMA-10 success band (literature-anchored, not in-harness calibrated
# until the baselines calibration leg runs).
NARMA10_BAND: tuple[float, float] = (0.15, 0.29)
# Pre-registered ESN halt point: at or above this value everything stops for
# a task-setup hunt.  The advisor wording is “far above band (~0.6)”.
ESN_HALT_ABOVE: float = 0.6
# Frozen-state decision thresholds (advisor-probe-suite spec).
RIDGE_PROMOTE_READOUT_BELOW: float = 0.45
RIDGE_INIT_REGIME_AT_OR_ABOVE: float = 0.6
# E0 corner decision thresholds (P2 rerun amendment).
E0_PASS_MC_ABOVE: float = 3.0
E0_PASS_RIDGE_BELOW: float = 0.40
E0_PASS_STATE_PR_MIN: float = 6.0
# Rail fraction disqualifier (spec).
RAIL_DISQUALIFY_ABOVE: float = 0.05
# Train-stream washout for frozen-state / E0 probes (matches the
# canonical --ridge-diagnostic washout=200).
PROBE_WASHOUT: int = 200


@dataclass
class FrozenStateRow:
    """One frozen-state probe row."""

    config_tag: str
    nrmse_ridge_full: float
    nrmse_ridge_lag5: float
    nrmse_mlp_full: float
    r2_ridge_full: float
    participation_ratio: float
    nrmse_gate_zero_ridge_full: float
    mc_total: float
    n_params_ridge: int
    n_params_mlp: int
    # PASS/FAIL on the ridge-from-states decision rule.
    readout_fix_promoted: bool  # ridge_full < RIDGE_PROMOTE_READOUT_BELOW
    init_regime_sole_primary: bool  # ridge_full >= RIDGE_INIT_REGIME_AT_OR_ABOVE


@dataclass
class E0SweepRow:
    """One E0 sweep corner, including its boundary-only pass."""

    config_tag: str
    hidden_dim: int
    n_params: int
    gm_init: float
    leak_mode: str  # "slow-fixed" | "randomized" | "constant:<val>"
    drive: float
    ridge_nrmse: float
    ridge_r2: float
    mc_total_washout_corrected: float
    state_pr: float  # participation ratio of washed full states
    jac_max_abs: float
    jac_min_abs: float
    jac_mean_abs: float
    jac_rank_proxy: float  # participation ratio of |eig|
    rail_frac: float
    sat_max_ratio: float  # max|x| / x_max
    gate_zero_ridge_nrmse: float
    gate_zero_ridge_r2: float
    gate_zero_mc_total: float
    gate_zero_state_pr: float
    gate_zero_jac_max_abs: float
    gate_zero_jac_min_abs: float
    gate_zero_jac_mean_abs: float
    gate_zero_jac_rank_proxy: float
    gate_zero_rail_frac: float
    gate_zero_sat_max_ratio: float
    rail_disqualified: bool
    pass_init_regime: bool
    pass_rail: bool  # NOT rail_disqualified
    pass_all: bool  # pass_init_regime AND pass_rail
    gate_zero_rail_disqualified: bool
    gate_zero_pass_init_regime: bool
    gate_zero_pass_rail: bool
    gate_zero_pass_all: bool


def participation_ratio(states: torch.Tensor) -> float:
    """Participation ratio of the (washed) state covariance eigenvalues.

    PR = (sum lambda)^2 / sum(lambda^2).  For a uniform spectrum of rank
    r, PR ~ r.  For a rank-1 spectrum PR ~ 1.  Used as a rank-collapse
    diagnostic in :func:`frozen_state_probe`.

    Also called with shape ``(1, D)`` for the Jacobian eigenvalue vector;
    in that case the input is treated as the spectrum directly (no
    centering / covariance).  Returns NaN if the spectrum is degenerate
    or non-finite.
    """
    if states.dim() == 1:
        spec = states.clamp_min(0.0)
        s = float(spec.sum().item())
        sq = float(spec.pow(2).sum().item())
        if sq <= 1e-12:
            return float("nan")
        return (s * s) / sq
    if states.dim() != 2 or states.shape[0] < 2:
        return float("nan")
    states_c = states - states.mean(dim=0, keepdim=True)
    cov = (states_c.T @ states_c) / max(states.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    s = float(eig.sum().item())
    sq = float(eig.pow(2).sum().item())
    if sq <= 1e-12:
        return float("nan")
    return (s * s) / sq


def _ridge_fit_predict(
    X: torch.Tensor, y: torch.Tensor, l2: float = 1e-2,
) -> torch.Tensor:
    """Closed-form ridge fit ``W = (X^T X + l2 I)^-1 X^T y`` with bias."""
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    XtX = X_aug.T @ X_aug + l2 * torch.eye(X_aug.shape[1], device=X.device)
    W = torch.linalg.solve(XtX, X_aug.T @ y)
    return W


def _stack_lag_features(states: torch.Tensor, n_lags: int = 5) -> torch.Tensor:
    """Build a ``(T - n_lags + 1, N * n_lags)`` lag-stacked feature matrix.

    Row ``i`` contains the newest state first:
    ``[x[i+n_lags-1], x[i+n_lags-2], ..., x[i]]`` flattened.  Thus row
    ``i`` must be compared with target ``y[i+n_lags-1]``.
    """
    if states.dim() != 2:
        raise ValueError(f"lag stacking requires a (T, N) matrix, got {tuple(states.shape)}")
    if n_lags < 1:
        raise ValueError(f"n_lags must be positive, got {n_lags}")
    T, N = states.shape
    if T < n_lags:
        raise ValueError(f"need at least {n_lags} states for lag stacking, got {T}")
    rows = []
    for lag in range(n_lags):
        rows.append(states[n_lags - 1 - lag: T - lag])
    return torch.cat(rows, dim=1)


def _huber_loss(pred: torch.Tensor, target: torch.Tensor,
                delta: float = 1.0) -> torch.Tensor:
    return nn.functional.huber_loss(pred, target, delta=delta)


def _train_small_mlp(
    X: torch.Tensor, y: torch.Tensor, *,
    hidden_dim: int = 32, epochs: int = 300, lr: float = 1e-3,
    seed: int = 0, device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a tiny 1-hidden-layer MLP as a probe readout.

    Two-pass protocol: 80% train, 20% val with early-stop on val NRMSE.
    Returns the trained model and a metrics dict.  Robust to ``epochs=0``
    and tiny validation sets (falls back to train-only when val < 2).
    """
    torch.manual_seed(seed)
    N = X.shape[1]
    model = nn.Sequential(
        nn.Linear(N, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 80/20 split with a fixed permutation so the run is reproducible.
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(0.8 * n)
    X_tr, y_tr = X[perm[:cut]].to(device), y[perm[:cut]].to(device)
    X_va, y_va = X[perm[cut:]].to(device), y[perm[cut:]].to(device)
    best_val: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    patience = max(20, epochs // 10)
    stagnant = 0
    epochs_run = 0
    for ep in range(epochs):
        epochs_run = ep + 1
        model.train()
        # Full-batch (probe, not training): small data, fast.
        optim.zero_grad()
        pred = model(X_tr).squeeze(-1)
        loss = _huber_loss(pred, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        model.eval()
        if X_va.shape[0] >= 2:
            with torch.no_grad():
                v_pred = model(X_va).squeeze(-1)
            val_n = ne.nrmse(v_pred, y_va)
        else:
            # Tiny-val fallback: use train NRMSE as a proxy so the probe
            # still returns a number instead of crashing on an empty split.
            with torch.no_grad():
                v_pred = model(X_tr).squeeze(-1)
            val_n = ne.nrmse(v_pred, y_tr)
        if best_val is None or val_n + 1e-6 < best_val:
            best_val = val_n
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_nrmse": float(best_val if best_val is not None
                                       else float("nan")), "epochs_run": epochs_run}


def _gate_zero_eval_forward(net: nn.Module, u_seq: torch.Tensor) -> torch.Tensor:
    """Run the eval path with core-edge gates forced to a very negative value.

    Returns ``all_states`` of shape ``(T, B, N)``.  Original gates are
    restored on exit, including on multi-stage nets.
    """
    if not hasattr(net, "core"):
        raise ValueError("net has no .core attribute; expected KirchhoffNetWithIO")
    if not net.core.stages:
        raise ValueError("gate-zero probe requires at least one stage")
    saved: list[tuple[nn.Module, torch.Tensor]] = []
    try:
        for stage in net.core.stages:
            if hasattr(stage, "z_logits"):
                saved.append((stage, stage.z_logits.detach().clone()))
                stage.z_logits.data.fill_(-12.0)
        # Build inputs identical to ``_evaluate_fabric_direct``:
        stage = net.core.stages[0]
        t_span = net.core.stage_times[0]
        num_steps = net.core.stage_steps[0]
        stage_width = net.hid_count + net.proj_count + net.output_ode_count
        x0 = u_seq.new_zeros(1, stage_width)
        with torch.no_grad():
            all_states = stage._forward_heun_sequence(
                x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_seq,
            )
        return all_states
    finally:
        for stage, value in saved:
            stage.z_logits.data.copy_(value)


def memory_capacity_washout(
    states: torch.Tensor, targets: torch.Tensor, *,
    washout: int = PROBE_WASHOUT, max_delay: int = 20, ridge_l2: float = 1e-2,
) -> tuple[list[float], float]:
    """Washout-corrected memory capacity (advisor-probe-suite spec).

    Discards the first ``washout`` rows of ``states`` before fitting the
    per-delay ridge regressions; the legacy ``memory_capacity`` keeps
    them (no-washout caveat for slow leaks).
    """
    if not torch.isfinite(states).all() or not torch.isfinite(targets).all():
        return [float("nan")] * max_delay, float("nan")
    if washout >= states.shape[0]:
        return [float("nan")] * max_delay, float("nan")
    s_w = states[washout:]
    t_w = targets[washout:]
    return ne.memory_capacity(s_w, t_w, max_delay=max_delay, ridge_l2=ridge_l2)


def frozen_state_probe(
    net: nn.Module, u_train: torch.Tensor, y_train: torch.Tensor, *,
    washout: int = PROBE_WASHOUT, mlp_hidden: int = 32, mlp_epochs: int = 300,
    mlp_seed: int = 0, device: str = "cpu",
    also_eval_gate_zero: bool = True, config_tag: str = "",
) -> FrozenStateRow:
    """Probe a trained (or untrained) net for information presence.

    Fits (a) ridge + small MLP on full hidden states, (b) ridge on a
    5-lag state stack, (c) participation ratio of the washed covariance,
    (d) optionally ridge with core-edge gates forced to zero at eval.

    All metrics are eval-only; the net weights are not touched (the
    gate-zero add-on saves/restores ``z_logits``).
    """
    stages = list(net.core.stages)
    if len(stages) != 1:
        raise NotImplementedError(
            "frozen_state_probe currently supports single-stage nets only"
        )
    stage = stages[0]
    hid_count = int(net.hid_count)
    proj_count = int(net.proj_count)
    state_width = hid_count + proj_count + int(net.output_ode_count)
    x0 = u_train.new_zeros(1, state_width)
    t_span = float(net.core.stage_times[0])
    num_steps = int(net.core.stage_steps[0])

    # Full states on the training stream (single stream, batch B=1).
    # Wrap in ``torch.no_grad`` so the Heun graph does not attach to
    # ``hidden`` and the small-MLP backward pass can run independently.
    with torch.no_grad():
        all_states = stage._forward_heun_sequence(
            x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_train,
        )
    states_full = all_states[:, 0, :].detach()  # (T, N)
    hidden = states_full[:, :hid_count].detach()  # (T, hid_count)
    if u_train.dim() != 1 or y_train.dim() != 1:
        raise ValueError(
            "frozen_state_probe requires one-dimensional input/target streams, "
            f"got {tuple(u_train.shape)} and {tuple(y_train.shape)}"
        )
    if hidden.shape[0] != y_train.shape[0]:
        raise ValueError(
            "input and target streams must have the same length, got "
            f"{hidden.shape[0]} and {y_train.shape[0]}"
        )
    if hidden.shape[0] <= washout:
        raise ValueError(
            f"need more than {washout} washed-out samples, got {hidden.shape[0]}"
        )
    y = y_train.to(device)
    hidden_w = hidden[washout:]
    y_w = y[washout:]

    # (a) Ridge from full hidden states.
    W = _ridge_fit_predict(hidden_w, y_w, l2=1e-2)
    y_pred = torch.cat([hidden_w, torch.ones(hidden_w.shape[0], 1, device=hidden_w.device)], dim=1) @ W
    nrmse_ridge = ne.nrmse(y_pred, y_w)
    r2_ridge = ne.r2(y_pred, y_w)
    n_params_ridge = hidden_w.shape[1] + 1

    # (b) MLP from full hidden states (small, ~32 hidden units).
    mlp, mlp_metrics = _train_small_mlp(
        hidden_w, y_w, hidden_dim=mlp_hidden, epochs=mlp_epochs,
        lr=1e-3, seed=mlp_seed, device=device,
    )
    with torch.no_grad():
        mlp_pred = mlp(hidden_w).squeeze(-1)
    nrmse_mlp = ne.nrmse(mlp_pred, y_w)
    n_params_mlp = sum(p.numel() for p in mlp.parameters() if p.requires_grad)

    # (c) Ridge from 5-lag state stack.  Stacked row ``i`` ends at state
    # ``i+n_lags-1``, so the first washed row (whose newest state is
    # ``hidden[washout]``) is at stacked index ``washout-(n_lags-1)`` and
    # must be compared with target ``y[washout]``.
    n_lags = 5
    if washout < n_lags - 1:
        raise ValueError(
            f"washout must be at least {n_lags - 1} for a {n_lags}-lag stack, "
            f"got {washout}"
        )
    stacked = _stack_lag_features(hidden, n_lags=n_lags)
    stacked_w = stacked[washout - (n_lags - 1):]
    y_aligned = y[washout: washout + stacked_w.shape[0]]
    if stacked_w.shape[0] != y_aligned.shape[0]:
        raise ValueError(
            "lagged states and aligned targets have different lengths: "
            f"{stacked_w.shape[0]} != {y_aligned.shape[0]}"
        )
    W_lag = _ridge_fit_predict(stacked_w, y_aligned, l2=1e-2)
    pred_lag = torch.cat(
        [stacked_w, torch.ones(stacked_w.shape[0], 1, device=stacked_w.device)], dim=1,
    ) @ W_lag
    nrmse_ridge_lag = ne.nrmse(pred_lag, y_aligned)

    # (d) Participation ratio on washed states (full state, all dims).
    pr = participation_ratio(states_full[washout:])

    # (e) Memory capacity on the full states (washout-corrected).
    _, mc_total = memory_capacity_washout(
        states_full, u_train, washout=washout, max_delay=20, ridge_l2=1e-2,
    )

    # (f) Gate-zero add-on: ridge from hidden states with core-edge gates off.
    nrmse_gate_zero = float("nan")
    if also_eval_gate_zero and hasattr(stage, "z_logits"):
        gate_states = _gate_zero_eval_forward(net, u_train)
        hidden_gz = gate_states[:, 0, :hid_count]
        hidden_gz_w = hidden_gz[washout:]
        W_gz = _ridge_fit_predict(hidden_gz_w, y_w, l2=1e-2)
        pred_gz = torch.cat(
            [hidden_gz_w, torch.ones(hidden_gz_w.shape[0], 1, device=hidden_gz_w.device)],
            dim=1,
        ) @ W_gz
        nrmse_gate_zero = ne.nrmse(pred_gz, y_w)

    return FrozenStateRow(
        config_tag=config_tag,
        nrmse_ridge_full=nrmse_ridge,
        nrmse_ridge_lag5=nrmse_ridge_lag,
        nrmse_mlp_full=nrmse_mlp,
        r2_ridge_full=r2_ridge,
        participation_ratio=pr,
        nrmse_gate_zero_ridge_full=nrmse_gate_zero,
        mc_total=mc_total,
        n_params_ridge=n_params_ridge,
        n_params_mlp=n_params_mlp,
        readout_fix_promoted=nrmse_ridge < RIDGE_PROMOTE_READOUT_BELOW,
        init_regime_sole_primary=nrmse_ridge >= RIDGE_INIT_REGIME_AT_OR_ABOVE,
    )


def apply_gain_override(
    net: nn.Module, *,
    gm_init: float, isat_init: float | None = None,
    leak_mode: str = "slow-fixed", raw_leak_init_seed: int = 0,
    raw_leak_init_mean: float = -3.0, raw_leak_init_std: float = 1.0,
) -> dict[str, Any]:
    """Override the gain / leak init of a built fabric net (post-build).

    Args:
        net: A built ``KirchhoffNetWithIO`` (output of
            ``_build_fabric_net`` or ``build_net_from_config``).
        gm_init: Value to fill ``gm_raw`` with on every cell library.
        isat_init: Value to fill ``isat_raw`` with. ``None`` (default)
            mirrors ``gm_init`` (symmetric override).
        leak_mode: One of ``"slow-fixed"`` (non-programmable leak
            ``0.0486``), ``"randomized"`` (programmable leak filled
            with mean=``raw_leak_init_mean``, std=``raw_leak_init_std``
            using ``raw_leak_init_seed``), or a numeric value used as a
            fixed scalar leak.
        raw_leak_init_seed / raw_leak_init_mean / raw_leak_init_std:
            Controls the randomized ``raw_leak`` init.

    Returns:
        A small dict with the overrides applied (for the probe row log).
        The ``cell_library.py`` defaults are NEVER modified.
    """
    if isat_init is None:
        isat_init = gm_init

    if not hasattr(net, "core"):
        raise ValueError("apply_gain_override requires a KirchhoffNetWithIO net")

    log: dict[str, Any] = {
        "gm_init": float(gm_init),
        "isat_init": float(isat_init),
        "leak_mode": leak_mode,
    }

    for stage_idx, stage in enumerate(net.core.stages):
        for lib_name in ("cell_lib", "boundary_cell_lib", "output_ode_cell_lib"):
            lib = getattr(stage, lib_name, None)
            if lib is None:
                continue
            if hasattr(lib, "gm_raw"):
                with torch.no_grad():
                    lib.gm_raw.data.fill_(float(gm_init))
            if hasattr(lib, "isat_raw"):
                with torch.no_grad():
                    lib.isat_raw.data.fill_(float(isat_init))

        # Leak dispatch.
        if leak_mode == "randomized":
            if not hasattr(stage, "raw_leak"):
                raise ValueError(
                    f"leak_mode='randomized' requires programmable leak "
                    f"on stage (no raw_leak found)."
                )
            gen = torch.Generator(device="cpu").manual_seed(int(raw_leak_init_seed))
            with torch.no_grad():
                stage.raw_leak.data.copy_(
                    torch.randn(stage.raw_leak.shape, generator=gen)
                    * float(raw_leak_init_std)
                    + float(raw_leak_init_mean)
                )
            stage.leak_mode = "programmable"
        elif leak_mode == "slow-fixed":
            stage.leak_mode = "non-programmable"
            stage.leak_constant = float(_CONFIG_INIT["leak_constant"])
            # ``raw_leak`` (if it exists) is now a no-op parameter that
            # the optimizer still updates; we leave it at the constructor
            # default so existing checkpoints load byte-identically.
            # The forward path picks up ``leak_constant`` instead.
        elif isinstance(leak_mode, str) and leak_mode.startswith("hetero:"):
            # Step 4 hook: hetero-leak init via log-uniform tau.
            # Format: ``hetero:<tau_lo>:<tau_hi>`` (e.g.
            # ``hetero:1.0:40.0``). Colons, NOT commas: the e0 CLI
            # splits --leak-grid on commas, so a comma inside the token
            # would fracture it into bogus modes (Alliance 2026-09-12:
            # "unknown leak_mode 'tau_hi=40.0'"). Colons never appear in
            # floats, so this form survives grid splitting intact.
            from narma_revised_plan import hetero_leak_init as _hetero
            _rest = leak_mode[len("hetero:"):]
            try:
                _lo_s, _hi_s = _rest.split(":")
                tau_lo, tau_hi = float(_lo_s), float(_hi_s)
            except ValueError:
                raise ValueError(
                    f"hetero leak_mode must be 'hetero:<tau_lo>:<tau_hi>', "
                    f"got {leak_mode!r}"
                )
            hl_log = _hetero(
                stage, tau_lo=tau_lo, tau_hi=tau_hi,
                seed=int(raw_leak_init_seed),
            )
            log.update({f"stage{stage_idx}_{k}": v for k, v in hl_log.items()})
        else:
            try:
                leak_val = float(leak_mode)
            except (TypeError, ValueError):
                raise ValueError(
                    f"unknown leak_mode {leak_mode!r}; expected "
                    "'slow-fixed' | 'randomized' | 'hetero:<tau_lo>:<tau_hi>' "
                    "| <numeric scalar>"
                )
            stage.leak_mode = "non-programmable"
            stage.leak_constant = leak_val
        log[f"stage{stage_idx}_leak_mode"] = stage.leak_mode
        if hasattr(stage, "leak_constant"):
            log[f"stage{stage_idx}_leak_constant"] = float(stage.leak_constant)

    return log


def _select_transition_points(
    states_full: torch.Tensor, u_seq: torch.Tensor, *,
    washout: int, n_samples: int,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Select deterministic mid-stream state/input transitions.

    Transition ``t`` uses the actually observed state ``states_full[t]``
    and the following input ``u_seq[t+1]``, so the Jacobian is evaluated
    at a genuine operating point of the collected trajectory.
    """
    if n_samples < 1:
        raise ValueError(f"need at least one Jacobian sample, got {n_samples}")
    if u_seq.dim() != 1 or states_full.dim() != 2:
        raise ValueError(
            "transition selection requires (T,) inputs and (T, N) states, got "
            f"{tuple(u_seq.shape)} and {tuple(states_full.shape)}"
        )
    if states_full.shape[0] != u_seq.shape[0]:
        raise ValueError(
            "states and inputs must have the same length, got "
            f"{states_full.shape[0]} and {u_seq.shape[0]}"
        )
    usable = states_full.shape[0] - washout - 1
    if usable < 1:
        raise ValueError(
            f"need at least one post-washout transition, got T={states_full.shape[0]} "
            f"with washout={washout}"
        )
    count = min(n_samples, usable)
    indices = torch.linspace(
        washout, states_full.shape[0] - 2, steps=count,
    ).round().to(torch.long).unique()
    transitions: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for transition_index in indices.tolist():
        x_from = states_full[int(transition_index)].detach().clone().requires_grad_(True)
        u_next = u_seq[int(transition_index) + 1].reshape(1, 1)
        transitions.append((x_from, u_next, int(transition_index)))
    return transitions


def _one_sample_transition(
    stage: nn.Module, x_from: torch.Tensor, u_next: torch.Tensor,
    dt: float, num_steps: int,
) -> torch.Tensor:
    """Evaluate one observed sample transition, exactly as the sequence path.

    This reproduces :meth:`DifferentialStage._forward_heun_sequence` for one
    sample, including VCA gating and frozen-current semantics, so its
    Jacobian is the Jacobian of the actually used discrete map.
    """
    gate_core_cached = None
    if stage._vca_core_enabled and u_next is not None and stage.vca_v_core is not None:
        gate_core_cached = stage._compute_core_gate(u_next)
    i_edge_const = None
    if stage.freeze_read or stage.core_refresh_interval > 0:
        i_edge_const = stage._compute_i_edge_const(x_from, gate_core_cached)
    i_boundary_const = (
        stage._compute_frozen_boundary(u_next, x_from)
        if stage.freeze_boundary else None
    )
    i_readout_const = (
        stage._compute_frozen_readout(u_next, x_from)
        if stage.freeze_temporal_read else None
    )
    x_next = stage._heun_steps(
        x_from, u_next, dt, num_steps,
        i_edge_const, i_boundary_const, i_readout_const,
        gate_core_cached,
    )
    return x_next.view(-1)


def jacobian_eigs(
    stage: nn.Module,
    transitions: list[tuple[torch.Tensor, torch.Tensor, int]], *,
    dt: float = 0.125, num_steps: int = 8,
) -> list[dict[str, float]]:
    """Jacobian eigenvalue instrument on observed single-sample transitions.

    Each transition is ``(state[t], input[t+1], t)``.  The closure
    differentiates the same one-sample Heun map that generated
    ``state[t+1]`` and reports its eigenvalues.

    Args:
        stage: A ``DifferentialStage``.
        transitions: One or more observed transitions.
        dt: Heun timestep (``t_span / num_steps``).
        num_steps: Heun steps per sample window.

    Returns:
        List of dicts (one per transition) with keys ``transition_index``,
        ``state_dim``, ``max_abs``, ``min_abs``, ``mean_abs``, and
        ``rank_proxy`` (PR over |eig|).
    """
    if not transitions:
        raise ValueError("jacobian_eigs requires at least one transition")
    stage.eval()
    rows: list[dict[str, float]] = []
    for x_from, u_next, transition_index in transitions:
        if x_from.dim() != 1:
            raise ValueError(
                f"transition states must be one-dimensional, got {tuple(x_from.shape)}"
            )

        def transition_map(x_flat: torch.Tensor) -> torch.Tensor:
            x = x_flat.view(1, -1)
            return _one_sample_transition(stage, x, u_next, dt, num_steps)

        J = torch.autograd.functional.jacobian(
            transition_map, x_from, create_graph=False,
        )
        eig = torch.linalg.eigvals(J)
        abs_eig = eig.abs()
        rows.append({
            "transition_index": float(transition_index),
            "state_dim": float(x_from.numel()),
            "max_abs": float(abs_eig.max().item()),
            "min_abs": float(abs_eig.min().item()),
            "mean_abs": float(abs_eig.mean().item()),
            "rank_proxy": float(participation_ratio(abs_eig)),
        })
    return rows


def _score_state_trajectory(
    *, stage: nn.Module, states_full: torch.Tensor,
    u_seq: torch.Tensor, y_seq: torch.Tensor,
    washout: int, jacobian_samples: int,
    t_span: float, num_steps: int,
) -> dict[str, float]:
    """Score one collected trajectory with the complete E0 instrument set."""
    if states_full.dim() != 2 or u_seq.dim() != 1 or y_seq.dim() != 1:
        raise ValueError(
            "trajectory scoring requires (T, N) states and (T,) input/targets, got "
            f"{tuple(states_full.shape)}, {tuple(u_seq.shape)}, {tuple(y_seq.shape)}"
        )
    if not (
        states_full.shape[0] == u_seq.shape[0] == y_seq.shape[0]
        and states_full.shape[0] > washout
    ):
        raise ValueError(
            "states, inputs, and targets must have the same length above washout, got "
            f"{states_full.shape[0]}, {u_seq.shape[0]}, {y_seq.shape[0]} "
            f"with washout={washout}"
        )
    x_max = float(stage.x_max)
    with torch.no_grad():
        sat_max = float(states_full.abs().max().item())
        rail_frac = float((states_full.abs() > 0.9 * x_max).float().mean().item())
    state_pr = participation_ratio(states_full[washout:])
    dt = t_span / num_steps
    transitions = _select_transition_points(
        states_full, u_seq, washout=washout, n_samples=jacobian_samples,
    )
    j_rows = jacobian_eigs(stage, transitions, dt=dt, num_steps=num_steps)
    X = states_full[washout:]
    y = y_seq[washout:]
    W = _ridge_fit_predict(X, y, l2=1e-2)
    pred = torch.cat(
        [X, torch.ones(X.shape[0], 1, device=X.device)], dim=1,
    ) @ W
    _, mc = memory_capacity_washout(
        states_full, u_seq, washout=washout, max_delay=20,
    )
    return {
        "ridge_nrmse": ne.nrmse(pred, y),
        "ridge_r2": ne.r2(pred, y),
        "mc_total": float(mc),
        "state_pr": float(state_pr),
        "jac_max_abs": float(max(r["max_abs"] for r in j_rows)),
        "jac_min_abs": float(min(r["min_abs"] for r in j_rows)),
        "jac_mean_abs": float(
            sum(r["mean_abs"] for r in j_rows) / len(j_rows)
        ),
        "jac_rank_proxy": float(max(r["rank_proxy"] for r in j_rows)),
        "rail_frac": float(rail_frac),
        "sat_max_ratio": float(sat_max / x_max) if x_max > 0 else float("nan"),
    }


def _select_corner_subset(
    drive_grid: Iterable[float], leak_grid: Iterable[Any],
    gain_grid: Iterable[float], max_corners: int | None,
) -> list[tuple[float, Any, float]]:
    """Select the sweep corners without changing their traversal order.

    The full sweep traverses drive, then leak, then gain.  A positive
    ``max_corners`` returns exactly that prefix; otherwise the full grid
    is returned.
    """
    full = [
        (float(drive), leak_mode, float(gm_init))
        for drive in drive_grid
        for leak_mode in leak_grid
        for gm_init in gain_grid
    ]
    if not full:
        raise ValueError("gain, leak, and drive grids must all be non-empty")
    if max_corners is None or max_corners <= 0 or max_corners >= len(full):
        return full
    return full[:max_corners]


def _e0_row(
    net: nn.Module, u_train: torch.Tensor, y_train: torch.Tensor, *,
    order: int, seed: int, device: str,
    gm_init: float, leak_mode: Any, drive: float,
    raw_leak_init_seed: int = 0, washout: int = PROBE_WASHOUT,
    jacobian_samples: int = 3,
) -> E0SweepRow:
    """Score one E0 sweep corner from normal and boundary-only passes."""
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "E0 corners currently support single-stage nets only"
        )
    if u_train.dim() != 1 or y_train.dim() != 1:
        raise ValueError(
            "E0 corners require one-dimensional input/target streams, got "
            f"{tuple(u_train.shape)} and {tuple(y_train.shape)}"
        )
    if u_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "E0 input and target streams must have the same length, got "
            f"{u_train.shape[0]} and {y_train.shape[0]}"
        )
    if u_train.shape[0] <= washout:
        raise ValueError(
            f"need more than {washout} washed-out samples, got {u_train.shape[0]}"
        )
    net.eval()
    # Gain + leak override (cell_library defaults untouched).  Drive is
    # already represented by the scaled input sequence.
    apply_gain_override(
        net, gm_init=gm_init, isat_init=gm_init, leak_mode=leak_mode,
        raw_leak_init_seed=raw_leak_init_seed,
    )
    stage = net.core.stages[0]
    state_width = net.hid_count + net.proj_count + net.output_ode_count
    x0 = u_train.new_zeros(1, state_width)
    t_span = float(net.core.stage_times[0])
    num_steps = int(net.core.stage_steps[0])
    with torch.no_grad():
        all_states = stage._forward_heun_sequence(
            x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_train,
        )
    states_full = all_states[:, 0, :].detach()
    normal = _score_state_trajectory(
        stage=stage, states_full=states_full,
        u_seq=u_train, y_seq=y_train,
        washout=washout, jacobian_samples=jacobian_samples,
        t_span=t_span, num_steps=num_steps,
    )

    # Boundary-only pass: suppress core-edge gates transiently, then score
    # the same instruments on the resulting trajectory.
    gate_states_full = _gate_zero_eval_forward(net, u_train)[:, 0, :].detach()
    boundary_only = _score_state_trajectory(
        stage=stage, states_full=gate_states_full,
        u_seq=u_train, y_seq=y_train,
        washout=washout, jacobian_samples=jacobian_samples,
        t_span=t_span, num_steps=num_steps,
    )

    rail_disqualified = normal["rail_frac"] > RAIL_DISQUALIFY_ABOVE
    pass_init_regime = (
        normal["mc_total"] > E0_PASS_MC_ABOVE
        and normal["ridge_nrmse"] <= E0_PASS_RIDGE_BELOW
        and normal["state_pr"] >= E0_PASS_STATE_PR_MIN
    )
    gate_zero_rail_disqualified = (
        boundary_only["rail_frac"] > RAIL_DISQUALIFY_ABOVE
    )
    gate_zero_pass_init_regime = (
        boundary_only["mc_total"] > E0_PASS_MC_ABOVE
        and boundary_only["ridge_nrmse"] <= E0_PASS_RIDGE_BELOW
        and boundary_only["state_pr"] >= E0_PASS_STATE_PR_MIN
    )
    hidden_dim = int(net.hid_count)
    n_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    config_tag = (
        f"order{order}_seed{seed}_{device}_tanhfree"
        f"_h{hidden_dim}_k{int(stage.core_refresh_interval)}"
        f"_tspan{t_span:g}_steps{num_steps}"
        f"_gm{float(gm_init):g}_leak{leak_mode}_drive{float(drive):g}"
        f"_washout{washout}_jac{jacobian_samples}"
    )
    return E0SweepRow(
        config_tag=config_tag,
        hidden_dim=hidden_dim,
        n_params=n_params,
        gm_init=float(gm_init),
        leak_mode=str(leak_mode),
        drive=float(drive),
        ridge_nrmse=normal["ridge_nrmse"],
        ridge_r2=normal["ridge_r2"],
        mc_total_washout_corrected=normal["mc_total"],
        state_pr=normal["state_pr"],
        jac_max_abs=normal["jac_max_abs"],
        jac_min_abs=normal["jac_min_abs"],
        jac_mean_abs=normal["jac_mean_abs"],
        jac_rank_proxy=normal["jac_rank_proxy"],
        rail_frac=normal["rail_frac"],
        sat_max_ratio=normal["sat_max_ratio"],
        gate_zero_ridge_nrmse=boundary_only["ridge_nrmse"],
        gate_zero_ridge_r2=boundary_only["ridge_r2"],
        gate_zero_mc_total=boundary_only["mc_total"],
        gate_zero_state_pr=boundary_only["state_pr"],
        gate_zero_jac_max_abs=boundary_only["jac_max_abs"],
        gate_zero_jac_min_abs=boundary_only["jac_min_abs"],
        gate_zero_jac_mean_abs=boundary_only["jac_mean_abs"],
        gate_zero_jac_rank_proxy=boundary_only["jac_rank_proxy"],
        gate_zero_rail_frac=boundary_only["rail_frac"],
        gate_zero_sat_max_ratio=boundary_only["sat_max_ratio"],
        rail_disqualified=rail_disqualified,
        pass_init_regime=pass_init_regime,
        pass_rail=not rail_disqualified,
        pass_all=pass_init_regime and not rail_disqualified,
        gate_zero_rail_disqualified=gate_zero_rail_disqualified,
        gate_zero_pass_init_regime=gate_zero_pass_init_regime,
        gate_zero_pass_rail=not gate_zero_rail_disqualified,
        gate_zero_pass_all=(
            gate_zero_pass_init_regime and not gate_zero_rail_disqualified
        ),
    )


def e0_sweep(
    order: int, seed: int, *, device: str = "cpu",
    gain_grid: Iterable[float] = E0_GAIN_GRID,
    leak_grid: Iterable[Any] = E0_LEAK_GRID,
    drive_grid: Iterable[float] = E0_DRIVE_GRID,
    n_streams: int = 4, train_samples_per_stream: int = 2500,
    washout: int = PROBE_WASHOUT, jacobian_samples: int = 3,
    t_span: float = 1.0, num_steps: int = 8, hidden_dim: int = 25,
    net_factory: Any | None = None,
    selected_corners: Iterable[tuple[float, Any, float]] | None = None,
    use_small_world: bool = False,
    small_world_k: int = 4,
    small_world_p: float = 0.2,
    small_world_seed: int = 1,
) -> list[E0SweepRow]:
    """Run the locked E0 (zero-training) gain/leak/drive sweep.

    Per-corner: build a fresh net, override gains/leak, scale drive,
    collect full states once on the first train stream, fit ridge, compute
    washout-corrected MC, evaluate Jacobian eigenvalues at observed
    mid-stream transitions, and measure rail fraction.  Returns a list of
    :class:`E0SweepRow`.

    The canonical training pipeline generates ``n_streams`` streams, but an
    E0 probe is eval-only and uses only stream zero, matching the legacy
    MC input.  Stream zero is seed-identical whether the other streams are
    generated or not.

    Args:
        net_factory: Optional callable ``() -> nn.Module``.  If omitted,
            uses the canonical NARMA fabric built via
            ``narma_experiment._build_fabric_net``.
        selected_corners: Optional explicit ``(drive, leak, gm)`` list.
            Used by ``--max-corners`` so smoke runs take an exact prefix of
            the traversal order.
        use_small_world: If True, build the NARMA net with
            ``hidden_family='small_world'`` (Step 4 grid axis). Replaces
            the canonical 5x5 torus with a Watts-Strogatz graph of the
            same node count. Random-sign fan-out is NOT yet wired (plan
            sec. 7 notes this can be added once a corner passes).
        small_world_k / p / seed: Parameters of the Watts-Strogatz
            graph. See ``narma_revised_plan.build_small_world_narma_preset``.
    """
    if n_streams < 1:
        raise ValueError(f"n_streams must be positive, got {n_streams}")
    if train_samples_per_stream <= washout:
        raise ValueError(
            f"need more than {washout} samples, got {train_samples_per_stream}"
        )
    if jacobian_samples < 1:
        raise ValueError(f"jacobian_samples must be positive, got {jacobian_samples}")
    if t_span <= 0:
        raise ValueError(f"t_span must be positive, got {t_span}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if hidden_dim < 1:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if selected_corners is None:
        corners = _select_corner_subset(drive_grid, leak_grid, gain_grid, None)
    else:
        corners = [(float(drive), leak_mode, float(gm_init))
                   for drive, leak_mode, gm_init in selected_corners]
        if not corners:
            raise ValueError("selected_corners must be non-empty")

    # Stream zero is the probe stream; other generated streams preserve the
    # canonical training-stream construction.
    u_train_raw, y_train_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream_raw = u_train_raw[0]
    y_stream_raw = y_train_raw[0]

    # Resolve the default net factory: small_world preset when
    # ``use_small_world`` is True, canonical torus otherwise. Both
    # factories ignore their arguments and always rebuild via the seeded
    # RNG so corner reproducibility holds (corners share the same seed).
    if net_factory is None:
        if use_small_world:
            from topology import build_net_from_config
            from cell_library import make_cell_library
            from config import make_narma_preset as _make_narma_preset
            from narma_revised_plan import build_small_world_narma_preset

            def _small_world_factory() -> nn.Module:
                preset = build_small_world_narma_preset(
                    order=order, hidden_dim=hidden_dim,
                    num_steps_per_sample=num_steps,
                    t_span=t_span, small_world_k=small_world_k,
                    small_world_p=small_world_p,
                    small_world_seed=small_world_seed,
                    core_refresh_interval=0,
                    leak_constant=None,
                )
                torch.manual_seed(seed)
                cell_lib = make_cell_library("tanh_free")
                return build_net_from_config(
                    cfg=preset, cell_lib=cell_lib,
                    boundary_fan_out=preset["boundary_fan_out"],
                    enable_temporal_readout=True,
                    freeze_read=False,
                )

            net_factory = _small_world_factory
        else:
            def _torus_factory() -> nn.Module:
                net, _t_span, _num_steps = ne._build_fabric_net(
                    order=order, seed=seed, freeze_read=False,
                    t_span=t_span, num_steps=num_steps,
                    cell_library="tanh_free",
                    hidden_dim=hidden_dim,
                    core_refresh_interval=0,
                    leak_constant=None,
                    compile_sequence=False,
                )
                return net

            net_factory = _torus_factory

    rows: list[E0SweepRow] = []
    scaled_cache: dict[float, torch.Tensor] = {}
    for drive, leak_mode, gm_init in corners:
        drive_value = float(drive)
        if drive_value not in scaled_cache:
            scaled_cache[drive_value] = ne._scale_drive(
                u_stream_raw, bipolar=True, order=order,
                input_scale=drive_value,
            )
        u_drive = scaled_cache[drive_value].to(device)
        y_drive = y_stream_raw.to(device)
        # Build a fresh net for each corner so the overridden init is
        # reproducible and no corner inherits another corner's override.
        net = net_factory()
        net.to(device)
        rows.append(_e0_row(
            net, u_drive, y_drive,
            order=order, seed=seed, device=device,
            gm_init=float(gm_init), leak_mode=leak_mode,
            drive=drive_value, raw_leak_init_seed=seed,
            washout=washout, jacobian_samples=jacobian_samples,
        ))
    return rows


# ---------------------------------------------------------------------------
# Baselines calibration
# ---------------------------------------------------------------------------

def _classify_esn(
    esn_nrmse: float, halt_above: float, band: tuple[float, float],
) -> str:
    """Apply the pre-registered ESN calibration decision without running data."""
    if esn_nrmse >= halt_above:
        return "halt-task-setup"
    if band[0] <= esn_nrmse <= band[1]:
        return "fabric-guilty"
    return "band-unknown"


def run_baselines_calibration(
    order: int, seed: int, *, device: str = "cpu",
    halt_above: float = ESN_HALT_ABOVE, band: tuple[float, float] = NARMA10_BAND,
) -> dict[str, Any]:
    """Run ``run_baselines`` and apply the halt-or-continue decision.

    The supplied band is calibrated for NARMA-10; callers must not use this
    helper for another order unless they also supply that order's band.

    Returns:
        Dict with ``baselines`` (raw ``run_baselines`` output),
        ``esn_nrmse`` (the ESN figure), and ``decision`` in
        ``{"halt-task-setup", "fabric-guilty", "band-unknown"}``.
    """
    if order != 10 and band == NARMA10_BAND:
        raise ValueError(
            "the default calibration band is NARMA-10-specific; supply an "
            f"explicit band for order {order}"
        )
    bls = ne.run_baselines(order=order, seed=seed, device=device)
    esn_nrmse = float(bls["esn"]["nrmse"])
    decision = _classify_esn(esn_nrmse, halt_above, band)
    return {
        "order": order,
        "seed": seed,
        "device": device,
        "baselines": bls,
        "esn_nrmse": esn_nrmse,
        "decision": decision,
        "halt_above": halt_above,
        "band": list(band),
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def write_probe_csv(path: Path, rows: list[dict[str, Any]],
                    fieldnames: list[str] | None = None) -> None:
    """Write probe rows to CSV with the given (or inferred) fieldnames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        # Stable order: keys of first row, then any new keys in order seen.
        seen: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_probe_txt(path: Path, header: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--order", type=int, choices=[10, 20], default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, default=Path("./output/advisor_probes"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Advisor-ordered zero-training probes for the NARMA reservoir.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- baselines calibration ---
    p_bl = sub.add_parser("baselines", help="Run in-harness baselines calibration.")
    _add_common_args(p_bl)

    # --- frozen-state probe ---
    p_fr = sub.add_parser(
        "frozen",
        help="Eval-only probe on a trained/native fabric checkpoint.",
    )
    _add_common_args(p_fr)
    p_fr.add_argument("--ckpt", type=Path, required=True,
                      help="Path to a fabric .pt checkpoint.")
    p_fr.add_argument("--cell-library", default="tanh_free")
    p_fr.add_argument("--core-refresh-interval", type=int, default=0)
    p_fr.add_argument("--freeze-read", action="store_true")
    p_fr.add_argument("--leak-constant", type=float, default=None)
    p_fr.add_argument("--input-scale", type=float, default=1.0)
    p_fr.add_argument("--n-streams", type=int, default=4)
    p_fr.add_argument("--train-samples", type=int, default=2500)
    p_fr.add_argument("--mlp-hidden", type=int, default=32)
    p_fr.add_argument("--mlp-epochs", type=int, default=300)
    p_fr.add_argument("--no-gate-zero", action="store_true")
    p_fr.add_argument("--t-span", type=float, default=1.0)
    p_fr.add_argument("--num-steps", type=int, default=8)

    # --- E0 sweep (caller can shrink the grid) ---
    p_e0 = sub.add_parser("e0", help="Run the zero-training 36-point gain/leak/drive sweep.")
    _add_common_args(p_e0)
    p_e0.add_argument("--n-streams", type=int, default=4)
    p_e0.add_argument("--train-samples", type=int, default=2500)
    p_e0.add_argument("--jacobian-samples", type=int, default=3)
    p_e0.add_argument("--t-span", type=float, default=1.0)
    p_e0.add_argument("--num-steps", type=int, default=8)
    p_e0.add_argument("--hidden-dim", type=int, default=25)
    p_e0.add_argument("--gain-grid", type=str, default="-5,-2,0,1.5",
                      help="Comma-separated gm_init values (default -5,-2,0,1.5)")
    p_e0.add_argument(
        "--leak-grid", type=str, default="slow-fixed,randomized,0.15",
        help="Comma-separated leak modes; numeric -> fixed scalar leak. "
             "Step 4 hetero-leak token: 'hetero:1.0:40.0' (per-node "
             "log-uniform time constants; colons, never commas, so the "
             "token survives grid splitting).",
    )
    p_e0.add_argument("--drive-grid", type=str, default="0.25,0.5,1.0")
    p_e0.add_argument("--max-corners", type=int, default=0,
                      help="If >0, run exactly the first N corners in "
                           "drive/leak/gain traversal order (smoke).")
    p_e0.add_argument("--use-small-world", action="store_true",
                      help="Step 4 grid axis: build the NARMA net with "
                           "hidden_family='small_world' instead of the "
                           "canonical torus (same node count).")
    p_e0.add_argument("--small-world-k", type=int, default=4)
    p_e0.add_argument("--small-world-p", type=float, default=0.2)
    p_e0.add_argument("--small-world-seed", type=int, default=1)

    args = parser.parse_args(argv)
    # Pre-registered thresholds and locked grids are NARMA-10-specific.
    # Library functions remain order-agnostic for future calibration work.
    if args.order != 10:
        parser.error("advisor-probe CLI decisions are pre-registered for --order 10 only")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "baselines":
        res = run_baselines_calibration(args.order, args.seed, device=args.device)
        esn = res["esn_nrmse"]
        print(f"  baselines seed={args.seed}: ESN NRMSE={esn:.4f}")
        for name, v in res["baselines"].items():
            print(f"    {name:>12}  NRMSE={v['nrmse']:.4f}  R^2={v['r2']:.4f}  "
                  f"params={v.get('n_params', '')}")
        print(f"  decision: {res['decision']} (band {res['band']}, halt>={res['halt_above']})")
        (args.output / "baselines.json").write_text(json.dumps({
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "esn_nrmse": esn,
            "decision": res["decision"],
            "thresholds": {
                "band": res["band"],
                "halt_above": res["halt_above"],
            },
            "baselines": {
                k: {kk: float(vv) if isinstance(vv, (int, float)) else vv
                    for kk, vv in v.items()}
                for k, v in res["baselines"].items()
            },
        }, indent=2))
        return 0

    if args.mode == "frozen":
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        net, t_span_used, num_steps_used = ne._build_fabric_net(
            order=args.order, seed=args.seed, freeze_read=args.freeze_read,
            t_span=args.t_span, num_steps=args.num_steps,
            cell_library=args.cell_library,
            core_refresh_interval=args.core_refresh_interval,
            leak_constant=args.leak_constant,
            compile_sequence=False,
        )
        net.load_state_dict(ckpt["model_state"], strict=True)
        net.to(args.device)
        u_train_raw, _ = ne._gen_narma_train_streams(
            args.order, args.seed, args.n_streams, args.train_samples,
        )
        u_train = ne._scale_drive(
            u_train_raw[0], bipolar=True, order=args.order,
            input_scale=args.input_scale,
        )
        u_train, y_train = u_train.to(args.device), _[0].to(args.device)
        config_tag = (
            f"ckpt{Path(args.ckpt).name}_order{args.order}_seed{args.seed}_"
            f"{args.cell_library}_k{args.core_refresh_interval}_"
            f"freeze{int(args.freeze_read)}_leak{args.leak_constant}_"
            f"drive{args.input_scale:g}_washout{PROBE_WASHOUT}_"
            f"mlp{args.mlp_hidden}x{args.mlp_epochs}"
        )
        row = frozen_state_probe(
            net, u_train, y_train,
            mlp_hidden=args.mlp_hidden, mlp_epochs=args.mlp_epochs,
            mlp_seed=args.seed, device=args.device,
            also_eval_gate_zero=not args.no_gate_zero,
            config_tag=config_tag,
        )
        row_dict = asdict(row)
        print(f"  FROZEN_STATE probe on {args.ckpt}:")
        for k, v in row_dict.items():
            print(f"    {k}={v}")
        # Decision.
        if row.readout_fix_promoted:
            decision = "READOUT-FIX-LEADS"
        elif row.init_regime_sole_primary:
            decision = "INIT-REGIME-SOLE-PRIMARY"
        else:
            decision = "AMBIGUOUS"
        print(f"  DECISION: {decision} "
              f"(thresholds: promote<{RIDGE_PROMOTE_READOUT_BELOW}, "
              f"init>={RIDGE_INIT_REGIME_AT_OR_ABOVE})")
        out = {
            "ckpt": str(args.ckpt),
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "cell_library": args.cell_library,
            "core_refresh_interval": args.core_refresh_interval,
            "freeze_read": args.freeze_read,
            "leak_constant": args.leak_constant,
            "input_scale": args.input_scale,
            "n_streams": args.n_streams,
            "train_samples": args.train_samples,
            "washout": PROBE_WASHOUT,
            "mlp_hidden": args.mlp_hidden,
            "mlp_epochs": args.mlp_epochs,
            "gate_zero": not args.no_gate_zero,
            "row": row_dict,
            "decision": decision,
            "thresholds": {
                "promote_readout_below": RIDGE_PROMOTE_READOUT_BELOW,
                "init_regime_at_or_above": RIDGE_INIT_REGIME_AT_OR_ABOVE,
            },
        }
        (args.output / "frozen_state.json").write_text(json.dumps(out, indent=2))
        write_probe_csv(args.output / "frozen_state.csv", [row_dict])
        return 0

    if args.mode == "e0":
        gain_grid = tuple(float(s) for s in args.gain_grid.split(",") if s.strip())
        leak_grid: list[Any] = []
        for tok in args.leak_grid.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                leak_grid.append(float(tok))
            except ValueError:
                leak_grid.append(tok)
        drive_grid = tuple(float(s) for s in args.drive_grid.split(",") if s.strip())
        selected_corners = _select_corner_subset(
            drive_grid, leak_grid, gain_grid,
            args.max_corners if args.max_corners > 0 else None,
        )
        t0 = time.time()
        rows = e0_sweep(
            order=args.order, seed=args.seed, device=args.device,
            gain_grid=gain_grid, leak_grid=leak_grid, drive_grid=drive_grid,
            n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            hidden_dim=args.hidden_dim,
            selected_corners=selected_corners,
            use_small_world=args.use_small_world,
            small_world_k=args.small_world_k,
            small_world_p=args.small_world_p,
            small_world_seed=args.small_world_seed,
        )
        elapsed = time.time() - t0
        row_dicts = [asdict(r) for r in rows]
        # CSV
        write_probe_csv(args.output / "e0_sweep.csv", row_dicts)
        # Plain-text summary.
        hdr = (f"E0 sweep -- order={args.order} seed={args.seed} "
               f"{len(rows)} corners in {elapsed:.1f}s")
        lines = [
            f"{'gm':>6} {'leak':>28} {'drive':>5} {'ridge':>7} "
            f"{'mc':>6} {'sPR':>6} {'gzR':>7} {'gzMC':>6} "
            f"{'gzPR':>6} {'rail%':>6} {'pass':>4} {'gzpass':>6}",
        ]
        for r in rows:
            lines.append(
                f"{r.gm_init:>6.2f} {r.leak_mode:>28} {r.drive:>5.2f} "
                f"{r.ridge_nrmse:>7.4f} {r.mc_total_washout_corrected:>6.2f} "
                f"{r.state_pr:>6.2f} {r.gate_zero_ridge_nrmse:>7.4f} "
                f"{r.gate_zero_mc_total:>6.2f} {r.gate_zero_state_pr:>6.2f} "
                f"{100.0 * r.rail_frac:>5.1f}% "
                f"{'PASS' if r.pass_all else 'FAIL':>4} "
                f"{'PASS' if r.gate_zero_pass_all else 'FAIL':>6}"
            )
        write_probe_txt(args.output / "e0_sweep.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        # Pre-registered sweep decision.
        passing = [r for r in rows if r.pass_all]
        rail_pass = [r for r in rows if r.pass_rail]
        print()
        if passing:
            best = min(passing, key=lambda r: r.ridge_nrmse)
            print(
                f"  DECISION: INIT-SEARCH-WARRANTED. {len(passing)} corner(s) "
                f"with MC>{E0_PASS_MC_ABOVE}, ridge<={E0_PASS_RIDGE_BELOW}, "
                f"state-PR>={E0_PASS_STATE_PR_MIN}, and rail pass; "
                f"best ridge={best.ridge_nrmse:.4f} at "
                f"gm={best.gm_init}, leak={best.leak_mode}, drive={best.drive}. "
                f"Next: init search + 1 confirmation train."
            )
        elif rail_pass and not passing:
            print(
                f"  DECISION: NO-CORNER-BREAKS-RIDGE. {len(rail_pass)} corner(s) "
                f"pass rail but none reach MC>{E0_PASS_MC_ABOVE}, "
                f"ridge<={E0_PASS_RIDGE_BELOW}, and "
                f"state-PR>={E0_PASS_STATE_PR_MIN}. Next: structural talk "
                f"(systolic delay-line candidate; depth-per-sample stays paused)."
            )
        else:
            print(
                f"  DECISION: ALL-CORNERS-RAIL-DISQUALIFIED. Init regime "
                f"shape does not break the failure mode; treat as "
                f"structural suppression."
            )
        # Also dump a structured summary.
        (args.output / "e0_sweep.json").write_text(json.dumps({
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "canonical_net": {
                "cell_library": "tanh_free",
                "freeze_read": False,
                "core_refresh_interval": 0,
                "t_span": args.t_span,
                "num_steps": args.num_steps,
                "hidden_dim": args.hidden_dim,
                "use_small_world": args.use_small_world,
                "small_world_k": args.small_world_k,
                "small_world_p": args.small_world_p,
                "small_world_seed": args.small_world_seed,
            },
            "n_streams": args.n_streams,
            "train_samples": args.train_samples,
            "washout": PROBE_WASHOUT,
            "jacobian_samples": args.jacobian_samples,
            "grids": {
                "gain": list(gain_grid),
                "leak": [str(value) for value in leak_grid],
                "drive": list(drive_grid),
            },
            "corner_selection": (
                f"first {args.max_corners}" if args.max_corners > 0 else "full grid"
            ),
            "n_corners": len(rows),
            "elapsed_s": elapsed,
            "thresholds": {
                "mc_above": E0_PASS_MC_ABOVE,
                "ridge_below_or_equal": E0_PASS_RIDGE_BELOW,
                "state_pr_min": E0_PASS_STATE_PR_MIN,
                "rail_disqualify_above": RAIL_DISQUALIFY_ABOVE,
            },
            "rows": row_dicts,
        }, indent=2))
        return 0

    parser.error(f"unknown mode {args.mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
