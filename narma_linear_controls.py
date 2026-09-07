"""Calibrated memory-suppressor controls for the NARMA-10 fabric plateau.

Implements the R0 Ridge-instrument reconciliation matrix, the C0 ESN
calibration control, the C1 linear-reservoir-in-fabric control at
hidden-49 with spectral radius pinned near 0.95, and the C2/C3/C4
single-element bisection restoring boundary-OTA input, tanh core, and
compliance one at a time.  All controls are eval-only.

The module is *side-effect free at import*; only directly-invoked
functions build nets, mutate init, or read checkpoints.

Linear-reservoir monkey-patching (C1):

- The fabric stage's ``cell_lib.forward`` is replaced at runtime to
  return ``G_edge * (x_src - x_dst)`` (per-edge resistive KCL).
- The stage's ``cell_lib.resistive_current`` is replaced to return
  zeros (the linear KCL already provides the resistive path, so the
  parallel shunt must not double-count).
- The stage's ``boundary_cell_lib.forward`` is replaced to return
  ``G_in * u_src`` (linear input injection).
- Core edge gates (``z_logits``) and boundary edge gates
  (``boundary_z_logits``) are set to ``+12`` so the per-edge mask is
  ~1.0 — gate-zero overrides alone would kill the resistive shunt, but
  the cell_lib replacement above makes the path gate-independent.
- ``clip_current`` is zeroed; ``x_max`` is pushed far outside the
  operating range; ``raw_leak`` is randomized per-node under the
  programmable leak mode.
- ``cell_library.py`` init defaults are NEVER edited; the
  ``probe_audit2.py`` suite asserts the original ``gm_raw`` /
  ``isat_raw`` / ``raw_leak`` snapshots are preserved across probes.

Reporting:

- Each control writes a JSON artifact and a CSV artifact under the
  leg output dir, with config tags, thresholds, seeds, spectral
  measurements, per-delay MC curves, Jacobian spectra, saturation
  histograms, and paired deltas.  All gate decisions (matched parity,
  MC, PR, rail) appear in the JSON for audit reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import narma_advisor_probes as npr  # noqa: E402
import narma_experiment as ne  # noqa: E402


# ---------------------------------------------------------------------------
# Locked control thresholds (linear-control-probes spec).
# ---------------------------------------------------------------------------

PROBE_WASHOUT: int = npr.PROBE_WASHOUT
CANONICAL_T_SPAN: float = 1.0
CANONICAL_NUM_STEPS: int = 8
CANONICAL_HIDDEN: int = 49
CANONICAL_DRIVE_SCALE: float = 1.0
CANONICAL_X_MAX_LIN: float = 20.0  # far-rail for the linear reservoir
CANONICAL_CLIP_LIN: float = 0.0  # clip disabled in C1

# R0 reconciliation: E0 Ridge must be at most matched-raw-delay Ridge + 0.03.
R0_MATCHED_PARITY_TOL: float = 0.03

# C1 PASS criteria (linear-control-probes spec, c1b-revision amendment).
# Parity gate DROPPED for linear controls (tapped delay line is the optimal
# finite-window linear basis by construction).  Retained as a diagnostic and
# as a hard pass gate for nonlinear fabrics only (not exercised here).
C1_PASS_MC_ABOVE: float = 3.0
C1_PASS_PR_MIN: float = 6.0
C1_PASS_RIDGE_MATCHED: float = R0_MATCHED_PARITY_TOL  # diagnostic only
C1_SPECTRAL_TARGET_LOW: float = 0.93
C1_SPECTRAL_TARGET_HIGH: float = 0.97
C1_SPECTRAL_TRANSITION_MIN: int = 3
C1_TUNE_MAX_ITERS: int = 16
C1_TUNE_TOL: float = 0.005

# C1b orthogonal-mixing revision (c1b-revision plan, c1b-protocol spec).
# Linear controls use a tightened memory gate (MC >= 4 ~= 0.5x C0 reference
# MC_ref = 8.07) and a uniform slow-leak reservoir with orthogonal W tuned
# to the J-band [0.97, 0.99].  Parity gate is dropped (diagnostic only).
C1B_PASS_MC_ABOVE: float = 5.0  # pre-registered MC 5-15
C1B_PASS_RAIL_BELOW: float = 0.05
C1B_SPECTRAL_TARGET_LOW: float = 0.97
C1B_SPECTRAL_TARGET_HIGH: float = 0.99
C1B_LEAK_VALUE: float = 0.05  # uniform slow leak, non-programmable
C1B_W_DIM: int = 50  # 50x50 incl. output node (matches hidden_dim convention)
C1B_EXPECTED_MC_LOW: float = 5.0
C1B_EXPECTED_MC_HIGH: float = 15.0
C1B_TUNE_MAX_ITERS: int = 24
C1B_TUNE_TOL: float = 0.005

# C1b-real resistive-core variant.  Same MC/rail gates, but installed by
# driving core gm/isat to their minima (shunt dominates) and reusing the
# J-tuner against the real rhs.
#
# Spectral-gate deviation (audit finding, 2026-09-07): the C1b max|J|
# band [0.97, 0.99] is UNREACHABLE for a conductance Laplacian.  The
# Laplacian always has a near-zero mode, so max|J| is pinned at the
# leak mode ~(1-dt*leak)^steps ~= 0.945 regardless of G (verified:
# G/4 and G*4 both give max 0.945x); max|J|(G) is U-shaped with its
# minimum at G->0, and the [0.97, 0.99] band is only crossed on the
# unstable shoulder.  Scaling G DOWN uniformizes the spectrum instead
# (tiny-net measurement: min 0.8335 at default G -> 0.9126 at G/4,
# max pinned 0.945x).  C1b-real therefore gates on UNIFORMITY
# (min|J| >= floor, default 0.90) with max|J| <= cap (default 0.995)
# as the stability guard; max|J| vs the C1b band is reported as a
# diagnostic only.
C1B_REAL_GM_MIN_RATIO: float = 0.0  # gm_raw -> very negative => gm ~ gm_min
C1B_REAL_ISAT_MIN_RATIO: float = 0.0
C1B_REAL_RAIL_BELOW: float = C1B_PASS_RAIL_BELOW
C1B_REAL_UNIFORM_MIN: float = 0.90
C1B_REAL_STABILITY_CAP: float = 0.995
C1B_REAL_MIN_SCALE: float = 1e-3  # never decouple below this of default G

# C3 tanh-crossover sweep (c1b-revision amendment).
# On a passing C1b base, sweep core gm_raw fills and track the gain where
# nonlinearity arrives before MC collapses.
C3_GM_GRID: tuple[float, ...] = (-8.0, -5.0, -2.0, 0.0, 1.5, 3.0)
C3_MC_COLLAPSE_RATIO: float = 0.5  # MC <= 0.5 * c1b_mc counts as collapsed

# R0 raw-delay Ridge window for "matched parity" comparison.
R0_RAW_DELAY_TAPS: int = 20

# Sidecar protocol for the tuned C1b base.
C1B_SIDECAR_NAME: str = "c1b_tuned_base.pt"
C1B_SIDECAR_JSON: str = "c1b_tuned_base.json"

# Long-stream R0 reference (cross-validated vs the k8 frozen-state 0.7194).
R0_LONG_STREAM_LEN: int = 10000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RidgeRow:
    """One Ridge readout row at a given diagnostic configuration."""

    config_tag: str
    diagnostic: str  # "e0_full_state" | "legacy_hidden_only" | "raw_delay"
    nrmse: float
    r2: float
    n_features: int
    drive_scale: float
    stream_length: int
    refresh: str
    note: str = ""


@dataclass
class R0ReconciliationReport:
    """Aggregate R0 reconciliation matrix."""

    order: int
    seed: int
    device: str
    n_streams: int
    train_samples_per_stream: int
    washout: int
    canonical_net_kwargs: dict
    rows: list[RidgeRow]
    matched_parity_rule: dict
    n_corners: int
    elapsed_s: float
    note: str = ""


@dataclass
class InstrumentRow:
    """Generic instrument row (Ridge, MC, PR, Jacobian, saturation)."""

    config_tag: str
    nrmse: float
    r2: float
    mc_total: float
    mc_per_delay: list[float]
    state_pr: float
    jac_max_abs: float
    jac_min_abs: float
    jac_mean_abs: float
    jac_rank_proxy: float
    jac_eig_abs: list[float] = field(default_factory=list)
    sat_max_ratio: float = float("nan")
    rail_frac: float = float("nan")
    n_params: int = 0
    note: str = ""
    # Standardized-state PR (zero-mean, unit-variance per node).  Optional
    # for backward compatibility with audit2 checks that build InstrumentRow
    # from the c1b c1-style kwargs.  NaN if not computed.
    state_pr_standardized: float = float("nan")


@dataclass
class LinearReservoirState:
    """Snapshot of monkey-patched fabric state (for clean restoration)."""

    cell_lib_forward: Any
    cell_lib_resistive: Any | None
    boundary_cell_lib_forward: Any | None
    boundary_cell_lib_resistive: Any | None
    output_ode_cell_lib_forward: Any | None
    output_ode_cell_lib_resistive: Any | None
    z_logits: torch.Tensor
    boundary_z_logits: torch.Tensor
    output_ode_z_logits: torch.Tensor
    raw_leak: torch.Tensor | None
    x_max: float
    clip_current: float
    leak_mode: str
    leak_constant: float | None
    drive_current: Any
    G_edge: torch.Tensor
    G_in: torch.Tensor


@dataclass
class LinearReservoirReport:
    """C1 linear-reservoir-in-fabric control report."""

    order: int
    seed: int
    device: str
    hidden_dim: int
    n_streams: int
    train_samples_per_stream: int
    washout: int
    n_params: int
    canonical_target_radius_low: float
    canonical_target_radius_high: float
    n_tuning_iterations: int
    tuning_history: list[dict]
    instrument: InstrumentRow
    matched_raw_delay_nrmse: float
    matched_parity_tolerance: float
    matched_parity_delta: float
    pass_mc: bool
    pass_pr: bool
    pass_ridge_matched: bool
    pass_all: bool
    nrmse_hidden: float
    r2_hidden: float
    mc_total_hidden: float
    state_pr_hidden: float
    note: str = ""


@dataclass
class C1bLinearReservoirReport:
    """C1b orthogonal-mixing revision control report.

    Same instrument set as ``LinearReservoirReport`` (per the c1b-protocol
    spec) with parity DROPPED for linear controls and PR reported as both
    raw and standardized.  The tuned W + leak vector + G_in are persisted
    to ``C1B_SIDECAR_NAME`` so C2/C3/C4 and the Alliance bisection legs
    reinstall the same base.
    """

    order: int
    seed: int
    device: str
    hidden_dim: int
    n_streams: int
    train_samples_per_stream: int
    washout: int
    n_params: int
    n_tuning_iterations: int
    tuning_history: list[dict]
    instrument: InstrumentRow
    # Diagnostic only -- no longer a pass gate for linear legs.
    matched_raw_delay_nrmse: float
    matched_parity_tolerance: float
    matched_parity_delta: float
    # Leak mode is forced uniform in C1b; the install path always uses
    # ``leak_value=C1B_LEAK_VALUE``.
    leak_value: float
    spectral_target_low: float
    spectral_target_high: float
    pre_registered_mc_low: float
    pre_registered_mc_high: float
    # W construction provenance -- aid reproducibility / audit.
    w_seed: int
    w_construction: str  # "qr_orthogonal_from_seed_ginibre"
    # Standardized-state PR diagnostic (raw + standardized).
    state_pr_raw: float
    state_pr_standardized: float
    # Pass gates.
    pass_mc: bool
    pass_rail: bool
    pass_spectral: bool
    pass_all: bool
    # Sidecar pointer (relative path; absolute on disk for the run).
    sidecar_path: str = ""
    sidecar_json_path: str = ""
    note: str = ""


@dataclass
class C1bRealReport:
    """C1b-real resistive-core variant report.

    Installed via the real rhs (gm/isat -> min, softplus-inverse G) without
    any rhs override.  Same gates as C1b.
    """

    order: int
    seed: int
    device: str
    hidden_dim: int
    n_streams: int
    train_samples_per_stream: int
    washout: int
    n_params: int
    n_tuning_iterations: int
    tuning_history: list[dict]
    instrument: InstrumentRow
    matched_raw_delay_nrmse: float
    matched_parity_tolerance: float
    matched_parity_delta: float
    # C1b-real specifics.
    gm_raw_fill: float
    isat_raw_fill: float
    boundary_drive_scale: float
    leak_value: float
    spectral_target_low: float
    spectral_target_high: float
    state_pr_raw: float
    state_pr_standardized: float
    pass_mc: bool
    pass_rail: bool
    pass_spectral: bool
    pass_all: bool
    note: str = ""


@dataclass
class C3SweepRow:
    """C3 tanh-crossover sweep row."""

    config_tag: str
    gm_raw_fill: float
    mc_total: float
    ridge_nrmse: float
    ridge_r2: float
    state_pr_raw: float
    state_pr_standardized: float
    jac_max_abs: float
    rail_frac: float
    sat_max_ratio: float
    collapsed: bool  # True iff MC <= C3_MC_COLLAPSE_RATIO * c1b_mc
    note: str = ""


@dataclass
class C3SweepReport:
    """C3 tanh-crossover sweep report."""

    order: int
    seed: int
    device: str
    hidden_dim: int
    c1b_path: str
    c1b_mc_total: float
    gm_grid: list[float]
    rows: list[C3SweepRow]
    crossover_gm_raw_fill: float | None
    note: str = ""


@dataclass
class TunedBaseSidecar:
    """JSON pointer + portable payload for the tuned C1b base.

    Used by C2/C3/C4 bisection and the Alliance bisection legs to
    reinstall the exact tuned W/leak/G_in without retuning.  Payload
    buffers are stored in the sibling ``.pt`` file (path stored here).
    """

    schema_version: int
    w_seed: int
    leak_value: float
    spectral_target_low: float
    spectral_target_high: float
    hidden_dim: int
    hidden_dim_with_output: int
    n_tuning_iterations: int
    final_spectral_radius: float
    sidecar_pt_path: str
    n_params: int
    c1b_metrics: dict
    created_at: str
    note: str = ""


@dataclass
class BisectionReport:
    """C2/C3/C4 single-element restoration report."""

    order: int
    seed: int
    device: str
    hidden_dim: int
    n_streams: int
    train_samples_per_stream: int
    washout: int
    c1_instrument: InstrumentRow
    legs: list[InstrumentRow]
    deltas_vs_c1: list[dict]
    suppressor: str
    c1b_sidecar_path: str = ""  # populated when running off a C1b sidecar
    note: str = ""


# ---------------------------------------------------------------------------
# Generic helpers (re-exported from narma_advisor_probes where possible)
# ---------------------------------------------------------------------------


def _ridge_fit_predict(
    X: torch.Tensor, y: torch.Tensor, l2: float = 1e-2,
) -> torch.Tensor:
    """Closed-form ridge fit ``W = (X^T X + l2 I)^-1 X^T y`` with bias."""
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    XtX = X_aug.T @ X_aug + l2 * torch.eye(X_aug.shape[1], device=X.device)
    return torch.linalg.solve(XtX, X_aug.T @ y)


def _raw_delay_features(u: torch.Tensor, n_taps: int) -> tuple[torch.Tensor, int]:
    """Build a raw-delay tapped feature matrix (ESN-style input-only Ridge)."""
    if u.dim() != 1:
        raise ValueError(f"raw delay features require (T,) input, got {tuple(u.shape)}")
    T = u.shape[0]
    if T < n_taps:
        raise ValueError(f"need at least {n_taps} samples, got {T}")
    rows = []
    for lag in range(n_taps):
        rows.append(u[n_taps - 1 - lag: T - lag])
    return torch.stack(rows, dim=1), n_taps - 1


def _raw_delay_ridge(
    u_seq: torch.Tensor, y_seq: torch.Tensor, *,
    n_taps: int = R0_RAW_DELAY_TAPS, washout: int = PROBE_WASHOUT,
    l2: float = 1e-2,
) -> dict[str, float]:
    """Fit a raw-delay Ridge on the E0 input sequence (no fabric involved)."""
    X, align = _raw_delay_features(u_seq, n_taps=n_taps)
    y_aligned = y_seq[align:]
    X_w = X[washout:]
    y_w = y_aligned[washout:]
    if X_w.shape[0] == 0:
        return {"nrmse": float("nan"), "r2": float("nan"), "n_features": n_taps + 1}
    W = _ridge_fit_predict(X_w, y_w, l2=l2)
    X_aug = torch.cat([X_w, torch.ones(X_w.shape[0], 1, device=X_w.device)], dim=1)
    pred = X_aug @ W
    return {
        "nrmse": ne.nrmse(pred, y_w),
        "r2": ne.r2(pred, y_w),
        "n_features": n_taps + 1,
    }


def _fabric_full_state_collect(
    net: nn.Module, u_seq: torch.Tensor, *, t_span: float, num_steps: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect ``(T, N)`` full states and return ``(states_full, hidden)``."""
    net.eval()
    net.to(device)
    u_seq = u_seq.to(device)
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "linear-control probes currently support single-stage nets only"
        )
    stage = net.core.stages[0]
    state_width = net.hid_count + net.proj_count + net.output_ode_count
    x0 = u_seq.new_zeros(1, state_width)
    with torch.no_grad():
        all_states = stage._forward_heun_sequence(
            x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_seq,
        )
    states_full = all_states[:, 0, :].detach()
    hidden = states_full[:, : net.hid_count].detach()
    return states_full, hidden


def _safe_participation_ratio(states: torch.Tensor) -> float:
    """Participation ratio with a guarded covariance eigendecomposition."""
    try:
        return float(npr.participation_ratio(states))
    except Exception:
        return float("nan")


def _instrument_trajectory(
    *, stage: nn.Module, states_full: torch.Tensor,
    u_seq: torch.Tensor, y_seq: torch.Tensor, washout: int,
    jacobian_samples: int, t_span: float, num_steps: int,
) -> dict[str, Any]:
    """Score one collected trajectory with the full E0 instrument set.

    Mirrors ``narma_advisor_probes._score_state_trajectory`` but uses the
    guarded eigenvalue path (:func:`_safe_abs_eigvals`,
    :func:`_safe_participation_ratio`) so ill-conditioned linear-reservoir
    Jacobians cannot crash the probe with a LAPACK error.
    """
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
    u_seq = u_seq.to(states_full.device)
    y_seq = y_seq.to(states_full.device)
    x_max = float(stage.x_max)
    with torch.no_grad():
        sat_max = float(states_full.abs().max().item())
        rail_frac = float((states_full.abs() > 0.9 * x_max).float().mean().item())
    state_pr = _safe_participation_ratio(states_full[washout:])
    rows, _ = _jac_eigs_per_transition(
        stage, states_full, u_seq,
        washout=washout, n_samples=jacobian_samples,
        t_span=t_span, num_steps=num_steps,
    )
    finite_rows = [r for r in rows if math.isfinite(r["max_abs"])]
    if finite_rows:
        jac_max_abs = float(max(r["max_abs"] for r in finite_rows))
        jac_min_abs = float(min(r["min_abs"] for r in finite_rows))
        jac_mean_abs = float(
            sum(r["mean_abs"] for r in finite_rows) / len(finite_rows)
        )
        jac_rank_proxy = float(max(r["rank_proxy"] for r in finite_rows))
    else:
        jac_max_abs = jac_min_abs = jac_mean_abs = jac_rank_proxy = float("nan")
    X = states_full[washout:]
    y = y_seq[washout:]
    if X.shape[0] == 0 or not torch.isfinite(X).all() or not torch.isfinite(y).all():
        ridge_nrmse, ridge_r2 = float("nan"), float("nan")
    else:
        W = _ridge_fit_predict(X, y, l2=1e-2)
        pred = torch.cat(
            [X, torch.ones(X.shape[0], 1, device=X.device)], dim=1,
        ) @ W
        ridge_nrmse, ridge_r2 = ne.nrmse(pred, y), ne.r2(pred, y)
    _, mc = memory_capacity_washout_safe(
        states_full, u_seq, washout=washout, max_delay=20,
    )
    return {
        "ridge_nrmse": ridge_nrmse,
        "ridge_r2": ridge_r2,
        "mc_total": float(mc),
        "state_pr": float(state_pr),
        "jac_max_abs": jac_max_abs,
        "jac_min_abs": jac_min_abs,
        "jac_mean_abs": jac_mean_abs,
        "jac_rank_proxy": jac_rank_proxy,
        "rail_frac": float(rail_frac),
        "sat_max_ratio": float(sat_max / x_max) if x_max > 0 else float("nan"),
    }


def memory_capacity_washout_safe(
    states: torch.Tensor, targets: torch.Tensor, *,
    washout: int, max_delay: int = 20, ridge_l2: float = 1e-2,
) -> tuple[list[float], float]:
    """Washout-corrected MC with a guarded covariance path."""
    try:
        return npr.memory_capacity_washout(
            states, targets, washout=washout,
            max_delay=max_delay, ridge_l2=ridge_l2,
        )
    except Exception:
        return [float("nan")] * max_delay, float("nan")


def _per_delay_mc(
    states: torch.Tensor, targets: torch.Tensor, *,
    washout: int, max_delay: int = 20, ridge_l2: float = 1e-2,
    use_svd_fallback: bool = True,
) -> tuple[list[float], float]:
    """Per-delay memory capacity (washout-corrected).

    When ``use_svd_fallback`` is True (default), each per-delay ridge
    solve is checked for ill-conditioning: ``cond(XtX) > 1e10`` falls
    back to a truncated-SVD pseudoinverse (skips the ill-conditioned
    lag, which would otherwise produce absurd MC estimates such as the
    legacy delay-19 ``-159`` row).
    """
    targets = targets.to(states.device)
    if not torch.isfinite(states).all() or not torch.isfinite(targets).all():
        return [float("nan")] * max_delay, float("nan")
    if washout >= states.shape[0]:
        return [float("nan")] * max_delay, float("nan")
    if not use_svd_fallback:
        return ne.memory_capacity(
            states[washout:], targets[washout:],
            max_delay=max_delay, ridge_l2=ridge_l2,
        )
    s_w = states[washout:]
    t_w = targets[washout:]
    r2_list: list[float] = []
    T = s_w.shape[0]
    for k in range(1, max_delay + 1):
        if T - k <= 0:
            r2_list.append(0.0)
            continue
        X = s_w[k:]
        y = t_w[: T - k]
        X_aug = torch.cat(
            [X, torch.ones(X.shape[0], 1, device=X.device)], dim=1,
        )
        XtX = X_aug.T @ X_aug + ridge_l2 * torch.eye(
            X_aug.shape[1], device=X.device,
        )
        try:
            cond = float(torch.linalg.cond(XtX).item())
        except Exception:
            cond = float("inf")
        if math.isfinite(cond) and cond < 1e10:
            try:
                W = torch.linalg.solve(XtX, X_aug.T @ y)
                y_pred = X_aug @ W
                r2_k = ne.r2(y_pred, y)
                r2_list.append(float(r2_k))
                continue
            except Exception:
                pass
        # Truncated-SVD ridge fallback.  The closed-form ridge
        # solution in SVD coordinates is
        # ``W = V @ ((S / (S^2 + l2)) * (U^T y))``; dropping singular
        # vectors below ``1e-8 * max(S)`` keeps an ill-conditioned lag
        # from producing absurd MC estimates (e.g. the legacy
        # delay-19 ``-159`` row).  A fully degenerate lag scores 0.0.
        try:
            U, S, Vt = torch.linalg.svd(X_aug, full_matrices=False)
            max_s = float(S.max().item())
            if not math.isfinite(max_s) or max_s <= 0:
                r2_list.append(0.0)
                continue
            keep = S > 1e-8 * max_s
            if not bool(keep.any()):
                r2_list.append(0.0)
                continue
            U_k = U[:, keep]
            S_k = S[keep]
            Vt_k = Vt[keep, :]
            filt = S_k / (S_k * S_k + ridge_l2)
            W_full = Vt_k.T @ (filt.unsqueeze(1) * (U_k.T @ y.unsqueeze(1)))
            y_pred = (X_aug @ W_full).squeeze(1)
            r2_k = ne.r2(y_pred, y)
            r2_list.append(float(r2_k))
        except Exception:
            r2_list.append(0.0)
    mc_total = sum(max(r, 0.0) for r in r2_list)
    return r2_list, mc_total


def _standardized_state_pr(states: torch.Tensor) -> float:
    """Participation ratio on standardized states (zero-mean, unit-variance).

    The raw-state PR is dominated by slow-leak modes that never decorrelate
    within the stream length; the standardized-state PR isolates the rank
    structure of the state-covariance geometry instead.  Returns NaN if
    the standardization is degenerate (any column has zero variance).
    """
    s = states
    if s.dim() != 2 or s.shape[0] < 2:
        return float("nan")
    s_c = s - s.mean(dim=0, keepdim=True)
    s_std = s.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-9)
    s_norm = s_c / s_std
    cov = (s_norm.T @ s_norm) / max(s_norm.shape[0] - 1, 1)
    try:
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    except Exception:
        return float("nan")
    s_val = float(eig.sum().item())
    sq = float(eig.pow(2).sum().item())
    if sq <= 1e-12:
        return float("nan")
    return (s_val * s_val) / sq


def _flatten_abs_eigs(jac_rows: list[dict[str, float]]) -> list[float]:
    """Flatten per-transition Jacobian eigenvalue magnitudes into one list."""
    if not jac_rows:
        return []
    out: list[float] = []
    for row in jac_rows:
        out.extend([
            float(v) for v in row.get("eig_abs", []) if math.isfinite(float(v))
        ])
    if not out:
        return []
    return out


def _safe_abs_eigvals(matrix: torch.Tensor) -> torch.Tensor:
    """Robust ``eigvals`` for ill-conditioned Jacobians.

    Falls back to a CPU float64 solve with NaN replacement when the
    default backend (LAPACK via MKL on Windows) signals bad parameters
    (typically NaN/Inf or extreme condition numbers after the linear
    reservoir's per-edge conductances amplify the KCL).  Returns
    absolute eigenvalues as a flat tensor.
    """
    try:
        eig = torch.linalg.eigvals(matrix)
        abs_e = eig.abs()
        if not torch.isfinite(abs_e).all():
            raise RuntimeError("non-finite eigenvalues")
        return abs_e
    except Exception:
        m = matrix.detach().to(dtype=torch.float64, device="cpu")
        if not torch.isfinite(m).all():
            return torch.tensor([], dtype=torch.float32)
        try:
            eig = torch.linalg.eigvals(m)
        except Exception:
            return torch.tensor([], dtype=torch.float32)
        return eig.abs().to(dtype=torch.float32)


def _jac_eigs_per_transition(
    stage: nn.Module, states_full: torch.Tensor, u_seq: torch.Tensor, *,
    washout: int, n_samples: int, t_span: float, num_steps: int,
) -> tuple[list[dict[str, float]], list[list[float]]]:
    """Return ``(rows, eig_abs_per_transition)``.

    Eigenvalues per transition are recorded for the schema; the rolled-up
    ``max_abs`` / ``min_abs`` / ``mean_abs`` / ``rank_proxy`` are reported
    by :func:`_collect_jacobian_eigs`.  Uses :func:`_safe_abs_eigvals`
    so the CPU LAPACK backend does not fail on the linear reservoir's
    sometimes ill-conditioned per-sample Jacobians.
    """
    # The state collector moves its local drive copy to the compute
    # device, but the caller's sequence may still be CPU (CUDA run).
    u_seq = u_seq.to(states_full.device)
    transitions = npr._select_transition_points(
        states_full, u_seq, washout=washout, n_samples=n_samples,
    )
    dt = t_span / num_steps
    rows: list[dict[str, float]] = []
    eig_abs_per_transition: list[list[float]] = []
    for x_from, u_next, transition_index in transitions:
        def transition_map(x_flat: torch.Tensor) -> torch.Tensor:
            x = x_flat.view(1, -1)
            return npr._one_sample_transition(
                stage, x, u_next, dt, num_steps,
            )

        J = torch.autograd.functional.jacobian(
            transition_map, x_from.detach().clone().requires_grad_(True),
            create_graph=False,
        )
        abs_eigs = _safe_abs_eigvals(J)
        if abs_eigs.numel() == 0:
            rows.append({
                "transition_index": float(transition_index),
                "state_dim": float(x_from.numel()),
                "max_abs": float("nan"),
                "min_abs": float("nan"),
                "mean_abs": float("nan"),
                "rank_proxy": float("nan"),
            })
            eig_abs_per_transition.append([])
            continue
        rows.append({
            "transition_index": float(transition_index),
            "state_dim": float(x_from.numel()),
            "max_abs": float(abs_eigs.max().item()),
            "min_abs": float(abs_eigs.min().item()),
            "mean_abs": float(abs_eigs.mean().item()),
            "rank_proxy": float(npr.participation_ratio(abs_eigs)),
        })
        eig_abs_per_transition.append([float(v) for v in abs_eigs.tolist()])
    return rows, eig_abs_per_transition


def _stage_width(net: nn.Module) -> int:
    return int(net.hid_count + net.proj_count + net.output_ode_count)


def _drive_rms(u_scaled: torch.Tensor) -> float:
    """RMS of the canonical drive-1.0 volt sequence."""
    if u_scaled.numel() == 0:
        return 0.0
    return float(u_scaled.pow(2).mean().sqrt().item())


# ---------------------------------------------------------------------------
# Linear reservoir monkey-patching (C1)
# ---------------------------------------------------------------------------


def install_linear_reservoir(
    net: nn.Module, *,
    G_edge_seed: int,
    G_in_seed: int,
    leak_seed: int,
    target_x_max: float = CANONICAL_X_MAX_LIN,
    clip_current: float = CANONICAL_CLIP_LIN,
    drive_rms_target: float | None = None,
    u_seq_for_rms: torch.Tensor | None = None,
) -> LinearReservoirState:
    """Convert a built NARMA fabric into a linear reservoir at runtime.

    Kept for backward compatibility with the audit harness; the active
    linear-reservoir install path is :func:`install_linear_reservoir_v2`,
    which rebinds the linear KCL closure on each ``tune_to_target_radius``
    iteration via a mutable ``_lin_G_edge`` attribute on ``cell_lib``.
    """
    return install_linear_reservoir_v2(
        net,
        G_edge_seed=G_edge_seed, G_in_seed=G_in_seed, leak_seed=leak_seed,
        target_x_max=target_x_max, clip_current=clip_current,
        drive_rms_target=drive_rms_target, u_seq_for_rms=u_seq_for_rms,
    )


def restore_linear_reservoir(
    net: nn.Module, saved: LinearReservoirState,
) -> None:
    """Restore the fabric to its pre-monkey-patched state."""
    return restore_linear_reservoir_v2(net, saved)


# ---------------------------------------------------------------------------
# Spectral-radius measurement and tuning
# ---------------------------------------------------------------------------


def measure_spectral_radius(
    net: nn.Module, u_seq: torch.Tensor, *, t_span: float, num_steps: int,
    washout: int = PROBE_WASHOUT, n_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure per-sample Jacobian spectral radius at observed transitions."""
    states_full, _ = _fabric_full_state_collect(
        net, u_seq, t_span=t_span, num_steps=num_steps, device=device,
    )
    rows, eig_per_trans = _jac_eigs_per_transition(
        net.core.stages[0], states_full, u_seq,
        washout=washout, n_samples=n_samples,
        t_span=t_span, num_steps=num_steps,
    )
    abs_eigs = _flatten_abs_eigs(rows + [{"eig_abs": eigs}
                                          for eigs in eig_per_trans])
    return {
        "max_abs": float(max(r["max_abs"] for r in rows)),
        "min_abs": float(min(r["min_abs"] for r in rows)),
        "mean_abs": float(
            sum(r["mean_abs"] for r in rows) / max(len(rows), 1)
        ),
        "rank_proxy": float(max(r["rank_proxy"] for r in rows)),
        "n_transitions": len(rows),
        "abs_eigs_per_transition": eig_per_trans,
        "abs_eigs": abs_eigs,
        "states_full": states_full,
        "jacobian_rows": rows,
    }


def tune_to_target_radius(
    net: nn.Module, u_seq: torch.Tensor, *,
    t_span: float, num_steps: int,
    target_low: float = C1_SPECTRAL_TARGET_LOW,
    target_high: float = C1_SPECTRAL_TARGET_HIGH,
    tol: float = C1_TUNE_TOL,
    max_iters: int = C1_TUNE_MAX_ITERS,
    washout: int = PROBE_WASHOUT,
    n_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    device: str = "cpu",
) -> tuple[float, list[dict]]:
    """Iteratively rescale ``W`` until spectral radius lands in band.

    The "W matrix" in the linear-reservoir-in-fabric install is stored
    on the stage (``stage._lin_G_edge`` is repurposed to hold the dense
    W).  We scale the entire W in place when the measured per-sample
    Jacobian spectral radius is off-target.
    """
    stage = net.core.stages[0]
    history: list[dict] = []
    last_radius = float("nan")
    for it in range(max_iters):
        spec = measure_spectral_radius(
            net, u_seq, t_span=t_span, num_steps=num_steps,
            washout=washout, n_samples=n_samples, device=device,
        )
        current = spec["max_abs"]
        last_radius = current
        history.append({
            "iter": it,
            "max_abs": current,
            "min_abs": spec["min_abs"],
            "mean_abs": spec["mean_abs"],
            "rank_proxy": spec["rank_proxy"],
            "n_transitions": spec["n_transitions"],
        })
        if math.isfinite(current) and target_low - tol <= current <= target_high + tol:
            return current, history
        W = getattr(stage, "_lin_G_edge", None)
        if W is None:
            return current, history
        if not math.isfinite(current) or current <= 0:
            # Ill-conditioned Jacobian: halve W and retry rather than
            # propagating NaN into the rescale.
            new_W = (W.detach() * 0.5).clone()
        else:
            target = 0.5 * (target_low + target_high)
            # Relationship between per-sample |J| and W spectral radius is
            # nonlinear; we use a per-iteration step of the form
            # W_new = W_old * (target / current) to converge in a few iters.
            scale = float(target) / max(current, 1e-6)
            # Damp the step size to avoid overshoot.
            scale = 0.5 * (scale + 1.0)
            new_W = (W.detach() * scale).clone()
        setattr(stage, "_lin_G_edge", new_W)
        cell_lib = stage.cell_lib
        setattr(cell_lib, "_lin_G_edge", new_W)
    return last_radius, history


def _current_W(stage: nn.Module) -> torch.Tensor | None:
    """Read the dense linear-reservoir ``W`` from the install snapshot."""
    return getattr(stage, "_lin_G_edge", None)


def install_linear_reservoir_v2(
    net: nn.Module, *,
    G_edge_seed: int,
    G_in_seed: int,
    leak_seed: int,
    target_x_max: float = CANONICAL_X_MAX_LIN,
    clip_current: float = CANONICAL_CLIP_LIN,
    drive_rms_target: float | None = None,
    u_seq_for_rms: torch.Tensor | None = None,
) -> LinearReservoirState:
    """Linear-reservoir install with a stable dense ``W @ x`` map.

    The fabric's per-edge KCL contract (random ``G_edge ~ N(0, 1)``)
    yields an inherently unstable linear operator because the per-edge
    conductances mix into a positive-real-part spectrum.  A proper
    ESN-style linear reservoir uses a dense ``W`` rescaled to a target
    spectral radius, applied as ``KCL = W @ x`` with the leak giving
    the damping.  This implementation overrides ``stage.rhs`` to
    compute that stable ``W @ x`` map directly (the cell_lib is bypassed
    by the patched rhs, so the gate/mask machinery is irrelevant).
    """
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "linear reservoir monkey-patch supports single-stage nets only"
        )
    stage = net.core.stages[0]
    cell_lib = stage.cell_lib
    boundary_cell_lib = stage.boundary_cell_lib
    output_ode_cell_lib = stage.output_ode_cell_lib

    n_nodes = int(stage.num_nodes)
    n_hidden = int(net.hid_count)
    n_boundary_edges = int(stage.boundary_src.shape[0])

    g_in_gen = torch.Generator(device="cpu").manual_seed(int(G_in_seed))
    if n_boundary_edges > 0:
        G_in = torch.randn(n_boundary_edges, generator=g_in_gen)
        if drive_rms_target is not None and u_seq_for_rms is not None:
            actual = _drive_rms(u_seq_for_rms.to(torch.float32))
            if actual > 0:
                G_in = G_in * (float(drive_rms_target) / actual)
    else:
        G_in = torch.empty(0)

    g_w_gen = torch.Generator(device="cpu").manual_seed(int(G_edge_seed))
    W_raw = torch.randn(n_nodes, n_nodes, generator=g_w_gen)
    # Rescale W so the per-substep Jacobian (with leak and dt=0.125)
    # is in the canonical mid-band.  The per-substep Jacobian magnitude
    # is approximately |1 + dt*(lambda_W - leak) + 0.5*dt^2*(lambda_W - leak)^2|;
    # with leak=0.05 and dt=0.125, for real lambda_W this is
    # 1 + 0.125*(lambda_W - 0.05) for the dominant eigenvalue.
    # We pick a target of ~0.95 for the dominant |lambda| of (W-leak)
    # (i.e., target radius for the underlying W).
    eig_raw = torch.linalg.eigvals(W_raw)
    rho_raw = float(eig_raw.abs().max().item())
    # Seed W with a small spectral radius; the spectral tuner
    # (tune_to_target_radius) measures the true per-sample Heun-map
    # Jacobian at observed transitions and rescales W until it lands
    # in [0.93, 0.97].  The seed must be STABLE on iteration 0
    # (per-sample |J| = exp(rho_W - leak) < 1 requires rho_W below the
    # ~0.05 mean leak), otherwise diverged states poison every later
    # measurement and the tuner cannot converge.
    target_rho_W = 0.02
    if rho_raw > 0:
        W_natural = W_raw * (target_rho_W / rho_raw)
    else:
        W_natural = W_raw

    saved = LinearReservoirState(
        cell_lib_forward=cell_lib.forward,
        cell_lib_resistive=(
            cell_lib.resistive_current if hasattr(cell_lib, "resistive_current")
            else None
        ),
        boundary_cell_lib_forward=(
            boundary_cell_lib.forward
            if boundary_cell_lib is not None and hasattr(boundary_cell_lib, "forward")
            else None
        ),
        boundary_cell_lib_resistive=(
            boundary_cell_lib.resistive_current
            if boundary_cell_lib is not None
            and hasattr(boundary_cell_lib, "resistive_current")
            else None
        ),
        output_ode_cell_lib_forward=(
            output_ode_cell_lib.forward
            if output_ode_cell_lib is not None
            and hasattr(output_ode_cell_lib, "forward")
            else None
        ),
        output_ode_cell_lib_resistive=(
            output_ode_cell_lib.resistive_current
            if output_ode_cell_lib is not None
            and hasattr(output_ode_cell_lib, "resistive_current")
            else None
        ),
        z_logits=stage.z_logits.detach().clone(),
        boundary_z_logits=(
            stage.boundary_z_logits.detach().clone()
            if stage.boundary_z_logits is not None
            else torch.empty(0)
        ),
        output_ode_z_logits=(
            stage.output_ode_z_logits.detach().clone()
            if hasattr(stage, "output_ode_z_logits")
            and stage.output_ode_z_logits is not None
            else torch.empty(0)
        ),
        raw_leak=(
            stage.raw_leak.detach().clone()
            if hasattr(stage, "raw_leak") and stage.raw_leak is not None
            else None
        ),
        x_max=float(stage.x_max),
        clip_current=float(stage.clip_current),
        leak_mode=str(stage.leak_mode),
        leak_constant=(
            float(stage.leak_constant) if hasattr(stage, "leak_constant")
            else float("nan")
        ),
        drive_current=stage.drive_current,
        G_edge=W_natural.detach().clone(),
        G_in=G_in.detach().clone(),
    )

    with torch.no_grad():
        stage.z_logits.data.fill_(12.0)
        if stage.boundary_z_logits is not None:
            stage.boundary_z_logits.data.fill_(12.0)
        if (
            hasattr(stage, "output_ode_z_logits")
            and stage.output_ode_z_logits is not None
        ):
            stage.output_ode_z_logits.data.fill_(12.0)
    stage.x_max = float(target_x_max)
    stage.clip_current = float(clip_current)

    leak_gen = torch.Generator(device="cpu").manual_seed(int(leak_seed))
    # Log-uniform per-node leaks.  The per-sample map is leak-dominated
    # (|J| ~= exp(rho_W - min_leak)), so the SLOWEST leak sets the
    # spectral radius: [0.06, 0.3] puts J near exp(-0.04) ~= 0.96
    # (inside the [0.93, 0.97] band) while keeping DC gains (1/leak)
    # within a 5x spread so slow nodes cannot drown the covariance.
    log_lo, log_hi = math.log(0.06), math.log(0.3)
    u_leak = torch.rand(n_nodes, generator=leak_gen) * (log_hi - log_lo) + log_lo
    leak_targets = u_leak.exp()
    leak_raw = torch.log(torch.expm1(leak_targets)).clamp_min(-20.0)
    if stage.leak_mode != "programmable":
        stage.leak_mode = "programmable"
        if not hasattr(stage, "raw_leak") or stage.raw_leak is None:
            stage.raw_leak = nn.Parameter(torch.full((n_nodes,), -3.0))
    with torch.no_grad():
        if stage.raw_leak.shape != leak_raw.shape:
            stage.raw_leak = nn.Parameter(leak_raw.clone())
        else:
            stage.raw_leak.data.copy_(leak_raw)

    # Save the stage.rhs so we can patch a stable linear-reservoir rhs.
    saved_rhs = stage.rhs
    setattr(stage, "_orig_rhs", saved_rhs)

    # The dense W lives on stage._lin_G_edge (cell_lib also has a copy
    # for the audit harness).  The bisection legs flip the restore
    # flags below; the patched rhs branches on them so C2/C3 actually
    # change the dynamics (reverting cell_lib.forward alone would be a
    # no-op because the patched rhs bypasses the cell lib).
    setattr(cell_lib, "_lin_G_edge", W_natural.detach().clone())
    setattr(stage, "_lin_G_edge", W_natural.detach().clone())
    setattr(boundary_cell_lib, "_lin_G_in", G_in.detach().clone())
    setattr(stage, "_lin_c_eff", float(getattr(stage, "c_eff", 1.0)))
    setattr(stage, "_lin_injection_dst", stage.boundary_dst.detach().clone())
    setattr(stage, "_lin_injection_src", stage.boundary_src.detach().clone())
    setattr(stage, "_lin_restore_boundary", False)
    setattr(stage, "_lin_restore_tanh", False)
    setattr(stage, "_lin_orig_cell_forward", saved.cell_lib_forward)
    setattr(stage, "_lin_orig_boundary_forward", saved.boundary_cell_lib_forward)

    def _linear_reservoir_rhs(
        x: torch.Tensor,
        u: torch.Tensor | None = None,
        x_drive: torch.Tensor | None = None,
        drive_scale: float = 0.0,
        leak_floor: float | None = None,
        i_edge_const: torch.Tensor | None = None,
        i_boundary_const: torch.Tensor | None = None,
        i_readout_const: torch.Tensor | None = None,
        vca_gate_core: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Linear reservoir dynamics:
        # dx/dt = (core_acc + boundary_acc - leak * x - clip) / c_eff.
        # The restore flags (C2/C3 bisection) swap one family back to
        # its canonical cell-library evaluation.
        c_eff = float(getattr(stage, "_lin_c_eff", 1.0))
        restore_tanh = bool(getattr(stage, "_lin_restore_tanh", False))
        restore_boundary = bool(getattr(stage, "_lin_restore_boundary", False))
        if restore_tanh:
            orig_cell_forward = getattr(stage, "_lin_orig_cell_forward", None)
            if orig_cell_forward is None:
                return saved_rhs(
                    x, u=u, x_drive=x_drive, drive_scale=drive_scale,
                    leak_floor=leak_floor,
                    i_edge_const=i_edge_const,
                    i_boundary_const=i_boundary_const,
                    i_readout_const=i_readout_const,
                    vca_gate_core=vca_gate_core,
                )
            x_src = x[:, stage.src]
            x_dst = x[:, stage.dst]
            i_edge = orig_cell_forward(
                x_src=x_src, x_dst=x_dst, x_max=stage.x_max,
            )
            edge_mask = torch.sigmoid(stage.z_logits)
            if stage.budget_enabled:
                edge_mask = edge_mask * stage._compute_budget_gate()
            i_edge = i_edge * edge_mask.unsqueeze(0)
            if vca_gate_core is not None:
                i_edge = i_edge * vca_gate_core
            acc = torch.zeros_like(x, dtype=torch.float32)
            acc.index_add_(1, stage.dst, i_edge.float())
            if not stage.read_only_source:
                acc.index_add_(1, stage.src, -i_edge.float())
            acc = acc.to(dtype=x.dtype)
        else:
            W = getattr(stage, "_lin_G_edge", None)
            if W is None:
                return saved_rhs(
                    x, u=u, x_drive=x_drive, drive_scale=drive_scale,
                    leak_floor=leak_floor,
                    i_edge_const=i_edge_const,
                    i_boundary_const=i_boundary_const,
                    i_readout_const=i_readout_const,
                    vca_gate_core=vca_gate_core,
                )
            acc = x @ W.to(x.device, x.dtype).T
        if restore_boundary:
            orig_b_forward = getattr(stage, "_lin_orig_boundary_forward", None)
            if (
                orig_b_forward is not None and u is not None
                and stage._has_boundary and stage.boundary_src.numel() > 0
            ):
                dev = x.device
                u_src = u.to(dev)[:, stage.boundary_src.to(dev)]
                x_dst_b = x[:, stage.boundary_dst.to(dev)]
                i_b = orig_b_forward(
                    x_src=u_src, x_dst=x_dst_b, x_max=stage.x_max,
                )
                i_b = i_b * torch.sigmoid(stage.boundary_z_logits).unsqueeze(0)
                acc_b = torch.zeros_like(acc, dtype=torch.float32)
                acc_b.index_add_(1, stage.boundary_dst.to(dev), i_b.float())
                acc = (acc.float() + acc_b).to(dtype=acc.dtype)
        else:
            if u is not None and stage._has_boundary:
                G_in_t = getattr(boundary_cell_lib, "_lin_G_in", None)
                if G_in_t is not None:
                    b_src = getattr(stage, "_lin_injection_src", None)
                    b_dst = getattr(stage, "_lin_injection_dst", None)
                    if b_src is not None and b_dst is not None and G_in_t.numel() > 0:
                        # All stored tensors are plain CPU attributes
                        # (setattr does not register buffers) and the
                        # caller may pass a CPU drive with a CUDA state,
                        # so everything follows the state device here.
                        dev = x.device
                        u_src = u.to(dev)[:, b_src.to(dev)]
                        i_b = (
                            G_in_t.to(dev, x.dtype).unsqueeze(0) * u_src
                        ).to(acc.dtype)
                        acc_b = torch.zeros_like(acc)
                        acc_b.index_add_(1, b_dst.to(dev), i_b)
                        acc = acc + acc_b
        # Leak.
        leak = stage._effective_leak(leak_floor=leak_floor).unsqueeze(0).to(x.device, x.dtype)
        leak_term = leak * x
        # Clip.
        clip = torch.sigmoid((x - stage.x_max) / stage.clip_softness)
        clip = clip - torch.sigmoid((-x - stage.x_max) / stage.clip_softness)
        clip_term = stage.clip_current * clip
        return (acc - leak_term - clip_term) / c_eff

    stage.rhs = _linear_reservoir_rhs

    def _zero_resistive(x_src: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x_src)

    if hasattr(cell_lib, "resistive_current"):
        cell_lib.resistive_current = _zero_resistive
    if (
        boundary_cell_lib is not None
        and hasattr(boundary_cell_lib, "resistive_current")
    ):
        boundary_cell_lib.resistive_current = _zero_resistive
    if (
        output_ode_cell_lib is not None
        and hasattr(output_ode_cell_lib, "resistive_current")
    ):
        output_ode_cell_lib.resistive_current = _zero_resistive

    return saved


def restore_linear_reservoir_v2(
    net: nn.Module, saved: LinearReservoirState,
) -> None:
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "linear reservoir restore supports single-stage nets only"
        )
    stage = net.core.stages[0]
    cell_lib = stage.cell_lib
    boundary_cell_lib = stage.boundary_cell_lib
    output_ode_cell_lib = stage.output_ode_cell_lib
    # The install function swapped stage.rhs with a linear-reservoir rhs
    # closure.  We cannot recover the original rhs by attribute lookup
    # (it was never stored); instead the audit calls
    # ``stage.rhs = stage._orig_rhs`` if present.  We do the same here
    # defensively.
    if hasattr(stage, "_orig_rhs"):
        stage.rhs = stage._orig_rhs
        delattr(stage, "_orig_rhs")
    cell_lib.forward = saved.cell_lib_forward
    if saved.cell_lib_resistive is not None and hasattr(cell_lib, "resistive_current"):
        cell_lib.resistive_current = saved.cell_lib_resistive
    if (
        boundary_cell_lib is not None
        and saved.boundary_cell_lib_forward is not None
        and hasattr(boundary_cell_lib, "forward")
    ):
        boundary_cell_lib.forward = saved.boundary_cell_lib_forward
    if (
        boundary_cell_lib is not None
        and saved.boundary_cell_lib_resistive is not None
        and hasattr(boundary_cell_lib, "resistive_current")
    ):
        boundary_cell_lib.resistive_current = saved.boundary_cell_lib_resistive
    if (
        output_ode_cell_lib is not None
        and saved.output_ode_cell_lib_forward is not None
        and hasattr(output_ode_cell_lib, "forward")
    ):
        output_ode_cell_lib.forward = saved.output_ode_cell_lib_forward
    if (
        output_ode_cell_lib is not None
        and saved.output_ode_cell_lib_resistive is not None
        and hasattr(output_ode_cell_lib, "resistive_current")
    ):
        output_ode_cell_lib.resistive_current = saved.output_ode_cell_lib_resistive
    with torch.no_grad():
        stage.z_logits.data.copy_(saved.z_logits)
        if stage.boundary_z_logits is not None:
            stage.boundary_z_logits.data.copy_(saved.boundary_z_logits)
        if (
            hasattr(stage, "output_ode_z_logits")
            and stage.output_ode_z_logits is not None
            and stage.output_ode_z_logits.shape == saved.output_ode_z_logits.shape
        ):
            stage.output_ode_z_logits.data.copy_(saved.output_ode_z_logits)
        if (
            saved.raw_leak is not None
            and hasattr(stage, "raw_leak")
            and stage.raw_leak is not None
            and stage.raw_leak.shape == saved.raw_leak.shape
        ):
            stage.raw_leak.data.copy_(saved.raw_leak)
    stage.x_max = saved.x_max
    stage.clip_current = saved.clip_current
    stage.leak_mode = saved.leak_mode
    if hasattr(stage, "leak_constant"):
        stage.leak_constant = saved.leak_constant
    stage.drive_current = saved.drive_current
    if hasattr(cell_lib, "_lin_G_edge"):
        delattr(cell_lib, "_lin_G_edge")
    if boundary_cell_lib is not None and hasattr(boundary_cell_lib, "_lin_G_in"):
        delattr(boundary_cell_lib, "_lin_G_in")
    for attr in (
        "_lin_G_edge", "_lin_c_eff", "_lin_injection_dst",
        "_lin_injection_src", "_lin_restore_boundary", "_lin_restore_tanh",
        "_lin_orig_cell_forward", "_lin_orig_boundary_forward",
    ):
        if hasattr(stage, attr):
            delattr(stage, attr)


# ---------------------------------------------------------------------------
# C1b orthogonal-mixing revision (c1b-revision plan, c1b-protocol spec).
# ---------------------------------------------------------------------------


def _qr_orthogonal_W(
    *, seed: int, dim: int, target_radius: float,
) -> torch.Tensor:
    """Build an orthogonal W via QR-of-seeded-Ginibre, scaled to ``target_radius``.

    The QR factor is unique up to a column-sign ambiguity; we leave the
    columns as-is (sign-flips only matter for eigenvector polarity, not
    for the spectral radius).  Output shape: ``(dim, dim)``.
    """
    if dim < 2:
        raise ValueError(f"QR-orthogonal W requires dim >= 2, got {dim}")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(dim, dim, generator=gen)
    # QR with the reduced mode returns ``(dim, dim)`` factors.
    Q, _ = torch.linalg.qr(raw, mode="reduced")
    # Rescale to the target spectral radius.  For a numerically
    # orthogonal Q, all |eig| are 1; multiply by ``target_radius``.
    return Q * float(target_radius)


def _make_c1b_rhs(
    stage: nn.Module, boundary_cell_lib: nn.Module | None,
    saved_rhs: Any,
):
    """Build the C1b right-hand side closure for one install.

    Default path: ``dx/dt = (W @ x + G_in @ u - leak * x - clip) / c_eff``.
    When ``stage._lin_restore_tanh`` is set (C3 tanh-crossover sweep),
    the core term is evaluated through the saved cell-library forward
    (honoring the caller's ``gm_raw``/``isat_raw`` fills) while the
    boundary stays on the linear ``G_in`` injection.  Shared by
    :func:`install_linear_reservoir_c1b` and
    :func:`reinstall_tuned_c1b` so the sweep and the base cannot drift.
    """
    def _c1b_rhs(
        x: torch.Tensor,
        u: torch.Tensor | None = None,
        x_drive: torch.Tensor | None = None,
        drive_scale: float = 0.0,
        leak_floor: float | None = None,
        i_edge_const: torch.Tensor | None = None,
        i_boundary_const: torch.Tensor | None = None,
        i_readout_const: torch.Tensor | None = None,
        vca_gate_core: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c_eff = float(getattr(stage, "_lin_c_eff", 1.0))
        if bool(getattr(stage, "_lin_restore_tanh", False)):
            # C3 sweep path: real tanh core (with the sweep's gm/isat
            # fills), linear boundary injection below.
            orig_cell_forward = getattr(stage, "_lin_orig_cell_forward", None)
            if orig_cell_forward is None:
                return saved_rhs(
                    x, u=u, x_drive=x_drive, drive_scale=drive_scale,
                    leak_floor=leak_floor,
                    i_edge_const=i_edge_const,
                    i_boundary_const=i_boundary_const,
                    i_readout_const=i_readout_const,
                    vca_gate_core=vca_gate_core,
                )
            x_src = x[:, stage.src]
            x_dst = x[:, stage.dst]
            i_edge = orig_cell_forward(
                x_src=x_src, x_dst=x_dst, x_max=stage.x_max,
            )
            edge_mask = torch.sigmoid(stage.z_logits)
            if stage.budget_enabled:
                edge_mask = edge_mask * stage._compute_budget_gate()
            i_edge = i_edge * edge_mask.unsqueeze(0)
            if vca_gate_core is not None:
                i_edge = i_edge * vca_gate_core
            acc = torch.zeros_like(x, dtype=torch.float32)
            acc.index_add_(1, stage.dst, i_edge.float())
            if not stage.read_only_source:
                acc.index_add_(1, stage.src, -i_edge.float())
            acc = acc.to(dtype=x.dtype)
        else:
            W_local = getattr(stage, "_lin_G_edge", None)
            if W_local is None:
                return saved_rhs(
                    x, u=u, x_drive=x_drive, drive_scale=drive_scale,
                    leak_floor=leak_floor,
                    i_edge_const=i_edge_const,
                    i_boundary_const=i_boundary_const,
                    i_readout_const=i_readout_const,
                    vca_gate_core=vca_gate_core,
                )
            acc = x @ W_local.to(x.device, x.dtype).T
        if u is not None and stage._has_boundary:
            G_in_t = (
                getattr(boundary_cell_lib, "_lin_G_in", None)
                if boundary_cell_lib is not None else None
            )
            if G_in_t is not None:
                b_src = getattr(stage, "_lin_injection_src", None)
                b_dst = getattr(stage, "_lin_injection_dst", None)
                if (
                    b_src is not None and b_dst is not None
                    and G_in_t.numel() > 0
                ):
                    dev = x.device
                    u_src = u.to(dev)[:, b_src.to(dev)]
                    i_b = (
                        G_in_t.to(dev, x.dtype).unsqueeze(0) * u_src
                    ).to(acc.dtype)
                    acc_b = torch.zeros_like(acc)
                    acc_b.index_add_(1, b_dst.to(dev), i_b)
                    acc = acc + acc_b
        leak = stage._effective_leak(leak_floor=leak_floor).unsqueeze(0).to(x.device, x.dtype)
        leak_term = leak * x
        clip = torch.sigmoid((x - stage.x_max) / stage.clip_softness)
        clip = clip - torch.sigmoid((-x - stage.x_max) / stage.clip_softness)
        clip_term = stage.clip_current * clip
        return (acc - leak_term - clip_term) / c_eff

    return _c1b_rhs


def install_linear_reservoir_c1b(
    net: nn.Module, *,
    w_seed: int, g_in_seed: int,
    target_radius: float = C1B_LEAK_VALUE,
    leak_value: float = C1B_LEAK_VALUE,
    target_x_max: float = CANONICAL_X_MAX_LIN,
    clip_current: float = CANONICAL_CLIP_LIN,
    drive_rms_target: float | None = None,
    u_seq_for_rms: torch.Tensor | None = None,
) -> LinearReservoirState:
    """Install C1b's orthogonal-mixing linear reservoir.

    Differences from :func:`install_linear_reservoir_v2`:

    - W is QR-of-seeded-Ginibre orthogonal (initial) and auto-scaled to
      the J-band by the spectral tuner.  The initial scale is set to
      ``leak_value`` so the per-sample |J| ≈ exp(rho_W - leak) ≈
      exp(leak_value - leak_value) = 1, which leaves the spectral tuner
      a stable starting point inside the band.
    - Leak is UNIFORM (``leak_value``), non-programmable.  We still set
      the stage to ``leak_mode='programmable'`` and broadcast the leak
      across all nodes (so the original cell_lib defaults remain
      untouched and the audit harness can introspect the install), but
      the per-node vector is identical for every node.
    - The rhs honors ``stage._lin_restore_tanh`` (C3 tanh-crossover
      sweep): core through the saved cell-library forward, boundary
      stays linear.  Shared factory :func:`_make_c1b_rhs`.
    """
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "C1b install supports single-stage nets only"
        )
    stage = net.core.stages[0]
    cell_lib = stage.cell_lib
    boundary_cell_lib = stage.boundary_cell_lib
    output_ode_cell_lib = stage.output_ode_cell_lib

    n_nodes = int(stage.num_nodes)
    n_boundary_edges = int(stage.boundary_src.shape[0])

    g_in_gen = torch.Generator(device="cpu").manual_seed(int(g_in_seed))
    if n_boundary_edges > 0:
        G_in = torch.randn(n_boundary_edges, generator=g_in_gen)
        if drive_rms_target is not None and u_seq_for_rms is not None:
            actual = _drive_rms(u_seq_for_rms.to(torch.float32))
            if actual > 0:
                G_in = G_in * (float(drive_rms_target) / actual)
    else:
        G_in = torch.empty(0)

    # Build orthogonal W via QR-of-seeded-Ginibre; rescale to the
    # initial radius.  The spectral tuner will adjust this once the
    # per-sample Jacobian is measured.
    W = _qr_orthogonal_W(seed=w_seed, dim=n_nodes, target_radius=target_radius)

    saved = LinearReservoirState(
        cell_lib_forward=cell_lib.forward,
        cell_lib_resistive=(
            cell_lib.resistive_current if hasattr(cell_lib, "resistive_current")
            else None
        ),
        boundary_cell_lib_forward=(
            boundary_cell_lib.forward
            if boundary_cell_lib is not None and hasattr(boundary_cell_lib, "forward")
            else None
        ),
        boundary_cell_lib_resistive=(
            boundary_cell_lib.resistive_current
            if boundary_cell_lib is not None
            and hasattr(boundary_cell_lib, "resistive_current")
            else None
        ),
        output_ode_cell_lib_forward=(
            output_ode_cell_lib.forward
            if output_ode_cell_lib is not None
            and hasattr(output_ode_cell_lib, "forward")
            else None
        ),
        output_ode_cell_lib_resistive=(
            output_ode_cell_lib.resistive_current
            if output_ode_cell_lib is not None
            and hasattr(output_ode_cell_lib, "resistive_current")
            else None
        ),
        z_logits=stage.z_logits.detach().clone(),
        boundary_z_logits=(
            stage.boundary_z_logits.detach().clone()
            if stage.boundary_z_logits is not None
            else torch.empty(0)
        ),
        output_ode_z_logits=(
            stage.output_ode_z_logits.detach().clone()
            if hasattr(stage, "output_ode_z_logits")
            and stage.output_ode_z_logits is not None
            else torch.empty(0)
        ),
        raw_leak=(
            stage.raw_leak.detach().clone()
            if hasattr(stage, "raw_leak") and stage.raw_leak is not None
            else None
        ),
        x_max=float(stage.x_max),
        clip_current=float(stage.clip_current),
        leak_mode=str(stage.leak_mode),
        leak_constant=(
            float(stage.leak_constant) if hasattr(stage, "leak_constant")
            else float("nan")
        ),
        drive_current=stage.drive_current,
        G_edge=W.detach().clone(),
        G_in=G_in.detach().clone(),
    )

    with torch.no_grad():
        stage.z_logits.data.fill_(12.0)
        if stage.boundary_z_logits is not None:
            stage.boundary_z_logits.data.fill_(12.0)
        if (
            hasattr(stage, "output_ode_z_logits")
            and stage.output_ode_z_logits is not None
        ):
            stage.output_ode_z_logits.data.fill_(12.0)
    stage.x_max = float(target_x_max)
    stage.clip_current = float(clip_current)

    # Uniform leak (programmable mode keeps the install surface uniform
    # with the existing C1 path; raw_leak is filled with the softplus-
    # inverse of ``leak_value`` so softplus(raw_leak) == leak_value).
    leak_raw_value = float(math.log(math.expm1(float(leak_value))))
    if stage.leak_mode != "programmable":
        stage.leak_mode = "programmable"
        if not hasattr(stage, "raw_leak") or stage.raw_leak is None:
            stage.raw_leak = nn.Parameter(torch.full((n_nodes,), leak_raw_value))
    leak_raw = torch.full((n_nodes,), leak_raw_value)
    with torch.no_grad():
        if stage.raw_leak.shape != leak_raw.shape:
            stage.raw_leak = nn.Parameter(leak_raw.clone())
        else:
            stage.raw_leak.data.copy_(leak_raw)

    saved_rhs = stage.rhs
    setattr(stage, "_orig_rhs", saved_rhs)

    setattr(cell_lib, "_lin_G_edge", W.detach().clone())
    setattr(stage, "_lin_G_edge", W.detach().clone())
    setattr(boundary_cell_lib, "_lin_G_in", G_in.detach().clone())
    setattr(stage, "_lin_c_eff", float(getattr(stage, "c_eff", 1.0)))
    setattr(stage, "_lin_injection_dst", stage.boundary_dst.detach().clone())
    setattr(stage, "_lin_injection_src", stage.boundary_src.detach().clone())
    setattr(stage, "_lin_restore_boundary", False)
    setattr(stage, "_lin_restore_tanh", False)
    setattr(stage, "_lin_orig_cell_forward", saved.cell_lib_forward)
    setattr(stage, "_lin_orig_boundary_forward", saved.boundary_cell_lib_forward)

    stage.rhs = _make_c1b_rhs(stage, boundary_cell_lib, saved_rhs)

    def _zero_resistive(x_src: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x_src)

    if hasattr(cell_lib, "resistive_current"):
        cell_lib.resistive_current = _zero_resistive
    if (
        boundary_cell_lib is not None
        and hasattr(boundary_cell_lib, "resistive_current")
    ):
        boundary_cell_lib.resistive_current = _zero_resistive
    if (
        output_ode_cell_lib is not None
        and hasattr(output_ode_cell_lib, "resistive_current")
    ):
        output_ode_cell_lib.resistive_current = _zero_resistive

    return saved


def tune_c1b_to_j_band(
    net: nn.Module, u_seq: torch.Tensor, *,
    t_span: float, num_steps: int,
    target_low: float = C1B_SPECTRAL_TARGET_LOW,
    target_high: float = C1B_SPECTRAL_TARGET_HIGH,
    tol: float = C1B_TUNE_TOL,
    max_iters: int = C1B_TUNE_MAX_ITERS,
    washout: int = PROBE_WASHOUT,
    n_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    device: str = "cpu",
) -> tuple[float, list[dict]]:
    """Tune the QR-orthogonal W until the per-sample Jacobian sits in the J-band.

    Identical scaling rule to :func:`tune_to_target_radius` (damped
    multiplicative rescale, halving on non-finite Jacobians) but pinned
    to the C1b target band [0.97, 0.99].
    """
    stage = net.core.stages[0]
    history: list[dict] = []
    last_radius = float("nan")
    for it in range(max_iters):
        spec = measure_spectral_radius(
            net, u_seq, t_span=t_span, num_steps=num_steps,
            washout=washout, n_samples=n_samples, device=device,
        )
        current = spec["max_abs"]
        last_radius = current
        history.append({
            "iter": it,
            "max_abs": current,
            "min_abs": spec["min_abs"],
            "mean_abs": spec["mean_abs"],
            "rank_proxy": spec["rank_proxy"],
            "n_transitions": spec["n_transitions"],
        })
        if math.isfinite(current) and target_low - tol <= current <= target_high + tol:
            return current, history
        W = getattr(stage, "_lin_G_edge", None)
        if W is None:
            return current, history
        if not math.isfinite(current) or current <= 0:
            new_W = (W.detach() * 0.5).clone()
        else:
            target = 0.5 * (target_low + target_high)
            scale = float(target) / max(current, 1e-6)
            scale = 0.5 * (scale + 1.0)
            new_W = (W.detach() * scale).clone()
        setattr(stage, "_lin_G_edge", new_W)
        cell_lib = stage.cell_lib
        setattr(cell_lib, "_lin_G_edge", new_W)
    return last_radius, history


def save_tuned_base(
    *, W: torch.Tensor, G_in: torch.Tensor, leak_value: float,
    w_seed: int, hidden_dim: int, hidden_dim_with_output: int,
    spectral_target_low: float, spectral_target_high: float,
    n_tuning_iterations: int, final_spectral_radius: float,
    n_params: int, c1b_metrics: dict,
    sidecar_dir: Path,
) -> TunedBaseSidecar:
    """Persist the tuned C1b base to a ``.pt`` + JSON pointer pair.

    The ``.pt`` file stores ``{"W": W, "G_in": G_in}`` so C2/C3/C4 and
    the Alliance bisection legs can reinstall the exact tuned base
    without retuning.  The JSON pointer carries the metadata for
    reproducibility.
    """
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    pt_path = sidecar_dir / C1B_SIDECAR_NAME
    json_path = sidecar_dir / C1B_SIDECAR_JSON
    torch.save({"W": W.detach().cpu(), "G_in": G_in.detach().cpu()}, pt_path)
    payload = TunedBaseSidecar(
        schema_version=1,
        w_seed=int(w_seed),
        leak_value=float(leak_value),
        spectral_target_low=float(spectral_target_low),
        spectral_target_high=float(spectral_target_high),
        hidden_dim=int(hidden_dim),
        hidden_dim_with_output=int(hidden_dim_with_output),
        n_tuning_iterations=int(n_tuning_iterations),
        final_spectral_radius=float(final_spectral_radius),
        sidecar_pt_path=str(pt_path),
        n_params=int(n_params),
        c1b_metrics={k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in c1b_metrics.items()},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        note="C1b orthogonal-mixing tuned base.",
    )
    json_path.write_text(json.dumps(asdict(payload), indent=2))
    return payload


def load_tuned_base(sidecar_dir: Path) -> dict:
    """Load the tuned C1b base from a previously persisted sidecar pair.

    Returns the dict ``{"W": ..., "G_in": ..., "meta": <TunedBaseSidecar>}``.
    """
    pt_path = sidecar_dir / C1B_SIDECAR_NAME
    json_path = sidecar_dir / C1B_SIDECAR_JSON
    if not pt_path.exists():
        raise FileNotFoundError(f"C1b sidecar .pt missing: {pt_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"C1b sidecar .json missing: {json_path}")
    state = torch.load(pt_path, map_location="cpu")
    meta = json.loads(json_path.read_text())
    if not isinstance(state, dict) or "W" not in state or "G_in" not in state:
        raise ValueError(
            f"C1b sidecar .pt must contain {{'W', 'G_in'}}, got keys "
            f"{list(state.keys()) if isinstance(state, dict) else type(state).__name__}"
        )
    return {"W": state["W"], "G_in": state["G_in"], "meta": meta}


def reinstall_tuned_c1b(
    net: nn.Module, *, W: torch.Tensor, G_in: torch.Tensor,
    leak_value: float, drive_rms_target: float | None = None,
    u_seq_for_rms: torch.Tensor | None = None,
) -> LinearReservoirState:
    """Reinstall a previously tuned C1b base into a freshly built net.

    Mirrors :func:`install_linear_reservoir_c1b` but uses the persisted
    W/G_in instead of building a fresh orthogonal matrix.  This is the
    install path used by C2/C3/C4 and the Alliance bisection legs.
    """
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "reinstall_tuned_c1b supports single-stage nets only"
        )
    stage = net.core.stages[0]
    cell_lib = stage.cell_lib
    boundary_cell_lib = stage.boundary_cell_lib
    output_ode_cell_lib = stage.output_ode_cell_lib
    n_nodes = int(stage.num_nodes)

    # Rescale G_in to the requested drive RMS (caller has the same drive
    # as the original C1b install; the scaling factor is rederived).
    G_in_local = G_in.detach().clone()
    if drive_rms_target is not None and u_seq_for_rms is not None:
        actual = _drive_rms(u_seq_for_rms.to(torch.float32))
        if actual > 0:
            G_in_local = G_in_local * (
                float(drive_rms_target) / max(actual, 1e-9)
            )

    saved = LinearReservoirState(
        cell_lib_forward=cell_lib.forward,
        cell_lib_resistive=(
            cell_lib.resistive_current if hasattr(cell_lib, "resistive_current")
            else None
        ),
        boundary_cell_lib_forward=(
            boundary_cell_lib.forward
            if boundary_cell_lib is not None
            and hasattr(boundary_cell_lib, "forward")
            else None
        ),
        boundary_cell_lib_resistive=(
            boundary_cell_lib.resistive_current
            if boundary_cell_lib is not None
            and hasattr(boundary_cell_lib, "resistive_current")
            else None
        ),
        output_ode_cell_lib_forward=(
            output_ode_cell_lib.forward
            if output_ode_cell_lib is not None
            and hasattr(output_ode_cell_lib, "forward")
            else None
        ),
        output_ode_cell_lib_resistive=(
            output_ode_cell_lib.resistive_current
            if output_ode_cell_lib is not None
            and hasattr(output_ode_cell_lib, "resistive_current")
            else None
        ),
        z_logits=stage.z_logits.detach().clone(),
        boundary_z_logits=(
            stage.boundary_z_logits.detach().clone()
            if stage.boundary_z_logits is not None
            else torch.empty(0)
        ),
        output_ode_z_logits=(
            stage.output_ode_z_logits.detach().clone()
            if hasattr(stage, "output_ode_z_logits")
            and stage.output_ode_z_logits is not None
            else torch.empty(0)
        ),
        raw_leak=(
            stage.raw_leak.detach().clone()
            if hasattr(stage, "raw_leak") and stage.raw_leak is not None
            else None
        ),
        x_max=float(stage.x_max),
        clip_current=float(stage.clip_current),
        leak_mode=str(stage.leak_mode),
        leak_constant=(
            float(stage.leak_constant) if hasattr(stage, "leak_constant")
            else float("nan")
        ),
        drive_current=stage.drive_current,
        G_edge=W.detach().clone(),
        G_in=G_in_local.detach().clone(),
    )

    with torch.no_grad():
        stage.z_logits.data.fill_(12.0)
        if stage.boundary_z_logits is not None:
            stage.boundary_z_logits.data.fill_(12.0)
        if (
            hasattr(stage, "output_ode_z_logits")
            and stage.output_ode_z_logits is not None
        ):
            stage.output_ode_z_logits.data.fill_(12.0)
    stage.x_max = float(CANONICAL_X_MAX_LIN)
    stage.clip_current = float(CANONICAL_CLIP_LIN)

    leak_raw_value = float(math.log(math.expm1(float(leak_value))))
    if stage.leak_mode != "programmable":
        stage.leak_mode = "programmable"
        if not hasattr(stage, "raw_leak") or stage.raw_leak is None:
            stage.raw_leak = nn.Parameter(torch.full((n_nodes,), leak_raw_value))
    leak_raw = torch.full((n_nodes,), leak_raw_value)
    with torch.no_grad():
        if stage.raw_leak.shape != leak_raw.shape:
            stage.raw_leak = nn.Parameter(leak_raw.clone())
        else:
            stage.raw_leak.data.copy_(leak_raw)

    saved_rhs = stage.rhs
    setattr(stage, "_orig_rhs", saved_rhs)
    setattr(cell_lib, "_lin_G_edge", W.detach().clone())
    setattr(stage, "_lin_G_edge", W.detach().clone())
    setattr(boundary_cell_lib, "_lin_G_in", G_in_local.detach().clone())
    setattr(stage, "_lin_c_eff", float(getattr(stage, "c_eff", 1.0)))
    setattr(stage, "_lin_injection_dst", stage.boundary_dst.detach().clone())
    setattr(stage, "_lin_injection_src", stage.boundary_src.detach().clone())
    setattr(stage, "_lin_restore_boundary", False)
    setattr(stage, "_lin_restore_tanh", False)
    setattr(stage, "_lin_orig_cell_forward", saved.cell_lib_forward)
    setattr(stage, "_lin_orig_boundary_forward", saved.boundary_cell_lib_forward)

    stage.rhs = _make_c1b_rhs(stage, boundary_cell_lib, saved_rhs)

    def _zero_resistive(x_src: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x_src)

    if hasattr(cell_lib, "resistive_current"):
        cell_lib.resistive_current = _zero_resistive
    if (
        boundary_cell_lib is not None
        and hasattr(boundary_cell_lib, "resistive_current")
    ):
        boundary_cell_lib.resistive_current = _zero_resistive
    if (
        output_ode_cell_lib is not None
        and hasattr(output_ode_cell_lib, "resistive_current")
    ):
        output_ode_cell_lib.resistive_current = _zero_resistive

    return saved


# ---------------------------------------------------------------------------
# C1b probe (orchestrates install + tuning + instrument + sidecar write)
# ---------------------------------------------------------------------------


def c1b_orthogonal_reservoir(
    *, order: int = 10, seed: int = 0, device: str = "cpu",
    hidden_dim: int = CANONICAL_HIDDEN,
    n_streams: int = 1, train_samples_per_stream: int = 300,
    washout: int = PROBE_WASHOUT, max_delay: int = 20,
    jacobian_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    t_span: float = CANONICAL_T_SPAN, num_steps: int = CANONICAL_NUM_STEPS,
    drive_scale: float = CANONICAL_DRIVE_SCALE,
    g_in_seed: int = 0, w_seed: int = 0,
    leak_value: float = C1B_LEAK_VALUE,
    target_low: float = C1B_SPECTRAL_TARGET_LOW,
    target_high: float = C1B_SPECTRAL_TARGET_HIGH,
    matched_parity_tol: float = R0_MATCHED_PARITY_TOL,
    sidecar_dir: Path | None = None,
) -> C1bLinearReservoirReport:
    """Run the C1b orthogonal-mixing revision control.

    Differences from :func:`c1_linear_reservoir`:
      - W = QR-of-seeded-Ginibre orthogonal, auto-scaled to [0.97, 0.99].
      - Leak = uniform ``leak_value`` (default 0.05), non-programmable.
      - Pass gates: MC >= 5 (was 3), rails <= 5%, J in [0.97, 0.99].
      - Parity gate DROPPED for linear controls (diagnostic only).
      - Persists tuned W/leak/G_in to a sidecar pair.
    """
    if order != 10:
        raise ValueError("C1b is calibrated for NARMA-10 only")
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    u_scaled = ne._scale_drive(
        u_stream, bipolar=True, order=order, input_scale=drive_scale,
    )
    u_scaled = u_scaled.to(device)
    y_stream = y_stream.to(device)
    drive_rms = _drive_rms(u_scaled.to(torch.float32))

    net, _, _ = ne._build_fabric_net(
        order=order, seed=seed, freeze_read=False,
        t_span=t_span, num_steps=num_steps, cell_library="tanh_free",
        core_refresh_interval=0, leak_constant=None,
        compile_sequence=False, hidden_dim=hidden_dim,
    )
    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    saved = install_linear_reservoir_c1b(
        net,
        w_seed=w_seed, g_in_seed=g_in_seed,
        target_radius=float(leak_value),  # neutral starting point
        leak_value=float(leak_value),
        target_x_max=CANONICAL_X_MAX_LIN,
        clip_current=CANONICAL_CLIP_LIN,
        drive_rms_target=drive_rms, u_seq_for_rms=u_scaled,
    )
    try:
        radius, history = tune_c1b_to_j_band(
            net, u_scaled, t_span=t_span, num_steps=num_steps,
            target_low=target_low, target_high=target_high,
            tol=C1B_TUNE_TOL, max_iters=C1B_TUNE_MAX_ITERS,
            washout=washout, n_samples=jacobian_samples, device=device,
        )
        spec = measure_spectral_radius(
            net, u_scaled, t_span=t_span, num_steps=num_steps,
            washout=washout, n_samples=jacobian_samples, device=device,
        )
        states_full = spec["states_full"]
        jac_rows = spec["jacobian_rows"]
        abs_eigs: list[float] = []
        for eigs in spec["abs_eigs_per_transition"]:
            abs_eigs.extend([float(v) for v in eigs if math.isfinite(float(v))])
        instrument = _instrument_trajectory(
            stage=net.core.stages[0], states_full=states_full,
            u_seq=u_scaled, y_seq=y_stream, washout=washout,
            jacobian_samples=jacobian_samples,
            t_span=t_span, num_steps=num_steps,
        )
        # Use SVD-fallback per-delay MC for robustness on long streams.
        mc_per, mc_total = _per_delay_mc(
            states_full, u_scaled, washout=washout, max_delay=max_delay,
            use_svd_fallback=True,
        )
        sat_max = float(states_full.abs().max().item())
        rail_frac = float(
            (states_full.abs() > 0.9 * CANONICAL_X_MAX_LIN).float().mean().item()
        )
        # Hidden-only Ridge (legacy hidden-band diagnostic).
        hidden = states_full[:, :hidden_dim].detach()
        X_w = hidden[washout:]
        y_w = y_stream[washout:]
        W_h = _ridge_fit_predict(X_w, y_w)
        X_aug = torch.cat(
            [X_w, torch.ones(X_w.shape[0], 1, device=X_w.device)], dim=1,
        )
        h_pred = X_aug @ W_h
        nrmse_hidden = float(ne.nrmse(h_pred, y_w))
        r2_hidden = float(ne.r2(h_pred, y_w))
        _, mc_total_hidden = _per_delay_mc(
            hidden, u_scaled, washout=washout, max_delay=max_delay,
            use_svd_fallback=True,
        )
        mc_total_hidden = float(mc_total_hidden)
        state_pr_hidden = _safe_participation_ratio(hidden[washout:])
        state_pr_hidden_standardized = _standardized_state_pr(
            states_full[washout:]
        )
        # Matched parity diagnostic (NOT a pass gate for linear controls).
        raw = _raw_delay_ridge(
            u_scaled, y_stream, n_taps=R0_RAW_DELAY_TAPS, washout=washout,
        )
        matched_delta = float(instrument["ridge_nrmse"] - raw["nrmse"])
        # Pass gates (c1b-protocol spec): MC >= 5 AND rails <= 5% AND J in band.
        pass_mc = bool(mc_total >= C1B_PASS_MC_ABOVE)
        pass_rail = bool(rail_frac <= C1B_PASS_RAIL_BELOW)
        pass_spectral = bool(
            math.isfinite(radius)
            and target_low - C1B_TUNE_TOL <= radius <= target_high + C1B_TUNE_TOL
        )
        pass_all = bool(pass_mc and pass_rail and pass_spectral)
        row = InstrumentRow(
            config_tag=(
                f"c1b_orth_seed{seed}_{device}_h{hidden_dim}"
                f"_tspan{t_span:g}_steps{num_steps}"
                f"_drive{drive_scale:g}_washout{washout}"
                f"_ws{w_seed}_gi{g_in_seed}_leak{leak_value:g}"
                f"_target[{target_low:g},{target_high:g}]"
            ),
            nrmse=float(instrument["ridge_nrmse"]),
            r2=float(instrument["ridge_r2"]),
            mc_total=float(mc_total),
            mc_per_delay=[float(v) for v in mc_per],
            state_pr=float(instrument["state_pr"]),
            jac_max_abs=float(instrument["jac_max_abs"]),
            jac_min_abs=float(instrument["jac_min_abs"]),
            jac_mean_abs=float(instrument["jac_mean_abs"]),
            jac_rank_proxy=float(instrument["jac_rank_proxy"]),
            jac_eig_abs=abs_eigs,
            sat_max_ratio=float(sat_max / CANONICAL_X_MAX_LIN) if CANONICAL_X_MAX_LIN > 0 else float("nan"),
            rail_frac=rail_frac,
            n_params=n_params,
            note="C1b orthogonal-mixing linear reservoir",
            state_pr_standardized=float(state_pr_hidden_standardized),
        )
        sidecar_payload: TunedBaseSidecar | None = None
        sidecar_path = ""
        sidecar_json_path = ""
        if sidecar_dir is not None:
            W_t = net.core.stages[0]._lin_G_edge.detach().cpu()
            G_in_t = (
                net.core.stages[0].boundary_cell_lib._lin_G_in.detach().cpu()
                if hasattr(net.core.stages[0].boundary_cell_lib, "_lin_G_in")
                else torch.empty(0)
            )
            metrics = {
                "mc_total": mc_total,
                "nrmse": float(instrument["ridge_nrmse"]),
                "r2": float(instrument["ridge_r2"]),
                "rail_frac": rail_frac,
                "sat_max_ratio": float(
                    sat_max / CANONICAL_X_MAX_LIN
                ) if CANONICAL_X_MAX_LIN > 0 else float("nan"),
                "state_pr": float(instrument["state_pr"]),
                "state_pr_standardized": float(state_pr_hidden_standardized),
                "jac_max_abs": float(instrument["jac_max_abs"]),
                "matched_raw_delay_nrmse": float(raw["nrmse"]),
                "matched_parity_delta": float(matched_delta),
                "nrmse_hidden": nrmse_hidden,
                "r2_hidden": r2_hidden,
                "mc_total_hidden": mc_total_hidden,
                "pass_mc": pass_mc,
                "pass_rail": pass_rail,
                "pass_spectral": pass_spectral,
                "pass_all": pass_all,
            }
            sidecar_payload = save_tuned_base(
                W=W_t, G_in=G_in_t, leak_value=float(leak_value),
                w_seed=int(w_seed), hidden_dim=int(hidden_dim),
                hidden_dim_with_output=int(_stage_width(net)),
                spectral_target_low=float(target_low),
                spectral_target_high=float(target_high),
                n_tuning_iterations=len(history),
                final_spectral_radius=float(radius),
                n_params=int(n_params),
                c1b_metrics=metrics,
                sidecar_dir=sidecar_dir,
            )
            sidecar_path = sidecar_payload.sidecar_pt_path
            sidecar_json_path = str(sidecar_dir / C1B_SIDECAR_JSON)
        return C1bLinearReservoirReport(
            order=order, seed=seed, device=device, hidden_dim=hidden_dim,
            n_streams=n_streams, train_samples_per_stream=train_samples_per_stream,
            washout=washout, n_params=n_params,
            n_tuning_iterations=len(history), tuning_history=history,
            instrument=row,
            matched_raw_delay_nrmse=float(raw["nrmse"]),
            matched_parity_tolerance=float(matched_parity_tol),
            matched_parity_delta=matched_delta,
            leak_value=float(leak_value),
            spectral_target_low=float(target_low),
            spectral_target_high=float(target_high),
            pre_registered_mc_low=C1B_EXPECTED_MC_LOW,
            pre_registered_mc_high=C1B_EXPECTED_MC_HIGH,
            w_seed=int(w_seed),
            w_construction="qr_orthogonal_from_seed_ginibre",
            state_pr_raw=float(state_pr_hidden),
            state_pr_standardized=float(state_pr_hidden_standardized),
            pass_mc=pass_mc, pass_rail=pass_rail,
            pass_spectral=pass_spectral, pass_all=pass_all,
            sidecar_path=sidecar_path,
            sidecar_json_path=sidecar_json_path,
            note=(
                "C1b PASS gates: MC>=5, rails<=5%, J in [0.97,0.99]. "
                "Parity diagnostic only (dropped for linear controls)."
            ),
        )
    finally:
        restore_linear_reservoir_v2(net, saved)


# ---------------------------------------------------------------------------
# C1b-real resistive-core variant (c1b-protocol spec)
# ---------------------------------------------------------------------------


def install_linear_reservoir_c1b_real(
    net: nn.Module, *,
    w_seed: int, g_in_seed: int,
    leak_value: float = C1B_LEAK_VALUE,
    boundary_drive_scale: float = 0.5,
    gm_raw_fill: float = C1B_REAL_GM_MIN_RATIO * 10.0,
    isat_raw_fill: float = C1B_REAL_ISAT_MIN_RATIO * 10.0,
    target_x_max: float = CANONICAL_X_MAX_LIN,
    clip_current: float = CANONICAL_CLIP_LIN,
    drive_rms_target: float | None = None,
    u_seq_for_rms: torch.Tensor | None = None,
) -> dict:
    """Install C1b-real: resistive-core linear reservoir, no rhs override.

    Differences from :func:`install_linear_reservoir_c1b`:

    - The ``stage.rhs`` is NOT replaced.  The reservoir uses the real
      cell-library rhs path.
    - Core edge ``gm_raw`` / ``isat_raw`` are filled so gm and isat land
      near their minima (``sigmoid(x) * (max-min) + min`` -> ~min),
      which makes the resistive shunt dominate the per-edge KCL.
    - Boundary OTA gm_raw/isat_raw are left at their constructor
      defaults (the boundary path remains a small-signal linear
      injector -- the user scales the boundary drive via
      ``boundary_drive_scale`` for the C1b-real leg).
    - The spectral tuner (:func:`tune_c1b_real_to_j_band`) rescales the
      core ``g_resistive_raw`` field itself (softplus-domain, i.e. the
      "softplus-inverse G targets" of the c1b-protocol spec), starting
      from the constructor default.
    - ``w_seed`` is reserved for CLI parity with C1b and unused in the
      resistive path (there is no orthogonal matrix here).

    Returns:
        A dict with the per-stage overrides (gm/isat snapshots, leak,
        raw_leak, G_in, etc.) for the audit harness; restoring is
        done via :func:`restore_linear_reservoir_c1b_real`.
    """
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "C1b-real install supports single-stage nets only"
        )
    stage = net.core.stages[0]
    cell_lib = stage.cell_lib
    n_nodes = int(stage.num_nodes)
    n_boundary_edges = int(stage.boundary_src.shape[0])

    g_in_gen = torch.Generator(device="cpu").manual_seed(int(g_in_seed))
    G_in = torch.randn(n_boundary_edges, generator=g_in_gen)
    if drive_rms_target is not None and u_seq_for_rms is not None:
        actual = _drive_rms(u_seq_for_rms.to(torch.float32))
        if actual > 0:
            G_in = G_in * (float(drive_rms_target) / actual)
    G_in = G_in * float(boundary_drive_scale)

    # Snapshot everything we'll touch.
    snapshots: dict[str, torch.Tensor] = {}
    if hasattr(cell_lib, "gm_raw"):
        snapshots["cell_lib.gm_raw"] = cell_lib.gm_raw.detach().clone()
        snapshots["cell_lib.isat_raw"] = cell_lib.isat_raw.detach().clone()
    if hasattr(stage, "boundary_cell_lib") and stage.boundary_cell_lib is not None:
        b_lib = stage.boundary_cell_lib
        if hasattr(b_lib, "gm_raw"):
            snapshots["boundary_cell_lib.gm_raw"] = b_lib.gm_raw.detach().clone()
        if hasattr(b_lib, "isat_raw"):
            snapshots["boundary_cell_lib.isat_raw"] = b_lib.isat_raw.detach().clone()
    if hasattr(stage, "output_ode_cell_lib") and stage.output_ode_cell_lib is not None:
        ro_lib = stage.output_ode_cell_lib
        if hasattr(ro_lib, "gm_raw"):
            snapshots["output_ode_cell_lib.gm_raw"] = ro_lib.gm_raw.detach().clone()
        if hasattr(ro_lib, "isat_raw"):
            snapshots["output_ode_cell_lib.isat_raw"] = ro_lib.isat_raw.detach().clone()

    z_snapshot = stage.z_logits.detach().clone()
    bz_snapshot = (
        stage.boundary_z_logits.detach().clone()
        if stage.boundary_z_logits is not None else torch.empty(0)
    )
    raw_leak_snapshot = (
        stage.raw_leak.detach().clone()
        if hasattr(stage, "raw_leak") and stage.raw_leak is not None
        else torch.empty(0)
    )
    x_max_snapshot = float(stage.x_max)
    clip_snapshot = float(stage.clip_current)
    leak_mode_snapshot = str(stage.leak_mode)
    leak_constant_snapshot = (
        float(stage.leak_constant) if hasattr(stage, "leak_constant")
        else float("nan")
    )

    # Apply gm/isat fills so the cell library operates in the small-gm
    # regime (sigmoid(fill) -> ~0 -> gm ~ gm_min).  ``gm_raw_fill = -5``
    # is the conservative "near-zero sigmoid" point used by the E0
    # sweep; ``gm_raw_fill = -10`` collapses gm even closer to gm_min.
    with torch.no_grad():
        if hasattr(cell_lib, "gm_raw"):
            cell_lib.gm_raw.data.fill_(float(gm_raw_fill))
        if hasattr(cell_lib, "isat_raw"):
            cell_lib.isat_raw.data.fill_(float(isat_raw_fill))
        # Gates open (the install stays gate-independent for symmetry
        # with C1b; the G overlay below is the working map).
        stage.z_logits.data.fill_(12.0)
        if stage.boundary_z_logits is not None:
            stage.boundary_z_logits.data.fill_(12.0)
        # Boundary OTA default raws are kept; the G_in rescaling is
        # the user-facing drive knob.
    stage.x_max = float(target_x_max)
    stage.clip_current = float(clip_current)

    # Uniform leak.
    leak_raw_value = float(math.log(math.expm1(float(leak_value))))
    if stage.leak_mode != "programmable":
        stage.leak_mode = "programmable"
        if not hasattr(stage, "raw_leak") or stage.raw_leak is None:
            stage.raw_leak = nn.Parameter(torch.full((n_nodes,), leak_raw_value))
    leak_raw = torch.full((n_nodes,), leak_raw_value)
    with torch.no_grad():
        if stage.raw_leak.shape != leak_raw.shape:
            stage.raw_leak = nn.Parameter(leak_raw.clone())
        else:
            stage.raw_leak.data.copy_(leak_raw)

    # Resistive-shunt working point: the real ``rhs`` stays installed,
    # so the J-tuner (:func:`tune_c1b_real_to_j_band`) rescales the
    # core ``g_resistive_raw`` field itself (softplus-domain, i.e. the
    # "softplus-inverse G targets" of the c1b-protocol spec).  The
    # constructor default (softplus(-5) ~= 0.0067 per edge) is kept as
    # the neutral starting point; snapshot it for restore.
    if hasattr(cell_lib, "g_resistive_raw"):
        snapshots["cell_lib.g_resistive_raw"] = (
            cell_lib.g_resistive_raw.detach().clone()
        )
    else:
        raise ValueError(
            "C1b-real requires a cell library with g_resistive_raw "
            f"(got {type(cell_lib).__name__})"
        )

    return {
        "snapshots": snapshots,
        "z_snapshot": z_snapshot,
        "bz_snapshot": bz_snapshot,
        "raw_leak_snapshot": raw_leak_snapshot,
        "x_max_snapshot": x_max_snapshot,
        "clip_snapshot": clip_snapshot,
        "leak_mode_snapshot": leak_mode_snapshot,
        "leak_constant_snapshot": leak_constant_snapshot,
        "G_in": G_in.detach().clone(),
        "n_nodes": n_nodes,
        "boundary_drive_scale": float(boundary_drive_scale),
    }


def _resistive_G(cell_lib: nn.Module) -> torch.Tensor:
    """Current per-edge shunt conductance ``softplus(g_resistive_raw)``."""
    return torch.nn.functional.softplus(cell_lib.g_resistive_raw.detach())


def _set_resistive_G(cell_lib: nn.Module, G_new: torch.Tensor) -> None:
    """Write per-edge conductances via the softplus inverse.

    ``g_resistive_raw = log(exp(G) - 1)`` with guards so large/small
    targets cannot produce non-finite raws.
    """
    G_safe = G_new.detach().to(cell_lib.g_resistive_raw.device).clamp(1e-6, 20.0)
    raw = torch.log(torch.expm1(G_safe).clamp_min(1e-12))
    with torch.no_grad():
        cell_lib.g_resistive_raw.data.copy_(raw)


def restore_linear_reservoir_c1b_real(
    net: nn.Module, saved: dict,
) -> None:
    """Restore the fabric to its pre-C1b-real install state."""
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "C1b-real restore supports single-stage nets only"
        )
    stage = net.core.stages[0]
    cell_lib = stage.cell_lib
    snapshots = saved["snapshots"]
    for key, value in snapshots.items():
        lib_name, raw_name = key.split(".")
        lib = getattr(stage, lib_name, None)
        if lib is not None and hasattr(lib, raw_name):
            with torch.no_grad():
                getattr(lib, raw_name).data.copy_(value)
    with torch.no_grad():
        stage.z_logits.data.copy_(saved["z_snapshot"])
        if (
            stage.boundary_z_logits is not None
            and saved["bz_snapshot"].numel() > 0
            and stage.boundary_z_logits.shape == saved["bz_snapshot"].shape
        ):
            stage.boundary_z_logits.data.copy_(saved["bz_snapshot"])
        if (
            saved["raw_leak_snapshot"].numel() > 0
            and hasattr(stage, "raw_leak")
            and stage.raw_leak is not None
            and stage.raw_leak.shape == saved["raw_leak_snapshot"].shape
        ):
            stage.raw_leak.data.copy_(saved["raw_leak_snapshot"])
    stage.x_max = saved["x_max_snapshot"]
    stage.clip_current = saved["clip_snapshot"]
    stage.leak_mode = saved["leak_mode_snapshot"]
    if hasattr(stage, "leak_constant"):
        stage.leak_constant = saved["leak_constant_snapshot"]


def tune_c1b_real_to_j_band(
    net: nn.Module, u_seq: torch.Tensor, *,
    t_span: float, num_steps: int,
    target_low: float = C1B_SPECTRAL_TARGET_LOW,
    target_high: float = C1B_SPECTRAL_TARGET_HIGH,
    tol: float = C1B_TUNE_TOL,
    max_iters: int = C1B_TUNE_MAX_ITERS,
    washout: int = PROBE_WASHOUT,
    n_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    device: str = "cpu",
) -> tuple[float, list[dict]]:
    """Tune core ``g_resistive_raw`` until the spectrum is uniformly slow.

    Reuses :func:`measure_spectral_radius` against the REAL rhs.
    Uniformity gate: ``min|J| >= uniform_floor`` with ``max|J| <=
    stability_cap`` as guard.  Both violations scale ``G`` DOWN
    (halving on a non-finite Jacobian): lowering ``G`` pulls the fast
    Laplacian modes up toward the pinned leak mode, shrinking the
    spread.  ``target_low``/``target_high`` are the uniformity floor
    and stability cap (NOT the C1b max-band -- see the module
    constants).  Stops early at ``C1B_REAL_MIN_SCALE`` of the install
    ``G`` so the net can never decouple to zero.

    Returns ``(min_abs, history)`` -- the uniform (slowest-decaying
    non-leak-mode) edge, unlike the C1b tuner which returns max.
    """
    uniform_floor = float(target_low)
    stability_cap = float(target_high)
    stage = net.core.stages[0]
    cell_lib = stage.cell_lib
    history: list[dict] = []
    last_min = float("nan")
    if not hasattr(cell_lib, "g_resistive_raw"):
        return last_min, history
    G_install = _resistive_G(cell_lib)
    cumulative = 1.0
    for it in range(max_iters):
        spec = measure_spectral_radius(
            net, u_seq, t_span=t_span, num_steps=num_steps,
            washout=washout, n_samples=n_samples, device=device,
        )
        cur_max = spec["max_abs"]
        cur_min = spec["min_abs"]
        last_min = cur_min
        uniform = (
            math.isfinite(cur_min) and math.isfinite(cur_max)
            and cur_min >= uniform_floor - tol
            and cur_max <= stability_cap + tol
        )
        history.append({
            "iter": it,
            "max_abs": cur_max,
            "min_abs": cur_min,
            "mean_abs": spec["mean_abs"],
            "rank_proxy": spec["rank_proxy"],
            "n_transitions": spec["n_transitions"],
            "uniform": bool(uniform),
            "g_scale": cumulative,
        })
        if uniform:
            return cur_min, history
        if cumulative <= C1B_REAL_MIN_SCALE:
            return cur_min, history
        G_cur = _resistive_G(cell_lib)
        _set_resistive_G(cell_lib, G_cur * 0.5)
        cumulative *= 0.5
    return last_min, history


def c1b_real_resistive(
    *, order: int = 10, seed: int = 0, device: str = "cpu",
    hidden_dim: int = CANONICAL_HIDDEN,
    n_streams: int = 1, train_samples_per_stream: int = 300,
    washout: int = PROBE_WASHOUT, max_delay: int = 20,
    jacobian_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    t_span: float = CANONICAL_T_SPAN, num_steps: int = CANONICAL_NUM_STEPS,
    drive_scale: float = CANONICAL_DRIVE_SCALE,
    g_in_seed: int = 0, w_seed: int = 0,
    leak_value: float = C1B_LEAK_VALUE,
    boundary_drive_scale: float = 0.5,
    gm_raw_fill: float = -10.0, isat_raw_fill: float = -10.0,
    uniform_min: float = C1B_REAL_UNIFORM_MIN,
    stability_cap: float = C1B_REAL_STABILITY_CAP,
    matched_parity_tol: float = R0_MATCHED_PARITY_TOL,
) -> C1bRealReport:
    """Run the C1b-real resistive-core variant control.

    Spectral gate is uniformity (``min|J| >= uniform_min`` with
    ``max|J| <= stability_cap``), NOT the C1b max-band -- see the
    module constants for the audit finding.
    """
    if order != 10:
        raise ValueError("C1b-real is calibrated for NARMA-10 only")
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    u_scaled = ne._scale_drive(
        u_stream, bipolar=True, order=order, input_scale=drive_scale,
    )
    u_scaled = u_scaled.to(device)
    y_stream = y_stream.to(device)
    drive_rms = _drive_rms(u_scaled.to(torch.float32))

    net, _, _ = ne._build_fabric_net(
        order=order, seed=seed, freeze_read=False,
        t_span=t_span, num_steps=num_steps, cell_library="tanh_free",
        core_refresh_interval=0, leak_constant=None,
        compile_sequence=False, hidden_dim=hidden_dim,
    )
    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    saved = install_linear_reservoir_c1b_real(
        net,
        w_seed=w_seed, g_in_seed=g_in_seed,
        leak_value=float(leak_value),
        boundary_drive_scale=float(boundary_drive_scale),
        gm_raw_fill=float(gm_raw_fill),
        isat_raw_fill=float(isat_raw_fill),
        target_x_max=CANONICAL_X_MAX_LIN,
        clip_current=CANONICAL_CLIP_LIN,
        drive_rms_target=drive_rms, u_seq_for_rms=u_scaled,
    )
    try:
        uniform_edge, history = tune_c1b_real_to_j_band(
            net, u_scaled, t_span=t_span, num_steps=num_steps,
            target_low=uniform_min, target_high=stability_cap,
            tol=C1B_TUNE_TOL, max_iters=C1B_TUNE_MAX_ITERS,
            washout=washout, n_samples=jacobian_samples, device=device,
        )
        spec = measure_spectral_radius(
            net, u_scaled, t_span=t_span, num_steps=num_steps,
            washout=washout, n_samples=jacobian_samples, device=device,
        )
        states_full = spec["states_full"]
        jac_rows = spec["jacobian_rows"]
        abs_eigs: list[float] = []
        for eigs in spec["abs_eigs_per_transition"]:
            abs_eigs.extend([float(v) for v in eigs if math.isfinite(float(v))])
        instrument = _instrument_trajectory(
            stage=net.core.stages[0], states_full=states_full,
            u_seq=u_scaled, y_seq=y_stream, washout=washout,
            jacobian_samples=jacobian_samples,
            t_span=t_span, num_steps=num_steps,
        )
        mc_per, mc_total = _per_delay_mc(
            states_full, u_scaled, washout=washout, max_delay=max_delay,
            use_svd_fallback=True,
        )
        sat_max = float(states_full.abs().max().item())
        rail_frac = float(
            (states_full.abs() > 0.9 * CANONICAL_X_MAX_LIN).float().mean().item()
        )
        hidden = states_full[:, :hidden_dim].detach()
        X_w = hidden[washout:]
        y_w = y_stream[washout:]
        W_h = _ridge_fit_predict(X_w, y_w)
        X_aug = torch.cat(
            [X_w, torch.ones(X_w.shape[0], 1, device=X_w.device)], dim=1,
        )
        h_pred = X_aug @ W_h
        nrmse_hidden = float(ne.nrmse(h_pred, y_w))
        r2_hidden = float(ne.r2(h_pred, y_w))
        _, mc_total_hidden = _per_delay_mc(
            hidden, u_scaled, washout=washout, max_delay=max_delay,
            use_svd_fallback=True,
        )
        mc_total_hidden = float(mc_total_hidden)
        state_pr_hidden = _safe_participation_ratio(hidden[washout:])
        state_pr_std = _standardized_state_pr(states_full[washout:])
        raw = _raw_delay_ridge(
            u_scaled, y_stream, n_taps=R0_RAW_DELAY_TAPS, washout=washout,
        )
        matched_delta = float(instrument["ridge_nrmse"] - raw["nrmse"])
        pass_mc = bool(mc_total >= C1B_PASS_MC_ABOVE)
        pass_rail = bool(rail_frac <= C1B_REAL_RAIL_BELOW)
        # Uniformity gate (see module constants): slowest non-leak mode
        # above the floor and nothing above the stability cap.
        final_max = float(instrument["jac_max_abs"])
        final_min = float(instrument["jac_min_abs"])
        pass_spectral = bool(
            math.isfinite(uniform_edge)
            and uniform_edge >= uniform_min - C1B_TUNE_TOL
            and math.isfinite(final_max)
            and final_max <= stability_cap + C1B_TUNE_TOL
        )
        pass_all = bool(pass_mc and pass_rail and pass_spectral)
        row = InstrumentRow(
            config_tag=(
                f"c1b_real_seed{seed}_{device}_h{hidden_dim}"
                f"_tspan{t_span:g}_steps{num_steps}"
                f"_drive{drive_scale:g}_bds{boundary_drive_scale:g}"
                f"_washout{washout}"
                f"_ws{w_seed}_gi{g_in_seed}_leak{leak_value:g}"
                f"_gm{gm_raw_fill:g}_is{isat_raw_fill:g}"
                f"_uniform{uniform_min:g}_cap{stability_cap:g}"
            ),
            nrmse=float(instrument["ridge_nrmse"]),
            r2=float(instrument["ridge_r2"]),
            mc_total=float(mc_total),
            mc_per_delay=[float(v) for v in mc_per],
            state_pr=float(instrument["state_pr"]),
            jac_max_abs=float(instrument["jac_max_abs"]),
            jac_min_abs=float(instrument["jac_min_abs"]),
            jac_mean_abs=float(instrument["jac_mean_abs"]),
            jac_rank_proxy=float(instrument["jac_rank_proxy"]),
            jac_eig_abs=abs_eigs,
            sat_max_ratio=float(sat_max / CANONICAL_X_MAX_LIN) if CANONICAL_X_MAX_LIN > 0 else float("nan"),
            rail_frac=rail_frac,
            n_params=n_params,
            note="C1b-real resistive-core linear reservoir",
            state_pr_standardized=float(state_pr_std),
        )
        return C1bRealReport(
            order=order, seed=seed, device=device, hidden_dim=hidden_dim,
            n_streams=n_streams, train_samples_per_stream=train_samples_per_stream,
            washout=washout, n_params=n_params,
            n_tuning_iterations=len(history), tuning_history=history,
            instrument=row,
            matched_raw_delay_nrmse=float(raw["nrmse"]),
            matched_parity_tolerance=float(matched_parity_tol),
            matched_parity_delta=matched_delta,
            gm_raw_fill=float(gm_raw_fill),
            isat_raw_fill=float(isat_raw_fill),
            boundary_drive_scale=float(boundary_drive_scale),
            leak_value=float(leak_value),
            spectral_target_low=float(uniform_min),
            spectral_target_high=float(stability_cap),
            state_pr_raw=float(state_pr_hidden),
            state_pr_standardized=float(state_pr_std),
            pass_mc=pass_mc, pass_rail=pass_rail,
            pass_spectral=pass_spectral, pass_all=pass_all,
            note=(
                "C1b-real PASS gates: MC>=5, rails<=5%, spectrum "
                "uniform (min|J|>=floor, max|J|<=cap). The C1b max-band "
                "[0.97,0.99] is unreachable for a Laplacian (max pinned "
                "at the leak mode ~0.945); max vs that band is diagnostic."
            ),
        )
    finally:
        restore_linear_reservoir_c1b_real(net, saved)


# ---------------------------------------------------------------------------
# C3 tanh-crossover sweep (c1b-revision amendment)
# ---------------------------------------------------------------------------


def c3_gm_grid_sweep(
    *, c1b_sidecar_dir: Path, gm_grid: tuple[float, ...] = C3_GM_GRID,
    order: int = 10, seed: int = 0, device: str = "cpu",
    hidden_dim: int = CANONICAL_HIDDEN,
    n_streams: int = 1, train_samples_per_stream: int = 300,
    washout: int = PROBE_WASHOUT, max_delay: int = 20,
    jacobian_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    t_span: float = CANONICAL_T_SPAN, num_steps: int = CANONICAL_NUM_STEPS,
    drive_scale: float = CANONICAL_DRIVE_SCALE,
) -> C3SweepReport:
    """Run the C3 tanh-crossover sweep on a passing C1b base.

    For each ``gm_raw_fill`` in ``gm_grid``, install the tuned C1b base
    (from the sidecar pair) and overwrite the core edge ``gm_raw`` and
    ``isat_raw`` with that fill; the boundary-OTA cell library is left
    alone.  The crossover is the FIRST ``gm_raw_fill`` where the core
    tanh arrives before MC collapses (MC <= ``C3_MC_COLLAPSE_RATIO`` *
    ``c1b_mc``); the operating point on the real fabric init is then
    chosen below that fill.
    """
    base = load_tuned_base(c1b_sidecar_dir)
    W = base["W"]
    G_in = base["G_in"]
    meta = base["meta"]
    leak_value = float(meta["leak_value"])
    c1b_metrics = meta.get("c1b_metrics", {})
    c1b_mc = float(c1b_metrics.get("mc_total", float("nan")))

    u_raw, y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    u_scaled = ne._scale_drive(
        u_stream, bipolar=True, order=order, input_scale=drive_scale,
    )
    u_scaled = u_scaled.to(device)
    y_stream = y_stream.to(device)
    drive_rms = _drive_rms(u_scaled.to(torch.float32))

    rows: list[C3SweepRow] = []
    crossover: float | None = None
    for gm_fill in gm_grid:
        net, _, _ = ne._build_fabric_net(
            order=order, seed=seed, freeze_read=False,
            t_span=t_span, num_steps=num_steps, cell_library="tanh_free",
            core_refresh_interval=0, leak_constant=None,
            compile_sequence=False, hidden_dim=hidden_dim,
        )
        saved = reinstall_tuned_c1b(
            net, W=W, G_in=G_in, leak_value=leak_value,
            drive_rms_target=drive_rms, u_seq_for_rms=u_scaled,
        )
        try:
            stage = net.core.stages[0]
            cell_lib = stage.cell_lib
            # Snapshot and overwrite gm_raw/isat_raw.  The C1b rhs only
            # routes the core through the cell library when the restore
            # flag is set -- without it the fills would be dead (the
            # linear W@x path ignores gm entirely).
            gm_snap = cell_lib.gm_raw.detach().clone()
            isat_snap = cell_lib.isat_raw.detach().clone()
            with torch.no_grad():
                cell_lib.gm_raw.data.fill_(float(gm_fill))
                cell_lib.isat_raw.data.fill_(float(gm_fill))
            stage._lin_restore_tanh = True
            # Measure states and run instruments.
            spec = measure_spectral_radius(
                net, u_scaled, t_span=t_span, num_steps=num_steps,
                washout=washout, n_samples=jacobian_samples, device=device,
            )
            states_full = spec["states_full"]
            jac_rows = spec["jacobian_rows"]
            instrument = _instrument_trajectory(
                stage=stage, states_full=states_full,
                u_seq=u_scaled, y_seq=y_stream, washout=washout,
                jacobian_samples=jacobian_samples,
                t_span=t_span, num_steps=num_steps,
            )
            mc_per, mc_total = _per_delay_mc(
                states_full, u_scaled, washout=washout, max_delay=max_delay,
                use_svd_fallback=True,
            )
            sat_max = float(states_full.abs().max().item())
            rail_frac = float(
                (states_full.abs() > 0.9 * CANONICAL_X_MAX_LIN).float().mean().item()
            )
            state_pr_raw = float(instrument["state_pr"])
            state_pr_std = _standardized_state_pr(states_full[washout:])
            jac_max = float(
                max(r["max_abs"] for r in jac_rows)
                if jac_rows else float("nan")
            )
            collapse = (
                math.isfinite(c1b_mc) and c1b_mc > 0
                and mc_total <= C3_MC_COLLAPSE_RATIO * c1b_mc
            )
            row = C3SweepRow(
                config_tag=(
                    f"c3_sweep_seed{seed}_{device}_h{hidden_dim}"
                    f"_drive{drive_scale:g}_gm{gm_fill:g}"
                ),
                gm_raw_fill=float(gm_fill),
                mc_total=float(mc_total),
                ridge_nrmse=float(instrument["ridge_nrmse"]),
                ridge_r2=float(instrument["ridge_r2"]),
                state_pr_raw=state_pr_raw,
                state_pr_standardized=float(state_pr_std),
                jac_max_abs=jac_max,
                rail_frac=float(rail_frac),
                sat_max_ratio=float(sat_max / CANONICAL_X_MAX_LIN) if CANONICAL_X_MAX_LIN > 0 else float("nan"),
                collapsed=bool(collapse),
                note=(
                    "C3 sweep row on C1b base; tanh crossover detector."
                ),
            )
            rows.append(row)
            if crossover is None and collapse:
                crossover = float(gm_fill)
        finally:
            # Restore gm/isat (the reinstall snapshot already restores
            # everything else on restore_linear_reservoir_v2).
            with torch.no_grad():
                cell_lib.gm_raw.data.copy_(gm_snap)
                cell_lib.isat_raw.data.copy_(isat_snap)
            restore_linear_reservoir_v2(net, saved)
    return C3SweepReport(
        order=order, seed=seed, device=device, hidden_dim=hidden_dim,
        c1b_path=str(c1b_sidecar_dir),
        c1b_mc_total=c1b_mc,
        gm_grid=[float(g) for g in gm_grid],
        rows=rows,
        crossover_gm_raw_fill=crossover,
        note=(
            "C3 tanh-crossover sweep on the persisted C1b base. "
            "Crossover is the first gm_raw_fill where MC collapses."
        ),
    )


# ---------------------------------------------------------------------------
# R0 reconciliation matrix
# ---------------------------------------------------------------------------


def _build_canonical_fabric(
    *, order: int, seed: int, refresh: int, freeze_read: bool,
    hidden_dim: int, t_span: float, num_steps: int, cell_library: str,
) -> nn.Module:
    net, ts, ns = ne._build_fabric_net(
        order=order, seed=seed, freeze_read=freeze_read,
        t_span=t_span, num_steps=num_steps, cell_library=cell_library,
        core_refresh_interval=refresh, leak_constant=None,
        compile_sequence=False, hidden_dim=hidden_dim,
    )
    return net


def _e0_ridge_on_state(
    states_full: torch.Tensor, y_seq: torch.Tensor, *,
    washout: int, l2: float = 1e-2,
) -> dict[str, float]:
    y_seq = y_seq.to(states_full.device)
    X_w = states_full[washout:]
    y_w = y_seq[washout:]
    if X_w.shape[0] == 0:
        return {"nrmse": float("nan"), "r2": float("nan"), "n_features": int(states_full.shape[1])}
    W = _ridge_fit_predict(X_w, y_w, l2=l2)
    X_aug = torch.cat([X_w, torch.ones(X_w.shape[0], 1, device=X_w.device)], dim=1)
    pred = X_aug @ W
    return {
        "nrmse": ne.nrmse(pred, y_w),
        "r2": ne.r2(pred, y_w),
        "n_features": int(X_w.shape[1]),
    }


def _legacy_hidden_only_ridge(
    net: nn.Module, u_seq: torch.Tensor, y_seq: torch.Tensor, *,
    t_span: float, num_steps: int, washout: int = PROBE_WASHOUT,
    device: str = "cpu", l2: float = 1e-2,
) -> dict[str, float]:
    states_full, hidden = _fabric_full_state_collect(
        net, u_seq, t_span=t_span, num_steps=num_steps, device=device,
    )
    y_seq = y_seq.to(hidden.device)
    X_w = hidden[washout:]
    y_w = y_seq[washout:]
    if X_w.shape[0] == 0:
        return {"nrmse": float("nan"), "r2": float("nan"), "n_features": int(hidden.shape[1])}
    W = _ridge_fit_predict(X_w, y_w, l2=l2)
    X_aug = torch.cat([X_w, torch.ones(X_w.shape[0], 1, device=X_w.device)], dim=1)
    pred = X_aug @ W
    return {
        "nrmse": ne.nrmse(pred, y_w),
        "r2": ne.r2(pred, y_w),
        "n_features": int(hidden.shape[1]),
    }


def r0_reconciliation(
    *, order: int = 10, seed: int = 0, device: str = "cpu",
    n_streams: int = 1, train_samples_per_stream: int = 300,
    washout: int = PROBE_WASHOUT, hidden_dim: int = 25,
    cell_library: str = "tanh_free",
    t_span: float = CANONICAL_T_SPAN, num_steps: int = CANONICAL_NUM_STEPS,
) -> R0ReconciliationReport:
    """Run the R0 reconciliation matrix on one fixed seed and stream."""
    if order != 10:
        raise ValueError("R0 reconciliation is calibrated for NARMA-10 only")
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    drive = ne._scale_drive(
        u_stream, bipolar=True, order=order, input_scale=CANONICAL_DRIVE_SCALE,
    )
    drive = drive.to(device)
    y_stream = y_stream.to(device)

    rows: list[RidgeRow] = []
    t0 = time.time()

    base_kwargs = {
        "order": order, "seed": seed,
        "hidden_dim": hidden_dim, "t_span": t_span, "num_steps": num_steps,
        "cell_library": cell_library,
    }

    # Factor 1: E0 full-state Ridge (canonical) vs legacy hidden-only.
    net_e0 = _build_canonical_fabric(
        order=order, seed=seed, refresh=0, freeze_read=False,
        hidden_dim=hidden_dim, t_span=t_span, num_steps=num_steps,
        cell_library=cell_library,
    )
    states_full, _ = _fabric_full_state_collect(
        net_e0, drive, t_span=t_span, num_steps=num_steps, device=device,
    )
    full = _e0_ridge_on_state(states_full, y_stream, washout=washout)
    rows.append(RidgeRow(
        config_tag="r0_e0_full_state",
        diagnostic="e0_full_state",
        nrmse=full["nrmse"], r2=full["r2"], n_features=full["n_features"],
        drive_scale=CANONICAL_DRIVE_SCALE, stream_length=int(drive.shape[0]),
        refresh="k0", note="E0 full-state Ridge, identical net & data",
    ))
    legacy = _legacy_hidden_only_ridge(
        net_e0, drive, y_stream,
        t_span=t_span, num_steps=num_steps, washout=washout, device=device,
    )
    rows.append(RidgeRow(
        config_tag="r0_legacy_hidden_only",
        diagnostic="legacy_hidden_only",
        nrmse=legacy["nrmse"], r2=legacy["r2"], n_features=legacy["n_features"],
        drive_scale=CANONICAL_DRIVE_SCALE, stream_length=int(drive.shape[0]),
        refresh="k0",
        note="legacy hidden-only Ridge, same net & data as full-state",
    ))

    # Factor 2: k8 frozen vs fully dynamic core.
    for refresh, label in [(8, "k8_frozen"), (0, "k0_dynamic")]:
        net = _build_canonical_fabric(
            order=order, seed=seed, refresh=refresh, freeze_read=(refresh > 0),
            hidden_dim=hidden_dim, t_span=t_span, num_steps=num_steps,
            cell_library=cell_library,
        )
        states, _ = _fabric_full_state_collect(
            net, drive, t_span=t_span, num_steps=num_steps, device=device,
        )
        out = _e0_ridge_on_state(states, y_stream, washout=washout)
        rows.append(RidgeRow(
            config_tag=f"r0_refresh_{label}",
            diagnostic="e0_full_state",
            nrmse=out["nrmse"], r2=out["r2"], n_features=out["n_features"],
            drive_scale=CANONICAL_DRIVE_SCALE, stream_length=int(drive.shape[0]),
            refresh=label, note=f"core_refresh_interval={refresh}",
        ))

    # Factor 3: drive scale 1.0 vs 0.5.
    for drive_scale in (1.0, 0.5):
        net = _build_canonical_fabric(
            order=order, seed=seed, refresh=0, freeze_read=False,
            hidden_dim=hidden_dim, t_span=t_span, num_steps=num_steps,
            cell_library=cell_library,
        )
        u_scaled = ne._scale_drive(
            u_stream, bipolar=True, order=order, input_scale=drive_scale,
        ).to(device)
        states, _ = _fabric_full_state_collect(
            net, u_scaled, t_span=t_span, num_steps=num_steps, device=device,
        )
        out = _e0_ridge_on_state(states, y_stream, washout=washout)
        rows.append(RidgeRow(
            config_tag=f"r0_drive_{drive_scale:g}",
            diagnostic="e0_full_state",
            nrmse=out["nrmse"], r2=out["r2"], n_features=out["n_features"],
            drive_scale=drive_scale, stream_length=int(u_scaled.shape[0]),
            refresh="k0",
            note=f"input_scale={drive_scale}",
        ))

    # Factor 4: stream length 300 vs 10000 (fixed washout).
    long_u_raw, long_y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=1, n=10000,
    )
    long_u = ne._scale_drive(
        long_u_raw[0], bipolar=True, order=order, input_scale=CANONICAL_DRIVE_SCALE,
    ).to(device)
    long_y = long_y_raw[0].to(device)
    net = _build_canonical_fabric(
        order=order, seed=seed, refresh=0, freeze_read=False,
        hidden_dim=hidden_dim, t_span=t_span, num_steps=num_steps,
        cell_library=cell_library,
    )
    long_states, _ = _fabric_full_state_collect(
        net, long_u, t_span=t_span, num_steps=num_steps, device=device,
    )
    long_out = _e0_ridge_on_state(long_states, long_y, washout=washout)
    rows.append(RidgeRow(
        config_tag="r0_stream_long",
        diagnostic="e0_full_state",
        nrmse=long_out["nrmse"], r2=long_out["r2"], n_features=long_out["n_features"],
        drive_scale=CANONICAL_DRIVE_SCALE, stream_length=int(long_u.shape[0]),
        refresh="k0", note="stream length 10000 at fixed washout",
    ))

    # Factor 5: raw-delay Ridge on the E0 stream.
    raw = _raw_delay_ridge(
        drive, y_stream, n_taps=R0_RAW_DELAY_TAPS, washout=washout,
    )
    rows.append(RidgeRow(
        config_tag="r0_raw_delay",
        diagnostic="raw_delay",
        nrmse=raw["nrmse"], r2=raw["r2"], n_features=raw["n_features"],
        drive_scale=CANONICAL_DRIVE_SCALE, stream_length=int(drive.shape[0]),
        refresh="k0",
        note=f"{R0_RAW_DELAY_TAPS}-tap raw-delay Ridge, identical stream & washout",
    ))

    raw_nrmse = raw["nrmse"]
    e0_nrmse = full["nrmse"]
    parity_delta = float(e0_nrmse - raw_nrmse)
    matched = {
        "rule": "E0_Ridge <= raw_delay_Ridge + 0.03",
        "tolerance": R0_MATCHED_PARITY_TOL,
        "raw_delay_nrmse": float(raw_nrmse),
        "e0_full_state_nrmse": float(e0_nrmse),
        "delta_e0_minus_raw": parity_delta,
        "rule_holds": bool(parity_delta <= R0_MATCHED_PARITY_TOL),
    }

    return R0ReconciliationReport(
        order=order, seed=seed, device=device,
        n_streams=n_streams, train_samples_per_stream=train_samples_per_stream,
        washout=washout,
        canonical_net_kwargs=base_kwargs,
        rows=rows,
        matched_parity_rule=matched,
        n_corners=len(rows),
        elapsed_s=float(time.time() - t0),
        note="R0 reconciliation matrix (5 factors); see matched_parity_rule.",
    )


# ---------------------------------------------------------------------------
# C0 ESN calibration control
# ---------------------------------------------------------------------------


def _esn_collect_states(
    esn: ne.ESN, u_seq: torch.Tensor, *, device: str = "cpu",
) -> torch.Tensor:
    """Run the ESN over a sequence and return ``(T, n_reservoir)`` states.

    The ESN weights live on CPU (no ``.to()`` protocol), so the
    recurrence always runs on CPU and the result is moved to ``device``
    afterwards.
    """
    states = esn._run(u_seq.to("cpu"))
    return states.detach().to(device)


def _esn_jacobian_eigs(
    esn: ne.ESN, esn_states: torch.Tensor, u_seq: torch.Tensor, *,
    washout: int, n_samples: int, max_eigs: int = 1024,
) -> tuple[list[dict[str, float]], list[list[float]]]:
    """Analytic Jacobian eigenvalues for the ESN at observed transitions.

    The ESN map is ``x_{t+1} = (1-leak)*x_t + leak*tanh(W*x_t + W_in*u)``,
    so ``J = (1-leak)*I + leak*diag(sech^2(pre))*W`` with
    ``pre = W*x[t] + W_in*u[t+1]``.  ESN tanh states live in [-1, 1];
    rail fraction is reported as N/A.
    """
    if esn_states.shape[0] < washout + 2:
        raise ValueError(
            f"need at least washout+2 states, got {esn_states.shape[0]}"
        )
    if u_seq.shape[0] != esn_states.shape[0]:
        raise ValueError(
            "ESN states and inputs must have the same length, got "
            f"{esn_states.shape[0]} and {u_seq.shape[0]}"
        )
    n_nodes = esn_states.shape[1]
    n = min(n_samples, esn_states.shape[0] - washout - 1)
    indices = torch.linspace(
        washout, esn_states.shape[0] - 2, steps=n,
    ).round().to(torch.long).unique().tolist()
    dev = esn_states.device
    W = esn.W.detach().to(device=dev, dtype=torch.float64)
    W_in = esn.W_in.detach().to(device=dev, dtype=torch.float64).squeeze(-1)
    leak = float(esn.leak)
    rows: list[dict[str, float]] = []
    eig_per_trans: list[list[float]] = []
    for t in indices:
        x = esn_states[t].detach().to(dtype=torch.float64)
        u_next = float(u_seq[t + 1].item())
        pre = W @ x + W_in * u_next
        gain = 1.0 - torch.tanh(pre).pow(2)
        J = (1.0 - leak) * torch.eye(
            n_nodes, dtype=torch.float64, device=dev
        ) + leak * (gain.unsqueeze(1) * W)
        try:
            eig = torch.linalg.eigvals(J)
            abs_e = eig.abs()
            if not torch.isfinite(abs_e).all():
                raise RuntimeError("non-finite ESN eigenvalues")
        except Exception:
            abs_e = torch.tensor([], dtype=torch.float64)
        if abs_e.numel() == 0:
            rows.append({
                "transition_index": float(t),
                "state_dim": float(n_nodes),
                "max_abs": float("nan"),
                "min_abs": float("nan"),
                "mean_abs": float("nan"),
                "rank_proxy": float("nan"),
            })
            eig_per_trans.append([])
            continue
        abs_e_truncated = abs_e[:max_eigs]
        rows.append({
            "transition_index": float(t),
            "state_dim": float(n_nodes),
            "max_abs": float(abs_e.max().item()),
            "min_abs": float(abs_e.min().item()),
            "mean_abs": float(abs_e.mean().item()),
            "rank_proxy": float(npr.participation_ratio(abs_e.to(torch.float32))),
        })
        eig_per_trans.append([float(v) for v in abs_e_truncated.tolist()])
    return rows, eig_per_trans


def c0_esn_calibration(
    *, order: int = 10, seed: int = 0, n_reservoir: int = 25,
    spectral_radius: float = 0.9, input_scaling: float = 1.0,
    leak: float = 1.0, ridge_l2: float = 1e-2,
    n_streams: int = 1, train_samples_per_stream: int = 300,
    washout: int = PROBE_WASHOUT, max_delay: int = 20,
    jacobian_samples: int = 3, device: str = "cpu",
) -> tuple[InstrumentRow, dict]:
    """Feed the ESN hidden states through the identical instrument set."""
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0].to(device)
    y_stream = y_raw[0].to(device)
    esn = ne.ESN(
        n_reservoir=n_reservoir, spectral_radius=spectral_radius,
        input_scaling=input_scaling, leak=leak,
        ridge_l2=ridge_l2, seed=seed,
    )
    states = _esn_collect_states(esn, u_stream, device=device)
    mc_per, mc_total = _per_delay_mc(
        states, u_stream, washout=washout, max_delay=max_delay, ridge_l2=ridge_l2,
    )
    X = states[washout:]
    y = y_stream[washout:]
    W = _ridge_fit_predict(X, y, l2=ridge_l2)
    pred = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1) @ W
    jac_rows, eig_per_trans = _esn_jacobian_eigs(
        esn, states, u_stream, washout=washout, n_samples=jacobian_samples,
    )
    abs_eigs = _flatten_abs_eigs(
        [{"eig_abs": eigs} for eigs in eig_per_trans]
    )
    pr = float(npr.participation_ratio(states[washout:]))
    state_pr = pr
    note = (
        "ESN states live in [-1, 1]; rail fraction reported as N/A per "
        "linear-control-probes spec."
    )
    row = InstrumentRow(
        config_tag=(
            f"c0_esn_seed{seed}_{device}_h{n_reservoir}"
            f"_sr{spectral_radius:g}_is{input_scaling:g}"
            f"_leak{leak:g}_washout{washout}"
        ),
        nrmse=float(ne.nrmse(pred, y)),
        r2=float(ne.r2(pred, y)),
        mc_total=float(mc_total),
        mc_per_delay=[float(v) for v in mc_per],
        state_pr=state_pr,
        jac_max_abs=float(max(r["max_abs"] for r in jac_rows)),
        jac_min_abs=float(min(r["min_abs"] for r in jac_rows)),
        jac_mean_abs=float(
            sum(r["mean_abs"] for r in jac_rows) / max(len(jac_rows), 1)
        ),
        jac_rank_proxy=float(max(r["rank_proxy"] for r in jac_rows)),
        jac_eig_abs=abs_eigs,
        sat_max_ratio=float("nan"),
        rail_frac=float("nan"),
        n_params=int(n_reservoir * n_reservoir + n_reservoir + n_reservoir + 1),
        note=note,
    )
    return row, {
        "n_reservoir": n_reservoir,
        "spectral_radius": spectral_radius,
        "input_scaling": input_scaling,
        "leak": leak,
        "washout": washout,
        "n_per_delay": len(mc_per),
        "eig_per_transition_n": len(eig_per_trans),
        "abs_eigs_count": len(abs_eigs),
        "esn_states_shape": list(states.shape),
    }


# ---------------------------------------------------------------------------
# C1 linear-reservoir-in-fabric control
# ---------------------------------------------------------------------------


def c1_linear_reservoir(
    *, order: int = 10, seed: int = 0, device: str = "cpu",
    hidden_dim: int = CANONICAL_HIDDEN,
    n_streams: int = 1, train_samples_per_stream: int = 300,
    washout: int = PROBE_WASHOUT, max_delay: int = 20,
    jacobian_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    t_span: float = CANONICAL_T_SPAN, num_steps: int = CANONICAL_NUM_STEPS,
    drive_scale: float = CANONICAL_DRIVE_SCALE,
    target_x_max: float = CANONICAL_X_MAX_LIN,
    clip_current: float = CANONICAL_CLIP_LIN,
    g_edge_seed: int = 0, g_in_seed: int = 0, leak_seed: int = 0,
    matched_parity_tol: float = R0_MATCHED_PARITY_TOL,
) -> LinearReservoirReport:
    """Run the C1 linear-reservoir-in-fabric control."""
    if order != 10:
        raise ValueError("C1 is calibrated for NARMA-10 only")
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    u_scaled = ne._scale_drive(
        u_stream, bipolar=True, order=order, input_scale=drive_scale,
    )
    # Normalize stream devices up front: helpers move what they touch,
    # but keeping one device throughout avoids any CPU/CUDA mixing.
    u_scaled = u_scaled.to(device)
    y_stream = y_stream.to(device)
    drive_rms = _drive_rms(u_scaled.to(torch.float32))

    net, ts, ns = ne._build_fabric_net(
        order=order, seed=seed, freeze_read=False,
        t_span=t_span, num_steps=num_steps, cell_library="tanh_free",
        core_refresh_interval=0, leak_constant=None,
        compile_sequence=False, hidden_dim=hidden_dim,
    )
    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    saved = install_linear_reservoir_v2(
        net,
        G_edge_seed=g_edge_seed, G_in_seed=g_in_seed, leak_seed=leak_seed,
        target_x_max=target_x_max, clip_current=clip_current,
        drive_rms_target=drive_rms, u_seq_for_rms=u_scaled,
    )
    try:
        radius, history = tune_to_target_radius(
            net, u_scaled, t_span=t_span, num_steps=num_steps,
            washout=washout, n_samples=jacobian_samples, device=device,
        )
        spec = measure_spectral_radius(
            net, u_scaled, t_span=t_span, num_steps=num_steps,
            washout=washout, n_samples=jacobian_samples, device=device,
        )
        states_full = spec["states_full"]
        jac_rows = spec["jacobian_rows"]
        # Per-transition eigenvalue lists.
        abs_eigs: list[float] = []
        for eigs in spec["abs_eigs_per_transition"]:
            abs_eigs.extend([float(v) for v in eigs if math.isfinite(float(v))])
        # MC, Ridge, PR, rail, sat.
        instrument = _instrument_trajectory(
            stage=net.core.stages[0], states_full=states_full,
            u_seq=u_scaled, y_seq=y_stream, washout=washout,
            jacobian_samples=jacobian_samples, t_span=t_span, num_steps=num_steps,
        )
        mc_per, mc_total = _per_delay_mc(
            states_full, u_scaled, washout=washout, max_delay=max_delay,
            use_svd_fallback=True,
        )
        sat_max = float(states_full.abs().max().item())
        rail_frac = float(
            (states_full.abs() > 0.9 * target_x_max).float().mean().item()
        )
        # Hidden-only Ridge/MC/PR (for the second reporting band).
        hidden = states_full[:, :hidden_dim].detach()
        X_w = hidden[washout:]
        y_w = y_stream[washout:]
        W = _ridge_fit_predict(X_w, y_w)
        X_aug = torch.cat(
            [X_w, torch.ones(X_w.shape[0], 1, device=X_w.device)], dim=1,
        )
        h_pred = X_aug @ W
        nrmse_hidden = float(ne.nrmse(h_pred, y_w))
        r2_hidden = float(ne.r2(h_pred, y_w))
        _, mc_total_hidden = _per_delay_mc(
            hidden, u_scaled, washout=washout, max_delay=max_delay,
            use_svd_fallback=True,
        )
        mc_total_hidden = float(mc_total_hidden)
        state_pr_hidden = _safe_participation_ratio(hidden[washout:])
        state_pr_standardized = _standardized_state_pr(states_full[washout:])
        # Matched parity: compare against raw-delay Ridge on the same stream.
        raw = _raw_delay_ridge(
            u_scaled, y_stream, n_taps=R0_RAW_DELAY_TAPS, washout=washout,
        )
        matched_delta = float(instrument["ridge_nrmse"] - raw["nrmse"])
        pass_mc = bool(mc_total > C1_PASS_MC_ABOVE)
        # Parity is diagnostic only for linear controls (c1b-revision),
        # but C1 itself still has it as a pass gate so the existing audit
        # invariants stay green until C1 is fully retired by C1b on the
        # canonical runs.
        pass_pr = bool(state_pr_hidden >= C1_PASS_PR_MIN)
        pass_ridge = bool(matched_delta <= matched_parity_tol)
        pass_all = bool(pass_mc and pass_pr and pass_ridge)
        row = InstrumentRow(
            config_tag=(
                f"c1_linear_seed{seed}_{device}_h{hidden_dim}"
                f"_tspan{t_span:g}_steps{num_steps}"
                f"_drive{drive_scale:g}_washout{washout}"
                f"_ge{g_edge_seed}_gi{g_in_seed}_lk{leak_seed}"
                f"_target[{C1_SPECTRAL_TARGET_LOW:g},{C1_SPECTRAL_TARGET_HIGH:g}]"
            ),
            nrmse=float(instrument["ridge_nrmse"]),
            r2=float(instrument["ridge_r2"]),
            mc_total=float(mc_total),
            mc_per_delay=[float(v) for v in mc_per],
            state_pr=float(instrument["state_pr"]),
            jac_max_abs=float(instrument["jac_max_abs"]),
            jac_min_abs=float(instrument["jac_min_abs"]),
            jac_mean_abs=float(instrument["jac_mean_abs"]),
            jac_rank_proxy=float(instrument["jac_rank_proxy"]),
            jac_eig_abs=abs_eigs,
            sat_max_ratio=float(sat_max / target_x_max) if target_x_max > 0 else float("nan"),
            rail_frac=rail_frac,
            n_params=n_params,
            note="C1 linear-reservoir-in-fabric control",
            state_pr_standardized=float(state_pr_standardized),
        )
        return LinearReservoirReport(
            order=order, seed=seed, device=device, hidden_dim=hidden_dim,
            n_streams=n_streams, train_samples_per_stream=train_samples_per_stream,
            washout=washout, n_params=n_params,
            canonical_target_radius_low=C1_SPECTRAL_TARGET_LOW,
            canonical_target_radius_high=C1_SPECTRAL_TARGET_HIGH,
            n_tuning_iterations=len(history),
            tuning_history=history,
            instrument=row,
            matched_raw_delay_nrmse=float(raw["nrmse"]),
            matched_parity_tolerance=float(matched_parity_tol),
            matched_parity_delta=matched_delta,
            pass_mc=pass_mc, pass_pr=pass_pr, pass_ridge_matched=pass_ridge,
            pass_all=pass_all,
            nrmse_hidden=nrmse_hidden,
            r2_hidden=r2_hidden,
            mc_total_hidden=mc_total_hidden,
            state_pr_hidden=state_pr_hidden,
            note=(
                "C1 PASS criteria: MC>3, PR>=6, Ridge within "
                "matched-parity tolerance."
            ),
        )
    finally:
        restore_linear_reservoir_v2(net, saved)


# ---------------------------------------------------------------------------
# C2/C3/C4 bisection
# ---------------------------------------------------------------------------


def c2_c3_c4_bisection(
    c1_report: LinearReservoirReport, *,
    order: int = 10, seed: int = 0, device: str = "cpu",
    hidden_dim: int = CANONICAL_HIDDEN,
    n_streams: int = 1, train_samples_per_stream: int = 300,
    washout: int = PROBE_WASHOUT, max_delay: int = 20,
    jacobian_samples: int = C1_SPECTRAL_TRANSITION_MIN,
    t_span: float = CANONICAL_T_SPAN, num_steps: int = CANONICAL_NUM_STEPS,
    drive_scale: float = CANONICAL_DRIVE_SCALE,
    matched_parity_tol: float = R0_MATCHED_PARITY_TOL,
    c3_gm_init: float = 0.0,
) -> BisectionReport:
    """Run the C2/C3/C4 single-element bisection onto a passing C1."""
    if not c1_report.pass_all:
        return BisectionReport(
            order=order, seed=seed, device=device, hidden_dim=hidden_dim,
            n_streams=n_streams, train_samples_per_stream=train_samples_per_stream,
            washout=washout, c1_instrument=c1_report.instrument,
            legs=[], deltas_vs_c1=[],
            suppressor="c1-not-pass",
            note="C2/C3/C4 skipped because C1 did not pass.",
        )
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    u_scaled = ne._scale_drive(
        u_stream, bipolar=True, order=order, input_scale=drive_scale,
    )
    u_scaled = u_scaled.to(device)
    y_stream = y_stream.to(device)
    drive_rms = _drive_rms(u_scaled.to(torch.float32))

    legs: list[InstrumentRow] = []
    deltas: list[dict] = []

    # C2: restore boundary-OTA input path (full cell-lib boundary), keep
    # core linear, rails far.
    leg_c2 = _bisect_leg(
        order=order, seed=seed, device=device,
        hidden_dim=hidden_dim, t_span=t_span, num_steps=num_steps,
        drive_scale=drive_scale, u_scaled=u_scaled, y_stream=y_stream,
        washout=washout, jacobian_samples=jacobian_samples,
        max_delay=max_delay,
        restore_boundary=True, restore_tanh_core=False, restore_compliance=False,
        drive_rms=drive_rms, matched_parity_tol=matched_parity_tol,
        leg_tag="c2_boundary_input_restored",
    )
    legs.append(leg_c2)
    deltas.append(_delta_vs_c1(c1_report.instrument, leg_c2, "C2"))

    # C3: restore tanh_free core, keep boundary linear injection, no clip.
    leg_c3 = _bisect_leg(
        order=order, seed=seed, device=device,
        hidden_dim=hidden_dim, t_span=t_span, num_steps=num_steps,
        drive_scale=drive_scale, u_scaled=u_scaled, y_stream=y_stream,
        washout=washout, jacobian_samples=jacobian_samples,
        max_delay=max_delay,
        restore_boundary=False, restore_tanh_core=True, restore_compliance=False,
        drive_rms=drive_rms, matched_parity_tol=matched_parity_tol,
        leg_tag="c3_tanh_core_restored", c3_gm_init=c3_gm_init,
    )
    legs.append(leg_c3)
    deltas.append(_delta_vs_c1(c1_report.instrument, leg_c3, "C3"))

    # C4: restore compliance and clip rail handling, keep core linear,
    # boundary linear injection.
    leg_c4 = _bisect_leg(
        order=order, seed=seed, device=device,
        hidden_dim=hidden_dim, t_span=t_span, num_steps=num_steps,
        drive_scale=drive_scale, u_scaled=u_scaled, y_stream=y_stream,
        washout=washout, jacobian_samples=jacobian_samples,
        max_delay=max_delay,
        restore_boundary=False, restore_tanh_core=False, restore_compliance=True,
        drive_rms=drive_rms, matched_parity_tol=matched_parity_tol,
        leg_tag="c4_compliance_restored",
    )
    legs.append(leg_c4)
    deltas.append(_delta_vs_c1(c1_report.instrument, leg_c4, "C4"))

    suppressor = _identify_suppressor(c1_report.instrument, legs, deltas)
    return BisectionReport(
        order=order, seed=seed, device=device, hidden_dim=hidden_dim,
        n_streams=n_streams, train_samples_per_stream=train_samples_per_stream,
        washout=washout, c1_instrument=c1_report.instrument,
        legs=legs, deltas_vs_c1=deltas, suppressor=suppressor,
        note="C2/C3/C4: single-element restoration onto passing C1.",
    )


def _bisect_leg(
    *, order: int, seed: int, device: str, hidden_dim: int,
    t_span: float, num_steps: int, drive_scale: float,
    u_scaled: torch.Tensor, y_stream: torch.Tensor,
    washout: int, jacobian_samples: int, max_delay: int,
    restore_boundary: bool, restore_tanh_core: bool, restore_compliance: bool,
    drive_rms: float, matched_parity_tol: float,
    leg_tag: str, c3_gm_init: float = 0.0,
) -> InstrumentRow:
    """Run one bisection leg with the requested elements restored.

    The restore flags branch inside the patched ``stage.rhs`` (reverting
    ``cell_lib.forward`` alone would be a no-op because the patched rhs
    bypasses the cell lib).  ``c3_gm_init`` fills the core
    ``gm_raw``/``isat_raw`` for the C3 leg (best-sweep gain; restored on
    exit so the install snapshot stays clean).
    """
    net, _, _ = ne._build_fabric_net(
        order=order, seed=seed, freeze_read=False,
        t_span=t_span, num_steps=num_steps, cell_library="tanh_free",
        core_refresh_interval=0, leak_constant=None,
        compile_sequence=False, hidden_dim=hidden_dim,
    )
    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    saved = install_linear_reservoir_v2(
        net,
        G_edge_seed=0, G_in_seed=0, leak_seed=0,
        target_x_max=(config_default_xmax() if restore_compliance
                      else CANONICAL_X_MAX_LIN),
        clip_current=(config_default_clip() if restore_compliance
                      else CANONICAL_CLIP_LIN),
        drive_rms_target=drive_rms, u_seq_for_rms=u_scaled,
    )
    stage = net.core.stages[0]
    gm_snapshot: dict[str, torch.Tensor] = {}
    try:
        if restore_boundary:
            stage._lin_restore_boundary = True
        if restore_tanh_core:
            stage._lin_restore_tanh = True
            for lib_name in ("cell_lib", "boundary_cell_lib", "output_ode_cell_lib"):
                lib = getattr(stage, lib_name, None)
                if lib is None:
                    continue
                for raw_name in ("gm_raw", "isat_raw"):
                    if hasattr(lib, raw_name):
                        param = getattr(lib, raw_name)
                        gm_snapshot[f"{lib_name}.{raw_name}"] = param.detach().clone()
                        with torch.no_grad():
                            param.data.fill_(float(c3_gm_init))
        spec = measure_spectral_radius(
            net, u_scaled, t_span=t_span, num_steps=num_steps,
            washout=washout, n_samples=jacobian_samples, device=device,
        )
        states_full = spec["states_full"]
        abs_eigs: list[float] = []
        for eigs in spec["abs_eigs_per_transition"]:
            abs_eigs.extend([float(v) for v in eigs if math.isfinite(float(v))])
        instrument = _instrument_trajectory(
            stage=net.core.stages[0], states_full=states_full,
            u_seq=u_scaled, y_seq=y_stream, washout=washout,
            jacobian_samples=jacobian_samples,
            t_span=t_span, num_steps=num_steps,
        )
        mc_per, mc_total = _per_delay_mc(
            states_full, u_scaled, washout=washout, max_delay=max_delay,
            use_svd_fallback=True,
        )
        x_max = float(net.core.stages[0].x_max)
        sat_max = float(states_full.abs().max().item())
        rail_frac = float(
            (states_full.abs() > 0.9 * x_max).float().mean().item()
        )
        state_pr_std_local = _standardized_state_pr(states_full[washout:])
        row = InstrumentRow(
            config_tag=(
                f"{leg_tag}_seed{seed}_{device}_h{hidden_dim}"
                f"_drive{drive_scale:g}_washout{washout}"
                f"_tspan{t_span:g}_steps{num_steps}"
            ),
            nrmse=float(instrument["ridge_nrmse"]),
            r2=float(instrument["ridge_r2"]),
            mc_total=float(mc_total),
            mc_per_delay=[float(v) for v in mc_per],
            state_pr=float(instrument["state_pr"]),
            jac_max_abs=float(instrument["jac_max_abs"]),
            jac_min_abs=float(instrument["jac_min_abs"]),
            jac_mean_abs=float(instrument["jac_mean_abs"]),
            jac_rank_proxy=float(instrument["jac_rank_proxy"]),
            jac_eig_abs=abs_eigs,
            sat_max_ratio=float(sat_max / x_max) if x_max > 0 else float("nan"),
            rail_frac=rail_frac,
            n_params=n_params,
            note=(
                f"restored: boundary={restore_boundary}, "
                f"tanh_core={restore_tanh_core}, "
                f"compliance={restore_compliance}"
            ),
        )
        return row
    finally:
        if gm_snapshot:
            for key, value in gm_snapshot.items():
                lib_name, raw_name = key.split(".")
                lib = getattr(net.core.stages[0], lib_name, None)
                if lib is not None and hasattr(lib, raw_name):
                    with torch.no_grad():
                        getattr(lib, raw_name).data.copy_(value)
        restore_linear_reservoir_v2(net, saved)


def _delta_vs_c1(
    c1: InstrumentRow, leg: InstrumentRow, tag: str,
) -> dict:
    """Compute paired C1-versus-restoration deltas for one leg."""
    return {
        "leg": tag,
        "delta_ridge_nrmse": float(leg.nrmse - c1.nrmse),
        "delta_mc_total": float(leg.mc_total - c1.mc_total),
        "delta_state_pr": float(leg.state_pr - c1.state_pr),
        "delta_jac_max_abs": float(leg.jac_max_abs - c1.jac_max_abs),
        "delta_rail_frac": float(leg.rail_frac - c1.rail_frac),
        "delta_sat_max_ratio": float(leg.sat_max_ratio - c1.sat_max_ratio),
        "c1_pass": c1.note,
        "leg_pass_mc": bool(leg.mc_total > C1_PASS_MC_ABOVE),
        "leg_pass_pr": bool(leg.state_pr >= C1_PASS_PR_MIN),
    }


def _identify_suppressor(
    c1: InstrumentRow, legs: list[InstrumentRow], deltas: list[dict],
) -> str:
    """Identify the first restoration collapsing MC and PR."""
    if not legs:
        return "no-legs"
    for leg, delta in zip(legs, deltas):
        mc_collapsed = leg.mc_total <= max(c1.mc_total - 0.5, 1.0)
        pr_collapsed = leg.state_pr <= max(c1.state_pr - 1.0, 2.0)
        if mc_collapsed and pr_collapsed:
            return delta["leg"]
    return "no-collapse-detected"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def config_default_clip() -> float:
    """Read the default ``clip_current`` from ``config.PHYS``."""
    try:
        from config import PHYS
        return float(PHYS["clip_current"])
    except Exception:
        return 0.05


def config_default_xmax() -> float:
    """Read the canonical voltage rail ``x_max`` from ``config.PHYS``."""
    try:
        from config import PHYS
        return float(PHYS["x_max"])
    except Exception:
        return 3.0


def _serialize_rows(rows: list[Any]) -> list[dict]:
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append({k: _jsonable(v) for k, v in r.items()})
        else:
            out.append({k: _jsonable(v) for k, v in asdict(r).items()})
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_probe_csv(path: Path, rows: list[dict],
                    fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
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


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--order", type=int, choices=[10, 20], default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, default=Path("./output/linear_controls"))
    p.add_argument("--n-streams", type=int, default=1)
    p.add_argument("--train-samples", type=int, default=300)
    p.add_argument("--washout", type=int, default=PROBE_WASHOUT)
    p.add_argument("--hidden-dim", type=int, default=CANONICAL_HIDDEN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Linear reservoir controls for the NARMA-10 fabric plateau.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_r0 = sub.add_parser("r0", help="R0 Ridge-instrument reconciliation matrix.")
    _add_common(p_r0)
    p_r0.add_argument("--t-span", type=float, default=CANONICAL_T_SPAN)
    p_r0.add_argument("--num-steps", type=int, default=CANONICAL_NUM_STEPS)

    p_c0 = sub.add_parser(
        "c0", help="C0 ESN-through-E0-instruments calibration control."
    )
    _add_common(p_c0)
    p_c0.add_argument("--n-reservoir", type=int, default=25)
    p_c0.add_argument("--spectral-radius", type=float, default=0.9)
    p_c0.add_argument("--input-scaling", type=float, default=1.0)
    p_c0.add_argument("--leak", type=float, default=1.0)
    p_c0.add_argument("--jacobian-samples", type=int, default=3)
    p_c0.add_argument("--max-delay", type=int, default=20)

    p_c1 = sub.add_parser(
        "c1", help="C1 linear-reservoir-in-fabric control."
    )
    _add_common(p_c1)
    p_c1.add_argument("--t-span", type=float, default=CANONICAL_T_SPAN)
    p_c1.add_argument("--num-steps", type=int, default=CANONICAL_NUM_STEPS)
    p_c1.add_argument("--jacobian-samples", type=int, default=C1_SPECTRAL_TRANSITION_MIN)
    p_c1.add_argument("--drive-scale", type=float, default=CANONICAL_DRIVE_SCALE)
    p_c1.add_argument("--g-edge-seed", type=int, default=0)
    p_c1.add_argument("--g-in-seed", type=int, default=0)
    p_c1.add_argument("--leak-seed", type=int, default=0)
    p_c1.add_argument("--max-delay", type=int, default=20)

    p_c234 = sub.add_parser(
        "c2c3c4",
        help="C2/C3/C4 bisection onto a passing C1.",
    )
    _add_common(p_c234)
    p_c234.add_argument("--t-span", type=float, default=CANONICAL_T_SPAN)
    p_c234.add_argument("--num-steps", type=int, default=CANONICAL_NUM_STEPS)
    p_c234.add_argument("--jacobian-samples", type=int, default=C1_SPECTRAL_TRANSITION_MIN)
    p_c234.add_argument("--drive-scale", type=float, default=CANONICAL_DRIVE_SCALE)
    p_c234.add_argument("--max-delay", type=int, default=20)
    p_c234.add_argument(
        "--c3-gm-init", type=float, default=0.0,
        help="Raw gm/isat fill for the C3 tanh-core restoration leg "
             "(best E0 sweep gain; default 0.0 until the sweep lands).",
    )
    p_c234.add_argument(
        "--c1-json", type=Path, required=True,
        help="JSON report from a prior C1 run; bisection runs only when "
             "the report's pass_all flag is true.",
    )

    p_c1b = sub.add_parser(
        "c1b",
        help="C1b orthogonal-mixing revision: QR-orthogonal W, uniform slow "
             "leak, J-band [0.97, 0.99], MC>=5 gate. Persists tuned base.",
    )
    _add_common(p_c1b)
    p_c1b.add_argument("--t-span", type=float, default=CANONICAL_T_SPAN)
    p_c1b.add_argument("--num-steps", type=int, default=CANONICAL_NUM_STEPS)
    p_c1b.add_argument(
        "--jacobian-samples", type=int, default=C1_SPECTRAL_TRANSITION_MIN,
    )
    p_c1b.add_argument("--drive-scale", type=float, default=CANONICAL_DRIVE_SCALE)
    p_c1b.add_argument("--w-seed", type=int, default=0)
    p_c1b.add_argument("--g-in-seed", type=int, default=0)
    p_c1b.add_argument(
        "--leak-value", type=float, default=C1B_LEAK_VALUE,
        help="Uniform slow-leak value (default 0.05).",
    )
    p_c1b.add_argument(
        "--j-low", type=float, default=C1B_SPECTRAL_TARGET_LOW,
        help="Lower bound of the J-band (default 0.97).",
    )
    p_c1b.add_argument(
        "--j-high", type=float, default=C1B_SPECTRAL_TARGET_HIGH,
        help="Upper bound of the J-band (default 0.99).",
    )
    p_c1b.add_argument("--max-delay", type=int, default=20)
    p_c1b.add_argument(
        "--no-sidecar", action="store_true",
        help="Skip writing the tuned-base sidecar (useful for CPU smokes).",
    )

    p_c1b_real = sub.add_parser(
        "c1b_real",
        help="C1b-real resistive-core variant: gm/isat -> min, softplus-"
             "inverse G overlay, real rhs (no override). Same gates as C1b.",
    )
    _add_common(p_c1b_real)
    p_c1b_real.add_argument("--t-span", type=float, default=CANONICAL_T_SPAN)
    p_c1b_real.add_argument("--num-steps", type=int, default=CANONICAL_NUM_STEPS)
    p_c1b_real.add_argument(
        "--jacobian-samples", type=int, default=C1_SPECTRAL_TRANSITION_MIN,
    )
    p_c1b_real.add_argument(
        "--drive-scale", type=float, default=CANONICAL_DRIVE_SCALE,
    )
    p_c1b_real.add_argument(
        "--boundary-drive-scale", type=float, default=0.5,
        help="Rescale the boundary-OTA G_in drive (default 0.5 small-signal).",
    )
    p_c1b_real.add_argument("--w-seed", type=int, default=0)
    p_c1b_real.add_argument("--g-in-seed", type=int, default=0)
    p_c1b_real.add_argument(
        "--leak-value", type=float, default=C1B_LEAK_VALUE,
        help="Uniform slow-leak value (default 0.05).",
    )
    p_c1b_real.add_argument(
        "--gm-raw-fill", type=float, default=-10.0,
        help="Fill value for cell_lib.gm_raw (very negative => gm ~ gm_min).",
    )
    p_c1b_real.add_argument(
        "--isat-raw-fill", type=float, default=-10.0,
        help="Fill value for cell_lib.isat_raw (very negative => isat ~ isat_min).",
    )
    p_c1b_real.add_argument(
        "--j-uniform-min", type=float, default=C1B_REAL_UNIFORM_MIN,
        help="Uniformity floor: min|J| must reach this (default 0.90).",
    )
    p_c1b_real.add_argument(
        "--j-stability-cap", type=float, default=C1B_REAL_STABILITY_CAP,
        help="Stability cap: max|J| must stay below this (default 0.995).",
    )
    p_c1b_real.add_argument("--max-delay", type=int, default=20)

    p_c3 = sub.add_parser(
        "c3_sweep",
        help="C3 tanh-crossover sweep: install the tuned C1b base, sweep "
             "core gm_raw fills, track the first gm where MC collapses.",
    )
    _add_common(p_c3)
    p_c3.add_argument("--t-span", type=float, default=CANONICAL_T_SPAN)
    p_c3.add_argument("--num-steps", type=int, default=CANONICAL_NUM_STEPS)
    p_c3.add_argument(
        "--jacobian-samples", type=int, default=C1_SPECTRAL_TRANSITION_MIN,
    )
    p_c3.add_argument(
        "--drive-scale", type=float, default=CANONICAL_DRIVE_SCALE,
    )
    p_c3.add_argument("--max-delay", type=int, default=20)
    p_c3.add_argument(
        "--c1b-sidecar-dir", type=Path, required=True,
        help="Directory containing the tuned C1b sidecar "
             f"({C1B_SIDECAR_NAME} + {C1B_SIDECAR_JSON}).",
    )

    args = parser.parse_args(argv)
    if args.order != 10:
        parser.error(
            "linear-control CLI decisions are pre-registered for --order 10 only"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "r0":
        report = r0_reconciliation(
            order=args.order, seed=args.seed, device=args.device,
            n_streams=args.n_streams, train_samples_per_stream=args.train_samples,
            washout=args.washout, hidden_dim=args.hidden_dim,
            t_span=args.t_span, num_steps=args.num_steps,
        )
        rows = _serialize_rows(report.rows)
        write_probe_csv(args.output / "r0_reconciliation.csv", rows)
        hdr = (
            f"R0 reconciliation -- order={args.order} seed={args.seed} "
            f"{report.n_corners} corners in {report.elapsed_s:.1f}s"
        )
        lines = [
            f"{'tag':>30} {'diag':>20} {'NRMSE':>7} {'R^2':>7} "
            f"{'#feat':>5} {'drive':>5} {'T':>5} {'refresh':>10}",
        ]
        for r in report.rows:
            lines.append(
                f"{r.config_tag:>30} {r.diagnostic:>20} "
                f"{r.nrmse:>7.4f} {r.r2:>7.4f} {r.n_features:>5d} "
                f"{r.drive_scale:>5.2f} {r.stream_length:>5d} {r.refresh:>10}"
            )
        lines.append("")
        rule = report.matched_parity_rule
        lines.append(
            f"matched_parity: raw_delay_nrmse={rule['raw_delay_nrmse']:.4f} "
            f"e0_full_state_nrmse={rule['e0_full_state_nrmse']:.4f} "
            f"delta={rule['delta_e0_minus_raw']:+.4f} "
            f"tolerance={rule['tolerance']:.3f} "
            f"rule_holds={rule['rule_holds']}"
        )
        write_probe_txt(args.output / "r0_reconciliation.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        (args.output / "r0_reconciliation.json").write_text(json.dumps({
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "n_streams": args.n_streams,
            "train_samples_per_stream": args.train_samples,
            "washout": args.washout,
            "canonical_net_kwargs": report.canonical_net_kwargs,
            "matched_parity_rule": report.matched_parity_rule,
            "rows": rows,
            "n_corners": report.n_corners,
            "elapsed_s": report.elapsed_s,
            "note": report.note,
        }, indent=2))
        return 0

    if args.mode == "c0":
        row, meta = c0_esn_calibration(
            order=args.order, seed=args.seed,
            n_reservoir=args.n_reservoir, spectral_radius=args.spectral_radius,
            input_scaling=args.input_scaling, leak=args.leak,
            n_streams=args.n_streams, train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples, device=args.device,
        )
        rd = _serialize_rows([row])[0]
        write_probe_csv(args.output / "c0_esn.csv", [rd])
        hdr = (
            f"C0 ESN calibration -- order={args.order} seed={args.seed} "
            f"h={args.n_reservoir} sr={args.spectral_radius:g}"
        )
        lines = [
            f"  config_tag={row.config_tag}",
            f"  nrmse={row.nrmse:.4f}  r2={row.r2:.4f}",
            f"  mc_total={row.mc_total:.3f}  state_pr={row.state_pr:.2f}",
            f"  jac_max_abs={row.jac_max_abs:.4f}  jac_min_abs={row.jac_min_abs:.4f}",
            f"  jac_mean_abs={row.jac_mean_abs:.4f}  jac_rank_proxy={row.jac_rank_proxy:.2f}",
            f"  rail_frac=N/A  sat_max_ratio=N/A  (ESN states in [-1,1])",
            f"  per_delay_mc[:10]={[round(v, 3) for v in row.mc_per_delay[:10]]}",
        ]
        write_probe_txt(args.output / "c0_esn.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        (args.output / "c0_esn.json").write_text(json.dumps({
            "order": args.order, "seed": args.seed, "device": args.device,
            "config": meta,
            "row": rd,
            "thresholds": {
                "mc_above_3_8": (3.0, 8.0),
                "pr_around_15_25": (15.0, 25.0),
            },
            "note": row.note,
        }, indent=2))
        return 0

    if args.mode == "c1":
        report = c1_linear_reservoir(
            order=args.order, seed=args.seed, device=args.device,
            hidden_dim=args.hidden_dim,
            n_streams=args.n_streams, train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            drive_scale=args.drive_scale,
            g_edge_seed=args.g_edge_seed, g_in_seed=args.g_in_seed,
            leak_seed=args.leak_seed,
        )
        rd = _serialize_rows([report.instrument])[0]
        write_probe_csv(args.output / "c1_linear_reservoir.csv", [rd])
        tuning_dicts = _serialize_rows(report.tuning_history)
        write_probe_csv(
            args.output / "c1_tuning_history.csv", tuning_dicts,
            fieldnames=list(tuning_dicts[0].keys()) if tuning_dicts else None,
        )
        hdr = (
            f"C1 linear reservoir -- order={args.order} seed={args.seed} "
            f"h={args.hidden_dim} t_span={args.t_span:g} steps={args.num_steps}"
        )
        lines = [
            f"  config_tag={report.instrument.config_tag}",
            f"  nrmse={report.instrument.nrmse:.4f}  r2={report.instrument.r2:.4f}",
            f"  mc_total={report.instrument.mc_total:.3f}  "
            f"state_pr={report.instrument.state_pr:.2f}",
            f"  jac_max_abs={report.instrument.jac_max_abs:.4f}  "
            f"jac_min_abs={report.instrument.jac_min_abs:.4f}",
            f"  jac_mean_abs={report.instrument.jac_mean_abs:.4f}  "
            f"jac_rank_proxy={report.instrument.jac_rank_proxy:.2f}",
            f"  rail_frac={report.instrument.rail_frac:.4f}  "
            f"sat_max_ratio={report.instrument.sat_max_ratio:.4f}",
            f"  matched_parity_delta={report.matched_parity_delta:+.4f}  "
            f"tolerance={report.matched_parity_tolerance:.3f}",
            f"  pass_mc={report.pass_mc} pass_pr={report.pass_pr} "
            f"pass_ridge_matched={report.pass_ridge_matched} "
            f"pass_all={report.pass_all}",
            f"  n_tuning_iterations={report.n_tuning_iterations}",
        ]
        write_probe_txt(args.output / "c1_linear_reservoir.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        (args.output / "c1_linear_reservoir.json").write_text(json.dumps({
            "order": report.order, "seed": report.seed,
            "device": report.device, "hidden_dim": report.hidden_dim,
            "n_streams": report.n_streams,
            "train_samples_per_stream": report.train_samples_per_stream,
            "washout": report.washout,
            "n_params": report.n_params,
            "canonical_target_radius_low": report.canonical_target_radius_low,
            "canonical_target_radius_high": report.canonical_target_radius_high,
            "n_tuning_iterations": report.n_tuning_iterations,
            "tuning_history": tuning_dicts,
            "instrument": rd,
            "matched_raw_delay_nrmse": report.matched_raw_delay_nrmse,
            "matched_parity_tolerance": report.matched_parity_tolerance,
            "matched_parity_delta": report.matched_parity_delta,
            "pass_mc": report.pass_mc,
            "pass_pr": report.pass_pr,
            "pass_ridge_matched": report.pass_ridge_matched,
            "pass_all": report.pass_all,
            "nrmse_hidden": report.nrmse_hidden,
            "r2_hidden": report.r2_hidden,
            "mc_total_hidden": report.mc_total_hidden,
            "state_pr_hidden": report.state_pr_hidden,
            "note": report.note,
        }, indent=2))
        return 0

    if args.mode == "c2c3c4":
        c1_json = json.loads(args.c1_json.read_text())
        c1_report = LinearReservoirReport(
            order=c1_json["order"], seed=c1_json["seed"],
            device=c1_json["device"], hidden_dim=c1_json["hidden_dim"],
            n_streams=c1_json["n_streams"],
            train_samples_per_stream=c1_json["train_samples_per_stream"],
            washout=c1_json["washout"],
            n_params=c1_json["n_params"],
            canonical_target_radius_low=c1_json["canonical_target_radius_low"],
            canonical_target_radius_high=c1_json["canonical_target_radius_high"],
            n_tuning_iterations=c1_json["n_tuning_iterations"],
            tuning_history=c1_json["tuning_history"],
            instrument=InstrumentRow(**{
                k: v for k, v in c1_json["instrument"].items()
                if k in InstrumentRow.__dataclass_fields__
            }),
            matched_raw_delay_nrmse=c1_json["matched_raw_delay_nrmse"],
            matched_parity_tolerance=c1_json["matched_parity_tolerance"],
            matched_parity_delta=c1_json["matched_parity_delta"],
            pass_mc=c1_json["pass_mc"], pass_pr=c1_json["pass_pr"],
            pass_ridge_matched=c1_json["pass_ridge_matched"],
            pass_all=c1_json["pass_all"],
            nrmse_hidden=c1_json["nrmse_hidden"],
            r2_hidden=c1_json["r2_hidden"],
            mc_total_hidden=c1_json["mc_total_hidden"],
            state_pr_hidden=c1_json["state_pr_hidden"],
            note=c1_json["note"],
        )
        report = c2_c3_c4_bisection(
            c1_report,
            order=args.order, seed=args.seed, device=args.device,
            hidden_dim=args.hidden_dim,
            n_streams=args.n_streams, train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            drive_scale=args.drive_scale,
            c3_gm_init=args.c3_gm_init,
        )
        rows = _serialize_rows([report.c1_instrument] + report.legs)
        write_probe_csv(args.output / "c2c3c4.csv", rows)
        write_probe_csv(
            args.output / "c2c3c4_deltas.csv",
            _serialize_rows(report.deltas_vs_c1),
        )
        hdr = (
            f"C2/C3/C4 bisection -- order={args.order} seed={args.seed} "
            f"h={args.hidden_dim} suppressor={report.suppressor}"
        )
        lines = [
            f"  C1   mc={report.c1_instrument.mc_total:.3f}  "
            f"pr={report.c1_instrument.state_pr:.2f}  "
            f"ridge={report.c1_instrument.nrmse:.4f}",
        ]
        for leg, delta in zip(report.legs, report.deltas_vs_c1):
            lines.append(
                f"  {delta['leg']}  mc={leg.mc_total:.3f}  "
                f"pr={leg.state_pr:.2f}  ridge={leg.nrmse:.4f}  "
                f"delta_mc={delta['delta_mc_total']:+.3f}  "
                f"delta_pr={delta['delta_state_pr']:+.2f}"
            )
        lines.append(f"  suppressor: {report.suppressor}")
        write_probe_txt(args.output / "c2c3c4.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        (args.output / "c2c3c4.json").write_text(json.dumps({
            "order": report.order, "seed": report.seed,
            "device": report.device, "hidden_dim": report.hidden_dim,
            "n_streams": report.n_streams,
            "train_samples_per_stream": report.train_samples_per_stream,
            "washout": report.washout,
            "c1_instrument": _serialize_rows([report.c1_instrument])[0],
            "legs": _serialize_rows(report.legs),
            "deltas_vs_c1": _serialize_rows(report.deltas_vs_c1),
            "suppressor": report.suppressor,
            "note": report.note,
        }, indent=2))
        return 0

    if args.mode == "c1b":
        sidecar_dir = None if args.no_sidecar else args.output
        report = c1b_orthogonal_reservoir(
            order=args.order, seed=args.seed, device=args.device,
            hidden_dim=args.hidden_dim,
            n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            drive_scale=args.drive_scale,
            w_seed=args.w_seed, g_in_seed=args.g_in_seed,
            leak_value=args.leak_value,
            target_low=args.j_low, target_high=args.j_high,
            sidecar_dir=sidecar_dir,
        )
        rd = _serialize_rows([report.instrument])[0]
        write_probe_csv(args.output / "c1b.csv", [rd])
        tuning_dicts = _serialize_rows(report.tuning_history)
        if tuning_dicts:
            write_probe_csv(
                args.output / "c1b_tuning_history.csv", tuning_dicts,
                fieldnames=list(tuning_dicts[0].keys()),
            )
        hdr = (
            f"C1b orthogonal -- order={args.order} seed={args.seed} "
            f"h={args.hidden_dim} t_span={args.t_span:g} steps={args.num_steps} "
            f"leak={report.leak_value:g} target=[{args.j_low:g},{args.j_high:g}]"
        )
        lines = [
            f"  config_tag={report.instrument.config_tag}",
            f"  nrmse={report.instrument.nrmse:.4f}  r2={report.instrument.r2:.4f}",
            f"  mc_total={report.instrument.mc_total:.3f}  "
            f"state_pr={report.instrument.state_pr:.2f}  "
            f"state_pr_std={report.state_pr_standardized:.2f}",
            f"  jac_max_abs={report.instrument.jac_max_abs:.4f}  "
            f"jac_min_abs={report.instrument.jac_min_abs:.4f}",
            f"  rail_frac={report.instrument.rail_frac:.4f}  "
            f"sat_max_ratio={report.instrument.sat_max_ratio:.4f}",
            f"  matched_parity_delta={report.matched_parity_delta:+.4f}  "
            f"tolerance={report.matched_parity_tolerance:.3f} (diagnostic)",
            f"  pass_mc={report.pass_mc} pass_rail={report.pass_rail} "
            f"pass_spectral={report.pass_spectral} pass_all={report.pass_all}",
            f"  n_tuning_iterations={report.n_tuning_iterations}",
            f"  pre_registered_mc=[{report.pre_registered_mc_low:g},"
            f"{report.pre_registered_mc_high:g}]",
            f"  sidecar={report.sidecar_path or 'NONE'}",
            f"  sidecar_json={report.sidecar_json_path or 'NONE'}",
        ]
        write_probe_txt(args.output / "c1b.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        (args.output / "c1b.json").write_text(json.dumps({
            "order": report.order, "seed": report.seed,
            "device": report.device, "hidden_dim": report.hidden_dim,
            "n_streams": report.n_streams,
            "train_samples_per_stream": report.train_samples_per_stream,
            "washout": report.washout,
            "n_params": report.n_params,
            "n_tuning_iterations": report.n_tuning_iterations,
            "tuning_history": tuning_dicts,
            "instrument": rd,
            "matched_raw_delay_nrmse": report.matched_raw_delay_nrmse,
            "matched_parity_tolerance": report.matched_parity_tolerance,
            "matched_parity_delta": report.matched_parity_delta,
            "leak_value": report.leak_value,
            "spectral_target_low": report.spectral_target_low,
            "spectral_target_high": report.spectral_target_high,
            "pre_registered_mc_low": report.pre_registered_mc_low,
            "pre_registered_mc_high": report.pre_registered_mc_high,
            "w_seed": report.w_seed,
            "w_construction": report.w_construction,
            "state_pr_raw": report.state_pr_raw,
            "state_pr_standardized": report.state_pr_standardized,
            "pass_mc": report.pass_mc,
            "pass_rail": report.pass_rail,
            "pass_spectral": report.pass_spectral,
            "pass_all": report.pass_all,
            "sidecar_path": report.sidecar_path,
            "sidecar_json_path": report.sidecar_json_path,
            "note": report.note,
        }, indent=2))
        return 0

    if args.mode == "c1b_real":
        report = c1b_real_resistive(
            order=args.order, seed=args.seed, device=args.device,
            hidden_dim=args.hidden_dim,
            n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            drive_scale=args.drive_scale,
            w_seed=args.w_seed, g_in_seed=args.g_in_seed,
            leak_value=args.leak_value,
            boundary_drive_scale=args.boundary_drive_scale,
            gm_raw_fill=args.gm_raw_fill,
            isat_raw_fill=args.isat_raw_fill,
            uniform_min=args.j_uniform_min,
            stability_cap=args.j_stability_cap,
        )
        rd = _serialize_rows([report.instrument])[0]
        write_probe_csv(args.output / "c1b_real.csv", [rd])
        tuning_dicts = _serialize_rows(report.tuning_history)
        if tuning_dicts:
            write_probe_csv(
                args.output / "c1b_real_tuning_history.csv", tuning_dicts,
                fieldnames=list(tuning_dicts[0].keys()),
            )
        hdr = (
            f"C1b-real resistive -- order={args.order} seed={args.seed} "
            f"h={args.hidden_dim} t_span={args.t_span:g} steps={args.num_steps} "
            f"leak={report.leak_value:g} gm={report.gm_raw_fill:g} "
            f"isat={report.isat_raw_fill:g} "
            f"uniform_min={args.j_uniform_min:g} "
            f"stability_cap={args.j_stability_cap:g}"
        )
        lines = [
            f"  config_tag={report.instrument.config_tag}",
            f"  nrmse={report.instrument.nrmse:.4f}  r2={report.instrument.r2:.4f}",
            f"  mc_total={report.instrument.mc_total:.3f}  "
            f"state_pr={report.instrument.state_pr:.2f}  "
            f"state_pr_std={report.state_pr_standardized:.2f}",
            f"  jac_max_abs={report.instrument.jac_max_abs:.4f}",
            f"  rail_frac={report.instrument.rail_frac:.4f}  "
            f"sat_max_ratio={report.instrument.sat_max_ratio:.4f}",
            f"  pass_mc={report.pass_mc} pass_rail={report.pass_rail} "
            f"pass_spectral={report.pass_spectral} pass_all={report.pass_all}",
            f"  n_tuning_iterations={report.n_tuning_iterations}",
        ]
        write_probe_txt(args.output / "c1b_real.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        (args.output / "c1b_real.json").write_text(json.dumps({
            "order": report.order, "seed": report.seed,
            "device": report.device, "hidden_dim": report.hidden_dim,
            "n_streams": report.n_streams,
            "train_samples_per_stream": report.train_samples_per_stream,
            "washout": report.washout,
            "n_params": report.n_params,
            "n_tuning_iterations": report.n_tuning_iterations,
            "tuning_history": tuning_dicts,
            "instrument": rd,
            "matched_raw_delay_nrmse": report.matched_raw_delay_nrmse,
            "matched_parity_tolerance": report.matched_parity_tolerance,
            "matched_parity_delta": report.matched_parity_delta,
            "gm_raw_fill": report.gm_raw_fill,
            "isat_raw_fill": report.isat_raw_fill,
            "boundary_drive_scale": report.boundary_drive_scale,
            "spectral_target_low": report.spectral_target_low,
            "spectral_target_high": report.spectral_target_high,
            "state_pr_raw": report.state_pr_raw,
            "state_pr_standardized": report.state_pr_standardized,
            "pass_mc": report.pass_mc,
            "pass_rail": report.pass_rail,
            "pass_spectral": report.pass_spectral,
            "pass_all": report.pass_all,
            "note": report.note,
        }, indent=2))
        return 0

    if args.mode == "c3_sweep":
        report = c3_gm_grid_sweep(
            c1b_sidecar_dir=args.c1b_sidecar_dir,
            order=args.order, seed=args.seed, device=args.device,
            hidden_dim=args.hidden_dim,
            n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            drive_scale=args.drive_scale,
        )
        rows = _serialize_rows(report.rows)
        write_probe_csv(args.output / "c3_sweep.csv", rows)
        hdr = (
            f"C3 tanh-crossover sweep -- order={args.order} seed={args.seed} "
            f"h={args.hidden_dim} c1b_mc={report.c1b_mc_total:.3f} "
            f"crossover={report.crossover_gm_raw_fill}"
        )
        lines = [
            f"  c1b_mc={report.c1b_mc_total:.3f}",
            f"  crossover_gm_raw_fill={report.crossover_gm_raw_fill}",
            f"  {'gm_raw_fill':>12} {'mc':>8} {'ridge':>8} {'rail':>8} "
            f"{'jac_max':>8} {'collapsed':>10}",
        ]
        for r in report.rows:
            lines.append(
                f"  {r.gm_raw_fill:>12.3f} {r.mc_total:>8.3f} "
                f"{r.ridge_nrmse:>8.4f} {r.rail_frac:>8.4f} "
                f"{r.jac_max_abs:>8.4f} {str(r.collapsed):>10}"
            )
        write_probe_txt(args.output / "c3_sweep.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        (args.output / "c3_sweep.json").write_text(json.dumps({
            "order": report.order, "seed": report.seed,
            "device": report.device, "hidden_dim": report.hidden_dim,
            "c1b_path": report.c1b_path,
            "c1b_mc_total": report.c1b_mc_total,
            "gm_grid": report.gm_grid,
            "rows": rows,
            "crossover_gm_raw_fill": report.crossover_gm_raw_fill,
            "note": report.note,
        }, indent=2))
        return 0

    parser.error(f"unknown mode {args.mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
