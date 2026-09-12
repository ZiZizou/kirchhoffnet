"""Revised NARMA-plan probes and helpers (Steps 1, 2, 4 of the plan).

Implements the measurement-first additions from
``docs/narma-revised-plan.md`` on top of the existing
``narma_advisor_probes`` and ``narma_linear_controls`` modules. Side-effect
free at import time.

Provides:

1. **S1-extended memory probes** (Step 1 of the plan):

   - :func:`y_memory_capacity_curve` -- output-trace MC (per-delay curve).
     Reuses ``narma_experiment.memory_capacity`` with the standardized test
     ``y`` as the regression target instead of ``u``. Measures whether the
     accumulator + hidden channel reconstructs the Sigma-y functional the
     task needs (plan sec. 4b, sec. 2 C2).
   - :func:`ipc2_curve` -- quadratic IPC per-delay curve. Regresses against
     ``u(n-k)^2`` (centered) for each delay. The total IPC-2 = sum over k
     of the R^2 of those ridge fits (Dambre convention, plan sec. 4c).
   - :func:`nar10_product_pairs_r2` -- the few ``u(n-i)*u(n-j)`` pairs the
     NARMA-10 hard term depends on (lag 9 × lag 0, plus its neighbors).
     Same ridge solve, single R^2 per pair. Cheap cross-check against
     IPC-2: a near-zero IPC-2 *plus* zero pair R^2 means the lag-9
     product path is provably absent.
   - :func:`s1_extended_probe` -- wrapper returning all three on the same
     collected states; one CLI ``memory`` invocation runs everything.

2. **Tuned-ESN calibration** (Step 2):

   - :func:`tuned_esn_calibration` -- broader ESN grid (spectral radius x
     leak x size x input_scaling x ridge_lambda, 288 corners). Identical
     stream / washout / seeds / metrics as the fabric legs; per-corner
     NRMSE/R^2/MC/PR. Decision rule: tuned ESN clearly below ~0.4 -> band
     stands; tuned ESN stuck >= ~0.4 -> band miscalibrated, retire
     NARMA-10 as the gate.

3. **Hetero-leak init** (Step 4 hook):

   - :func:`hetero_leak_init` -- per-node time constants tau drawn
     log-uniform in [tau_lo, tau_hi]; converts to a raw_leak vector via
     softplus-inverse and writes it into the stage. Independent of the
     existing ``apply_gain_override(leak_mode='randomized')`` which uses
     a normal draw on raw_leak (not the same distribution).
   - :func:`build_small_world_narma_preset` -- NARMA preset override that
     flips ``hidden_family`` from ``torus`` to ``small_world`` with the
     given ``k``, ``p``, ``seed``; passes through to ``build_net_from_config``.

The CLI in :mod:`main` exposes ``memory``, ``esn-cal``, and
``hetero-leak-init`` modes for smoke runs. Heavy grid sweeps are meant
for Alliance GPU; local smoke is sized so a 2-corner E0 / 1-ESN-corner
calibration finishes in seconds.
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
from typing import Any, Iterable

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import narma_advisor_probes as npr  # noqa: E402
import narma_experiment as ne  # noqa: E402
from config import (  # noqa: E402
    INIT as _CONFIG_INIT,
    make_narma_preset as _make_narma_preset_factory,
)

# ---------------------------------------------------------------------------
# Step 1: S1-extended memory probes
# ---------------------------------------------------------------------------

# Pre-registered decision thresholds (plan sec. 1 / 4). The IPC-2 / y-MC
# thresholds are reference points only (anchored against C0-ESN); the
# plan only commits to "presence vs absence" verdicts.
IPC2_PRESENCE_THRESHOLD: float = 0.05
Y_MC_PRESENCE_THRESHOLD: float = 0.05
# NARMA-10 product pairs to score explicitly. The hard term is
# 1.5 * u(n-9) * u(n) + ... so we cover the lag-9 axis and a few
# neighbors. Pairs are (i, j) with i > j (so the pair is u(n-i)*u(n-j)).
NAR10_PRODUCT_PAIRS: tuple[tuple[int, int], ...] = (
    (9, 0), (9, 1), (9, 2),
    (10, 0), (10, 1),
    (8, 0), (8, 1),
)


@dataclass
class S1ExtendedRow:
    """One S1-extended probe row (frozen-state)."""

    config_tag: str
    nrmse_ridge_full: float
    r2_ridge_full: float
    nrmse_ridge_hidden: float
    r2_ridge_hidden: float
    # u-MC (existing instrument, retained for comparability).
    u_mc_total: float
    u_mc_per_delay: list[float]
    # y-MC: regress states -> past y targets (output-trace).
    y_mc_total: float
    y_mc_per_delay: list[float]
    # IPC-2 per delay (Dambre).
    ipc2_total: float
    ipc2_per_delay: list[float]
    # NARMA-10 specific: u(n-i) * u(n-j) pair R^2 for the few pairs the
    # hard term depends on.
    product_pair_r2: dict[str, float]
    # Decision helpers.
    ipc2_present: bool
    y_mc_present: bool
    product_present: bool


def y_memory_capacity_curve(
    states: torch.Tensor, y_train: torch.Tensor, *,
    washout: int = npr.PROBE_WASHOUT, max_delay: int = 20,
    ridge_l2: float = 1e-2,
) -> tuple[list[float], float]:
    """Output-trace memory capacity (plan sec. 2 C2, sec. 4b).

    Identical machinery to ``memory_capacity`` but with the standardized
    train/test target ``y`` as the regression signal instead of ``u``.
    Discards the first ``washout`` states to match the probe convention
    (same washout-corrected caveat as :func:`memory_capacity_washout`).
    """
    if not torch.isfinite(states).all() or not torch.isfinite(y_train).all():
        return [float("nan")] * max_delay, float("nan")
    if washout >= states.shape[0]:
        return [float("nan")] * max_delay, float("nan")
    s_w = states[washout:]
    y_w = y_train[washout:]
    return ne.memory_capacity(s_w, y_w, max_delay=max_delay, ridge_l2=ridge_l2)


def ipc2_curve(
    states: torch.Tensor, u_seq: torch.Tensor, *,
    washout: int = npr.PROBE_WASHOUT, max_delay: int = 20,
    ridge_l2: float = 1e-2,
) -> tuple[list[float], float]:
    """Quadratic IPC per-delay curve (plan sec. 4c).

    For each delay ``k`` we build the centered quadratic feature
    ``t_k = u(n-k)^2 - mean(u^2)`` over the post-washout window and fit
    a ridge ``states -> t_k``. The R^2 of that fit is IPC-2_k; total
    IPC-2 = sum_k max(R^2, 0). Sum is clamped at 0 so individual delay
    blowups cannot poison the total (parity with ``memory_capacity``).
    """
    if not torch.isfinite(states).all() or not torch.isfinite(u_seq).all():
        return [float("nan")] * max_delay, float("nan")
    if washout >= states.shape[0]:
        return [float("nan")] * max_delay, float("nan")
    if u_seq.dim() != 1 or states.dim() != 2:
        raise ValueError(
            f"ipc2_curve requires (T,) inputs and (T, N) states, got "
            f"{tuple(u_seq.shape)} and {tuple(states.shape)}"
        )
    if u_seq.shape[0] != states.shape[0]:
        raise ValueError(
            f"u and states must have the same length, got {u_seq.shape[0]} "
            f"and {states.shape[0]}"
        )
    s_w = states[washout:]
    u_w = u_seq[washout:]
    # Center the quadratic: subtract the mean so the feature has zero
    # mean (ridge cannot exploit a constant). This matches the standard
    # Dambre convention for orthogonalized polynomial channels.
    u2 = u_w.pow(2)
    u2_centered = u2 - u2.mean()
    # Degenerate guard: a zero-variance target carries no information and
    # the shared ``r2()`` helper returns 1.0 on it (perfect prediction of
    # a constant). Score 0.0 so constant signals cannot fake IPC-2
    # presence. (Real NARMA streams always have u^2 variance > 0.)
    if float(u2_centered.var().item()) < 1e-12:
        return [0.0] * max_delay, 0.0
    r2_list: list[float] = []
    for k in range(1, max_delay + 1):
        if s_w.shape[0] - k <= 0:
            r2_list.append(0.0)
            continue
        X = s_w[k:]
        t = u2_centered[: s_w.shape[0] - k]
        X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        XtX = X_aug.T @ X_aug + ridge_l2 * torch.eye(
            X_aug.shape[1], device=X.device,
        )
        W = torch.linalg.solve(XtX, X_aug.T @ t)
        t_pred = X_aug @ W
        r2_list.append(ne.r2(t_pred, t))
    total = sum(max(r, 0.0) for r in r2_list)
    return r2_list, float(total)


def nar10_product_pairs_r2(
    states: torch.Tensor, u_seq: torch.Tensor, *,
    washout: int = npr.PROBE_WASHOUT,
    pairs: Iterable[tuple[int, int]] = NAR10_PRODUCT_PAIRS,
    ridge_l2: float = 1e-2,
) -> dict[str, float]:
    """Per-pair ridge R^2 of ``u(n-i) * u(n-j)`` from washed states.

    The NARMA-10 hard term is ``1.5 * u(n-9) * u(n) + 0.1`` (plus a few
    linear + constant terms). A non-zero R^2 on the (9, 0) pair means
    the lag-9 product path exists in the reservoir's readout-visible
    representation; near-zero on every pair means the multiplicative
    mechanism is provably absent.
    """
    if not torch.isfinite(states).all() or not torch.isfinite(u_seq).all():
        return {f"({i},{j})": float("nan") for i, j in pairs}
    if u_seq.dim() != 1 or states.dim() != 2:
        raise ValueError(
            f"nar10_product_pairs_r2 requires (T,) inputs and (T, N) states, "
            f"got {tuple(u_seq.shape)} and {tuple(states.shape)}"
        )
    if u_seq.shape[0] != states.shape[0]:
        raise ValueError(
            f"u and states must have the same length, got {u_seq.shape[0]} "
            f"and {states.shape[0]}"
        )
    out: dict[str, float] = {}
    for (i, j) in pairs:
        max_lag = max(i, j)
        if washout + max_lag >= u_seq.shape[0]:
            out[f"({i},{j})"] = float("nan")
            continue
        s_w = states[washout:]
        u_w = u_seq[washout:]
        # target t = u(n-i) * u(n-j) -> we want t at sample n; row n in
        # the post-washout window corresponds to original index
        # n + washout. So t[n] = u[(n + washout) - i] * u[(n + washout) - j].
        # Vectorised: t = u_w[:-max_lag] (slice off the tail that can't
        # see back i / j steps). For row n of t, the indices used are
        # u_w[n + (max_lag - i)] and u_w[n + (max_lag - j)].
        u_shift_i = u_w[max_lag - i: u_w.shape[0] - i]
        u_shift_j = u_w[max_lag - j: u_w.shape[0] - j]
        # Pair the shifted views: align so that row n of both uses the
        # same n. Easiest: both views above are length u_w.shape[0] - max_lag.
        if u_shift_i.shape[0] != u_shift_j.shape[0]:
            out[f"({i},{j})"] = float("nan")
            continue
        t = u_shift_i * u_shift_j
        # Center the target (constant subtraction; ridge handles the
        # constant via the bias column). Zero-variance guard: a constant
        # product (e.g. zero drive) carries no information; score 0.0
        # instead of the degenerate r2() == 1.0.
        t_centered = t - t.mean()
        if float(t_centered.var().item()) < 1e-12:
            out[f"({i},{j})"] = 0.0
            continue
        # States aligned to t: row n of t corresponds to original sample
        # washout + n + max_lag; in the post-washout state window that is
        # index n + max_lag.
        X = s_w[max_lag: s_w.shape[0]]
        if X.shape[0] != t_centered.shape[0]:
            out[f"({i},{j})"] = float("nan")
            continue
        X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        XtX = X_aug.T @ X_aug + ridge_l2 * torch.eye(
            X_aug.shape[1], device=X.device,
        )
        W = torch.linalg.solve(XtX, X_aug.T @ t_centered)
        t_pred = X_aug @ W
        out[f"({i},{j})"] = float(ne.r2(t_pred, t_centered))
    return out


def _collect_states_single_stage(
    net: nn.Module, u_seq: torch.Tensor, *,
    device: str = "cpu",
) -> torch.Tensor:
    """Forward ``u_seq`` through a single-stage fabric and return ``(T, N)`` states."""
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "_collect_states_single_stage supports single-stage nets only"
        )
    stage = net.core.stages[0]
    state_width = net.hid_count + net.proj_count + net.output_ode_count
    x0 = u_seq.new_zeros(1, state_width)
    t_span = float(net.core.stage_times[0])
    num_steps = int(net.core.stage_steps[0])
    u_seq = u_seq.to(device)
    with torch.no_grad():
        all_states = stage._forward_heun_sequence(
            x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_seq,
        )
    return all_states[:, 0, :].detach().to(device)


def s1_extended_probe(
    net: nn.Module, u_train: torch.Tensor, y_train: torch.Tensor, *,
    washout: int = npr.PROBE_WASHOUT, ridge_l2: float = 1e-2,
    device: str = "cpu", config_tag: str = "",
) -> S1ExtendedRow:
    """S1-extended: u-MC + y-MC + IPC-2 + product pairs on the same states.

    Args:
        net: A (built, possibly trained) ``KirchhoffNetWithIO`` with one
            stage; the per-stage hidden / proj / accumulator widths are
            read from ``net.hid_count`` / ``net.proj_count`` /
            ``net.output_ode_count``.
        u_train: Scaled input drive ``(T,)`` (after ``_scale_drive``).
        y_train: Standardized training targets ``(T,)`` for the y-MC
            leg. ``u`` is taken straight from ``u_train`` for the
            u-MC / IPC-2 / product-pair legs.
        washout: Initial samples to discard before fitting ridge.
        ridge_l2: L2 regularization for every ridge leg.
        device: ``"cpu"`` or ``"cuda"``.
        config_tag: String written verbatim into the row for CSV join.
    """
    if u_train.dim() != 1 or y_train.dim() != 1:
        raise ValueError(
            f"s1_extended_probe requires one-dimensional input/target streams, "
            f"got {tuple(u_train.shape)} and {tuple(y_train.shape)}"
        )
    if u_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"u and y streams must have the same length, got {u_train.shape[0]} "
            f"and {y_train.shape[0]}"
        )
    states = _collect_states_single_stage(net, u_train, device=device)
    if states.shape[0] <= washout:
        raise ValueError(
            f"need more than {washout} washed-out samples, got {states.shape[0]}"
        )
    # Hidden slice for the dense-readout-comparable ridge probe.
    hid_count = int(net.hid_count)
    hidden = states[:, :hid_count]

    def _ridge_nrmse_r2(X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        X_w = X[washout:]
        y_w = y[washout:]
        W = npr._ridge_fit_predict(X_w, y_w, l2=ridge_l2)
        pred = torch.cat(
            [X_w, torch.ones(X_w.shape[0], 1, device=X_w.device)], dim=1,
        ) @ W
        return float(ne.nrmse(pred, y_w)), float(ne.r2(pred, y_w))

    nrmse_full, r2_full = _ridge_nrmse_r2(states, y_train.to(device))
    nrmse_hid, r2_hid = _ridge_nrmse_r2(hidden, y_train.to(device))

    u_per, u_mc_total = npr.memory_capacity_washout(
        states, u_train.to(device), washout=washout, max_delay=20,
        ridge_l2=ridge_l2,
    )
    y_per, y_mc_total = y_memory_capacity_curve(
        states, y_train.to(device), washout=washout, max_delay=20,
        ridge_l2=ridge_l2,
    )
    ipc2_per, ipc2_total = ipc2_curve(
        states, u_train.to(device), washout=washout, max_delay=20,
        ridge_l2=ridge_l2,
    )
    pair_r2 = nar10_product_pairs_r2(
        states, u_train.to(device), washout=washout, ridge_l2=ridge_l2,
    )

    return S1ExtendedRow(
        config_tag=config_tag,
        nrmse_ridge_full=nrmse_full,
        r2_ridge_full=r2_full,
        nrmse_ridge_hidden=nrmse_hid,
        r2_ridge_hidden=r2_hid,
        u_mc_total=float(u_mc_total),
        u_mc_per_delay=u_per,
        y_mc_total=float(y_mc_total),
        y_mc_per_delay=y_per,
        ipc2_total=float(ipc2_total),
        ipc2_per_delay=ipc2_per,
        product_pair_r2=pair_r2,
        ipc2_present=ipc2_total > IPC2_PRESENCE_THRESHOLD,
        y_mc_present=y_mc_total > Y_MC_PRESENCE_THRESHOLD,
        product_present=any(
            v > IPC2_PRESENCE_THRESHOLD for v in pair_r2.values()
            if math.isfinite(v)
        ),
    )


# ---------------------------------------------------------------------------
# Step 2: Tuned-ESN calibration
# ---------------------------------------------------------------------------

ESN_TUNED_GRID: dict[str, tuple[Any, ...]] = {
    "spectral_radius": (0.7, 0.9, 1.0, 1.1),
    "leak": (0.3, 0.5, 0.8, 1.0),
    "n_reservoir": (25, 50, 100),
    "input_scaling": (0.2, 0.5, 1.0),
    "ridge_lambda": (1e-4, 1e-2),
}
# Plan gate: tuned ESN clearly below ~0.4 -> band stands; stuck >= ~0.4
# -> band miscalibrated, retire NARMA-10 as the gate.
ESN_CAL_BAND_STANDS_BELOW: float = 0.4


@dataclass
class ESNCalRow:
    """One tuned-ESN corner."""
    spectral_radius: float
    leak: float
    n_reservoir: int
    input_scaling: float
    ridge_lambda: float
    nrmse: float
    r2: float
    mc_total: float
    state_pr: float
    n_params: int


def tuned_esn_calibration(
    order: int, seed: int, *, device: str = "cpu",
    spectral_radius: Iterable[float] = ESN_TUNED_GRID["spectral_radius"],
    leak: Iterable[float] = ESN_TUNED_GRID["leak"],
    n_reservoir: Iterable[int] = ESN_TUNED_GRID["n_reservoir"],
    input_scaling: Iterable[float] = ESN_TUNED_GRID["input_scaling"],
    ridge_lambda: Iterable[float] = ESN_TUNED_GRID["ridge_lambda"],
    max_corners: int = 0,
    train_samples: int = 2500,
    washout: int = npr.PROBE_WASHOUT,
) -> list[ESNCalRow]:
    """Run the tuned-ESN calibration grid (plan sec. 5).

    Each corner: standard :class:`ESN` from ``narma_experiment``,
    refit on the full NARMA train stream, evaluate on a fresh test
    stream with the same washout protocol as the fabric legs. Also
    reports MC and participation-ratio on the train trajectory.
    """
    from narma_experiment import ESN  # local import to keep module-side-effect-free
    u_train, y_train = ne.narma(train_samples, order=order, seed=seed)
    u_test, y_test = ne.narma(1000, order=order, seed=seed + 10000)
    corners = [
        (float(sr), float(lk), int(nr), float(ins), float(rl))
        for sr in spectral_radius
        for lk in leak
        for nr in n_reservoir
        for ins in input_scaling
        for rl in ridge_lambda
    ]
    if max_corners > 0:
        corners = corners[:max_corners]
    rows: list[ESNCalRow] = []
    for sr, lk, nr, ins, rl in corners:
        torch.manual_seed(seed)
        esn = ESN(
            n_reservoir=nr, spectral_radius=sr,
            input_scaling=ins, leak=lk, ridge_l2=rl, seed=seed,
        )
        esn.fit(u_train, y_train)
        y_pred_test = esn.predict(u_test)
        nrmse_val = ne.nrmse(y_pred_test[washout:], y_test[washout:])
        r2_val = ne.r2(y_pred_test[washout:], y_test[washout:])
        # Train-trajectory MC / PR (ridge over the train states; ESN
        # does NOT cache ``train_states_``, so we re-collect via _run).
        try:
            train_states = esn._run(u_train)
            if train_states.shape[0] > washout:
                _, mc_total = npr.memory_capacity_washout(
                    train_states, u_train, washout=washout, max_delay=20,
                    ridge_l2=rl,
                )
                state_pr = float(npr.participation_ratio(train_states[washout:]))
            else:
                mc_total = float("nan")
                state_pr = float("nan")
        except (ValueError, RuntimeError):
            mc_total = float("nan")
            state_pr = float("nan")
        trained = nr + 1
        rows.append(ESNCalRow(
            spectral_radius=sr, leak=lk, n_reservoir=nr,
            input_scaling=ins, ridge_lambda=rl,
            nrmse=float(nrmse_val), r2=float(r2_val),
            mc_total=float(mc_total), state_pr=float(state_pr),
            n_params=trained,
        ))
    return rows


# ---------------------------------------------------------------------------
# Step 4 hooks: hetero-leak init + small-world NARMA preset
# ---------------------------------------------------------------------------


def hetero_leak_init(
    stage: nn.Module, *, tau_lo: float = 1.0, tau_hi: float = 40.0,
    seed: int = 0, mean: float | None = None, std: float | None = None,
) -> dict[str, Any]:
    """Per-node hetero-leak init: tau log-uniform in [tau_lo, tau_hi].

    The reservoir clockwork-RNN style: half the nodes become fast taps
    (small tau), half slow traces (large tau). Converts tau -> leak rate
    (1/tau) -> softplus-inverse raw_leak, so ``softplus(raw_leak)`` after
    the override equals the desired leak rate per node.

    Args:
        stage: A ``DifferentialStage`` (must have ``raw_leak``).
        tau_lo / tau_hi: Time-constant range (samples). Must satisfy
            ``0 < tau_lo <= tau_hi`` and both ``tau_lo * dt <= 1`` for the
            Heun stability to remain valid (we do NOT clip; we just warn
            at audit time).
        seed: RNG seed for the tau draw.
        mean / std: Optional normal-noise overlay added to the raw_leak
            in addition to the log-uniform draw; ``None`` = no overlay.

    Returns:
        Dict with the override log (tau stats + leak stats) for the
        probe-row audit trail. ``cell_library.py`` init defaults are
        NEVER touched.
    """
    if not hasattr(stage, "raw_leak"):
        raise ValueError(
            "hetero_leak_init requires a programmable leak (stage.raw_leak)."
        )
    if tau_lo <= 0 or tau_hi < tau_lo:
        raise ValueError(
            f"need 0 < tau_lo <= tau_hi, got tau_lo={tau_lo}, tau_hi={tau_hi}"
        )
    n_nodes = int(stage.raw_leak.shape[0])
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    log_tau = torch.empty(n_nodes).uniform_(0.0, 1.0, generator=gen)
    log_tau = log_tau * (math.log(tau_hi) - math.log(tau_lo)) + math.log(tau_lo)
    tau = log_tau.exp()
    leak = 1.0 / tau  # rate per sample
    # softplus inverse: y = log(exp(x) - 1) for y > 0, ~ y for small y
    safe_leak = leak.clamp_min(1e-6)
    raw_leak = safe_leak + torch.log(-torch.expm1(-safe_leak))  # numerically stable
    if mean is not None or std is not None:
        m = 0.0 if mean is None else float(mean)
        s = 0.0 if std is None else float(std)
        raw_leak = raw_leak + torch.randn(n_nodes, generator=gen) * s + m
    with torch.no_grad():
        stage.raw_leak.data.copy_(raw_leak.to(stage.raw_leak.device))
    stage.leak_mode = "programmable"
    return {
        "leak_mode": "hetero-log-uniform",
        "tau_lo": float(tau_lo),
        "tau_hi": float(tau_hi),
        "tau_median": float(tau.median().item()),
        "tau_min": float(tau.min().item()),
        "tau_max": float(tau.max().item()),
        "raw_leak_mean": float(raw_leak.mean().item()),
        "raw_leak_std": float(raw_leak.std().item()),
    }


def build_small_world_narma_preset(
    order: int, *, hidden_dim: int = 25, num_steps_per_sample: int = 8,
    t_span: float = 1.0, small_world_k: int = 4, small_world_p: float = 0.2,
    small_world_seed: int = 1, core_refresh_interval: int = 0,
    leak_constant: float | None = None,
) -> dict:
    """NARMA preset with ``hidden_family='small_world'`` (Step 4 grid axis).

    Mirrors ``config.make_narma_preset`` but flips the hidden topology
    from the canonical 5x5 torus to a Watts-Strogatz small-world graph
    (matches ``config.make_friedman2_preset``'s small_world path).

    The plan sec. 7 calls for ``random-sign sparse fan-out`` to pair with
    small-world; that requires per-edge boundary input gains and is
    deferred to a follow-up. This helper only flips the topology so the
    E0 sweep can isolate the small-world effect first.
    """
    if small_world_k < 2 or small_world_k % 2 != 0 or small_world_k >= 16:
        raise ValueError(
            f"small_world_k must be even, >=2, and <16, got {small_world_k}"
        )
    if not (0.0 <= small_world_p <= 1.0):
        raise ValueError(
            f"small_world_p must be in [0, 1], got {small_world_p}"
        )
    preset = _make_narma_preset_factory(
        order=order, hidden_dim=hidden_dim, num_stages=1,
        num_steps_per_sample=num_steps_per_sample,
        output_ode_count=1, bidirectional=False,
        edge_repeats=1, t_span=t_span,
        core_refresh_interval=core_refresh_interval,
        leak_constant=leak_constant,
    )
    side = int(round(hidden_dim ** 0.5))
    if side * side == hidden_dim:
        # Override the torus kwarg with small-world kwargs.
        preset["stages"][0]["hidden_family"] = "small_world"
        preset["stages"][0]["hidden_kwargs"] = {
            "k": int(small_world_k),
            "p": float(small_world_p),
            "seed": int(small_world_seed),
            "bidirectional": False,
        }
    else:
        # Non-square width: still build small-world; same kwargs apply
        # (the topology builder accepts small_world for any num_hidden).
        preset["stages"][0]["hidden_family"] = "small_world"
        preset["stages"][0]["hidden_kwargs"] = {
            "k": int(small_world_k),
            "p": float(small_world_p),
            "seed": int(small_world_seed),
            "bidirectional": False,
        }
    return preset


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def write_revised_csv(
    path: Path, rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    """Write probe rows (lists + dicts serialised to JSON strings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    serialised: list[dict[str, Any]] = []
    for r in rows:
        s = {}
        for k, v in r.items():
            if isinstance(v, list):
                s[k] = json.dumps(v)
            elif isinstance(v, dict):
                s[k] = json.dumps(v)
            else:
                s[k] = v
        serialised.append(s)
    if fieldnames is None:
        seen: list[str] = []
        for r in serialised:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in serialised:
            w.writerow(r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--order", type=int, choices=[10, 20], default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, default=Path("./output/narma_revised"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Revised NARMA-plan probes: S1-extended memory curves, "
            "tuned-ESN calibration, hetero-leak init, small-world "
            "preset."
        ),
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- memory (S1-extended) ---
    p_mem = sub.add_parser(
        "memory",
        help="Run the S1-extended memory probe on a fabric checkpoint.",
    )
    _add_common(p_mem)
    p_mem.add_argument("--ckpt", type=Path, required=True,
                       help="Path to a fabric .pt checkpoint.")
    p_mem.add_argument("--cell-library", default="tanh_free")
    p_mem.add_argument("--core-refresh-interval", type=int, default=0)
    p_mem.add_argument("--freeze-read", action="store_true")
    p_mem.add_argument("--leak-constant", type=float, default=None)
    p_mem.add_argument("--input-scale", type=float, default=1.0)
    p_mem.add_argument("--n-streams", type=int, default=4)
    p_mem.add_argument("--train-samples", type=int, default=2500)

    # --- esn-cal ---
    p_esn = sub.add_parser(
        "esn-cal",
        help="Tuned-ESN calibration grid sweep.",
    )
    _add_common(p_esn)
    p_esn.add_argument("--max-corners", type=int, default=0,
                       help="0 = full grid (288 corners). Positive N takes the "
                            "first N corners for a smoke run.")
    p_esn.add_argument("--train-samples", type=int, default=2500)
    p_esn.add_argument("--spectral-radius", default="0.7,0.9,1.0,1.1")
    p_esn.add_argument("--leak", default="0.3,0.5,0.8,1.0")
    p_esn.add_argument("--n-reservoir", default="25,50,100")
    p_esn.add_argument("--input-scaling", default="0.2,0.5,1.0")
    p_esn.add_argument("--ridge-lambda", default="1e-4,1e-2")

    # --- hetero-leak-init ---
    p_hl = sub.add_parser(
        "hetero-leak-init",
        help="Build a fresh NARMA net + apply hetero-leak init (audit smoke).",
    )
    _add_common(p_hl)
    p_hl.add_argument("--hidden-dim", type=int, default=25)
    p_hl.add_argument("--tau-lo", type=float, default=1.0)
    p_hl.add_argument("--tau-hi", type=float, default=40.0)
    p_hl.add_argument("--t-span", type=float, default=1.0)
    p_hl.add_argument("--num-steps", type=int, default=8)

    args = parser.parse_args(argv)
    if args.order != 10:
        parser.error("revised-plan probes are pre-registered for --order 10 only")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "memory":
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        net, t_span_used, num_steps_used = ne._build_fabric_net(
            order=args.order, seed=args.seed, freeze_read=args.freeze_read,
            t_span=None, num_steps=None, cell_library=args.cell_library,
            core_refresh_interval=args.core_refresh_interval,
            leak_constant=args.leak_constant,
            compile_sequence=False,
        )
        net.load_state_dict(ckpt["model_state"], strict=True)
        net.to(args.device)
        u_train_raw, y_train_raw = ne._gen_narma_train_streams(
            args.order, args.seed, args.n_streams, args.train_samples,
        )
        u_train = ne._scale_drive(
            u_train_raw[0], bipolar=True, order=args.order,
            input_scale=args.input_scale,
        ).to(args.device)
        y_train = y_train_raw[0].to(args.device)
        config_tag = (
            f"ckpt{Path(args.ckpt).name}_order{args.order}_seed{args.seed}_"
            f"{args.cell_library}_k{args.core_refresh_interval}_"
            f"freeze{int(args.freeze_read)}_leak{args.leak_constant}_"
            f"drive{args.input_scale:g}_washout{npr.PROBE_WASHOUT}"
        )
        row = s1_extended_probe(
            net, u_train, y_train, device=args.device, config_tag=config_tag,
        )
        row_d = asdict(row)
        print(f"  S1-EXTENDED probe on {args.ckpt}:")
        for k, v in row_d.items():
            print(f"    {k}={v}")
        # Decision: any non-zero IPC-2 / y-MC / product presence?
        present_bits = []
        if row.ipc2_present:
            present_bits.append("IPC-2")
        if row.y_mc_present:
            present_bits.append("y-MC")
        if row.product_present:
            present_bits.append("NAR10-product")
        decision = " + ".join(present_bits) if present_bits else "ABSENT"
        print(f"  DECISION: {decision} (presence threshold "
              f">{IPC2_PRESENCE_THRESHOLD} for IPC-2 and product pairs, "
              f">{Y_MC_PRESENCE_THRESHOLD} for y-MC)")
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
            "washout": npr.PROBE_WASHOUT,
            "row": row_d,
            "decision": decision,
            "thresholds": {
                "ipc2_presence": IPC2_PRESENCE_THRESHOLD,
                "y_mc_presence": Y_MC_PRESENCE_THRESHOLD,
            },
            "product_pairs": list(NAR10_PRODUCT_PAIRS),
        }
        (args.output / "s1_extended.json").write_text(json.dumps(out, indent=2))
        write_revised_csv(args.output / "s1_extended.csv", [row_d])
        return 0

    if args.mode == "esn-cal":
        sr = tuple(float(s) for s in args.spectral_radius.split(",") if s.strip())
        lk = tuple(float(s) for s in args.leak.split(",") if s.strip())
        nr = tuple(int(s) for s in args.n_reservoir.split(",") if s.strip())
        ins = tuple(float(s) for s in args.input_scaling.split(",") if s.strip())
        rl = tuple(float(s) for s in args.ridge_lambda.split(",") if s.strip())
        t0 = time.time()
        rows = tuned_esn_calibration(
            order=args.order, seed=args.seed, device=args.device,
            spectral_radius=sr, leak=lk, n_reservoir=nr,
            input_scaling=ins, ridge_lambda=rl,
            max_corners=args.max_corners,
            train_samples=args.train_samples,
        )
        elapsed = time.time() - t0
        rows_d = [asdict(r) for r in rows]
        write_revised_csv(args.output / "esn_cal.csv", rows_d)
        # Decision: best NRMSE across corners.
        if rows:
            best = min(rows, key=lambda r: r.nrmse)
            band_stands = best.nrmse < ESN_CAL_BAND_STANDS_BELOW
            verdict = (
                f"BAND-STANDS (best NRMSE {best.nrmse:.4f} < "
                f"{ESN_CAL_BAND_STANDS_BELOW})"
                if band_stands else
                f"BAND-MISCALIBRATED (best NRMSE {best.nrmse:.4f} >= "
                f"{ESN_CAL_BAND_STANDS_BELOW}; consider retiring NARMA-10)"
            )
        else:
            verdict = "NO-CORNERS-RAN"
        print(
            f"  Tuned-ESN calibration: {len(rows)} corners in {elapsed:.1f}s\n"
            f"  DECISION: {verdict}"
        )
        for r in rows_d[:10]:
            print(f"    sr={r['spectral_radius']:.2f} leak={r['leak']:.2f} "
                  f"nr={r['n_reservoir']:3d} ins={r['input_scaling']:.2f} "
                  f"rl={r['ridge_lambda']:.0e} -> "
                  f"NRMSE={r['nrmse']:.4f} R2={r['r2']:.4f} "
                  f"MC={r['mc_total']:.2f} PR={r['state_pr']:.2f}")
        (args.output / "esn_cal.json").write_text(json.dumps({
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "n_corners": len(rows),
            "elapsed_s": elapsed,
            "thresholds": {
                "band_stands_below": ESN_CAL_BAND_STANDS_BELOW,
            },
            "grids": {
                "spectral_radius": list(sr),
                "leak": list(lk),
                "n_reservoir": list(nr),
                "input_scaling": list(ins),
                "ridge_lambda": list(rl),
            },
            "best_nrmse": (
                float(best.nrmse) if rows else None
            ),
            "decision": verdict,
            "rows": rows_d,
        }, indent=2))
        return 0

    if args.mode == "hetero-leak-init":
        net, t_span_used, num_steps_used = ne._build_fabric_net(
            order=args.order, seed=args.seed, freeze_read=False,
            t_span=args.t_span, num_steps=args.num_steps,
            cell_library="tanh_free",
            core_refresh_interval=0,
            leak_constant=None,
            compile_sequence=False,
            hidden_dim=args.hidden_dim,
        )
        log = hetero_leak_init(
            net.core.stages[0], tau_lo=args.tau_lo, tau_hi=args.tau_hi,
            seed=args.seed,
        )
        # Audit: forward a small stream and check the states stay finite.
        u_smoke, _ = ne._gen_narma_train_streams(
            args.order, args.seed, 1, 200,
        )
        u_smoke = ne._scale_drive(
            u_smoke[0], bipolar=True, order=args.order, input_scale=1.0,
        )
        with torch.no_grad():
            states = _collect_states_single_stage(net, u_smoke)
        sat_max = float(states.abs().max().item())
        x_max = float(net.core.stages[0].x_max)
        sat_ratio = sat_max / x_max if x_max > 0 else float("nan")
        print(
            f"  HETERO-LEAK init: tau_lo={args.tau_lo} tau_hi={args.tau_hi} "
            f"seed={args.seed} -> log: {log}\n"
            f"  smoke forward: T={states.shape[0]} N={states.shape[1]} "
            f"sat_max_ratio={sat_ratio:.3f} (x_max={x_max})"
        )
        out = {
            "tau_lo": args.tau_lo,
            "tau_hi": args.tau_hi,
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "hetero_log": log,
            "smoke": {
                "T": int(states.shape[0]),
                "N": int(states.shape[1]),
                "sat_max_ratio": float(sat_ratio),
                "x_max": float(x_max),
                "all_finite": bool(torch.isfinite(states).all().item()),
            },
        }
        (args.output / "hetero_leak_init.json").write_text(
            json.dumps(out, indent=2),
        )
        return 0

    parser.error(f"unknown mode {args.mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
