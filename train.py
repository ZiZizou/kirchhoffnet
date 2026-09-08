"""Loss function, regularizers, and training loop for the differential KirchhoffNet.

The loss combines:
  - task loss (MSE / BCE / MAE / residual+solution)
  - edge_gate regularizer (CP): Σ_e σ(z_logits) — active edge count proxy
  - power regularizer (CP): Σ_e z_e·m_e·gm_e — static power proxy
  - rail regularizer: ReLU² quadratic barrier for trajectory excursions beyond x_max

Regularizers are scheduled with a staged warm-up: ``[0, W)`` → off,
``[W, W+A)`` → linear anneal from 0 to full value,
``[W+A, ∞)`` → full value (RR-A + CP).
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

from config import (
    LAMBDAS,
    OPTIM,
    SCHEDULE_FOUR_PHASE,
    SCHEDULE_THREE_PHASE,
    SOLVER,
    TAU,
    VARIATION,
)
from sim_context import SimContext, sample_random_context
from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
from cell_library import (
    AntiParallelFreeTanhLibrary,
    FreeTanhLibrary,
    RealisticTanhLibrary,
    RealisticTanhUpgradeLibrary,
    SimpleEdgeLibrary,
)
from io_mapper import OutputAffine


__all__ = [
    "compute_loss",
    "compute_solver_loss",
    "residual_loss",
    "solution_loss",
    "tau_for_epoch",
    "reg_schedule",
    "apply_reg_schedule",
    "phase_for_epoch",
    "phase_boundaries",
    "three_phase_tau",
    "three_phase_lambdas",
    "four_phase_boundaries",
    "phase_for_epoch_four",
    "four_phase_tau",
    "four_phase_lambdas",
    "make_optimizer",
    "apply_ablation",
    "train_epoch",
    "budget_frac_for_epoch",
    "budget_temperature_for_epoch",
]


# ---------- regularizers ----------

def _stage_edge_gates(stage) -> torch.Tensor:
    """Edge gate values z_e = σ(z_logits), shape [E]."""
    return torch.sigmoid(stage.z_logits)


def _stage_node_gates(stage) -> torch.Tensor:
    """Node gate values u_j = σ(u_logits), shape [N].

    DEPRECATED (deprecate-node-gates): node gates are no longer used in
    the forward pass, in any regularizer, or in pruning. Returns an
    all-ones tensor so any stray caller gets a no-op result instead of
    corrupting the dynamics.
    """
    import warnings as _warnings
    _warnings.warn(
        "train._stage_node_gates() is deprecated (deprecate-node-gates); "
        "node gates are no longer used. Returns an all-ones tensor.",
        DeprecationWarning,
        stacklevel=2,
    )
    return torch.ones(stage.num_nodes, device=stage.u_logits.device,
                       dtype=stage.u_logits.dtype)


def _stage_rail_loss(stage, traj: torch.Tensor) -> torch.Tensor:
    """Mean over (batch, node, time) of ReLU^2(|x| - x_max).

    ReLU² barrier (three-phase-schedule/rail-loss-fix): exactly zero loss
    and zero gradient when all voltages are within [-x_max, x_max].
    Quadratic growth beyond bounds gives a smooth, differentiable penalty
    for voltage excursions. Replaces the previous softplus formulation
    whose non-zero floor (softplus(-x_max) ≈ 0.049 at x_max=3.0) dragged
    the task loss by ~16% even when no voltage was out of bounds.
    """
    excess = F.relu(traj.abs() - stage.x_max)
    return excess.pow(2).mean()


def _stage_tanh_sat_loss(stage, traj: torch.Tensor) -> torch.Tensor:
    """Mean over (batch, edges, time) of tanh(u)^2 for FreeTanhLibrary stages.

    Penalizes edges whose tanh input magnitude is large (operating in the
    saturated region of tanh, where the cell output is near ±Isat and small
    input changes produce negligible output changes). Returns zero for stages
    whose cell library is not a ``FreeTanhLibrary``.

    The tanh input ``u`` is recomputed from the library's effective parameters
    and the per-step trajectory voltages:

        u = gm * (s * (A * Vsrc - B * Vdest) + theta)

    Args:
        stage: ``DifferentialStage`` with a ``cell_lib`` attribute.
        traj: ``[batch, num_nodes, num_steps+1]`` node-voltage trajectory.

    Returns:
        Scalar tensor equal to ``tanh(u).pow(2).mean()`` over (batch, edges,
        time). Returns 0 for non-``FreeTanhLibrary`` stages.
    """
    lib = stage.cell_lib
    if not isinstance(lib, FreeTanhLibrary):
        return traj.new_zeros(())
    if stage.src.numel() == 0:
        return traj.new_zeros(())

    A = F.softplus(lib.a_raw).unsqueeze(0).unsqueeze(-1)        # [1, E, 1]
    B = F.softplus(lib.b_raw).unsqueeze(0).unsqueeze(-1)        # [1, E, 1]
    s = torch.sign(lib.s_raw)
    s_ste = s + lib.s_raw - lib.s_raw.detach()                  # [E]
    gm_uncl = torch.sigmoid(lib.gm_raw)
    gm = lib.gm_min + (lib.gm_max - lib.gm_min) * gm_uncl       # [E]
    gm_e = gm.unsqueeze(0).unsqueeze(-1)                        # [1, E, 1]

    # traj: [B, N, T] -> gather per-edge src/dst voltages -> [B, E, T]
    x_src = traj.index_select(1, stage.src)
    x_dst = traj.index_select(1, stage.dst)

    pre = s_ste.unsqueeze(0).unsqueeze(-1) * (A * x_src - B * x_dst)
    if lib._bias_enabled:
        theta = lib.theta_raw.unsqueeze(0).unsqueeze(-1)        # [1, E, 1]
        pre = pre + theta
    u = gm_e * pre                                              # [B, E, T]

    return torch.tanh(u).pow(2).mean()


def _stage_sparsity_loss(stage, traj: torch.Tensor) -> torch.Tensor:
    """Mean |cell device param| over FreeTanhLibrary families (weight sparsity).

    L1 weight-sparsity penalty on the tanh_free device parameters. For each
    stage, walks ``cell_lib``, ``boundary_cell_lib``, ``output_ode_cell_lib``,
    ``ref_cell_lib``, and ``readout_sense_cell_lib`` (when each is a
    ``FreeTanhLibrary``) and averages ``|p|`` over the non-sign parameters
    (``s_raw`` is a discrete ±1 with STE and is excluded). Returns zero for
    stages with no FreeTanh library.

    Complements ``edge_gate`` (active-edge *count*) by applying magnitude
    pressure to the device weights themselves. ``reg_scale`` from the caller
    applies the staged warm-up.
    """
    seed = traj.new_zeros(())
    parts = []
    for lib in (
        getattr(stage, "cell_lib", None),
        getattr(stage, "boundary_cell_lib", None),
        getattr(stage, "output_ode_cell_lib", None),
        getattr(stage, "ref_cell_lib", None),
        # B2: shared-sense readout family gets the same magnitude pressure as
        # the legacy mesh so the optimizer can prune senses alongside
        # boundary / ref / mesh devices.
        getattr(stage, "readout_sense_cell_lib", None),
    ):
        if lib is None or not isinstance(lib, FreeTanhLibrary):
            continue
        total = None
        num = 0
        for name, p in lib.named_parameters(recurse=False):
            if name == "s_raw":
                continue
            v = p.abs().sum()
            total = v if total is None else total + v
            num += int(p.numel())
        if total is not None:
            parts.append(total / max(1, num))
    if not parts:
        return seed
    out = parts[0]
    for v in parts[1:]:
        out = out + v
    return out / len(parts)


def _stage_entropy_loss(stage, traj: torch.Tensor) -> torch.Tensor:
    """Mean binary entropy of edge gate probabilities (crisp-selection pressure).

    ``H(p) = -p·log2(p) - (1-p)·log2(1-p)`` averaged over the soft gate
    values ``p = σ(z_logits)`` of the core, boundary, and temporal-readout
    edges. Gate entropy is maximized at ``p = 0.5`` (all gates half-open) and
    goes to zero as gates commit to 0 or 1. Penalizing ``H`` pushes the gates
    toward crisp 0/1 selection, which pairs with ``edge_gate`` for pruning.
    """
    seed = traj.new_zeros(())
    gates = []
    if getattr(stage, "z_logits", None) is not None:
        gates.append(torch.sigmoid(stage.z_logits))
    if getattr(stage, "boundary_z_logits", None) is not None:
        gates.append(torch.sigmoid(stage.boundary_z_logits))
    if getattr(stage, "output_ode_z_logits", None) is not None:
        gates.append(torch.sigmoid(stage.output_ode_z_logits))
    # F3 parity: shared-sense gates must also receive crisp-selection
    # entropy pressure so they commit alongside the other edge families.
    if getattr(stage, "readout_sense_z_logits", None) is not None:
        gates.append(torch.sigmoid(stage.readout_sense_z_logits))
    if not gates:
        return seed
    eps = 1e-6
    h = seed
    count = 0
    for p in gates:
        pc = p.clamp(eps, 1.0 - eps)
        h = h + (-(pc * torch.log2(pc)) - ((1.0 - pc) * torch.log2(1.0 - pc))).mean()
        count += 1
    return h / count


def _device_l2_penalty(net) -> torch.Tensor:
    """Scale-invariant L2 penalty over the ``tanh_free`` device-family params.

    Walks the unwrapped KirchhoffNetWithIO and sums the squared raw params of
    every partition the user asked us to keep small:

      - ``FreeTanhLibrary`` cell params: ``a_raw``, ``b_raw``, ``gm_raw``,
        ``isat_raw``, ``theta_raw``, ``g_resistive_raw`` on ``cell_lib``,
        ``boundary_cell_lib``, ``output_ode_cell_lib``, ``ref_cell_lib``,
        and ``readout_sense_cell_lib`` (when each exists and is a
        ``FreeTanhLibrary``). ``s_raw`` is skipped because it is a discrete
        ±1 sign with STE; penalizing it would push ``sign(0) = 0`` and kill
        edges.
      - Shared-sense readout scalars + dense crossbar: ``readout_sense_z_logits``
        and ``raw_vref_sense`` (treated as plain L2 targets, same as the
        OutputAffine / VCA group below). The dense ``readout_crossbar_W`` is
        included as a plain L2 target for the same reason.
      - VCA params: ``vca_W``, ``vca_W_core``, ``vca_v_boundary``,
        ``vca_v_readout``, ``vca_v_core``.
      - Interstage residual transfer: ``StageTransfer.residual_w1/w2/w3/vth``
        (only present when the activation is a residual variant).
      - Temporal-readout decoder: ``OutputAffine.gain/bias/tanh_gain/
        relu_gain/relu_threshold`` (only when the readout uses OutputAffine).

    Returns the *mean* of ``p.pow(2)`` over the union of penalized tensors
    (so the pressure per parameter is independent of how many params the
    active configuration has). Returns 0 when no penalized params exist.
    Unwraps ``DataParallel`` and the ``KirchhoffNetNoiseWrapper`` so the
    same call site works with or without multi-GPU / noise-aware training.
    """
    raw = net
    if isinstance(raw, torch.nn.DataParallel):
        raw = raw.module
    if hasattr(raw, "base") and hasattr(raw, "_stage_noise_std"):
        raw = raw.base

    sum_sq = None
    num = 0
    device = None
    dtype = None

    def _accum(p: torch.Tensor) -> None:
        nonlocal sum_sq, num, device, dtype
        if p is None:
            return
        s = p.pow(2).sum()
        sum_sq = s if sum_sq is None else sum_sq + s
        num += int(p.numel())
        if device is None:
            device = p.device
            dtype = p.dtype

    stages = getattr(getattr(raw, "core", raw), "stages", [])
    for stage in stages:
        for lib in (
            getattr(stage, "cell_lib", None),
            getattr(stage, "boundary_cell_lib", None),
            getattr(stage, "output_ode_cell_lib", None),
            getattr(stage, "ref_cell_lib", None),
            getattr(stage, "readout_sense_cell_lib", None),
        ):
            if isinstance(lib, FreeTanhLibrary):
                for name, p in lib.named_parameters():
                    if name == "s_raw":
                        continue
                    _accum(p)
        # Shared-sense readout scalars + the dense crossbar (plain L2
        # targets, like the OutputAffine / VCA params below).
        _accum(getattr(stage, "readout_sense_z_logits", None))
        _accum(getattr(stage, "raw_vref_sense", None))
        _accum(getattr(stage, "readout_crossbar_W", None))
        for vca_name in (
            "vca_W",
            "vca_W_core",
            "vca_v_boundary",
            "vca_v_readout",
            "vca_v_core",
        ):
            _accum(getattr(stage, vca_name, None))

    transfers = getattr(getattr(raw, "core", raw), "transfers", [])
    for tr in transfers:
        for name in ("residual_w1", "residual_w2", "residual_w3", "residual_vth"):
            _accum(getattr(tr, name, None))

    om = getattr(raw, "output_mapper", None)
    if isinstance(om, OutputAffine):
        for name in ("gain", "bias", "tanh_gain", "relu_gain", "relu_threshold"):
            _accum(getattr(om, name, None))

    if sum_sq is None or num == 0:
        try:
            ref = next(raw.parameters())
            return torch.zeros((), device=ref.device, dtype=ref.dtype)
        except StopIteration:
            return torch.zeros(())
    return sum_sq / float(num)


def _compute_regularizers(
    net: KirchhoffNetWithIO | KirchhoffNet,
    trajs: list[torch.Tensor],
    lambdas: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Shared regularizer computation: edge_gate, rail, tanh_sat, sparsity, entropy.

    Complexity terms:
      - edge_gate   : Σ_e σ(z_logits) over core, boundary, ref, and
                       temporal-readout (output_ode) edges (active edge count).
      - rail        : ReLU²(|x| - x_max).mean()        (voltage excursion)
      - tanh_sat    : tanh(u)².mean()                  (FreeTanhLibrary saturation)
      - sparsity    : mean |device cell param| over FreeTanhLibrary families
      - entropy     : mean binary entropy of the soft edge gates

    Returns ``(loss_edge_gate, loss_rail, loss_tanh_sat,
    loss_sparsity, loss_entropy)``.
    """
    stages = net.core.stages if isinstance(net, KirchhoffNetWithIO) else net.stages
    loss_edge_gate = loss_rail = loss_tanh_sat = trajs[0].new_zeros(())
    loss_sparsity = loss_entropy = trajs[0].new_zeros(())
    for stage, traj in zip(stages, trajs):
        z = _stage_edge_gates(stage)

        loss_edge_gate = loss_edge_gate + z.sum()
        # Include boundary-fan-out edge gates when present so the
        # structural regularizer also prunes the boundary read-in edges.
        if getattr(stage, "boundary_z_logits", None) is not None:
            loss_edge_gate = loss_edge_gate + torch.sigmoid(
                stage.boundary_z_logits
            ).sum()
        # Include temporal-readout edge gates when present so the
        # structural regularizer also prunes the readout OTA edges.
        if getattr(stage, "output_ode_z_logits", None) is not None:
            loss_edge_gate = loss_edge_gate + torch.sigmoid(
                stage.output_ode_z_logits
            ).sum()
        # Include shared-sense edge gates (F3 parity with the legacy mesh):
        # the sense bank has its own per-sense z_logits that must receive the
        # same L1 pressure so the optimizer can prune senses alongside the
        # boundary / mesh families.
        if getattr(stage, "readout_sense_z_logits", None) is not None:
            loss_edge_gate = loss_edge_gate + torch.sigmoid(
                stage.readout_sense_z_logits
            ).sum()
        loss_rail = loss_rail + _stage_rail_loss(stage, traj)
        loss_tanh_sat = loss_tanh_sat + _stage_tanh_sat_loss(stage, traj)
        loss_sparsity = loss_sparsity + _stage_sparsity_loss(stage, traj)
        loss_entropy = loss_entropy + _stage_entropy_loss(stage, traj)

    return (loss_edge_gate, loss_rail, loss_tanh_sat,
            loss_sparsity, loss_entropy)


# ---------- regularizer warm-up schedule (RR-A + CP) ----------

_REG_KEYS = (
    "edge_gate",
    "tanh_sat",
    "sparsity",
    "entropy",
)


def reg_schedule(epoch: int, *, warmup: int | None = None, anneal: int | None = None) -> float:
    """Return a scalar in [0, 1] scaling the staged regularizers (RR-A).

    - epoch < warmup: 0.0 (network learns freely).
    - warmup <= epoch < warmup + anneal: linear ramp from 0 → 1.
    - epoch >= warmup + anneal: 1.0 (full penalty).

    Defaults: ``warmup = OPTIM["reg_warmup_epochs"] = 50``,
    ``anneal = OPTIM["reg_anneal_epochs"] = 50``.

    Note: ``rail`` is NOT in ``_REG_KEYS``. The rail regularizer is a safety
    voltage clamp on differential node states (clamps x_j to ±x_max via
    soft sigmoid), not a structural complexity regularizer.
    """
    if warmup is None:
        warmup = int(OPTIM.get("reg_warmup_epochs", 50))
    if anneal is None:
        anneal = int(OPTIM.get("reg_anneal_epochs", 50))
    if epoch < warmup:
        return 0.0
    if epoch >= warmup + anneal:
        return 1.0
    return float(epoch - warmup + 1) / float(max(1, anneal))


def apply_reg_schedule(lambdas: dict, epoch: int, **kw) -> dict:
    """Return a copy of ``lambdas`` with each ``_REG_KEYS`` entry scaled by
    ``reg_schedule(epoch)``. ``rail`` is left untouched."""
    scale = reg_schedule(epoch, **kw)
    out = dict(lambdas)
    for k in _REG_KEYS:
        if k in out:
            out[k] = float(out[k]) * scale
    return out


# ---------- three-phase schedule (three-phase-schedule plan) ----------

def phase_boundaries(total_epochs: int, schedule_cfg: dict | None = None) -> tuple[int, int, int]:
    """Compute the (a_end, b_end, c_end) epoch boundaries for the three-phase schedule.

    The three phases have fixed epoch counts derived from the schedule
    config's phase fractions (frac_a/frac_b/frac_c). Returns the boundary
    epochs in the closed interval [0, total_epochs]. Phase C ends at
    total_epochs (i.e. c_end == total_epochs).

    Args:
        total_epochs: Total epoch budget.
        schedule_cfg: A ``SCHEDULE_THREE_PHASE`` dict. Defaults to the
            module-level ``SCHEDULE_THREE_PHASE``.

    Returns:
        ``(a_end, b_end, c_end)`` epoch boundaries with
        ``0 <= a_end <= b_end <= c_end == total_epochs``.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_THREE_PHASE
    frac_a = float(schedule_cfg.get("frac_a", 0.30))
    frac_b = float(schedule_cfg.get("frac_b", 0.40))
    frac_c = float(schedule_cfg.get("frac_c", 0.30))
    total_frac = frac_a + frac_b + frac_c
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"three_phase schedule fractions must sum to 1.0, got {total_frac} "
            f"(frac_a={frac_a}, frac_b={frac_b}, frac_c={frac_c})"
        )
    a_end = int(round(total_epochs * frac_a))
    b_end = int(round(total_epochs * (frac_a + frac_b)))
    c_end = total_epochs
    if a_end < 1:
        a_end = 1
    if b_end < a_end + 1:
        b_end = a_end + 1
    return (a_end, b_end, c_end)


def phase_for_epoch(epoch: int, total_epochs: int, schedule_cfg: dict | None = None) -> str:
    """Return the active phase name (``'A'``, ``'B'``, or ``'C'``) for a given epoch.

    Boundary semantics:
      - Phase A: ``0 <= epoch < a_end``
      - Phase B: ``a_end <= epoch < b_end``
      - Phase C: ``b_end <= epoch < c_end`` (c_end == total_epochs)

    Pruning happens at the epoch ``b_end`` (the first epoch of Phase C is
    the retrain of the freshly-pruned network).
    """
    a_end, b_end, _ = phase_boundaries(total_epochs, schedule_cfg)
    if epoch < a_end:
        return "A"
    if epoch < b_end:
        return "B"
    return "C"


def three_phase_tau(epoch: int, total_epochs: int, schedule_cfg: dict | None = None) -> float:
    """Tau value for the current epoch under the three-phase schedule.

    Phase A: fixed at ``tau_a`` (default 1.0).
    Phase B: anneals from ``tau_b_init`` to ``tau_b_final`` (default 1.0→0.6).
    Phase C: anneals from ``tau_c_init`` to ``tau_c_final`` (default 0.6→0.1).

    Reuses the existing ``tau_for_epoch`` exponential-decay + linear-hardening
    behavior, scoped to each phase's local epoch window.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_THREE_PHASE
    a_end, b_end, _ = phase_boundaries(total_epochs, schedule_cfg)

    if epoch < a_end:
        return float(schedule_cfg.get("tau_a", 1.0))

    if epoch < b_end:
        tau_init = float(schedule_cfg.get("tau_b_init", 1.0))
        tau_final = float(schedule_cfg.get("tau_b_final", 0.6))
        local_epoch = epoch - a_end
        local_total = max(1, b_end - a_end)
        return tau_for_epoch(local_epoch, total_epochs=local_total,
                             tau_init=tau_init, tau_final=tau_final)

    tau_init = float(schedule_cfg.get("tau_c_init", 0.6))
    tau_final = float(schedule_cfg.get("tau_c_final", 0.1))
    local_epoch = epoch - b_end
    local_total = max(1, total_epochs - b_end)
    return tau_for_epoch(local_epoch, total_epochs=local_total,
                         tau_init=tau_init, tau_final=tau_final)


def three_phase_lambdas(
    epoch: int,
    total_epochs: int,
    base_lambdas: dict,
    schedule_cfg: dict | None = None,
) -> dict:
    """Effective lambda dict for the current epoch under the three-phase schedule.

    Phase A: all structural lambdas (sparsity, edge_gate, node_gate, power,
      capacitance) are zeroed. ``rail`` is left at its base value as a
      safety net. No warmup.

    Phase B: the Phase B target lambdas (``lambdas_b`` in the schedule
      config) are linearly ramped from 0 to full over the first
      ``warmup_frac_b`` of Phase B's epoch window, then held at full.

    Phase C: the Phase C retrain lambdas (``lambdas_c``) are applied at
      full strength from the start of the phase (no warmup — pruning has
      already happened at the B→C boundary, the surviving network should
      re-fit quickly).

    All returned dicts include ``rail`` (preserved from ``base_lambdas``).
    Returns a fresh dict; ``base_lambdas`` is not mutated.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_THREE_PHASE
    a_end, b_end, _ = phase_boundaries(total_epochs, schedule_cfg)
    rail_val = float(base_lambdas.get("rail", 0.0))
    out = {"rail": rail_val}

    if epoch < a_end:
        # Phase A: zero all structural lambdas. Rail is the only active term.
        for k in _REG_KEYS:
            out[k] = 0.0
        return out

    if epoch < b_end:
        # Phase B: ramp lambdas_b from 0 to full over warmup window.
        lambdas_b = schedule_cfg.get("lambdas_b", {})
        warmup_frac = float(schedule_cfg.get("warmup_frac_b", 1.0 / 6.0))
        local_epoch = epoch - a_end
        local_total = max(1, b_end - a_end)
        warmup_epochs = max(1, int(round(warmup_frac * local_total)))
        if local_epoch < warmup_epochs:
            scale = float(local_epoch + 1) / float(warmup_epochs)
        else:
            scale = 1.0
        for k in _REG_KEYS:
            out[k] = float(lambdas_b.get(k, 0.0)) * scale
        return out

    # Phase C: apply lambdas_c at full strength, no warmup.
    lambdas_c = schedule_cfg.get("lambdas_c", {})
    for k in _REG_KEYS:
        out[k] = float(lambdas_c.get(k, 0.0))
    return out


# ---------- four-phase schedule (four-phase-redesign plan) ----------

def four_phase_boundaries(
    total_epochs: int, schedule_cfg: dict | None = None
) -> tuple[int, int, int, int]:
    """Compute (a_end, b1_end, b2_end, c_end) epoch boundaries for the
    four-phase schedule (four-phase-redesign/Phase 3b).

    Phase fractions must sum to 1.0 (frac_a + frac_b1 + frac_b2 + frac_c = 1).
    Each end is computed by rounding total_epochs * cumulative_frac, with
    the constraint that adjacent phases differ by at least 1 epoch.

    Returns (a_end, b1_end, b2_end, c_end) with
    ``0 < a_end < b1_end < b2_end < c_end == total_epochs``.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_FOUR_PHASE
    frac_a = float(schedule_cfg.get("frac_a", 0.25))
    frac_b1 = float(schedule_cfg.get("frac_b1", 0.20))
    frac_b2 = float(schedule_cfg.get("frac_b2", 0.25))
    frac_c = float(schedule_cfg.get("frac_c", 0.30))
    total_frac = frac_a + frac_b1 + frac_b2 + frac_c
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"four_phase schedule fractions must sum to 1.0, got {total_frac} "
            f"(frac_a={frac_a}, frac_b1={frac_b1}, frac_b2={frac_b2}, frac_c={frac_c})"
        )
    a_end = int(round(total_epochs * frac_a))
    b1_end = int(round(total_epochs * (frac_a + frac_b1)))
    b2_end = int(round(total_epochs * (frac_a + frac_b1 + frac_b2)))
    c_end = total_epochs
    if a_end < 1:
        a_end = 1
    if b1_end < a_end + 1:
        b1_end = a_end + 1
    if b2_end < b1_end + 1:
        b2_end = b1_end + 1
    return (a_end, b1_end, b2_end, c_end)


def phase_for_epoch_four(
    epoch: int, total_epochs: int, schedule_cfg: dict | None = None
) -> str:
    """Return the active phase name for the four-phase schedule.

    Returns one of: 'A', 'B1', 'B2', 'C'.
    Boundary semantics:
      - Phase A:  ``0 <= epoch < a_end``
      - Phase B1: ``a_end <= epoch < b1_end``
      - Phase B2: ``b1_end <= epoch < b2_end``
      - Phase C:  ``b2_end <= epoch < c_end`` (c_end == total_epochs)
    """
    a_end, b1_end, b2_end, _ = four_phase_boundaries(total_epochs, schedule_cfg)
    if epoch < a_end:
        return "A"
    if epoch < b1_end:
        return "B1"
    if epoch < b2_end:
        return "B2"
    return "C"


def four_phase_tau(
    epoch: int, total_epochs: int, schedule_cfg: dict | None = None
) -> float:
    """Tau value for the current epoch under the four-phase schedule.

    Phase A:  fixed at ``tau_a`` (default 1.0).
    Phase B1: anneals ``tau_b1_init`` -> ``tau_b1_final`` (1.0 -> 0.6).
    Phase B2: anneals ``tau_b2_init`` -> ``tau_b2_final`` (0.6 -> 0.4).
    Phase C:  anneals ``tau_c_init`` -> ``tau_c_final`` (0.4 -> 0.1).

    Tau is continuous at all phase boundaries (end of B1 == start of B2
    at 0.6; end of B2 == start of C at 0.4 by construction).
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_FOUR_PHASE
    a_end, b1_end, b2_end, _ = four_phase_boundaries(total_epochs, schedule_cfg)

    if epoch < a_end:
        return float(schedule_cfg.get("tau_a", 1.0))

    if epoch < b1_end:
        tau_init = float(schedule_cfg.get("tau_b1_init", 1.0))
        tau_final = float(schedule_cfg.get("tau_b1_final", 0.6))
        local_epoch = epoch - a_end
        local_total = max(1, b1_end - a_end)
        return tau_for_epoch(local_epoch, total_epochs=local_total,
                             tau_init=tau_init, tau_final=tau_final)

    if epoch < b2_end:
        tau_init = float(schedule_cfg.get("tau_b2_init", 0.6))
        tau_final = float(schedule_cfg.get("tau_b2_final", 0.4))
        local_epoch = epoch - b1_end
        local_total = max(1, b2_end - b1_end)
        return tau_for_epoch(local_epoch, total_epochs=local_total,
                             tau_init=tau_init, tau_final=tau_final)

    tau_init = float(schedule_cfg.get("tau_c_init", 0.4))
    tau_final = float(schedule_cfg.get("tau_c_final", 0.1))
    local_epoch = epoch - b2_end
    local_total = max(1, total_epochs - b2_end)
    return tau_for_epoch(local_epoch, total_epochs=local_total,
                         tau_init=tau_init, tau_final=tau_final)


def four_phase_lambdas(
    epoch: int,
    total_epochs: int,
    base_lambdas: dict,
    schedule_cfg: dict | None = None,
) -> dict:
    """Effective lambda dict for the current epoch under the four-phase
    schedule (four-phase-redesign/Phase 3b).

    Phase A:  all structural lambdas = 0. ``rail`` preserved.
    Phase B1: lambdas_b1 ramped from 0 to full over warmup_frac_b1.
    Phase B2: lambdas_b2 ramped from 0 to full over warmup_frac_b2.
    Phase C:  lambdas_c at full strength (no warmup).

    The output dict always includes ``rail`` from ``base_lambdas``.
    Returns a fresh dict; ``base_lambdas`` is not mutated.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_FOUR_PHASE
    a_end, b1_end, b2_end, _ = four_phase_boundaries(total_epochs, schedule_cfg)
    rail_val = float(base_lambdas.get("rail", 0.0))
    out = {"rail": rail_val}

    if epoch < a_end:
        for k in _REG_KEYS:
            out[k] = 0.0
        return out

    if epoch < b1_end:
        lambdas_b1 = schedule_cfg.get("lambdas_b1", {})
        warmup_frac = float(schedule_cfg.get("warmup_frac_b1", 0.25))
        local_epoch = epoch - a_end
        local_total = max(1, b1_end - a_end)
        warmup_epochs = max(1, int(round(warmup_frac * local_total)))
        if local_epoch < warmup_epochs:
            scale = float(local_epoch + 1) / float(warmup_epochs)
        else:
            scale = 1.0
        for k in _REG_KEYS:
            out[k] = float(lambdas_b1.get(k, 0.0)) * scale
        return out

    if epoch < b2_end:
        lambdas_b2 = schedule_cfg.get("lambdas_b2", {})
        warmup_frac = float(schedule_cfg.get("warmup_frac_b2", 0.25))
        local_epoch = epoch - b1_end
        local_total = max(1, b2_end - b1_end)
        warmup_epochs = max(1, int(round(warmup_frac * local_total)))
        if local_epoch < warmup_epochs:
            scale = float(local_epoch + 1) / float(warmup_epochs)
        else:
            scale = 1.0
        for k in _REG_KEYS:
            out[k] = float(lambdas_b2.get(k, 0.0)) * scale
        return out

    lambdas_c = schedule_cfg.get("lambdas_c", {})
    for k in _REG_KEYS:
        out[k] = float(lambdas_c.get(k, 0.0))
    return out


def four_phase_kd_active(
    epoch: int,
    total_epochs: int,
    schedule_cfg: dict | None = None,
) -> bool:
    """Whether teacher distillation should be active for the current epoch
    (four-phase-redesign/Phase 3c).

    KD is active in Phase B1 and B2 only (not A, not C). During Phase A
    we are fitting the soft teacher; during Phase C we are retraining the
    pruned compact model and want to fit the task directly.
    """
    phase = phase_for_epoch_four(epoch, total_epochs, schedule_cfg)
    return phase in ("B1", "B2")


# ---------- readiness-based prune trigger (four-phase-redesign) ----------




def compute_loss(
    net: KirchhoffNetWithIO | KirchhoffNet,
    x0: torch.Tensor,
    target: torch.Tensor,
    ctx: SimContext,
    task_fn,
    lambdas: dict | None = None,
    return_parts: bool = False,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    reg_scale: float = 1.0,
    solver: str = "heun",
    deq_cfg: dict | None = None,
    teacher: KirchhoffNetWithIO | KirchhoffNet | None = None,
    kd_lambda: float = 0.0,
    device_l2_lambda: float = 0.0,
):
    """Compute total loss = task (optionally + KD) + regularizers.

    If net is a KirchhoffNetWithIO, x0 is the raw input u. If it is a plain
    KirchhoffNet, x0 is the already-bounded initial differential state.

    ``reg_scale`` is a multiplicative factor on edge_gate / rail: pass
    ``reg_schedule(epoch)`` from the training loop to implement staged
    warm-up. Defaults to 1.0 (no warm-up).

    If `amp` is True, wraps forward+loss in torch.cuda.amp.autocast for
    mixed-precision training. Caller is responsible for GradScaler.

    When ``teacher`` is not None, adds a knowledge-distillation term
    ``kd_lambda * MSE(student_out, teacher_out)`` to the task loss.
    """
    if lambdas is None:
        lambdas = LAMBDAS

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype) if amp else _NullContext()
    )
    with autocast_ctx:
        if isinstance(net, KirchhoffNetWithIO):
            out, trajs = net(x0, store_trajectory=True,
                             solver=solver, deq_cfg=deq_cfg)
        else:
            out, trajs = net(x0, store_trajectory=True,
                             solver=solver, deq_cfg=deq_cfg)

        loss_task = task_fn(out, target)

        if teacher is not None:
            with torch.no_grad():
                teacher_out, _ = teacher(x0, solver=solver, deq_cfg=deq_cfg)
            loss_kd = kd_lambda * F.mse_loss(out, teacher_out.detach())
            loss_task = loss_task + loss_kd

        if trajs is None:
            zero = loss_task.new_zeros((), requires_grad=True)
            if return_parts:
                return loss_task, zero, {"task": float(loss_task.item())}
            return loss_task, zero

        loss_edge_gate, loss_rail, loss_tanh_sat, \
            loss_sparsity, loss_entropy = _compute_regularizers(net, trajs, lambdas)

        loss_skip_l2 = 0.0
        skip_l2_lambda = float(lambdas.get("skip_linear_l2", 0.0))
        if skip_l2_lambda > 0:
            net_skip = net
            if hasattr(net_skip, "module") and isinstance(net_skip, torch.nn.DataParallel):
                net_skip = net_skip.module
            if hasattr(net_skip, "base") and hasattr(net_skip, "_stage_noise_std"):
                net_skip = net_skip.base
            if getattr(net_skip, "skip_linear_enabled", False):
                p = net_skip.skip_linear
                loss_skip_l2 = skip_l2_lambda * (
                    p.weight.pow(2).sum()
                    + (p.bias.pow(2).sum() if p.bias is not None else torch.zeros((), device=p.weight.device))
                )

        loss_device_l2 = 0.0
        device_l2_lambda = float(device_l2_lambda)
        if device_l2_lambda > 0:
            loss_device_l2 = device_l2_lambda * _device_l2_penalty(net)

        total_task = loss_task + float(lambdas.get("rail", 0.0)) * loss_rail
        structural = reg_scale * (
            float(lambdas.get("edge_gate", 0.0)) * loss_edge_gate
            + float(lambdas.get("tanh_sat", 0.0)) * loss_tanh_sat
            + float(lambdas.get("sparsity", 0.0)) * loss_sparsity
            + float(lambdas.get("entropy", 0.0)) * loss_entropy
        ) + loss_skip_l2 + loss_device_l2

    if return_parts:
        parts = {
            "task": float(loss_task.item()) if torch.is_tensor(loss_task) else float(loss_task),
            "edge_gate": float(loss_edge_gate.item()),
            "rail": float(loss_rail.item()),
            "tanh_sat": float(loss_tanh_sat.item()),
            "sparsity": float(loss_sparsity.item()),
            "entropy": float(loss_entropy.item()),
            "skip_linear_l2": float(loss_skip_l2.item()) if torch.is_tensor(loss_skip_l2) else float(loss_skip_l2),
            "device_l2": float(loss_device_l2.item()) if torch.is_tensor(loss_device_l2) else float(loss_device_l2),
            "reg_scale": float(reg_scale),
            "total": float((total_task + structural).item()),
        }
        return total_task, structural, parts
    return total_task, structural


class _NullContext:
    """A no-op context manager for use as a placeholder for autocast."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ---------- sparse linear solver losses ----------

def residual_loss(x_pred: torch.Tensor, b: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Mean squared residual ||A x_pred - b||^2 over batch.

    Args:
        x_pred: [batch, n] network output (predicted solution).
        b: [batch, n] right-hand side.
        A: [batch, n, n] (or broadcastable) system matrix.

    Returns:
        Scalar mean squared residual.
    """
    if A.dim() == 2:
        Ax = x_pred @ A.T
    else:
        Ax = torch.bmm(A, x_pred.unsqueeze(-1)).squeeze(-1)
    residual = Ax - b
    return residual.pow(2).mean()


def solution_loss(x_pred: torch.Tensor, x_star: torch.Tensor) -> torch.Tensor:
    """Mean squared error ||x_pred - x_star||^2."""
    return (x_pred - x_star).pow(2).mean()


def compute_solver_loss(
    net: KirchhoffNetWithIO | KirchhoffNet,
    b: torch.Tensor,
    x_star: torch.Tensor,
    A: torch.Tensor,
    ctx: SimContext,
    lambdas: dict | None = None,
    return_parts: bool = False,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    reg_scale: float = 1.0,
):
    """Solver loss = residual + 0.1 * solution + regularizers.

    Same regularizer set as compute_loss (edge_gate, rail). Task loss is the
    residual plus a small direct solution error to stabilize early training.
    ``reg_scale`` applies the staged warm-up factor.

    If `amp` is True, wraps forward+loss in torch.cuda.amp.autocast for
    mixed-precision training. Caller is responsible for GradScaler.
    """
    if lambdas is None:
        lambdas = LAMBDAS

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype) if amp else _NullContext()
    )
    with autocast_ctx:
        out, trajs = net(b, store_trajectory=True)

        loss_res = residual_loss(out, b, A)
        loss_sol = solution_loss(out, x_star)
        loss_task = loss_res + 0.1 * loss_sol

        if trajs is None:
            if return_parts:
                return loss_task, loss_task.new_zeros(()), {
                    "task": float(loss_task.item()),
                    "residual": float(loss_res.item()),
                    "solution": float(loss_sol.item()),
                }
            return loss_task, loss_task.new_zeros(())

        loss_edge_gate, loss_rail, loss_tanh_sat, \
            loss_sparsity, loss_entropy = _compute_regularizers(net, trajs, lambdas)

        total_task = (
            loss_task
            + float(lambdas.get("rail", 0.0)) * loss_rail
        )
        structural = reg_scale * (
            float(lambdas.get("edge_gate", 0.0)) * loss_edge_gate
            + float(lambdas.get("tanh_sat", 0.0)) * loss_tanh_sat
            + float(lambdas.get("sparsity", 0.0)) * loss_sparsity
            + float(lambdas.get("entropy", 0.0)) * loss_entropy
        )

    if return_parts:
        parts = {
            "task": float(loss_task.item()),
            "residual": float(loss_res.item()),
            "solution": float(loss_sol.item()),
            "edge_gate": float(loss_edge_gate.item()),
            "rail": float(loss_rail.item()),
            "tanh_sat": float(loss_tanh_sat.item()),
            "sparsity": float(loss_sparsity.item()),
            "entropy": float(loss_entropy.item()),
            "reg_scale": float(reg_scale),
            "total": float((total_task + structural).item()),
        }
        return total_task, structural, parts
    return total_task, structural


# ---------- tau annealing ----------

def tau_for_epoch(
    epoch: int,
    total_epochs: int | None = None,
    tau_init: float | None = None,
    tau_final: float | None = None,
) -> float:
    """Monotonic exponential decay (R6.1); smooth linear hardening in the
    last fraction of training (R5-prune-retrain-fixes).

    tau(epoch) = max(tau_min, tau_init * exp(-epoch / decay_half_life))
    for the first (1 - 2*hardening_epoch_frac) of training, then linearly
    interpolates from tau_base down to tau_final across the next
    2*hardening_epoch_frac window. The default ``hardening_epoch_frac=0.1``
    keeps the exponential floor until 80% of training, then linearly
    ramps to ``tau_final`` by 100% of training (no step jump).

    When ``tau_init`` or ``tau_final`` is provided, it overrides the
    corresponding ``TAU`` config values. This enables two-phase tau for
    pruning (R2-phase-tau): pre-prune uses ``tau_final=final_pretrain``
    to cap the floor at a gentle value, then retrain uses
    ``tau_init=final_pretrain, tau_final=final`` for a continuous
    schedule. The ``tau_min`` floor is automatically set to
    ``max(config_min, tau_final)`` to prevent non-monotonic artifacts
    when the phase target is higher than the global minimum.
    """
    if total_epochs is None:
        total_epochs = OPTIM["epochs"]
    if tau_init is None:
        tau_init = float(TAU["init"])
    if tau_final is None:
        tau_final = float(TAU["final"])
    config_min = float(TAU.get("min", 0.15))
    hardening_frac = float(TAU.get("hardening_epoch_frac", 0.1))
    decay_half_life = total_epochs * 0.5
    tau_base = max(config_min, tau_init * math.exp(-epoch / decay_half_life))
    tau_base = max(tau_base, tau_final)

    if hardening_frac <= 0.0:
        return tau_base

    hardening_start = int(total_epochs * (1.0 - 2.0 * hardening_frac))
    hardening_end = total_epochs
    if epoch < hardening_start:
        return tau_base
    if epoch >= hardening_end:
        return tau_final
    span = max(1, hardening_end - hardening_start)
    progress = (epoch - hardening_start) / span
    return tau_base + (tau_final - tau_base) * progress


# ---------- degree-budget annealing (degree-budget-topk plan) ----------

def budget_frac_for_epoch(
    epoch: int,
    total_epochs: int,
    frac_start: float = 1.0,
    frac_end: float = 0.75,
    anneal_frac: float = 0.8,
) -> float:
    """Linearly anneal the degree-budget fraction from ``frac_start`` to ``frac_end``.

    The fraction is the proportion of each node's incoming edges that the
    budget keeps open (via per-group softmax renormalization). ``frac=1.0``
    means no restriction (every group keeps all incident edges);
    ``frac=0.0`` disables the budget entirely. Annealing is spread over
    the first ``anneal_frac`` of ``total_epochs``; beyond that the value
    is clamped to ``frac_end``.

    This is a continuous float, unlike the previous integer-``k`` annealing
    that over-pruned high-degree nodes. Each group independently computes
    ``k_eff = max(1, round(count * frac))`` at the call site
    (``DifferentialStage._budget_group_mask``), so all node types get a
    uniform proportion of their incoming connections.

    Args:
        epoch: Current epoch index (0-based).
        total_epochs: Total number of training epochs (the annealing
            denominator).
        frac_start: Initial budget fraction (permissive). Default 1.0
            (no restriction at the start of training).
        frac_end: Final budget fraction (restrictive). Default 0.75
            (75% retention at the prune point).
        anneal_frac: Fraction of ``total_epochs`` over which to anneal.
            Default 0.8 (i.e. the last 20% of training freezes at frac_end).

    Returns:
        Float budget fraction for this epoch, in ``[frac_end, frac_start]``.
        Negative outputs are clamped to 0.0.
    """
    if frac_start < frac_end:
        warnings.warn(
            f"budget_frac_for_epoch: frac_start={frac_start} < frac_end={frac_end}, "
            f"annealing will become more permissive over time (likely a bug)"
        )
    if total_epochs <= 0:
        return max(0.0, float(frac_end))
    max_anneal_epochs = max(1, int(total_epochs * anneal_frac))
    if epoch >= max_anneal_epochs:
        return max(0.0, float(frac_end))
    denom = max(1, max_anneal_epochs - 1)
    alpha = epoch / denom
    frac = (1.0 - alpha) * float(frac_start) + alpha * float(frac_end)
    return max(0.0, frac)


def budget_temperature_for_epoch(
    epoch: int,
    total_epochs: int,
    temp_start: float = 1.0,
    temp_end: float = 0.1,
    anneal_frac: float = 0.8,
) -> float:
    """Linearly anneal the degree-budget softmax temperature.

    Annealing is spread over the first ``anneal_frac`` of ``total_epochs``.
    Beyond that, the value is clamped to ``temp_end`` (sharper competition).

    Smaller temperature → sharper softmax → closer to hard top-k. Larger
    temperature → softer → more uniform sharing of the budget.

    Args:
        epoch: Current epoch index (0-based).
        total_epochs: Total number of training epochs.
        temp_start: Initial temperature (typically 1.0, soft).
        temp_end: Final temperature (typically 0.1, sharp).
        anneal_frac: Fraction of ``total_epochs`` over which to anneal.

    Returns:
        Float temperature for this epoch, clamped to be > 0.
    """
    if total_epochs <= 0:
        return max(1e-6, temp_end)
    max_anneal_epochs = max(1, int(total_epochs * anneal_frac))
    if epoch >= max_anneal_epochs:
        return max(1e-6, temp_end)
    denom = max(1, max_anneal_epochs - 1)
    alpha = epoch / denom
    T = (1.0 - alpha) * temp_start + alpha * temp_end
    return max(1e-6, T)


# ---------- optimizer ----------

def make_optimizer(
    net: torch.nn.Module,
    lr: float | None = None,
    weight_decay: float | None = None,
    stage_lr_scale: float = 1.0,
    mapper_lr_scale: float = 1.0,
    struct_lr_scale: float = 1.0,
    dyn_lr_scale: float = 1.0,
):
    """Build the AdamW optimizer.

    Args:
        net: The KirchhoffNetWithIO (or KirchhoffNet) to optimize. May be
            wrapped in DataParallel; the function unwraps internally for
            name inspection but uses the same parameter tensors.
        lr: Base learning rate (default: ``OPTIM['lr']``).
        weight_decay: L2 penalty (default: ``OPTIM['weight_decay']``).
        stage_lr_scale: Per-stage geometric LR multiplier (stage-lr-scaling).
            When ``1.0`` (default), falls back to a single param group for
            backward compatibility. When ``>1.0``, the i-th stage (0-indexed
            from first) gets ``lr * stage_lr_scale ** (S - 1 - i)`` where
            S is the number of stages. Earlier stages (smaller gradient
            norm) receive proportionally larger updates to compensate for
            vanishing gradients through deep ODE stacks.

            Example with ``stage_lr_scale=10`` and 3 stages:
              stage 0 → lr × 100, stage 1 → lr × 10, stage 2 → lr × 1

            Ignored when ``struct_lr_scale != 1.0`` or ``dyn_lr_scale != 1.0``
            (flat grouping active).
        mapper_lr_scale: Multiplier on base LR for the I/O mapper parameter
            group (input_mapper + output_mapper). When ``1.0`` (default) and
            all other scales == 1.0, falls back to a single param group.
            When ``<1.0``, mappers learn more slowly — useful when mapper
            gradient norms dominate core by ~300×.
        struct_lr_scale: Multiplier on base LR for structural core params
            (``z_logits``). Default 1.0 (no change). Use >1.0 (e.g. 2.0)
            to boost learning of edge gates. When != 1.0, uses flat global
            groups and ignores ``stage_lr_scale``.
        dyn_lr_scale: Multiplier on base LR for sensitive dynamical params
            (``raw_leak``, ``raw_drive_g``). Default 1.0 (no change). Use
            <1.0 to protect solver stability. When != 1.0, uses flat global
            groups and ignores ``stage_lr_scale``.
    """
    if lr is None:
        lr = OPTIM["lr"]
    if weight_decay is None:
        weight_decay = OPTIM["weight_decay"]
    for name, sc in [
        ("stage_lr_scale", stage_lr_scale),
        ("mapper_lr_scale", mapper_lr_scale),
        ("struct_lr_scale", struct_lr_scale),
        ("dyn_lr_scale", dyn_lr_scale),
    ]:
        if sc <= 0.0:
            raise ValueError(f"{name} must be positive, got {sc}")

    if (
        stage_lr_scale == 1.0
        and mapper_lr_scale == 1.0
        and struct_lr_scale == 1.0
        and dyn_lr_scale == 1.0
    ):
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    raw = net.module if isinstance(net, torch.nn.DataParallel) else net
    groups: list[dict] = []

    if struct_lr_scale != 1.0 or dyn_lr_scale != 1.0:
        if stage_lr_scale != 1.0:
            warnings.warn(
                f"stage_lr_scale={stage_lr_scale} ignored because "
                f"struct/dyn LR scales are active (flat grouping)."
            )
        struct_params: list[torch.nn.Parameter] = []
        dyn_params: list[torch.nn.Parameter] = []
        mapper_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, p in raw.named_parameters():
            if "input_mapper" in name or "output_mapper" in name:
                mapper_params.append(p)
            elif name.endswith(".z_logits"):
                struct_params.append(p)
            elif (
                name.endswith(".vca_W")
                or name.endswith(".vca_W_core")
                or name.endswith(".vca_v_boundary")
                or name.endswith(".vca_v_readout")
                or name.endswith(".vca_v_core")
            ):
                # VCA routing-structure params: shared input projection +
                # per-edge tap embeddings for each gated family. Allocated
                # to the structural LR group (4x base by convention) so the
                # gate signal can learn alongside z_logits.
                struct_params.append(p)
            elif (name.endswith(".raw_leak") or name.endswith(".raw_drive_g")
                  or name.endswith(".clip_sharpness_raw")
                  or name.startswith("gln_rails.")):
                dyn_params.append(p)
            else:
                other_params.append(p)
        if other_params:
            groups.append({"params": other_params, "lr": lr})
        if mapper_params:
            groups.append({"params": mapper_params, "lr": lr * mapper_lr_scale})
        if struct_params:
            groups.append({"params": struct_params, "lr": lr * struct_lr_scale})
        if dyn_params:
            groups.append({"params": dyn_params, "lr": lr * dyn_lr_scale})
    else:
        core = getattr(raw, "core", raw)
        stages = getattr(core, "stages", None)
        num_stages = len(stages) if stages is not None else 0
        if stage_lr_scale > 1.0 and num_stages <= 1:
            warnings.warn(
                f"stage_lr_scale={stage_lr_scale} has no effect with "
                f"{num_stages} stage(s); all parameters will use base LR."
            )

        stage_params: dict[int, list[torch.nn.Parameter]] = {}
        mapper_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, p in raw.named_parameters():
            if ".stages." in name:
                try:
                    idx = int(name.split(".stages.")[1].split(".")[0])
                except (ValueError, IndexError):
                    other_params.append(p)
                    continue
                stage_params.setdefault(idx, []).append(p)
            elif "input_mapper" in name or "output_mapper" in name:
                mapper_params.append(p)
            else:
                other_params.append(p)

        if stage_lr_scale == 1.0:
            core_params = other_params
            for stage_list in stage_params.values():
                core_params.extend(stage_list)
            if core_params:
                groups.append({"params": core_params, "lr": lr})
        else:
            for i in sorted(stage_params.keys()):
                stage_lr = lr * (stage_lr_scale ** (num_stages - 1 - i))
                groups.append({"params": stage_params[i], "lr": stage_lr})
            if other_params:
                groups.append({"params": other_params, "lr": lr})
        if mapper_params:
            groups.append({"params": mapper_params, "lr": lr * mapper_lr_scale})

    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


# ---------- one-epoch loop ----------

def apply_ablation(net, ablation: str) -> None:
    """Apply a structural ablation in-place to a KirchhoffNetWithIO (R2.1-R2.4).

    - 'none': no change.
    - 'mapper-only': set every stage's t_span to 0.0 so the ODE core is
      identity and only the I/O mappers are exercised.
    - 'empty-graph': zero out the COO edge lists of every stage so no
      inter-node coupling exists, while still exercising the I/O mappers
      and stage evolution (which trivially returns x0).
    """
    if ablation == "none":
        return
    if ablation == "mapper-only":
        for i, _ in enumerate(net.core.stages):
            net.core.stage_times[i] = 0.0
        return
    if ablation == "empty-graph":
        import torch.nn as nn
        for stage in net.core.stages:
            stage.src = stage.src.new_zeros(0)
            stage.dst = stage.dst.new_zeros(0)
            cell_lib = getattr(stage, 'cell_lib', None)
            if isinstance(cell_lib, SimpleEdgeLibrary):
                stage.cell_lib.param = nn.Parameter(stage.cell_lib.param.new_zeros(3, 0))
            elif isinstance(cell_lib, RealisticTanhLibrary):
                stage.cell_lib.alpha_raw = nn.Parameter(stage.cell_lib.alpha_raw.new_zeros(0))
                if hasattr(stage.cell_lib, "bias_raw"):
                    stage.cell_lib.bias_raw = nn.Parameter(stage.cell_lib.bias_raw.new_zeros(0))
            elif isinstance(cell_lib, RealisticTanhUpgradeLibrary):
                for name in ("alpha_raw", "gm_raw", "isat_raw"):
                    setattr(
                        stage.cell_lib,
                        name,
                        nn.Parameter(getattr(stage.cell_lib, name).new_zeros(0)),
                    )
                if hasattr(stage.cell_lib, "bias_raw"):
                    stage.cell_lib.bias_raw = nn.Parameter(stage.cell_lib.bias_raw.new_zeros(0))
            elif isinstance(cell_lib, FreeTanhLibrary):
                for name in ("a_raw", "b_raw", "s_raw", "gm_raw", "isat_raw", "g_resistive_raw"):
                    setattr(
                        stage.cell_lib,
                        name,
                        nn.Parameter(getattr(stage.cell_lib, name).new_zeros(0)),
                    )
                if hasattr(stage.cell_lib, "theta_raw"):
                    stage.cell_lib.theta_raw = nn.Parameter(stage.cell_lib.theta_raw.new_zeros(0))
                if getattr(stage.cell_lib, "_parallel_tanh_mult_enabled", False):
                    for name in ("gm_x_raw", "gm_y_raw", "isat_parallel_raw"):
                        setattr(
                            stage.cell_lib,
                            name,
                            nn.Parameter(getattr(stage.cell_lib, name).new_zeros(0)),
                        )
            elif isinstance(cell_lib, AntiParallelFreeTanhLibrary):
                for name in ("kappa_raw", "gm_raw", "isat_raw"):
                    setattr(
                        stage.cell_lib,
                        name,
                        nn.Parameter(getattr(stage.cell_lib, name).new_zeros(0)),
                    )
                if hasattr(stage.cell_lib, "theta_raw"):
                    stage.cell_lib.theta_raw = nn.Parameter(stage.cell_lib.theta_raw.new_zeros(0))
            if hasattr(stage, "raw_leak"):
                stage.raw_leak = nn.Parameter(stage.raw_leak.new_zeros(stage.num_nodes))
            stage.z_logits = nn.Parameter(stage.z_logits.new_zeros(0))
        return
    raise ValueError(f"Unknown ablation: {ablation!r}")


def train_epoch(
    net: KirchhoffNetWithIO,
    loader,
    optimizer,
    task_fn,
    ctx_factory,
    epoch: int,
    total_epochs: int | None = None,
    lambdas: dict | None = None,
    grad_clip_norm: float | None = None,
    log_every: int = 0,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
):
    """Run one training epoch. ctx_factory(batch_size, num_edges_total, device) -> SimContext.

    `loader` is an iterable of (u, target) batches. For our DifferentialStage
    the relevant edge count is the sum across all stages.

    Regularizers are scheduled with :func:`reg_schedule` so the network
    learns freely in the first ``reg_warmup_epochs`` and the penalties
    anneal in linearly over the next ``reg_anneal_epochs``.

    If `scaler` is provided, uses AMP with `scaler.scale(loss).backward()`,
    `scaler.step(optimizer)`, `scaler.update()`. Otherwise standard
    `.backward()` and `optimizer.step()`.
    """
    if grad_clip_norm is None:
        grad_clip_norm = OPTIM["grad_clip_norm"]

    tau = tau_for_epoch(epoch, total_epochs=total_epochs)
    reg_scale = reg_schedule(epoch)
    net.train()
    total_loss = 0.0
    n_batches = 0
    for i, (u, target) in enumerate(loader):
        ctx = ctx_factory(u.size(0), device=u.device)
        optimizer.zero_grad()
        loss_task, loss_structural, parts = compute_loss(
            net, u, target, ctx, task_fn, lambdas=lambdas,
            return_parts=True, amp=amp, reg_scale=reg_scale,
        )
        if scaler is not None:
            ( scaler.scale(loss_task) + scaler.scale(loss_structural) ).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            (loss_task + loss_structural).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        total_loss += float((loss_task + loss_structural).item())
        n_batches += 1
        if log_every and (i % log_every == 0):
            print(f"  epoch {epoch} batch {i} loss={parts['total']:.4f} task={parts['task']:.4f} reg_scale={reg_scale:.2f}")
    return {"avg_loss": total_loss / max(1, n_batches), "tau": tau, "reg_scale": reg_scale}


# ---------- convenience ctx factory ----------

def default_ctx_factory(net: KirchhoffNetWithIO):
    """Build a ctx_factory closure tied to a specific net and device.

    Usage:
        factory = default_ctx_factory(net)
        ctx = factory(batch_size=128, device=u.device)

    Note (RR-C): ``temp_choices`` is no longer passed — ``temp_c`` is
    deprecated and ``sample_random_context`` always returns
    ``VARIATION["temp_c_default"]``.
    """
    def _factory(batch_size: int, device: torch.device | str = "cpu", **kwargs):
        total_edges = sum(s.num_edges() for s in net.core.stages)
        return sample_random_context(
            num_edges=total_edges,
            device=device,
            **{
                "gain_shift_std": VARIATION["global_gain_shift_std"],
                "mismatch_std": VARIATION["edge_mismatch_std"],
                "global_isat_shift_std": VARIATION.get("global_isat_shift_std", 0.0),
                "isat_mismatch_std": VARIATION.get("edge_isat_mismatch_std", 0.0),
            },
        )
    return _factory
