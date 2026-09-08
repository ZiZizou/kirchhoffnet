"""Per-edge current source library for the reduced differential KirchhoffNet.

Provides simple per-edge devices that compute edge currents directly from
the source and destination node voltages without cell-type selection:

  - SimpleEdgeLibrary:  I = ReLU(p0*Vsrc + p1*Vdest + p2)  or  I = tanh(...)
  - RealisticTanhLibrary:  I = tanh(A*Vsrc - B*Vdest + C), A+B=1
  - RealisticTanhUpgradeLibrary:  I = Isat * tanh(gm*(A*Vsrc - B*Vdest) + C)
  - FreeTanhLibrary:  I = Isat * tanh(gm*(s*(A*Vsrc - B*Vdest) + theta))
  - AntiParallelFreeTanhLibrary:  I = Isat * tanh(gm * relu(kappa*(Vsrc - Vdst) - theta))

All devices are single-cell (no library selection). Compliance gating
turns the edge off as |x_src| or |x_dst| approaches x_max.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ANTI_PARALLEL_GM_MAX,
    ANTI_PARALLEL_GM_MIN,
    ANTI_PARALLEL_ISAT_MAX,
    ANTI_PARALLEL_ISAT_MIN,
    ANTI_PARALLEL_KAPPA_MAX,
    ANTI_PARALLEL_KAPPA_MIN,
    ANTI_PARALLEL_THETA_MAX,
    CELL_LIBRARIES,
    PHYS,
    TANH_REALISTIC_GM_MAX,
    TANH_REALISTIC_GM_MIN,
    TANH_REALISTIC_ISAT_MAX,
    TANH_REALISTIC_ISAT_MIN,
    cells_to_tensor_dict,
)


__all__ = [
    "SimpleEdgeLibrary",
    "RealisticTanhLibrary",
    "RealisticTanhUpgradeLibrary",
    "FreeTanhLibrary",
    "AntiParallelFreeTanhLibrary",
    "make_cell_library",
]


class SimpleEdgeLibrary(nn.Module):
    """Single-cell edge device with per-edge learnable parameters.

    Two modes:
      - ``mode="relu"``:  I = ReLU(p0 * Vsrc + p1 * Vdest + p2)
      - ``mode="tanh"``:  I = tanh(p0 * Vsrc + p1 * Vdest + p2)

    Holds a single ``nn.Parameter`` of shape ``[3, E]`` for per-edge
    learnable weights. Compliance gating (src/dst rail clamp) is applied.
    """

    def __init__(self, num_edges: int, mode: str) -> None:
        super().__init__()
        if mode not in ("relu", "tanh"):
            raise ValueError(f"SimpleEdgeLibrary mode must be 'relu' or 'tanh', got {mode!r}")
        self._mode = mode
        self._act = torch.relu if mode == "relu" else torch.tanh
        self.param = nn.Parameter(torch.randn(3, num_edges))
        self._beta_softness = float(PHYS["beta_softness"])

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
        batch, num_edges = x_src.shape
        u = (self.param[0].unsqueeze(0) * x_src
             + self.param[1].unsqueeze(0) * x_dst
             + self.param[2].unsqueeze(0))
        i_cell = self._act(u)

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        gate = gate_src * gate_dst

        return gate * i_cell

    def compile_forward(self, backend: str = "inductor"):
        pass


class RealisticTanhLibrary(nn.Module):
    """Per-edge constrained-differential tanh device.

    Per-edge formula::

        I = tanh(A * Vsrc - B * Vdest + C)

    where ``A = sigmoid(alpha_raw)``, ``B = 1 - A`` (so ``A, B > 0`` and
    ``A + B = 1`` exactly), and ``C = bias_raw`` when ``bias_enabled=True``
    (otherwise ``C = 0`` exactly, no parameter, no gradient).

    Holds one or two learnable parameters of shape ``[E]``:
      - ``alpha_raw``: unconstrained source/destination mix coefficient (always present).
      - ``bias_raw``:  additive pre-tanh bias, init=0; only present when
        ``bias_enabled=True``.

    Compliance gating (src/dst rail clamp) is applied after the tanh evaluation.
    """

    def __init__(self, num_edges: int, bias_enabled: bool = False) -> None:
        super().__init__()
        self._bias_enabled = bool(bias_enabled)
        self.alpha_raw = nn.Parameter(torch.randn(num_edges))
        if self._bias_enabled:
            self.bias_raw = nn.Parameter(torch.zeros(num_edges))
        self._beta_softness = float(PHYS["beta_softness"])

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
        A = torch.sigmoid(self.alpha_raw).unsqueeze(0)  # [1, E]
        B = 1.0 - A
        bias = self.bias_raw.unsqueeze(0) if self._bias_enabled else 0.0
        u = A * x_src - B * x_dst + bias
        i_cell = torch.tanh(u)

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        gate = gate_src * gate_dst

        return gate * i_cell

    def compile_forward(self, backend: str = "inductor"):
        pass


class RealisticTanhUpgradeLibrary(nn.Module):
    """Per-edge saturated realistic tanh device with bounded gain/saturation.

    Per-edge formula::

        I = Isat * tanh(gm * (A * Vsrc - B * Vdest) + C)

    Parameterization (all per-edge, shape ``[E]``):
      - ``alpha_raw`` → A = sigmoid(alpha_raw), B = 1 - A (so A, B > 0, A+B=1).
      - ``gm_raw``    → gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw).
      - ``isat_raw``  → Isat = isat_min + (isat_max - isat_min) * sigmoid(isat_raw).
      - ``bias_raw``  → C; only present when ``bias_enabled=True`` (init=0).
        When disabled, ``C = 0`` exactly and ``bias_raw`` does not exist.

    Boundary defaults come from ``config.TANH_REALISTIC_*`` and can be
    overridden per-instance via constructor kwargs. Initial ``gm`` and
    ``Isat`` sit at the geometric midpoint (sigmoid(0)=0.5 → (min+max)/2).

    Compliance gating applied after tanh.
    """

    def __init__(
        self,
        num_edges: int,
        gm_min: float | None = None,
        gm_max: float | None = None,
        isat_min: float | None = None,
        isat_max: float | None = None,
        bias_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.gm_min = float(gm_min if gm_min is not None else TANH_REALISTIC_GM_MIN)
        self.gm_max = float(gm_max if gm_max is not None else TANH_REALISTIC_GM_MAX)
        self.isat_min = float(isat_min if isat_min is not None else TANH_REALISTIC_ISAT_MIN)
        self.isat_max = float(isat_max if isat_max is not None else TANH_REALISTIC_ISAT_MAX)
        if self.gm_max <= self.gm_min:
            raise ValueError(
                f"RealisticTanhUpgradeLibrary requires gm_max > gm_min, "
                f"got [{self.gm_min}, {self.gm_max}]"
            )
        if self.isat_max <= self.isat_min:
            raise ValueError(
                f"RealisticTanhUpgradeLibrary requires isat_max > isat_min, "
                f"got [{self.isat_min}, {self.isat_max}]"
            )

        self._bias_enabled = bool(bias_enabled)
        self.alpha_raw = nn.Parameter(torch.randn(num_edges))
        self.gm_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        self.isat_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        if self._bias_enabled:
            self.bias_raw = nn.Parameter(torch.zeros(num_edges))

        self._beta_softness = float(PHYS["beta_softness"])

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
        sig_alpha = torch.sigmoid(self.alpha_raw)
        A = sig_alpha.unsqueeze(0)  # [1, E]
        B = 1.0 - A
        sig_gm = torch.sigmoid(self.gm_raw)
        gm = self.gm_min + (self.gm_max - self.gm_min) * sig_gm  # [E]
        sig_isat = torch.sigmoid(self.isat_raw)
        isat = (self.isat_min + (self.isat_max - self.isat_min) * sig_isat).clamp_min(1e-6)  # [E]
        bias = self.bias_raw.unsqueeze(0) if self._bias_enabled else 0.0
        u = (A * x_src - B * x_dst) * gm.unsqueeze(0) + bias
        i_cell = isat.unsqueeze(0) * torch.tanh(u)

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        gate = gate_src * gate_dst

        return gate * i_cell

    def compile_forward(self, backend: str = "inductor"):
        pass


class FreeTanhLibrary(nn.Module):
    """Per-edge signed-realistic tanh device with independent A/B and STE sign,
    in parallel with an optimizable resistive shunt.

    Per-edge formula (leaky tanh + linear shunt)::

        u = gm * (s * (A * Vsrc - B * Vdest) + theta)
        I_tanh      = I_sat * (tanh(u) + ALPHA * u)
        I_resistive = G * (Vsrc - Vdest),   G = softplus(g_resistive_raw) > 0
        I_total     = I_tanh + I_resistive

    The ``ALPHA * u`` linear term guarantees a non-zero gradient even when
    ``tanh(u)`` is saturated (gradient floor = ``ALPHA``), so the cell never
    becomes a flat plateau for backward signal.

    The resistive shunt is exposed via a dedicated method so that, under
    ``--freeze-read``, the tanh contribution can be frozen at the initial state
    while the resistive term keeps routing current from evolving voltages.

    Parameterization (all per-edge, shape ``[E]``):
      - ``a_raw``   → A = softplus(a_raw), A >= 0, no upper bound, independent of B.
      - ``b_raw``   → B = softplus(b_raw), B >= 0, no upper bound, no sum constraint.
      - ``s_raw``   → s = sign(s_raw) (forward discrete ±1; backward uses STE).
      - ``gm_raw``  → gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw).
      - ``isat_raw``→ Isat = clamp_min(isat_min + (isat_max - isat_min) * sigmoid(isat_raw), 1e-6).
      - ``theta_raw``→ theta; only present when ``bias_enabled=True`` (init=0).
      - ``g_resistive_raw``→ G = softplus(g_resistive_raw), positive shunt
        conductance (init=-5 → G ≈ 0.0067).

    Boundary defaults come from ``config.TANH_REALISTIC_*``.
    Compliance gating applied after tanh.
    """

    # Leaky tanh slope: output = Isat * (tanh(u) + ALPHA * u). Sets the
    # minimum gradient magnitude so backward signal never vanishes in
    # saturation. Mirrors the LeakyReLU default of 0.01.
    ALPHA = 0.01

    def __init__(
        self,
        num_edges: int,
        gm_min: float | None = None,
        gm_max: float | None = None,
        isat_min: float | None = None,
        isat_max: float | None = None,
        bias_enabled: bool = False,
        parallel_tanh_mult_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.gm_min = float(gm_min if gm_min is not None else TANH_REALISTIC_GM_MIN)
        self.gm_max = float(gm_max if gm_max is not None else TANH_REALISTIC_GM_MAX)
        self.isat_min = float(isat_min if isat_min is not None else TANH_REALISTIC_ISAT_MIN)
        self.isat_max = float(isat_max if isat_max is not None else TANH_REALISTIC_ISAT_MAX)
        if self.gm_max <= self.gm_min:
            raise ValueError(
                f"FreeTanhLibrary requires gm_max > gm_min, "
                f"got [{self.gm_min}, {self.gm_max}]"
            )
        if self.isat_max <= self.isat_min:
            raise ValueError(
                f"FreeTanhLibrary requires isat_max > isat_min, "
                f"got [{self.isat_min}, {self.isat_max}]"
            )

        self._bias_enabled = bool(bias_enabled)
        self._parallel_tanh_mult_enabled = bool(parallel_tanh_mult_enabled)
        # Initialize a_raw, b_raw near 0 so softplus gives small values (~0.7)
        # instead of randn() which can cause huge A,B and gradient explosion.
        self.a_raw = nn.Parameter(torch.zeros(num_edges))
        self.b_raw = nn.Parameter(torch.zeros(num_edges))
        # Initialize s_raw to small random values so sign() gives ±1, not 0.
        # STE allows gradients to flow; forward needs non-zero sign for tanh path.
        self.s_raw = nn.Parameter(torch.randn(num_edges) * 0.1)
        # Initialize gm_raw, isat_raw more negative so initial gm, isat are near
        # minimum (0.01) to avoid initial gradient explosion in evolving core.
        # -5.0 -> sigmoid(-5) ≈ 0.0067 -> gm ≈ 0.01 + 9.99*0.0067 ≈ 0.077
        self.gm_raw = nn.Parameter(torch.full((num_edges,), -5.0))
        self.isat_raw = nn.Parameter(torch.full((num_edges,), -5.0))
        if self._bias_enabled:
            self.theta_raw = nn.Parameter(torch.zeros(num_edges))
        # Experimental parallel tanh-multiplier cell:
        #   I_parallel = Isat * tanh(gm_x * Vsrc) * tanh(gm_y * Vdest)
        if self._parallel_tanh_mult_enabled:
            self.gm_x_raw = nn.Parameter(torch.full((num_edges,), -2.3))
            self.gm_y_raw = nn.Parameter(torch.full((num_edges,), -2.3))
            self.isat_parallel_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        # This branch is always active, including when ``freeze_read`` is
        # enabled.  Start it in the same low-current regime as the OTA path:
        # softplus(-5) ≈ 0.0067, rather than a stiff 0.693 per-edge shunt.
        self.g_resistive_raw = nn.Parameter(torch.full((num_edges,), -5.0))

        self._beta_softness = float(PHYS["beta_softness"])

    def current_gm(self) -> torch.Tensor:
        """Current gm values ``[E]`` from the bounded sigmoid map.

        This is the GLN *base* ``gm0``: the static gm the cell would use
        without rails. GLN multiplies it by ``exp(delta)`` and re-clamps to
        ``[gm_min, gm_max]`` (see ``gln_rails.GLNRails``).
        """
        sig_gm = torch.sigmoid(self.gm_raw)
        return self.gm_min + (self.gm_max - self.gm_min) * sig_gm        # [E]

    def _tanh_core(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        gm_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Internal: leaky-tanh current (no compliance gating).

        Returns ``I_sat * (tanh(u) + ALPHA * u)`` with shape ``[B, E]`` —
        broadcastable against ``x_src``/``x_dst`` which are ``[B, E]``.

        ``gm_override`` (GLN path): caller-supplied modulated gm, shape
        ``[B, E]`` (or ``[1, E]``), used instead of the static sigmoid map.
        The stage calls this with the GLN-modulated gm for boundary /
        readout-sense families only; the core family always uses the static
        map (GLN never touches core edges in v1).
        """
        A = torch.nn.functional.softplus(self.a_raw).unsqueeze(0)        # [1, E]
        B = torch.nn.functional.softplus(self.b_raw).unsqueeze(0)        # [1, E]
        s = torch.sign(self.s_raw)                                       # [E]
        s_ste = s + self.s_raw - self.s_raw.detach()                     # STE: [E]
        if gm_override is not None:
            gm = gm_override                                             # [B, E] or [1, E]
        else:
            sig_gm = torch.sigmoid(self.gm_raw)
            gm = self.gm_min + (self.gm_max - self.gm_min) * sig_gm      # [E]
            gm = gm.unsqueeze(0)                                         # [1, E]
        sig_isat = torch.sigmoid(self.isat_raw)
        isat = (self.isat_min + (self.isat_max - self.isat_min) * sig_isat).clamp_min(1e-6)  # [E]
        theta = self.theta_raw.unsqueeze(0) if self._bias_enabled else 0.0
        u = (s_ste.unsqueeze(0) * (A * x_src - B * x_dst) + theta) * gm
        return isat.unsqueeze(0) * (torch.tanh(u) + self.ALPHA * u)

    def forward_tanh(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
        gm_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Leaky-tanh current with compliance gating, no resistive shunt.

        Use this when the resistive term must be excluded (e.g. ``--freeze-read``
        wants to freeze only the tanh contribution at ``x0``).
        Returns ``[batch, E]`` gated by ``gate_src * gate_dst``.

        ``gm_override`` (GLN path): modulated gm ``[B, E]`` from
        :meth:`GLNRails.modulate_gm_z`, used instead of the static sigmoid
        map (see :meth:`_tanh_core`).
        """
        i_cell = self._tanh_core(x_src, x_dst, gm_override=gm_override)  # [1, E] or [B, E]
        if self._parallel_tanh_mult_enabled:
            # NOTE (GLN): the parallel tanh-multiplier branch has its OWN
            # per-edge gm_x/gm_y parameters (independent of the OTA gm the
            # GLN modulates), so ``gm_override`` intentionally applies to the
            # main OTA path only. PARALLEL_TANH_MULT_ENABLED defaults False
            # and is not used by GLN configs (tanh_free core without the
            # parallel multiplier); revisit if the branch is ever gated.
            sig_gm_x = torch.sigmoid(self.gm_x_raw)
            gm_x = self.gm_min + (self.gm_max - self.gm_min) * sig_gm_x # [E]
            sig_gm_y = torch.sigmoid(self.gm_y_raw)
            gm_y = self.gm_min + (self.gm_max - self.gm_min) * sig_gm_y # [E]
            sig_isat_p = torch.sigmoid(self.isat_parallel_raw)
            isat_p = (
                self.isat_min
                + (self.isat_max - self.isat_min) * sig_isat_p
            ).clamp_min(1e-6)                                           # [E]
            i_parallel = isat_p.unsqueeze(0) * torch.tanh(
                gm_x.unsqueeze(0) * x_src
            ) * torch.tanh(gm_y.unsqueeze(0) * x_dst)                   # [1, E]
            i_cell = i_cell + i_parallel
        i_cell = i_cell.expand(x_src.shape[0], -1)                      # [B, E]
        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        return (gate_src * gate_dst) * i_cell

    def resistive_current(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Resistive shunt current ``G * (Vsrc - Vdest)``, no compliance gating.

        Returns shape ``[batch, E]``. Always uses evolving voltages (intended
        to bypass ``--freeze-read``).
        """
        G = torch.nn.functional.softplus(self.g_resistive_raw).unsqueeze(0)  # [1, E]
        return G * (x_src - x_dst)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
        """Full cell current: leaky tanh + resistive shunt.

        Only the tanh contribution is compliance-gated; the resistive shunt
        is a physical conductor that always routes current based on the
        voltage difference (intentional — see ``resistive_current``).
        Stage-level edge_mask / budget_gate are applied separately in
        ``DifferentialStage.rhs()``.
        """
        i_tanh = self.forward_tanh(x_src, x_dst, x_max)
        i_res = self.resistive_current(x_src, x_dst)
        return i_tanh + i_res

    def compile_forward(self, backend: str = "inductor"):
        pass


class AntiParallelFreeTanhLibrary(nn.Module):
    """Per-edge rectified differential OTA slice for antiparallel edge fabrics.

    Per-edge formula::

        v   = x_src - x_dst
        q   = relu(kappa * v - theta)
        i   = Isat * tanh(gm * q)

    Parameterization (all per-edge, shape ``[E]``):
      - ``kappa_raw`` → kappa = kappa_min + (kappa_max - kappa_min) * sigmoid(kappa_raw)
        (tied differential gain: A = B = kappa; guarantees common-mode rejection
        and exact zero current at equal voltages).
      - ``gm_raw``    → gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw)
      - ``isat_raw``  → Isat = clamp_min(isat_min + (isat_max - isat_min) * sigmoid(isat_raw), 1e-6)
      - ``theta_raw`` → theta = theta_max * sigmoid(theta_raw); only present when
        ``theta_enabled=True`` (default ``False`` → theta is hardcoded 0,
        no parameter, no gradient).

    Boundary defaults come from ``config.ANTI_PARALLEL_*`` and can be
    overridden per-instance via constructor kwargs.

    Antiparallel pairs are realized by the topology layer (each undirected
    adjacency becomes two directed edges via ``bidirectional=True`` in grid /
    torus / small_world topology kwargs). The library itself implements a
    single directed branch with rectified, nonnegative current. At
    ``x_src = x_dst`` the branch produces exactly zero current (since
    ``q = relu(-theta) = 0`` when ``theta = 0``).

    Compliance gating (src/dst rail clamp) is applied after the tanh evaluation.
    """

    def __init__(
        self,
        num_edges: int,
        kappa_min: float | None = None,
        kappa_max: float | None = None,
        gm_min: float | None = None,
        gm_max: float | None = None,
        isat_min: float | None = None,
        isat_max: float | None = None,
        theta_max: float | None = None,
        theta_enabled: bool = False,
        use_isat_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.kappa_min = float(kappa_min if kappa_min is not None else ANTI_PARALLEL_KAPPA_MIN)
        self.kappa_max = float(kappa_max if kappa_max is not None else ANTI_PARALLEL_KAPPA_MAX)
        self.gm_min = float(gm_min if gm_min is not None else ANTI_PARALLEL_GM_MIN)
        self.gm_max = float(gm_max if gm_max is not None else ANTI_PARALLEL_GM_MAX)
        self.isat_min = float(isat_min if isat_min is not None else ANTI_PARALLEL_ISAT_MIN)
        self.isat_max = float(isat_max if isat_max is not None else ANTI_PARALLEL_ISAT_MAX)
        self.theta_max = float(theta_max if theta_max is not None else ANTI_PARALLEL_THETA_MAX)
        if self.kappa_max <= self.kappa_min:
            raise ValueError(
                f"AntiParallelFreeTanhLibrary requires kappa_max > kappa_min, "
                f"got [{self.kappa_min}, {self.kappa_max}]"
            )
        if self.gm_max <= self.gm_min:
            raise ValueError(
                f"AntiParallelFreeTanhLibrary requires gm_max > gm_min, "
                f"got [{self.gm_min}, {self.gm_max}]"
            )
        if self.isat_max <= self.isat_min:
            raise ValueError(
                f"AntiParallelFreeTanhLibrary requires isat_max > isat_min, "
                f"got [{self.isat_min}, {self.isat_max}]"
            )

        self._theta_enabled = bool(theta_enabled)
        self._use_isat_normalization = bool(use_isat_normalization)
        self.kappa_raw = nn.Parameter(torch.randn(num_edges))
        self.gm_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        self.isat_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        if self._theta_enabled:
            self.theta_raw = nn.Parameter(torch.zeros(num_edges))

        self._beta_softness = float(PHYS["beta_softness"])

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(self.kappa_raw)  # [E]
        sig_gm = torch.sigmoid(self.gm_raw)
        gm = self.gm_min + (self.gm_max - self.gm_min) * sig_gm  # [E]
        sig_isat = torch.sigmoid(self.isat_raw)
        isat = (self.isat_min + (self.isat_max - self.isat_min) * sig_isat).clamp_min(1e-6)  # [E]
        if self._theta_enabled:
            theta = self.theta_max * torch.sigmoid(self.theta_raw)  # [E]
        else:
            theta = torch.zeros_like(kappa)  # [E]

        v = x_src - x_dst  # [B, E]
        q = F.relu(kappa.unsqueeze(0) * v - theta.unsqueeze(0))  # [B, E]

        if self._use_isat_normalization:
            u = gm.unsqueeze(0) * q / isat.unsqueeze(0)
        else:
            u = gm.unsqueeze(0) * q
        i_cell = isat.unsqueeze(0) * torch.tanh(u)

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        gate = gate_src * gate_dst

        return gate * i_cell

    def compile_forward(self, backend: str = "inductor"):
        pass


def make_cell_library(
    library_name: str,
    num_edges: int | None = None,
    gm_min: float | None = None,
    gm_max: float | None = None,
    isat_min: float | None = None,
    isat_max: float | None = None,
) -> SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary:
    """Factory: returns the appropriate edge-library class.

    ``relu`` / ``tanh`` → ``SimpleEdgeLibrary``.
    ``tanh_realistic`` → ``RealisticTanhLibrary``.
    ``tanh_realistic_upgrade`` → ``RealisticTanhUpgradeLibrary``.
    ``tanh_free`` → ``FreeTanhLibrary``.
    ``tanh_anti`` → ``AntiParallelFreeTanhLibrary``.

    ``gm_min`` / ``gm_max`` / ``isat_min`` / ``isat_max`` override the
    bounded sigmoid parameterization rails for the ``tanh_realistic_upgrade``,
    ``tanh_free``, and ``tanh_anti`` libraries (defaults from config
    ``TANH_REALISTIC_*`` / ``ANTI_PARALLEL_*``). When any of the four is
    given, all others fall back to their config defaults; the library
    constructors validate ``max > min``.

    The factory reads ``BIAS_ENABLED`` from the corresponding
    ``CELL_LIBRARIES[library_name]`` entry (default ``False``) and passes
    it to ``RealisticTanhLibrary`` / ``RealisticTanhUpgradeLibrary`` /
    ``FreeTanhLibrary``. For ``tanh_anti``, ``THETA_ENABLED`` is read
    instead and passed as ``theta_enabled``. For ``tanh_free``, the
    experimental ``PARALLEL_TANH_MULT_ENABLED`` flag is also read and
    passed as ``parallel_tanh_mult_enabled``.

    When ``num_edges`` is ``None`` (template), the returned library
    is created with ``num_edges=1`` and must be replaced with a per-stage
    instance (matching the actual edge count) before use.
    """
    n = num_edges if num_edges is not None else 1
    if library_name in ("relu", "tanh"):
        return SimpleEdgeLibrary(num_edges=n, mode=library_name)
    if library_name == "tanh_realistic":
        bias_enabled = bool(CELL_LIBRARIES[library_name].get("BIAS_ENABLED", False))
        return RealisticTanhLibrary(num_edges=n, bias_enabled=bias_enabled)
    if library_name == "tanh_realistic_upgrade":
        bias_enabled = bool(CELL_LIBRARIES[library_name].get("BIAS_ENABLED", False))
        return RealisticTanhUpgradeLibrary(
            num_edges=n, bias_enabled=bias_enabled,
            gm_min=gm_min, gm_max=gm_max,
            isat_min=isat_min, isat_max=isat_max)
    if library_name == "tanh_free":
        bias_enabled = bool(CELL_LIBRARIES[library_name].get("BIAS_ENABLED", False))
        parallel_tanh_mult_enabled = bool(
            CELL_LIBRARIES[library_name].get("PARALLEL_TANH_MULT_ENABLED", False)
        )
        return FreeTanhLibrary(
            num_edges=n,
            bias_enabled=bias_enabled,
            parallel_tanh_mult_enabled=parallel_tanh_mult_enabled,
            gm_min=gm_min, gm_max=gm_max,
            isat_min=isat_min, isat_max=isat_max,
        )
    if library_name == "tanh_anti":
        theta_enabled = bool(CELL_LIBRARIES[library_name].get("THETA_ENABLED", False))
        return AntiParallelFreeTanhLibrary(
            num_edges=n, theta_enabled=theta_enabled,
            gm_min=gm_min, gm_max=gm_max,
            isat_min=isat_min, isat_max=isat_max)
    raise ValueError(
        f"Unknown cell library: {library_name!r}. "
        f"Available: 'relu', 'tanh', 'tanh_realistic', "
        f"'tanh_realistic_upgrade', 'tanh_free', 'tanh_anti'."
    )
