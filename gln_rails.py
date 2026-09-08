"""GLN rails (plan F2): input-conditioned ``log-gm`` modulation.

Analog story: the rails are programmable bias currents derived from the
input ``u``; each rail is a tanh half-space ``z_b = tanh(alpha_b * (a_b . u
+ c_b))`` with ``alpha_b > 0`` (softplus). Per gated edge ``e`` in a family
(boundary OTAs, readout-sense OTAs) the static gm from the cell library's
bounded sigmoid map becomes the *base* ``gm0`` and rails add a log-space
modulation:

    delta_e  = sum_b W[e, b] * z_b
    gm_e     = clamp(gm0_e * exp(delta_e), gm_min, gm_max)

Identity init (load-bearing): ``W = P @ Q`` is zero-initialized, so with
``W = 0`` every edge keeps ``gm_e = gm0_e`` exactly (``exp(0) = 1``) and the
epoch-0 forward is identical to the no-GLN model. ``a, c`` start at zero
(``z ~= 0`` as well, which also zeroes the first-order sensitivity of the
modulation until P/Q move).

v1 scope (per the plan):
  - One shared ``GLNRails`` core ``(a, c, alpha_raw)`` for the whole net
    (tied across stages) plus a factorized ``P[E, rank] @ Q[rank, B]`` edge
    mix per family (also tied across stages).
  - Families: ``boundary`` and ``readout`` (shared-sense OTAs). Core hidden
    edges and resistive shunts are never gated.
  - ``gm_min`` / ``gm_max`` are the same rails as F1 (the BO-searchable
    ``gm_max`` remains the clamp ceiling after modulation).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GLNRails", "gln_param_count", "inv_softplus"]


def inv_softplus(y: float) -> float:
    """``softplus^{-1}(y)`` so ``F.softplus(inv_softplus(y)) == y``."""
    return math.log(math.expm1(float(y)))


class _EdgeMix(nn.Module):
    """Factorized per-family edge mixing ``W = P @ Q`` (shape ``[E, B]``).

    Zero-initialized so ``W = 0`` at startup (identity: ``gm = gm0``). Keeps
    the per-family parameter cost at ``E * rank + rank * B`` instead of the
    dense ``E * B``.
    """

    def __init__(self, n_edges: int, rank: int, n_rails: int) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.zeros(int(n_edges), int(rank)))
        self.Q = nn.Parameter(torch.zeros(int(rank), int(n_rails)))

    def weight(self) -> torch.Tensor:
        return self.P @ self.Q  # [E, B]


class GLNRails(nn.Module):
    """Shared input-conditioned rails + per-family log-gm edge modulation.

    Args:
        in_dim: Width of the raw input ``u`` seen by the stage RHS (the
            boundary-terminal voltage vector, ``[B, in_dim]``).
        B: Number of tanh half-space rails (default 4).
        rank: Factorization rank of each family's edge mix ``P @ Q``
            (default 2).
        gm_min: Lower clamp of the modulated gm (default 0.01).
        gm_max: Upper clamp of the modulated gm (default 10.0). This is the
            same ceiling as F1's searchable ``gm_max``.
        alpha_init: Initial rail steepness before the softplus map (default
            1.0). ``a, c`` start at zero, so the rails output ``z ~= 0`` at
            init regardless of ``alpha``; the identity property does not
            depend on ``alpha``.
    """

    def __init__(
        self,
        in_dim: int,
        B: int = 4,
        rank: int = 2,
        gm_min: float = 0.01,
        gm_max: float = 10.0,
        alpha_init: float = 1.0,
    ) -> None:
        super().__init__()
        if int(B) < 1:
            raise ValueError(f"GLNRails requires B >= 1, got {B}")
        if int(rank) < 1:
            raise ValueError(f"GLNRails requires rank >= 1, got {rank}")
        if gm_max <= gm_min:
            raise ValueError(
                f"GLNRails requires gm_max > gm_min, got [{gm_min}, {gm_max}]"
            )
        self.B = int(B)
        self.rank = int(rank)
        self.gm_min = float(gm_min)
        self.gm_max = float(gm_max)
        self.in_dim = int(in_dim)
        # Rails: a [B, in_dim], c [B], alpha_raw [B] (softplus -> alpha > 0).
        self.a = nn.Parameter(torch.zeros(self.B, self.in_dim))
        self.c = nn.Parameter(torch.zeros(self.B))
        self.alpha_raw = nn.Parameter(
            torch.full((self.B,), inv_softplus(alpha_init))
        )
        self.families = nn.ModuleDict()

    def add_family(self, name: str, n_edges: int) -> None:
        """Register an edge family with ``n_edges`` gated edges."""
        if name in self.families:
            raise ValueError(f"GLNRails family {name!r} already registered")
        if int(n_edges) < 1:
            raise ValueError(
                f"GLNRails family {name!r} needs n_edges >= 1, got {n_edges}"
            )
        self.families[name] = _EdgeMix(int(n_edges), self.rank, self.B)

    def rails(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the ``[Bch, B]`` rail activations from input ``u``.

        ``z_b = tanh(alpha_b * (a_b . u + c_b))`` with ``alpha_b =
        softplus(alpha_raw_b) + 1e-6 > 0``. Called once per stage entry (u is
        constant per sample during the ODE integration).
        """
        alpha = F.softplus(self.alpha_raw) + 1e-6          # [B]
        z = torch.tanh(alpha * (u @ self.a.T + self.c))    # [Bch, B]
        return z

    def modulate_gm_z(
        self, gm0: torch.Tensor, z: torch.Tensor, family: str
    ) -> torch.Tensor:
        """Modulate base gm with precomputed rails ``z`` (``[Bch, B]``).

        ``gm = clamp(gm0 * exp(z @ W.T), gm_min, gm_max)`` with ``W = P @ Q``.
        ``gm0`` is ``[E]`` (broadcast against ``[Bch, B]`` z) or ``[Bch, E]``.
        Returns ``[Bch, E]``. At ``W = 0`` the result is exactly ``gm0``
        (``exp(0) = 1``, and ``gm0`` already lies in ``[gm_min, gm_max]``).
        """
        if family not in self.families:
            raise ValueError(
                f"GLNRails has no family {family!r}; registered: "
                f"{sorted(self.families)}"
            )
        W = self.families[family].weight()                 # [E, B]
        delta = z @ W.T                                    # [Bch, E]
        gm = gm0 * torch.exp(delta)
        return gm.clamp(self.gm_min, self.gm_max)

    def modulate_gm(
        self, gm0: torch.Tensor, u: torch.Tensor, family: str
    ) -> torch.Tensor:
        """``rails(u)`` + :meth:`modulate_gm_z` (convenience for callers that
        do not precompute ``z``)."""
        return self.modulate_gm_z(gm0, self.rails(u), family)


def gln_param_count(
    *,
    in_dim: int,
    e_bound: int,
    e_read: int,
    B: int = 4,
    rank: int = 2,
    share_rails: bool = True,
    share_edge_mix_across_stages: bool = True,
    num_stages: int = 1,
) -> int:
    """Analytic trainable-param count for the v1 shared GLN.

    Rails: ``B * (in_dim + 2)`` (``a``, ``c``, ``alpha_raw``) once when
    shared across stages. Per family: ``E * rank + rank * B`` (``P``, ``Q``),
    times ``num_stages`` only when the edge mix is per-stage.
    """
    rails = int(B) * (int(in_dim) + 2)
    fam = lambda e: int(e) * int(rank) + int(rank) * int(B)
    body = fam(e_bound) + fam(e_read)
    if not share_rails:
        rails *= int(num_stages)
        body *= int(num_stages)
    elif not share_edge_mix_across_stages:
        body *= int(num_stages)
    return rails + body