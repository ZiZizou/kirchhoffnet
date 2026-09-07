"""Shared joint feasible-architecture sampling for the BO controllers.

Single source of truth ported from ``fixed-distillation-bayes-opt.py``:
every controller precomputes the set of architectures whose trainable
parameter count lies under the soft cap ``budget * (1 + tolerance)`` and
samples one joint tuple via a single ``suggest_categorical`` over the
feasible indices. This guarantees every suggested architecture is feasible
(no sample-then-reject, no ``TrialPruned`` on budget misses) while keeping
the Optuna distribution fixed across trials (avoids the
``suggest_int``-bounds-change prohibition).

Key deviation from fixed-distillation's conditional per-(layers, layernorm)
width sampling: a single joint tuple keeps one fixed categorical
distribution per study, which is what RDBStorage-backed TPE requires.

Budget misses that survive sampling (e.g. the subprocess preflight measuring
a count that disagrees with the analytic/build count) return a finite graded
penalty — never ``TrialPruned`` or ``inf`` — because pruned trials crash the
multi-objective TPE Parzen estimator (optuna#5260) and non-finite values
crash it too (optuna#3676).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parent

# Finite penalty used in lieu of TrialPruned/inf for any infeasible
# configuration or failed preflight. See module docstring.
PENALTY_VALUES: tuple[float, float] = (1.0, 1e6)


def over_budget_objective(actual_params: int, budget: int, base: float) -> float:
    """Finite graded objective for architectures above the soft cap.

    ``base * (params / budget) ** 2`` — sorts after every real objective but
    stays finite so TPE's Parzen estimator never sees inf (optuna#3676).
    """
    ratio = actual_params / max(1, budget)
    return float(base) * ratio * ratio


def sample_arch_idx(trial: optuna.Trial, name: str, arches: list) -> int:
    """Sample one joint tuple index with a fixed-distribution categorical.

    A single categorical over ``range(len(arches))`` keeps the Optuna
    distribution identical across trials regardless of which tuple TPE
    currently favors.
    """
    if not arches:
        raise ValueError(
            f"cannot sample '{name}': feasible architecture list is empty. "
            "Widen --param-tolerance or adjust --param-budget."
        )
    return int(trial.suggest_categorical(name, list(range(len(arches)))))


# ---------------------------------------------------------------------------
# Resume fingerprint guard
# ---------------------------------------------------------------------------

def make_sampling_fingerprint(components: dict) -> str:
    """Stable JSON string identifying the (budget, tolerance, controller, etc.)
    of a sampling configuration. Stored in ``study.user_attrs`` so resuming
    against an incompatible ``study.db`` fails fast instead of crashing the
    first trial with ``CategoricalDistribution does not support dynamic value
    space`` (RDBStorage rejects mismatched distributions on second commit).
    """
    return json.dumps(components, sort_keys=True, separators=(",", ":"))


def check_sampling_fingerprint(study: optuna.Study, fingerprint: str) -> None:
    """Compare ``fingerprint`` against ``study.user_attrs['sampling_fingerprint']``.

    No-op on a fresh study (fingerprint not yet recorded). Raises ``RuntimeError``
    on mismatch with a clear remediation message; old studies must be moved to
    a fresh ``--output`` directory (or ``study.db`` deleted) because the
    SQLite distributions of past trials cannot be migrated.
    """
    existing = study.user_attrs.get("sampling_fingerprint")
    if existing is None:
        return
    if existing != fingerprint:
        raise RuntimeError(
            "refusing to resume study with different sampling configuration:\n"
            f"  stored:   {existing}\n"
            f"  requested:{fingerprint}\n"
            "Pass --output to a fresh directory (or delete the existing "
            "study.db). Old studies cannot be migrated because Optuna's "
            "RDBStorage rejects distributions that change across trials "
            "('CategoricalDistribution does not support dynamic value space')."
        )


# ---------------------------------------------------------------------------
# Plain MLP (fixed-harness style: first layer in_dim->W, head out_dim)
# ---------------------------------------------------------------------------

def mlp_param_count(width: int, layers: int, in_dim: int, out_dim: int,
                    layernorm: bool = False) -> int:
    """Analytic trainable-param count for a plain MLP.

    First layer ``in_dim->W`` ((in_dim+1)W), (L-1) hidden layers W->W
    (W^2+W each), head ``W->out_dim`` (out_dim*(W+1)), plus
    ``2*W*L`` for LayerNorm weight+bias when enabled.

    For the fixed-distillation harness (in_dim=4, out_dim=7) this matches
    the harness ``--count-params-only`` (verified: 48x3 -> 5287).
    """
    if width < 1 or layers < 1:
        raise ValueError("width and layers must be positive")
    trunk = (in_dim + 1) * width + (layers - 1) * (width * width + width)
    head = out_dim * (width + 1)
    ln = 2 * width * layers if layernorm else 0
    return trunk + head + ln


def mlp_feasible_arches(*, soft_limit: int, in_dim: int, out_dim: int,
                        layers_range: tuple[int, int] = (1, 5),
                        width_range: tuple[int, int] = (8, 256),
                        ln_options: tuple[bool, ...] = (False, True),
                        min_width: int = 1) -> list[tuple[int, int, bool]]:
    """All ``(layers, width, layernorm)`` tuples with params <= soft_limit.

    Unlike the legacy derive-at-max-width scheme, this explores the full
    depth/width degeneracy under the cap. LayerNorm parameters are included
    in the count (the generative-plain closed form omitted them and was
    optimistic by design — a known invalid-config source).
    """
    lo, hi = layers_range
    wlo, whi = width_range
    arches: list[tuple[int, int, bool]] = []
    for layers in range(lo, hi + 1):
        for width in range(max(wlo, min_width), whi + 1):
            for ln in ln_options:
                if mlp_param_count(width, layers, in_dim, out_dim, ln) <= soft_limit:
                    arches.append((layers, width, ln))
    return arches


def mlp_feasible_width_ranges(*, soft_limit: int, in_dim: int, out_dim: int,
                              layers_range: tuple[int, int],
                              ln: bool = False,
                              max_sweep_width: int = 4096) -> tuple[int, int]:
    """Tight (min, max) width bounds across ``layers_range`` under the soft cap.

    Uniform categorical sampling over a wide sweep (e.g. width 1..4096) is
    dominated by tiny models — the trial budget ends up wasted far under
    ``--param-budget``. This helper finds the union of feasible widths across
    the requested layer counts and returns the tight inclusive bounds, so the
    downstream enumerator stays small AND every arch is near-budget.

    Returns ``(1, 0)`` when no layer admits any positive width (caller should
    fail fast via :func:`require_feasible` — an enumerator over ``(1, 0)``
    yields no entries, so the empty-list guard triggers correctly).
    """
    lo, hi = layers_range
    w_min, w_max = max_sweep_width, 1
    for layers in range(lo, hi + 1):
        for width in range(1, max_sweep_width + 1):
            if mlp_param_count(width, layers, in_dim, out_dim, ln) <= soft_limit:
                w_min = min(w_min, width)
                w_max = max(w_max, width)
    return (1, 0) if w_max < 1 else (w_min, w_max)


# ---------------------------------------------------------------------------
# CTLE RegimeAwareMoE student
# ---------------------------------------------------------------------------

def moe_param_count(trunk_width: int, trunk_layers: int, num_experts: int,
                    input_dim: int = 4, layernorm: bool = False) -> int:
    """Exact trainable count of the CTLE RegimeAwareMoE student.

    Trunk: (input_dim+1)W + (L-1)(W^2+W) (plus 2WL when layernorm is on;
    the harness keeps ``MOE_USE_LAYERNORM=False`` by default). Routing: two
    bias-free input_dim->E heads (2*input_dim*E). Experts: E x (W->7) heads.
    Matches ``sum(p.numel() for p in student.parameters())``.
    """
    if trunk_width < 1 or trunk_layers < 1 or num_experts < 1:
        raise ValueError("MoE dimensions must be positive")
    trunk = (input_dim + 1) * trunk_width \
        + (trunk_layers - 1) * (trunk_width ** 2 + trunk_width)
    ln = 2 * trunk_width * trunk_layers if layernorm else 0
    routes = 2 * input_dim * num_experts
    experts = num_experts * (7 * trunk_width + 7)
    return trunk + ln + routes + experts


def moe_feasible_arches(*, soft_limit: int, layers_range: tuple[int, int] = (2, 3),
                        experts_range: tuple[int, int] = (2, 4),
                        width_range: tuple[int, int] = (32, 64),
                        input_dim: int = 4) -> list[tuple[int, int, int]]:
    """All ``(layers, experts, width)`` MoE tuples with params <= soft_limit."""
    arches: list[tuple[int, int, int]] = []
    for layers in range(layers_range[0], layers_range[1] + 1):
        for experts in range(experts_range[0], experts_range[1] + 1):
            for width in range(width_range[0], width_range[1] + 1):
                if moe_param_count(width, layers, experts, input_dim) <= soft_limit:
                    arches.append((layers, experts, width))
    return arches


# ---------------------------------------------------------------------------
# KirchhoffNet (build-based exact counts, cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _knet_param_count_cached(num_hidden: int, num_stages: int,
                             small_world_k: int, vca_rank: int,
                             in_dim: int, out_dim: int,
                             fanout_count: int, use_robust_input: bool,
                             moe_num_experts: int, moe_gate_rank: int,
                             dagger: bool, readout_mode: str = "ota_mesh",
                             readout_senses: int = 1) -> int:
    """Exact KNet trainable-param count by building the net (no training).

    The count depends on hidden, stages, k, rank AND fanout for the generic
    path (empirically verified: friedman2 h25/st5 k4->7061, k6->9561,
    rank4->8471, fanout1->6973), so the tuple is cached in full. t_span,
    num_steps, x_max, gm/isat, lr/wd/batch and freeze flags never move the
    count.

    ``dagger=True`` reproduces the CTLE dagger harness construction
    (fixed-distillation pattern); ``dagger=False`` reproduces the generic
    ``train_script.py`` construction. ``moe_num_experts``/``moe_gate_rank``
    are readout/gate dimensions on the CTLE dagger path only.

    ``readout_mode``/``readout_senses`` mirror the shared-sense-crossbar
    plan's builder kwargs: ``"ota_mesh"`` + 1 (legacy h*d_out mesh),
    ``"shared_sense"`` + 1 (one sense per hidden node + dense crossbar), or
    ``"shared_sense"`` + 2 (two senses per node). The accumulator tail is
    always enabled (``enable_temporal_readout=True``) in every mode.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cell_library import make_cell_library
    from topology import build_net_from_config
    from config import SOLVER

    if dagger:
        # CTLE dagger harness (dagger-nuance-distillation-kirchhoffnet.py /
        # fixed-mlp-distillation-kirchhoffnet.py KNet): 4-dim specs, out 7,
        # no robust input, dense-read sparse-proj.
        num_inputs = 4
        eff_out = 7
        eff_robust = False
        eff_fanout = fanout_count
    else:
        num_inputs = in_dim
        eff_out = out_dim
        eff_robust = use_robust_input
        eff_fanout = fanout_count

    stage_t_span = SOLVER["t_span"] / num_stages
    stage_steps = max(1, int(round(SOLVER["num_steps"] / num_stages)))
    fanout = {
        i: ([i, i + num_inputs] if eff_fanout >= 2 else [i])
        for i in range(num_inputs)
    }
    cfg = {
        "stages": [{
            "num_inputs": num_inputs, "num_hidden": num_hidden,
            "num_proj": 0, "num_outputs": 0,
            "hidden_family": "small_world",
            "hidden_kwargs": {"k": small_world_k, "p": 0.2, "seed": 1,
                              "bidirectional": False},
            "input_pattern": "all_to_all", "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all",
            "edge_repeats": 2, "t_span": stage_t_span, "num_steps": stage_steps,
        } for _ in range(num_stages)],
        "out_dim": eff_out, "write_mode": "sparse_proj", "read_mode": "dense",
        "use_robust_input": eff_robust,
    }
    net = build_net_from_config(
        cfg, cell_lib=make_cell_library("tanh_free"),
        leak_mode="non-programmable", freeze_read=True,
        interstage_activation="residual-relu-tanh",
        boundary_fan_out=fanout,
        enable_temporal_readout=True, x_max=4.0,
        readout_mode=readout_mode,
        readout_senses_per_node=int(readout_senses),
        vca_enabled=True, vca_rank=vca_rank, vca_core_enabled=True,
        vca_gate_shunt=False, vca_separate_core_bus=True, vca_bias=False,
    )
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def knet_param_count(num_hidden: int, num_stages: int, small_world_k: int,
                     vca_rank: int, *, in_dim: int, out_dim: int,
                     fanout_count: int = 2, use_robust_input: bool = False,
                     moe_num_experts: int = 0, moe_gate_rank: int = 0,
                     dagger: bool = False, readout_mode: str = "ota_mesh",
                     readout_senses: int = 1) -> int:
    """Public wrapper around the cached build-based KNet count."""
    return _knet_param_count_cached(
        int(num_hidden), int(num_stages), int(small_world_k), int(vca_rank),
        int(in_dim), int(out_dim), int(fanout_count), bool(use_robust_input),
        int(moe_num_experts), int(moe_gate_rank), bool(dagger),
        readout_mode, int(readout_senses),
    )


_knet_feasible_cache: dict[tuple, list[tuple]] = {}


def knet_feasible_arches(*, soft_limit: int, in_dim: int, out_dim: int,
                         hidden_range: tuple[int, int] = (8, 32),
                         stages_range: tuple[int, int] = (1, 10),
                         k_choices: tuple[int, ...] = (2, 4, 6, 8),
                         rank_range: tuple[int, int] = (1, 8),
                         fanout_choices: tuple[int, ...] = (2,),
                         use_robust_input: bool = False,
                         dagger: bool = False,
                         readout_mode: str = "ota_mesh",
                         readout_senses: int = 1,
                         moe_experts_choices: tuple[int, ...] = (),
                         moe_gate_rank_choices: tuple[int, ...] = (),
                         require_moe: bool = False) -> list[tuple]:
    """All feasible KNet tuples with build-count <= soft_limit, cached.

    Generic path tuples: ``(hidden, stages, k, rank, fanout_count)``.
    CTLE dagger path tuples (``require_moe=True``):
    ``(hidden, stages, k, rank, fanout_count, moe_num_experts, moe_gate_rank)``.
    ``hidden >= in_dim * fanout_count`` is enforced (boundary fan-out needs
    enough distinct targets). Cached per window so the build cost is paid
    once per study.

    ``readout_mode`` / ``readout_senses`` are fixed per-study constants
    (not sampled dims), forwarded into every ``knet_param_count`` call.
    """
    key = (soft_limit, in_dim, out_dim, hidden_range, stages_range,
           k_choices, rank_range, fanout_choices, use_robust_input,
           dagger, readout_mode, int(readout_senses),
           moe_experts_choices, moe_gate_rank_choices, require_moe)
    if key in _knet_feasible_cache:
        return _knet_feasible_cache[key]

    arches: list[tuple] = []
    seen: set[tuple] = set()
    for hidden in range(hidden_range[0], hidden_range[1] + 1):
        for fanout in fanout_choices:
            if hidden < in_dim * fanout:
                continue
            for stages in range(stages_range[0], stages_range[1] + 1):
                for k in k_choices:
                    if k >= hidden:
                        continue
                    for rank in range(rank_range[0], rank_range[1] + 1):
                        if require_moe:
                            if not moe_experts_choices or not moe_gate_rank_choices:
                                raise ValueError(
                                    "require_moe=True needs moe_experts_choices "
                                    "and moe_gate_rank_choices")
                            for experts in moe_experts_choices:
                                for gate_rank in moe_gate_rank_choices:
                                    try:
                                        p = knet_param_count(
                                            hidden, stages, k, rank,
                                            in_dim=in_dim, out_dim=out_dim,
                                            fanout_count=fanout,
                                            use_robust_input=use_robust_input,
                                            moe_num_experts=experts,
                                            moe_gate_rank=gate_rank,
                                            dagger=True,
                                            readout_mode=readout_mode,
                                            readout_senses=readout_senses)
                                    except Exception:
                                        continue
                                    if p <= soft_limit:
                                        arch = (hidden, stages, k, rank,
                                                fanout, experts, gate_rank)
                                        if arch not in seen:
                                            seen.add(arch)
                                            arches.append(arch)
                        else:
                            try:
                                p = knet_param_count(
                                    hidden, stages, k, rank,
                                    in_dim=in_dim, out_dim=out_dim,
                                    fanout_count=fanout,
                                    use_robust_input=use_robust_input,
                                    dagger=False,
                                    readout_mode=readout_mode,
                                    readout_senses=readout_senses)
                            except Exception:
                                continue
                            if p <= soft_limit:
                                arch = (hidden, stages, k, rank, fanout)
                                if arch not in seen:
                                    seen.add(arch)
                                    arches.append(arch)
    _knet_feasible_cache[key] = arches
    return arches


def require_feasible(arches: list, kind: str, soft_limit: int) -> list:
    """Fail fast (like fixed-distillation main) when the window admits zero arches."""
    if not arches:
        raise ValueError(
            f"soft cap {soft_limit} admits no {kind} architecture. "
            "Widen --param-tolerance or adjust --param-budget."
        )
    return arches
