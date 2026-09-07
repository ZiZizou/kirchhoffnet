"""Topology generators for the differential KirchhoffNet.

Three-layer API:
  1. Primitives: line_graph, ring_graph, grid_graph,
     small_world_graph, torus_graph, empty_graph
  2. Connectors: connect_bipartite, connect_projection
  3. Composer:   StageTopologyBuilder, MultiStageTopology.from_config()

Key design: input/output edges are NOT ODE edges. They are write-path
initialization and readout taps. The ODE core only evolves hidden + projection
nodes. topology_to_stage() remaps global node IDs to compact 0..N-1 for the
stage's internal state.

All presets come from config.PRESETS.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn

from config import PRESETS, VCA
from differential_stage import DifferentialStage
from cell_library import (
    AntiParallelFreeTanhLibrary,
    FreeTanhLibrary,
    RealisticTanhLibrary,
    RealisticTanhUpgradeLibrary,
    SimpleEdgeLibrary,
    make_cell_library,
)


__all__ = [
    "SparseTopology",
    "line_graph",
    "ring_graph",
    "grid_graph",
    "small_world_graph",
    "torus_graph",
    "empty_graph",
    "repeat_edges",
    "connect_bipartite",
    "connect_projection",
    "StageTopologyBuilder",
    "MultiStageTopology",
    "validate_topology",
    "validate_topology_degrees",
    "topology_to_stage",
    "build_net_from_preset",
    "build_net_from_config",
    "prune_stage",
    "prune_network",
]


EDGE_TYPE_INPUT = "input"
EDGE_TYPE_HIDDEN = "hidden"
EDGE_TYPE_PROJ = "proj"
EDGE_TYPE_OUTPUT = "output"

NODE_KIND_INPUT = "input"
NODE_KIND_HIDDEN = "hidden"
NODE_KIND_PROJ = "proj"
NODE_KIND_OUTPUT = "output"


@dataclass
class SparseTopology:
    """Universal sparse graph representation for a stage.

    src/dst are parallel lists of node IDs in the global stage node space
    (input + hidden + proj + output). They are NOT yet remapped to a stage's
    internal compact space.
    """

    num_nodes: int
    src: list[int] = field(default_factory=list)
    dst: list[int] = field(default_factory=list)
    edge_type: list[str] = field(default_factory=list)
    node_kind: list[str] = field(default_factory=list)
    input_node_ids: list[int] = field(default_factory=list)
    output_node_ids: list[int] = field(default_factory=list)
    hidden_node_ids: list[int] = field(default_factory=list)
    proj_node_ids: list[int] = field(default_factory=list)

    def num_edges(self) -> int:
        return len(self.src)


# ---------- primitives ----------

def line_graph(n_nodes: int, radius: int = 1, bidirectional: bool = False) -> SparseTopology:
    """1D chain; node i connects to i+1..i+radius.

    Emits a single directed edge per neighbor pair by default. L/S cells
    (odd I-V) provide implicit bidirectional conduction via sign reversal.

    When ``bidirectional=True``, also emits the reverse direction for every
    edge: for each (i, j) edge, an additional (j, i) edge is added. Edge
    count is exactly 2× the single-direction count. No self-loops are
    introduced since the original pairs have i != j.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if radius < 1:
        raise ValueError("radius must be >= 1")
    src, dst = [], []
    for i in range(n_nodes):
        for r in range(1, radius + 1):
            j = i + r
            if j < n_nodes:
                src.append(i); dst.append(j)
                if bidirectional:
                    src.append(j); dst.append(i)
    return SparseTopology(
        num_nodes=n_nodes,
        src=src, dst=dst,
        edge_type=[EDGE_TYPE_HIDDEN] * len(src),
        node_kind=[NODE_KIND_HIDDEN] * n_nodes,
        hidden_node_ids=list(range(n_nodes)),
    )


def ring_graph(n_nodes: int, radius: int = 1, bidirectional: bool = False) -> SparseTopology:
    """1D ring with wrap-around; useful for periodic signals.

    Emits a single directed edge per neighbor pair: each node i connects to
    (i+r) mod n_nodes for r=1..radius. Reverse direction is implicit via
    sign reversal of L/S cell currents.

    When ``bidirectional=True``, also emits the reverse direction for every
    edge: for each (i, j) edge, an additional (j, i) edge is added. Edge
    count is exactly 2× the single-direction count. No self-loops are
    introduced since the original pairs have i != j (radius * 2 < n_nodes).
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if radius * 2 >= n_nodes:
        raise ValueError("radius * 2 must be < n_nodes for ring_graph")
    src, dst = [], []
    for i in range(n_nodes):
        for r in range(1, radius + 1):
            j = (i + r) % n_nodes
            src.append(i); dst.append(j)
            if bidirectional:
                src.append(j); dst.append(i)
    return SparseTopology(
        num_nodes=n_nodes,
        src=src, dst=dst,
        edge_type=[EDGE_TYPE_HIDDEN] * len(src),
        node_kind=[NODE_KIND_HIDDEN] * n_nodes,
        hidden_node_ids=list(range(n_nodes)),
    )


def grid_graph(height: int, width: int, kernel_size: int = 3, bidirectional: bool = False) -> SparseTopology:
    """2D local grid; node id = row * width + col.

    Emits a single directed edge per unique neighbor pair (i, j) with j>i
    by default. The single-edge representation matches a 2-terminal
    electrical branch: L/S cells (odd I-V) carry current either direction
    via sign reversal, and P cells (asymmetric, softplus threshold)
    conduct only when V_src - V_dst > theta.

    When ``bidirectional=True``, also emits the reverse direction for every
    edge: for each (i, j) edge, an additional (j, i) edge is added. Edge
    count is exactly 2× the single-direction count. No self-loops are
    introduced since the original pairs have j > i (i != j).
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    n = height * width
    pad = kernel_size // 2
    src, dst = [], []
    for r in range(height):
        for c in range(width):
            i = r * width + c
            for dr in range(-pad, pad + 1):
                for dc in range(-pad, pad + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        j = nr * width + nc
                        if j > i:
                            src.append(i); dst.append(j)
                            if bidirectional:
                                src.append(j); dst.append(i)
    return SparseTopology(
        num_nodes=n,
        src=src, dst=dst,
        edge_type=[EDGE_TYPE_HIDDEN] * len(src),
        node_kind=[NODE_KIND_HIDDEN] * n,
        hidden_node_ids=list(range(n)),
    )


def small_world_graph(
    n_nodes: int,
    k: int = 4,
    p: float = 0.3,
    seed: int = 0,
    bidirectional: bool = False,
) -> SparseTopology:
    """Watts-Strogatz small-world graph.

    Starts with a ring lattice where each node connects to its k nearest
    neighbors (k must be even). Then each edge (i, j) is rewired with
    probability p: the destination j is replaced by a uniformly random
    distinct node j' (not i, and not already an edge of i). This produces
    the characteristic small-world property: high clustering coefficient
    plus short average path length.

    Emits a single directed edge per undirected pair by default (j > i).
    L/S cells (odd I-V) provide implicit bidirectional conduction via sign
    reversal. With ``bidirectional=True``, both (i, j) and (j, i) are
    emitted (2x edge count) so asymmetric cells (P/rectifier) get true
    bidirectional capability.

    Args:
        n_nodes: Number of hidden nodes (must be >= 2).
        k: Each node's degree in the initial ring lattice (even, >= 2,
            < n_nodes).
        p: Rewiring probability in [0, 1]. p=0 recovers the ring lattice;
            p=1 produces a random regular graph (no self-loops, no parallel
            edges).
        seed: RNG seed for rewiring determinism.
        bidirectional: If True, emit both directions for every undirected pair.

    Raises:
        ValueError: On invalid n_nodes / k / p.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    if k < 2 or k % 2 != 0:
        raise ValueError(f"k must be even and >= 2, got {k}")
    if k >= n_nodes:
        raise ValueError(f"k must be < n_nodes, got k={k}, n_nodes={n_nodes}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    half = k // 2
    # Build initial ring lattice as undirected pairs {(min, max)}.
    pairs: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        for r in range(1, half + 1):
            j = (i + r) % n_nodes
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))

    # Rewire with probability p.
    rng = random.Random(seed)
    rewired_pairs: set[tuple[int, int]] = set()
    for (a, b) in pairs:
        if p > 0.0 and rng.random() < p:
            # Pick a new neighbor for a: must be != a, and (a, new) not already
            # an edge in either direction. Try a bounded number of times
            # before giving up (rare in practice; preserves no-self-loop
            # / no-parallel-edge invariants).
            new_b = b
            attempts = 0
            while True:
                cand = rng.randrange(n_nodes)
                if cand == a:
                    attempts += 1
                    if attempts > 100:
                        break
                    continue
                pair = (a, cand) if a < cand else (cand, a)
                if pair in pairs or pair in rewired_pairs:
                    attempts += 1
                    if attempts > 100:
                        break
                    continue
                new_b = cand
                break
            lo, hi = (a, new_b) if a < new_b else (new_b, a)
            rewired_pairs.add((lo, hi))
        else:
            rewired_pairs.add((a, b))

    src: list[int] = []
    dst: list[int] = []
    for (a, b) in rewired_pairs:
        src.append(a); dst.append(b)
        if bidirectional:
            src.append(b); dst.append(a)

    return SparseTopology(
        num_nodes=n_nodes,
        src=src, dst=dst,
        edge_type=[EDGE_TYPE_HIDDEN] * len(src),
        node_kind=[NODE_KIND_HIDDEN] * n_nodes,
        hidden_node_ids=list(range(n_nodes)),
    )


def torus_graph(
    height: int,
    width: int,
    kernel_size: int = 3,
    bidirectional: bool = False,
) -> SparseTopology:
    """2D grid with periodic boundary conditions (wrap-around).

    Same neighbor structure as :func:`grid_graph`, but row and column indices
    wrap around: ``nr = (r + dr) % height``, ``nc = (c + dc) % width``. Every
    node has the same number of neighbors (no boundary nodes), giving a
    regular lattice with smaller diameter than the non-periodic grid.

    Node id = ``row * width + col``. Emits a single directed edge per
    undirected pair (j > i) by default; L/S cells provide implicit
    bidirectional conduction. With ``bidirectional=True``, both (i, j) and
    (j, i) are emitted for asymmetric cells.

    Args:
        height: Number of rows (must be >= 1).
        width: Number of columns (must be >= 1).
        kernel_size: Neighborhood half-size (positive odd integer). 3 means
            each node connects to its 8 immediate neighbors on the torus.
        bidirectional: If True, emit both directions for every undirected pair.

    Raises:
        ValueError: On non-positive height/width, or invalid kernel_size.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    n = height * width
    pad = kernel_size // 2
    src, dst = [], []
    for r in range(height):
        for c in range(width):
            i = r * width + c
            for dr in range(-pad, pad + 1):
                for dc in range(-pad, pad + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = (r + dr) % height
                    nc = (c + dc) % width
                    j = nr * width + nc
                    if j > i:
                        src.append(i); dst.append(j)
                        if bidirectional:
                            src.append(j); dst.append(i)
    return SparseTopology(
        num_nodes=n,
        src=src, dst=dst,
        edge_type=[EDGE_TYPE_HIDDEN] * len(src),
        node_kind=[NODE_KIND_HIDDEN] * n,
        hidden_node_ids=list(range(n)),
    )


def empty_graph(n_nodes: int) -> SparseTopology:
    """No edges. Useful for ablation or pure projection-node stages."""
    return SparseTopology(
        num_nodes=n_nodes,
        src=[], dst=[], edge_type=[],
        node_kind=[NODE_KIND_HIDDEN] * n_nodes,
        hidden_node_ids=list(range(n_nodes)),
    )


def repeat_edges(topo: SparseTopology, n: int) -> SparseTopology:
    """Duplicate every ``EDGE_TYPE_HIDDEN`` edge ``n`` times (total count).

    Non-hidden edges (``input``, ``proj``, ``output``) are NOT duplicated.
    Each repeated edge gets independent per-edge parameters (cell-type
    logits, gate, multiplier) in ``DifferentialStage``; their currents
    sum naturally in KCL via scatter-add (physically: parallel branches).

    Args:
        topo: Source topology.
        n: Total number of copies per hidden edge. ``n=1`` is identity
            (no duplication). ``n>=2`` produces parallel edges. Must be
            ``>= 1``.

    Returns:
        New ``SparseTopology`` with the same non-hidden edges and
        ``n`` hidden-edge copies per original hidden edge.
    """
    if n < 1:
        raise ValueError(f"repeat_edges requires n >= 1, got {n}")
    if n == 1:
        # Identity: return a shallow copy to keep the API symmetric.
        return SparseTopology(
            num_nodes=topo.num_nodes,
            src=list(topo.src),
            dst=list(topo.dst),
            edge_type=list(topo.edge_type),
            node_kind=list(topo.node_kind),
            input_node_ids=list(topo.input_node_ids),
            output_node_ids=list(topo.output_node_ids),
            hidden_node_ids=list(topo.hidden_node_ids),
            proj_node_ids=list(topo.proj_node_ids),
        )
    new_src: list[int] = []
    new_dst: list[int] = []
    new_type: list[str] = []
    for s, d, t in zip(topo.src, topo.dst, topo.edge_type):
        if t == EDGE_TYPE_HIDDEN:
            for _ in range(n):
                new_src.append(s)
                new_dst.append(d)
                new_type.append(t)
        else:
            new_src.append(s)
            new_dst.append(d)
            new_type.append(t)
    return SparseTopology(
        num_nodes=topo.num_nodes,
        src=new_src,
        dst=new_dst,
        edge_type=new_type,
        node_kind=list(topo.node_kind),
        input_node_ids=list(topo.input_node_ids),
        output_node_ids=list(topo.output_node_ids),
        hidden_node_ids=list(topo.hidden_node_ids),
        proj_node_ids=list(topo.proj_node_ids),
    )


# ---------- connectors ----------

def connect_bipartite(
    src_ids: Iterable[int],
    dst_ids: Iterable[int],
    pattern: str = "all_to_all",
) -> tuple[list[int], list[int]]:
    """Connect two disjoint node sets with the given pattern.

    pattern: 'all_to_all', 'one_to_one', 'none'.
    """
    src_ids = list(src_ids)
    dst_ids = list(dst_ids)
    src, dst = [], []
    if pattern == "all_to_all":
        for s in src_ids:
            for d in dst_ids:
                src.append(s); dst.append(d)
    elif pattern == "one_to_one":
        if len(src_ids) != len(dst_ids):
            raise ValueError(
                f"one_to_one requires equal-length lists, got {len(src_ids)} vs {len(dst_ids)}"
            )
        for s, d in zip(src_ids, dst_ids):
            src.append(s); dst.append(d)
    elif pattern == "none":
        pass
    else:
        raise ValueError(f"Unsupported bipartite pattern: {pattern!r}")
    return src, dst


def connect_projection(
    hidden_ids: Iterable[int],
    proj_ids: Iterable[int],
    pattern: str = "all_to_all",
) -> tuple[list[int], list[int]]:
    """Unidirectional bipartite from hidden to projection nodes.

    Each hidden node connects to projection nodes via the given pattern.
    L/S cells (odd I-V) handle reverse (proj->hidden) flow via sign
    reversal of current on the same edge.
    """
    return connect_bipartite(hidden_ids, proj_ids, pattern)


# ---------- composer ----------

class StageTopologyBuilder:
    """Assembles a full stage from input, hidden, projection, and output sub-graphs.

    Node ID allocation:
        [0 .. n_in-1]                     = inputs
        [n_in .. n_in+n_h-1]              = hidden
        [n_in+n_h .. n_in+n_h+n_p-1]      = projection
        [n_in+n_h+n_p .. N-1]             = outputs
    """

    def __init__(self, num_inputs: int, num_outputs: int, num_hidden: int, num_proj: int = 0) -> None:
        if min(num_inputs, num_outputs, num_hidden, num_proj) < 0:
            raise ValueError("Negative node counts not allowed")
        self.n_in = int(num_inputs)
        self.n_out = int(num_outputs)
        self.n_h = int(num_hidden)
        self.n_p = int(num_proj)

        self.in_ids = list(range(num_inputs))
        self.hid_ids = list(range(num_inputs, num_inputs + num_hidden))
        self.proj_ids = list(
            range(num_inputs + num_hidden, num_inputs + num_hidden + num_proj)
        )
        self.out_ids = list(
            range(
                num_inputs + num_hidden + num_proj,
                num_inputs + num_hidden + num_proj + num_outputs,
            )
        )
        self.total_nodes = num_inputs + num_hidden + num_proj + num_outputs

    def build(
        self,
        hidden_topo: SparseTopology,
        input_pattern: str = "one_to_one",
        output_pattern: str = "all_to_all",
        proj_pattern: str = "all_to_all",
    ) -> SparseTopology:
        if hidden_topo.num_nodes != self.n_h:
            raise ValueError(
                f"hidden_topo has {hidden_topo.num_nodes} nodes, expected {self.n_h}"
            )

        src, dst = [], []
        edge_type = []
        node_kind = (
            [NODE_KIND_INPUT] * self.n_in
            + [NODE_KIND_HIDDEN] * self.n_h
            + [NODE_KIND_PROJ] * self.n_p
            + [NODE_KIND_OUTPUT] * self.n_out
        )

        offset = self.n_in
        for s, d in zip(hidden_topo.src, hidden_topo.dst):
            src.append(s + offset)
            dst.append(d + offset)
            edge_type.append(EDGE_TYPE_HIDDEN)

        s, d = connect_bipartite(self.in_ids, self.hid_ids, input_pattern)
        src.extend(s); dst.extend(d); edge_type.extend([EDGE_TYPE_INPUT] * len(s))

        if self.n_p > 0:
            s, d = connect_projection(self.hid_ids, self.proj_ids, proj_pattern)
            src.extend(s); dst.extend(d); edge_type.extend([EDGE_TYPE_PROJ] * len(s))

        if self.n_out > 0:
            source_pool = self.hid_ids + self.proj_ids
            s, d = connect_bipartite(source_pool, self.out_ids, output_pattern)
            src.extend(s); dst.extend(d); edge_type.extend([EDGE_TYPE_OUTPUT] * len(s))

        return SparseTopology(
            num_nodes=self.total_nodes,
            src=src, dst=dst, edge_type=edge_type, node_kind=node_kind,
            input_node_ids=list(self.in_ids),
            output_node_ids=list(self.out_ids),
            hidden_node_ids=list(self.hid_ids),
            proj_node_ids=list(self.proj_ids),
        )


class MultiStageTopology:
    """Holds a list of SparseTopology objects, one per stage."""

    def __init__(self, stage_topologies: list[SparseTopology]) -> None:
        self.stages = list(stage_topologies)

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, idx: int) -> SparseTopology:
        return self.stages[idx]

    @staticmethod
    def from_config(configs: list[dict]) -> "MultiStageTopology":
        """Build a multi-stage topology from a list of per-stage config dicts."""
        topologies = []
        for cfg in configs:
            builder = StageTopologyBuilder(
                num_inputs=cfg["num_inputs"],
                num_outputs=cfg["num_outputs"],
                num_hidden=cfg["num_hidden"],
                num_proj=cfg.get("num_proj", 0),
            )
            family = cfg.get("hidden_family", "line")
            hidden_kwargs = dict(cfg.get("hidden_kwargs", {}))
            if family == "line":
                hid = line_graph(cfg["num_hidden"], **hidden_kwargs)
            elif family == "ring":
                hid = ring_graph(cfg["num_hidden"], **hidden_kwargs)
            elif family == "grid":
                hid = grid_graph(**hidden_kwargs)
            elif family == "small_world":
                hid = small_world_graph(cfg["num_hidden"], **hidden_kwargs)
            elif family == "torus":
                hid = torus_graph(**hidden_kwargs)
            elif family == "empty":
                hid = empty_graph(cfg["num_hidden"])
            else:
                raise ValueError(f"Unknown hidden_family: {family!r}")
            edge_repeats = int(cfg.get("edge_repeats", 1))
            if edge_repeats > 1:
                hid = repeat_edges(hid, edge_repeats)
            topo = builder.build(
                hid,
                input_pattern=cfg.get("input_pattern", "one_to_one"),
                output_pattern=cfg.get("output_pattern", "all_to_all"),
                proj_pattern=cfg.get("proj_pattern", "all_to_all"),
            )
            topologies.append(topo)
        return MultiStageTopology(topologies)


# ---------- pruning support (CP-5) ----------

def _bfs_undirected(num_nodes: int, src_list, dst_list, sources):
    """BFS over the undirected graph defined by (src, dst) edge list.

    Returns a list of (parent, distance) per node, where parent=-1 indicates
    source. If sources is empty, returns all -1 distances.
    """
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for s, d in zip(src_list, dst_list):
        s = int(s); d = int(d)
        if 0 <= s < num_nodes and 0 <= d < num_nodes and s != d:
            adj[s].append(d)
            adj[d].append(s)
    parent = [-1] * num_nodes
    dist = [-1] * num_nodes
    queue = []
    for src in sources:
        if 0 <= src < num_nodes and dist[src] < 0:
            dist[src] = 0
            queue.append(src)
    head = 0
    while head < len(queue):
        u = queue[head]; head += 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return parent, dist


def _connected_components(num_nodes: int, src_list, dst_list):
    """Return a list of node-id sets, one per connected component."""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for s, d in zip(src_list, dst_list):
        s = int(s); d = int(d)
        if 0 <= s < num_nodes and 0 <= d < num_nodes and s != d:
            adj[s].append(d)
            adj[d].append(s)
    seen = [False] * num_nodes
    components = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        comp = []
        queue = [start]
        seen[start] = True
        head = 0
        while head < len(queue):
            u = queue[head]; head += 1
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
        components.append(set(comp))
    return components


def validate_topology_degrees(
    src: list[int],
    dst: list[int],
    num_nodes: int,
    write_idx: list[int] | None,
    read_idx: list[int] | None,
) -> None:
    """Hard-error check: every (write_idx, read_idx) pair must be >1 hop apart.

    A 1-hop edge from a write_idx node to a read_idx node creates a direct
    input-to-output bypass that defeats capacitor dynamics. Built topologies
    that violate this must be redesigned (use a different graph, or pick
    different write_idx/read_idx).

    This check is silent if either write_idx or read_idx is None.
    """
    if write_idx is None or read_idx is None:
        return
    if not write_idx or not read_idx:
        return
    _, dists = _bfs_undirected(num_nodes, src, dst, write_idx)
    for w in write_idx:
        for r in read_idx:
            d = dists[r] if 0 <= r < num_nodes else -1
            if 0 < d <= 1:
                raise ValueError(
                    f"Topology degree-of-separation violation: write_idx node {w} "
                    f"is within {d} hop(s) of read_idx node {r} "
                    f"(must be >1). Choose a topology with greater degree of "
                    f"separation, or pick different write_idx/read_idx."
                )


def prune_stage(
    stage,
    edge_threshold: float = 0.01,
    node_threshold: float = 0.01,
    transfer_params: bool = True,
    write_idx: list[int] | None = None,
    read_idx: list[int] | None = None,
    protected_nodes: set[int] | None = None,
    min_read_nodes: int = 1,
    prune_nodes_by_gate: bool = False,
    leak_mode: str = "programmable",
    leak_constant: float | None = None,
    freeze_read: bool | None = None,
    freeze_boundary: bool | None = None,
    freeze_temporal_read: bool | None = None,
) -> tuple["DifferentialStage", dict[int, int]]:
    """Rebuild a DifferentialStage with edges and nodes removed.

    Edge pruning uses a joint Z+gate criterion: an edge is kept if
    ``(1 - P(Z)) * σ(z_logits) > edge_threshold``. This folds the Z-cell
    probability (gm_Z ≈ 0 ⇒ no current) and the edge gate (σ(z_logits) ≈ 0)
    into a single effective-activity score.

    Node pruning behavior (DEPRECATED: ``prune_nodes_by_gate`` is always
    treated as ``False``; node pruning is connectivity-only):
      - ``True``: emits a ``DeprecationWarning`` and is treated as
        ``False`` — no independent node pruning. All nodes start alive;
        they are only removed by the connectivity backstop (dead island
        purge) if they become fully disconnected from I/O after edge
        pruning.

    ``protected_nodes`` are forced to survive pruning regardless of their
    gate value. This is the input-side guard: write targets (the hidden
    nodes that receive input writes) are passed in here so they can never
    be silently pruned. The protection runs before the connectivity
    backstop so protected nodes also anchor their component.

    When ``write_idx`` and ``read_idx`` are both provided, a connectivity
    backstop runs after pruning:
      - BFS from write_idx; verify all read_idx are reachable.
      - Remove any node not in a component that contains at least one I/O
        node (dead-island purge); protected nodes are also part of ``io_nodes``.
      - Re-filter edges for surviving nodes.
      - If fewer than ``min_read_nodes`` read nodes survive, raise an error.

    Args:
        stage: Trained DifferentialStage with z_logits and u_logits.
        edge_threshold: Joint Z+gate threshold for edges.
        node_threshold: Gate threshold for nodes (only used when
            ``prune_nodes_by_gate=True``).
        transfer_params: If True, copy surviving logits/raw_mult/raw_leak
            values into the new stage. If False, the new stage starts with
            default initialization (used when retraining from scratch).
        write_idx: Optional compact node ids of write locations. Used for the
            connectivity backstop. Pass None to skip the check (intermediate
            stages in multi-stage networks).
        read_idx: Optional compact node ids of read locations. Used for the
            connectivity backstop. Pass None to skip the check.
        protected_nodes: Optional set of compact node ids that are forced to
            survive pruning. Use for write targets that must remain alive.
        min_read_nodes: Minimum number of read_idx nodes that must survive
            after pruning. If fewer survive, raises ValueError. Default 1.
        prune_nodes_by_gate: If True, prune nodes by ``σ(u_logits)``
            independently (legacy behavior). If False, skip node-gate
            pruning and only remove nodes via the connectivity backstop.

    Returns:
        A tuple ``(new_stage, node_remap)`` where:
          - ``new_stage`` is a new DifferentialStage with filtered src/dst,
            logits, raw_mult, z_logits (kept edges only), filtered
            raw_leak, u_logits (kept nodes only), and compact node IDs
            (0..N_new-1). When ``transfer_params=True`` all per-edge/
            per-node parameters are copied from the original stage.
          - ``node_remap`` is a dict mapping old compact node id -> new
            compact node id for all surviving nodes. Used by callers to
            remap I/O indices and transfer per-node I/O mapper weights.
    """
    from differential_stage import DifferentialStage

    # Shared-sense + crossbar readout is not yet supported by the pruner
    # (it has no notion of the sense bank or the dense crossbar).
    if getattr(stage, "_has_shared_readout", False):
        raise ValueError(
            "prune_stage: shared_sense readout not yet supported (the "
            "pruner cannot rebuild the sense bank + crossbar); use "
            "readout_mode='ota_mesh' for pruning runs."
        )

    z = stage.edge_gates().detach().cpu()
    # When budget was enabled during training, combine sigmoid * budget gate
    # as the effective edge mask. Budget-losers with still-positive sigmoid
    # should not survive pruning — their effective contribution is ≈ 0.
    _budget_enabled = getattr(stage, 'budget_enabled', False)
    if _budget_enabled:
        _bg = stage._compute_budget_gate().detach().cpu()
        z = z * _bg
    eff_score = z  # gate alone determines activity (no Z cell)

    keep_edge = eff_score > edge_threshold
    # DEPRECATED (deprecate-node-gates): prune_nodes_by_gate is ignored.
    # When True is passed, emit a deprecation warning and fall through to
    # the connectivity-only behavior (all nodes start alive; the backstop
    # removes fully disconnected nodes).
    if prune_nodes_by_gate:
        import warnings as _w
        _w.warn(
            "prune_nodes_by_gate=True is deprecated (deprecate-node-gates); "
            "treated as False. Node pruning is connectivity-only.",
            DeprecationWarning,
            stacklevel=2,
        )
        prune_nodes_by_gate = False
    keep_node = torch.ones(stage.num_nodes, dtype=torch.bool)

    src_old = stage.src.detach().cpu()
    dst_old = stage.dst.detach().cpu()

    # ----- protected nodes: forced to survive -----
    # Input-side guard: write targets must never be pruned, regardless of
    # their gate value. Applied before the connectivity backstop so they
    # also anchor the I/O-connected component.
    if protected_nodes is not None:
        for idx in protected_nodes:
            if 0 <= int(idx) < stage.num_nodes:
                keep_node[int(idx)] = True

    # ----- connectivity backstop -----
    enforce_io = write_idx is not None and read_idx is not None and len(write_idx) > 0 and len(read_idx) > 0
    if enforce_io:
        keep_edge_tmp = keep_edge & keep_node[src_old] & keep_node[dst_old]
        surv_src = src_old[keep_edge_tmp].tolist()
        surv_dst = dst_old[keep_edge_tmp].tolist()

        # BFS from write_idx; verify read_idx reachable
        _, dists = _bfs_undirected(stage.num_nodes, surv_src, surv_dst, list(write_idx))
        unreachable_reads = [r for r in read_idx if dists[r] < 0]
        if unreachable_reads:
            hint = (
                f"lower edge_threshold={edge_threshold}"
                " (node-gate pruning is deprecated and disabled)"
            )
            raise ValueError(
                f"prune_stage: read_idx {unreachable_reads} are unreachable from "
                f"write_idx {write_idx} after gate-based pruning. {hint.capitalize()}."
            )

        # Identify dead islands: nodes not in any I/O-connected component
        components = _connected_components(stage.num_nodes, surv_src, surv_dst)
        io_nodes = set(int(x) for x in write_idx) | set(int(x) for x in read_idx)
        if protected_nodes is not None:
            io_nodes |= {int(x) for x in protected_nodes}
        io_components = [c for c in components if not c.isdisjoint(io_nodes)]
        if not io_components:
            raise ValueError(
                f"prune_stage: no I/O-connected components remain after pruning. "
                f"Lower edge_threshold={edge_threshold} or node_threshold={node_threshold}."
            )
        io_keep = set().union(*io_components)
        dead_island_nodes = set(range(stage.num_nodes)) - io_keep
        for n in dead_island_nodes:
            keep_node[n] = False

    # ----- edge-only dead island purge (without I/O indices) -----
    # When no I/O indices are provided, the connectivity backstop above
    # doesn't run. In edge-only mode we still need to remove nodes that
    # have no surviving incident edges (they are dead islands).
    if not enforce_io and not prune_nodes_by_gate:
        surv_edges = keep_edge & keep_node[src_old] & keep_node[dst_old]
        has_incident = torch.zeros(stage.num_nodes, dtype=torch.bool)
        has_incident.scatter_(0, src_old[surv_edges], True)
        has_incident.scatter_(0, dst_old[surv_edges], True)
        if write_idx is not None:
            for idx in write_idx:
                has_incident[int(idx)] = True
        if read_idx is not None:
            for idx in read_idx:
                has_incident[int(idx)] = True
        if protected_nodes is not None:
            for idx in protected_nodes:
                has_incident[int(idx)] = True
        keep_node = keep_node & has_incident

    # ----- min_read_nodes guard -----
    # After dead-island purge, count surviving read nodes. If fewer than
    # min_read_nodes survive, abort the prune (readout would be empty).
    if read_idx is not None and len(read_idx) > 0:
        surviving_reads = [r for r in read_idx if bool(keep_node[int(r)])]
        if len(surviving_reads) < min_read_nodes:
            raise ValueError(
                f"prune_stage: only {len(surviving_reads)}/{len(read_idx)} read nodes "
                f"survived pruning (min_read_nodes={min_read_nodes}). "
                f"Lower edge_threshold={edge_threshold} or node_threshold={node_threshold}."
            )

    # Build node ID remapping: old_global -> new_compact.
    new_ids = torch.full((stage.num_nodes,), -1, dtype=torch.long)
    new_ids[keep_node] = torch.arange(int(keep_node.sum().item()))

    # Re-filter edges: keep only edges where both endpoints survive
    edge_mask = keep_edge & keep_node[src_old] & keep_node[dst_old]
    if int(edge_mask.sum().item()) == 0:
        hint = (
            f"lower edge_threshold={edge_threshold}"
            " (node-gate pruning is deprecated and disabled)"
        )
        raise ValueError(
            f"prune_stage: pruning removed all edges; consider {hint}"
        )

    new_src = new_ids[src_old[edge_mask]].tolist()
    new_dst = new_ids[dst_old[edge_mask]].tolist()

    num_nodes_new = int(keep_node.sum().item())
    node_idx_old = torch.nonzero(keep_node, as_tuple=True)[0]
    node_remap: dict[int, int] = {
        int(old_id): int(new_id)
        for old_id, new_id in zip(
            node_idx_old.tolist(),
            new_ids[keep_node].tolist(),
        )
    }

    # Remap write_idx for persistent drive when the original stage had drive.
    new_write_idx = None
    if hasattr(stage, '_has_drive') and stage._has_drive:
        drive_surviving = [int(idx) for idx in stage._drive_idx.tolist() if int(idx) in node_remap]
        new_write_idx = [node_remap[int(idx)] for idx in stage._drive_idx.tolist()
                         if int(idx) in node_remap] if drive_surviving else None

    is_simple = isinstance(stage.cell_lib, (SimpleEdgeLibrary, RealisticTanhLibrary, RealisticTanhUpgradeLibrary, FreeTanhLibrary, AntiParallelFreeTanhLibrary))
    is_simple_classic = isinstance(stage.cell_lib, SimpleEdgeLibrary)
    is_realistic = isinstance(stage.cell_lib, RealisticTanhLibrary)
    is_realistic_upgrade = isinstance(stage.cell_lib, RealisticTanhUpgradeLibrary)
    is_free_tanh = isinstance(stage.cell_lib, FreeTanhLibrary)
    is_anti_parallel = isinstance(stage.cell_lib, AntiParallelFreeTanhLibrary)
    if is_simple_classic:
        new_lib = SimpleEdgeLibrary(num_edges=len(new_src), mode=stage.cell_lib._mode)
    elif is_realistic:
        new_lib = RealisticTanhLibrary(
            num_edges=len(new_src),
            bias_enabled=stage.cell_lib._bias_enabled,
        )
    elif is_realistic_upgrade:
        old = stage.cell_lib
        new_lib = RealisticTanhUpgradeLibrary(
            num_edges=len(new_src),
            gm_min=old.gm_min,
            gm_max=old.gm_max,
            isat_min=old.isat_min,
            isat_max=old.isat_max,
            bias_enabled=old._bias_enabled,
        )
    elif is_free_tanh:
        old = stage.cell_lib
        new_lib = FreeTanhLibrary(
            num_edges=len(new_src),
            gm_min=old.gm_min,
            gm_max=old.gm_max,
            isat_min=old.isat_min,
            isat_max=old.isat_max,
            bias_enabled=old._bias_enabled,
            parallel_tanh_mult_enabled=old._parallel_tanh_mult_enabled,
        )
    elif is_anti_parallel:
        old = stage.cell_lib
        new_lib = AntiParallelFreeTanhLibrary(
            num_edges=len(new_src),
            kappa_min=old.kappa_min,
            kappa_max=old.kappa_max,
            gm_min=old.gm_min,
            gm_max=old.gm_max,
            isat_min=old.isat_min,
            isat_max=old.isat_max,
            theta_max=old.theta_max,
            theta_enabled=old._theta_enabled,
            use_isat_normalization=old._use_isat_normalization,
        )
    else:
        new_lib = stage.cell_lib

    has_ref = getattr(stage, '_has_ref', False)

    # Ref cell library recreation (for ref edges per node)
    if has_ref and stage.ref_cell_lib is not None:
        old_ref = stage.ref_cell_lib
        if isinstance(old_ref, SimpleEdgeLibrary):
            new_ref_lib = SimpleEdgeLibrary(num_edges=num_nodes_new, mode=old_ref._mode)
        elif isinstance(old_ref, RealisticTanhLibrary):
            new_ref_lib = RealisticTanhLibrary(
                num_edges=num_nodes_new, bias_enabled=old_ref._bias_enabled,
            )
        elif isinstance(old_ref, RealisticTanhUpgradeLibrary):
            new_ref_lib = RealisticTanhUpgradeLibrary(
                num_edges=num_nodes_new, gm_min=old_ref.gm_min, gm_max=old_ref.gm_max,
                isat_min=old_ref.isat_min, isat_max=old_ref.isat_max,
                bias_enabled=old_ref._bias_enabled,
            )
        elif isinstance(old_ref, FreeTanhLibrary):
            new_ref_lib = FreeTanhLibrary(
                num_edges=num_nodes_new, gm_min=old_ref.gm_min, gm_max=old_ref.gm_max,
                isat_min=old_ref.isat_min, isat_max=old_ref.isat_max,
                bias_enabled=old_ref._bias_enabled,
            )
        elif isinstance(old_ref, AntiParallelFreeTanhLibrary):
            new_ref_lib = AntiParallelFreeTanhLibrary(
                num_edges=num_nodes_new, kappa_min=old_ref.kappa_min, kappa_max=old_ref.kappa_max,
                gm_min=old_ref.gm_min, gm_max=old_ref.gm_max,
                isat_min=old_ref.isat_min, isat_max=old_ref.isat_max,
                theta_max=old_ref.theta_max, theta_enabled=old_ref._theta_enabled,
                use_isat_normalization=old_ref._use_isat_normalization,
            )
        else:
            new_ref_lib = old_ref
    else:
        new_ref_lib = None

    new_stage = DifferentialStage(
        num_nodes=num_nodes_new,
        src=new_src,
        dst=new_dst,
        cell_lib=new_lib,
        c_eff=stage.c_eff,
        x_max=stage.x_max,
        clip_current=stage.clip_current,
        clip_softness=stage.clip_softness,
        write_idx=new_write_idx,
        leak_mode=leak_mode,
        leak_constant=leak_constant,
        freeze_read=stage.freeze_read if freeze_read is None else freeze_read,
        freeze_boundary=stage.freeze_boundary if freeze_boundary is None else freeze_boundary,
        freeze_temporal_read=stage.freeze_temporal_read if freeze_temporal_read is None else freeze_temporal_read,
        boundary_cell_lib=stage.boundary_cell_lib,
        enable_ref_edges=has_ref,
        ref_cell_lib=new_ref_lib,
        vca_enabled=False,
        vca_rank=VCA["rank"],
        vca_in_dim=0,
        core_refresh_interval=int(getattr(stage, "core_refresh_interval", 0)),
    )

    if transfer_params:
        with torch.no_grad():
            edge_idx_old = torch.nonzero(edge_mask, as_tuple=True)[0]
            if is_simple_classic:
                new_stage.cell_lib.param.data.copy_(stage.cell_lib.param.data[:, edge_idx_old].cpu())
            elif is_realistic:
                new_stage.cell_lib.alpha_raw.data.copy_(
                    stage.cell_lib.alpha_raw.data[edge_idx_old].cpu()
                )
                if hasattr(new_stage.cell_lib, "bias_raw") and hasattr(stage.cell_lib, "bias_raw"):
                    new_stage.cell_lib.bias_raw.data.copy_(
                        stage.cell_lib.bias_raw.data[edge_idx_old].cpu()
                    )
            elif is_realistic_upgrade:
                for name in ("alpha_raw", "gm_raw", "isat_raw"):
                    new_stage.cell_lib.get_parameter(name).data.copy_(
                        stage.cell_lib.get_parameter(name).data[edge_idx_old].cpu()
                    )
                if hasattr(new_stage.cell_lib, "bias_raw") and hasattr(stage.cell_lib, "bias_raw"):
                    new_stage.cell_lib.bias_raw.data.copy_(
                        stage.cell_lib.bias_raw.data[edge_idx_old].cpu()
                    )
            elif is_free_tanh:
                for name in ("a_raw", "b_raw", "s_raw", "gm_raw", "isat_raw"):
                    new_stage.cell_lib.get_parameter(name).data.copy_(
                        stage.cell_lib.get_parameter(name).data[edge_idx_old].cpu()
                    )
                if hasattr(new_stage.cell_lib, "theta_raw") and hasattr(stage.cell_lib, "theta_raw"):
                    new_stage.cell_lib.theta_raw.data.copy_(
                        stage.cell_lib.theta_raw.data[edge_idx_old].cpu()
                    )
            elif is_anti_parallel:
                for name in ("kappa_raw", "gm_raw", "isat_raw"):
                    new_stage.cell_lib.get_parameter(name).data.copy_(
                        stage.cell_lib.get_parameter(name).data[edge_idx_old].cpu()
                    )
                if hasattr(new_stage.cell_lib, "theta_raw") and hasattr(stage.cell_lib, "theta_raw"):
                    new_stage.cell_lib.theta_raw.data.copy_(
                        stage.cell_lib.theta_raw.data[edge_idx_old].cpu()
                    )
            old_z_logits = stage.z_logits.data[edge_idx_old].cpu()
            if _budget_enabled:
                # Rescale z_logits so that σ(z_new) = σ(z_old) * budget_gate_old.
                # This folds the budget attenuation into the surviving edge
                # gates so that when budget is disabled at Phase C start the
                # effective edge mask is identical to the budget-on regime.
                _bg_surv = _bg[edge_idx_old]
                combined = torch.sigmoid(old_z_logits) * _bg_surv
                combined = torch.clamp(combined, min=1e-6, max=1.0 - 1e-6)
                old_z_logits = torch.logit(combined)
            new_stage.z_logits.data.copy_(old_z_logits)
        with torch.no_grad():
            if hasattr(stage, 'raw_leak') and hasattr(new_stage, 'raw_leak'):
                new_stage.raw_leak.data.copy_(stage.raw_leak.data[node_idx_old].cpu())
            new_stage.u_logits.data.copy_(stage.u_logits.data[node_idx_old].cpu())
        # Transfer raw_drive_g for surviving driven nodes.
        if new_write_idx is not None and hasattr(stage, 'raw_drive_g'):
            old_drive_idx_list = stage._drive_idx.tolist()
            surv_mask = torch.tensor(
                [int(idx) in node_remap for idx in old_drive_idx_list],
                dtype=torch.bool,
            )
            with torch.no_grad():
                new_stage.raw_drive_g.data.copy_(stage.raw_drive_g.data[surv_mask].cpu())

        # Transfer reference-edge parameters (if stage had ref edges).
        if has_ref and transfer_params:
            with torch.no_grad():
                if hasattr(new_stage, "raw_vref") and hasattr(stage, "raw_vref"):
                    new_stage.raw_vref.data.copy_(stage.raw_vref.data.cpu())
                if hasattr(new_stage, "ref_z_logits") and hasattr(stage, "ref_z_logits"):
                    old_ref_z = stage.ref_z_logits.data[node_idx_old].cpu()
                    if getattr(stage, "budget_enabled", False):
                        _bg = stage._compute_budget_gate().detach().cpu()
                        combined = torch.sigmoid(old_ref_z) * _bg[node_idx_old]
                        combined = combined.clamp(1e-6, 1.0 - 1e-6)
                        old_ref_z = torch.logit(combined)
                    new_stage.ref_z_logits.data.copy_(old_ref_z)
                # Transfer per-node ref_cell_lib parameters (one ref edge per node).
                if hasattr(new_stage, "ref_cell_lib") and new_stage.ref_cell_lib is not None \
                   and hasattr(stage, "ref_cell_lib") and stage.ref_cell_lib is not None:
                    rsrc = stage.ref_cell_lib
                    rdst = new_stage.ref_cell_lib
                    if hasattr(rdst, "param") and hasattr(rsrc, "param"):
                        rdst.param.data.copy_(rsrc.param.data[:, node_idx_old].cpu())
                    elif hasattr(rdst, "alpha_raw") and hasattr(rsrc, "alpha_raw"):
                        rdst.alpha_raw.data.copy_(rsrc.alpha_raw.data[node_idx_old].cpu())
                        if hasattr(rdst, "bias_raw") and hasattr(rsrc, "bias_raw"):
                            rdst.bias_raw.data.copy_(rsrc.bias_raw.data[node_idx_old].cpu())
                    elif hasattr(rdst, "kappa_raw") and hasattr(rsrc, "kappa_raw"):
                        for name in ("kappa_raw", "gm_raw", "isat_raw"):
                            getattr(rdst, name).data.copy_(
                                getattr(rsrc, name).data[node_idx_old].cpu()
                            )
                        if hasattr(rdst, "theta_raw") and hasattr(rsrc, "theta_raw"):
                            rdst.theta_raw.data.copy_(rsrc.theta_raw.data[node_idx_old].cpu())
                    elif hasattr(rdst, "a_raw") and hasattr(rsrc, "a_raw"):
                        for name in ("a_raw", "b_raw", "s_raw", "gm_raw", "isat_raw"):
                            getattr(rdst, name).data.copy_(
                                getattr(rsrc, name).data[node_idx_old].cpu()
                            )
                        if hasattr(rdst, "theta_raw") and hasattr(rsrc, "theta_raw"):
                            rdst.theta_raw.data.copy_(rsrc.theta_raw.data[node_idx_old].cpu())
                    elif hasattr(rdst, "alpha_raw") and hasattr(rsrc, "alpha_raw"):
                        for name in ("alpha_raw", "gm_raw", "isat_raw"):
                            getattr(rdst, name).data.copy_(
                                getattr(rsrc, name).data[node_idx_old].cpu()
                            )
                        if hasattr(rdst, "bias_raw") and hasattr(rsrc, "bias_raw"):
                            rdst.bias_raw.data.copy_(rsrc.bias_raw.data[node_idx_old].cpu())

    return new_stage, node_remap


def prune_network(
    net: "KirchhoffNet",
    edge_threshold: float = 0.01,
    node_threshold: float = 0.01,
    transfer_params: bool = True,
    write_idx: list[int] | None = None,
    read_idx: list[int] | None = None,
    min_read_nodes: int = 1,
    prune_nodes_by_gate: bool = False,
    freeze_read: bool | None = None,
    freeze_boundary: bool | None = None,
    freeze_temporal_read: bool | None = None,
) -> tuple["KirchhoffNet", list[dict[int, int]]]:
    """Apply prune_stage to every stage of a KirchhoffNet core.

    Stage widths may change after pruning, so StageTransfer modules are
    reinitialized to match the new active-node counts. Returns a new
    KirchhoffNet with the same t_span/num_steps.

    ``write_idx`` and ``read_idx`` (when provided) are routed to the first
    and last stage respectively for the connectivity backstop. Intermediate
    stages skip the check.

    ``write_idx`` is also passed to stage 0 (and single-stage networks) as
    ``protected_nodes`` so write targets are guaranteed to survive pruning.
    Read nodes are NOT protected; elastic readout pruning is allowed, but
    the prune fails if fewer than ``min_read_nodes`` read nodes survive.

    ``prune_nodes_by_gate`` DEPRECATED (deprecate-node-gates): forwarded
    to each stage's ``prune_stage`` but has no effect; node pruning is
    always connectivity-only.

    Returns:
        A tuple ``(new_core, stage_remaps)`` where:
          - ``new_core`` is the pruned KirchhoffNet core.
          - ``stage_remaps`` is a list of dicts, one per stage, mapping
            old compact node id -> new compact node id for surviving
            nodes. ``stage_remaps[0]`` is used to remap write targets,
            ``stage_remaps[-1]`` is used to remap read targets.
    """
    from kirchhoff_net import KirchhoffNet

    n_stages = len(net.stages)
    new_stages = []
    stage_remaps: list[dict[int, int]] = []
    for i, s in enumerate(net.stages):
        if n_stages == 1:
            wi, ri = write_idx, read_idx
            protected = set(write_idx) if write_idx else set()
        elif i == 0:
            wi, ri = write_idx, None
            protected = set(write_idx) if write_idx else set()
        elif i == n_stages - 1:
            wi, ri = None, read_idx
            protected = set()
        else:
            wi, ri = None, None
            protected = set()
        # Driven nodes must survive pruning in every stage (persistent drive).
        if hasattr(s, '_has_drive') and s._has_drive:
            protected.update(int(idx) for idx in s._drive_idx.tolist())
        protected = protected if protected else None
        new_s, remap = prune_stage(
            s,
            edge_threshold=edge_threshold,
            node_threshold=node_threshold,
            transfer_params=transfer_params,
            write_idx=wi,
            read_idx=ri,
            protected_nodes=protected,
            min_read_nodes=min_read_nodes,
            prune_nodes_by_gate=prune_nodes_by_gate,
            freeze_read=freeze_read,
            freeze_boundary=freeze_boundary,
            freeze_temporal_read=freeze_temporal_read,
        )
        new_stages.append(new_s)
        stage_remaps.append(remap)
    new_widths = [s.num_nodes for s in new_stages]
    new_transfers = []
    from stage_transfer import StageTransfer
    for i in range(len(new_stages) - 1):
        activation = getattr(net.transfers[i], 'activation', 'none')
        residual_rank = getattr(net.transfers[i], 'residual_rank', -1)
        drive_indices = None
        if hasattr(new_stages[i], '_has_drive') and new_stages[i]._has_drive:
            drive_indices = new_stages[i]._drive_idx.tolist()
        new_transfers.append(StageTransfer(new_widths[i], new_widths[i + 1],
                                           activation=activation,
                                           drive_mask=drive_indices,
                                           residual_rank=residual_rank))

    return KirchhoffNet(
        stages=new_stages,
        transfers=new_transfers,
        stage_times=list(net.stage_times),
        stage_steps=list(net.stage_steps),
    ), stage_remaps


# ---------- integration with DifferentialStage ----------

def validate_topology(topo: SparseTopology, max_hidden_density: float = 0.5) -> None:
    """Assert that topo satisfies the spec's sanity checks. Raises ValueError on failure."""
    if len(topo.src) != len(topo.dst):
        raise ValueError("src/dst length mismatch")
    if len(topo.edge_type) != len(topo.src):
        raise ValueError("edge_type length must equal src/dst length")
    if len(topo.node_kind) != topo.num_nodes:
        raise ValueError("node_kind length must equal num_nodes")
    if topo.src or topo.dst:
        if max(topo.src + topo.dst, default=-1) >= topo.num_nodes:
            raise ValueError("Edge endpoint >= num_nodes")
    for s, d in zip(topo.src, topo.dst):
        if s == d:
            raise ValueError(f"Self-loop not allowed: edge {s}->{d}")
    n_h = len(topo.hidden_node_ids)
    if n_h > 0:
        # Density is measured on UNIQUE directed (src, dst) pairs, not raw
        # edge count. This keeps the metric aligned with graph connectivity
        # rather than counting parallel branches (which are intentional
        # duplicates created by ``repeat_edges``). For bidirectional graphs
        # each undirected pair contributes 2 directed pairs (i->j, j->i),
        # and for ``edge_repeats=N`` each pair still counts as 1 unique
        # directed edge. The check is intentionally permissive to allow
        # bidirectional mode (e.g., grid_graph 5x5 with kernel_size=3
        # reaches 0.27 bidirectional).
        max_edges = n_h * (n_h - 1)  # max directed pairs
        unique_directed_pairs = len({
            (s, d)
            for s, d, t in zip(topo.src, topo.dst, topo.edge_type)
            if t == EDGE_TYPE_HIDDEN
        })
        if n_h > 32 and (unique_directed_pairs / max_edges) > max_hidden_density:
            raise ValueError(
                f"Hidden core too dense: {unique_directed_pairs} unique directed "
                f"pairs / {max_edges} max > {max_hidden_density}"
            )
    for i in topo.input_node_ids:
        if i not in topo.src:
            raise ValueError(f"Input node {i} has no outgoing edge")
    for o in topo.output_node_ids:
        if o not in topo.dst:
            raise ValueError(f"Output node {o} has no incoming edge")


def topology_to_stage(
    topo: SparseTopology,
    cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary,
    c_eff: float | None = None,
    x_max: float | None = None,
    clip_current: float | None = None,
    clip_softness: float | None = None,
    write_idx: list[int] | None = None,
    leak_mode: str = "programmable",
    leak_constant: float | None = None,
    read_only_source: bool = False,
    freeze_read: bool = False,
    freeze_boundary: bool = False,
    freeze_temporal_read: bool = False,
    boundary_src: list[int] | None = None,
    boundary_dst: list[int] | None = None,
    boundary_cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary | None = None,
    enable_ref_edges: bool = False,
    output_ode_src: list[int] | None = None,
    output_ode_dst: list[int] | None = None,
    output_ode_cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary | None = None,
    readout_mode: str = "ota_mesh",
    readout_senses_per_node: int = 1,
    readout_sense_src: list[int] | None = None,
    readout_sense_cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary | None = None,
    readout_crossbar_shape: tuple[int, int] | None = None,
    output_ode_node_count: int = 0,
    vca_enabled: bool = False,
    vca_rank: int = 2,
    vca_in_dim: int = 0,
    vca_core_enabled: bool = False,
    vca_gate_shunt: bool = False,
    vca_separate_core_bus: bool = False,
    vca_bias: bool | None = None,
    core_refresh_interval: int = 0,
) -> tuple[DifferentialStage, list[int], dict[int, int]]:
    """Convert a SparseTopology into a DifferentialStage.

    Input and output edges are filtered out (they are write/read taps, not
    ODE branches). Hidden + projection nodes are remapped to compact 0..N-1.

    Args:
        write_idx: Hidden-node indices (in user-facing 0..hid_count-1
            coordinates) that should receive persistent drive current.
            These indices map directly to the first hid_count positions
            of the compact state vector.
        leak_mode: ``"programmable"`` (default) or ``"non-programmable"``.
            Controls whether the stage has a learnable per-node ``raw_leak``
            or a fixed constant.
        leak_constant: Fixed leak value when ``leak_mode="non-programmable"``.
        vca_enabled: When ``True``, the stage builds per-edge VCA
            (Voltage-Controlled Amplifier) embeddings for boundary and
            temporal-readout edges plus a shared input projection matrix.
            The gate per gated edge is ``2*sigmoid(b_e + (u^T W) v_e)`` when
            bias is enabled (otherwise ``2*sigmoid((u^T W) v_e)``). Requires at
            least one of ``boundary_src`` or ``output_ode_src`` to be
            non-empty (else raises ``ValueError``).
        vca_rank: Projection rank ``r`` for the low-rank VCA. Default 2.
        vca_in_dim: Input feature dimension used to size ``W``. Must
            match the network-wide ``in_dim``. Default 0 (unset).
            ``None`` uses ``config.INIT["leak_constant"]``.
        freeze_read: When ``True``, edge currents are computed once from the
            initial state and held constant across all Heun / DEQ iterations
            inside the stage. Forwarded to ``DifferentialStage``.
        boundary_src: List of input-terminal indices for sparse OTA edges
            from fixed-voltage boundary terminals into the dynamic fabric.
            Forwarded to ``DifferentialStage`` unchanged.
        boundary_dst: List of target dynamic-node indices for the boundary
            OTA edges (compact coordinates, must be in ``[0, num_nodes)``).
        boundary_cell_lib: Cell library instance used to compute boundary
            edge currents; must match the cell type of ``cell_lib`` and be
            sized for ``len(boundary_src)`` edges.
        enable_ref_edges: When ``True``, every node gets one OTA edge to a
            global per-stage learnable reference voltage ``Vref``. Forwarded
            to ``DifferentialStage``.
        output_ode_src: List of source node indices (compact coordinates) for
            temporal-readout OTA edges. The source is a hidden or projection
            node and is read-only (its voltage drives the OTA current, no
            current is drained from it). Forwarded to ``DifferentialStage``
            unchanged.
        output_ode_dst: List of destination node indices (compact coordinates,
            must be in ``[0, num_nodes)``) for temporal-readout OTA edges.
            Typically the output ODE accumulator region
            ``[len(active_nodes), num_nodes)`` but any valid index works.
        output_ode_cell_lib: Cell library instance used to compute temporal-
            readout edge currents ``I_OTA(x_src, x_dst)``. Must match the
            cell type of ``cell_lib`` and be sized for ``len(output_ode_src)``
            edges. Required when ``output_ode_src``/``output_ode_dst`` are
            provided.
        readout_mode: ``"ota_mesh"`` (default) builds the legacy h*d_out
            temporal-readout OTA mesh. ``"shared_sense"`` builds one sense
            OTA per hidden node (optionally two at different operating
            points) + a dense crossbar readout instead; mutually exclusive
            with ``output_ode_src``/``output_ode_dst``. Forwarded to
            ``DifferentialStage``.
        readout_senses_per_node: ``1`` or ``2`` senses per hidden node in
            shared mode (``2`` adds an init jitter to the second copy's gm so
            the operating points differ).
        readout_sense_src: List of compact source node indices, one per
            sense cell (``s*h`` entries). Forwarded to ``DifferentialStage``.
        readout_sense_cell_lib: Template cell library sized for the sense
            bank; cloned per stage so each stage owns its OTA parameters.
        readout_crossbar_shape: ``(d_out, s*h)`` crossbar shape; forwarded to
            ``DifferentialStage``.
        output_ode_node_count: Number of output ODE accumulator nodes appended
            to this stage's state. When ``> 0``, ``num_nodes`` passed to
            ``DifferentialStage`` is ``len(active_nodes) + output_ode_node_count``
            so the ODE state vector includes both the core nodes and the
            output accumulators. Defaults to ``0``.

    Returns:
        stage: DifferentialStage
        active_nodes: list of global node ids that are evolved by this stage
        id_map: dict mapping global node id -> compact id
    """
    core_mask = [t in (EDGE_TYPE_HIDDEN, EDGE_TYPE_PROJ) for t in topo.edge_type]
    core_src = [topo.src[i] for i, m in enumerate(core_mask) if m]
    core_dst = [topo.dst[i] for i, m in enumerate(core_mask) if m]

    active_nodes = sorted(set(topo.hidden_node_ids + topo.proj_node_ids))
    id_map = {old: new for new, old in enumerate(active_nodes)}

    remapped_src = [id_map[s] for s in core_src]
    remapped_dst = [id_map[d] for d in core_dst]

    if isinstance(cell_lib, SimpleEdgeLibrary):
        cell_lib = SimpleEdgeLibrary(num_edges=len(remapped_src), mode=cell_lib._mode)
    elif isinstance(cell_lib, RealisticTanhLibrary):
        cell_lib = RealisticTanhLibrary(
            num_edges=len(remapped_src),
            bias_enabled=cell_lib._bias_enabled,
        )
    elif isinstance(cell_lib, RealisticTanhUpgradeLibrary):
        cell_lib = RealisticTanhUpgradeLibrary(
            num_edges=len(remapped_src),
            gm_min=cell_lib.gm_min,
            gm_max=cell_lib.gm_max,
            isat_min=cell_lib.isat_min,
            isat_max=cell_lib.isat_max,
            bias_enabled=cell_lib._bias_enabled,
        )
    elif isinstance(cell_lib, FreeTanhLibrary):
        cell_lib = FreeTanhLibrary(
            num_edges=len(remapped_src),
            gm_min=cell_lib.gm_min,
            gm_max=cell_lib.gm_max,
            isat_min=cell_lib.isat_min,
            isat_max=cell_lib.isat_max,
            bias_enabled=cell_lib._bias_enabled,
            parallel_tanh_mult_enabled=cell_lib._parallel_tanh_mult_enabled,
        )
    elif isinstance(cell_lib, AntiParallelFreeTanhLibrary):
        cell_lib = AntiParallelFreeTanhLibrary(
            num_edges=len(remapped_src),
            kappa_min=cell_lib.kappa_min,
            kappa_max=cell_lib.kappa_max,
            gm_min=cell_lib.gm_min,
            gm_max=cell_lib.gm_max,
            isat_min=cell_lib.isat_min,
            isat_max=cell_lib.isat_max,
            theta_max=cell_lib.theta_max,
            theta_enabled=cell_lib._theta_enabled,
            use_isat_normalization=cell_lib._use_isat_normalization,
        )

    # Reference-edge cell library: one OTA per node (num_nodes edges). Same
    # cell type/config as the core lib so reference edges behave like normal
    # OTA cells, just connected to a global per-stage learnable Vref instead
    # of inter-node voltages.
    ref_cell_lib = None
    if enable_ref_edges:
        n_ref = len(active_nodes) + int(output_ode_node_count)
        if isinstance(cell_lib, SimpleEdgeLibrary):
            ref_cell_lib = SimpleEdgeLibrary(num_edges=n_ref, mode=cell_lib._mode)
        elif isinstance(cell_lib, RealisticTanhLibrary):
            ref_cell_lib = RealisticTanhLibrary(
                num_edges=n_ref,
                bias_enabled=cell_lib._bias_enabled,
            )
        elif isinstance(cell_lib, RealisticTanhUpgradeLibrary):
            ref_cell_lib = RealisticTanhUpgradeLibrary(
                num_edges=n_ref,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                bias_enabled=cell_lib._bias_enabled,
            )
        elif isinstance(cell_lib, FreeTanhLibrary):
            ref_cell_lib = FreeTanhLibrary(
                num_edges=n_ref,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                bias_enabled=cell_lib._bias_enabled,
                parallel_tanh_mult_enabled=cell_lib._parallel_tanh_mult_enabled,
            )
        elif isinstance(cell_lib, AntiParallelFreeTanhLibrary):
            ref_cell_lib = AntiParallelFreeTanhLibrary(
                num_edges=n_ref,
                kappa_min=cell_lib.kappa_min,
                kappa_max=cell_lib.kappa_max,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                theta_max=cell_lib.theta_max,
                theta_enabled=cell_lib._theta_enabled,
                use_isat_normalization=cell_lib._use_isat_normalization,
            )

    # Shared-sense readout: clone the template sense cell library per stage
    # (mirrors the output_ode clone below) so each stage owns its own sense
    # OTA parameters. The sense src layout in ``build_net_from_config`` is
    # **interleaved** (one node's s senses sit adjacent in the buffer), so
    # ``readout_sense_cell_lib.gm_raw`` is arranged as
    # ``[sense0_node0, sense1_node0, sense0_node1, sense1_node1, ...]`` for
    # ``s=2``. With ``readout_senses_per_node == 2`` we add a +0.5 init
    # jitter to the k=1 sense of every node (``gm_raw[1::2]``) so the two
    # senses on the same node start at different saturation points; the k=0
    # sense of every node stays at the default ``-5.0`` low-current init.
    n_sense_clone = len(readout_sense_src) if readout_sense_src else 0
    if readout_sense_cell_lib is not None and n_sense_clone > 0:
        if isinstance(readout_sense_cell_lib, SimpleEdgeLibrary):
            readout_sense_cell_lib = SimpleEdgeLibrary(
                num_edges=n_sense_clone, mode=readout_sense_cell_lib._mode,
            )
        elif isinstance(readout_sense_cell_lib, RealisticTanhLibrary):
            readout_sense_cell_lib = RealisticTanhLibrary(
                num_edges=n_sense_clone,
                bias_enabled=readout_sense_cell_lib._bias_enabled,
            )
        elif isinstance(readout_sense_cell_lib, RealisticTanhUpgradeLibrary):
            readout_sense_cell_lib = RealisticTanhUpgradeLibrary(
                num_edges=n_sense_clone,
                gm_min=readout_sense_cell_lib.gm_min,
                gm_max=readout_sense_cell_lib.gm_max,
                isat_min=readout_sense_cell_lib.isat_min,
                isat_max=readout_sense_cell_lib.isat_max,
                bias_enabled=readout_sense_cell_lib._bias_enabled,
            )
        elif isinstance(readout_sense_cell_lib, FreeTanhLibrary):
            readout_sense_cell_lib = FreeTanhLibrary(
                num_edges=n_sense_clone,
                gm_min=readout_sense_cell_lib.gm_min,
                gm_max=readout_sense_cell_lib.gm_max,
                isat_min=readout_sense_cell_lib.isat_min,
                isat_max=readout_sense_cell_lib.isat_max,
                bias_enabled=readout_sense_cell_lib._bias_enabled,
                parallel_tanh_mult_enabled=readout_sense_cell_lib._parallel_tanh_mult_enabled,
            )
            if int(readout_senses_per_node) == 2:
                # Interleaved layout: odd-indexed cells are sense k=1 of
                # each hidden node. Jitter only those so every node
                # contributes one default-init and one shifted-init sense.
                with torch.no_grad():
                    readout_sense_cell_lib.gm_raw.data[1::2] += 0.5
        elif isinstance(readout_sense_cell_lib, AntiParallelFreeTanhLibrary):
            readout_sense_cell_lib = AntiParallelFreeTanhLibrary(
                num_edges=n_sense_clone,
                kappa_min=readout_sense_cell_lib.kappa_min,
                kappa_max=readout_sense_cell_lib.kappa_max,
                gm_min=readout_sense_cell_lib.gm_min,
                gm_max=readout_sense_cell_lib.gm_max,
                isat_min=readout_sense_cell_lib.isat_min,
                isat_max=readout_sense_cell_lib.isat_max,
                theta_max=readout_sense_cell_lib.theta_max,
                theta_enabled=readout_sense_cell_lib._theta_enabled,
                use_isat_normalization=readout_sense_cell_lib._use_isat_normalization,
            )

    # Clone the temporal-readout cell library per stage so each stage owns
    # its own OTA parameter set for the readout edges. Without this clone,
    # the same library object would be registered as a submodule on every
    # DifferentialStage and its parameters would appear N times in
    # named_parameters() (causing the optimizer to apply N gradient steps
    # per optimization pass). Same cloning pattern as the core cell_lib
    # above and ref_cell_lib above.
    if output_ode_cell_lib is not None and len(output_ode_src or []) > 0:
        n_out_ode = len(output_ode_src)
        if isinstance(cell_lib, SimpleEdgeLibrary):
            output_ode_cell_lib = SimpleEdgeLibrary(
                num_edges=n_out_ode, mode=cell_lib._mode,
            )
        elif isinstance(cell_lib, RealisticTanhLibrary):
            output_ode_cell_lib = RealisticTanhLibrary(
                num_edges=n_out_ode,
                bias_enabled=cell_lib._bias_enabled,
            )
        elif isinstance(cell_lib, RealisticTanhUpgradeLibrary):
            output_ode_cell_lib = RealisticTanhUpgradeLibrary(
                num_edges=n_out_ode,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                bias_enabled=cell_lib._bias_enabled,
            )
        elif isinstance(cell_lib, FreeTanhLibrary):
            output_ode_cell_lib = FreeTanhLibrary(
                num_edges=n_out_ode,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                bias_enabled=cell_lib._bias_enabled,
                parallel_tanh_mult_enabled=cell_lib._parallel_tanh_mult_enabled,
            )
        elif isinstance(cell_lib, AntiParallelFreeTanhLibrary):
            output_ode_cell_lib = AntiParallelFreeTanhLibrary(
                num_edges=n_out_ode,
                kappa_min=cell_lib.kappa_min,
                kappa_max=cell_lib.kappa_max,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                theta_max=cell_lib.theta_max,
                theta_enabled=cell_lib._theta_enabled,
                use_isat_normalization=cell_lib._use_isat_normalization,
            )

    stage = DifferentialStage(
        num_nodes=len(active_nodes) + int(output_ode_node_count),
        src=remapped_src,
        dst=remapped_dst,
        cell_lib=cell_lib,
        c_eff=c_eff,
        x_max=x_max,
        clip_current=clip_current,
        clip_softness=clip_softness,
        write_idx=write_idx,
        leak_mode=leak_mode,
        leak_constant=leak_constant,
        read_only_source=read_only_source,
        freeze_read=freeze_read,
        freeze_boundary=freeze_boundary,
        freeze_temporal_read=freeze_temporal_read,
        boundary_src=boundary_src,
        boundary_dst=boundary_dst,
        boundary_cell_lib=boundary_cell_lib,
        enable_ref_edges=enable_ref_edges,
        ref_cell_lib=ref_cell_lib,
        output_ode_src=output_ode_src,
        output_ode_dst=output_ode_dst,
        output_ode_cell_lib=output_ode_cell_lib,
        readout_mode=readout_mode,
        readout_senses_per_node=readout_senses_per_node,
        readout_sense_src=readout_sense_src,
        readout_sense_cell_lib=readout_sense_cell_lib,
        readout_crossbar_shape=readout_crossbar_shape,
        vca_enabled=vca_enabled,
        vca_rank=vca_rank,
        vca_in_dim=vca_in_dim,
        vca_core_enabled=vca_core_enabled,
        vca_gate_shunt=vca_gate_shunt,
        vca_separate_core_bus=vca_separate_core_bus,
        vca_bias=vca_bias,
        core_refresh_interval=core_refresh_interval,
    )
    return stage, active_nodes, id_map


# ---------- factory: config -> network ----------

def build_net_from_preset(
    preset_name: str,
    cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary,
    write_mode: str | None = None,
    read_mode: str | None = None,
    write_idx: list[int] | None = None,
    read_idx: list[int] | None = None,
    enable_drive: bool = False,
    drive_mode: str = "fan_out",
    leak_mode: str = "programmable",
    leak_constant: float | None = None,
    encoder_type: str | None = None,
    decoder_type: str | None = None,
    encoder_hidden_dim: int | None = None,
    decoder_hidden_dim: int | None = None,
    read_only_source: bool = False,
    interstage_activation: str = "none",
    interstage_residual_rank: int = -1,
    freeze_read: bool = False,
    freeze_boundary: bool = False,
    freeze_temporal_read: bool = False,
    enable_skip_linear: bool = False,
    boundary_fan_out: dict[int, list[int]] | None = None,
    enable_ref_edges: bool = False,
    enable_temporal_readout: bool = False,
    readout_mode: str = "ota_mesh",
    readout_senses_per_node: int = 1,
    vca_enabled: bool = False,
    vca_rank: int | None = None,
    vca_core_enabled: bool = False,
    vca_gate_shunt: bool = False,
    vca_separate_core_bus: bool = False,
    vca_bias: bool | None = None,
    x_max: float | None = None,
    c_eff: float | None = None,
    core_refresh_interval: int = 0,
):
    """Build a full KirchhoffNetWithIO from a config.PRESETS entry.

    ``readout_mode`` / ``readout_senses_per_node`` select the readout
    family (see ``build_net_from_config``): ``ota_mesh`` (default, legacy
    h*d_out temporal OTA mesh), ``shared_sense`` with 1 sense per hidden
    node, or ``shared_sense`` with 2 senses per node (the two copies start
    at different gm operating points so they explore distinct saturating
    regimes). ``enable_temporal_readout`` must be ``True`` whenever the
    readout tail (``out_dim`` output-ODE accumulator nodes appended to each
    stage) is wanted — i.e. for all three CLI ``--readout`` choices
    (temporal, shared, shared-x2) — while ``readout_mode`` switches only
    the edge family (mesh vs sense bank).

    Resolution precedence: explicit value > preset value > hardcoded default.

    * write_mode:  "one_to_one" | "dense" | "fan_out" | "sparse_proj".
                   When ``None`` (default), use the preset's ``write_mode``
                   if present, else ``"one_to_one"``.
    * read_mode:   "sparse" | "dense".  When ``None`` (default), use the
                   preset's ``read_mode`` if present, else ``"sparse"``.
    * write_idx / read_idx: explicit index lists override preset values.
    * drive_mode:  "fan_out" | "projection". When ``enable_drive=True``,
                   controls the per-stage drive mapper architecture.
                   ``"fan_out"`` (default) uses FanOutInputMapper with
                   per-input scalars (gain, bias) per driven node. With
                   ``"projection"`` for write_mode='sparse_proj' or
                   'one_to_one', uses ProjectedSparseInputMapper with a
                   learned nn.Linear projection matching the input mapper.
                   Only meaningful when enable_drive=True.
    * leak_mode / leak_constant: forwarded to :func:`topology_to_stage`.
    * encoder_type: "linear" | "residual_tanh". When "residual_tanh",
                    ``InputMapper`` is replaced by ``ResidualTanhInputMapper``
                    (dense write mode only). Sparse write modes fall back
                    to their standard mapper with a warning. Defaults to
                    "linear" (current behavior).
    * decoder_type: "linear" | "residual_tanh". When "residual_tanh",
                    ``OutputMapper`` is replaced by
                    ``ResidualTanhOutputMapper``. Grouped readout is
                    mutually exclusive (raises ``ValueError``). Defaults
                    to "linear".
    * encoder_hidden_dim / decoder_hidden_dim: hidden width of the
                    ResidualTanhEncoder tanh branch (default 64). Only
                    used when the corresponding type is "residual_tanh".
    * freeze_read: When ``True``, edge currents are computed once from the
                    initial state and held constant across all Heun / DEQ
                    iterations inside every stage. Defaults to ``False``.
    * freeze_boundary: When ``True``, boundary fan-out tanh edge currents
                    are computed once from ``(u, x0)`` and held constant
                    across all Heun / DEQ iterations. The family's resistive
                    shunt (when present) stays dynamic. Independent of
                    ``freeze_read``. No-op when boundary fan-out is not
                    configured. Defaults to ``False``.
    * freeze_temporal_read: When ``True``, temporal-readout tanh edge
                    currents are computed once from ``x0`` and held
                    constant across all Heun / DEQ iterations. The
                    family's resistive shunt (when present) stays
                    dynamic. Independent of ``freeze_read`` and
                    ``freeze_boundary``. No-op when temporal readout is
                    not configured. Defaults to ``False``.
    * boundary_fan_out: dict mapping input indices to lists of target
                    hidden-node indices. When provided, the input signal
                    is treated as fixed-voltage boundary terminals with
                    sparse OTA edges into the dynamic fabric. All dynamic
                    nodes start at zero (no initial-condition write).
    * enable_ref_edges: When ``True``, every ODE node gets one OTA edge to
                    a global per-stage learnable reference voltage ``Vref``
                    (scalar, constrained to ``[0, x_max]``). Implements
                    programmable unary nonlinearities (thresholding,
                    saturation, soft activation, bias injection, etc.)
                    without introducing a heterogeneous cell type. The
                    reference edges use a separate cell library sized to
                    ``num_nodes`` (one OTA per node). Default ``False``.
    * enable_temporal_readout: When ``True``, append ``out_dim`` extra
                    output ODE accumulator nodes to each stage's state
                    (after hidden+proj). All hidden nodes connect all-to-
                    all to each output ODE node via one-way OTA edges
                    (source read-only, destination writable). At readout
                    time the output ODE node voltages are scaled by a
                    learnable affine layer (``OutputAffine``). The
                    ``OutputMapper`` linear projection is bypassed.
                    Mutually exclusive with ``--decoder-type residual_tanh``
                    and ``--grouped-readout``. Requires all stages to have
                    the same width. Default ``False``.
    """
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name!r}. Available: {list(PRESETS)}")
    cfg = dict(PRESETS[preset_name])
    if write_mode is not None:
        cfg["write_mode"] = write_mode
    if read_mode is not None:
        cfg["read_mode"] = read_mode
    if write_idx is not None:
        cfg["write_idx"] = list(write_idx)
    if read_idx is not None:
        cfg["read_idx"] = list(read_idx)
    if encoder_type is not None:
        cfg["encoder_type"] = encoder_type
    if decoder_type is not None:
        cfg["decoder_type"] = decoder_type
    if encoder_hidden_dim is not None:
        cfg["encoder_hidden_dim"] = int(encoder_hidden_dim)
    if decoder_hidden_dim is not None:
        cfg["decoder_hidden_dim"] = int(decoder_hidden_dim)
    if freeze_read:
        cfg["freeze_read"] = True
    if freeze_boundary:
        cfg["freeze_boundary"] = True
    if freeze_temporal_read:
        cfg["freeze_temporal_read"] = True
    if vca_enabled:
        cfg["vca_enabled"] = True
        if vca_rank is not None:
            cfg["vca_rank"] = int(vca_rank)
    return build_net_from_config(
        cfg, cell_lib=cell_lib, enable_drive=enable_drive,
        drive_mode=drive_mode,
        leak_mode=leak_mode, leak_constant=leak_constant,
        read_only_source=read_only_source,
        interstage_activation=interstage_activation,
        interstage_residual_rank=interstage_residual_rank,
        freeze_read=freeze_read,
        freeze_boundary=freeze_boundary,
        freeze_temporal_read=freeze_temporal_read,
        enable_skip_linear=enable_skip_linear,
        boundary_fan_out=boundary_fan_out,
        enable_ref_edges=enable_ref_edges,
        enable_temporal_readout=enable_temporal_readout,
        readout_mode=readout_mode,
        readout_senses_per_node=readout_senses_per_node,
        vca_enabled=vca_enabled,
        vca_rank=vca_rank,
        vca_core_enabled=vca_core_enabled,
        vca_gate_shunt=vca_gate_shunt,
        vca_separate_core_bus=vca_separate_core_bus,
        vca_bias=vca_bias,
        x_max=x_max,
        c_eff=c_eff,
        core_refresh_interval=core_refresh_interval,
    )


def build_net_from_config(
    cfg: dict,
    cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary,
    enable_drive: bool = False,
    drive_mode: str = "fan_out",
    leak_mode: str | None = None,
    leak_constant: float | None = None,
    read_only_source: bool = False,
    interstage_activation: str = "none",
    interstage_residual_rank: int = -1,
    freeze_read: bool = False,
    freeze_boundary: bool = False,
    freeze_temporal_read: bool = False,
    enable_skip_linear: bool = False,
    boundary_fan_out: dict[int, list[int]] | None = None,
    enable_ref_edges: bool = False,
    enable_temporal_readout: bool = False,
    readout_mode: str = "ota_mesh",
    readout_senses_per_node: int = 1,
    vca_enabled: bool = False,
    vca_rank: int | None = None,
    vca_core_enabled: bool = False,
    vca_gate_shunt: bool = False,
    vca_separate_core_bus: bool = False,
    vca_bias: bool | None = None,
    x_max: float | None = None,
    c_eff: float | None = None,
    core_refresh_interval: int = 0,
):
    """Build a KirchhoffNetWithIO from a full config dict.

    ``readout_mode`` (``"ota_mesh"`` default | ``"shared_sense"``) selects
    the readout edge family built when ``enable_temporal_readout=True``:

      - ``"ota_mesh"``: legacy h*d_out temporal-readout OTA mesh — every
        hidden node connects to every output ODE accumulator via its own
        cell + ``output_ode_z_logits`` gate + VCA tap.
      - ``"shared_sense"``: one sense OTA per hidden node (or two, see
        ``readout_senses_per_node``) driven against a per-stage learnable
        Vref rail, plus a plain dense crossbar ``W [d_out x s*h]`` mixing
        the sense currents into the accumulator tail. The crossbar has no
        gates and no VCA; ``vca_v_readout`` gates the sense currents only.
        Requires every stage to have the same width (checked by
        ``KirchhoffNetWithIO``).

    ``readout_senses_per_node`` must be ``1`` or ``2``; ``2`` gives the two
    senses of a node different gm operating points at init (second copy
    jittered ``+0.5`` on ``gm_raw`` per stage).

    ``leak_mode`` and ``leak_constant`` can be specified either explicitly or
    via the ``cfg`` dict (``cfg['leak_mode']`` / ``cfg['leak_constant']``).
    Explicit kwargs take precedence.

    ``drive_mode`` ("fan_out" | "projection") controls per-stage drive
    mapper architecture when ``enable_drive=True``. See build_net_from_preset
    for details.

    ``freeze_read`` can be specified either explicitly or via
    ``cfg['freeze_read']``. Explicit kwargs take precedence.

    ``freeze_boundary`` can be specified either explicitly or via
    ``cfg['freeze_boundary']``. Explicit kwargs take precedence. No-op when
    boundary fan-out is not configured.

    ``freeze_temporal_read`` can be specified either explicitly or via
    ``cfg['freeze_temporal_read']``. Explicit kwargs take precedence.
    No-op when temporal readout is not configured.

    ``vca_enabled`` (``False`` default): enable low-rank input-driven VCA
    (Voltage-Controlled Amplifier) gating on boundary and temporal-readout
    edges. ``vca_rank`` (default ``cfg['vca_rank']`` then
    ``config.VCA['rank']``) controls the projection rank ``r``. The VCA
    gate per gated edge is ``2*sigmoid(b_e + (u^T W) v_e)`` when enabled,
    where ``W`` (in_dim x rank)
    is a shared input projection and ``v_e`` (rank) is a per-edge
    embedding. Requires at least one of ``boundary_fan_out`` or
    ``enable_temporal_readout``; VCA only modulates unfrozen edges that
    have access to the input features.

    ``encoder_type`` / ``decoder_type`` ("linear" | "residual_tanh")
    select non-linear encoder/decoder variants built on top of
    ``io_mapper.ResidualTanhEncoder``. ``encoder_hidden_dim`` and
    ``decoder_hidden_dim`` control the hidden width of the residual
    tanh branch (default 64). Defaults are read from the ``cfg`` dict
    when not supplied via the API.

    ``enable_ref_edges`` (``False`` default): every ODE node gets one
    OTA edge to a global per-stage learnable ``Vref``. Forwarded to
    :func:`topology_to_stage` for each stage.

    ``enable_temporal_readout`` (``False`` default): append ``out_dim``
    extra output ODE accumulator nodes to each stage. Hidden nodes
    connect all-to-all to each output ODE node via one-way OTA edges
    (source read-only, destination writable). Output ODE node voltages
    are read at the end and scaled by a learnable ``OutputAffine``
    layer (gain * V + bias + tanh_gain * tanh(V)), bypassing the linear
    ``OutputMapper`` projection. Requires all stages to have the same
    width. Mutually exclusive with ``decoder_type='residual_tanh'`` and
    ``grouped_readout``.
    """
    if leak_mode is None:
        leak_mode = cfg.get("leak_mode", "programmable")
    if leak_constant is None:
        leak_constant = cfg.get("leak_constant", None)
    freeze_read = bool(cfg.get("freeze_read", freeze_read))
    freeze_boundary = bool(cfg.get("freeze_boundary", freeze_boundary))
    freeze_temporal_read = bool(cfg.get("freeze_temporal_read", freeze_temporal_read))
    if drive_mode not in ("fan_out", "projection"):
        raise ValueError(
            f"drive_mode must be 'fan_out' or 'projection', got {drive_mode!r}"
        )
    # Resolve core_refresh_interval: cfg > explicit kwarg (matches the
    # freeze_read/freeze_boundary/freeze_temporal_read precedence above).
    core_refresh_interval = int(cfg.get("core_refresh_interval", core_refresh_interval))
    if core_refresh_interval < 0:
        raise ValueError(
            f"core_refresh_interval must be >= 0, got {core_refresh_interval}"
        )
    encoder_type = cfg.get("encoder_type", "linear")
    decoder_type = cfg.get("decoder_type", "linear")
    encoder_hidden_dim = int(cfg.get("encoder_hidden_dim", 64))
    decoder_hidden_dim = int(cfg.get("decoder_hidden_dim", 64))
    if encoder_type not in ("linear", "residual_tanh"):
        raise ValueError(
            f"encoder_type must be 'linear' or 'residual_tanh', got {encoder_type!r}"
        )
    if decoder_type not in ("linear", "residual_tanh"):
        raise ValueError(
            f"decoder_type must be 'linear' or 'residual_tanh', got {decoder_type!r}"
        )
    # VCA rank resolution: explicit kwarg > cfg > config default.
    vca_enabled_effective = bool(vca_enabled or cfg.get("vca_enabled", False))
    vca_bias_effective = bool(cfg.get("vca_bias", VCA.get("bias", False))) if vca_bias is None else bool(vca_bias)
    if vca_enabled_effective:
        vca_rank_effective = int(
            vca_rank if vca_rank is not None
            else cfg.get("vca_rank", VCA["rank"])
        )
        if vca_rank_effective < VCA["min_rank"]:
            raise ValueError(
                f"VCA rank must be >= {VCA['min_rank']}, got {vca_rank_effective}"
            )
    else:
        vca_rank_effective = VCA["rank"]
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import (
        InputMapper,
        RobustInputMapper,
        OutputMapper,
        SparseInputMapper,
        ProjectedSparseInputMapper,
        FanOutInputMapper,
        NullInputMapper,
        GroupedOutputMapper,
        ResidualTanhInputMapper,
        ResidualTanhOutputMapper,
    )
    from stage_transfer import StageTransfer

    stages_cfg = cfg["stages"]
    use_robust = cfg.get("use_robust_input", False)
    out_dim = cfg.get("out_dim", 1)
    write_mode = cfg.get("write_mode", "one_to_one")
    read_mode = cfg.get("read_mode", "sparse")
    if write_mode not in ("one_to_one", "dense", "fan_out", "sparse_proj"):
        raise ValueError(
            f"write_mode must be 'one_to_one', 'dense', 'fan_out', or 'sparse_proj', "
            f"got {write_mode!r}"
        )
    if read_mode not in ("sparse", "dense"):
        raise ValueError(
            f"read_mode must be 'sparse' or 'dense', got {read_mode!r}"
        )

    multi = MultiStageTopology.from_config(stages_cfg)
    in_dim = stages_cfg[0]["num_inputs"]
    first_topo = multi.stages[0]
    first_hid = list(first_topo.hidden_node_ids)
    first_proj = list(first_topo.proj_node_ids)
    last_topo = multi.stages[-1]
    last_hid = list(last_topo.hidden_node_ids)
    last_proj = list(last_topo.proj_node_ids)
    n_first_hid = len(first_hid)
    final_state_dim = len(last_hid) + len(last_proj)

    # Boundary-fan-out mode: input is treated as fixed-voltage boundary
    # terminals rather than as an initial-condition write. All dynamic
    # nodes start at zero; the input signal is injected continuously via
    # sparse OTA edges computed in each stage's RHS.
    boundary_src: list[int] | None = None
    boundary_dst: list[int] | None = None
    boundary_cell_lib = None
    enable_boundary = False
    if boundary_fan_out is not None:
        enable_boundary = True
        # Validate keys cover [0, in_dim).
        missing_b = [i for i in range(in_dim) if i not in boundary_fan_out]
        if missing_b:
            raise ValueError(
                f"boundary_fan_out: missing input indices {missing_b}; "
                f"must cover [0, {in_dim})"
            )
        # Validate target indices are in [0, n_first_hid) and unique across inputs.
        all_b_targets: list[int] = []
        for i, targets in boundary_fan_out.items():
            if i < 0 or i >= in_dim:
                raise ValueError(
                    f"boundary_fan_out: input key {i} out of range [0, {in_dim})"
                )
            for t in targets:
                if t < 0 or t >= n_first_hid:
                    raise ValueError(
                        f"boundary_fan_out: input {i} target {t} out of "
                        f"range [0, {n_first_hid})"
                    )
                all_b_targets.append(t)
        if len(all_b_targets) != len(set(all_b_targets)):
            dupes = sorted(
                {t for t in all_b_targets if all_b_targets.count(t) > 1}
            )
            raise ValueError(
                f"boundary_fan_out: duplicate target nodes {dupes}"
            )
        # Build flat lists: for each input i, for each target j, push (i, j).
        boundary_src = []
        boundary_dst = []
        for i in range(in_dim):
            for t in boundary_fan_out[i]:
                boundary_src.append(int(i))
                boundary_dst.append(int(t))
        # Build a fresh cell library instance matching the cell type of
        # ``cell_lib`` but sized for the boundary-edge count. We carry the
        # same cell-type config (bias_enabled, gm bounds, theta, etc.) so
        # the boundary OTA behaves identically to core edges.
        n_boundary = len(boundary_src)
        if isinstance(cell_lib, SimpleEdgeLibrary):
            boundary_cell_lib = SimpleEdgeLibrary(
                num_edges=n_boundary, mode=cell_lib._mode,
            )
        elif isinstance(cell_lib, RealisticTanhLibrary):
            boundary_cell_lib = RealisticTanhLibrary(
                num_edges=n_boundary,
                bias_enabled=cell_lib._bias_enabled,
            )
        elif isinstance(cell_lib, RealisticTanhUpgradeLibrary):
            boundary_cell_lib = RealisticTanhUpgradeLibrary(
                num_edges=n_boundary,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                bias_enabled=cell_lib._bias_enabled,
            )
        elif isinstance(cell_lib, FreeTanhLibrary):
            boundary_cell_lib = FreeTanhLibrary(
                num_edges=n_boundary,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                bias_enabled=cell_lib._bias_enabled,
                parallel_tanh_mult_enabled=cell_lib._parallel_tanh_mult_enabled,
            )
        elif isinstance(cell_lib, AntiParallelFreeTanhLibrary):
            boundary_cell_lib = AntiParallelFreeTanhLibrary(
                num_edges=n_boundary,
                kappa_min=cell_lib.kappa_min,
                kappa_max=cell_lib.kappa_max,
                gm_min=cell_lib.gm_min,
                gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min,
                isat_max=cell_lib.isat_max,
                theta_max=cell_lib.theta_max,
                theta_enabled=cell_lib._theta_enabled,
                use_isat_normalization=cell_lib._use_isat_normalization,
            )
        else:
            raise ValueError(
                f"build_net_from_config: unsupported cell_lib type for "
                f"boundary edges: {type(cell_lib).__name__}"
            )

    # Readout mode validation (shared-sense-crossbar plan).
    # ``readout_mode`` picks the readout edge family built when
    # ``enable_temporal_readout=True``; ``readout_senses_per_node`` only
    # matters for ``shared_sense``.
    if readout_mode not in ("ota_mesh", "shared_sense"):
        raise ValueError(
            f"readout_mode must be 'ota_mesh' or 'shared_sense', got "
            f"{readout_mode!r}"
        )
    if int(readout_senses_per_node) not in (1, 2):
        raise ValueError(
            f"readout_senses_per_node must be 1 or 2, got "
            f"{readout_senses_per_node}"
        )
    enable_temporal_readout_effective = bool(enable_temporal_readout)
    if readout_mode == "shared_sense" and not enable_temporal_readout_effective:
        raise ValueError(
            "build_net_from_config: readout_mode='shared_sense' requires "
            "enable_temporal_readout=True (the sense crossbar writes into "
            "the output-ODE accumulator tail)."
        )
    if readout_mode == "ota_mesh" and int(readout_senses_per_node) != 1:
        # The legacy mesh has no concept of multi-sense per node; reject
        # nonsensical combinations early instead of silently ignoring the
        # senses count (F9).
        raise ValueError(
            "build_net_from_config: readout_mode='ota_mesh' requires "
            "readout_senses_per_node=1; "
            f"got {readout_senses_per_node!r}. Use --readout 'shared-x2' "
            "for the 2-sense shared readout."
        )

    # Local helper: fresh cell library of the same concrete type/config as
    # the core ``cell_lib``, sized ``n`` (identical to the boundary/
    # output_ode clone ladders used elsewhere in this function).
    def _readout_lib_clone(n: int):
        if isinstance(cell_lib, SimpleEdgeLibrary):
            return SimpleEdgeLibrary(num_edges=n, mode=cell_lib._mode)
        if isinstance(cell_lib, RealisticTanhLibrary):
            return RealisticTanhLibrary(
                num_edges=n, bias_enabled=cell_lib._bias_enabled,
            )
        if isinstance(cell_lib, RealisticTanhUpgradeLibrary):
            return RealisticTanhUpgradeLibrary(
                num_edges=n,
                gm_min=cell_lib.gm_min, gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min, isat_max=cell_lib.isat_max,
                bias_enabled=cell_lib._bias_enabled,
            )
        if isinstance(cell_lib, FreeTanhLibrary):
            return FreeTanhLibrary(
                num_edges=n,
                gm_min=cell_lib.gm_min, gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min, isat_max=cell_lib.isat_max,
                bias_enabled=cell_lib._bias_enabled,
                parallel_tanh_mult_enabled=cell_lib._parallel_tanh_mult_enabled,
            )
        if isinstance(cell_lib, AntiParallelFreeTanhLibrary):
            return AntiParallelFreeTanhLibrary(
                num_edges=n,
                kappa_min=cell_lib.kappa_min, kappa_max=cell_lib.kappa_max,
                gm_min=cell_lib.gm_min, gm_max=cell_lib.gm_max,
                isat_min=cell_lib.isat_min, isat_max=cell_lib.isat_max,
                theta_max=cell_lib.theta_max,
                theta_enabled=cell_lib._theta_enabled,
                use_isat_normalization=cell_lib._use_isat_normalization,
            )
        raise ValueError(
            f"build_net_from_config: unsupported cell_lib type for "
            f"readout edges: {type(cell_lib).__name__}"
        )

    # Readout tail mode: append ``out_dim`` extra output ODE accumulator
    # nodes to each stage's state, after the hidden and projection nodes.
    # At readout time the accumulator node voltages are scaled by a
    # learnable ``OutputAffine`` layer, bypassing the linear ``OutputMapper``
    # projection. The accumulator tail is common to both readout families
    # (temporal mesh and shared sense); ``readout_mode`` selects the edge
    # family that charges it.
    output_ode_src: list[int] | None = None
    output_ode_dst: list[int] | None = None
    output_ode_cell_lib = None
    readout_sense_src: list[int] | None = None
    readout_sense_cell_lib = None
    readout_crossbar_shape: tuple[int, int] | None = None
    output_ode_count = 0
    if enable_temporal_readout_effective:
        # Mutual exclusion with residual_tanh decoder: explicit digital
        # nonlinearity over the readout contradicts the analog-readout
        # contract.
        if decoder_type == "residual_tanh":
            raise ValueError(
                "build_net_from_config: enable_temporal_readout=True is "
                "incompatible with decoder_type='residual_tanh' (the "
                "temporal-readout path uses OutputAffine)."
            )
        if cfg.get("grouped_readout") is not None:
            raise ValueError(
                "build_net_from_config: enable_temporal_readout=True is "
                "incompatible with grouped_readout."
            )
        output_ode_count = int(out_dim)
        if readout_mode == "shared_sense":
            s = int(readout_senses_per_node)
            # One sense per hidden node (optionally s=2). Sense k of node h
            # lives at flat index h*s + k, so ``readout_sense_src`` is the
            # hidden node index repeated s times (the two senses of a node
            # start at different operating points via a per-stage gm jitter
            # applied in topology_to_stage).
            readout_sense_src = [
                int(h) for h in range(n_first_hid) for _ in range(s)
            ]
            readout_crossbar_shape = (
                int(out_dim), s * n_first_hid,
            )
            readout_sense_cell_lib = _readout_lib_clone(
                s * n_first_hid
            )
        else:
            # ota_mesh (legacy temporal readout): ``out_dim`` output ODE
            # nodes. Each hidden node connects to each output ODE node via
            # a one-way OTA edge. ``output_ode_dst`` uses compact
            # coordinates [hid_count + proj_count, hid_count + proj_count +
            # out_dim) so the edges reference the accumulator region at the
            # tail of the ODE state vector.
            proj_count_first = len(first_proj)
            core_node_count_first = n_first_hid + proj_count_first
            output_ode_src = []
            output_ode_dst = []
            for h_idx in range(n_first_hid):
                for o_idx in range(output_ode_count):
                    output_ode_src.append(int(h_idx))
                    output_ode_dst.append(core_node_count_first + o_idx)
            # Fresh cell library for the mesh readout edges. Same
            # type/config as the core cell_lib so readout OTAs behave
            # identically to core edges.
            output_ode_cell_lib = _readout_lib_clone(len(output_ode_src))

    # Resolve write_idx and input mapper.
    fan_out_map = None
    if enable_boundary:
        # Boundary mode: input_mapper returns zeros; boundary edges inject
        # the signal. write_idx remains None (no persistent drive needed).
        input_mapper = NullInputMapper(out_dim=n_first_hid)
        write_idx_arg = None
    elif write_mode == "one_to_one":
        preset_write_idx = cfg.get("write_idx")
        if preset_write_idx is None:
            preset_write_idx = list(range(min(in_dim, n_first_hid)))
        if len(preset_write_idx) != in_dim:
            raise ValueError(
                f"write_idx length {len(preset_write_idx)} must equal in_dim {in_dim} "
                f"for one_to_one mode"
            )
        input_mapper = SparseInputMapper(
            in_dim=in_dim, out_dim=n_first_hid, write_idx=preset_write_idx,
            x_max=x_max,
        )
        write_idx_arg = list(preset_write_idx)
    elif write_mode == "fan_out":
        fan_out_map = cfg.get("write_fan_out")
        if fan_out_map is None:
            raise ValueError(
                "write_mode='fan_out' requires 'write_fan_out' in config: "
                "dict mapping input index to list of target hidden-node indices"
            )
        input_mapper = FanOutInputMapper(
            in_dim=in_dim, out_dim=n_first_hid, fan_out_map=fan_out_map,
            x_max=x_max,
        )
        write_idx_arg = sorted(
            {t for targets in fan_out_map.values() for t in targets}
        )
    elif write_mode == "sparse_proj":
        preset_write_idx = cfg.get("write_idx")
        if preset_write_idx is None:
            raise ValueError(
                "write_mode='sparse_proj' requires 'write_idx' in config: "
                "list of hidden-node indices (length >= in_dim)"
            )
        if len(preset_write_idx) < in_dim:
            raise ValueError(
                f"write_mode='sparse_proj' requires len(write_idx) >= in_dim; "
                f"got len(write_idx)={len(preset_write_idx)}, in_dim={in_dim}"
            )
        input_mapper = ProjectedSparseInputMapper(
            in_dim=in_dim, out_dim=n_first_hid, write_idx=preset_write_idx,
            x_max=x_max,
        )
        write_idx_arg = list(preset_write_idx)
    else:
        MapperCls = RobustInputMapper if use_robust else InputMapper
        if encoder_type == "residual_tanh":
            input_mapper = ResidualTanhInputMapper(
                in_dim=in_dim,
                hidden_dim=encoder_hidden_dim,
                out_dim=n_first_hid,
                x_max=x_max,
            )
        else:
            input_mapper = MapperCls(in_dim=in_dim, out_dim=n_first_hid,
                                     x_max=x_max)
        write_idx_arg = None
        if enable_drive:
            raise ValueError(
                "enable_drive=True requires write_mode='fan_out', "
                "'sparse_proj', or 'one_to_one' "
                f"(got write_mode={write_mode!r})"
            )

    # Build stages with optional write_idx for persistent drive.
    stage_modules = []
    transfers = []
    stage_times = []
    stage_steps = []
    first_id_map: dict[int, int] = {}

    for stage_idx, topo in enumerate(multi.stages):
        stage, active_nodes, id_map = topology_to_stage(
            topo, cell_lib=cell_lib, write_idx=write_idx_arg if enable_drive else None,
            leak_mode=leak_mode, leak_constant=leak_constant,
            read_only_source=read_only_source,
            freeze_read=freeze_read,
            freeze_boundary=freeze_boundary,
            freeze_temporal_read=freeze_temporal_read,
            boundary_src=boundary_src,
            boundary_dst=boundary_dst,
            boundary_cell_lib=boundary_cell_lib,
            enable_ref_edges=enable_ref_edges,
            output_ode_src=output_ode_src,
            output_ode_dst=output_ode_dst,
            output_ode_cell_lib=output_ode_cell_lib,
            readout_mode=readout_mode,
            readout_senses_per_node=int(readout_senses_per_node),
            readout_sense_src=readout_sense_src,
            readout_sense_cell_lib=readout_sense_cell_lib,
            readout_crossbar_shape=readout_crossbar_shape,
            output_ode_node_count=output_ode_count,
            vca_enabled=vca_enabled_effective,
            vca_rank=vca_rank_effective,
            vca_in_dim=in_dim,
            vca_core_enabled=vca_core_enabled,
            vca_gate_shunt=vca_gate_shunt,
            vca_separate_core_bus=vca_separate_core_bus,
            vca_bias=vca_bias_effective,
            x_max=x_max,
            c_eff=c_eff,
            core_refresh_interval=core_refresh_interval,
        )
        stage_modules.append(stage)
        stage_times.append(float(stages_cfg[stage_idx].get("t_span", 0.5)))
        stage_steps.append(int(stages_cfg[stage_idx].get("num_steps", 20)))
        if stage_idx == 0:
            first_id_map = id_map

        if stage_idx < len(multi) - 1:
            next_topo = multi.stages[stage_idx + 1]
            next_active = sorted(
                set(next_topo.hidden_node_ids + next_topo.proj_node_ids)
            )
            transfer_drive_mask = (
                list(write_idx_arg) if (enable_drive and write_idx_arg is not None) else None
            )
            # When temporal readout is active, each stage's state includes
            # output_ode_count extra output accumulator nodes appended after
            # the core nodes. StageTransfer must be sized for the full state
            # width so the output ODE slice is preserved (not truncated) when
            # the state passes between stages.
            transfer_in_nodes = len(active_nodes)
            transfer_out_nodes = len(next_active)
            if enable_temporal_readout_effective:
                transfer_in_nodes += output_ode_count
                transfer_out_nodes += output_ode_count
            transfers.append(StageTransfer(transfer_in_nodes, transfer_out_nodes,
                                           activation=interstage_activation,
                                           drive_mask=transfer_drive_mask,
                                           residual_rank=interstage_residual_rank))

    core = KirchhoffNet(
        stages=stage_modules,
        transfers=transfers,
        stage_times=stage_times,
        stage_steps=stage_steps,
    )

    # Build per-stage drive mappers when persistent drive is enabled.
    drive_mappers_list = None
    if enable_drive:
        if fan_out_map is not None:
            # write_mode == 'fan_out': use FanOutInputMapper directly.
            drive_mappers_list = [
                FanOutInputMapper(
                    in_dim=in_dim,
                    out_dim=len(first_hid),
                    fan_out_map=fan_out_map,
                    x_max=x_max,
                )
                for _ in range(len(stages_cfg))
            ]
        elif write_mode == "sparse_proj" and write_idx_arg is not None:
            # write_mode == 'sparse_proj': drive targets come from write_idx.
            # Default ('fan_out' drive_mode): round-robin assign write_idx
            # entries to inputs to form a fan_out_map. 'projection' mode:
            # use ProjectedSparseInputMapper with learned nn.Linear.
            if drive_mode == "projection":
                drive_mappers_list = [
                    ProjectedSparseInputMapper(
                        in_dim=in_dim,
                        out_dim=len(first_hid),
                        write_idx=list(write_idx_arg),
                        x_max=x_max,
                    )
                    for _ in range(len(stages_cfg))
                ]
            else:
                drive_fan_out = {
                    i: list(write_idx_arg[i::in_dim])
                    for i in range(in_dim)
                }
                drive_mappers_list = [
                    FanOutInputMapper(
                        in_dim=in_dim,
                        out_dim=len(first_hid),
                        fan_out_map=drive_fan_out,
                        x_max=x_max,
                    )
                    for _ in range(len(stages_cfg))
                ]
        elif write_mode == "one_to_one" and write_idx_arg is not None:
            # write_mode == 'one_to_one': drive targets come from write_idx.
            # Both modes work here: 'projection' uses ProjectedSparseInputMapper
            # (matches input mapper architecture), 'fan_out' uses 1-to-1
            # FanOutInputMapper (one input per driven node).
            if drive_mode == "projection":
                drive_mappers_list = [
                    ProjectedSparseInputMapper(
                        in_dim=in_dim,
                        out_dim=len(first_hid),
                        write_idx=list(write_idx_arg),
                        x_max=x_max,
                    )
                    for _ in range(len(stages_cfg))
                ]
            else:
                # 1-to-1: input i -> [write_idx_arg[i]]
                drive_fan_out = {i: [write_idx_arg[i]] for i in range(in_dim)}
                drive_mappers_list = [
                    FanOutInputMapper(
                        in_dim=in_dim,
                        out_dim=len(first_hid),
                        fan_out_map=drive_fan_out,
                        x_max=x_max,
                    )
                    for _ in range(len(stages_cfg))
                ]
        # else (write_mode == 'dense'): drive_mappers_list stays None.
        # The dense-mode ValueError above already rejected this case, so
        # reaching here means enable_drive=True without a write_idx — which
        # only happens for dense mode. Leave drive disabled (no mappers).

    grouped_cfg = cfg.get("grouped_readout")
    if enable_temporal_readout_effective:
        # Temporal readout overrides the standard OutputMapper / sparse
        # read / dense read / grouped read selections. OutputAffine
        # learns gain + bias for the output ODE node voltages (the
        # output ODE slice is read directly by KirchhoffNetWithIO).
        from io_mapper import OutputAffine
        output_mapper = OutputAffine(out_dim=out_dim)
        # read_idx_arg is unused; KirchhoffNetWithIO targets the output
        # ODE slice via its own output_ode_count bookkeeping.
        read_idx_arg = None
    elif grouped_cfg is not None:
        nodes_per_target = int(grouped_cfg.get("nodes_per_target", 0))
        readout_offset = int(grouped_cfg.get("offset", 0))
        if nodes_per_target <= 0:
            raise ValueError(
                f"grouped_readout: nodes_per_target must be > 0, got {nodes_per_target}"
            )
        required = readout_offset + out_dim * nodes_per_target
        if final_state_dim < required:
            raise ValueError(
                f"grouped_readout needs final_state_dim >= {required} "
                f"(offset={readout_offset}, out_dim={out_dim}, "
                f"nodes_per_target={nodes_per_target}); got final_state_dim={final_state_dim}. "
                f"Increase --num-hidden/--grid-size."
            )
        if decoder_type == "residual_tanh":
            raise ValueError(
                f"decoder_type='residual_tanh' is incompatible with "
                f"grouped_readout (use the standard linear GroupedOutputMapper "
                f"or disable grouped readout)."
            )
        output_mapper = GroupedOutputMapper(
            nodes_per_target=nodes_per_target,
            num_targets=out_dim,
            node_dim=final_state_dim,
            offset=readout_offset,
        )
        # Pass full state through KirchhoffNetWithIO; the mapper does its
        # own contiguous windowing.
        read_idx_arg = list(range(final_state_dim))
    elif read_mode == "sparse":
        preset_read_idx = cfg.get("read_idx")
        if preset_read_idx is None:
            if len(last_hid) > 0:
                preset_read_idx = [len(last_hid) - 1]
            else:
                preset_read_idx = [0]
        if any(i < 0 or i >= final_state_dim for i in preset_read_idx):
            raise ValueError(
                f"read_idx entries must be in [0, {final_state_dim}), "
                f"got {preset_read_idx}"
            )
        read_idx_arg = list(preset_read_idx)
        if decoder_type == "residual_tanh":
            output_mapper = ResidualTanhOutputMapper(
                in_dim=final_state_dim,
                hidden_dim=decoder_hidden_dim,
                out_dim=out_dim,
                read_idx=preset_read_idx,
            )
        else:
            output_mapper = OutputMapper(
                node_dim=final_state_dim, out_dim=out_dim, read_idx=preset_read_idx
            )
    else:
        read_idx_arg = None
        read_dim = len(last_proj) if len(last_proj) > 0 else len(last_hid)
        if decoder_type == "residual_tanh":
            output_mapper = ResidualTanhOutputMapper(
                in_dim=read_dim,
                hidden_dim=decoder_hidden_dim,
                out_dim=out_dim,
            )
        else:
            output_mapper = OutputMapper(node_dim=read_dim, out_dim=out_dim)

    net = KirchhoffNetWithIO(
        input_mapper,
        core,
        output_mapper,
        hid_count=n_first_hid,
        proj_count=len(first_proj),
        final_hid_count=len(last_hid),
        final_proj_count=len(last_proj),
        write_idx=write_idx_arg,
        read_idx=read_idx_arg,
        enable_drive=enable_drive,
        drive_mappers=drive_mappers_list,
        enable_skip_linear=enable_skip_linear,
        skip_linear_in_dim=in_dim if enable_skip_linear else None,
        skip_linear_out_dim=out_dim if enable_skip_linear else None,
        enable_boundary=enable_boundary,
        boundary_fan_out=boundary_fan_out,
        enable_temporal_readout=enable_temporal_readout_effective,
        output_ode_count=output_ode_count,
        enable_vca=vca_enabled_effective,
        vca_rank=vca_rank_effective,
        vca_in_dim=in_dim,
        vca_bias=vca_bias_effective,
    )

    # Hard topology check: write_idx → read_idx must be >1 hop on the core
    # graph of the first and last stages. Skip when all read_idx target
    # projection nodes, since hidden<->proj direct edges are the intended
    # readout path (R1 spec). When read_idx mixes hidden and proj nodes,
    # filter to only the non-proj (hidden) entries — proj nodes are 1-hop
    # from any hidden write target by construction (all_to_all), so they
    # are exempt from the degree check.
    #
    # write_idx_arg and read_idx_arg are in pre-compact (user-facing)
    # coordinates, but the stage's src/dst are in compact (post-remap)
    # coordinates. Remap via first_id_map before calling validate.
    # StageTopologyBuilder offsets hidden node IDs by num_inputs (in_dim),
    # so user-facing hidden index h corresponds to topology node (h + in_dim).
    # Skip when grouped readout is active: read_idx covers all state nodes and
    # the >1-hop write→read constraint cannot be satisfied across the whole
    # state. GroupedOutputMapper handles its own node selection.
    if (
        grouped_cfg is None
        and not enable_temporal_readout_effective
        and write_idx_arg is not None
        and read_idx_arg is not None
    ):
        all_read_are_proj = all(r >= n_first_hid for r in read_idx_arg)
        if not all_read_are_proj:
            hidden_read_idx = [r for r in read_idx_arg if r < n_first_hid]
            if hidden_read_idx:
                first_stage = stage_modules[0]
                compact_write = [
                    first_id_map[w + in_dim] for w in write_idx_arg
                    if (w + in_dim) in first_id_map
                ]
                compact_read = [
                    first_id_map[r + in_dim] for r in hidden_read_idx
                    if (r + in_dim) in first_id_map
                ]
                if compact_write and compact_read:
                    validate_topology_degrees(
                        src=first_stage.src.tolist(),
                        dst=first_stage.dst.tolist(),
                        num_nodes=first_stage.num_nodes,
                        write_idx=compact_write,
                        read_idx=compact_read,
                    )

    return net
