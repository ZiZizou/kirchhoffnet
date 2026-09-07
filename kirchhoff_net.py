"""KirchhoffNet: multi-stage ODE core and KirchhoffNetWithIO pipeline.

KirchhoffNet sequences DifferentialStage modules and StageTransfer modules.
KirchhoffNetWithIO wraps the core with an InputMapper and OutputMapper
that implement the write/evolve/read phases.

Honest I/O (R1): the write/evolve/read pipeline enforces a strict split
between hidden and projection nodes:
  - InputMapper writes ONLY to the hidden-node positions of the
    differential state vector.
  - Projection nodes start at zero (and remain zero until evolution moves
    them).
  - OutputMapper reads ONLY from projection-node positions; if a stage
    has no projection nodes, it falls back to hidden-node positions
    (with a warning) so backward compatibility is preserved.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from config import DRIVE
from differential_stage import DifferentialStage
from stage_transfer import StageTransfer


__all__ = ["KirchhoffNet", "KirchhoffNetWithIO", "format_parameter_breakdown"]


def format_parameter_breakdown(breakdown: dict) -> str:
    """Render a KirchhoffNetWithIO.parameter_breakdown() dict as aligned text.

    Output example (single-stage friedman2):

        input_mapper:           20
        output_mapper:        1169
        drive_mappers:           8
        stage_0:
          cell_lib:            320
          z_logits:             64
          u_logits:             16
          raw_leak:             16
          raw_drive_g:           4
          ── stage total:      420
        ───────────────────────
        total:               1605
    """
    groups = breakdown.get("groups", {})
    per_stage = breakdown.get("per_stage", {})
    total = breakdown.get("total", 0)
    label_w = max((len(k) for k in groups), default=0)
    label_w = max(label_w, 10)
    stage_label_w = max(
        (len(k) for k in per_stage), default=0,
    )
    stage_label_w = max(stage_label_w, 8)
    width = max(label_w + 18, stage_label_w + 18, 30)
    lines: list[str] = []
    for name in ("input_mapper", "output_mapper", "drive_mappers", "skip_linear"):
        if groups.get(name, 0) or name in ("input_mapper", "output_mapper", "drive_mappers"):
            lines.append(f"  {name:<{label_w}}: {groups.get(name, 0):>{width - label_w - 4}}")
    for stage_key in sorted(per_stage.keys()):
        lines.append(f"  {stage_key}:")
        bucket = per_stage[stage_key]
        for sub_name in (
            "cell_lib", "z_logits", "u_logits", "raw_leak", "raw_drive_g",
            "boundary_cell_lib", "boundary_z_logits",
            "raw_vref", "ref_z_logits", "ref_cell_lib",
            "output_ode_cell_lib", "output_ode_z_logits",
            "readout_sense_z_logits", "readout_sense_cell_lib",
            "readout_crossbar", "raw_vref_sense",
            "vca_W", "vca_v", "other",
        ):
            if bucket.get(sub_name, 0):
                lines.append(f"    {sub_name:<{stage_label_w}}: {bucket[sub_name]:>{width - stage_label_w - 6}}")
            stage_total = sum(bucket.values())
            lines.append(f"    {'── stage total:':<{stage_label_w + 2}} {stage_total}")
    lines.append("  " + "─" * (width - 2))
    lines.append(f"  {'total:':<{label_w + 2}} {total}")
    return "\n".join(lines)


class KirchhoffNet(nn.Module):
    """Multi-stage reduced differential KirchhoffNet ODE core.

    Each stage integrates its internal state for stage_times[i] over
    stage_steps[i] Heun steps. Between stages, StageTransfer truncates
    or zero-pads the state vector to match the next stage's width.

    Handles multi-stage edge_mismatch slicing internally: when ``ctx``
    carries an ``edge_mismatch`` tensor sized for the *sum* of all stages,
    ``forward()`` slices it into per-stage chunks so each stage receives
    mismatch matching its own edge count.
    """

    def __init__(
        self,
        stages: list[DifferentialStage],
        transfers: list[StageTransfer],
        stage_times: list[float],
        stage_steps: list[int],
    ) -> None:
        super().__init__()
        if len(stages) != len(stage_times) or len(stages) != len(stage_steps):
            raise ValueError(
                f"stages/stage_times/stage_steps length mismatch: "
                f"{len(stages)}, {len(stage_times)}, {len(stage_steps)}"
            )
        if len(transfers) != max(0, len(stages) - 1):
            raise ValueError(
                f"transfers length {len(transfers)} should be len(stages)-1 = {len(stages)-1}"
            )

        self.stages = nn.ModuleList(stages)
        self.transfers = nn.ModuleList(transfers)
        self.stage_times = [float(t) for t in stage_times]
        self.stage_steps = [int(n) for n in stage_steps]

        offset = 0
        self._edge_offsets = [offset]
        for s in self.stages:
            offset += s.num_edges()
            self._edge_offsets.append(offset)

    def forward(
        self,
        x0: torch.Tensor,
        store_trajectory: bool = False,
        drive_targets: list[torch.Tensor] | None = None,
        drive_scales: list[float] | None = None,
        solver: str = "heun",
        deq_cfg: dict | None = None,
        stage_noise_std: float = 0.0,
        stage_noise_generator: torch.Generator | None = None,
        u: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ):
        x = initial_state if initial_state is not None else x0
        all_trajs = []
        stage_outputs = []
        stage_infos = []
        for i, stage in enumerate(self.stages):
            x_drive_i = None if drive_targets is None else drive_targets[i]
            drive_scale_i = 0.0 if drive_scales is None else drive_scales[i]
            x, traj = stage(
                x0=x,
                t_span=self.stage_times[i],
                num_steps=self.stage_steps[i],
                store_trajectory=store_trajectory,
                x_drive=x_drive_i,
                drive_scale=drive_scale_i,
                solver=solver,
                deq_cfg=deq_cfg,
                u=u,
            )
            stage_outputs.append(x.detach())
            stage_infos.append(dict(getattr(stage, "last_deq_info", {}) or {}))
            if store_trajectory and traj is not None:
                all_trajs.append(traj)
            # kirchhoff-noise: per-stage additive Gaussian noise on the state
            # vector (thermal/IR-drop modeling on the analog voltage rails).
            if stage_noise_std > 0.0:
                if stage_noise_generator is not None:
                    noise = torch.empty_like(x)
                    noise.normal_(mean=0.0, std=stage_noise_std,
                                  generator=stage_noise_generator)
                    x = x + noise
                else:
                    x = x + torch.randn_like(x) * stage_noise_std
            if i < len(self.transfers):
                x = self.transfers[i](x)
        self.last_stage_outputs = stage_outputs
        self.last_stage_infos = stage_infos
        self.last_drive_targets = drive_targets
        self.last_drive_scales = list(drive_scales) if drive_scales is not None else None
        self.last_solver = solver
        return x, all_trajs if store_trajectory else None

    def parameter_breakdown(self) -> dict:
        """Return parameter counts per component for the regularizer / loss."""
        out = {
            "raw_leak_per_stage": [],
            "total_raw_leak": 0,
        }
        for s in self.stages:
            n = int(s.raw_leak.numel()) if hasattr(s, "raw_leak") else 0
            out["raw_leak_per_stage"].append(n)
            out["total_raw_leak"] += n
        return out


class KirchhoffNetWithIO(nn.Module):
    """Write/evolve/read pipeline: input_mapper -> core -> output_mapper.

    The pipeline enforces a hidden/projection split:
      - ``hid_count`` is the number of hidden nodes in the FIRST stage.
      - ``proj_count`` is the number of projection nodes in the FIRST stage.
      - The first-stage differential state vector is the concatenation
        ``[hidden(0..hid_count-1); projection(hid_count..hid_count+proj_count-1)]``.
      - InputMapper writes only to the hidden portion.
      - Projection portion is zero-initialized.
      - ``final_hid_count`` / ``final_proj_count`` describe the LAST stage's
        split; OutputMapper reads from the final projection portion (or
        final hidden portion if ``final_proj_count == 0``, with a warning,
        R1.4).

    For multi-stage networks with intermediate width changes, the read
    happens at the LAST stage's state width (after all StageTransfer
    operations). When all stages have the same width (the recommended
    case for paper v1, R5), first and final counts match.

    For mapper-only ablations, set ``t_span=0`` at the core level; the
    pipeline is unchanged.

    Sparse I/O (sparse-io-mapping spec):
      - ``write_idx``: list of hidden-node indices (length = in_dim).
        Each input feature `u_i` writes only to `h_{write_idx[i]}`.
        Hidden nodes NOT in `write_idx` are zero-initialized at t=0.
        When `None` (default), all hidden positions are written (dense
        mode, original behavior).
      - ``read_idx``: list of full-state indices to read from. When
        `None` (default), the read slice is the final projection
        portion (or final hidden portion, with the R1.4 warning). When
        provided, the OutputMapper must already be configured with the
        same `read_idx` (it will gather from the full state itself) and
        the pipeline hands the entire final state to the mapper.
    """

    def __init__(
        self,
        input_mapper,
        core: KirchhoffNet,
        output_mapper,
        hid_count: int,
        proj_count: int,
        final_hid_count: int | None = None,
        final_proj_count: int | None = None,
        write_idx: list[int] | None = None,
        read_idx: list[int] | None = None,
        enable_drive: bool = False,
        drive_mappers: list | None = None,
        drive_scales: list[float] | None = None,
        enable_skip_linear: bool = False,
        skip_linear_in_dim: int | None = None,
        skip_linear_out_dim: int | None = None,
        enable_boundary: bool = False,
        boundary_fan_out: dict[int, list[int]] | None = None,
        enable_temporal_readout: bool = False,
        output_ode_count: int = 0,
        enable_vca: bool = False,
        vca_rank: int | None = None,
        vca_in_dim: int | None = None,
        vca_bias: bool | None = None,
    ) -> None:
        super().__init__()
        if hid_count < 0 or proj_count < 0:
            raise ValueError(
                f"hid_count/proj_count must be non-negative, got {hid_count}/{proj_count}"
            )
        if hid_count + proj_count == 0:
            raise ValueError("KirchhoffNetWithIO requires hid_count + proj_count > 0")
        self.input_mapper = input_mapper
        self.core = core
        self.output_mapper = output_mapper
        self.hid_count = int(hid_count)
        self.proj_count = int(proj_count)
        self.final_hid_count = int(hid_count if final_hid_count is None else final_hid_count)
        self.final_proj_count = int(proj_count if final_proj_count is None else final_proj_count)
        self.write_idx = list(write_idx) if write_idx is not None else None
        self.read_idx = list(read_idx) if read_idx is not None else None
        self.enable_drive = bool(enable_drive)
        if self.enable_drive:
            if drive_mappers is None:
                raise ValueError("enable_drive=True requires drive_mappers list")
            if len(drive_mappers) != len(core.stages):
                raise ValueError(
                    f"drive_mappers length {len(drive_mappers)} must equal "
                    f"num_stages {len(core.stages)}"
                )
            self.drive_mappers = nn.ModuleList(drive_mappers)
            # Validate all stages have the same width for drive target shape match.
            stage_widths = [s.num_nodes for s in core.stages]
            if len(set(stage_widths)) != 1:
                raise ValueError(
                    f"enable_drive=True requires all stages to have the same width, "
                    f"got {stage_widths}"
                )
            self.drive_scales = [float(s) for s in (drive_scales or DRIVE["drive_scales"])]
            if len(self.drive_scales) != len(core.stages):
                self.drive_scales = self.drive_scales[:len(core.stages)]
                if len(self.drive_scales) < len(core.stages):
                    self.drive_scales = self.drive_scales + [0.0] * (
                        len(core.stages) - len(self.drive_scales)
                    )
        else:
            self.drive_mappers = None
            self.drive_scales = []
        if self.write_idx is not None:
            if any(i < 0 or i >= self.hid_count for i in self.write_idx):
                raise ValueError(
                    f"write_idx entries must be in [0, hid_count)={self.hid_count}, "
                    f"got {self.write_idx}"
                )
            if len(set(self.write_idx)) != len(self.write_idx):
                raise ValueError(
                    f"write_idx entries must be unique, got {self.write_idx}"
                )
        self.enable_temporal_readout = bool(enable_temporal_readout)
        self.output_ode_count = int(output_ode_count)
        if self.enable_temporal_readout:
            if self.output_ode_count <= 0:
                raise ValueError(
                    f"enable_temporal_readout=True requires output_ode_count > 0, "
                    f"got {self.output_ode_count}"
                )
            # Require all stages to have the same width so StageTransfer
            # doesn't need to remap the output ODE accumulator region
            # (which lives outside the topology's compact space).
            stage_widths = [s.num_nodes for s in core.stages]
            if len(set(stage_widths)) != 1:
                raise ValueError(
                    f"enable_temporal_readout=True requires all stages to "
                    f"have the same width, got {stage_widths}"
                )
            if (
                self.final_hid_count + self.final_proj_count + self.output_ode_count
                != stage_widths[0]
            ):
                raise ValueError(
                    f"enable_temporal_readout: final stage width "
                    f"{stage_widths[0]} must equal "
                    f"final_hid_count + final_proj_count + output_ode_count "
                    f"= {self.final_hid_count} + {self.final_proj_count} + "
                    f"{self.output_ode_count}"
                )
            # Read slice targets the output ODE accumulator region at the
            # tail of the ODE state vector.
            self.read_start = self.final_hid_count + self.final_proj_count
            self.read_dim = self.output_ode_count
        elif self.read_idx is not None:
            max_full = self.final_hid_count + self.final_proj_count
            if any(i < 0 or i >= max_full for i in self.read_idx):
                raise ValueError(
                    f"read_idx entries must be in [0, {max_full}), got {self.read_idx}"
                )
            self.read_start = 0
            self.read_dim = max_full
        elif self.final_proj_count == 0:
            warnings.warn(
                "KirchhoffNetWithIO: no projection nodes in final stage; "
                "OutputMapper falls back to reading from final hidden nodes (R1.4).",
                stacklevel=2,
            )
            self.read_start = 0
            self.read_dim = self.final_hid_count
        else:
            self.read_start = self.final_hid_count
            self.read_dim = self.final_proj_count
        self.read_slice = slice(self.read_start, self.read_start + self.read_dim)

        self.skip_linear_enabled = bool(enable_skip_linear)
        self.enable_boundary = bool(enable_boundary)
        self.boundary_fan_out = (
            None if boundary_fan_out is None
            else {int(k): [int(v) for v in tgts] for k, tgts in boundary_fan_out.items()}
        )
        # Post-readout residual-relu-tanh transfer (always-on, dataset-agnostic).
        # Placed after OutputMapper/OutputAffine so the settled temporal
        # accumulator readout gets one final learned nonlinearity before the
        # bounded-output sigmoid. Identity at init (w1=1, w2=w3=vth=0) so
        # epoch-0 behaviour is bit-identical to the pre-transfer baseline.
        # Hard-wired in KirchhoffNetWithIO so every future preset / problem
        # inherits it regardless of config/CLI.
        _out_dim_post = int(self.output_mapper.out_dim) if hasattr(self.output_mapper, "out_dim") else int(self.read_dim)
        self.post_readout_transfer = StageTransfer(
            in_nodes=_out_dim_post,
            out_nodes=_out_dim_post,
            activation="residual-relu-tanh",
        )
        # VCA (low-rank input-driven attention gating) lives on unfrozen
        # edges. Stored at the IO level so the forward pass knows to pass
        # ``u`` to every stage (boundary / temporal-readout edge currents
        # already need ``u``; VCA needs the raw input features for its
        # shared input projection ``W``).
        self.enable_vca = bool(enable_vca)
        self.vca_rank = int(vca_rank) if vca_rank is not None else 0
        self.vca_in_dim = int(vca_in_dim) if vca_in_dim is not None else 0
        # Stored for topology/API compatibility; per-edge VCA biases are
        # owned and applied by DifferentialStage instances.
        self.vca_bias = vca_bias
        if self.enable_boundary and self.boundary_fan_out is not None:
            stage_widths = [s.num_nodes for s in core.stages]
            if len(set(stage_widths)) != 1:
                raise ValueError(
                    f"enable_boundary=True requires all stages to have the same width, "
                    f"got {stage_widths}"
                )
        if self.skip_linear_enabled:
            if skip_linear_in_dim is None or skip_linear_out_dim is None:
                raise ValueError(
                    "enable_skip_linear=True requires skip_linear_in_dim and "
                    "skip_linear_out_dim"
                )
            self.skip_linear_in_dim = int(skip_linear_in_dim)
            self.skip_linear_out_dim = int(skip_linear_out_dim)
            self.skip_linear = nn.Linear(
                self.skip_linear_in_dim, self.skip_linear_out_dim, bias=True,
            )
            nn.init.xavier_uniform_(self.skip_linear.weight)
            with torch.no_grad():
                self.skip_linear.bias.zero_()
        else:
            self.skip_linear = None
            self.skip_linear_in_dim = None
            self.skip_linear_out_dim = None

    def _make_full_drive(self, hidden_drive: torch.Tensor) -> torch.Tensor:
        B = hidden_drive.shape[0]
        parts = [hidden_drive]
        if self.proj_count > 0:
            proj_zeros = hidden_drive.new_zeros(B, self.proj_count)
            parts.append(proj_zeros)
        if self.output_ode_count > 0:
            out_zeros = hidden_drive.new_zeros(B, self.output_ode_count)
            parts.append(out_zeros)
        if len(parts) == 1:
            return hidden_drive
        return torch.cat(parts, dim=1)

    def forward(
        self,
        u: torch.Tensor,
        store_trajectory: bool = False,
        solver: str = "heun",
        deq_cfg: dict | None = None,
        stage_noise_std: float = 0.0,
        stage_noise_generator: torch.Generator | None = None,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ):
        """Run the write/evolve/read pipeline on input ``u``.

        When ``initial_state`` is provided, it overrides the input-driven
        initial ODE state (input_mapper output, or zero-init under
        boundary mode). The shape must be ``[batch, hid_count + proj_count
        + output_ode_count]`` (i.e. the full stage width). This is the
        entry point for sequential/streaming workloads (NARMA, CTLE
        symbol-by-symbol, etc.) where the fabric state must carry across
        calls. See ``narma_experiment.py`` for the canonical usage.

        When ``return_final_state`` is True the final ODE state (after
        integration) is returned as a third tuple element so the caller
        can pass it back on the next call. Trajectory storage is forced
        off when ``return_final_state`` is set (trajectory tensors are
        batch-internal and not designed to be cached across calls).
        """
        if initial_state is not None:
            # Carry-over path: caller owns the initial state. Validate shape
            # matches the full stage width we expect.
            expected_width = self.hid_count + self.proj_count + self.output_ode_count
            if initial_state.size(1) != expected_width:
                raise ValueError(
                    f"initial_state has {initial_state.size(1)} dims; expected "
                    f"hid_count + proj_count + output_ode_count = {expected_width}"
                )
            x0_full = initial_state
        else:
            if self.enable_boundary:
                # Boundary-fan-out mode: all dynamic nodes start at zero.
                # The input signal enters only through boundary-terminal OTA
                # edges, computed per-step in each stage's RHS.
                x0 = u.new_zeros(u.size(0), self.hid_count)
            else:
                x0 = self.input_mapper(u)
            if x0.size(1) != self.hid_count:
                raise ValueError(
                    f"InputMapper output has {x0.size(1)} dims, expected hid_count={self.hid_count}"
                )
            # Append zero padding for projection and (when enabled) output
            # ODE accumulator nodes so the initial ODE state has the full
            # stage width. Output ODE nodes are always zero-initialized;
            # their current comes from the temporal-readout OTA edges only.
            parts = [x0]
            if self.proj_count > 0:
                parts.append(x0.new_zeros(x0.size(0), self.proj_count))
            if self.output_ode_count > 0:
                parts.append(x0.new_zeros(x0.size(0), self.output_ode_count))
            if len(parts) == 1:
                x0_full = x0
            else:
                x0_full = torch.cat(parts, dim=1)

        # Build per-stage drive targets when persistent drive is enabled.
        # Suppressed under boundary mode: drive targets would compete with
        # the boundary-terminal OTA currents and add a second input path.
        if self.enable_drive and not self.enable_boundary and self.drive_mappers is not None:
            drive_targets = []
            for dm in self.drive_mappers:
                hidden_drive = dm(u)
                drive_targets.append(self._make_full_drive(hidden_drive))
        else:
            drive_targets = None

        if all(t == 0.0 for t in self.core.stage_times):
            x_final, trajs = x0_full, None
        else:
            x_final, trajs = self.core(
                x0_full, store_trajectory=store_trajectory and not return_final_state,
                drive_targets=drive_targets,
                drive_scales=self.drive_scales if (self.enable_drive and not self.enable_boundary) else None,
                solver=solver,
                deq_cfg=deq_cfg,
                stage_noise_std=stage_noise_std,
                stage_noise_generator=stage_noise_generator,
                u=u if (self.enable_boundary or self.enable_vca) else None,
            )
        if self.read_idx is not None:
            x_read = x_final
        else:
            x_read = x_final[:, self.read_slice]
            if x_read.size(1) != self.read_dim:
                raise ValueError(
                    f"Final state has {x_final.size(1)} dims; read_slice gives "
                    f"{x_read.size(1)} dims; expected {self.read_dim}"
                )
        y = self.output_mapper(x_read)
        if self.skip_linear_enabled and self.skip_linear is not None:
            if u.size(-1) != self.skip_linear_in_dim:
                raise ValueError(
                    f"SkipLinear: input has {u.size(-1)} features, "
                    f"expected {self.skip_linear_in_dim}"
                )
            y = y + self.skip_linear(u)
        y = self.post_readout_transfer(y)
        if return_final_state:
            return y, None, x_final
        return y, trajs

    def parameter_breakdown(self) -> dict:
        """Return trainable-parameter counts grouped by component.

        Walks ``self.named_parameters()`` and buckets each tensor by its
        fully-qualified name prefix:

          - ``input_mapper.*``
          - ``output_mapper.*``
          - ``post_readout_transfer.*``  (post-OutputMapper residual-relu-tanh)
          - ``drive_mappers.*``
          - ``skip_linear.*``  (skip connection, enabled via enable_skip_linear)
          - ``core.stages.N.z_logits``  (structural)
          - ``core.stages.N.u_logits``  (structural, deprecated)
          - ``core.stages.N.raw_leak``  (dynamical)
          - ``core.stages.N.raw_drive_g``  (dynamical)
          - ``core.stages.N.cell_lib.*``  (per-edge device params)

        Returns a dict with per-group subtotals, per-stage ``stage_N``
        subtotals, and a ``total``. Use :func:`format_parameter_breakdown`
        for a human-readable rendering.
        """
        groups = {
            "input_mapper": 0,
            "output_mapper": 0,
            "post_readout_transfer": 0,
            "drive_mappers": 0,
            "skip_linear": 0,
        }
        per_stage: dict[str, dict[str, int]] = {}
        total = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            n = int(p.numel())
            total += n
            if name.startswith("input_mapper"):
                groups["input_mapper"] += n
            elif name.startswith("post_readout_transfer"):
                groups["post_readout_transfer"] += n
            elif name.startswith("output_mapper"):
                groups["output_mapper"] += n
            elif name.startswith("drive_mappers"):
                groups["drive_mappers"] += n
            elif name.startswith("skip_linear"):
                groups["skip_linear"] += n
            elif name.startswith("core.stages."):
                parts = name.split(".")
                if len(parts) < 4:
                    continue
                stage_idx = parts[2]
                tail = ".".join(parts[3:])
                stage_key = f"stage_{stage_idx}"
                stage_bucket = per_stage.setdefault(stage_key, {
                    "cell_lib": 0,
                    "z_logits": 0,
                    "u_logits": 0,
                    "raw_leak": 0,
                    "raw_drive_g": 0,
                    "boundary_z_logits": 0,
                    "boundary_cell_lib": 0,
                    "raw_vref": 0,
                    "ref_z_logits": 0,
                    "ref_cell_lib": 0,
                    "output_ode_z_logits": 0,
                    "output_ode_cell_lib": 0,
                    "readout_sense_z_logits": 0,
                    "readout_sense_cell_lib": 0,
                    "readout_crossbar": 0,
                    "raw_vref_sense": 0,
                    "vca_W": 0,
                    "vca_v": 0,
                    "other": 0,
                })
                matched = False
                if tail == "z_logits":
                    stage_bucket["z_logits"] += n; matched = True
                elif tail == "u_logits":
                    stage_bucket["u_logits"] += n; matched = True
                elif tail == "raw_leak":
                    stage_bucket["raw_leak"] += n; matched = True
                elif tail == "raw_drive_g":
                    stage_bucket["raw_drive_g"] += n; matched = True
                elif tail == "boundary_z_logits":
                    stage_bucket["boundary_z_logits"] += n; matched = True
                elif tail.startswith("boundary_cell_lib."):
                    stage_bucket["boundary_cell_lib"] += n; matched = True
                elif tail.startswith("cell_lib."):
                    stage_bucket["cell_lib"] += n; matched = True
                elif tail == "raw_vref":
                    stage_bucket["raw_vref"] += n; matched = True
                elif tail == "ref_z_logits":
                    stage_bucket["ref_z_logits"] += n; matched = True
                elif tail.startswith("ref_cell_lib."):
                    stage_bucket["ref_cell_lib"] += n; matched = True
                elif tail == "output_ode_z_logits":
                    stage_bucket["output_ode_z_logits"] += n; matched = True
                elif tail.startswith("output_ode_cell_lib."):
                    stage_bucket["output_ode_cell_lib"] += n; matched = True
                elif tail == "readout_sense_z_logits":
                    stage_bucket["readout_sense_z_logits"] += n; matched = True
                elif tail.startswith("readout_sense_cell_lib."):
                    stage_bucket["readout_sense_cell_lib"] += n; matched = True
                elif tail == "readout_crossbar_W":
                    stage_bucket["readout_crossbar"] += n; matched = True
                elif tail == "raw_vref_sense":
                    stage_bucket["raw_vref_sense"] += n; matched = True
                elif tail == "vca_W" or tail == "vca_W_core":
                    stage_bucket["vca_W"] += n; matched = True
                elif tail.startswith("vca_v_"):
                    stage_bucket["vca_v"] += n; matched = True
                if not matched:
                    stage_bucket["other"] += n
        return {
            "groups": groups,
            "per_stage": per_stage,
            "total": total,
        }

    def extra_repr(self) -> str:
        skip = (
            f", skip_linear={self.skip_linear_in_dim}->{self.skip_linear_out_dim}"
            if self.skip_linear_enabled else ""
        )
        return (
            f"hid_count={self.hid_count}, proj_count={self.proj_count}, "
            f"final_hid_count={self.final_hid_count}, final_proj_count={self.final_proj_count}, "
            f"write_idx={self.write_idx}, read_idx={self.read_idx}{skip}"
        )
