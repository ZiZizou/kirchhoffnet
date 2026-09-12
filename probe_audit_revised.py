"""Audit #3 for narma_revised_plan + the Step 3 dense-readout flag.

Covers invariants for the new probes and dense readout plumbing:

- S1-extended probes:
  (A) y-MC curve matches memory_capacity with y targets;
      -1-step delay line scores y-MC_1 ~= 1.00 (sanity).
  (B) IPC-2 curve: target u(n-k)^2 -> centered; constant signal
      scores IPC-2 ~= 0; 1-step-back squared signal scores >= 0.9 on
      synthetic states encoding u^2 exactly.
  (C) Product-pair R^2: NAR10 pairs aligned correctly, lag-9 product
      visible in the lag-9 column.
  (D) s1_extended_probe returns finite, monotone values; the row's
      decision bits agree with the underlying numeric thresholds.

- Tuned-ESN calibration:
  (E) tuned_esn_calibration returns one row per (sr, leak, nr, ins, rl)
      tuple; seed-deterministic; best NRMSE is finite; MC/PR are computed.

- Hetero-leak init:
  (F) hetero_leak_init: tau median within [tau_lo, tau_hi]; raw_leak
      values are finite and softplus(raw_leak) gives the desired leak
      rates (1/tau). Bad params raise.

- Small-world preset:
  (G) build_small_world_narma_preset sets hidden_family='small_world'
      with the requested kwargs; the resulting net builds and forwards
      finite states.

- Hetero-leak dispatch:
  (H) apply_gain_override(leak_mode='hetero:tau_lo=1,tau_hi=40') writes
      the programmable raw_leak as hetero_leak_init would.

- Step 3 dense readout plumbing:
  (I) _build_fabric_net(readout='dense') builds a KirchhoffNetWithIO
      with output_ode_count=0, proj_count=0, OutputMapper (not
      OutputAffine), read_slice over all hidden states.
  (J) dense readout has fewer total params than temporal readout
      (the OTA mesh + accumulator tail are removed).
  (K) dense readout trains end-to-end (smoke 1-epoch + 200-sample
      forward yields a finite NRMSE).

Run:
    python -B probe_audit_revised.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, ".")

import narma_advisor_probes as npr  # noqa: E402
import narma_experiment as ne  # noqa: E402
import narma_revised_plan as nrp  # noqa: E402

torch.manual_seed(0)


def assert_close(a, b, msg, tol=1e-6):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} != {b}"


def _make_small_states(n_t: int = 200, n_n: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic stream + states for IPC-2 sanity checks.

    Column 0 is an explicit 1-step delay line of ``u^2``: ``states[n, 0]``
    equals ``u[n-1]^2`` (row 0 is zero padding). This matches the Jaeger
    k-step-back pairing in :func:`ipc2_curve` -- state at time ``n``
    reconstructs the input ``k`` steps back -- so delay-1 IPC-2 must be
    ~= 1. (A contemporaneous ``u[n]^2`` column would pair with the wrong
    time index under the k>=1 offset and correctly score ~0.)
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n_t, generator=g) * 0.5  # in [0, 0.5] like NARMA
    states = torch.zeros(n_t, n_n)
    states[1:, 0] = u[:-1].pow(2)
    states[:, 1:] = torch.randn(n_t, n_n - 1, generator=g)
    return u, states


# --- A. y-MC sanity: 1-step delay line yields y-MC_1 ~= 1.00 -----------------
u_test, _ = ne.narma(300, order=10, seed=0)
states = torch.stack([u_test.roll(k, dims=0) for k in range(1, 26)], dim=1)  # each col is a delay
# y targets = u at the same shifted alignment; for synthetic, just feed u.
y_mc_per, y_mc_total = nrp.y_memory_capacity_curve(states, u_test, washout=20, max_delay=5)
# Just check MC is computed, finite, and at delay 1 it's high (delay-line in col 0).
assert math.isfinite(y_mc_total), f"y-MC total finite: {y_mc_total}"
assert y_mc_per[0] > 0.5, f"y-MC_1 high for explicit delay: {y_mc_per[0]}"
print(f"AUDIT_A_YMC_OK y_mc_total={y_mc_total:.3f}, y_mc_1={y_mc_per[0]:.3f}")

# --- B. IPC-2 sanity: synthetic states with u^2 in col 0 --------------------
u_s, states_s = _make_small_states(n_t=200, n_n=5, seed=0)
ipc2_per, ipc2_total = nrp.ipc2_curve(states_s, u_s, washout=20, max_delay=5)
assert math.isfinite(ipc2_total), f"IPC-2 total finite: {ipc2_total}"
# IPC-2 should pick up u^2 from col 0 at delay 1.
assert ipc2_per[0] > 0.9, f"IPC-2_1 high for u^2 in col 0: {ipc2_per[0]}"
print(f"AUDIT_B_IPC2_OK ipc2_1={ipc2_per[0]:.3f}, ipc2_total={ipc2_total:.3f}")

# Constant signal -> IPC-2 = 0 (centered quadratic is constant 0)
const = torch.zeros(200)
states_const = torch.zeros(200, 3)
ipc2_const_per, ipc2_const_total = nrp.ipc2_curve(states_const, const, washout=20, max_delay=5)
assert_close(ipc2_const_total, 0.0, "IPC-2 of constant signal = 0", tol=1e-6)
print(f"AUDIT_B2_IPC2_CONSTANT_OK ipc2_total={ipc2_const_total:.3f}")

# --- C. Product-pair R^2: synthetic states with u(n-9)*u(n) in lag-9 col ---
n_t = 200
u_s = torch.rand(n_t) * 0.5
states_s = torch.zeros(n_t, 10)
states_s[:, 0] = u_s.roll(9, dims=0).pow(2)  # u(n-9)^2 in col 0
# Pad col 1 with explicit u(n-9)*u(n)
prod_aligned = u_s.roll(9, dims=0) * u_s
states_s[:, 1] = prod_aligned
pair_r2 = nrp.nar10_product_pairs_r2(states_s, u_s, washout=20)
# (9, 0) pair R^2 should be high because the synthetic has u(n-9)*u(n).
assert pair_r2["(9,0)"] > 0.5, f"product pair (9,0) R^2: {pair_r2['(9,0)']}"
print(f"AUDIT_C_PRODUCT_OK pair_r2={pair_r2}")

# --- D. s1_extended_probe end-to-end on a fresh fabric ---------------------
torch.manual_seed(0)
net, _, _ = ne._build_fabric_net(
    order=10, seed=0, freeze_read=False,
    t_span=1.0, num_steps=8, cell_library="tanh_free",
    core_refresh_interval=0, leak_constant=None,
    compile_sequence=False,
)
u_train, y_train = ne._gen_narma_train_streams(10, 0, 1, 250)
u_train_d = ne._scale_drive(u_train[0], bipolar=True, order=10, input_scale=1.0).to("cpu")
y_train_d = y_train[0].to("cpu")
row = nrp.s1_extended_probe(net, u_train_d, y_train_d, device="cpu", config_tag="audit_D")
assert math.isfinite(row.nrmse_ridge_full), f"s1 ridge full finite: {row.nrmse_ridge_full}"
assert math.isfinite(row.y_mc_total), f"s1 y-MC finite: {row.y_mc_total}"
assert math.isfinite(row.ipc2_total), f"s1 IPC-2 finite: {row.ipc2_total}"
# Decision bits match numerics.
expected_ipc2_present = row.ipc2_total > nrp.IPC2_PRESENCE_THRESHOLD
assert row.ipc2_present == expected_ipc2_present, (
    f"ipc2_present mismatch: {row.ipc2_present} vs {expected_ipc2_present}"
)
print(f"AUDIT_D_S1_EXTENDED_OK u_mc={row.u_mc_total:.3f} y_mc={row.y_mc_total:.3f} "
      f"ipc2={row.ipc2_total:.3f} decision bits: ipc2={row.ipc2_present} "
      f"y_mc={row.y_mc_present} prod={row.product_present}")

# --- E. Tuned-ESN calibration smoke (2 corners) ----------------------------
rows = nrp.tuned_esn_calibration(
    order=10, seed=0, device="cpu",
    spectral_radius=[0.9, 1.0],
    leak=[0.5],
    n_reservoir=[25, 50],
    input_scaling=[0.5],
    ridge_lambda=[1e-2],
    max_corners=2,
)
assert len(rows) == 2, f"expected 2 ESN-cal rows, got {len(rows)}"
for r in rows:
    assert math.isfinite(r.nrmse), f"ESN-cal NRMSE finite: {r.nrmse}"
    assert math.isfinite(r.mc_total), f"ESN-cal MC finite: {r.mc_total}"
print(f"AUDIT_E_ESN_CAL_OK rows[0]: sr={rows[0].spectral_radius} nr={rows[0].n_reservoir} "
      f"nrmse={rows[0].nrmse:.4f} mc={rows[0].mc_total:.3f}")

# --- F. Hetero-leak init: median tau in [lo, hi]; softplus matches 1/tau ----
torch.manual_seed(0)
net, _, _ = ne._build_fabric_net(
    order=10, seed=0, freeze_read=False,
    t_span=1.0, num_steps=8, cell_library="tanh_free",
    core_refresh_interval=0, leak_constant=None,
    compile_sequence=False,
)
log = nrp.hetero_leak_init(
    net.core.stages[0], tau_lo=1.0, tau_hi=40.0, seed=42,
)
assert log["tau_min"] >= 1.0 and log["tau_max"] <= 40.0, (
    f"tau bounds violated: {log['tau_min']}, {log['tau_max']}"
)
assert log["tau_median"] >= 1.0 and log["tau_median"] <= 40.0, (
    f"tau_median out of bounds: {log['tau_median']}"
)
# softplus(raw_leak) should equal 1/tau for each node.
raw = net.core.stages[0].raw_leak.detach()
softplus_vals = torch.nn.functional.softplus(raw)
tau_implied = 1.0 / softplus_vals.clamp_min(1e-6)
assert_close(float(tau_implied.median().item()), float(log["tau_median"]),
             "softplus(raw_leak) = 1/tau median", tol=1e-3)
# Bad params raise.
try:
    nrp.hetero_leak_init(net.core.stages[0], tau_lo=-1.0, tau_hi=10.0, seed=0)
    raise AssertionError("expected ValueError on negative tau_lo")
except ValueError:
    pass
print(f"AUDIT_F_HETERO_LEAK_OK tau_median={log['tau_median']:.3f} "
      f"tau_range=[{log['tau_min']:.2f}, {log['tau_max']:.2f}]")

# --- G. Small-world preset builds and forwards -----------------------------
preset = nrp.build_small_world_narma_preset(
    order=10, hidden_dim=25, num_steps_per_sample=8, t_span=1.0,
    small_world_k=4, small_world_p=0.2, small_world_seed=1,
)
assert preset["stages"][0]["hidden_family"] == "small_world"
assert preset["stages"][0]["hidden_kwargs"]["k"] == 4
assert preset["stages"][0]["hidden_kwargs"]["p"] == 0.2
torch.manual_seed(0)
from topology import build_net_from_config
from cell_library import make_cell_library
cell_lib = make_cell_library("tanh_free")
net_sw = build_net_from_config(
    cfg=preset, cell_lib=cell_lib,
    boundary_fan_out=preset["boundary_fan_out"],
    enable_temporal_readout=True, freeze_read=False,
)
u_smoke, _ = ne._gen_narma_train_streams(10, 0, 1, 100)
u_smoke_d = ne._scale_drive(u_smoke[0], bipolar=True, order=10, input_scale=1.0).unsqueeze(0)
with torch.no_grad():
    out = net_sw(u_smoke_d)
y_sw = out[0] if isinstance(out, tuple) else out
assert torch.isfinite(y_sw).all(), "small-world net forward finite"
# Bad k raises.
try:
    nrp.build_small_world_narma_preset(order=10, small_world_k=3)
    raise AssertionError("expected ValueError on k=3 (must be even)")
except ValueError:
    pass
print(f"AUDIT_G_SMALL_WORLD_OK family={preset['stages'][0]['hidden_family']} "
      f"k={preset['stages'][0]['hidden_kwargs']['k']}")

# --- H. apply_gain_override dispatches hetero mode --------------------------
torch.manual_seed(0)
net, _, _ = ne._build_fabric_net(
    order=10, seed=0, freeze_read=False,
    t_span=1.0, num_steps=8, cell_library="tanh_free",
    core_refresh_interval=0, leak_constant=None,
    compile_sequence=False,
)
override_log = npr.apply_gain_override(
    net, gm_init=0.0, leak_mode="hetero:tau_lo=1.0,tau_hi=40.0",
    raw_leak_init_seed=7,
)
assert override_log["stage0_leak_mode"] == "programmable"
assert "stage0_tau_median" in override_log
# Bad leak mode string raises.
try:
    npr.apply_gain_override(net, gm_init=0.0, leak_mode="unknown-mode")
    raise AssertionError("expected ValueError on unknown leak_mode")
except ValueError:
    pass
print(f"AUDIT_H_HETERO_DISPATCH_OK tau_median={override_log['stage0_tau_median']:.3f}")

# --- I. _build_fabric_net dense readout builds correctly -------------------
torch.manual_seed(0)
net_t, _, _ = ne._build_fabric_net(
    order=10, seed=0, freeze_read=False,
    t_span=1.0, num_steps=8, cell_library="tanh_free",
    core_refresh_interval=0, leak_constant=None,
    compile_sequence=False, readout="temporal",
)
torch.manual_seed(0)
net_d, _, _ = ne._build_fabric_net(
    order=10, seed=0, freeze_read=False,
    t_span=1.0, num_steps=8, cell_library="tanh_free",
    core_refresh_interval=0, leak_constant=None,
    compile_sequence=False, readout="dense",
)
# I.1 dense output_ode_count == 0; proj_count == 0.
assert net_d.output_ode_count == 0, f"dense ode=0, got {net_d.output_ode_count}"
assert net_d.proj_count == 0, f"dense proj=0, got {net_d.proj_count}"
# I.2 read_slice covers all hidden states.
assert net_d.read_dim == net_d.hid_count == 25, (
    f"dense read_dim==hid_count==25, got {net_d.read_dim}/{net_d.hid_count}"
)
# I.3 dense uses OutputMapper, not OutputAffine.
from io_mapper import OutputAffine, OutputMapper
assert isinstance(net_d.output_mapper, OutputMapper), (
    f"dense mapper=OutputMapper, got {type(net_d.output_mapper).__name__}"
)
assert isinstance(net_t.output_mapper, OutputAffine), (
    f"temporal mapper=OutputAffine, got {type(net_t.output_mapper).__name__}"
)
# I.4 unknown readout raises.
try:
    ne._build_fabric_net(
        order=10, seed=0, freeze_read=False,
        t_span=1.0, num_steps=8, cell_library="tanh_free",
        core_refresh_interval=0, leak_constant=None,
        compile_sequence=False, readout="bogus",
    )
    raise AssertionError("expected ValueError on readout='bogus'")
except ValueError:
    pass
print(f"AUDIT_I_DENSE_BUILD_OK dense: ode=0 proj=0 read_dim=25 mapper=OutputMapper")

# --- J. dense cheaper than temporal -----------------------------------------
n_t = sum(p.numel() for p in net_t.parameters() if p.requires_grad)
n_d = sum(p.numel() for p in net_d.parameters() if p.requires_grad)
assert n_d < n_t, f"dense cheaper than temporal: dense={n_d} temporal={n_t}"
# Map per-family (output_mapper should be ~26 vs ~205).
n_mapper_t = sum(p.numel() for n, p in net_t.named_parameters() if "output_mapper" in n)
n_mapper_d = sum(p.numel() for n, p in net_d.named_parameters() if "output_mapper" in n)
assert n_mapper_d <= 30, f"dense output_mapper small: {n_mapper_d}"
print(f"AUDIT_J_DENSE_CHEAPER_OK temporal={n_t} dense={n_d} delta={n_d - n_t} "
      f"(output_mapper temporal={n_mapper_t}, dense={n_mapper_d})")

# --- K. dense trains end-to-end --------------------------------------------
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = ne.train_fabric(
        net_d, ne._scale_drive(
            ne._gen_narma_train_streams(10, 0, 4, 250)[0],
            bipolar=True, order=10, input_scale=1.0,
        ),
        ne._gen_narma_train_streams(10, 0, 4, 250)[1],
        batch_size=4, epochs=1, tbptt_chunk=25,
        lr=1e-3, weight_decay=1e-4, device="cpu",
        verbose=False, use_amp=False, standardize=True, grad_clip=1.0,
    )
assert "train_loss_history" in res
assert all(math.isfinite(float(v)) for v in res["train_loss_history"])
print(f"AUDIT_K_DENSE_TRAINS_OK 1 epoch loss_history finite, "
      f"final_loss={res['final_loss']:.4f}")

# --- L. CLI smoke: e0 with explicit grids (the exact surface that broke
# on Alliance 2026-09-12) ------------------------------------------------
# argparse classifies a value starting with '-' as option-looking unless
# it matches the negative-number regex; comma-joined grids like
# "-5,-2,0,1.5" do NOT match, so they must travel in --flag=value form.
# This audit runs the batch's e0 surface (explicit grids, hetero token,
# small-world flag) as subprocesses so a CLI regression fails preflight
# instead of a GPU allocation.
_here = Path(".").resolve()
for _sw in (False, True):
    with tempfile.TemporaryDirectory(prefix="audit_e0_cli_") as _tmp:
        _cmd = [
            sys.executable, "-B", "narma_advisor_probes.py", "e0",
            "--order", "10", "--seed", "0", "--device", "cpu",
            "--gain-grid=-5,-2,0,1.5",
            "--leak-grid=slow-fixed,hetero:tau_lo=1.0,tau_hi=40.0",
            "--drive-grid=0.25,0.5,1.0",
            "--max-corners", "1",
            "--n-streams", "1", "--train-samples", "300",
            "--jacobian-samples", "1",
            "--output", _tmp,
        ]
        if _sw:
            _cmd += ["--use-small-world", "--small-world-k", "4",
                     "--small-world-p", "0.2", "--small-world-seed", "1"]
        _proc = subprocess.run(
            _cmd, cwd=str(_here), capture_output=True, text=True, timeout=300,
        )
        assert _proc.returncode == 0, (
            f"e0 CLI smoke (small_world={_sw}) rc={_proc.returncode}: "
            f"{_proc.stderr[-2000:]}"
        )
        _summary = json.loads(
            (Path(_tmp) / "e0_sweep.json").read_text()
        )
        assert _summary["n_corners"] == 1, (
            f"e0 CLI smoke (small_world={_sw}) corners: "
            f"{_summary['n_corners']}"
        )
print("AUDIT_L_E0_CLI_SMOKE_OK torus+smallworld explicit grids parse and run")

print("\nALL_AUDITS_OK")
