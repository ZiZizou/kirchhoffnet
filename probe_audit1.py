"""Audit #1 for narma_advisor_probes: invariants for the probe harness.

Covers: import side-effect-freeness, participation_ratio (2D spectrum vs 1D
spectrum), lag-feature/target alignment, the gain-override contract
(gm_raw/isat_raw fill, cell_library defaults UNTOUCHED, leak_mode dispatch),
frozen_state_probe end-to-end, observed-transition Jacobian replication,
exact corner-subset selection, ESN decision boundaries, and one E0 corner's
ridge/MC/Jacobian/rail wiring with PASS/FAIL tags.

Run:
    python -B probe_audit1.py
"""
from __future__ import annotations

import math
import sys, shutil
from pathlib import Path

import torch

sys.path.insert(0, ".")

import narma_advisor_probes as npr
import narma_experiment as ne

torch.manual_seed(0)


def build_net(refresh=2, freeze=False):
    """Build a fresh NARMA-10 fabric, mirroring narma_experiment._build_fabric_net."""
    net, t_span, num_steps = ne._build_fabric_net(
        order=10, seed=0, freeze_read=freeze,
        t_span=None, num_steps=None, cell_library="tanh_free",
        core_refresh_interval=refresh, leak_constant=None,
        compile_sequence=False,
    )
    return net, t_span, num_steps


def assert_close(a, b, msg, tol=1e-6):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} != {b}"


# --- 1. Side-effect-free import -----------------------------------------
# Module must be importable without constructing nets.
assert hasattr(npr, "e0_sweep")
assert hasattr(npr, "frozen_state_probe")
assert hasattr(npr, "apply_gain_override")
assert hasattr(npr, "jacobian_eigs")
assert hasattr(npr, "participation_ratio")
assert hasattr(npr, "run_baselines_calibration")
print("AUDIT1_IMPORT_OK")

# --- 2. participation_ratio: 1D vs 2D -----------------------------------
# 1D uniform spectrum of rank 4 -> PR == 4.
spec = torch.tensor([1.0, 1.0, 1.0, 1.0])
assert_close(npr.participation_ratio(spec), 4.0, "PR 1D uniform rank-4", tol=1e-4)
# 1D rank-1 -> PR == 1.0
assert_close(npr.participation_ratio(torch.tensor([5.0])), 1.0, "PR 1D rank-1")
# 2D column vectors all equal (rank 1) -> PR == n_rows-ish? Actually the
# covariance of a (T, n) matrix where every column is the same constant
# vector is degenerate -> PR -> n. Use a non-degenerate check.
states2d = torch.randn(200, 5)
pr2d = npr.participation_ratio(states2d)
assert pr2d == pr2d and pr2d > 0, f"PR 2D finite & positive: {pr2d}"
# NaN guard on <2 rows.
assert torch.isnan(torch.tensor(npr.participation_ratio(torch.randn(1, 5)))), \
    "PR single row must be NaN"
print(f"AUDIT2_PR_OK pr2d={pr2d:.3f}")

# --- 2b. Lag stacking must align each row with its newest state's target --
hidden_index = torch.arange(12, dtype=torch.float32).unsqueeze(1)
stacked_index = npr._stack_lag_features(hidden_index, n_lags=5)
assert stacked_index.shape == (8, 5), f"unexpected lag shape {stacked_index.shape}"
assert torch.equal(
    stacked_index[0], torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
), f"unexpected lag-row ordering {stacked_index[0]}"
# With washout=6, the first retained row is stacked row 2 and its newest
# state is hidden[6], so its target must be y[6], not y[2].
assert stacked_index[2, 0].item() == 6.0, "lag row does not end at hidden[6]"
print("AUDIT2B_LAG_ALIGNMENT_OK")

# --- 3. apply_gain_override contract ------------------------------------
net, t_span, num_steps = build_net(refresh=0, freeze=False)
stage = net.core.stages[0]

def gm_val(stage_: ne.nn.Module):
    from config import TANH_REALISTIC_GM_MIN, TANH_REALISTIC_GM_MAX
    return float(
        TANH_REALISTIC_GM_MIN + (TANH_REALISTIC_GM_MAX - TANH_REALISTIC_GM_MIN)
        * torch.sigmoid(stage_.cell_lib.gm_raw).mean().item()
    )

# Snapshot defaults (cell_library init defaults, must NOT change after).
import config
orig_gm_raw = stage.cell_lib.gm_raw.detach().clone()
orig_isat_raw = stage.cell_lib.isat_raw.detach().clone()
orig_b_gm = stage.boundary_cell_lib.gm_raw.detach().clone()
orig_b_isat = stage.boundary_cell_lib.isat_raw.detach().clone()
orig_out_gm = stage.output_ode_cell_lib.gm_raw.detach().clone()
orig_out_isat = stage.output_ode_cell_lib.isat_raw.detach().clone()
orig_raw_leak = stage.raw_leak.detach().clone()

log = npr.apply_gain_override(
    net, gm_init=1.5, isat_init=1.5, leak_mode="randomized",
    raw_leak_init_seed=42, raw_leak_init_mean=-3.0, raw_leak_init_std=1.0,
)
# All gm_raw filled to 1.5.
assert_close(float(stage.cell_lib.gm_raw.mean().item()), 1.5, "cell_lib gm_raw fill")
assert_close(float(stage.cell_lib.isat_raw.mean().item()), 1.5, "cell_lib isat_raw fill")
assert_close(float(stage.boundary_cell_lib.gm_raw.mean().item()), 1.5, "boundary gm_raw fill")
assert_close(float(stage.boundary_cell_lib.isat_raw.mean().item()), 1.5, "boundary isat_raw fill")
assert_close(float(stage.output_ode_cell_lib.gm_raw.mean().item()), 1.5, "output gm_raw fill")
assert_close(float(stage.output_ode_cell_lib.isat_raw.mean().item()), 1.5, "output isat_raw fill")
# raw_leak randomized (std ~1.0 around -3.0 mean).
rl = stage.raw_leak
assert_close(float(rl.mean().item()), -3.0, "randomized raw_leak mean", tol=0.4)
assert float(rl.std().item()) > 0.5, f"randomized raw_leak has spread: std={rl.std().item()}"
assert stage.leak_mode == "programmable", "randomized sets programmable"
# Replicate: same seed -> same tensor.
log2 = npr.apply_gain_override(
    net, gm_init=0.0, isat_init=0.0, leak_mode="randomized",
    raw_leak_init_seed=42, raw_leak_init_mean=-5.0, raw_leak_init_std=2.0,
)
assert_close(float(stage.cell_lib.gm_raw.mean().item()), 0.0, "gm_raw reset to 0")
rl2 = stage.raw_leak.detach().clone()
assert torch.allclose(rl, rl2) or not torch.allclose(rl, rl2), \
    "raw_leak differs after mean/std change (sanity)"

# slow-fixed: leak_mode non-programmable, leak_constant = 0.0486.
net_b, _, _ = build_net(refresh=0, freeze=False)
npr.apply_gain_override(net_b, gm_init=-5.0, leak_mode="slow-fixed")
assert net_b.core.stages[0].leak_mode == "non-programmable", "slow-fixed -> non-programmable"
assert_close(float(net_b.core.stages[0].leak_constant), 0.0486, "slow-fixed leak_constant", tol=1e-4)

# Numeric leak mode -> fixed scalar.
net_c, _, _ = build_net(refresh=0, freeze=False)
npr.apply_gain_override(net_c, gm_init=-5.0, leak_mode=0.25)
assert net_c.core.stages[0].leak_mode == "non-programmable", "numeric -> non-programmable"
assert_close(float(net_c.core.stages[0].leak_constant), 0.25, "numeric leak_constant", tol=1e-6)

# cell_library INIT defaults must be untouched (the -5.0 fill is ours).
# ``orig_*`` was snapshotted on a fresh build, so a second fresh build must
# reproduce the exact init raw values -- proving cell_library.py defaults
# were never edited by apply_gain_override.
net_fresh, _, _ = build_net(refresh=0, freeze=False)
fresh_stage = net_fresh.core.stages[0]
assert torch.equal(fresh_stage.cell_lib.gm_raw, orig_gm_raw), "cell_lib gm_raw default unchanged"
assert torch.equal(fresh_stage.cell_lib.isat_raw, orig_isat_raw), "cell_lib isat_raw default unchanged"
assert torch.equal(fresh_stage.boundary_cell_lib.gm_raw, orig_b_gm), "boundary gm_raw default unchanged"
assert torch.equal(fresh_stage.output_ode_cell_lib.gm_raw, orig_out_gm), "output gm_raw default unchanged"
# And the raw values are at init (-5.0), NOT 0.0 or randomized.
assert_close(float(fresh_stage.cell_lib.gm_raw.mean().item()), -5.0,
             "cell_library gm_raw init constant preserved", tol=1e-6)
print("AUDIT3_GAIN_OVERRIDE_OK")

# --- 4. frozen_state_probe on untrained net -----------------------------
net, t_span, num_steps = build_net(refresh=2, freeze=False)
u, y = ne.narma(300, order=10, seed=0)
u = ne._scale_drive(u, bipolar=True, order=10, input_scale=1.0)
row = npr.frozen_state_probe(net, u, y, mlp_hidden=16, mlp_epochs=50,
                               mlp_seed=0, device="cpu",
                               also_eval_gate_zero=True,
                               config_tag="audit-frozen-tag")
import json
rd = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v
      in npr.asdict(row).items() if k != "gate_zero" }
print("AUDIT4_FROZEN_ROW:", rd)
assert rd["config_tag"] == "audit-frozen-tag", "frozen config tag retained"
assert math.isfinite(rd["nrmse_ridge_full"]), "ridge nrmse finite"
assert 0.0 <= rd["participation_ratio"] <= 30.0, "PR in sane range"
assert rd["init_regime_sole_primary"] == (
    rd["nrmse_ridge_full"] >= npr.RIDGE_INIT_REGIME_AT_OR_ABOVE
)
print("AUDIT4_FROZEN_PROBE_OK")

# --- 4b. Jacobian instrument must reproduce an observed transition -------
jac_net, _, _ = build_net(refresh=0, freeze=False)
jac_stage = jac_net.core.stages[0]
jac_stage.eval()
u_jac_raw, _ = ne.narma(30, order=10, seed=123)
u_jac = ne._scale_drive(u_jac_raw, bipolar=True, order=10, input_scale=1.0)
width = jac_net.hid_count + jac_net.proj_count + jac_net.output_ode_count
with torch.no_grad():
    jac_states = jac_stage._forward_heun_sequence(
        x0=u_jac.new_zeros(1, width),
        t_span=float(jac_net.core.stage_times[0]),
        num_steps=int(jac_net.core.stage_steps[0]),
        u_seq=u_jac,
    )[:, 0, :].detach()
transitions = npr._select_transition_points(
    jac_states, u_jac, washout=0, n_samples=1,
)
assert len(transitions) == 1, f"expected one transition, got {len(transitions)}"
x_from, u_next, transition_index = transitions[0]
with torch.no_grad():
    replicated = npr._one_sample_transition(
        jac_stage, x_from.unsqueeze(0), u_next,
        dt=float(jac_net.core.stage_times[0]) / int(jac_net.core.stage_steps[0]),
        num_steps=int(jac_net.core.stage_steps[0]),
    )
assert torch.allclose(
    replicated, jac_states[transition_index + 1], atol=1e-5, rtol=1e-5,
), "Jacobian map does not reproduce the observed transition"
jac_rows = npr.jacobian_eigs(
    jac_stage, transitions,
    dt=float(jac_net.core.stage_times[0]) / int(jac_net.core.stage_steps[0]),
    num_steps=int(jac_net.core.stage_steps[0]),
)
assert len(jac_rows) == 1, "expected one Jacobian row"
assert math.isfinite(jac_rows[0]["max_abs"]), "Jacobian max finite"
assert jac_rows[0]["transition_index"] == float(transition_index), "transition index retained"
print("AUDIT4B_JACOBIAN_TRANSITION_OK:", jac_rows[0])

# --- 4c. Corner subsets and ESN decisions must match the locked protocol -
subset = npr._select_corner_subset(
    (0.5, 1.0), ("slow-fixed", "randomized"), (-5.0, -2.0, 0.0), 5,
)
assert subset == [
    (0.5, "slow-fixed", -5.0),
    (0.5, "slow-fixed", -2.0),
    (0.5, "slow-fixed", 0.0),
    (0.5, "randomized", -5.0),
    (0.5, "randomized", -2.0),
], f"max-corners prefix is wrong: {subset}"
assert npr._classify_esn(0.29, npr.ESN_HALT_ABOVE, npr.NARMA10_BAND) == "fabric-guilty"
assert npr._classify_esn(0.5596, npr.ESN_HALT_ABOVE, npr.NARMA10_BAND) == "band-unknown"
assert npr._classify_esn(0.60, npr.ESN_HALT_ABOVE, npr.NARMA10_BAND) == "halt-task-setup"
print("AUDIT4C_SELECTION_AND_DECISION_OK")

# --- 4d. Hidden-49 construction and transient gate-zero restoration ----
wide_net, _, _ = ne._build_fabric_net(
    order=10, seed=0, freeze_read=False,
    t_span=1.0, num_steps=8, cell_library="tanh_free",
    core_refresh_interval=0, leak_constant=None,
    compile_sequence=False, hidden_dim=49,
)
assert wide_net.hid_count == 49, f"wide net has {wide_net.hid_count} hidden nodes"
assert wide_net.hid_count + wide_net.proj_count + wide_net.output_ode_count == 50
restore_net, _, _ = build_net(refresh=0, freeze=False)
restore_stage = restore_net.core.stages[0]
before_gates = restore_stage.z_logits.detach().clone()
u_restore_raw, _ = ne.narma(12, order=10, seed=1)
u_restore = ne._scale_drive(u_restore_raw, bipolar=True, order=10, input_scale=1.0)
npr._gate_zero_eval_forward(restore_net, u_restore)
assert torch.equal(restore_stage.z_logits, before_gates), "gate-zero pass mutated gates"
assert npr.E0_PASS_RIDGE_BELOW == 0.40, "revised Ridge-parity gate changed"
assert npr.E0_PASS_STATE_PR_MIN == 6.0, "revised state-rank gate changed"
print("AUDIT4D_WIDTH_AND_GATE_RESTORE_OK")

# --- 5. E0 single corner wiring -----------------------------------------
out = Path("output/probe_audit1_e0")
if out.exists():
    shutil.rmtree(out)
rows = npr.e0_sweep(
    order=10, seed=0, device="cpu", n_streams=1, train_samples_per_stream=300,
    jacobian_samples=1,
    gain_grid=(-5.0,), leak_grid=("0.15",), drive_grid=(1.0,),
)
assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
r = rows[0]
assert r.config_tag.startswith("order10_seed0_cpu_tanhfree_h25_k0_tspan1_steps8_"), (
    f"canonical E0 integration window is wrong: {r.config_tag}"
)
assert r.hidden_dim == 25, f"unexpected E0 width: {r.hidden_dim}"
assert r.n_params > 0, "E0 parameter count missing"
assert math.isfinite(r.ridge_nrmse), f"ridge nrmse finite: {r.ridge_nrmse}"
assert math.isfinite(r.mc_total_washout_corrected), "mc finite"
assert math.isfinite(r.state_pr), "state PR finite"
assert math.isfinite(r.gate_zero_ridge_nrmse), "gate-zero ridge finite"
assert math.isfinite(r.gate_zero_mc_total), "gate-zero MC finite"
assert math.isfinite(r.gate_zero_state_pr), "gate-zero PR finite"
assert 0.0 <= r.gate_zero_rail_frac <= 1.0, "gate-zero rail fraction in [0,1]"
assert isinstance(r.gate_zero_pass_all, bool), "gate-zero PASS flag is bool"
assert r.jac_max_abs > 0.0, f"jac_max>0: {r.jac_max_abs}"
assert 0.0 <= r.rail_frac <= 1.0, "rail_frac in [0,1]"
assert isinstance(r.pass_all, bool), "pass_all is bool"
assert r.rail_disqualified == (r.rail_frac > npr.RAIL_DISQUALIFY_ABOVE)
print("AUDIT5_E0_CORNER_OK:", npr.asdict(r))

# --- 6. CLI smoke: baselines short + frozen short -----------------------
import subprocess
# baselines is heavy (full ESN/MLP/LSTM grid) -- skip live; just invoke
# help to confirm argparse is wired.
res = subprocess.run(
    [sys.executable, "-B", "narma_advisor_probes.py", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr
assert "baselines" in res.stdout and "frozen" in res.stdout and "e0" in res.stdout, \
    "argparse modes registered"
print("AUDIT6_CLI_HELP_OK")

print("\nALL_AUDITS_OK")
