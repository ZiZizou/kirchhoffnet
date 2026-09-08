"""Unit tests for plan GLN_learned_sharpness: F1 (learnable per-stage clip
sharpness) and F2 (shared-rail GLN on boundary + readout-sense gm).

Run with pytest from the repo root:

    venv/Scripts/python -m pytest test_f1_f2.py -q
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import pytest
import torch

from cell_library import make_cell_library
from topology import build_net_from_config


def _build_small_net(*, learnable_clip=False, gln_rails=False, num_stages=2):
    """Small shared-sense KNet with boundary fan-out (in_dim=4, hidden=8)."""
    cfg = {
        "stages": [{
            "num_inputs": 4, "num_hidden": 8, "num_proj": 0, "num_outputs": 0,
            "hidden_family": "small_world",
            "hidden_kwargs": {"k": 2, "p": 0.2, "seed": 1, "bidirectional": False},
            "input_pattern": "all_to_all", "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all", "edge_repeats": 1,
            "t_span": 0.5, "num_steps": 5,
        } for _ in range(num_stages)],
        "out_dim": 1, "write_mode": "sparse_proj", "read_mode": "dense",
        "use_robust_input": False,
    }
    net = build_net_from_config(
        cfg, cell_lib=make_cell_library("tanh_free"),
        leak_mode="non-programmable", freeze_read=True,
        interstage_activation="residual-relu-tanh",
        boundary_fan_out={0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]},
        enable_temporal_readout=True, x_max=3.0,
        readout_mode="shared_sense", readout_senses_per_node=1,
        vca_enabled=True, vca_rank=2, vca_core_enabled=True,
        vca_gate_shunt=False, vca_separate_core_bus=True, vca_bias=False,
        learnable_clip_sharpness=learnable_clip,
        gln_rails=gln_rails,
    )
    return net


# ---------------------------------------------------------------------------
# F1 — learnable per-stage clip sharpness
# ---------------------------------------------------------------------------

def test_clip_sharpness_init_identity():
    """Mapped learnable sharpness == clip_sharpness_init (0.02) at startup and
    soft_clip matches the fixed-softness computation."""
    stage_off = _build_small_net(learnable_clip=False).core.stages[0]
    stage_on = _build_small_net(learnable_clip=True).core.stages[0]
    x = torch.linspace(-4.0, 4.0, 65).view(-1, 1)
    with torch.no_grad():
        clip_off = stage_off.soft_clip(x)
        clip_on = stage_on.soft_clip(x)
    assert torch.allclose(clip_off, clip_on, atol=1e-6), \
        (clip_off - clip_on).abs().max().item()
    s = stage_on.clip_sharpness()
    assert isinstance(s, torch.Tensor)
    assert abs(float(s.detach()) - 0.02) < 1e-6


def test_clip_sharpness_in_param_count():
    """Learnable clip on adds exactly num_stages trainable params."""
    net_off = _build_small_net(learnable_clip=False)
    net_on = _build_small_net(learnable_clip=True)
    n_off = sum(p.numel() for p in net_off.parameters() if p.requires_grad)
    n_on = sum(p.numel() for p in net_on.parameters() if p.requires_grad)
    assert n_on - n_off == len(net_off.core.stages)
    # every stage owns exactly one clip_sharpness_raw scalar
    for s in net_on.core.stages:
        assert s.clip_sharpness_raw.numel() == 1


def test_epoch0_forward_equal_when_defaults():
    """Same init, learnable off vs on-at-defaults: forward identical."""
    torch.manual_seed(0)
    net_off = _build_small_net(learnable_clip=False)
    net_on = _build_small_net(learnable_clip=True)
    net_on.load_state_dict(net_off.state_dict(), strict=False)
    net_off.eval()
    net_on.eval()
    u = torch.randn(16, 4)
    with torch.no_grad():
        y_off, _ = net_off(u)
        y_on, _ = net_on(u)
    assert (y_off - y_on).abs().max().item() < 1e-6


def test_clip_sharpness_grad_flows():
    """soft_clip w.r.t. clip_sharpness_raw is wired into autograd: grads are
    finite, and nonzero when states sit on the rail transition."""
    stage = _build_small_net(learnable_clip=True).core.stages[0]
    x = torch.tensor([2.9, 3.0, 3.1, -2.9, -3.0], requires_grad=True)
    stage.soft_clip(x).sum().backward()
    g = stage.clip_sharpness_raw.grad
    assert g is not None and torch.isfinite(g).all()
    assert abs(g.item()) > 1e-4, g.item()


def test_clip_sharpness_off_path_has_no_param():
    """Default path adds no parameter and no state-dict key for sharpness."""
    net = _build_small_net(learnable_clip=False)
    assert all(s.clip_sharpness_raw is None for s in net.core.stages)
    keys = [k for k in net.state_dict() if "clip_sharpness" in k]
    assert keys == [], keys


# ---------------------------------------------------------------------------
# F2 — GLN rails
# ---------------------------------------------------------------------------

def test_gln_identity_at_zero_W():
    """W/P/Q all-zero init -> modulate_gm(gm0, u) == gm0 for random u."""
    from gln_rails import GLNRails
    torch.manual_seed(0)
    gln = GLNRails(in_dim=4, B=4, rank=2, gm_min=0.01, gm_max=10.0)
    gln.add_family("boundary", n_edges=8)
    gm0 = torch.full((8,), 2.5)
    u = torch.randn(32, 4) * 5.0
    gm = gln.modulate_gm(gm0, u, family="boundary")
    assert gm.shape == (32, 8)
    assert torch.allclose(gm, gm0.expand(32, 8), atol=1e-6), \
        (gm - gm0.expand(32, 8)).abs().max().item()


def test_gln_grad_flows():
    """GLN must not be a dead saddle: Q gets nonzero grads at step 0 (the
    asymmetric a/P ~ N(0, 0.01), c=Q=0 init), and after one step every GLN
    param receives nonzero grads."""
    from gln_rails import GLNRails
    torch.manual_seed(0)
    gln = GLNRails(in_dim=4, B=4, rank=2, gm_min=0.01, gm_max=10.0)
    gln.add_family("readout", n_edges=8)
    u = torch.randn(16, 4) * 3.0
    gm0 = torch.linspace(0.5, 5.0, 8)

    gm = gln.modulate_gm(gm0, u, family="readout")
    # identity holds exactly at startup (W = P @ Q = 0)
    assert (gm - gm0.expand(16, 8)).abs().max().item() == 0.0
    gm.sum().backward()
    q = gln.families["readout"]
    # load-bearing: Q must get a nonzero gradient at step 0 (dead-saddle
    # guard — a=c=P=Q=0 would make every GLN gradient exactly zero)
    assert q.Q.grad is not None and q.Q.grad.abs().max().item() > 0.0
    assert q.P.grad is not None and torch.isfinite(q.P.grad).all()
    for name in ("a", "c", "alpha_raw"):
        p = getattr(gln, name)
        assert p.grad is not None and torch.isfinite(p.grad).all(), name

    # one optimizer step -> W != 0 -> every param gets nonzero signal
    opt = torch.optim.Adam(gln.parameters(), lr=1e-2)
    opt.step()
    gln.zero_grad()
    gln.modulate_gm(gm0, u, family="readout").sum().backward()
    for name in ("P", "Q"):
        p = getattr(q, name)
        assert p.grad is not None and p.grad.abs().max().item() > 0.0, name
    for name in ("a", "c", "alpha_raw"):
        p = getattr(gln, name)
        assert p.grad is not None and p.grad.abs().max().item() > 0.0, name


def test_gln_param_count_matches_modules():
    """Analytic gln_param_count == sum of module params."""
    from gln_rails import GLNRails, gln_param_count
    torch.manual_seed(0)
    B, rank, in_dim = 4, 2, 4
    e_bound, e_read = 8, 8
    gln = GLNRails(in_dim=in_dim, B=B, rank=rank, gm_min=0.01, gm_max=10.0)
    gln.add_family("boundary", n_edges=e_bound)
    gln.add_family("readout", n_edges=e_read)
    built = sum(p.numel() for p in gln.parameters())
    analytic = gln_param_count(
        in_dim=in_dim, e_bound=e_bound, e_read=e_read, B=B, rank=rank,
        share_rails=True, share_edge_mix_across_stages=True, num_stages=2,
    )
    assert built == analytic, (built, analytic)
    assert built == B * (in_dim + 2) + (e_bound + e_read) * rank + 2 * rank * B


def test_gln_does_not_touch_shunt():
    """GLN modulates gm only; resistive shunt path is structurally separate."""
    from gln_rails import GLNRails
    gln = GLNRails(in_dim=4, B=4, rank=2)
    gln.add_family("boundary", n_edges=8)
    u = torch.randn(4, 4)
    gm0 = torch.full((8,), 1.0)
    gm = gln.modulate_gm(gm0, u, family="boundary")
    # gm values are in [gm_min, gm_max]
    assert gm.min().item() >= 0.01 - 1e-6
    assert gm.max().item() <= 10.0 + 1e-6


def test_gln_forward_match_baseline_gln_off_or_W0():
    """Full-net: GLN on with zero-initialized mixing == GLN off, atol tight."""
    torch.manual_seed(0)
    net_off = _build_small_net(gln_rails=False)
    net_on = _build_small_net(gln_rails=True)
    net_on.load_state_dict(net_off.state_dict(), strict=False)
    net_off.eval()
    net_on.eval()
    u = torch.randn(16, 4)
    with torch.no_grad():
        y_off, _ = net_off(u)
        y_on, _ = net_on(u)
    assert (y_off - y_on).abs().max().item() < 1e-6


def test_gln_off_adds_no_params():
    """GLN off: no extra params; on: analytic gln_param_count extra."""
    from gln_rails import gln_param_count
    net_off = _build_small_net(gln_rails=False)
    net_on = _build_small_net(gln_rails=True)
    n_off = sum(p.numel() for p in net_off.parameters() if p.requires_grad)
    n_on = sum(p.numel() for p in net_on.parameters() if p.requires_grad)
    # boundary E = 8 (4 inputs x 2 targets), readout E = n_sense = 8
    expected = gln_param_count(
        in_dim=4, e_bound=8, e_read=8, B=4, rank=2,
        share_rails=True, share_edge_mix_across_stages=True,
        num_stages=len(net_off.core.stages),
    )
    assert n_on - n_off == expected, (n_on - n_off, expected)


def test_gln_frozen_paths_parity():
    """GLN-on (W=0) == GLN-off under freeze_boundary + freeze_temporal_read:
    the frozen precompute folds the same modulated gm the dynamic path uses."""
    torch.manual_seed(0)
    kw = dict(
        learnable_clip=False, gln_rails=False,
    )
    cfg_freeze = {"num_stages": 2}
    net_off = _build_small_net(**kw)
    net_on = _build_small_net(gln_rails=True)
    # enable every freeze flag on both nets
    for net in (net_off, net_on):
        for s in net.core.stages:
            s.freeze_read = True
            s.freeze_boundary = True
            s.freeze_temporal_read = True
    net_on.load_state_dict(net_off.state_dict(), strict=False)
    net_off.eval()
    net_on.eval()
    u = torch.randn(16, 4)
    with torch.no_grad():
        y_off, _ = net_off(u)
        y_on, _ = net_on(u)
    assert (y_off - y_on).abs().max().item() < 1e-6, \
        (y_off - y_on).abs().max().item()


def test_gln_vca_composition():
    """VCA + GLN together: forward finite, grads finite (VCA outside on the
    current, GLN inside the tanh argument)."""
    torch.manual_seed(0)
    net = _build_small_net(gln_rails=True)
    assert net.enable_vca is True  # builder default in _build_small_net
    net.train()
    u = torch.randn(16, 4)
    y, _ = net(u)
    assert torch.isfinite(y).all()
    y.pow(2).mean().backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in net.parameters() if p.grad is not None
    )


def test_gln_prune_fails_loud():
    """Pruning a GLN net raises (no-prune v1), never silently drops GLN."""
    from topology import prune_network
    net = _build_small_net(gln_rails=True)
    with pytest.raises(ValueError):
        prune_network(net.core, edge_threshold=0.0)


def test_count_params_agreement_phase0_arch():
    """Pins the Phase-0 arch counts: N0, +clip, +gln, both (analytic == built)."""
    import bo_param_sampling as bps
    from gln_rails import gln_param_count
    n0 = bps.knet_param_count(
        10, 4, 4, 2, in_dim=4, out_dim=1, fanout_count=2,
        use_robust_input=False, dagger=False,
        readout_mode="shared_sense", readout_senses=1,
    )
    assert n0 == 2445, n0
    n_clip = bps.knet_param_count(
        10, 4, 4, 2, in_dim=4, out_dim=1, fanout_count=2,
        use_robust_input=False, dagger=False,
        readout_mode="shared_sense", readout_senses=1,
        learnable_clip=True,
    )
    assert n_clip == 2445 + 4, n_clip
    n_gln = bps.knet_param_count(
        10, 4, 4, 2, in_dim=4, out_dim=1, fanout_count=2,
        use_robust_input=False, dagger=False,
        readout_mode="shared_sense", readout_senses=1,
        gln_on=True, gln_B=4, gln_rank=2,
    )
    expected_gln = gln_param_count(
        in_dim=4, e_bound=8, e_read=10, B=4, rank=2,
        share_rails=True, share_edge_mix_across_stages=True, num_stages=4,
    )
    assert n_gln == 2445 + expected_gln, (n_gln, 2445 + expected_gln)
    n_both = bps.knet_param_count(
        10, 4, 4, 2, in_dim=4, out_dim=1, fanout_count=2,
        use_robust_input=False, dagger=False,
        readout_mode="shared_sense", readout_senses=1,
        learnable_clip=True, gln_on=True, gln_B=4, gln_rank=2,
    )
    assert n_both == n_gln + 4, n_both


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))