# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.006957, "end_time": "2026-07-01T22:28:11.248628+00:00", "exception": false, "start_time": "2026-07-01T22:28:11.241671+00:00", "status": "completed"}
#  # CTLE Inverse Design: Variant A (FP32/FP16 Baseline)
#
#
#
#  **Goal:** Train a compact 8-layer MLP that learns the inverse design policy from the teacher (RealNVP flow + ZIG critic).
#
#
#
#  **This variant:** FP32 baseline, no QAT. Same architecture as Variant B for comparison.
#
#
#
#  **Key design:**
#
#  - Bounded log-space outputs (sigmoid + scale per param)
#
#  - Regime-aware loss (zero-inflated and ceiling regions handled)
#
#  - Adaptive teacher labeling (tiered candidate budget)
#
#  - Huber imitation + ZIG forward-consistency + physics loss

# %% papermill={"duration": 0.014837, "end_time": "2026-07-01T22:28:11.269131+00:00", "exception": false, "start_time": "2026-07-01T22:28:11.254294+00:00", "status": "completed"}
import os
# os.chdir('/home/annaik/Documents/train_ctle')

TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'
DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'
OUTPUT_DIR = '/home/annaik/Documents/dagger_output'
# BO wiring: keep Kaggle fallback for local smoke tests
if not os.path.isdir(DATA_DIR):
    _kaggle_fallback = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'
    if os.path.isdir(_kaggle_fallback):
        DATA_DIR = _kaggle_fallback


# %% [markdown] papermill={"duration": 0.005545, "end_time": "2026-07-01T22:28:11.279866+00:00", "exception": false, "start_time": "2026-07-01T22:28:11.274321+00:00", "status": "completed"}
#  ## Phase 1: Setup and Imports

# %% papermill={"duration": 7.953187, "end_time": "2026-07-01T22:28:19.238321+00:00", "exception": false, "start_time": "2026-07-01T22:28:11.285134+00:00", "status": "completed"}
import sys
import math
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import logging
import time
from functools import wraps

_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FMT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_DATE_FMT, force=True)
_logger = logging.getLogger("ctle_dagger")

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger or _logger
        self.start = None
        self.elapsed = None
    def __enter__(self):
        self.start = time.time()
        self.logger.info(f"[TIMER] {self.name} started")
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        self.logger.info(f"[TIMER] {self.name} done ({self.elapsed:.1f}s)")
    def __str__(self):
        return f"{self.elapsed:.1f}s" if self.elapsed is not None else "incomplete"

_logger.info(f"Using device: {DEVICE}")


# %% papermill={"duration": 0.01532, "end_time": "2026-07-01T22:28:19.260518+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.245198+00:00", "status": "completed"}
"""
===========================================================
HYPERPARAMETERS — edit all DAgger / training knobs here
===========================================================
"""
DAGGER_ITERATIONS     = 10        # max DAgger loop iterations
EPOCHS_PER_ITER       = 200       # student training epochs per iteration
BATCH_SIZE            = 256      # DataLoader batch size
ERROR_THRESHOLD       = 0.10     # 10% relative error per spec dimension → failure
VALIDATION_SIZE       = 2000     # validation specs sampled per evaluation
BOUNDARY_RATIO        = 0.50     # fraction of val specs from boundary regions
N_CANDIDATES_PER_SPEC = 3000     # teacher candidates per failed spec (interior)
N_CANDIDATES_BOUNDARY = 10000    # teacher candidates for boundary specs
VALIDITY_THRESHOLD    = 0.50     # non-negotiable validity floor for generated designs
DEGRADE_REL_THRESHOLD = 0.20     # count degradation only at 20%+ relative regression
MIN_DEGRADED_DIMS     = 2        # fail only when 2+ dimensions degrade meaningfully
LR_INITIAL            = 1e-3     # AdamW initial learning rate
LR_DECAY_AFTER_ITER   = 3        # iteration index (0-based) after which LR *= 0.5
LR_FLOOR             = 1e-4      # minimum LR during decay (was eta_min in scheduler)
FAILURE_CAP_RATIO     = 0.40     # max new failures / existing dataset size per iter
CONVERGENCE_THRESHOLD = 0.02     # early-stop if failure_rate < 2%
# Best-model selection (mirrors dagger-nuance-distillation-kirchhoffnet.py for apples-to-apples):
# A SINGLE shared eval set (COMMON_EVAL_SPECS) is used for both per-epoch early-stop
# tracking AND per-iteration validation reporting. Every iteration uses the *same* spec
# set so the per-iteration rates are directly comparable.
COMMON_EVAL_SIZE      = 2000      # size of the shared eval set
COMMON_EVAL_SEED      = 24681012  # fixed seed; identical set across iterations
EARLYSTOP_EVAL_EVERY  = 1         # evaluate failure rate every N epochs (1 = every epoch)
EARLYSTOP_LOG_EVERY   = 10        # emit verbose earlystop log every N epochs
EARLYSTOP_SKIP_EPOCHS = 5         # no best-tracking during first N epochs (warm-start guard)
EARLYSTOP_PATIENCE_EPOCHS = EPOCHS_PER_ITER  # full budget; loop never cut short on patience
MIN_FAILURE_IMPROVEMENT = 0.01    # absolute 1% improvement required to accept new best
DIVERGENCE_ABORT          = True  # abort iteration early on large sustained regression
DIVERGENCE_MARGIN         = 0.20  # absolute increase over prev (20 pts)
DIVERGENCE_CONSEC_EVALS   = 5     # consecutive above-margin evals before abort
COMMON_EVAL_SPECS = None          # built once before the DAgger loop
N_EMPIRICAL_SAMPLES   = 20000    # initial empirical distillation samples (was 5000)
N_CANDIDATES_INITIAL  = 5000     # candidates per spec for initial dataset build
WEIGHT_DECAY          = 1e-4     # AdamW weight decay
LOSS_WEIGHT_EMPIRIC   = 1.0      # loss weight for empirical fine-tune phase (Phase 2)
HARD_BUFFER_WEIGHT    = 10.0     # oversampling weight for newly added DAgger samples
BOUNDARY_ABS_TOLERANCES = {
    'height': (5.0, 0.5),
    'width':  (5.0, 2.0),
    'jitter': (90.0, 0.15),
    'power':  (0.003, 0.0003),
}
BOUNDARY_ABS_TOLERANCES_RELAXED = {
    'height': (5.0, 1.0),
    'width':  (5.0, 5.0),
    'jitter': (90.0, 0.20),
    'power':  (0.003, 0.0005),
}

# ── BO overrides for CTLE 4×100 proxy (mlp_bayes_opt.py) ─────────────────────
try:
    import argparse as _argparse
    _bo_parser = _argparse.ArgumentParser(add_help=False)
    _bo_parser.add_argument('--dagger-iterations', type=int, default=None)
    _bo_parser.add_argument('--epochs-per-iter', type=int, default=None)
    _bo_parser.add_argument('--common-eval-size', type=int, default=None)
    _bo_parser.add_argument('--earlystop-eval-every', type=int, default=None)
    _bo_parser.add_argument('--moe-trunk-width', type=int, default=None)
    _bo_parser.add_argument('--moe-trunk-layers', type=int, default=None)
    _bo_parser.add_argument('--moe-num-experts', type=int, default=None)
    _bo_parser.add_argument('--lr', type=float, default=None)
    _bo_parser.add_argument('--weight-decay', type=float, default=None)
    _bo_parser.add_argument('--batch-size', type=int, default=None)
    _bo_parser.add_argument('--output', type=str, default=None)
    _bo_parser.add_argument('--device', type=str, default=None)
    _bo_parser.add_argument('--data-dir', type=str, default=None)
    _bo_parser.add_argument('--teacher-dir', type=str, default=None)
    _bo_parser.add_argument('--seed', type=int, default=None)
    _bo_parser.add_argument('--input-preprocessing', choices=['knet', 'q75'], default=None)
    # Canonical Phase-A flags (plan canonical-ctle-unify).  When --canonical-dataset
    # is passed, the script bypasses DAgger and runs the static-data, Huber-only
    # Phase-A training mode (no relabeling, no surrogate-through terms).
    _bo_parser.add_argument('--canonical-dataset', type=str, default=None,
                            help='Path to canonical Phase-A .npz (specs + flow labels).')
    _bo_parser.add_argument('--prepare-canonical-dataset', type=str, default=None,
                            help='Path to write the canonical Phase-A .npz and exit (no training).')
    _bo_parser.add_argument('--phase-a-epochs', type=int, default=None,
                            help='Phase-A training epochs (overrides --epochs-per-iter when in Phase A).')
    _bo_parser.add_argument('--phase-a-boundary-ratio', type=float, default=0.5,
                            help='Boundary ratio used when preparing the canonical dataset.')
    _bo_parser.add_argument('--phase-a-n-samples', type=int, default=20000,
                            help='Number of specs in the canonical dataset.')
    _bo_parser.add_argument('--phase-a-validity-weight', type=float, default=0.0,
                            help='Weight on the ZIG-validity NLL -log(p_valid) term added to '
                                 'the Huber loss in Phase-A training (default 0.0 = legacy '
                                 'Huber-only). BO drivers pass 0.3.')
    _bo_parser.add_argument('--phase-a-validity-ramp-start', type=int, default=10,
                            help='First epoch (1-based) at which the validity term starts '
                                 'fading in (default 10).')
    _bo_parser.add_argument('--phase-a-validity-ramp-epochs', type=int, default=30,
                            help='Linear ramp length in epochs to full validity weight '
                                 '(default 30, i.e. full weight from epoch 40).')
    # Plan phase-a-mlp-guided-knet: MLP-guided 4-term loss flags. Defaults
    # match the KNet BO script (mlp 1.0 + fwd 1.0 + power 5e-4) so the two
    # controllers consume the same canonical dataset and train the same loss.
    _bo_parser.add_argument('--phase-a-mlp-teacher-ckpt', type=str, default=None,
                            help='Path to the frozen PlainMLP teacher checkpoint '
                                 '(e.g. trial_0019 dagger_student_plain.pt). Required for '
                                 'the MLP-guided loss and for preparing a schema-2 '
                                 'canonical dataset.')
    _bo_parser.add_argument('--phase-a-mlp-weight', type=float, default=1.0,
                            help='Weight on MSE(student_logits, mlp_teacher_logits) '
                                 '(default 1.0; 0.0 = legacy Huber-on-flow path).')
    _bo_parser.add_argument('--phase-a-fwd-weight', type=float, default=1.0,
                            help='Weight on the ZIG forward-consistency MSE '
                                 '(scaled spec space; default 1.0).')
    _bo_parser.add_argument('--phase-a-power-weight', type=float, default=0.0005,
                            help='Weight on the normalised absolute power term '
                                 '4*VDD*I / train_mean_power (default 5e-4; expected '
                                 'in [1e-4, 1e-3] after normalisation).')
    _bo_parser.add_argument('--phase-a-power-norm', type=str, default='train_mean',
                            choices=['train_mean'],
                            help='Power normalisation mode (only train_mean is '
                                 'supported; the constant is stored in the '
                                 'canonical dataset).')
    _bo_parser.add_argument('--phase-a-power-norm-const', type=float, default=None,
                            help='Override the train-mean power constant (defaults '
                                 'to the value stored in the canonical dataset).')
    _bo_args, _ = _bo_parser.parse_known_args()
    if _bo_args.dagger_iterations is not None:
        DAGGER_ITERATIONS = _bo_args.dagger_iterations
    if _bo_args.epochs_per_iter is not None:
        EPOCHS_PER_ITER = _bo_args.epochs_per_iter
        EARLYSTOP_PATIENCE_EPOCHS = EPOCHS_PER_ITER
    if _bo_args.common_eval_size is not None:
        COMMON_EVAL_SIZE = _bo_args.common_eval_size
    if _bo_args.earlystop_eval_every is not None:
        if _bo_args.earlystop_eval_every < 1:
            raise ValueError('--earlystop-eval-every must be >= 1')
        EARLYSTOP_EVAL_EVERY = _bo_args.earlystop_eval_every
    if _bo_args.moe_trunk_width is not None:
        MOE_TRUNK_WIDTH = _bo_args.moe_trunk_width
    if _bo_args.moe_trunk_layers is not None:
        MOE_TRUNK_LAYERS = _bo_args.moe_trunk_layers
    if _bo_args.moe_num_experts is not None:
        MOE_NUM_EXPERTS = _bo_args.moe_num_experts
    if _bo_args.lr is not None:
        LR_INITIAL = _bo_args.lr
    if _bo_args.weight_decay is not None:
        WEIGHT_DECAY = _bo_args.weight_decay
    if _bo_args.batch_size is not None:
        BATCH_SIZE = _bo_args.batch_size
    if _bo_args.output is not None:
        OUTPUT_DIR = _bo_args.output
    if _bo_args.device is not None:
        DEVICE = torch.device(_bo_args.device) if _bo_args.device != 'auto' else DEVICE
    if _bo_args.data_dir is not None:
        DATA_DIR = _bo_args.data_dir
    if _bo_args.teacher_dir is not None:
        TEACHER_DIR = _bo_args.teacher_dir
    if _bo_args.seed is not None:
        np.random.seed(_bo_args.seed)
        torch.manual_seed(_bo_args.seed)
except Exception as _e:
    _logger.warning(f"[BO] override parsing failed: {_e}")


# %% papermill={"duration": 0.086574, "end_time": "2026-07-01T22:28:19.352706+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.266132+00:00", "status": "completed"}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import joblib
from pathlib import Path

LOG_BOUNDS = [-15.0, 8]


# =============================================================================
# ZIG MODEL: HybridHurdleModel (MUST match trained architecture exactly)
# =============================================================================

def make_eye_head():
    """Small nonlinear head for positive magnitude prediction (instruction 6)."""
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.SiLU(),
        nn.LayerNorm(32),
        nn.Dropout(0),
        nn.Linear(32, 1),
    )


class HybridHurdleModel(nn.Module):
    """
    Trained ZIG model: hybrid hurdle architecture.
    If PER_TARGET_HURDLE: three separate validity heads (logit per metric).
    Otherwise: one shared validity head.
    Positive regression heads are always per-metric.
    """

    def __init__(self, dropout=0.10, per_target=False):
        super().__init__()
        self.per_target = per_target

        self.enc = nn.Sequential(
            nn.Linear(7, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
        )

        if per_target:
            self.valid_head_h = make_eye_head()
            self.valid_head_w = make_eye_head()
            self.valid_head_j = make_eye_head()
        else:
            self.valid_head = make_eye_head()

        self.log1p_h = make_eye_head()
        self.log1p_w = make_eye_head()
        self.log1p_j = make_eye_head()

        self.mu_power = nn.Linear(64, 1)
        self.log_sigma_power = nn.Linear(64, 1)

    def forward(self, x):
        h = self.enc(x)

        if self.per_target:
            valid_logit_h = self.valid_head_h(h).squeeze(-1)
            valid_logit_w = self.valid_head_w(h).squeeze(-1)
            valid_logit_j = self.valid_head_j(h).squeeze(-1)
            p_valid_h = torch.sigmoid(valid_logit_h)
            p_valid_w = torch.sigmoid(valid_logit_w)
            p_valid_j = torch.sigmoid(valid_logit_j)
            p_valid = torch.stack([p_valid_h, p_valid_w, p_valid_j], dim=1)
        else:
            valid_logit = self.valid_head(h).squeeze(-1)
            p_valid = torch.sigmoid(valid_logit)

        pred_h_log1p = F.softplus(self.log1p_h(h)).squeeze(-1)
        pred_w_log1p = F.softplus(self.log1p_w(h)).squeeze(-1)
        pred_j_log1p = F.softplus(self.log1p_j(h)).squeeze(-1)

        pred_h_pos = torch.expm1(torch.clamp(pred_h_log1p, max=8.0))
        pred_w_pos = torch.expm1(torch.clamp(pred_w_log1p, max=8.0))
        pred_j_pos = torch.expm1(torch.clamp(pred_j_log1p, max=8.0))

        if self.per_target:
            pred_h_soft = p_valid_h * pred_h_pos
            pred_w_soft = p_valid_w * pred_w_pos
            pred_j_soft = p_valid_j * pred_j_pos
        else:
            pred_h_soft = p_valid * pred_h_pos
            pred_w_soft = p_valid * pred_w_pos
            pred_j_soft = p_valid * pred_j_pos

        mu_power = self.mu_power(h).squeeze(-1)
        sigma_power = F.softplus(self.log_sigma_power(h)).squeeze(-1) + 1e-4

        out = {
            "pred_h_log1p": pred_h_log1p, "pred_w_log1p": pred_w_log1p, "pred_j_log1p": pred_j_log1p,
            "pred_h_pos": pred_h_pos, "pred_w_pos": pred_w_pos, "pred_j_pos": pred_j_pos,
            "pred_h_soft": pred_h_soft, "pred_w_soft": pred_w_soft, "pred_j_soft": pred_j_soft,
            "mu_power": mu_power, "sigma_power": sigma_power,
        }
        if self.per_target:
            out.update({"p_valid_h": p_valid_h, "p_valid_w": p_valid_w, "p_valid_j": p_valid_j,
                         "valid_logit_h": valid_logit_h, "valid_logit_w": valid_logit_w, "valid_logit_j": valid_logit_j})
        else:
            out.update({"p_valid": p_valid, "valid_logit": valid_logit})
        return out


class ZIGHurdleWrapper(nn.Module):
    """
    Frozen HybridHurdleModel wrapper for flow scoring.
    Returns regressor in flow's condition space: [power_s, jitter_s, height_s, width_s]
    Returns validity: product of per-metric validity probabilities.
    """
    def __init__(self, model_path, device, eye_scale_h, eye_scale_w, eye_scale_j,
                 scaler_y_p, per_target=False):
        super().__init__()
        self.device = device
        self.per_target = per_target

        self.model = HybridHurdleModel(dropout=0.0, per_target=per_target).to(device)
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Plain instance attributes (avoid register_buffer name conflicts with nn.Module)
        self._eye_scale_h = float(eye_scale_h)
        self._eye_scale_w = float(eye_scale_w)
        self._eye_scale_j = float(eye_scale_j)
        self._scaler_y_p_mean = float(scaler_y_p.mean_[0])
        self._scaler_y_p_scale = float(scaler_y_p.scale_[0])

    def _scale_eye_torch(self, raw_val, eye_scale):
        log_val = torch.log10(raw_val.clamp(min=1e-30))
        return log_val / eye_scale

    def _scale_power_torch(self, raw_val):
        log_val = torch.log10(raw_val.clamp(min=1e-30))
        return (log_val - self._scaler_y_p_mean) / (self._scaler_y_p_scale + 1e-8)

    def forward(self, x_scaled):
        p = self.model(x_scaled)

        h_soft = p['pred_h_soft']
        w_soft = p['pred_w_soft']
        j_soft = p['pred_j_soft']
        power_mu = p['mu_power']

        h_scaled = self._scale_eye_torch(h_soft, self._eye_scale_h)
        w_scaled = self._scale_eye_torch(w_soft, self._eye_scale_w)
        j_scaled = self._scale_eye_torch(j_soft, self._eye_scale_j)
        power_scaled = power_mu

        regressor = torch.stack([power_scaled, j_scaled, h_scaled, w_scaled], dim=-1)

        if self.per_target:
            validity = (p['p_valid_h'] * p['p_valid_w'] * p['p_valid_j']).clamp(1e-6, 1.0 - 1e-6)
        else:
            validity = p['p_valid'].clamp(1e-6, 1.0 - 1e-6)

        return {"regressor": regressor, "classifier": validity}


# =============================================================================
# SPLINE FLOW ARCHITECTURE
# =============================================================================

class RationalQuadraticSpline(nn.Module):
    """Batched rational-quadratic spline (num_bins=8, tail_bound=3.0)."""
    def __init__(self, num_bins=8, tail_bound=3.0):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound

    def _get_params(self, params):
        B, D, _ = params.shape
        nb = self.num_bins
        widths  = F.softmax(params[..., :nb], dim=-1) * (2 * self.tail_bound)
        heights = F.softmax(params[..., nb:2*nb], dim=-1) * (2 * self.tail_bound)
        derivs  = F.softplus(params[..., 2*nb:]) + 1e-2
        return widths, heights, derivs

    def _searchsorted(self, sorted_seq, values):
        idx = torch.searchsorted(sorted_seq.contiguous(), values.contiguous(), right=False)
        return idx.clamp(1, sorted_seq.shape[-1] - 1)

    def forward(self, x, params, inverse=False):
        B, D = x.shape
        widths, heights, derivs = self._get_params(params)

        x_k = torch.cumsum(widths, dim=-1) - self.tail_bound
        x_k = torch.cat([torch.full((B, D, 1), -self.tail_bound, device=x.device, dtype=x.dtype), x_k], dim=-1)
        y_k = torch.cumsum(heights, dim=-1) - self.tail_bound
        y_k = torch.cat([torch.full((B, D, 1), -self.tail_bound, device=x.device, dtype=x.dtype), y_k], dim=-1)
        delta = heights / (widths + 1e-8)

        if not inverse:
            x_exp = x.unsqueeze(-1)
            idx = self._searchsorted(x_k, x_exp)
            x0 = torch.gather(x_k, -1, idx - 1).squeeze(-1)
            x1 = torch.gather(x_k, -1, idx).squeeze(-1)
            y0 = torch.gather(y_k, -1, idx - 1).squeeze(-1)
            y1 = torch.gather(y_k, -1, idx).squeeze(-1)
            d0 = torch.gather(derivs, -1, idx - 1).squeeze(-1)
            d1 = torch.gather(derivs, -1, idx).squeeze(-1)
            delta = torch.gather(delta, -1, idx - 1).squeeze(-1)
            theta = (x - x0) / (x1 - x0 + 1e-8)
            theta = torch.clamp(theta, 0, 1)
            t1m = 1 - theta
            num = delta * theta * theta + d0 * theta * t1m
            den = delta + (d0 + d1 - 2 * delta) * theta * t1m
            y = y0 + (y1 - y0) * num / (den + 1e-8)
            deriv = (delta ** 2 * (d1 * theta ** 2 + 2 * delta * theta * t1m + d0 * t1m ** 2)) / (den ** 2 + 1e-8)
            logabsdet = torch.log(deriv + 1e-8)
            outside = (x.abs() >= self.tail_bound)
            y = torch.where(outside, x, y)
            logabsdet = torch.where(outside, torch.zeros_like(logabsdet), logabsdet)
            return y, logabsdet
        else:
            y_exp = x.unsqueeze(-1)
            idx = self._searchsorted(y_k, y_exp)
            x0 = torch.gather(x_k, -1, idx - 1).squeeze(-1)
            x1 = torch.gather(x_k, -1, idx).squeeze(-1)
            y0 = torch.gather(y_k, -1, idx - 1).squeeze(-1)
            y1 = torch.gather(y_k, -1, idx).squeeze(-1)
            d0 = torch.gather(derivs, -1, idx - 1).squeeze(-1)
            d1 = torch.gather(derivs, -1, idx).squeeze(-1)
            delta = torch.gather(delta, -1, idx - 1).squeeze(-1)
            s = (x - y0) / (y1 - y0 + 1e-8)
            s = torch.clamp(s, 0, 1)
            a = delta - d0 + s * (d0 + d1 - 2 * delta)
            b = d0 - s * (d0 + d1 - 2 * delta)
            c = -s * delta
            disc = b ** 2 - 4 * a * c
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0) + 1e-8)
            theta = torch.where(b >= 0, (-b - sqrt_disc) / (2 * a + 1e-8), (2 * c) / (-b + sqrt_disc + 1e-8))
            theta2 = c / (a * theta + 1e-8)
            use_alt = (theta < 0) | (theta > 1) | torch.isnan(theta)
            theta = torch.where(use_alt, theta2, theta)
            theta = torch.clamp(theta, 0, 1)
            x_inv = x0 + theta * (x1 - x0)
            t1m = 1 - theta
            den = delta + (d0 + d1 - 2 * delta) * theta * t1m
            deriv = (delta ** 2 * (d1 * theta ** 2 + 2 * delta * theta * t1m + d0 * t1m ** 2)) / (den ** 2 + 1e-8)
            logabsdet = -torch.log(deriv + 1e-8)
            outside = (x.abs() >= self.tail_bound)
            x_inv = torch.where(outside, x, x_inv)
            logabsdet = torch.where(outside, torch.zeros_like(logabsdet), logabsdet)
            return x_inv, logabsdet


class ConditionalSplineCoupling(nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim, num_layers, mask, num_bins=8, tail_bound=3.0):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)
        n_fixed = int(mask.sum().item())
        n_trans = dim - n_fixed
        layers = []
        in_dim = n_fixed + cond_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else n_trans * (3 * num_bins + 1)
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self.spline = RationalQuadraticSpline(num_bins=num_bins, tail_bound=tail_bound)
        self.n_trans = n_trans
        self.num_bins = num_bins

    def forward(self, x, cond, reverse=False):
        mask_bool = self.mask.bool()
        x_fixed = x[:, mask_bool]
        x_trans = x[:, ~mask_bool]
        params = self.net(torch.cat([x_fixed, cond], dim=-1))
        params = params.view(-1, self.n_trans, 3 * self.num_bins + 1)
        if not reverse:
            y_trans, logdet = self.spline(x_trans, params, inverse=False)
        else:
            y_trans, logdet = self.spline(x_trans, params, inverse=True)
        y = x.clone()
        y[:, ~mask_bool] = y_trans
        return y, logdet.sum(dim=-1)


class LearnedInvertibleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))

    def forward(self, x, cond=None, reverse=False):
        if not reverse:
            y = x @ self.W.t()
            logdet = torch.slogdet(self.W)[1].expand(x.size(0))
            return y, logdet
        else:
            W_inv = torch.inverse(self.W)
            y = x @ W_inv.t()
            logdet = -torch.slogdet(self.W)[1].expand(x.size(0))
            return y, logdet


class ActNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def forward(self, x, cond=None, reverse=False):
        if not self.initialized and self.training:
            with torch.no_grad():
                mean = x.mean(dim=0)
                std = x.std(dim=0) + 1e-6
                self.bias.data = -mean
                self.log_scale.data = -torch.log(std)
                self.initialized = True
        if not reverse:
            y = (x + self.bias) * torch.exp(self.log_scale)
            return y, self.log_scale.sum()
        else:
            y = x * torch.exp(-self.log_scale) - self.bias
            return y, -self.log_scale.sum()


class Permute(nn.Module):
    def __init__(self, dim):
        super().__init__()
        perm = torch.arange(dim - 1, -1, -1)
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', torch.argsort(perm))

    def forward(self, x, cond=None, reverse=False):
        if not reverse:
            return x[:, self.perm], 0.0
        else:
            return x[:, self.inv_perm], 0.0


class ConditionalSplineFlow(nn.Module):
    """
    Conditional Spline Flow matching trained ctle_conditional_flow.pt.
    Uses RationalQuadraticSpline coupling + learned invertible linear + actnorm.
    Has conditional base distribution (not fixed N(0,I)).
    """
    def __init__(self, dim=7, cond_dim=4, num_coupling=10, hidden_dim=256,
                 num_layers=3, num_bins=8, tail_bound=3.0):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.base_loc_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, dim)
        )
        self.base_scale_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, dim)
        )
        self.layers = nn.ModuleList()
        for i in range(num_coupling):
            mask = (torch.cat([torch.ones(3), torch.zeros(4)]) if i % 2 == 0
                    else torch.cat([torch.zeros(4), torch.ones(3)]))
            self.layers.append(
                ConditionalSplineCoupling(dim, hidden_dim, hidden_dim, num_layers, mask,
                                         num_bins=num_bins, tail_bound=tail_bound)
            )
            self.layers.append(ActNorm(dim))
            self.layers.append(LearnedInvertibleLinear(dim))

    def get_base_dist(self, cond):
        c = self.cond_embed(cond)
        loc = self.base_loc_net(c)
        scale = F.softplus(self.base_scale_net(c)) + 1e-3
        return dist.Independent(dist.Normal(loc, scale), 1)

    def forward(self, x, cond):
        c = self.cond_embed(cond)
        z = x
        logdet = 0.0
        for layer in self.layers:
            z, ld = layer(z, c)
            logdet = logdet + ld
        base = self.get_base_dist(cond)
        log_prob = base.log_prob(z) + logdet
        return z, log_prob

    def inverse(self, z, cond):
        c = self.cond_embed(cond)
        x = z
        for layer in reversed(self.layers):
            x, _ = layer(x, c, reverse=True)
        return x

    def sample(self, cond, num_samples=1, temperature=1.0):
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_rep = cond.repeat_interleave(num_samples, dim=0)
        c = self.cond_embed(cond_rep)
        loc = self.base_loc_net(c)
        scale = F.softplus(self.base_scale_net(c)) + 1e-3
        eps = torch.randn_like(loc)
        z = loc + temperature * eps * scale
        return self.inverse(z, cond_rep)


class FlowInference:
    """
    End-to-end inference for ConditionalSplineFlow with Q75 condition scaling.
    flow_scaler_C structure: {scaler_y_p, eye_scale_h, eye_scale_w, eye_scale_j, col_map}
    """
    def __init__(self, flow, scaler_X, flow_scaler_C, zig_wrapper, device):
        self.flow = flow
        self.scaler_X = scaler_X
        self.flow_scaler_C = flow_scaler_C
        self.zig = zig_wrapper
        self.device = device
        self._build_cond_scaler()

    def _build_cond_scaler(self):
        c = self.flow_scaler_C
        self._eye_scale_h = c['eye_scale_h']
        self._eye_scale_w = c['eye_scale_w']
        self._eye_scale_j = c['eye_scale_j']
        self.scaler_y_p = c['scaler_y_p']
        self.col_map = c['col_map']

    def scale_condition(self, cond_raw):
        """
        Scale raw conditions [power, jitter, height, width] to flow's condition space.
        - power: StandardScaler on log10(raw)
        - jitter/height/width: Q75 scaling (log10 / eye_scale)
        Returns torch tensor of shape (batch, 4).
        """
        if cond_raw.ndim == 1:
            cond_raw = cond_raw.reshape(1, -1)
        power_raw = cond_raw[:, 0:1]
        jitter_raw = cond_raw[:, 1:2]
        height_raw = cond_raw[:, 2:3]
        width_raw = cond_raw[:, 3:4]

        p_log = np.log10(np.clip(power_raw, 1e-12, None))
        p_scaled = self.scaler_y_p.transform(p_log)

        j_scaled = np.log10(np.clip(jitter_raw, 1e-12, None)) / self._eye_scale_j
        h_scaled = np.log10(np.clip(height_raw, 1e-12, None)) / self._eye_scale_h
        w_scaled = np.log10(np.clip(width_raw, 1e-12, None)) / self._eye_scale_w

        cond_scaled = np.concatenate([p_scaled, j_scaled, h_scaled, w_scaled], axis=1)
        return torch.from_numpy(cond_scaled.astype(np.float32)).to(self.device)

    def inverse_scale_params(self, x_scaled):
        if torch.is_tensor(x_scaled):
            x_np = x_scaled.detach().cpu().numpy()
        else:
            x_np = x_scaled
        x_log = x_np * self.scaler_X.scale_ + self.scaler_X.mean_
        x_log = np.clip(x_log, LOG_BOUNDS[0], LOG_BOUNDS[1])
        return 10.0 ** x_log

    def sample_and_rank(self, target_cond_raw, n=5000, top_k=10, valid_threshold=0.9,
                        diversity_weight=0.1, physics_weight=0.5, df_empirical=None, k_manifold=5):
        self.flow.eval()
        cond_s = self.scale_condition(target_cond_raw)
        target_arr = np.array(target_cond_raw).reshape(1, -1)

        def _sample_and_select(n_cands, min_threshold=0.3):
            cond_s_expanded = cond_s.repeat_interleave(n_cands, dim=0) if cond_s.size(0) == 1 else cond_s
            with torch.no_grad():
                samples_s = self.flow.sample(cond_s, num_samples=n_cands, temperature=1.0)
                zig_out = self.zig(samples_s)
                pred_specs = zig_out['regressor']
                validity = zig_out['classifier']

                x_phys = self.inverse_scale_params(samples_s.cpu().numpy())
                x_phys_t = torch.from_numpy(x_phys.astype(np.float32)).to(samples_s.device)

                cond_u_t = torch.from_numpy(
                    np.repeat(np.array(target_cond_raw).reshape(1, -1), n_cands, axis=0).astype(np.float32)
                ).to(samples_s.device)

                _, I, _, Rd, Cs, Rs, VDD = x_phys_t.unbind(dim=1)
                power_t, _, eh_t, _ = cond_u_t.unbind(dim=1)
                gm_target = 0.5 + 3.0 * torch.sigmoid(eh_t)
                gm_loss = (I - gm_target) ** 2
                vd = VDD - I * Rd
                sat_loss = F.relu(0.3 - vd)
                rc = Rs * Cs
                freq_loss = F.softplus(torch.log10(rc / 1e-9))
                power_est = 4.0 * VDD * I
                power_log_loss = (torch.log10(power_est + 1e-12) - torch.log10(power_t + 1e-12)) ** 2
                phys_pen = gm_loss + sat_loss + freq_loss + power_log_loss

            valid_mask = validity >= valid_threshold
            if valid_mask.sum() < top_k:
                quant = max(0.3, (top_k / n_cands) * 0.8)
                valid_mask = validity >= validity.quantile(quant)

            samples_s_val = samples_s[valid_mask]
            pred_specs_val = pred_specs[valid_mask]
            validity_val = validity[valid_mask]
            phys_pen_val = phys_pen[valid_mask]
            x_phys_t_val = x_phys_t[valid_mask]

            if len(samples_s_val) == 0:
                return None

            target = cond_s_expanded[valid_mask] if cond_s_expanded.size(0) > 1 else cond_s.squeeze(0)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            diff = pred_specs_val - target
            weighted_err = (diff ** 2).sum(dim=-1)
            log_validity = -torch.log(torch.clamp(validity_val, 1e-6, 1 - 1e-6))

            manifold_dist = torch.zeros_like(weighted_err)
            if df_empirical is not None and len(df_empirical) > 0:
                try:
                    emp_specs = df_empirical[['power', 'stage_2_jitter',
                                              'stage_2_eye_max_height', 'stage_2_eye_max_width']].values
                    emp_specs_log = np.log10(np.clip(emp_specs, 1e-12, None))
                    emp_params = df_empirical[PARAM_COLS].values
                    target_spec_log = np.log10(np.clip(target_arr[0], 1e-12, None))
                    dists = np.max(np.abs(emp_specs_log - target_spec_log), axis=1)
                    k_eff = min(k_manifold, len(emp_params))
                    knn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
                    neighbor_params = emp_params[knn_idx]
                    neighbor_median = np.median(neighbor_params, axis=0)
                    neighbor_median_log = np.log10(np.clip(neighbor_median, 1e-12, None))
                    x_phys_log = torch.log10(x_phys_t_val + 1e-30)
                    neighbor_median_log_t = torch.from_numpy(neighbor_median_log.astype(np.float32)).to(x_phys_log.device)
                    manifold_dist = torch.norm(x_phys_log - neighbor_median_log_t, dim=-1)
                except Exception:
                    manifold_dist = torch.zeros_like(weighted_err)

            gamma = 0.1
            score = weighted_err + log_validity + physics_weight * phys_pen_val + gamma * manifold_dist

            # Deterministic anchored tie-breaker: prefer lower estimated power, then lower manifold distance.
            power_anchor = (4.0 * x_phys_t_val[:, 6] * x_phys_t_val[:, 1]) if x_phys_t_val.shape[1] >= 7 else torch.zeros_like(score)
            score = score + 1e-6 * power_anchor + 1e-7 * manifold_dist

            selected_indices = []
            remaining = torch.arange(len(score), device=score.device)
            for _ in range(top_k):
                if len(remaining) == 0:
                    break
                best_idx_in_rem = torch.argmin(score[remaining])
                best_idx = remaining[best_idx_in_rem].item()
                selected_indices.append(best_idx)
                if len(selected_indices) < top_k:
                    mask = torch.ones(len(remaining), dtype=torch.bool, device=remaining.device)
                    mask[best_idx_in_rem] = False
                    remaining = remaining[mask]

            if len(selected_indices) == 0:
                return None

            top_idx = selected_indices[0]
            top_validity = validity_val[top_idx].item()
            top_manifold = manifold_dist[top_idx].item() if df_empirical is not None else 0.0
            passes_threshold = (top_validity >= min_threshold) or (top_manifold < 2.0)

            return {
                'samples': samples_s_val[selected_indices],
                'preds': pred_specs_val[selected_indices],
                'validity': validity_val[selected_indices],
                'scores': score[selected_indices],
                'passes': passes_threshold,
                'top_validity': top_validity,
                'top_manifold': top_manifold,
            }

        result = _sample_and_select(n, min_threshold=0.3)
        if result is None:
            return np.zeros((top_k, 7)), np.zeros((top_k, 4)), np.zeros(top_k)

        if not result['passes'] and n < 2048:
            result2 = _sample_and_select(2048, min_threshold=0.25)
            if result2 is not None:
                result = result2
        if not result['passes'] and n < 5000:
            result2 = _sample_and_select(5000, min_threshold=0.25)
            if result2 is not None and result2['passes']:
                result = result2

        best_samples_s = result['samples']
        best_preds = result['preds']
        best_validity = result['validity']
        best_phys = self.inverse_scale_params(best_samples_s.cpu())

        return best_phys, best_preds.cpu().numpy(), best_validity.cpu().numpy()

    def _pred_specs_physical_from_scaled(self, pred_specs_scaled):
        power_scaled = pred_specs_scaled[:, 0]
        j_scaled = pred_specs_scaled[:, 1]
        h_scaled = pred_specs_scaled[:, 2]
        w_scaled = pred_specs_scaled[:, 3]
        power_raw = 10 ** (power_scaled * self.scaler_y_p.scale_[0] + self.scaler_y_p.mean_[0])
        j_raw = 10 ** (j_scaled * self._eye_scale_j)
        h_raw = 10 ** (h_scaled * self._eye_scale_h)
        w_raw = 10 ** (w_scaled * self._eye_scale_w)
        if hasattr(power_raw, 'cpu'):
            power_raw = power_raw.detach().cpu().numpy()
            j_raw = j_raw.detach().cpu().numpy()
            h_raw = h_raw.detach().cpu().numpy()
            w_raw = w_raw.detach().cpu().numpy()
        return np.stack([power_raw, j_raw, h_raw, w_raw], axis=1)


def load_flow_inference(teacher_dir, device=None):
    """Load flow model, scalers, and ZIG wrapper for Q75 architecture."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flow = ConditionalSplineFlow(dim=7, cond_dim=4, num_coupling=10,
                                  hidden_dim=256, num_layers=3,
                                  num_bins=8, tail_bound=3.0).to(device)
    flow.load_state_dict(torch.load(os.path.join(teacher_dir, 'ctle_conditional_flow.pt'),
                                     map_location=device))
    flow.eval()

    scaler_X = joblib.load(os.path.join(teacher_dir, 'flow_scaler_X.pkl'))
    flow_scaler_C = joblib.load(os.path.join(teacher_dir, 'flow_scaler_C.pkl'))

    eye_scale_h = flow_scaler_C['eye_scale_h']
    eye_scale_w = flow_scaler_C['eye_scale_w']
    eye_scale_j = flow_scaler_C['eye_scale_j']
    scaler_y_p = flow_scaler_C['scaler_y_p']

    zig_wrapper = ZIGHurdleWrapper(
        os.path.join(teacher_dir, 'hybrid_hurdle_ctle_model.pt'),
        device,
        eye_scale_h=eye_scale_h,
        eye_scale_w=eye_scale_w,
        eye_scale_j=eye_scale_j,
        scaler_y_p=scaler_y_p,
        per_target=PER_TARGET_HURDLE
    )

    return FlowInference(flow, scaler_X, flow_scaler_C, zig_wrapper, device)


class FlowTeacherLabeler:
    """High-level interface: target specs -> canonical params via spline flow + ZIG."""
    def __init__(self, teacher_dir, device):
        self.teacher_dir = teacher_dir
        self.device = device
        self.fi = load_flow_inference(teacher_dir, device)
        self.flow = self.fi.flow
        self.zig = self.fi.zig

    def label_single(self, target_spec, n_candidates=5000, valid_threshold=0.9, top_k=1):
        if n_candidates < top_k:
            n_candidates = top_k
        best_phys, _, _ = self.fi.sample_and_rank(
            np.array(target_spec).reshape(1, -1),
            n=n_candidates,
            top_k=top_k,
            valid_threshold=valid_threshold,
            diversity_weight=0.1,
            physics_weight=0.5,
        )
        return best_phys

    def label_batch(self, target_specs, n_candidates=5000, valid_threshold=0.9, top_k=1, verbose=True):
        n = len(target_specs)
        results = []
        for i in range(n):
            if verbose and (i + 1) % 500 == 0:
                _logger.info(f"  Labeled {i+1}/{n}...")
            results.append(self.label_single(
                target_specs[i], n_candidates=n_candidates,
                valid_threshold=valid_threshold, top_k=top_k
            ))
        if top_k == 1:
            params = np.array(results).squeeze(1)
        else:
            params = np.array(results)
        passed_mask = filter_by_zig_validity(
            params, self.fi.zig.model, self.fi.scaler_X,
            threshold=0.5, device=self.device
        )
        _logger.info(f"  label_batch: {passed_mask.sum()}/{n} passed validity gate "
                     f"(threshold=0.5)")
        if passed_mask.sum() < n * 0.5:
            relaxed_mask = filter_by_zig_validity(
                params, self.fi.zig.model, self.fi.scaler_X,
                threshold=0.3, device=self.device
            )
            _logger.info(f"    relaxed (0.3): {relaxed_mask.sum()}/{n} passed")
        # Preserve a 1:1 mapping with the input specs; callers decide how to filter.
        return params



# %% papermill={"duration": 0.01134, "end_time": "2026-07-01T22:28:19.369617+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.358277+00:00", "status": "completed"}
# Flow model already defined in Cell 4 above


# %% [markdown] papermill={"duration": 0.005282, "end_time": "2026-07-01T22:28:19.380147+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.374865+00:00", "status": "completed"}
#  ## Phase 2: Load Teacher Artifacts

# %% [markdown] papermill={"duration": 0.005216, "end_time": "2026-07-01T22:28:19.390715+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.385499+00:00", "status": "completed"}
#  ## Phase 3: Define ZIG Forward Model (Teacher Surrogate)

# %% papermill={"duration": 0.010736, "end_time": "2026-07-01T22:28:19.406917+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.396181+00:00", "status": "completed"}
# ZIG model loaded above



# %% [markdown] papermill={"duration": 0.005401, "end_time": "2026-07-01T22:28:19.417688+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.412287+00:00", "status": "completed"}
#  ## Phase 4: Load Scalers

# %% papermill={"duration": 0.047013, "end_time": "2026-07-01T22:28:19.470172+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.423159+00:00", "status": "completed"}
# Parameter scaler — flow model (used for flow inference only)
flow_scaler_X = joblib.load(os.path.join(TEACHER_DIR, 'flow_scaler_X.pkl'))
scaler_X = flow_scaler_X

# ZIG model artifacts from normalizing-flow-ctle-improved-finalized.py
# The hurdle teacher has:
# - a StandardScaler for parameter inputs
# - a StandardScaler for log10(power)
# - Q75 eye scales stored in config (not separate sklearn scalers)
zig_scaler_X = joblib.load(os.path.join(TEACHER_DIR, 'hybrid_hurdle_scaler_X.pkl'))
zig_scaler_y_p = joblib.load(os.path.join(TEACHER_DIR, 'hybrid_hurdle_scaler_y_power.pkl'))
zig_config = joblib.load(os.path.join(TEACHER_DIR, 'hybrid_hurdle_config.pkl'))

# Flow condition scaling used by the conditional flow:
# - power uses StandardScaler(log10(power))
# - jitter/height/width use Q75 eye scales from the teacher pipeline
flow_scaler_C = joblib.load(os.path.join(TEACHER_DIR, 'flow_scaler_C.pkl'))
flow_scaler_y_p = flow_scaler_C['scaler_y_p']
eye_scale_h = float(zig_config['eye_scale_h'])
eye_scale_w = float(zig_config['eye_scale_w'])
eye_scale_j = float(zig_config['eye_scale_j'])
PER_TARGET_HURDLE = bool(zig_config.get('per_target_hurdle', False))

# ZIG-related functions (filter_by_zig_consistency, RegimeAwareLoss, compute_zig_score)
# must use ZIG's own scalers, NOT flow's scalers.
# Override globals: ZIG operations now use correct scalers.
scaler_X = zig_scaler_X
scaler_y_p = zig_scaler_y_p

_logger.info("Teacher artifacts loaded")
_logger.info(f"  flow scaler_X mean: {flow_scaler_X.mean_[:3]}...")
_logger.info(f"  zig scaler_X mean: {zig_scaler_X.mean_[:3]}...")
_logger.info(f"  flow scaler_y_p: mean={flow_scaler_y_p.mean_[0]:.4f}, scale={flow_scaler_y_p.scale_[0]:.4f}")
_logger.info(f"  zig scaler_y_p:  mean={zig_scaler_y_p.mean_[0]:.4f}, scale={zig_scaler_y_p.scale_[0]:.4f}")
_logger.info(f"  eye_scale_h (Q75 scale): {eye_scale_h:.6f}")
_logger.info(f"  eye_scale_w (Q75 scale): {eye_scale_w:.6f}")
_logger.info(f"  eye_scale_j (Q75 scale): {eye_scale_j:.6f}")
_logger.info(f"  per_target_hurdle: {PER_TARGET_HURDLE}")
_logger.info("NOTE: scaler_X/scaler_y_p overridden to ZIG's scalers for ZIG operations.")
for name, flow_val, zig_val in [
    ("height", float(flow_scaler_C['eye_scale_h']), eye_scale_h),
    ("width", float(flow_scaler_C['eye_scale_w']), eye_scale_w),
    ("jitter", float(flow_scaler_C['eye_scale_j']), eye_scale_j),
]:
    rel_diff = abs(flow_val - zig_val) / max(abs(zig_val), 1e-12)
    _logger.info(f"  eye_scale_{name}: flow={flow_val:.6f} zig={zig_val:.6f} rel_diff={rel_diff:.3e}")
    if rel_diff > 1e-6:
        _logger.warning(f"  eye_scale_{name} differs between flow_scaler_C and zig_config")



# %% papermill={"duration": 0.389838, "end_time": "2026-07-01T22:28:19.867008+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.477170+00:00", "status": "completed"}
# Load the trained HybridHurdleModel as ZIG
zig_model = HybridHurdleModel(dropout=0.0, per_target=PER_TARGET_HURDLE).to(DEVICE)
zig_model.load_state_dict(torch.load(os.path.join(TEACHER_DIR, 'hybrid_hurdle_ctle_model.pt'),
                                      map_location=DEVICE))
zig_model.eval()
for p in zig_model.parameters():
    p.requires_grad = False
_logger.info("HybridHurdleModel (ZIG) loaded from checkpoint")



# %% [markdown] papermill={"duration": 0.006327, "end_time": "2026-07-01T22:28:19.879549+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.873222+00:00", "status": "completed"}
#  ## Phase 5: Define Spec Ranges and Output Bounds

# %% papermill={"duration": 0.025777, "end_time": "2026-07-01T22:28:19.911524+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.885747+00:00", "status": "completed"}
# Spec ranges (from empirical ctle_ml_dataset.csv, 83k rows)
SPEC_RANGES = {
    'power': (0.0012, 0.012),
    'stage_2_eye_max_height': (0.0, 88.4),
    'stage_2_eye_max_width': (0.0, 98.5),
    'stage_2_jitter': (1.57, 100.0),
}

# Input: [power, jitter, height, width]
# These are the target specs we condition on
SPEC_INPUT_COLS = ['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']

# Output: [fW, current, ind, Rd, Cs, Rs, VDD]
PARAM_COLS = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD']

# Log10 bounds for output parameterization
# These map sigmoid output [0,1] -> physical values via: 10^(log_lo + (log_hi - log_lo)*sigmoid)
PARAM_LOG_BOUNDS = {
    'fW':   (np.log10(1e-6),  np.log10(10.0)),     # [−6, 1]
    'current': (np.log10(5e-4), np.log10(2.5)),   # [−3.3, 0.4]
    'ind':  (np.log10(1e-12), np.log10(3.0)),     # [−12, 0.48]
    'Rd':   (np.log10(10),    np.log10(1500)),    # [1, 3.18]
    'Cs':   (np.log10(1e-15), np.log10(1e-9)),    # [−15, −9]
    'Rs':   (np.log10(10),    np.log10(1500)),    # [1, 3.18]
    'VDD':  (np.log10(0.6),   np.log10(1.2)),     # [−0.22, 0.08]
}

def params_from_logits(logits, param_log_bounds=PARAM_LOG_BOUNDS):
    """Convert bounded logits to physical parameter values."""
    # logits: shape (..., 7), unbounded
    # Apply sigmoid to get [0, 1]
    probs = torch.sigmoid(logits)
    
    results = {}
    for i, (name, (log_lo, log_hi)) in enumerate(param_log_bounds.items()):
        log_val = log_lo + (log_hi - log_lo) * probs[..., i]
        results[name] = torch.pow(10, log_val)
    
    return results

def params_to_log_logits(params_dict, param_log_bounds=PARAM_LOG_BOUNDS):
    """Convert physical params back to unbounded logits (for loss computation)."""
    logits = []
    for name, (log_lo, log_hi) in param_log_bounds.items():
        log_val = torch.log10(torch.clamp(params_dict[name], min=1e-12))
        # Invert: prob = (log_val - log_lo) / (log_hi - log_lo)
        prob = (log_val - log_lo) / (log_hi - log_lo)
        prob = torch.clamp(prob, 0.0, 1.0)
        # Inverse sigmoid
        logits.append(torch.logit(prob.clamp(1e-6, 1-1e-6)))
    return torch.stack(logits, dim=-1)

_logger.info("Parameter log bounds defined:")
for k, v in PARAM_LOG_BOUNDS.items():
    _logger.info(f"  {k}: [{v[0]:.2f}, {v[1]:.2f}] -> [10^{v[0]:.1f}, 10^{v[1]:.1f}]")


# %% [markdown] papermill={"duration": 0.007104, "end_time": "2026-07-01T22:28:19.926015+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.918911+00:00", "status": "completed"}
#  ## Phase 6: Define Student MLP Architecture

# %% papermill={"duration": 0.037144, "end_time": "2026-07-01T22:28:19.970293+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.933149+00:00", "status": "completed"}
# =============================================================================
# STUDENT MODEL: Regime-Aware Mixture of Experts (MoE)
# =============================================================================
# Hyperparameters — edit here to modify MoE architecture sizes
MOE_TRUNK_WIDTH     = 44   # shared trunk hidden dimension (4 → trunk → trunk → trunk)
MOE_TRUNK_LAYERS    = 3     # number of trunk layers (excluding input)
MOE_NUM_EXPERTS     = 3     # number of localized experts (Interior, Boundary, High-Jitter/Power)
MOE_TRUNK_ACTIVATION = nn.SiLU  # activation function for trunk layers (SiLU or GELU)
MOE_USE_LAYERNORM   = False # LayerNorm in trunk (keep False — inputs already scaled)
# Shared input representation. KNet mode is the controlled four-feature default.
INPUT_PREPROCESSING = getattr(_bo_args, 'input_preprocessing', None) or 'knet'
KN_INPUT_LOG_MIN = np.zeros(4, dtype=np.float32)
KN_INPUT_LOG_MAX = np.ones(4, dtype=np.float32)
# Re-apply BO MoE overrides that were parsed before this block was defined
# (the initial BO block at ~161 runs before these defaults exist, so it cannot
# override them; re-apply here so --moe-trunk-width etc. from mlp_bayes_opt.py
# actually take effect for the RegimeAwareMoE instantiation at 2482).
try:
    if '_bo_args' in globals() and _bo_args is not None:
        if getattr(_bo_args, 'moe_trunk_width', None) is not None:
            MOE_TRUNK_WIDTH = _bo_args.moe_trunk_width
        if getattr(_bo_args, 'moe_trunk_layers', None) is not None:
            MOE_TRUNK_LAYERS = _bo_args.moe_trunk_layers
        if getattr(_bo_args, 'moe_num_experts', None) is not None:
            MOE_NUM_EXPERTS = _bo_args.moe_num_experts
except Exception as _e:
    _logger.warning(f"[BO] MoE override re-apply failed: {_e}")

class RegimeAwareMoE(nn.Module):
    """
    Regime-Aware Mixture of Experts for CTLE inverse design.
    
    Input:  [power, jitter, height, width] (4 dims, raw physical units)
    Output: logits for 7 params -> bounded via sigmoid + scale -> 10^value
    
    Architecture:
        - Shared trunk: 3-layer MLP (width=64), SiLU, no LayerNorm
        - Gating: input-driven Linear(4→3) + softmax → 3 expert weights
        - 3 Localized experts: Linear(64→7) each
        - Output: weighted combination of expert logits → bounded sigmoid output
    
    Size target: ~40k–60k total parameters
    Drop-in replacement for BoundedMLP in RegimeAwareLoss / StudentEvaluator / DAgger loop.
    """
    def __init__(self, trunk_width=MOE_TRUNK_WIDTH, trunk_layers=MOE_TRUNK_LAYERS,
                 num_experts=MOE_NUM_EXPERTS, param_log_bounds=PARAM_LOG_BOUNDS,
                 activation=MOE_TRUNK_ACTIVATION, use_log_features=True):
        super().__init__()
        self.trunk_width = trunk_width
        self.trunk_layers = trunk_layers
        self.num_experts = num_experts
        self.param_log_bounds = param_log_bounds
        self.activation = activation
        self.use_log_features = use_log_features

        input_dim = 4
        trunk_input_dim = input_dim if INPUT_PREPROCESSING == 'knet' else (input_dim * 2 if use_log_features else input_dim)
        trunk_dims = [trunk_input_dim] + [trunk_width] * trunk_layers

        trunk_layers_list = []
        for i in range(trunk_layers):
            trunk_layers_list.append(nn.Linear(trunk_dims[i], trunk_dims[i+1]))
            if MOE_USE_LAYERNORM:
                trunk_layers_list.append(nn.LayerNorm(trunk_dims[i+1]))
            trunk_layers_list.append(activation())
        self.trunk = nn.Sequential(*trunk_layers_list)

        self.gate = nn.Linear(trunk_input_dim, num_experts, bias=False)
        self.regime_classifier = nn.Linear(trunk_input_dim, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Linear(trunk_width, 7) for _ in range(num_experts)
        ])

        self.log_lo = nn.Parameter(torch.zeros(7), requires_grad=False)
        self.log_hi = nn.Parameter(torch.zeros(7), requires_grad=False)
        for i, (name, (lo, hi)) in enumerate(param_log_bounds.items()):
            self.log_lo.data[i] = lo
            self.log_hi.data[i] = hi

        self.scaler_p_scale = flow_scaler_C['scaler_y_p'].scale_[0]
        self.scaler_p_mean = flow_scaler_C['scaler_y_p'].mean_[0]
        self._eye_scale_j = float(flow_scaler_C['eye_scale_j'])
        self._eye_scale_h = float(flow_scaler_C['eye_scale_h'])
        self._eye_scale_w = float(flow_scaler_C['eye_scale_w'])

    def scale_input(self, x):
        power_log = torch.log10(x[..., 0].clamp(min=1e-12))
        if INPUT_PREPROCESSING == 'knet':
            logs = torch.stack([power_log, torch.log10(x[..., 1].clamp(min=1e-12)),
                                torch.log10(x[..., 2].clamp(min=1e-12)),
                                torch.log10(x[..., 3].clamp(min=1e-12))], dim=-1)
            lo = torch.as_tensor(KN_INPUT_LOG_MIN, dtype=x.dtype, device=x.device)
            hi = torch.as_tensor(KN_INPUT_LOG_MAX, dtype=x.dtype, device=x.device)
            return (2.0 * (logs - lo) / (hi - lo).clamp(min=1e-8) - 1.0).clamp(-4.0, 4.0)
        power_scaled = (power_log - self.scaler_p_mean) / self.scaler_p_scale
        jitter_scaled = torch.log10(x[..., 1].clamp(min=1e-12)) / self._eye_scale_j
        height_scaled = torch.log10(x[..., 2].clamp(min=1e-12)) / self._eye_scale_h
        width_scaled = torch.log10(x[..., 3].clamp(min=1e-12)) / self._eye_scale_w
        linear_scaled = torch.stack([power_scaled, jitter_scaled, height_scaled, width_scaled], dim=-1)
        if self.use_log_features:
            log_scaled = torch.stack([power_log, torch.log10(x[..., 1].clamp(min=1e-12)),
                                      torch.log10(x[..., 2].clamp(min=1e-12)),
                                      torch.log10(x[..., 3].clamp(min=1e-12))], dim=-1)
            return torch.cat([linear_scaled, log_scaled], dim=-1)
        return linear_scaled

    def get_regime_label(self, x):
        h = self.scale_input(x)
        target_h = x[..., 2]
        target_w = x[..., 3]
        target_j = x[..., 1]
        target_p = x[..., 0]
        regimes = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        regimes[(target_h >= 5) & (target_w >= 5) & (target_j <= 90) & (target_p >= 0.003)] = 0
        regimes[(target_h < 5) | (target_w < 5)] = 1
        regimes[(target_j > 90) | (target_p < 0.003)] = 2
        return regimes

    def forward(self, x, return_regime_loss=False):
        x_s = self.scale_input(x)
        h = self.trunk(x_s)
        gate_weights = F.softmax(self.gate(x_s), dim=-1)
        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=-1)
        logits = (expert_outputs * gate_weights.unsqueeze(-2)).sum(dim=-1)

        if return_regime_loss:
            regime_labels = self.get_regime_label(x)
            regime_logits = self.regime_classifier(x_s)
            regime_loss = F.cross_entropy(regime_logits, regime_labels)
            return logits, regime_loss
        return logits

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        results = {}
        for i, name in enumerate(self.param_log_bounds.keys()):
            log_val = self.log_lo[i] + (self.log_hi[i] - self.log_lo[i]) * probs[..., i]
            results[name] = torch.pow(10, log_val)
        return results

    def get_bounded_output(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        bounded_log = self.log_lo.unsqueeze(0) + (self.log_hi.unsqueeze(0) - self.log_lo.unsqueeze(0)) * probs
        physical = {name: torch.pow(10, bounded_log[:, i]) for i, name in enumerate(self.param_log_bounds.keys())}
        return logits, bounded_log, physical

    def extra_repr(self):
        return (f"trunk_width={self.trunk_width}, trunk_layers={self.trunk_layers}, "
                f"num_experts={self.num_experts}")


class BoundedMLP(nn.Module):
    """Compact MLP with bounded log-space outputs for CTLE inverse design.
    
    Input: [power, jitter, height, width] (4 dims)
    Output: logits for 7 params -> bounded via sigmoid + scale -> 10^value
    """
    def __init__(self, hidden_dim=128, num_layers=4, 
                 input_dim=4, output_dim=7,
                 param_log_bounds=PARAM_LOG_BOUNDS,
                 use_per_output_heads=False):
        super().__init__()
        self.param_log_bounds = param_log_bounds
        self.use_per_output_heads = use_per_output_heads
        
        layers = []
        dims = [4 if INPUT_PREPROCESSING == 'knet' else input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.GELU(),
                nn.LayerNorm(dims[i+1]),
            ])
        self.backbone = nn.Sequential(*layers)
        
        if use_per_output_heads:
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(output_dim)])
        else:
            self.head = nn.Linear(hidden_dim, output_dim)
        
        self.log_lo = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        self.log_hi = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        
        for i, (name, (lo, hi)) in enumerate(param_log_bounds.items()):
            self.log_lo.data[i] = lo
            self.log_hi.data[i] = hi
        
        self.scaler_p_scale = flow_scaler_C['scaler_y_p'].scale_[0]
        self.scaler_p_mean = flow_scaler_C['scaler_y_p'].mean_[0]
        self._eye_scale_j = float(flow_scaler_C['eye_scale_j'])
        self._eye_scale_h = float(flow_scaler_C['eye_scale_h'])
        self._eye_scale_w = float(flow_scaler_C['eye_scale_w'])
    
    def scale_input(self, x):
        """
        Scale input specs [power, jitter, height, width] using Q75 condition scaling.
        - power: StandardScaler on log10(power)
        - jitter/height/width: Q75 (log10(raw) / eye_scale)
        """
        power_log = torch.log10(x[..., 0].clamp(min=1e-12))
        if INPUT_PREPROCESSING == 'knet':
            logs = torch.stack([power_log, torch.log10(x[..., 1].clamp(min=1e-12)),
                                torch.log10(x[..., 2].clamp(min=1e-12)),
                                torch.log10(x[..., 3].clamp(min=1e-12))], dim=-1)
            lo = torch.as_tensor(KN_INPUT_LOG_MIN, dtype=x.dtype, device=x.device)
            hi = torch.as_tensor(KN_INPUT_LOG_MAX, dtype=x.dtype, device=x.device)
            return (2.0 * (logs - lo) / (hi - lo).clamp(min=1e-8) - 1.0).clamp(-4.0, 4.0)
        power_scaled = (power_log - self.scaler_p_mean) / self.scaler_p_scale
        jitter_scaled = torch.log10(x[..., 1].clamp(min=1e-12)) / self._eye_scale_j
        height_scaled = torch.log10(x[..., 2].clamp(min=1e-12)) / self._eye_scale_h
        width_scaled = torch.log10(x[..., 3].clamp(min=1e-12)) / self._eye_scale_w
        return torch.stack([power_scaled, jitter_scaled, height_scaled, width_scaled], dim=-1)

    def forward(self, x):
        """Forward pass. Returns unbounded logits."""
        x = self.scale_input(x)
        h = self.backbone(x)
        if self.use_per_output_heads:
            logits = torch.cat([head(h) for head in self.heads], dim=-1)
        else:
            logits = self.head(h)
        return logits
    
    def predict(self, x):
        """Forward pass with bounded output. Returns physical param dict."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        results = {}
        for i, name in enumerate(self.param_log_bounds.keys()):
            log_val = self.log_lo[i] + (self.log_hi[i] - self.log_lo[i]) * probs[..., i]
            results[name] = torch.pow(10, log_val)
        return results

    def get_bounded_output(self, x):
        """Returns (logits, bounded_log_values, physical_params)."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        bounded_log = self.log_lo.unsqueeze(0) + (self.log_hi.unsqueeze(0) - self.log_lo.unsqueeze(0)) * probs
        physical = {name: torch.pow(10, bounded_log[:, i]) for i, name in enumerate(self.param_log_bounds.keys())}
        return logits, bounded_log, physical


# %% papermill={"duration": 0.390682, "end_time": "2026-07-01T22:28:20.367544+00:00", "exception": false, "start_time": "2026-07-01T22:28:19.976862+00:00", "status": "completed"}
# Load historical data to get spec distributions
# import zipfile

CSV_PATHS = []
history_dir = DATA_DIR

csv_files = [
    'dataset_log_part_1_may3.csv', 'dataset_log_part_1_may4.csv',
    'dataset_log_part_2_may3.csv', 'dataset_log_part_2_may4.csv',
    'dataset_log_part_3_may3.csv', 'dataset_log_part_3_may4.csv',
    'dataset_log_part_4_may3.csv', 'dataset_log_part_4_may4.csv',
    'dataset_log_part_5_may3.csv', 'dataset_log_part_5_may4.csv',
    'dataset_log_part_6_may3.csv', 'dataset_log_part_6_may4.csv',
    'dataset_log_may_7.csv',
    'dataset_log_march12.csv','dataset_log_may_18.csv'
]

# # Try loading from the backup zip
# backup_zip = os.path.join(history_dir, 'history_csvs_backup.zip')
dfs = []
for fname in csv_files:
    fpath = os.path.join(history_dir, fname)
    
    if not os.path.exists(fpath):
        _logger.warning(f"  Warning: {fname} not found, skipping")
        continue
    df = pd.read_csv(fpath)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index = True)
_logger.info(f"Total rows: {len(combined)}")
_logger.info(f"Columns: {list(combined.columns)[:10]}...")


# %% papermill={"duration": 0.046427, "end_time": "2026-07-01T22:28:20.421150+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.374723+00:00", "status": "completed"}
# Map column names from history CSV to our naming
COL_MAPPING = {
    'eye_maxHeight_norm Vout_2 56G': 'stage_2_eye_max_height',
    'eye_maxWidth_norm Vout_2 56G': 'stage_2_eye_max_width',
    'eye_p2pJitterAverage_norm stage 2': 'stage_2_jitter',
}

# Filter and prepare data
required_cols = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD', 
                  'power', 'stage_2_eye_max_height', 'stage_2_eye_max_width', 'stage_2_jitter']

# Rename columns
combined = combined.rename(columns=COL_MAPPING)

# Keep only rows with all required cols
df = combined[required_cols].copy()

# Filter: jitter > 0, power > 0, no NaN
mask = (df['stage_2_jitter'] > 0) & (df['power'] > 0) & (df['stage_2_eye_max_height'] > 0) & (df['stage_2_eye_max_width']>0) & ~df.isna().any(axis=1)
df = df[mask].reset_index(drop=True)
_logger.info(f"Filtered rows: {len(df)}")

if len(df) > 0:
    _spec_logs = np.log10(np.clip(
        df[['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']].values,
        1e-12, None))
    _spec_lo = _spec_logs.min(axis=0)
    _spec_hi = _spec_logs.max(axis=0)
    _spec_pad = 0.05 * np.maximum(_spec_hi - _spec_lo, 1e-8)
    KN_INPUT_LOG_MIN = (_spec_lo - _spec_pad).astype(np.float32)
    KN_INPUT_LOG_MAX = (_spec_hi + _spec_pad).astype(np.float32)
    _logger.info(f"Input preprocessing: {INPUT_PREPROCESSING}; KNet log bounds="
                 f"{KN_INPUT_LOG_MIN.tolist()}..{KN_INPUT_LOG_MAX.tolist()}")

# Show spec distributions
for col in ['power', 'stage_2_eye_max_height', 'stage_2_eye_max_width', 'stage_2_jitter']:
    nz = df[df[col] > 0][col]
    _logger.info(f"{col}: n={len(nz)}, min={nz.min():.4f}, max={nz.max():.4f}, median={nz.median():.4f}")


# %% [markdown] papermill={"duration": 0.0067, "end_time": "2026-07-01T22:28:20.434866+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.428166+00:00", "status": "completed"}
#  ## Phase 8: Implement Teacher Inference (Canonical Labeling)

# %% papermill={"duration": 0.014828, "end_time": "2026-07-01T22:28:20.456337+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.441509+00:00", "status": "completed"}
def compute_zig_score(params_log, zig_model, scaler_X, target_specs,
                         eye_scale_h, eye_scale_w, eye_scale_j, scaler_y_p):
    """
    Compute ZIG-based score for a candidate parameter set using Q75 scaler structure.

    HybridHurdleModel outputs:
      pred_h_soft, pred_w_soft, pred_j_soft: already in physical units (p_valid * expm1(log1p_pred))
      mu_power: StandardScaler on log10(power) — needs StandardScaler inverse

    Returns dict with predicted physical specs and validity score.
    """
    x_scaled = scaler_X.transform(params_log)
    x_tensor = torch.from_numpy(x_scaled.astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        out = zig_model(x_tensor)

    # pred_h_soft/pred_w_soft/pred_j_soft are already in physical units (expm1 applied in model)
    pred_h = out['pred_h_soft'].cpu().numpy()
    pred_w = out['pred_w_soft'].cpu().numpy()
    pred_j = out['pred_j_soft'].cpu().numpy()

    # Power: StandardScaler inverse on log10 scale
    power_scaled = out['mu_power'].cpu().numpy() * scaler_y_p.scale_[0] + scaler_y_p.mean_[0]
    pred_p = 10 ** power_scaled

    # Validity: product of per-metric validity probabilities
    validity = out['p_valid'].cpu().numpy()

    return {
        'pred_height': pred_h,
        'pred_width': pred_w,
        'pred_jitter': pred_j,
        'pred_power': pred_p,
        'validity': validity,
    }



# %% [markdown] papermill={"duration": 0.006865, "end_time": "2026-07-01T22:28:20.470365+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.463500+00:00", "status": "completed"}
#  ## Phase 9: Build Distillation Dataset with Adaptive Labeling

# %% papermill={"duration": 0.02605, "end_time": "2026-07-01T22:28:20.503196+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.477146+00:00", "status": "completed"}
# Phase 9: Build Distillation Dataset with Flow-Generated Canonical Labels
#
# TWO-PHASE TRAINING:
#   Phase 1 (primary):   Train on flow-generated canonical labels via FlowTeacherLabeler
#   Phase 2 (fine-tune): Fine-tune on empirical data with tunable weight λ_fine
#
# The flow generates diverse candidates for each target spec, ZIG ranks them,
# and the best (canonical) candidate becomes the distillation label.
# Target specs are still sampled from empirical distribution (preserves spec-space coverage).

def sample_target_specs(df, n_samples, boundary_ratio=0.3):
    """Sample target specs with priority on boundary cases.
    
    boundary_ratio: fraction of samples drawn from near-boundary regions
    """
    n_boundary = int(n_samples * boundary_ratio)
    n_interior = n_samples - n_boundary
    
    targets = []
    
    # Boundary samples: near zero-inflated or ceiling regions
    # Zero-inflated: height or width near 0
    near_zero = df[(df['stage_2_eye_max_height'] < 5) | (df['stage_2_eye_max_width'] < 5)].sample(n_boundary, replace=True)
    for _, row in near_zero.iterrows():
        targets.append({
            'power': row['power'],
            'height': row['stage_2_eye_max_height'],
            'width': row['stage_2_eye_max_width'],
            'jitter': row['stage_2_jitter'],
            'params': row[PARAM_COLS].values,
            'region': 'zero_inflated',
        })
    
    # Interior samples: randomly from full dataset
    interior = df.sample(n_interior, replace=True)
    for _, row in interior.iterrows():
        targets.append({
            'power': row['power'],
            'height': row['stage_2_eye_max_height'],
            'width': row['stage_2_eye_max_width'],
            'jitter': row['stage_2_jitter'],
            'params': row[PARAM_COLS].values,
            'region': 'interior',
        })
    
    return targets

def is_boundary_case(specs):
    """Identify if a target spec is a boundary case needing Cadence validation."""
    if specs['height'] < 5 or specs['width'] < 5:
        return True
    if specs['jitter'] > 90:
        return True
    if specs['power'] > 0.011:
        return True
    return False

def create_blended_distillation_dataset(df, teacher_labeler=None, n_samples=5000, n_candidates=5000,
                                        boundary_ratio=0.3, tolerance=0.05):
    """
    Creates a dataset where labels are either:
    - empirical (real) parameters if a noisy spec matches an existing design within `tolerance`,
    - otherwise flow-generated canonical parameters.

    tolerance: relative tolerance (e.g., 0.05 = 5%) in each spec dimension.
    """

    if teacher_labeler is None:
        teacher_labeler = FlowTeacherLabeler(TEACHER_DIR, DEVICE)
    
    # 1. Sample target specs (clean) from empirical distribution
    targets = sample_target_specs(df, n_samples, boundary_ratio)
    
    # 2. Build nearest-neighbor index on empirical specs (log10 transformed for relative distances)
    emp_specs = df[['power', 'stage_2_jitter', 'stage_2_eye_max_height', 
                    'stage_2_eye_max_width']].values.copy()
    # Convert to log space so that relative differences become absolute differences
    emp_specs_log = np.log10(emp_specs + 1e-12)
    nn = NearestNeighbors(n_neighbors=1, metric='chebyshev')
    nn.fit(emp_specs_log)
    
    # 3. For each target, apply noise and decide label source
    data_list = []
    for i, t in enumerate(targets):
        # Generate noisy specs (same as in original create_flow_distillation_dataset)
        noisy_power = t['power'] * np.random.uniform(0.95, 1.05)
        noisy_height = t['height'] * np.random.uniform(0.9, 1.1) if t['height'] > 1 else np.random.uniform(0, 2)
        noisy_width = t['width'] * np.random.uniform(0.9, 1.1) if t['width'] > 1 else np.random.uniform(0, 2)
        noisy_jitter = np.clip(t['jitter'] * np.random.uniform(0.95, 1.05), 1.57, 100)
        noisy_specs = np.array([noisy_power, noisy_jitter, noisy_height, noisy_width])
        
        # Convert noisy specs to log10 space and query nearest empirical point
        noisy_specs_log = np.log10(noisy_specs + 1e-12)
        distances, indices = nn.kneighbors(noisy_specs_log.reshape(1, -1))
        # Chebyshev distance in log space = max relative difference in original space
        if distances[0, 0] < tolerance:
            # Close match → use empirical parameters (physical values)
            row = df.iloc[indices[0, 0]]
            params = row[PARAM_COLS].values
        else:
            # No close match → use flow teacher
            params = teacher_labeler.label_single(noisy_specs, n_candidates=n_candidates,
                                                  valid_threshold=0.9, top_k=1)
            params = params[0]  # label_single with top_k=1 returns (1,7)
        
        data_list.append({
            'power': noisy_specs[0],
            'height': noisy_specs[2],
            'width': noisy_specs[3],
            'jitter': noisy_specs[1],
            'params': params,
        })
    
    _logger.info(f"Blended distillation dataset: {len(data_list)} samples "
                 f"(empirical matches used where distance < {tolerance})")
    return data_list

def create_flow_distillation_dataset(df, n_samples=10000, n_candidates=5000,
                                    boundary_ratio=0.3, teacher_labeler=None):
    if teacher_labeler is None:
        teacher_labeler = FlowTeacherLabeler(TEACHER_DIR, DEVICE)

    # Step 1: Sample clean target specs (same as before)
    targets = sample_target_specs(df, n_samples, boundary_ratio)
    # Step 2: Add noise to the specs to create the actual inputs the student will see
    noisy_specs_batch = []
    for t in targets:
        # Same noise logic as before, but applied to each spec individually
        noisy_power = t['power'] * np.random.uniform(0.95, 1.05)
        noisy_height = t['height'] * np.random.uniform(0.9, 1.1) if t['height'] > 1 else np.random.uniform(0, 2)
        noisy_width = t['width'] * np.random.uniform(0.9, 1.1) if t['width'] > 1 else np.random.uniform(0, 2)
        noisy_jitter = np.clip(t['jitter'] * np.random.uniform(0.95, 1.05), 1.57, 100)
        noisy_specs_batch.append([noisy_power, noisy_jitter, noisy_height, noisy_width])
    noisy_specs_batch = np.array(noisy_specs_batch)

    # Step 3: Generate canonical labels from the NOISY specs (this is the key fix!)
    canonical_params_batch = teacher_labeler.label_batch(
        noisy_specs_batch,                    # <-- use noisy specs here
        n_candidates=n_candidates,
        valid_threshold=0.9,
        top_k=1,
        verbose=True
    )

    keep_valid = filter_by_zig_validity(
        canonical_params_batch, zig_model, scaler_X, threshold=0.5, device=DEVICE
    )
    n_invalid = int((~keep_valid).sum())
    if n_invalid > 0:
        _logger.info(f"Flow teacher produced {n_invalid}/{len(noisy_specs_batch)} low-validity initial labels; replacing with empirical k-NN fallback")
        for i in np.where(~keep_valid)[0]:
            canonical_params_batch[i] = empirical_fallback_label(noisy_specs_batch[i], df, k=3)

    # Step 4: Build the dataset list (now input = noisy_specs, output = labels for noisy specs)
    data_list = []
    for i, t in enumerate(targets):
        data_list.append({
            'power': noisy_specs_batch[i, 0],
            'height': noisy_specs_batch[i, 2],
            'width': noisy_specs_batch[i, 3],
            'jitter': noisy_specs_batch[i, 1],
            'params': canonical_params_batch[i],   # already the correct mapping
        })

    _logger.info(f"Flow distillation dataset: {len(data_list)} samples ready (noisy inputs → teacher labels for those inputs)")
    return data_list



# %% [markdown] papermill={"duration": 0.006736, "end_time": "2026-07-01T22:28:20.516495+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.509759+00:00", "status": "completed"}
#  ## Phase 10: Regime-Aware Loss Function

# %% papermill={"duration": 0.034162, "end_time": "2026-07-01T22:28:20.558452+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.524290+00:00", "status": "completed"}
def huber_loss(pred, target, delta=1.0):
    """Huber loss - robust to outliers."""
    diff = torch.abs(pred - target)
    return torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))

class RegimeAwareLoss(nn.Module):
    """
    Loss function for HybridHurdleModel + Q75 condition scaling.
    Forward consistency uses:
      - HybridHurdleModel output keys: pred_h_soft, pred_w_soft, pred_j_soft, mu_power, p_valid
      - Q75 inverse: raw = 10^(scaled * eye_scale) for eye metrics
      - StandardScaler inverse for power: raw = 10^(scaled * scale + mean)
    """
    def __init__(self, zig_model, scaler_X,
                 eye_scale_h, eye_scale_w, eye_scale_j, scaler_y_p,
                 alpha_spec=1.0, beta_phys=0.1, gamma_mono=0.01, alpha_invalid=0.5,
                 empirical_df=None, k_manifold=5, alpha_manifold=0.1):
        super().__init__()
        self.zig = zig_model
        self.scaler_X = scaler_X
        safe_scale = np.maximum(scaler_X.scale_, 1e-6).astype(np.float32)
        self._scaler_X_scale = safe_scale
        self._scaler_X_mean = scaler_X.mean_.astype(np.float32)
        self._scaler_y_p_scale = float(scaler_y_p.scale_[0])
        self._scaler_y_p_mean = float(scaler_y_p.mean_[0])
        self._eye_scale_h = float(eye_scale_h)
        self._eye_scale_w = float(eye_scale_w)
        self._eye_scale_j = float(eye_scale_j)
        self.alpha_spec = alpha_spec
        self.beta_phys = beta_phys
        self.gamma_mono = gamma_mono
        self.alpha_invalid = alpha_invalid
        self.empirical_df = empirical_df
        self.k_manifold = k_manifold
        self.alpha_manifold = alpha_manifold
        self._emp_nn = None
        self._emp_params_log = None
        if empirical_df is not None and len(empirical_df) > 0:
            emp_specs = empirical_df[['power', 'stage_2_jitter',
                                      'stage_2_eye_max_height', 'stage_2_eye_max_width']].values
            self._emp_specs_log = np.log10(np.clip(emp_specs, 1e-12, None))
            self._emp_params_log = np.log10(np.clip(empirical_df[PARAM_COLS].values, 1e-12, None))
            self._emp_k = min(k_manifold, len(self._emp_specs_log))
            self._emp_nn = NearestNeighbors(n_neighbors=self._emp_k, metric='chebyshev')
            self._emp_nn.fit(self._emp_specs_log)

    def forward(self, student, spec_targets, canonical_params, logits=None):
        spec_tensor = torch.stack([
            spec_targets['power'],
            spec_targets['jitter'],
            spec_targets['height'],
            spec_targets['width'],
        ], dim=-1)

        if logits is None:
            logits = student(spec_tensor)
        probs = torch.sigmoid(logits)
        log_lo = student.log_lo.unsqueeze(0)
        log_hi = student.log_hi.unsqueeze(0)
        bounded_log = log_lo + (log_hi - log_lo) * probs

        # 1. Imitation loss with teacher label clamping to student bounds
        clamped_params = torch.clamp(canonical_params, log_lo, log_hi)
        L_imit = huber_loss(bounded_log, clamped_params, delta=0.5).mean()

        # 2. Forward consistency with HybridHurdleModel outputs + Q75 scaling
        #    Dynamic lambda: disable surrogate loss when ZIG validity is low (boundary blindspot)
        x_mean_t = torch.from_numpy(self._scaler_X_mean).to(bounded_log.device)
        x_scale_t = torch.from_numpy(self._scaler_X_scale).to(bounded_log.device)
        x_scaled = (bounded_log - x_mean_t) / (x_scale_t + 1e-8)
        x_scaled = torch.clamp(x_scaled, -10.0, 10.0)

        out = self.zig(x_scaled)

        pred_h = out['pred_h_soft'].clamp(min=1e-12, max=1e12)
        pred_w = out['pred_w_soft'].clamp(min=1e-12, max=1e12)
        pred_j = out['pred_j_soft'].clamp(min=1e-12, max=1e12)

        power_scaled = out['mu_power'] * self._scaler_y_p_scale + self._scaler_y_p_mean
        pred_p = torch.clamp(10.0 ** power_scaled, min=1e-12, max=1e12)

        if 'p_valid' in out:
            p_valid = out['p_valid'].clamp(1e-6, 1.0 - 1e-6)
        else:
            p_valid = (out['p_valid_h'] * out['p_valid_w'] * out['p_valid_j']).clamp(1e-6, 1.0 - 1e-6)
        invalidity_score = 1.0 - p_valid

        # Dynamic lambda: zero out forward-ZIG loss on low-validity boundary regions
        lambda_zig = torch.where(p_valid > 0.6, torch.ones_like(p_valid), torch.zeros_like(p_valid))

        # Regime-aware masking
        target_h = spec_targets['height']
        target_w = spec_targets['width']
        target_j = spec_targets['jitter']
        target_p = spec_targets['power']

        abs_err_h = (pred_h - target_h).abs()
        abs_err_w = (pred_w - target_w).abs()
        abs_err_j = (pred_j - target_j).abs()
        abs_err_p = (pred_p - target_p).abs()
        eps = 1e-6

        err_h = torch.where(target_h < BOUNDARY_ABS_TOLERANCES['height'][0],
                            abs_err_h / BOUNDARY_ABS_TOLERANCES['height'][1],
                            abs_err_h / target_h.clamp(min=eps))
        err_w = torch.where(target_w < BOUNDARY_ABS_TOLERANCES['width'][0],
                            abs_err_w / BOUNDARY_ABS_TOLERANCES['width'][1],
                            abs_err_w / target_w.clamp(min=eps))
        is_high_jitter = target_j > BOUNDARY_ABS_TOLERANCES['jitter'][0]
        err_j = torch.where(is_high_jitter,
                             abs_err_j / target_j.clamp(min=eps) / BOUNDARY_ABS_TOLERANCES['jitter'][1],
                             abs_err_j / target_j.clamp(min=eps))
        err_p = torch.where(target_p < BOUNDARY_ABS_TOLERANCES['power'][0],
                            abs_err_p / BOUNDARY_ABS_TOLERANCES['power'][1],
                            abs_err_p / target_p.clamp(min=eps))

        # Forward-ZIG consistency loss with per-sample confidence gating.
        per_sample_forward = err_h**2 + err_w**2 + err_j**2 + err_p**2
        lambda_norm = lambda_zig.sum().clamp(min=1.0)
        L_forward_zig = (per_sample_forward * lambda_zig).sum() / lambda_norm
        L_spec = L_forward_zig  # alias for compatibility

        # 3. Manifold prior loss L_manifold (step 4)
        L_manifold = torch.tensor(0.0, device=bounded_log.device)
        if self._emp_nn is not None and self._emp_params_log is not None:
            try:
                spec_log_batch = np.log10(np.clip(spec_tensor.detach().cpu().numpy(), 1e-12, None))
                _, knn_idx = self._emp_nn.kneighbors(spec_log_batch, return_distance=True)
                neighbor_median_log = np.median(self._emp_params_log[knn_idx], axis=1)
                neighbor_median_log_t = torch.from_numpy(neighbor_median_log.astype(np.float32)).to(bounded_log.device)
                L_manifold = ((bounded_log - neighbor_median_log_t) ** 2).sum(dim=1).mean() * self.alpha_manifold
            except Exception:
                L_manifold = torch.tensor(0.0, device=bounded_log.device)

        # 4. Physics loss
        physical = {name: torch.pow(10, bounded_log[:, i]) for i, name in enumerate(PARAM_COLS)}
        rc_rd_cs = 1.0 / (physical['Rd'] * physical['Cs'] + 1e-12)
        rc_rs_cs = 1.0 / (physical['Rs'] * physical['Cs'] + 1e-12)
        L_phys_rc = torch.relu(torch.log10(rc_rd_cs + 1e-12) - 12).clamp(max=3) ** 2 + \
                    torch.relu(torch.log10(rc_rs_cs + 1e-12) - 12).clamp(max=3) ** 2
        power_approx = 4 * physical['VDD'] * physical['current']
        L_phys_power = torch.relu(target_p / (power_approx + 1e-12) - 5).clamp(max=2) ** 2 + \
                       torch.relu(power_approx / (target_p + 1e-12) - 5).clamp(max=2) ** 2
        L_phys = (L_phys_rc + L_phys_power).mean() * self.beta_phys

        # Invalid design penalty
        L_invalid = invalidity_score.mean()
        L_total = L_imit + self.alpha_spec * L_spec + L_phys + self.alpha_invalid * L_invalid + L_manifold

        if torch.isnan(L_total) or torch.isinf(L_total):
            return {
                'total': L_total.detach().clone(),
                'L_imit_t': L_imit.detach().clone(),
                'L_spec_t': L_spec.detach().clone(),
                'L_phys_t': L_phys.detach().clone(),
                'L_invalid_t': L_invalid.detach().clone(),
                'L_manifold_t': L_manifold.detach().clone() if isinstance(L_manifold, torch.Tensor) else torch.tensor(0.0),
                'imit': 0.0,
                'spec': 0.0,
                'phys': 0.0,
                'invalid': 0.0,
                'manifold': 0.0,
                '_nan': True,
            }

        return {
            'total': L_total,
            'L_imit_t': L_imit,
            'L_spec_t': L_spec,
            'L_phys_t': L_phys,
            'L_invalid_t': L_invalid,
            'L_manifold_t': L_manifold if isinstance(L_manifold, torch.Tensor) else torch.tensor(0.0, device=L_total.device),
            'imit': L_imit.item(),
            'spec': L_spec.item(),
            'phys': L_phys.item(),
            'invalid': L_invalid.item(),
            'manifold': L_manifold.item() if isinstance(L_manifold, torch.Tensor) else 0.0,
            '_nan': False,
        }

_logger.info("Regime-aware loss defined (Q75 + HybridHurdleModel)")



# %% [markdown] papermill={"duration": 0.007029, "end_time": "2026-07-01T22:28:20.572530+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.565501+00:00", "status": "completed"}
#  ## Phase 11: Dataset and DataLoader

# %% papermill={"duration": 0.054051, "end_time": "2026-07-01T22:28:20.633411+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.579360+00:00", "status": "completed"}
def check_label_quality(teacher_labeler, specs, zig, scaler_X, threshold=0.10):
    params = teacher_labeler.label_batch(specs, n_candidates=3000, top_k=1).squeeze(1)
    params_log = np.log10(np.clip(params, 1e-12, None))
    x_scaled = scaler_X.transform(params_log)
    x_t = torch.from_numpy(x_scaled.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        out = zig(x_t)

    pred_power = 10**(out['mu_power'].cpu().numpy() * scaler_y_p.scale_ + scaler_y_p.mean_)
    pred_h = out['pred_h_soft'].cpu().numpy()   # already physical
    pred_w = out['pred_w_soft'].cpu().numpy()
    pred_j = out['pred_j_soft'].cpu().numpy()

    target_p = specs[:, 0]; target_j = specs[:, 1]
    target_h = specs[:, 2]; target_w = specs[:, 3]

    err_p = np.abs(pred_power - target_p) / np.maximum(target_p, 1e-6)
    err_h = np.abs(pred_h - target_h) / np.maximum(target_h, 1e-6)
    err_w = np.abs(pred_w - target_w) / np.maximum(target_w, 1e-6)
    err_j = np.abs(pred_j - target_j) / np.maximum(target_j, 1e-6)
    max_err = np.maximum.reduce([err_p, err_h, err_w, err_j])
    fail = max_err > threshold
    return fail.mean(), max_err, params


class DistillationDataset(Dataset):
    """Rolling distillation dataset that grows via DAgger iterations.

    Tracks a 'hard buffer' — samples added in the last DAgger iteration —
    which are oversampled 10× during training via WeightedRandomSampler.
    """
    MAX_PARAMS = 7

    def __init__(self, data_list):
        self.data = data_list
        self._loader = None
        self._val_loader = None
        self._train_indices = None
        self._val_indices = None
        self._hard_buffer_start_idx = len(data_list)
        self._hard_indices = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        specs = torch.tensor([d['power'], d['jitter'], d['height'], d['width']], dtype=torch.float32)
        params_arr = np.asarray(d['params']).ravel()
        params_log = np.log10(np.clip(params_arr, 1e-12, None))
        pad_size = self.MAX_PARAMS - len(params_log)
        if pad_size > 0:
            params_log = np.pad(params_log, (0, pad_size), constant_values=-30.0)
        params = torch.tensor(params_log, dtype=torch.float32)
        return specs, params

    def append_samples(self, specs_list, params_list):
        """Append new (spec, params) pairs and invalidate the DataLoader cache.

        Newly added samples are marked as 'hard buffer' for oversampling.
        Newly added samples are added to the training pool (not val set).
        """
        self._hard_buffer_start_idx = len(self.data)
        self._hard_indices = []
        for spec, params in zip(specs_list, params_list):
            params_arr = np.asarray(params)
            if params_arr.ndim > 1:
                params_arr = params_arr.squeeze()
            params_arr = np.clip(np.asarray(params).ravel(), 1e-12, None)
            self.data.append({
                'power': float(spec[0]),
                'jitter': float(spec[1]),
                'height': float(spec[2]),
                'width': float(spec[3]),
                'params': params_arr,
            })
        self._hard_indices = list(range(self._hard_buffer_start_idx, len(self.data)))
        self._loader = None
        self._val_loader = None

    def split_train_val(self, val_frac=0.1):
        """Store train/val indices; val set is never used for training."""
        n = len(self.data)
        all_idx = np.random.permutation(n)
        n_val = int(val_frac * n)
        self._val_indices = all_idx[:n_val].tolist()
        self._train_indices = all_idx[n_val:].tolist()
        self._loader = None
        self._val_loader = None
        return self._train_indices, self._val_indices

    def get_loader(self, batch_size=256, shuffle=True, hard_weight=10.0):
        """Return DataLoader for train_indices only with WeightedRandomSampler.

        hard_weight: multiplier for sampling weight of newly added (hard) samples.
        Always rebuilds the sampler to include newly appended samples.
        """
        indices = self._train_indices if self._train_indices is not None else list(range(len(self.data)))
        train_subset = Subset(self, indices)
        if hard_weight > 1.0 and len(self._hard_indices) > 0:
            index_to_local = {idx: pos for pos, idx in enumerate(indices)}
            valid_hard = [i for i in self._hard_indices if i in index_to_local]
            weights = torch.ones(len(indices))
            for i_hard in valid_hard:
                weights[index_to_local[i_hard]] = hard_weight
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(indices),
                replacement=True,
            )
            self._loader = DataLoader(
                train_subset, batch_size=batch_size, sampler=sampler, drop_last=True
            )
        else:
            self._loader = DataLoader(train_subset, batch_size=batch_size,
                                     shuffle=shuffle,
                                     drop_last=True)
        return self._loader

    def get_val_loader(self, batch_size=256):
        """Return DataLoader for val_indices only."""
        if self._val_indices is None:
            return None
        if self._val_loader is None:
            val_subset = Subset(self, self._val_indices)
            self._val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        return self._val_loader


class StudentEvaluator:
    """
    Run student inference and flag failure cases using HybridHurdleModel + Q75 scaling.
    ZIG model output keys: pred_h_soft, pred_w_soft, pred_j_soft, mu_power, p_valid
    Q75 inverse for eye: raw = 10^(scaled * eye_scale)
    Power inverse: raw = 10^(scaled * scale + mean) via StandardScaler
    """
    def __init__(self, student, scaler_X, zig_model,
                 eye_scale_h, eye_scale_w, eye_scale_j, scaler_y_p, device):
        self.student = student
        self.scaler_X = scaler_X
        self.zig = zig_model
        self._eye_scale_h = float(eye_scale_h)
        self._eye_scale_w = float(eye_scale_w)
        self._eye_scale_j = float(eye_scale_j)
        self._scaler_y_p_scale = float(scaler_y_p.scale_[0])
        self._scaler_y_p_mean = float(scaler_y_p.mean_[0])
        self.device = device
        safe_scale = np.maximum(scaler_X.scale_, 1e-6).astype(np.float32)
        self._scaler_X_scale = safe_scale
        self._scaler_X_mean = scaler_X.mean_.astype(np.float32)

    def identify_failures(self, val_specs, threshold=0.10):
        """Return failure mask and per-spec metrics using validity-first adaptive degradation."""
        self.student.eval()

        with torch.no_grad():
            specs_t = torch.from_numpy(val_specs).float().to(self.device)
            logits = self.student(specs_t)
            probs = torch.sigmoid(logits)
            log_lo = self.student.log_lo.unsqueeze(0)
            log_hi = self.student.log_hi.unsqueeze(0)
            bounded_log = log_lo + (log_hi - log_lo) * probs

            x_mean_t = torch.from_numpy(self._scaler_X_mean).to(self.device)
            x_scale_t = torch.from_numpy(self._scaler_X_scale).to(self.device)
            x_scaled = (bounded_log - x_mean_t) / (x_scale_t + 1e-8)
            x_scaled = torch.clamp(x_scaled, -10.0, 10.0)

            out = self.zig(x_scaled)

            pred_h = out['pred_h_soft'].clamp(min=1e-12, max=1e12)
            pred_w = out['pred_w_soft'].clamp(min=1e-12, max=1e12)
            pred_j = out['pred_j_soft'].clamp(min=1e-12, max=1e12)
            power_scaled = out['mu_power'] * self._scaler_y_p_scale + self._scaler_y_p_mean
            pred_p = torch.clamp(10.0 ** power_scaled, 1e-12, 1e12)
            if 'p_valid' in out:
                p_valid = out['p_valid'].clamp(1e-6, 1.0 - 1e-6)
            else:
                p_valid = (out['p_valid_h'] * out['p_valid_w'] * out['p_valid_j']).clamp(1e-6, 1.0 - 1e-6)

            pred_specs = np.stack([
                pred_p.detach().cpu().numpy(),
                pred_j.detach().cpu().numpy(),
                pred_h.detach().cpu().numpy(),
                pred_w.detach().cpu().numpy(),
            ], axis=1)
            metric_dict = compute_forward_errors(pred_specs, val_specs)
            all_errors = np.stack([
                metric_dict['err_p'],
                metric_dict['err_j'],
                metric_dict['err_h'],
                metric_dict['err_w'],
            ], axis=1)
            nan_mask = np.isnan(all_errors).any(axis=1) | np.isnan(pred_specs).any(axis=1)
            invalid_mask = p_valid.detach().cpu().numpy() < VALIDITY_THRESHOLD
            failure_mask = nan_mask | invalid_mask | metric_dict['failure_mask']
            all_errors = np.where(nan_mask[:, np.newaxis], np.inf, all_errors)

            target_p = specs_t[:, 0]
            target_j = specs_t[:, 1]
            target_h = specs_t[:, 2]
            target_w = specs_t[:, 3]
            is_boundary = ((target_h < BOUNDARY_ABS_TOLERANCES['height'][0]) |
                           (target_w < BOUNDARY_ABS_TOLERANCES['width'][0]) |
                           (target_j > BOUNDARY_ABS_TOLERANCES['jitter'][0]) |
                           (target_p < BOUNDARY_ABS_TOLERANCES['power'][0])).cpu().numpy()
            boundary_failure = failure_mask & is_boundary
            interior_failure = failure_mask & ~is_boundary

        metrics = {'errors': all_errors, 'mean_errors': all_errors.mean(axis=1),
                   'boundary_failure': boundary_failure, 'interior_failure': interior_failure,
                   'is_boundary': is_boundary,
                   'pred_specs': pred_specs,
                   'target_specs': val_specs.copy(),
                   'dim_fail_rates': (all_errors >= DEGRADE_REL_THRESHOLD).mean(axis=0),
                   'degraded_dims': metric_dict['degraded_dims'],
                   'failure_mask': failure_mask,
                   'invalid_mask': invalid_mask,
                   'p_valid': p_valid.detach().cpu().numpy()}
        return failure_mask, metrics

    def identify_failures_relaxed(self, val_specs, threshold=1.0):
        """Compatibility wrapper for the adaptive validity-first metric."""
        self.student.eval()

        with torch.no_grad():
            specs_t = torch.from_numpy(val_specs).float().to(self.device)
            logits = self.student(specs_t)
            probs = torch.sigmoid(logits)
            log_lo = self.student.log_lo.unsqueeze(0)
            log_hi = self.student.log_hi.unsqueeze(0)
            bounded_log = log_lo + (log_hi - log_lo) * probs

            x_mean_t = torch.from_numpy(self._scaler_X_mean).to(self.device)
            x_scale_t = torch.from_numpy(self._scaler_X_scale).to(self.device)
            x_scaled = (bounded_log - x_mean_t) / (x_scale_t + 1e-8)
            x_scaled = torch.clamp(x_scaled, -10.0, 10.0)

            out = self.zig(x_scaled)

            pred_h = out['pred_h_soft'].clamp(min=1e-12, max=1e12)
            pred_w = out['pred_w_soft'].clamp(min=1e-12, max=1e12)
            pred_j = out['pred_j_soft'].clamp(min=1e-12, max=1e12)
            power_scaled = out['mu_power'] * self._scaler_y_p_scale + self._scaler_y_p_mean
            pred_p = torch.clamp(10.0 ** power_scaled, 1e-12, 1e12)
            if 'p_valid' in out:
                p_valid = out['p_valid'].clamp(1e-6, 1.0 - 1e-6)
            else:
                p_valid = (out['p_valid_h'] * out['p_valid_w'] * out['p_valid_j']).clamp(1e-6, 1.0 - 1e-6)

            pred_specs = np.stack([
                pred_p.detach().cpu().numpy(),
                pred_j.detach().cpu().numpy(),
                pred_h.detach().cpu().numpy(),
                pred_w.detach().cpu().numpy(),
            ], axis=1)
            metric_dict = compute_forward_errors_relaxed(pred_specs, val_specs)
            all_errors = np.stack([
                metric_dict['err_p'],
                metric_dict['err_j'],
                metric_dict['err_h'],
                metric_dict['err_w'],
            ], axis=1)
            nan_mask = np.isnan(all_errors).any(axis=1) | np.isnan(pred_specs).any(axis=1)
            invalid_mask = p_valid.detach().cpu().numpy() < VALIDITY_THRESHOLD
            failure_mask = nan_mask | invalid_mask | metric_dict['failure_mask']
            all_errors = np.where(nan_mask[:, np.newaxis], np.inf, all_errors)

            target_p = specs_t[:, 0]
            target_j = specs_t[:, 1]
            target_h = specs_t[:, 2]
            target_w = specs_t[:, 3]
            is_boundary = ((target_h < BOUNDARY_ABS_TOLERANCES['height'][0]) |
                           (target_w < BOUNDARY_ABS_TOLERANCES['width'][0]) |
                           (target_j > BOUNDARY_ABS_TOLERANCES['jitter'][0]) |
                           (target_p < BOUNDARY_ABS_TOLERANCES['power'][0])).cpu().numpy()
            boundary_failure = failure_mask & is_boundary
            interior_failure = failure_mask & ~is_boundary

        metrics = {'errors': all_errors, 'mean_errors': all_errors.mean(axis=1),
                   'boundary_failure': boundary_failure, 'interior_failure': interior_failure,
                   'is_boundary': is_boundary,
                   'pred_specs': pred_specs,
                   'target_specs': val_specs.copy(),
                   'dim_fail_rates': (all_errors >= DEGRADE_REL_THRESHOLD).mean(axis=0),
                   'degraded_dims': metric_dict['degraded_dims'],
                   'failure_mask': failure_mask,
                   'invalid_mask': invalid_mask,
                   'p_valid': p_valid.detach().cpu().numpy()}
        return failure_mask, metrics

class TeacherLabeler:
    """
    Wrapper around FlowTeacherLabeler that adds ZIG validity filtering.
    Accepts a FlowTeacherLabeler instance and uses it to generate labels
    for failed specs, filtering by HybridHurdleModel validity.
    """
    def __init__(self, teacher_labeler, zig_model, scaler_X, device):
        self.teacher_labeler = teacher_labeler
        self.zig = zig_model
        self.scaler_X = scaler_X
        self.device = device

    def compute_validity(self, params_log):
        """Compute ZIG validity for params in log10 space using HybridHurdleModel."""
        x_scaled = self.scaler_X.transform(params_log)
        x_t = torch.from_numpy(x_scaled.astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self.zig(x_t)
        if 'p_valid' in out:
            validity = out['p_valid'].cpu().numpy()
        else:
            validity = (out['p_valid_h'] * out['p_valid_w'] * out['p_valid_j']).cpu().numpy()
        return validity

    def generate_labels(self, failed_specs, sample_budget=3000, validity_threshold=0.9):
        """
        Query FlowTeacherLabeler for canonical params and filter by ZIG validity.
        Returns (specs_list, params_list).
        """
        failed_array = np.array(failed_specs)
        raw_params = self.teacher_labeler.label_batch(
            failed_array, n_candidates=sample_budget,
            valid_threshold=0.9, top_k=1, verbose=True
        ).squeeze(1)  # (N, 1, 7) -> (N, 7)
        params_log = np.log10(np.clip(raw_params, 1e-12, None))
        validities = self.compute_validity(params_log)

        keep_mask = validities >= validity_threshold
        kept_params = raw_params[keep_mask]
        kept_specs = failed_array[keep_mask]

        _logger.info(f"  TeacherLabeler: {keep_mask.sum()}/{len(failed_specs)} passed "
                    f"ZIG validity >= {validity_threshold}")

        result_specs = [spec for spec in kept_specs]
        result_params = [params for params in kept_params]
        return result_specs, result_params


def sample_validation_specs(df, n_samples=2000, boundary_ratio=0.5, seed=None):
    """Sample validation specs with heavy boundary emphasis for DAgger evaluation.

    If ``seed`` is not None, every random draw is deterministic so repeating the
    call with the same seed yields the exact same spec set. ``seed=None``
    preserves the historical unseeded behavior.
    """
    n_boundary = int(n_samples * boundary_ratio)
    n_interior = n_samples - n_boundary
    boundary_mask = (
        (df['stage_2_eye_max_height'] < BOUNDARY_ABS_TOLERANCES['height'][0]) |
        (df['stage_2_eye_max_width'] < BOUNDARY_ABS_TOLERANCES['width'][0]) |
        (df['stage_2_jitter'] > BOUNDARY_ABS_TOLERANCES['jitter'][0]) |
        (df['power'] < BOUNDARY_ABS_TOLERANCES['power'][0])
    )
    boundary_df = df[boundary_mask]
    interior_df = df[~boundary_mask]

    if len(boundary_df) == 0:
        boundary_df = df
    if len(interior_df) == 0:
        interior_df = df

    if seed is not None:
        boundary_rows = boundary_df.sample(n_boundary, replace=len(boundary_df) < n_boundary, random_state=seed)
        interior_rows = interior_df.sample(n_interior, replace=len(interior_df) < n_interior, random_state=seed)
    else:
        boundary_rows = boundary_df.sample(n_boundary, replace=len(boundary_df) < n_boundary)
        interior_rows = interior_df.sample(n_interior, replace=len(interior_df) < n_interior)
    sampled = pd.concat([boundary_rows, interior_rows], ignore_index=True)
    shuffle_seed = seed if seed is not None else 42
    sampled = sampled.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    return sampled[['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']].values.astype(np.float32)


def create_empirical_distillation_dataset(df, n_samples=10000, boundary_ratio=0.3):
    """Create empirical fine-tune dataset (Phase 2 fine-tuning). Returns a list of dicts."""
    targets = sample_target_specs(df, n_samples, boundary_ratio)
    data_list = []
    for t in targets:
        noisy_specs = {
            'power': t['power'] * np.random.uniform(0.95, 1.05),
            'height': t['height'] * np.random.uniform(0.9, 1.1) if t['height'] > 1 else np.random.uniform(0, 2),
            'width': t['width'] * np.random.uniform(0.9, 1.1) if t['width'] > 1 else np.random.uniform(0, 2),
            'jitter': np.clip(t['jitter'] * np.random.uniform(0.95, 1.05), 1.57, 100),
        }
        noisy_params = t['params'] * np.random.uniform(0.98, 1.02)
        phys_lo = np.array([10**v[0] for v in PARAM_LOG_BOUNDS.values()])
        phys_hi = np.array([10**v[1] for v in PARAM_LOG_BOUNDS.values()])
        noisy_params = np.clip(noisy_params, phys_lo, phys_hi)
        data_list.append({
            'power': noisy_specs['power'],
            'height': noisy_specs['height'],
            'width': noisy_specs['width'],
            'jitter': noisy_specs['jitter'],
            'params': noisy_params,
        })
    return data_list


# =====================================================================
# DAGGER DISTILLATION EXECUTION
# =====================================================================


# %% papermill={"duration": 0.019891, "end_time": "2026-07-01T22:28:20.660162+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.640271+00:00", "status": "completed"}
# =====================================================================
def train_epoch(student, train_loader, optimizer, criterion, device, loss_weight=1.0):
    student.train()
    total_loss = 0
    losses = {'total': 0, 'imit': 0, 'spec': 0, 'phys': 0, 'invalid': 0}
    
    for specs, params_target in train_loader:
        specs = specs.to(device)
        params_target = params_target.to(device)
        
        optimizer.zero_grad()
        
        spec_targets = {
            'power': specs[:, 0],
            'height': specs[:, 2],
            'width': specs[:, 3],
            'jitter': specs[:, 1],
        }
        
        loss_dict = criterion(student, spec_targets, params_target)
        
        # Apply loss weight (for empirical fine-tune phase)
        weighted_loss = loss_dict['total'] * loss_weight
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss_dict['total'].item()
        losses['total'] += loss_dict['total'].item()
        for k in ('imit', 'spec', 'phys', 'invalid'):
            losses[k] += loss_dict[k]
    
    n = len(train_loader)
    return {k: float(v/n) for k, v in losses.items()}

def eval_epoch(student, val_loader, criterion, device):
    student.eval()
    total_loss = 0
    losses = {'total': 0, 'imit': 0, 'spec': 0, 'phys': 0, 'invalid': 0, 'manifold': 0}
    
    with torch.no_grad():
        for specs, params_target in val_loader:
            specs = specs.to(device)
            params_target = params_target.to(device)
            
            spec_targets = {
                'power': specs[:, 0],
                'height': specs[:, 2],
                'width': specs[:, 3],
                'jitter': specs[:, 1],
            }
            
            loss_dict = criterion(student, spec_targets, params_target)
            
            total_loss += loss_dict['total'].item()
            for k in ('imit', 'spec', 'phys', 'invalid', 'manifold'):
                losses[k] += loss_dict[k]
    
    n = len(val_loader)
    return {k: float(v/n) for k, v in losses.items()}

_logger.info("Training functions defined (two-phase)")


# %% papermill={"duration": 2441.971062, "end_time": "2026-07-01T23:09:02.638092+00:00", "exception": false, "start_time": "2026-07-01T22:28:20.667030+00:00", "status": "completed"}
def filter_by_zig_consistency(specs_arr, params_arr, zig_model, scaler_X, scaler_y_p, threshold=ERROR_THRESHOLD, device=DEVICE):
    """Keep only (spec, param) pairs that are valid and pass adaptive degradation checks."""
    params_log = np.log10(np.clip(params_arr, 1e-12, None))
    x_scaled = scaler_X.transform(params_log)
    x_t = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
    with torch.no_grad():
        out = zig_model(x_t)

    pred_p = 10**(out['mu_power'].cpu().numpy() * scaler_y_p.scale_[0] + scaler_y_p.mean_[0])
    pred_h = out['pred_h_soft'].cpu().numpy()
    pred_w = out['pred_w_soft'].cpu().numpy()
    pred_j = out['pred_j_soft'].cpu().numpy()

    pred_specs = np.stack([pred_p, pred_j, pred_h, pred_w], axis=1)
    metrics = compute_forward_errors(pred_specs, specs_arr)
    if 'p_valid' in out:
        p_valid = out['p_valid'].cpu().numpy()
    else:
        p_valid = (out['p_valid_h'] * out['p_valid_w'] * out['p_valid_j']).cpu().numpy()
    keep = (p_valid >= VALIDITY_THRESHOLD) & (~metrics['failure_mask'])

    n = len(specs_arr)
    n_boundary = ((specs_arr[:, 2] < BOUNDARY_ABS_TOLERANCES['height'][0]) |
                  (specs_arr[:, 3] < BOUNDARY_ABS_TOLERANCES['width'][0]) |
                  (specs_arr[:, 1] > BOUNDARY_ABS_TOLERANCES['jitter'][0]) |
                  (specs_arr[:, 0] < BOUNDARY_ABS_TOLERANCES['power'][0])).sum()
    _logger.info(f"  filter_by_zig_consistency: {keep.sum()}/{n} passed  "
                 f"(boundary={n_boundary}, interior={n-n_boundary})")
    return keep


def filter_by_zig_validity(params_arr, zig_model, scaler_X, threshold=0.5, device=DEVICE):
    """Keep params where ZIG validity p_valid >= threshold."""
    params_log = np.log10(np.clip(params_arr, 1e-12, None))
    x_scaled = scaler_X.transform(params_log)
    x_t = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
    with torch.no_grad():
        out = zig_model(x_t)
    if 'p_valid' in out:
        validity = out['p_valid'].cpu().numpy()
    else:
        validity = (out['p_valid_h'] * out['p_valid_w'] * out['p_valid_j']).cpu().numpy()
    keep = validity >= threshold
    n = len(params_arr)
    _logger.info(f"  filter_by_zig_validity: {keep.sum()}/{n} passed (threshold={threshold})")
    return keep


def filter_by_zig_consistency_relaxed(specs_arr, params_arr, zig_model, scaler_X, scaler_y_p, device=DEVICE):
    """Keep only (spec, param) pairs that are valid and pass adaptive degradation checks."""
    params_log = np.log10(np.clip(params_arr, 1e-12, None))
    x_scaled = scaler_X.transform(params_log)
    x_t = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
    with torch.no_grad():
        out = zig_model(x_t)

    pred_p = 10**(out['mu_power'].cpu().numpy() * scaler_y_p.scale_[0] + scaler_y_p.mean_[0])
    pred_h = out['pred_h_soft'].cpu().numpy()
    pred_w = out['pred_w_soft'].cpu().numpy()
    pred_j = out['pred_j_soft'].cpu().numpy()

    pred_specs = np.stack([pred_p, pred_j, pred_h, pred_w], axis=1)
    metrics = compute_forward_errors_relaxed(pred_specs, specs_arr)
    if 'p_valid' in out:
        p_valid = out['p_valid'].cpu().numpy()
    else:
        p_valid = (out['p_valid_h'] * out['p_valid_w'] * out['p_valid_j']).cpu().numpy()
    keep = (p_valid >= VALIDITY_THRESHOLD) & (~metrics['failure_mask'])

    n = len(specs_arr)
    n_boundary = ((specs_arr[:, 2] < BOUNDARY_ABS_TOLERANCES_RELAXED['height'][0]) |
                  (specs_arr[:, 3] < BOUNDARY_ABS_TOLERANCES_RELAXED['width'][0]) |
                  (specs_arr[:, 1] > BOUNDARY_ABS_TOLERANCES_RELAXED['jitter'][0]) |
                  (specs_arr[:, 0] < BOUNDARY_ABS_TOLERANCES_RELAXED['power'][0])).sum()
    _logger.info(f"  filter_by_zig_consistency_relaxed: {keep.sum()}/{n} passed  "
                 f"(boundary={n_boundary}, interior={n-n_boundary})")
    return keep


def empirical_fallback_label(spec, df, k=3):
    """
    Return median parameters of k nearest empirical neighbors to `spec` in log-spec space.
    Uses Chebyshev distance (max relative difference) in log10-transformed spec space.
    Guarantees a physically plausible label for any spec.
    """
    spec_log = np.log10(np.clip(spec, 1e-12, None))
    emp_specs = df[['power', 'stage_2_jitter', 'stage_2_eye_max_height',
                    'stage_2_eye_max_width']].values
    emp_specs_log = np.log10(np.clip(emp_specs, 1e-12, None))
    dists = np.max(np.abs(emp_specs_log - spec_log), axis=1)
    k_eff = min(k, len(dists))
    idx = np.argpartition(dists, k_eff - 1)[:k_eff]
    return np.median(df.iloc[idx][PARAM_COLS].values, axis=0)


def is_boundary_spec(spec):
    """Return True if spec is in the boundary region (hard for the teacher/evaluator)."""
    return (spec[2] < BOUNDARY_ABS_TOLERANCES['height'][0] or
            spec[3] < BOUNDARY_ABS_TOLERANCES['width'][0] or
            spec[1] > BOUNDARY_ABS_TOLERANCES['jitter'][0] or
            spec[0] < BOUNDARY_ABS_TOLERANCES['power'][0])


def compute_forward_errors(pred_specs, target_specs, boundary_tols=BOUNDARY_ABS_TOLERANCES,
                           default_threshold=ERROR_THRESHOLD):
    """
    Compute adaptive degradation metrics for forward consistency.
    pred_specs / target_specs: arrays of shape (N, 4) in order [power, jitter, height, width]
    Improvements are treated as neutral; only meaningful regressions count.
    """
    pred_p = pred_specs[:, 0]
    pred_j = pred_specs[:, 1]
    pred_h = pred_specs[:, 2]
    pred_w = pred_specs[:, 3]
    target_p = target_specs[:, 0]
    target_j = target_specs[:, 1]
    target_h = target_specs[:, 2]
    target_w = target_specs[:, 3]
    eps = 1e-6

    degrade_p = np.maximum((pred_p - target_p) / np.maximum(target_p, eps), 0.0)
    degrade_j = np.maximum((pred_j - target_j) / np.maximum(target_j, eps), 0.0)
    degrade_h = np.maximum((target_h - pred_h) / np.maximum(target_h, eps), 0.0)
    degrade_w = np.maximum((target_w - pred_w) / np.maximum(target_w, eps), 0.0)

    degraded = np.stack([
        degrade_p >= DEGRADE_REL_THRESHOLD,
        degrade_j >= DEGRADE_REL_THRESHOLD,
        degrade_h >= DEGRADE_REL_THRESHOLD,
        degrade_w >= DEGRADE_REL_THRESHOLD,
    ], axis=1)
    degraded_dims = degraded.sum(axis=1)
    max_err = np.maximum.reduce([degrade_p, degrade_j, degrade_h, degrade_w])
    failure_mask = degraded_dims >= MIN_DEGRADED_DIMS
    return {
        'err_h': degrade_h,
        'err_w': degrade_w,
        'err_j': degrade_j,
        'err_p': degrade_p,
        'max_err': max_err,
        'degraded': degraded,
        'degraded_dims': degraded_dims,
        'failure_mask': failure_mask,
    }


def compute_forward_errors_relaxed(pred_specs, target_specs):
    """Alias of adaptive degradation metric for compatibility with existing callers."""
    return compute_forward_errors(pred_specs, target_specs, boundary_tols=BOUNDARY_ABS_TOLERANCES_RELAXED,
                                  default_threshold=ERROR_THRESHOLD)


def log_failure_breakdown(metrics, prefix=""):
    """Emit compact diagnostics for failure modes and per-dimension error rates."""
    if not metrics or 'errors' not in metrics:
        return
    errs = np.asarray(metrics['errors'])
    if errs.size == 0:
        return
    dim_names = ['power', 'jitter', 'height', 'width']
    fail_by_dim = (errs >= DEGRADE_REL_THRESHOLD).mean(axis=0)
    mean_err = errs.mean(axis=0)
    median_err = np.median(errs, axis=0)
    p90_err = np.percentile(errs, 90, axis=0)
    for i, name in enumerate(dim_names):
        _logger.info(
            f"{prefix}{name}: mean={mean_err[i]:.4f} median={median_err[i]:.4f} "
            f"p90={p90_err[i]:.4f} degrade_rate={fail_by_dim[i]*100:.1f}%"
        )

    if 'degraded_dims' in metrics:
        degraded_dims = np.asarray(metrics['degraded_dims'])
        fail_mask = np.asarray(metrics.get('failure_mask', degraded_dims >= MIN_DEGRADED_DIMS))
        _logger.info(
            f"{prefix}adaptive: valid_threshold={VALIDITY_THRESHOLD:.2f} "
            f"degrade_threshold={DEGRADE_REL_THRESHOLD*100:.0f}% "
            f"min_bad_dims={MIN_DEGRADED_DIMS} "
            f"mean_bad_dims={degraded_dims.mean():.2f} "
            f"failure_rate={fail_mask.mean()*100:.1f}%"
        )
    if 'invalid_mask' in metrics:
        invalid_mask = np.asarray(metrics['invalid_mask'])
        _logger.info(f"{prefix}validity: invalid_rate={(invalid_mask.mean()*100):.1f}%")

    pred_specs = metrics.get('pred_specs')
    target_specs = metrics.get('target_specs')
    if pred_specs is not None and target_specs is not None:
        pred_specs = np.asarray(pred_specs)
        target_specs = np.asarray(target_specs)
        pred_mean = pred_specs.mean(axis=0)
        tgt_mean = target_specs.mean(axis=0)
        pred_med = np.median(pred_specs, axis=0)
        tgt_med = np.median(target_specs, axis=0)
        for i, name in enumerate(dim_names):
            _logger.info(
                f"{prefix}{name} raw: pred_mean={pred_mean[i]:.4f} target_mean={tgt_mean[i]:.4f} "
                f"pred_med={pred_med[i]:.4f} target_med={tgt_med[i]:.4f}"
            )


def log_label_quality_summary(name, specs_arr, params_arr, zig_model, scaler_X, scaler_y_p, device=DEVICE):
    """Summarize label validity and forward consistency for a batch of labels."""
    if len(specs_arr) == 0 or len(params_arr) == 0:
        _logger.info(f"{name}: no samples")
        return
    keep_valid = filter_by_zig_validity(params_arr, zig_model, scaler_X, threshold=0.5, device=device)
    keep_strict = filter_by_zig_consistency(specs_arr, params_arr, zig_model, scaler_X, scaler_y_p,
                                            threshold=ERROR_THRESHOLD, device=device)
    keep_relaxed = filter_by_zig_consistency_relaxed(specs_arr, params_arr, zig_model, scaler_X, scaler_y_p,
                                                     device=device)
    _logger.info(
        f"{name}: validity@0.5={keep_valid.mean()*100:.1f}%  "
        f"strict_consistency={keep_strict.mean()*100:.1f}%  "
        f"relaxed_consistency={keep_relaxed.mean()*100:.1f}%"
    )


# =====================================================================
#  DAgger DISTILLATION LOOP
# =====================================================================

dagger_history = {
    'iteration':     [],
    'failure_rate': [],
    'dataset_size': [],
    'train_loss':   [],
    'boundary_failure_rate': [],
    'interior_failure_rate': [],
}

loss_history = {
    'iteration': [],
    'train_imit': [], 'train_spec': [], 'train_phys': [], 'train_invalid': [], 'train_total': [],
    'train_manifold': [],
    'val_imit': [],   'val_spec': [],   'val_phys': [],   'val_invalid': [],   'val_total': [],
    'val_manifold': [],
}

# NOTE: train_loss > val_loss is expected here due to WeightedRandomSampler oversampling
# the hard buffer 10x — ~65% of each training epoch consists of hard (recently-failed)
# samples. This is NOT overfitting; it reflects the intentional difficulty reweighting.
# Use val_* losses (and val failure rate) as the ground-truth training signal.

# Instantiate teacher_labeler and evaluator
teacher_labeler = FlowTeacherLabeler(TEACHER_DIR, DEVICE)
evaluator = StudentEvaluator(
    student=None, scaler_X=scaler_X, zig_model=zig_model,
    eye_scale_h=eye_scale_h, eye_scale_w=eye_scale_w, eye_scale_j=eye_scale_j,
    scaler_y_p=scaler_y_p, device=DEVICE
)

# Instantiate student (Regime-Aware MoE — drop-in replacement for BoundedMLP)
# Architecture hyperparameters are at the top of the MoE section above.
# To switch back to BoundedMLP, replace RegimeAwareMoE(...) with:
#   BoundedMLP(hidden_dim=128, num_layers=8, input_dim=4, output_dim=7,
#              param_log_bounds=PARAM_LOG_BOUNDS, use_per_output_heads=True)
student = RegimeAwareMoE(
    trunk_width=MOE_TRUNK_WIDTH,
    trunk_layers=MOE_TRUNK_LAYERS,
    num_experts=MOE_NUM_EXPERTS,
    param_log_bounds=PARAM_LOG_BOUNDS,
    activation=MOE_TRUNK_ACTIVATION,
).to(DEVICE)
_logger.info(f"Student model: {sum(p.numel() for p in student.parameters()):,} params  "
             f"(trunk_width={MOE_TRUNK_WIDTH}, trunk_layers={MOE_TRUNK_LAYERS}, experts={MOE_NUM_EXPERTS})")

# Optimizer and scheduler (defined once, reused across iterations)
optimizer = torch.optim.AdamW(student.parameters(), lr=LR_INITIAL, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS_PER_ITER, eta_min=LR_FLOOR
)

# ── Canonical Phase-A gate (plan canonical-ctle-unify) ─────────────────────
# Placed BEFORE the initial-dataset build so Phase-A trials never pay for the
# (expensive) DAgger teacher labelling.  Exits via sys.exit(0) in every
# Phase-A path, so the DAgger body below only runs without Phase-A flags.
if _bo_args.canonical_dataset is not None or _bo_args.prepare_canonical_dataset is not None:
    from ctle_dagger_common import (create_canonical_flow_dataset,
                                     load_canonical_flow_dataset, run_phase_a_training)
    # Plan phase-a-mlp-guided-knet: a guided (mlp/fwd/power > 0) build needs
    # the frozen MLP teacher to label the canonical dataset and stamp
    # schema_rev >= 2 with the 'mlp_logits_trial0019' column. A guided run
    # without a teacher ckpt cannot produce the required columns, so refuse
    # the build at the gate rather than silently stamp a schema-1 .npz that
    # a downstream guided trial would crash on (common.py raises at runtime
    # but with a less actionable message and after teacher-labelling has
    # already paid its cost).
    # NOTE: the guided weights default to ON (mlp 1.0 / fwd 1.0 / power
    # 5e-4, KNet parity), so a standalone --canonical-dataset invocation
    # without --phase-a-mlp-teacher-ckpt now raises; pass explicit 0.0
    # weights for the legacy Huber-on-flow path.
    _guided_requested = (
        float(_bo_args.phase_a_mlp_weight) > 0.0
        or float(_bo_args.phase_a_fwd_weight) > 0.0
        or float(_bo_args.phase_a_power_weight) > 0.0
    )
    if _guided_requested and _bo_args.phase_a_mlp_teacher_ckpt is None:
        raise ValueError(
            "Guided Phase-A loss requested (mlp/fwd/power weight > 0) but "
            "--phase-a-mlp-teacher-ckpt is unset. The schema-2 canonical "
            "dataset requires the frozen PlainMLP teacher logits column; "
            "either pass --phase-a-mlp-teacher-ckpt <path/to/dagger_student_plain.pt> "
            "or set all three guided weights to 0.0 to reproduce the legacy "
            "Huber-on-flow path."
        )
    phase_a_mlp_teacher = None
    phase_a_mlp_teacher_id = None
    if _bo_args.phase_a_mlp_teacher_ckpt is not None:
        from ctle_dagger_common import load_plain_mlp_teacher
        phase_a_mlp_teacher = load_plain_mlp_teacher(
            _bo_args.phase_a_mlp_teacher_ckpt, device=DEVICE,
        )
        phase_a_mlp_teacher.eval()
        for _p in phase_a_mlp_teacher.parameters():
            _p.requires_grad = False
        _ckpt_path = Path(_bo_args.phase_a_mlp_teacher_ckpt)
        _ckpt_tag = f"{_ckpt_path.parent.parent.name}/{_ckpt_path.parent.name}/{_ckpt_path.stem}"
        from ctle_dagger_common import _file_sha256
        phase_a_mlp_teacher_id = (
            f"{_ckpt_tag}:{_file_sha256(_bo_args.phase_a_mlp_teacher_ckpt)[:16]}"
        )
        _logger.info(f"[phaseA] MLP teacher loaded from "
                     f"{_bo_args.phase_a_mlp_teacher_ckpt} (id={phase_a_mlp_teacher_id})")
    if _bo_args.prepare_canonical_dataset is not None:
        target_path = Path(_bo_args.prepare_canonical_dataset)
        if target_path.exists() and not _bo_args.canonical_dataset:
            _logger.info(f"[phaseA] canonical dataset already exists at {target_path}; loading")
            canonical = load_canonical_flow_dataset(target_path)
        else:
            teacher_labeler = FlowTeacherLabeler(TEACHER_DIR, DEVICE)
            canonical = create_canonical_flow_dataset(
                df, teacher_labeler=teacher_labeler,
                n_samples=int(_bo_args.phase_a_n_samples),
                boundary_ratio=float(_bo_args.phase_a_boundary_ratio),
                seed=int(_bo_args.seed if _bo_args.seed is not None else 100),
                output_path=target_path,
                zig_model=zig_model, scaler_X=scaler_X, device=DEVICE,
                mlp_teacher=phase_a_mlp_teacher,
                mlp_teacher_id=phase_a_mlp_teacher_id,
                power_norm=str(_bo_args.phase_a_power_norm),
            )
        if not _bo_args.canonical_dataset:
            _logger.info("[phaseA] --prepare-canonical-dataset set: exiting before training")
            sys.exit(0)
    if _bo_args.canonical_dataset is not None:
        canonical = load_canonical_flow_dataset(_bo_args.canonical_dataset)
        phase_a_epochs = int(_bo_args.phase_a_epochs if _bo_args.phase_a_epochs is not None
                              else _bo_args.epochs_per_iter if _bo_args.epochs_per_iter is not None
                              else EPOCHS_PER_ITER)
        ctx_phase_a = {
            "DEVICE": DEVICE, "scaler_X": scaler_X, "scaler_y_p": scaler_y_p,
            "zig_model": zig_model,
            "eye_scale_h": eye_scale_h, "eye_scale_w": eye_scale_w, "eye_scale_j": eye_scale_j,
            "canonical_dataset": canonical,
            "epochs": phase_a_epochs,
            "batch_size": int(_bo_args.batch_size if _bo_args.batch_size is not None else BATCH_SIZE),
            "lr": float(_bo_args.lr if _bo_args.lr is not None else LR_INITIAL),
            "weight_decay": float(_bo_args.weight_decay if _bo_args.weight_decay is not None else WEIGHT_DECAY),
            "output_dir": OUTPUT_DIR,
            "grad_clip": 1.0,
            "val_eval_every": int(_bo_args.earlystop_eval_every if _bo_args.earlystop_eval_every is not None
                                   else EARLYSTOP_EVAL_EVERY),
            "earlystop_patience": int(EARLYSTOP_PATIENCE_EPOCHS),
            "error_threshold": ERROR_THRESHOLD,
            "validity_weight": float(_bo_args.phase_a_validity_weight),
            "validity_ramp_start": int(_bo_args.phase_a_validity_ramp_start),
            "validity_ramp_epochs": int(_bo_args.phase_a_validity_ramp_epochs),
            "input_preprocessing": _bo_args.input_preprocessing or "q75",
            "input_log_min": getattr(student, "input_log_min", None) if hasattr(student, "input_log_min") else None,
            "input_log_max": getattr(student, "input_log_max", None) if hasattr(student, "input_log_max") else None,
            # Plan phase-a-mlp-guided-knet: MLP-guided 4-term loss wiring.
            "mlp_weight": float(_bo_args.phase_a_mlp_weight),
            "fwd_weight": float(_bo_args.phase_a_fwd_weight),
            "power_weight": float(_bo_args.phase_a_power_weight),
            "power_norm_const": (
                float(_bo_args.phase_a_power_norm_const)
                if _bo_args.phase_a_power_norm_const is not None else None),
        }
        _logger.info(f"[phaseA] canonical-dataset gate engaged: {len(canonical['specs'])} specs, "
                     f"epochs={phase_a_epochs}, batch={ctx_phase_a['batch_size']}, "
                     f"lr={ctx_phase_a['lr']:.2e}, "
                     f"validity_w={ctx_phase_a['validity_weight']:.3f} "
                     f"ramp=[{ctx_phase_a['validity_ramp_start']}+{ctx_phase_a['validity_ramp_epochs']}] "
                     f"mlp_w={ctx_phase_a['mlp_weight']} fwd_w={ctx_phase_a['fwd_weight']} "
                     f"power_w={ctx_phase_a['power_weight']}")
        run_phase_a_training(student, ctx_phase_a, model_name="phase_a_moe")
        _logger.info("[phaseA] canonical-dataset gate: training complete; exiting before DAgger body")
        sys.exit(0)

# ── Initial dataset ──────────────────────────────────────────────────
with Timer("build initial flow distillation dataset"):
    initial_data = create_flow_distillation_dataset(
        df,
        n_samples=N_EMPIRICAL_SAMPLES,
        n_candidates=N_CANDIDATES_INITIAL,
        boundary_ratio=BOUNDARY_RATIO,
        teacher_labeler=teacher_labeler,
    )
_logger.info(f"Initial dataset (flow teacher labels): {len(initial_data)} samples")
distillation_dataset = DistillationDataset(initial_data)
all_specs = np.array([[d['power'], d['jitter'], d['height'], d['width']] for d in distillation_dataset.data])
all_params = np.array([d['params'] for d in distillation_dataset.data])
with Timer("ZIG forward-consistency check on initial dataset"):
    keep_strict = filter_by_zig_consistency(all_specs, all_params, zig_model, scaler_X, scaler_y_p, threshold=ERROR_THRESHOLD, device=DEVICE)
    keep_relaxed = filter_by_zig_consistency_relaxed(all_specs, all_params, zig_model, scaler_X, scaler_y_p, device=DEVICE)
_logger.info(f"Initial dataset: {len(distillation_dataset)} samples")
_logger.info(f"Initial dataset forward-consistency (strict): {keep_strict.mean()*100:.1f}%  ({keep_strict.sum()}/{len(keep_strict)})")
_logger.info(f"Initial dataset forward-consistency (relaxed): {keep_relaxed.mean()*100:.1f}%  ({keep_relaxed.sum()}/{len(keep_relaxed)})")
log_label_quality_summary("Initial dataset labels", all_specs, all_params, zig_model, scaler_X, scaler_y_p, device=DEVICE)

# Carve out 10% validation split (never augmented by DAgger)
train_subset, val_subset = distillation_dataset.split_train_val(val_frac=0.1)
train_loader = distillation_dataset.get_loader(batch_size=BATCH_SIZE, shuffle=True, hard_weight=10.0)

val_loader = distillation_dataset.get_val_loader(batch_size=BATCH_SIZE)
_logger.info(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

# ── Warmup wrapper for ZIG-dependent losses ─────────────────────────
class RampedRegimeLoss(nn.Module):
    def __init__(self, base_loss, warmup_epochs=5):
        super().__init__()
        self.base_loss = base_loss
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = min(epoch, self.warmup_epochs)

    def forward(self, student, spec_targets, canonical_params, logits=None, loss_dict=None):
        if loss_dict is None:
            loss_dict = self.base_loss(student, spec_targets, canonical_params, logits=logits)
        r = self.current_epoch / max(1, self.warmup_epochs)
        L_imit = loss_dict['L_imit_t']
        L_spec = loss_dict['L_spec_t'] * r
        L_phys = loss_dict['L_phys_t'] * r
        L_invalid = loss_dict['L_invalid_t'] * r
        L_manifold = loss_dict.get('L_manifold_t', torch.tensor(0.0, device=L_imit.device))
        L_total = L_imit + r * (loss_dict['total'] - L_imit)
        return {
            'total': L_total,
            'L_imit_t': L_imit,
            'L_spec_t': L_spec,
            'L_phys_t': L_phys,
            'L_invalid_t': L_invalid,
            'L_manifold_t': L_manifold,
            'imit': loss_dict['imit'],
            'spec': L_spec.item(),
            'phys': L_phys.item(),
            'invalid': L_invalid.item(),
            'manifold': loss_dict.get('manifold', 0.0),
            '_nan': loss_dict.get('_nan', False),
        }

# ── Criterion (RegimeAwareLoss needs Q75 scalers) ────────────────────
criterion = RegimeAwareLoss(
    zig_model=zig_model,
    scaler_X=scaler_X,
    eye_scale_h=eye_scale_h,
    eye_scale_w=eye_scale_w,
    eye_scale_j=eye_scale_j,
    scaler_y_p=scaler_y_p,
    alpha_spec=1.5,
    beta_phys=0.1,
    gamma_mono=0.01,
    alpha_invalid=1.5,
    empirical_df=df,
    k_manifold=5,
    alpha_manifold=0.1,
)
ramped_criterion = RampedRegimeLoss(criterion, warmup_epochs=5)

# =====================================================================
# DIAGNOSTIC SCRIPT — run before training to identify the bottleneck
# =====================================================================
def run_diagnostics(teacher_labeler, df, zig_model, scaler_X, scaler_y_p, device, n=200):
    """Diagnose whether the bottleneck is the teacher, ZIG critic, or student."""
    _logger.info("="*60)
    _logger.info("  RUNNING DIAGNOSTICS")
    _logger.info("="*60)

    # Sample boundary specs (returns np.ndarray shape (n, 4) in [power, jitter, height, width] order)
    boundary_specs = sample_validation_specs(df, n_samples=n, boundary_ratio=1.0)

    _logger.info(f"1. Teacher flow validity yield on {n} boundary specs (n_candidates=10000, valid=0.5):")
    try:
        teacher_labels = teacher_labeler.label_batch(
            boundary_specs, n_candidates=10000, valid_threshold=0.5, top_k=1, verbose=False
        )
        keep_flow = filter_by_zig_validity(
            teacher_labels.squeeze(), zig_model, scaler_X, threshold=0.5, device=device
        )
        _logger.info(f"   Flow teacher yield: {keep_flow.sum()}/{n} ({keep_flow.mean()*100:.1f}%)")
    except Exception as e:
        _logger.info(f"   Flow teacher error: {e}")

    _logger.info(f"2. Empirical k-NN validity yield on {n} boundary specs (k=3):")
    knn_labels = np.array([empirical_fallback_label(s, df, k=3) for s in boundary_specs])
    keep_knn = filter_by_zig_validity(
        knn_labels.squeeze(), zig_model, scaler_X, threshold=0.5, device=device
    )
    _logger.info(f"   k-NN empirical yield: {keep_knn.sum()}/{n} ({keep_knn.mean()*100:.1f}%)")

    _logger.info(f"3. ZIG validity on {min(1000, len(df))} real empirical rows:")
    real_subset = df.sample(min(1000, len(df)), random_state=42)
    real_params = real_subset[PARAM_COLS].values
    keep_real = filter_by_zig_validity(
        real_params, zig_model, scaler_X, threshold=0.5, device=device
    )
    _logger.info(f"   ZIG validity on real data: {keep_real.sum()}/{len(keep_real)} ({keep_real.mean()*100:.1f}%)")

    _logger.info("="*60)
    _logger.info("  DIAGNOSTICS COMPLETE")
    _logger.info("="*60)
    _logger.info("Interpretation:")
    _logger.info("  If (1) >50%:  teacher is fine → focus on loss/replay")
    _logger.info("  If (1) <50%:  flow teacher is weak → increase candidate budget")
    _logger.info("  If (2) <50%:  empirical data is weak → need better training set")
    _logger.info("  If (3) <50%:  ZIG p_valid threshold may be too high (lower to 0.3)")
    return {
        'flow_yield': keep_flow.mean() if 'keep_flow' in dir() else None,
        'knn_yield': keep_knn.mean(),
        'zig_real_acc': keep_real.mean(),
    }

# Run diagnostics
diag_results = run_diagnostics(teacher_labeler, df, zig_model, scaler_X, scaler_y_p, DEVICE, n=200)

# ── Shared COMMON_EVAL_SPECS ─────────────────────────────────────────
# Single fixed shared eval set used for both per-epoch early-stop tracking
# AND per-iteration validation reporting (mirrors dagger-nuance-distillation-
# kirchhoffnet.py). Every iteration uses the *same* spec set so the
# per-iteration failure rates are directly comparable.
if COMMON_EVAL_SPECS is None:
    with Timer("build COMMON_EVAL_SPECS shared eval set"):
        COMMON_EVAL_SPECS = sample_validation_specs(
            df,
            n_samples=COMMON_EVAL_SIZE,
            boundary_ratio=BOUNDARY_RATIO,
            seed=COMMON_EVAL_SEED,
        )
    _logger.info(f"  COMMON_EVAL_SPECS: shape={COMMON_EVAL_SPECS.shape}, "
                 f"seed={COMMON_EVAL_SEED}")
else:
    _logger.info(f"  COMMON_EVAL_SPECS already set (shape="
                 f"{COMMON_EVAL_SPECS.shape}); reusing")


# ── Main DAgger loop ─────────────────────────────────────────────────
for dagger_iter in range(DAGGER_ITERATIONS):
    _logger.info(f"{'='*60}")
    _logger.info(f"  DAgger Iteration {dagger_iter + 1}/{DAGGER_ITERATIONS}")
    _logger.info(f"{'='*60}")
    _logger.info(f"Dataset size: {len(distillation_dataset)}")

    train_loader = distillation_dataset.get_loader(batch_size=BATCH_SIZE, shuffle=True, hard_weight=HARD_BUFFER_WEIGHT)
    val_loader = distillation_dataset.get_val_loader(batch_size=BATCH_SIZE)

    # ── 3a. Train student with early stopping on failure rate ─────
    best_val_loss = float('inf')
    patience = 0
    best_state = None
    best_failure_state = None
    # Snapshot the carried-forward student and measure its baseline on the
    # shared COMMON_EVAL_SPECS set (iter 1 has no prior model -> 100%).
    prev_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
    if dagger_iter == 0:
        prev_failure_rate = 1.0
        _logger.info(f"  Prev-iter baseline: no prior model (iteration 1) -> "
                     f"prev_failure_rate=100.0%")
    else:
        evaluator.student = student
        _prev_mask, _prev_metrics = evaluator.identify_failures(
            COMMON_EVAL_SPECS, threshold=ERROR_THRESHOLD
        )
        prev_failure_rate = float(_prev_mask.mean())
        _logger.info(f"  Prev-iter baseline: prev_failure_rate="
                     f"{prev_failure_rate*100:.2f}% on COMMON_EVAL_SPECS "
                     f"(n={len(COMMON_EVAL_SPECS)})")
    best_failure_rate = prev_failure_rate

    _logger.info(f"  Shared eval set (COMMON_EVAL_SPECS): n={len(COMMON_EVAL_SPECS)} "
                 f"specs (seed={COMMON_EVAL_SEED}, every={EARLYSTOP_EVAL_EVERY}ep, "
                 f"log_every={EARLYSTOP_LOG_EVERY}ep, skip_first={EARLYSTOP_SKIP_EPOCHS}ep)")

    epoch_losses = {'total': [], 'imit': [], 'spec': [], 'phys': [], 'invalid': [], 'manifold': [], 'regime': []}
    n_batches = 0
    n_nonfinite_batches = 0
    _div_consec = 0
    diverged_at = None
    for epoch in range(EPOCHS_PER_ITER):
        ramped_criterion.set_epoch(epoch)
        student.train()
        losses = {'total': 0, 'imit': 0, 'spec': 0, 'phys': 0, 'invalid': 0, 'manifold': 0, 'regime': 0}
        for specs_batch, params_batch in train_loader:
            specs_batch = specs_batch.to(DEVICE)
            params_batch = params_batch.to(DEVICE)

            spec_targets = {
                'power':  specs_batch[:, 0],
                'jitter': specs_batch[:, 1],
                'height': specs_batch[:, 2],
                'width':  specs_batch[:, 3],
            }

            optimizer.zero_grad()
            logits, regime_loss = student(specs_batch, return_regime_loss=True)
            base_loss_dict = ramped_criterion.base_loss(student, spec_targets, params_batch, logits=logits)
            ramped_loss_dict = ramped_criterion(student, spec_targets, params_batch, logits=logits, loss_dict=base_loss_dict)
            total_loss = ramped_loss_dict['total'] + 0.1 * regime_loss
            if not total_loss.requires_grad or not torch.isfinite(total_loss):
                losses['total'] += float(total_loss)
                n_nonfinite_batches += 1
            else:
                total_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
                if math.isfinite(grad_norm):
                    optimizer.step()
                else:
                    n_nonfinite_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    if n_nonfinite_batches == 1 or n_nonfinite_batches % 50 == 0:
                        _logger.warning(
                            f"    [grad-guard] non-finite grad norm at epoch {epoch+1}; "
                            f"skipped optimizer step ({n_nonfinite_batches} skipped this iteration)")
                losses['total'] += total_loss.item()
            for k in ('imit', 'spec', 'phys', 'invalid', 'manifold'):
                losses[k] += ramped_loss_dict[k]
            losses['regime'] += regime_loss.item()

        n = len(train_loader)
        avg_loss = losses['total'] / n
        avg_imit = losses['imit'] / n
        avg_spec = losses['spec'] / n
        avg_phys = losses['phys'] / n
        avg_invalid = losses['invalid'] / n
        avg_manifold = losses['manifold'] / n
        avg_regime = losses['regime'] / n

        for k in epoch_losses:
            epoch_losses[k] += [losses[k] / n]
        n_batches += 1

        # Evaluate on validation set and step scheduler
        val_losses = eval_epoch(student, val_loader, ramped_criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _logger.info(f"  Epoch {epoch+1:3d}/{EPOCHS_PER_ITER}  "
                         f"train={avg_loss:.4f}(imit={avg_imit:.3f},spec={avg_spec:.3f},phys={avg_phys:.3f},invalid={avg_invalid:.3f},manifold={avg_manifold:.3f},regime={avg_regime:.3f})  "
                         f"val={val_losses['total']:.4f}(imit={val_losses['imit']:.3f},spec={val_losses['spec']:.3f},phys={val_losses['phys']:.3f},invalid={val_losses['invalid']:.3f},manifold={val_losses['manifold']:.3f})  "
                         f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Save best state by val loss (for loss-aligned checkpointing)
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        # Early stopping on failure rate: evaluate on the SHARED COMMON_EVAL_SPECS set.
        # - First EARLYSTOP_SKIP_EPOCHS epochs: no best-tracking (warm-start guard).
        # - New best only when it beats running best by >= MIN_FAILURE_IMPROVEMENT.
        # - Patience never triggers a break (EARLYSTOP_PATIENCE_EPOCHS = EPOCHS_PER_ITER).
        if (epoch + 1) % EARLYSTOP_EVAL_EVERY == 0 or epoch == EPOCHS_PER_ITER - 1:
            evaluator.student = student
            failure_mask_check, metrics_check = evaluator.identify_failures(
                COMMON_EVAL_SPECS, threshold=ERROR_THRESHOLD
            )
            current_failure_rate = failure_mask_check.mean()
            if (epoch + 1) > EARLYSTOP_SKIP_EPOCHS:
                if current_failure_rate < best_failure_rate - MIN_FAILURE_IMPROVEMENT:
                    best_failure_rate = current_failure_rate
                    patience = 0
                    best_failure_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                    _logger.info(f"    earlystop@{epoch+1:3d}: failure_rate="
                                 f"{current_failure_rate*100:.2f}% "
                                 f"(new best, prev={prev_failure_rate*100:.2f}%, "
                                 f"delta={(prev_failure_rate - current_failure_rate)*100:+.2f}pts)")
                else:
                    patience += 1
                    _logger.info(f"    earlystop@{epoch+1:3d}: failure_rate="
                                 f"{current_failure_rate*100:.2f}% "
                                 f"(no accept; best={best_failure_rate*100:.2f}%, "
                                 f"min_delta={MIN_FAILURE_IMPROVEMENT*100:.1f}%, "
                                 f"patience={patience})")
                if DIVERGENCE_ABORT and (
                        current_failure_rate >= prev_failure_rate + DIVERGENCE_MARGIN):
                    _div_consec += 1
                    if _div_consec >= DIVERGENCE_CONSEC_EVALS:
                        diverged_at = epoch + 1
                        _logger.warning(
                            f"    [divergence] failure_rate >= "
                            f"{(prev_failure_rate + DIVERGENCE_MARGIN)*100:.1f}% for "
                            f"{DIVERGENCE_CONSEC_EVALS} consecutive evals; aborting "
                            f"iteration {dagger_iter+1} at epoch {epoch+1} "
                            f"(carry-forward guardrail will restore the best/previous model)")
                        break
                else:
                    _div_consec = 0
            else:
                _logger.info(f"    earlystop@{epoch+1:3d}: failure_rate="
                             f"{current_failure_rate*100:.2f}% "
                             f"(skipped, epoch {epoch+1} <= "
                             f"EARLYSTOP_SKIP_EPOCHS={EARLYSTOP_SKIP_EPOCHS})")
            if (epoch + 1) % EARLYSTOP_LOG_EVERY == 0 or epoch == EPOCHS_PER_ITER - 1:
                log_failure_breakdown(metrics_check, prefix="    earlystop/")

    # ── Carry-forward model selection with regression guardrail ────────
    iter_outcome = "keep_new"
    if (best_failure_state is not None
            and best_failure_rate < prev_failure_rate - MIN_FAILURE_IMPROVEMENT):
        student.load_state_dict(best_failure_state)
        student.to(DEVICE)
        _logger.info(f"  Carry-forward: restored best-failure-rate model "
                     f"({best_failure_rate*100:.2f}% on COMMON_EVAL_SPECS, "
                     f"prev={prev_failure_rate*100:.2f}%, "
                     f"delta={(prev_failure_rate - best_failure_rate)*100:+.2f}pts)")
    elif prev_state is not None:
        student.load_state_dict(prev_state)
        student.to(DEVICE)
        iter_outcome = "fallback_prev"
        _logger.warning(
            f"  Carry-forward: iteration {dagger_iter+1} no >="
            f"{MIN_FAILURE_IMPROVEMENT*100:.1f}% improvement "
            f"(prev={prev_failure_rate*100:.2f}% -> "
            f"best={best_failure_rate*100:.2f}%); kept carried-forward model"
        )
    elif best_state is not None:
        student.load_state_dict(best_state)
        student.to(DEVICE)
        iter_outcome = "fallback_valloss"
        _logger.warning(f"  Carry-forward: no prev_state and no accepted "
                        f"best_failure_state; falling back to val-loss best "
                        f"(val={best_val_loss:.4f})")
    else:
        iter_outcome = "keep_last"
        _logger.warning(f"  Carry-forward: NO checkpoint captured this iteration "
                        f"(no best_failure_state, no prev_state, no best_state); "
                        f"keeping last-epoch student")

    best_loss = float(best_val_loss)

    # ── 3b. Evaluate on validation set (vectorized) — shared COMMON_EVAL_SPECS
    val_specs_arr = COMMON_EVAL_SPECS
    evaluator.student = student
    failure_mask, metrics = evaluator.identify_failures(val_specs_arr, threshold=ERROR_THRESHOLD)
    failure_errors = metrics['errors'].max(axis=1)

    failed_specs_arr = val_specs_arr[failure_mask]
    failure_rate = failure_mask.mean()
    if 'boundary_failure' in metrics and 'is_boundary' in metrics:
        boundary_mask_eval = metrics['is_boundary']
        interior_mask_eval = ~boundary_mask_eval
        boundary_failure_rate = metrics['boundary_failure'].sum() / max(1, boundary_mask_eval.sum())
        interior_failure_rate = metrics['interior_failure'].sum() / max(1, interior_mask_eval.sum())
    else:
        boundary_failure_rate = failure_rate
        interior_failure_rate = failure_rate
    _logger.info(f"Validation failure rate: {failure_rate*100:.1f}%  "
                f"({failure_mask.sum()}/{len(val_specs_arr)} specs)")
    _logger.info(f"  boundary: {boundary_failure_rate*100:.1f}%  "
                f"interior: {interior_failure_rate*100:.1f}%")
    log_failure_breakdown(metrics, prefix="  val/")

    # Per-iteration summary line (mirrors kirchhoffnet for apples-to-apples).
    iter_delta = (prev_failure_rate - failure_rate) * 100
    if diverged_at is not None:
        iter_outcome += f" (divergence-abort@ep{diverged_at})"
    if n_nonfinite_batches > 0:
        iter_outcome += f" (skipped {n_nonfinite_batches} non-finite batches)"
    _logger.info(
        f"  Iter {dagger_iter+1} final={failure_rate*100:.2f}% "
        f"vs prev={prev_failure_rate*100:.2f}% "
        f"(delta {iter_delta:+.2f} pts) -> {iter_outcome}"
    )

    # ── 3c. Convergence check ───────────────────────────────────────
    dagger_history['iteration'].append(dagger_iter + 1)
    dagger_history['failure_rate'].append(failure_rate)
    dagger_history['boundary_failure_rate'].append(boundary_failure_rate)
    dagger_history['interior_failure_rate'].append(interior_failure_rate)
    dagger_history['dataset_size'].append(len(distillation_dataset))
    dagger_history['train_loss'].append(best_loss)

    avg_train_imit = sum(epoch_losses['imit']) / len(epoch_losses['imit']) if epoch_losses['imit'] else 0.0
    avg_train_spec = sum(epoch_losses['spec']) / len(epoch_losses['spec']) if epoch_losses['spec'] else 0.0
    avg_train_phys = sum(epoch_losses['phys']) / len(epoch_losses['phys']) if epoch_losses['phys'] else 0.0
    avg_train_invalid = sum(epoch_losses['invalid']) / len(epoch_losses['invalid']) if epoch_losses['invalid'] else 0.0
    avg_train_total = sum(epoch_losses['total']) / len(epoch_losses['total']) if epoch_losses['total'] else 0.0
    avg_train_manifold = sum(epoch_losses['manifold']) / len(epoch_losses['manifold']) if epoch_losses['manifold'] else 0.0

    loss_history['iteration'].append(dagger_iter + 1)
    loss_history['train_imit'].append(avg_train_imit)
    loss_history['train_spec'].append(avg_train_spec)
    loss_history['train_phys'].append(avg_train_phys)
    loss_history['train_invalid'].append(avg_train_invalid)
    loss_history['train_total'].append(avg_train_total)
    loss_history['train_manifold'].append(avg_train_manifold)
    loss_history['val_imit'].append(val_losses['imit'])
    loss_history['val_spec'].append(val_losses['spec'])
    loss_history['val_phys'].append(val_losses['phys'])
    loss_history['val_invalid'].append(val_losses['invalid'])
    loss_history['val_total'].append(val_losses['total'])
    loss_history['val_manifold'].append(val_losses['manifold'])

    _logger.info(f"  Iter {dagger_iter+1} loss summary: train_total={avg_train_total:.3f}(imit={avg_train_imit:.3f},spec={avg_train_spec:.3f},phys={avg_train_phys:.3f},invalid={avg_train_invalid:.3f},manifold={avg_train_manifold:.3f})  "
                 f"val_total={val_losses['total']:.3f}(imit={val_losses['imit']:.3f},spec={val_losses['spec']:.3f},phys={val_losses['phys']:.3f},invalid={val_losses['invalid']:.3f},manifold={val_losses['manifold']:.3f})")

    if failure_rate < CONVERGENCE_THRESHOLD:
        _logger.info(f"CONVERGENCE MET (failure_rate={failure_rate*100:.2f}% < "
                     f"{CONVERGENCE_THRESHOLD*100}%)")
        break

    # ── 3d. Skip teacher labeling if no failures ────────────────────
    if len(failed_specs_arr) == 0:
        _logger.info("No failures detected — skipping teacher labeling.")
        continue

    # ── 3e. Data imbalance check ────────────────────────────────────
    cap = int(len(distillation_dataset) * FAILURE_CAP_RATIO)
    if len(failed_specs_arr) > cap:
        top_k_idx = np.argsort(failure_errors[failure_mask])[-cap:]
        failed_specs_arr = failed_specs_arr[top_k_idx]
        _logger.info(f"  Capped failures to top {cap} highest-error cases")

    # ── 3f. Reset optimizer LR and apply decay (floor at 1e-4) ─────
    new_lr = LR_INITIAL * (0.5 ** max(0, dagger_iter - LR_DECAY_AFTER_ITER))
    new_lr = max(new_lr, 1e-4)
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr
    # Recreate cosine scheduler so it restarts from the reset LR (otherwise the
    # continuously-stepped scheduler overrides the manual reset on next step
    # and its phase wraps across iterations).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PER_ITER, eta_min=LR_FLOOR
    )
    _logger.info(f"  LR reset to {new_lr:.2e}; cosine scheduler restarted for next iteration")

    # ── 3g. Teacher labeling with k-NN empirical fallback ─────────────
    _logger.info(f"  Querying teacher for {len(failed_specs_arr)} failed specs ...")

    # Separate boundary and interior specs for tiered teacher budget
    boundary_mask = np.array([is_boundary_spec(s) for s in failed_specs_arr])
    boundary_specs = failed_specs_arr[boundary_mask]
    interior_specs = failed_specs_arr[~boundary_mask]

    new_labels = np.zeros((len(failed_specs_arr), 7))  # (N, 7) params
    label_source = np.zeros(len(failed_specs_arr), dtype=int)  # 0=unlabeled, 1=flow, 2=knn

    # Interior specs: standard flow labeling
    if len(interior_specs) > 0:
        _logger.info(f"  Interior specs ({len(interior_specs)}): using flow teacher")
        new_labels_int = teacher_labeler.label_batch(
            interior_specs,
            n_candidates=N_CANDIDATES_PER_SPEC,
            valid_threshold=0.9,
            top_k=1,
            verbose=False,
        )
        keep_int = filter_by_zig_validity(
            new_labels_int.squeeze(), zig_model, scaler_X, threshold=0.5, device=DEVICE
        )
        idx_int = np.where(~boundary_mask)[0]
        if keep_int.sum() > 0:
            new_labels[idx_int[keep_int]] = new_labels_int[keep_int]
            label_source[idx_int[keep_int]] = 1

    # Boundary specs: high-budget flow + empirical k-NN fallback
    if len(boundary_specs) > 0:
        _logger.info(f"  Boundary specs ({len(boundary_specs)}): high-budget flow + k-NN fallback")
        new_labels_bnd = teacher_labeler.label_batch(
            boundary_specs,
            n_candidates=10000,
            valid_threshold=0.5,
            top_k=1,
            verbose=False,
        )
        keep_bnd = filter_by_zig_validity(
            new_labels_bnd.squeeze(), zig_model, scaler_X, threshold=0.5, device=DEVICE
        )
        idx_bnd = np.where(boundary_mask)[0]
        if keep_bnd.sum() > 0:
            new_labels[idx_bnd[keep_bnd]] = new_labels_bnd[keep_bnd]
            label_source[idx_bnd[keep_bnd]] = 1

        still_failed_bnd = idx_bnd[~keep_bnd] if keep_bnd.sum() > 0 else idx_bnd
        if len(still_failed_bnd) > 0:
            _logger.info(f"  k-NN fallback for {len(still_failed_bnd)} boundary specs where flow failed")
            for i, fi in enumerate(still_failed_bnd):
                spec = failed_specs_arr[fi]
                fallback_params = empirical_fallback_label(spec, df, k=3)
                new_labels[fi] = fallback_params
                label_source[fi] = 2
            _logger.info(f"  k-NN fallback: {(label_source[still_failed_bnd] == 2).sum()}/{len(still_failed_bnd)} labeled")

    _logger.info(f"  Teacher labels: flow={int((label_source==1).sum())}, "
                 f"k-NN={int((label_source==2).sum())}, total={int((label_source>0).sum())}/{len(label_source)}")

    final_keep_mask = label_source > 0
    final_specs = failed_specs_arr[final_keep_mask]
    final_labels = new_labels[final_keep_mask]

    # ── 3h. Dataset aggregation ──────────────────────────────────────
    if len(final_specs) > 0:
        log_label_quality_summary("  DAgger labels", final_specs, final_labels, zig_model, scaler_X, scaler_y_p, device=DEVICE)
        distillation_dataset.append_samples(final_specs, final_labels)
        _logger.info(f"  Added {len(final_labels)} labeled failures.  Dataset now: "
                     f"{len(distillation_dataset)}")

_logger.info("DAgger training complete")

_logger.info("DAgger training complete - no Phase 2 fine-tuning (pure manifold-constrained distillation)")



# %% papermill={"duration": 0.068704, "end_time": "2026-07-01T23:09:02.784150+00:00", "exception": false, "start_time": "2026-07-01T23:09:02.715446+00:00", "status": "completed"}
def evaluate_student(student, df_eval, device, eye_scale_h, eye_scale_w, eye_scale_j,
                     scaler_y_p, scaler_X, zig_model):
    """Run final evaluation on a held-out set of specs."""
    evaluator = StudentEvaluator(
        student, scaler_X, zig_model,
        eye_scale_h, eye_scale_w, eye_scale_j, scaler_y_p, device
    )
    specs = df_eval[['power', 'stage_2_eye_max_height',
                     'stage_2_eye_max_width', 'stage_2_jitter']].values.astype(np.float32)
    specs = specs[:, [0, 3, 1, 2]]
    failure_mask, metrics = evaluator.identify_failures(specs, threshold=ERROR_THRESHOLD)
    failure_rate = failure_mask.mean()
    _logger.info(f"Test failure rate: {failure_rate*100:.2f}%  ({failure_mask.sum()}/{len(specs)} specs)")
    if 'boundary_failure' in metrics and 'is_boundary' in metrics:
        boundary_mask_eval = metrics['is_boundary']
        interior_mask_eval = ~boundary_mask_eval
        boundary_rate = metrics['boundary_failure'].sum() / max(1, boundary_mask_eval.sum())
        interior_rate = metrics['interior_failure'].sum() / max(1, interior_mask_eval.sum())
        _logger.info(f"  boundary: {boundary_rate*100:.2f}%  "
                     f"interior: {interior_rate*100:.2f}%")
    _logger.info(f"  Mean errors — power: {metrics['errors'][:,0].mean():.4f}, "
                 f"jitter: {metrics['errors'][:,1].mean():.4f}, "
                 f"height: {metrics['errors'][:,2].mean():.4f}, "
                 f"width: {metrics['errors'][:,3].mean():.4f}")
    return metrics



# %% papermill={"duration": 0.962692, "end_time": "2026-07-01T23:09:03.802976+00:00", "exception": false, "start_time": "2026-07-01T23:09:02.840284+00:00", "status": "completed"}
# %%
# DAgger loop has completed; plot failure-rate trajectory
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(dagger_history['iteration'], dagger_history['failure_rate'], 'r-o', label='overall')
if 'boundary_failure_rate' in dagger_history and dagger_history['boundary_failure_rate']:
    axes[0].plot(dagger_history['iteration'], dagger_history['boundary_failure_rate'],
                 'b--o', label='boundary', alpha=0.6)
    axes[0].plot(dagger_history['iteration'], dagger_history['interior_failure_rate'],
                 'g--o', label='interior', alpha=0.6)
axes[0].set_xlabel('DAgger Iteration')
axes[0].set_ylabel('Failure Rate')
axes[0].set_title('Validation Failure Rate vs Iteration')
axes[0].axhline(CONVERGENCE_THRESHOLD, color='g', linestyle='--',
                label=f'Convergence={CONVERGENCE_THRESHOLD}')
axes[0].legend()

axes[1].plot(dagger_history['iteration'], dagger_history['dataset_size'], 'b-o')
axes[1].set_xlabel('DAgger Iteration')
axes[1].set_ylabel('Dataset Size')
axes[1].set_title('Rolling Dataset Size')

axes[2].plot(dagger_history['iteration'], dagger_history['train_loss'], 'm-o')
axes[2].set_xlabel('DAgger Iteration')
axes[2].set_ylabel('Best Train Loss')
axes[2].set_title('Training Loss per Iteration')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dagger_curves.png'), dpi=150)
plt.show()
_logger.info("DAgger training curves saved.")

# =====================================================================
# Final evaluation on held-out test set
# =====================================================================
df_eval = df.sample(min(1000, len(df)), random_state=42)
eval_results = evaluate_student(
    student, df_eval, DEVICE,
    eye_scale_h, eye_scale_w, eye_scale_j,
    scaler_y_p, scaler_X, zig_model
)

# =====================================================================
# Save model and export ONNX
# =====================================================================
#os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = 'dagger_student_moe'
torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.pt'))
joblib.dump(scaler_X, os.path.join(OUTPUT_DIR, 'scaler_X.pkl'))
joblib.dump(flow_scaler_C, os.path.join(OUTPUT_DIR, 'flow_scaler_C.pkl'))
#import json as _json
#with open(os.path.join(OUTPUT_DIR, 'param_log_bounds.json'), 'w') as _f:
#    _json.dump({k: list(v) for k, v in PARAM_LOG_BOUNDS.items()}, _f, indent=2)
#with open(os.path.join(OUTPUT_DIR, 'dagger_history.json'), 'w') as _f:
#    _json.dump(dagger_history, _f, indent=2)
#with open(os.path.join(OUTPUT_DIR, 'loss_history.json'), 'w') as _f:
#    _json.dump(loss_history, _f, indent=2)
#print(f"DAgger student model saved to {OUTPUT_DIR}")
#
#import subprocess
#subprocess.run(["pip", "install", "-U", "onnxscript", "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#import torch.onnx
#dummy_input = torch.zeros(1, 4).to(DEVICE)
#onnx_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.onnx')
#torch.onnx.export(
#    student,
#    dummy_input,
#    onnx_path,
#    input_names=['specs'],
#    output_names=['logits'],
#    dynamic_axes={'specs': {0: 'batch'}, 'logits': {0: 'batch'}},
#    opset_version=13,
#)
#print(f"ONNX exported to {onnx_path}")

# =====================================================================
# Save test set predictions to CSV
# =====================================================================
student.eval()
test_specs = df_eval[['power', 'stage_2_jitter',
                       'stage_2_eye_max_height', 'stage_2_eye_max_width']].values.astype(np.float32)
test_specs_t = torch.from_numpy(test_specs).to(DEVICE)

pred_dict = student.predict(test_specs_t)
pred_params = np.stack([pred_dict[name].cpu().detach().numpy()
                        for name in PARAM_COLS], axis=1)

pred_specs_zig = eval_results['pred_specs']
target_specs_zig = eval_results['target_specs']

csv_data = {
    'power':        test_specs[:, 0],
    'jitter':       test_specs[:, 1],
    'height':       test_specs[:, 2],
    'width':        test_specs[:, 3],
    'pred_fW':      pred_params[:, 0],
    'pred_current': pred_params[:, 1],
    'pred_ind':     pred_params[:, 2],
    'pred_Rd':      pred_params[:, 3],
    'pred_Cs':      pred_params[:, 4],
    'pred_Rs':      pred_params[:, 5],
    'pred_VDD':     pred_params[:, 6],
    'zigi_power':   pred_specs_zig[:, 0],
    'zigi_jitter':  pred_specs_zig[:, 1],
    'zigi_height':  pred_specs_zig[:, 2],
    'zigi_width':   pred_specs_zig[:, 3],
    'target_power': target_specs_zig[:, 0],
    'target_jitter':target_specs_zig[:, 1],
    'target_height':target_specs_zig[:, 2],
    'target_width': target_specs_zig[:, 3],
}
df_csv = pd.DataFrame(csv_data)
csv_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_predictions.csv')
df_csv.to_csv(csv_path, index=False)
print(f"Test predictions CSV saved to {csv_path}")

print("\n=== DAgger Variant A Complete ===")
print(f"Outputs saved to: {OUTPUT_DIR}")


# ============================================================
# Demo: Student Predictions (Bounded MLP) for Specific Target Specs
# ============================================================
# Input: [power, jitter, height, width] — raw physical units
# Output: [fW, current, ind, Rd, Cs, Rs, VDD] — raw physical units

LABELS = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD']

target_specs = np.array([
    [0.01044, 37.91, 21.14, 62.43],
    [0.00709, 11.45, 33.33, 88.68],
    [0.01107, 8.79,  29.37, 91.69],
    [0.00607, 16.59, 60.29, 83.90],
    [0.00229, 29.61,  4.44, 70.82],
    [0.00235, 27.55, 20.23, 72.58],
])

print(f"\n{'='*80}")
print(f"{'Target Specs':^40} | {'Student Predicted Parameters (7 CTO parameters)':^38}")
print(f"{'Power':>8} {'Jitter':>8} {'Height':>8} {'Width':>8} | "
      f"{'fW':>9} {'current':>9} {'ind':>9} {'Rd':>9} {'Cs':>9} {'Rs':>9} {'VDD':>9}")
print(f"{'='*80}")

student.eval()
with torch.no_grad():
    for spec in target_specs:
        power, jitter, height, width = spec
        spec_tensor = torch.tensor([[power, jitter, height, width]], dtype=torch.float32).to(DEVICE)
        pred_dict = student.predict(spec_tensor)
        pred_values = [pred_dict[name].item() for name in LABELS]
        print(f"{power:>8.5f} {jitter:>8.3f} {height:>8.3f} {width:>8.3f} | " +
              " ".join(f"{v:>12.4e}" for v in pred_values))
print(f"{'='*80}")

print("\nInput scaling (BoundedMLP.scale_input):")
print(f"  power  : log10(raw) -> StandardScaler -> (log10 - mean) / scale")
print(f"  jitter : log10(raw) -> Q75           -> log10 / eye_scale_j")
print(f"  height : log10(raw) -> Q75           -> log10 / eye_scale_h")
print(f"  width  : log10(raw) -> Q75           -> log10 / eye_scale_w")
print(f"\nOutput (bounded log-space -> physical): 10^(log_lo + (log_hi - log_lo)*sigmoid)")
for name, (lo, hi) in PARAM_LOG_BOUNDS.items():
    print(f"  {name:>8}: [{10**lo:.4e}, {10**hi:.4e}]")
