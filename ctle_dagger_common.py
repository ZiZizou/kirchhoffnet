"""Shared infrastructure for CTLE DAgger training.

Used by:
  - generative-distillation-improved-dagger-nuance-mlp.py (RegimeAwareMoE)
  - generative-distillation-plain-mlp.py (PlainMLP single-head control)
  - dagger-mlp-distillation-kirchhoffnet.py (LocalKirchhoffStudentWrapper)

This module is import-side-effect free: heavy artifact loads and the DAgger loop
are exposed as :func:`setup` and :func:`run_dagger_training`. Importing symbols
(class/function/constant definitions) does not touch disk or run training.
"""

from __future__ import annotations

import json
import math
import os
import time
import logging
import warnings
from pathlib import Path
from typing import Optional, Callable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.neighbors import NearestNeighbors
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FMT = "%H:%M:%S"
_logger = logging.getLogger("ctle_dagger")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False


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


LOG_BOUNDS = [-15.0, 8]

DEFAULT_TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'
DEFAULT_DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'
DEFAULT_OUTPUT_DIR = '/home/annaik/Documents/dagger_output'

SPEC_RANGES = {
    'power': (0.0012, 0.012),
    'stage_2_eye_max_height': (0.0, 88.4),
    'stage_2_eye_max_width': (0.0, 98.5),
    'stage_2_jitter': (1.57, 100.0),
}

SPEC_INPUT_COLS = ['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']

PARAM_COLS = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD']

PARAM_LOG_BOUNDS = {
    'fW':   (np.log10(1e-6),  np.log10(10.0)),
    'current': (np.log10(5e-4), np.log10(2.5)),
    'ind':  (np.log10(1e-12), np.log10(3.0)),
    'Rd':   (np.log10(10),    np.log10(1500)),
    'Cs':   (np.log10(1e-15), np.log10(1e-9)),
    'Rs':   (np.log10(10),    np.log10(1500)),
    'VDD':  (np.log10(0.6),   np.log10(1.2)),
}

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

HPARAMS_DEFAULTS = {
    'DAGGER_ITERATIONS':     10,
    'EPOCHS_PER_ITER':       200,
    'BATCH_SIZE':            256,
    'ERROR_THRESHOLD':       0.10,
    'VALIDATION_SIZE':       2000,
    'BOUNDARY_RATIO':        0.50,
    'N_CANDIDATES_PER_SPEC': 3000,
    'N_CANDIDATES_BOUNDARY': 10000,
    'VALIDITY_THRESHOLD':    0.50,
    'DEGRADE_REL_THRESHOLD': 0.20,
    'MIN_DEGRADED_DIMS':     2,
    'LR_INITIAL':            1e-3,
    'LR_DECAY_AFTER_ITER':   3,
    'LR_FLOOR':              1e-4,
    'FAILURE_CAP_RATIO':     0.40,
    'CONVERGENCE_THRESHOLD': 0.02,
    'COMMON_EVAL_SIZE':      2000,
    'COMMON_EVAL_SEED':      24681012,
    'EARLYSTOP_EVAL_EVERY':  1,
    'EARLYSTOP_LOG_EVERY':   10,
    'EARLYSTOP_SKIP_EPOCHS': 5,
    'EARLYSTOP_PATIENCE_EPOCHS': 200,
    'MIN_FAILURE_IMPROVEMENT': 0.01,
    'DIVERGENCE_ABORT':      True,
    'DIVERGENCE_MARGIN':     0.20,
    'DIVERGENCE_CONSEC_EVALS': 5,
    'N_EMPIRICAL_SAMPLES':   20000,
    'N_CANDIDATES_INITIAL':  5000,
    'WEIGHT_DECAY':          1e-4,
    'LOSS_WEIGHT_EMPIRIC':   1.0,
    'HARD_BUFFER_WEIGHT':    10.0,
    'WARMUP_EPOCHS':         5,
    'ALPHA_SPEC':            1.5,
    'BETA_PHYS':             0.1,
    'GAMMA_MONO':            0.01,
    'ALPHA_INVALID':         1.5,
    'K_MANIFOLD':            5,
    'ALPHA_MANIFOLD':        0.1,
}

DATASET_CSV_FILES = [
    'dataset_log_part_1_may3.csv', 'dataset_log_part_1_may4.csv',
    'dataset_log_part_2_may3.csv', 'dataset_log_part_2_may4.csv',
    'dataset_log_part_3_may3.csv', 'dataset_log_part_3_may4.csv',
    'dataset_log_part_4_may3.csv', 'dataset_log_part_4_may4.csv',
    'dataset_log_part_5_may3.csv', 'dataset_log_part_5_may4.csv',
    'dataset_log_part_6_may3.csv', 'dataset_log_part_6_may4.csv',
    'dataset_log_may_7.csv',
    'dataset_log_march12.csv', 'dataset_log_may_18.csv',
]

COL_MAPPING = {
    'eye_maxHeight_norm Vout_2 56G': 'stage_2_eye_max_height',
    'eye_maxWidth_norm Vout_2 56G': 'stage_2_eye_max_width',
    'eye_p2pJitterAverage_norm stage 2': 'stage_2_jitter',
}


# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------
def plain_param_count(trunk_width: int, trunk_layers: int) -> int:
    """Exact trainable parameter count for :class:`PlainMLP`.

    trunk = 9W + (L-1)(W^2 + W)        (8 -> W first layer, then W -> W per layer, biases)
    head  = 7W + 7                      (single Linear(W -> 7))
    """
    if trunk_width < 1 or trunk_layers < 1:
        raise ValueError("trunk dimensions must be positive")
    trunk = 9 * trunk_width + (trunk_layers - 1) * (trunk_width ** 2 + trunk_width)
    head = 7 * trunk_width + 7
    return trunk + head


def moe_param_count(trunk_width: int, trunk_layers: int, num_experts: int) -> int:
    """Trainable param count for :class:`RegimeAwareMoE` (mirrors mlp_bayes_opt.py)."""
    if trunk_width < 1 or trunk_layers < 1 or num_experts < 1:
        raise ValueError("MoE dimensions must be positive")
    trunk = 9 * trunk_width + (trunk_layers - 1) * (trunk_width ** 2 + trunk_width)
    routes = 16 * num_experts
    experts = num_experts * (7 * trunk_width + 7)
    return trunk + routes + experts


def derive_plain_width(budget: int, trunk_layers: int) -> int:
    """Largest integer W >= 1 with :func:`plain_param_count` <= budget.

    plain(W,L) = (L-1)*W^2 + (L+15)*W + 7, so for L>1 the quadratic is
    a=L-1, b=L+15, c=7-budget. Previous b=16 was only correct for L=1.

    Returns -1 if no valid W exists for the requested layer count.
    """
    if trunk_layers < 1:
        raise ValueError("trunk_layers must be >= 1")
    a = trunk_layers - 1
    b = trunk_layers + 15
    c = 7 - budget
    if a == 0:
        h_exact = (budget - 7) / b
        h = max(1, round(h_exact))
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return -1
        h_exact = (-b + math.sqrt(disc)) / (2 * a)
        if h_exact <= 0:
            return -1
        h = max(1, round(h_exact))
    while h > 0 and plain_param_count(h, trunk_layers) > budget:
        h -= 1
    return h if h >= 1 else -1


# ---------------------------------------------------------------------------
# Output parameter transforms
# ---------------------------------------------------------------------------
def params_from_logits(logits, param_log_bounds=PARAM_LOG_BOUNDS):
    """Convert bounded logits to physical parameter values."""
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
        prob = (log_val - log_lo) / (log_hi - log_lo)
        prob = torch.clamp(prob, 0.0, 1.0)
        logits.append(torch.logit(prob.clamp(1e-6, 1 - 1e-6)))
    return torch.stack(logits, dim=-1)


# ---------------------------------------------------------------------------
# Teacher: Hybrid Hurdle ZIG model
# ---------------------------------------------------------------------------
def make_eye_head():
    """Small nonlinear head for positive magnitude prediction."""
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.SiLU(),
        nn.LayerNorm(32),
        nn.Dropout(0),
        nn.Linear(32, 1),
    )


class HybridHurdleModel(nn.Module):
    """Trained ZIG model: hybrid hurdle architecture.

    If PER_TARGET_HURDLE: three separate validity heads (logit per metric).
    Otherwise: one shared validity head. Positive regression heads are per-metric.
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
    """Frozen HybridHurdleModel wrapper for flow scoring.

    Returns regressor in flow's condition space: [power_s, jitter_s, height_s, width_s].
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


# ---------------------------------------------------------------------------
# Teacher: spline flow
# ---------------------------------------------------------------------------
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
    """Conditional Spline Flow matching trained ctle_conditional_flow.pt.

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
    """End-to-end inference for ConditionalSplineFlow with Q75 condition scaling.

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


def load_flow_inference(teacher_dir, device=None, per_target=False):
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
        per_target=per_target
    )
    return FlowInference(flow, scaler_X, flow_scaler_C, zig_wrapper, device)


class FlowTeacherLabeler:
    """High-level interface: target specs -> canonical params via spline flow + ZIG."""
    def __init__(self, teacher_dir, device, per_target=False):
        self.teacher_dir = teacher_dir
        self.device = device
        self.per_target = per_target
        self.fi = load_flow_inference(teacher_dir, device, per_target=per_target)
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
        return params


# ---------------------------------------------------------------------------
# Teacher: PlainMLP (single-head MLP control)
# ---------------------------------------------------------------------------
def _plain_mlp_arch_from_state(state):
    """Infer the structural PlainMLP options from a state dict.

    This is intentionally based on tensor shapes/key positions rather than on
    ``teacher_config.json``.  Older runs either did not write that file or
    wrote metadata that no longer matched the checkpoint.
    """
    linear_weights = sorted(
        (int(k.split('.')[1]), v) for k, v in state.items()
        if k.startswith('trunk.') and k.endswith('.weight') and getattr(v, 'ndim', 0) == 2
    )
    if not linear_weights:
        raise ValueError("PlainMLP checkpoint has no trunk Linear weights")
    trunk_layers = len(linear_weights)
    trunk_width = int(linear_weights[0][1].shape[0])
    input_dim = int(linear_weights[0][1].shape[1])
    if input_dim not in (4, 8):
        raise ValueError(f"Unsupported PlainMLP input dimension in checkpoint: {input_dim}")
    use_log_features = input_dim == 8

    # With the current Sequential layout, LayerNorm parameters are 1-D
    # weights at the indices immediately following each Linear layer.
    linear_indices = {idx for idx, _ in linear_weights}
    norm_indices = {
        int(k.split('.')[1]) for k, v in state.items()
        if k.startswith('trunk.') and k.endswith('.weight')
        and getattr(v, 'ndim', 0) == 1
    }
    use_layernorm = any((idx + 1) in norm_indices for idx in linear_indices)
    return trunk_width, trunk_layers, use_layernorm, use_log_features


def load_plain_mlp_teacher(ckpt_path, device=None, *, trunk_width=48, trunk_layers=3,
                             activation=None, use_layernorm=False, input_preprocessing="knet",
                             use_log_features=None):
    """Load a frozen PlainMLP teacher from a ``dagger_student_plain.pt`` checkpoint.

    Resolution order (first hit wins for architecture):
      1. Embedded ``{"config": {...}}`` inside the checkpoint (if ``torch.save({"state_dict":..., "config":...})`` was used).
      2. Sidecar ``teacher_config.json`` co-located with the checkpoint.
      3. Explicit kwargs / defaults.

    For ``schema_version >= 2`` sidecars/embedded configs the scaler constants
    (``scaler_p_*``, ``eye_scale_*``, ``input_log_min/max``) are auto-attached
    so the caller does **not** need to call ``teacher.attach_scaler(...)``.
    Legacy checkpoints (no config or v1) fall back to shape inference via
    :func:`_plain_mlp_arch_from_state` and still require manual ``attach_scaler``.

    One-liner for any ``output_dir`` produced by
    ``generative-distillation-plain-mlp.py`` (always writes a v2 sidecar, with
    BO-locked ``input_preprocessing == "knet"`` and ``use_log_features == False``)::

        from ctle_dagger_common import load_plain_mlp_teacher
        model = load_plain_mlp_teacher("path/to/dagger_student_plain.pt")
        model.eval()
        preds = model.predict(specs_tensor)  # knet 4-feature min-max log scaled, clipped [-4,4]
    """
    if device is None:
        try:
            device = _active_device()
        except RuntimeError:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = str(ckpt_path)
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                             "teacher_config.json")
    sidecar_cfg = None
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                sidecar_cfg = json.load(f)
        except Exception as e:
            _logger.warning(f"[MLP-TEACHER] Failed to read teacher_config.json ({cfg_path}): "
                            f"{e}; using CLI defaults")
            sidecar_cfg = None

    # Load checkpoint first so embedded config (if any) can take precedence.
    raw = torch.load(ckpt_path, map_location=device)
    embedded_cfg = None
    if isinstance(raw, dict) and "config" in raw and isinstance(raw["config"], dict) and "trunk_width" in raw["config"]:
        embedded_cfg = raw["config"]
        _logger.info(f"[MLP-TEACHER] found embedded config in checkpoint (schema v{embedded_cfg.get('schema_version', '?')})")

    # Architecture source: embedded > sidecar > kwargs
    arch_cfg = embedded_cfg if isinstance(embedded_cfg, dict) else sidecar_cfg
    if isinstance(arch_cfg, dict) and "trunk_width" in arch_cfg:
        try:
            trunk_width = int(arch_cfg.get("trunk_width", trunk_width))
            trunk_layers = int(arch_cfg.get("trunk_layers", trunk_layers))
            act_name = str(arch_cfg.get("activation", "silu")).lower()
            use_layernorm = bool(arch_cfg.get("use_layernorm", use_layernorm))
            input_preprocessing = str(arch_cfg.get("input_preprocessing", input_preprocessing))
            if use_log_features is None and "use_log_features" in arch_cfg:
                use_log_features = bool(arch_cfg["use_log_features"])
            activation = {"silu": nn.SiLU, "gelu": nn.GELU}.get(act_name, nn.SiLU)
            src = "embedded config" if arch_cfg is embedded_cfg else "teacher_config.json"
            _logger.info(f"[MLP-TEACHER] loaded {src}: W={trunk_width} L={trunk_layers} "
                         f"act={act_name} LN={use_layernorm} prep={input_preprocessing}")
        except Exception as e:
            _logger.warning(f"[MLP-TEACHER] Failed to apply arch from config: {e}; using CLI defaults")

    # Extract state_dict (supports raw OrderedDict and {"state_dict": ...} wrappers)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    inferred = _plain_mlp_arch_from_state(state)
    inferred_width, inferred_layers, inferred_ln, inferred_log = inferred
    requested = (int(trunk_width), int(trunk_layers), bool(use_layernorm),
                 (input_preprocessing == "q75") if use_log_features is None else bool(use_log_features))
    actual = (inferred_width, inferred_layers, inferred_ln, inferred_log)
    if requested != actual:
        _logger.warning(
            "[MLP-TEACHER] CLI/config architecture %s does not match checkpoint %s; "
            "using checkpoint architecture",
            requested, actual,
        )
    trunk_width, trunk_layers, use_layernorm, use_log_features = actual
    if activation is None:
        activation = nn.SiLU
    teacher = PlainMLP(
        trunk_width=trunk_width,
        trunk_layers=trunk_layers,
        activation=activation,
        use_layernorm=use_layernorm,
        use_log_features=use_log_features,
    )
    teacher.load_state_dict(state)
    teacher = teacher.to(device)

    # Auto-attach scaler from v2 config (embedded > sidecar) if available.
    # Legacy configs without scaler still require manual attach_scaler().
    v2_cfg = None
    if isinstance(embedded_cfg, dict) and int(embedded_cfg.get("schema_version", 0)) >= 2:
        v2_cfg = embedded_cfg
    elif isinstance(sidecar_cfg, dict) and int(sidecar_cfg.get("schema_version", 0)) >= 2:
        v2_cfg = sidecar_cfg
    if isinstance(v2_cfg, dict) and "scaler" in v2_cfg:
        try:
            scaler = v2_cfg.get("scaler") or {}
            inp_min = v2_cfg.get("input_log_min")
            inp_max = v2_cfg.get("input_log_max")
            # Normalise lists -> np arrays (or None)
            if inp_min is not None:
                inp_min = np.asarray(inp_min, dtype=np.float32)
            if inp_max is not None:
                inp_max = np.asarray(inp_max, dtype=np.float32)
            teacher.attach_scaler(
                scaler_p_scale=float(scaler.get("scaler_p_scale")),
                scaler_p_mean=float(scaler.get("scaler_p_mean")),
                eye_scale_j=float(scaler.get("eye_scale_j")),
                eye_scale_h=float(scaler.get("eye_scale_h")),
                eye_scale_w=float(scaler.get("eye_scale_w")),
                input_preprocessing=str(v2_cfg.get("input_preprocessing", input_preprocessing)),
                input_log_min=inp_min,
                input_log_max=inp_max,
            )
            _logger.info(f"[MLP-TEACHER] auto-attached scaler from v{v2_cfg.get('schema_version')} config "
                         f"(prep={teacher.input_preprocessing})")
        except Exception as e:
            _logger.warning(f"[MLP-TEACHER] Failed to auto-attach scaler from v2 config: {e}; "
                            "caller must call attach_scaler() manually")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    _logger.info(
        f"[MLP-TEACHER] Loaded PlainMLP teacher from {ckpt_path}: "
        f"W={trunk_width} L={trunk_layers} act={activation.__name__} "
        f"LN={use_layernorm} input_preprocessing={input_preprocessing}"
    )
    return teacher


class MlpTeacherLabeler:
    """Batched deterministic teacher: target specs -> canonical params via PlainMLP.

    The PlainMLP is a single forward pass per spec; no candidate sampling /
    ranking is involved (unlike :class:`FlowTeacherLabeler`). Labels are the
    predicted 7 physical params in :data:`PARAM_COLS` order.
    """
    def __init__(self, teacher: PlainMLP, device=None, batch_size=1024):
        self.teacher = teacher
        if device is not None:
            self.device = device
        else:
            try:
                self.device = _active_device()
            except RuntimeError:
                # Fall back to teacher's device if active_ctx not set
                try:
                    self.device = next(teacher.parameters()).device
                except StopIteration:
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = int(batch_size)

    @torch.no_grad()
    def label_batch(self, target_specs):
        """target_specs: np.ndarray shape (N, 4) in [power, jitter, height, width] order.
        Returns np.ndarray shape (N, 7) in :data:`PARAM_COLS` order, physical units.
        """
        specs = np.asarray(target_specs, dtype=np.float32)
        if specs.ndim == 1:
            specs = specs.reshape(1, -1)
        n = len(specs)
        out = np.zeros((n, 7), dtype=np.float64)
        param_names = list(self.teacher.param_log_bounds.keys())
        for s in range(0, n, self.batch_size):
            batch = torch.from_numpy(specs[s:s + self.batch_size]).to(self.device)
            preds = self.teacher.predict(batch)
            for j, name in enumerate(param_names):
                out[s:s + len(batch), j] = preds[name].detach().cpu().numpy().astype(np.float64)
        return out

    def label_single(self, target_spec):
        return self.label_batch(np.array(target_spec, dtype=np.float32).reshape(1, -1))[0]


def create_mlp_distillation_dataset(df, n_samples=10000, boundary_ratio=0.3,
                                     teacher_labeler: MlpTeacherLabeler | None = None,
                                     scaler_X=None, zig_model=None, device=None):
    """Build a distillation dataset with frozen PlainMLP teacher labels.

    Mirrors :func:`create_flow_distillation_dataset` spec sampling (empirical
    spec noise) but labels via a single deterministic MLP forward (no
    candidate budget, no flow sampling). ZIG validity @ 0.5 filters invalid
    labels; rows that fail validity are replaced by empirical k-NN fallback.

    scaler_X/zig_model/device default to _active_ctx when not passed, so
    standalone scripts that already hold globals can pass them explicitly
    without needing setup().
    """
    if teacher_labeler is None:
        raise RuntimeError("create_mlp_distillation_dataset requires a MlpTeacherLabeler")
    targets = sample_target_specs(df, n_samples, boundary_ratio)
    noisy_specs_batch = []
    for t in targets:
        noisy_power = t['power'] * np.random.uniform(0.95, 1.05)
        noisy_height = t['height'] * np.random.uniform(0.9, 1.1) if t['height'] > 1 else np.random.uniform(0, 2)
        noisy_width = t['width'] * np.random.uniform(0.9, 1.1) if t['width'] > 1 else np.random.uniform(0, 2)
        noisy_jitter = np.clip(t['jitter'] * np.random.uniform(0.95, 1.05), 1.57, 100)
        noisy_specs_batch.append([noisy_power, noisy_jitter, noisy_height, noisy_width])
    noisy_specs_batch = np.array(noisy_specs_batch, dtype=np.float64)
    canonical_params_batch = teacher_labeler.label_batch(noisy_specs_batch)
    if scaler_X is None or zig_model is None or device is None:
        try:
            _sX, _sY = _active_scalers()
            _zm = _active_zig()
            _dev = _active_device()
        except RuntimeError:
            _sX = _zm = _dev = None
        if scaler_X is None:
            scaler_X = _sX
        if zig_model is None:
            zig_model = _zm
        if device is None:
            device = _dev
    if scaler_X is None or zig_model is None or device is None:
        raise RuntimeError("create_mlp_distillation_dataset requires scaler_X/zig_model/device — "
                           "pass explicitly or call setup()/_set_active_context first")
    DEVICE_local = device
    keep_valid = filter_by_zig_validity(
        canonical_params_batch, zig_model, scaler_X, threshold=0.5, device=DEVICE_local
    )
    n_invalid = int((~keep_valid).sum())
    if n_invalid > 0:
        _logger.info(f"MLP teacher produced {n_invalid}/{len(noisy_specs_batch)} low-validity labels; "
                     f"replacing with empirical k-NN fallback")
        for i in np.where(~keep_valid)[0]:
            canonical_params_batch[i] = empirical_fallback_label(noisy_specs_batch[i], df, k=3)
    data_list = []
    for i in range(len(targets)):
        data_list.append({
            'power': float(noisy_specs_batch[i, 0]),
            'height': float(noisy_specs_batch[i, 2]),
            'width': float(noisy_specs_batch[i, 3]),
            'jitter': float(noisy_specs_batch[i, 1]),
            'params': canonical_params_batch[i],
        })
    _logger.info(f"MLP distillation dataset: {len(data_list)} samples ready "
                 f"(noisy inputs -> frozen MLP labels)")
    return data_list


# ---------------------------------------------------------------------------
# Active context used by helper functions that need scalers/ZIG/DEVICE
# ---------------------------------------------------------------------------
_active_ctx: dict = {}


def _set_active_context(ctx: dict):
    global _active_ctx
    _active_ctx = ctx


def _active_device():
    if 'DEVICE' not in _active_ctx:
        raise RuntimeError("_active_ctx['DEVICE'] not set — call setup(args) before helpers that need DEVICE.")
    return _active_ctx['DEVICE']


def _active_zig():
    if 'zig_model' not in _active_ctx:
        raise RuntimeError("_active_ctx['zig_model'] not set — call setup(args) before helpers that need ZIG.")
    return _active_ctx['zig_model']


def _active_scalers():
    if 'scaler_X' not in _active_ctx or 'scaler_y_p' not in _active_ctx:
        raise RuntimeError("_active_ctx scalers not set — call setup(args) before helpers that need scalers.")
    return _active_ctx['scaler_X'], _active_ctx['scaler_y_p']


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------
class RegimeAwareMoE(nn.Module):
    """Regime-Aware Mixture of Experts for CTLE inverse design.

    Input:  [power, jitter, height, width] (4 dims, raw physical units)
    Output: logits for 7 params -> bounded via sigmoid + scale -> 10^value

    Architecture:
        - Shared trunk: L layers of Linear(W->W)+Act
        - Gating: input-driven Linear(trunk_input_dim->E, bias=False) + softmax
        - E Localized experts: each Linear(W->7)
        - Output: weighted combination of expert logits -> bounded sigmoid output

    Call :meth:`attach_scaler` (via :func:`setup`) before :meth:`forward`.
    """
    def __init__(self, trunk_width=44, trunk_layers=3,
                 num_experts=3, param_log_bounds=PARAM_LOG_BOUNDS,
                 activation=nn.SiLU, use_log_features=True,
                 scaler_p_scale=None, scaler_p_mean=None,
                 eye_scale_j=None, eye_scale_h=None, eye_scale_w=None):
        super().__init__()
        self.trunk_width = trunk_width
        self.trunk_layers = trunk_layers
        self.num_experts = num_experts
        self.param_log_bounds = param_log_bounds
        self.activation = activation
        self.use_log_features = use_log_features
        input_dim = 4
        trunk_input_dim = input_dim * 2 if use_log_features else input_dim
        trunk_dims = [trunk_input_dim] + [trunk_width] * trunk_layers
        trunk_layers_list = []
        for i in range(trunk_layers):
            trunk_layers_list.append(nn.Linear(trunk_dims[i], trunk_dims[i+1]))
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
        self.scaler_p_scale = scaler_p_scale
        self.scaler_p_mean = scaler_p_mean
        self._eye_scale_j = float(eye_scale_j) if eye_scale_j is not None else None
        self._eye_scale_h = float(eye_scale_h) if eye_scale_h is not None else None
        self._eye_scale_w = float(eye_scale_w) if eye_scale_w is not None else None
        self.input_preprocessing = 'knet'
        self.input_log_min = None
        self.input_log_max = None

    def attach_scaler(self, scaler_p_scale, scaler_p_mean,
                      eye_scale_j, eye_scale_h, eye_scale_w,
                      input_preprocessing='knet', input_log_min=None,
                      input_log_max=None):
        self.scaler_p_scale = float(scaler_p_scale)
        self.scaler_p_mean = float(scaler_p_mean)
        self._eye_scale_j = float(eye_scale_j)
        self._eye_scale_h = float(eye_scale_h)
        self._eye_scale_w = float(eye_scale_w)
        self.input_preprocessing = str(input_preprocessing)
        self.input_log_min = None if input_log_min is None else np.asarray(input_log_min, dtype=np.float32)
        self.input_log_max = None if input_log_max is None else np.asarray(input_log_max, dtype=np.float32)
        if self.input_preprocessing == 'knet' and (self.input_log_min is None or self.input_log_max is None):
            raise ValueError("KNet preprocessing requires input_log_min/input_log_max")

    def _require_scaler(self):
        if (self.scaler_p_scale is None or self.scaler_p_mean is None
                or self._eye_scale_j is None or self._eye_scale_h is None
                or self._eye_scale_w is None):
            raise RuntimeError("RegimeAwareMoE: scaler constants not attached. "
                               "Call attach_scaler() before forward().")

    def scale_input(self, x):
        self._require_scaler()
        power_log = torch.log10(x[..., 0].clamp(min=1e-12))
        if self.input_preprocessing == 'knet':
            lo = torch.as_tensor(self.input_log_min, dtype=x.dtype, device=x.device)
            hi = torch.as_tensor(self.input_log_max, dtype=x.dtype, device=x.device)
            logs = torch.stack([power_log, torch.log10(x[..., 1].clamp(min=1e-12)),
                                torch.log10(x[..., 2].clamp(min=1e-12)),
                                torch.log10(x[..., 3].clamp(min=1e-12))], dim=-1)
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
    """Compact MLP with bounded log-space outputs for CTLE inverse design (legacy).

    Kept for backward compatibility with prior experiments; new code should use
    :class:`RegimeAwareMoE` or :class:`PlainMLP`, which attach scaler constants
    explicitly via :meth:`attach_scaler`.
    """
    def __init__(self, hidden_dim=128, num_layers=4,
                 input_dim=4, output_dim=7,
                 param_log_bounds=PARAM_LOG_BOUNDS,
                 use_per_output_heads=False):
        super().__init__()
        self.param_log_bounds = param_log_bounds
        self.use_per_output_heads = use_per_output_heads
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
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


class PlainMLP(nn.Module):
    """Single-head MLP control for the CTLE MoE experiment.

    Mirrors :class:`RegimeAwareMoE` trunk (SiLU, no LN by default) with a single
    ``Linear(W -> 7)`` output head. No gating, no regime classifier, no
    auxiliary regime cross-entropy loss.

    The class default ``input_preprocessing='knet'`` (4 log/min-max features
    clipped to ``[-4, 4]``); ``use_log_features=True`` only takes effect when
    the harness explicitly switches to ``q75`` preprocessing. The
    ``generative-distillation-plain-mlp.py`` harness forces
    ``use_log_features=False`` whenever preprocessing is ``knet`` (the BO
    controller's locked default).

    Parameters
    ----------
    trunk_width, trunk_layers, activation, use_layernorm, use_log_features
        Defaults match the current best MoE shape (``width=45, layers=3, SiLU,
        no LN, log_features=True``) for an apples-to-apples "same backbone"
        comparison under the legacy Q75 path. Under the knet path
        ``use_log_features`` is forced to False (4->W first layer) regardless
        of the constructor default. ``--param-budget`` can derive
        ``trunk_width`` automatically via :func:`derive_plain_width`.
    """
    def __init__(self, trunk_width=45, trunk_layers=3,
                 activation=nn.SiLU, use_layernorm=False, use_log_features=True,
                 param_log_bounds=PARAM_LOG_BOUNDS,
                 scaler_p_scale=None, scaler_p_mean=None,
                 eye_scale_j=None, eye_scale_h=None, eye_scale_w=None):
        super().__init__()
        self.trunk_width = trunk_width
        self.trunk_layers = trunk_layers
        self.activation = activation
        self.use_layernorm = use_layernorm
        self.use_log_features = use_log_features
        self.param_log_bounds = param_log_bounds

        input_dim = 4
        trunk_input_dim = input_dim * 2 if use_log_features else input_dim
        trunk_dims = [trunk_input_dim] + [trunk_width] * trunk_layers
        trunk_layers_list = []
        for i in range(trunk_layers):
            trunk_layers_list.append(nn.Linear(trunk_dims[i], trunk_dims[i+1]))
            if use_layernorm:
                trunk_layers_list.append(nn.LayerNorm(trunk_dims[i+1]))
            trunk_layers_list.append(activation())
        self.trunk = nn.Sequential(*trunk_layers_list)

        self.head = nn.Linear(trunk_width, 7)

        self.log_lo = nn.Parameter(torch.zeros(7), requires_grad=False)
        self.log_hi = nn.Parameter(torch.zeros(7), requires_grad=False)
        for i, (name, (lo, hi)) in enumerate(param_log_bounds.items()):
            self.log_lo.data[i] = lo
            self.log_hi.data[i] = hi

        self.scaler_p_scale = scaler_p_scale
        self.scaler_p_mean = scaler_p_mean
        self._eye_scale_j = float(eye_scale_j) if eye_scale_j is not None else None
        self._eye_scale_h = float(eye_scale_h) if eye_scale_h is not None else None
        self._eye_scale_w = float(eye_scale_w) if eye_scale_w is not None else None
        self.input_preprocessing = 'knet'
        self.input_log_min = None
        self.input_log_max = None

    def attach_scaler(self, scaler_p_scale, scaler_p_mean,
                      eye_scale_j, eye_scale_h, eye_scale_w,
                      input_preprocessing='knet', input_log_min=None,
                      input_log_max=None):
        """Inject scaler constants required by :meth:`scale_input`.

        For ``input_preprocessing='knet'`` the :meth:`scale_input` knet
        branch (4 log/min-max features clipped to [-4, 4]) requires
        ``input_log_min`` and ``input_log_max``. For ``'q75'`` it
        additionally requires the Q75/StandardScaler ``scaler_p_*`` and
        ``eye_scale_*`` constants. ``setup()``/``load_plain_mlp_teacher()``
        always call this before any forward pass.
        """
        self.scaler_p_scale = float(scaler_p_scale)
        self.scaler_p_mean = float(scaler_p_mean)
        self._eye_scale_j = float(eye_scale_j)
        self._eye_scale_h = float(eye_scale_h)
        self._eye_scale_w = float(eye_scale_w)
        self.input_preprocessing = str(input_preprocessing)
        self.input_log_min = None if input_log_min is None else np.asarray(input_log_min, dtype=np.float32)
        self.input_log_max = None if input_log_max is None else np.asarray(input_log_max, dtype=np.float32)
        if self.input_preprocessing == 'knet' and (self.input_log_min is None or self.input_log_max is None):
            raise ValueError("KNet preprocessing requires input_log_min/input_log_max")

    def _require_scaler(self):
        if (self.scaler_p_scale is None or self.scaler_p_mean is None
                or self._eye_scale_j is None or self._eye_scale_h is None
                or self._eye_scale_w is None):
            raise RuntimeError("PlainMLP: scaler constants not attached. "
                               "Call attach_scaler() before forward().")

    def scale_input(self, x):
        self._require_scaler()
        power_log = torch.log10(x[..., 0].clamp(min=1e-12))
        if self.input_preprocessing == 'knet':
            lo = torch.as_tensor(self.input_log_min, dtype=x.dtype, device=x.device)
            hi = torch.as_tensor(self.input_log_max, dtype=x.dtype, device=x.device)
            logs = torch.stack([power_log, torch.log10(x[..., 1].clamp(min=1e-12)),
                                torch.log10(x[..., 2].clamp(min=1e-12)),
                                torch.log10(x[..., 3].clamp(min=1e-12))], dim=-1)
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

    def forward(self, x):
        """Forward pass returning unbounded logits (B, 7)."""
        x_s = self.scale_input(x)
        h = self.trunk(x_s)
        return self.head(h)

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
                f"use_layernorm={self.use_layernorm}, use_log_features={self.use_log_features}")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def huber_loss(pred, target, delta=1.0):
    """Huber loss - robust to outliers."""
    diff = torch.abs(pred - target)
    return torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))


class RegimeAwareLoss(nn.Module):
    """Loss function for HybridHurdleModel + Q75 condition scaling."""
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
        clamped_params = torch.clamp(canonical_params, log_lo, log_hi)
        L_imit = huber_loss(bounded_log, clamped_params, delta=0.5).mean()
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
        lambda_zig = torch.where(p_valid > 0.6, torch.ones_like(p_valid), torch.zeros_like(p_valid))
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
        per_sample_forward = err_h**2 + err_w**2 + err_j**2 + err_p**2
        lambda_norm = lambda_zig.sum().clamp(min=1.0)
        L_forward_zig = (per_sample_forward * lambda_zig).sum() / lambda_norm
        L_spec = L_forward_zig
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
        physical = {name: torch.pow(10, bounded_log[:, i]) for i, name in enumerate(PARAM_COLS)}
        rc_rd_cs = 1.0 / (physical['Rd'] * physical['Cs'] + 1e-12)
        rc_rs_cs = 1.0 / (physical['Rs'] * physical['Cs'] + 1e-12)
        L_phys_rc = torch.relu(torch.log10(rc_rd_cs + 1e-12) - 12).clamp(max=3) ** 2 + \
                    torch.relu(torch.log10(rc_rs_cs + 1e-12) - 12).clamp(max=3) ** 2
        power_approx = 4 * physical['VDD'] * physical['current']
        L_phys_power = torch.relu(target_p / (power_approx + 1e-12) - 5).clamp(max=2) ** 2 + \
                       torch.relu(power_approx / (target_p + 1e-12) - 5).clamp(max=2) ** 2
        L_phys = (L_phys_rc + L_phys_power).mean() * self.beta_phys
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
                'imit': 0.0, 'spec': 0.0, 'phys': 0.0, 'invalid': 0.0, 'manifold': 0.0,
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


class RampedRegimeLoss(nn.Module):
    """Warmup wrapper that scales ZIG/physics/manifold by r = epoch / warmup_epochs."""
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


# ---------------------------------------------------------------------------
# Spec sampling / dataset creators
# ---------------------------------------------------------------------------
def sample_target_specs(df, n_samples, boundary_ratio=0.3):
    """Sample target specs with priority on boundary cases."""
    n_boundary = int(n_samples * boundary_ratio)
    n_interior = n_samples - n_boundary
    targets = []
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


def create_flow_distillation_dataset(df, n_samples=10000, n_candidates=5000,
                                    boundary_ratio=0.3, teacher_labeler=None):
    if teacher_labeler is None:
        teacher_labeler = FlowTeacherLabeler(DEFAULT_TEACHER_DIR, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    targets = sample_target_specs(df, n_samples, boundary_ratio)
    noisy_specs_batch = []
    for t in targets:
        noisy_power = t['power'] * np.random.uniform(0.95, 1.05)
        noisy_height = t['height'] * np.random.uniform(0.9, 1.1) if t['height'] > 1 else np.random.uniform(0, 2)
        noisy_width = t['width'] * np.random.uniform(0.9, 1.1) if t['width'] > 1 else np.random.uniform(0, 2)
        noisy_jitter = np.clip(t['jitter'] * np.random.uniform(0.95, 1.05), 1.57, 100)
        noisy_specs_batch.append([noisy_power, noisy_jitter, noisy_height, noisy_width])
    noisy_specs_batch = np.array(noisy_specs_batch)
    canonical_params_batch = teacher_labeler.label_batch(
        noisy_specs_batch, n_candidates=n_candidates,
        valid_threshold=0.9, top_k=1, verbose=True
    )
    scaler_X, scaler_y_p = _active_scalers()
    zig_model = _active_zig()
    DEVICE_local = _active_device()
    keep_valid = filter_by_zig_validity(
        canonical_params_batch, zig_model, scaler_X, threshold=0.5, device=DEVICE_local
    )
    n_invalid = int((~keep_valid).sum())
    if n_invalid > 0:
        _logger.info(f"Flow teacher produced {n_invalid}/{len(noisy_specs_batch)} low-validity initial labels; replacing with empirical k-NN fallback")
        for i in np.where(~keep_valid)[0]:
            canonical_params_batch[i] = empirical_fallback_label(noisy_specs_batch[i], df, k=3)
    data_list = []
    for i, _ in enumerate(targets):
        data_list.append({
            'power': noisy_specs_batch[i, 0],
            'height': noisy_specs_batch[i, 2],
            'width': noisy_specs_batch[i, 3],
            'jitter': noisy_specs_batch[i, 1],
            'params': canonical_params_batch[i],
        })
    _logger.info(f"Flow distillation dataset: {len(data_list)} samples ready (noisy inputs -> teacher labels for those inputs)")
    return data_list


# ---------------------------------------------------------------------------
# Canonical Phase-A dataset builder (plan canonical-ctle-unify)
# ---------------------------------------------------------------------------
# Frozen min-power tie-break rule:
#   1. Flow samples `top_k=8` candidate param vectors per spec.
#   2. Keep only candidates with surrogate-validity >= 0.9, or (relaxed fallback)
#      >= 0.5 if no candidate clears the strict gate.
#   3. Among the surviving ZIG-valid candidates, pick the one whose predicted
#      forward power `4*VDD*I` is smallest (min-power tie-break; min-power is
#      the industrial objective for CTLE).
# The rule is chosen on physical grounds (lowest-power viable design) and is
# frozen before either Phase-A BO run; never tuned per student.
# Spec: docs/specs/canonical-ctle-bench.md (canonical-ctle-unify).
PHASE_A_TIE_BREAK = "min_power"
PHASE_A_TOP_K = 8
PHASE_A_VALID_THRESHOLD = 0.9
PHASE_A_RELAXED_THRESHOLD = 0.5

_PHASE_A_FEATURE_ORDER = ("power", "stage_2_jitter", "stage_2_eye_max_height", "stage_2_eye_max_width")


def _predict_forward_power(params_arr: np.ndarray) -> np.ndarray:
    """Estimate surrogate forward power (W) for min-power tie-break.

    Uses the same 4*VDD*I approximation the flow teacher's physics-penalty
    term relies on (`ctle_dagger_common.py` sample_and_rank).  Indices are
    PARAM_COLS order: fW(0), current(1), ind(2), Rd(3), Cs(4), Rs(5), VDD(6).
    """
    if params_arr.size == 0 or params_arr.shape[-1] < 7:
        return np.full(params_arr.shape[:-1], np.inf, dtype=np.float64)
    return 4.0 * params_arr[..., 6] * params_arr[..., 1]


def _canonical_dataset_fingerprint(seed: int, teacher_identity: str,
                                  n_samples: int, boundary_ratio: float,
                                  top_k: int, valid_threshold: float,
                                  relaxed_threshold: float, tie_break: str) -> str:
    payload = {
        "seed": int(seed),
        "teacher_identity": str(teacher_identity),
        "n_samples": int(n_samples),
        "boundary_ratio": float(boundary_ratio),
        "top_k": int(top_k),
        "valid_threshold": float(valid_threshold),
        "relaxed_threshold": float(relaxed_threshold),
        "tie_break": str(tie_break),
        "feature_order": list(_PHASE_A_FEATURE_ORDER),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def create_canonical_flow_dataset(
    df,
    teacher_labeler=None,
    n_samples: int = 20000,
    boundary_ratio: float = 0.5,
    seed: int = 100,
    n_candidates: int = 5000,
    top_k: int = PHASE_A_TOP_K,
    valid_threshold: float = PHASE_A_VALID_THRESHOLD,
    relaxed_threshold: float = PHASE_A_RELAXED_THRESHOLD,
    output_path: str | os.PathLike | None = None,
    zig_model=None,
    scaler_X=None,
    device=None,
):
    """Canonical Phase-A dataset: flow-labelled, min-power tie-break, frozen.

    Returns a dict with the same fields DistillationDataset indexes, plus a
    sha256 fingerprint and the per-spec predicted forward power (for auditing).
    If ``output_path`` is given, also writes a compressed .npz with the same
    schema (specs, params_log, predicted_power, fingerprint, train_idx/val_idx/test_idx)
    that both BO drivers and all harnesses load identically.

    The dataset is **static**: every trial sees byte-identical labels and
    splits.  No DAgger relabeling is performed; loss is Huber-on-log-params
    only (the rest of the RegimeAwareLoss machinery is held out so that
    Phase A isolates model capacity + optimization from surrogate-gradient
    credit assignment).

    Plan phase-a-fixes, bug 2: ``zig_model``/``scaler_X``/``device`` are
    optional explicit kwargs, each falling back independently to
    ``_active_zig()`` / ``_active_scalers()[0]`` / ``_active_device()`` when
    None (same per-arg pattern as :func:`create_mlp_distillation_dataset`).
    Harnesses that wire their own locals (nuance, MoE) pass all three and
    never touch the module-level ``_active_ctx``; existing ``setup()``-based
    callers (plain-mlp) keep working unchanged.
    """
    if teacher_labeler is None:
        teacher_labeler = FlowTeacherLabeler(
            DEFAULT_TEACHER_DIR,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    specs_arr = sample_validation_specs(
        df, n_samples=n_samples, boundary_ratio=boundary_ratio, seed=seed
    )
    _logger.info(
        f"[phaseA] sampling {len(specs_arr)} specs "
        f"(boundary_ratio={boundary_ratio}, seed={seed}); flow top_k={top_k}"
    )

    # Use :meth:`label_single` per spec so the top_k candidates are
    # returned as (top_k, 7) without relying on :meth:`label_batch`, which
    # is currently hard-coded for top_k=1 and would route a 3D candidate
    # tensor through a 2D-only StandardScaler inside filter_by_zig_validity.
    # For the canonical builder this is the deterministic, self-contained
    # path; the existing DAgger ``label_batch(top_k=1)`` calls are unchanged.
    labels_topk = np.zeros((len(specs_arr), top_k, 7), dtype=np.float64)
    for i in range(len(specs_arr)):
        top = teacher_labeler.label_single(
            specs_arr[i], n_candidates=n_candidates,
            valid_threshold=valid_threshold, top_k=top_k,
        )
        labels_topk[i] = np.asarray(top, dtype=np.float64).reshape(top_k, 7)

    zig_model = zig_model if zig_model is not None else _active_zig()
    scaler_X = scaler_X if scaler_X is not None else _active_scalers()[0]
    device = device if device is not None else _active_device()
    n_invalid = 0
    n_relaxed = 0
    chosen = np.zeros((len(specs_arr), 7), dtype=np.float64)
    predicted_power = np.zeros(len(specs_arr), dtype=np.float64)
    for i in range(len(specs_arr)):
        cands = labels_topk[i]
        keep = filter_by_zig_validity(
            cands, zig_model, scaler_X, threshold=valid_threshold, device=device,
        )
        if not keep.any():
            keep = filter_by_zig_validity(
                cands, zig_model, scaler_X, threshold=relaxed_threshold, device=device,
            )
            n_relaxed += int(keep.sum())
        if not keep.any():
            chosen[i] = cands[0]
            n_invalid += 1
            predicted_power[i] = _predict_forward_power(chosen[i:i + 1])[0]
            continue
        cands_valid = cands[keep]
        power = _predict_forward_power(cands_valid)
        order = np.argsort(power)
        chosen[i] = cands_valid[order[0]]
        predicted_power[i] = float(power[order[0]])

    _logger.info(
        f"[phaseA] tie-break={PHASE_A_TIE_BREAK}: "
        f"{n_invalid}/{len(specs_arr)} fell back to top-1 flow pick "
        f"({n_relaxed} candidates needed the {relaxed_threshold} relaxed gate)"
    )

    teacher_identity = str(getattr(teacher_labeler, "teacher_dir", DEFAULT_TEACHER_DIR))
    fingerprint = _canonical_dataset_fingerprint(
        seed=seed,
        teacher_identity=teacher_identity,
        n_samples=n_samples,
        boundary_ratio=boundary_ratio,
        top_k=top_k,
        valid_threshold=valid_threshold,
        relaxed_threshold=relaxed_threshold,
        tie_break=PHASE_A_TIE_BREAK,
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(specs_arr))
    n_train = int(0.8 * len(specs_arr))
    n_val = int(0.1 * len(specs_arr))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    # knet log/min-max input bounds over the canonical specs (+5% pad), mirroring
    # the fixed-distillation harness.  Students with knet input preprocessing
    # (KNet, PlainMLPStudent) require these via attach_scaler; storing them in
    # the .npz keeps every trial byte-identical without recomputation.
    spec_logs = np.log10(np.clip(specs_arr, 1e-12, None))
    lo, hi = spec_logs.min(axis=0), spec_logs.max(axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-8)
    payload = {
        "specs": specs_arr.astype(np.float32),
        "params": chosen.astype(np.float32),
        "predicted_power": predicted_power.astype(np.float32),
        "fingerprint": np.array(fingerprint),
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "tie_break": np.array(PHASE_A_TIE_BREAK),
        "input_log_min": (lo - pad).astype(np.float32),
        "input_log_max": (hi + pad).astype(np.float32),
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **payload)
        _logger.info(f"[phaseA] wrote canonical dataset -> {out} "
                     f"(fingerprint={fingerprint[:12]}...)")
    return payload


def load_canonical_flow_dataset(path: str | os.PathLike) -> dict:
    """Load a canonical Phase-A .npz produced by :func:`create_canonical_flow_dataset`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Canonical dataset not found: {p}")
    npz = np.load(p, allow_pickle=False)
    required = {"specs", "params", "predicted_power", "fingerprint",
                "train_idx", "val_idx", "test_idx",
                "input_log_min", "input_log_max"}
    missing = required - set(npz.files)
    if missing:
        raise RuntimeError(f"Canonical dataset {p} missing fields: {sorted(missing)}")
    return {key: npz[key] for key in npz.files}


def create_blended_distillation_dataset(df, teacher_labeler=None, n_samples=5000, n_candidates=5000,
                                        boundary_ratio=0.3, tolerance=0.05):
    """Creates a dataset where labels are either empirical or flow-generated."""
    if teacher_labeler is None:
        teacher_labeler = FlowTeacherLabeler(DEFAULT_TEACHER_DIR, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    targets = sample_target_specs(df, n_samples, boundary_ratio)
    emp_specs = df[['power', 'stage_2_jitter', 'stage_2_eye_max_height',
                    'stage_2_eye_max_width']].values.copy()
    emp_specs_log = np.log10(emp_specs + 1e-12)
    nn = NearestNeighbors(n_neighbors=1, metric='chebyshev')
    nn.fit(emp_specs_log)
    data_list = []
    for t in targets:
        noisy_power = t['power'] * np.random.uniform(0.95, 1.05)
        noisy_height = t['height'] * np.random.uniform(0.9, 1.1) if t['height'] > 1 else np.random.uniform(0, 2)
        noisy_width = t['width'] * np.random.uniform(0.9, 1.1) if t['width'] > 1 else np.random.uniform(0, 2)
        noisy_jitter = np.clip(t['jitter'] * np.random.uniform(0.95, 1.05), 1.57, 100)
        noisy_specs = np.array([noisy_power, noisy_jitter, noisy_height, noisy_width])
        noisy_specs_log = np.log10(noisy_specs + 1e-12)
        distances, indices = nn.kneighbors(noisy_specs_log.reshape(1, -1))
        if distances[0, 0] < tolerance:
            row = df.iloc[indices[0, 0]]
            params = row[PARAM_COLS].values
        else:
            params = teacher_labeler.label_single(noisy_specs, n_candidates=n_candidates,
                                                  valid_threshold=0.9, top_k=1)
            params = params[0]
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DistillationDataset(Dataset):
    """Rolling distillation dataset that grows via DAgger iterations.

    Tracks a 'hard buffer' - samples added in the last DAgger iteration -
    which are oversampled 10x during training via WeightedRandomSampler.
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
        n = len(self.data)
        all_idx = np.random.permutation(n)
        n_val = int(val_frac * n)
        self._val_indices = all_idx[:n_val].tolist()
        self._train_indices = all_idx[n_val:].tolist()
        self._loader = None
        self._val_loader = None
        return self._train_indices, self._val_indices

    def get_loader(self, batch_size=256, shuffle=True, hard_weight=10.0):
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
            self._loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        return self._loader

    def get_val_loader(self, batch_size=256):
        if self._val_indices is None:
            return None
        if self._val_loader is None:
            val_subset = Subset(self, self._val_indices)
            self._val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        return self._val_loader


# ---------------------------------------------------------------------------
# Student evaluator
# ---------------------------------------------------------------------------
class StudentEvaluator:
    """Run student inference and flag failure cases using HybridHurdleModel + Q75 scaling."""
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
                pred_p.detach().cpu().numpy(), pred_j.detach().cpu().numpy(),
                pred_h.detach().cpu().numpy(), pred_w.detach().cpu().numpy(),
            ], axis=1)
            metric_dict = compute_forward_errors(
                pred_specs, val_specs, default_threshold=threshold,
            )
            all_errors = np.stack([
                metric_dict['err_p'], metric_dict['err_j'],
                metric_dict['err_h'], metric_dict['err_w'],
            ], axis=1)
            nan_mask = np.isnan(all_errors).any(axis=1) | np.isnan(pred_specs).any(axis=1)
            invalid_mask = p_valid.detach().cpu().numpy() < _active_ctx.get('VALIDITY_THRESHOLD', 0.5)
            failure_mask = nan_mask | invalid_mask | metric_dict['failure_mask']
            all_errors = np.where(nan_mask[:, np.newaxis], np.inf, all_errors)
            target_p = specs_t[:, 0]; target_j = specs_t[:, 1]
            target_h = specs_t[:, 2]; target_w = specs_t[:, 3]
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
                   'dim_fail_rates': (all_errors >= _active_ctx.get('DEGRADE_REL_THRESHOLD', 0.20)).mean(axis=0),
                   'degraded_dims': metric_dict['degraded_dims'],
                   'failure_mask': failure_mask,
                   'invalid_mask': invalid_mask,
                   'p_valid': p_valid.detach().cpu().numpy()}
        return failure_mask, metrics


class TeacherLabeler:
    """Wrapper around FlowTeacherLabeler that adds ZIG validity filtering."""
    def __init__(self, teacher_labeler, zig_model, scaler_X, device):
        self.teacher_labeler = teacher_labeler
        self.zig = zig_model
        self.scaler_X = scaler_X
        self.device = device

    def compute_validity(self, params_log):
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
        failed_array = np.array(failed_specs)
        raw_params = self.teacher_labeler.label_batch(
            failed_array, n_candidates=sample_budget,
            valid_threshold=0.9, top_k=1, verbose=True
        ).squeeze(1)
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


# ---------------------------------------------------------------------------
# Validation sampling
# ---------------------------------------------------------------------------
def sample_validation_specs(df, n_samples=2000, boundary_ratio=0.5, seed=None):
    """Sample validation specs with heavy boundary emphasis for DAgger evaluation."""
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


# ---------------------------------------------------------------------------
# Filtering / labeling helpers
# ---------------------------------------------------------------------------
def filter_by_zig_consistency(specs_arr, params_arr, zig_model, scaler_X, scaler_y_p,
                              threshold=0.10, device=None):
    if device is None:
        device = _active_device()
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
    keep = (p_valid >= _active_ctx.get('VALIDITY_THRESHOLD', 0.5)) & (~metrics['failure_mask'])
    n = len(specs_arr)
    n_boundary = ((specs_arr[:, 2] < BOUNDARY_ABS_TOLERANCES['height'][0]) |
                  (specs_arr[:, 3] < BOUNDARY_ABS_TOLERANCES['width'][0]) |
                  (specs_arr[:, 1] > BOUNDARY_ABS_TOLERANCES['jitter'][0]) |
                  (specs_arr[:, 0] < BOUNDARY_ABS_TOLERANCES['power'][0])).sum()
    _logger.info(f"  filter_by_zig_consistency: {keep.sum()}/{n} passed  "
                 f"(boundary={n_boundary}, interior={n-n_boundary})")
    return keep


def filter_by_zig_validity(params_arr, zig_model, scaler_X, threshold=0.5, device=None):
    if device is None:
        device = _active_device()
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


def filter_by_zig_consistency_relaxed(specs_arr, params_arr, zig_model, scaler_X, scaler_y_p,
                                       device=None):
    if device is None:
        device = _active_device()
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
    keep = (p_valid >= _active_ctx.get('VALIDITY_THRESHOLD', 0.5)) & (~metrics['failure_mask'])
    n = len(specs_arr)
    n_boundary = ((specs_arr[:, 2] < BOUNDARY_ABS_TOLERANCES_RELAXED['height'][0]) |
                  (specs_arr[:, 3] < BOUNDARY_ABS_TOLERANCES_RELAXED['width'][0]) |
                  (specs_arr[:, 1] > BOUNDARY_ABS_TOLERANCES_RELAXED['jitter'][0]) |
                  (specs_arr[:, 0] < BOUNDARY_ABS_TOLERANCES_RELAXED['power'][0])).sum()
    _logger.info(f"  filter_by_zig_consistency_relaxed: {keep.sum()}/{n} passed  "
                 f"(boundary={n_boundary}, interior={n-n_boundary})")
    return keep


def empirical_fallback_label(spec, df, k=3):
    spec_log = np.log10(np.clip(spec, 1e-12, None))
    emp_specs = df[['power', 'stage_2_jitter', 'stage_2_eye_max_height',
                    'stage_2_eye_max_width']].values
    emp_specs_log = np.log10(np.clip(emp_specs, 1e-12, None))
    dists = np.max(np.abs(emp_specs_log - spec_log), axis=1)
    k_eff = min(k, len(dists))
    idx = np.argpartition(dists, k_eff - 1)[:k_eff]
    return np.median(df.iloc[idx][PARAM_COLS].values, axis=0)


def is_boundary_spec(spec):
    return (spec[2] < BOUNDARY_ABS_TOLERANCES['height'][0] or
            spec[3] < BOUNDARY_ABS_TOLERANCES['width'][0] or
            spec[1] > BOUNDARY_ABS_TOLERANCES['jitter'][0] or
            spec[0] < BOUNDARY_ABS_TOLERANCES['power'][0])


def compute_forward_errors(pred_specs, target_specs,
                            boundary_tols=None,
                            default_threshold=None):
    if boundary_tols is None:
        boundary_tols = BOUNDARY_ABS_TOLERANCES
    # Audit fix (phase-a validity bundle): default_threshold was accepted but
    # never used — every caller silently got 0.20.  An explicitly passed
    # threshold now governs the degrade gate; None preserves the legacy
    # 0.20 default bit-identically for all existing callers.
    if default_threshold is None:
        degrade_thr = _active_ctx.get('DEGRADE_REL_THRESHOLD', 0.20)
    else:
        degrade_thr = float(default_threshold)
    min_bad = _active_ctx.get('MIN_DEGRADED_DIMS', 2)
    pred_p = pred_specs[:, 0]; pred_j = pred_specs[:, 1]
    pred_h = pred_specs[:, 2]; pred_w = pred_specs[:, 3]
    target_p = target_specs[:, 0]; target_j = target_specs[:, 1]
    target_h = target_specs[:, 2]; target_w = target_specs[:, 3]
    eps = 1e-6
    degrade_p = np.maximum((pred_p - target_p) / np.maximum(target_p, eps), 0.0)
    degrade_j = np.maximum((pred_j - target_j) / np.maximum(target_j, eps), 0.0)
    degrade_h = np.maximum((target_h - pred_h) / np.maximum(target_h, eps), 0.0)
    degrade_w = np.maximum((target_w - pred_w) / np.maximum(target_w, eps), 0.0)
    degraded = np.stack([
        degrade_p >= degrade_thr,
        degrade_j >= degrade_thr,
        degrade_h >= degrade_thr,
        degrade_w >= degrade_thr,
    ], axis=1)
    degraded_dims = degraded.sum(axis=1)
    max_err = np.maximum.reduce([degrade_p, degrade_j, degrade_h, degrade_w])
    failure_mask = degraded_dims >= min_bad
    return {
        'err_h': degrade_h, 'err_w': degrade_w,
        'err_j': degrade_j, 'err_p': degrade_p,
        'max_err': max_err,
        'degraded': degraded,
        'degraded_dims': degraded_dims,
        'failure_mask': failure_mask,
    }


def compute_forward_errors_relaxed(pred_specs, target_specs):
    return compute_forward_errors(pred_specs, target_specs, boundary_tols=BOUNDARY_ABS_TOLERANCES_RELAXED)


def log_failure_breakdown(metrics, prefix=""):
    if not metrics or 'errors' not in metrics:
        return
    errs = np.asarray(metrics['errors'])
    if errs.size == 0:
        return
    dim_names = ['power', 'jitter', 'height', 'width']
    degrade_thr = _active_ctx.get('DEGRADE_REL_THRESHOLD', 0.20)
    validity_thr = _active_ctx.get('VALIDITY_THRESHOLD', 0.5)
    fail_by_dim = (errs >= degrade_thr).mean(axis=0)
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
        fail_mask = np.asarray(metrics.get('failure_mask', degraded_dims >= _active_ctx.get('MIN_DEGRADED_DIMS', 2)))
        _logger.info(
            f"{prefix}adaptive: valid_threshold={validity_thr:.2f} "
            f"degrade_threshold={degrade_thr*100:.0f}% "
            f"min_bad_dims={_active_ctx.get('MIN_DEGRADED_DIMS', 2)} "
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


def log_label_quality_summary(name, specs_arr, params_arr, zig_model, scaler_X, scaler_y_p, device=None):
    if device is None:
        device = _active_device()
    if len(specs_arr) == 0 or len(params_arr) == 0:
        _logger.info(f"{name}: no samples")
        return
    keep_valid = filter_by_zig_validity(params_arr, zig_model, scaler_X, threshold=0.5, device=device)
    keep_strict = filter_by_zig_consistency(specs_arr, params_arr, zig_model, scaler_X, scaler_y_p, device=device)
    keep_relaxed = filter_by_zig_consistency_relaxed(specs_arr, params_arr, zig_model, scaler_X, scaler_y_p, device=device)
    _logger.info(
        f"{name}: validity@0.5={keep_valid.mean()*100:.1f}%  "
        f"strict_consistency={keep_strict.mean()*100:.1f}%  "
        f"relaxed_consistency={keep_relaxed.mean()*100:.1f}%"
    )


def compute_zig_score(params_log, zig_model, scaler_X, target_specs,
                       eye_scale_h, eye_scale_w, eye_scale_j, scaler_y_p):
    x_scaled = scaler_X.transform(params_log)
    x_tensor = torch.from_numpy(x_scaled.astype(np.float32))
    DEVICE_local = _active_device()
    x_tensor = x_tensor.to(DEVICE_local)
    with torch.no_grad():
        out = zig_model(x_tensor)
    pred_h = out['pred_h_soft'].cpu().numpy()
    pred_w = out['pred_w_soft'].cpu().numpy()
    pred_j = out['pred_j_soft'].cpu().numpy()
    power_scaled = out['mu_power'].cpu().numpy() * scaler_y_p.scale_[0] + scaler_y_p.mean_[0]
    pred_p = 10 ** power_scaled
    validity = out['p_valid'].cpu().numpy() if 'p_valid' in out else None
    return {
        'pred_height': pred_h, 'pred_width': pred_w, 'pred_jitter': pred_j,
        'pred_power': pred_p, 'validity': validity,
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_epoch(student, train_loader, optimizer, criterion, device, loss_weight=1.0,
                regime_loss_fn=None, regime_loss_weight=0.0):
    student.train()
    total_loss = 0
    losses = {'total': 0, 'imit': 0, 'spec': 0, 'phys': 0, 'invalid': 0, 'regime': 0}
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
        logits, regime_loss = regime_loss_fn(student, specs) if regime_loss_fn else (student(specs), None)
        loss_dict = criterion(student, spec_targets, params_target, logits=logits)
        total = loss_dict['total'] * loss_weight
        if regime_loss is not None:
            total = total + regime_loss_weight * regime_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        total_loss += loss_dict['total'].item()
        losses['total'] += loss_dict['total'].item()
        for k in ('imit', 'spec', 'phys', 'invalid'):
            losses[k] += loss_dict[k]
        losses['regime'] += regime_loss.item() if regime_loss is not None else 0.0
    n = len(train_loader)
    return {k: float(v/n) for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Phase-A training mode (plan canonical-ctle-unify)
# ---------------------------------------------------------------------------
# Static-data, Huber-on-log-params-only training mode for the canonical CTLE
# benchmark. No DAgger relabeling, no scheduler restarts, no surrogate-through
# gradient terms — those are deliberately held out so Phase A isolates model
# capacity + optimization from surrogate-gradient credit assignment. The
# KNet harness adds the differential-LR-group recipe from the Friedman wins.
# Spec: docs/specs/canonical-ctle-bench.md + docs/specs/canonical-phasea-run.md.

class _PhaseACanonicalDataset(Dataset):
    """Static loader for the canonical Phase-A .npz (specs + flow labels)."""

    MAX_PARAMS = 7

    def __init__(self, canonical: dict, split: str):
        self.specs = np.asarray(canonical["specs"], dtype=np.float32)
        params = np.asarray(canonical["params"], dtype=np.float32)
        params_log = np.log10(np.clip(params, 1e-12, None))
        pad = self.MAX_PARAMS - params_log.shape[-1]
        if pad > 0:
            params_log = np.pad(params_log, ((0, 0), (0, pad)), constant_values=-30.0)
        self.params_log = params_log.astype(np.float32)
        idx = canonical[f"{split}_idx"]
        self.indices = np.asarray(idx, dtype=np.int64).tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        j = self.indices[idx]
        specs = torch.from_numpy(self.specs[j])
        params = torch.from_numpy(self.params_log[j])
        return specs, params


def _phase_a_loaders(canonical: dict, batch_size: int):
    train_ds = _PhaseACanonicalDataset(canonical, "train")
    val_ds = _PhaseACanonicalDataset(canonical, "val")
    test_ds = _PhaseACanonicalDataset(canonical, "test")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def _phase_a_validity_nll(student, logits, scaler_X, zig_model):
    """Differentiable ZIG-validity NLL for Phase-A training: ``-log(p_valid)``.

    Mirrors the student→ZIG path in :meth:`StudentEvaluator.identify_failures`
    and :meth:`RegimeAwareLoss.forward` (sigmoid → bounded log-params →
    ZIG scaling → frozen ZIG forward), but stays entirely in torch (no
    detach/cpu/numpy) so gradients flow back into the student.  The ZIG
    weights themselves are frozen (``requires_grad=False`` at load); only
    the student receives gradients.

    Returns a scalar tensor (mean NLL over the batch).  NLL (rather than the
    linear ``1 - p_valid`` used by ``RegimeAwareLoss.L_invalid``) gives gradient
    ``-1/p``: hardest push exactly where specs slide down the validity cliff
    toward ``p -> 0``, relaxing once valid.
    """
    probs = torch.sigmoid(logits)
    log_lo = student.log_lo.unsqueeze(0)
    log_hi = student.log_hi.unsqueeze(0)
    bounded_log = log_lo + (log_hi - log_lo) * probs
    x_mean_t = torch.from_numpy(
        scaler_X.mean_.astype(np.float32)
    ).to(bounded_log.device)
    x_scale_t = torch.from_numpy(
        np.maximum(scaler_X.scale_, 1e-6).astype(np.float32)
    ).to(bounded_log.device)
    x_scaled = (bounded_log - x_mean_t) / (x_scale_t + 1e-8)
    x_scaled = torch.clamp(x_scaled, -10.0, 10.0)
    out = zig_model(x_scaled)
    if 'p_valid' in out:
        p_valid = out['p_valid'].clamp(1e-6, 1.0 - 1e-6)
    else:
        p_valid = (out['p_valid_h'] * out['p_valid_w'] * out['p_valid_j']).clamp(1e-6, 1.0 - 1e-6)
    return (-torch.log(p_valid)).mean()


def run_phase_a_training(student, ctx, *, model_name: str = "phase_a_student"):
    """Static, single-pass canonical-Phase-A training + ZIG evaluation.

    Parameters expected in ``ctx`` (subset of the DAgger context):
      - DEVICE, scaler_X, scaler_y_p, zig_model, eye_scale_{h,w,j}
      - canonical_dataset: dict returned by :func:`create_canonical_flow_dataset`
        (or loaded via :func:`load_canonical_flow_dataset`)
      - epochs (int), batch_size (int), lr (float), weight_decay (float)
      - output_dir (str)
    Optional:
      - grad_clip (float, default 1.0)
      - val_eval_every (int, default 1) — runs StudentEvaluator on the val split
        every N epochs; the best-failure checkpoint is the one carried forward.
      - earlystop_patience (int, default 30) — counted in *eval cycles*
        (only epochs that actually run the ZIG eval advance the counter).
      - error_threshold (float, default 0.10 — passes through to identify_failures)
      - validity_weight (float, default 0.0) — weight on the ZIG-validity NLL
        ``-log(p_valid)`` term.  0.0 reproduces the legacy Huber-only path
        bit-identically (the ZIG forward is skipped entirely).
      - validity_ramp_start (int, default 10) — first epoch (1-based) at which
        the validity term starts fading in; earlier epochs are Huber-only so
        the readout first fits the labels before validity is enforced.
      - validity_ramp_epochs (int, default 30) — linear ramp length in epochs
        from ramp start to full ``validity_weight``.
      - mapper_lr_scale / struct_lr_scale / dyn_lr_scale (floats) — KNet Friedman
        recipe; passed straight to the AdamW group builder.  If absent or 1.0
        the student is trained with a single AdamW group (safe for MLP).
    """
    DEVICE = ctx["DEVICE"]
    scaler_X = ctx["scaler_X"]
    scaler_y_p = ctx["scaler_y_p"]
    zig_model = ctx["zig_model"]
    eye_scale_h = ctx["eye_scale_h"]
    eye_scale_w = ctx["eye_scale_w"]
    eye_scale_j = ctx["eye_scale_j"]
    canonical = ctx["canonical_dataset"]
    epochs = int(ctx["epochs"])
    batch_size = int(ctx["batch_size"])
    lr = float(ctx["lr"])
    weight_decay = float(ctx["weight_decay"])
    output_dir = ctx["output_dir"]
    grad_clip = float(ctx.get("grad_clip", 1.0))
    val_eval_every = int(ctx.get("val_eval_every", 1))
    earlystop_patience = int(ctx.get("earlystop_patience", 30))
    error_threshold = float(ctx.get("error_threshold", 0.10))
    validity_weight = float(ctx.get("validity_weight", 0.0))
    validity_ramp_start = int(ctx.get("validity_ramp_start", 10))
    validity_ramp_epochs = int(ctx.get("validity_ramp_epochs", 30))
    mapper_lr_scale = float(ctx.get("mapper_lr_scale", 1.0))
    struct_lr_scale = float(ctx.get("struct_lr_scale", 1.0))
    dyn_lr_scale = float(ctx.get("dyn_lr_scale", 1.0))
    # knet input bounds: explicit ctx wins, else the canonical .npz bounds
    # (always present since the builder stores them).  Required by
    # attach_scaler for knet-preprocessing students.
    input_log_min = ctx.get("input_log_min", None)
    input_log_max = ctx.get("input_log_max", None)
    if input_log_min is None and "input_log_min" in canonical:
        input_log_min = np.asarray(canonical["input_log_min"], dtype=np.float32)
    if input_log_max is None and "input_log_max" in canonical:
        input_log_max = np.asarray(canonical["input_log_max"], dtype=np.float32)

    if hasattr(student, "attach_scaler"):
        student.attach_scaler(
            scaler_p_scale=scaler_y_p.scale_[0],
            scaler_p_mean=scaler_y_p.mean_[0],
            eye_scale_j=eye_scale_j,
            eye_scale_h=eye_scale_h,
            eye_scale_w=eye_scale_w,
            input_preprocessing=ctx.get("input_preprocessing", "q75"),
            input_log_min=input_log_min,
            input_log_max=input_log_max,
        )
    student = student.to(DEVICE)

    train_loader, val_loader, test_loader = _phase_a_loaders(canonical, batch_size)
    _logger.info(
        f"[phaseA] split sizes: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)} "
        f"epochs={epochs} batch={batch_size} lr={lr:.2e}"
    )

    optimizer = _build_phase_a_optimizer(
        student, lr=lr, weight_decay=weight_decay,
        mapper_lr_scale=mapper_lr_scale,
        struct_lr_scale=struct_lr_scale,
        dyn_lr_scale=dyn_lr_scale,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_valid_nll": [], "validity_w_eff": [],
               "val_failure_rate": [], "val_boundary_failure": [],
               "val_interior_failure": []}
    best_val_failure = float("inf")
    best_state = None
    best_val_loss = float("inf")
    best_val_loss_state = None
    no_improve = 0
    checkpoint_path = os.path.join(output_dir, "phase_a_last.pt")
    best_path = os.path.join(output_dir, "phase_a_best.pt")
    os.makedirs(output_dir, exist_ok=True)

    evaluator = StudentEvaluator(
        student=student, scaler_X=scaler_X, zig_model=zig_model,
        eye_scale_h=eye_scale_h, eye_scale_w=eye_scale_w,
        eye_scale_j=eye_scale_j, scaler_y_p=scaler_y_p, device=DEVICE,
    )

    for epoch in range(epochs):
        student.train()
        # Ramped validity weight (1-based epochs): 0 up to ramp start, linear
        # ramp over ramp_epochs, full weight thereafter.  validity_weight=0.0
        # keeps the legacy Huber-only path (ZIG forward skipped entirely).
        if validity_weight > 0 and validity_ramp_epochs > 0:
            ramp_frac = ((epoch + 1) - validity_ramp_start) / validity_ramp_epochs
            w_eff = validity_weight * min(1.0, max(0.0, ramp_frac))
        else:
            w_eff = validity_weight
        total_loss = 0.0
        total_huber = 0.0
        total_valid_nll = 0.0
        n_batches = 0
        for specs_batch, params_target in train_loader:
            specs_batch = specs_batch.to(DEVICE)
            params_target = params_target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = student(specs_batch)
            huber_term = huber_loss(logits, params_target, delta=0.5).mean()
            if w_eff > 0:
                valid_nll = _phase_a_validity_nll(
                    student, logits, scaler_X, zig_model,
                )
                loss = huber_term + w_eff * valid_nll
            else:
                valid_nll = None
                loss = huber_term
            if not torch.isfinite(loss):
                _logger.warning(
                    f"[phaseA] non-finite loss at epoch {epoch+1}, batch {n_batches}; skipping"
                )
                n_batches += 1
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            total_huber += huber_term.item()
            if valid_nll is not None:
                total_valid_nll += valid_nll.item()
            n_batches += 1
        scheduler.step()
        avg_train = total_loss / max(1, n_batches)
        avg_huber = total_huber / max(1, n_batches)
        avg_valid_nll = total_valid_nll / max(1, n_batches) if w_eff > 0 else float("nan")
        avg_val = _phase_a_huber_eval(student, val_loader, DEVICE)
        # ZIG failure eval runs only on eval epochs (or the final epoch).
        # Patience is counted in *eval cycles*: non-eval epochs must not
        # advance no_improve (previously they did via the nan placeholder,
        # burning patience ~val_eval_every times too fast).
        eval_this_epoch = (epoch + 1) % val_eval_every == 0 or epoch == epochs - 1
        val_failure_rate = float("nan")
        boundary_failure_rate = float("nan")
        interior_failure_rate = float("nan")
        if eval_this_epoch:
            evaluator.student = student
            val_specs = canonical["specs"][np.asarray(canonical["val_idx"], dtype=np.int64)]
            failure_mask, metrics = evaluator.identify_failures(
                val_specs, threshold=error_threshold,
            )
            val_failure_rate = float(failure_mask.mean())
            if "boundary_failure" in metrics and "is_boundary" in metrics:
                bm = metrics["is_boundary"]
                boundary_failure_rate = float(
                    metrics["boundary_failure"].sum() / max(1, bm.sum())
                )
                interior_failure_rate = float(
                    metrics["interior_failure"].sum() / max(1, (~bm).sum())
                )
            if val_failure_rate < best_val_failure:
                best_val_failure = val_failure_rate
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train)
        history["train_valid_nll"].append(avg_valid_nll)
        history["validity_w_eff"].append(w_eff)
        history["val_loss"].append(avg_val)
        history["val_failure_rate"].append(val_failure_rate)
        history["val_boundary_failure"].append(boundary_failure_rate)
        history["val_interior_failure"].append(interior_failure_rate)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_val_loss_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        torch.save(
            {"epoch": epoch, "model": student.state_dict(),
             "optimizer": optimizer.state_dict(), "history": history,
             "best_val_failure": best_val_failure},
            checkpoint_path,
        )
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            if eval_this_epoch:
                fail_str = (
                    f"val_fail={val_failure_rate*100:5.2f}% "
                    f"(bnd={boundary_failure_rate*100:5.2f}% int={interior_failure_rate*100:5.2f}%) "
                    f"best={best_val_failure*100:5.2f}%"
                )
            else:
                fail_str = "val_fail=  --% (no ZIG eval this epoch)"
            _logger.info(
                f"[phaseA] epoch {epoch+1:3d}/{epochs}  "
                f"train_huber={avg_huber:.4f}  valid_nll={avg_valid_nll:5.4f}  "
                f"w_valid={w_eff:.3f}  val_huber={avg_val:.4f}  "
                f"{fail_str} patience={no_improve}/{earlystop_patience}"
            )
        if earlystop_patience > 0 and no_improve >= earlystop_patience:
            _logger.info(
                f"[phaseA] early stop: no best-failure improvement in "
                f"{earlystop_patience} eval cycles; restoring best-failure checkpoint"
            )
            break

    if best_state is not None:
        student.load_state_dict(best_state)
        student.to(DEVICE)
        torch.save(best_state, best_path)
    elif best_val_loss_state is not None:
        student.load_state_dict(best_val_loss_state)
        student.to(DEVICE)
        torch.save(best_val_loss_state, best_path)

    final = {}
    for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        evaluator.student = student
        idx_arr = canonical[f"{split_name}_idx"]
        specs_arr = canonical["specs"][np.asarray(idx_arr, dtype=np.int64)]
        failure_mask, metrics = evaluator.identify_failures(
            specs_arr, threshold=error_threshold,
        )
        boundary_failure_rate = float("nan")
        interior_failure_rate = float("nan")
        if "boundary_failure" in metrics and "is_boundary" in metrics:
            bm = metrics["is_boundary"]
            boundary_failure_rate = float(metrics["boundary_failure"].sum() / max(1, bm.sum()))
            interior_failure_rate = float(metrics["interior_failure"].sum() / max(1, (~bm).sum()))
        final[split_name] = {
            "failure_rate": float(failure_mask.mean()),
            "boundary_failure_rate": boundary_failure_rate,
            "interior_failure_rate": interior_failure_rate,
            "dim_fail_rates": (
                np.asarray(metrics["errors"]) >= float(ctx.get("DEGRADE_REL_THRESHOLD", 0.20))
            ).mean(axis=0).tolist(),
        }

    log_path = os.path.join(output_dir, "phase_a_history.json")
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump({"history": history, "final": final, "args": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "model_name": model_name,
            "fingerprint": _canonical_fingerprint_str(canonical),
            "mapper_lr_scale": mapper_lr_scale, "struct_lr_scale": struct_lr_scale,
            "dyn_lr_scale": dyn_lr_scale,
            "validity_weight": validity_weight,
            "validity_ramp_start": validity_ramp_start,
            "validity_ramp_epochs": validity_ramp_epochs,
            "error_threshold": error_threshold,
            "val_eval_every": val_eval_every,
            "earlystop_patience": earlystop_patience,
        }}, handle, indent=2, default=str)
    _logger.info(f"[phaseA] wrote {log_path}")
    _logger.info(
        f"[phaseA] final failure: train={final['train']['failure_rate']*100:5.2f}% "
        f"val={final['val']['failure_rate']*100:5.2f}% test={final['test']['failure_rate']*100:5.2f}%"
    )
    return {
        "best_val_failure_rate": float(best_val_failure),
        "final": final,
        "history": history,
        "checkpoint": best_path,
    }


def _canonical_fingerprint_str(canonical: dict) -> str | None:
    fp = canonical.get("fingerprint", None)
    if fp is None:
        return None
    try:
        return str(np.asarray(fp).item())
    except (ValueError, TypeError):
        return str(fp)


def init_output_affine_bias_from_labels(student, train_params_physical: np.ndarray,
                                        param_log_bounds: dict | None = None) -> bool:
    """Init a KNet OutputAffine bias to train-label means (Friedman recipe).

    With gain=1 and silent nonlinear branches, ``bias=logit(mean position)``
    makes the readout start at the dataset mean instead of mid-range, removing
    a constant offset the ODE would otherwise have to learn.  Returns True when
    an OutputAffine-style ``output_mapper`` with a 7-dim bias was found and
    initialised, False otherwise (safe no-op for MLP students).
    """
    bounds = param_log_bounds or PARAM_LOG_BOUNDS
    names = list(bounds.keys())
    inner = getattr(student, "net", student)
    mapper = getattr(inner, "output_mapper", None)
    bias = getattr(mapper, "bias", None)
    if bias is None or not isinstance(bias, torch.nn.Parameter):
        return False
    if bias.numel() != len(names):
        return False
    phys = np.asarray(train_params_physical, dtype=np.float64)
    if phys.ndim != 2 or phys.shape[1] != len(names) or len(phys) == 0:
        return False
    logits = np.zeros(len(names), dtype=np.float32)
    for j, name in enumerate(names):
        lo, hi = bounds[name]
        span = hi - lo
        pos = (np.log10(np.clip(phys[:, j], 1e-12, None)) - lo) / max(span, 1e-8)
        pos = float(np.clip(pos.mean(), 1e-3, 1.0 - 1e-3))
        logits[j] = float(np.log(pos / (1.0 - pos)))
    with torch.no_grad():
        bias.data = torch.as_tensor(logits, dtype=bias.dtype, device=bias.device)
    _logger.info(
        "[phaseA] OutputAffine bias initialised to train-label means: "
        + ", ".join(f"{n}={v:+.3f}" for n, v in zip(names, logits))
    )
    return True


def _phase_a_huber_eval(student, val_loader, device):
    student.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for specs_batch, params_target in val_loader:
            specs_batch = specs_batch.to(device)
            params_target = params_target.to(device)
            logits = student(specs_batch)
            loss = huber_loss(logits, params_target, delta=0.5).mean()
            if torch.isfinite(loss):
                total_loss += loss.item()
                n_batches += 1
    return total_loss / max(1, n_batches)


def _build_phase_a_optimizer(student, *, lr, weight_decay,
                             mapper_lr_scale=1.0, struct_lr_scale=1.0,
                             dyn_lr_scale=1.0):
    """AdamW with optional differential LR groups (Friedman KNet recipe).

    Single-group when all scales are 1.0 (safe for MLP / Plain students).
    Group naming follows the same conventions used by the Friedman KNet
    harness (input_mapper / output_mapper / post_readout_transfer /
    z_logits / u_logits / raw_leak / raw_drive_g / cell_lib.* / vca_*).
    """
    mapper_params, struct_params, dyn_params = [], [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if any(tag in n for tag in ("input_mapper.", "output_mapper.",
                                    "post_readout_transfer.")):
            mapper_params.append(p)
        elif any(tag in n for tag in ("z_logits", "u_logits", "raw_leak",
                                      "raw_drive_g", "vca_")):
            struct_params.append(p)
        else:
            dyn_params.append(p)
    if mapper_lr_scale == 1.0 and struct_lr_scale == 1.0 and dyn_lr_scale == 1.0:
        return torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    group_list = []
    if mapper_params:
        group_list.append({"params": mapper_params, "lr": lr * mapper_lr_scale,
                           "weight_decay": weight_decay})
    if struct_params:
        group_list.append({"params": struct_params, "lr": lr * struct_lr_scale,
                           "weight_decay": weight_decay})
    if dyn_params:
        group_list.append({"params": dyn_params, "lr": lr * dyn_lr_scale,
                           "weight_decay": weight_decay})
    if not group_list:
        return torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(group_list, lr=lr, weight_decay=weight_decay)


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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def run_diagnostics(teacher_labeler, df, zig_model, scaler_X, scaler_y_p, device, n=200):
    _logger.info("="*60)
    _logger.info("  RUNNING DIAGNOSTICS")
    _logger.info("="*60)
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
    _logger.info("  If (1) >50%:  teacher is fine -> focus on loss/replay")
    _logger.info("  If (1) <50%:  flow teacher is weak -> increase candidate budget")
    _logger.info("  If (2) <50%:  empirical data is weak -> need better training set")
    _logger.info("  If (3) <50%:  ZIG p_valid threshold may be too high (lower to 0.3)")
    return {
        'flow_yield': keep_flow.mean() if 'keep_flow' in dir() else None,
        'knn_yield': keep_knn.mean(),
        'zig_real_acc': keep_real.mean(),
    }


# ---------------------------------------------------------------------------
# Setup: load teacher artifacts, CSV, scalers, ZIG
# ---------------------------------------------------------------------------
def setup(args) -> dict:
    """Load teacher artifacts, CSV data, ZIG model, scalers.

    Side-effect: populates the module-level ``_active_ctx`` used by helper
    functions that need scaler/ZIG/DEVICE. Returns a context dict the caller
    passes into :func:`run_dagger_training`.
    """
    teacher_dir = args.teacher_dir
    data_dir = args.data_dir
    output_dir = args.output
    device = (torch.device(args.device) if args.device != 'auto'
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    _logger.info(f"Using device: {device}")
    _logger.info(f"  TEACHER_DIR={teacher_dir}")
    _logger.info(f"  DATA_DIR={data_dir}")
    _logger.info(f"  OUTPUT_DIR={output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    flow_scaler_X = joblib.load(os.path.join(teacher_dir, 'flow_scaler_X.pkl'))
    zig_scaler_X = joblib.load(os.path.join(teacher_dir, 'hybrid_hurdle_scaler_X.pkl'))
    zig_scaler_y_p = joblib.load(os.path.join(teacher_dir, 'hybrid_hurdle_scaler_y_power.pkl'))
    zig_config = joblib.load(os.path.join(teacher_dir, 'hybrid_hurdle_config.pkl'))
    flow_scaler_C = joblib.load(os.path.join(teacher_dir, 'flow_scaler_C.pkl'))
    flow_scaler_y_p = flow_scaler_C['scaler_y_p']
    eye_scale_h = float(zig_config['eye_scale_h'])
    eye_scale_w = float(zig_config['eye_scale_w'])
    eye_scale_j = float(zig_config['eye_scale_j'])
    per_target_hurdle = bool(zig_config.get('per_target_hurdle', False))
    _logger.info("Teacher artifacts loaded")
    _logger.info(f"  flow scaler_X mean: {flow_scaler_X.mean_[:3]}...")
    _logger.info(f"  zig scaler_X mean: {zig_scaler_X.mean_[:3]}...")
    _logger.info(f"  flow scaler_y_p: mean={flow_scaler_y_p.mean_[0]:.4f}, scale={flow_scaler_y_p.scale_[0]:.4f}")
    _logger.info(f"  zig scaler_y_p:  mean={zig_scaler_y_p.mean_[0]:.4f}, scale={zig_scaler_y_p.scale_[0]:.4f}")
    _logger.info(f"  eye_scale_h (Q75 scale): {eye_scale_h:.6f}")
    _logger.info(f"  eye_scale_w (Q75 scale): {eye_scale_w:.6f}")
    _logger.info(f"  eye_scale_j (Q75 scale): {eye_scale_j:.6f}")
    _logger.info(f"  per_target_hurdle: {per_target_hurdle}")

    zig_model = HybridHurdleModel(dropout=0.0, per_target=per_target_hurdle).to(device)
    zig_model.load_state_dict(torch.load(os.path.join(teacher_dir, 'hybrid_hurdle_ctle_model.pt'),
                                          map_location=device))
    zig_model.eval()
    for p in zig_model.parameters():
        p.requires_grad = False
    _logger.info("HybridHurdleModel (ZIG) loaded from checkpoint")

    with Timer("load historical dataset"):
        dfs = []
        for fname in DATASET_CSV_FILES:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                _logger.warning(f"  Warning: {fname} not found, skipping")
                continue
            dfs.append(pd.read_csv(fpath))
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.rename(columns=COL_MAPPING)
        required_cols = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD',
                          'power', 'stage_2_eye_max_height', 'stage_2_eye_max_width', 'stage_2_jitter']
        df = combined[required_cols].copy()
        mask = ((df['stage_2_jitter'] > 0) & (df['power'] > 0)
                & (df['stage_2_eye_max_height'] > 0) & (df['stage_2_eye_max_width'] > 0)
                & ~df.isna().any(axis=1))
        df = df[mask].reset_index(drop=True)
    _logger.info(f"Filtered rows: {len(df)}")
    spec_logs = np.log10(np.clip(df[['power', 'stage_2_jitter',
                                     'stage_2_eye_max_height', 'stage_2_eye_max_width']].values,
                                  1e-12, None))
    input_log_min = (spec_logs.min(axis=0) - 0.05 * np.maximum(spec_logs.max(axis=0) - spec_logs.min(axis=0), 1e-8)).astype(np.float32)
    input_log_max = (spec_logs.max(axis=0) + 0.05 * np.maximum(spec_logs.max(axis=0) - spec_logs.min(axis=0), 1e-8)).astype(np.float32)
    input_preprocessing = getattr(args, 'input_preprocessing', 'knet')
    _logger.info(f"Input preprocessing: {input_preprocessing} ({'4 features, min-max, clip [-4,4]' if input_preprocessing == 'knet' else 'legacy Q75 + optional raw-log features'})")
    for col in ['power', 'stage_2_eye_max_height', 'stage_2_eye_max_width', 'stage_2_jitter']:
        nz = df[df[col] > 0][col]
        _logger.info(f"{col}: n={len(nz)}, min={nz.min():.4f}, max={nz.max():.4f}, median={nz.median():.4f}")

    teacher_labeler = FlowTeacherLabeler(teacher_dir, device, per_target=per_target_hurdle)

    ctx = {
        'TEACHER_DIR': teacher_dir,
        'DATA_DIR': data_dir,
        'OUTPUT_DIR': output_dir,
        'DEVICE': device,
        'scaler_X': zig_scaler_X,
        'scaler_y_p': zig_scaler_y_p,
        'flow_scaler_X': flow_scaler_X,
        'flow_scaler_C': flow_scaler_C,
        'eye_scale_h': eye_scale_h,
        'eye_scale_w': eye_scale_w,
        'eye_scale_j': eye_scale_j,
        'zig_model': zig_model,
        'per_target_hurdle': per_target_hurdle,
        'df': df,
        'teacher_labeler': teacher_labeler,
        'input_preprocessing': input_preprocessing,
        'input_log_min': input_log_min,
        'input_log_max': input_log_max,
    }
    for k, v in HPARAMS_DEFAULTS.items():
        ctx[k] = v
    for k, v in vars(args).items():
        if v is not None and k.upper() in HPARAMS_DEFAULTS:
            ctx[k.upper()] = v
    if getattr(args, 'param_budget', None) is not None:
        ctx['PARAM_BUDGET'] = args.param_budget
    ctx['EARLYSTOP_PATIENCE_EPOCHS'] = ctx['EPOCHS_PER_ITER']

    _set_active_context(ctx)

    return ctx


# ---------------------------------------------------------------------------
# DAgger runner
# ---------------------------------------------------------------------------
def run_dagger_training(student, ctx, *, regime_loss_fn=None, regime_loss_weight=0.0,
                        model_name='dagger_student'):
    """Run the full DAgger training loop + diagnostics + evaluation + plotting.

    Parameters
    ----------
    student : nn.Module
        Must support ``forward(x) -> logits`` and expose ``log_lo``, ``log_hi``.
    regime_loss_fn : callable or None
        ``(student, specs) -> (logits, regime_loss_or_None)``. When None, no
        auxiliary loss is added. For MoE, pass a callable that returns
        ``(logits, regime_loss)`` and ``regime_loss_weight > 0``.
    """
    DEVICE = ctx['DEVICE']
    teacher_labeler = ctx['teacher_labeler']
    zig_model = ctx['zig_model']
    scaler_X = ctx['scaler_X']
    scaler_y_p = ctx['scaler_y_p']
    eye_scale_h = ctx['eye_scale_h']
    eye_scale_w = ctx['eye_scale_w']
    eye_scale_j = ctx['eye_scale_j']
    df = ctx['df']
    OUTPUT_DIR = ctx['OUTPUT_DIR']

    DAGGER_ITERATIONS = ctx['DAGGER_ITERATIONS']
    EPOCHS_PER_ITER = ctx['EPOCHS_PER_ITER']
    BATCH_SIZE = ctx['BATCH_SIZE']
    ERROR_THRESHOLD = ctx['ERROR_THRESHOLD']
    BOUNDARY_RATIO = ctx['BOUNDARY_RATIO']
    N_CANDIDATES_PER_SPEC = ctx['N_CANDIDATES_PER_SPEC']
    N_CANDIDATES_BOUNDARY = ctx['N_CANDIDATES_BOUNDARY']
    VALIDITY_THRESHOLD = ctx['VALIDITY_THRESHOLD']
    DEGRADE_REL_THRESHOLD = ctx['DEGRADE_REL_THRESHOLD']
    MIN_DEGRADED_DIMS = ctx['MIN_DEGRADED_DIMS']
    LR_INITIAL = ctx['LR_INITIAL']
    LR_DECAY_AFTER_ITER = ctx['LR_DECAY_AFTER_ITER']
    LR_FLOOR = ctx['LR_FLOOR']
    FAILURE_CAP_RATIO = ctx['FAILURE_CAP_RATIO']
    CONVERGENCE_THRESHOLD = ctx['CONVERGENCE_THRESHOLD']
    COMMON_EVAL_SIZE = ctx['COMMON_EVAL_SIZE']
    COMMON_EVAL_SEED = ctx['COMMON_EVAL_SEED']
    EARLYSTOP_EVAL_EVERY = ctx['EARLYSTOP_EVAL_EVERY']
    EARLYSTOP_LOG_EVERY = ctx['EARLYSTOP_LOG_EVERY']
    EARLYSTOP_SKIP_EPOCHS = ctx['EARLYSTOP_SKIP_EPOCHS']
    EARLYSTOP_PATIENCE_EPOCHS = ctx['EARLYSTOP_PATIENCE_EPOCHS']
    MIN_FAILURE_IMPROVEMENT = ctx['MIN_FAILURE_IMPROVEMENT']
    DIVERGENCE_ABORT = ctx['DIVERGENCE_ABORT']
    DIVERGENCE_MARGIN = ctx['DIVERGENCE_MARGIN']
    DIVERGENCE_CONSEC_EVALS = ctx['DIVERGENCE_CONSEC_EVALS']
    N_EMPIRICAL_SAMPLES = ctx['N_EMPIRICAL_SAMPLES']
    N_CANDIDATES_INITIAL = ctx['N_CANDIDATES_INITIAL']
    WEIGHT_DECAY = ctx['WEIGHT_DECAY']
    HARD_BUFFER_WEIGHT = ctx['HARD_BUFFER_WEIGHT']
    WARMUP_EPOCHS = ctx['WARMUP_EPOCHS']
    ALPHA_SPEC = ctx['ALPHA_SPEC']
    BETA_PHYS = ctx['BETA_PHYS']
    GAMMA_MONO = ctx['GAMMA_MONO']
    ALPHA_INVALID = ctx['ALPHA_INVALID']
    K_MANIFOLD = ctx['K_MANIFOLD']
    ALPHA_MANIFOLD = ctx['ALPHA_MANIFOLD']

    if hasattr(student, 'attach_scaler'):
        student.attach_scaler(
            scaler_p_scale=scaler_y_p.scale_[0],
            scaler_p_mean=scaler_y_p.mean_[0],
            eye_scale_j=eye_scale_j,
            eye_scale_h=eye_scale_h,
            eye_scale_w=eye_scale_w,
            input_preprocessing=ctx.get('input_preprocessing', 'q75'),
            input_log_min=ctx.get('input_log_min'),
            input_log_max=ctx.get('input_log_max'),
        )
    student = student.to(DEVICE)

    with Timer("RUNNING DIAGNOSTICS"):
        run_diagnostics(teacher_labeler, df, zig_model, scaler_X, scaler_y_p, DEVICE, n=200)

    with Timer("build COMMON_EVAL_SPECS shared eval set"):
        COMMON_EVAL_SPECS = sample_validation_specs(
            df, n_samples=COMMON_EVAL_SIZE, boundary_ratio=BOUNDARY_RATIO, seed=COMMON_EVAL_SEED
        )
    _logger.info(f"  COMMON_EVAL_SPECS: shape={COMMON_EVAL_SPECS.shape}, seed={COMMON_EVAL_SEED}")

    criterion = RegimeAwareLoss(
        zig_model=zig_model, scaler_X=scaler_X,
        eye_scale_h=eye_scale_h, eye_scale_w=eye_scale_w, eye_scale_j=eye_scale_j,
        scaler_y_p=scaler_y_p,
        alpha_spec=ALPHA_SPEC, beta_phys=BETA_PHYS, gamma_mono=GAMMA_MONO, alpha_invalid=ALPHA_INVALID,
        empirical_df=df, k_manifold=K_MANIFOLD, alpha_manifold=ALPHA_MANIFOLD,
    )
    ramped_criterion = RampedRegimeLoss(criterion, warmup_epochs=WARMUP_EPOCHS)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR_INITIAL, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_ITER, eta_min=LR_FLOOR)

    with Timer("build initial flow distillation dataset"):
        initial_data = create_flow_distillation_dataset(
            df, n_samples=N_EMPIRICAL_SAMPLES,
            n_candidates=N_CANDIDATES_INITIAL,
            boundary_ratio=BOUNDARY_RATIO,
            teacher_labeler=teacher_labeler,
        )
    _logger.info(f"Initial dataset (flow teacher labels): {len(initial_data)} samples")
    distillation_dataset = DistillationDataset(initial_data)
    train_subset, val_subset = distillation_dataset.split_train_val(val_frac=0.1)
    train_loader = distillation_dataset.get_loader(batch_size=BATCH_SIZE, shuffle=True, hard_weight=HARD_BUFFER_WEIGHT)
    val_loader = distillation_dataset.get_val_loader(batch_size=BATCH_SIZE)
    _logger.info(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

    dagger_history = {'iteration': [], 'failure_rate': [], 'dataset_size': [],
                       'train_loss': [], 'boundary_failure_rate': [], 'interior_failure_rate': []}
    loss_history = {'iteration': [], 'train_imit': [], 'train_spec': [], 'train_phys': [],
                    'train_invalid': [], 'train_total': [], 'train_manifold': [],
                    'val_imit': [], 'val_spec': [], 'val_phys': [], 'val_invalid': [],
                    'val_total': [], 'val_manifold': []}

    for dagger_iter in range(DAGGER_ITERATIONS):
        _logger.info(f"{'='*60}")
        _logger.info(f"  DAgger Iteration {dagger_iter + 1}/{DAGGER_ITERATIONS}")
        _logger.info(f"{'='*60}")
        _logger.info(f"Dataset size: {len(distillation_dataset)}")
        train_loader = distillation_dataset.get_loader(batch_size=BATCH_SIZE, shuffle=True, hard_weight=HARD_BUFFER_WEIGHT)
        val_loader = distillation_dataset.get_val_loader(batch_size=BATCH_SIZE)

        best_val_loss = float('inf')
        best_state = None
        best_failure_state = None
        prev_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        evaluator = StudentEvaluator(
            student=student, scaler_X=scaler_X, zig_model=zig_model,
            eye_scale_h=eye_scale_h, eye_scale_w=eye_scale_w, eye_scale_j=eye_scale_j,
            scaler_y_p=scaler_y_p, device=DEVICE,
        )
        if dagger_iter == 0:
            prev_failure_rate = 1.0
            _logger.info("  Prev-iter baseline: no prior model (iteration 1) -> prev_failure_rate=100.0%")
        else:
            evaluator.student = student
            _prev_mask, _ = evaluator.identify_failures(COMMON_EVAL_SPECS, threshold=ERROR_THRESHOLD)
            prev_failure_rate = float(_prev_mask.mean())
            _logger.info(f"  Prev-iter baseline: prev_failure_rate={prev_failure_rate*100:.2f}% on COMMON_EVAL_SPECS (n={len(COMMON_EVAL_SPECS)})")
        best_failure_rate = prev_failure_rate
        _logger.info(f"  Shared eval set (COMMON_EVAL_SPECS): n={len(COMMON_EVAL_SPECS)} specs "
                     f"(seed={COMMON_EVAL_SEED}, every={EARLYSTOP_EVAL_EVERY}ep, "
                     f"log_every={EARLYSTOP_LOG_EVERY}ep, skip_first={EARLYSTOP_SKIP_EPOCHS}ep)")

        epoch_losses = {'total': [], 'imit': [], 'spec': [], 'phys': [], 'invalid': [], 'manifold': []}
        n_nonfinite_batches = 0
        _div_consec = 0
        diverged_at = None

        for epoch in range(EPOCHS_PER_ITER):
            ramped_criterion.set_epoch(epoch)
            student.train()
            losses = {'total': 0, 'imit': 0, 'spec': 0, 'phys': 0, 'invalid': 0, 'manifold': 0}
            for specs_batch, params_batch in train_loader:
                specs_batch = specs_batch.to(DEVICE)
                params_batch = params_batch.to(DEVICE)
                spec_targets = {
                    'power': specs_batch[:, 0],
                    'jitter': specs_batch[:, 1],
                    'height': specs_batch[:, 2],
                    'width': specs_batch[:, 3],
                }
                optimizer.zero_grad()
                logits, regime_loss = (regime_loss_fn(student, specs_batch)
                                       if regime_loss_fn else (student(specs_batch), None))
                base_loss_dict = ramped_criterion.base_loss(student, spec_targets, params_batch, logits=logits)
                ramped_loss_dict = ramped_criterion(student, spec_targets, params_batch,
                                                    logits=logits, loss_dict=base_loss_dict)
                total_loss = ramped_loss_dict['total']
                if regime_loss is not None:
                    total_loss = total_loss + regime_loss_weight * regime_loss
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
                            _logger.warning(f"    [grad-guard] non-finite grad norm at epoch {epoch+1}; skipped optimizer step ({n_nonfinite_batches} skipped this iteration)")
                    losses['total'] += total_loss.item()
                for k in ('imit', 'spec', 'phys', 'invalid', 'manifold'):
                    losses[k] += ramped_loss_dict[k]

            n = len(train_loader)
            avg_loss = losses['total'] / n
            avg_imit = losses['imit'] / n
            avg_spec = losses['spec'] / n
            avg_phys = losses['phys'] / n
            avg_invalid = losses['invalid'] / n
            avg_manifold = losses['manifold'] / n
            for k in epoch_losses:
                epoch_losses[k] += [losses[k] / n]

            val_losses = eval_epoch(student, val_loader, ramped_criterion, DEVICE)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                _logger.info(f"  Epoch {epoch+1:3d}/{EPOCHS_PER_ITER}  "
                             f"train={avg_loss:.4f}(imit={avg_imit:.3f},spec={avg_spec:.3f},phys={avg_phys:.3f},invalid={avg_invalid:.3f},manifold={avg_manifold:.3f})  "
                             f"val={val_losses['total']:.4f}(imit={val_losses['imit']:.3f},spec={val_losses['spec']:.3f},phys={val_losses['phys']:.3f},invalid={val_losses['invalid']:.3f},manifold={val_losses['manifold']:.3f})  "
                             f"lr={optimizer.param_groups[0]['lr']:.2e}")
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

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
                        _logger.info(f"    earlystop@{epoch+1:3d}: failure_rate="
                                     f"{current_failure_rate*100:.2f}% "
                                     f"(no accept; best={best_failure_rate*100:.2f}%, "
                                     f"min_delta={MIN_FAILURE_IMPROVEMENT*100:.1f}%)")
                    if DIVERGENCE_ABORT and (
                            current_failure_rate >= prev_failure_rate + DIVERGENCE_MARGIN):
                        _div_consec += 1
                        if _div_consec >= DIVERGENCE_CONSEC_EVALS:
                            diverged_at = epoch + 1
                            _logger.warning(
                                f"    [divergence] failure_rate >= "
                                f"{(prev_failure_rate + DIVERGENCE_MARGIN)*100:.1f}% for "
                                f"{DIVERGENCE_CONSEC_EVALS} consecutive evals; aborting "
                                f"iteration {dagger_iter+1} at epoch {epoch+1}")
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
            _logger.warning("  Carry-forward: NO checkpoint captured this iteration; "
                            "keeping last-epoch student")

        best_loss = float(best_val_loss)

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

        if len(failed_specs_arr) == 0:
            _logger.info("No failures detected - skipping teacher labeling.")
            continue

        cap = int(len(distillation_dataset) * FAILURE_CAP_RATIO)
        if len(failed_specs_arr) > cap:
            top_k_idx = np.argsort(failure_errors[failure_mask])[-cap:]
            failed_specs_arr = failed_specs_arr[top_k_idx]
            _logger.info(f"  Capped failures to top {cap} highest-error cases")

        new_lr = LR_INITIAL * (0.5 ** max(0, dagger_iter - LR_DECAY_AFTER_ITER))
        new_lr = max(new_lr, 1e-4)
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS_PER_ITER, eta_min=LR_FLOOR
        )
        _logger.info(f"  LR reset to {new_lr:.2e}; cosine scheduler restarted for next iteration")

        _logger.info(f"  Querying teacher for {len(failed_specs_arr)} failed specs ...")
        boundary_mask_iter = np.array([is_boundary_spec(s) for s in failed_specs_arr])
        boundary_specs_iter = failed_specs_arr[boundary_mask_iter]
        interior_specs = failed_specs_arr[~boundary_mask_iter]

        new_labels = np.zeros((len(failed_specs_arr), 7))
        label_source = np.zeros(len(failed_specs_arr), dtype=int)

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
            idx_int = np.where(~boundary_mask_iter)[0]
            if keep_int.sum() > 0:
                new_labels[idx_int[keep_int]] = new_labels_int[keep_int]
                label_source[idx_int[keep_int]] = 1

        if len(boundary_specs_iter) > 0:
            _logger.info(f"  Boundary specs ({len(boundary_specs_iter)}): high-budget flow + k-NN fallback")
            new_labels_bnd = teacher_labeler.label_batch(
                boundary_specs_iter,
                n_candidates=N_CANDIDATES_BOUNDARY,
                valid_threshold=0.5,
                top_k=1,
                verbose=False,
            )
            keep_bnd = filter_by_zig_validity(
                new_labels_bnd.squeeze(), zig_model, scaler_X, threshold=0.5, device=DEVICE
            )
            idx_bnd = np.where(boundary_mask_iter)[0]
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

        if len(final_specs) > 0:
            log_label_quality_summary("  DAgger labels", final_specs, final_labels, zig_model, scaler_X, scaler_y_p, device=DEVICE)
            distillation_dataset.append_samples(final_specs, final_labels)
            _logger.info(f"  Added {len(final_labels)} labeled failures.  Dataset now: "
                         f"{len(distillation_dataset)}")

    _logger.info("DAgger training complete")

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
    plt.close(fig)
    _logger.info("DAgger training curves saved.")

    df_eval = df.sample(min(1000, len(df)), random_state=42)
    specs_eval = df_eval[['power', 'stage_2_eye_max_height',
                          'stage_2_eye_max_width', 'stage_2_jitter']].values.astype(np.float32)
    specs_eval = specs_eval[:, [0, 3, 1, 2]]
    evaluator = StudentEvaluator(
        student=student, scaler_X=scaler_X, zig_model=zig_model,
        eye_scale_h=eye_scale_h, eye_scale_w=eye_scale_w, eye_scale_j=eye_scale_j,
        scaler_y_p=scaler_y_p, device=DEVICE,
    )
    test_failure_mask, test_metrics = evaluator.identify_failures(specs_eval, threshold=ERROR_THRESHOLD)
    test_failure_rate = test_failure_mask.mean()
    _logger.info(f"Test failure rate: {test_failure_rate*100:.2f}%  "
                 f"({test_failure_mask.sum()}/{len(specs_eval)} specs)")
    if 'boundary_failure' in test_metrics and 'is_boundary' in test_metrics:
        boundary_mask_eval = test_metrics['is_boundary']
        interior_mask_eval = ~boundary_mask_eval
        boundary_rate = test_metrics['boundary_failure'].sum() / max(1, boundary_mask_eval.sum())
        interior_rate = test_metrics['interior_failure'].sum() / max(1, interior_mask_eval.sum())
        _logger.info(f"  boundary: {boundary_rate*100:.2f}%  "
                     f"interior: {interior_rate*100:.2f}%")
    _logger.info(f"  Mean errors - power: {test_metrics['errors'][:,0].mean():.4f}, "
                 f"jitter: {test_metrics['errors'][:,1].mean():.4f}, "
                 f"height: {test_metrics['errors'][:,2].mean():.4f}, "
                 f"width: {test_metrics['errors'][:,3].mean():.4f}")

    torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, f'{model_name}.pt'))
    joblib.dump(scaler_X, os.path.join(OUTPUT_DIR, 'scaler_X.pkl'))
    joblib.dump(ctx['flow_scaler_C'], os.path.join(OUTPUT_DIR, 'flow_scaler_C.pkl'))
    _logger.info(f"Saved student + scalers to {OUTPUT_DIR}")

    student.eval()
    test_specs = df_eval[['power', 'stage_2_jitter',
                           'stage_2_eye_max_height', 'stage_2_eye_max_width']].values.astype(np.float32)
    test_specs_t = torch.from_numpy(test_specs).to(DEVICE)
    pred_dict = student.predict(test_specs_t)
    pred_params = np.stack([pred_dict[name].cpu().detach().numpy()
                             for name in PARAM_COLS], axis=1)
    pred_specs_zig = test_metrics['pred_specs']
    target_specs_zig = test_metrics['target_specs']
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
    csv_path = os.path.join(OUTPUT_DIR, f'{model_name}_predictions.csv')
    df_csv.to_csv(csv_path, index=False)
    _logger.info(f"Test predictions CSV saved to {csv_path}")

    LABELS = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD']
    target_specs_demo = np.array([
        [0.01044, 37.91, 21.14, 62.43],
        [0.00709, 11.45, 33.33, 88.68],
        [0.01107, 8.79,  29.37, 91.69],
        [0.00607, 16.59, 60.29, 83.90],
        [0.00229, 29.61,  4.44, 70.82],
        [0.00235, 27.55, 20.23, 72.58],
    ])
    _logger.info("="*80)
    _logger.info(f"{'Target Specs':^40} | {'Student Predicted Parameters (7 CTO parameters)':^38}")
    with torch.no_grad():
        for spec in target_specs_demo:
            power, jitter, height, width = spec
            spec_tensor = torch.tensor([[power, jitter, height, width]], dtype=torch.float32).to(DEVICE)
            pred_dict = student.predict(spec_tensor)
            pred_values = [pred_dict[name].item() for name in LABELS]
            _logger.info(f"{power:>8.5f} {jitter:>8.3f} {height:>8.3f} {width:>8.3f} | "
                         + " ".join(f"{v:>12.4e}" for v in pred_values))
    _logger.info("="*80)

    return {
        'model_name': model_name,
        'student': student,
        'dagger_history': dagger_history,
        'loss_history': loss_history,
        'test_failure_rate': float(test_failure_rate),
        'boundary_rate': float(boundary_rate) if 'boundary_rate' in dir() else None,
        'interior_rate': float(interior_rate) if 'interior_rate' in dir() else None,
        'output_dir': OUTPUT_DIR,
        'predictions_csv': csv_path,
        'test_metrics': test_metrics,
        'params_count': sum(p.numel() for p in student.parameters() if p.requires_grad),
    }
