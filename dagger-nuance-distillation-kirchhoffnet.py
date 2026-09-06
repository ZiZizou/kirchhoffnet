"""CTLE Inverse Design: KirchhoffNet Student (DAgger)

Standalone training script (originally a jupytext `.py` of a notebook).
Run directly: ``python dagger-nuance-distillation-kirchhoffnet.py`` after
editing the path constants below to point at local teacher artifacts and
data.

Goal: train an ODE-based KirchhoffNet student that learns the inverse
design policy from the teacher (RealNVP flow + ZIG critic) using the full
DAgger distillation loop.

This variant uses the LOCAL KirchhoffNet codebase (an ODE-based fabric
built via ``topology.build_net_from_config``), wrapped to provide the same
``forward`` / ``predict`` / ``log_lo`` / ``log_hi`` API as the MLP student
so the DAgger loop, RegimeAwareLoss, and StudentEvaluator work unchanged.

Student architecture (configured via the ``KN_*`` constants below):
- input scaling: per-dim log10 + min-max -> [-1, +1] (computed from the
  empirical df with a 5% margin), rail-clamped to [-KN_INPUT_RAIL, +KN_INPUT_RAIL]
  (= [-4, 4], matching the local x_max=4.0).
- fabric: 3-stage KirchhoffNetWithIO built with small_world (k=4, p=0.2)
  hidden topology, 23 hidden nodes per stage, edge_repeats=2, boundary
  fan-out map tying each input terminal to specific hidden nodes,
  temporal-readout appending 7 output ODE accumulators per stage.
- circuit: FreeTanhLibrary edges per stage, non-programmable leak,
  freeze_read, interstage activation "residual-relu-tanh".
- output: gain/bias + tanh/relu residual branches of the 7 output-ODE
  node voltages -> bounded via sigmoid + log_lo/log_hi buffers.

Key design:
- LocalKirchhoffStudentWrapper provides the MLP-compatible API
  (forward / predict / log_lo / log_hi / scale_input /
  get_bounded_output / count_params_by_component).
- Regime-aware loss preserved: imitation + ZIG forward-consistency +
  physics + invalid + manifold.
- Adaptive teacher labeling (tiered candidate budget).
- Huber imitation + ZIG forward-consistency + physics loss.
"""

import os
import sys
import json
import time
import math
import logging
import subprocess
import hashlib
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths to edit for local runs (these default to Kaggle-mount paths).
TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'
DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'
OUTPUT_DIR = '/home/annaik/Documents/dagger_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Checkpoint / resume helpers.
#
# Two complementary mechanisms:
# 1. ``dagger_checkpoint.pt`` — a rolling, atomic checkpoint saved at the
#    baseline (after initial dataset build) and at the end of every DAgger
#    iteration. Contains full dataset, student/optimizer/scheduler state,
#    histories, and RNG state. On startup it's loaded and the run resumes at
#    ``dagger_iter = dagger_iter + 1`` from the latest checkpoint.
# 2. ``initial_dataset_<hash>.pkl`` — a joblib cache of the ~20k-sample
#    initial distillation dataset, keyed by a hash of the labeling-relevant
#    hyperparams + df fingerprint. Consulted only on a *fresh* start (no
#    checkpoint) so the most expensive single step is never re-run.
#
# A crash mid-iteration loses at most that one iteration (per the
# iteration-end-only granularity choice); the dataset is only mutated at
# iteration end, so there is no double-append risk on resume.
# ----------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'dagger_checkpoint.pt')
_CKPT_VERSION = 1


def _config_hash():
    """sha256 over the labeling-relevant hyperparams + df fingerprint.

    Hash inputs include the dataset-length bounds so a stale cache can never
    be silently reused after the underlying historical CSV changes.
    """
    try:
        df_n = int(len(df))
    except Exception:
        df_n = -1
    try:
        spec_cols = ['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']
        if 'df' in globals() and df_n > 0:
            spec_log = np.log10(np.clip(df[spec_cols].values, 1e-12, None))
            spec_min = np.round(spec_log.min(axis=0), 6).tolist()
            spec_max = np.round(spec_log.max(axis=0), 6).tolist()
        else:
            spec_min, spec_max = [], []
    except Exception:
        spec_min, spec_max = [], []
    payload = {
        'N_EMPIRICAL_SAMPLES': N_EMPIRICAL_SAMPLES,
        'N_CANDIDATES_INITIAL': N_CANDIDATES_INITIAL,
        'BOUNDARY_RATIO': BOUNDARY_RATIO,
        'N_CANDIDATES_PER_SPEC': N_CANDIDATES_PER_SPEC,
        'N_CANDIDATES_BOUNDARY': N_CANDIDATES_BOUNDARY,
        'np_seed': _NP_SEED,
        'torch_seed': _TORCH_SEED,
        'PARAM_COLS': list(PARAM_COLS),
        'df_n': df_n,
        'spec_min': spec_min,
        'spec_max': spec_max,
        'TEACHER_DIR': TEACHER_DIR,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()
    return h[:16]


def _initial_dataset_cache_path():
    cache_dir = INITIAL_DATASET_CACHE_DIR or OUTPUT_DIR
    return os.path.join(cache_dir, f'initial_dataset_{_config_hash()}.pkl')


def load_checkpoint():
    """Load the latest checkpoint, returning ``None`` if missing or corrupt."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        return torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to load checkpoint at {CHECKPOINT_PATH}: {e}; ignoring")
        return None


def save_checkpoint(ckpt):
    """Atomically save checkpoint to ``CHECKPOINT_PATH`` (tmp + replace)."""
    tmp = CHECKPOINT_PATH + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, CHECKPOINT_PATH)


def _capture_rng():
    states = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            states['torch_cuda'] = torch.cuda.get_rng_state_all()
        except Exception:
            states['torch_cuda'] = None
    else:
        states['torch_cuda'] = None
    return states


def _restore_rng(states):
    if states is None:
        return
    try:
        np.random.set_state(states['np'])
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to restore numpy RNG state: {e}")
    try:
        torch.random.set_rng_state(states['torch'])
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to restore torch RNG state: {e}")
    cuda_states = states.get('torch_cuda')
    if cuda_states is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            _logger.warning(f"[CKPT] Failed to restore cuda RNG state: {e}")


def _ckpt_baseline_payload(dagger_iter=0, converged=False):
    """Build a baseline checkpoint dict (called after initial build + split)."""
    payload = {
        'format_version': _CKPT_VERSION,
        'dagger_iter': int(dagger_iter),
        'converged': bool(converged),
        'student_state': student.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'dataset': list(distillation_dataset.data),
        'train_indices': list(distillation_dataset._train_indices) if distillation_dataset._train_indices is not None else None,
        'val_indices': list(distillation_dataset._val_indices) if distillation_dataset._val_indices is not None else None,
        'hard_buffer_start_idx': int(distillation_dataset._hard_buffer_start_idx),
        'hard_indices': list(distillation_dataset._hard_indices),
        'dagger_history': {k: list(v) for k, v in dagger_history.items()},
        'loss_history': {k: list(v) for k, v in loss_history.items()},
        'rng': _capture_rng(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if globals().get('COMMON_EVAL_SPECS') is not None:
        payload['common_eval_specs'] = np.asarray(globals()['COMMON_EVAL_SPECS']).tolist()
    return payload


def _restore_dataset_from_ckpt(ckpt):
    """Rebuild the DistillationDataset from a checkpoint and restore indices."""
    distillation_dataset.data = list(ckpt['dataset'])
    distillation_dataset._train_indices = list(ckpt['train_indices']) if ckpt['train_indices'] is not None else None
    distillation_dataset._val_indices = list(ckpt['val_indices']) if ckpt['val_indices'] is not None else None
    distillation_dataset._hard_buffer_start_idx = int(ckpt['hard_buffer_start_idx'])
    distillation_dataset._hard_indices = list(ckpt['hard_indices'])
    distillation_dataset._loader = None
    distillation_dataset._val_loader = None


def _restore_histories_from_ckpt(ckpt):
    """Restore the dagger_history / loss_history dicts from a checkpoint.

    Only updates keys that are present in the checkpoint payload so a
    newer schema (extra metric) still gets its empty list from the
    module-level initializers.
    """
    for hist_name, target in (('dagger_history', dagger_history), ('loss_history', loss_history)):
        saved = ckpt.get(hist_name)
        if not isinstance(saved, dict):
            continue
        for k, v in saved.items():
            target[k] = list(v)


def _restore_common_eval_specs_from_ckpt(ckpt):
    """Restore the shared COMMON_EVAL_SPECS ndarray from a checkpoint, if present.

    If the checkpoint has no ``common_eval_specs`` field (older schema or a
    pre-existing checkpoint from before this change), leave the module-level
    ``COMMON_EVAL_SPECS`` as the freshly-built value so the resume still
    functions; only override when the field is present and well-formed.
    """
    if 'COMMON_EVAL_SPECS' not in globals():
        return
    saved = ckpt.get('common_eval_specs')
    if saved is None:
        return
    try:
        restored = np.asarray(saved, dtype=np.float32)
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to restore common_eval_specs: {e}; keeping rebuilt set")
        return
    if restored.ndim != 2 or restored.shape[1] != 4:
        _logger.warning(f"[CKPT] Restored common_eval_specs has unexpected shape "
                        f"{restored.shape}; keeping rebuilt set")
        return
    globals()['COMMON_EVAL_SPECS'] = restored
    _logger.info(f"[CKPT] Restored COMMON_EVAL_SPECS from checkpoint: "
                 f"shape={restored.shape}")

# Make the local KirchhoffNet codebase importable. This file lives in the
# repo root, so adding the script directory is sufficient.
_LOCAL_KIRCHHOFF_DIR = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_KIRCHHOFF_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_KIRCHHOFF_DIR)

from config import SOLVER, VCA
from cell_library import make_cell_library
from topology import build_net_from_config
from kirchhoff_net import format_parameter_breakdown



_NP_SEED = 42
_TORCH_SEED = 42

np.random.seed(_NP_SEED)
torch.manual_seed(_TORCH_SEED)

DEVICE = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FMT = "%H:%M:%S"


class _Tee:
    """File-like stream that mirrors writes to an original stream and a log
    file handle, flushing after every write so the log grows continuously
    for live monitoring (e.g. `tail -f` / `Get-Content -Wait`).

    Cross-platform: the log file is always UTF-8; console writes are
    encoding-safe (replace on failure) so Windows cp1252 consoles and
    redirected pipes never raise for non-ASCII chars (e.g. the `─`
    literals at lines 1322 / 2197-2199 of this script).
    """

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, message):
        try:
            self._stream.write(message)
        except UnicodeEncodeError:
            enc = getattr(self._stream, "encoding", "ascii") or "ascii"
            self._stream.write(
                message.encode(enc, errors="replace").decode(enc, errors="replace")
            )
        try:
            self._log_fh.write(message)
            self._log_fh.flush()
        except (ValueError, OSError):
            pass
        try:
            self._stream.flush()
        except (ValueError, OSError, AttributeError):
            pass

    def flush(self):
        try:
            self._log_fh.flush()
        except (ValueError, OSError):
            pass
        try:
            self._stream.flush()
        except (ValueError, OSError, AttributeError):
            pass

    def isatty(self):
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    class _Buffer:
        def __init__(self, tee):
            self._tee = tee

        def write(self, data):
            if isinstance(data, (bytes, bytearray)):
                try:
                    msg = data.decode("utf-8", errors="replace")
                except Exception:
                    msg = data.decode("ascii", errors="replace")
            else:
                msg = data
            self._tee.write(msg)
            return len(data)

        def flush(self):
            self._tee.flush()

    @property
    def buffer(self):
        return self._Buffer(self)


_LOG_FILE = os.path.join(OUTPUT_DIR, "dagger_training.log")
_log_fh = open(_LOG_FILE, "a", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FMT,
    datefmt=_DATE_FMT,
    force=True,
    stream=sys.stderr,
)
_logger = logging.getLogger("ctle_dagger_kirchhoff")

# Run-header banner: marks each run's block in the appended log file.
try:
    _git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
    ).decode("utf-8", errors="replace").strip()
except Exception:
    _git_commit = None

_logger.info("=" * 80)
_logger.info(
    f"=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} "
    f"(pid {os.getpid()}, platform={sys.platform}) ==="
)
if _git_commit:
    _logger.info(f"git_commit={_git_commit}")
_logger.info(f"log_file={_LOG_FILE}")
_logger.info("=" * 80)

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


def _log_hyperparameters():
    """Emit a full per-run hyperparameter dump to the log file.

    Both human-readable ``[CFG] name=value`` lines and one machine-readable
    ``[CFG_JSON] {...}`` line are written so runs in the appended log file
    can be eyeballed and programmatically diffed. Captures KN_INPUT_LOG_MIN/MAX
    at call time (after the df-driven rebind) and the regime-aware loss
    weights promoted to module-level constants.
    """
    _HYPERPARAMS = [
        # --- Paths / runtime context ---
        ("TEACHER_DIR", TEACHER_DIR),
        ("DATA_DIR", DATA_DIR),
        ("OUTPUT_DIR", OUTPUT_DIR),
        ("DEVICE", str(DEVICE)),
        ("dataset_rows", len(globals()["df"])) if "df" in globals() else ("dataset_rows", "n/a"),
        # --- DAgger / training knobs ---
        ("DAGGER_ITERATIONS", DAGGER_ITERATIONS),
        ("EPOCHS_PER_ITER", EPOCHS_PER_ITER),
        ("BATCH_SIZE", BATCH_SIZE),
        ("ERROR_THRESHOLD", ERROR_THRESHOLD),
        ("VALIDATION_SIZE", VALIDATION_SIZE),
        ("BOUNDARY_RATIO", BOUNDARY_RATIO),
        ("N_CANDIDATES_PER_SPEC", N_CANDIDATES_PER_SPEC),
        ("N_CANDIDATES_BOUNDARY", N_CANDIDATES_BOUNDARY),
        ("VALIDITY_THRESHOLD", VALIDITY_THRESHOLD),
        ("DEGRADE_REL_THRESHOLD", DEGRADE_REL_THRESHOLD),
        ("MIN_DEGRADED_DIMS", MIN_DEGRADED_DIMS),
        ("LR_INITIAL", LR_INITIAL),
        ("LR_DECAY_AFTER_ITER", LR_DECAY_AFTER_ITER),
        ("LR_FLOOR", LR_FLOOR),
        ("FAILURE_CAP_RATIO", FAILURE_CAP_RATIO),
        ("CONVERGENCE_THRESHOLD", CONVERGENCE_THRESHOLD),
        ("EARLYSTOP_EVAL_EVERY", EARLYSTOP_EVAL_EVERY),
        ("EARLYSTOP_LOG_EVERY", EARLYSTOP_LOG_EVERY),
        ("EARLYSTOP_SKIP_EPOCHS", EARLYSTOP_SKIP_EPOCHS),
        ("EARLYSTOP_PATIENCE_EPOCHS", EARLYSTOP_PATIENCE_EPOCHS),
        ("COMMON_EVAL_SIZE", COMMON_EVAL_SIZE),
        ("COMMON_EVAL_SEED", COMMON_EVAL_SEED),
        ("MIN_FAILURE_IMPROVEMENT", MIN_FAILURE_IMPROVEMENT),
        ("DIVERGENCE_ABORT", DIVERGENCE_ABORT),
        ("DIVERGENCE_MARGIN", DIVERGENCE_MARGIN),
        ("DIVERGENCE_CONSEC_EVALS", DIVERGENCE_CONSEC_EVALS),
        ("N_EMPIRICAL_SAMPLES", N_EMPIRICAL_SAMPLES),
        ("N_CANDIDATES_INITIAL", N_CANDIDATES_INITIAL),
        ("WEIGHT_DECAY", WEIGHT_DECAY),
        ("LOSS_WEIGHT_EMPIRIC", LOSS_WEIGHT_EMPIRIC),
        ("HARD_BUFFER_WEIGHT", HARD_BUFFER_WEIGHT),
        ("BOUNDARY_ABS_TOLERANCES", BOUNDARY_ABS_TOLERANCES),
        ("BOUNDARY_ABS_TOLERANCES_RELAXED", BOUNDARY_ABS_TOLERANCES_RELAXED),
        # --- Regime-aware loss weights ---
        ("ALPHA_SPEC", ALPHA_SPEC),
        ("BETA_PHYS", BETA_PHYS),
        ("GAMMA_MONO", GAMMA_MONO),
        ("ALPHA_INVALID", ALPHA_INVALID),
        ("K_MANIFOLD", K_MANIFOLD),
        ("ALPHA_MANIFOLD", ALPHA_MANIFOLD),
        ("WARMUP_EPOCHS", WARMUP_EPOCHS),
        ("PER_TARGET_HURDLE", PER_TARGET_HURDLE),
        # --- Local KirchhoffNet topology ---
        ("KN_NUM_STAGES", KN_NUM_STAGES),
        ("KN_NUM_HIDDEN", KN_NUM_HIDDEN),
        ("KN_SMALL_WORLD_K", KN_SMALL_WORLD_K),
        ("KN_SMALL_WORLD_P", KN_SMALL_WORLD_P),
        ("KN_SMALL_WORLD_SEED", KN_SMALL_WORLD_SEED),
        ("KN_EDGE_REPEATS", KN_EDGE_REPEATS),
        ("KN_CELL_LIBRARY", KN_CELL_LIBRARY),
        ("KN_LEAK_MODE", KN_LEAK_MODE),
        ("KN_INTERSTAGE_ACTIVATION", KN_INTERSTAGE_ACTIVATION),
        ("KN_FREEZE_READ", KN_FREEZE_READ),
        ("KN_TEMPORAL_READOUT", KN_TEMPORAL_READOUT),
        ("KN_BOUNDARY_FAN_OUT", KN_BOUNDARY_FAN_OUT),
        ("KN_INPUT_RAIL", KN_INPUT_RAIL),
        ("KN_X_MAX", KN_X_MAX),
        ("KN_VCA", KN_VCA),
        ("KN_VCA_RANK", KN_VCA_RANK if KN_VCA_RANK is not None else int(VCA["rank"])),
        ("KN_VCA_CORE", KN_VCA_CORE),
        ("KN_VCA_GATE_SHUNT", KN_VCA_GATE_SHUNT),
        ("KN_VCA_SEPARATE_CORE_BUS", KN_VCA_SEPARATE_CORE_BUS),
        ("KN_INPUT_LOG_PAD_FRAC", KN_INPUT_LOG_PAD_FRAC),
        ("KN_INPUT_LOG_MIN", KN_INPUT_LOG_MIN),
        ("KN_INPUT_LOG_MAX", KN_INPUT_LOG_MAX),
        ("LOG_BOUNDS", LOG_BOUNDS),
    ]

    def _coerce(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    _logger.info("-" * 80)
    _logger.info("[CFG] Run hyperparameters:")
    _cfg_dict = {}
    for _name, _val in _HYPERPARAMS:
        _val_coerced = _coerce(_val)
        _cfg_dict[_name] = _val_coerced
        try:
            _val_repr = json.dumps(_val_coerced, default=str)
        except (TypeError, ValueError):
            _val_repr = repr(_val_coerced)
        _logger.info(f"[CFG] {_name}={_val_repr}")
    _logger.info("[CFG_JSON] " + json.dumps(_cfg_dict, default=str))
    _logger.info("-" * 80)


_logger.info(f"Using device: {DEVICE}")


"""
===========================================================
HYPERPARAMETERS — edit all DAgger / training knobs here
===========================================================
"""
DAGGER_ITERATIONS     = 10        # max DAgger loop iterations
EPOCHS_PER_ITER       = 200       # student training epochs per iteration
BATCH_SIZE            = 256      # DataLoader batch size
INITIAL_DATASET_CACHE_DIR = None  # optional shared cache for repeated BO trials
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
MAPPER_LR_SCALE       = 1.0      # I/O mapper + OutputAffine readout (base * 1.0)
STRUCT_LR_SCALE       = 4.0      # structural core params: name.endswith('.z_logits')
DYN_LR_SCALE          = 1.0      # sensitive dynamical params: raw_leak, raw_drive_g
FAILURE_CAP_RATIO     = 0.40     # max new failures / existing dataset size per iter
CONVERGENCE_THRESHOLD = 0.02     # early-stop if failure_rate < 2%
# Best-model selection (plan dagger_knet_best_model_selection_improvement):
# A SINGLE shared eval set is built once (COMMON_EVAL_SPECS) and used for both
# per-epoch early-stop tracking AND the per-iteration "Validation failure rate"
# report. Every iteration and every resume uses the *same* spec set, so the
# per-iteration failure rates are directly comparable and monotone-improvement
# can be enforced, not assumed.
COMMON_EVAL_SIZE      = 2000      # size of the shared eval set
COMMON_EVAL_SEED      = 24681012  # fixed seed; identical set across iterations + resumes
EARLYSTOP_EVAL_EVERY  = 1         # evaluate failure rate every N epochs (1 = every epoch)
EARLYSTOP_LOG_EVERY   = 10        # emit verbose earlystop log every N epochs
EARLYSTOP_SKIP_EPOCHS = 5         # no best-tracking during the first N epochs of an
                                  # iteration (prevents warm-started epoch-1 snapshot
                                  # from winning immediately)
EARLYSTOP_PATIENCE_EPOCHS = EPOCHS_PER_ITER  # effectively disables in-budget early
                                              # stopping; loop always runs full budget
MIN_FAILURE_IMPROVEMENT = 0.01    # absolute (1%) improvement required to accept a
                                  # new within-iteration best over prev_failure_rate
# Divergence abort (safety valve for the full-budget policy above): if training
# catastrophically diverges (failure rate explodes far ABOVE the previous
# iteration's baseline) for several consecutive evals, the rest of the budget
# would be wasted — abort the iteration and let the carry-forward guardrail
# restore the best/previous model. Never fires on normal noise: it requires a
# large sustained REGRESSION, not merely a failure to improve.
DIVERGENCE_ABORT          = True  # enable the mid-iteration divergence abort
DIVERGENCE_MARGIN         = 0.20  # absolute failure-rate INCREASE over prev (20 pts)
DIVERGENCE_CONSEC_EVALS   = 5     # consecutive above-margin evals before aborting

# Module-level slot for the shared eval set (built lazily below the DAgger
# hyperparameter block, before the training loop). Declared here so the
# resume-from-checkpoint helper can overwrite it as soon as ckpt is loaded.
COMMON_EVAL_SPECS = None
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

# ---------------------------------------------------------------
# Regime-aware loss weights (used by RegimeAwareLoss + RampedRegimeLoss)
# ---------------------------------------------------------------
ALPHA_SPEC      = 1.5    # weight on L_spec (target-spec regression)
BETA_PHYS       = 0.1    # weight on L_phys (power / RC physics priors)
GAMMA_MONO      = 0.01   # weight on L_mono (monotonicity penalty)
ALPHA_INVALID   = 1.5    # weight on L_invalid (invalid-zone penalty)
K_MANIFOLD      = 5      # k-NN neighbor count for manifold regularization
ALPHA_MANIFOLD  = 0.1    # weight on L_manifold (k-NN empirical pull)
WARMUP_EPOCHS   = 5      # epochs over which L_spec/L_phys/L_invalid ramp from 0

# ---------------------------------------------------------------
# Local KirchhoffNet student topology
# Replicates the friedman2 train_script.py preset with num_stages and
# num_hidden tuned so the param count lands in the 5e3-6e3 target range.
# ---------------------------------------------------------------
KN_NUM_STAGES       = 4       # ODE circuit stages (3 tunes param count to ~5755 with out_dim=7)
KN_NUM_HIDDEN       = 14      # hidden nodes per stage (>= max boundary fan-out target)
KN_SMALL_WORLD_K    = 4       # Watts-Strogatz ring-lattice degree
KN_SMALL_WORLD_P    = 0.2     # Watts-Strogatz rewiring probability
KN_SMALL_WORLD_SEED = 1       # RNG seed for rewiring
KN_EDGE_REPEATS     = 2       # parallel edges per unique hidden pair
KN_CELL_LIBRARY     = "tanh_free"   # FreeTanhLibrary (per-edge tanh OTA)
KN_LEAK_MODE        = "non-programmable"  # fixed leak_constant (config default 0.0486)
KN_INTERSTAGE_ACTIVATION = "residual-relu-tanh"  # interstage StageTransfer activation
KN_FREEZE_READ          = True      # precompute edge currents once from state (frozen OTA read core)
KN_TEMPORAL_READOUT     = True      # append per-stage output ODE accumulators (temporal-readout readout)
KN_BOUNDARY_FAN_OUT = '{"0": [2, 4], "1": [1, 3], "2": [12, 5], "3": [7, 9]}'  # 4 inputs -> 8 boundary edges
KN_INPUT_RAIL       = 4.0     # clamp normalized input to [-KN_INPUT_RAIL, KN_INPUT_RAIL] (x_max=4.0)

KN_X_MAX            = 4.0     # local knet rail (overrides PHYS["x_max"]=3.0 via build_net_from_config x_max kwarg)
KN_GM_MAX           = None    # cell-library gm_max override (None -> config default 10.0); set via --kn-gm-max
KN_ISAT_MAX         = None    # cell-library isat_max override (None -> config default 10.0); set via --kn-isat-max

# VCA (Voltage-Controlled Amplifier) gating knobs — mirrored from train_script.py
# argparse flags so the DAgger student matches the canonical train.py surface.
# Identity-at-init is preserved: with --vca on, vca_W is zero-init and the
# 2-sigma gate evaluates to 1.0 at step 0, so the VCA-off baseline behavior is
# reproduced bit-for-bit at epoch 0 regardless of the chosen flags.
KN_VCA                   = True  # --vca: enable low-rank input-driven VCA on boundary / temporal-readout edges
KN_VCA_CORE              = True  # --vca-core: also gate core (hidden) edges (auto-enabled if no boundary / readout families)
KN_VCA_BIAS              = bool(VCA.get("bias", False))  # --vca-bias: affine per-edge offsets
KN_VCA_GATE_SHUNT        = False  # --vca-gate-shunt: also gate parallel resistive shunt with the core VCA gate (input-dependent routing)
KN_VCA_SEPARATE_CORE_BUS = True  # --vca-separate-core-bus: give core family its own vca_W_core projection (ablation)
KN_VCA_RANK              = 2   # projection rank r; None -> config.VCA['rank'] (must be >= VCA['min_rank'])

# Fallback per-dim log10 spec bounds used when historical data is unavailable.
# Order: [power, jitter, height, width]. Rebound from df after data load below.
KN_INPUT_LOG_MIN_DEFAULT = np.array([np.log10(1e-4), np.log10(1.0), np.log10(0.5), np.log10(1.0)], dtype=np.float32)
KN_INPUT_LOG_MAX_DEFAULT = np.array([np.log10(1e-1), np.log10(1e3), np.log10(2e2), np.log10(5e2)], dtype=np.float32)
KN_INPUT_LOG_MIN = KN_INPUT_LOG_MIN_DEFAULT
KN_INPUT_LOG_MAX = KN_INPUT_LOG_MAX_DEFAULT
KN_INPUT_LOG_PAD_FRAC = 0.05  # widen each bound by this fraction of the dim range


LOG_BOUNDS = [-15.0, 8]

# ── BO overrides for CTLE 4×100 proxy (kn_bayes_opt.py) ─────────────────────
# Allows `python dagger-nuance-distillation-kirchhoffnet.py --dagger-iterations 4
# --epochs-per-iter 100 ...` to override hyperparams without editing the file.
# Unknown args are ignored so normal `python script.py` still uses defaults.
# Default BEFORE the BO-override try block so a parse failure inside try
# can never leave this undefined for the checkpoint gate (~line 3059).
# Audit fix (phase-a-fixes follow-up): previously defined only inside try.
_phase_a_requested = False
try:
    import argparse as _argparse
    _bo_parser = _argparse.ArgumentParser(add_help=False)
    _bo_parser.add_argument('--dagger-iterations', type=int, default=None)
    _bo_parser.add_argument('--epochs-per-iter', type=int, default=None)
    _bo_parser.add_argument('--common-eval-size', type=int, default=None)
    _bo_parser.add_argument('--common-eval-seed', type=int, default=None)
    _bo_parser.add_argument('--earlystop-eval-every', type=int, default=None)
    _bo_parser.add_argument('--batch-size', type=int, default=None)
    _bo_parser.add_argument('--initial-dataset-cache-dir', type=str, default=None)
    _bo_parser.add_argument('--count-params-only', action='store_true',
                            help='Construct the student, print trainable parameter count, and exit.')
    _bo_parser.add_argument('--kn-num-stages', type=int, default=None)
    _bo_parser.add_argument('--kn-num-hidden', type=int, default=None)
    _bo_parser.add_argument('--kn-small-world-k', type=int, default=None)
    _bo_parser.add_argument('--kn-small-world-p', type=float, default=None)
    _bo_parser.add_argument('--kn-vca-rank', type=int, default=None)
    _bo_parser.add_argument('--vca-bias', action='store_true', default=False,
                            help='Enable affine per-edge VCA biases.')
    _bo_parser.add_argument('--kn-x-max', type=float, default=None)
    _bo_parser.add_argument('--kn-gm-max', type=float, default=None)
    _bo_parser.add_argument('--kn-isat-max', type=float, default=None)
    _bo_parser.add_argument('--t-span', type=float, default=None)
    _bo_parser.add_argument('--boundary-fan-out', type=str, default=None)
    _bo_parser.add_argument('--lr', type=float, default=None)
    _bo_parser.add_argument('--weight-decay', type=float, default=None)
    _bo_parser.add_argument('--output', type=str, default=None)
    _bo_parser.add_argument('--device', type=str, default=None)
    _bo_parser.add_argument('--data-dir', type=str, default=None)
    _bo_parser.add_argument('--teacher-dir', type=str, default=None)
    _bo_parser.add_argument('--seed', type=int, default=None)
    # Canonical Phase-A flags (plan canonical-ctle-unify).  When --canonical-dataset
    # is passed, the script bypasses DAgger and runs the static-data, Huber-only
    # Phase-A training mode (no relabeling, no surrogate-through terms).  When
    # --prepare-canonical-dataset is passed, only the canonical .npz is built
    # and the script exits.
    from pathlib import Path  # local import keeps the top-of-file imports tidy
    _bo_parser.add_argument('--canonical-dataset', type=str, default=None,
                            help='Path to canonical Phase-A .npz (specs + flow labels). '
                                 'When provided, DAgger is bypassed and run_phase_a_training is used.')
    _bo_parser.add_argument('--prepare-canonical-dataset', type=str, default=None,
                            help='Path to write the canonical Phase-A .npz and exit (no training).')
    _bo_parser.add_argument('--phase-a-epochs', type=int, default=None,
                            help='Phase-A training epochs (overrides --epochs-per-iter when in Phase A).')
    _bo_parser.add_argument('--phase-a-boundary-ratio', type=float, default=0.5,
                            help='Boundary ratio used when preparing the canonical dataset (default 0.5).')
    _bo_parser.add_argument('--phase-a-n-samples', type=int, default=20000,
                            help='Number of specs in the canonical dataset (default 20000).')
    # KNet Friedman recipe (differential LR groups + cell bounds).  All are
    # no-ops when left at their default — single AdamW group with the base LR
    # is used, matching the previous MLP/Plain student behavior.
    _bo_parser.add_argument('--kn-mapper-lr-scale', type=float, default=1.0,
                            help='Multiplier on the LR for input_mapper / output_mapper / '
                                 'post_readout_transfer groups (Friedman recipe; default 1.0).')
    _bo_parser.add_argument('--kn-struct-lr-scale', type=float, default=1.0,
                            help='Multiplier on the LR for z/u logits, raw_leak, VCA groups '
                                 '(Friedman recipe; default 1.0).')
    _bo_parser.add_argument('--kn-dyn-lr-scale', type=float, default=1.0,
                            help='Multiplier on the LR for the remaining dynamic (cell library) '
                                 'parameters (Friedman recipe; default 1.0).')
    _bo_args, _ = _bo_parser.parse_known_args()
    # Phase-A request flag (plan phase-a-fixes, bug 1): when set, the
    # checkpoint restore below MUST be skipped so a stale shared-default
    # checkpoint (e.g. /home/annaik/.../dagger_checkpoint.pt from a prior
    # 52-edge / VCA-rank-3 architecture) never silently warm-starts a
    # Phase-A BO trial of the current 56-edge / VCA-rank-2 architecture.
    _phase_a_requested = (
        _bo_args.prepare_canonical_dataset is not None
        or _bo_args.canonical_dataset is not None
    )
    if _bo_args.dagger_iterations is not None:
        DAGGER_ITERATIONS = _bo_args.dagger_iterations
    if _bo_args.epochs_per_iter is not None:
        EPOCHS_PER_ITER = _bo_args.epochs_per_iter
        # keep early-stop patience consistent with new budget
        EARLYSTOP_PATIENCE_EPOCHS = EPOCHS_PER_ITER
    if _bo_args.common_eval_size is not None:
        COMMON_EVAL_SIZE = _bo_args.common_eval_size
    if _bo_args.common_eval_seed is not None:
        COMMON_EVAL_SEED = _bo_args.common_eval_seed
    if _bo_args.earlystop_eval_every is not None:
        if _bo_args.earlystop_eval_every < 1:
            raise ValueError('--earlystop-eval-every must be >= 1')
        EARLYSTOP_EVAL_EVERY = _bo_args.earlystop_eval_every
    if _bo_args.batch_size is not None:
        if _bo_args.batch_size < 1:
            raise ValueError('--batch-size must be >= 1')
        BATCH_SIZE = _bo_args.batch_size
    if _bo_args.initial_dataset_cache_dir is not None:
        INITIAL_DATASET_CACHE_DIR = _bo_args.initial_dataset_cache_dir
        os.makedirs(INITIAL_DATASET_CACHE_DIR, exist_ok=True)
    if _bo_args.kn_num_stages is not None:
        KN_NUM_STAGES = _bo_args.kn_num_stages
    if _bo_args.kn_num_hidden is not None:
        KN_NUM_HIDDEN = _bo_args.kn_num_hidden
    if _bo_args.kn_small_world_k is not None:
        KN_SMALL_WORLD_K = _bo_args.kn_small_world_k
    if _bo_args.kn_small_world_p is not None:
        KN_SMALL_WORLD_P = _bo_args.kn_small_world_p
    if _bo_args.kn_vca_rank is not None:
        KN_VCA_RANK = _bo_args.kn_vca_rank
    if _bo_args.vca_bias:
        KN_VCA_BIAS = True
    if _bo_args.kn_x_max is not None:
        KN_X_MAX = _bo_args.kn_x_max
    if _bo_args.boundary_fan_out is not None:
        KN_BOUNDARY_FAN_OUT = _bo_args.boundary_fan_out
    if _bo_args.lr is not None:
        LR_INITIAL = _bo_args.lr
    if _bo_args.weight_decay is not None:
        WEIGHT_DECAY = _bo_args.weight_decay
    if _bo_args.output is not None:
        # OUTPUT_DIR is overridden after the module-level checkpoint path was
        # initialized.  Keep the checkpoint colocated with the requested run
        # so repeated invocations can resume one trial's DAgger prefix without
        # ever restoring another trial's state.
        OUTPUT_DIR = _bo_args.output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'dagger_checkpoint.pt')
    if _bo_args.device is not None:
        DEVICE = torch.device(_bo_args.device) if _bo_args.device != 'auto' else DEVICE
    if _bo_args.data_dir is not None:
        DATA_DIR = _bo_args.data_dir
    if _bo_args.teacher_dir is not None:
        TEACHER_DIR = _bo_args.teacher_dir
    if _bo_args.seed is not None:
        _NP_SEED = _bo_args.seed
        _TORCH_SEED = _bo_args.seed
        np.random.seed(_NP_SEED)
        torch.manual_seed(_TORCH_SEED)
    # t-span overrides SOLVER global (used in build_net_from_config via stage_t_span)
    if _bo_args.t_span is not None:
        try:
            SOLVER["t_span"] = float(_bo_args.t_span)
        except Exception:
            pass
    # kn-gm/isat overrides: forwarded into the student constructor so the
    # cell library is built with the BO-suggested bounds (param-neutral:
    # bounds change no parameter count, only the sigmoid operating range).
    if _bo_args.kn_gm_max is not None:
        KN_GM_MAX = float(_bo_args.kn_gm_max)
    if _bo_args.kn_isat_max is not None:
        KN_ISAT_MAX = float(_bo_args.kn_isat_max)
except Exception as _e:
    _logger.warning(f"[BO] override parsing failed: {_e}")


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


# ----------------------------------------------------------------------------
# Load teacher artifacts (flow + ZIG).
# ----------------------------------------------------------------------------
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

# ZIG-related code must use ZIG's own scalers, NOT flow's scalers.
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


# Load the trained HybridHurdleModel as ZIG
zig_model = HybridHurdleModel(dropout=0.0, per_target=PER_TARGET_HURDLE).to(DEVICE)
zig_model.load_state_dict(torch.load(os.path.join(TEACHER_DIR, 'hybrid_hurdle_ctle_model.pt'),
                                      map_location=DEVICE))
zig_model.eval()
for p in zig_model.parameters():
    p.requires_grad = False
_logger.info("HybridHurdleModel (ZIG) loaded from checkpoint")


# ----------------------------------------------------------------------------
# Output parameterization.
# ----------------------------------------------------------------------------
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

_logger.info("Parameter log bounds defined:")
for k, v in PARAM_LOG_BOUNDS.items():
    _logger.info(f"  {k}: [{v[0]:.2f}, {v[1]:.2f}] -> [10^{v[0]:.1f}, 10^{v[1]:.1f}]")


# =============================================================================
# STUDENT MODEL: Local KirchhoffNet wrapper
# =============================================================================
# Provides MLP-compatible API (forward / predict / log_lo / log_hi /
# get_bounded_output / scale_input / count_params_by_component) so the
# DAgger loop, RegimeAwareLoss, and StudentEvaluator work unchanged.


def _parse_boundary_fan_out(spec):
    """Parse boundary fan-out JSON spec into a {input_index: [targets...]} dict.

    Accepts either a JSON string (CLI-style) or an already-parsed dict.
    """
    if isinstance(spec, dict):
        return {int(k): list(v) for k, v in spec.items()}
    parsed = json.loads(spec)
    return {int(k): list(v) for k, v in parsed.items()}


class LocalKirchhoffStudentWrapper(nn.Module):
    """Wrapper around a local KirchhoffNetWithIO that exposes the same
    `forward` / `predict` / `log_lo` / `log_hi` / `scale_input` API used by
    the original MLP-based student. The local fabric is built via
    ``topology.build_net_from_config`` using a small-world hidden topology,
    boundary fan-out (input terminals wired to specific hidden nodes),
    temporal-readout (each stage appends ``out_dim`` output-ODE accumulators
    read out through a learnable ``OutputAffine`` layer), non-programmable
    leak, freeze-read tanh contributions, and a residual-relu-tanh
    interstage activation.

    Forward:
        specs (..., 4) raw physical units [power, jitter, height, width]
            └─ scale_input(...)                                # log10 + per-dim min-max -> [-1, +1], clamp to [-rail, rail]
                └─ KirchhoffNetWithIO (boundary-mode)          # input -> 4-dim boundary voltages
                    └─ 3 ODE stages w/ interstage activation
                        └─ OutputAffine -> logits (..., 7)
        bounded via sigmoid + log_lo / log_hi buffers
    """

    def __init__(self,
                 num_stages: int = 3,
                 num_hidden: int = 23,
                 small_world_k: int = 4,
                 small_world_p: float = 0.2,
                 small_world_seed: int = 1,
                 edge_repeats: int = 2,
                 cell_library: str = "tanh_free",
                 leak_mode: str = "non-programmable",
                 interstage_activation: str = "residual-relu-tanh",
                 freeze_read: bool = True,
                 boundary_fan_out=None,
                 enable_temporal_readout: bool = True,
                 num_targets: int = 7,
                 input_rail: float = KN_INPUT_RAIL,
                 x_max: float = KN_X_MAX,
                 param_log_bounds=None,
                 input_log_min=None,
                 input_log_max=None,
                 stage_t_span: float | None = None,
                 stage_num_steps: int | None = None,
                 seed: int | None = None,
                 vca_enabled: bool = False,
                 vca_rank: int | None = None,
                 vca_core_enabled: bool = False,
                 vca_gate_shunt: bool = False,
                 vca_separate_core_bus: bool = False,
                 vca_bias: bool = False,
                 gm_max: float | None = None,
                 isat_max: float | None = None):
        super().__init__()

        if param_log_bounds is None:
            param_log_bounds = PARAM_LOG_BOUNDS
        if input_log_min is None:
            input_log_min = KN_INPUT_LOG_MIN
        if input_log_max is None:
            input_log_max = KN_INPUT_LOG_MAX

        self.num_stages = int(num_stages)
        self.num_hidden = int(num_hidden)
        self.small_world_k = int(small_world_k)
        self.small_world_p = float(small_world_p)
        self.small_world_seed = int(small_world_seed)
        self.edge_repeats = int(edge_repeats)
        self.cell_library = cell_library
        self.leak_mode = leak_mode
        self.interstage_activation = interstage_activation
        self.freeze_read = bool(freeze_read)
        self.boundary_fan_out = _parse_boundary_fan_out(boundary_fan_out)
        self.enable_temporal_readout = bool(enable_temporal_readout)
        self.num_targets = int(num_targets)
        self.input_rail = float(input_rail)
        self.x_max = float(x_max)
        self.param_log_bounds = param_log_bounds

        self._seed = int(seed if seed is not None else small_world_seed)

        # VCA gating (mirrors train_script.py --vca / --vca-core /
        # --vca-gate-shunt / --vca-separate-core-bus / --vca-rank).
        # Rank resolution mirrors train_script.py: explicit kwarg >
        # config.VCA['rank']. Rank is only validated when vca_enabled=True
        # (matches train_script.py:3393-3398); when vca_enabled=False the
        # rank is unused and no validation is performed.
        self.vca_enabled = bool(vca_enabled)
        self.vca_rank = (
            int(vca_rank) if vca_rank is not None else int(VCA["rank"])
        )
        if self.vca_enabled and self.vca_rank < VCA["min_rank"]:
            raise ValueError(
                f"vca_rank must be >= {VCA['min_rank']}, got {self.vca_rank}"
            )
        self.vca_core_enabled = bool(vca_core_enabled)
        self.vca_gate_shunt = bool(vca_gate_shunt)
        self.vca_separate_core_bus = bool(vca_separate_core_bus)
        self.vca_bias = bool(vca_bias)
        # Cell-bound overrides (BO search dims; None -> config defaults).
        # gm_min/isat_min stay fixed at config defaults (0.01), matching
        # train_script.py and kn_bayes_opt.py conventions.
        self.gm_max = float(gm_max) if gm_max is not None else None
        self.isat_max = float(isat_max) if isat_max is not None else None


        self.register_buffer("input_log_min", torch.as_tensor(input_log_min, dtype=torch.float32))
        self.register_buffer("input_log_max", torch.as_tensor(input_log_max, dtype=torch.float32))
        # Cumulative diagnostic counters for the normalized input rail.
        self._clip_elements = 0
        self._input_elements = 0

        # Bounded-output buffers.
        self.log_lo = nn.Parameter(torch.zeros(num_targets), requires_grad=False)
        self.log_hi = nn.Parameter(torch.zeros(num_targets), requires_grad=False)
        for i, (name, (lo, hi)) in enumerate(param_log_bounds.items()):
            self.log_lo.data[i] = lo
            self.log_hi.data[i] = hi

        # Build the local KirchhoffNetWithIO via build_net_from_config.
        t_span_stage = stage_t_span if stage_t_span is not None else SOLVER["t_span"] / num_stages
        num_steps_stage = stage_num_steps if stage_num_steps is not None else max(1, int(round(SOLVER["num_steps"] / num_stages)))
        cfg = {
            "stages": [
                {
                    "num_inputs": 4,
                    "num_hidden": num_hidden,
                    "num_proj": 0,
                    "num_outputs": 0,
                    "hidden_family": "small_world",
                    "hidden_kwargs": {
                        "k": int(small_world_k),
                        "p": float(small_world_p),
                        "seed": int(small_world_seed),
                        "bidirectional": False,
                    },
                    "input_pattern": "all_to_all",
                    "output_pattern": "all_to_all",
                    "proj_pattern": "all_to_all",
                    "edge_repeats": int(edge_repeats),
                    "t_span": float(t_span_stage),
                    "num_steps": int(num_steps_stage),
                }
                for _ in range(num_stages)
            ],
            "out_dim": int(num_targets),
            "write_mode": "sparse_proj",
            "read_mode": "dense",
            "use_robust_input": False,
        }
        cell_lib_template = make_cell_library(
            cell_library, gm_max=self.gm_max, isat_max=self.isat_max,
        )
        self.net = build_net_from_config(
            cfg,
            cell_lib=cell_lib_template,
            leak_mode=leak_mode,
            freeze_read=freeze_read,
            interstage_activation=interstage_activation,
            boundary_fan_out=self.boundary_fan_out,
            enable_temporal_readout=enable_temporal_readout,

            x_max=self.x_max,
            vca_enabled=self.vca_enabled,
            vca_rank=self.vca_rank,
            vca_core_enabled=self.vca_core_enabled,
            vca_gate_shunt=self.vca_gate_shunt,
            vca_separate_core_bus=self.vca_separate_core_bus,
            vca_bias=self.vca_bias,
        )
        self._state_dim = num_hidden
        # local readout size is num_targets; the OutputAffine returns logits.
        self._output_dim = int(num_targets)

    def scale_input(self, x: torch.Tensor) -> torch.Tensor:
        """Per-dim log10 + min-max normalization for the 4 raw spec dims -> boundary voltages.

        For each of [power, jitter, height, width]:
            u_i = 2 * (log10(x_i) - input_log_min[i]) / (input_log_max[i] - input_log_min[i]) - 1
        so the empirical range maps linearly to [-1, +1] (with a small margin
        applied to ``input_log_min/max`` at data-load time). Log space preserves
        relative resolution across the wide multi-decade raw spec ranges.

        Result is rail-clamped to [-input_rail, +input_rail] (default 4.0) as a
        compliance guard so the boundary OTA gates stay well below the rail.
        """
        eps = 1e-12
        lo = self.input_log_min.to(x)
        hi = self.input_log_max.to(x)
        log_x = torch.log10(x.clamp(min=eps))
        span = (hi - lo).clamp(min=1e-8)
        u = 2.0 * (log_x - lo) / span - 1.0
        with torch.no_grad():
            self._clip_elements += int((u.abs() >= self.input_rail).sum().item())
            self._input_elements += int(u.numel())
        return u.clamp(min=-self.input_rail, max=self.input_rail)

    def clipping_stats(self) -> dict[str, float | int]:
        """Return cumulative input-rail clipping statistics for this run."""
        clipped = self._clip_elements
        total = self._input_elements
        return {
            'clipped_elements': clipped,
            'input_elements': total,
            'clip_fraction': clipped / max(1, total),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through KirchhoffNetWithIO. Returns unbounded 7 logits.

        Compatible with the MLP student API consumed by RegimeAwareLoss.
        """
        u = self.scale_input(x)
        logits, _trajs = self.net(u, store_trajectory=False, solver="heun")
        return logits

    def predict(self, x: torch.Tensor) -> dict:
        """Returns dict of physical params (10**bounded_log) for the 7 targets."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        results = {}
        for i, name in enumerate(self.param_log_bounds.keys()):
            log_val = self.log_lo[i] + (self.log_hi[i] - self.log_lo[i]) * probs[..., i]
            results[name] = torch.pow(10.0, log_val)
        return results

    def get_bounded_output(self, x: torch.Tensor):
        """Returns (logits, bounded_log (...,7), physical_params (dict))."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        bounded_log = self.log_lo.unsqueeze(0) + (self.log_hi.unsqueeze(0) - self.log_lo.unsqueeze(0)) * probs
        physical = {name: torch.pow(10.0, bounded_log[:, i])
                    for i, name in enumerate(self.param_log_bounds.keys())}
        return logits, bounded_log, physical

    def extra_repr(self):
        return (f"num_stages={self.num_stages}, num_hidden={self.num_hidden}, "
                f"cell_library={self.cell_library}, leak_mode={self.leak_mode}, "
                f"freeze_read={self.freeze_read}, "
                f"interstage_activation={self.interstage_activation}, "
                f"temporal_readout={self.enable_temporal_readout}, "
                f"vca_enabled={self.vca_enabled}, vca_rank={self.vca_rank}, "
                f"vca_core={self.vca_core_enabled}, "
                f"vca_gate_shunt={self.vca_gate_shunt}, "
                f"vca_separate_core_bus={self.vca_separate_core_bus}, "
                f"out_dim={self._output_dim}")

    def count_params_by_component(self):
        """Return ``{encoder, circuit, projector, total}`` param counts.

        Maps the local knet's :meth:`parameter_breakdown` groups onto the
        ``encoder / circuit / projector / total`` schema the DAgger
        diagnostics expect.
        """
        bd = self.net.parameter_breakdown()
        groups = bd.get("groups", {})
        total = bd.get("total", 0)
        per_stage = bd.get("per_stage", {})
        encoder = int(groups.get("input_mapper", 0))
        projector = int(groups.get("output_mapper", 0)) + int(groups.get("skip_linear", 0))
        drive = int(groups.get("drive_mappers", 0))
        circuit = int(total) - encoder - projector - drive
        return {
            "encoder": encoder,
            "circuit": circuit,
            "projector": projector,
            "total": int(total),
        }


# ----------------------------------------------------------------------------
# Load historical data to get spec distributions.
# ----------------------------------------------------------------------------
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

# Show spec distributions
for col in ['power', 'stage_2_eye_max_height', 'stage_2_eye_max_width', 'stage_2_jitter']:
    nz = df[df[col] > 0][col]
    _logger.info(f"{col}: n={len(nz)}, min={nz.min():.4f}, max={nz.max():.4f}, median={nz.median():.4f}")


# Rebind KN_INPUT_LOG_MIN/MAX from the empirical spec distributions so
# log10 + min-max -> [-1, 1] covers the full data range with small margin.
if len(df) > 0:
    _spec_log = np.log10(np.clip(
        df[['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']].values,
        1e-12, None,
    ))
    _lo = _spec_log.min(axis=0)
    _hi = _spec_log.max(axis=0)
    _pad = KN_INPUT_LOG_PAD_FRAC * np.maximum(_hi - _lo, 1e-8)
    KN_INPUT_LOG_MIN = (_lo - _pad).astype(np.float32)
    KN_INPUT_LOG_MAX = (_hi + _pad).astype(np.float32)
    for _i, _name in enumerate(['power', 'jitter', 'height', 'width']):
        _logger.info(
            f"KN input log bound [{_name}]: "
            f"[{10 ** KN_INPUT_LOG_MIN[_i]:.4e}, {10 ** KN_INPUT_LOG_MAX[_i]:.4e}] "
            f"(log10 [{KN_INPUT_LOG_MIN[_i]:.3f}, {KN_INPUT_LOG_MAX[_i]:.3f}])"
        )


# ----------------------------------------------------------------------------
# Build distillation dataset with adaptive labeling.
#
# The flow generates diverse candidates for each target spec, ZIG ranks them,
# and the best (canonical) candidate becomes the distillation label. Target
# specs are sampled from the empirical distribution to preserve spec-space
# coverage.
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Regime-aware loss function.
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Dataset and DataLoader.
# ----------------------------------------------------------------------------
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
                   'p_valid': p_valid.detach().cpu().numpy(),
                   'probs': probs.detach().cpu().numpy(),
                   'logits': logits.detach().cpu().numpy()}
        return failure_mask, metrics


def sample_validation_specs(df, n_samples=2000, boundary_ratio=0.5, seed=None):
    """Sample validation specs with heavy boundary emphasis for DAgger evaluation.

    If ``seed`` is not None, every random draw (boundary/interior row selection and
    final shuffle) is deterministic, so repeating the call with the same seed yields
    the exact same spec set. ``seed=None`` preserves the historical unseeded behavior
    used by the per-iteration final validation.
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


# =====================================================================
# DAGGER DISTILLATION EXECUTION
# =====================================================================


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


def log_param_bound_feasibility(name, params_arr, log_lo=None, log_hi=None,
                                param_cols=PARAM_COLS, edge_eps=0.02):
    """Log per-param buckets of where teacher labels sit within [log_lo, log_hi].

    For each output param i, computes ``frac = (log10(p) - log_lo[i]) / (log_hi[i] - log_lo[i])``
    (clipped to 1e-12 for log10) and reports the fraction of labels in five buckets:

      * ``below``  — frac < 0           (label cannot be represented)
      * ``at_lo``  — frac in [0, edge_eps]
      * ``mid``    — frac in (edge_eps, 1 - edge_eps)
      * ``at_hi``  — frac in [1 - edge_eps, 1]
      * ``above``  — frac > 1           (label cannot be represented)

    Any non-zero ``below`` or ``above`` bucket is a strong signal that the
    output ``log_lo/log_hi`` bounds are too tight for the task.
    """
    if params_arr is None or len(params_arr) == 0:
        _logger.info(f"[BOUNDS] {name}: no samples")
        return
    try:
        if log_lo is None or log_hi is None:
            lo_arr = np.array([PARAM_LOG_BOUNDS[k][0] for k in param_cols], dtype=np.float64)
            hi_arr = np.array([PARAM_LOG_BOUNDS[k][1] for k in param_cols], dtype=np.float64)
        else:
            lo_arr = np.asarray(log_lo, dtype=np.float64).flatten()
            hi_arr = np.asarray(log_hi, dtype=np.float64).flatten()
        if len(lo_arr) != len(param_cols) or len(hi_arr) != len(param_cols):
            raise ValueError(f"log_lo/log_hi length {len(lo_arr)}/{len(hi_arr)} != #param_cols {len(param_cols)}")
        params = np.asarray(params_arr, dtype=np.float64)
        if params.ndim != 2 or params.shape[1] != len(param_cols):
            raise ValueError(f"params_arr shape {params.shape} does not match #param_cols {len(param_cols)}")
        safe = np.clip(params, 1e-12, None)
        log_p = np.log10(safe)
        span = (hi_arr - lo_arr)
        if np.any(span <= 0):
            raise ValueError("non-positive log span in PARAM_LOG_BOUNDS")
        frac = (log_p - lo_arr) / span
        frac = np.nan_to_num(frac, nan=0.5, posinf=1.0, neginf=0.0)
        below = (frac < 0).mean(axis=0) * 100
        at_lo = ((frac >= 0) & (frac <= edge_eps)).mean(axis=0) * 100
        mid = ((frac > edge_eps) & (frac < 1 - edge_eps)).mean(axis=0) * 100
        at_hi = ((frac >= 1 - edge_eps) & (frac <= 1)).mean(axis=0) * 100
        above = (frac > 1).mean(axis=0) * 100
        _logger.info(
            f"[BOUNDS] {name} feasibility (frac in [log_lo, log_hi], edges={edge_eps:.2f}):"
        )
        for i, pname in enumerate(param_cols):
            _logger.info(
                f"  {pname:>8}: below={below[i]:4.1f}%  at_lo={at_lo[i]:4.1f}%  "
                f"mid={mid[i]:5.1f}%  at_hi={at_hi[i]:4.1f}%  above={above[i]:4.1f}%"
            )
    except Exception as e:
        _logger.warning(f"[BOUNDS] Failed to compute feasibility for {name}: {e}")


def log_saturation_breakdown(metrics, prefix="  val/", param_cols=PARAM_COLS, edge_eps=0.02):
    """Log how close the student's sigmoid outputs are to the log_lo/log_hi edges.

    Reads ``metrics['probs']`` (sigmoid outputs) and ``metrics['logits']``
    (pre-sigmoid) — both added to the metrics dict by ``identify_failures``.
    Reports per-param sigmoid mean/median, fraction pinned near 0 / 1, and
    ``max |logit|`` as a proxy for how hard the readout is being driven.
    """
    try:
        probs = metrics.get('probs') if metrics else None
        logits = metrics.get('logits') if metrics else None
        if probs is None:
            _logger.info(f"{prefix}saturation: (no probs in metrics — skipping)")
            return
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != len(param_cols):
            _logger.info(f"{prefix}saturation: unexpected probs shape {probs.shape} — skipping")
            return
        probs = np.nan_to_num(probs, nan=0.5)
        if logits is not None:
            logits = np.asarray(logits, dtype=np.float64)
            logits = np.nan_to_num(logits, nan=0.0)
            max_abs = np.abs(logits).max(axis=0)
        else:
            max_abs = np.full(len(param_cols), float('nan'))
        mean = probs.mean(axis=0)
        med = np.median(probs, axis=0)
        sat_lo = (probs <= edge_eps).mean(axis=0) * 100
        sat_hi = (probs >= 1 - edge_eps).mean(axis=0) * 100
        _logger.info(f"{prefix}saturation (sigmoid -> 0=log_lo, 1=log_hi, edges={edge_eps:.2f}):")
        for i, pname in enumerate(param_cols):
            _logger.info(
                f"  {pname:>8}: mean={mean[i]:.3f}  med={med[i]:.3f}  "
                f"sat_lo={sat_lo[i]:4.1f}%  sat_hi={sat_hi[i]:4.1f}%  max|logit|={max_abs[i]:.2f}"
            )
    except Exception as e:
        _logger.warning(f"{prefix}saturation logging failed: {e}")


def log_rail_probe(student, specs_arr, n=256, x_max=None, device=DEVICE):
    """Forward a subsample with trajectory storage; log node-voltage magnitude vs x_max.

    Per ODE stage, reports ``max|x|/x_max`` and the fraction of node-timepoints
    with ``|x| > 0.9*x_max`` (i.e., the tanh cells saturating against the local
    rail). A high rail fraction means the circuit dynamics are clamped and the
    effective expressivity is bounded by ``x_max``.
    """
    if x_max is None:
        x_max = KN_X_MAX
    if student is None or specs_arr is None or len(specs_arr) == 0:
        _logger.info(f"[RAIL] no specs — skipping (x_max={x_max})")
        return
    # Output ODE accumulators are the last ``output_ode_count`` entries of each
    # stage's ODE state vector (differential_stage.py:334). The readout is the
    # accumulator value at the final timepoint (t_span); if it is still changing
    # there, the effective output is a mid-transient integral (unsettled readout).
    out_dim = int(getattr(student.net, "output_ode_count", 0))
    try:
        subsample = specs_arr[: min(n, len(specs_arr))]
        specs_t = torch.from_numpy(subsample.astype(np.float32)).to(device)
        was_training = student.training
        student.eval()
        with torch.no_grad():
            u = student.scale_input(specs_t)
            _, trajs = student.net(u, store_trajectory=True, solver="heun")
        if was_training:
            student.train()
        if not trajs:
            _logger.info(f"[RAIL] no per-stage trajectories returned (x_max={x_max})")
            return
        _logger.info(f"[RAIL] node-voltage saturation vs x_max={x_max} (n={len(subsample)}):")
        for s, traj in enumerate(trajs):
            if traj is None:
                _logger.info(f"  stage {s}: (no trajectory)")
                continue
            v = traj.detach().abs()
            v_flat = torch.nan_to_num(v.reshape(-1), nan=0.0)
            max_v = v_flat.max().item()
            max_ratio = max_v / x_max if x_max > 0 else float('inf')
            frac = (v_flat > 0.9 * x_max).float().mean().item() * 100
            _logger.info(
                f"  stage {s}: max|x|/x_max={max_ratio:.2f}  "
                f"frac(|x|>0.9*x_max)={frac:4.1f}%"
            )
            _log_readout_settling(traj, out_dim=out_dim, s=s)
    except Exception as e:
        _logger.warning(f"[RAIL] probe failed: {e}")


def _log_readout_settling(traj, out_dim, s, first_frac=0.25, last_frac=0.25):
    """Report whether the output ODE accumulator readout is still in flight.

    Estimates the accumulator time-derivative over the early window vs the
    final window of each stage's trajectory. ``vel_last/vel_first`` near 1
    means the accumulator is still integrating (readout is a mid-transient
    value); a small ratio means it has settled before ``t_span``. Uses only the
    last ``output_ode_count`` columns, which are the accumulator nodes.
    """
    if out_dim <= 0:
        return
    try:
        a = traj[..., -out_dim:, :].float()  # (B, out_dim, T)
        if a.dim() == 2:
            a = a.unsqueeze(0)
        B, Oc, T = a.shape
        if T < 4 or Oc < 1:
            _logger.info("  readout: too few time steps to measure settling")
            return
        abs_a = torch.nan_to_num(a.detach().abs(), nan=0.0)
        mean_acc = abs_a.mean().item()
        max_acc = abs_a.max().item()
        # finite-difference derivative magnitudes
        d = torch.diff(a, dim=2).abs()  # (B, Oc, T-1)
        i_first = max(1, int(first_frac * (T - 1)))
        i_last = max(1, int(last_frac * (T - 1)))
        vel_first = d[..., :i_first].mean().item()
        vel_last = d[..., -i_last:].mean().item()
        ratio = (vel_last / (vel_first + 1e-12)) if vel_first > 0 else float('inf')
        state = "SETTLED" if ratio < 0.5 else ("RISING" if ratio > 1.0 else "MIXED")
        # fraction of (sample,accumulator) entries still moving in the last window
        any_move = (d[..., -i_last:].abs() > 1e-6).float().mean().item() * 100
        _logger.info(
            f"    readout: mean|acc|={mean_acc:.4f}  max={max_acc:.4f}  "
            f"vel_first={vel_first:.2e}  vel_last={vel_last:.2e}  "
            f"ratio={ratio:.2f} [{state}]  %moving_last={any_move:.1f}"
        )
    except Exception as e:
        _logger.warning(f"  [readout] settling probe failed on stage {s}: {e}")


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

# Instantiate student (Local KirchhoffNet wrapper)
# Topology hyperparameters are at the top of the notebook
# (KN_NUM_STAGES, KN_NUM_HIDDEN, KN_SMALL_WORLD_K/P/SEED, KN_EDGE_REPEATS,
#  KN_CELL_LIBRARY, KN_LEAK_MODE, KN_INTERSTAGE_ACTIVATION,
#  KN_BOUNDARY_FAN_OUT, KN_INPUT_RAIL).
student = LocalKirchhoffStudentWrapper(
    num_stages=KN_NUM_STAGES,
    num_hidden=KN_NUM_HIDDEN,
    small_world_k=KN_SMALL_WORLD_K,
    small_world_p=KN_SMALL_WORLD_P,
    small_world_seed=KN_SMALL_WORLD_SEED,
    edge_repeats=KN_EDGE_REPEATS,
    cell_library=KN_CELL_LIBRARY,
    leak_mode=KN_LEAK_MODE,
    interstage_activation=KN_INTERSTAGE_ACTIVATION,
    freeze_read=KN_FREEZE_READ,
    boundary_fan_out=KN_BOUNDARY_FAN_OUT,
    enable_temporal_readout=KN_TEMPORAL_READOUT,
    num_targets=7,
    input_rail=KN_INPUT_RAIL,
    x_max=KN_X_MAX,
    param_log_bounds=PARAM_LOG_BOUNDS,
    seed=KN_SMALL_WORLD_SEED,
    vca_enabled=KN_VCA,
    vca_rank=KN_VCA_RANK,
    vca_core_enabled=KN_VCA_CORE,
    vca_gate_shunt=KN_VCA_GATE_SHUNT,
    vca_separate_core_bus=KN_VCA_SEPARATE_CORE_BUS,
    vca_bias=KN_VCA_BIAS,
    gm_max=KN_GM_MAX,
    isat_max=KN_ISAT_MAX,
).to(DEVICE)

_logger.info(
    f"Student VCA config: enabled={KN_VCA}, rank="
    f"{int(KN_VCA_RANK) if KN_VCA_RANK is not None else int(VCA['rank'])}, "
    f"core={KN_VCA_CORE}, bias={KN_VCA_BIAS}, gate_shunt={KN_VCA_GATE_SHUNT}, "
    f"separate_core_bus={KN_VCA_SEPARATE_CORE_BUS} "
    f"(mirrors train_script.py --vca/--vca-core/--vca-gate-shunt/"
    f"--vca-separate-core-bus)"
)

# No `student.prepare(...)` step needed — local KirchhoffNetWithIO
# builds its src/dst indices eagerly inside build_net_from_config.

_param_counts = student.count_params_by_component()
_logger.info(
    f"Student (Local KirchhoffNet): "
    f"encoder={_param_counts['encoder']}, "
    f"circuit={_param_counts['circuit']}, "
    f"projector={_param_counts['projector']}, "
    f"total={_param_counts['total']:,} params  "
    f"(num_stages={KN_NUM_STAGES}, num_hidden={KN_NUM_HIDDEN}, "
    f"small_world_k={KN_SMALL_WORLD_K}, small_world_p={KN_SMALL_WORLD_P}, "
    f"edge_repeats={KN_EDGE_REPEATS}, cell_library={KN_CELL_LIBRARY}, "
    f"leak_mode={KN_LEAK_MODE}, interstage_activation={KN_INTERSTAGE_ACTIVATION}, "
     f"freeze_read={KN_FREEZE_READ}, temporal_readout={KN_TEMPORAL_READOUT}, input_rail={KN_INPUT_RAIL}, "
     f"x_max={KN_X_MAX}, gm_max={KN_GM_MAX}, isat_max={KN_ISAT_MAX})"
)
_n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
_logger.info(f"Student trainable params: {_n_trainable:,}")
_bd = student.net.parameter_breakdown()
_logger.info(f"Student param breakdown:\n{format_parameter_breakdown(_bd)}")

# BO's parameter preflight must stop before teacher setup, initial-label
# generation, or any DAgger iteration. ``print`` provides a stable contract
# for kn_bayes_opt.py's subprocess parser.
if getattr(_bo_args, 'count_params_only', False):
    print(f"trainable params: {_n_trainable:,}", flush=True)
    raise SystemExit(0)

# Optimizer and scheduler (defined once, reused across iterations).
# Wrapper's parameters() exposes the entire local KirchhoffNetWithIO
# (fabric stages, transfers, InputMapper, OutputAffine). Per-group LR scales
#   - mapper: InputMapper + OutputAffine (output_mapper)
#   - struct: name.endswith('.z_logits') (per-stage edge-gate logits)
#             + .vca_W / .vca_W_core / .vca_v_{boundary,readout,core}
#             (VCA routing-structure params, train.py:1102-1108)
#   - dyn:    name.endswith('.raw_leak') or '.raw_drive_g'
#   - other:  everything else (stages/transfers/vref/etc.)
# The 'lr_scale' key is stored in each group dict so the per-iteration LR
# reset at dagger:3004-3010 can multiply the base LR by the group's scale.
def _make_dagger_optimizer(student, lr, weight_decay):
    _mapper, _struct, _dyn, _other = [], [], [], []
    for _name, _p in student.named_parameters():
        if not _p.requires_grad:
            continue
        if "input_mapper" in _name or "output_mapper" in _name or "post_readout_transfer" in _name:
            _mapper.append(_p)
        elif _name.endswith(".z_logits"):
            _struct.append(_p)
        elif (
            _name.endswith(".vca_W")
            or _name.endswith(".vca_W_core")
            or _name.endswith(".vca_v_boundary")
            or _name.endswith(".vca_v_readout")
            or _name.endswith(".vca_v_core")
        ):
            # VCA routing-structure params (shared input projection +
            # per-edge tap embeddings for each gated family). Mirrors
            # train.py make_optimizer classification (train.py:1102-1108):
            # routed to the structural LR group (4x base by convention)
            # so the gate signal learns alongside z_logits.
            _struct.append(_p)
        elif _name.endswith(".raw_leak") or _name.endswith(".raw_drive_g"):
            _dyn.append(_p)
        else:
            _other.append(_p)
    groups = []
    if _other:
        groups.append({"params": _other, "lr": lr, "lr_scale": 1.0})
    if _mapper:
        groups.append({"params": _mapper, "lr": lr * MAPPER_LR_SCALE, "lr_scale": MAPPER_LR_SCALE})
    if _struct:
        groups.append({"params": _struct, "lr": lr * STRUCT_LR_SCALE, "lr_scale": STRUCT_LR_SCALE})
    if _dyn:
        groups.append({"params": _dyn, "lr": lr * DYN_LR_SCALE, "lr_scale": DYN_LR_SCALE})
    _logger.info(f"[OPT] Param groups: other={sum(p.numel() for p in _other)} "
                 f"({len(_other)} tensors, lr={lr:.2e}); "
                 f"mapper={sum(p.numel() for p in _mapper)} "
                 f"({len(_mapper)} tensors, lr={lr*MAPPER_LR_SCALE:.2e}); "
                 f"struct={sum(p.numel() for p in _struct)} "
                 f"({len(_struct)} tensors, lr={lr*STRUCT_LR_SCALE:.2e}); "
                 f"dyn={sum(p.numel() for p in _dyn)} "
                 f"({len(_dyn)} tensors, lr={lr*DYN_LR_SCALE:.2e})")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


optimizer = _make_dagger_optimizer(student, LR_INITIAL, WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS_PER_ITER, eta_min=LR_FLOOR
)

# Load any existing checkpoint BEFORE the initial-build guard so we can either
# resume from it (skipping the expensive initial labeling and re-splitting)
# or fall through to the fresh-start path. The entire resume restore —
# model/optimizer/scheduler state, RNG, dataset, and histories — is wrapped
# in one try/except so any legacy/incompatible field gracefully degrades to
# a fresh start instead of crashing.
#
# Plan phase-a-fixes, bug 1: Phase-A runs (--prepare-canonical-dataset or
# --canonical-dataset) skip the restore entirely.  The shared default
# CHECKPOINT_PATH may carry a stale model from a different architecture
# (different edge count / VCA rank); even strict=False masking is unsafe
# because the VCA core weights would silently warm-start the BO trial.
ckpt = None
if not _phase_a_requested:
    ckpt = load_checkpoint()
distillation_dataset = None
if ckpt is not None:
    try:
        # strict=False so pre-post_readout_transfer checkpoints (without the
        # 28-param residual-relu-tanh post-readout stage) still load — missing
        # keys keep identity-at-init (w1=1, w2=w3=vth=0).
        student.load_state_dict(ckpt['student_state'], strict=False)
        student.to(DEVICE)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'rng' in ckpt:
            _restore_rng(ckpt['rng'])
        distillation_dataset = DistillationDataset([])
        _restore_dataset_from_ckpt(ckpt)
        _restore_histories_from_ckpt(ckpt)
        _restore_common_eval_specs_from_ckpt(ckpt)
        _logger.info(
            f"[CKPT] Resumed from {CHECKPOINT_PATH}  "
            f"next_iter={int(ckpt['dagger_iter'])}, "
            f"dataset_size={len(ckpt.get('dataset', []))}, "
            f"format_version={ckpt.get('format_version')}"
        )
        _logger.info(f"[CKPT] Restored DistillationDataset from checkpoint: "
                     f"{len(distillation_dataset)} samples, "
                     f"train={len(distillation_dataset._train_indices)}, "
                     f"val={len(distillation_dataset._val_indices)}, "
                     f"hard_buffer_start_idx={distillation_dataset._hard_buffer_start_idx}")
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to restore from checkpoint: {e}; "
                        f"starting fresh and overwriting checkpoint")
        ckpt = None
        distillation_dataset = None

# ── Canonical Phase-A gate (plan canonical-ctle-unify) ─────────────────────
# Placed BEFORE the initial-dataset build so Phase-A trials never pay for the
# (expensive) DAgger teacher labelling.  If --canonical-dataset is passed (or
# --prepare-canonical-dataset), the DAgger loop is bypassed entirely: the
# static-data, Huber-only Phase-A training mode lets the BO drivers run a
# fully teacher-comparable, distribution-stationary CTLE inverse benchmark.
if _bo_args.prepare_canonical_dataset is not None or _bo_args.canonical_dataset is not None:
    from ctle_dagger_common import (create_canonical_flow_dataset,
                                     load_canonical_flow_dataset, run_phase_a_training,
                                     init_output_affine_bias_from_labels)
    if _bo_args.prepare_canonical_dataset is not None:
        target_path = Path(_bo_args.prepare_canonical_dataset)
        if target_path.exists() and not _bo_args.canonical_dataset:
            _logger.info(f"[phaseA] canonical dataset already exists at {target_path}; loading instead")
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
            )
        if not _bo_args.canonical_dataset:
            _logger.info("[phaseA] --prepare-canonical-dataset set: exiting before training")
            sys.exit(0)
    if _bo_args.canonical_dataset is not None:
        canonical = load_canonical_flow_dataset(_bo_args.canonical_dataset)
        train_idx = np.asarray(canonical["train_idx"], dtype=np.int64)
        init_output_affine_bias_from_labels(
            student, np.asarray(canonical["params"])[train_idx],
        )
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
            "input_preprocessing": "knet",
            "mapper_lr_scale": float(_bo_args.kn_mapper_lr_scale),
            "struct_lr_scale": float(_bo_args.kn_struct_lr_scale),
            "dyn_lr_scale": float(_bo_args.kn_dyn_lr_scale),
        }
        _logger.info(f"[phaseA] canonical-dataset gate engaged: {len(canonical['specs'])} specs, "
                     f"epochs={phase_a_epochs}, batch={ctx_phase_a['batch_size']}, "
                     f"lr={ctx_phase_a['lr']:.2e}, "
                     f"diff-LR scales mapper/struct/dyn="
                     f"{_bo_args.kn_mapper_lr_scale}/{_bo_args.kn_struct_lr_scale}/{_bo_args.kn_dyn_lr_scale}")
        run_phase_a_training(student, ctx_phase_a, model_name="phase_a_knet")
        _logger.info("[phaseA] canonical-dataset gate: training complete; exiting before DAgger body")
        sys.exit(0)

# ── Initial dataset ──────────────────────────────────────────────────
if ckpt is not None:
    pass  # already restored above
else:
    # Fresh-start path: consult the initial-dataset cache (config-hashed),
    # else run the (expensive) teacher labeling and write the cache. The cache
    # load is guarded because a corrupt or partial file (from an interrupted
    # prior write) must not crash the script; the dump is atomic so a kill
    # mid-write leaves the prior valid file untouched.
    _cache_path = _initial_dataset_cache_path()
    initial_data = None
    if os.path.exists(_cache_path):
        try:
            with Timer("load initial distillation dataset from cache"):
                initial_data = joblib.load(_cache_path)
            _logger.info(f"[CKPT] Initial dataset loaded from cache: {_cache_path} "
                         f"({len(initial_data)} samples, hash={os.path.basename(_cache_path)})")
        except Exception as e:
            _logger.warning(f"[CKPT] Failed to load initial-dataset cache ({_cache_path}): "
                            f"{e}; rebuilding from scratch")
            initial_data = None
    if initial_data is None:
        with Timer("build initial flow distillation dataset"):
            initial_data = create_flow_distillation_dataset(
                df,
                n_samples=N_EMPIRICAL_SAMPLES,
                n_candidates=N_CANDIDATES_INITIAL,
                boundary_ratio=BOUNDARY_RATIO,
                teacher_labeler=teacher_labeler,
            )
        try:
            _cache_tmp = _cache_path + '.tmp'
            joblib.dump(initial_data, _cache_tmp)
            os.replace(_cache_tmp, _cache_path)
            _logger.info(f"[CKPT] Initial dataset cached at {_cache_path} "
                         f"(hash={os.path.basename(_cache_path)})")
        except Exception as e:
            _logger.warning(f"[CKPT] Failed to write initial dataset cache: {e}")
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
    log_param_bound_feasibility("Initial dataset", all_params)

    # Carve out 10% validation split (never augmented by DAgger)
    train_subset, val_subset = distillation_dataset.split_train_val(val_frac=0.1)

    # Baseline checkpoint: anchors the run at dagger_iter=0 so a crash during
    # the first iteration's training or teacher labeling can resume cleanly.
    try:
        save_checkpoint(_ckpt_baseline_payload(dagger_iter=0))
        _logger.info(f"[CKPT] Baseline checkpoint saved to {CHECKPOINT_PATH}")
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to write baseline checkpoint: {e}")

train_loader = distillation_dataset.get_loader(batch_size=BATCH_SIZE, shuffle=True, hard_weight=10.0)

val_loader = distillation_dataset.get_val_loader(batch_size=BATCH_SIZE)
_logger.info(f"Train: {len(distillation_dataset._train_indices)}, "
             f"Val: {len(distillation_dataset._val_indices)}")

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
    alpha_spec=ALPHA_SPEC,
    beta_phys=BETA_PHYS,
    gamma_mono=GAMMA_MONO,
    alpha_invalid=ALPHA_INVALID,
    empirical_df=df,
    k_manifold=K_MANIFOLD,
    alpha_manifold=ALPHA_MANIFOLD,
)
ramped_criterion = RampedRegimeLoss(criterion, warmup_epochs=WARMUP_EPOCHS)

# Dump the full hyperparameter set to the log file at the start of each run.
_log_hyperparameters()

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

# Run diagnostics (skip on resume: already logged in the prior run, and the
# teacher-labeling call inside is expensive — we don't want to re-pay it).
if ckpt is None:
    run_diagnostics(teacher_labeler, df, zig_model, scaler_X, scaler_y_p, DEVICE, n=200)
else:
    _logger.info("[CKPT] Skipping run_diagnostics on resume")


# ── Shared COMMON_EVAL_SPECS ─────────────────────────────────────────
# Single fixed shared eval set used for both per-epoch early-stop tracking
# AND per-iteration "Validation failure rate" reporting (plan
# dagger_knet_best_model_selection_improvement). Built fresh on a fresh start;
# on resume, _restore_common_eval_specs_from_ckpt already overwrote this from
# the checkpoint if the field was present.
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
_start_iter = int(ckpt['dagger_iter']) if ckpt is not None else 0
if ckpt is not None and ckpt.get('converged', False):
    _logger.info(f"[CKPT] Resumed checkpoint has converged=True; skipping training loop "
                 f"and regenerating final artifacts")
    _start_iter = DAGGER_ITERATIONS
if _start_iter >= DAGGER_ITERATIONS:
    _logger.info(f"[CKPT] Resumed dagger_iter={_start_iter} >= DAGGER_ITERATIONS="
                 f"{DAGGER_ITERATIONS}; skipping training loop")
for dagger_iter in range(_start_iter, DAGGER_ITERATIONS):
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
    # best_failure_rate is initialized below to prev_failure_rate (iter 1 → 1.0)
    # so the first within-iteration accept threshold is the carried-forward baseline.

    # Snapshot the carried-forward student (last iteration's restored best)
    # and measure its baseline failure rate on COMMON_EVAL_SPECS. Iteration 1
    # has no previous iteration → use 1.0 so any reasonable model passes the
    # MIN_FAILURE_IMPROVEMENT threshold on the first iteration.
    prev_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
    if dagger_iter == 0:
        prev_failure_rate = 1.0
        _logger.info(f"  Prev-iter baseline: no prior model (iteration 1) → "
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

    epoch_losses = {'total': [], 'imit': [], 'spec': [], 'phys': [], 'invalid': [], 'manifold': []}
    n_batches = 0
    n_nonfinite_batches = 0   # batches skipped by the NaN/grad guards this iteration
    _div_consec = 0           # consecutive divergence-margin evals (see DIVERGENCE_*)
    diverged_at = None        # epoch at which the iteration was aborted, if any
    for epoch in range(EPOCHS_PER_ITER):
        ramped_criterion.set_epoch(epoch)
        student.train()
        losses = {'total': 0, 'imit': 0, 'spec': 0, 'phys': 0, 'invalid': 0, 'manifold': 0}
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
            # LocalKirchhoffStudentWrapper returns just logits (no regime branch).
            logits = student(specs_batch)
            base_loss_dict = ramped_criterion.base_loss(student, spec_targets, params_batch, logits=logits)
            ramped_loss_dict = ramped_criterion(student, spec_targets, params_batch, logits=logits, loss_dict=base_loss_dict)
            total_loss = ramped_loss_dict['total']
            if not total_loss.requires_grad or not torch.isfinite(total_loss):
                # NaN/Inf batch (flagged by the loss's internal guard or a
                # non-finite ramped total): skip the update entirely so a
                # poisoned batch can never reach optimizer.step().
                losses['total'] += float(total_loss)
                n_nonfinite_batches += 1
            else:
                total_loss.backward()
                # clip_grad_norm_ over the wrapper's parameters covers the local
                # KirchhoffNet fabric. Its return value is the PRE-clip total
                # norm: if backward exploded (stiff ODE), the norm is Inf/NaN
                # and clipping itself maps Inf grads to NaN — stepping here
                # would write NaN into every weight (permanent 100% failure).
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

        n = len(train_loader)
        avg_loss = losses['total'] / n
        avg_imit = losses['imit'] / n
        avg_spec = losses['spec'] / n
        avg_phys = losses['phys'] / n
        avg_invalid = losses['invalid'] / n
        avg_manifold = losses['manifold'] / n

        for k in epoch_losses:
            epoch_losses[k] += [losses[k] / n]
        n_batches += 1

        # Evaluate on validation set and step scheduler
        val_losses = eval_epoch(student, val_loader, ramped_criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _logger.info(f"  Epoch {epoch+1:3d}/{EPOCHS_PER_ITER}  "
                         f"train={avg_loss:.4f}(imit={avg_imit:.3f},spec={avg_spec:.3f},phys={avg_phys:.3f},invalid={avg_invalid:.3f},manifold={avg_manifold:.3f})  "
                         f"val={val_losses['total']:.4f}(imit={val_losses['imit']:.3f},spec={val_losses['spec']:.3f},phys={val_losses['phys']:.3f},invalid={val_losses['invalid']:.3f},manifold={val_losses['manifold']:.3f})  "
                         f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Save best state by val loss (for loss-aligned checkpointing)
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        # Early stopping on failure rate: evaluate every EARLYSTOP_EVAL_EVERY epochs
        # on the SHARED COMMON_EVAL_SPECS set (built once before the DAgger loop).
        # - During the first EARLYSTOP_SKIP_EPOCHS epochs, no best-tracking is
        #   performed so the warm-started epoch-1 snapshot can't win immediately.
        # - A new best is accepted only when it beats the running best by at
        #   least MIN_FAILURE_IMPROVEMENT (absolute), enforcing monotone
        #   improvement across the in-iteration budget.
        # - EARLYSTOP_PATIENCE_EPOCHS = EPOCHS_PER_ITER (full budget) by design,
        #   so the inner loop is never cut short.
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
                # Divergence abort: a large SUSTAINED regression above the
                # previous iteration's baseline means this iteration's training
                # is destroying the model (e.g. NaN-poisoned or loss-ramp
                # collapse). The remaining budget would be wasted — abort and
                # let the carry-forward guardrail below restore the best/prev
                # model. Requires DIVERGENCE_CONSEC_EVALS consecutive evals
                # above prev + DIVERGENCE_MARGIN, so normal noise never fires it.
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
    # Priority:
    #   1. best_failure_state if it beat prev_failure_rate by >= MIN_FAILURE_IMPROVEMENT
    #      (new within-iteration best is strictly better on COMMON_EVAL_SPECS)
    #   2. prev_state (carried-forward from last iteration; only fallback for
    #      "no meaningful improvement" — protects against regression per the plan)
    #   3. best_state (val-loss best) — legacy fallback for the rare edge case
    #      where we have a loss-best but neither a failure-state nor a prev_state
    #   4. last-epoch student (final fallback)
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

    # ── 3b. Evaluate on the SHARED COMMON_EVAL_SPECS set (vectorized) ─────
    # Same 2000-spec set used for per-epoch early-stop tracking, so the
    # per-iteration rate reported here is directly comparable to last
    # iteration's (monotone-improvement table).
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
    log_saturation_breakdown(metrics, prefix="  val/")
    log_param_bound_feasibility("Rolling dataset", np.array([d['params'] for d in distillation_dataset.data]))
    log_rail_probe(student, val_specs_arr)

    # Per-iteration summary line: monotone-improvement table.
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

    # Use flags instead of break/continue so the iteration-end checkpoint
    # below always runs (regardless of path through the body).
    converged = False
    skip_labeling = False

    if failure_rate < CONVERGENCE_THRESHOLD:
        _logger.info(f"CONVERGENCE MET (failure_rate={failure_rate*100:.2f}% < "
                     f"{CONVERGENCE_THRESHOLD*100}%)")
        converged = True

    # ── 3d. Skip teacher labeling if no failures ────────────────────
    if not converged and len(failed_specs_arr) == 0:
        _logger.info("No failures detected — skipping teacher labeling.")
        skip_labeling = True

    # ── 3e. Data imbalance check ────────────────────────────────────
    if not converged and not skip_labeling:
        cap = int(len(distillation_dataset) * FAILURE_CAP_RATIO)
        if len(failed_specs_arr) > cap:
            top_k_idx = np.argsort(failure_errors[failure_mask])[-cap:]
            failed_specs_arr = failed_specs_arr[top_k_idx]
            _logger.info(f"  Capped failures to top {cap} highest-error cases")

    # ── 3f. Reset optimizer LR and apply decay (floor at 1e-4) ─────
    if not converged and not skip_labeling:
        new_lr = LR_INITIAL * (0.5 ** max(0, dagger_iter - LR_DECAY_AFTER_ITER))
        new_lr = max(new_lr, 1e-4)
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr * pg.get('lr_scale', 1.0)
        _lr_per_group = ', '.join('{:.2e}'.format(pg['lr']) for pg in optimizer.param_groups)
        # Recreate the cosine scheduler so it restarts from the reset LR.
        # Without this, the continuously-stepped CosineAnnealingLR overrides
        # the manual reset on its next step() and its phase wraps across
        # iterations (T_max=EPOCHS_PER_ITER), so some iterations would START
        # at the full initial LR — instantly destroying the warm-started model.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS_PER_ITER, eta_min=LR_FLOOR
        )
        _logger.info(f"  LR reset to {new_lr:.2e} (per-group: {_lr_per_group}); "
                     f"cosine scheduler restarted for next iteration")

    # ── 3g. Teacher labeling with k-NN empirical fallback ─────────────
    final_specs = np.empty((0, 4), dtype=np.float32)
    final_labels = np.empty((0, 7), dtype=np.float32)
    if not converged and not skip_labeling:
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
        log_param_bound_feasibility("DAgger labels", final_labels)
        distillation_dataset.append_samples(final_specs, final_labels)
        _logger.info(f"  Added {len(final_labels)} labeled failures.  Dataset now: "
                     f"{len(distillation_dataset)}")

    # ── 3i. Iteration-end checkpoint ─────────────────────────────────
    # Save unconditionally at the end of every iteration (normal path,
    # no-failure skip, and convergence). Next resume starts at dagger_iter+1.
    try:
        save_checkpoint(_ckpt_baseline_payload(dagger_iter=dagger_iter + 1, converged=converged))
        _logger.info(f"[CKPT] Iteration {dagger_iter+1} checkpoint saved "
                     f"(next_iter={dagger_iter+1}, dataset={len(distillation_dataset)})")
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to write iteration-end checkpoint: {e}")

    if converged:
        break

_logger.info("DAgger training complete")

_logger.info("DAgger training complete - no Phase 2 fine-tuning (pure manifold-constrained distillation)")


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
    log_saturation_breakdown(metrics, prefix="  test/")
    return metrics


# ----------------------------------------------------------------------------
# Final DAgger plots, evaluation, and artifact save.
# ----------------------------------------------------------------------------
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
_clip_stats = student.clipping_stats()
_logger.info(
    "KNet input clipping: %d/%d elements (%.4f%%) hit [-%.1f, %.1f] during run",
    _clip_stats['clipped_elements'], _clip_stats['input_elements'],
    100.0 * _clip_stats['clip_fraction'], student.input_rail, student.input_rail,
)

# =====================================================================
# Save model
# =====================================================================

MODEL_NAME = 'dagger_student_kirchhoff'
torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.pt'))
joblib.dump(scaler_X, os.path.join(OUTPUT_DIR, 'scaler_X.pkl'))
joblib.dump(flow_scaler_C, os.path.join(OUTPUT_DIR, 'flow_scaler_C.pkl'))

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
# Demo: Student Predictions (KirchhoffNet) for Specific Target Specs
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

print("\nInput scaling (LocalKirchhoffStudentWrapper.scale_input):")
print(f"  per dim: log10(raw) -> min-max -> 2*(log - lo)/(hi - lo) - 1, clamp to [-{KN_INPUT_RAIL}, {KN_INPUT_RAIL}]")
for _i, _name in enumerate(['power', 'jitter', 'height', 'width']):
    print(f"  {_name:<7}: log10 range [{KN_INPUT_LOG_MIN[_i]:.3f}, {KN_INPUT_LOG_MAX[_i]:.3f}] "
          f"-> raw [{10 ** KN_INPUT_LOG_MIN[_i]:.4e}, {10 ** KN_INPUT_LOG_MAX[_i]:.4e}]")
print(f"\nOutput (bounded log-space -> physical): 10^(log_lo + (log_hi - log_lo)*sigmoid)")
for name, (lo, hi) in PARAM_LOG_BOUNDS.items():
    print(f"  {name:>8}: [{10**lo:.4e}, {10**hi:.4e}]")
