"""
spice_failure_check.py
======================

True-SPICE failure-rate harness for ONE trained checkpoint (KNet DAgger
student or PlainMLP BO winner), driven through the 7-param ideal-Bob Cadence
flow on ssr0.

Stages
------
1. Inference:  load checkpoint, batch-forward N target specs -> 7 physical
   params [fW, current, ind, Rd, Cs, Rs, VDD] in PARAM_COLS order.
2. Ocean:      per-spec `update_ocn_ctle` (use_new_templates=True) +
   `ocean -nograph` via csh PDK wrapper + `parse_ctle_csv`. ssr0-only.
3. Scoring:    ZIG-proxy degrade rule (>=2 dims degraded at 0.10) on the
   Cadence-measured [power, jitter, height, width] vs the original target.

Local-mode defaults run inference + scoring against a mocked parser so the
file py_compiles, the CLI parses, and the math is exercised without Cadence
or torch CUDA. Set --enable-ocean to drive true Spectre sims on ssr0.

Pulled-and-run flow on ssr0
---------------------------
    # From this repo's root on ssr0 (or copy the single file next to train_ctle):
    python spice_failure_check.py \
        --model-kind mlp \
        --ckpt <path>/dagger_student_plain.pt \
        --specs-npz <path>/canonical_ctle.npz \
        --n-specs 50 \
        --work-root ~/simulation/SPICE_failure \
        --template-dir /home/annaik/Documents/train_ctle/ocn_template_ideal_bob \
        --enable-ocean

KNet / external bridge (no auto-rebuild yet)
--------------------------------------------
    # 1. Run your own inference -> (N,7) log10 params in PARAM_COLS order:
    #    pred_params_log.csv  (header: fW,current,ind,Rd,Cs,Rs,VDD)
    # 2. python spice_failure_check.py --model-kind knet --ckpt <trial_ckpt> \
    #        --specs-npz $CANONICAL --n-specs 50 \
    #        --pred-params-csv pred_params_log.csv \
    #        --work-root ~/simulation/SPICE_failure \
    #        --template-dir .../ocn_template_ideal_bob --enable-ocean
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Project constants (mirrors ctle_dagger_common / dagger-nuance-distillation).
# Re-declared locally so this file is import-side-effect-free.
# ---------------------------------------------------------------------------

PARAM_COLS = ["fW", "current", "ind", "Rd", "Cs", "Rs", "VDD"]

# Physical log10 bounds; match ctle_dagger_common.PARAM_LOG_BOUNDS.
PARAM_LOG_BOUNDS: dict[str, tuple[float, float]] = {
    "fW":     (np.log10(1e-6),  np.log10(10.0)),
    "current": (np.log10(5e-4), np.log10(2.5)),
    "ind":    (np.log10(1e-12), np.log10(3.0)),
    "Rd":     (np.log10(10),    np.log10(1500)),
    "Cs":     (np.log10(1e-15), np.log10(1e-9)),
    "Rs":     (np.log10(10),    np.log10(1500)),
    "VDD":    (np.log10(0.6),   np.log10(1.2)),
}

SPEC_INPUT_COLS = ["power", "stage_2_jitter", "stage_2_eye_max_height",
                   "stage_2_eye_max_width"]

# Default measured-spec -> target-spec key map (56G stage-2).
DEFAULT_KEY_MAP = {
    "power":   "Power",
    "jitter":  "eye_p2pJitterAverage_norm stage 2",
    "height":  "eye_maxHeight_norm Vout_2 56G",
    "width":   "eye_maxWidth_norm Vout_2 56G",
}


logger = logging.getLogger("spice_failure_check")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spice_failure_check",
        description=(
            "True-SPICE failure rate for ONE trained checkpoint via the 7-param "
            "ideal-Bob Cadence flow on ssr0."
        ),
    )
    # ----- model -----
    p.add_argument("--model-kind", choices=["knet", "mlp"], required=True,
                   help="Which checkpoint family to load.")
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to the trained .pt checkpoint.")
    p.add_argument("--device", default="cpu",
                   help="Torch device for inference (default: cpu; 'cuda' on GPU hosts).")

    # KNet rebuild hints (preferred: --kn-config-json; fallback: explicit flags).
    p.add_argument("--kn-config-json", type=Path, default=None,
                   help="Optional trial config JSON with all --kn-* fields.")

    # MLP loader knobs (fallback when teacher_config.json is absent).
    p.add_argument("--mlp-width", type=int, default=48)
    p.add_argument("--mlp-layers", type=int, default=3)
    p.add_argument("--mlp-activation", choices=["silu", "gelu"], default="silu")
    p.add_argument("--mlp-use-layernorm", action="store_true")
    p.add_argument("--input-preprocessing", choices=["knet", "q75"], default="knet")

    # ----- specs -----
    spec_src = p.add_mutually_exclusive_group(required=True)
    spec_src.add_argument("--specs-npz", type=Path, default=None,
                          help="Canonical Phase-A .npz with 'specs' key "
                               "(N×4 in SPEC_INPUT_COLS order).")
    spec_src.add_argument("--specs-csv", type=Path, default=None,
                          help="CSV with header SPEC_INPUT_COLS (must have at "
                               "least those 4 columns).")
    p.add_argument("--pred-params-csv", type=Path, default=None,
                   help="Optional (N,7) log10-params CSV in PARAM_COLS order. "
                        "When given, Stage-1 inference is SKIPPED and these "
                        "params are used directly (KNet/external bridge path). "
                        "Must have exactly --n-specs rows matching the sampled specs.")
    p.add_argument("--n-specs", type=int, default=20,
                   help="Number of specs to evaluate (subsampled if dataset is larger).")
    p.add_argument("--spec-offset", type=int, default=0,
                   help="Subsample start offset (seeding-controlled).")
    p.add_argument("--seed", type=int, default=100,
                   help="Subsample seed for --n-specs deterministic slice.")

    # ----- Ocean / sim -----
    p.add_argument("--enable-ocean", action="store_true",
                   help="Drive true Cadence Spectre sims. WITHOUT this flag the "
                        "script runs in local mode (mocked parser + scoring only).")
    p.add_argument("--template-dir", type=Path,
                   default=Path("/home/annaik/Documents/train_ctle/ocn_template_ideal_bob"),
                   help="Directory with template_{tb}.ocn (ideal-Bob 7-param flow).")
    p.add_argument("--tb-ids", default="1,2,3,4,5,6,7,9",
                   help="Comma-separated TB ids to round-robin across (template_8 absent upstream).")
    p.add_argument("--work-root", type=Path,
                   default=Path("~/simulation/SPICE_failure"),
                   help="Per-spec working directory root.")
    p.add_argument("--run-tag", default=None,
                   help="Subdir under --work-root (default: ckpt basename + ts).")
    p.add_argument("--sim-timeout", type=int, default=600,
                   help="Per-spec sim timeout (seconds).")
    p.add_argument("--jobs", type=int, default=None,
                   help="Parallel Ocean workers (default: len(tb-ids)).")
    p.add_argument("--pdk-source-cmd",
                   default="source /home/annaik/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh",
                   help="csh 'source' line for the GF22 PDK.")
    p.add_argument("--ocean-bin", default="ocean",
                   help="Cadence Ocean binary name (default: ocean).")
    p.add_argument("--no-delete-history", action="store_true",
                   help="Skip maestro-history cleanup (only relevant with --enable-ocean).")

    # ----- scoring -----
    p.add_argument("--degrade-thr", type=float, default=0.10,
                   help="Relative degrade threshold per dim (default 0.10).")
    p.add_argument("--min-degraded-dims", type=int, default=2,
                   help="Number of dims that must degrade to count a spec as failure.")
    p.add_argument("--key-map", default=None,
                   help='JSON dict {"power":"Power","jitter":"...",...} overriding defaults.')

    # ----- modes -----
    p.add_argument("--dry-run", action="store_true",
                   help="Stage-1 + Stage-2 OCN scaffolding only; no Spectre launch.")
    p.add_argument("--smoke", action="store_true",
                   help="Force n-specs=2, 1 TB, dry-run by default. For local smoke tests.")
    p.add_argument("--offline-csv", type=Path, default=None,
                   help="Re-score an existing results.csv without running anything.")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Skip specs with valid parsed.json (default).")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Re-run all specs even if parsed.json exists.")
    p.add_argument("--dump-keys", action="store_true",
                   help="Parse one result.csv and print its keys; exit. Requires --offline-csv or --resume with at least 1 done spec.")
    p.add_argument("--mock-template", action="store_true",
                   help="Synthesize a minimal template_{tb}.ocn under --template-dir "
                        "when real templates are absent (local dry-run only).")

    p.add_argument("--log-level", default="INFO")
    return p


# ---------------------------------------------------------------------------
# Spec sampling
# ---------------------------------------------------------------------------

def load_specs(args: argparse.Namespace) -> np.ndarray:
    """Return (N, 4) float32 array in SPEC_INPUT_COLS order."""
    if args.specs_npz is not None:
        try:
            data = np.load(args.specs_npz, allow_pickle=True)
        except Exception as e:
            raise SystemExit(f"[specs] failed to load --specs-npz {args.specs_npz}: {e}")
        if "specs" not in data:
            raise SystemExit(
                f"[specs] {args.specs_npz} has no 'specs' key; keys={list(data.keys())}")
        specs = np.asarray(data["specs"], dtype=np.float32)
        if specs.ndim != 2 or specs.shape[1] != 4:
            raise SystemExit(f"[specs] expected (N,4), got {specs.shape}")
    else:
        specs = _read_specs_csv(args.specs_csv)

    n = specs.shape[0]
    if args.smoke:
        n_take = min(2, n)
        return specs[:n_take].copy()
    if args.n_specs is not None and args.n_specs > 0:
        offset = max(0, min(args.spec_offset, n - 1))
        rng = np.random.default_rng(args.seed)
        idx = np.arange(n)
        # Deterministic seeded subsample: shuffle, then slice [offset:offset+n_specs].
        rng.shuffle(idx)
        idx = np.sort(idx[offset:offset + args.n_specs])
        return specs[idx].copy()
    return specs.copy()


def _read_specs_csv(path: Path) -> np.ndarray:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in SPEC_INPUT_COLS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"[specs-csv] {path} missing columns: {missing}")
        rows = [[float(r[c]) for c in SPEC_INPUT_COLS] for r in reader]
    if not rows:
        raise SystemExit(f"[specs-csv] {path} is empty")
    return np.asarray(rows, dtype=np.float32)


def _read_pred_params_csv(path: Path, n_expected: int) -> np.ndarray:
    """Read an (N,7) log10-params CSV in PARAM_COLS order (KNet bridge path)."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in PARAM_COLS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"[pred-params-csv] {path} missing columns: {missing}")
        try:
            rows = [[float(r[c]) for c in PARAM_COLS] for r in reader]
        except ValueError as e:
            raise SystemExit(f"[pred-params-csv] {path} has non-numeric entries: {e}")
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape != (n_expected, len(PARAM_COLS)):
        raise SystemExit(
            f"[pred-params-csv] {path} shape {arr.shape} != "
            f"({n_expected}, {len(PARAM_COLS)}); it must match the sampled specs.")
    if not np.all(np.isfinite(arr)):
        raise SystemExit(f"[pred-params-csv] {path} contains non-finite values.")
    return arr


# ---------------------------------------------------------------------------
# Stage 1 — inference
# ---------------------------------------------------------------------------

def _clamp_log_bounds(log_arr: np.ndarray) -> np.ndarray:
    out = log_arr.copy()
    for i, name in enumerate(PARAM_COLS):
        lo, hi = PARAM_LOG_BOUNDS[name]
        out[:, i] = np.clip(out[:, i], lo, hi)
    return out


def _log_to_physical(log_arr: np.ndarray) -> np.ndarray:
    return np.power(10.0, _clamp_log_bounds(log_arr)).astype(np.float32)


class MlpTeacher:
    """Lightweight stand-in for ctle_dagger_common.PlainMLP.

    The actual PlainMLP class lives in ctle_dagger_common; we import it lazily
    so that a `--help` run (or mocked smoke test) never touches torch at all.

    CLI fallbacks (used only when neither the sidecar nor the embedded
    checkpoint config supplies architecture). Pass via constructor kwargs.
    """

    def __init__(self, state_path: Path, config_path: Optional[Path] = None,
                 _mlp_width: int = 48, _mlp_layers: int = 3,
                 _mlp_activation: str = "silu",
                 _mlp_use_layernorm: bool = False,
                 _input_preprocessing: str = "knet"):
        self.state_path = state_path
        self.config_path = config_path
        self._mlp_width = int(_mlp_width)
        self._mlp_layers = int(_mlp_layers)
        self._mlp_activation = str(_mlp_activation).lower()
        self._mlp_use_layernorm = bool(_mlp_use_layernorm)
        self._input_preprocessing = str(_input_preprocessing)
        self._teacher = None
        self._scaler = None

    def load(self, device: str):
        if self._teacher is not None:
            return self._teacher
        import torch  # local import — see docstring.
        try:
            raw = torch.load(self.state_path, map_location="cpu",
                             weights_only=True)
        except Exception:
            raw = torch.load(self.state_path, map_location="cpu",
                             weights_only=False)
        try:
            from ctle_dagger_common import PlainMLP  # type: ignore
        except Exception as e:
            raise SystemExit(
                f"[mlp-load] cannot import PlainMLP from ctle_dagger_common: {e}. "
                "Run from the kirchhoffnet_realistic directory or add it to PYTHONPATH.")
        cfg = None
        if self.config_path is not None and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as e:
                logger.warning("failed to read %s: %s; using defaults",
                               self.config_path, e)
        if cfg is None and isinstance(raw, dict) and "config" in raw:
            cfg = raw["config"]
        arch = {
            "trunk_width": int((cfg or {}).get("trunk_width", self._mlp_width)),
            "trunk_layers": int((cfg or {}).get("trunk_layers", self._mlp_layers)),
            "activation": str((cfg or {}).get("activation", self._mlp_activation)).lower(),
            "use_layernorm": bool((cfg or {}).get("use_layernorm", self._mlp_use_layernorm)),
            "input_preprocessing": str((cfg or {}).get("input_preprocessing",
                                                       self._input_preprocessing)),
            "use_log_features": (cfg or {}).get("use_log_features", None),
        }
        act_map = {"silu": torch.nn.SiLU, "gelu": torch.nn.GELU}
        if arch["activation"] not in act_map:
            raise SystemExit(
                f"[mlp-load] unsupported activation {arch['activation']!r}; "
                f"expected one of {sorted(act_map)}.")
        act_cls = act_map[arch["activation"]]
        eff_log_features = arch["use_log_features"]
        if arch["input_preprocessing"] == "knet" and eff_log_features is None:
            eff_log_features = False
        elif eff_log_features is None:
            eff_log_features = True
        self._teacher = PlainMLP(
            trunk_width=arch["trunk_width"],
            trunk_layers=arch["trunk_layers"],
            activation=act_cls,
            use_layernorm=arch["use_layernorm"],
            use_log_features=bool(eff_log_features),
        )
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        try:
            self._teacher.load_state_dict(state)
        except Exception as e:
            raise SystemExit(
                f"[mlp-load] state_dict mismatch for arch "
                f"(W={arch['trunk_width']}, L={arch['trunk_layers']}, "
                f"act={arch['activation']}, LN={arch['use_layernorm']}, "
                f"logfeat={eff_log_features}, prep={arch['input_preprocessing']}): "
                f"{e}. Check --mlp-* flags vs embedded/sidecar config.")

        # Optional v2 scaler attach. Validate keys first so a partial config
        # produces a clear warning instead of a TypeError mid-attach.
        scaler = (cfg or {}).get("scaler")
        if isinstance(scaler, dict):
            prep = str(arch["input_preprocessing"])
            need = ["scaler_p_scale", "scaler_p_mean",
                    "eye_scale_j", "eye_scale_h", "eye_scale_w"]
            if prep == "knet":
                have_min = cfg.get("input_log_min") is not None
                have_max = cfg.get("input_log_max") is not None
                missing = [k for k in need if scaler.get(k) is None]
                if missing or not (have_min and have_max):
                    logger.warning(
                        "scaler config incomplete for knet prep "
                        "(missing=%s, input_log_min=%s, input_log_max=%s); "
                        "forward() will fail until attach_scaler() is called manually",
                        missing, have_min, have_max)
                else:
                    self._attach(v2_cfg=cfg, scaler=scaler, arch=arch)
            else:
                missing = [k for k in need if scaler.get(k) is None]
                if missing:
                    logger.warning(
                        "scaler config incomplete for q75 prep (missing=%s); "
                        "forward() will fail until attach_scaler() is called manually",
                        missing)
                else:
                    self._attach(v2_cfg=cfg, scaler=scaler, arch=arch)
        self._teacher = self._teacher.to(device)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad = False
        self._scaler = arch
        return self._teacher

    def _attach(self, v2_cfg: dict, scaler: dict, arch: dict) -> None:
        try:
            self._teacher.attach_scaler(
                scaler_p_scale=float(scaler["scaler_p_scale"]),
                scaler_p_mean=float(scaler["scaler_p_mean"]),
                eye_scale_j=float(scaler["eye_scale_j"]),
                eye_scale_h=float(scaler["eye_scale_h"]),
                eye_scale_w=float(scaler["eye_scale_w"]),
                input_preprocessing=str(arch["input_preprocessing"]),
                input_log_min=np.asarray(v2_cfg.get("input_log_min"), dtype=np.float32)
                               if v2_cfg.get("input_log_min") is not None else None,
                input_log_max=np.asarray(v2_cfg.get("input_log_max"), dtype=np.float32)
                               if v2_cfg.get("input_log_max") is not None else None,
            )
        except Exception as e:
            logger.warning("attach_scaler failed: %s (continuing)", e)

    @property
    def architecture(self) -> dict:
        return self._scaler or {}


def predict_mlp(teacher: MlpTeacher, specs: np.ndarray, device: str,
                batch_size: int = 512) -> np.ndarray:
    """Run the MLP teacher and return (N, 7) log10 params in PARAM_COLS order.

    Mirrors ``PlainMLP.predict()``: sigmoid(logits) -> affine log map using
    the model's own ``log_lo/log_hi`` buffers.
    """
    import torch
    mdl = teacher.load(device)
    out = np.zeros((specs.shape[0], len(PARAM_COLS)), dtype=np.float32)
    log_lo = mdl.log_lo.detach().cpu().numpy().astype(np.float32)
    log_hi = mdl.log_hi.detach().cpu().numpy().astype(np.float32)
    for i in range(0, specs.shape[0], batch_size):
        x = torch.from_numpy(specs[i:i + batch_size]).to(device)
        with torch.no_grad():
            logits = mdl(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        log_arr = log_lo + (log_hi - log_lo) * probs
        out[i:i + batch_size] = log_arr
    return out


# ---------------------------------------------------------------------------
# Stage 2 — Ocean driver (ssr0 only)
# ---------------------------------------------------------------------------

def _delete_maestro_history(tb_idx: int) -> None:
    """Best-effort maestro history cleanup. Mirrors ocean_utils_ctle.delete_maestro_history."""
    paths = [
        f"~/workarea_GF22_FDX_EXT/ECE740_local/tb_channel_CTLE_ML_{tb_idx}/maestro/history/*",
        f"~/simulation/ECE740_local/tb_channel_CTLE_ML_{tb_idx}/maestro/history/*",
        f"~/simulation/ECE740_local/tb_channel_CTLE_ML_{tb_idx}/maestro/results/maestro/*",
        f"~/workarea_GF22_FDX_EXT/ECE740_local/tb_channel_CTLE_ML_{tb_idx}/maestro/test_states/*",
    ]
    cmds = "\n".join(f"rm -r {os.path.expanduser(p)}" for p in paths)
    subprocess.run(cmds, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def write_spec_ocn(template_dir: Path, tb_idx: int, var_map: dict,
                   csv_path: Path, mock_template: bool = False) -> str:
    """Render one OCN for a (tb, params) tuple via regex-update of the template.

    If ``mock_template`` is True, a minimal stand-in template is materialized
    on first use so dry-runs without the full ideal-Bob tree still proceed.

    Returns the on-disk ocn path. Reuses the regex approach from
    ocean_utils_ctle.update_ocn_ctle with use_new_templates=True (ECE740 +
    tb_channel_CTLE_ML_{tb_idx} + 7-param schema).
    """
    import re
    base = template_dir / f"template_{tb_idx}.ocn"
    if not base.exists():
        if mock_template:
            template_dir.mkdir(parents=True, exist_ok=True)
            base.write_text(_SYNTHETIC_TEMPLATE_BODY, encoding="utf-8")
        else:
            raise FileNotFoundError(f"template missing: {base}")
    content = base.read_text(encoding="utf-8")

    # Swap schematic lib/cell/view to ECE740 + tb_channel_CTLE_ML_{tb_idx}.
    # NOTE: real ideal-Bob templates spell the lib "ECE740_local" and the
    # TargetCellView call carries NO ?mode arg, so both are optional here.
    content = re.sub(
        r'"ECE740(?:_local)?"\s+"tb_channel_CTLE_ML_\d+"\s+"schematic"',
        lambda m: f'"ECE740" "tb_channel_CTLE_ML_{tb_idx}" "schematic"',
        content,
    )
    content = re.sub(
        r'ocnxlTargetCellView\(\s*"[^"]*"\s*"[^"]*"\s*"maestro"\s*(?:\?mode\s*"[^"]*"\s*)?\)',
        lambda m: f'ocnxlTargetCellView( "ECE740" "tb_channel_CTLE_ML_{tb_idx}" "maestro" ?mode "r" )',
        content,
    )
    # Remap the design() testbench pointer too; otherwise every worker would
    # simulate TB 1's schematic regardless of tb_idx.
    content = re.sub(
        r'design\(\s*"[^"]*"\s*"tb_channel_CTLE_ML_\d+"\s*"schematic"\s*\)',
        lambda m: f'design( "ECE740" "tb_channel_CTLE_ML_{tb_idx}" "schematic")',
        content,
    )
    # Force batch / non-interactive.
    content = re.sub(
        r'envSetVal\(\s*"maestro\.simulation"\s+"interactiveA"\s+\'boolean\s+t\s*\)',
        lambda m: 'envSetVal("maestro.simulation" "interactiveA" \'boolean nil)',
        content,
    )
    content = re.sub(
        r'envSetVal\(\s*"spectre\.envOpts"\s+"controlMode"\s+\'string\s+"interactive"\s*\)',
        lambda m: 'envSetVal("spectre.envOpts" "controlMode" \'string "batch")',
        content,
    )
    # Strip old desVar for the 7 tunables and re-emit ours.
    for name in ("VDD", "Rs", "Cs", "Rd", "ind", "current", "fW"):
        content = re.sub(
            rf'desVar\(\s*"{name}"\s+[^)]+\)',
            lambda m, n=name, v=var_map[name]: f'desVar(   "{n}" {v}  )',
            content,
        )
    # Repoint export. POSIX separators: OCN runs on Linux; str(Path) on a
    # Windows dev box would otherwise emit backslashes into the script.
    csv_str = csv_path.as_posix() if isinstance(csv_path, Path) else str(csv_path).replace("\\", "/")
    content = re.sub(
        r'ocnxlExportOutputView\(\s*"[^"]*"\s+"Detail"\s*\)',
        lambda m: f'ocnxlExportOutputView( "{csv_str}" "Detail")',
        content,
    )
    return content


def run_simulation(ocn_file: Path, cwd: Path, pdk_source: str,
                   ocean_bin: str = "ocean", timeout: int = 600) -> float:
    wrapper = cwd / "run_ocean.csh"
    ocn_posix = ocn_file.as_posix() if isinstance(ocn_file, Path) else str(ocn_file).replace("\\", "/")
    wrapper.write_text(
        "#!/bin/csh -f\n"
        "cd /home/annaik/workarea_GF22_FDX_EXT\n"
        f"{pdk_source}\n"
        f"{ocean_bin} -nograph << EOF\n"
        f'load "{ocn_posix}"\n'
        "exit()\n"
        "EOF\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    t0 = time.time()
    try:
        subprocess.run(
            ["/bin/csh", str(wrapper)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("ocean timeout after %ds on %s", timeout, ocn_file)
        return -1.0
    return time.time() - t0


def _wait_for_csv(path: Path, timeout: int, interval: float = 5.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(interval)
    return False


def _parse_ctle_csv(csv_path: Path) -> dict:
    """Re-implementation of ocean_utils_ctle.parse_ctle_csv (avoids the import)."""
    if not csv_path.exists():
        return {"_error": 1}
    out = {}
    error_count = 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                key = row[1].strip()
                val = row[2].strip()
                if not key or not val or val in ("eval err", "sim err"):
                    if val in ("eval err", "sim err"):
                        error_count += 1
                    continue
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    except Exception as e:
        out["_error"] = str(e)
        return out
    out["_error_count"] = error_count
    return out


# Minimal synthetic template (only desVars the regex replaces + required
# header lines). Used for local dry-runs when --mock-template is set.
_SYNTHETIC_TEMPLATE_BODY = """\
;====================Set to Maestro mode assembler ============================
ocnSetXLMode("assembler")
ocnxlProjectDir( "~/simulation" )
ocnxlTargetCellView( "ECE740_local" "tb_channel_CTLE_ML_1" "maestro" )
ocnxlResultsLocation( "" )
ocnxlSimResultsLocation( "" )
ocnxlMaxJobFail( 20 )

;====================== Tests setup ============================================

;---------- Test "ECE740_tb_channel_CTLE_ML_1_1" -------------
ocnxlBeginTest("ECE740_tb_channel_CTLE_ML_1_1")
simulator( 'spectre )
design( "ECE740" "tb_channel_CTLE_ML_1" "schematic")
modelFile( '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre") )
analysis('ac ?start "0.1G" ?stop "100G")
analysis('tran ?stop "20n")
analysis('dc ?saveOppoint t)
desVar( "VDD" 1.0 )
desVar( "Rs" 100 )
desVar( "Cs" 1e-12 )
desVar( "Rd" 100 )
desVar( "ind" 1e-9 )
desVar( "current" 1e-3 )
desVar( "fW" 1e-6 )
desVar( "Rload" 50 )
desVar( "Rsource" 0 )
desVar( "L" 20n )
desVar( "Vin_DC" 0.7 )
desVar( "Vin_AC" 1 )
desVar( "V_zero" 0.5 )
desVar( "V_one" 0.9 )
desVar( "bitperiod" "1/56G" )
desVar( "wireopt" 19 )
envSetVal("spectre.envOpts" "controlMode" 'string "interactive")
envSetVal("maestro.simulation" "interactiveA" 'boolean t)
ocnxlOutputExpr( "(VAR(\"current\") * 4 * VAR(\"VDD\"))" ?name "power" ?evalType 'point)
ocnxlOutputExpr( "1.0" ?name "eye_maxHeight_norm Vout_2 56G" ?evalType 'point)
ocnxlOutputExpr( "1.0" ?name "eye_maxWidth_norm Vout_2 56G" ?evalType 'point)
ocnxlOutputExpr( "1.0" ?name "eye_p2pJitterAverage_norm stage 2" ?evalType 'point)
ocnxlEndTest()

ocnxlLocalParametricSet( "ECE740_tb_channel_CTLE_ML_1_1" '("wireopt" "bitperiod" "V_one" "V_zero" "Vin_AC" "Vin_DC" "L" "Rsource" "Rload" "fW" "current" "ind" "Rd" "Cs" "Rs" "VDD"))
ocnxlRun( ?mode 'sweepsAndCorners ?nominalCornerEnabled t ?allCornersEnabled t ?allSweepsEnabled t)
ocnxlOutputSummary(?exprSummary t ?specSummary t ?detailed t ?wave t)
ocnxlOpenResults()

ocnxlExportOutputView( "./CSV_results/placeholder/result.csv" "Detail")

ocnxlEndXLMode("assembler")
"""


# Mocked parser used by --smoke / local mode without Cadence.
def _mocked_parser(idx: int, target_spec: np.ndarray) -> dict:
    """Deterministic stand-in: returns the target as the 'measured' value
    for power/jitter (no degradation) and 80% of target for height/width
    (mild degradation). One spec in 5 has an error injection."""
    is_err = (idx % 5 == 4)
    if is_err:
        return {"_error": 1}
    p, j, h, w = (float(x) for x in target_spec)
    return {
        "Power": p * 1.05,
        "eye_p2pJitterAverage_norm stage 2": j * 1.03,
        "eye_maxHeight_norm Vout_2 56G": h * 0.78,
        "eye_maxWidth_norm Vout_2 56G": w * 0.85,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _ocean_worker(spec_idx: int, spec: np.ndarray, params_log: np.ndarray,
                  tb_idx: int, spec_dir: Path, template_dir: Path,
                  pdk_source: str, sim_timeout: int, ocean_bin: str,
                  do_launch: bool, do_history_delete: bool,
                  mock_template: bool = False) -> dict:
    """Single spec pipeline. Writes OCN (+ sim launch) and parses result CSV."""
    spec_dir.mkdir(parents=True, exist_ok=True)
    params_phys = _log_to_physical(params_log[spec_idx:spec_idx + 1])[0]
    var_map = {name: float(params_phys[i]) for i, name in enumerate(PARAM_COLS)}
    # Save inputs.
    (spec_dir / "params.json").write_text(json.dumps(var_map, indent=2))
    (spec_dir / "target.json").write_text(
        json.dumps({SPEC_INPUT_COLS[i]: float(spec[i])
                    for i in range(4)}, indent=2))

    ocn_text = write_spec_ocn(template_dir, tb_idx, var_map,
                              spec_dir / "result.csv",
                              mock_template=mock_template)
    ocn_path = spec_dir / "spec.ocn"
    ocn_path.write_text(ocn_text, encoding="utf-8")

    if not do_launch:
        parsed = _mocked_parser(spec_idx, spec)
        parsed["_mocked"] = True
    else:
        if do_history_delete:
            _delete_maestro_history(tb_idx)
        run_simulation(ocn_path, spec_dir, pdk_source,
                       ocean_bin=ocean_bin, timeout=sim_timeout)
        if not _wait_for_csv(spec_dir / "result.csv", timeout=sim_timeout):
            parsed = {"_error": 1, "_reason": "timeout"}
        else:
            parsed = _parse_ctle_csv(spec_dir / "result.csv")
    (spec_dir / "parsed.json").write_text(json.dumps(parsed, indent=2))
    return parsed


def _worker_entry(payload: tuple) -> dict:
    (spec_idx, spec, params_log, tb_idx, spec_dir_str, template_dir_str,
     pdk_source, sim_timeout, ocean_bin, do_launch, do_history_delete,
     mock_template) = payload
    return _ocean_worker(
        spec_idx, spec, params_log, tb_idx,
        Path(spec_dir_str), Path(template_dir_str), pdk_source,
        sim_timeout, ocean_bin, do_launch, do_history_delete,
        mock_template=mock_template,
    )


# ---------------------------------------------------------------------------
# Stage 3 — scoring
# ---------------------------------------------------------------------------

def _one_sided_degrade(pred: float, target: float, kind: str) -> float:
    """Mirror of compute_forward_errors in ctle_dagger_common."""
    eps = 1e-6
    target = max(float(target), eps)
    pred = float(pred)
    if kind == "power" or kind == "jitter":
        return max((pred - target) / target, 0.0)
    if kind == "height" or kind == "width":
        return max((target - pred) / target, 0.0)
    raise ValueError(kind)


def score_spec(target: np.ndarray, measured: dict, key_map: dict,
               degrade_thr: float, min_degraded_dims: int) -> dict:
    """Recompute the ZIG-proxy degrade rule from Cadence-measured values."""
    row = {
        "power": float(target[0]),
        "jitter": float(target[1]),
        "height": float(target[2]),
        "width":  float(target[3]),
    }
    measured_ok = True
    sim_error = bool(measured.get("_error"))
    pred = {}
    errs = {}
    degraded = []
    for dim in ("power", "jitter", "height", "width"):
        key = key_map[dim]
        if key not in measured or sim_error:
            measured_ok = False
            pred[dim] = float("nan")
            errs[dim] = float("inf")
            degraded.append(True)
            continue
        try:
            v = float(measured[key])
        except (TypeError, ValueError):
            measured_ok = False
            pred[dim] = float("nan")
            errs[dim] = float("inf")
            degraded.append(True)
            continue
        if not np.isfinite(v):
            measured_ok = False
            pred[dim] = float("nan")
            errs[dim] = float("inf")
            degraded.append(True)
            continue
        v = v
        if v <= 0.0:
            measured_ok = False
            pred[dim] = float("nan")
            errs[dim] = float("inf")
            degraded.append(True)
            continue
        pred[dim] = v
        errs[dim] = _one_sided_degrade(v, row[dim], dim)
        degraded.append(errs[dim] >= degrade_thr)
    deg_count = int(sum(degraded))
    failure = (not measured_ok) or (deg_count >= min_degraded_dims)
    return {
        "target": row,
        "measured": pred,
        "errors": errs,
        "degraded": degraded,
        "degraded_dims": deg_count,
        "failure": bool(failure),
        "error": bool(sim_error),
    }


def write_results_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx",
            "tgt_power", "tgt_jitter", "tgt_height", "tgt_width",
            "pred_power", "pred_jitter", "pred_height", "pred_width",
            "err_power", "err_jitter", "err_height", "err_width",
            "degraded_dims", "failure", "error",
        ])
        for r in rows:
            t = r["target"]; m = r["measured"]; e = r["errors"]
            w.writerow([
                r["idx"],
                t["power"], t["jitter"], t["height"], t["width"],
                m["power"], m["jitter"], m["height"], m["width"],
                e["power"], e["jitter"], e["height"], e["width"],
                r["degraded_dims"], int(r["failure"]), int(r["error"]),
            ])


def summarize(rows: list[dict], args: argparse.Namespace, ckpt_sha: str,
              template_sha: str) -> dict:
    n = len(rows)
    failures = sum(1 for r in rows if r["failure"])
    errors = sum(1 for r in rows if r["error"])
    dim_fail = np.zeros(4, dtype=np.int64)
    for r in rows:
        for i, d in enumerate(("power", "jitter", "height", "width")):
            if r["degraded"][i]:
                dim_fail[i] += 1
    dim_rates = (dim_fail / max(n, 1)).tolist()
    return {
        "n": n,
        "failures": failures,
        "failure_rate": failures / max(n, 1),
        "error_rate": errors / max(n, 1),
        "dim_degrade_rates": {
            "power":   dim_rates[0],
            "jitter":  dim_rates[1],
            "height":  dim_rates[2],
            "width":   dim_rates[3],
        },
        "ckpt": str(args.ckpt),
        "ckpt_sha256": ckpt_sha,
        "template_sha256": template_sha,
        "degrade_thr": args.degrade_thr,
        "min_degraded_dims": args.min_degraded_dims,
        "key_map": args.key_map_dict,
        "model_kind": args.model_kind,
        "specs_source": str(args.specs_npz or args.specs_csv),
        "n_specs_requested": args.n_specs,
        "spec_offset": args.spec_offset,
        "seed": args.seed,
        "enable_ocean": args.enable_ocean,
        "dry_run": args.dry_run,
        "hostname": socket.gethostname(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(p: Path) -> str:
    """Stable hash of (path, mtime, size) for every .ocn under p."""
    if not p.exists():
        return "missing"
    files = sorted(p.glob("*.ocn"))
    h = hashlib.sha256()
    for f in files:
        st = f.stat()
        h.update(str(f.relative_to(p)).encode())
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
    return h.hexdigest()


def parse_key_map(raw: Optional[str]) -> dict:
    if not raw:
        return dict(DEFAULT_KEY_MAP)
    try:
        m = json.loads(raw)
    except Exception as e:
        raise SystemExit(f"[key-map] invalid JSON: {e}")
    missing = [k for k in DEFAULT_KEY_MAP if k not in m]
    if missing:
        raise SystemExit(f"[key-map] missing keys: {missing}")
    return {k: str(m[k]) for k in DEFAULT_KEY_MAP}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_offline_score(args: argparse.Namespace) -> int:
    """Re-score an existing results.csv without re-running anything."""
    if args.offline_csv is None:
        raise SystemExit("--offline-csv requires a path")
    args.key_map_dict = parse_key_map(args.key_map)
    key_map = args.key_map_dict
    rows = []
    with open(args.offline_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tgt = np.array([
                float(r["tgt_power"]), float(r["tgt_jitter"]),
                float(r["tgt_height"]), float(r["tgt_width"]),
            ], dtype=np.float32)
            measured = {
                key_map["power"]:  float(r["pred_power"]),
                key_map["jitter"]: float(r["pred_jitter"]),
                key_map["height"]: float(r["pred_height"]),
                key_map["width"]:  float(r["pred_width"]),
            }
            score = score_spec(tgt, measured, key_map,
                               args.degrade_thr, args.min_degraded_dims)
            score["idx"] = int(r["idx"])
            rows.append(score)
    out = args.offline_csv.with_name("summary_offline.json")
    out.write_text(json.dumps(summarize(
        rows, args, sha256_file(args.ckpt) if args.ckpt.exists() else "n/a",
        sha256_dir(args.template_dir)), indent=2))
    print(json.dumps(summarize(
        rows, args, sha256_file(args.ckpt) if args.ckpt.exists() else "n/a",
        sha256_dir(args.template_dir)), indent=2))
    return 0


def run_dump_keys(args: argparse.Namespace) -> int:
    if args.run_tag is None:
        raise SystemExit("[dump-keys] requires --run-tag")
    spec_root = Path(os.path.expanduser(str(args.work_root))) / args.run_tag
    if not spec_root.exists():
        raise SystemExit(f"[dump-keys] {spec_root} does not exist")
    for child in sorted(spec_root.glob("spec_*")):
        rcsv = child / "result.csv"
        if rcsv.exists():
            parsed = _parse_ctle_csv(rcsv)
            print(f"# {child.name} keys: {sorted(parsed.keys())}")
            return 0
    raise SystemExit(f"[dump-keys] no result.csv under {spec_root}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.key_map_dict = parse_key_map(args.key_map)

    if args.smoke:
        args.n_specs = 2
        args.tb_ids = args.tb_ids.split(",")[0]
        if not args.enable_ocean and not args.dry_run:
            args.dry_run = True

    if args.offline_csv is not None:
        return run_offline_score(args)
    if args.dump_keys:
        return run_dump_keys(args)

    # 1. Specs.
    specs = load_specs(args)
    logger.info("loaded %d specs (shape=%s)", specs.shape[0], specs.shape)

    # 2. Setup work dir.
    if args.run_tag is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.run_tag = f"{args.ckpt.stem}_{args.model_kind}_{ts}"
    work_root = args.work_root.expanduser() / args.run_tag
    work_root.mkdir(parents=True, exist_ok=True)
    args.work_root = work_root  # downstream uses this.

    tb_ids = [int(x) for x in args.tb_ids.split(",") if x.strip()]
    if not tb_ids:
        raise SystemExit("--tb-ids must list at least one TB id")
    n_jobs = args.jobs if args.jobs else len(tb_ids)
    n_jobs = max(1, min(n_jobs, len(tb_ids)))

    # 3. Inference (skipped when --pred-params-csv is supplied).
    if args.pred_params_csv is not None:
        params_log = _read_pred_params_csv(args.pred_params_csv, len(specs))
        logger.info("stage-1 skipped: loaded %s from %s",
                    params_log.shape, args.pred_params_csv)
    elif args.model_kind == "mlp":
        teacher = MlpTeacher(args.ckpt,
                             args.ckpt.with_name("teacher_config.json"),
                             _mlp_width=args.mlp_width,
                             _mlp_layers=args.mlp_layers,
                             _mlp_activation=args.mlp_activation,
                             _mlp_use_layernorm=args.mlp_use_layernorm,
                             _input_preprocessing=args.input_preprocessing)
        params_log = predict_mlp(teacher, specs, args.device)
    else:
        raise SystemExit(
            "KNet checkpoints cannot be rebuilt by this script yet. "
            "Pre-run inference yourself, save (N,7) log10 params in "
            "PARAM_COLS order, and pass them via --pred-params-csv "
            "(Stage-1 is then skipped; Ocean + scoring run normally). "
            "--kn-config-json is reserved for a future auto-rebuild."
        )

    # Save stage-1 outputs.
    np.savez_compressed(
        work_root / "stage1_outputs.npz",
        pred_params_log=params_log,
        target_specs=specs,
    )
    with open(work_root / "pred_params_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PARAM_COLS)
        for row in params_log:
            w.writerow([f"{x:.6e}" for x in row])
    with open(work_root / "target_specs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SPEC_INPUT_COLS)
        for row in specs:
            w.writerow([f"{x:.6e}" for x in row])
    logger.info("stage-1 done: saved to %s", work_root)

    # 4. Manifest (placeholder; written after stage-2 to capture post-mock template hash).
    manifest_template_sha = sha256_dir(args.template_dir)
    cli_serializable = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            cli_serializable[k] = str(v)
        elif isinstance(v, (list, tuple)):
            cli_serializable[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            cli_serializable[k] = v
    manifest = {
        "cli": cli_serializable,
        "ckpt_sha256": sha256_file(args.ckpt) if args.ckpt.exists() else "missing",
        "template_sha256": manifest_template_sha,
        "n_specs": int(specs.shape[0]),
        "tb_ids": tb_ids,
        "n_jobs": n_jobs,
        "enable_ocean": args.enable_ocean,
        "dry_run": args.dry_run,
    }

    # 5. Stage 2 — Ocean.
    do_launch = args.enable_ocean and not args.dry_run
    do_history_delete = do_launch and not args.no_delete_history

    payloads = []
    for i, spec in enumerate(specs):
        tb_idx = tb_ids[i % len(tb_ids)]
        spec_dir = work_root / f"spec_{i:05d}"
        parsed_path = spec_dir / "parsed.json"
        if args.resume and parsed_path.exists():
            try:
                prev = json.loads(parsed_path.read_text())
                # Mocked entries (dry-run/local) must NOT satisfy a later
                # --enable-ocean resume; only real ocean results carry over.
                if do_launch and prev.get("_mocked"):
                    pass
                else:
                    continue
            except Exception:
                pass
        payloads.append((
            i, spec, params_log, tb_idx, str(spec_dir),
            str(args.template_dir), args.pdk_source_cmd, args.sim_timeout,
            args.ocean_bin, do_launch, do_history_delete,
            args.mock_template,
        ))

    if not payloads:
        logger.info("resume: %d specs already done; skipping stage 2",
                    specs.shape[0])
    else:
        logger.info("stage-2: launching %d specs (do_launch=%s, jobs=%d)",
                    len(payloads), do_launch, n_jobs)
        if n_jobs == 1:
            for p in payloads:
                _worker_entry(p)
        else:
            # Default start method (fork on Linux, spawn elsewhere): the
            # payload is plain picklable data and _worker_entry is module-level.
            ctx = mp.get_context()
            with ctx.Pool(processes=n_jobs) as pool:
                pool.map(_worker_entry, payloads)

    # Refresh template sha now that --mock-template may have populated it.
    manifest["template_sha256"] = sha256_dir(args.template_dir)
    (work_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 6. Stage 3 — scoring.
    rows = []
    for i, spec in enumerate(specs):
        spec_dir = work_root / f"spec_{i:05d}"
        parsed_path = spec_dir / "parsed.json"
        if not parsed_path.exists():
            rows.append({
                "idx": i, "target": {k: float(spec[j]) for j, k in enumerate(SPEC_INPUT_COLS)},
                "measured": {k: float("nan") for k in ("power", "jitter", "height", "width")},
                "errors": {k: float("inf") for k in ("power", "jitter", "height", "width")},
                "degraded": [True, True, True, True],
                "degraded_dims": 4,
                "failure": True,
                "error": True,
            })
            continue
        measured = json.loads(parsed_path.read_text())
        score = score_spec(spec, measured, args.key_map_dict,
                           args.degrade_thr, args.min_degraded_dims)
        score["idx"] = i
        rows.append(score)

    results_csv = work_root / "results.csv"
    write_results_csv(rows, results_csv)
    summary = summarize(rows, args, manifest["ckpt_sha256"],
                        manifest["template_sha256"])
    (work_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
