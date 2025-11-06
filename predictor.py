# model_predictor.py
# Reproduces notebook Logistic probabilities 1:1 using exported artifacts.
# Artifacts expected in ./models :
#   - logistic_signed_coefficients.json  (dict: feature -> signed coef)
#   - logistic_model_meta.json           (dict: {"feature_names": [...], "intercept": float})
#   - scaler_mean.json                   (dict: feature -> mean)
#   - scaler_scale.json                  (dict: feature -> scale)
#
# Usage:
#   from model_predictor import predict_logistic_from_artifacts
#   pA, pB, used = predict_logistic_from_artifacts(teamA_dict, teamB_dict, location="home")
#
# Set DEBUG=True to print any missing features that get filled with 0.0.

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

ARTIFACTS = Path("models")
DEBUG = False  # set True to log missing features mapping

def _load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def load_logistic_artifacts(artifacts_dir: Path = ARTIFACTS) -> Tuple[List[str], List[float], float, List[float], List[float]]:
    meta_path   = artifacts_dir / "logistic_model_meta.json"
    coef_path   = artifacts_dir / "logistic_signed_coefficients.json"
    mean_path   = artifacts_dir / "scaler_mean.json"
    scale_path  = artifacts_dir / "scaler_scale.json"

    meta   = _load_json(meta_path)
    coefs  = _load_json(coef_path)
    means  = _load_json(mean_path)
    scales = _load_json(scale_path)

    feature_names = list(meta["feature_names"])
    intercept     = float(meta.get("intercept", 0.0))

    # Align vectors to feature_names order (exactly like notebook)
    coef_vec  = [float(coefs.get(f, 0.0))  for f in feature_names]
    mean_vec  = [float(means.get(f, 0.0))  for f in feature_names]
    scale_vec = [float(scales.get(f, 1.0)) for f in feature_names]

    return feature_names, coef_vec, intercept, mean_vec, scale_vec

def _safe_get(d: Dict, k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _home_advantage_value(location: str | None) -> float:
    loc = (location or "").strip().lower()
    if loc == "home": return 1.0
    if loc == "away": return -1.0
    return 0.0  # neutral or unknown

def build_features_from_live(teamA: Dict, teamB: Dict, location: str, feature_names: List[str]) -> Tuple[List[float], Dict[str, float]]:
    """
    Build the exact feature vector used in the notebook from live stats dicts.
    Expected live keys (adjust mapping below if your data_loader uses different names):
        AdjO, AdjD,
        eFG_off, eFG_def,
        ThreeP_off, ThreeP_def,
        TwoP_off, TwoP_def,
        FT_off, FT_def,
        OR_off, DR_def,
        TOV_off, TOV_def,
        Tempo
    Unknown features are filled with 0.0 to preserve the trained layout.
    """
    A = dict(teamA)
    B = dict(teamB)

    f = {}

    # Core
    f["home_advantage"] = _home_advantage_value(location)
    f["OffRating_diff"] = _safe_get(A, "AdjO")      - _safe_get(B, "AdjO")
    f["DefRating_diff"] = _safe_get(A, "AdjD")      - _safe_get(B, "AdjD")

    # Offense diffs (A - B)
    f["Off_eFG_diff"]   = _safe_get(A, "eFG_off")   - _safe_get(B, "eFG_off")
    f["Off_3P%_diff"]   = _safe_get(A, "ThreeP_off")- _safe_get(B, "ThreeP_off")
    f["Off_2P%_diff"]   = _safe_get(A, "TwoP_off")  - _safe_get(B, "TwoP_off")
    f["Off_FT%_diff"]   = _safe_get(A, "FT_off")    - _safe_get(B, "FT_off")
    f["Off_3PA%_diff"]  = _safe_get(A, "ThreePA_off", 0.0) - _safe_get(B, "ThreePA_off", 0.0)
    f["Off_TOV%_diff"]  = _safe_get(A, "TOV_off")   - _safe_get(B, "TOV_off")
    f["Off_FTR_diff"]   = _safe_get(A, "FTR_off", 0.0)     - _safe_get(B, "FTR_off", 0.0)
    f["Off_A%_diff"]    = _safe_get(A, "AST_off", 0.0)     - _safe_get(B, "AST_off", 0.0)
    f["Off_OR%_diff"]   = _safe_get(A, "OR_off")    - _safe_get(B, "OR_off")

    # Defense diffs (A - B)
    f["Def_eFG_allowed_diff"]  = _safe_get(A, "eFG_def")    - _safe_get(B, "eFG_def")
    f["Def_3P%_allowed_diff"]  = _safe_get(A, "ThreeP_def") - _safe_get(B, "ThreeP_def")
    f["Def_2P%_allowed_diff"]  = _safe_get(A, "TwoP_def")   - _safe_get(B, "TwoP_def")
    f["Def_FT%_allowed_diff"]  = _safe_get(A, "FT_def")     - _safe_get(B, "FT_def")
    f["Def_3PA%_allowed_diff"] = _safe_get(A, "ThreePA_def", 0.0) - _safe_get(B, "ThreePA_def", 0.0)
    f["Def_TOV%_diff"]         = _safe_get(A, "TOV_def")    - _safe_get(B, "TOV_def")
    f["Def_DR%_diff"]          = _safe_get(A, "DR_def", 0.0)       - _safe_get(B, "DR_def", 0.0)
    f["Def_BLK%_diff"]         = _safe_get(A, "BLK_def", 0.0)      - _safe_get(B, "BLK_def", 0.0)
    f["Def_STL%_diff"]         = _safe_get(A, "STL_def", 0.0)      - _safe_get(B, "STL_def", 0.0)

    # Tempo / pace
    f["Poss_diff"] = _safe_get(A, "Tempo", 0.0) - _safe_get(B, "Tempo", 0.0)

    # Interactions (present in your notebook)
    f["home_x_offrat"] = f["home_advantage"] * f["OffRating_diff"]
    f["home_x_defrat"] = f["home_advantage"] * f["DefRating_diff"]

    # Assemble vector in the exact order expected by the notebook model
    x = []
    for name in feature_names:
        if name in f:
            x.append(f[name])
        else:
            x.append(0.0)
            if DEBUG:
                print(f"[predictor] Missing live feature '{name}', filled 0.0")
    return x, f

def _sigmoid(z: float) -> float:
    # numerically stable-ish sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def predict_logistic_from_artifacts(teamA: Dict, teamB: Dict, location: str, artifacts_dir: Path = ARTIFACTS):
    """
    Returns (pA, pB, used_features_dict).
    pA is the probability the model assigns to Team A (first argument) winning.
    """
    feature_names, coef_vec, intercept, mean_vec, scale_vec = load_logistic_artifacts(artifacts_dir)
    x_raw, used = build_features_from_live(teamA, teamB, location, feature_names)

    # Standardize using the *training* scaler statistics
    x_std = []
    for v, m, s in zip(x_raw, mean_vec, scale_vec):
        if s == 0.0:
            x_std.append(0.0)
        else:
            x_std.append((v - m) / s)

    # Linear model
    z = intercept + sum(c * v for c, v in zip(coef_vec, x_std))
    pA = _sigmoid(z)
    pB = 1.0 - pA
    return pA, pB, used
