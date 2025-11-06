#data_loader.py

from __future__ import annotations

from pathlib import Path
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

DEFAULT_SEASON = None  #None = latest KenPom; or set e.g., 2025
CACHE_DIR = Path("data"); CACHE_DIR.mkdir(exist_ok=True, parents=True)
CACHE_FILE = lambda season: CACHE_DIR / (f"kenpom_team_summaries_{season}.csv" if season else "kenpom_team_summaries.csv")

#Optional kenpompy import (graceful if not installed)
try:
    from kenpompy.utils import login
    from kenpompy.summary import get_team_summaries  #common entry in kenpompy (package >= 0.2.x)
    HAVE_KENPOMPY = True
except Exception:
    HAVE_KENPOMPY = False


class CredentialError(RuntimeError): ...
class FetchError(RuntimeError): ...


def _get_kenpom_browser():
    """Create a KenPom session using env credentials; helpful error if missing/invalid."""
    user = os.getenv("KP_USERNAME")
    pwd  = os.getenv("KP_PASSWORD")
    if not user or not pwd:
        raise CredentialError(
            "KenPom credentials missing. Set environment variables KP_USERNAME and KP_PASSWORD."
        )
    try:
        return login(user, pwd)
    except Exception as e:
        raise CredentialError(f"KenPom login failed. Verify your subscription/credentials. Underlying: {e}")


def _fetch_kenpom_df(season: Optional[int]) -> pd.DataFrame:
    """Fetch team summaries via kenpompy, optionally for a given season."""
    if not HAVE_KENPOMPY:
        raise FetchError("kenpompy is not installed in this environment.")
    br = _get_kenpom_browser()
    try:
        df = get_team_summaries(br, season=season)
        #Persist a cache CSV for offline or later runs
        try:
            CACHE_FILE(season).parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(CACHE_FILE(season), index=False)
        except Exception:
            pass
        return df
    except Exception as e:
        raise FetchError(f"Failed to fetch team summaries from KenPom: {e}")


def _load_cached_df(season: Optional[int]) -> pd.DataFrame:
    """Load a pre-downloaded CSV when online fetch isn't available."""
    cf = CACHE_FILE(season)
    if not cf.exists():
        raise FetchError(
            f"Cache file not found: {cf}\n"
            "Provide kenpompy+credentials for live fetch, or add a CSV at that path."
        )
    return pd.read_csv(cf)


def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _percent_to_float(x):
    """Convert '52.3' or '52.3%' to 0.523; leave None if conversion fails."""
    if x is None:
        return None
    if isinstance(x, str):
        xs = x.strip()
        if xs.endswith('%'):
            xs = xs[:-1]
        try:
            return float(xs) / 100.0
        except Exception:
            return None
    try:
        #if already fraction like 0.523, keep; if 52.3 assume percent
        v = float(x)
        return v/100.0 if v > 1.5 else v
    except Exception:
        return None


def _map_row(row: pd.Series) -> Dict[str, float]:
    """
    Map a single DataFrame row (KenPom team summaries) to the live feature dict.
    Adjust the column names if your kenpompy version uses different labels.
    Common columns in kenpompy.summary.get_team_summaries():
      - 'Team', 'AdjO', 'AdjD', 'AdjT'
      - 'eFG_O', 'eFG_D', 'TOV_O', 'TOV_D', 'ORB', 'DRB', 'FT', 'FTR_O', 'FTR_D',
      - '3P_O', '3P_D', '2P_O', '2P_D', etc.
    """
    #Try multiple aliases per stat to be resilient across versions
    def pick(cols, default=None, pct=False):
        for c in cols:
            if c in row and pd.notna(row[c]):
                return _percent_to_float(row[c]) if pct else _coerce_float(row[c])
        return default

    out = {
        #Core efficiencies
        "AdjO": pick(["AdjO", "AdjOE", "Adj. Off"], 0.0, pct=False),
        "AdjD": pick(["AdjD", "AdjDE", "Adj. Def"], 0.0, pct=False),
        "Tempo": pick(["AdjT", "Tempo", "Adj. Tempo"], 0.0, pct=False),

        #Shooting offense/defense (fractions 0..1)
        "eFG_off":   pick(["eFG_O", "eFG%_O", "eFG Off"],  None, pct=True),
        "eFG_def":   pick(["eFG_D", "eFG%_D", "eFG Def"],  None, pct=True),
        "ThreeP_off":pick(["3P_O", "3P%_O", "3P Off"],     None, pct=True),
        "ThreeP_def":pick(["3P_D", "3P%_D", "3P Def"],     None, pct=True),
        "TwoP_off":  pick(["2P_O", "2P%_O", "2P Off"],     None, pct=True),
        "TwoP_def":  pick(["2P_D", "2P%_D", "2P Def"],     None, pct=True),
        "FT_off":    pick(["FT%_O", "FT_O", "FT Off"],     None, pct=True),
        "FT_def":    pick(["FT%_D", "FT_D", "FT Def"],     None, pct=True),

        #Possession components / rates
        "OR_off":    pick(["ORB", "OR%_O", "OR Off"],      None, pct=True),
        "DR_def":    pick(["DRB", "DR%_D", "DR Def"],      None, pct=True),
        "TOV_off":   pick(["TOV_O", "TOV%_O", "TO%_O"],    None, pct=True),
        "TOV_def":   pick(["TOV_D", "TOV%_D", "TO%_D"],    None, pct=True),
    }

    #Some KenPom tables use FTR_O / FTR_D instead of FT%
    ftr_o = pick(["FTR_O", "FTR Off"], None, pct=True)
    ftr_d = pick(["FTR_D", "FTR Def"], None, pct=True)
    if out["FT_off"] is None and ftr_o is not None:
        out["FT_off"] = ftr_o
    if out["FT_def"] is None and ftr_d is not None:
        out["FT_def"] = ftr_d

    #Final safety: default None -> 0.0 to keep the vector complete
    for k, v in list(out.items()):
        if v is None:
            out[k] = 0.0
    return out


def load_team_stats(season: Optional[int] = DEFAULT_SEASON) -> Dict[str, Dict[str, float]]:
    """
    Returns a mapping: team_name -> feature dict required by the predictor.
    Tries kenpompy+env creds first; falls back to a local CSV cache if present.
    """
    #Attempt live fetch
    df: Optional[pd.DataFrame] = None
    if HAVE_KENPOMPY:
        try:
            df = _fetch_kenpom_df(season)
        except (CredentialError, FetchError) as e:
            #Print a short, helpful message and fall back to cache
            print(f"[data_loader] {e}. Trying local cache...")

    #Fallback to cache if live fetch unavailable
    if df is None:
        df = _load_cached_df(season)

    #Basic team name column
    team_col = None
    for c in ["Team", "School", "TeamName", "team"]:
        if c in df.columns:
            team_col = c
            break
    if team_col is None:
        raise FetchError("Could not find team name column in fetched/cache data.")

    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        name = str(row[team_col]).strip()
        out[name] = _map_row(row)

    return out


#Minimal self-test for local dev
if __name__ == "__main__":
    try:
        teams = load_team_stats()
        print(f"Loaded {len(teams)} teams.")
        #print a sample
        for i, (k, v) in enumerate(teams.items()):
            print(k, {kk: round(vv, 4) for kk, vv in list(v.items())[:6]})
            if i > 3:
                break
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
