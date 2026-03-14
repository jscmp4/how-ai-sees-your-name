#!/usr/bin/env python3
"""控制种族/文化变量后重跑所有验证实验。

核心思路: 用名字在各州的分布模式估算"文化主流度"(white-state affinity),
然后在控制了文化主流度之后, 看WEAT-outcome关联是否仍然存在。

输出: data/controlled_validation.json
"""

import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
DATA = PROJECT / "data"
RAW = DATA / "raw"

# Force UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_en_scores():
    with open(DATA / "en_name_scores.json", encoding="utf-8") as f:
        return json.load(f)


def get_weat(en_scores, name):
    if not name:
        return None
    for v in [name, name.capitalize(), name.lower()]:
        if v in en_scores:
            return en_scores[v].get("composite", None)
    return None


def first_name(full):
    if not isinstance(full, str):
        return None
    if "," in full:
        parts = full.split(",")
        return parts[1].strip().split()[0].capitalize() if len(parts) >= 2 else None
    return full.strip().split()[0].capitalize()


# 2020 Census: % non-Hispanic white by state
STATE_WHITE = {
    "ME": 90.8, "VT": 89.8, "WV": 90.1, "NH": 87.2, "WY": 83.2,
    "MT": 83.4, "IA": 82.7, "ND": 81.5, "ID": 81.6, "KY": 81.3,
    "SD": 78.4, "NE": 77.5, "WI": 78.6, "MN": 77.6, "UT": 75.4,
    "IN": 76.8, "MO": 76.5, "OH": 75.4, "KS": 72.4, "PA": 73.5,
    "OR": 71.7, "MI": 72.4, "AK": 58.0, "CO": 65.1, "CT": 63.2,
    "MA": 67.6, "RI": 66.4, "AR": 68.5, "TN": 70.9, "AL": 62.4,
    "SC": 62.1, "NC": 61.2, "VA": 60.3, "DE": 57.8, "LA": 55.8,
    "FL": 51.5, "NY": 52.5, "NJ": 51.9, "GA": 50.1, "IL": 58.3,
    "AZ": 53.4, "NV": 45.9, "MD": 47.2, "MS": 55.4, "TX": 39.7,
    "CA": 34.7, "NM": 36.5, "HI": 21.7, "DC": 37.5, "WA": 63.8,
    "OK": 60.8,
}


def build_cultural_proxy(en_scores):
    """Build white-state affinity as cultural mainstream proxy."""
    state_dir = RAW / "ssa_baby_names" / "state"
    name_state = defaultdict(lambda: defaultdict(int))

    for fname in os.listdir(state_dir):
        if not fname.endswith(".TXT"):
            continue
        st = fname[:2]
        df = pd.read_csv(state_dir / fname, header=None,
                         names=["state", "sex", "year", "name", "count"])
        for _, row in df[df["year"] >= 2010].iterrows():
            name_state[row["name"]][st] += row["count"]

    cultural = {}
    for name_str, st_counts in name_state.items():
        total = sum(st_counts.values())
        if total < 100:
            continue
        white_aff = sum(STATE_WHITE.get(st, 65) * cnt
                        for st, cnt in st_counts.items()) / total
        weat = get_weat(en_scores, name_str)
        if weat is not None:
            cultural[name_str] = {
                "white_affinity": white_aff,
                "weat": weat,
                "n": total,
            }

    return cultural


def run_chicago_controlled(en_scores, cultural):
    """Re-run Chicago salary with culture control."""
    print("=" * 70)
    print("Experiment A: Chicago Salary (controlled for cultural background)")
    print("=" * 70)

    chi = pd.read_csv(RAW / "us_salary" / "chicago_salaries.csv")
    chi["first"] = chi["name"].apply(first_name)
    chi = chi.dropna(subset=["first", "annual_salary"])
    chi = chi[chi["annual_salary"] > 10000]

    chi["culture"] = chi["first"].apply(
        lambda n: cultural.get(n, cultural.get(
            n.capitalize() if n else "", {})).get("white_affinity")
    )
    chi["weat"] = chi["first"].apply(lambda n: get_weat(en_scores, n))
    chi = chi.dropna(subset=["culture", "weat"])

    # Original (uncontrolled)
    r_raw, p_raw = stats.pearsonr(chi["weat"], chi["annual_salary"])

    # Partial correlation: residualize salary on culture, then correlate with WEAT
    lr = LinearRegression()
    lr.fit(chi[["culture"]], chi["annual_salary"])
    salary_resid = chi["annual_salary"] - lr.predict(chi[["culture"]])
    r_partial, p_partial = stats.pearsonr(chi["weat"], salary_resid)

    # Also residualize WEAT on culture (double residualization)
    lr2 = LinearRegression()
    lr2.fit(chi[["culture"]], chi["weat"])
    weat_resid = chi["weat"] - lr2.predict(chi[["culture"]])
    r_double, p_double = stats.pearsonr(weat_resid, salary_resid)

    # Within-tier analysis
    chi["tier"] = pd.qcut(chi["culture"], 3,
                           labels=["minority_leaning", "mixed", "mainstream"])
    tier_results = {}
    for tier in ["minority_leaning", "mixed", "mainstream"]:
        sub = chi[chi["tier"] == tier]
        if len(sub) > 100:
            r, p = stats.pearsonr(sub["weat"], sub["annual_salary"])
            tier_results[tier] = {
                "n": len(sub),
                "r": round(float(r), 4),
                "p": float(p),
                "mean_salary": round(float(sub["annual_salary"].mean()), 0),
            }

    print(f"  Raw (no control):        r={r_raw:+.4f}  p={p_raw:.2e}")
    print(f"  Salary residualized:     r={r_partial:+.4f}  p={p_partial:.2e}")
    print(f"  Double residualized:     r={r_double:+.4f}  p={p_double:.2e}")
    print()
    for tier, d in tier_results.items():
        sig = "***" if d["p"] < 0.001 else "**" if d["p"] < 0.01 else "*" if d["p"] < 0.05 else "ns"
        print(f"  Within {tier:18s}: n={d['n']:>5d}  r={d['r']:+.4f} {sig}")

    return {
        "raw_r": round(float(r_raw), 4),
        "partial_r_salary": round(float(r_partial), 4),
        "partial_p_salary": float(p_partial),
        "double_partial_r": round(float(r_double), 4),
        "double_partial_p": float(p_double),
        "within_tier": tier_results,
    }


def run_elite_controlled(en_scores, cultural):
    """Re-run elite analysis within mainstream names only."""
    print()
    print("=" * 70)
    print("Experiment B: Elite Names (mainstream-only subset)")
    print("=" * 70)

    # Define mainstream = top 50% white_affinity
    cdf = pd.DataFrame.from_dict(cultural, orient="index")
    median_wa = cdf["white_affinity"].median()
    mainstream = set(cdf[cdf["white_affinity"] >= median_wa].index)
    print(f"  Mainstream name set: {len(mainstream)} names (white_aff >= {median_wa:.1f}%)")

    # Control baseline: mainstream SSA names
    random.seed(42)
    ms_ssa = [n for n in en_scores if n in mainstream]
    ctrl = [en_scores[n].get("composite", 0)
            for n in random.sample(ms_ssa, min(3000, len(ms_ssa)))]

    # Also build full (uncontrolled) baseline
    all_ssa = list(en_scores.keys())
    ctrl_all = [en_scores[n].get("composite", 0)
                for n in random.sample(all_ssa, 5000)]

    elite_files = {
        "Billionaires": ("data/raw/elite_names/billionaires.csv", "csv"),
        "Nobel": ("data/raw/elite_names/nobel_laureates.json", "json_dict"),
        "Senators": ("data/raw/elite_names/us_senators.json", "json_list"),
        "Olympic": ("data/raw/elite_names/olympic_gold.json", "json_list"),
    }

    results = {}
    for cat, (fpath, fmt) in elite_files.items():
        fp = PROJECT / fpath
        if not fp.exists():
            continue
        if fmt == "csv":
            df = pd.read_csv(fp)
            names = [first_name(str(n)) for n in df.iloc[:, 0].drop_duplicates()]
        elif fmt == "json_dict":
            with open(fp) as f:
                data = json.load(f)
            names = [first_name(d["name"] if isinstance(d, dict) else d) for d in data]
        else:
            with open(fp) as f:
                data = json.load(f)
            names = [first_name(d) for d in data]

        # All names
        all_scores = [s for s in (get_weat(en_scores, n) for n in names if n)
                      if s is not None]
        # Mainstream only
        ms_scores = [s for s in (get_weat(en_scores, n)
                     for n in names if n and n in mainstream) if s is not None]

        if len(ms_scores) >= 10:
            t_all, p_all = stats.ttest_ind(all_scores, ctrl_all)
            t_ms, p_ms = stats.ttest_ind(ms_scores, ctrl)
            diff_all = np.mean(all_scores) - np.mean(ctrl_all)
            diff_ms = np.mean(ms_scores) - np.mean(ctrl)

            sig_all = "***" if p_all < 0.001 else "ns"
            sig_ms = "***" if p_ms < 0.001 else "ns"

            print(f"  {cat:15s}:")
            print(f"    Uncontrolled:  n={len(all_scores):>4d}  delta={diff_all:>+.4f} {sig_all}")
            print(f"    Mainstream:    n={len(ms_scores):>4d}  delta={diff_ms:>+.4f} {sig_ms}")

            results[cat] = {
                "uncontrolled": {
                    "n": len(all_scores),
                    "diff": round(float(diff_all), 4),
                    "p": float(p_all),
                },
                "mainstream_only": {
                    "n": len(ms_scores),
                    "diff": round(float(diff_ms), 4),
                    "p": float(p_ms),
                },
            }

    return results


def run_billionaire_controlled(en_scores, cultural):
    """Re-run self-made vs inherited within mainstream names."""
    print()
    print("=" * 70)
    print("Experiment C: Self-Made vs Inherited (mainstream-only)")
    print("=" * 70)

    cdf = pd.DataFrame.from_dict(cultural, orient="index")
    mainstream = set(cdf[cdf["white_affinity"] >= cdf["white_affinity"].median()].index)

    bill = pd.read_csv(RAW / "elite_names" / "billionaires.csv")
    bill_u = bill.drop_duplicates(subset="name").copy()
    bill_u["first"] = bill_u["name"].apply(first_name)
    bill_u["weat"] = bill_u["first"].apply(lambda n: get_weat(en_scores, n))
    bill_u["is_ms"] = bill_u["first"].apply(lambda n: n in mainstream if n else False)

    # Uncontrolled
    sm_all = bill_u[bill_u["wealth.type"].isin(
        ["founder non-finance", "self-made finance"])]["weat"].dropna()
    ih_all = bill_u[bill_u["wealth.type"] == "inherited"]["weat"].dropna()

    # Mainstream only
    ms = bill_u[bill_u["is_ms"]]
    sm_ms = ms[ms["wealth.type"].isin(
        ["founder non-finance", "self-made finance"])]["weat"].dropna()
    ih_ms = ms[ms["wealth.type"] == "inherited"]["weat"].dropna()

    results = {}

    if len(sm_all) > 5 and len(ih_all) > 5:
        t1, p1 = stats.ttest_ind(sm_all, ih_all)
        print(f"  Uncontrolled:  SM={sm_all.mean():.4f} (n={len(sm_all)})"
              f"  IH={ih_all.mean():.4f} (n={len(ih_all)})"
              f"  diff={sm_all.mean()-ih_all.mean():+.4f}  p={p1:.4f}")
        results["uncontrolled"] = {
            "sm_mean": round(float(sm_all.mean()), 4),
            "ih_mean": round(float(ih_all.mean()), 4),
            "diff": round(float(sm_all.mean() - ih_all.mean()), 4),
            "p": float(p1),
        }

    if len(sm_ms) > 5 and len(ih_ms) > 5:
        t2, p2 = stats.ttest_ind(sm_ms, ih_ms)
        print(f"  Mainstream:    SM={sm_ms.mean():.4f} (n={len(sm_ms)})"
              f"  IH={ih_ms.mean():.4f} (n={len(ih_ms)})"
              f"  diff={sm_ms.mean()-ih_ms.mean():+.4f}  p={p2:.4f}")
        results["mainstream_only"] = {
            "sm_mean": round(float(sm_ms.mean()), 4),
            "ih_mean": round(float(ih_ms.mean()), 4),
            "diff": round(float(sm_ms.mean() - ih_ms.mean()), 4),
            "p": float(p2),
        }

    return results


def main():
    en_scores = load_en_scores()
    print("Building cultural proxy from SSA state data...")
    cultural = build_cultural_proxy(en_scores)
    print(f"Cultural proxy: {len(cultural)} names\n")

    results = {
        "method": "White-state affinity as cultural mainstream proxy (Census 2020 demographics x SSA state-level names 2010-2024)",
        "chicago_salary": run_chicago_controlled(en_scores, cultural),
        "elite_names": run_elite_controlled(en_scores, cultural),
        "selfmade_inherited": run_billionaire_controlled(en_scores, cultural),
    }

    out = DATA / "controlled_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Does controlling for culture change the conclusions?")
    print("=" * 70)
    ch = results["chicago_salary"]
    print(f"  Chicago salary: raw r={ch['raw_r']:+.4f} -> controlled r={ch['double_partial_r']:+.4f}")
    for cat, d in results["elite_names"].items():
        u = d["uncontrolled"]
        c = d["mainstream_only"]
        print(f"  {cat}: uncontrolled delta={u['diff']:+.4f} -> mainstream delta={c['diff']:+.4f}")
    sm = results["selfmade_inherited"]
    if "uncontrolled" in sm and "mainstream_only" in sm:
        print(f"  Self-made vs Inherited: uncontrolled={sm['uncontrolled']['diff']:+.4f}"
              f" -> mainstream={sm['mainstream_only']['diff']:+.4f}")


if __name__ == "__main__":
    main()
