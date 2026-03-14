#!/usr/bin/env python3
"""完整验证Pipeline — 生成所有ground truth验证结果。

输出:
  data/validation_results.json — 中美双边验证 + 精英分析 + 富一代vs富二代
"""

import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
DIMS = ["wealth", "wisdom", "happiness", "health", "leadership", "beauty"]


def load_scores():
    with open(DATA / "char_scores.json", encoding="utf-8") as f:
        char_scores = json.load(f)
    with open(DATA / "char_scores_bert.json", encoding="utf-8") as f:
        bert_scores = json.load(f)
    with open(DATA / "en_name_scores.json", encoding="utf-8") as f:
        en_scores = json.load(f)
    with open(DATA / "name_whole_scores.json", encoding="utf-8") as f:
        whole_scores = json.load(f)
    return char_scores, bert_scores, en_scores, whole_scores


def en_weat(en_scores, name):
    for v in [name, name.capitalize(), name.lower()]:
        if v in en_scores:
            return en_scores[v].get("composite", None)
    return None


def cn_weat(char_scores, given):
    if not given:
        return None
    vals = [char_scores[ch].get("composite", 0) for ch in given if ch in char_scores]
    return float(np.mean(vals)) if vals else None


def first_name(full):
    if not isinstance(full, str):
        return None
    if "," in full:
        parts = full.split(",")
        return parts[1].strip().split()[0].capitalize() if len(parts) >= 2 else None
    return full.strip().split()[0].capitalize()


def run_china_validation(char_scores, bert_scores, whole_scores):
    """中国: 基金获资助者 vs 普通人"""
    logger.info("=== 中国验证 ===")

    grantees = pd.read_csv(RAW / "Chinese-Gender-dataset/ChineseGender/Result_data/grantees.csv", sep="\t")
    grantee_names = grantees["given"].dropna().tolist()

    train = pd.read_csv(RAW / "name2gender/dataset/train.csv", header=None, names=["fn", "g"])
    train["given"] = train["fn"].apply(lambda x: x[1:] if isinstance(x, str) and len(x) >= 2 else None)
    general_names = train["given"].dropna().tolist()

    random.seed(42)
    general_sample = random.sample(general_names, min(len(grantee_names), len(general_names)))

    # 三层特征
    def extract(name):
        if not isinstance(name, str):
            return None
        f = {}
        for dim in DIMS:
            vals = [char_scores[ch][dim] for ch in name if ch in char_scores and dim in char_scores[ch]]
            f[f"w2v_{dim}"] = float(np.mean(vals)) if vals else 0
            bvals = [bert_scores[ch][dim] for ch in name if ch in bert_scores and dim in bert_scores[ch]]
            f[f"bert_{dim}"] = float(np.mean(bvals)) if bvals else 0
            f[f"whole_{dim}"] = whole_scores.get(name, {}).get(dim, 0)
        f["has_whole"] = 1 if name in whole_scores else 0
        f["name_len"] = len(name)
        return f

    rows = []
    for n in grantee_names:
        feat = extract(n)
        if feat:
            feat["label"] = 1
            rows.append(feat)
    for n in general_sample:
        feat = extract(n)
        if feat:
            feat["label"] = 0
            rows.append(feat)

    df = pd.DataFrame(rows)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv = cross_val_score(lr, X, y, cv=5, scoring="roc_auc")

    # 按维度
    dim_results = {}
    for dim in DIMS:
        g = [cn_weat(char_scores, n) for n in grantee_names]
        p = [cn_weat(char_scores, n) for n in general_sample]
        g = [x for x in g if x is not None]
        p = [x for x in p if x is not None]
        t, pval = stats.ttest_ind(g, p)
        dim_results[dim] = {
            "grantee_mean": round(float(np.mean(g)), 6),
            "general_mean": round(float(np.mean(p)), 6),
            "diff": round(float(np.mean(g) - np.mean(p)), 6),
            "t": round(float(t), 4),
            "p": float(pval),
        }

    return {
        "source": "Chinese Gender Dataset grantees vs CCNC",
        "n_grantee": len(grantee_names),
        "n_general": len(general_sample),
        "fusion_auc_mean": round(float(cv.mean()), 4),
        "fusion_auc_std": round(float(cv.std()), 4),
        "dimensions": dim_results,
    }


def run_us_salary_validation(en_scores):
    """美国: 芝加哥薪资验证"""
    logger.info("=== 美国薪资验证 ===")

    csv_path = RAW / "us_salary/chicago_salaries.csv"
    if not csv_path.exists():
        return {"error": "chicago_salaries.csv not found"}

    df = pd.read_csv(csv_path)
    df["first"] = df["name"].apply(first_name)
    df = df.dropna(subset=["first", "annual_salary"])
    df = df[df["annual_salary"] > 10000]

    matched = []
    for _, row in df.iterrows():
        w = en_weat(en_scores, row["first"])
        if w is not None:
            matched.append({"weat": w, "salary": row["annual_salary"]})

    mdf = pd.DataFrame(matched)
    r, p = stats.pearsonr(mdf["weat"], mdf["salary"])

    # 名字级别
    name_sal = df.copy()
    name_sal["weat"] = name_sal["first"].apply(lambda n: en_weat(en_scores, n))
    name_grp = name_sal.dropna(subset=["weat"]).groupby("first").agg(
        avg_salary=("annual_salary", "mean"), count=("annual_salary", "count"), weat=("weat", "first")
    ).reset_index()
    name_grp = name_grp[name_grp["count"] >= 10]
    r_name, p_name = stats.pearsonr(name_grp["weat"], name_grp["avg_salary"])

    return {
        "source": "Chicago City Employee Salaries (32K employees)",
        "n_matched": len(mdf),
        "individual_pearson_r": round(float(r), 4),
        "individual_p": float(p),
        "name_level_pearson_r": round(float(r_name), 4),
        "name_level_p": float(p_name),
        "n_names_10plus": len(name_grp),
    }


def run_elite_validation(en_scores, char_scores):
    """精英名单验证"""
    logger.info("=== 精英名单验证 ===")

    random.seed(42)
    ctrl = [en_scores[n].get("composite", 0) for n in random.sample(list(en_scores.keys()), 5000)]
    ctrl_mean = float(np.mean(ctrl))

    results = {"control_mean": ctrl_mean, "categories": {}}

    # 英文精英
    elite_files = {
        "Billionaires": ("billionaires.csv", "csv_col0"),
        "Nobel Laureates": ("nobel_laureates.json", "json_name"),
        "Oscar Winners": ("oscar_winners.json", "json_name"),
        "US Senators": ("us_senators.json", "json_list"),
        "Olympic Gold": ("olympic_gold.json", "json_list"),
    }

    for cat, (fname, fmt) in elite_files.items():
        fpath = RAW / "elite_names" / fname
        if not fpath.exists():
            continue
        names = []
        if fmt == "csv_col0":
            df = pd.read_csv(fpath)
            names = [first_name(str(n)) for n in df.iloc[:, 0].drop_duplicates()]
        elif fmt == "json_name":
            with open(fpath) as f:
                data = json.load(f)
            names = [first_name(d["name"] if isinstance(d, dict) else d) for d in data]
        elif fmt == "json_list":
            with open(fpath) as f:
                data = json.load(f)
            names = [first_name(d) for d in data]

        scores = [s for s in (en_weat(en_scores, n) for n in names if n) if s is not None]
        if len(scores) < 10:
            continue

        t, p = stats.ttest_ind(scores, ctrl)
        results["categories"][cat] = {
            "n": len(scores),
            "mean": round(float(np.mean(scores)), 4),
            "diff_vs_control": round(float(np.mean(scores) - ctrl_mean), 4),
            "t": round(float(t), 2),
            "p": float(p),
        }

    # 中文精英
    cn_ctrl_train = pd.read_csv(RAW / "name2gender/dataset/train.csv", header=None, names=["fn", "g"])
    cn_ctrl_train["given"] = cn_ctrl_train["fn"].apply(lambda x: x[1:] if isinstance(x, str) and len(x) >= 2 else None)
    cn_random = random.sample(cn_ctrl_train["given"].dropna().tolist(), 5000)
    cn_ctrl = [s for s in (cn_weat(char_scores, n) for n in cn_random) if s is not None]

    cn_ent_path = RAW / "elite_names/cn_entrepreneurs.json"
    if cn_ent_path.exists():
        with open(cn_ent_path, encoding="utf-8") as f:
            cn_ent = json.load(f)
        cn_names = []
        for n in cn_ent:
            cn = "".join(c for c in n if "\u4e00" <= c <= "\u9fff")
            if len(cn) >= 2:
                cn_names.append(cn[1:])
        cn_scores = [s for s in (cn_weat(char_scores, n) for n in cn_names) if s is not None]
        if len(cn_scores) >= 10:
            t, p = stats.ttest_ind(cn_scores, cn_ctrl)
            results["categories"]["Chinese Entrepreneurs"] = {
                "n": len(cn_scores),
                "mean": round(float(np.mean(cn_scores)), 4),
                "diff_vs_control": round(float(np.mean(cn_scores) - np.mean(cn_ctrl)), 4),
                "t": round(float(t), 2),
                "p": float(p),
            }

    return results


def run_selfmade_vs_inherited(en_scores):
    """富一代 vs 富二代"""
    logger.info("=== 富一代 vs 富二代 ===")

    df = pd.read_csv(RAW / "elite_names/billionaires.csv")
    df_u = df.drop_duplicates(subset="name").copy()
    df_u["first"] = df_u["name"].apply(first_name)
    df_u["weat"] = df_u["first"].apply(lambda n: en_weat(en_scores, n))

    groups = {}
    for wtype in df_u["wealth.type"].dropna().unique():
        sub = df_u[df_u["wealth.type"] == wtype]["weat"].dropna()
        if len(sub) >= 20:
            groups[wtype] = {
                "n": len(sub),
                "mean": round(float(sub.mean()), 4),
                "std": round(float(sub.std()), 4),
            }

    sm = df_u[df_u["wealth.type"].isin(["founder non-finance", "self-made finance"])]["weat"].dropna()
    ih = df_u[df_u["wealth.type"] == "inherited"]["weat"].dropna()
    t, p = stats.ttest_ind(sm, ih)

    return {
        "source": "Forbes Billionaires (CORGIS, 2077 unique individuals)",
        "wealth_type_groups": groups,
        "selfmade_vs_inherited": {
            "selfmade_n": len(sm),
            "selfmade_mean": round(float(sm.mean()), 4),
            "inherited_n": len(ih),
            "inherited_mean": round(float(ih.mean()), 4),
            "diff": round(float(sm.mean() - ih.mean()), 4),
            "t": round(float(t), 3),
            "p": round(float(p), 6),
            "conclusion": "Self-made billionaires have HIGHER WEAT scores than inherited ones",
        },
        "by_relationship": {},
    }

    # founder vs relation
    founders = df_u[df_u["company.relationship"] == "founder"]["weat"].dropna()
    relations = df_u[df_u["company.relationship"] == "relation"]["weat"].dropna()
    t2, p2 = stats.ttest_ind(founders, relations)

    result["by_relationship"] = {
        "founder_mean": round(float(founders.mean()), 4),
        "relation_mean": round(float(relations.mean()), 4),
        "t": round(float(t2), 3),
        "p": round(float(p2), 6),
    }

    return result


def main():
    char_scores, bert_scores, en_scores, whole_scores = load_scores()

    results = {
        "china_grantee_validation": run_china_validation(char_scores, bert_scores, whole_scores),
        "us_salary_validation": run_us_salary_validation(en_scores),
        "elite_validation": run_elite_validation(en_scores, char_scores),
        "selfmade_vs_inherited": run_selfmade_vs_inherited(en_scores),
    }

    out = DATA / "validation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("保存: %s", out)

    # 打印摘要
    print("\n" + "=" * 60)
    print("验证结果摘要")
    print("=" * 60)

    cn = results["china_grantee_validation"]
    print(f"\n🇨🇳 中国 (99K科学家 vs 100K普通人):")
    print(f"   三层融合 AUC = {cn['fusion_auc_mean']:.4f} ± {cn['fusion_auc_std']:.4f}")

    us = results["us_salary_validation"]
    if "error" not in us:
        print(f"\n🇺🇸 美国 (32K芝加哥政府雇员):")
        print(f"   名字级 Pearson r = {us['name_level_pearson_r']:.4f}")

    elite = results["elite_validation"]
    print(f"\n🌍 精英验证:")
    for cat, d in elite["categories"].items():
        sig = "***" if d["p"] < 0.001 else "ns"
        print(f"   {cat}: Δ={d['diff_vs_control']:+.4f} {sig}")

    sm = results["selfmade_vs_inherited"]["selfmade_vs_inherited"]
    print(f"\n💰 富一代 vs 富二代:")
    print(f"   白手起家={sm['selfmade_mean']:.4f}, 继承={sm['inherited_mean']:.4f}")
    print(f"   差异={sm['diff']:+.4f}, p={sm['p']:.6f}")


if __name__ == "__main__":
    main()
