#!/usr/bin/env python3
"""预计算"名字-成就"关联分析数据。

对比基金获资助者（成功学者）与普通人名字在WEAT各维度上的差异。
输出:
  data/success_analysis.json — 统计分析结果
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIMS = ["wealth", "wisdom", "happiness", "health", "leadership", "beauty"]


def main():
    # 加载预计算字得分
    with open(PROJECT_ROOT / "data" / "char_scores.json", encoding="utf-8") as f:
        char_scores = json.load(f)

    # 加载成功者数据（获基金资助的科学家/学者）
    grantees = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "Chinese-Gender-dataset" /
        "ChineseGender" / "Result_data" / "grantees.csv",
        sep="\t"
    )
    grantee_names = grantees["given"].dropna().tolist()
    logger.info("成功者名字: %d", len(grantee_names))

    # 加载普通人名字
    ccnc = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "name2gender" / "dataset" / "train.csv",
        header=None, names=["fullname", "gender"]
    )
    ccnc["given"] = ccnc["fullname"].apply(
        lambda x: x[1:] if isinstance(x, str) and len(x) >= 2 else None
    )
    general_names = ccnc["given"].dropna().tolist()
    logger.info("普通人名字: %d", len(general_names))

    random.seed(42)
    general_sample = random.sample(general_names, min(100000, len(general_names)))

    # 计算函数
    def name_dim_score(name, dim):
        if not isinstance(name, str):
            return None
        vals = [char_scores[ch][dim] for ch in name
                if ch in char_scores and dim in char_scores[ch]]
        return float(np.mean(vals)) if vals else None

    def name_composite(name):
        vals = [name_dim_score(name, d) for d in DIMS]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    # 分维度计算
    results = {"dimensions": {}, "composite": {}, "meta": {}}

    for dim in DIMS:
        logger.info("计算维度: %s", dim)
        g_scores = [s for s in (name_dim_score(n, dim) for n in grantee_names) if s is not None]
        p_scores = [s for s in (name_dim_score(n, dim) for n in general_sample) if s is not None]

        t_stat, p_val = stats.ttest_ind(g_scores, p_scores)
        cohens_d = (np.mean(g_scores) - np.mean(p_scores)) / np.sqrt(
            (np.std(g_scores)**2 + np.std(p_scores)**2) / 2)

        results["dimensions"][dim] = {
            "grantee_mean": round(float(np.mean(g_scores)), 6),
            "grantee_std": round(float(np.std(g_scores)), 6),
            "general_mean": round(float(np.mean(p_scores)), 6),
            "general_std": round(float(np.std(p_scores)), 6),
            "diff": round(float(np.mean(g_scores) - np.mean(p_scores)), 6),
            "t_stat": round(float(t_stat), 4),
            "p_value": float(p_val),
            "cohens_d": round(float(cohens_d), 4),
            "n_grantee": len(g_scores),
            "n_general": len(p_scores),
        }

    # 综合得分
    g_comp = [s for s in (name_composite(n) for n in grantee_names) if s is not None]
    p_comp = [s for s in (name_composite(n) for n in general_sample) if s is not None]
    t_stat, p_val = stats.ttest_ind(g_comp, p_comp)
    cohens_d = (np.mean(g_comp) - np.mean(p_comp)) / np.sqrt(
        (np.std(g_comp)**2 + np.std(p_comp)**2) / 2)

    results["composite"] = {
        "grantee_mean": round(float(np.mean(g_comp)), 6),
        "general_mean": round(float(np.mean(p_comp)), 6),
        "diff": round(float(np.mean(g_comp) - np.mean(p_comp)), 6),
        "t_stat": round(float(t_stat), 4),
        "p_value": float(p_val),
        "cohens_d": round(float(cohens_d), 4),
        "n_grantee": len(g_comp),
        "n_general": len(p_comp),
    }

    # 成功者中高频用字的WEAT得分
    char_freq_grantee = {}
    for name in grantee_names:
        if isinstance(name, str):
            for ch in name:
                char_freq_grantee[ch] = char_freq_grantee.get(ch, 0) + 1

    top_grantee_chars = sorted(char_freq_grantee.items(), key=lambda x: x[1], reverse=True)[:50]
    results["top_grantee_chars"] = []
    for ch, freq in top_grantee_chars:
        entry = {"char": ch, "freq": freq}
        if ch in char_scores:
            entry["scores"] = char_scores[ch]
        results["top_grantee_chars"].append(entry)

    results["meta"] = {
        "grantee_source": "Chinese Gender Dataset - grantees.csv (基金获资助者/科学家)",
        "general_source": "CCNC dataset - train.csv (普通人名字)",
        "n_grantee_total": len(grantee_names),
        "n_general_total": len(general_sample),
    }

    # 保存
    out_path = PROJECT_ROOT / "data" / "success_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("保存到 %s", out_path)

    # 打印摘要
    print("\n" + "=" * 60)
    print("名字-成就关联分析结果")
    print("=" * 60)
    print(f"综合: 成功者={results['composite']['grantee_mean']:.6f}  "
          f"普通人={results['composite']['general_mean']:.6f}  "
          f"t={results['composite']['t_stat']:.2f}  "
          f"p={results['composite']['p_value']:.2e}  "
          f"d={results['composite']['cohens_d']:.4f}")
    print()
    for dim in DIMS:
        d = results["dimensions"][dim]
        sig = "***" if d["p_value"] < 0.001 else "**" if d["p_value"] < 0.01 else "*" if d["p_value"] < 0.05 else "ns"
        print(f"  {dim:12s}: Δ={d['diff']:+.6f}  {sig}")


if __name__ == "__main__":
    main()
