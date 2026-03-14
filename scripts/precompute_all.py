#!/usr/bin/env python3
"""一键预计算所有数据：扩展字库WEAT + 名字频率 + SSA趋势。

输出:
  data/char_scores.json      — 3000+字的6维WEAT得分 (覆盖旧文件)
  data/char_neighbors.json   — 邻居词 (覆盖旧文件)
  data/name_freq_zh.json     — 中文名字频率数据
  data/ssa_trends.json       — SSA英文名历史趋势
"""

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import WEAT_ATTRIBUTES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIMS = list(WEAT_ATTRIBUTES.keys())
DATA_DIR = PROJECT_ROOT / "data"


def compute_weat_score(model, char, pos_words, neg_words):
    if char not in model:
        return None
    vec = model[char]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    pos = [float(np.dot(vec, model[w]) / (norm * np.linalg.norm(model[w])))
           for w in pos_words if w in model and np.linalg.norm(model[w]) > 0]
    neg = [float(np.dot(vec, model[w]) / (norm * np.linalg.norm(model[w])))
           for w in neg_words if w in model and np.linalg.norm(model[w]) > 0]
    if not pos or not neg:
        return None
    return float(np.mean(pos) - np.mean(neg))


def compute_neighbors(model, char, topn=10):
    if char not in model:
        return []
    try:
        neighbors = model.most_similar(char, topn=topn + 20)
        filtered = []
        for word, score in neighbors:
            if any('\u4e00' <= c <= '\u9fff' for c in word):
                filtered.append([word, round(float(score), 4)])
            if len(filtered) >= topn:
                break
        return filtered
    except Exception:
        return []


def phase1_char_scores():
    """Phase 1: 扩展字库WEAT得分 (3000+ chars)"""
    logger.info("=" * 60)
    logger.info("Phase 1: 扩展字库WEAT得分")
    logger.info("=" * 60)

    # 加载字符列表
    with open(DATA_DIR / "name_chars_top3000.json", encoding="utf-8") as f:
        data = json.load(f)
    chars = data["chars"]
    logger.info("目标字符数: %d", len(chars))

    # 加载模型
    logger.info("加载 text2vec 词向量...")
    from text2vec import Word2Vec
    w2v = Word2Vec("w2v-light-tencent-chinese")
    model = w2v.w2v
    logger.info("模型词汇量: %d", len(model))

    found = [c for c in chars if c in model]
    logger.info("覆盖: %d/%d (%.1f%%)", len(found), len(chars), 100 * len(found) / len(chars))

    # WEAT得分
    char_scores = {}
    for i, char in enumerate(found):
        if (i + 1) % 500 == 0:
            logger.info("  WEAT进度: %d/%d", i + 1, len(found))
        scores = {}
        for dim, attrs in WEAT_ATTRIBUTES.items():
            score = compute_weat_score(model, char, attrs["positive"], attrs["negative"])
            if score is not None:
                scores[dim] = round(score, 6)
        if scores:
            scores["composite"] = round(float(np.mean(list(scores.values()))), 6)
            char_scores[char] = scores

    logger.info("WEAT得分: %d 个字", len(char_scores))

    # 邻居词
    char_neighbors = {}
    for i, char in enumerate(found):
        if (i + 1) % 500 == 0:
            logger.info("  邻居进度: %d/%d", i + 1, len(found))
        neighbors = compute_neighbors(model, char)
        if neighbors:
            char_neighbors[char] = neighbors

    logger.info("邻居词: %d 个字", len(char_neighbors))

    # 保存
    with open(DATA_DIR / "char_scores.json", "w", encoding="utf-8") as f:
        json.dump(char_scores, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / "char_neighbors.json", "w", encoding="utf-8") as f:
        json.dump(char_neighbors, f, ensure_ascii=False, indent=2)

    return char_scores


def phase2_name_freq():
    """Phase 2: 中文名字频率数据"""
    logger.info("=" * 60)
    logger.info("Phase 2: 中文名字频率数据")
    logger.info("=" * 60)

    # CCNC
    ccnc = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "name2gender" / "dataset" / "ccnc.csv",
        sep="\t", names=["surname", "given", "fullname", "gender"], header=0,
    )
    train = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "name2gender" / "dataset" / "train.csv",
        header=None, names=["fullname", "gender"],
    )
    train["given"] = train["fullname"].apply(
        lambda x: x[1:] if isinstance(x, str) and len(x) >= 2 else None
    )

    # 合并名字频率
    name_counter = Counter()
    gender_counter = defaultdict(lambda: {"M": 0, "F": 0})

    for _, row in ccnc.iterrows():
        name = row.get("given")
        gender = row.get("gender")
        if isinstance(name, str) and len(name) >= 1:
            name_counter[name] += 1
            if gender in ("M", "男"):
                gender_counter[name]["M"] += 1
            elif gender in ("F", "女"):
                gender_counter[name]["F"] += 1

    for _, row in train.iterrows():
        name = row.get("given")
        gender = row.get("gender")
        if isinstance(name, str) and len(name) >= 1:
            name_counter[name] += 1
            if gender in ("男",):
                gender_counter[name]["M"] += 1
            elif gender in ("女",):
                gender_counter[name]["F"] += 1

    logger.info("不同名字数: %d", len(name_counter))

    # 取 Top 10000 构建频率表
    top_names = name_counter.most_common(10000)
    total = sum(name_counter.values())

    freq_data = {}
    cumulative = 0
    for rank, (name, count) in enumerate(top_names, 1):
        cumulative += count
        gc = gender_counter[name]
        total_gc = gc["M"] + gc["F"]
        freq_data[name] = {
            "count": count,
            "rank": rank,
            "percentile": round(cumulative / total * 100, 2),
            "female_ratio": round(gc["F"] / total_gc * 100, 1) if total_gc > 0 else 50.0,
        }

    with open(DATA_DIR / "name_freq_zh.json", "w", encoding="utf-8") as f:
        json.dump(freq_data, f, ensure_ascii=False)
    logger.info("保存: name_freq_zh.json (%d names)", len(freq_data))


def phase3_ssa_trends():
    """Phase 3: SSA英文名历史趋势"""
    logger.info("=" * 60)
    logger.info("Phase 3: SSA英文名历史趋势")
    logger.info("=" * 60)

    ssa_dir = PROJECT_ROOT / "data" / "raw" / "ssa_baby_names"

    # 按年度汇总
    yearly = defaultdict(lambda: defaultdict(int))  # name → {year: count}
    yearly_total = defaultdict(int)  # year → total

    for f in sorted(ssa_dir.glob("yob*.txt")):
        year = int(f.stem[3:])
        df = pd.read_csv(f, header=None, names=["name", "sex", "count"])
        for _, row in df.iterrows():
            yearly[row["name"]][year] += row["count"]
            yearly_total[year] += row["count"]

    logger.info("年份范围: %d-%d", min(yearly_total), max(yearly_total))
    logger.info("总名字数: %d", len(yearly))

    # 只保留有一定流行度的名字的趋势 (总使用>500)
    trends = {}
    for name, year_counts in yearly.items():
        total = sum(year_counts.values())
        if total < 500:
            continue

        # 每10年取一个数据点 + 最近5年每年一个
        points = {}
        for year in range(1880, 2020, 10):
            count = year_counts.get(year, 0)
            pct = count / yearly_total.get(year, 1) * 10000  # per 10K births
            points[str(year)] = round(pct, 2)

        for year in range(2020, max(yearly_total) + 1):
            count = year_counts.get(year, 0)
            pct = count / yearly_total.get(year, 1) * 10000
            points[str(year)] = round(pct, 2)

        # 找峰值年
        peak_year = max(year_counts, key=year_counts.get)
        peak_count = year_counts[peak_year]

        trends[name] = {
            "total": total,
            "peak_year": peak_year,
            "peak_count": peak_count,
            "trend": points,
        }

    with open(DATA_DIR / "ssa_trends.json", "w", encoding="utf-8") as f:
        json.dump(trends, f)
    logger.info("保存: ssa_trends.json (%d names)", len(trends))


def main():
    start = time.time()
    phase1_char_scores()
    phase2_name_freq()
    phase3_ssa_trends()
    logger.info("=" * 60)
    logger.info("全部完成，耗时 %.1f 秒", time.time() - start)


if __name__ == "__main__":
    main()
