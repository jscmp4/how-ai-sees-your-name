#!/usr/bin/env python3
"""预计算英文名WEAT得分。

输出:
  data/en_name_scores.json — {name: {wealth: 0.12, ...}, ...}
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import WEAT_ATTRIBUTES_EN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIMS = list(WEAT_ATTRIBUTES_EN.keys())


def main():
    start = time.time()

    # 加载 GloVe
    glove_path = PROJECT_ROOT / "data" / "raw" / "glove.6B.300d.txt"
    logger.info("加载 GloVe...")
    model = KeyedVectors.load_word2vec_format(str(glove_path), binary=False, no_header=True)
    logger.info("加载完成: %d words", len(model))

    # 收集所有SSA名字
    ssa_dir = PROJECT_ROOT / "data" / "raw" / "ssa_baby_names"
    all_names = {}
    for f in sorted(ssa_dir.glob("yob*.txt")):
        import pandas as pd
        df = pd.read_csv(f, header=None, names=["name", "sex", "count"])
        for _, row in df.iterrows():
            n = row["name"]
            all_names[n] = all_names.get(n, 0) + row["count"]

    logger.info("SSA 总名字数: %d", len(all_names))

    # 计算每个在GloVe中的名字的得分
    results = {}
    count = 0
    for name, freq in all_names.items():
        lower = name.lower()
        if lower not in model:
            continue

        vec = model[lower]
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue

        scores = {}
        for dim in DIMS:
            pos_sims = []
            for w in WEAT_ATTRIBUTES_EN[dim]["positive"]:
                if w in model:
                    w_norm = np.linalg.norm(model[w])
                    if w_norm > 0:
                        pos_sims.append(float(np.dot(vec, model[w]) / (norm * w_norm)))
            neg_sims = []
            for w in WEAT_ATTRIBUTES_EN[dim]["negative"]:
                if w in model:
                    w_norm = np.linalg.norm(model[w])
                    if w_norm > 0:
                        neg_sims.append(float(np.dot(vec, model[w]) / (norm * w_norm)))
            if pos_sims and neg_sims:
                scores[dim] = round(float(np.mean(pos_sims) - np.mean(neg_sims)), 6)

        if scores:
            scores["composite"] = round(float(np.mean(list(scores.values()))), 6)
            scores["frequency"] = freq
            results[name] = scores
            count += 1

        if count % 2000 == 0 and count > 0:
            logger.info("进度: %d names", count)

    logger.info("计算完成: %d 个英文名有得分", len(results))

    # 保存
    out_path = PROJECT_ROOT / "data" / "en_name_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("保存到 %s", out_path)

    # Top 20
    ranked = sorted(results.items(), key=lambda x: x[1].get("composite", 0), reverse=True)
    print("\nTop 20 英文名:")
    for name, s in ranked[:20]:
        print(f"  {name:15s} comp={s['composite']:.4f}  freq={s.get('frequency',0):>8d}")

    logger.info("耗时 %.1f 秒", time.time() - start)


if __name__ == "__main__":
    main()
