#!/usr/bin/env python3
"""运行WEAT实验 — 计算候选名字在各维度上的embedding偏好得分。

用法:
    python scripts/run_weat.py [--model tencent|text2vec] [--limit N] [--output results/weat_results.csv]
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import (
    TENCENT_EMBEDDING_PATH,
    WEAT_ATTRIBUTES,
    WEAT_ATTRIBUTES_EN,
    GLOVE_EMBEDDING_PATH,
    INITIAL_CHINESE_CANDIDATES,
    INITIAL_ENGLISH_CANDIDATES,
    RESULTS_DIR,
    SCORING_WEIGHTS,
)
from src.embeddings.chinese_vectors import load_tencent_word2vec, load_text2vec_tencent
from src.embeddings.english_vectors import load_glove, get_name_vector_en
from src.weat.calculator import batch_weat, weat_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_chinese_weat(model_type: str = "text2vec", limit: int | None = None):
    """运行中文WEAT实验。"""
    logger.info("=" * 60)
    logger.info("中文WEAT实验")
    logger.info("=" * 60)

    # 加载模型
    if model_type == "tencent":
        model = load_tencent_word2vec(TENCENT_EMBEDDING_PATH, limit=limit)
    else:
        model = load_text2vec_tencent()

    # WEAT权重（只取WEAT相关维度）
    weat_dims = {"wealth", "wisdom", "happiness", "health", "leadership", "beauty"}
    weights = {k: v for k, v in SCORING_WEIGHTS.items() if k in weat_dims}

    # 批量计算
    candidates = INITIAL_CHINESE_CANDIDATES
    logger.info("候选名字数: %d", len(candidates))

    profiles = batch_weat(model, candidates, WEAT_ATTRIBUTES, weights)

    # 输出结果
    rows = []
    for p in profiles:
        row = {"name": p.name, "composite": round(p.composite_score, 6)}
        for dim, result in p.dimension_scores.items():
            row[f"{dim}_score"] = round(result.score, 6)
            row[f"{dim}_pos"] = round(result.pos_similarity, 6)
            row[f"{dim}_neg"] = round(result.neg_similarity, 6)
            row[f"{dim}_cov"] = round(result.coverage, 4)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    return df


def run_english_weat(limit: int | None = None):
    """运行英文WEAT实验。"""
    logger.info("=" * 60)
    logger.info("英文WEAT实验")
    logger.info("=" * 60)

    model = load_glove(GLOVE_EMBEDDING_PATH, limit=limit)

    weat_dims = {"wealth", "wisdom", "happiness", "health", "leadership", "beauty"}
    weights = {k: v for k, v in SCORING_WEIGHTS.items() if k in weat_dims}

    candidates = INITIAL_ENGLISH_CANDIDATES
    logger.info("候选英文名数: %d", len(candidates))

    # 英文名需要转小写来匹配GloVe
    # 先检查哪些名字在模型中
    found = [n for n in candidates if any(v in model for v in [n, n.lower(), n.capitalize()])]
    logger.info("在GloVe中找到的名字: %d/%d", len(found), len(candidates))

    profiles = batch_weat(model, [n.lower() for n in candidates], WEAT_ATTRIBUTES_EN, weights)

    # 恢复原始大小写
    name_map = {n.lower(): n for n in candidates}
    rows = []
    for p in profiles:
        original_name = name_map.get(p.name, p.name)
        row = {"name": original_name, "composite": round(p.composite_score, 6)}
        for dim, result in p.dimension_scores.items():
            row[f"{dim}_score"] = round(result.score, 6)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    return df


def main():
    parser = argparse.ArgumentParser(description="运行WEAT偏好实验")
    parser.add_argument("--model", default="text2vec", choices=["tencent", "text2vec"],
                        help="中文词向量模型")
    parser.add_argument("--limit", type=int, default=None, help="词向量加载限制")
    parser.add_argument("--skip-english", action="store_true", help="跳过英文WEAT")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 中文WEAT
    zh_df = run_chinese_weat(model_type=args.model, limit=args.limit)
    zh_path = output_dir / "weat_chinese_results.csv"
    zh_df.to_csv(zh_path, encoding="utf-8-sig")
    logger.info("中文WEAT结果已保存: %s", zh_path)

    print("\n" + "=" * 60)
    print("中文名字WEAT综合排名 Top 20:")
    print("=" * 60)
    print(zh_df.head(20).to_string())

    # 英文WEAT
    if not args.skip_english:
        try:
            en_df = run_english_weat(limit=args.limit)
            en_path = output_dir / "weat_english_results.csv"
            en_df.to_csv(en_path, encoding="utf-8-sig")
            logger.info("英文WEAT结果已保存: %s", en_path)

            print("\n" + "=" * 60)
            print("英文名字WEAT综合排名 Top 20:")
            print("=" * 60)
            print(en_df.head(20).to_string())
        except FileNotFoundError as e:
            logger.warning("跳过英文WEAT: %s", e)


if __name__ == "__main__":
    main()
