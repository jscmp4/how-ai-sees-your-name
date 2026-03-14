#!/usr/bin/env python3
"""运行LLM第一印象测试 — 用API测量AI对名字的隐性联想。

用法:
    python scripts/run_llm_test.py [--api anthropic|openai] [--repeat 10] [--top N]

注意: 需要设置环境变量 ANTHROPIC_API_KEY 或 OPENAI_API_KEY
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import (
    INITIAL_CHINESE_CANDIDATES,
    INITIAL_ENGLISH_CANDIDATES,
    LLM_REPEAT_COUNT,
    LLM_TEMPERATURE,
    RESULTS_DIR,
)
from src.llm.first_impression import batch_test_impressions, test_name_impression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def results_to_dataframe(results) -> pd.DataFrame:
    """将LLM测试结果转为DataFrame。"""
    rows = []
    for r in results:
        rows.append({
            "name": r.name,
            "economic_avg": round(r.avg_economic, 2),
            "happiness_avg": round(r.avg_happiness, 2),
            "social_avg": round(r.avg_social, 2),
            "composite_llm": round(r.composite_llm_score, 2),
            "n_responses": len(r.raw_responses),
            "top_impressions": "; ".join(
                [", ".join(imp[:3]) for imp in r.impressions[:3]]
            ),
            "top_occupations": "; ".join(
                [", ".join(occ[:2]) for occ in r.occupations[:3]]
            ),
        })
    return pd.DataFrame(rows).sort_values("composite_llm", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="运行LLM第一印象测试")
    parser.add_argument("--api", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--repeat", type=int, default=LLM_REPEAT_COUNT)
    parser.add_argument("--top", type=int, default=None,
                        help="只测试WEAT排名前N的名字 (需先运行run_weat.py)")
    parser.add_argument("--skip-english", action="store_true")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定候选名字
    zh_candidates = INITIAL_CHINESE_CANDIDATES
    en_candidates = INITIAL_ENGLISH_CANDIDATES

    if args.top:
        weat_csv = output_dir / "weat_chinese_results.csv"
        if weat_csv.exists():
            weat_df = pd.read_csv(weat_csv)
            zh_candidates = weat_df.head(args.top)["name"].tolist()
            logger.info("使用WEAT前 %d 名中文名字", len(zh_candidates))

        weat_en_csv = output_dir / "weat_english_results.csv"
        if weat_en_csv.exists():
            weat_en_df = pd.read_csv(weat_en_csv)
            en_candidates = weat_en_df.head(args.top)["name"].tolist()

    # 中文名字测试
    logger.info("=" * 60)
    logger.info("中文名字LLM第一印象测试 (%d个名字, 每个%d次)", len(zh_candidates), args.repeat)
    logger.info("=" * 60)

    zh_results = batch_test_impressions(
        zh_candidates, repeat=args.repeat, api=args.api,
        language="zh", temperature=LLM_TEMPERATURE,
    )
    zh_df = results_to_dataframe(zh_results)
    zh_path = output_dir / "llm_chinese_results.csv"
    zh_df.to_csv(zh_path, index=False, encoding="utf-8-sig")
    logger.info("中文LLM结果已保存: %s", zh_path)

    print("\n" + "=" * 60)
    print("中文名字LLM第一印象排名 Top 20:")
    print("=" * 60)
    print(zh_df.head(20).to_string(index=False))

    # 保存详细JSON
    zh_detail = []
    for r in zh_results:
        zh_detail.append({
            "name": r.name,
            "economic_scores": r.economic_scores,
            "happiness_scores": r.happiness_scores,
            "social_scores": r.social_scores,
            "impressions": r.impressions,
            "occupations": r.occupations,
        })
    with open(output_dir / "llm_chinese_detail.json", "w", encoding="utf-8") as f:
        json.dump(zh_detail, f, ensure_ascii=False, indent=2)

    # 英文名字测试
    if not args.skip_english:
        logger.info("=" * 60)
        logger.info("英文名字LLM第一印象测试 (%d个名字)", len(en_candidates))
        logger.info("=" * 60)

        en_results = batch_test_impressions(
            en_candidates, repeat=args.repeat, api=args.api,
            language="en", temperature=LLM_TEMPERATURE,
        )
        en_df = results_to_dataframe(en_results)
        en_path = output_dir / "llm_english_results.csv"
        en_df.to_csv(en_path, index=False, encoding="utf-8-sig")
        logger.info("英文LLM结果已保存: %s", en_path)

        print("\n" + "=" * 60)
        print("英文名字LLM第一印象排名 Top 20:")
        print("=" * 60)
        print(en_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
