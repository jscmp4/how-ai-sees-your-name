#!/usr/bin/env python3
"""完整Pipeline — 依次运行WEAT、LLM测试、综合评分、可视化。

用法:
    python scripts/run_full_pipeline.py [--skip-llm] [--skip-english] [--model text2vec]

这是项目的主入口脚本，一键完成所有分析。
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import (
    INITIAL_CHINESE_CANDIDATES,
    INITIAL_ENGLISH_CANDIDATES,
    RESULTS_DIR,
    SCORING_WEIGHTS,
    TENCENT_EMBEDDING_PATH,
    GLOVE_EMBEDDING_PATH,
    WEAT_ATTRIBUTES,
    WEAT_ATTRIBUTES_EN,
    LLM_REPEAT_COUNT,
    LLM_TEMPERATURE,
)
from src.weat.calculator import batch_weat
from src.scoring.composite import compute_composite_scores, scores_to_dataframe
from src.visualization.plots import (
    radar_chart,
    multi_radar_chart,
    heatmap,
    ranking_bar_chart,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="How AI Sees Your Name — 完整Pipeline")
    parser.add_argument("--model", default="text2vec", choices=["tencent", "text2vec"])
    parser.add_argument("--skip-llm", action="store_true", help="跳过LLM测试 (节省API费用)")
    parser.add_argument("--skip-english", action="store_true", help="跳过英文分析")
    parser.add_argument("--llm-api", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--llm-repeat", type=int, default=LLM_REPEAT_COUNT)
    parser.add_argument("--llm-top", type=int, default=20,
                        help="只对WEAT排名前N的名字做LLM测试")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Phase 1: 加载词向量
    # ============================================================
    logger.info("=" * 60)
    logger.info("Phase 1: 加载词向量模型")
    logger.info("=" * 60)

    if args.model == "tencent":
        from src.embeddings.chinese_vectors import load_tencent_word2vec
        zh_model = load_tencent_word2vec(TENCENT_EMBEDDING_PATH)
    else:
        from src.embeddings.chinese_vectors import load_text2vec_tencent
        zh_model = load_text2vec_tencent()

    # ============================================================
    # Phase 2: 中文WEAT分析
    # ============================================================
    logger.info("=" * 60)
    logger.info("Phase 2: 中文WEAT分析")
    logger.info("=" * 60)

    zh_candidates = INITIAL_CHINESE_CANDIDATES
    weat_dims = {"wealth", "wisdom", "happiness", "health", "leadership", "beauty"}
    weat_weights = {k: v for k, v in SCORING_WEIGHTS.items() if k in weat_dims}

    zh_profiles = batch_weat(zh_model, zh_candidates, WEAT_ATTRIBUTES, weat_weights)

    logger.info("WEAT分析完成，Top 10:")
    for i, p in enumerate(zh_profiles[:10], 1):
        dims_str = " | ".join(f"{d}={r.score:.4f}" for d, r in p.dimension_scores.items())
        logger.info("  #%d %s (%.4f) — %s", i, p.name, p.composite_score, dims_str)

    # ============================================================
    # Phase 3: LLM第一印象测试 (可选)
    # ============================================================
    llm_results = None
    if not args.skip_llm:
        logger.info("=" * 60)
        logger.info("Phase 3: LLM第一印象测试 (Top %d)", args.llm_top)
        logger.info("=" * 60)

        from src.llm.first_impression import batch_test_impressions

        top_names = [p.name for p in zh_profiles[:args.llm_top]]
        llm_results = batch_test_impressions(
            top_names, repeat=args.llm_repeat,
            api=args.llm_api, language="zh",
            temperature=LLM_TEMPERATURE,
        )
    else:
        logger.info("跳过LLM测试")

    # ============================================================
    # Phase 4: 综合评分
    # ============================================================
    logger.info("=" * 60)
    logger.info("Phase 4: 综合评分")
    logger.info("=" * 60)

    composite_scores = compute_composite_scores(
        weat_profiles=zh_profiles,
        llm_results=llm_results,
        weights=SCORING_WEIGHTS,
    )

    df = scores_to_dataframe(composite_scores)
    csv_path = output_dir / "final_chinese_ranking.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("综合排名已保存: %s", csv_path)

    print("\n" + "=" * 60)
    print("中文名字最终综合排名 Top 20")
    print("=" * 60)
    print(df.head(20).to_string(index=False))

    # ============================================================
    # Phase 5: 可视化
    # ============================================================
    logger.info("=" * 60)
    logger.info("Phase 5: 可视化")
    logger.info("=" * 60)

    # Top 5 雷达图对比
    top5_scores = {}
    for ns in composite_scores[:5]:
        # 只取WEAT维度 + llm + frequency
        top5_scores[ns.name] = {
            k: v for k, v in ns.normalized_scores.items()
            if k in weat_dims | {"llm", "frequency"}
        }

    multi_radar_chart(
        top5_scores,
        output_path=output_dir / "top5_radar.png",
        title="Top 5 中文名字对比雷达图",
    )

    # 单独雷达图（Top 1）
    if composite_scores:
        best = composite_scores[0]
        radar_chart(
            best.name,
            {k: v for k, v in best.normalized_scores.items()
             if k in weat_dims | {"llm", "frequency"}},
            output_path=output_dir / f"radar_{best.name}.png",
        )

    # 热力图
    heatmap(df, output_path=output_dir / "heatmap_top30.png", top_n=30)

    # 排名条形图
    ranking_bar_chart(df, output_path=output_dir / "ranking_top20.png", top_n=20)

    # ============================================================
    # Phase 6 (可选): 英文分析
    # ============================================================
    if not args.skip_english:
        try:
            logger.info("=" * 60)
            logger.info("Phase 6: 英文名字分析")
            logger.info("=" * 60)

            from src.embeddings.english_vectors import load_glove
            en_model = load_glove(GLOVE_EMBEDDING_PATH)

            en_profiles = batch_weat(
                en_model,
                [n.lower() for n in INITIAL_ENGLISH_CANDIDATES],
                WEAT_ATTRIBUTES_EN,
                weat_weights,
            )

            name_map = {n.lower(): n for n in INITIAL_ENGLISH_CANDIDATES}
            en_rows = []
            for p in en_profiles:
                original = name_map.get(p.name, p.name)
                row = {"name": original, "composite": round(p.composite_score, 6)}
                for dim, result in p.dimension_scores.items():
                    row[f"{dim}_score"] = round(result.score, 6)
                en_rows.append(row)

            en_df = pd.DataFrame(en_rows)
            en_path = output_dir / "final_english_ranking.csv"
            en_df.to_csv(en_path, index=False)
            logger.info("英文排名已保存: %s", en_path)

            print("\n" + "=" * 60)
            print("英文名字WEAT排名 Top 20")
            print("=" * 60)
            print(en_df.head(20).to_string(index=False))

        except FileNotFoundError as e:
            logger.warning("跳过英文分析: %s", e)

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"结果目录: {output_dir}")
    print(f"生成文件:")
    for f in sorted(output_dir.glob("*")):
        if f.name != ".gitkeep":
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
