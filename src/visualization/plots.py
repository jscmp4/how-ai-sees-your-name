"""可视化模块 — 雷达图、热力图、排名图"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试设置中文字体
_CN_FONTS = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]

def _setup_chinese_font():
    """配置matplotlib中文字体支持。"""
    for font_name in _CN_FONTS:
        fonts = fm.findSystemFonts()
        for f in fonts:
            try:
                prop = fm.FontProperties(fname=f)
                if font_name.lower() in prop.get_name().lower():
                    plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams.get("font.sans-serif", [])
                    plt.rcParams["axes.unicode_minus"] = False
                    logger.info("使用中文字体: %s", font_name)
                    return
            except Exception:
                continue

    logger.warning("未找到合适的中文字体，图表中的中文可能显示为方块")

_setup_chinese_font()

# ── 维度名称映射 ──
DIM_LABELS = {
    "wealth": "财富",
    "wisdom": "智慧",
    "happiness": "幸福",
    "health": "健康",
    "leadership": "领导力",
    "beauty": "美感",
    "llm": "LLM印象",
    "frequency": "频率适中",
}


def radar_chart(
    name: str,
    scores: dict[str, float],
    output_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """为单个名字绘制雷达图。

    Parameters
    ----------
    name : 名字
    scores : {dimension: normalized_score (0-100)}
    output_path : 保存路径，None则不保存
    """
    dims = list(scores.keys())
    labels = [DIM_LABELS.get(d, d) for d in dims]
    values = [scores[d] for d in dims]

    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color="#2196F3")
    ax.plot(angles, values, "o-", linewidth=2, color="#1565C0")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(title or f"「{name}」综合评分雷达图", fontsize=16, pad=20)

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("雷达图已保存: %s", output_path)

    return fig


def multi_radar_chart(
    names_scores: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    title: str = "Top名字对比雷达图",
    max_names: int = 5,
) -> plt.Figure:
    """多个名字的雷达图叠加对比。"""
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
              "#00BCD4", "#795548", "#607D8B"]

    items = list(names_scores.items())[:max_names]
    if not items:
        return plt.figure()

    dims = list(items[0][1].keys())
    labels = [DIM_LABELS.get(d, d) for d in dims]
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (name, scores) in enumerate(items):
        values = [scores[d] for d in dims] + [scores[dims[0]]]
        color = colors[i % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("对比雷达图已保存: %s", output_path)

    return fig


def heatmap(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "名字维度得分热力图",
    top_n: int = 30,
) -> plt.Figure:
    """绘制名字×维度的热力图。

    Parameters
    ----------
    df : scores_to_dataframe() 输出的DataFrame
    """
    import seaborn as sns

    # 筛选归一化列
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    if not norm_cols:
        logger.warning("DataFrame中没有找到 _norm 列")
        return plt.figure()

    plot_df = df.head(top_n).set_index("name")[norm_cols]
    plot_df.columns = [DIM_LABELS.get(c.replace("_norm", ""), c) for c in norm_cols]

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    sns.heatmap(
        plot_df, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=0, vmax=100, ax=ax, linewidths=0.5,
    )
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("")
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("热力图已保存: %s", output_path)

    return fig


def ranking_bar_chart(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "名字综合评分排名",
    top_n: int = 20,
) -> plt.Figure:
    """绘制综合得分水平条形图。"""
    plot_df = df.head(top_n).sort_values("total_score")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(plot_df["name"], plot_df["total_score"], color="#2196F3", edgecolor="#1565C0")
    ax.set_xlabel("综合得分", fontsize=13)
    ax.set_title(title, fontsize=16)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}", ha="left", va="center", fontsize=10)

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("排名图已保存: %s", output_path)

    return fig
