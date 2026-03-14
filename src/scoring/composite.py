"""综合评分系统 — 将WEAT、LLM、频率等多维度得分归一化并加权合并"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.weat.calculator import NameWEATProfile
from src.llm.first_impression import LLMImpressionResult

logger = logging.getLogger(__name__)


@dataclass
class NameScore:
    """单个名字的综合评分。"""
    name: str
    dimension_scores: dict[str, float] = field(default_factory=dict)  # 原始得分
    normalized_scores: dict[str, float] = field(default_factory=dict)  # 归一化0-100
    total_score: float = 0.0
    rank: int = 0


def normalize_scores(values: list[float], target_min: float = 0, target_max: float = 100) -> list[float]:
    """Min-Max归一化到 [target_min, target_max]。"""
    if not values:
        return []
    arr = np.array(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return [50.0] * len(values)
    normalized = (arr - vmin) / (vmax - vmin) * (target_max - target_min) + target_min
    return normalized.tolist()


def frequency_optimality_score(
    frequency_percentile: float,
    optimal_low: float = 0.15,
    optimal_high: float = 0.60,
) -> float:
    """频率适中度得分。

    在 [optimal_low, optimal_high] 百分位区间内的名字得满分，
    过高或过低频率逐渐扣分。
    """
    if optimal_low <= frequency_percentile <= optimal_high:
        return 100.0

    if frequency_percentile < optimal_low:
        # 太罕见
        return max(0, 100 * frequency_percentile / optimal_low)
    else:
        # 太常见
        return max(0, 100 * (1 - frequency_percentile) / (1 - optimal_high))


def compute_composite_scores(
    weat_profiles: list[NameWEATProfile],
    llm_results: list[LLMImpressionResult] | None = None,
    frequency_data: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
) -> list[NameScore]:
    """计算所有候选名字的综合得分。

    Parameters
    ----------
    weat_profiles : WEAT分析结果
    llm_results : LLM第一印象结果 (可选)
    frequency_data : 频率数据DataFrame，含 name, frequency_percentile 列 (可选)
    weights : 各维度权重
    """
    if weights is None:
        from config.settings import SCORING_WEIGHTS
        weights = SCORING_WEIGHTS

    # 建立名字到各数据源的映射
    name_set = {p.name for p in weat_profiles}
    llm_map = {}
    if llm_results:
        llm_map = {r.name: r for r in llm_results}

    freq_map = {}
    if frequency_data is not None and not frequency_data.empty:
        for _, row in frequency_data.iterrows():
            freq_map[row["name"]] = row.get("frequency_percentile", 0.5)

    # 1. 收集各维度的原始分
    raw_scores: dict[str, dict[str, float]] = {}

    for profile in weat_profiles:
        name = profile.name
        raw_scores[name] = {}

        # WEAT维度
        for dim, result in profile.dimension_scores.items():
            raw_scores[name][dim] = result.score

        # LLM得分 (0-10 → 直接使用)
        if name in llm_map:
            raw_scores[name]["llm"] = llm_map[name].composite_llm_score
        else:
            raw_scores[name]["llm"] = 5.0  # 默认中位数

        # 频率适中度
        percentile = freq_map.get(name, 0.5)
        raw_scores[name]["frequency"] = frequency_optimality_score(percentile)

    # 2. 按维度归一化
    all_names = list(raw_scores.keys())
    all_dims = set()
    for scores in raw_scores.values():
        all_dims.update(scores.keys())

    normalized: dict[str, dict[str, float]] = {n: {} for n in all_names}

    for dim in all_dims:
        values = [raw_scores[n].get(dim, 0.0) for n in all_names]
        norm_values = normalize_scores(values)
        for name, nv in zip(all_names, norm_values):
            normalized[name][dim] = nv

    # 3. 加权求和
    results = []
    for name in all_names:
        total = 0.0
        total_weight = 0.0

        for dim, w in weights.items():
            if dim in normalized[name]:
                total += normalized[name][dim] * w
                total_weight += w

        if total_weight > 0:
            total /= total_weight
            total *= total_weight  # 保持实际加权和

        ns = NameScore(
            name=name,
            dimension_scores={d: raw_scores[name].get(d, 0.0) for d in all_dims},
            normalized_scores=normalized[name],
            total_score=total,
        )
        results.append(ns)

    # 排名
    results.sort(key=lambda x: x.total_score, reverse=True)
    for i, ns in enumerate(results):
        ns.rank = i + 1

    return results


def scores_to_dataframe(scores: list[NameScore]) -> pd.DataFrame:
    """将评分结果转换为DataFrame方便导出和可视化。"""
    rows = []
    for ns in scores:
        row = {
            "rank": ns.rank,
            "name": ns.name,
            "total_score": round(ns.total_score, 2),
        }
        for dim, val in sorted(ns.normalized_scores.items()):
            row[f"{dim}_norm"] = round(val, 2)
        for dim, val in sorted(ns.dimension_scores.items()):
            row[f"{dim}_raw"] = round(val, 4)
        rows.append(row)

    return pd.DataFrame(rows)
