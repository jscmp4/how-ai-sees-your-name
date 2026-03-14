"""WEAT (Word Embedding Association Test) 计算器

基于 Caliskan et al. (2017) "Semantics derived automatically from language
corpora contain human-like biases" 的方法。
"""

import logging
from dataclasses import dataclass

import numpy as np
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


@dataclass
class WEATResult:
    """单个维度的WEAT测试结果。"""
    dimension: str
    name: str
    score: float            # positive = 偏向正面属性
    pos_similarity: float   # 与正面属性词的平均cosine相似度
    neg_similarity: float   # 与负面属性词的平均cosine相似度
    coverage: float         # 有多少比例的属性词在模型中


@dataclass
class NameWEATProfile:
    """一个名字在所有维度上的WEAT得分汇总。"""
    name: str
    dimension_scores: dict[str, WEATResult]
    composite_score: float  # 加权综合得分


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量的cosine相似度。"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def weat_single_word(
    model: KeyedVectors,
    target: str,
    positive_attrs: list[str],
    negative_attrs: list[str],
) -> tuple[float, float, float, float]:
    """计算单个目标词相对于正面/负面属性词的WEAT关联得分。

    Returns
    -------
    (score, pos_sim, neg_sim, coverage)
    """
    if target not in model:
        return (0.0, 0.0, 0.0, 0.0)

    target_vec = model[target]
    total_attrs = len(positive_attrs) + len(negative_attrs)

    pos_sims = []
    for attr in positive_attrs:
        if attr in model:
            pos_sims.append(cosine_similarity(target_vec, model[attr]))

    neg_sims = []
    for attr in negative_attrs:
        if attr in model:
            neg_sims.append(cosine_similarity(target_vec, model[attr]))

    found = len(pos_sims) + len(neg_sims)
    coverage = found / total_attrs if total_attrs > 0 else 0.0

    pos_mean = float(np.mean(pos_sims)) if pos_sims else 0.0
    neg_mean = float(np.mean(neg_sims)) if neg_sims else 0.0

    return (pos_mean - neg_mean, pos_mean, neg_mean, coverage)


def weat_name(
    model: KeyedVectors,
    name: str,
    positive_attrs: list[str],
    negative_attrs: list[str],
    dimension: str = "",
) -> WEATResult:
    """计算一个名字（可能是多字）在一个维度上的WEAT得分。

    策略：
    1. 先尝试整个名字作为一个token
    2. 再逐字计算，取平均
    """
    # 尝试整词
    score, pos_sim, neg_sim, coverage = weat_single_word(
        model, name, positive_attrs, negative_attrs
    )

    if coverage > 0:
        return WEATResult(
            dimension=dimension, name=name,
            score=score, pos_similarity=pos_sim,
            neg_similarity=neg_sim, coverage=coverage,
        )

    # 逐字分析
    char_scores = []
    char_pos = []
    char_neg = []
    char_cov = []

    for char in name:
        s, p, n, c = weat_single_word(model, char, positive_attrs, negative_attrs)
        if c > 0:
            char_scores.append(s)
            char_pos.append(p)
            char_neg.append(n)
            char_cov.append(c)

    if not char_scores:
        logger.warning("名字 '%s' 中没有任何字符在词向量模型中", name)
        return WEATResult(
            dimension=dimension, name=name,
            score=0.0, pos_similarity=0.0,
            neg_similarity=0.0, coverage=0.0,
        )

    return WEATResult(
        dimension=dimension,
        name=name,
        score=float(np.mean(char_scores)),
        pos_similarity=float(np.mean(char_pos)),
        neg_similarity=float(np.mean(char_neg)),
        coverage=float(np.mean(char_cov)),
    )


def weat_profile(
    model: KeyedVectors,
    name: str,
    attributes: dict[str, dict[str, list[str]]],
    weights: dict[str, float] | None = None,
) -> NameWEATProfile:
    """计算一个名字在所有维度上的WEAT得分汇总。

    Parameters
    ----------
    model : 词向量模型
    name : 候选名字
    attributes : {dimension: {"positive": [...], "negative": [...]}}
    weights : {dimension: weight}，如为None则等权
    """
    dimension_scores = {}

    for dim, attrs in attributes.items():
        result = weat_name(
            model, name,
            attrs["positive"], attrs["negative"],
            dimension=dim,
        )
        dimension_scores[dim] = result

    # 计算加权综合得分
    if weights is None:
        weights = {dim: 1.0 / len(attributes) for dim in attributes}

    composite = 0.0
    total_weight = 0.0
    for dim, result in dimension_scores.items():
        w = weights.get(dim, 0.0)
        if result.coverage > 0:
            composite += result.score * w
            total_weight += w

    if total_weight > 0:
        composite /= total_weight

    return NameWEATProfile(
        name=name,
        dimension_scores=dimension_scores,
        composite_score=composite,
    )


def batch_weat(
    model: KeyedVectors,
    names: list[str],
    attributes: dict[str, dict[str, list[str]]],
    weights: dict[str, float] | None = None,
) -> list[NameWEATProfile]:
    """批量计算多个名字的WEAT得分，按综合得分降序排列。"""
    profiles = []
    for name in names:
        profile = weat_profile(model, name, attributes, weights)
        profiles.append(profile)

    profiles.sort(key=lambda p: p.composite_score, reverse=True)
    return profiles
