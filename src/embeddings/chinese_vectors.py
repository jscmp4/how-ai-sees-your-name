"""中文词向量加载器 — 支持腾讯AI Lab词向量和北师大词向量"""

import logging
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


def load_tencent_word2vec(path: str | Path, limit: int | None = None) -> KeyedVectors:
    """加载腾讯AI Lab中文词向量 (word2vec txt格式)。

    Parameters
    ----------
    path : 词向量文件路径 (.txt)
    limit : 只加载前N个词，None=全量

    Returns
    -------
    gensim.models.KeyedVectors
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"词向量文件未找到: {path}\n"
            "请先运行 scripts/download_data.py 或手动下载腾讯词向量。"
        )

    logger.info("正在加载腾讯词向量: %s (limit=%s) ...", path, limit)
    model = KeyedVectors.load_word2vec_format(str(path), binary=False, limit=limit)
    logger.info("加载完成，词汇量: %d，维度: %d", len(model), model.vector_size)
    return model


def load_text2vec_tencent() -> KeyedVectors:
    """通过 text2vec 包加载轻量版腾讯词向量。

    需要先 pip install text2vec
    """
    try:
        from text2vec import Word2Vec

        logger.info("正在通过 text2vec 加载轻量版腾讯词向量...")
        model = Word2Vec("w2v-light-tencent-chinese")
        return model.w2v
    except ImportError:
        raise ImportError("请先安装 text2vec: pip install text2vec")


def load_bnu_word2vec(path: str | Path, limit: int | None = None) -> KeyedVectors:
    """加载北师大中文词向量 (word2vec txt格式)。

    Parameters
    ----------
    path : 词向量文件路径
    limit : 只加载前N个词
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"北师大词向量文件未找到: {path}")

    logger.info("正在加载北师大词向量: %s ...", path)
    model = KeyedVectors.load_word2vec_format(str(path), binary=False, limit=limit)
    logger.info("加载完成，词汇量: %d，维度: %d", len(model), model.vector_size)
    return model


def get_char_vectors(model: KeyedVectors, name: str) -> list[tuple[str, np.ndarray]]:
    """提取名字中每个字符的向量。

    Returns
    -------
    [(char, vector), ...] — 只包含模型中存在的字符
    """
    results = []
    for char in name:
        if char in model:
            results.append((char, model[char]))
        else:
            logger.debug("字符 '%s' 不在词向量模型中", char)
    return results


def get_name_vector(model: KeyedVectors, name: str) -> np.ndarray | None:
    """获取名字的向量表示。

    策略：
    1. 如果整个名字在模型中，直接返回
    2. 否则取各字符向量的平均
    3. 如果没有任何字符在模型中，返回 None
    """
    if name in model:
        return model[name]

    char_vecs = get_char_vectors(model, name)
    if not char_vecs:
        return None

    return np.mean([v for _, v in char_vecs], axis=0)
