"""英文词向量加载器 — 支持GloVe和NamePrism"""

import logging
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


def load_glove(path: str | Path, limit: int | None = None) -> KeyedVectors:
    """加载GloVe英文词向量。

    GloVe原始格式不含header行，需要用 no_header=True。

    Parameters
    ----------
    path : glove文件路径 (如 glove.6B.300d.txt)
    limit : 只加载前N个词
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"GloVe文件未找到: {path}\n"
            "请先下载: https://nlp.stanford.edu/data/glove.6B.zip"
        )

    logger.info("正在加载GloVe词向量: %s ...", path)
    model = KeyedVectors.load_word2vec_format(
        str(path), binary=False, no_header=True, limit=limit
    )
    logger.info("加载完成，词汇量: %d，维度: %d", len(model), model.vector_size)
    return model


def get_name_vector_en(model: KeyedVectors, name: str) -> np.ndarray | None:
    """获取英文名字的向量。

    尝试顺序：原始 → 小写 → 首字母大写
    """
    for variant in [name, name.lower(), name.capitalize()]:
        if variant in model:
            return model[variant]
    return None
