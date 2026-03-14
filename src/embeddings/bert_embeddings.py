"""BERT/Transformer上下文相关embedding — 用于更精细的语义相似度计算"""

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class BertEmbedder:
    """使用中文BERT模型计算句子/词的上下文embedding。"""

    def __init__(self, model_name: str = "hfl/chinese-macbert-base"):
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("请安装 transformers 和 torch: pip install transformers torch")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("加载BERT模型: %s (device=%s)", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._torch = torch

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """将文本列表编码为向量 (CLS token pooling)。

        Returns
        -------
        np.ndarray of shape (len(texts), hidden_size)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt"
            ).to(self.device)

            with self._torch.no_grad():
                outputs = self.model(**encoded)

            # CLS token
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)

    def similarity(self, text_a: str, text_b: str) -> float:
        """计算两个文本的cosine相似度。"""
        vecs = self.encode([text_a, text_b])
        cos = np.dot(vecs[0], vecs[1]) / (
            np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
        )
        return float(cos)

    def name_attribute_similarity(
        self, name: str, attributes: list[str]
    ) -> float:
        """计算名字与属性词列表的平均cosine相似度。

        通过将名字和属性词放入短句模板来获取上下文embedding。
        """
        name_text = f"她叫{name}"
        attr_texts = [f"她很{a}" for a in attributes]

        name_vec = self.encode([name_text])[0]
        attr_vecs = self.encode(attr_texts)

        similarities = []
        for attr_vec in attr_vecs:
            cos = np.dot(name_vec, attr_vec) / (
                np.linalg.norm(name_vec) * np.linalg.norm(attr_vec)
            )
            similarities.append(cos)

        return float(np.mean(similarities))
