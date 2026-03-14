#!/usr/bin/env python3
"""用中文BERT计算字的上下文语义得分（第二套评分体系）。

使用 hfl/chinese-macbert-base，通过句子模板获取上下文embedding。
输出:
  data/char_scores_bert.json — {char: {wealth: 0.12, ...}, ...}
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import WEAT_ATTRIBUTES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIMS = list(WEAT_ATTRIBUTES.keys())
DATA_DIR = PROJECT_ROOT / "data"
MODEL_NAME = "hfl/chinese-macbert-base"


def encode_texts(model, tokenizer, texts, device, batch_size=64):
    """批量编码文本为CLS向量。"""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=32, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_vecs.append(cls)
    return np.vstack(all_vecs)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # 加载模型
    logger.info("加载 %s ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    logger.info("模型加载完成")

    # 加载字符列表
    with open(DATA_DIR / "name_chars_top3000.json", encoding="utf-8") as f:
        chars = json.load(f)["chars"]
    logger.info("字符数: %d", len(chars))

    # 预编码属性词（用句子模板）
    logger.info("编码属性词...")
    attr_vecs = {}
    for dim, attrs in WEAT_ATTRIBUTES.items():
        pos_texts = [f"她很{w}" for w in attrs["positive"]]
        neg_texts = [f"她很{w}" for w in attrs["negative"]]
        pos_vecs = encode_texts(model, tokenizer, pos_texts, device)
        neg_vecs = encode_texts(model, tokenizer, neg_texts, device)
        attr_vecs[dim] = {
            "pos_mean": pos_vecs.mean(axis=0),
            "neg_mean": neg_vecs.mean(axis=0),
        }

    # 批量编码字符（用名字模板）
    logger.info("编码 %d 个字符...", len(chars))
    char_texts = [f"她叫小{ch}" for ch in chars]  # 用"小X"作为单字名模板
    char_vecs = encode_texts(model, tokenizer, char_texts, device, batch_size=128)
    logger.info("编码完成: shape=%s", char_vecs.shape)

    # 计算WEAT得分
    logger.info("计算BERT WEAT得分...")
    results = {}
    for i, char in enumerate(chars):
        if (i + 1) % 500 == 0:
            logger.info("  进度: %d/%d", i + 1, len(chars))

        vec = char_vecs[i]
        scores = {}
        for dim in DIMS:
            pos_sim = cosine_sim(vec, attr_vecs[dim]["pos_mean"])
            neg_sim = cosine_sim(vec, attr_vecs[dim]["neg_mean"])
            scores[dim] = round(pos_sim - neg_sim, 6)

        scores["composite"] = round(float(np.mean(list(scores.values()))), 6)
        results[char] = scores

    logger.info("完成: %d 个字", len(results))

    # 保存
    out_path = DATA_DIR / "char_scores_bert.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("保存: %s", out_path)

    # Top 10
    ranked = sorted(results.items(), key=lambda x: x[1]["composite"], reverse=True)
    print("\nBERT Top 15 综合:")
    for ch, s in ranked[:15]:
        print(f"  {ch} = {s['composite']:.4f}")

    logger.info("耗时 %.1f 秒", time.time() - start)


if __name__ == "__main__":
    main()
