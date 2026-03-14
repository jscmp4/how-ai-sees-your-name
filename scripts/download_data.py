#!/usr/bin/env python3
"""数据下载脚本 — 下载所需的词向量和名字数据集。

用法:
    python scripts/download_data.py [--all | --tencent | --glove | --names]
"""

import argparse
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path, desc: str = ""):
    """带进度条下载文件。"""
    if dest.exists():
        logger.info("文件已存在，跳过: %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("下载 %s → %s", desc or url, dest)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_tencent_embedding():
    """下载轻量版腾讯词向量（通过text2vec）。"""
    logger.info("=" * 60)
    logger.info("安装 text2vec (含轻量版腾讯词向量) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "text2vec", "-q"])

    logger.info("text2vec 安装完成。")
    logger.info("轻量版词向量会在首次使用时自动下载到缓存。")
    logger.info("")
    logger.info("如需完整版腾讯词向量 (16GB)，请手动下载:")
    logger.info("  https://ai.tencent.com/ailab/nlp/en/embedding.html")
    logger.info("  下载后放到: %s", DATA_RAW / "tencent-ailab-embedding-zh-d200-v0.2.0-s.txt")


def download_glove():
    """下载GloVe英文词向量。"""
    logger.info("=" * 60)
    logger.info("下载 GloVe 6B 词向量 ...")

    zip_path = DATA_RAW / "glove.6B.zip"
    url = "https://nlp.stanford.edu/data/glove.6B.zip"

    download_file(url, zip_path, desc="glove.6B.zip")

    target = DATA_RAW / "glove.6B.300d.txt"
    if not target.exists():
        logger.info("解压 glove.6B.300d.txt ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("glove.6B.300d.txt", DATA_RAW)
        logger.info("解压完成: %s", target)


def download_chinese_names():
    """克隆 Chinese Gender Dataset。"""
    logger.info("=" * 60)
    dest = DATA_RAW / "Chinese-Gender-dataset"
    if dest.exists():
        logger.info("Chinese Gender Dataset 已存在: %s", dest)
        return

    logger.info("克隆 Chinese Gender Dataset ...")
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/tongt1213/Chinese-Gender-dataset.git",
        str(dest),
    ])
    logger.info("克隆完成: %s", dest)


def main():
    parser = argparse.ArgumentParser(description="下载 How AI Sees Your Name 项目所需数据")
    parser.add_argument("--all", action="store_true", help="下载所有数据")
    parser.add_argument("--tencent", action="store_true", help="安装text2vec (腾讯词向量)")
    parser.add_argument("--glove", action="store_true", help="下载GloVe英文词向量")
    parser.add_argument("--names", action="store_true", help="下载中文名字数据集")
    args = parser.parse_args()

    if not any([args.all, args.tencent, args.glove, args.names]):
        args.all = True

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if args.all or args.tencent:
        download_tencent_embedding()

    if args.all or args.glove:
        download_glove()

    if args.all or args.names:
        download_chinese_names()

    logger.info("=" * 60)
    logger.info("数据下载完成！")


if __name__ == "__main__":
    main()
