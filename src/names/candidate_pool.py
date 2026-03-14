"""候选名字池构建 — 从数据集筛选 + 手动候选合并"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_chinese_gender_dataset(dataset_dir: str | Path) -> pd.DataFrame:
    """加载Chinese Gender Dataset。

    期望数据目录中有 CSV 文件，包含 name, gender, frequency 等列。
    自动扫描目录下的 csv 文件。
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"数据集目录不存在: {dataset_dir}\n"
            "请运行: git clone https://github.com/tongt1213/Chinese-Gender-dataset.git "
            f"到 {dataset_dir}"
        )

    csv_files = list(dataset_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {dataset_dir} 中未找到CSV文件")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8")
            dfs.append(df)
            logger.info("加载 %s: %d 条记录", f.name, len(df))
        except Exception as e:
            logger.warning("跳过文件 %s: %s", f, e)

    if not dfs:
        raise ValueError("未能成功加载任何CSV文件")

    return pd.concat(dfs, ignore_index=True)


def filter_female_names(
    df: pd.DataFrame,
    gender_col: str = "gender",
    name_col: str = "name",
    female_value: str = "F",
    min_frequency: int | None = None,
    max_frequency: int | None = None,
    freq_col: str = "frequency",
) -> list[str]:
    """从数据集中筛选女性名字。

    Parameters
    ----------
    df : 名字数据集DataFrame
    gender_col : 性别列名
    name_col : 名字列名
    female_value : 女性性别标识值
    min_frequency / max_frequency : 频率过滤
    freq_col : 频率列名
    """
    mask = df[gender_col] == female_value

    if min_frequency is not None and freq_col in df.columns:
        mask &= df[freq_col] >= min_frequency

    if max_frequency is not None and freq_col in df.columns:
        mask &= df[freq_col] <= max_frequency

    names = df.loc[mask, name_col].dropna().unique().tolist()
    logger.info("筛选出 %d 个女性名字", len(names))
    return names


def build_candidate_pool(
    dataset_names: list[str] | None = None,
    manual_names: list[str] | None = None,
    max_total: int = 5000,
) -> list[str]:
    """合并数据集筛选名字和手动候选名字，去重。"""
    pool = set()

    if manual_names:
        pool.update(manual_names)

    if dataset_names:
        pool.update(dataset_names)

    pool = [n for n in pool if n and len(n) >= 1]

    if len(pool) > max_total:
        # 优先保留手动名字，然后随机抽样
        manual_set = set(manual_names or [])
        manual_in_pool = [n for n in pool if n in manual_set]
        rest = [n for n in pool if n not in manual_set]

        import random
        random.shuffle(rest)
        pool = manual_in_pool + rest[: max_total - len(manual_in_pool)]

    logger.info("候选名字池大小: %d", len(pool))
    return sorted(pool)


def get_name_frequency_stats(
    df: pd.DataFrame,
    name_col: str = "name",
    freq_col: str = "frequency",
) -> pd.DataFrame:
    """获取名字频率统计，用于频率适中度打分。"""
    if freq_col not in df.columns:
        logger.warning("数据集中没有频率列 '%s'", freq_col)
        return pd.DataFrame()

    stats = df.groupby(name_col)[freq_col].sum().reset_index()
    stats.columns = ["name", "total_frequency"]
    stats = stats.sort_values("total_frequency", ascending=False).reset_index(drop=True)
    stats["frequency_rank"] = stats.index + 1
    stats["frequency_percentile"] = stats["frequency_rank"] / len(stats)
    return stats
