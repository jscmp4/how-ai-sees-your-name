#!/usr/bin/env python3
"""预计算汉字WEAT得分和邻居词。

输出:
  data/char_scores.json  — {char: {wealth: 0.12, wisdom: 0.08, ...}, ...}
  data/char_neighbors.json — {char: [["邻居1", 0.89], ["邻居2", 0.85], ...], ...}

用法:
  python scripts/precompute_char_scores.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import WEAT_ATTRIBUTES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 适合人名的常用汉字（约1800字） ──
# 来源: 常见取名用字 + 现代汉语常用字表中语义适合人名的字
# 分类整理方便维护

# 自然/天象
NATURE = (
    "天日月星辰云霞雪霜露雨虹风雷电晨曦晖暮"
    "春夏秋冬朝夕晓暖晴岚霁曙旭昕晏晔煦熙"
    "山川河海湖泉溪江波澜潮源流涧泓渊淼瀚"
    "林木森松柏桐桑梧柳杨榆楠榕枫桦槐檀杉"
    "花草兰竹梅莲荷菊萱芝芸蕙蕊芷荟茗薇藤"
    "玉石珀琥珊瑚玛瑙翡翠琉璃珍宝瑾瑜璟琬"
)

# 品德/气质
VIRTUE = (
    "仁义礼智信忠孝廉勇毅诚谦和善良慈悲恩"
    "德惠泽润恕容雅正直刚柔温谨敬慎恭俭让"
    "淳朴质真纯净洁素清雅静安宁泰宜怡然悠"
    "端庄敏睿慧聪颖达通博厚深远弘广博渊明"
    "贤哲圣英豪俊杰卓越翰逸轩昂凯悦乐欣畅"
)

# 美好/寓意
AUSPICIOUS = (
    "福禄寿喜庆祥瑞嘉吉康健宁安平和丰盈裕"
    "荣华富贵昌盛兴旺繁茂隆盛显赫崇尚尊贵"
    "美丽秀雅婉约柔婷娴淑妍媛姝妙佳颜彤丹"
    "灵巧曼妙翩飘舞韵律歌诗词曲文章书翰墨"
    "成功业绩勋卓著立志向望期冀愿梦想希盼"
)

# 色彩/光影
COLOR_LIGHT = (
    "红朱赤丹绯绛紫青蓝碧绿翠黄金银白皓素"
    "黛墨灰彩艳绮绚绘辉耀灿烁焕煌照映晃亮"
    "光明暗影华彩虹霓润泽鲜妍艳丽娟秀娇柔"
)

# 动作/状态
ACTION = (
    "飞翔翱鹏鸿展翼腾跃奔驰驾航行远游历遊"
    "思念想忆怀感悟觉知晓解悉透彻圆满全备"
    "生长育养培植树建造创新开拓进取奋发图强"
    "守护卫保佑助扶持携带引领导启迪化育"
)

# 人物/身份
PERSON = (
    "君王侯公卿将相师帅长官臣士民人儿女子男"
    "父母兄弟姐妹亲友朋伴侣偶翁媪叟童少"
)

# 器物/象征
SYMBOL = (
    "鼎钟鼓琴瑟筝笛箫琵琶剑刀弓矛盾冠冕衣"
    "裳履屏帘幕帐旗帜旌幡印玺符令牌笏简册"
    "镜灯烛台炉鼎壶杯盏盘碗钵瓶罐缸缶甑"
)

# 动物
ANIMAL = (
    "龙凤麟鹤鹏雁鸿鸥鹭莺燕鸽鹊鸾鹄鹃鸳鸯"
    "虎豹狮象熊鹿马骏骐麒驹羊犬兔猫蝶蜂蚕"
    "鱼鲤鲲鲸蛟蛙龟鳌蟒螭鳳"
)

# 数字/方位/天干地支
MISC = (
    "一二三四五六七八九十百千万亿兆"
    "东西南北中上下左右前后内外"
    "甲乙丙丁戊己庚辛壬癸"
    "子丑寅卯辰巳午未申酉戌亥"
)

# 常用名字高频字（从实际取名统计中补充）
HIGH_FREQ_NAME = (
    "梓涵轩宇浩然诗琪雨萱子墨欣怡佳琪思源"
    "若溪语嫣沐阳紫萱韵竹依然一诺可馨心怡"
    "雅琪语桐若曦梦瑶嘉怡锦程辰星煦宁晏清"
    "珺瑶璟怡琬琰悠然晴川芊羽微澜清歌念卿"
    "望舒疏影锦书如歌初见安澜泓清霁月星河"
    "映雪晓棠沐晴栖桐素心拾光盈盈明珠瑞雪"
    "嘉禾之恒以沫亦辰昭华承泽铭哲奕辰睿渊"
    "彦博俊熙浩宇天佑泽楷瑾萱靖琪书瑶映彤"
    "若彤婧琪佩玲慧敏晓蕾雅芙馨月碧瑶冰洁"
    "芮溪楚瑜知行乐天朗逸文韬武略鸿飞逸飞"
    "思齐学勤任重道远临风沐雨含章锦绣华章"
)

# 额外补充的取名常用字
EXTRA = (
    "啟珏琅珂珞珈珮琛琦琮琳琪琰瑗瑶瑭璐璇"
    "璋璞璟环璧瓒瓘衡鑫淏渝澈瀛湘滢沁沐汐"
    "漪潼澄浠洛淇涵清潇漫澜沅泽洋津浚波渡"
    "峰岭峻崇嵘嵩岩峡巍屹岑昆仑恒嶷桀尧舜"
    "禹汤文武周秦楚齐鲁燕赵韩魏晋隋唐宋元"
    "程颐朱熹陆王阳明曾参颜回冉求仲由季路"
    "诺谣瑾瑜珩璠玥珣珹琤瑀瑄瑆瑢瑨瑱瑳瑹"
    "筠箐簪筱篁笙笺筝筵簌翊翎翌翡翥翰翮翼"
    "罡霖霏霓霭霆霈霁霄霂雯雰霞雪零雫雹需"
)


def build_char_set() -> list[str]:
    """构建去重的取名汉字集。"""
    all_chars = set()
    for group in [NATURE, VIRTUE, AUSPICIOUS, COLOR_LIGHT, ACTION,
                  PERSON, SYMBOL, ANIMAL, MISC, HIGH_FREQ_NAME, EXTRA]:
        for char in group:
            if '\u4e00' <= char <= '\u9fff':  # 只保留CJK统一汉字
                all_chars.add(char)
    return sorted(all_chars)


def compute_weat_score(model, char: str, pos_words: list[str], neg_words: list[str]) -> float | None:
    """计算单个汉字对一组正面/负面属性词的WEAT得分。"""
    if char not in model:
        return None

    char_vec = model[char]
    char_norm = np.linalg.norm(char_vec)
    if char_norm == 0:
        return None

    pos_sims = []
    for w in pos_words:
        if w in model:
            w_vec = model[w]
            w_norm = np.linalg.norm(w_vec)
            if w_norm > 0:
                pos_sims.append(float(np.dot(char_vec, w_vec) / (char_norm * w_norm)))

    neg_sims = []
    for w in neg_words:
        if w in model:
            w_vec = model[w]
            w_norm = np.linalg.norm(w_vec)
            if w_norm > 0:
                neg_sims.append(float(np.dot(char_vec, w_vec) / (char_norm * w_norm)))

    if not pos_sims or not neg_sims:
        return None

    return float(np.mean(pos_sims) - np.mean(neg_sims))


def compute_neighbors(model, char: str, topn: int = 10) -> list[list]:
    """获取一个字在向量空间中最近的邻居词。"""
    if char not in model:
        return []

    try:
        neighbors = model.most_similar(char, topn=topn + 20)
        # 过滤掉纯标点和单个字母，保留有意义的词
        filtered = []
        for word, score in neighbors:
            if len(word) >= 1 and any('\u4e00' <= c <= '\u9fff' for c in word):
                filtered.append([word, round(float(score), 4)])
            if len(filtered) >= topn:
                break
        return filtered
    except Exception as e:
        logger.warning("获取 '%s' 邻居失败: %s", char, e)
        return []


def main():
    start_time = time.time()

    # 构建字符集
    chars = build_char_set()
    logger.info("取名汉字集大小: %d", len(chars))

    # 加载模型
    logger.info("正在加载 text2vec 轻量版腾讯词向量...")
    from text2vec import Word2Vec
    w2v = Word2Vec("w2v-light-tencent-chinese")
    model = w2v.w2v
    logger.info("模型加载完成，词汇量: %d，维度: %d", len(model), model.vector_size)

    # 检查覆盖率
    found = [c for c in chars if c in model]
    logger.info("字符覆盖率: %d/%d (%.1f%%)", len(found), len(chars), 100 * len(found) / len(chars))

    # 计算每个字的6维WEAT得分
    logger.info("计算WEAT得分...")
    char_scores = {}
    for i, char in enumerate(found):
        if (i + 1) % 200 == 0:
            logger.info("  进度: %d/%d", i + 1, len(found))

        scores = {}
        for dim, attrs in WEAT_ATTRIBUTES.items():
            score = compute_weat_score(model, char, attrs["positive"], attrs["negative"])
            if score is not None:
                scores[dim] = round(score, 6)

        if scores:
            # 计算综合分 (等权平均)
            scores["composite"] = round(float(np.mean(list(scores.values()))), 6)
            char_scores[char] = scores

    logger.info("成功计算 %d 个字的WEAT得分", len(char_scores))

    # 计算邻居词
    logger.info("计算邻居词...")
    char_neighbors = {}
    for i, char in enumerate(found):
        if (i + 1) % 200 == 0:
            logger.info("  进度: %d/%d", i + 1, len(found))

        neighbors = compute_neighbors(model, char, topn=10)
        if neighbors:
            char_neighbors[char] = neighbors

    logger.info("成功计算 %d 个字的邻居词", len(char_neighbors))

    # 输出
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "char_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(char_scores, f, ensure_ascii=False, indent=2)
    logger.info("得分已保存: %s (%d chars)", scores_path, len(char_scores))

    neighbors_path = output_dir / "char_neighbors.json"
    with open(neighbors_path, "w", encoding="utf-8") as f:
        json.dump(char_neighbors, f, ensure_ascii=False, indent=2)
    logger.info("邻居已保存: %s (%d chars)", neighbors_path, len(char_neighbors))

    elapsed = time.time() - start_time
    logger.info("预计算完成，耗时 %.1f 秒", elapsed)

    # 打印一些有趣的统计
    print("\n" + "=" * 60)
    print("各维度Top 10汉字")
    print("=" * 60)
    for dim in list(WEAT_ATTRIBUTES.keys()) + ["composite"]:
        ranked = sorted(char_scores.items(), key=lambda x: x[1].get(dim, -999), reverse=True)
        top10 = [(c, s[dim]) for c, s in ranked[:10] if dim in s]
        print(f"\n{dim}:")
        for c, s in top10:
            print(f"  {c} = {s:.4f}")


if __name__ == "__main__":
    main()
