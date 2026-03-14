"""全局配置"""

from pathlib import Path

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── 词向量路径 ──
TENCENT_EMBEDDING_PATH = DATA_RAW / "tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"
GLOVE_EMBEDDING_PATH = DATA_RAW / "glove.6B.300d.txt"

# ── 名字数据集路径 ──
CHINESE_GENDER_DATASET_DIR = DATA_RAW / "Chinese-Gender-dataset"

# ── LLM API ──
LLM_TEMPERATURE = 0.7
LLM_REPEAT_COUNT = 10  # 每个名字测试次数

# ── 评分权重 ──
SCORING_WEIGHTS = {
    "wealth":     0.20,  # 财富联想
    "wisdom":     0.15,  # 智慧联想
    "happiness":  0.15,  # 幸福联想
    "health":     0.10,  # 健康联想（降低少许，原文未单独列出大权重）
    "leadership": 0.10,  # 领导力联想
    "beauty":     0.10,  # 美感联想
    "llm":        0.15,  # LLM第一印象
    "frequency":  0.05,  # 频率适中度（相对权重微调，总和=1）
}

# ── WEAT属性词集 ──
WEAT_ATTRIBUTES = {
    "wealth": {
        "positive": ["富裕", "成功", "财富", "繁荣", "富有", "兴旺", "昌盛", "富贵"],
        "negative": ["贫穷", "贫困", "穷困", "匮乏", "拮据", "窘迫", "落魄", "潦倒"],
    },
    "wisdom": {
        "positive": ["智慧", "聪明", "博学", "睿智", "才华", "才能", "天赋", "英明"],
        "negative": ["愚蠢", "愚昧", "无知", "笨拙", "迟钝", "呆滞", "糊涂", "蒙昧"],
    },
    "happiness": {
        "positive": ["幸福", "快乐", "美满", "喜悦", "甜蜜", "愉悦", "欢乐", "美好"],
        "negative": ["痛苦", "悲伤", "不幸", "苦难", "忧愁", "哀伤", "凄凉", "悲惨"],
    },
    "health": {
        "positive": ["健康", "长寿", "活力", "强壮", "茁壮", "康健", "矫健", "充沛"],
        "negative": ["疾病", "虚弱", "病痛", "衰弱", "体弱", "羸弱", "病弱", "衰败"],
    },
    "leadership": {
        "positive": ["领袖", "领导", "杰出", "卓越", "精英", "优秀", "出众", "非凡"],
        "negative": ["平庸", "普通", "庸碌", "平凡", "碌碌", "寻常", "一般", "渺小"],
    },
    "beauty": {
        "positive": ["美丽", "优雅", "端庄", "秀美", "婉约", "清秀", "出众", "灵秀"],
        "negative": ["丑陋", "粗俗", "难看", "庸俗", "邋遢", "猥琐", "鄙陋", "丑恶"],
    },
}

# ── 英文WEAT属性词集 ──
WEAT_ATTRIBUTES_EN = {
    "wealth": {
        "positive": ["wealthy", "successful", "rich", "prosperous", "affluent",
                      "fortune", "luxury", "thriving"],
        "negative": ["poor", "poverty", "destitute", "impoverished", "bankrupt",
                      "broke", "needy", "indigent"],
    },
    "wisdom": {
        "positive": ["intelligent", "brilliant", "wise", "clever", "genius",
                      "talented", "gifted", "knowledgeable"],
        "negative": ["stupid", "ignorant", "foolish", "dumb", "incompetent",
                      "clueless", "dim", "dense"],
    },
    "happiness": {
        "positive": ["happy", "joyful", "blissful", "cheerful", "delighted",
                      "content", "elated", "wonderful"],
        "negative": ["miserable", "sad", "unhappy", "depressed", "sorrowful",
                      "gloomy", "wretched", "tragic"],
    },
    "health": {
        "positive": ["healthy", "strong", "vigorous", "robust", "energetic",
                      "vital", "fit", "resilient"],
        "negative": ["sick", "weak", "frail", "diseased", "fragile",
                      "feeble", "ailing", "infirm"],
    },
    "leadership": {
        "positive": ["leader", "outstanding", "excellent", "exceptional", "elite",
                      "distinguished", "remarkable", "extraordinary"],
        "negative": ["mediocre", "ordinary", "average", "unremarkable", "mundane",
                      "forgettable", "common", "insignificant"],
    },
    "beauty": {
        "positive": ["beautiful", "elegant", "graceful", "lovely", "gorgeous",
                      "radiant", "charming", "exquisite"],
        "negative": ["ugly", "hideous", "unattractive", "repulsive", "grotesque",
                      "homely", "unsightly", "plain"],
    },
}

# ── 候选中文名字（初始池，可从数据集扩充） ──
INITIAL_CHINESE_CANDIDATES = [
    # 经典高雅
    "思琪", "梓涵", "诗涵", "欣妍", "语桐", "若曦", "芷若", "清妍",
    "雨桐", "子涵", "可馨", "诗蕊", "梦瑶", "佳怡", "心怡", "雅琪",
    # 文学气质
    "念初", "知许", "映雪", "晓棠", "语嫣", "沐晴", "栖桐", "望舒",
    "疏影", "锦书", "如歌", "兰芝", "清欢", "初见", "安然", "素心",
    # 大气端庄
    "明珠", "瑞雪", "嘉禾", "锦程", "泓清", "辰星", "霁月", "星河",
    "云舒", "澜清", "煦宁", "晏清", "瑾瑜", "璟怡", "琬琰", "珺瑶",
    # 灵动温婉
    "悠然", "晴川", "芊羽", "紫萱", "韵竹", "若水", "初夏", "暖阳",
    "微澜", "清歌", "念卿", "浅吟", "轻舟", "拾光", "漫漫", "盈盈",
]

# ── 候选英文名字 ──
INITIAL_ENGLISH_CANDIDATES = [
    # Classic / Elegant
    "Sophia", "Olivia", "Charlotte", "Eleanor", "Victoria",
    "Isabella", "Audrey", "Vivian", "Celeste", "Aurora",
    # Modern / Trendy
    "Nova", "Luna", "Aria", "Isla", "Ivy",
    "Stella", "Iris", "Chloe", "Zoe", "Mia",
    # Strong / Distinguished
    "Alexandra", "Catherine", "Elizabeth", "Margaret", "Valentina",
    "Genevieve", "Penelope", "Josephine", "Evangeline", "Seraphina",
    # Nature / Artistic
    "Lily", "Violet", "Rose", "Jasmine", "Hazel",
    "Scarlett", "Ruby", "Amber", "Jade", "Willow",
]
