# How AI Sees Your Name

输入任意名字，看它在AI向量空间中的"命运"：财富、智慧、幸福、健康、领导力、美感六维评分。

> 通过分析AI模型中名字的向量表示（word embedding），揭示名字在AI系统中的隐性语义联想。支持中文和英文名字分析。

## 核心思路

AI模型的embedding空间中，名字与"财富""成功""幸福""智慧"等正面概念的距离各不相同。在AI深度渗透生活的时代（简历筛选、内容推荐等），名字在向量空间中的位置可能带来隐性影响。

本项目用 WEAT（Word Embedding Association Test）方法量化这种关联，并通过交互式Web应用让任何人都可以探索自己名字在AI眼中的"画像"。

## 方法论

### 实验1: WEAT偏好测量
使用 [Word Embedding Association Test](https://science.sciencemag.org/content/356/6334/183) 测量每个候选名字在embedding空间中与正面/负面概念的关联强度。

**6个评估维度：** 财富 · 智慧 · 幸福 · 健康 · 领导力 · 美感

### 实验2: LLM第一印象测试
通过API批量测试Claude/GPT对名字的"第一印象"——包括职业猜测、经济状况、社会地位等评分。

### 实验3: 综合评分
加权合并WEAT得分、LLM印象、名字频率适中度等多维度指标。

## 项目结构

```
how-ai-sees-your-name/
├── config/
│   └── settings.py              # 全局配置、属性词集、候选名字
├── src/
│   ├── embeddings/
│   │   ├── chinese_vectors.py   # 腾讯/北师大中文词向量加载
│   │   ├── english_vectors.py   # GloVe英文词向量加载
│   │   └── bert_embeddings.py   # BERT上下文embedding
│   ├── weat/
│   │   └── calculator.py        # WEAT得分计算器
│   ├── names/
│   │   └── candidate_pool.py    # 候选名字池构建
│   ├── llm/
│   │   └── first_impression.py  # LLM第一印象测试
│   ├── scoring/
│   │   └── composite.py         # 综合评分系统
│   └── visualization/
│       └── plots.py             # 雷达图、热力图、排名图
├── scripts/
│   ├── download_data.py         # 数据下载
│   ├── run_weat.py              # 运行WEAT实验
│   ├── run_llm_test.py          # 运行LLM测试
│   └── run_full_pipeline.py     # 一键完整Pipeline
├── data/raw/                    # 原始数据 (git忽略)
├── results/                     # 分析结果
├── requirements.txt
└── .gitignore
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

```bash
python scripts/download_data.py --all
```

这会：
- 安装 `text2vec`（含轻量版腾讯中文词向量）
- 下载 GloVe 英文词向量
- 克隆 Chinese Gender Dataset

### 3. 一键运行

```bash
# 完整Pipeline（含LLM测试，需要API key）
python scripts/run_full_pipeline.py

# 只跑WEAT（不需要API，不花钱）
python scripts/run_full_pipeline.py --skip-llm

# 使用OpenAI替代Anthropic
python scripts/run_full_pipeline.py --llm-api openai
```

### 4. 单独运行各实验

```bash
# 只跑WEAT
python scripts/run_weat.py --model text2vec

# 只跑LLM测试（对WEAT前20名）
python scripts/run_llm_test.py --api anthropic --top 20 --repeat 5
```

## 评分维度与权重

| 维度 | 权重 | 数据来源 |
|------|------|----------|
| 财富联想 | 20% | WEAT |
| 智慧联想 | 15% | WEAT |
| 幸福联想 | 15% | WEAT |
| 领导力 | 10% | WEAT |
| 美感 | 10% | WEAT |
| LLM印象 | 15% | Claude/GPT API |
| 频率适中 | 5% | Chinese Gender Dataset |
| 健康联想 | 10% | WEAT |

## 数据来源

- **腾讯AI Lab中文词向量** — 1200万词，200维
- **GloVe** — Stanford NLP, 6B tokens, 300维
- **Chinese Gender Dataset** — 105万名字，Nature Scientific Data (2025)
- **WEAT方法** — Caliskan et al., Science (2017)

## 环境变量

LLM测试需要设置API密钥：

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# 或
export OPENAI_API_KEY=sk-...
```

## 参考文献

- Caliskan, Bryson & Narayanan (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*
- Garg et al. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. *PNAS*
- Bolukbasi et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? *NeurIPS*
- Shi & Tong (2025). An Open Dataset of Chinese Name-to-Gender Associations. *Nature Scientific Data*

## License

MIT
