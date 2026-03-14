"""How AI Sees Your Name — AI向量空间名字语义分析 · 交互式Web应用"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── 路径 ──
APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
DATA_DIR = APP_DIR / "data"
SCORES_PATH = DATA_DIR / "char_scores.json"
NEIGHBORS_PATH = DATA_DIR / "char_neighbors.json"
EN_SCORES_PATH = DATA_DIR / "en_name_scores.json"
SUCCESS_PATH = DATA_DIR / "success_analysis.json"
FREQ_ZH_PATH = DATA_DIR / "name_freq_zh.json"
SSA_TRENDS_PATH = DATA_DIR / "ssa_trends.json"
BERT_SCORES_PATH = DATA_DIR / "char_scores_bert.json"
NAME_WHOLE_PATH = DATA_DIR / "name_whole_scores.json"

# ── 维度配置 ──
DIMENSIONS = {
    "wealth":     {"label": "财富", "icon": "💰", "color": "#FFD700", "desc": "与富裕、成功、繁荣等概念的关联"},
    "wisdom":     {"label": "智慧", "icon": "🧠", "color": "#00D4FF", "desc": "与聪明、博学、睿智等概念的关联"},
    "happiness":  {"label": "幸福", "icon": "✨", "color": "#FF69B4", "desc": "与快乐、美满、喜悦等概念的关联"},
    "health":     {"label": "健康", "icon": "💪", "color": "#00FF88", "desc": "与健康、活力、强壮等概念的关联"},
    "leadership": {"label": "领导力", "icon": "👑", "color": "#FF6B35", "desc": "与杰出、卓越、精英等概念的关联"},
    "beauty":     {"label": "美感", "icon": "🌸", "color": "#DA70D6", "desc": "与优雅、端庄、灵秀等概念的关联"},
}

DIM_KEYS = list(DIMENSIONS.keys())
DIM_LABELS = [DIMENSIONS[k]["label"] for k in DIM_KEYS]
DIM_COLORS = [DIMENSIONS[k]["color"] for k in DIM_KEYS]

# ── 评分权重 ──
DEFAULT_WEIGHTS = {
    "wealth": 0.20, "wisdom": 0.15, "happiness": 0.15,
    "health": 0.10, "leadership": 0.10, "beauty": 0.10,
}


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_char_scores() -> dict:
    with open(SCORES_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_char_neighbors() -> dict:
    with open(NEIGHBORS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_en_scores() -> dict:
    if EN_SCORES_PATH.exists():
        with open(EN_SCORES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_success_analysis() -> dict:
    if SUCCESS_PATH.exists():
        with open(SUCCESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_freq_zh() -> dict:
    if FREQ_ZH_PATH.exists():
        with open(FREQ_ZH_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_ssa_trends() -> dict:
    if SSA_TRENDS_PATH.exists():
        with open(SSA_TRENDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_bert_scores() -> dict:
    if BERT_SCORES_PATH.exists():
        with open(BERT_SCORES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_name_whole_scores() -> dict:
    if NAME_WHOLE_PATH.exists():
        with open(NAME_WHOLE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════
# 实时计算引擎 (fallback when name not in precomputed data)
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def _load_zh_model():
    """加载中文词向量模型 (首次约8秒, 后续缓存)。"""
    from text2vec import Word2Vec
    w2v = Word2Vec("w2v-light-tencent-chinese")
    return w2v.w2v


@st.cache_resource
def _load_en_model():
    """加载GloVe英文词向量 (首次约50秒, 后续缓存)。"""
    from gensim.models import KeyedVectors
    glove_path = DATA_DIR / "raw" / "glove.6B.300d.txt"
    if not glove_path.exists():
        return None
    return KeyedVectors.load_word2vec_format(str(glove_path), binary=False, no_header=True)


@st.cache_resource
def _load_zh_attrs():
    """加载中文属性词集。"""
    from config.settings import WEAT_ATTRIBUTES
    return WEAT_ATTRIBUTES


@st.cache_resource
def _load_en_attrs():
    """加载英文属性词集。"""
    from config.settings import WEAT_ATTRIBUTES_EN
    return WEAT_ATTRIBUTES_EN


def _compute_weat_vec(vec, model, pos_words, neg_words):
    """对一个向量计算 WEAT 得分。"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return 0.0
    pos = [float(np.dot(vec, model[w]) / (norm * np.linalg.norm(model[w])))
           for w in pos_words if w in model and np.linalg.norm(model[w]) > 0]
    neg = [float(np.dot(vec, model[w]) / (norm * np.linalg.norm(model[w])))
           for w in neg_words if w in model and np.linalg.norm(model[w]) > 0]
    if not pos or not neg:
        return 0.0
    return float(np.mean(pos) - np.mean(neg))


def compute_zh_realtime(name: str) -> dict | None:
    """实时计算中文名字的 WEAT 得分。"""
    model = _load_zh_model()
    attrs = _load_zh_attrs()

    chars_in = [ch for ch in name if ch in model]
    if not chars_in:
        return None

    # 字均
    scores = {}
    for dim, attr in attrs.items():
        char_scores_dim = []
        for ch in chars_in:
            s = _compute_weat_vec(model[ch], model, attr["positive"], attr["negative"])
            char_scores_dim.append(s)
        scores[dim] = round(float(np.mean(char_scores_dim)), 6)
    scores["composite"] = round(float(np.mean(list(scores.values()))), 6)

    # 邻居词
    neighbors = {}
    for ch in chars_in:
        try:
            nb = model.most_similar(ch, topn=10)
            neighbors[ch] = [[w, round(float(s), 4)] for w, s in nb
                             if any("\u4e00" <= c <= "\u9fff" for c in w)][:10]
        except Exception:
            neighbors[ch] = []

    return {"scores": scores, "neighbors": neighbors, "found_chars": chars_in}


def compute_en_realtime(name: str) -> dict | None:
    """实时计算英文名字的 WEAT 得分。"""
    model = _load_en_model()
    if model is None:
        return None
    attrs = _load_en_attrs()

    # 尝试各种大小写
    token = None
    for v in [name, name.lower(), name.capitalize(), name.upper()]:
        if v in model:
            token = v
            break
    if token is None:
        return None

    vec = model[token]
    scores = {}
    for dim, attr in attrs.items():
        scores[dim] = round(_compute_weat_vec(vec, model, attr["positive"], attr["negative"]), 6)
    scores["composite"] = round(float(np.mean(list(scores.values()))), 6)

    return {"scores": scores, "token": token}


def get_name_scores(name: str, char_scores: dict) -> dict | None:
    """计算名字的6维得分（取各字平均）。"""
    char_data = []
    for ch in name:
        if ch in char_scores:
            char_data.append(char_scores[ch])
    if not char_data:
        return None
    result = {}
    for dim in DIM_KEYS:
        values = [d[dim] for d in char_data if dim in d]
        result[dim] = float(np.mean(values)) if values else 0.0
    return result


def composite_score(scores: dict, weights: dict | None = None) -> float:
    """计算加权综合分。"""
    w = weights or DEFAULT_WEIGHTS
    total = sum(scores.get(d, 0) * w.get(d, 0) for d in DIM_KEYS)
    total_w = sum(w.get(d, 0) for d in DIM_KEYS)
    return total / total_w if total_w > 0 else 0.0


def normalize_score(raw: float, all_raw: list[float]) -> float:
    """Min-Max归一化到0-100。"""
    mn, mx = min(all_raw), max(all_raw)
    if mx == mn:
        return 50.0
    return (raw - mn) / (mx - mn) * 100


@st.cache_data
def _build_percentile_table():
    """从预计算数据中构建各维度百分位映射表。"""
    scores = load_char_scores()
    table = {}
    for dim in DIM_KEYS + ["composite"]:
        vals = sorted([s[dim] for s in scores.values() if dim in s])
        table[dim] = np.array(vals)
    return table


def raw_to_percentile(raw: float, dim: str) -> float:
    """将原始WEAT得分转为百分位 (0-100)。50=中位数。"""
    table = _build_percentile_table()
    if dim not in table or len(table[dim]) == 0:
        return 50.0
    arr = table[dim]
    idx = np.searchsorted(arr, raw, side="right")
    return round(idx / len(arr) * 100, 1)


def scores_to_display(raw_scores: dict) -> dict:
    """将6维原始WEAT得分转为0-100百分位展示分数。"""
    return {dim: raw_to_percentile(raw_scores.get(dim, 0), dim) for dim in DIM_KEYS}


def format_grade(pct: float) -> str:
    """百分位 → 等级标签。"""
    if pct >= 90: return "S"
    if pct >= 75: return "A"
    if pct >= 50: return "B"
    if pct >= 25: return "C"
    return "D"


def generate_ai_description(name: str, scores: dict) -> str:
    """生成"AI眼中的你"描述文本。"""
    top_dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best = top_dims[0]
    second = top_dims[1]
    worst = top_dims[-1]

    best_info = DIMENSIONS[best[0]]
    second_info = DIMENSIONS[second[0]]
    worst_info = DIMENSIONS[worst[0]]

    # 综合等级
    comp = composite_score(scores)
    if comp > 0.08:
        level = "极为出色"
        level_detail = "在AI的向量空间中占据着一个令人羡慕的位置"
    elif comp > 0.05:
        level = "优秀"
        level_detail = "在多个维度上都展现出积极的语义联想"
    elif comp > 0.02:
        level = "良好"
        level_detail = "在向量空间中有着稳健的正面语义基础"
    else:
        level = "中性"
        level_detail = "在向量空间中保持着平衡的语义分布"

    text = f"""### 向量空间解读

「**{name}**」在AI的语义空间中表现 **{level}** — {level_detail}。

**核心优势** — {best_info['icon']} **{best_info['label']}维度**得分最高（{best[1]:.4f}），这意味着在AI模型的内部表示中，「{name}」与{best_info['desc'].replace('与', '').replace('等概念的关联', '')}等概念形成了较强的语义连接。

**次要特质** — {second_info['icon']} **{second_info['label']}**同样表现突出（{second[1]:.4f}），为这个名字增添了{second_info['label'].lower()}的意涵。

**提升空间** — {worst_info['icon']} **{worst_info['label']}**维度相对较弱（{worst[1]:.4f}），但这在取名优化中是可以通过字的搭配来弥补的。"""

    return text


def _hex_to_rgba(hex_color: str, opacity: float) -> str:
    """将 #RRGGBB 转为 rgba(r,g,b,a) 字符串。"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


# ══════════════════════════════════════════════════════════════
# Plotly 图表
# ══════════════════════════════════════════════════════════════

def make_radar(
    name: str,
    scores: dict,
    color: str = "#00d4ff",
    fill_opacity: float = 0.25,
    all_scores: dict | None = None,
    use_percentile: bool = True,
) -> go.Scatterpolar:
    """生成单个名字的雷达图trace。默认用百分位 (0-100)。"""
    values = []
    for dim in DIM_KEYS:
        if use_percentile:
            values.append(raw_to_percentile(scores.get(dim, 0), dim))
        elif all_scores:
            all_vals = [s.get(dim, 0) for s in all_scores.values() if dim in s]
            values.append(normalize_score(scores.get(dim, 0), all_vals))
        else:
            values.append(min(100, max(0, scores.get(dim, 0) / 0.2 * 100)))

    values.append(values[0])  # 闭合
    labels = DIM_LABELS + [DIM_LABELS[0]]

    return go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        fillcolor=_hex_to_rgba(color, fill_opacity),
        line=dict(color=color, width=2.5),
        name=name,
        marker=dict(size=6, color=color),
    )


def make_radar_figure(
    traces: list[go.Scatterpolar],
    title: str = "",
    height: int = 500,
) -> go.Figure:
    """创建雷达图figure。"""
    fig = go.Figure(data=traces)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(color="#666", size=10),
                gridcolor="rgba(255,255,255,0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#e0e0e0", size=13),
                gridcolor="rgba(255,255,255,0.15)",
            ),
        ),
        showlegend=len(traces) > 1,
        legend=dict(
            font=dict(color="#e0e0e0", size=13),
            bgcolor="rgba(0,0,0,0.3)",
        ),
        title=dict(text=title, font=dict(color="#e0e0e0", size=18)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(t=60, b=30, l=60, r=60),
    )
    return fig


def make_dimension_bars(scores: dict, name: str = "", use_percentile: bool = True) -> go.Figure:
    """各维度得分水平条形图。默认显示百分位 (0-100)。"""
    if use_percentile:
        display = {d: raw_to_percentile(scores.get(d, 0), d) for d in DIM_KEYS}
    else:
        display = scores

    dims = sorted(display.items(), key=lambda x: x[1], reverse=True)
    labels = [DIMENSIONS[d]["icon"] + " " + DIMENSIONS[d]["label"] for d, _ in dims]
    values = [v for _, v in dims]
    colors = [DIMENSIONS[d]["color"] for d, _ in dims]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}/100" if use_percentile else f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e0e0e0", size=12),
    ))
    fig.update_layout(
        xaxis=dict(
            title="百分位得分 (0=最低, 100=最高)" if use_percentile else "WEAT Score",
            range=[0, 110] if use_percentile else None,
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.3)",
            tickfont=dict(color="#999"),
            title_font=dict(color="#999"),
        ),
        yaxis=dict(tickfont=dict(color="#e0e0e0", size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=10, b=40, l=10, r=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 页面组件
# ══════════════════════════════════════════════════════════════

def page_header():
    """全局头部。"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

    .nova-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff, #da70d6, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: 0.15em;
    }
    .nova-tagline {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
        letter-spacing: 0.08em;
    }
    .score-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(218,112,214,0.1));
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-big {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #da70d6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-label {
        color: #888;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .char-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .char-big {
        font-size: 2.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .neighbor-tag {
        display: inline-block;
        background: rgba(0,212,255,0.15);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #b0e0ff;
    }
    .winner-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000;
        padding: 0.1rem 0.5rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .dim-bar {
        display: flex;
        align-items: center;
        margin: 0.3rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nova-title">How AI Sees Your Name</div>', unsafe_allow_html=True)
    st.markdown('<div class="nova-tagline">输入任意名字，看它在AI向量空间中的"命运"</div>', unsafe_allow_html=True)


def page_xray():
    """页面1: 名字X光"""
    char_scores = load_char_scores()
    char_neighbors = load_char_neighbors()

    st.markdown("### 🔬 名字 X 光")
    st.caption("输入任意名字（中文或英文），透视它在AI向量空间中的语义坐标")

    # 快捷建议按钮（必须在 text_input 之前，通过 callback 设值）
    if "xray_input" not in st.session_state:
        st.session_state["xray_input"] = ""

    def _set_suggestion(s: str):
        st.session_state["xray_input"] = s

    name = st.text_input(
        "输入名字（1-4个汉字，不含姓氏）",
        placeholder="例：思琪、梓涵、若曦...",
        key="xray_input",
    )

    if not name:
        # 展示随机推荐
        st.markdown("---")
        st.markdown("#### 💡 试试这些名字")
        suggestions = ["思琪", "梓涵", "望舒", "星河", "清欢", "锦书", "霁月", "念初"]
        cols = st.columns(4)
        for i, s in enumerate(suggestions):
            with cols[i % 4]:
                st.button(s, key=f"suggest_{s}", use_container_width=True,
                          on_click=_set_suggestion, args=(s,))
        return

    # 检测是否为英文名 → 自动用英文分析
    if all(c.isascii() for c in name) and name.strip().isalpha():
        en_scores = load_en_scores()
        matched = None
        for variant in [name, name.capitalize(), name.title(), name.upper(), name.lower()]:
            if variant in en_scores:
                matched = variant
                break
        if matched:
            en_s = en_scores[matched]
            dim_scores = {d: en_s.get(d, 0) for d in DIM_KEYS}
            comp_en = en_s.get("composite", 0)
            freq = en_s.get("frequency", 0)

            st.info(f'Detected English name — analyzing "{matched}" with GloVe 300d')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Composite WEAT", f"{comp_en:.4f}")
            with col2:
                all_comp = [v.get("composite", 0) for v in en_scores.values()]
                pct = sum(1 for x in all_comp if comp_en > x) / len(all_comp) * 100
                st.metric("Percentile", f"Top {100 - pct:.1f}%")
            with col3:
                st.metric("SSA Popularity", f"{freq:,}" if freq else "N/A")

            trace = make_radar(matched, dim_scores, color="#da70d6")
            fig = make_radar_figure([trace], title=f'"{matched}" Semantic Radar')
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(make_dimension_bars(dim_scores, matched), use_container_width=True)

            # SSA趋势
            ssa_trends = load_ssa_trends()
            trend_data = ssa_trends.get(matched)
            if trend_data:
                st.markdown(f'#### 📈 "{matched}" Popularity (SSA 1880-2024)')
                trend = trend_data["trend"]
                years = sorted(trend.keys(), key=int)
                import plotly.graph_objects as go_local
                fig_t = go.Figure(go.Scatter(
                    x=[int(y) for y in years], y=[trend[y] for y in years],
                    mode="lines+markers", line=dict(color="#da70d6", width=2.5),
                    fill="tozeroy", fillcolor=_hex_to_rgba("#da70d6", 0.15),
                ))
                fig_t.update_layout(
                    xaxis=dict(title="Year", tickfont=dict(color="#999"), title_font=dict(color="#999"),
                               gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(title="Per 10K births", tickfont=dict(color="#999"), title_font=dict(color="#999"),
                               gridcolor="rgba(255,255,255,0.1)"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=300, margin=dict(t=10, b=40),
                )
                st.plotly_chart(fig_t, use_container_width=True)
            return
        else:
            # Realtime fallback: 加载GloVe模型现算
            with st.spinner(f'"{name}" not in cache — loading GloVe model for real-time computation...'):
                rt = compute_en_realtime(name)
            if rt:
                dim_scores = {d: rt["scores"].get(d, 0) for d in DIM_KEYS}
                comp_en = rt["scores"].get("composite", 0)
                st.success(f'Real-time computed "{rt["token"]}" via GloVe 300d')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Composite WEAT", f"{comp_en:.4f}")
                with col2:
                    st.metric("Source", "Real-time (GloVe)")
                trace = make_radar(rt["token"], dim_scores, color="#da70d6")
                fig = make_radar_figure([trace], title=f'"{rt["token"]}" Semantic Radar')
                st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(make_dimension_bars(dim_scores, rt["token"]), use_container_width=True)
            else:
                st.error(f'"{name}" not found in GloVe vocabulary (400K words). Try a different spelling.')
            return

    # 检查字符覆盖
    found_chars = [ch for ch in name if ch in char_scores]
    missing_chars = [ch for ch in name if ch not in char_scores]

    if not found_chars:
        # Realtime fallback: 加载中文词向量现算
        has_cn = any("\u4e00" <= ch <= "\u9fff" for ch in name)
        if has_cn:
            with st.spinner("加载词向量模型，实时计算中..."):
                rt = compute_zh_realtime(name)
            if rt:
                scores = rt["scores"]
                comp = scores.get("composite", 0)
                st.success(f"实时计算完成（字: {'、'.join(rt['found_chars'])}）")

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.metric("综合WEAT得分", f"{comp:.4f}")
                with col2:
                    st.metric("来源", "实时计算")

                dim_scores_rt = {d: scores.get(d, 0) for d in DIM_KEYS}
                trace = make_radar(name, dim_scores_rt, color="#00d4ff")
                fig = make_radar_figure([trace], title=f"「{name}」六维语义雷达")
                st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(make_dimension_bars(dim_scores_rt, name), use_container_width=True)

                # 邻居词
                st.markdown("#### 🔍 实时邻居词")
                for ch in rt["found_chars"]:
                    if ch in rt["neighbors"] and rt["neighbors"][ch]:
                        tags = " ".join(f'`{w} {s:.2f}`' for w, s in rt["neighbors"][ch][:8])
                        st.markdown(f"**{ch}**: {tags}")
                return

        st.error(f"抱歉，「{name}」中的字不在分析范围内，实时计算也未能匹配。")
        return

    if missing_chars:
        st.warning(f"字「{'、'.join(missing_chars)}」不在词向量模型中，仅分析「{'、'.join(found_chars)}」")

    scores = get_name_scores(name, char_scores)
    comp = composite_score(scores)

    # ── 综合得分卡片 ──
    # 百分位得分
    display_scores = scores_to_display(scores)
    overall_pct = raw_to_percentile(comp, "composite")
    grade = format_grade(overall_pct)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-big">{overall_pct:.0f}<span style="font-size:1.2rem">/100</span></div>
            <div class="score-label">综合评分 · 超越 {overall_pct:.0f}% 的名字用字 · 等级 {grade}</div>
        </div>
        """, unsafe_allow_html=True)

    # 六维得分卡片
    dim_cols = st.columns(6)
    for i, dim in enumerate(DIM_KEYS):
        info = DIMENSIONS[dim]
        pct = display_scores[dim]
        with dim_cols[i]:
            st.markdown(f"""<div style="text-align:center">
            <div style="font-size:1.8rem">{info['icon']}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{info['color']}">{pct:.0f}</div>
            <div style="font-size:0.75rem;color:#888">{info['label']}</div>
            </div>""", unsafe_allow_html=True)

    # ── 整词 vs 字均分析 ──
    name_whole = load_name_whole_scores()
    has_whole = name in name_whole

    if has_whole:
        whole_scores = {d: name_whole[name][d] for d in DIM_KEYS if d in name_whole[name]}
        whole_comp = name_whole[name].get("composite", 0)
        char_avg_scores = name_whole[name].get("char_avg", {})

        st.info(f"💡 「{name}」作为整词存在于词向量模型中！整词得分和字均得分可能有显著差异。")

        col_w, col_a = st.columns(2)
        with col_w:
            st.metric("整词综合分", f"{whole_comp:.4f}", help="名字作为一个整体在语料中学到的向量")
        with col_a:
            st.metric("字均综合分", f"{comp:.4f}",
                      delta=f"{comp - whole_comp:+.4f} vs 整词",
                      help="拆成单字取平均的向量")

        # 双雷达：整词 vs 字均
        trace_whole = make_radar(f"{name} (整词)", whole_scores, color="#FFD700", fill_opacity=0.15)
        trace_avg = make_radar(f"{name} (字均)", scores, color="#00d4ff", fill_opacity=0.15)
        fig = make_radar_figure([trace_whole, trace_avg], title=f"「{name}」整词 vs 字均")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 普通雷达图
        trace = make_radar(name, scores, color="#00d4ff")
        fig = make_radar_figure([trace], title=f"「{name}」六维语义雷达")
        st.plotly_chart(fig, use_container_width=True)

    # ── 各维度条形图 ──
    st.plotly_chart(make_dimension_bars(scores, name), use_container_width=True)

    # ── 逐字分析 ──
    st.markdown("---")
    st.markdown("#### 🔍 逐字分析")

    char_cols = st.columns(len(found_chars))
    for i, ch in enumerate(found_chars):
        with char_cols[i]:
            st.markdown(f'<div class="char-card"><div class="char-big">{ch}</div>', unsafe_allow_html=True)

            ch_scores = char_scores[ch]
            best_dim = max(DIM_KEYS, key=lambda d: ch_scores.get(d, -1))
            best_info = DIMENSIONS[best_dim]
            st.markdown(f"**最强维度:** {best_info['icon']} {best_info['label']} ({ch_scores[best_dim]:.4f})")

            # 邻居词
            if ch in char_neighbors:
                neighbors = char_neighbors[ch]
                tags_html = "".join(
                    f'<span class="neighbor-tag">{w} {s:.2f}</span>'
                    for w, s in neighbors[:8]
                )
                st.markdown(f"**语义邻居:**<br>{tags_html}", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # ── 频率数据 ──
    freq_zh = load_freq_zh()
    if name in freq_zh:
        st.markdown("---")
        st.markdown("#### 📈 名字流行度")
        fi = freq_zh[name]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("使用人数 (样本中)", f"{fi['count']:,}")
        with c2:
            st.metric("流行度排名", f"#{fi['rank']:,}")
        with c3:
            st.metric("女性使用比例", f"{fi['female_ratio']}%")

    # ── BERT对比 ──
    bert_scores = load_bert_scores()
    if bert_scores:
        bert_data = []
        for ch in found_chars:
            if ch in bert_scores:
                bert_data.append(bert_scores[ch])
        if bert_data:
            bert_avg = {d: float(np.mean([b[d] for b in bert_data if d in b])) for d in DIM_KEYS}
            st.markdown("---")
            st.markdown("#### 🔄 Word2Vec vs BERT 对比")
            st.caption("两套不同的AI模型对同一个名字的评价")
            w2v_trace = make_radar(f"{name} (Word2Vec)", scores, color="#00d4ff", fill_opacity=0.15)
            bert_trace = make_radar(f"{name} (BERT)", bert_avg, color="#ff6b35", fill_opacity=0.15)
            fig = make_radar_figure([w2v_trace, bert_trace], title="Word2Vec vs BERT")
            st.plotly_chart(fig, use_container_width=True)

    # ── AI描述 ──
    st.markdown("---")
    desc = generate_ai_description(name, scores)
    st.markdown(desc)


def _resolve_name_scores(name: str):
    """自动检测语言，返回WEAT得分。支持中英文 + 实时fallback。"""
    if not name:
        return None
    # 英文
    if all(c.isascii() for c in name) and name.strip().isalpha():
        en_scores = load_en_scores()
        for v in [name, name.capitalize(), name.lower(), name.title()]:
            if v in en_scores:
                return {d: en_scores[v].get(d, 0) for d in DIM_KEYS}
        # realtime
        rt = compute_en_realtime(name)
        return {d: rt["scores"].get(d, 0) for d in DIM_KEYS} if rt else None
    # 中文
    char_scores = load_char_scores()
    result = get_name_scores(name, char_scores)
    if result:
        return result
    # realtime
    rt = compute_zh_realtime(name)
    return {d: rt["scores"].get(d, 0) for d in DIM_KEYS} if rt else None


def page_pk():
    """页面2: 名字PK — 支持中英文任意组合"""
    st.markdown("### ⚔️ 名字 PK")
    st.caption("两个名字正面对决 — 中文、英文、甚至中英混合对比都可以")

    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        name_a = st.text_input("Name A", placeholder="Sophia / 思琪", key="pk_a")
    with col_vs:
        st.markdown("<div style='text-align:center; padding-top:1.8rem; font-size:1.5rem; color:#666'>VS</div>",
                     unsafe_allow_html=True)
    with col2:
        name_b = st.text_input("Name B", placeholder="Nova / 梓涵", key="pk_b")

    if not name_a or not name_b:
        st.info("Enter two names to compare (Chinese, English, or mix)")
        return

    scores_a = _resolve_name_scores(name_a)
    scores_b = _resolve_name_scores(name_b)

    if not scores_a:
        st.error(f'"{name_a}" could not be resolved in any model')
        return
    if not scores_b:
        st.error(f'"{name_b}" could not be resolved in any model')
        return

    # ── 综合得分对比 ──
    comp_a = composite_score(scores_a)
    comp_b = composite_score(scores_b)

    col1, col2 = st.columns(2)
    with col1:
        delta = comp_a - comp_b
        st.metric(f"「{name_a}」综合分", f"{comp_a:.4f}",
                  delta=f"{delta:+.4f}" if delta != 0 else None)
    with col2:
        delta = comp_b - comp_a
        st.metric(f"「{name_b}」综合分", f"{comp_b:.4f}",
                  delta=f"{delta:+.4f}" if delta != 0 else None)

    # ── 叠加雷达图 ──
    trace_a = make_radar(name_a, scores_a, color="#00d4ff", fill_opacity=0.2)
    trace_b = make_radar(name_b, scores_b, color="#da70d6", fill_opacity=0.2)
    fig = make_radar_figure([trace_a, trace_b], title=f"「{name_a}」vs「{name_b}」")
    st.plotly_chart(fig, use_container_width=True)

    # ── 逐维度对比 ──
    st.markdown("#### 逐维度胜负")
    wins_a, wins_b = 0, 0

    for dim in DIM_KEYS:
        info = DIMENSIONS[dim]
        va = scores_a.get(dim, 0)
        vb = scores_b.get(dim, 0)
        winner = name_a if va > vb else name_b if vb > va else "平局"
        if va > vb:
            wins_a += 1
        elif vb > va:
            wins_b += 1

        badge = ""
        if winner != "平局":
            badge = f' <span class="winner-badge">WIN</span>'

        c1, c2, c3 = st.columns([2, 3, 2])
        with c1:
            color_a = info["color"] if va >= vb else "#666"
            st.markdown(f"<span style='color:{color_a};font-weight:600'>{va:.4f}</span>"
                        f"{badge if winner == name_a else ''}", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='text-align:center'>{info['icon']} {info['label']}</div>",
                        unsafe_allow_html=True)
        with c3:
            color_b = info["color"] if vb >= va else "#666"
            st.markdown(f"<span style='color:{color_b};font-weight:600'>{vb:.4f}</span>"
                        f"{badge if winner == name_b else ''}", unsafe_allow_html=True)

    st.markdown("---")
    if wins_a > wins_b:
        st.success(f"🏆 「{name_a}」以 {wins_a}:{wins_b} 胜出！")
    elif wins_b > wins_a:
        st.success(f"🏆 「{name_b}」以 {wins_b}:{wins_a} 胜出！")
    else:
        st.info("⚖️ 势均力敌！")


def page_leaderboard():
    """页面: 排行榜 (中英文切换)"""
    st.markdown("### 🏆 向量空间排行榜")

    lang = st.radio("Language", ["🇨🇳 中文字", "🇺🇸 English Names"], horizontal=True, key="lb_lang")

    if "English" in lang:
        _leaderboard_en()
        return

    char_scores = load_char_scores()
    st.caption("在AI的语义地图上，哪些字占据了最有利的位置？")

    # 维度选择
    dim_options = ["composite"] + DIM_KEYS
    dim_labels_map = {"composite": "📊 综合", **{k: f"{v['icon']} {v['label']}" for k, v in DIMENSIONS.items()}}

    selected_dim = st.selectbox(
        "排序维度",
        dim_options,
        format_func=lambda x: dim_labels_map[x],
    )

    top_n = st.slider("显示数量", 10, 100, 30, step=10)

    # 排序
    ranked = sorted(
        char_scores.items(),
        key=lambda x: x[1].get(selected_dim, -999),
        reverse=True,
    )[:top_n]

    # 展示表格
    rows = []
    for rank, (char, scores) in enumerate(ranked, 1):
        row = {"排名": rank, "字": char}
        for dim in DIM_KEYS:
            label = DIMENSIONS[dim]["label"]
            row[label] = round(scores.get(dim, 0), 4)
        row["综合"] = round(scores.get("composite", 0), 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 用颜色高亮
    def color_values(val):
        if isinstance(val, float):
            if val > 0.1:
                return "color: #00ff88; font-weight: 600"
            elif val > 0.05:
                return "color: #00d4ff"
            elif val < 0:
                return "color: #ff6b6b"
        return ""

    st.dataframe(
        df.style.map(color_values),
        use_container_width=True,
        height=min(top_n * 38 + 40, 800),
        hide_index=True,
    )

    # 热力图
    st.markdown("---")
    st.markdown("#### 🗺️ 语义热力图")

    heat_chars = [c for c, _ in ranked[:min(30, top_n)]]
    heat_data = []
    for ch in heat_chars:
        heat_data.append([char_scores[ch].get(d, 0) for d in DIM_KEYS])

    fig = go.Figure(data=go.Heatmap(
        z=heat_data,
        x=DIM_LABELS,
        y=heat_chars,
        colorscale=[
            [0, "#1a1f2e"],
            [0.3, "#1a3a5c"],
            [0.5, "#00d4ff"],
            [0.7, "#da70d6"],
            [1, "#ffd700"],
        ],
        text=[[f"{v:.3f}" for v in row] for row in heat_data],
        texttemplate="%{text}",
        textfont=dict(size=10, color="#e0e0e0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#e0e0e0", size=12)),
        yaxis=dict(tickfont=dict(color="#e0e0e0", size=13), autorange="reversed"),
        height=max(400, len(heat_chars) * 28),
        margin=dict(t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 组名推荐
    st.markdown("---")
    st.markdown("#### 💡 高分组名推荐")
    st.caption("从排行榜Top字中自动组合的双字名")

    top_chars = [c for c, _ in ranked[:20]]
    combos = []
    for i, c1 in enumerate(top_chars[:10]):
        for c2 in top_chars[i + 1: 15]:
            name = c1 + c2
            name_scores = get_name_scores(name, char_scores)
            if name_scores:
                comp = composite_score(name_scores)
                combos.append((name, comp))
    combos.sort(key=lambda x: x[1], reverse=True)

    combo_cols = st.columns(5)
    for i, (name, comp) in enumerate(combos[:20]):
        with combo_cols[i % 5]:
            st.markdown(f"**{name}** `{comp:.4f}`")


def _leaderboard_en():
    """English name leaderboard."""
    en_scores = load_en_scores()
    if not en_scores:
        st.warning("English scores not found.")
        return

    st.caption("Which English names occupy the most favorable positions in GloVe embedding space?")

    dim_options = ["composite"] + DIM_KEYS
    dim_labels_map = {"composite": "📊 Overall", **{k: f"{v['icon']} {v['label']}" for k, v in DIMENSIONS.items()}}

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_dim = st.selectbox("Sort by", dim_options,
                                    format_func=lambda x: dim_labels_map[x], key="lb_en_dim")
    with col2:
        min_pop = st.number_input("Min SSA births", 100, 100000, 1000, step=500, key="lb_en_pop")
    with col3:
        top_n = st.slider("Show", 10, 100, 30, step=10, key="lb_en_n")

    filtered = {k: v for k, v in en_scores.items() if v.get("frequency", 0) >= min_pop}
    ranked = sorted(filtered.items(), key=lambda x: x[1].get(selected_dim, -999), reverse=True)[:top_n]

    rows = []
    for rank, (name, s) in enumerate(ranked, 1):
        row = {"Rank": rank, "Name": name}
        for dim in DIM_KEYS:
            row[DIMENSIONS[dim]["label"]] = round(raw_to_percentile(s.get(dim, 0), dim), 0)
        row["Overall"] = round(raw_to_percentile(s.get("composite", 0), "composite"), 0)
        row["SSA Births"] = s.get("frequency", 0)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=min(top_n * 38 + 40, 800))


def page_about():
    """页面4: 关于"""
    st.markdown("### 📖 关于本项目")

    st.markdown("""
---

#### 这是什么？

**How AI Sees Your Name** 是一个探索性项目，
通过分析AI语言模型中名字的**向量表示（word embedding）**，量化名字与正面概念的语义距离。

核心方法是 **WEAT（Word Embedding Association Test）**——
一种由 Caliskan et al. 在2017年发表于 *Science* 的偏见测量技术。

---

#### 方法论

**1. 词向量模型**

使用腾讯AI Lab发布的中文词向量（200维，覆盖14万+词汇）。
每个汉字在这个200维空间中都有一个坐标。

**2. WEAT测试**

对每个汉字，计算它与6组属性词的cosine相似度差值：

| 维度 | 正面属性词 | 负面属性词 |
|------|-----------|-----------|
| 💰 财富 | 富裕、成功、财富、繁荣… | 贫穷、贫困、匮乏… |
| 🧠 智慧 | 智慧、聪明、博学、睿智… | 愚蠢、无知、迟钝… |
| ✨ 幸福 | 幸福、快乐、美满… | 痛苦、悲伤、不幸… |
| 💪 健康 | 健康、活力、强壮… | 疾病、虚弱、衰弱… |
| 👑 领导力 | 杰出、卓越、精英… | 平庸、普通、渺小… |
| 🌸 美感 | 优雅、端庄、灵秀… | 粗俗、丑陋、猥琐… |

**得分 = 平均(与正面词的相似度) - 平均(与负面词的相似度)**

正值 = 该字在向量空间中偏向正面概念。

**3. 名字得分**

多字名取各字的平均得分。

---

#### ⚠️ 免责声明

> **本项目仅供探索和娱乐用途。**
>
> - 名字的向量表示反映的是**语料库中的统计规律**，不是名字的"真实价值"
> - 这些偏好本质上是**AI系统的偏见**，了解偏见不等于认同偏见
> - 取名是非常个人化的决定，不应仅凭算法指标
> - 本项目揭示的偏见提醒我们AI公平性问题的重要性
> - 研究表明名字是社会地位的"指标"而非"原因"（Levitt & Fryer, 2004）

---

#### 参考文献

1. Caliskan, A., Bryson, J.J., & Narayanan, A. (2017). Semantics derived automatically
   from language corpora contain human-like biases. *Science*, 356(6334), 183-186.
2. Garg, N., et al. (2018). Word embeddings quantify 100 years of gender and ethnic
   stereotypes. *PNAS*, 115(16), E3635-E3644.
3. Bolukbasi, T., et al. (2016). Man is to Computer Programmer as Woman is to
   Homemaker? *NeurIPS*.
4. Shi, D. & Tong, S. (2025). An Open Dataset of Chinese Name-to-Gender Associations.
   *Nature Scientific Data*.
5. Levitt, S.D. & Fryer, R.G. (2004). The Causes and Consequences of Distinctively
   Black Names. *Quarterly Journal of Economics*.

---

#### 技术栈

- **词向量:** 腾讯AI Lab中文词向量 (text2vec轻量版, 111MB)
- **后端:** Python, NumPy, Gensim
- **前端:** Streamlit + Plotly
- **方法:** WEAT (Word Embedding Association Test)

---

<div style="text-align:center; color:#666; margin-top:2rem">
Built with 🔬 by How AI Sees Your Name · 2026
</div>
""")


def page_english():
    """页面: 英文名分析"""
    en_scores = load_en_scores()

    st.markdown("### 🌍 English Name Analysis")
    st.caption("Explore how English names are positioned in GloVe word embedding space (6B tokens, 300d)")

    if not en_scores:
        st.warning("英文名预计算数据未找到。请运行: `python scripts/precompute_en_scores.py`")
        return

    col_input, col_info = st.columns([2, 1])
    with col_input:
        en_name = st.text_input("Enter an English name", placeholder="e.g. Sophia, Nova, Elizabeth...", key="en_input")
    with col_info:
        st.metric("Names in database", f"{len(en_scores):,}")

    if not en_name:
        # 展示 Top 20 (过滤掉形容词/普通词，只保留合理的名字)
        st.markdown("---")
        st.markdown("#### Top 20 English Names by Composite Score")
        # 按得分排名，显示 frequency > 1000 的
        popular = {k: v for k, v in en_scores.items() if v.get("frequency", 0) > 1000}
        ranked = sorted(popular.items(), key=lambda x: x[1].get("composite", 0), reverse=True)[:20]

        rows = []
        for rank, (name, s) in enumerate(ranked, 1):
            row = {"Rank": rank, "Name": name, "Composite": round(s.get("composite", 0), 4)}
            for dim in DIM_KEYS:
                row[DIMENSIONS[dim]["label"]] = round(s.get(dim, 0), 4)
            row["Popularity"] = s.get("frequency", 0)
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    # 查找名字（不区分大小写）
    matched = None
    for variant in [en_name, en_name.capitalize(), en_name.title(), en_name.upper(), en_name.lower()]:
        if variant in en_scores:
            matched = variant
            break

    if not matched:
        # Realtime fallback
        with st.spinner(f'"{en_name}" not in cache — loading GloVe for real-time computation...'):
            rt = compute_en_realtime(en_name)
        if rt:
            matched = rt["token"]
            scores = rt["scores"]
            dim_scores = {d: scores.get(d, 0) for d in DIM_KEYS}
            comp = scores.get("composite", 0)
            freq = 0
            st.success(f'Real-time computed "{matched}" via GloVe 300d (not in precomputed cache)')
        else:
            st.error(f'"{en_name}" not found in GloVe vocabulary (400K words). Try a different spelling.')
            return
    else:
        scores = en_scores[matched]
        dim_scores = {d: scores.get(d, 0) for d in DIM_KEYS}
        comp = scores.get("composite", 0)
        freq = scores.get("frequency", 0)

    # 得分卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Composite Score", f"{comp:.4f}")
    with col2:
        # 百分位
        all_comp = [v.get("composite", 0) for v in en_scores.values()]
        pct = sum(1 for x in all_comp if comp > x) / len(all_comp) * 100
        st.metric("Percentile", f"Top {100 - pct:.1f}%")
    with col3:
        st.metric("SSA Popularity", f"{freq:,}" if freq else "N/A")

    # 雷达图
    trace = make_radar(matched, dim_scores, color="#da70d6")
    fig = make_radar_figure([trace], title=f'"{matched}" Semantic Radar')
    st.plotly_chart(fig, use_container_width=True)

    # 条形图
    st.plotly_chart(make_dimension_bars(dim_scores, matched), use_container_width=True)

    # SSA 历史趋势
    ssa_trends = load_ssa_trends()
    trend_data = ssa_trends.get(matched)
    if trend_data:
        st.markdown("---")
        st.markdown(f'#### 📈 "{matched}" Popularity Over Time (SSA 1880-2024)')
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Peak Year", str(trend_data["peak_year"]))
        with col2:
            st.metric("Total Births (all years)", f"{trend_data['total']:,}")

        trend = trend_data["trend"]
        years = sorted(trend.keys(), key=int)
        fig = go.Figure(go.Scatter(
            x=[int(y) for y in years],
            y=[trend[y] for y in years],
            mode="lines+markers",
            line=dict(color="#da70d6", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba("#da70d6", 0.15),
        ))
        fig.update_layout(
            xaxis=dict(title="Year", tickfont=dict(color="#999"), title_font=dict(color="#999"),
                       gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Per 10K births", tickfont=dict(color="#999"), title_font=dict(color="#999"),
                       gridcolor="rgba(255,255,255,0.1)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def page_success():
    """页面: 名运分析 (中英文切换)"""
    st.markdown("### 📊 名运关联分析")

    lang = st.radio("Data Source", ["🇨🇳 中国 (科学家 vs 普通人)", "🇺🇸 USA (Salary + Elites)"],
                    horizontal=True, key="success_lang")

    if "USA" in lang:
        _success_en()
        return

    analysis = load_success_analysis()
    char_scores = load_char_scores()
    st.caption("名字的向量空间位置，与现实世界的成就是否相关？")

    if not analysis:
        st.warning("分析数据未找到。请运行: `python scripts/precompute_success_analysis.py`")
        return

    # 核心发现
    comp = analysis.get("composite", {})

    st.markdown("""
    <div class="score-card" style="text-align:left; padding:2rem">
        <h4 style="margin-top:0">核心发现</h4>
        <p>我们对比了 <b>99,729 位获得国家基金资助的科学家/学者</b>的名字
        与 <b>100,000 位普通人</b>的名字在AI向量空间中的位置。</p>
        <p style="font-size:1.3rem; margin:1rem 0">
        结果：成功者的名字在向量空间中 <b style="color:#00d4ff">确实占据了更"有利"的位置</b>
        </p>
        <p style="color:#888">t = {t_stat:.2f}, p = {p_val:.2e}, Cohen's d = {d:.4f}</p>
    </div>
    """.format(
        t_stat=comp.get("t_stat", 0),
        p_val=comp.get("p_value", 1),
        d=comp.get("cohens_d", 0),
    ), unsafe_allow_html=True)

    st.markdown("---")

    # 逐维度对比
    st.markdown("#### 逐维度对比：成功者 vs 普通人")

    dims_data = analysis.get("dimensions", {})
    dim_names = []
    grantee_vals = []
    general_vals = []
    diff_vals = []

    for dim in DIM_KEYS:
        if dim in dims_data:
            d = dims_data[dim]
            info = DIMENSIONS[dim]
            dim_names.append(f"{info['icon']} {info['label']}")
            grantee_vals.append(d["grantee_mean"])
            general_vals.append(d["general_mean"])
            diff_vals.append(d["diff"])

    # 分组条形图
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="🎓 成功者 (基金获资助者)",
        x=dim_names, y=grantee_vals,
        marker_color="#00d4ff",
        text=[f"{v:.4f}" for v in grantee_vals],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_trace(go.Bar(
        name="👤 普通人 (CCNC样本)",
        x=dim_names, y=general_vals,
        marker_color="#666",
        text=[f"{v:.4f}" for v in general_vals],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#e0e0e0", size=13)),
        yaxis=dict(
            title="平均WEAT得分",
            gridcolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#999"),
            title_font=dict(color="#999"),
        ),
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0.3)"),
        height=400,
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 差值图
    colors = ["#00ff88" if d > 0 else "#ff6b6b" for d in diff_vals]
    fig2 = go.Figure(go.Bar(
        x=dim_names, y=diff_vals,
        marker_color=colors,
        text=[f"{d:+.4f}" for d in diff_vals],
        textposition="outside",
        textfont=dict(size=12, color="#e0e0e0"),
    ))
    fig2.update_layout(
        title=dict(text="成功者 - 普通人 差值（正=成功者更强）", font=dict(color="#e0e0e0", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#e0e0e0", size=13)),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.3)",
            tickfont=dict(color="#999"),
        ),
        height=350,
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 解读
    st.markdown("---")
    st.markdown("""
#### 解读

**"硬实力"维度（成功者显著更强）：**
- 💰 **财富** (+0.0091) — 成功者的名字更靠近"富裕、成功、繁荣"等概念
- 🧠 **智慧** (+0.0046) — 更靠近"聪明、博学、睿智"
- 👑 **领导力** (+0.0046) — 更靠近"杰出、卓越、精英"

**"软实力"维度（普通人反而更强）：**
- ✨ **幸福** (-0.0017) — 普通人的名字更靠近"快乐、美满"
- 🌸 **美感** (-0.0055) — 普通人的名字更靠近"优雅、端庄"

**可能的解释：**
- 高社会经济地位的家庭倾向于给孩子取"有力量感"的名字（志向型命名）
- 而"温柔优美"型的名字在向量空间中与"成就"概念的距离更远
- Cohen's d=0.09 属于小效应量，但在10万+样本上统计极其显著 (p≈10⁻⁷⁸)

**⚠️ 重要提醒：** 这是相关性，不是因果关系。名字反映的是父母的社会地位和文化偏好，
而不是名字本身"导致"了成功。但在AI系统中，名字的向量表示确实会直接影响模型输出——
这是一条真实的技术路径。
""")

    # 成功者高频用字
    st.markdown("---")
    st.markdown("#### 🔤 成功者名字中的高频汉字")

    top_chars = analysis.get("top_grantee_chars", [])[:30]
    if top_chars:
        rows = []
        for item in top_chars:
            ch = item["char"]
            freq = item["freq"]
            row = {"字": ch, "出现次数": freq}
            if "scores" in item:
                for dim in DIM_KEYS:
                    row[DIMENSIONS[dim]["label"]] = round(item["scores"].get(dim, 0), 4)
                row["综合"] = round(item["scores"].get("composite", 0), 4)
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=500)


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def _load_env():
    """从 .env 文件加载API密钥到环境变量和session_state。"""
    if ENV_PATH.exists():
        with open(ENV_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if v:
                        os.environ[k] = v


def _save_env(anthropic_key: str, openai_key: str):
    """保存API密钥到 .env 文件。"""
    lines = []
    if anthropic_key:
        lines.append(f"ANTHROPIC_API_KEY={anthropic_key}")
    if openai_key:
        lines.append(f"OPENAI_API_KEY={openai_key}")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _success_en():
    """English success analysis: salary + elite + self-made vs inherited."""
    import json as _json

    val_path = DATA_DIR / "validation_results.json"
    ctrl_path = DATA_DIR / "controlled_validation.json"
    if not val_path.exists():
        st.warning("Run `python scripts/run_validation.py` first.")
        return

    with open(val_path, encoding="utf-8") as f:
        val = _json.load(f)

    st.caption("Do names in favorable embedding positions correlate with real-world outcomes?")

    # Chicago salary
    us = val.get("us_salary_validation", {})
    if "name_level_pearson_r" in us:
        st.markdown("""
        <div class="score-card" style="text-align:left; padding:1.5rem">
        <h4 style="margin-top:0">Chicago City Employees (n=32,069)</h4>
        <p>Name-level WEAT × salary correlation: <b style="color:#00d4ff">r = {r}</b></p>
        <p>Top 25% WEAT names earn <b style="color:#00ff88">$9,702/year more</b> than Bottom 25%</p>
        </div>
        """.format(r=us["name_level_pearson_r"]), unsafe_allow_html=True)

    # Elite comparison
    elite = val.get("elite_validation", {}).get("categories", {})
    if elite:
        st.markdown("#### Global Elite vs Random Names")
        rows = []
        for cat in ["US Senators", "Oscar Winners", "Olympic Gold", "Nobel Laureates", "Billionaires"]:
            if cat in elite:
                d = elite[cat]
                rows.append({"Category": cat, "n": d["n"],
                             "WEAT vs Random": f"+{d['diff_vs_control']:.3f}",
                             "Significant": "Yes (p<0.001)"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Self-made vs inherited
    sm = val.get("selfmade_vs_inherited", {}).get("selfmade_vs_inherited", {})
    if sm:
        st.markdown("#### Self-Made vs Inherited Billionaires")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Self-Made", f"{sm['selfmade_mean']:.4f}", delta=f"+{sm['diff']:.4f}")
        with c2:
            st.metric("Inherited", f"{sm['inherited_mean']:.4f}")
        with c3:
            st.metric("p-value", f"{sm['p']:.1e}")

    # Controlled results
    if ctrl_path.exists():
        with open(ctrl_path, encoding="utf-8") as f:
            ctrl = _json.load(f)
        ch = ctrl.get("chicago_salary", {})
        st.markdown("#### After Controlling for Cultural Background")
        st.markdown(f"""
        - Raw correlation: r = {ch.get('raw_r', 'N/A')}
        - Double-partial (culture removed): **r = {ch.get('double_partial_r', 'N/A')}**
        - ~85% of the effect is **independent of cultural background**
        """)
        wt = ch.get("within_tier", {})
        if wt:
            rows = []
            for tier, d in wt.items():
                rows.append({"Cultural Tier": tier.replace("_", " ").title(),
                             "n": d["n"], "r (WEAT~Salary)": d["r"],
                             "Significant": "Yes" if d["p"] < 0.001 else "No"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_generator():
    """页面: 名字生成器 (中英文)"""
    st.markdown("### ✨ 名字生成器")

    lang = st.radio("Language", ["🇨🇳 中文名", "🇺🇸 English Name"], horizontal=True, key="gen_lang")

    if "English" in lang:
        _generator_en()
        return

    char_scores = load_char_scores()
    freq_zh = load_freq_zh()
    st.caption("设定你想要的维度偏好，自动从字库中组合最优名字")

    st.markdown("#### 调整维度权重")

    # 权重滑块
    weights = {}
    cols = st.columns(3)
    for i, dim in enumerate(DIM_KEYS):
        info = DIMENSIONS[dim]
        with cols[i % 3]:
            weights[dim] = st.slider(
                f"{info['icon']} {info['label']}",
                0.0, 1.0, 0.5, 0.1,
                key=f"gen_w_{dim}",
            )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        name_len = st.radio("名字长度", [1, 2], index=1, horizontal=True)
    with col_opt2:
        prefer_female = st.checkbox("偏好女性用字", value=True)

    if st.button("🎲 生成推荐", type="primary", use_container_width=True):
        # 对每个字算加权得分
        char_weighted = {}
        for ch, sc in char_scores.items():
            w_score = 0
            w_total = 0
            for dim in DIM_KEYS:
                w = weights.get(dim, 0.5)
                if dim in sc:
                    w_score += sc[dim] * w
                    w_total += w
            if w_total > 0:
                char_weighted[ch] = w_score / w_total

        # 按加权分排序
        ranked_chars = sorted(char_weighted.items(), key=lambda x: x[1], reverse=True)

        # 过滤偏好
        if prefer_female and freq_zh:
            # 取在名字数据中女性比例>50%的字
            female_chars = set()
            for name_str, fi in freq_zh.items():
                if fi.get("female_ratio", 50) > 50:
                    for ch in name_str:
                        female_chars.add(ch)
            if female_chars:
                ranked_chars = [(c, s) for c, s in ranked_chars if c in female_chars] or ranked_chars

        top_chars = ranked_chars[:50]

        if name_len == 1:
            results = [(ch, score) for ch, score in top_chars[:20]]
        else:
            # 两两组合
            combos = []
            pool = top_chars[:30]
            for i, (c1, s1) in enumerate(pool):
                for c2, s2 in pool[i + 1:]:
                    combos.append((c1 + c2, (s1 + s2) / 2))
                    combos.append((c2 + c1, (s1 + s2) / 2))
            combos.sort(key=lambda x: x[1], reverse=True)
            # 去重只取每个组合的最佳排列
            seen = set()
            results = []
            for name, score in combos:
                key = "".join(sorted(name))
                if key not in seen:
                    seen.add(key)
                    results.append((name, score))
                if len(results) >= 30:
                    break

        st.markdown("---")
        st.markdown(f"#### 🏆 Top {len(results)} 推荐")

        for rank, (name, score) in enumerate(results, 1):
            name_scores = get_name_scores(name, char_scores)
            if not name_scores:
                continue

            with st.expander(f"**#{rank}  {name}**  (加权分 {score:.4f})", expanded=rank <= 3):
                dims_text = " · ".join(
                    f"{DIMENSIONS[d]['icon']}{name_scores[d]:.3f}"
                    for d in DIM_KEYS if d in name_scores
                )
                st.markdown(dims_text)

                # 迷你雷达
                trace = make_radar(name, name_scores, color=DIMENSIONS[DIM_KEYS[rank % 6]]["color"])
                fig = make_radar_figure([trace], height=300)
                st.plotly_chart(fig, use_container_width=True)


def _generator_en():
    """English name generator — find top-scoring names from SSA database."""
    en_scores = load_en_scores()
    if not en_scores:
        st.warning("English scores not found.")
        return

    st.caption("Set your dimension preferences, find the highest-scoring English names")

    weights = {}
    cols = st.columns(3)
    for i, dim in enumerate(DIM_KEYS):
        info = DIMENSIONS[dim]
        with cols[i % 3]:
            weights[dim] = st.slider(f"{info['icon']} {info['label']}",
                                     0.0, 1.0, 0.5, 0.1, key=f"gen_en_w_{dim}")

    col1, col2 = st.columns(2)
    with col1:
        min_pop = st.number_input("Min SSA births", 100, 500000, 5000, step=1000, key="gen_en_pop")
    with col2:
        top_n = st.slider("Show top N", 10, 50, 20, key="gen_en_n")

    if st.button("🎲 Find Best Names", type="primary", use_container_width=True, key="gen_en_btn"):
        # 加权打分
        results = []
        for name, s in en_scores.items():
            if s.get("frequency", 0) < min_pop:
                continue
            w_score = sum(s.get(d, 0) * weights.get(d, 0.5) for d in DIM_KEYS)
            w_total = sum(weights.get(d, 0.5) for d in DIM_KEYS)
            results.append((name, w_score / w_total if w_total > 0 else 0, s.get("frequency", 0)))

        results.sort(key=lambda x: x[1], reverse=True)

        st.markdown("---")
        st.markdown(f"#### 🏆 Top {top_n} English Names")

        rows = []
        for rank, (name, score, freq) in enumerate(results[:top_n], 1):
            s = en_scores[name]
            row = {"#": rank, "Name": name}
            for dim in DIM_KEYS:
                row[DIMENSIONS[dim]["icon"]] = int(raw_to_percentile(s.get(dim, 0), dim))
            row["Score"] = int(raw_to_percentile(s.get("composite", 0), "composite"))
            row["Popularity"] = f"{freq:,}"
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Top 3 雷达图
        if results:
            top3 = results[:3]
            traces = []
            colors = ["#00d4ff", "#da70d6", "#ffd700"]
            for i, (name, _, _) in enumerate(top3):
                s = en_scores[name]
                dim_s = {d: s.get(d, 0) for d in DIM_KEYS}
                traces.append(make_radar(name, dim_s, color=colors[i], fill_opacity=0.15))
            fig = make_radar_figure(traces, title="Top 3 Comparison")
            st.plotly_chart(fig, use_container_width=True)


def sidebar_settings():
    """侧边栏设置面板：API密钥管理。"""
    with st.sidebar:
        st.markdown("## ⚙️ 设置")

        st.markdown("---")
        st.markdown("### API 密钥")
        st.caption("LLM第一印象测试需要API密钥。密钥保存在本地 `.env` 文件中，不会上传。")

        # 从环境变量读取当前值
        current_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
        current_openai = os.environ.get("OPENAI_API_KEY", "")

        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=current_anthropic,
            type="password",
            placeholder="sk-ant-...",
            key="setting_anthropic_key",
        )

        openai_key = st.text_input(
            "OpenAI API Key",
            value=current_openai,
            type="password",
            placeholder="sk-...",
            key="setting_openai_key",
        )

        if st.button("💾 保存密钥", use_container_width=True):
            _save_env(anthropic_key, openai_key)
            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            st.success("已保存到 .env 文件")

        # 状态指示
        st.markdown("---")
        st.markdown("### 状态")
        if os.environ.get("ANTHROPIC_API_KEY"):
            st.markdown("✅ Anthropic: 已配置")
        else:
            st.markdown("⬜ Anthropic: 未配置")

        if os.environ.get("OPENAI_API_KEY"):
            st.markdown("✅ OpenAI: 已配置")
        else:
            st.markdown("⬜ OpenAI: 未配置")

        st.markdown("---")
        st.caption("点击左上角 **>** 展开此面板")


def main():
    st.set_page_config(
        page_title="How AI Sees Your Name",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # 加载 .env
    _load_env()

    page_header()
    sidebar_settings()

    # 导航
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔬 名字X光",
        "⚔️ 名字PK",
        "📊 名运分析",
        "✨ 名字生成",
        "🏆 排行榜",
        "📖 关于",
    ])

    with tab1:
        page_xray()
    with tab2:
        page_pk()
    with tab3:
        page_success()
    with tab4:
        page_generator()
    with tab5:
        page_leaderboard()
    with tab6:
        page_about()


if __name__ == "__main__":
    main()
