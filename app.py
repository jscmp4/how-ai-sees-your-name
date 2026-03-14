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
) -> go.Scatterpolar:
    """生成单个名字的雷达图trace。"""
    values = []
    for dim in DIM_KEYS:
        if all_scores:
            all_vals = [s.get(dim, 0) for s in all_scores.values() if dim in s]
            values.append(normalize_score(scores.get(dim, 0), all_vals))
        else:
            # 使用绝对值映射：0 → 0分，0.2 → 100分
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


def make_dimension_bars(scores: dict, name: str = "") -> go.Figure:
    """各维度得分水平条形图。"""
    dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    labels = [DIMENSIONS[d]["icon"] + " " + DIMENSIONS[d]["label"] for d, _ in dims]
    values = [v for _, v in dims]
    colors = [DIMENSIONS[d]["color"] for d, _ in dims]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e0e0e0", size=12),
    ))
    fig.update_layout(
        xaxis=dict(
            title="WEAT得分（正面关联 - 负面关联）",
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
    st.caption("输入任意中文名字，透视它在AI向量空间中的语义坐标")

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

    # 检查字符覆盖
    found_chars = [ch for ch in name if ch in char_scores]
    missing_chars = [ch for ch in name if ch not in char_scores]

    if not found_chars:
        st.error(f"抱歉，「{name}」中的字不在我们的分析范围内。请尝试其他名字。")
        return

    if missing_chars:
        st.warning(f"字「{'、'.join(missing_chars)}」不在词向量模型中，仅分析「{'、'.join(found_chars)}」")

    scores = get_name_scores(name, char_scores)
    comp = composite_score(scores)

    # ── 综合得分卡片 ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 百分位排名
        all_composites = []
        for s in char_scores.values():
            all_composites.append(s.get("composite", 0))
        percentile = sum(1 for x in all_composites if comp > x) / len(all_composites) * 100

        st.markdown(f"""
        <div class="score-card">
            <div class="score-big">{comp:.4f}</div>
            <div class="score-label">综合WEAT得分 · 超越 {percentile:.0f}% 的汉字</div>
        </div>
        """, unsafe_allow_html=True)

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


def page_pk():
    """页面2: 名字PK"""
    char_scores = load_char_scores()

    st.markdown("### ⚔️ 名字 PK")
    st.caption("两个名字正面对决，看谁在向量空间中更占优势")

    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        name_a = st.text_input("名字 A", placeholder="例：思琪", key="pk_a")
    with col_vs:
        st.markdown("<div style='text-align:center; padding-top:1.8rem; font-size:1.5rem; color:#666'>VS</div>",
                     unsafe_allow_html=True)
    with col2:
        name_b = st.text_input("名字 B", placeholder="例：梓涵", key="pk_b")

    if not name_a or not name_b:
        st.info("请输入两个名字开始对比")
        return

    scores_a = get_name_scores(name_a, char_scores)
    scores_b = get_name_scores(name_b, char_scores)

    if not scores_a:
        st.error(f"「{name_a}」中的字不在分析范围内")
        return
    if not scores_b:
        st.error(f"「{name_b}」中的字不在分析范围内")
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
    """页面3: 排行榜"""
    char_scores = load_char_scores()

    st.markdown("### 🏆 向量空间排行榜")
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
        st.error(f'"{en_name}" not found in GloVe vocabulary. Try a different spelling.')
        return

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
    """页面: 名运分析"""
    analysis = load_success_analysis()
    char_scores = load_char_scores()

    st.markdown("### 📊 名运关联分析")
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


def page_generator():
    """页面: 名字生成器"""
    char_scores = load_char_scores()
    freq_zh = load_freq_zh()

    st.markdown("### ✨ 名字生成器")
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔬 名字X光",
        "⚔️ 名字PK",
        "🌍 English",
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
        page_english()
    with tab4:
        page_success()
    with tab5:
        page_generator()
    with tab6:
        page_leaderboard()
    with tab7:
        page_about()


if __name__ == "__main__":
    main()
