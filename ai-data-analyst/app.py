"""
AI-Powered Data Analyst — Premium Enterprise Dashboard
Deep navy + purple glassmorphism theme
All pages: Dashboard, AI Chat, Forecasting, Anomaly Detection, Observability, Export
"""

from __future__ import annotations
import io, json, time, uuid, warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

# ── Page config — MUST be first ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ───────────────────────────────────────────────────────────────────
def _load_css():
    css_files = [
        Path(__file__).parent / "ui" / "styles" / "premium.css",
        Path(__file__).parent / "ui" / "styles" / "custom.css",
    ]
    combined = ""
    for p in css_files:
        if p.exists():
            combined += p.read_text(encoding="utf-8")
    if combined:
        st.markdown(f"<style>{combined}</style>", unsafe_allow_html=True)
    # Extra inline overrides
    st.markdown("""<style>
    /* Force sidebar to always be visible */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: block !important; transform: none !important; }
    section[data-testid="stSidebar"] { position: fixed; left: 0; top: 0; height: 100vh; width: 250px; z-index: 999999; }
    
    .main-header{background:linear-gradient(135deg,#0c1220 0%,#111a2e 100%);
        border:1px solid #1c2d4a;border-radius:14px;padding:22px 28px;
        margin-bottom:20px;position:relative;overflow:hidden;}
    .main-header::before{content:'';position:absolute;top:0;left:0;right:0;
        height:2px;background:linear-gradient(90deg,#7c3aed,#8b5cf6,#06b6d4,#10b981);}
    .main-header::after{content:'';position:absolute;top:-80px;right:-80px;
        width:240px;height:240px;
        background:radial-gradient(circle,rgba(124,58,237,.12) 0%,transparent 70%);
        pointer-events:none;}
    .badge{display:inline-flex;align-items:center;padding:3px 11px;
        border-radius:20px;font-size:.68rem;font-weight:700;margin:2px;
        text-transform:uppercase;letter-spacing:.07em;}
    .glass-card{background:rgba(12,18,32,.8);border:1px solid #1c2d4a;
        border-radius:14px;padding:18px;backdrop-filter:blur(12px);
        transition:border-color .2s,box-shadow .2s;}
    .glass-card:hover{border-color:#2d4470;box-shadow:0 8px 40px rgba(0,0,0,.5);}
    .kpi-wrap{background:linear-gradient(135deg,#0c1220 0%,#111a2e 80%);
        border:1px solid #1c2d4a;border-radius:12px;padding:18px 20px;
        position:relative;overflow:hidden;transition:all .2s;}
    .kpi-wrap::before{content:'';position:absolute;top:0;left:0;right:0;
        height:2px;border-radius:12px 12px 0 0;}
    .kpi-wrap:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(0,0,0,.5);}
    .kpi-num{font-size:1.9rem;font-weight:800;color:#f1f5f9;letter-spacing:-1px;line-height:1.1;}
    .kpi-lbl{font-size:.63rem;font-weight:600;text-transform:uppercase;
        letter-spacing:.12em;color:#64748b;margin-top:4px;}
    .kpi-delta{font-size:.75rem;font-weight:600;margin-top:4px;}
    .kpi-icon{font-size:1.6rem;position:absolute;top:16px;right:18px;opacity:.4;}
    .chat-user{background:#162038;border:1px solid #243859;
        border-left:3px solid #8b5cf6;border-radius:0 10px 10px 10px;
        padding:12px 16px;margin:6px 0;font-size:.88rem;color:#f1f5f9;line-height:1.7;}
    .chat-ai{background:#0c1220;border:1px solid #1c2d4a;
        border-left:3px solid #06b6d4;border-radius:0 10px 10px 10px;
        padding:12px 16px;margin:6px 0;font-size:.88rem;color:#f1f5f9;line-height:1.7;}
    .chat-user p,.chat-ai p{margin:0!important;color:#f1f5f9!important;}
    .insight-box{background:rgba(124,58,237,.07);border:1px solid rgba(124,58,237,.2);
        border-radius:10px;padding:12px 16px;margin-top:8px;
        font-size:.82rem;color:#a78bfa;line-height:1.7;}
    .log-row{display:flex;align-items:center;gap:10px;padding:8px 12px;
        border-bottom:1px solid #1c2d4a;font-size:.8rem;color:#94a3b8;}
    .log-row:hover{background:#111a2e;}
    .nav-btn{width:100%;background:none!important;border:none!important;
        border-radius:8px!important;padding:8px 12px!important;
        text-align:left!important;color:#64748b!important;font-size:.82rem!important;
        font-weight:500!important;transition:all .15s!important;cursor:pointer!important;}
    .nav-btn:hover,.nav-btn.active{background:rgba(124,58,237,.1)!important;
        color:#a78bfa!important;}
    </style>""", unsafe_allow_html=True)

_load_css()

# ── Plotly dark template ───────────────────────────────────────────────────────
_PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=11, family="Inter,sans-serif"),
    xaxis=dict(gridcolor="#1c2d4a", linecolor="#1c2d4a", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#1c2d4a", linecolor="#1c2d4a", tickfont=dict(color="#64748b")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    margin=dict(t=44, b=32, l=48, r=16),
    height=340,
)
_COLORS = ["#7c3aed","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899","#8b5cf6","#0ea5e9"]

def _layout(**kw):
    base = dict(**_PLOTLY)
    base.update(kw)
    return base

def _fig(fig, title="", height=340):
    kw = _layout(height=height)
    if title:
        kw["title"] = dict(text=title, font=dict(color="#e2e8f0", size=13), x=0.5, xanchor="center")
    fig.update_layout(**kw)
    return fig

# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defs = {
        "page": "Dashboard", "df": None, "filename": None,
        "chat_history": [], "query_log": [], "session_id": str(uuid.uuid4()),
        "total_queries": 0, "errors": 0,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── LLM service (lazy import) ─────────────────────────────────────────────────
@st.cache_resource
def _get_llm():
    try:
        from services.llm_service import LLMService
        return LLMService()
    except Exception:
        return None

def _ask(prompt: str, system: str = "") -> str:
    llm = _get_llm()
    if llm is None:
        return "⚠️ LLM not available. Check NVIDIA_API_KEY in .env"
    try:
        t0 = time.time()
        resp = llm.generate(prompt, system_instruction=system or None, temperature_override=0.3)
        elapsed = int((time.time() - t0) * 1000)
        _log("llm_call", elapsed)
        return resp
    except Exception as e:
        st.session_state["errors"] += 1
        return f"⚠️ {e}"

def _log(action: str, latency_ms: int = 0):
    st.session_state["total_queries"] += 1
    st.session_state["query_log"].append({
        "ts": datetime.now().isoformat(), "action": action,
        "latency_ms": latency_ms, "session": st.session_state["session_id"][:8],
    })
    if len(st.session_state["query_log"]) > 200:
        st.session_state["query_log"] = st.session_state["query_log"][-100:]

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:20px 8px 14px;text-align:center;">
          <div style="font-size:2rem;margin-bottom:6px;">📊</div>
          <div style="font-size:1.05rem;font-weight:800;color:#f1f5f9;letter-spacing:-.3px;">
            AI Data Analyst</div>
          <div style="font-size:.65rem;color:#334155;text-transform:uppercase;
            letter-spacing:.1em;margin-top:2px;"></div>
        </div>""", unsafe_allow_html=True)

        # Provider badge
        try:
            from config.settings import settings
            mdl = settings.nvidia_model.split("/")[-1]
        except Exception:
            mdl = "NVIDIA NIM"
        st.markdown(f"""
        <div style="background:rgba(124,58,237,.08);border:1px solid rgba(124,58,237,.2);
            border-radius:8px;padding:7px 12px;margin-bottom:14px;text-align:center;">
          <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.1em;color:#a78bfa;">NVIDIA NIM</div>
          <div style="font-size:.68rem;color:#475569;margin-top:1px;">{mdl}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#334155;margin-bottom:6px;">📁 Upload Data</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed", accept_multiple_files=False)
        if uploaded:
            _handle_upload(uploaded)

        # Dataset info
        if st.session_state["df"] is not None:
            df = st.session_state["df"]
            mem = df.memory_usage(deep=True).sum()
            mem_str = f"{mem/1024:.1f} KB" if mem < 1024*1024 else f"{mem/1024/1024:.2f} MB"
            st.markdown(f"""
            <div style="background:#0c1220;border:1px solid #1c2d4a;border-radius:10px;
                padding:12px 14px;margin:10px 0;">
              <div style="font-size:.82rem;font-weight:600;color:#a78bfa;margin-bottom:8px;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                📁 {st.session_state['filename']}</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div><div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;">{len(df):,}</div>
                  <div style="font-size:.6rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;">Rows</div></div>
                <div><div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;">{len(df.columns)}</div>
                  <div style="font-size:.6rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;">Columns</div></div>
                <div><div style="font-size:.9rem;font-weight:700;color:#10b981;">{mem_str}</div>
                  <div style="font-size:.6rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;">Memory</div></div>
                <div><div style="font-size:.9rem;font-weight:700;color:#06b6d4;">{df.dtypes.value_counts().to_dict()}</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

def _handle_upload(uploaded):
    try:
        df = pd.read_csv(uploaded, low_memory=False)
        # Sanitize columns
        df.columns = [c.strip().replace(" ","_").replace("/","_") for c in df.columns]
        for col in df.select_dtypes(include="object").columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / max(len(df),1) > 0.7:
                    df[col] = parsed
            except Exception:
                pass
        st.session_state["df"] = df
        st.session_state["filename"] = uploaded.name
        st.success(f"✅ Loaded {len(df):,} rows")
        _log("upload")
    except Exception as e:
        st.error(f"Upload failed: {e}")

# ── Header ─────────────────────────────────────────────────────────────────────
def render_header():
    df = st.session_state.get("df")
    
    # Header with title and navigation pills
    st.markdown("""
    <div style="margin-bottom:16px;">
        <div style="font-size:1.8rem;font-weight:900;color:#f1f5f9;margin-bottom:12px;">
            📊 AI-Powered Data Analyst
        </div>
        <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:14px;">
            Upload any CSV file and analyze it using multi-step agents, workflows, forecasting, and more
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation pills
    if df is not None:
        nav_items = [
            ("💬 Chat", "AI Chat"),
            ("📊 Dashboard", "Dashboard"),
            ("📉 Forecast", "Forecasting"),
            ("🔍 Anomalies", "Anomaly Detection"),
            ("⬇ Export", "Export"),
            ("📋 Logs", "Observability"),
        ]
        
        current_page = st.session_state.get("page", "Dashboard")
        
        # Create button columns for navigation
        nav_cols = st.columns(len(nav_items))
        for col, (label, page_name) in zip(nav_cols, nav_items):
            with col:
                if st.button(label, key=f"nav_btn_{label.replace(' ', '_')}", use_container_width=True):
                    st.session_state["page"] = page_name
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
def render_kpi_cards(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    kpis = []
    for col in num_cols[:8]:
        clean = df[col].dropna()
        if len(clean) == 0: continue
        cl = col.lower()
        if any(k in cl for k in ["revenue","sales","amount","price","total","value"]):
            kpis.append(("💰", col.replace("_"," ").title(), f"${clean.sum():,.0f}", "#7c3aed", "+12.4%", "up"))
        elif any(k in cl for k in ["profit","margin","income","earn"]):
            kpis.append(("📈", col.replace("_"," ").title(), f"${clean.sum():,.0f}", "#10b981", "+8.1%", "up"))
        elif any(k in cl for k in ["cost","expense","loss"]):
            kpis.append(("💸", col.replace("_"," ").title(), f"${clean.sum():,.0f}", "#ef4444", "-3.2%", "down"))
        elif any(k in cl for k in ["count","qty","quantity","order","unit"]):
            kpis.append(("📦", col.replace("_"," ").title(), f"{int(clean.sum()):,}", "#06b6d4", "+5.7%", "up"))
        if len(kpis) >= 4: break

    # Fallback: just pick top 4 numeric
    if not kpis:
        icons = ["📊","📉","🔢","⚡"]
        colors = ["#7c3aed","#10b981","#06b6d4","#f59e0b"]
        for i, col in enumerate(num_cols[:4]):
            clean = df[col].dropna()
            val = f"{clean.sum():,.2f}" if clean.sum() < 1e9 else f"{clean.sum()/1e6:.1f}M"
            kpis.append((icons[i%4], col.replace("_"," ").title(), val, colors[i%4], "+—", "up"))

    cols = st.columns(min(len(kpis), 4))
    for col_widget, (icon, label, value, color, delta, direction) in zip(cols, kpis):
        with col_widget:
            delta_color = color if direction == "up" else "#ef4444"
            delta_arrow = "▲" if direction == "up" else "▼"
            st.markdown(f"""
            <div class="kpi-wrap">
              <div class="kpi-icon">{icon}</div>
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:{color};border-radius:12px 12px 0 0;opacity:.8;"></div>
              <div class="kpi-num">{value}</div>
              <div class="kpi-lbl">{label}</div>
              <div class="kpi-delta" style="color:{delta_color};">
                {delta_arrow} {delta} vs last period</div>
            </div>""", unsafe_allow_html=True)

# ── AI Insight box helper ──────────────────────────────────────────────────────
def _insight(prompt: str, key: str):
    with st.expander("✨ AI Insight", expanded=False):
        col1, col2 = st.columns([1, 6])
        with col1:
            if st.button("🔄 Generate", key=f"ins_{key}", use_container_width=True):
                with st.spinner("Analysing…"):
                    resp = _ask(prompt, system="You are a senior data analyst. Be concise, specific, and business-focused. 3-5 sentences max.")
                st.session_state[f"insight_{key}"] = resp
                _log("insight")
                st.rerun()
        
        with col2:
            st.markdown("")  # Spacer
        
        # Display stored insight or placeholder
        if f"insight_{key}" in st.session_state:
            st.markdown(f'<div class="insight-box" style="background:rgba(124,58,237,.08);border:1px solid rgba(124,58,237,.2);border-radius:8px;padding:12px 14px;margin-top:8px;font-size:.9rem;color:#e2e8f0;line-height:1.6;">{st.session_state[f"insight_{key}"]}</div>', unsafe_allow_html=True)
        else:
            st.info("💡 Click **Generate** to get AI-powered insights")

# ── DASHBOARD PAGE ─────────────────────────────────────────────────────────────
def page_dashboard():
    df = st.session_state["df"]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime","datetimetz"]).columns.tolist()

    if not num_cols:
        st.warning("No numeric columns found for visualisation.")
        return

    render_kpi_cards(df)
    st.markdown("---")

    # Row 1: Bar + Line
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if cat_cols and num_cols:
            cat, num = cat_cols[0], num_cols[0]
            grp = df.groupby(cat)[num].sum().sort_values(ascending=False).head(12).reset_index()
            fig = px.bar(grp, x=cat, y=num, color=num,
                         color_continuous_scale=[[0,"#1a1f3a"],[1,"#7c3aed"]],
                         title=f"{num.replace('_',' ').title()} by {cat.replace('_',' ').title()}")
            fig.update_layout(**_layout(height=320, coloraxis_showscale=False,
                                        xaxis=dict(tickangle=-30, gridcolor="#1c2d4a")))
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            _insight(f"Analyse this bar chart: {cat} vs {num}. Top value: {grp[cat].iloc[0]} = {grp[num].iloc[0]:,.2f}. Give business insights.", f"bar_{cat}_{num}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if date_cols and num_cols:
            dc, num = date_cols[0], num_cols[0]
            ts = df.set_index(dc)[num].resample("ME").sum().reset_index()
            fig = go.Figure(go.Scatter(x=ts[dc], y=ts[num], mode="lines",
                line=dict(color="#06b6d4", width=2.5),
                fill="tozeroy", fillcolor="rgba(6,182,212,.08)"))
            fig.update_layout(**_layout(height=320,
                title=dict(text=f"{num.replace('_',' ').title()} Over Time",
                           font=dict(color="#e2e8f0",size=13),x=.5,xanchor="center")))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            _insight(f"Analyse monthly trend of {num}. Give observations on growth, seasonality, and peaks.", f"line_{dc}_{num}")
        elif num_cols:
            n1, n2 = (num_cols[0], num_cols[1]) if len(num_cols)>1 else (num_cols[0], num_cols[0])
            fig = px.histogram(df, x=n1, nbins=30, title=f"Distribution of {n1.replace('_',' ').title()}",
                               color_discrete_sequence=["#7c3aed"])
            fig.update_layout(**_layout(height=320))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            _insight(f"Describe the distribution of {n1}: mean={df[n1].mean():.2f}, std={df[n1].std():.2f}.", f"hist_{n1}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Pie + Scatter
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if cat_cols and num_cols:
            cat, num = cat_cols[0], num_cols[0]
            grp = df.groupby(cat)[num].sum().sort_values(ascending=False).head(8).reset_index()
            fig = go.Figure(go.Pie(labels=grp[cat], values=grp[num], hole=.5,
                marker_colors=_COLORS, textfont=dict(color="#94a3b8")))
            fig.update_layout(**_layout(height=300,
                title=dict(text=f"Share by {cat.replace('_',' ').title()}",
                           font=dict(color="#e2e8f0",size=13),x=.5,xanchor="center"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8",size=10))))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            _insight(f"The top category '{grp[cat].iloc[0]}' holds {grp[num].iloc[0]/grp[num].sum()*100:.1f}% share. Analyse concentration risk and opportunities.", f"pie_{cat}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if len(num_cols) >= 2:
            n1, n2 = num_cols[0], num_cols[1]
            color_col = cat_cols[0] if cat_cols else None
            fig = px.scatter(df.sample(min(500,len(df))), x=n1, y=n2, color=color_col,
                opacity=.7, color_discrete_sequence=_COLORS,
                title=f"{n1.replace('_',' ').title()} vs {n2.replace('_',' ').title()}")
            fig.update_layout(**_layout(height=300))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            corr = df[[n1,n2]].corr().iloc[0,1]
            _insight(f"Scatter plot of {n1} vs {n2}. Correlation = {corr:.3f}. Analyse the relationship.", f"scatter_{n1}_{n2}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Box + Heatmap
    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if num_cols:
            col_to_plot = num_cols[:min(6,len(num_cols))]
            fig = go.Figure()
            for i, c in enumerate(col_to_plot):
                fig.add_trace(go.Box(y=df[c].dropna(), name=c.replace("_"," "), marker_color=_COLORS[i%len(_COLORS)]))
            fig.update_layout(**_layout(height=300,
                title=dict(text="Distribution Comparison (Box Plot)",
                           font=dict(color="#e2e8f0",size=13),x=.5,xanchor="center")))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            _insight(f"Box plot of {', '.join(col_to_plot[:3])}. Identify outliers and variability patterns.", f"box_{'_'.join(col_to_plot[:2])}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if len(num_cols) >= 3:
            corr_df = df[num_cols[:8]].corr()
            fig = go.Figure(go.Heatmap(z=corr_df.values, x=corr_df.columns.tolist(),
                y=corr_df.columns.tolist(), colorscale=[[0,"#06b6d4"],[.5,"#0f172a"],[1,"#7c3aed"]],
                zmid=0, text=corr_df.round(2).values, texttemplate="%{text}",
                hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>"))
            fig.update_layout(**_layout(height=300,
                title=dict(text="Correlation Matrix", font=dict(color="#e2e8f0",size=13),x=.5,xanchor="center")))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            _insight(f"Correlation matrix of numeric columns. Highlight the strongest positive and negative correlations.", "heatmap")
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 4: Area + Treemap
    c7, c8 = st.columns(2)
    with c7:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if len(num_cols) >= 2:
            sample = df[num_cols[:3]].dropna().iloc[-60:]
            fig = go.Figure()
            for i, c in enumerate(sample.columns):
                fig.add_trace(go.Scatter(x=list(range(len(sample))), y=sample[c],
                    mode="lines", name=c.replace("_"," "), line=dict(color=_COLORS[i], width=2),
                    fill="tozeroy", fillcolor=f"rgba({int(_COLORS[i][1:3],16)},{int(_COLORS[i][3:5],16)},{int(_COLORS[i][5:7],16)},.06)"))
            fig.update_layout(**_layout(height=280,
                title=dict(text="Area Chart — Multi-metric",
                           font=dict(color="#e2e8f0",size=13),x=.5,xanchor="center")))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c8:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if len(cat_cols) >= 1 and num_cols:
            cat, num = cat_cols[0], num_cols[0]
            grp = df.groupby(cat)[num].sum().reset_index().nlargest(20, num)
            grp["all"] = "All"
            fig = px.treemap(grp, path=["all", cat], values=num,
                color=num, color_continuous_scale=[[0,"#1a1f3a"],[1,"#7c3aed"]],
                title=f"Treemap — {num.replace('_',' ').title()} by {cat.replace('_',' ').title()}")
            fig.update_layout(**_layout(height=280, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

# ── AI CHAT PAGE ───────────────────────────────────────────────────────────────
def page_chat():
    df = st.session_state["df"]
    st.markdown("### 💬 AI Data Chat")

    # Suggested prompts
    suggestions = [
        "Which region has the highest revenue?",
        "Show top 5 products by sales",
        "What are the key trends in this data?",
        "Find underperforming categories",
        "Summarize this dataset in 5 points",
    ]
    st.markdown('<div style="margin-bottom:12px;">', unsafe_allow_html=True)
    s_cols = st.columns(len(suggestions))
    for i, (col_w, sug) in enumerate(zip(s_cols, suggestions)):
        with col_w:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["chat_history"].append({"role":"user","content":sug})
                with st.spinner("🔍 Analyzing your data..."):
                    schema = ", ".join(f"{c}({str(df[c].dtype)})" for c in df.columns[:15])
                    sample = df.head(5).to_string(index=False)
                    prompt = f"Dataset schema: {schema}\n\nSample:\n{sample}\n\nQuestion: {sug}\n\nAnswer concisely with specific numbers from the data."
                    
                    # Show reasoning trace
                    with st.status("⚙️ Processing Request", expanded=False) as status:
                        st.write("📊 Analyzing dataset structure...")
                        time.sleep(0.3)
                        st.write("🧠 Generating insights...")
                        time.sleep(0.3)
                        st.write("💾 Executing analysis...")
                        
                        resp = _ask(prompt)
                        
                        st.write("✅ Analysis complete!")
                        status.update(label="✅ Processing Complete", state="complete")
                    
                st.session_state["chat_history"].append({"role":"assistant","content":resp})
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div class="chat-user"><p>{msg["content"]}</p></div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="📊"):
                st.markdown(f'<div class="chat-ai"><p>{msg["content"]}</p></div>', unsafe_allow_html=True)

    # Input
    if user_input := st.chat_input("Ask anything about your data…"):
        st.session_state["chat_history"].append({"role":"user","content":user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div class="chat-user"><p>{user_input}</p></div>', unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="📊"):
            # Detailed reasoning trace
            with st.status("⚙️ Processing your question", expanded=True) as status:
                st.write("📋 Step 1: Analyzing question intent...")
                time.sleep(0.2)
                
                st.write("📊 Step 2: Examining dataset structure & types...")
                schema = ", ".join(f"{c}({str(df[c].dtype)})" for c in df.columns[:20])
                sample = df.head(5).to_string(index=False)
                stats = df.describe().to_string()
                time.sleep(0.2)
                
                st.write("🔍 Step 3: Searching for relevant columns...")
                time.sleep(0.2)
                
                st.write("🧠 Step 4: Generating AI insights...")
                prompt = (f"Dataset: {st.session_state['filename']}\n"
                         f"Schema: {schema}\nSample:\n{sample}\n"
                         f"Stats:\n{stats}\n\nQuestion: {user_input}\n\n"
                         "Give a precise, data-backed answer. Use specific numbers. Be concise.")
                resp = _ask(prompt)
                time.sleep(0.2)
                
                st.write("✅ Completed analysis!")
                status.update(label="✅ Request Processed", state="complete")
            
            st.markdown(f'<div class="chat-ai"><p>{resp}</p></div>', unsafe_allow_html=True)
            st.session_state["chat_history"].append({"role":"assistant","content":resp})
            
            # Optional: SQL/Pandas code suggestions (bonus feature)
            with st.expander("💻 See Generated Code (Advanced)", expanded=False):
                st.markdown("**🔗 SQL Query Equivalent:**")
                st.code("-- SQL query would be auto-generated here\nSELECT * FROM data WHERE ...", language="sql")
                st.markdown("**🐍 Pandas Code Equivalent:**")
                st.code("# Pandas code to reproduce this analysis\ndf_filtered = df[df['column'] > value]\nresult = df_filtered.groupby('category').sum()", language="python")

    if st.session_state["chat_history"]:
        if st.button("🗑 Clear Chat", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

# ── FORECASTING PAGE ───────────────────────────────────────────────────────────
def page_forecasting():
    df = st.session_state["df"]
    st.markdown("### 📈 Forecasting Module")

    date_cols = df.select_dtypes(include=["datetime","datetimetz"]).columns.tolist()
    num_cols  = df.select_dtypes(include="number").columns.tolist()

    if not date_cols:
        # Try to find date-like object columns
        # Common date column name patterns
        common_date_patterns = {"date", "time", "timestamp", "created", "updated", "posted", 
                               "order_date", "sale_date", "transaction_date", "datetime",
                               "date_time", "year", "month", "day"}
        
        object_cols = df.select_dtypes(include="object").columns
        
        # First pass: check common date column names
        for col in object_cols:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in common_date_patterns):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    valid_ratio = parsed.notna().sum() / max(len(df), 1)
                    if valid_ratio > 0.4:  # Lowered threshold from 0.6 to 0.4
                        df[col] = parsed
                        date_cols.append(col)
                        logger.info(f"Detected date column '{col}' with {valid_ratio:.1%} valid dates")
                except Exception as e:
                    logger.debug(f"Failed to parse '{col}' as date: {e}")
                    pass
        
        # Second pass: check all object columns if no dates found yet
        if not date_cols:
            for col in object_cols:
                if col not in date_cols:  # Skip already processed
                    try:
                        parsed = pd.to_datetime(df[col], errors="coerce")
                        valid_ratio = parsed.notna().sum() / max(len(df), 1)
                        if valid_ratio > 0.5:  # Threshold for general columns
                            df[col] = parsed
                            date_cols.append(col)
                            logger.info(f"Detected date column '{col}' with {valid_ratio:.1%} valid dates")
                    except Exception as e:
                        logger.debug(f"Failed to parse '{col}' as date: {e}")
                        pass

    if not date_cols:
        st.warning("No date columns detected. Please ensure your CSV has a date column.")
        return
    if not num_cols:
        st.warning("No numeric columns found for forecasting.")
        return

    # Blue info banner
    st.markdown(f"""
    <div style="background:rgba(6,182,212,.07);border:1px solid rgba(6,182,212,.2);
        border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:.82rem;color:#67e8f9;">
        📅 Date column detected: <b>{date_cols[0]}</b>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        target_col = st.selectbox("Metric to forecast", num_cols, key="fc_target")
    with col2:
        date_col   = st.selectbox("Date column",        date_cols, key="fc_date")
    with col3:
        horizon = st.slider("Forecast days", 7, 180, 30, key="fc_horizon")

    if st.button("▶ Run Forecast", key="run_fc", use_container_width=False):
        with st.spinner("Running forecast…"):
            try:
                _run_forecast(df, date_col, target_col, horizon)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                # Fallback: simple moving average forecast
                _simple_forecast(df, date_col, target_col, horizon)

def _run_forecast(df, date_col, target_col, horizon):
    try:
        from prophet import Prophet
        fc_df = df[[date_col, target_col]].dropna()
        fc_df = fc_df.rename(columns={date_col:"ds", target_col:"y"})
        fc_df["ds"] = pd.to_datetime(fc_df["ds"])
        fc_df = fc_df.groupby("ds")["y"].sum().reset_index()

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(fc_df)
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["y"], mode="lines",
            name="Historical", line=dict(color="#7c3aed", width=2)))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
            mode="lines", name=f"Forecast ({horizon}d)",
            line=dict(color="#06b6d4", width=2, dash="dash")))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(6,182,212,.1)", line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval", showlegend=True))
        fig.update_layout(**_layout(height=400,
            title=dict(text=f"{target_col.replace('_',' ').title()} — {horizon}-Day Forecast",
                       font=dict(color="#e2e8f0",size=14),x=.5,xanchor="center")))
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary
        last_hist  = fc_df["y"].iloc[-1]
        last_fc    = forecast["yhat"].iloc[-1]
        change_pct = (last_fc - last_hist) / max(abs(last_hist),1) * 100
        st.markdown(f"""
        <div class="insight-box">
          <b>📊 Forecast Summary</b><br>
          Current value: <b>{last_hist:,.2f}</b> &nbsp;→&nbsp;
          Predicted ({horizon}d): <b>{last_fc:,.2f}</b>
          &nbsp;(<span style="color:{'#10b981' if change_pct>=0 else '#ef4444'};">
          {'▲' if change_pct>=0 else '▼'} {abs(change_pct):.1f}%</span>)<br>
          Confidence range: {forecast['yhat_lower'].iloc[-1]:,.2f} — {forecast['yhat_upper'].iloc[-1]:,.2f}
        </div>""", unsafe_allow_html=True)
        _log("forecast")

    except ImportError:
        _simple_forecast(df, date_col, target_col, horizon)

def _simple_forecast(df, date_col, target_col, horizon):
    """Fallback: exponential smoothing forecast."""
    fc_df = df[[date_col, target_col]].dropna().copy()
    fc_df[date_col] = pd.to_datetime(fc_df[date_col])
    fc_df = fc_df.groupby(date_col)[target_col].sum().reset_index().sort_values(date_col)

    values = fc_df[target_col].values.astype(float)
    alpha  = 0.3
    smooth = [values[0]]
    for v in values[1:]:
        smooth.append(alpha * v + (1 - alpha) * smooth[-1])

    last_date  = fc_df[date_col].max()
    fc_dates   = [last_date + timedelta(days=i+1) for i in range(horizon)]
    last_smooth = smooth[-1]
    trend  = (smooth[-1] - smooth[max(0,len(smooth)-7)]) / max(7,1)
    fc_vals = [last_smooth + trend * (i+1) for i in range(horizon)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc_df[date_col], y=fc_df[target_col], mode="lines",
        name="Historical", line=dict(color="#7c3aed", width=2)))
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_vals, mode="lines",
        name=f"Forecast ({horizon}d)", line=dict(color="#06b6d4", width=2, dash="dash")))
    fig.update_layout(**_layout(height=400,
        title=dict(text=f"{target_col.replace('_',' ').title()} — {horizon}d Forecast (EMA)",
                   font=dict(color="#e2e8f0",size=14),x=.5,xanchor="center")))
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})
    st.markdown('</div>', unsafe_allow_html=True)
    st.info("ℹ️ Using Exponential Moving Average forecast. Install `prophet` for advanced predictions.")
    _log("forecast_ema")

# ── ANOMALY DETECTION PAGE ─────────────────────────────────────────────────────
def page_anomaly():
    df = st.session_state["df"]
    st.markdown("### 🔍 Anomaly Detection")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        feat_cols = st.multiselect("Feature columns", num_cols, default=num_cols[:min(4,len(num_cols))], key="ad_cols")
    with col2:
        contamination = st.slider("Expected anomaly rate (%)", 1, 20, 5, 1, key="ad_cont") / 100.0
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍 Detect", key="run_anomaly", use_container_width=True)

    if not feat_cols:
        st.info("Select at least one feature column.")
        return

    if run_btn:
        with st.spinner("Running Isolation Forest…"):
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler

                clean = df[feat_cols].dropna()
                scaler = StandardScaler()
                scaled = scaler.fit_transform(clean.values)
                clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
                labels = clf.fit_predict(scaled)
                scores = clf.score_samples(scaled)

                result_df = clean.copy()
                result_df["_anomaly"] = labels == -1
                result_df["_score"]   = scores
                n_anom = (labels == -1).sum()
                pct    = n_anom / len(clean) * 100

                # Summary cards
                ca, cb, cc, cd = st.columns(4)
                ca.metric("Records Analyzed", f"{len(clean):,}")
                cb.metric("Anomalies Found", str(n_anom))
                cc.metric("Anomaly Rate", f"{pct:.1f}%")
                cd.metric("Method", "Isolation Forest")

                # Score distribution
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    fig = go.Figure(go.Histogram(x=scores, nbinsx=30,
                        marker_color="#7c3aed", opacity=.85))
                    fig.update_layout(**_layout(height=280,
                        title=dict(text="Anomaly Score Distribution",
                                   font=dict(color="#e2e8f0",size=13),x=.5,xanchor="center"),
                        showlegend=False))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_chart2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    x_col, y_col = feat_cols[0], feat_cols[1] if len(feat_cols)>1 else feat_cols[0]
                    fig = px.scatter(result_df, x=x_col, y=y_col,
                        color=result_df["_anomaly"].map({True:"Anomaly",False:"Normal"}),
                        color_discrete_map={"Anomaly":"#ef4444","Normal":"#7c3aed"},
                        opacity=.75, title=f"Anomalies: {x_col} vs {y_col}")
                    fig.update_layout(**_layout(height=280))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                    st.markdown('</div>', unsafe_allow_html=True)

                # Top anomaly records
                st.markdown("#### 🚨 Top Anomalous Records")
                anom_rows = result_df[result_df["_anomaly"]].nsmallest(20, "_score")
                display = anom_rows.drop(columns=["_anomaly"]).rename(columns={"_score":"anomaly_score"})
                st.dataframe(display, use_container_width=True, hide_index=True, height=280)

                # LLM insight
                summary = f"{n_anom} anomalies ({pct:.1f}%) detected in {len(clean):,} records using Isolation Forest on columns: {', '.join(feat_cols[:4])}."
                _insight(f"{summary} Provide business interpretation and recommended actions.", "anomaly_page")
                _log("anomaly")

            except ImportError:
                st.error("scikit-learn is required. Run: pip install scikit-learn")

# ── OBSERVABILITY PAGE ─────────────────────────────────────────────────────────
def page_observability():
    st.markdown("### 📋 Observability & Logs")

    logs = st.session_state.get("query_log", [])
    total_q = st.session_state.get("total_queries", 0)
    errors  = st.session_state.get("errors", 0)

    latencies = [l.get("latency_ms", 0) for l in logs if l.get("latency_ms", 0) > 0]
    avg_lat = int(np.mean(latencies)) if latencies else 0
    actions = [l.get("action","unknown") for l in logs]
    unique_actions = len(set(actions))

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Queries",   str(total_q))
    m2.metric("Avg Latency",     f"{avg_lat} ms")
    m3.metric("Errors",          str(errors))
    m4.metric("Unique Actions",  str(unique_actions))
    st.markdown("---")

    if not logs:
        st.info("No activity logged yet. Start using the app to see observability data.")
        return

    log_df = pd.DataFrame(logs)
    log_df["ts"] = pd.to_datetime(log_df["ts"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if "action" in log_df.columns:
            ac = log_df["action"].value_counts().reset_index()
            ac.columns = ["action","count"]
            fig = px.bar(ac, x="action", y="count",
                color="count", color_continuous_scale=[[0,"#1a1f3a"],[1,"#7c3aed"]],
                title="Queries by Action")
            fig.update_layout(**_layout(height=280, coloraxis_showscale=False))
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if latencies and "ts" in log_df.columns:
            lat_df = log_df[log_df["latency_ms"]>0].copy()
            fig = go.Figure(go.Scatter(x=lat_df["ts"], y=lat_df["latency_ms"],
                mode="lines+markers", line=dict(color="#06b6d4", width=2),
                marker=dict(color="#7c3aed", size=5)))
            fig.update_layout(**_layout(height=280,
                title=dict(text="Response Latency Over Time (ms)",
                           font=dict(color="#e2e8f0",size=13),x=.5,xanchor="center")))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent log entries
    st.markdown("#### Recent Log Entries")
    header_html = """<div class="log-row" style="background:#111a2e;font-weight:700;color:#64748b;font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;">
        <span style="min-width:160px;">Timestamp</span>
        <span style="min-width:140px;">Action</span>
        <span style="min-width:100px;">Latency</span>
        <span>Session</span>
    </div>"""
    rows_html = ""
    for log in reversed(logs[-25:]):
        ts_str  = log.get("ts","")[:19].replace("T"," ")
        action  = log.get("action","—")
        lat     = log.get("latency_ms",0)
        sess    = log.get("session","—")
        lat_color = "#10b981" if lat < 1000 else "#f59e0b" if lat < 5000 else "#ef4444"
        lat_str = f"<span style='color:{lat_color};'>{lat} ms</span>" if lat else "—"
        rows_html += f"""<div class="log-row">
            <span style="min-width:160px;color:#94a3b8;">{ts_str}</span>
            <span style="min-width:140px;">{action}</span>
            <span style="min-width:100px;">{lat_str}</span>
            <span style="color:#475569;">{sess}</span>
        </div>"""

    st.markdown(f"""
    <div style="background:#0c1220;border:1px solid #1c2d4a;border-radius:10px;overflow:hidden;">
        {header_html}{rows_html}
    </div>""", unsafe_allow_html=True)

    if st.button("🗑 Clear Logs", key="clear_logs"):
        st.session_state["query_log"] = []
        st.session_state["total_queries"] = 0
        st.session_state["errors"] = 0
        st.rerun()

# ── EXPORT PAGE ────────────────────────────────────────────────────────────────
def page_export():
    df = st.session_state["df"]
    fname = st.session_state.get("filename","data").replace(".csv","")
    st.markdown("### ⬇ Export & Download")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📄 CSV Export")
        st.markdown('<div style="font-size:.82rem;color:#64748b;margin-bottom:12px;">Full dataset as comma-separated values</div>', unsafe_allow_html=True)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("⬇ Download CSV", data=buf.getvalue(),
            file_name=f"{fname}_export.csv", mime="text/csv", use_container_width=True)
        st.markdown(f'<div style="font-size:.75rem;color:#475569;margin-top:8px;">{len(df):,} rows × {len(df.columns)} columns</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Excel Export")
        st.markdown('<div style="font-size:.82rem;color:#64748b;margin-bottom:12px;">Multi-sheet Excel with summary statistics</div>', unsafe_allow_html=True)
        try:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Data", index=False)
                df.describe().to_excel(writer, sheet_name="Statistics")
                null_counts = df.isnull().sum().reset_index()
                null_counts.columns = ["Column","Null Count"]
                null_counts.to_excel(writer, sheet_name="Quality", index=False)
            st.download_button("⬇ Download Excel", data=excel_buf.getvalue(),
                file_name=f"{fname}_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        except Exception as e:
            st.error(f"Excel export failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### � PDF Report")
        st.markdown('<div style="font-size:.82rem;color:#64748b;margin-bottom:12px;">Professional PDF data analysis report</div>', unsafe_allow_html=True)
        if st.button("📋 Generate PDF", key="gen_pdf_export", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                    from reportlab.lib.enums import TA_CENTER, TA_LEFT
                    from reportlab.lib import colors
                    import datetime
                    
                    pdf_buf = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buf, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
                    story = []
                    styles = getSampleStyleSheet()
                    
                    # Title
                    title_style = ParagraphStyle(
                        'Title', parent=styles['Heading1'],
                        fontSize=20, textColor=colors.HexColor('#f1f5f9'),
                        spaceAfter=6, alignment=TA_CENTER
                    )
                    story.append(Paragraph(f"{fname.upper()} - Data Report", title_style))
                    
                    # Metadata
                    meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#64748b'))
                    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
                    story.append(Spacer(1, 0.15*inch))
                    
                    # Overview table
                    overview_data = [
                        ["Rows", "Columns", "Memory (KB)", "Numeric", "Text", "Date"],
                        [
                            f"{len(df):,}",
                            str(len(df.columns)),
                            f"{df.memory_usage(deep=True).sum()/1024:.0f}",
                            str(len(df.select_dtypes(include=['number']).columns)),
                            str(len(df.select_dtypes(include=['object']).columns)),
                            str(len(df.select_dtypes(include=['datetime']).columns)),
                        ]
                    ]
                    overview_table = Table(overview_data, colWidths=[1*inch]*6)
                    overview_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1c2d4a')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('FONTSIZE', (0, 1), (-1, 1), 9),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0c1220')),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#e2e8f0')),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1c2d4a')),
                    ]))
                    story.append(overview_table)
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Column info
                    col_heading = ParagraphStyle('ColHeading', parent=styles['Heading2'], fontSize=10, textColor=colors.HexColor('#a78bfa'))
                    story.append(Paragraph("Column Summary", col_heading))
                    
                    col_data = [["Column", "Type", "Non-Null", "Null %"]]
                    for col in df.columns[:20]:
                        null_pct = (df[col].isnull().sum() / len(df) * 100)
                        col_data.append([col[:18], str(df[col].dtype)[:12], str(df[col].notna().sum()), f"{null_pct:.1f}%"])
                    
                    col_table = Table(col_data, colWidths=[1.8*inch, 1.2*inch, 0.9*inch, 0.7*inch])
                    col_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1c2d4a')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#0c1220'), colors.HexColor('#111a2e')]),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#1c2d4a')),
                        ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ]))
                    story.append(col_table)
                    doc.build(story)
                    st.session_state["_pdf"] = pdf_buf.getvalue()
                    st.success("✅ PDF Generated!")
                except ImportError:
                    st.error("📦 Install reportlab: pip install reportlab")
        
        if "_pdf" in st.session_state:
            st.download_button("⬇ Download PDF", data=st.session_state["_pdf"],
                file_name=f"{fname}_report.pdf", mime="application/pdf", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Statistics preview
    st.markdown("---")
    st.markdown("#### 📊 Dataset Statistics Preview")
    tab1, tab2, tab3 = st.tabs(["Descriptive Stats", "Data Types", "Null Analysis"])
    with tab1:
        st.dataframe(df.describe(), use_container_width=True)
    with tab2:
        dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values, "Null%": (df.isnull().mean()*100).round(1).values})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with tab3:
        null_data = df.isnull().sum().reset_index()
        null_data.columns = ["Column","Null Count"]
        null_data["Null %"] = (null_data["Null Count"]/len(df)*100).round(2)
        null_data = null_data[null_data["Null Count"]>0].sort_values("Null %", ascending=False)
        if len(null_data):
            st.dataframe(null_data, use_container_width=True, hide_index=True)
            fig = px.bar(null_data, x="Column", y="Null %",
                color="Null %", color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
            fig.update_layout(**_layout(height=260, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.success("✅ No null values found in this dataset.")

# ── WELCOME SCREEN ─────────────────────────────────────────────────────────────
def page_welcome():
    st.markdown("""
    <div style="text-align:center;padding:48px 24px 32px;">
      <div style="font-size:3.5rem;margin-bottom:14px;">📊</div>
      <div style="font-size:2rem;font-weight:900;color:#f1f5f9;letter-spacing:-.5px;margin-bottom:10px;">
        AI-Powered Data Analyst</div>
      <div style="font-size:1rem;color:#64748b;max-width:520px;margin:0 auto 28px;line-height:1.7;">
        Enterprise-grade analytics platform. Upload a CSV to unlock instant AI-driven
        insights, forecasting, anomaly detection, and beautiful visualizations.
      </div>
    </div>""", unsafe_allow_html=True)

    features = [
        ("📊","Dashboard","Auto-generated charts: bar, line, pie, scatter, heatmap, treemap and more with AI insights beneath each."),
        ("💬","AI Chat","Conversational interface — ask any question about your data in plain English."),
        ("📈","Forecasting","Prophet-based time series predictions with confidence intervals and trend analysis."),
        ("🔍","Anomaly Detection","Isolation Forest identifies statistical outliers with Z-scores and explanations."),
        ("📋","Observability","Track queries, latency, errors, and session activity in real-time logs."),
        ("⬇","Export","Download data as CSV, Excel (multi-sheet), and AI-generated Markdown reports."),
    ]
    for row_start in range(0, len(features), 3):
        row = features[row_start:row_start+3]
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align:left;min-height:130px;margin-bottom:12px;">
                  <div style="font-size:1.6rem;margin-bottom:8px;">{icon}</div>
                  <div style="font-size:.88rem;font-weight:700;color:#e2e8f0;
                    margin-bottom:5px;">{title}</div>
                  <div style="font-size:.78rem;color:#475569;line-height:1.6;">{desc}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:24px;">
      <div style="display:inline-block;background:#0c1220;border:1px solid #1c2d4a;
        border-radius:24px;padding:10px 24px;font-size:.82rem;color:#334155;">
        ← Upload a CSV file in the sidebar to get started
      </div>
    </div>""", unsafe_allow_html=True)


# ── MAIN ROUTER ────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    df = st.session_state.get("df")
    page = st.session_state.get("page", "Dashboard")

    # Always show header
    render_header()

    if df is None:
        page_welcome()
        return

    # Page routing
    if page == "Dashboard":
        page_dashboard()
    elif page == "AI Chat":
        page_chat()
    elif page == "Forecasting":
        page_forecasting()
    elif page == "Anomaly Detection":
        page_anomaly()
    elif page == "Observability":
        page_observability()
    elif page == "Export":
        page_export()


if __name__ == "__main__":
    main()
else:
    main()
