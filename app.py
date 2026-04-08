from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.advanced_lab import (
    bootstrap_corridor_statistics,
    monte_carlo_risk,
    train_advanced_classifier,
    train_regression_suite,
)
from src.dataset_hub import (
    crawl_detection_dataset_sources,
    export_dataset_bundle,
    import_dataset_from_kaggle,
    import_dataset_from_url,
    import_dataset_from_upload,
    load_dataset_registry_df,
    scan_raw_datasets,
    sync_dataset_samples_to_vision,
)
from src.emergency_location import build_network_data, compute_station_metrics, optimize_emergency_centers
from src.llm_api import LLMSettings, enhance_answer_with_llm, is_llm_configured, test_llm_connection
from src.model_lab import corridor_deep_forecast, detect_anomaly_points, feature_importance_from_rf, train_model_zoo
from src.traffic_data import collect_online_hotspots, corridor_list, load_hotspots_data, load_or_build_traffic_data
from src.traffic_model import explain_next_hour_drivers, forecast_corridor, train_traffic_model
from src.traffic_qa import answer_query
from src.traffic_strategy import (
    compute_city_kpis,
    generate_micro_policies,
    rank_hot_corridors,
)
from src.vision_lab import compare_yolo_versions, detect_available_device, run_vision_detection

px.defaults.template = "plotly_white"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "city_data"
HOTSPOT_FILE = DATA_DIR / "chengdu_hotspots_online.csv"
USER_DB_FILE = DATA_DIR / "user_accounts.json"
VISION_SAMPLE_DIR = DATA_DIR / "vision_samples"
VISION_SAMPLE_SOURCE = DATA_DIR / "_tmp_demo_img.png"
高德瓦片源 = [
    "https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=8&x={x}&y={y}&z={z}",
    "https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=8&x={x}&y={y}&z={z}",
    "https://webrd03.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=8&x={x}&y={y}&z={z}",
    "https://webrd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=8&x={x}&y={y}&z={z}",
]

基础模型中文名 = {
    "RandomForest": "随机森林",
    "GradientBoosting": "梯度提升树",
    "NeuralNet(MLP)": "多层感知机",
}

热点字段中文名 = {
    "published_at": "发布时间",
    "keyword": "关键词",
    "tag": "事件类别",
    "impact_score": "影响评分",
    "title": "热点标题",
}

热点类别中文名 = {
    "incident": "事故事件",
    "event": "活动事件",
    "transit": "公共交通",
    "weather": "天气影响",
    "general": "综合资讯",
    "fallback": "示例资讯",
}

基础特征中文名 = {
    "hour": "小时",
    "weekday": "星期",
    "is_weekend": "是否周末",
    "is_holiday": "是否节假日",
    "rain_intensity": "降雨强度",
    "event_intensity": "活动冲击",
    "incident_count": "事故数量",
    "roadwork_flag": "施工影响",
    "flow": "车流量",
    "capacity": "道路容量",
    "demand_capacity_ratio": "需求压力比",
    "avg_speed": "平均速度",
    "occupancy": "占有率",
    "bus_delay_min": "公交延误分钟",
    "metro_inflow": "地铁客流",
}

主题主色 = "#4f6f68"
主题次色 = "#7e948f"
主题强调 = "#93514b"
主题暖色 = "#b77b5a"
主题冷高亮 = "#4f6f68"
统一连续色阶 = ["#edf2f0", "#d7e2de", "#b4c6c1", "#7f9a93", "#4f6f68"]

DEMO_ACCOUNTS = {
    "评审专家": "chengdu2026",
    "管理员": "admin2026",
    "演示账号": "demo2026",
}

ROUTE_GEOMETRY: Dict[str, List[tuple[float, float]]] = {
    "一环路北段": [(30.679, 104.045), (30.682, 104.072), (30.679, 104.103)],
    "一环路南段": [(30.636, 104.041), (30.632, 104.074), (30.629, 104.105)],
    "人民南路": [(30.632, 104.073), (30.615, 104.073), (30.590, 104.074)],
    "天府大道北段": [(30.640, 104.076), (30.615, 104.077), (30.592, 104.078)],
    "天府大道南段": [(30.592, 104.078), (30.570, 104.078), (30.546, 104.079)],
    "科华路": [(30.617, 104.090), (30.604, 104.089), (30.582, 104.086)],
    "东大街": [(30.659, 104.075), (30.657, 104.088), (30.655, 104.101)],
    "蜀都大道": [(30.665, 104.045), (30.666, 104.069), (30.666, 104.095)],
    "中环路东段": [(30.690, 104.091), (30.660, 104.112), (30.625, 104.123)],
    "双流机场高速": [(30.606, 104.066), (30.575, 104.045), (30.537, 104.018)],
    "成龙大道": [(30.625, 104.102), (30.620, 104.122), (30.617, 104.145)],
    "凤凰山环线": [(30.714, 104.037), (30.707, 104.058), (30.700, 104.081)],
}

DATASET_BACKUP: List[Dict[str, str]] = [
    {
        "名称": "城市多摄像头交通挑战数据集",
        "说明": "覆盖道路监控视频、车辆行为识别、跨镜追踪，适合交通视觉与路网分析。",
        "链接": "https://www.aicitychallenge.org/ai-city-challenge-dataset-access/",
    },
    {
        "名称": "自动驾驶场景视觉数据集",
        "说明": "包含目标检测、可行驶区域、车道线等任务，适合目标检测与多任务学习。",
        "链接": "https://github.com/bdd100k/bdd100k",
    },
    {
        "名称": "城市交通车辆检测数据集",
        "说明": "经典车辆检测与跟踪数据，适合车流量识别与检测算法对比。",
        "链接": "https://sites.google.com/view/daweidu/projects/ua-detrac",
    },
    {
        "名称": "低空视角交通视觉数据集",
        "说明": "覆盖密集目标场景，可用于复杂交通环境下的小目标检测。",
        "链接": "http://aiskyeye.com/",
    },
]


def inject_style() -> None:
    st.markdown(
        """
<style>
:root {
  --bg:#f3f4f1;
  --panel:#ffffff;
  --panel-soft:#f8f9f6;
  --ink-900:#1f2a28;
  --ink-700:#3f4b48;
  --ink-500:#71807b;
  --line:#d7ddd8;
  --accent:#4f6f68;
  --accent-soft:#d7e2df;
  --danger:#93514b;
}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(980px 540px at 5% -24%, rgba(79,111,104,.12), transparent 45%),
    radial-gradient(860px 480px at 105% -16%, rgba(215,226,223,.55), transparent 42%),
    linear-gradient(180deg,#f7f7f5 0%, var(--bg) 100%);
}

[data-testid="stHeader"]{
  background: rgba(247,247,245,0.85);
  backdrop-filter: blur(8px);
}

[data-testid="stSidebar"]{
  background: #f7f8f6;
  border-right: 1px solid #d7ddd8;
}

[data-testid="stSidebar"] *{
  color: #1f2a28;
}

.main .block-container{
  max-width: 1400px;
  padding-top: 2.0rem;
  padding-bottom: 2.0rem;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", "Heiti SC", sans-serif;
  line-height: 1.45;
}

[data-testid="stVerticalBlock"] > [data-testid="element-container"]{
  margin-bottom: .45rem;
}

div[data-testid="stPlotlyChart"]{
  border:1px solid var(--line);
  border-radius: 16px;
  padding: .45rem .5rem .2rem .5rem;
  background: var(--panel);
  box-shadow: 0 8px 22px rgba(26,39,36,.05);
}

div[data-testid="stDataFrame"]{
  border:1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
  background: var(--panel);
}

.hero{
  border-radius: 18px;
  padding: 22px 24px;
  margin-bottom: 14px;
  border: 1px solid var(--line);
  background:
    radial-gradient(1000px 380px at -15% -40%, rgba(79,111,104,.16), transparent 45%),
    linear-gradient(180deg,#ffffff 0%, #f6f7f4 100%);
  color: var(--ink-900);
  box-shadow: 0 10px 24px rgba(26,39,36,.06);
  display: grid;
  grid-template-columns: 1.35fr .95fr;
  gap: 14px;
}

.hero-kicker{
  font-size: .78rem;
  letter-spacing: .1em;
  color: var(--ink-500);
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.hero h1{
  margin: 0;
  font-size: 2.03rem;
  line-height: 1.12;
  letter-spacing: .01em;
  font-weight: 800;
}

.hero p{
  margin: 10px 0 0 0;
  color: var(--ink-700);
  font-size: .98rem;
  max-width: 95%;
}

.chip{
  display:inline-flex;
  align-items:center;
  gap:5px;
  margin-top:10px;
  margin-right:8px;
  padding:5px 11px;
  border-radius:999px;
  border:1px solid var(--line);
  color: var(--ink-700);
  background: #ffffff;
  font-size:.78rem;
}

.hero-side{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  align-content: start;
}

.hero-stat{
  border:1px solid var(--line);
  border-radius: 12px;
  background: #fff;
  padding: 10px 12px;
}

.hero-stat .k{
  color: var(--ink-500);
  font-size: .75rem;
}

.hero-stat .v{
  color: var(--ink-900);
  font-size: 1.18rem;
  font-weight: 700;
  margin-top: 2px;
}

.section-title{
  font-weight: 750;
  color: var(--ink-900);
  font-size: 1.13rem;
  margin: 5px 0 9px 0;
  display: flex;
  align-items: center;
  gap: 0;
}

.section-sub{
  color: var(--ink-500);
  font-size: .86rem;
  margin-top: -2px;
  margin-bottom: 6px;
}

.note{
  color: var(--ink-500);
  font-size:.9rem;
  line-height: 1.55;
}

div[data-testid="stMetric"]{
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 11px;
  background: var(--panel);
  box-shadow: 0 4px 10px rgba(20,30,28,.03);
}

div[data-testid="stMetric"] label{
  color: var(--ink-500) !important;
  font-size: .95rem !important;
  font-weight: 600 !important;
  line-height: 1.35 !important;
}

div[data-testid="stMetricValue"]{
  color: var(--ink-900) !important;
}

div[data-testid="stMetricValue"] > div{
  font-size: clamp(2.05rem, 2.6vw, 2.75rem) !important;
  font-weight: 760 !important;
  line-height: 1.08 !important;
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] * ,
[data-testid="stMultiSelect"] * ,
[data-testid="stNumberInput"] input,
[data-testid="stFileUploader"] *,
[data-testid="stButton"] button{
  font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif !important;
}

[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] button[kind="primary"]{
  border-radius: 11px;
  border: 1px solid rgba(255,255,255,.2);
  background: rgba(255,255,255,.06);
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div{
  background: rgba(255,255,255,.22) !important;
}

.side-title{
  display:flex;
  align-items:center;
  gap:7px;
  font-size:1.35rem;
  font-weight:800;
  margin-bottom:8px;
}

.side-subtitle{
  display:flex;
  align-items:center;
  gap:7px;
  font-size:.93rem;
  font-weight:700;
  margin: 6px 0 8px 0;
}

.control-bar{
  border:1px solid var(--line);
  border-radius: 14px;
  background: linear-gradient(180deg,#ffffff 0%, #f8faf8 100%);
  padding: 12px 12px 10px 12px;
  box-shadow: 0 6px 14px rgba(20,30,28,.04);
  margin-bottom: 12px;
}

.user-badge{
  display:inline-flex;
  align-items:center;
  gap:4px;
  padding: 6px 10px;
  border-radius: 999px;
  border:1px solid var(--line);
  background:#fff;
  color: var(--ink-700);
  font-size: .82rem;
}

button[data-baseweb="tab"]{
  font-weight: 620 !important;
  color:var(--ink-500) !important;
  border-radius: 10px !important;
  padding: 0 14px !important;
  height: 42px !important;
  white-space: nowrap !important;
}

div[data-baseweb="tab-list"]{
  gap: 4px;
  background: var(--panel-soft);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 3px;
  margin-bottom: 12px;
}

button[data-baseweb="tab"][aria-selected="true"]{
  color: var(--ink-900) !important;
  background: #ffffff !important;
  border-bottom: none !important;
  box-shadow: 0 1px 0 rgba(10,12,11,.04);
}

.stButton > button{
  min-height: 42px;
  font-size: 0.97rem;
  font-weight: 640;
  letter-spacing: .01em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.2;
}

.ref-card{
  border:1px solid var(--line);
  background:var(--panel);
  border-radius: 12px;
  padding: 11px 12px;
  margin-bottom: 10px;
}

.insight-card{
  border:1px solid #d0dad6;
  background: linear-gradient(180deg,#ffffff 0%, #f8fbfa 100%);
  border-radius: 12px;
  padding: 9px 12px 8px 12px;
  margin: 5px 0 10px 0;
}

.insight-title{
  color:#2f403c;
  font-size:.90rem;
  font-weight:700;
  margin-bottom: 3px;
}

.insight-card ul{
  margin: 0;
  padding-left: 1.05rem;
  color:#4b5c57;
  font-size:.84rem;
  line-height:1.45;
}

#MainMenu, footer {
  visibility: hidden;
}

@media (max-width: 980px){
  .hero{
    grid-template-columns: 1fr;
  }
  .hero h1{
    font-size: 1.55rem;
  }
  .hero-side{
    grid-template-columns: 1fr 1fr;
  }
  button[data-baseweb="tab"]{
    padding: 0 10px !important;
    font-size: .88rem !important;
    height: 38px !important;
  }
}

@media (max-width: 640px){
  .main .block-container{
    padding-top: 1.35rem;
    padding-left: .75rem;
    padding-right: .75rem;
  }
  .hero{
    padding: 16px;
  }
  .hero-side{
    grid-template-columns: 1fr;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(df: pd.DataFrame, hotspots_df: pd.DataFrame) -> None:
    latest_ts = pd.to_datetime(df["timestamp"], errors="coerce").max()
    latest_str = latest_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(latest_ts) else "-"
    st.markdown(
        f"""
<div class="hero">
  <div>
    <div class="hero-kicker">城市交通智能决策平台</div>
    <h1>蓉城智行 · 城市交通智能中枢</h1>
    <p>以“城市拥堵治理”为核心，融合机器学习、时序预测、视觉感知与策略沙盘，形成可展示、可解释、可决策的一体化系统。</p>
    <span class="chip">样本 {len(df):,} 条</span>
    <span class="chip">热点 {len(hotspots_df)} 条</span>
    <span class="chip">机器学习与深度学习</span>
    <span class="chip">目标检测接口</span>
    <span class="chip">策略推演</span>
  </div>
  <div class="hero-side">
    <div class="hero-stat"><div class="k">核心走廊</div><div class="v">{len(ROUTE_GEOMETRY)}</div></div>
    <div class="hero-stat"><div class="k">覆盖区域</div><div class="v">{df['district'].nunique()}</div></div>
    <div class="hero-stat"><div class="k">模型类型</div><div class="v">随机森林 / 多层感知机</div></div>
    <div class="hero-stat"><div class="k">数据更新时间</div><div class="v" style="font-size:0.95rem;">{latest_str}</div></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _light(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#2c3b37", size=13, family="PingFang SC, Microsoft YaHei, Noto Sans SC, sans-serif"),
        title=dict(font=dict(size=21, color="#253430")),
        colorway=[主题主色, 主题次色, 主题强调, 主题暖色, "#8e9d99"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=12, color="#31423d"),
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="#d1dcd8",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#dce3e0",
        zeroline=False,
        showline=True,
        linecolor="#bcc9c4",
        linewidth=1.2,
        title_font=dict(size=14, color="#334641"),
        tickfont=dict(size=12, color="#4a5b56"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#dce3e0",
        zeroline=False,
        showline=True,
        linecolor="#bcc9c4",
        linewidth=1.2,
        title_font=dict(size=14, color="#334641"),
        tickfont=dict(size=12, color="#4a5b56"),
    )
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(font=dict(color="#32443f", size=13)),
            tickfont=dict(color="#41534e", size=12),
            outlinecolor="#c3d0cb",
            outlinewidth=1,
            bgcolor="rgba(255,255,255,0.82)",
        )
    )
    fig.update_annotations(font=dict(color="#2f3f3a", size=12))
    return fig


def _show(fig: go.Figure, key: str | None = None) -> None:
    fig2 = _light(fig)

    has_legend = any(
        (getattr(trace, "showlegend", True) is not False) and bool(getattr(trace, "name", ""))
        for trace in list(fig2.data)
    )
    if has_legend:
        margin_obj = getattr(fig2.layout, "margin", None)
        current_margin: Dict[str, object] = {}
        if margin_obj is not None:
            if hasattr(margin_obj, "to_plotly_json"):
                try:
                    current_margin = margin_obj.to_plotly_json() or {}
                except Exception:
                    current_margin = {}
            elif isinstance(margin_obj, dict):
                current_margin = margin_obj
            else:
                try:
                    current_margin = dict(margin_obj)
                except Exception:
                    current_margin = {}

        def _as_int(v: object, default: int = 0) -> int:
            try:
                return int(float(v))
            except Exception:
                return default

        left = max(_as_int(current_margin.get("l", 0), 0), 28)
        right = max(_as_int(current_margin.get("r", 0), 0), 18)
        top = max(_as_int(current_margin.get("t", 0), 0), 88)
        bottom = max(_as_int(current_margin.get("b", 0), 0), 44)
        fig2.update_layout(
            margin=dict(l=left, r=right, t=top, b=bottom),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(size=12, color="#31423d"),
                bgcolor="rgba(255,255,255,0.78)",
                bordercolor="#d1dcd8",
                borderwidth=1,
            ),
        )

    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False}, key=key)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
<div class="section-title">{title}</div>
<div class="section-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def _feature_name_cn(name: str) -> str:
    feat = str(name)
    if feat.startswith("num__"):
        feat = feat[5:]
    if feat.startswith("cat__"):
        feat = feat[5:]

    if feat.startswith("district_"):
        return f"区县：{feat.split('district_', 1)[1]}"
    if feat.startswith("corridor_"):
        return f"走廊：{feat.split('corridor_', 1)[1]}"
    if feat.startswith("weather_"):
        weather_value = feat.split("weather_", 1)[1]
        weather_cn = {
            "smooth": "畅通态",
            "busy": "缓行态",
            "severe": "拥堵态",
            "sunny": "晴天",
            "cloudy": "多云",
            "light_rain": "小雨",
            "heavy_rain": "大雨",
        }.get(weather_value, weather_value)
        return f"天气：{weather_cn}"

    return 基础特征中文名.get(feat, feat.replace("_", ""))


def _hotspot_table_cn(hotspots_df: pd.DataFrame) -> pd.DataFrame:
    show_cols = [c for c in ["published_at", "keyword", "tag", "impact_score", "title"] if c in hotspots_df.columns]
    if not show_cols:
        return pd.DataFrame()

    hot_show = hotspots_df.sort_values("impact_score", ascending=False).head(10)[show_cols].copy()
    if "published_at" in hot_show.columns:
        hot_show["published_at"] = pd.to_datetime(hot_show["published_at"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "tag" in hot_show.columns:
        hot_show["tag"] = hot_show["tag"].astype(str).map(热点类别中文名).fillna(hot_show["tag"].astype(str))
    hot_show = hot_show.rename(columns=热点字段中文名)
    return hot_show


def chart_insight(title: str, points: List[str]) -> None:
    if not st.session_state.get("show_chart_notes", True):
        return
    if not points:
        return
    items = "".join([f"<li>{str(p)}</li>" for p in points[:3]])
    st.markdown(
        f"""
<div class="insight-card">
  <div class="insight-title">{title}</div>
  <ul>{items}</ul>
</div>
        """,
        unsafe_allow_html=True,
    )


def _ensure_vision_demo_samples() -> List[Path]:
    VISION_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    real_files = sorted([p for p in VISION_SAMPLE_DIR.glob("real_traffic_*") if p.is_file()], key=lambda x: x.name)
    dataset_files = sorted([p for p in VISION_SAMPLE_DIR.glob("dataset_sample_*.jpg") if p.is_file()], key=lambda x: x.name)
    if dataset_files:
        merged = dataset_files[:18] + real_files[:12]
        return merged if merged else dataset_files
    if len(real_files) >= 6:
        return real_files[:16]

    sample_specs = [
        ("sample_mainroad_day.png", "主干道白天"),
        ("sample_mainroad_peak.png", "主干道晚高峰"),
        ("sample_intersection_busy.png", "十字路口高峰"),
        ("sample_rainy_crossing.png", "雨天路口"),
        ("sample_night_traffic.png", "夜间主干道"),
        ("sample_elevated_dense.png", "高架路拥堵"),
        ("sample_school_zone.png", "学校周边"),
        ("sample_station_zone.png", "地铁站周边"),
    ]
    targets = [VISION_SAMPLE_DIR / file_name for file_name, _ in sample_specs]
    if all(p.exists() and p.stat().st_size > 10_000 for p in targets):
        return targets

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return [p for p in targets if p.exists()]

    def _draw_vehicle(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, color: str) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=7, fill=color, outline="#1f2a28", width=2)
        draw.rectangle((x + 6, y + 5, x + w - 6, y + h // 2), fill="#d8e6f0")

    def _build_scene(seed: int, style: str) -> "Image.Image":
        rng = np.random.default_rng(seed)
        w, h = 1280, 720

        sky = "#d9e7f2"
        ground = "#ced6dc"
        road = "#636a72"
        lane = "#f3f4f1"
        if style == "night":
            sky, ground, road, lane = "#1f2730", "#2b333b", "#3a424a", "#cad2d9"
        elif style == "rain":
            sky, ground, road, lane = "#bcc6cf", "#b9c2c9", "#5c6570", "#d8dce0"

        img = Image.new("RGB", (w, h), sky)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 220, w, h), fill=ground)
        draw.polygon([(120, h), (430, 260), (850, 260), (1160, h)], fill=road)

        # 车道线
        for k in range(1, 4):
            x_start = 120 + k * (1040 // 4)
            x_end = 430 + k * (420 // 4)
            for s in np.linspace(0, 1, 12):
                y0 = int(680 - s * 380)
                y1 = y0 - 16
                x0 = int(x_start - s * (x_start - x_end))
                x1 = int(x0 - 6)
                draw.rectangle((x1, y1, x0 + 6, y0), fill=lane)

        # 横向道路（用于十字路口/学校/地铁站场景）
        if style in {"cross", "school", "station", "rain"}:
            draw.polygon([(0, 470), (420, 410), (860, 410), (1280, 470), (1280, 560), (0, 560)], fill=road)
            # 斑马线
            for i in range(10):
                zx = 520 + i * 22
                draw.rectangle((zx, 430, zx + 12, 462), fill="#efefef")

        # 车辆
        density = {"peak": 20, "cross": 18, "night": 14, "rain": 16, "elevated": 24, "school": 17, "station": 19}.get(style, 12)
        palette = ["#ef4444", "#f59e0b", "#3b82f6", "#22c55e", "#f97316", "#14b8a6", "#9ca3af"]
        for _ in range(density):
            y = int(rng.integers(300, 670))
            lane_center = int(rng.choice([330, 490, 650, 810, 960]))
            scale = max(0.65, min(1.7, (y - 260) / 260))
            vw = int(rng.integers(34, 52) * scale)
            vh = int(rng.integers(20, 30) * scale)
            x = lane_center - vw // 2 + int(rng.integers(-14, 14))
            color = str(rng.choice(palette))
            _draw_vehicle(draw, x, y, vw, vh, color)

        # 行人
        if style in {"school", "station", "cross"}:
            for _ in range(12):
                px = int(rng.integers(460, 830))
                py = int(rng.integers(390, 540))
                draw.ellipse((px, py, px + 8, py + 8), fill="#2f3f3a")
                draw.rectangle((px + 2, py + 8, px + 6, py + 18), fill="#2f3f3a")

        # 雨线/夜间光带
        if style == "rain":
            for _ in range(180):
                rx = int(rng.integers(0, w))
                ry = int(rng.integers(0, h))
                draw.line((rx, ry, rx - 8, ry + 20), fill="#e8edf2", width=1)
        if style == "night":
            for _ in range(45):
                lx = int(rng.integers(280, 1020))
                ly = int(rng.integers(320, 700))
                draw.line((lx, ly, lx + 22, ly), fill="#f8e16a", width=2)

        # 高架
        if style == "elevated":
            draw.polygon([(130, 330), (410, 250), (900, 250), (1170, 330), (1170, 370), (130, 370)], fill="#4e5660")
            for _ in range(14):
                ex = int(rng.integers(180, 1090))
                ey = int(rng.integers(270, 342))
                _draw_vehicle(draw, ex, ey, 32, 18, str(rng.choice(palette)))

        return img

    style_order = ["day", "peak", "cross", "rain", "night", "elevated", "school", "station"]
    for idx, out_path in enumerate(targets):
        style = style_order[idx] if idx < len(style_order) else "day"
        try:
            scene = _build_scene(seed=2026 + idx * 13, style=style)
            scene.save(out_path)
        except Exception:
            continue
    return [p for p in targets if p.exists()]


def _load_real_sample_title_map() -> Dict[str, str]:
    meta_path = VISION_SAMPLE_DIR / "real_photo_sources.json"
    mapping: Dict[str, str] = {}
    if not meta_path.exists():
        return mapping
    try:
        rows = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return mapping
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            file_name = str(row.get("local_file", "")).strip()
            title = str(row.get("title", "")).strip().replace("File:", "")
            if title.lower().endswith((".jpg", ".jpeg", ".png")):
                title = title.rsplit(".", 1)[0]
            t_low = title.lower()
            if "intersection" in t_low:
                scene = "城市路口"
            elif "ring road" in t_low or "highway" in t_low or "i-26" in t_low:
                scene = "高速路段"
            elif "bridge" in t_low:
                scene = "桥梁道路"
            elif "street" in t_low or "boulevard" in t_low:
                scene = "城市道路"
            elif "traffic" in t_low:
                scene = "道路交通"
            else:
                scene = "交通主干道"
            if file_name:
                mapping[file_name] = f"实拍{idx} {scene}"
    except Exception:
        return {}
    return mapping


def _vision_sample_display_name(path: Path) -> str:
    real_map = _load_real_sample_title_map()
    if path.name in real_map:
        return real_map[path.name]
    if path.name.startswith("dataset_sample_"):
        seg = path.stem.split("_")
        seq = seg[-1] if len(seg) >= 1 else ""
        dataset_code = seg[2][:6] if len(seg) >= 3 else "外部"
        return f"数据样本{seq}（{dataset_code}）"
    name_map = {
        "sample_mainroad_day.png": "示例1 主干道白天",
        "sample_mainroad_peak.png": "示例2 主干道晚高峰",
        "sample_intersection_busy.png": "示例3 十字路口高峰",
        "sample_rainy_crossing.png": "示例4 雨天路口",
        "sample_night_traffic.png": "示例5 夜间主干道",
        "sample_elevated_dense.png": "示例6 高架路拥堵",
        "sample_school_zone.png": "示例7 学校周边",
        "sample_station_zone.png": "示例8 地铁站周边",
        "示例图_主干道白天.png": "示例(旧) 主干道白天",
        "示例图_高峰期路口.png": "示例(旧) 高峰期路口",
        "示例图_阴雨场景.png": "示例(旧) 阴雨场景",
    }
    return name_map.get(path.name, path.stem.replace("_", " "))


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}::{password}".encode("utf-8")).hexdigest()


def _default_user_db() -> Dict[str, Dict[str, Dict[str, str]]]:
    users: Dict[str, Dict[str, str]] = {}
    for username, password in DEMO_ACCOUNTS.items():
        salt = f"demo_{username}"
        users[username] = {
            "salt": salt,
            "pwd_hash": _hash_password(password, salt),
            "role": "演示账号",
            "created_at": "2026-04-07 00:00:00",
        }
    return {"users": users}


def _save_user_db(db: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    USER_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_DB_FILE.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_user_db() -> Dict[str, Dict[str, Dict[str, str]]]:
    USER_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    if USER_DB_FILE.exists():
        try:
            db = json.loads(USER_DB_FILE.read_text(encoding="utf-8"))
            if isinstance(db, dict) and isinstance(db.get("users"), dict):
                # 补齐默认演示账号，防止文件被外部覆盖
                changed = False
                for username, password in DEMO_ACCOUNTS.items():
                    if username not in db["users"]:
                        salt = f"demo_{username}"
                        db["users"][username] = {
                            "salt": salt,
                            "pwd_hash": _hash_password(password, salt),
                            "role": "演示账号",
                            "created_at": "2026-04-07 00:00:00",
                        }
                        changed = True
                if changed:
                    _save_user_db(db)
                return db
        except Exception:
            pass
    db = _default_user_db()
    _save_user_db(db)
    return db


def _verify_user(username: str, password: str) -> tuple[bool, str]:
    uname = username.strip()
    if not uname or not password:
        return False, "账号和密码不能为空。"
    db = _load_user_db()
    user = db.get("users", {}).get(uname)
    if not user:
        return False, "账号不存在，请先注册。"
    salt = str(user.get("salt", ""))
    pwd_hash = str(user.get("pwd_hash", ""))
    if _hash_password(password, salt) != pwd_hash:
        return False, "密码错误，请重试。"
    return True, "登录成功。"


def _register_user(username: str, password: str) -> tuple[bool, str]:
    uname = username.strip()
    if len(uname) < 2 or len(uname) > 20:
        return False, "账号长度需在2到20位之间。"
    if any(ch.isspace() for ch in uname):
        return False, "账号不能包含空格。"
    if len(password) < 6:
        return False, "密码长度至少6位。"

    db = _load_user_db()
    users = db.get("users", {})
    if uname in users:
        return False, "账号已存在，请更换账号名。"

    salt = hashlib.sha256(f"{uname}-{datetime.now().isoformat()}".encode("utf-8")).hexdigest()[:16]
    users[uname] = {
        "salt": salt,
        "pwd_hash": _hash_password(password, salt),
        "role": "注册用户",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    db["users"] = users
    _save_user_db(db)
    return True, "注册成功，请返回登录。"


def show_login() -> bool:
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "login_user" not in st.session_state:
        st.session_state["login_user"] = ""

    if st.session_state["logged_in"]:
        return True

    st.markdown(
        """
<div style="
max-width:540px;
margin: 4.5rem auto 0 auto;
padding: 26px 24px;
border:1px solid #d7ddd8;
border-radius: 16px;
background:#ffffff;
box-shadow:0 12px 28px rgba(26,39,36,.08);">
  <div style="font-size:1.62rem;font-weight:800;color:#1f2a28;">蓉城智行系统登录</div>
  <div style="margin-top:8px;color:#71807b;">请输入账号与密码进入交通智能决策平台</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    col_l, col_m, col_r = st.columns([1.05, 1.8, 1.05])
    login_btn = False
    register_btn = False
    login_user = ""
    login_pwd = ""
    reg_user = ""
    reg_pwd = ""
    reg_pwd2 = ""
    reg_agree = False

    with col_m:
        t_login, t_register = st.tabs(["登录", "注册"])
        with t_login:
            login_user = st.text_input("账号", value=st.session_state.get("login_user_input", ""), key="login_user_input", placeholder="例如：评审专家")
            login_pwd = st.text_input("密码", value="", type="password", key="login_pwd_input", placeholder="请输入密码")
            login_btn = st.button("登录系统", key="btn_login", use_container_width=True)
            st.caption("演示账号：评审专家（默认口令可使用团队预置账号）")
        with t_register:
            reg_user = st.text_input("注册账号", value="", key="register_user_input", placeholder="2-20位，不含空格")
            reg_pwd = st.text_input("设置密码", value="", type="password", key="register_pwd_input", placeholder="至少6位")
            reg_pwd2 = st.text_input("确认密码", value="", type="password", key="register_pwd_confirm_input", placeholder="再次输入密码")
            reg_agree = st.checkbox("我同意用于本地演示账号管理", value=True, key="register_agree")
            register_btn = st.button("注册账号", key="btn_register", use_container_width=True)
            user_count = len(_load_user_db().get("users", {}))
            st.caption(f"当前可登录账号数：{user_count}")

    if login_btn:
        ok, msg = _verify_user(login_user, login_pwd)
        if ok:
            st.session_state["logged_in"] = True
            st.session_state["login_user"] = login_user.strip()
            st.rerun()
        st.error(msg)

    if register_btn:
        if not reg_agree:
            st.warning("请先勾选同意项后再注册。")
        elif reg_pwd != reg_pwd2:
            st.error("两次输入的密码不一致。")
        else:
            ok, msg = _register_user(reg_user, reg_pwd)
            if ok:
                st.session_state["login_user_input"] = reg_user.strip()
                st.success(msg)
            else:
                st.error(msg)
    return False


@st.cache_data(show_spinner=False)
def load_data_cached(rebuild_nonce: int) -> pd.DataFrame:
    force = rebuild_nonce > 0
    return load_or_build_traffic_data(DATA_DIR, force_regenerate=force, max_days=35).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_hotspots_cached(hotspot_nonce: int) -> pd.DataFrame:
    _ = hotspot_nonce
    return load_hotspots_data(DATA_DIR)


@st.cache_resource(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda d: f"{len(d)}-{str(d['timestamp'].min())}-{str(d['timestamp'].max())}"},
)
def train_baseline_cached(df: pd.DataFrame):
    return train_traffic_model(df)


@st.cache_resource(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda d: f"{len(d)}-{str(d['timestamp'].min())}-{str(d['timestamp'].max())}"},
)
def train_lab_cached(df: pd.DataFrame):
    return train_model_zoo(df)


@st.cache_resource(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda d: f"{len(d)}-{str(d['timestamp'].min())}-{str(d['timestamp'].max())}"},
)
def advanced_cls_cached(df: pd.DataFrame):
    return train_advanced_classifier(df)


@st.cache_resource(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda d: f"{len(d)}-{str(d['timestamp'].min())}-{str(d['timestamp'].max())}"},
)
def advanced_reg_cached(df: pd.DataFrame):
    return train_regression_suite(df)


@st.cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda d: f"{len(d)}-{str(d['timestamp'].min())}-{str(d['timestamp'].max())}"},
)
def bootstrap_cached(df: pd.DataFrame):
    return bootstrap_corridor_statistics(df, top_n=10, n_boot=350)


@st.cache_data(show_spinner=False)
def emergency_network_cached(rebuild_nonce: int):
    _ = rebuild_nonce
    return build_network_data()


@st.cache_data(
    show_spinner=False,
    hash_funcs={
        pd.DataFrame: lambda d: f"{len(d)}-{str(d.iloc[0,0]) if len(d) else '0'}",
    },
)
def emergency_metrics_cached(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, traffic_df: pd.DataFrame):
    return compute_station_metrics(nodes_df, edges_df, traffic_df)


def _build_llm_settings(api_base: str, api_key: str, model: str, timeout: int = 45) -> LLMSettings:
    return LLMSettings(
        api_base=api_base.strip(),
        api_key=api_key.strip(),
        model=model.strip(),
        timeout=int(timeout),
    )


def _scenario_dict(event_boost: float, transit_boost: float, signal_opt: float, incident_ctl: float) -> Dict[str, float]:
    return {
        "event_boost": float(event_boost),
        "transit_boost": float(transit_boost),
        "signal_optimization": float(signal_opt),
        "incident_control": float(incident_ctl),
    }


def _build_corridor_map_df(df: pd.DataFrame, lookback_hours: int = 6) -> pd.DataFrame:
    latest_ts = pd.to_datetime(df["timestamp"]).max()
    recent = df[df["timestamp"] >= latest_ts - pd.Timedelta(hours=lookback_hours - 1)].copy()
    agg = (
        recent.groupby("corridor", as_index=False)
        .agg(
            district=("district", lambda s: s.mode().iloc[0] if not s.mode().empty else str(s.iloc[0])),
            congestion_index=("congestion_index", "mean"),
            avg_speed=("avg_speed", "mean"),
            severe_ratio=("congestion_level", lambda s: float((s == 2).mean())),
            incidents=("incident_count", "sum"),
        )
        .sort_values("congestion_index", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def _route_color(v: float) -> str:
    if v >= 88:
        return "#93514b"
    if v >= 78:
        return "#b77b5a"
    if v >= 67:
        return "#b89b70"
    if v >= 55:
        return "#7e948f"
    return "#4f6f68"


def _route_map(map_df: pd.DataFrame) -> go.Figure:
    score_map = map_df.set_index("corridor")["congestion_index"].to_dict()
    speed_map = map_df.set_index("corridor")["avg_speed"].to_dict()
    severe_map = map_df.set_index("corridor")["severe_ratio"].to_dict()
    district_map = map_df.set_index("corridor")["district"].to_dict()
    incident_map = map_df.set_index("corridor")["incidents"].to_dict()

    fig = go.Figure()

    for corridor, points in ROUTE_GEOMETRY.items():
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        score = float(score_map.get(corridor, 55.0))
        avg_speed = float(speed_map.get(corridor, 40.0))
        severe_ratio = float(severe_map.get(corridor, 0.0))
        incidents = float(incident_map.get(corridor, 0.0))
        district = str(district_map.get(corridor, "-"))
        width = 3.5 + np.clip((score - 50.0) / 18.0, 0.0, 4.8)
        color = _route_color(score)

        # 光晕底线，提升高质感与可读性
        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=width + 4.0, color="rgba(255,255,255,0.72)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=width, color=color),
                name=corridor,
                showlegend=False,
                hovertemplate=(
                    f"<b>{corridor}</b><br>"
                    f"区域: {district}<br>"
                    f"拥堵指数: {score:.1f}<br>"
                    f"严重拥堵占比: {severe_ratio:.0%}<br>"
                    f"平均速度: {avg_speed:.1f} 公里/小时<br>"
                    f"事件数: {incidents:.0f}<extra></extra>"
                ),
            )
        )

        center_idx = len(points) // 2
        fig.add_trace(
            go.Scattermapbox(
                lat=[points[center_idx][0]],
                lon=[points[center_idx][1]],
                mode="markers",
                marker=dict(size=8, color=color, opacity=0.92),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=700,
        margin=dict(l=8, r=8, t=6, b=6),
        mapbox=dict(
            style="white-bg",
            layers=[{"sourcetype": "raster", "source": 高德瓦片源, "below": "traces"}],
            center=dict(lat=30.642, lon=104.077),
            zoom=10.25,
        ),
        paper_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def _emergency_network_map(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> go.Figure:
    pos = node_df.set_index("站点")[["纬度", "经度"]].to_dict("index")
    fig = go.Figure()

    for _, e in edge_df.iterrows():
        a = str(e["起点"])
        b = str(e["终点"])
        if a not in pos or b not in pos:
            continue
        fig.add_trace(
            go.Scattermapbox(
                lat=[float(pos[a]["纬度"]), float(pos[b]["纬度"])],
                lon=[float(pos[a]["经度"]), float(pos[b]["经度"])],
                mode="lines",
                line=dict(width=2.2, color="rgba(79,111,104,0.36)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    normal = node_df[node_df["是否入选"] == 0]
    chosen = node_df[node_df["是否入选"] == 1]
    base = node_df[node_df["是否基线站"] == 1]

    fig.add_trace(
        go.Scattermapbox(
            lat=normal["纬度"],
            lon=normal["经度"],
            mode="markers",
            marker=dict(
                size=(8 + normal["综合重要度"] * 18).clip(8, 21),
                color="#7e948f",
                opacity=0.78,
            ),
            text=normal["站点"],
            customdata=np.stack([normal["区县"], normal["预计救援时间分钟"]], axis=1),
            hovertemplate="<b>%{text}</b><br>区县: %{customdata[0]}<br>预计时间: %{customdata[1]:.1f} 分钟<extra></extra>",
            name="普通站点",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=chosen["纬度"],
            lon=chosen["经度"],
            mode="markers+text",
            marker=dict(
                size=(13 + chosen["综合重要度"] * 20).clip(14, 28),
                color="#93514b",
                opacity=0.95,
            ),
            text=chosen["站点"],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>优化入选站<extra></extra>",
            name="优化入选站",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=base["纬度"],
            lon=base["经度"],
            mode="markers",
            marker=dict(size=16, color="#4f6f68", opacity=0.42),
            hovertemplate="<b>%{text}</b><br>基线站点<extra></extra>",
            text=base["站点"],
            name="基线站点",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=700,
        margin=dict(l=6, r=6, t=6, b=6),
        mapbox=dict(
            style="white-bg",
            layers=[{"sourcetype": "raster", "source": 高德瓦片源, "below": "traces"}],
            center=dict(lat=30.66, lon=104.08),
            zoom=9.95,
        ),
        paper_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def _risk_compare_chart(base_fc: pd.DataFrame, plan_fc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base_fc["timestamp"],
            y=base_fc["risk_score"],
            mode="lines+markers",
            name="基线风险",
            line=dict(color="#9aa6a2", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plan_fc["timestamp"],
            y=plan_fc["risk_score"],
            mode="lines+markers",
            name="策略后风险",
            line=dict(color="#4f6f68", width=3),
            marker=dict(size=5),
        )
    )
    fig.update_layout(title="24小时风险对比", yaxis_title="风险分值(0-100)", xaxis_title="时间")
    return fig


def main() -> None:
    st.set_page_config(page_title="蓉城智行-城市交通智能系统", layout="wide", initial_sidebar_state="collapsed")
    inject_style()

    if not show_login():
        return

    if "rebuild_nonce" not in st.session_state:
        st.session_state["rebuild_nonce"] = 0
    if "hotspot_nonce" not in st.session_state:
        st.session_state["hotspot_nonce"] = 0
    if "lab_ready" not in st.session_state:
        st.session_state["lab_ready"] = False
    if "lab_result" not in st.session_state:
        st.session_state["lab_result"] = None
    if "advanced_result_pack" not in st.session_state:
        st.session_state["advanced_result_pack"] = None
    if "vision_last_result" not in st.session_state:
        st.session_state["vision_last_result"] = None
    if "event_boost" not in st.session_state:
        st.session_state["event_boost"] = 0.45
    if "transit_boost" not in st.session_state:
        st.session_state["transit_boost"] = 0.55
    if "signal_opt" not in st.session_state:
        st.session_state["signal_opt"] = 0.60
    if "incident_ctl" not in st.session_state:
        st.session_state["incident_ctl"] = 0.50
    if "use_llm" not in st.session_state:
        st.session_state["use_llm"] = False
    if "api_base" not in st.session_state:
        st.session_state["api_base"] = ""
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = ""
    if "timeout_s" not in st.session_state:
        st.session_state["timeout_s"] = 45
    if "show_chart_notes" not in st.session_state:
        st.session_state["show_chart_notes"] = True

    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    row_left, row_right = st.columns([1.25, 1.75])
    with row_left:
        st.markdown(
            f'<span class="user-badge">当前用户：{st.session_state.get("login_user", "-")}</span>',
            unsafe_allow_html=True,
        )
    with row_right:
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("重建数据", use_container_width=True):
                st.session_state["rebuild_nonce"] += 1
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        with b2:
            if st.button("更新热点", use_container_width=True):
                try:
                    collect_online_hotspots(DATA_DIR, out_file=HOTSPOT_FILE, max_items=170)
                    st.session_state["hotspot_nonce"] += 1
                    st.cache_data.clear()
                    st.success("在线热点已更新。")
                except Exception as exc:
                    st.warning(f"在线抓取失败，已保留本地热点数据：{exc}")
        with b3:
            if st.button("退出登录", use_container_width=True):
                st.session_state["logged_in"] = False
                st.session_state["login_user"] = ""
                st.rerun()

    p1, p2 = st.columns(2)
    with p1:
        with st.expander("策略参数设置", expanded=False):
            st.session_state["event_boost"] = st.slider("活动冲击强度", 0.0, 1.0, float(st.session_state["event_boost"]), 0.01)
            st.session_state["transit_boost"] = st.slider("公交地铁分流能力", 0.0, 1.0, float(st.session_state["transit_boost"]), 0.01)
            st.session_state["signal_opt"] = st.slider("信号优化强度", 0.0, 1.0, float(st.session_state["signal_opt"]), 0.01)
            st.session_state["incident_ctl"] = st.slider("事故快处能力", 0.0, 1.0, float(st.session_state["incident_ctl"]), 0.01)
    with p2:
        with st.expander("智能问答接口设置", expanded=False):
            st.session_state["use_llm"] = st.toggle("启用智能回答增强", value=bool(st.session_state["use_llm"]))
            st.session_state["api_base"] = st.text_input("接口地址", value=st.session_state.get("api_base", ""))
            st.session_state["api_key"] = st.text_input("接口密钥", value=st.session_state.get("api_key", ""), type="password")
            st.session_state["llm_model"] = st.text_input("模型名称", value=st.session_state.get("llm_model", ""))
            st.session_state["timeout_s"] = int(
                st.number_input("超时（秒）", min_value=10, max_value=120, value=int(st.session_state["timeout_s"]), step=5)
            )
            if st.button("测试接口连接", use_container_width=True):
                settings = _build_llm_settings(
                    st.session_state["api_base"],
                    st.session_state["api_key"],
                    st.session_state["llm_model"],
                    int(st.session_state["timeout_s"]),
                )
                ok, msg = test_llm_connection(settings)
                if ok:
                    st.success("连接成功")
                else:
                    st.error(msg)
    n1, n2 = st.columns([1.05, 1.2])
    with n1:
        st.session_state["show_chart_notes"] = st.toggle(
            "显示图表解读",
            value=bool(st.session_state["show_chart_notes"]),
            help="开启后在关键图下方显示简要洞察。",
        )
    with n2:
        st.caption("建议答辩时开启图表解读；演示走查时可关闭，页面更简洁。")
    st.markdown("</div>", unsafe_allow_html=True)

    event_boost = float(st.session_state["event_boost"])
    transit_boost = float(st.session_state["transit_boost"])
    signal_opt = float(st.session_state["signal_opt"])
    incident_ctl = float(st.session_state["incident_ctl"])
    use_llm = bool(st.session_state["use_llm"])
    api_base = str(st.session_state["api_base"])
    api_key = str(st.session_state["api_key"])
    llm_model = str(st.session_state["llm_model"])
    timeout_s = int(st.session_state["timeout_s"])

    df = load_data_cached(st.session_state["rebuild_nonce"])
    hotspots_df = load_hotspots_cached(st.session_state["hotspot_nonce"])
    model_bundle = train_baseline_cached(df)
    map_df = _build_corridor_map_df(df, lookback_hours=6)
    rank_df = rank_hot_corridors(df, lookback_hours=24)
    corridors = corridor_list()

    tabs = st.tabs(["总览驾驶舱", "模型实验室", "深度预测", "视觉感知", "策略推演", "应急站选址"])

    with tabs[0]:
        render_hero(df, hotspots_df)
        kpis = compute_city_kpis(df)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("近24小时平均速度（公里/小时）", f"{kpis['avg_speed']:.1f}")
        c2.metric("近24小时拥堵指数", f"{kpis['avg_index']:.1f}")
        c3.metric("严重拥堵占比", f"{kpis['severe_ratio']:.1%}")
        c4.metric("近24小时事件总数", f"{kpis['incident_total']:.0f}")
        c5.metric("热点条数", f"{len(hotspots_df)}")

        section_header("成都交通拓扑地图", "最近6小时聚合路网态势")
        _show(_route_map(map_df), key="route-map")
        st.caption("地图底图来源：高德电子地图（中文标注）")
        if not map_df.empty:
            top_row = map_df.sort_values("congestion_index", ascending=False).iloc[0]
            chart_insight(
                "路网态势解读",
                [
                    f"当前拥堵最高走廊为「{top_row['corridor']}」，拥堵指数约 {float(top_row['congestion_index']):.1f}。",
                    "红色线段越粗表示风险越高，适合用于答辩时直观说明治理优先级。",
                    "建议优先结合事件热点与事故点，制定分时分段治理策略。",
                ],
            )

        latest_ts = pd.to_datetime(df["timestamp"]).max()
        line_df = (
            df[df["timestamp"] >= latest_ts - pd.Timedelta(days=7)]
            .groupby("timestamp", as_index=False)["congestion_index"]
            .mean()
            .sort_values("timestamp")
        )
        top_corridors = rank_df.head(8).sort_values("corridor_score", ascending=True)
        heat_df = (
            df[df["timestamp"] >= latest_ts - pd.Timedelta(hours=71)]
            .groupby(["corridor", "hour"], as_index=False)["congestion_index"]
            .mean()
        )

        x1, x2 = st.columns([1.45, 1.0])
        with x1:
            fig_line = px.line(
                line_df,
                x="timestamp",
                y="congestion_index",
                title="城市拥堵指数趋势（最近7天）",
                markers=True,
                color_discrete_sequence=[主题主色],
                labels={"timestamp": "时间", "congestion_index": "拥堵指数"},
            )
            _show(fig_line, key="city-line")
            if not line_df.empty:
                chart_insight(
                    "趋势图解读",
                    [
                        f"最近7天全市平均拥堵指数波动区间约 {float(line_df['congestion_index'].min()):.1f} - {float(line_df['congestion_index'].max()):.1f}。",
                        "曲线的上升段通常对应通勤高峰和活动时段叠加。",
                    ],
                )
        with x2:
            fig_rank = px.bar(
                top_corridors,
                x="corridor_score",
                y="corridor",
                orientation="h",
                color="severe_ratio",
                color_continuous_scale=统一连续色阶,
                title="高风险走廊排名（近24小时）",
                labels={"corridor_score": "综合风险分", "corridor": "走廊", "severe_ratio": "严重拥堵占比"},
            )
            _show(fig_rank, key="rank-bar")
            if not top_corridors.empty:
                chart_insight(
                    "高风险走廊解读",
                    [
                        f"高风险前3走廊为：{'、'.join(top_corridors.sort_values('corridor_score', ascending=False).head(3)['corridor'].astype(str).tolist())}。",
                        "综合风险分由拥堵、严重占比、事件数共同决定，便于解释资源投放依据。",
                    ],
                )

        h1, h2 = st.columns([1.2, 1.0])
        with h1:
            hour_axis = list(range(24))
            heat_mat = heat_df.pivot_table(index="corridor", columns="hour", values="congestion_index", aggfunc="mean")
            heat_mat = heat_mat.reindex(columns=hour_axis)

            ordered_corridors = [str(c) for c in rank_df["corridor"].tolist() if str(c) in heat_mat.index]
            for c_name in heat_mat.index.tolist():
                if c_name not in ordered_corridors:
                    ordered_corridors.append(str(c_name))
            heat_mat = heat_mat.reindex(ordered_corridors)

            if heat_mat.isna().any().any():
                row_mean = heat_mat.mean(axis=1)
                heat_mat = heat_mat.apply(lambda r: r.fillna(row_mean.get(r.name, np.nan)), axis=1)
                heat_mat = heat_mat.fillna(float(np.nanmean(heat_mat.values)))

            fig_heat = go.Figure(
                go.Heatmap(
                    z=heat_mat.values,
                    x=hour_axis,
                    y=heat_mat.index.tolist(),
                    colorscale=[
                        [0.0, "#f6f7f5"],
                        [0.2, "#dde5e2"],
                        [0.45, "#b8c9c4"],
                        [0.7, "#7f9690"],
                        [1.0, "#4f6f68"],
                    ],
                    zmin=float(np.nanmin(heat_mat.values)),
                    zmax=float(np.nanmax(heat_mat.values)),
                    xgap=1.2,
                    ygap=1.2,
                    colorbar=dict(title="拥堵指数", thickness=14, len=0.88),
                    hovertemplate="走廊：%{y}<br>小时：%{x}:00<br>拥堵指数：%{z:.1f}<extra></extra>",
                )
            )
            fig_heat.update_layout(
                title={"text": "走廊分时拥堵热力图（24小时）", "x": 0.01, "xanchor": "left", "font": {"size": 22, "color": "#253430"}},
                xaxis_title="小时",
                yaxis_title="走廊",
                height=480,
                margin=dict(l=26, r=12, t=74, b=58),
                font=dict(color="#2c3b37"),
                plot_bgcolor="#ffffff",
            )
            fig_heat.update_xaxes(
                tickmode="array",
                tickvals=[0, 3, 6, 9, 12, 15, 18, 21],
                ticktext=["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
                showgrid=False,
                linecolor="#c6d1cd",
                tickfont=dict(size=12, color="#43554f"),
            )
            fig_heat.update_yaxes(showgrid=False, linecolor="#c6d1cd", tickfont=dict(size=12, color="#43554f"))
            _show(fig_heat, key="heatmap")
            chart_insight(
                "分时热力解读",
                [
                    "横向看单行可识别单走廊全天拥堵节律，纵向看同一时刻可比较多走廊风险。",
                    "建议将 07:00-09:00 与 17:00-19:00 作为信号优化与分流重点窗口。",
                ],
            )
        with h2:
            section_header("在线热点摘要", "实时事件提要与影响评分")
            if hotspots_df.empty:
                st.info("暂无在线热点数据。")
            else:
                hot_show = _hotspot_table_cn(hotspots_df)
                st.dataframe(hot_show, use_container_width=True, height=325, hide_index=True)

    with tabs[1]:
        section_header("模型实验室", "多模型分类、回归与统计推断一体化评估")
        st.markdown('<div class="note">本页是硬核算法中心：基础模型、集成模型、回归模型、置信区间分析统一展示。</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("基线分类准确率", f"{model_bundle.metrics['accuracy']:.3f}")
        m2.metric("基线宏平均分", f"{model_bundle.metrics['macro_f1']:.3f}")
        m3.metric("测试样本量", f"{int(model_bundle.metrics['test_size'])}")

        with st.expander("查看基线模型特征贡献", expanded=True):
            imp = model_bundle.feature_importance.head(16).copy()
            imp["feature_cn"] = imp["feature"].astype(str).map(_feature_name_cn)
            fig_imp = px.bar(
                imp.sort_values("importance", ascending=True),
                x="importance",
                y="feature_cn",
                orientation="h",
                color="importance",
                color_continuous_scale=统一连续色阶,
                title="基线随机森林特征重要性",
                labels={"importance": "重要度", "feature_cn": "特征"},
            )
            _show(fig_imp, key="base-imp")

        st.markdown("---")
        if st.button("启动基础模型对比", use_container_width=False):
            st.session_state["lab_ready"] = True
            st.session_state["lab_result"] = None

        lab = st.session_state.get("lab_result")
        if st.session_state["lab_ready"] and lab is None:
            lab_prog = st.progress(5, text="基础模型对比：准备中")
            lab_note = st.empty()
            lab_note.caption("正在加载数据并执行训练，首次运行会稍慢……")
            lab = train_lab_cached(df)
            lab_prog.progress(100, text="基础模型对比：已完成")
            lab_note.success("基础模型对比计算完成。")
            st.session_state["lab_result"] = lab

        if st.session_state["lab_ready"] and lab is not None:
            base_board = lab.leaderboard.copy()
            if "模型" in base_board.columns:
                base_board["模型"] = base_board["模型"].astype(str).map(lambda x: 基础模型中文名.get(x, x))
            base_board = base_board.rename(columns={"Accuracy": "准确率", "Macro-F1": "宏平均分", "训练时长(s)": "训练时长(秒)"})
            st.dataframe(base_board, use_container_width=True, hide_index=True)

            model_col = str(lab.leaderboard.columns[0])
            raw_models = [str(v) for v in lab.leaderboard[model_col].tolist()]
            cn_to_raw: Dict[str, str] = {}
            for raw_name in raw_models:
                cn_name = 基础模型中文名.get(raw_name, raw_name)
                cn_to_raw[cn_name] = raw_name
            selected_model_cn = st.selectbox("查看基础模型细节", options=list(cn_to_raw.keys()))
            selected_model = cn_to_raw[selected_model_cn]
            cm_df = lab.confusion[selected_model]
            fig_cm = px.imshow(
                cm_df.values,
                x=cm_df.columns,
                y=cm_df.index,
                text_auto=True,
                color_continuous_scale=统一连续色阶,
                title=f"{selected_model_cn} 混淆矩阵",
            )
            _show(fig_cm, key="cm")

            if selected_model == "RandomForest":
                rf_pipe = lab.model_cache["RandomForest"]
                rf_imp = feature_importance_from_rf(rf_pipe).head(18)
                rf_imp["feature_cn"] = rf_imp["feature"].astype(str).map(_feature_name_cn)
                fig_rf = px.bar(
                    rf_imp.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature_cn",
                    orientation="h",
                    color="importance",
                    color_continuous_scale=统一连续色阶,
                    title="随机森林可解释特征图谱",
                    labels={"importance": "重要度", "feature_cn": "特征"},
                )
                _show(fig_rf, key="rf-imp")
        else:
            st.info("点击“启动基础模型对比”后查看随机森林、梯度提升与神经网络结果。")

        st.markdown("---")
        section_header("高级算法评估", "高精度加权集成 + 回归模型 + 自助法置信区间")
        if st.button("启动高级算法评估", use_container_width=False):
            st.session_state["advanced_ready"] = True
            st.session_state["advanced_result_pack"] = None

        advanced_pack = st.session_state.get("advanced_result_pack")
        if st.session_state.get("advanced_ready", False) and advanced_pack is None:
            adv_prog = st.progress(5, text="高级算法评估：准备中")
            adv_note = st.empty()
            adv_note.caption("步骤 1/3：分类模型评估")
            cls_res = advanced_cls_cached(df)
            adv_prog.progress(40, text="高级算法评估：分类模型完成")
            adv_note.caption("步骤 2/3：回归模型评估")
            reg_res = advanced_reg_cached(df)
            adv_prog.progress(75, text="高级算法评估：回归模型完成")
            adv_note.caption("步骤 3/3：自助法置信区间")
            boot_df = bootstrap_cached(df)
            adv_prog.progress(100, text="高级算法评估：已完成")
            adv_note.success("高级算法评估计算完成。")
            advanced_pack = {"cls_res": cls_res, "reg_res": reg_res, "boot_df": boot_df}
            st.session_state["advanced_result_pack"] = advanced_pack

        if st.session_state.get("advanced_ready", False) and advanced_pack is not None:
            cls_res = advanced_pack["cls_res"]
            reg_res = advanced_pack["reg_res"]
            boot_df = advanced_pack["boot_df"]
            st.markdown("**分类模型排行榜**")
            st.dataframe(cls_res.排行榜, use_container_width=True, hide_index=True)

            choose_cls = st.selectbox("查看分类模型混淆矩阵", options=cls_res.排行榜["模型"].tolist(), key="cls-matrix")
            cls_cm = cls_res.混淆矩阵[choose_cls]
            fig_cls_cm = px.imshow(
                cls_cm.values,
                x=cls_cm.columns,
                y=cls_cm.index,
                text_auto=True,
                color_continuous_scale=统一连续色阶,
                title=f"{choose_cls} 混淆矩阵",
            )
            _show(fig_cls_cm, key="cls-cm")

            weight_df = pd.DataFrame(
                [{"模型": k, "权重": v} for k, v in cls_res.集成权重.items()]
            ).sort_values("权重", ascending=False)
            fig_w = px.bar(weight_df, x="模型", y="权重", title="高精度加权集成权重", color="权重", color_continuous_scale=统一连续色阶)
            _show(fig_w, key="ensemble-w")

            st.markdown("**回归模型排行榜**")
            st.dataframe(reg_res.排行榜, use_container_width=True, hide_index=True)
            best_reg = str(reg_res.排行榜.iloc[0]["模型"])
            reg_curve = reg_res.预测序列[best_reg].copy().tail(180)
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=reg_curve["时间"], y=reg_curve["真实值"], mode="lines", name="真实值", line=dict(color="#7e948f", width=2)))
            fig_reg.add_trace(go.Scatter(x=reg_curve["时间"], y=reg_curve["预测值"], mode="lines", name="预测值", line=dict(color="#93514b", width=2)))
            fig_reg.update_layout(title=f"{best_reg} 预测拟合曲线", yaxis_title="拥堵指数", xaxis_title="时间")
            _show(fig_reg, key="reg-fit")

            if not boot_df.empty:
                fig_boot = go.Figure()
                fig_boot.add_trace(
                    go.Scatter(
                        x=boot_df["拥堵均值"],
                        y=boot_df["走廊"],
                        mode="markers",
                        marker=dict(size=10, color="#4f6f68"),
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=(boot_df["95%上界"] - boot_df["拥堵均值"]).values,
                            arrayminus=(boot_df["拥堵均值"] - boot_df["95%下界"]).values,
                            color="#8ea09b",
                            thickness=1.2,
                        ),
                    )
                )
                fig_boot.update_layout(title="走廊拥堵均值的95%置信区间（自助抽样）", xaxis_title="拥堵指数", yaxis_title="走廊")
                _show(fig_boot, key="boot-ci")
                st.dataframe(boot_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        section_header("深度时序预测与异常检测", "多步预测与异常识别联动")
        selected_corridor = st.selectbox("预测走廊", corridors, index=0, key="deep-corridor")
        horizon = st.slider("预测时长（小时）", 12, 48, 24, 1, key="deep-horizon")

        deep_fc, meta = corridor_deep_forecast(df, selected_corridor, horizon=horizon, lag=8)
        history = df[df["corridor"] == selected_corridor].sort_values("timestamp").tail(120)

        fig_deep = go.Figure()
        fig_deep.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history["congestion_index"],
                mode="lines",
                name="历史拥堵指数",
                line=dict(color="#7e948f", width=2),
            )
        )
        if not deep_fc.empty:
            fig_deep.add_trace(
                go.Scatter(
                    x=deep_fc["timestamp"],
                    y=deep_fc["pred_congestion_index"],
                    mode="lines+markers",
                    name="深度预测",
                    line=dict(color="#4f6f68", width=3, dash="dot"),
                    marker=dict(size=6),
                )
            )
        fig_deep.update_layout(title=f"{selected_corridor} 时序预测曲线", yaxis_title="拥堵指数", xaxis_title="时间")
        _show(fig_deep, key="deep-fc")
        chart_insight(
            "时序预测解读",
            [
                "深色虚线为未来预测走势，可直观看到潜在风险抬升时段。",
                "建议在预测峰值前 1-2 小时预置信号配时和分流方案。",
            ],
        )

        if deep_fc.empty:
            st.warning("该走廊可用样本不足，暂时无法完成深度时序建模。")
        else:
            severe_hours = int((deep_fc["pred_level"].astype(str) == "拥堵").sum())
            peak = float(deep_fc["pred_congestion_index"].max())
            d1, d2, d3 = st.columns(3)
            d1.metric("测试集平均绝对误差", f"{meta['mae']:.2f}")
            d2.metric("未来拥堵小时数", f"{severe_hours}")
            d3.metric("预测峰值", f"{peak:.1f}")

        an_df = detect_anomaly_points(df, selected_corridor)
        if not an_df.empty:
            show_an = an_df.tail(240).copy()
            show_an["状态"] = np.where(show_an["is_anomaly"] == 1, "异常", "正常")
            # Plotly 的 marker size 要求非负，做平移缩放避免运行时报错
            min_score = float(show_an["anomaly_score"].min())
            show_an["anomaly_size"] = ((show_an["anomaly_score"] - min_score) + 0.2) * 7.0
            fig_an = px.scatter(
                show_an,
                x="timestamp",
                y="congestion_index",
                color="状态",
                size="anomaly_size",
                size_max=16,
                color_discrete_map={"正常": 主题次色, "异常": 主题强调},
                title="异常拥堵点识别（孤立森林）",
                labels={"timestamp": "时间", "congestion_index": "拥堵指数", "anomaly_size": "异常强度"},
            )
            _show(fig_an, key="anomaly")

    with tabs[3]:
        section_header("视觉感知模块（目标检测）", "实时检测、YOLO多版本对比、数据集抓取导入与导出")
        st.markdown(
            '<div class="note">本模块已升级为可落地流程：支持模型对比、在线抓取候选数据集、去重导入、历史记录与压缩导出。</div>',
            unsafe_allow_html=True,
        )
        sample_paths = _ensure_vision_demo_samples()
        vision_tabs = st.tabs(["实时检测", "模型对比", "数据集管理"])

        with vision_tabs[0]:
            c1, c2 = st.columns([1.05, 0.95])
            with c1:
                conf = st.slider("检测置信度阈值", 0.1, 0.9, 0.25, 0.05, key="vision-conf-single")
                model_name = st.text_input("检测模型版本", value="yolo11n.pt", key="vision-model-single")
                dev_id, dev_name, has_gpu = detect_available_device()
                dev_options = ["自动", "CPU"] + (["GPU"] if has_gpu else [])
                pref = str(os.getenv("VISION_DEVICE_DEFAULT", "auto")).strip().lower()
                default_idx = 0
                if pref in {"gpu", "cuda", "cuda:0"} and "GPU" in dev_options:
                    default_idx = dev_options.index("GPU")
                elif pref == "cpu":
                    default_idx = dev_options.index("CPU")
                dev_choice = st.selectbox("推理设备", options=dev_options, index=default_idx, key="vision-device-single")
                if has_gpu:
                    st.caption(f"已检测到GPU：{dev_name}")
                else:
                    st.caption("未检测到可用GPU，当前使用CPU。")
                if dev_choice == "GPU":
                    run_device = "cuda:0"
                elif dev_choice == "CPU":
                    run_device = "cpu"
                else:
                    run_device = "auto"

                speed_mode = st.select_slider("推理速度模式", options=["极速", "平衡", "高精度"], value="平衡", key="vision-speed-mode")
                if speed_mode == "极速":
                    max_infer_edge, imgsz = 640, 640
                elif speed_mode == "高精度":
                    max_infer_edge, imgsz = 1280, 1280
                else:
                    max_infer_edge, imgsz = 960, 960
                image_file = st.file_uploader(
                    "上传交通场景图片（支持常见图片格式）",
                    type=["jpg", "jpeg", "png"],
                    key="vision-uploader-single",
                )
            with c2:
                sample_display_map = {p.name: _vision_sample_display_name(p) for p in sample_paths}
                options = ["不使用示例图"] + [p.name for p in sample_paths]
                selected_sample = st.selectbox(
                    "或选择内置示例图",
                    options=options,
                    format_func=lambda x: sample_display_map.get(x, x),
                    key="vision-sample-single",
                )
                use_sample = st.button("使用示例图进行识别", use_container_width=True, key="vision-use-sample-single")
                if selected_sample != "不使用示例图":
                    preview_path = next((p for p in sample_paths if p.name == selected_sample), None)
                    if preview_path is not None:
                        st.image(
                            preview_path.read_bytes(),
                            caption=f"当前选择：{sample_display_map.get(preview_path.name, preview_path.name)}",
                            use_container_width=True,
                        )

            with st.expander("查看示例图库（点击可先预览效果）", expanded=False):
                gallery_cols = st.columns(4)
                for idx, p in enumerate(sample_paths):
                    with gallery_cols[idx % 4]:
                        st.image(p.read_bytes(), caption=_vision_sample_display_name(p), use_container_width=True)

            image_bytes: bytes | None = None
            image_name = "上传图像"
            if image_file is not None:
                image_bytes = image_file.getvalue()
                image_name = image_file.name
            elif use_sample and selected_sample != "不使用示例图":
                chosen = next((p for p in sample_paths if p.name == selected_sample), None)
                if chosen is not None:
                    image_bytes = chosen.read_bytes()
                    image_name = chosen.name
                    st.info(f"已加载示例图：{chosen.name}")

            if image_bytes is not None:
                st.caption("图像已准备就绪，点击下方“开始识别”执行推理。")
            else:
                st.info("请上传图片或选择示例图后再识别。")

            run_single_detect = st.button("开始识别", use_container_width=True, key="vision-run-single")
            if run_single_detect:
                if image_bytes is None:
                    st.warning("当前没有可识别图像，请先上传或选择示例图。")
                else:
                    single_bar = st.progress(3, text="识别任务：准备中")
                    single_note = st.empty()

                    def _single_progress(pct: int, detail: str) -> None:
                        single_bar.progress(max(3, min(99, int(pct))), text=f"识别任务：{max(3, min(99, int(pct)))}%")
                        single_note.caption(detail)

                    result = run_vision_detection(
                        image_bytes,
                        conf=conf,
                        model_name=model_name.strip() or "yolo11n.pt",
                        max_infer_edge=max_infer_edge,
                        imgsz=imgsz,
                        device=run_device,
                        progress_callback=_single_progress,
                    )
                    single_bar.progress(100, text="识别任务：已完成")
                    single_note.success("识别完成。")

                    st.session_state["vision_last_result"] = {
                        "result": result,
                        "image_name": image_name,
                        "run_device": run_device,
                    }

                    total_targets = int(result.object_table["??"].sum()) if not result.object_table.empty else 0
                    avg_conf = float(result.object_table["平均置信度"].mean()) if not result.object_table.empty else 0.0
                    history = st.session_state.setdefault("vision_detect_history", [])
                    history.append(
                        {
                            "??": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "??": image_name,
                            "模型版本": result.model_name,
                            "推理设备": run_device,
                            "运行状态": result.engine,
                            "检测目标数": total_targets,
                            "平均置信度": round(avg_conf, 3),
                            "推理耗时毫秒": round(float(result.latency_ms), 1),
                        }
                    )
                    st.session_state["vision_detect_history"] = history[-120:]

            last_pack = st.session_state.get("vision_last_result")
            if isinstance(last_pack, dict) and last_pack.get("result") is not None:
                result = last_pack["result"]
                img_left, img_right = st.columns(2)
                with img_left:
                    st.image(result.original_image_bytes, caption="原始图像", use_container_width=True)
                with img_right:
                    st.image(result.annotated_image_bytes, caption="目标检测结果（带框）", use_container_width=True)
                _show(result.figure, key="vision-fig-single")
                st.dataframe(result.object_table, use_container_width=True, hide_index=True)

                total_targets = int(result.object_table["??"].sum()) if not result.object_table.empty else 0
                avg_conf = float(result.object_table["平均置信度"].mean()) if not result.object_table.empty else 0.0
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("模型版本", result.model_name)
                m2.metric("检测目标数", f"{total_targets}")
                m3.metric("平均置信度", f"{avg_conf:.3f}")
                m4.metric("推理耗时（毫秒）", f"{result.latency_ms:.1f}")
                st.download_button(
                    "下载检测结果图（PNG）",
                    data=result.annotated_image_bytes,
                    file_name=f"检测结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    key=f"vision-download-{datetime.now().strftime('%H%M%S%f')}",
                )

                if result.engine == "高精度识别引擎":
                    if total_targets > 0:
                        st.success("推理状态：已完成高精度目标识别。")
                    else:
                        st.info("推理状态：模型已运行，但在当前阈值下未检出目标。")
                else:
                    st.warning("推理状态：当前环境未加载可用高精度模型，已停止绘制演示框。建议切换到 GPU 环境后重试。")
            history_df = pd.DataFrame(st.session_state.get("vision_detect_history", []))
            if not history_df.empty:
                st.markdown("**检测记录（最近120次）**")
                st.dataframe(history_df.sort_values("时间", ascending=False), use_container_width=True, hide_index=True)
                hist_csv = history_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("导出检测记录CSV", data=hist_csv, file_name="视觉检测记录.csv", mime="text/csv")

            chart_insight(
                "实时检测解读",
                [
                    "本页支持直接切换模型版本，便于在答辩现场演示不同 YOLO 版本的推理差异。",
                    "检测记录会自动留痕并支持CSV导出，可作为实验过程证据和复盘材料。",
                ],
            )

        with vision_tabs[1]:
            left, right = st.columns([1.1, 0.9])
            with left:
                model_options = [
                    "yolov8n.pt",
                    "yolov8s.pt",
                    "yolov8m.pt",
                    "yolov9t.pt",
                    "yolov9s.pt",
                    "yolov9m.pt",
                    "yolov10n.pt",
                    "yolov10s.pt",
                    "yolov10m.pt",
                    "yolo11n.pt",
                    "yolo11s.pt",
                    "yolo11m.pt",
                ]
                preset_models = st.multiselect(
                    "预置模型版本（跨代可多选）",
                    options=model_options,
                    default=["yolov8n.pt", "yolov9s.pt", "yolov10n.pt", "yolo11n.pt"],
                    key="vision-compare-models",
                )
                extra_models_text = st.text_input(
                    "补充模型版本或别名（用逗号分隔）",
                    value="v8s,v9c,v10s,v11s",
                    key="vision-compare-extra-models",
                    help="示例：v8n,v9s,v10m,v11x 或 yolov8n.pt,yolov9c.pt",
                )
                conf_compare = st.slider("对比阈值", 0.1, 0.9, 0.25, 0.05, key="vision-conf-compare")
                st.caption("建议答辩默认展示：v8n / v9s / v10n / v11n 的同图对比。")
                dev_id_cmp, dev_name_cmp, has_gpu_cmp = detect_available_device()
                dev_options_cmp = ["自动", "CPU"] + (["GPU"] if has_gpu_cmp else [])
                pref_cmp = str(os.getenv("VISION_DEVICE_DEFAULT", "auto")).strip().lower()
                default_idx_cmp = 0
                if pref_cmp in {"gpu", "cuda", "cuda:0"} and "GPU" in dev_options_cmp:
                    default_idx_cmp = dev_options_cmp.index("GPU")
                elif pref_cmp == "cpu":
                    default_idx_cmp = dev_options_cmp.index("CPU")
                dev_choice_cmp = st.selectbox("对比推理设备", options=dev_options_cmp, index=default_idx_cmp, key="vision-device-compare")
                if dev_choice_cmp == "GPU":
                    compare_device = "cuda:0"
                elif dev_choice_cmp == "CPU":
                    compare_device = "cpu"
                else:
                    compare_device = "auto"
                speed_mode_cmp = st.select_slider("对比速度模式", options=["极速", "平衡", "高精度"], value="平衡", key="vision-speed-mode-compare")
                if speed_mode_cmp == "极速":
                    max_infer_edge_cmp, imgsz_cmp = 640, 640
                elif speed_mode_cmp == "高精度":
                    max_infer_edge_cmp, imgsz_cmp = 1280, 1280
                else:
                    max_infer_edge_cmp, imgsz_cmp = 960, 960
            with right:
                compare_uploads = st.file_uploader(
                    "上传用于对比的多张图片",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    key="vision-uploader-compare",
                )
                default_sample_set = [p.name for p in sample_paths]
                selected_compare_samples = st.multiselect(
                    "或追加示例图进入对比集",
                    options=default_sample_set,
                    default=default_sample_set,
                    format_func=lambda x: _vision_sample_display_name(Path(x)),
                    key="vision-compare-samples",
                )

            compare_items: List[tuple[str, bytes]] = []
            for f in compare_uploads or []:
                try:
                    compare_items.append((f.name, f.getvalue()))
                except Exception:
                    continue
            for name in selected_compare_samples:
                p = next((x for x in sample_paths if x.name == name), None)
                if p is not None:
                    compare_items.append((name, p.read_bytes()))

            extra_models = [x.strip() for x in str(extra_models_text).replace("，", ",").split(",") if x.strip()]
            model_versions = []
            for m in list(preset_models) + extra_models:
                if m not in model_versions:
                    model_versions.append(m)

            compare_btn = st.button("开始YOLO多版本对比", use_container_width=True, key="vision-compare-run")
            if compare_btn:
                if not compare_items:
                    st.warning("请至少提供1张对比图片。")
                elif not model_versions:
                    st.warning("请至少选择1个模型版本。")
                else:
                    cmp_bar = st.progress(2, text="模型对比任务：准备中")
                    cmp_note = st.empty()

                    def _compare_progress(done: int, total: int, detail: str) -> None:
                        pct = 2 if total <= 0 else max(2, min(99, int(done * 100 / total)))
                        cmp_bar.progress(pct, text=f"模型对比任务：{pct}%")
                        cmp_note.caption(f"{detail}?{done}/{total}?")

                    board = compare_yolo_versions(
                        compare_items,
                        model_versions=model_versions,
                        conf=conf_compare,
                        max_infer_edge=max_infer_edge_cmp,
                        imgsz=imgsz_cmp,
                        device=compare_device,
                        progress_callback=_compare_progress,
                    )
                    cmp_bar.progress(100, text="模型对比任务：已完成")
                    cmp_note.success("YOLO 多版本对比完成。")
                    if board.empty:
                        st.warning("未生成可用对比结果，请检查输入图片和模型版本。")
                    else:
                        st.session_state["vision_compare_board"] = board

            board_df = st.session_state.get("vision_compare_board")
            if isinstance(board_df, pd.DataFrame) and not board_df.empty:
                st.dataframe(board_df, use_container_width=True, hide_index=True)

                fig_latency = px.bar(
                    board_df,
                    x="模型版本",
                    y="平均耗时毫秒",
                    color="运行状态",
                    title="不同模型推理耗时对比",
                    color_discrete_map={"高精度识别": 主题主色, "部分回退": 主题暖色, "演示回退": 主题次色},
                )
                _show(fig_latency, key="vision-compare-latency")

                fig_obj = px.bar(
                    board_df,
                    x="模型版本",
                    y="检测总目标数",
                    color="模型版本",
                    title="不同模型检测目标总量对比",
                    color_discrete_sequence=统一连续色阶,
                )
                _show(fig_obj, key="vision-compare-objects")

                fig_tradeoff = px.scatter(
                    board_df,
                    x="平均耗时毫秒",
                    y="平均置信度",
                    color="运行状态",
                    size="检测总目标数",
                    hover_name="模型版本",
                    title="模型精度-速度权衡分布",
                    color_discrete_map={"高精度识别": 主题主色, "部分回退": 主题暖色, "演示回退": 主题次色},
                    labels={"平均耗时毫秒": "平均耗时(ms)", "平均置信度": "平均置信度"},
                )
                _show(fig_tradeoff, key="vision-compare-tradeoff")

                chart_insight(
                    "模型对比解读",
                    [
                        "建议优先关注“平均置信度高且耗时适中”的模型，用于线上部署更稳妥。",
                        "若运行状态出现“演示回退”，说明当前环境缺少对应权重或依赖，可作为后续部署优化点。",
                    ],
                )
            else:
                st.info("点击“开始YOLO多版本对比”后查看结果。")

        with vision_tabs[2]:
            st.markdown("**在线抓取与导入管理**")
            d1, d2 = st.columns([1.0, 1.3])
            with d1:
                if st.button("抓取候选数据集", use_container_width=True, key="dataset-crawl"):
                    with st.spinner("正在抓取可用数据源..."):
                        cand = crawl_detection_dataset_sources(timeout_s=20, max_items=120)
                    st.session_state["vision_dataset_candidates"] = cand
                    st.success(f"抓取完成，共 {len(cand)} 条候选记录。")

                url_input = st.text_input("手动导入链接", value="", key="dataset-import-url")
                alias_input = st.text_input("导入别名（可选）", value="", key="dataset-import-alias")
                if st.button("导入链接数据集", use_container_width=True, key="dataset-import-url-btn"):
                    prog = st.progress(4, text="链接导入：准备中")
                    note = st.empty()

                    def _url_prog(pct: int, detail: str) -> None:
                        prog.progress(max(4, min(99, int(pct))), text=f"链接导入：{max(4, min(99, int(pct)))}%")
                        note.caption(detail)

                    ok, msg, _ = import_dataset_from_url(DATA_DIR, url=url_input, alias=alias_input, progress_callback=_url_prog)
                    prog.progress(100, text="链接导入：已完成")
                    if ok:
                        st.success(msg)
                    else:
                        st.warning(msg)

                kaggle_slug = st.text_input(
                    "Kaggle 标识（username/dataset-name）",
                    value="",
                    key="dataset-kaggle-slug",
                )
                kaggle_alias = st.text_input("Kaggle 导入别名（可选）", value="", key="dataset-kaggle-alias")
                st.caption("提示：首次使用 Kaggle 导入前，请先在本机完成 Kaggle API 配置。")
                if st.button("导入 Kaggle 数据集", use_container_width=True, key="dataset-import-kaggle-btn"):
                    prog_kg = st.progress(4, text="Kaggle导入：准备中")
                    note_kg = st.empty()

                    def _kg_prog(pct: int, detail: str) -> None:
                        prog_kg.progress(max(4, min(99, int(pct))), text=f"Kaggle导入：{max(4, min(99, int(pct)))}%")
                        note_kg.caption(detail)

                    ok, msg, _ = import_dataset_from_kaggle(
                        DATA_DIR,
                        dataset_slug=kaggle_slug,
                        alias=kaggle_alias,
                        progress_callback=_kg_prog,
                    )
                    prog_kg.progress(100, text="Kaggle导入：已完成")
                    if ok:
                        st.success(msg)
                    else:
                        st.warning(msg)

                if st.button("扫描 raw 目录并入库", use_container_width=True, key="dataset-scan-raw-btn"):
                    added_n, skipped_n, notices = scan_raw_datasets(DATA_DIR)
                    if added_n > 0:
                        st.success(f"扫描完成：新增 {added_n} 个，跳过 {skipped_n} 个。")
                    else:
                        st.info(f"扫描完成：未新增，跳过 {skipped_n} 个。")
                    for x in notices[:3]:
                        st.caption(x)

                upload_files = st.file_uploader(
                    "本地导入压缩包（支持多选）",
                    type=["zip", "rar", "7z", "tar", "gz", "tgz"],
                    accept_multiple_files=True,
                    key="dataset-upload-files",
                )
                upload_alias = st.text_input("本地导入统一别名前缀（可选）", value="", key="dataset-upload-alias")
                if st.button("导入本地数据集", use_container_width=True, key="dataset-upload-btn"):
                    files = upload_files or []
                    if not files:
                        st.warning("请至少选择一个本地压缩包。")
                    else:
                        success_n = 0
                        warn_msgs: List[str] = []
                        for idx, f in enumerate(files, start=1):
                            alias = f"{upload_alias}_{idx}" if upload_alias else ""
                            ok, msg, _ = import_dataset_from_upload(DATA_DIR, file_name=f.name, file_bytes=f.getvalue(), alias=alias)
                            if ok:
                                success_n += 1
                            else:
                                warn_msgs.append(f"{f.name}: {msg}")
                        if success_n:
                            st.success(f"本地导入完成，成功 {success_n} 个文件。")
                        for wm in warn_msgs[:5]:
                            st.warning(wm)

            with d2:
                candidates = st.session_state.get("vision_dataset_candidates")
                if not isinstance(candidates, pd.DataFrame) or candidates.empty:
                    candidates = pd.DataFrame(
                        [
                            {
                                "数据集名称": x["名称"],
                                "下载链接": x["链接"],
                                "来源": "官方入口",
                                "类型": "交通检测",
                                "备注": x["说明"],
                                "Kaggle标识": "",
                            }
                            for x in DATASET_BACKUP
                        ]
                    )
                st.dataframe(candidates, use_container_width=True, hide_index=True)

                if not candidates.empty and "下载链接" in candidates.columns:
                    labels = [
                        f"{row.get('数据集名称', '未命名')}｜{row.get('来源', '未知来源')}"
                        for _, row in candidates.iterrows()
                    ]
                    quick_index = st.selectbox("快速导入候选源", options=list(range(len(labels))), format_func=lambda i: labels[i], key="dataset-quick-index")
                    if st.button("导入当前候选源", use_container_width=True, key="dataset-quick-btn"):
                        row = candidates.iloc[int(quick_index)]
                        quick_url = str(row.get("下载链接", "")).strip()
                        quick_alias = str(row.get("数据集名称", "")).strip()
                        quick_kaggle = str(row.get("Kaggle标识", "")).strip()
                        if quick_kaggle and quick_kaggle != "username/dataset-name":
                            ok, msg, _ = import_dataset_from_kaggle(DATA_DIR, dataset_slug=quick_kaggle, alias=quick_alias)
                        else:
                            ok, msg, _ = import_dataset_from_url(DATA_DIR, url=quick_url, alias=quick_alias)
                        if ok:
                            st.success(msg)
                        else:
                            st.warning(msg)

            st.markdown("**已导入数据集台账**")
            registry_df = load_dataset_registry_df(DATA_DIR)
            if registry_df.empty:
                st.info("当前还没有导入记录。")
            else:
                st.dataframe(registry_df, use_container_width=True, hide_index=True)
                registry_labels = {
                    str(r["编号"]): f"{r['数据集名称']}（{r['文件大小(MB)']}MB）"
                    for _, r in registry_df.iterrows()
                }
                selected_ids = st.multiselect(
                    "选择要导出的数据集",
                    options=list(registry_labels.keys()),
                    format_func=lambda x: registry_labels.get(x, x),
                    key="dataset-export-ids",
                )
                bundle_name = st.text_input("导出压缩包名称", value="目标检测数据集打包", key="dataset-export-name")
                if st.button("导出选中数据集", use_container_width=True, key="dataset-export-btn"):
                    ok, msg, out_path = export_dataset_bundle(DATA_DIR, selected_ids=selected_ids, bundle_name=bundle_name)
                    if ok and out_path is not None and out_path.exists():
                        st.success(f"{msg} 文件：{out_path.name}")
                        st.download_button(
                            "下载导出压缩包",
                            data=out_path.read_bytes(),
                            file_name=out_path.name,
                            mime="application/zip",
                            key=f"dataset-download-{out_path.name}",
                        )
                    else:
                        st.warning(msg)

                st.markdown("**同步数据集样本到视觉模块**")
                sync_ids = st.multiselect(
                    "选择要同步样本的数据集（不选则默认最新3个）",
                    options=list(registry_labels.keys()),
                    format_func=lambda x: registry_labels.get(x, x),
                    key="dataset-sync-ids",
                )
                sync_count = st.slider("同步图片数量", 6, 80, 24, 2, key="dataset-sync-count")
                if st.button("同步到视觉示例库", use_container_width=True, key="dataset-sync-btn"):
                    ok, msg, new_paths = sync_dataset_samples_to_vision(
                        DATA_DIR,
                        VISION_SAMPLE_DIR,
                        selected_ids=sync_ids,
                        max_images=int(sync_count),
                    )
                    if ok:
                        st.success(msg)
                        pre_cols = st.columns(4)
                        for i, p in enumerate(new_paths[:8]):
                            with pre_cols[i % 4]:
                                st.image(p.read_bytes(), caption=_vision_sample_display_name(p), use_container_width=True)
                    else:
                        st.warning(msg)

            chart_insight(
                "数据集管理解读",
                [
                    "系统会对链接与文件哈希做双重去重，避免重复抓取和重复入库。",
                    "支持 Kaggle 标识一键导入，并可扫描本地 raw 目录自动登记历史数据。",
                    "导出功能会附带清单文件，便于后续训练复现和成果归档提交。",
                ],
            )

    with tabs[4]:
        section_header("策略推演与智能问答", "策略参数联动、风险对比、自动建议")
        target_corridor = st.selectbox("推演走廊", corridors, index=2, key="plan-corridor")
        scenario = _scenario_dict(event_boost, transit_boost, signal_opt, incident_ctl)
        baseline_scenario = _scenario_dict(0.0, 0.0, 0.0, 0.0)

        base_fc = forecast_corridor(df, model_bundle, target_corridor, horizon_hours=24, scenario=baseline_scenario)
        plan_fc = forecast_corridor(df, model_bundle, target_corridor, horizon_hours=24, scenario=scenario)

        base_kpis = compute_city_kpis(df, base_fc)
        plan_kpis = compute_city_kpis(df, plan_fc)
        severe_drop = float(base_kpis["forecast_severe_hours"] - plan_kpis["forecast_severe_hours"])
        speed_gain = float(plan_kpis["forecast_avg_speed"] - base_kpis["forecast_avg_speed"])
        peak_risk_drop = float(base_kpis["forecast_peak_risk"] - plan_kpis["forecast_peak_risk"])

        s1, s2, s3 = st.columns(3)
        s1.metric("严重拥堵小时变化", f"{-severe_drop:+.1f}", help="负值代表减少")
        s2.metric("预测平均速度变化", f"{speed_gain:+.2f} 公里/小时")
        s3.metric("峰值风险变化", f"{-peak_risk_drop:+.2f}", help="负值代表下降")

        _show(_risk_compare_chart(base_fc, plan_fc), key="risk-compare")
        chart_insight(
            "策略对比解读",
            [
                "两条曲线间距越大，说明策略对风险抑制越明显。",
                "若高峰时段差值持续扩大，代表策略在关键时段有效。",
            ],
        )

        mc = monte_carlo_risk(plan_fc, trials=900)
        if not mc.样本分布.empty:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("峰值风险中位数", f"{mc.摘要.get('峰值风险中位数', 0.0):.2f}")
            mc2.metric("峰值风险95分位", f"{mc.摘要.get('峰值风险95分位', 0.0):.2f}")
            mc3.metric("平均风险中位数", f"{mc.摘要.get('平均风险中位数', 0.0):.2f}")
            mc4.metric("高风险小时均值", f"{mc.摘要.get('高风险小时数均值', 0.0):.2f}")

            fig_mc = px.histogram(
                mc.样本分布,
                x="峰值风险",
                nbins=30,
                title="蒙特卡洛仿真：峰值风险分布",
                color_discrete_sequence=[主题主色],
            )
            _show(fig_mc, key="mc-peak")
            chart_insight(
                "仿真解读",
                [
                    "分布越集中，说明策略稳定性越好；右尾越长表示极端风险仍需重点关注。",
                ],
            )

        d_left, d_right = st.columns([1.1, 1.0])
        with d_left:
            drivers = explain_next_hour_drivers(plan_fc)
            drivers_cn = drivers.rename(columns={"driver": "影响因子", "score": "影响值", "direction": "作用方向"})
            fig_driver = px.bar(
                drivers_cn,
                x="影响值",
                y="影响因子",
                color="作用方向",
                orientation="h",
                title="下一小时关键驱动因素",
                color_discrete_map={"推高拥堵": 主题强调, "缓解拥堵": 主题主色},
            )
            _show(fig_driver, key="driver")

        with d_right:
            st.markdown('<div class="section-title">自动策略建议</div>', unsafe_allow_html=True)
            policies = generate_micro_policies(plan_fc, hotspots_df, scenario=scenario, top_n=8)
            for idx, line in enumerate(policies, start=1):
                st.markdown(f"{idx}. {line}")

        st.markdown("---")
        st.subheader("智能问答副驾")
        question = st.text_input("输入答辩问题（例如：为什么这个走廊风险高？）", value="为什么这个走廊风险高，策略后改善了什么？")
        if st.button("生成回答", use_container_width=False):
            deterministic = answer_query(question, plan_kpis, rank_df, plan_fc, hotspots_df)
            final_answer = deterministic

            settings = _build_llm_settings(api_base, api_key, llm_model, int(timeout_s))
            if use_llm and is_llm_configured(settings):
                context = (
                    f"目标走廊: {target_corridor}; "
                    f"策略参数: {scenario}; "
                    f"预测峰值风险: {plan_kpis['forecast_peak_risk']:.2f}; "
                    f"预测平均速度: {plan_kpis['forecast_avg_speed']:.2f} 公里/小时"
                )
                ok, llm_text = enhance_answer_with_llm(settings, question, deterministic, context)
                if ok:
                    final_answer = llm_text
                else:
                    st.warning(f"大模型增强失败，已回退规则回答：{llm_text}")

            st.success(final_answer)

    with tabs[5]:
        section_header("轨道交通应急站选址", "复杂网络中心性 + 近似优化选址 + 覆盖能力评估")
        st.markdown('<div class="note">该模块用于回答“站点建在哪里最有效”：在约束站点数量下，最小化平均与最大救援时间，并提升网络覆盖率。</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        center_count = c1.slider("建设站点数量", 3, 10, 6, 1)
        speed_kmh = c2.slider("救援通行速度(公里/小时)", 24, 60, 36, 1)
        dispatch_min = c3.slider("出警准备时间(分钟)", 2, 12, 4, 1)
        cover_time = c4.slider("覆盖阈值(分钟)", 8, 20, 12, 1)

        nodes_df, edges_df = emergency_network_cached(st.session_state["rebuild_nonce"])
        station_metrics = emergency_metrics_cached(nodes_df, edges_df, df)
        em_res = optimize_emergency_centers(
            station_metrics=station_metrics,
            edges=edges_df,
            center_count=center_count,
            speed_kmh=float(speed_kmh),
            dispatch_min=float(dispatch_min),
            cover_time_min=float(cover_time),
        )

        avg_gain = em_res.baseline_summary["平均救援时间"] - em_res.summary["平均救援时间"]
        max_gain = em_res.baseline_summary["最大救援时间"] - em_res.summary["最大救援时间"]
        cover_gain = em_res.summary["覆盖率"] - em_res.baseline_summary["覆盖率"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("优化后平均救援时间", f"{em_res.summary['平均救援时间']:.2f} 分钟", delta=f"{avg_gain:+.2f}")
        s2.metric("优化后最大救援时间", f"{em_res.summary['最大救援时间']:.2f} 分钟", delta=f"{max_gain:+.2f}")
        s3.metric("覆盖率", f"{em_res.summary['覆盖率']:.1%}", delta=f"{cover_gain:+.1%}")
        s4.metric("高风险覆盖缺口", f"{em_res.summary['高风险覆盖缺口']:.1%}")

        _show(_emergency_network_map(em_res.node_df, em_res.edge_df), key="em-map")

        left, right = st.columns([1.1, 1.0])
        with left:
            rank = em_res.node_df.sort_values("综合重要度", ascending=False).head(15).copy()
            rank["站点状态"] = rank["是否入选"].map({1: "优化入选", 0: "候选站点"})
            fig_rank = px.bar(
                rank.sort_values("综合重要度", ascending=True),
                x="综合重要度",
                y="站点",
                orientation="h",
                color="站点状态",
                color_discrete_map={"候选站点": "#b5c4bf", "优化入选": 主题主色},
                title="站点综合重要度排名（前15）",
                labels={"综合重要度": "综合重要度", "站点": "站点", "站点状态": "类型"},
            )
            _show(fig_rank, key="em-rank")
            chart_insight(
                "站点排名解读",
                [
                    "同色表示同一类型站点，避免误把颜色深浅理解为数值大小。",
                    "靠前站点表示综合重要度更高，适合作为优先建设候选。",
                ],
            )
        with right:
            comp = pd.DataFrame(
                [
                    {"方案": "基线方案", **em_res.baseline_summary},
                    {"方案": "优化方案", **em_res.summary},
                ]
            )
            st.dataframe(comp, use_container_width=True, hide_index=True)
            chart_insight(
                "方案对比解读",
                [
                    "优化方案的平均救援时间与最大救援时间越低越好。",
                    "覆盖率越高且高风险覆盖缺口越小，说明方案更稳健。",
                ],
            )
            st.markdown(f"**优化入选站点**：{'、'.join(em_res.chosen_centers)}")
            st.markdown(f"**基线站点**：{'、'.join(em_res.baseline_centers)}")

        st.subheader("站点-服务中心分配明细")
        show_service = em_res.service_df.sort_values("预计救援时间分钟", ascending=False).head(20)
        st.dataframe(show_service, use_container_width=True, hide_index=True)
        chart_insight(
            "分配明细解读",
            [
                "表格按预计救援时间从高到低展示，越靠前越值得优先治理。",
            ],
        )

        st.subheader("备用数据集清单")
        ds_df = pd.DataFrame(DATASET_BACKUP)
        st.dataframe(ds_df[["名称", "说明"]], use_container_width=True, hide_index=True)
        with st.expander("查看数据源链接", expanded=False):
            for idx, row in enumerate(DATASET_BACKUP, start=1):
                st.markdown(f"{idx}. [{row['名称']} 数据源链接]({row['链接']})")


if __name__ == "__main__":
    main()
