from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_city_kpis(base_df: pd.DataFrame, forecast_df: pd.DataFrame | None = None) -> Dict[str, float]:
    latest_ts = pd.to_datetime(base_df["timestamp"].max())
    recent = base_df[base_df["timestamp"] >= latest_ts - pd.Timedelta(hours=23)].copy()

    severe_ratio = float((recent["congestion_level"] == 2).mean()) if not recent.empty else 0.0
    avg_speed = float(recent["avg_speed"].mean()) if not recent.empty else 0.0
    avg_index = float(recent["congestion_index"].mean()) if not recent.empty else 0.0
    incidents = float(recent["incident_count"].sum()) if not recent.empty else 0.0

    out = {
        "avg_speed": round(avg_speed, 2),
        "avg_index": round(avg_index, 2),
        "severe_ratio": round(severe_ratio, 4),
        "incident_total": round(incidents, 1),
    }

    if forecast_df is not None and not forecast_df.empty:
        out["forecast_severe_hours"] = float((forecast_df["pred_level"] == 2).sum())
        out["forecast_peak_risk"] = float(forecast_df["risk_score"].max())
        out["forecast_avg_speed"] = float(forecast_df["expected_speed"].mean())
    else:
        out["forecast_severe_hours"] = 0.0
        out["forecast_peak_risk"] = 0.0
        out["forecast_avg_speed"] = 0.0

    return out


def rank_hot_corridors(base_df: pd.DataFrame, lookback_hours: int = 24) -> pd.DataFrame:
    latest_ts = pd.to_datetime(base_df["timestamp"].max())
    recent = base_df[base_df["timestamp"] >= latest_ts - pd.Timedelta(hours=lookback_hours - 1)].copy()

    g = (
        recent.groupby("corridor", as_index=False)
        .agg(
            avg_index=("congestion_index", "mean"),
            avg_speed=("avg_speed", "mean"),
            severe_ratio=("congestion_level", lambda s: float((s == 2).mean())),
            incidents=("incident_count", "sum"),
        )
    )
    g["corridor_score"] = g["avg_index"] * 0.55 + g["severe_ratio"] * 40 + g["incidents"] * 1.5 - g["avg_speed"] * 0.08
    g = g.sort_values("corridor_score", ascending=False).reset_index(drop=True)
    g.index = g.index + 1
    g.insert(0, "rank", g.index)
    return g


def generate_micro_policies(
    forecast_df: pd.DataFrame,
    hotspots_df: pd.DataFrame,
    scenario: Dict[str, float] | None = None,
    top_n: int = 6,
) -> List[str]:
    if scenario is None:
        scenario = {}

    if forecast_df.empty:
        return ["暂无预测结果，先运行“预测中心”模块再生成策略。"]

    severe = forecast_df[forecast_df["pred_level"] == 2]
    busy = forecast_df[forecast_df["pred_level"] >= 1]
    peak_ts = forecast_df.sort_values("risk_score", ascending=False).head(1)["timestamp"].astype(str).tolist()
    peak_time = peak_ts[0] if peak_ts else "未来时段"

    event_boost = float(scenario.get("event_boost", 0.0))
    transit_boost = float(scenario.get("transit_boost", 0.0))
    signal_opt = float(scenario.get("signal_optimization", 0.0))
    incident_ctl = float(scenario.get("incident_control", 0.0))

    suggestions: List[str] = []
    suggestions.append(f"在 {peak_time} 前后 90 分钟执行“绿波带+动态相位”联动，优先消化主干道排队。")

    if len(severe) >= 4:
        suggestions.append("将高风险路口设置分级管控：一级路口布警，二级路口启用可变车道，三级路口以诱导分流为主。")
    else:
        suggestions.append("以柔性管控为主：高峰时段压缩次干道进口放行，保证主走廊通行效率。")

    if float(forecast_df["incident_count"].mean()) >= 0.8:
        suggestions.append("建立“事故 15 分钟快处机制”：路警联动、拖车前置、事故图片 AI 分级派单。")

    if float(forecast_df["event_intensity"].mean()) >= 0.8 or event_boost > 0.6:
        suggestions.append("赛事/演唱会场景执行“散场双波峰”策略：散场前地铁增开班次，散场后公交接驳与网约车分区上客并行。")

    if transit_boost >= 0.5:
        suggestions.append("强化轨道分流：地铁站口-热点商圈设置 5 分钟接驳循环线，降低核心路段小汽车出行占比。")

    if signal_opt >= 0.4:
        suggestions.append("对连续拥堵路段实施信号自适应算法，滚动重算配时周期，按 15 分钟更新一次。")

    if incident_ctl >= 0.4:
        suggestions.append("部署重点路口视频巡检模型，提前识别违停与轻微擦碰风险，避免次生拥堵。")

    if not hotspots_df.empty:
        top_hot = hotspots_df.sort_values("impact_score", ascending=False).head(2)["title"].astype(str).tolist()
        if top_hot:
            suggestions.append(f"将热点事件纳入当天运行图，优先关注：{top_hot[0]}。")
            if len(top_hot) > 1:
                suggestions.append(f"次级关注：{top_hot[1]}。")

    unique: List[str] = []
    for s in suggestions:
        if s not in unique:
            unique.append(s)
    return unique[:top_n]


def build_competition_alignment() -> pd.DataFrame:
    rows = [
        ("主题创意(30%)", "成都治堵+活动客流+极端天气的复合真实场景", "热点联动、可解释场景建模、城市治理价值"),
        ("技术方案(30%)", "多源时空数据+拥堵分类模型+情景仿真", "完整技术链路、可复现实验流程"),
        ("功能效果(20%)", "24小时预测、风险预警、一点一策建议", "可运行演示、结果可视化、决策可落地"),
        ("作品呈现(20%)", "驾驶舱式界面+AI问答+答辩流程面板", "展示冲击力强，便于现场讲解"),
    ]
    return pd.DataFrame(rows, columns=["评审项", "本作品对应实现", "预期加分点"])
