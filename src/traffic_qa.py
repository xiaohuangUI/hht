from __future__ import annotations

from typing import Dict

import pandas as pd


def _safe_top_corridor(rank_df: pd.DataFrame, n: int = 3) -> str:
    if rank_df.empty or "corridor" not in rank_df.columns:
        return "暂无可用走廊排名数据"
    top = rank_df.head(n)
    lines = []
    for _, row in top.iterrows():
        name = str(row.get("corridor", "-"))
        score = float(row.get("corridor_score", 0.0))
        severe_ratio = float(row.get("severe_ratio", 0.0))
        lines.append(f"{name}(风险分{score:.1f}, 严重占比{severe_ratio:.0%})")
    return "、".join(lines)


def _safe_peak_time(forecast_df: pd.DataFrame) -> str:
    if forecast_df.empty:
        return "暂无预测结果"
    top = forecast_df.sort_values("risk_score", ascending=False).head(1)
    if top.empty:
        return "暂无预测结果"
    return str(pd.to_datetime(top.iloc[0]["timestamp"]).strftime("%m-%d %H:%M"))


def _safe_hotspots(hotspots_df: pd.DataFrame, n: int = 2) -> str:
    if hotspots_df.empty or "title" not in hotspots_df.columns:
        return "暂无热点事件"
    top = hotspots_df.sort_values("impact_score", ascending=False).head(n)["title"].astype(str).tolist()
    if not top:
        return "暂无热点事件"
    return "；".join(top)


def _overview_answer(kpis: Dict[str, float], rank_df: pd.DataFrame, forecast_df: pd.DataFrame, hotspots_df: pd.DataFrame) -> str:
    avg_speed = float(kpis.get("avg_speed", 0.0))
    avg_index = float(kpis.get("avg_index", 0.0))
    severe_ratio = float(kpis.get("severe_ratio", 0.0))
    severe_hours = float(kpis.get("forecast_severe_hours", 0.0))
    peak_risk = float(kpis.get("forecast_peak_risk", 0.0))
    peak_time = _safe_peak_time(forecast_df)
    top_corridors = _safe_top_corridor(rank_df, n=3)
    hot_titles = _safe_hotspots(hotspots_df, n=2)

    return (
        f"当前城市近24小时平均速度约 {avg_speed:.1f} 公里/小时，平均拥堵指数 {avg_index:.1f}，严重拥堵占比 {severe_ratio:.1%}。"
        f"未来24小时预计严重拥堵 {severe_hours:.0f} 小时，峰值风险约 {peak_risk:.1f}（高峰时段 {peak_time}）。"
        f"高风险走廊主要是：{top_corridors}。重点关注热点：{hot_titles}。"
    )


def answer_query(
    question: str,
    kpis: Dict[str, float],
    rank_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    hotspots_df: pd.DataFrame,
) -> str:
    q = (question or "").strip()
    if not q:
        return "请输入问题，例如：为什么这个走廊风险高？策略后改善了什么？"

    lower_q = q.lower()
    top_corridors = _safe_top_corridor(rank_df, n=3)
    peak_time = _safe_peak_time(forecast_df)
    hot_titles = _safe_hotspots(hotspots_df, n=2)

    if any(k in q for k in ["最堵", "高风险", "风险最高", "拥堵最严重", "走廊"]) and ("为什么" not in q):
        return f"当前高风险走廊排名靠前的是：{top_corridors}。建议优先把信号优化和事故快处资源投向这些走廊。"

    if any(k in q for k in ["什么时候", "何时", "高峰", "峰值", "哪段时间"]):
        severe_hours = int(kpis.get("forecast_severe_hours", 0.0))
        peak_risk = float(kpis.get("forecast_peak_risk", 0.0))
        return f"预测峰值时段在 {peak_time}，峰值风险约 {peak_risk:.1f}。未来24小时预计严重拥堵约 {severe_hours} 小时。"

    if any(k in q for k in ["为什么", "原因", "成因", "驱动"]):
        if forecast_df.empty:
            return "当前没有可解释的预测结果，请先运行策略推演模块。"
        row = forecast_df.sort_values("risk_score", ascending=False).iloc[0]
        ratio = float(row.get("demand_capacity_ratio", 1.0))
        incident = float(row.get("incident_count", 0.0))
        event_intensity = float(row.get("event_intensity", 0.0))
        metro = float(row.get("metro_inflow", 0.0))
        return (
            f"风险高主要由四个因子叠加：需求压力比约 {ratio:.2f}、事件冲击 {event_intensity:.2f}、"
            f"事故水平 {incident:.1f}、公共交通分流量 {metro:.0f}。其中前两项是当前最主要抬升项。"
        )

    if any(k in q for k in ["改善", "优化效果", "策略后", "提升", "收益"]):
        severe_hours = float(kpis.get("forecast_severe_hours", 0.0))
        avg_speed = float(kpis.get("forecast_avg_speed", 0.0))
        peak_risk = float(kpis.get("forecast_peak_risk", 0.0))
        return (
            f"策略执行后，预测平均速度约 {avg_speed:.2f} 公里/小时，未来24小时严重拥堵约 {severe_hours:.0f} 小时，"
            f"峰值风险约 {peak_risk:.1f}。建议继续提升信号自适应与地铁接驳强度，进一步压低风险峰值。"
        )

    if any(k in q for k in ["热点", "舆情", "新闻", "线上"]):
        return f"当前高影响热点包括：{hot_titles}。建议把热点时段提前纳入交通组织方案，避免活动冲击放大。"

    if any(k in lower_q for k in ["指标", "总览", "概况", "整体"]):
        return _overview_answer(kpis, rank_df, forecast_df, hotspots_df)

    return _overview_answer(kpis, rank_df, forecast_df, hotspots_df)
