from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmergencyLocationResult:
    node_df: pd.DataFrame
    edge_df: pd.DataFrame
    service_df: pd.DataFrame
    summary: Dict[str, float]
    baseline_summary: Dict[str, float]
    chosen_centers: List[str]
    baseline_centers: List[str]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def _metro_lines() -> Dict[str, List[Tuple[str, float, float, str]]]:
    return {
        "一号线": [
            ("升仙湖", 30.716, 104.083, "金牛"),
            ("火车北站", 30.696, 104.079, "金牛"),
            ("人民北路", 30.681, 104.076, "青羊"),
            ("文殊院", 30.672, 104.078, "青羊"),
            ("骡马市", 30.666, 104.073, "青羊"),
            ("天府广场", 30.659, 104.074, "锦江"),
            ("锦江宾馆", 30.648, 104.075, "锦江"),
            ("华西坝", 30.640, 104.076, "武侯"),
            ("省体育馆", 30.632, 104.076, "武侯"),
            ("倪家桥", 30.621, 104.076, "武侯"),
            ("火车南站", 30.606, 104.074, "高新"),
            ("金融城", 30.593, 104.078, "高新"),
            ("孵化园", 30.585, 104.078, "高新"),
            ("世纪城", 30.573, 104.080, "高新"),
        ],
        "二号线": [
            ("犀浦", 30.756, 103.969, "郫都"),
            ("天河路", 30.739, 103.983, "郫都"),
            ("茶店子客运站", 30.713, 104.034, "金牛"),
            ("羊犀立交", 30.693, 104.047, "金牛"),
            ("一品天下", 30.684, 104.052, "金牛"),
            ("蜀汉路东", 30.679, 104.060, "金牛"),
            ("白果林", 30.671, 104.064, "青羊"),
            ("中医大省医院", 30.667, 104.067, "青羊"),
            ("人民公园", 30.663, 104.070, "青羊"),
            ("天府广场", 30.659, 104.074, "锦江"),
            ("春熙路", 30.657, 104.083, "锦江"),
            ("东门大桥", 30.653, 104.093, "锦江"),
            ("牛王庙", 30.646, 104.102, "锦江"),
            ("塔子山公园", 30.636, 104.118, "成华"),
            ("成都东客站", 30.625, 104.141, "成华"),
        ],
        "三号线": [
            ("军区总医院", 30.708, 104.116, "成华"),
            ("北湖大道", 30.708, 104.103, "成华"),
            ("动物园", 30.695, 104.100, "成华"),
            ("李家沱", 30.686, 104.094, "成华"),
            ("红星桥", 30.675, 104.088, "锦江"),
            ("市二医院", 30.666, 104.084, "锦江"),
            ("春熙路", 30.657, 104.083, "锦江"),
            ("新南门", 30.649, 104.087, "锦江"),
            ("磨子桥", 30.629, 104.083, "武侯"),
            ("省体育馆", 30.632, 104.076, "武侯"),
            ("衣冠庙", 30.620, 104.066, "武侯"),
            ("红牌楼", 30.612, 104.051, "武侯"),
            ("太平园", 30.593, 104.042, "武侯"),
        ],
    }


def build_network_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    lines = _metro_lines()
    station_map: Dict[str, Dict[str, object]] = {}
    edge_rows: List[Dict[str, object]] = []

    for line_name, stops in lines.items():
        for station, lat, lon, district in stops:
            if station not in station_map:
                station_map[station] = {
                    "站点": station,
                    "纬度": float(lat),
                    "经度": float(lon),
                    "区县": district,
                    "线路集合": {line_name},
                }
            else:
                station_map[station]["线路集合"].add(line_name)  # type: ignore[index]

        for i in range(len(stops) - 1):
            a = stops[i]
            b = stops[i + 1]
            dist = _haversine_km(a[1], a[2], b[1], b[2])
            edge_rows.append(
                {
                    "起点": a[0],
                    "终点": b[0],
                    "线路": line_name,
                    "距离公里": round(float(dist), 3),
                }
            )

    nodes = pd.DataFrame(station_map.values())
    nodes["线路"] = nodes["线路集合"].apply(lambda s: "、".join(sorted(list(s))))  # type: ignore[arg-type]
    nodes = nodes.drop(columns=["线路集合"]).sort_values("站点").reset_index(drop=True)
    edges = pd.DataFrame(edge_rows).reset_index(drop=True)
    return nodes, edges


def _build_adj(nodes: pd.DataFrame, edges: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    station_list = nodes["站点"].astype(str).tolist()
    idx = {s: i for i, s in enumerate(station_list)}
    n = len(station_list)
    inf = 1e15
    dist = np.full((n, n), inf, dtype=float)
    hops = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(hops, 0.0)

    for _, row in edges.iterrows():
        a, b = str(row["起点"]), str(row["终点"])
        i, j = idx[a], idx[b]
        d = float(row["距离公里"])
        if d < dist[i, j]:
            dist[i, j] = d
            dist[j, i] = d
        hops[i, j] = 1.0
        hops[j, i] = 1.0
    return dist, hops, station_list


def _floyd_warshall(mat: np.ndarray) -> np.ndarray:
    d = mat.copy()
    n = d.shape[0]
    for k in range(n):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    return d


def _brandes_unweighted(station_list: List[str], edges: pd.DataFrame) -> Dict[str, float]:
    # 无权图介数中心性（Brandes）
    n = len(station_list)
    idx = {s: i for i, s in enumerate(station_list)}
    adj: List[List[int]] = [[] for _ in range(n)]
    for _, row in edges.iterrows():
        i = idx[str(row["起点"])]
        j = idx[str(row["终点"])]
        adj[i].append(j)
        adj[j].append(i)

    cb = np.zeros(n, dtype=float)
    for s in range(n):
        stack: List[int] = []
        pred: List[List[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0
        dist = -np.ones(n, dtype=int)
        dist[s] = 0
        queue: List[int] = [s]
        qh = 0
        while qh < len(queue):
            v = queue[qh]
            qh += 1
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = np.zeros(n, dtype=float)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    if n > 2:
        cb /= ((n - 1) * (n - 2) / 2.0)
    return {station_list[i]: float(cb[i]) for i in range(n)}


def compute_station_metrics(nodes: pd.DataFrame, edges: pd.DataFrame, traffic_df: pd.DataFrame) -> pd.DataFrame:
    dist, hops, station_list = _build_adj(nodes, edges)
    sp = _floyd_warshall(dist)
    hp = _floyd_warshall(hops)
    n = len(station_list)

    degree = np.zeros(n, dtype=float)
    for _, row in edges.iterrows():
        i = station_list.index(str(row["起点"]))
        j = station_list.index(str(row["终点"]))
        degree[i] += 1.0
        degree[j] += 1.0
    degree_c = degree / max(1.0, (n - 1))

    closeness = np.zeros(n, dtype=float)
    for i in range(n):
        s = np.sum(sp[i, :][np.isfinite(sp[i, :])])
        closeness[i] = float((n - 1) / (s + 1e-9)) if s > 0 else 0.0

    between_map = _brandes_unweighted(station_list, edges)
    betweenness = np.array([between_map[s] for s in station_list], dtype=float)

    # 用交通数据构建站点需求权重
    latest = pd.to_datetime(traffic_df["timestamp"]).max()
    recent = traffic_df[traffic_df["timestamp"] >= latest - pd.Timedelta(hours=23)].copy()
    district_risk = recent.groupby("district")["congestion_index"].mean().to_dict()
    base_risk = float(np.mean(list(district_risk.values()))) if district_risk else 75.0
    demand = []
    for _, row in nodes.iterrows():
        d = str(row["区县"])
        risk = float(district_risk.get(d, base_risk))
        demand.append(risk)
    demand = np.array(demand, dtype=float)
    demand = (demand - demand.min()) / (demand.max() - demand.min() + 1e-9)
    demand = 0.45 + demand  # [0.45, 1.45]

    out = nodes.copy()
    out["度中心性"] = degree_c
    out["紧密中心性"] = closeness / (closeness.max() + 1e-9)
    out["介数中心性"] = betweenness / (betweenness.max() + 1e-9)
    out["需求权重"] = demand
    out["综合重要度"] = (
        0.22 * out["度中心性"]
        + 0.30 * out["紧密中心性"]
        + 0.33 * out["介数中心性"]
        + 0.15 * out["需求权重"]
    )
    out = out.sort_values("综合重要度", ascending=False).reset_index(drop=True)
    return out


def _evaluate_scheme(
    all_sp: np.ndarray,
    station_list: List[str],
    selected_idx: List[int],
    demand_weights: np.ndarray,
    speed_kmh: float,
    dispatch_min: float,
    cover_time_min: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    # 时间 = 距离/速度*60 + dispatch
    near_dist = np.min(all_sp[:, selected_idx], axis=1)
    times = near_dist / max(speed_kmh, 1e-6) * 60.0 + dispatch_min
    avg_t = float(np.average(times, weights=demand_weights))
    max_t = float(np.max(times))
    cover = float(np.mean(times <= cover_time_min))
    severe = float(np.mean(times > cover_time_min + 3.0))
    score = 0.6 * avg_t + 0.4 * max_t
    summary = {
        "平均救援时间": round(avg_t, 2),
        "最大救援时间": round(max_t, 2),
        "覆盖率": round(cover, 4),
        "高风险覆盖缺口": round(severe, 4),
        "综合目标值": round(score, 3),
    }
    return summary, times


def optimize_emergency_centers(
    station_metrics: pd.DataFrame,
    edges: pd.DataFrame,
    center_count: int = 6,
    speed_kmh: float = 36.0,
    dispatch_min: float = 4.0,
    cover_time_min: float = 12.0,
) -> EmergencyLocationResult:
    nodes = station_metrics.copy()
    station_list = nodes["站点"].astype(str).tolist()
    dist, _, _ = _build_adj(nodes[["站点"]], edges)
    all_sp = _floyd_warshall(dist)
    idx = {s: i for i, s in enumerate(station_list)}

    demand = nodes["需求权重"].values.astype(float)
    imp = nodes["综合重要度"].values.astype(float)
    demand_w = 0.65 * demand + 0.35 * (imp / (imp.max() + 1e-9))

    # 基线：按历史常见枢纽设定
    baseline_names = ["火车北站", "天府广场", "火车南站"]
    baseline_idx = [idx[s] for s in baseline_names if s in idx]
    while len(baseline_idx) < min(center_count, 3):
        baseline_idx.append(int(np.argmax(imp)))
    baseline_summary, baseline_times = _evaluate_scheme(
        all_sp=all_sp,
        station_list=station_list,
        selected_idx=baseline_idx,
        demand_weights=demand_w,
        speed_kmh=speed_kmh,
        dispatch_min=dispatch_min,
        cover_time_min=cover_time_min,
    )

    # 贪心加局部替换：近似求解 p-中位 + p-中心
    selected: List[int] = []
    candidate = list(range(len(station_list)))
    for _ in range(center_count):
        best_j = None
        best_obj = 1e18
        for j in candidate:
            trial = selected + [j]
            summary, _ = _evaluate_scheme(
                all_sp=all_sp,
                station_list=station_list,
                selected_idx=trial,
                demand_weights=demand_w,
                speed_kmh=speed_kmh,
                dispatch_min=dispatch_min,
                cover_time_min=cover_time_min,
            )
            obj = summary["综合目标值"]
            if obj < best_obj:
                best_obj = obj
                best_j = j
        if best_j is None:
            break
        selected.append(best_j)
        candidate.remove(best_j)

    # 单步局部替换优化
    improved = True
    while improved:
        improved = False
        base_summary, _ = _evaluate_scheme(
            all_sp=all_sp,
            station_list=station_list,
            selected_idx=selected,
            demand_weights=demand_w,
            speed_kmh=speed_kmh,
            dispatch_min=dispatch_min,
            cover_time_min=cover_time_min,
        )
        base_obj = base_summary["综合目标值"]
        for i, sel in enumerate(selected):
            for cand in [c for c in range(len(station_list)) if c not in selected]:
                trial = selected.copy()
                trial[i] = cand
                trial_summary, _ = _evaluate_scheme(
                    all_sp=all_sp,
                    station_list=station_list,
                    selected_idx=trial,
                    demand_weights=demand_w,
                    speed_kmh=speed_kmh,
                    dispatch_min=dispatch_min,
                    cover_time_min=cover_time_min,
                )
                if trial_summary["综合目标值"] < base_obj - 1e-6:
                    selected = sorted(trial)
                    improved = True
                    break
            if improved:
                break

    summary, times = _evaluate_scheme(
        all_sp=all_sp,
        station_list=station_list,
        selected_idx=selected,
        demand_weights=demand_w,
        speed_kmh=speed_kmh,
        dispatch_min=dispatch_min,
        cover_time_min=cover_time_min,
    )

    # 最近服务站
    selected_mat = all_sp[:, selected]
    nearest_col = np.argmin(selected_mat, axis=1)
    nearest_dist = np.min(selected_mat, axis=1)
    nearest_station = [station_list[selected[c]] for c in nearest_col]
    service_df = pd.DataFrame(
        {
            "站点": station_list,
            "服务中心": nearest_station,
            "最短路距离公里": nearest_dist.round(3),
            "预计救援时间分钟": (nearest_dist / max(speed_kmh, 1e-6) * 60.0 + dispatch_min).round(2),
        }
    )

    nodes_out = nodes.copy()
    chosen_set = {station_list[i] for i in selected}
    baseline_set = {station_list[i] for i in baseline_idx}
    nodes_out["是否入选"] = nodes_out["站点"].isin(chosen_set).astype(int)
    nodes_out["是否基线站"] = nodes_out["站点"].isin(baseline_set).astype(int)
    nodes_out["预计救援时间分钟"] = times

    return EmergencyLocationResult(
        node_df=nodes_out,
        edge_df=edges.copy(),
        service_df=service_df,
        summary=summary,
        baseline_summary=baseline_summary,
        chosen_centers=sorted(list(chosen_set)),
        baseline_centers=sorted(list(baseline_set)),
    )
