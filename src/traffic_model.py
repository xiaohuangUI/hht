from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS: List[str] = [
    "district",
    "corridor",
    "hour",
    "weekday",
    "is_weekend",
    "is_holiday",
    "weather",
    "rain_intensity",
    "event_intensity",
    "incident_count",
    "roadwork_flag",
    "flow",
    "capacity",
    "demand_capacity_ratio",
    "avg_speed",
    "occupancy",
    "bus_delay_min",
    "metro_inflow",
]

NUMERIC_COLS: List[str] = [
    "hour",
    "weekday",
    "is_weekend",
    "is_holiday",
    "rain_intensity",
    "event_intensity",
    "incident_count",
    "roadwork_flag",
    "flow",
    "capacity",
    "demand_capacity_ratio",
    "avg_speed",
    "occupancy",
    "bus_delay_min",
    "metro_inflow",
]

CATEGORICAL_COLS: List[str] = ["district", "corridor", "weather"]

LEVEL_MAP = {0: "畅通", 1: "缓行", 2: "拥堵"}


@dataclass(frozen=True)
class TrafficModelBundle:
    pipeline: Pipeline
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    baseline_numeric: Dict[str, float]


def _feature_importance_frame(pipeline: Pipeline, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    prep: ColumnTransformer = pipeline.named_steps["prep"]  # type: ignore[assignment]
    model: RandomForestClassifier = pipeline.named_steps["model"]  # type: ignore[assignment]

    feat_names = list(prep.get_feature_names_out())
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
    return df.reset_index(drop=True)


def train_traffic_model(df: pd.DataFrame) -> TrafficModelBundle:
    data = df.sort_values("timestamp").reset_index(drop=True)
    x = data[FEATURE_COLUMNS].copy()
    y = data["congestion_level"].astype(int).copy()

    split_idx = max(int(len(data) * 0.82), 300)
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=140,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )

    pipeline = Pipeline(steps=[("prep", prep), ("model", model)])
    pipeline.fit(x_train, y_train)

    pred = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "test_size": float(len(x_test)),
    }

    imp_df = _feature_importance_frame(pipeline, NUMERIC_COLS, CATEGORICAL_COLS)
    baseline_numeric = x_train[NUMERIC_COLS].mean(numeric_only=True).to_dict()

    return TrafficModelBundle(
        pipeline=pipeline,
        metrics=metrics,
        feature_importance=imp_df,
        baseline_numeric=baseline_numeric,
    )


def predict_rows(bundle: TrafficModelBundle, feature_df: pd.DataFrame) -> pd.DataFrame:
    x = feature_df[FEATURE_COLUMNS].copy()
    probs = bundle.pipeline.predict_proba(x)
    pred = np.argmax(probs, axis=1)

    out = feature_df.copy()
    out["prob_smooth"] = probs[:, 0]
    out["prob_busy"] = probs[:, 1]
    out["prob_severe"] = probs[:, 2]
    out["pred_level"] = pred
    out["pred_label"] = out["pred_level"].map(LEVEL_MAP)

    out["risk_score"] = np.clip(
        out["prob_busy"] * 55
        + out["prob_severe"] * 90
        + out["incident_count"] * 4
        + out["event_intensity"] * 5,
        0,
        100,
    )
    out["expected_speed"] = np.clip(
        64
        - out["demand_capacity_ratio"] * 23
        - out["pred_level"] * 7
        - out["rain_intensity"] * 5,
        8,
        65,
    )
    return out


def _future_row_from_history(history: pd.DataFrame, ts: pd.Timestamp, corridor: str) -> Dict[str, object]:
    c_hist = history[history["corridor"] == corridor].copy()
    if c_hist.empty:
        c_hist = history.copy()

    hour = int(ts.hour)
    weekday = int(ts.weekday())
    similar = c_hist[(c_hist["hour"] == hour) & (c_hist["weekday"] == weekday)]
    if similar.empty:
        similar = c_hist[c_hist["hour"] == hour]
    if similar.empty:
        similar = c_hist.tail(96)

    row = similar.median(numeric_only=True).to_dict()
    latest = c_hist.sort_values("timestamp").tail(1).iloc[0]

    base = {
        "timestamp": ts,
        "district": str(latest["district"]),
        "corridor": corridor,
        "hour": hour,
        "weekday": weekday,
        "is_weekend": int(weekday >= 5),
        "is_holiday": int((weekday >= 5) and (hour >= 10 and hour <= 22)),
        "weather": str(latest["weather"]),
        "rain_intensity": float(row.get("rain_intensity", latest["rain_intensity"])),
        "event_intensity": float(row.get("event_intensity", latest["event_intensity"])),
        "incident_count": float(row.get("incident_count", latest["incident_count"])),
        "roadwork_flag": float(row.get("roadwork_flag", latest["roadwork_flag"])),
        "flow": float(row.get("flow", latest["flow"])),
        "capacity": float(row.get("capacity", latest["capacity"])),
        "demand_capacity_ratio": float(row.get("demand_capacity_ratio", latest["demand_capacity_ratio"])),
        "avg_speed": float(row.get("avg_speed", latest["avg_speed"])),
        "occupancy": float(row.get("occupancy", latest["occupancy"])),
        "bus_delay_min": float(row.get("bus_delay_min", latest["bus_delay_min"])),
        "metro_inflow": float(row.get("metro_inflow", latest["metro_inflow"])),
    }
    return base


def forecast_corridor(
    df: pd.DataFrame,
    bundle: TrafficModelBundle,
    corridor: str,
    horizon_hours: int = 24,
    scenario: Dict[str, float] | None = None,
) -> pd.DataFrame:
    if scenario is None:
        scenario = {}
    event_boost = float(scenario.get("event_boost", 0.0))
    transit_boost = float(scenario.get("transit_boost", 0.0))
    signal_optimization = float(scenario.get("signal_optimization", 0.0))
    incident_control = float(scenario.get("incident_control", 0.0))

    history = df.sort_values("timestamp").copy()
    last_ts = pd.to_datetime(history["timestamp"].max())
    future_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="H")

    rows: List[Dict[str, object]] = []
    for ts in future_ts:
        row = _future_row_from_history(history, ts=ts, corridor=corridor)

        row["event_intensity"] = float(row["event_intensity"]) * (1 + 0.26 * event_boost)
        row["metro_inflow"] = float(row["metro_inflow"]) * (1 + 0.22 * transit_boost)
        row["incident_count"] = max(0.0, float(row["incident_count"]) * (1 - 0.35 * incident_control))
        row["bus_delay_min"] = max(0.3, float(row["bus_delay_min"]) * (1 - 0.17 * signal_optimization))
        row["demand_capacity_ratio"] = float(row["demand_capacity_ratio"]) * (1 + 0.08 * event_boost - 0.11 * signal_optimization)
        row["flow"] = float(row["flow"]) * (1 + 0.06 * event_boost - 0.09 * signal_optimization)
        row["avg_speed"] = max(6.0, float(row["avg_speed"]) * (1 - 0.06 * event_boost + 0.12 * signal_optimization))
        row["occupancy"] = float(np.clip(float(row["occupancy"]) * (1 + 0.05 * event_boost - 0.1 * signal_optimization), 0.1, 0.99))

        rows.append(row)

    future_df = pd.DataFrame(rows)
    pred_df = predict_rows(bundle, future_df)
    pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)
    return pred_df


def explain_next_hour_drivers(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame(columns=["driver", "score", "direction"])

    row = forecast_df.iloc[0]
    drivers = [
        ("需求压力", float(row["demand_capacity_ratio"]) * 1.2),
        ("事件冲击", float(row["event_intensity"]) * 1.5),
        ("事故风险", float(row["incident_count"]) * 1.6),
        ("公交延误", float(row["bus_delay_min"]) / 7.0),
        ("天气影响", float(row["rain_intensity"]) * 1.8),
        ("轨道分流", -float(row["metro_inflow"]) / 12000.0),
    ]

    out = pd.DataFrame(drivers, columns=["driver", "score"])
    out["direction"] = np.where(out["score"] >= 0, "推高拥堵", "缓解拥堵")
    out["abs_score"] = out["score"].abs()
    out = out.sort_values("abs_score", ascending=False).drop(columns=["abs_score"]).reset_index(drop=True)
    return out
