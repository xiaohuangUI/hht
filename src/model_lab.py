from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
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

LEVEL_LABELS = {0: "畅通", 1: "缓行", 2: "拥堵"}


@dataclass(frozen=True)
class LabResult:
    leaderboard: pd.DataFrame
    model_cache: Dict[str, Pipeline]
    confusion: Dict[str, pd.DataFrame]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = df.sort_values("timestamp").reset_index(drop=True)
    x = data[FEATURE_COLUMNS].copy()
    y = data["congestion_level"].astype(int).copy()
    split_idx = max(int(len(data) * 0.8), 1200)
    return x.iloc[:split_idx], x.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def train_model_zoo(df: pd.DataFrame) -> LabResult:
    x_train, x_test, y_train, y_test = _split(df)
    prep = _build_preprocessor()

    model_defs = {
        "RandomForest": RandomForestClassifier(
            n_estimators=160,
            max_depth=11,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=90,
            learning_rate=0.08,
            max_depth=3,
            random_state=42,
        ),
        "NeuralNet(MLP)": MLPClassifier(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=0.0015,
            max_iter=110,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        ),
    }

    rows: List[Dict[str, float | str]] = []
    model_cache: Dict[str, Pipeline] = {}
    confusion: Dict[str, pd.DataFrame] = {}

    for name, estimator in model_defs.items():
        pipe = Pipeline([("prep", prep), ("model", estimator)])
        t0 = perf_counter()
        pipe.fit(x_train, y_train)
        sec = perf_counter() - t0

        pred = pipe.predict(x_test)
        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred, average="macro"))

        rows.append(
            {
                "模型": name,
                "Accuracy": round(acc, 4),
                "Macro-F1": round(f1, 4),
                "训练时长(s)": round(sec, 2),
                "测试样本": int(len(x_test)),
            }
        )
        model_cache[name] = pipe

        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=["真实-畅通", "真实-缓行", "真实-拥堵"], columns=["预测-畅通", "预测-缓行", "预测-拥堵"])
        confusion[name] = cm_df

    leaderboard = pd.DataFrame(rows).sort_values(["Macro-F1", "Accuracy"], ascending=False).reset_index(drop=True)
    return LabResult(leaderboard=leaderboard, model_cache=model_cache, confusion=confusion)


def feature_importance_from_rf(pipe: Pipeline) -> pd.DataFrame:
    prep: ColumnTransformer = pipe.named_steps["prep"]  # type: ignore[assignment]
    model: RandomForestClassifier = pipe.named_steps["model"]  # type: ignore[assignment]
    feats = list(prep.get_feature_names_out())
    imp = model.feature_importances_
    return pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)


def corridor_deep_forecast(df: pd.DataFrame, corridor: str, horizon: int = 24, lag: int = 8) -> Tuple[pd.DataFrame, Dict[str, float]]:
    s = (
        df[df["corridor"] == corridor]
        .sort_values("timestamp")[["timestamp", "congestion_index"]]
        .dropna()
        .reset_index(drop=True)
    )
    if len(s) < lag + 40:
        return pd.DataFrame(), {"mae": 0.0, "samples": float(len(s))}

    values = s["congestion_index"].values.astype(float)
    x, y = [], []
    for i in range(lag, len(values)):
        x.append(values[i - lag : i])
        y.append(values[i])
    x = np.array(x)
    y = np.array(y)

    split_idx = max(int(len(x) * 0.8), 80)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    reg = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.002,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.12,
    )
    reg.fit(x_train, y_train)
    pred_test = reg.predict(x_test)
    mae = float(mean_absolute_error(y_test, pred_test))

    # Recursive forecast
    seq = values[-lag:].copy()
    preds = []
    for _ in range(horizon):
        p = float(reg.predict(seq.reshape(1, -1))[0])
        p = float(np.clip(p, 0.0, 100.0))
        preds.append(p)
        seq = np.roll(seq, -1)
        seq[-1] = p

    last_ts = pd.to_datetime(s["timestamp"].iloc[-1])
    future_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon, freq="H")
    out = pd.DataFrame({"timestamp": future_ts, "pred_congestion_index": preds})
    out["pred_level"] = pd.cut(out["pred_congestion_index"], bins=[-1, 48, 72, 200], labels=["畅通", "缓行", "拥堵"])
    return out, {"mae": mae, "samples": float(len(x_test))}


def detect_anomaly_points(df: pd.DataFrame, corridor: str) -> pd.DataFrame:
    s = (
        df[df["corridor"] == corridor]
        .sort_values("timestamp")[["timestamp", "congestion_index", "avg_speed", "incident_count", "demand_capacity_ratio"]]
        .dropna()
    )
    if len(s) < 80:
        return pd.DataFrame(columns=["timestamp", "congestion_index", "anomaly_score", "is_anomaly"])

    feats = s[["congestion_index", "avg_speed", "incident_count", "demand_capacity_ratio"]].values
    iso = IsolationForest(n_estimators=180, contamination=0.07, random_state=42)
    iso.fit(feats)
    raw = iso.decision_function(feats)
    pred = iso.predict(feats)

    out = s.copy()
    out["anomaly_score"] = -raw
    out["is_anomaly"] = (pred == -1).astype(int)
    return out.reset_index(drop=True)
