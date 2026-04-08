from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
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

LEVEL_NAMES = {0: "畅通", 1: "缓行", 2: "拥堵"}


@dataclass(frozen=True)
class 高级分类结果:
    排行榜: pd.DataFrame
    混淆矩阵: Dict[str, pd.DataFrame]
    集成权重: Dict[str, float]


@dataclass(frozen=True)
class 回归结果:
    排行榜: pd.DataFrame
    预测序列: Dict[str, pd.DataFrame]


@dataclass(frozen=True)
class 蒙特卡洛结果:
    样本分布: pd.DataFrame
    摘要: Dict[str, float]


def _build_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("数值", "passthrough", NUMERIC_COLS),
            ("类别", _build_ohe(), CATEGORICAL_COLS),
        ]
    )


def _split_three_way(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    data = df.sort_values("timestamp").reset_index(drop=True)
    x = data[FEATURE_COLUMNS].copy()
    y = data["congestion_level"].astype(int).copy()
    n = len(data)
    i1 = max(int(n * 0.70), 900)
    i2 = max(int(n * 0.85), i1 + 120)
    i2 = min(i2, n - 60)
    return x.iloc[:i1], x.iloc[i1:i2], x.iloc[i2:], y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:]


def train_advanced_classifier(df: pd.DataFrame) -> 高级分类结果:
    x_train, x_val, x_test, y_train, y_val, y_test = _split_three_way(df)
    prep = _build_preprocessor()

    model_defs = {
        "逻辑回归": LogisticRegression(max_iter=1400, solver="saga", random_state=42, n_jobs=1),
        "随机森林": RandomForestClassifier(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
        "极端随机树": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "梯度提升树": GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.06,
            max_depth=3,
            random_state=42,
        ),
        "神经网络": MLPClassifier(
            hidden_layer_sizes=(120, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=0.0012,
            max_iter=150,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.12,
        ),
    }

    rows: List[Dict[str, float | str]] = []
    confusion: Dict[str, pd.DataFrame] = {}
    proba_test: Dict[str, np.ndarray] = {}
    proba_val: Dict[str, np.ndarray] = {}

    for name, estimator in model_defs.items():
        pipe = Pipeline([("预处理", prep), ("模型", estimator)])
        t0 = perf_counter()
        pipe.fit(x_train, y_train)
        sec = perf_counter() - t0

        pred_test = pipe.predict(x_test)
        pred_val = pipe.predict(x_val)
        p_test = pipe.predict_proba(x_test)
        p_val = pipe.predict_proba(x_val)
        proba_test[name] = p_test
        proba_val[name] = p_val

        rows.append(
            {
                "模型": name,
                "准确率": round(float(accuracy_score(y_test, pred_test)), 4),
                "宏平均F1": round(float(f1_score(y_test, pred_test, average="macro")), 4),
                "宏平均精确率": round(float(precision_score(y_test, pred_test, average="macro")), 4),
                "宏平均召回率": round(float(recall_score(y_test, pred_test, average="macro")), 4),
                "卡帕系数": round(float(cohen_kappa_score(y_test, pred_test)), 4),
                "训练时长(秒)": round(float(sec), 2),
            }
        )

        cm = confusion_matrix(y_test, pred_test, labels=[0, 1, 2])
        confusion[name] = pd.DataFrame(
            cm,
            index=["真实-畅通", "真实-缓行", "真实-拥堵"],
            columns=["预测-畅通", "预测-缓行", "预测-拥堵"],
        )

        # 记录验证集分数，后续用于加权集成
        _ = pred_val

    # 基于验证集做权重搜索，构建高精度加权集成
    candidate = ["随机森林", "梯度提升树", "神经网络"]
    weights = {}
    for c in candidate:
        if c not in proba_val:
            weights[c] = 0.0

    best = {"f1": -1.0, "w": (0.4, 0.3, 0.3)}
    for w1 in range(1, 9):
        for w2 in range(1, 9):
            w3 = 10 - w1 - w2
            if w3 <= 0:
                continue
            a, b, c = w1 / 10.0, w2 / 10.0, w3 / 10.0
            p = a * proba_val["随机森林"] + b * proba_val["梯度提升树"] + c * proba_val["神经网络"]
            pred = np.argmax(p, axis=1)
            f1 = float(f1_score(y_val, pred, average="macro"))
            if f1 > best["f1"]:
                best = {"f1": f1, "w": (a, b, c)}

    wa, wb, wc = best["w"]  # type: ignore[index]
    p_test_mix = wa * proba_test["随机森林"] + wb * proba_test["梯度提升树"] + wc * proba_test["神经网络"]
    pred_mix = np.argmax(p_test_mix, axis=1)
    rows.append(
        {
            "模型": "高精度加权集成",
            "准确率": round(float(accuracy_score(y_test, pred_mix)), 4),
            "宏平均F1": round(float(f1_score(y_test, pred_mix, average="macro")), 4),
            "宏平均精确率": round(float(precision_score(y_test, pred_mix, average="macro")), 4),
            "宏平均召回率": round(float(recall_score(y_test, pred_mix, average="macro")), 4),
            "卡帕系数": round(float(cohen_kappa_score(y_test, pred_mix)), 4),
            "训练时长(秒)": 0.0,
        }
    )
    confusion["高精度加权集成"] = pd.DataFrame(
        confusion_matrix(y_test, pred_mix, labels=[0, 1, 2]),
        index=["真实-畅通", "真实-缓行", "真实-拥堵"],
        columns=["预测-畅通", "预测-缓行", "预测-拥堵"],
    )

    leaderboard = pd.DataFrame(rows).sort_values(["宏平均F1", "准确率"], ascending=False).reset_index(drop=True)
    ensemble_weights = {"随机森林": round(wa, 2), "梯度提升树": round(wb, 2), "神经网络": round(wc, 2)}
    return 高级分类结果(排行榜=leaderboard, 混淆矩阵=confusion, 集成权重=ensemble_weights)


def train_regression_suite(df: pd.DataFrame) -> 回归结果:
    data = df.sort_values("timestamp").reset_index(drop=True)
    x = data[FEATURE_COLUMNS].copy()
    y = data["congestion_index"].astype(float).copy()
    split_idx = max(int(len(data) * 0.83), 1200)

    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    ts_test = pd.to_datetime(data.iloc[split_idx:]["timestamp"]).reset_index(drop=True)

    prep = _build_preprocessor()
    model_defs = {
        "随机森林回归": RandomForestRegressor(n_estimators=220, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=1),
        "梯度提升回归": GradientBoostingRegressor(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=42),
        "岭回归": Ridge(alpha=1.0),
    }

    rows: List[Dict[str, float | str]] = []
    pred_map: Dict[str, pd.DataFrame] = {}
    for name, estimator in model_defs.items():
        pipe = Pipeline([("预处理", prep), ("模型", estimator)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        rows.append(
            {
                "模型": name,
                "平均绝对误差": round(float(mean_absolute_error(y_test, pred)), 4),
                "均方根误差": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
                "R2": round(float(r2_score(y_test, pred)), 4),
                "样本数": int(len(y_test)),
            }
        )
        pred_map[name] = pd.DataFrame({"时间": ts_test, "真实值": y_test.reset_index(drop=True), "预测值": pred})

    leaderboard = pd.DataFrame(rows).sort_values("平均绝对误差", ascending=True).reset_index(drop=True)
    return 回归结果(排行榜=leaderboard, 预测序列=pred_map)


def bootstrap_corridor_statistics(df: pd.DataFrame, top_n: int = 8, n_boot: int = 500) -> pd.DataFrame:
    work = df.copy()
    recent = work[work["timestamp"] >= pd.to_datetime(work["timestamp"]).max() - pd.Timedelta(days=7)]
    rank = (
        recent.groupby("corridor", as_index=False)
        .agg(均值=("congestion_index", "mean"), 样本数=("congestion_index", "size"))
        .sort_values("均值", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(42)
    rows: List[Dict[str, float | str]] = []
    for _, row in rank.iterrows():
        corridor = str(row["corridor"])
        vals = recent[recent["corridor"] == corridor]["congestion_index"].dropna().values.astype(float)
        if len(vals) < 20:
            continue
        boots = []
        n = len(vals)
        for _ in range(n_boot):
            sample = rng.choice(vals, size=n, replace=True)
            boots.append(float(np.mean(sample)))
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5]).tolist()
        rows.append(
            {
                "走廊": corridor,
                "拥堵均值": round(float(np.mean(vals)), 2),
                "95%下界": round(float(ci_low), 2),
                "95%上界": round(float(ci_high), 2),
                "波动系数": round(float(np.std(vals) / (np.mean(vals) + 1e-6)), 3),
                "样本数": int(n),
            }
        )
    return pd.DataFrame(rows).sort_values("拥堵均值", ascending=False).reset_index(drop=True)


def monte_carlo_risk(forecast_df: pd.DataFrame, trials: int = 800) -> 蒙特卡洛结果:
    if forecast_df.empty:
        return 蒙特卡洛结果(样本分布=pd.DataFrame(columns=["峰值风险", "平均风险", "高风险小时数"]), 摘要={})

    risk = forecast_df["risk_score"].astype(float).values
    rng = np.random.default_rng(42)
    sigma = max(float(np.std(risk) * 0.10), 1.8)

    sims = risk[None, :] + rng.normal(0.0, sigma, size=(trials, len(risk)))
    sims = np.clip(sims, 0.0, 100.0)

    peak = sims.max(axis=1)
    mean = sims.mean(axis=1)
    high_hours = (sims >= 80.0).sum(axis=1)
    dist = pd.DataFrame({"峰值风险": peak, "平均风险": mean, "高风险小时数": high_hours})
    summary = {
        "峰值风险中位数": round(float(np.median(peak)), 2),
        "峰值风险95分位": round(float(np.quantile(peak, 0.95)), 2),
        "平均风险中位数": round(float(np.median(mean)), 2),
        "高风险小时数均值": round(float(np.mean(high_hours)), 2),
    }
    return 蒙特卡洛结果(样本分布=dist, 摘要=summary)
