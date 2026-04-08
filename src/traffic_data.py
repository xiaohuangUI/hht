from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import quote_plus
import html
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests


CITY_CORRIDORS: List[Dict[str, object]] = [
    {"corridor": "一环路北段", "district": "金牛", "free_speed": 58, "capacity": 5600, "transit_factor": 0.82},
    {"corridor": "一环路南段", "district": "武侯", "free_speed": 56, "capacity": 5400, "transit_factor": 0.86},
    {"corridor": "人民南路", "district": "武侯", "free_speed": 54, "capacity": 5100, "transit_factor": 0.92},
    {"corridor": "天府大道北段", "district": "高新", "free_speed": 62, "capacity": 6200, "transit_factor": 0.88},
    {"corridor": "天府大道南段", "district": "天府新区", "free_speed": 66, "capacity": 6500, "transit_factor": 0.8},
    {"corridor": "科华路", "district": "锦江", "free_speed": 50, "capacity": 4600, "transit_factor": 0.78},
    {"corridor": "东大街", "district": "锦江", "free_speed": 46, "capacity": 4300, "transit_factor": 0.94},
    {"corridor": "蜀都大道", "district": "青羊", "free_speed": 48, "capacity": 4700, "transit_factor": 0.9},
    {"corridor": "中环路东段", "district": "成华", "free_speed": 55, "capacity": 5400, "transit_factor": 0.72},
    {"corridor": "双流机场高速", "district": "双流", "free_speed": 72, "capacity": 7000, "transit_factor": 0.55},
    {"corridor": "成龙大道", "district": "锦江", "free_speed": 63, "capacity": 6400, "transit_factor": 0.62},
    {"corridor": "凤凰山环线", "district": "金牛", "free_speed": 52, "capacity": 5000, "transit_factor": 0.74},
]

WEATHER_STATES = [
    ("sunny", 0.0, 1.0),
    ("cloudy", 0.12, 0.85),
    ("light_rain", 0.35, 0.58),
    ("heavy_rain", 0.7, 0.25),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


@dataclass(frozen=True)
class HotspotCollectResult:
    success: bool
    rows: int
    output_file: str
    message: str
    source_counts: Dict[str, int]


def _hour_intensity(hour: int) -> float:
    if 7 <= hour <= 9:
        return 1.0
    if 17 <= hour <= 19:
        return 1.05
    if 10 <= hour <= 16:
        return 0.62
    if 20 <= hour <= 22:
        return 0.44
    return 0.24


def _sample_weather(rng: np.random.Generator) -> tuple[str, float]:
    labels = [x[0] for x in WEATHER_STATES]
    rain = [x[1] for x in WEATHER_STATES]
    prob = [x[2] for x in WEATHER_STATES]
    idx = int(rng.choice(np.arange(len(labels)), p=np.array(prob) / np.sum(prob)))
    return labels[idx], float(rain[idx])


def _event_intensity(ts: pd.Timestamp, corridor: str) -> float:
    score = 0.0
    hour = ts.hour
    weekday = ts.weekday()

    if corridor == "凤凰山环线" and weekday in {4, 5} and 18 <= hour <= 22:
        score += 1.45
    if corridor in {"东大街", "科华路"} and weekday in {5, 6} and 13 <= hour <= 20:
        score += 1.05
    if corridor in {"双流机场高速", "天府大道南段"} and hour in {8, 9, 21, 22}:
        score += 0.55
    if corridor in {"蜀都大道", "人民南路"} and weekday in {0, 1, 2, 3, 4} and 8 <= hour <= 10:
        score += 0.65

    return float(score)


def _congestion_label(index: float) -> tuple[int, str]:
    if index >= 72:
        return 2, "severe"
    if index >= 48:
        return 1, "busy"
    return 0, "smooth"


def generate_city_traffic_data(days: int = 56, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = (pd.Timestamp.now().floor("H") - pd.Timedelta(days=days))
    hours = pd.date_range(start=start, periods=days * 24, freq="H")

    records: List[Dict[str, object]] = []

    for ts in hours:
        hour = int(ts.hour)
        weekday = int(ts.weekday())
        is_weekend = int(weekday >= 5)
        is_holiday = int(is_weekend and rng.random() < 0.22)

        for c in CITY_CORRIDORS:
            corridor = str(c["corridor"])
            district = str(c["district"])
            free_speed = float(c["free_speed"])
            capacity = float(c["capacity"])
            transit_factor = float(c["transit_factor"])

            weather, rain_intensity = _sample_weather(rng)
            event_intensity = _event_intensity(ts, corridor)

            peak = _hour_intensity(hour)
            weekend_relief = -0.08 if is_weekend and hour in range(7, 19) else 0.0
            weekend_leisure = 0.12 if is_weekend and hour in range(13, 22) else 0.0
            weather_penalty = rain_intensity * 0.22

            demand_ratio = (
                0.34
                + 0.58 * peak
                + weekend_relief
                + weekend_leisure
                + 0.11 * event_intensity
                + weather_penalty
                + rng.normal(0, 0.06)
            )
            demand_ratio = float(np.clip(demand_ratio, 0.25, 1.48))
            flow = int(np.clip(capacity * demand_ratio, 850, capacity * 1.25))

            incident_base = 0.15 + 0.65 * max(demand_ratio - 0.85, 0) + 0.75 * rain_intensity
            incident_count = int(np.clip(rng.poisson(incident_base), 0, 5))
            roadwork_flag = int(rng.random() < 0.022)

            speed = (
                free_speed * (1 - 0.64 * (demand_ratio ** 1.48))
                - incident_count * 2.4
                - roadwork_flag * 3.8
                - rain_intensity * 5.0
                + rng.normal(0, 2.2)
            )
            speed = float(np.clip(speed, 8, free_speed))

            occupancy = 0.24 + 0.56 * demand_ratio + rain_intensity * 0.06 + incident_count * 0.038 + rng.normal(0, 0.03)
            occupancy = float(np.clip(occupancy, 0.18, 0.98))

            metro_inflow = (
                2200
                + 7800 * transit_factor * (0.45 + 0.65 * peak + 0.22 * event_intensity)
                + rng.normal(0, 320)
            )
            metro_inflow = float(np.clip(metro_inflow, 900, 22000))

            bus_delay_min = (
                2.4
                + 18.0 * max(demand_ratio - 0.62, 0)
                + 2.1 * incident_count
                + 4.2 * rain_intensity
                + roadwork_flag * 3.6
                + rng.normal(0, 1.25)
            )
            bus_delay_min = float(np.clip(bus_delay_min, 0.8, 35))

            congestion_index = (
                102
                - speed * 1.35
                + occupancy * 42
                + bus_delay_min * 0.95
                + incident_count * 6.8
                + event_intensity * 5.2
                + roadwork_flag * 4.6
            )
            congestion_index = float(np.clip(congestion_index, 0, 100))
            level, label = _congestion_label(congestion_index)

            records.append(
                {
                    "timestamp": ts,
                    "district": district,
                    "corridor": corridor,
                    "hour": hour,
                    "weekday": weekday,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "weather": weather,
                    "rain_intensity": round(rain_intensity, 3),
                    "event_intensity": round(event_intensity, 3),
                    "incident_count": incident_count,
                    "roadwork_flag": roadwork_flag,
                    "flow": flow,
                    "capacity": int(capacity),
                    "demand_capacity_ratio": round(demand_ratio, 4),
                    "avg_speed": round(speed, 3),
                    "occupancy": round(occupancy, 4),
                    "bus_delay_min": round(bus_delay_min, 3),
                    "metro_inflow": round(metro_inflow, 1),
                    "congestion_index": round(congestion_index, 3),
                    "congestion_level": level,
                    "congestion_label": label,
                }
            )

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def load_or_build_traffic_data(
    data_dir: Path,
    force_regenerate: bool = False,
    max_days: int = 35,
) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "chengdu_traffic_hourly.csv"

    if path.exists() and not force_regenerate:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        df = generate_city_traffic_data(days=max_days, seed=42)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    if not df.empty and max_days > 0:
        latest = pd.to_datetime(df["timestamp"].max())
        df = df[pd.to_datetime(df["timestamp"]) >= latest - pd.Timedelta(days=max_days)]

    return df.sort_values("timestamp").reset_index(drop=True)


def _clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_rss_items(url: str, source: str, timeout: int = 10, max_items: int = 45) -> List[Dict[str, str]]:
    resp = requests.get(url, timeout=timeout, headers=HEADERS)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = []
    for item in root.findall("./channel/item")[:max_items]:
        title = _clean_text(item.findtext("title", default=""))
        desc = _clean_text(item.findtext("description", default=""))
        link = item.findtext("link", default="")
        pub = item.findtext("pubDate", default="")
        text = f"{title} {desc}".strip()
        if not text:
            continue
        items.append(
            {
                "title": title,
                "summary": desc,
                "text": text,
                "link": link,
                "published_at": pub,
                "source": source,
            }
        )
    return items


def _hotspot_tag(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["事故", "追尾", "封闭", "施工", "地灾"]):
        return "incident"
    if any(k in t for k in ["会展", "演出", "活动", "景区", "大型集会"]):
        return "event"
    if any(k in t for k in ["地铁", "公交", "换乘"]):
        return "transit"
    if any(k in t for k in ["暴雨", "强对流", "降雨", "台风"]):
        return "weather"
    return "general"


def _impact_score(text: str) -> float:
    score = 35.0
    weights = {
        "拥堵": 9,
        "治堵": 7,
        "封闭": 8,
        "事故": 11,
        "暴雨": 8,
        "演出": 5,
        "会展": 5,
        "活动": 4,
        "地铁": 5,
        "公交": 4,
        "机场": 5,
        "高峰": 6,
    }
    for k, v in weights.items():
        if k in text:
            score += v
    return float(np.clip(score, 20, 95))


def collect_online_hotspots(
    output_file: Path,
    keywords: Sequence[str] | None = None,
    timeout: int = 10,
) -> HotspotCollectResult:
    if keywords is None:
        keywords = [
            "成都 治堵",
            "成都 交通 拥堵",
            "成都 地铁 客流",
            "成都 会展 交通",
            "成都 景区 交通",
            "四川 暴雨 交通",
        ]

    records: List[Dict[str, str]] = []
    errors: List[str] = []

    for kw in keywords:
        google_url = (
            "https://news.google.com/rss/search?q="
            + quote_plus(kw)
            + "&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        )
        bing_url = (
            "https://www.bing.com/news/search?q="
            + quote_plus(kw)
            + "&format=rss&mkt=zh-CN"
        )
        try:
            for item in _fetch_rss_items(google_url, source="google_news", timeout=timeout, max_items=28):
                item["keyword"] = kw
                records.append(item)
        except Exception as e:
            errors.append(f"Google[{kw}] {e}")

        try:
            for item in _fetch_rss_items(bing_url, source="bing_news", timeout=timeout, max_items=28):
                item["keyword"] = kw
                records.append(item)
        except Exception:
            # Bing RSS occasionally returns HTML in some networks.
            pass

    if not records:
        now = pd.Timestamp.now()
        fallback = [
            "成都中心城区重点时段开展拥堵治理行动，强化路口精细化配时。",
            "大型会展活动与周边商圈叠加，夜间客流显著上升。",
            "节假日核心景区周边道路出现阶段性拥堵。",
            "通勤早高峰地铁换乘站客流增长，公交接驳需求提升。",
        ]
        for i, t in enumerate(fallback):
            records.append(
                {
                    "title": t,
                    "summary": t,
                    "text": t,
                    "link": "",
                    "published_at": (now - pd.Timedelta(hours=i * 4)).isoformat(),
                    "source": "fallback_demo",
                    "keyword": "fallback",
                }
            )

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["title", "published_at", "source"]).reset_index(drop=True)
    df["impact_score"] = df["text"].astype(str).map(_impact_score).round(2)
    df["tag"] = df["text"].astype(str).map(_hotspot_tag)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.sort_values(["impact_score", "published_at"], ascending=[False, False]).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    src_counts = df["source"].value_counts().to_dict()
    msg = f"在线热点更新完成，共 {len(df)} 条。"
    if errors:
        msg += f" 部分源不可达：{'; '.join(errors[:2])}"

    return HotspotCollectResult(
        success=True,
        rows=len(df),
        output_file=str(output_file),
        message=msg,
        source_counts=src_counts,
    )


def load_hotspots_data(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "chengdu_hotspots_online.csv"
    if not path.exists():
        return pd.DataFrame(columns=["title", "summary", "text", "link", "published_at", "source", "keyword", "impact_score", "tag"])
    df = pd.read_csv(path)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df


def corridor_list() -> List[str]:
    return [str(c["corridor"]) for c in CITY_CORRIDORS]
