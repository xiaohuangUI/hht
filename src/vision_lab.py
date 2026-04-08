from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw

LABEL_ZH = {
    "car": "小汽车",
    "bus": "公交车",
    "person": "行人",
    "motorcycle": "摩托车",
    "truck": "货车",
    "bicycle": "自行车",
}

MSG_OK = "高精度目标检测推理完成。"
MSG_EMPTY = "模型已运行，当前阈值下未检出目标。"
MSG_FALLBACK = "当前环境未启用高精度模型，无法输出可靠检测框。"


@dataclass(frozen=True)
class VisionResult:
    figure: go.Figure
    object_table: pd.DataFrame
    engine: str
    message: str
    latency_ms: float
    model_name: str
    original_image_bytes: bytes
    annotated_image_bytes: bytes


_MODEL_CACHE: Dict[str, object] = {}
_ULTRALYTICS_CFG_DIR = Path(__file__).resolve().parents[1] / "data" / "city_data" / "ultralytics_cfg"
_VISION_BRIDGE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vision_infer_worker.py"
_VISION_TMP_DIR = Path(__file__).resolve().parents[1] / "data" / "city_data" / "tmp" / "vision_bridge"
_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "data" / "city_data" / "weights"


def detect_available_device() -> Tuple[str, str, bool]:
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            return "cuda:0", str(torch.cuda.get_device_name(0)), True
    except Exception:
        pass
    return "cpu", "CPU", False


def _prepare_ultralytics_env() -> None:
    try:
        _ULTRALYTICS_CFG_DIR.mkdir(parents=True, exist_ok=True)
        cfg = str(_ULTRALYTICS_CFG_DIR)
        os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", cfg)
        os.environ.setdefault("YOLO_CONFIG_DIR", cfg)
    except Exception:
        pass


def _safe_progress(cb: Optional[Callable[[int, str], None]], pct: int, text: str) -> None:
    if cb is None:
        return
    try:
        cb(int(max(0, min(100, pct))), str(text))
    except Exception:
        pass


def normalize_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return "yolo11n.pt"

    key = raw.lower().replace(" ", "")
    alias_map = {
        "v8n": "yolov8n.pt",
        "v8s": "yolov8s.pt",
        "v8m": "yolov8m.pt",
        "v8l": "yolov8l.pt",
        "v8x": "yolov8x.pt",
        "v9t": "yolov9t.pt",
        "v9s": "yolov9s.pt",
        "v9m": "yolov9m.pt",
        "v9c": "yolov9c.pt",
        "v9e": "yolov9e.pt",
        "v10n": "yolov10n.pt",
        "v10s": "yolov10s.pt",
        "v10m": "yolov10m.pt",
        "v10b": "yolov10b.pt",
        "v10l": "yolov10l.pt",
        "v10x": "yolov10x.pt",
        "v11n": "yolo11n.pt",
        "v11s": "yolo11s.pt",
        "v11m": "yolo11m.pt",
        "v11l": "yolo11l.pt",
        "v11x": "yolo11x.pt",
    }
    if key in alias_map:
        return alias_map[key]

    if key.startswith("yolo") and not key.endswith(".pt"):
        return f"{key}.pt"
    return raw


def _resolve_model_reference(model_name: str) -> str:
    text = str(model_name or "").strip()
    if not text:
        return "yolo11n.pt"
    p = Path(text)
    if p.exists():
        return str(p.resolve())
    local_p = _WEIGHTS_DIR / text
    if local_p.exists():
        return str(local_p.resolve())
    return text


def _image_to_array(image_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    out = BytesIO()
    Image.fromarray(arr).save(out, format="PNG")
    return out.getvalue()


def _resize_array(arr: np.ndarray, max_edge: int) -> Tuple[np.ndarray, float]:
    h, w = arr.shape[:2]
    if max_edge <= 0:
        return arr, 1.0
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return arr, 1.0

    scale = float(max_edge) / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
    return resized, scale


def _fallback_detection() -> Tuple[List[Dict[str, float]], str]:
    return [], MSG_FALLBACK


def _bridge_detect_with_pytorch_env(
    arr: np.ndarray,
    conf: float,
    model_name: str,
    imgsz: int,
    device: str,
) -> Tuple[List[Dict[str, float]], str]:
    if not _VISION_BRIDGE_SCRIPT.exists():
        return _fallback_detection()

    _VISION_TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="vision_bridge_", dir=str(_VISION_TMP_DIR)))
    input_path = temp_dir / "input.png"
    output_path = temp_dir / "output.json"
    try:
        Image.fromarray(arr).save(input_path, format="PNG")
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return _fallback_detection()

    base_cmd = [
        str(_VISION_BRIDGE_SCRIPT),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        str(model_name),
        "--conf",
        str(float(conf)),
        "--imgsz",
        str(int(imgsz)),
        "--device",
        str(device),
        "--cfg-dir",
        str(_ULTRALYTICS_CFG_DIR),
    ]
    cmd_candidates = [
        ["conda", "run", "-n", "pytorch", "python"] + base_cmd,
        ["python"] + base_cmd,
    ]

    result: Tuple[List[Dict[str, float]], str] = _fallback_detection()
    try:
        for cmd in cmd_candidates:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(_VISION_BRIDGE_SCRIPT.parent.parent),
                    capture_output=True,
                    text=True,
                    timeout=300,
                    shell=False,
                )
            except Exception:
                continue

            if proc.returncode != 0:
                continue
            if not output_path.exists():
                continue
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if bool(payload.get("ok")):
                boxes = payload.get("boxes", [])
                if isinstance(boxes, list):
                    if len(boxes) == 0:
                        result = ([], MSG_EMPTY)
                    else:
                        result = ([x for x in boxes if isinstance(x, dict)], MSG_OK)
                    break
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return result


def _draw_figure(arr: np.ndarray, boxes: List[Dict[str, float]]) -> go.Figure:
    h, w, _ = arr.shape
    fig = go.Figure()
    fig.add_trace(go.Image(z=arr))

    palette = {
        "car": "#ef4444",
        "bus": "#f59e0b",
        "person": "#0ea5e9",
        "motorcycle": "#22c55e",
        "truck": "#f97316",
        "bicycle": "#14b8a6",
    }

    for b in boxes:
        label = str(b.get("label", ""))
        color = palette.get(label, "#ef4444")
        label_cn = LABEL_ZH.get(label, label)
        x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])

        fig.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line=dict(color=color, width=2),
            fillcolor="rgba(0,0,0,0)",
        )
        fig.add_annotation(
            x=x1,
            y=max(0, y1 - 8),
            text=f"{label_cn} {float(b.get('conf', 0.0)):.2f}",
            showarrow=False,
            align="left",
            font=dict(color="#111827", size=12),
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor=color,
            borderwidth=1,
        )

    fig.update_layout(
        title="交通视觉识别结果",
        margin=dict(l=8, r=8, t=40, b=8),
        height=460,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, w]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[h, 0]),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
    )
    return fig


def _draw_annotated_image(arr: np.ndarray, boxes: List[Dict[str, float]]) -> bytes:
    img = Image.fromarray(arr.copy())
    draw = ImageDraw.Draw(img)

    palette = {
        "car": "#ef4444",
        "bus": "#f59e0b",
        "person": "#0ea5e9",
        "motorcycle": "#22c55e",
        "truck": "#f97316",
        "bicycle": "#14b8a6",
    }

    for b in boxes:
        label = str(b.get("label", ""))
        label_cn = LABEL_ZH.get(label, label)
        color = palette.get(label, "#ef4444")

        x1, y1, x2, y2 = [int(round(float(v))) for v in [b["x1"], b["y1"], b["x2"], b["y2"]]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        text = f"{label_cn} {float(b.get('conf', 0.0)):.2f}"
        tx = max(0, x1 + 2)
        ty = max(0, y1 - 22)
        try:
            text_box = draw.textbbox((tx, ty), text)
            draw.rectangle(text_box, fill=(255, 255, 255))
        except Exception:
            draw.rectangle((tx, ty, tx + max(90, len(text) * 12), ty + 18), fill=(255, 255, 255))
        draw.text((tx + 2, ty + 1), text, fill="#1f2a28")

    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _yolo_detect(
    arr: np.ndarray,
    conf: float,
    model_name: str = "yolo11n.pt",
    imgsz: int = 960,
    device: str = "auto",
) -> Tuple[List[Dict[str, float]], str]:
    _prepare_ultralytics_env()
    key = normalize_model_name(model_name)
    model_ref = _resolve_model_reference(key)

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return _bridge_detect_with_pytorch_env(arr, conf=conf, model_name=model_ref, imgsz=imgsz, device=device)

    try:
        model = _MODEL_CACHE.get(model_ref)
        if model is None:
            model = YOLO(model_ref)
            _MODEL_CACHE[model_ref] = model

        predict_kwargs: Dict[str, object] = {
            "conf": float(conf),
            "imgsz": int(imgsz),
            "verbose": False,
        }
        dev = str(device or "auto").strip().lower()
        if dev and dev != "auto":
            predict_kwargs["device"] = dev
            if dev.startswith("cuda"):
                predict_kwargs["half"] = True

        result = model.predict(arr, **predict_kwargs)[0]
        names = result.names
        boxes_out: List[Dict[str, float]] = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                label = str(names.get(int(cls[i]), int(cls[i])))
                boxes_out.append(
                    {
                        "label": label,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "conf": float(confs[i]),
                    }
                )

        if not boxes_out:
            return [], MSG_EMPTY
        return boxes_out, MSG_OK
    except Exception:
        return _bridge_detect_with_pytorch_env(arr, conf=conf, model_name=model_ref, imgsz=imgsz, device=device)


def run_vision_detection(
    image_bytes: bytes,
    conf: float = 0.25,
    model_name: str = "yolo11n.pt",
    max_display_edge: int = 1440,
    max_infer_edge: int = 960,
    imgsz: int = 960,
    device: str = "auto",
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> VisionResult:
    _safe_progress(progress_callback, 8, "正在读取图像")
    arr_raw = _image_to_array(image_bytes)

    _safe_progress(progress_callback, 18, "正在生成展示分辨率")
    arr_display, display_scale = _resize_array(arr_raw, max_display_edge)

    _safe_progress(progress_callback, 30, "正在生成推理分辨率")
    arr_infer, infer_scale = _resize_array(arr_raw, max_infer_edge)

    norm_name = normalize_model_name(model_name)

    _safe_progress(progress_callback, 45, "正在执行模型推理")
    t0 = perf_counter()
    boxes_infer, msg = _yolo_detect(arr_infer, conf=conf, model_name=norm_name, imgsz=imgsz, device=device)
    latency_ms = (perf_counter() - t0) * 1000.0

    _safe_progress(progress_callback, 78, "正在生成结果图")
    scale_to_display = float(display_scale) / float(infer_scale if infer_scale > 0 else 1.0)
    boxes: List[Dict[str, float]] = []
    for b in boxes_infer:
        boxes.append(
            {
                "label": str(b["label"]),
                "x1": float(b["x1"]) * scale_to_display,
                "y1": float(b["y1"]) * scale_to_display,
                "x2": float(b["x2"]) * scale_to_display,
                "y2": float(b["y2"]) * scale_to_display,
                "conf": float(b["conf"]),
            }
        )

    fig = _draw_figure(arr_display, boxes)
    annotated = _draw_annotated_image(arr_display, boxes)
    preview = _array_to_png_bytes(arr_display)

    _safe_progress(progress_callback, 92, "正在整理统计信息")
    if not boxes:
        table = pd.DataFrame(columns=["类别", "数量", "平均置信度"])
    else:
        df = pd.DataFrame(boxes)
        df["类别中文"] = df["label"].astype(str).map(lambda x: LABEL_ZH.get(x, x))
        table = (
            df.groupby("类别中文", as_index=False)
            .agg(数量=("类别中文", "count"), 平均置信度=("conf", "mean"))
            .rename(columns={"类别中文": "类别"})
            .sort_values("数量", ascending=False)
        )
        table["平均置信度"] = table["平均置信度"].round(3)

    engine = "高精度识别引擎" if msg in {MSG_OK, MSG_EMPTY} else "回退模式"
    _safe_progress(progress_callback, 100, "识别完成")

    return VisionResult(
        figure=fig,
        object_table=table,
        engine=engine,
        message=msg,
        latency_ms=float(latency_ms),
        model_name=norm_name,
        original_image_bytes=preview,
        annotated_image_bytes=annotated,
    )


def compare_yolo_versions(
    image_items: List[Tuple[str, bytes]],
    model_versions: List[str],
    conf: float = 0.25,
    max_infer_edge: int = 960,
    imgsz: int = 960,
    device: str = "auto",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    clean_models: List[str] = []
    for m in model_versions:
        mm = normalize_model_name(str(m))
        if mm and mm not in clean_models:
            clean_models.append(mm)
    if not clean_models:
        return pd.DataFrame(
            columns=["模型版本", "样本数", "检测总目标数", "平均置信度", "平均耗时毫秒", "运行状态"]
        )

    prepared_arrays: List[np.ndarray] = []
    for _, image_bytes in image_items:
        try:
            arr = _image_to_array(image_bytes)
            arr_infer, _ = _resize_array(arr, max_infer_edge)
            prepared_arrays.append(arr_infer)
        except Exception:
            continue
    if not prepared_arrays:
        return pd.DataFrame(
            columns=["模型版本", "样本数", "检测总目标数", "平均置信度", "平均耗时毫秒", "运行状态"]
        )

    total_steps = int(len(clean_models) * len(prepared_arrays))
    done_steps = 0

    for model_name in clean_models:
        total_objs = 0
        conf_list: List[float] = []
        latencies: List[float] = []
        fallback_count = 0

        for arr in prepared_arrays:
            t0 = perf_counter()
            boxes, msg = _yolo_detect(arr, conf=conf, model_name=model_name, imgsz=imgsz, device=device)
            latencies.append((perf_counter() - t0) * 1000.0)

            if msg not in {MSG_OK, MSG_EMPTY}:
                fallback_count += 1

            total_objs += int(len(boxes))
            conf_list.extend([float(b.get("conf", 0.0)) for b in boxes])

            done_steps += 1
            if progress_callback is not None:
                try:
                    progress_callback(done_steps, total_steps, f"{model_name} 推理中")
                except Exception:
                    pass

        avg_conf = float(np.mean(conf_list)) if conf_list else 0.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        status = (
            "高精度识别"
            if fallback_count == 0
            else ("部分回退" if fallback_count < len(prepared_arrays) else "演示回退")
        )

        rows.append(
            {
                "模型版本": model_name,
                "样本数": int(len(prepared_arrays)),
                "检测总目标数": int(total_objs),
                "平均置信度": round(avg_conf, 3),
                "平均耗时毫秒": round(avg_latency, 1),
                "运行状态": status,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["运行状态", "平均耗时毫秒"], ascending=[True, True])
        .reset_index(drop=True)
    )
