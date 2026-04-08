from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    total = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        total += len(list(folder.glob(ext)))
    return total


def _to_float(row: pd.Series, keys: List[str]) -> float | None:
    for key in keys:
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                continue
    return None


def get_local_training_status(data_dir: Path, repo_root: Path) -> Dict[str, object]:
    raw_dir = data_dir / "vision_datasets" / "raw"
    train_root = data_dir / "vision_datasets" / "training" / "visdrone_traffic_yolo"
    data_yaml = train_root / "data.yaml"
    summary_path = train_root / "summary.json"
    weights_dir = data_dir / "weights"

    raw_archives = sorted([p for p in raw_dir.glob("*.zip") if p.is_file()], key=lambda x: x.name.lower())
    train_images = _count_images(train_root / "images" / "train")
    val_images = _count_images(train_root / "images" / "val")
    train_labels = len(list((train_root / "labels" / "train").glob("*.txt"))) if (train_root / "labels" / "train").exists() else 0
    val_labels = len(list((train_root / "labels" / "val").glob("*.txt"))) if (train_root / "labels" / "val").exists() else 0
    model_weights = sorted([p.name for p in weights_dir.glob("*.pt") if p.is_file()])

    summary_data: Dict[str, object] = {}
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary_data = {}

    dataset_ready = bool(data_yaml.exists() and train_images > 0 and val_images > 0 and train_labels > 0 and val_labels > 0)
    default_model = weights_dir / "yolo11n.pt"
    if not default_model.exists():
        default_model = weights_dir / "yolov8n.pt"
    if not default_model.exists():
        default_model = Path("yolo11n.pt")

    return {
        "raw_archive_count": len(raw_archives),
        "raw_archives": [str(p) for p in raw_archives],
        "train_images": train_images,
        "val_images": val_images,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "dataset_ready": dataset_ready,
        "dataset_root": str(train_root.resolve()),
        "data_yaml": str(data_yaml.resolve()) if data_yaml.exists() else "",
        "summary_path": str(summary_path.resolve()) if summary_path.exists() else "",
        "summary_data": summary_data,
        "weights": model_weights,
        "default_model": str(default_model),
        "train_script": str((repo_root / "scripts" / "train_yolo_local.py").resolve()),
        "prepare_script": str((repo_root / "scripts" / "prepare_traffic_training_data.py").resolve()),
    }


def build_training_commands(status: Dict[str, object]) -> Dict[str, str]:
    data_yaml = str(status.get("data_yaml", "") or "")
    model = str(status.get("default_model", "data/city_data/weights/yolo11n.pt"))
    train_script = str(status.get("train_script", "scripts/train_yolo_local.py"))
    prepare_script = str(status.get("prepare_script", "scripts/prepare_traffic_training_data.py"))

    if data_yaml:
        data_arg = f'--data "{data_yaml}"'
    else:
        data_arg = '--data "data/city_data/vision_datasets/training/visdrone_traffic_yolo/data.yaml"'

    gpu_cmd = (
        f'conda run -n pytorch python "{train_script}" '
        f'{data_arg} --model "{model}" --device cuda:0 --epochs 80 --imgsz 960 --batch 8 --workers 6 --name traffic_yolo11n_gpu'
    )
    cpu_cmd = (
        f'python "{train_script}" {data_arg} --model "{model}" --device cpu '
        f'--epochs 20 --imgsz 640 --batch 4 --workers 2 --name traffic_yolo_cpu'
    )
    prepare_cmd = f'python "{prepare_script}"'

    return {"prepare": prepare_cmd, "gpu": gpu_cmd, "cpu": cpu_cmd}


def list_local_training_runs(repo_root: Path, limit: int = 30) -> pd.DataFrame:
    run_roots = [repo_root / "runs" / "detect", repo_root / "runs" / "detect_traffic", repo_root / "runs" / "train"]
    rows: List[Dict[str, object]] = []

    for root in run_roots:
        if not root.exists():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            results_csv = run_dir / "results.csv"
            if not results_csv.exists():
                continue
            try:
                df = pd.read_csv(results_csv)
            except Exception:
                continue
            if df.empty:
                continue
            last = df.iloc[-1]

            map50 = _to_float(last, ["metrics/mAP50(B)", "metrics/mAP50-95(B)"])
            map5095 = _to_float(last, ["metrics/mAP50-95(B)", "metrics/mAP50(B)"])
            precision = _to_float(last, ["metrics/precision(B)"])
            recall = _to_float(last, ["metrics/recall(B)"])
            train_loss = _to_float(last, ["train/box_loss", "train/cls_loss"])
            val_loss = _to_float(last, ["val/box_loss", "val/cls_loss"])
            epoch = int(last["epoch"]) if "epoch" in last and pd.notna(last["epoch"]) else int(len(df) - 1)

            best_pt = run_dir / "weights" / "best.pt"
            last_pt = run_dir / "weights" / "last.pt"
            rows.append(
                {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir.resolve()),
                    "epoch": epoch,
                    "mAP50": map50,
                    "mAP50-95": map5095,
                    "precision": precision,
                    "recall": recall,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_pt": str(best_pt.resolve()) if best_pt.exists() else "",
                    "last_pt": str(last_pt.resolve()) if last_pt.exists() else "",
                    "updated_at": run_dir.stat().st_mtime,
                }
            )

    if not rows:
        return pd.DataFrame()

    board = pd.DataFrame(rows).sort_values("updated_at", ascending=False).head(limit).reset_index(drop=True)
    board["updated_at"] = pd.to_datetime(board["updated_at"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    return board


def load_training_curve(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    curve = pd.DataFrame()
    curve["epoch"] = df["epoch"] if "epoch" in df.columns else range(len(df))
    curve["mAP50"] = df["metrics/mAP50(B)"] if "metrics/mAP50(B)" in df.columns else None
    curve["mAP50-95"] = df["metrics/mAP50-95(B)"] if "metrics/mAP50-95(B)" in df.columns else None
    curve["precision"] = df["metrics/precision(B)"] if "metrics/precision(B)" in df.columns else None
    curve["recall"] = df["metrics/recall(B)"] if "metrics/recall(B)" in df.columns else None
    curve["train_loss"] = df["train/box_loss"] if "train/box_loss" in df.columns else None
    curve["val_loss"] = df["val/box_loss"] if "val/box_loss" in df.columns else None
    return curve.dropna(axis=1, how="all")
