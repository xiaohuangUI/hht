from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return repo_root / p


def _resolve_model(repo_root: Path, model: str) -> str:
    model_path = _resolve_path(repo_root, model)
    if model_path.exists():
        return str(model_path)

    alias = Path(model).name
    if alias and not alias.endswith(".pt"):
        alias = f"{alias}.pt"

    weights_candidate = repo_root / "data" / "city_data" / "weights" / alias
    if weights_candidate.exists():
        return str(weights_candidate)

    return model


def _ensure_ultralytics_cfg(repo_root: Path) -> None:
    cfg_dir = repo_root / "data" / "city_data" / "ultralytics_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(cfg_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 YOLO 训练脚本（交通场景）")
    parser.add_argument("--data", default="data/city_data/vision_datasets/training/visdrone_traffic_yolo/data.yaml")
    parser.add_argument("--model", default="data/city_data/weights/yolo11n.pt")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="")
    parser.add_argument("--project", default="runs/detect_traffic")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root()
    _ensure_ultralytics_cfg(repo_root)

    data_yaml = _resolve_path(repo_root, args.data)
    if not data_yaml.exists():
        print(f"[错误] 未找到数据配置文件: {data_yaml}")
        print("[提示] 先执行: python scripts/prepare_traffic_training_data.py")
        return 2

    model_ref = _resolve_model(repo_root, args.model)
    project_dir = _resolve_path(repo_root, args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        print(f"[错误] 缺少训练依赖: {exc}")
        print("[提示] 在训练环境中安装: pip install ultralytics torch torchvision")
        return 3

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    run_name = args.name.strip() or f"traffic_{Path(str(model_ref)).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("=== 训练配置 ===")
    print(f"data:    {data_yaml}")
    print(f"model:   {model_ref}")
    print(f"device:  {device}")
    print(f"epochs:  {args.epochs}")
    print(f"imgsz:   {args.imgsz}")
    print(f"batch:   {args.batch}")
    print(f"workers: {args.workers}")
    print(f"project: {project_dir}")
    print(f"name:    {run_name}")

    model = YOLO(model_ref)
    results = model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        device=device,
        project=str(project_dir),
        name=run_name,
        patience=int(args.patience),
        cache=bool(args.cache),
        resume=bool(args.resume),
        exist_ok=bool(args.exist_ok),
        verbose=True,
    )

    run_dir = Path(getattr(model.trainer, "save_dir", project_dir / run_name))
    results_csv = run_dir / "results.csv"
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    last_metrics = {}
    if results_csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(results_csv)
            if not df.empty:
                last = df.iloc[-1].to_dict()
                keys = [
                    "epoch",
                    "metrics/mAP50(B)",
                    "metrics/mAP50-95(B)",
                    "metrics/precision(B)",
                    "metrics/recall(B)",
                    "train/box_loss",
                    "val/box_loss",
                ]
                last_metrics = {k: last.get(k) for k in keys if k in last}
        except Exception:
            last_metrics = {}

    output_dir = repo_root / "data" / "city_data" / "training_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir.resolve()),
        "results_csv": str(results_csv.resolve()) if results_csv.exists() else "",
        "best_pt": str(best_pt.resolve()) if best_pt.exists() else "",
        "last_pt": str(last_pt.resolve()) if last_pt.exists() else "",
        "device": device,
        "model": str(model_ref),
        "data_yaml": str(data_yaml.resolve()),
        "metrics_last": last_metrics,
    }
    (output_dir / "latest_train_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 训练完成 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    _ = results
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
