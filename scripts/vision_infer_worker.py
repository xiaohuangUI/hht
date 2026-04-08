from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="跨环境YOLO推理工作进程")
    parser.add_argument("--input", required=True, help="输入图像路径")
    parser.add_argument("--output", required=True, help="输出JSON路径")
    parser.add_argument("--model", default="yolo11n.pt", help="模型名称")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=960, help="推理尺寸")
    parser.add_argument("--device", default="auto", help="推理设备")
    parser.add_argument("--cfg-dir", default="", help="Ultralytics配置目录")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_dir = str(args.cfg_dir or "").strip()
    if cfg_dir:
        os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", cfg_dir)
        os.environ.setdefault("YOLO_CONFIG_DIR", cfg_dir)

    payload: Dict[str, object] = {"ok": False, "boxes": [], "error": ""}
    try:
        from ultralytics import YOLO  # type: ignore

        img = Image.open(args.input).convert("RGB")
        arr = np.array(img)
        model = YOLO(str(args.model))

        kwargs: Dict[str, object] = {
            "conf": float(args.conf),
            "imgsz": int(args.imgsz),
            "verbose": False,
        }
        dev = str(args.device or "auto").strip().lower()
        if dev and dev != "auto":
            kwargs["device"] = dev
            if dev.startswith("cuda"):
                kwargs["half"] = True

        result = model.predict(arr, **kwargs)[0]
        boxes_out: List[Dict[str, float]] = []
        names = result.names

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

        payload = {"ok": True, "boxes": boxes_out, "error": ""}
    except Exception as exc:
        payload = {"ok": False, "boxes": [], "error": str(exc)}

    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
