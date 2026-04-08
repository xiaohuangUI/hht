from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "data" / "city_data" / "ultralytics_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(cfg_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg_dir))

    print("=== Python 解释器 ===")
    print(sys.executable)

    print("\n=== PyTorch / CUDA ===")
    try:
        import torch  # type: ignore

        print(f"torch 版本: {torch.__version__}")
        cuda_ok = bool(torch.cuda.is_available())
        print(f"CUDA 可用: {cuda_ok}")
        if cuda_ok:
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"torch 不可用: {e}")

    print("\n=== Ultralytics ===")
    try:
        import ultralytics  # type: ignore

        print(f"ultralytics 版本: {getattr(ultralytics, '__version__', 'unknown')}")
    except Exception as e:
        print(f"ultralytics 不可用: {e}")


if __name__ == "__main__":
    main()
