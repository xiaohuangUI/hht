from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.dataset_hub import (
    import_dataset_from_kaggle,
    import_dataset_from_url,
    scan_raw_datasets,
    sync_dataset_samples_to_vision,
)

DATA_DIR = BASE_DIR / "data" / "city_data"
SAMPLE_DIR = DATA_DIR / "vision_samples"

PRESET_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
]


def _print_progress(pct: int, detail: str) -> None:
    print(f"[{max(0, min(100, int(pct))):>3}%] {detail}")


def _run_url_import(urls: List[str], alias_prefix: str = "") -> None:
    if not urls:
        return
    print(f"\n开始导入 URL 数据源，共 {len(urls)} 条...")
    for idx, url in enumerate(urls, start=1):
        alias = f"{alias_prefix}_url_{idx}" if alias_prefix else ""
        print(f"\n[{idx}/{len(urls)}] {url}")
        ok, msg, _ = import_dataset_from_url(DATA_DIR, url=url, alias=alias, progress_callback=_print_progress)
        print("结果:", msg if ok else f"失败 - {msg}")


def _run_kaggle_import(slugs: List[str], alias_prefix: str = "") -> None:
    if not slugs:
        return
    print(f"\n开始导入 Kaggle 数据源，共 {len(slugs)} 条...")
    for idx, slug in enumerate(slugs, start=1):
        alias = f"{alias_prefix}_kg_{idx}" if alias_prefix else ""
        print(f"\n[{idx}/{len(slugs)}] {slug}")
        ok, msg, _ = import_dataset_from_kaggle(DATA_DIR, dataset_slug=slug, alias=alias, progress_callback=_print_progress)
        print("结果:", msg if ok else f"失败 - {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="交通视觉数据集抓取与同步脚本")
    parser.add_argument("--url", action="append", default=[], help="可重复传入数据集下载链接")
    parser.add_argument("--kaggle", action="append", default=[], help="可重复传入Kaggle标识，如 user/dataset")
    parser.add_argument("--preset", action="store_true", help="导入内置示例数据源")
    parser.add_argument("--alias-prefix", default="", help="导入别名前缀")
    parser.add_argument("--scan-raw", action="store_true", help="扫描 raw 目录并补录台账")
    parser.add_argument("--sync-images", type=int, default=24, help="导入后同步到视觉样本库的图片数量，0表示不同步")
    args = parser.parse_args()

    urls = list(args.url or [])
    if args.preset:
        for u in PRESET_URLS:
            if u not in urls:
                urls.append(u)
    kaggle_slugs = list(args.kaggle or [])

    if not urls and not kaggle_slugs and not args.scan_raw:
        print("未提供数据源。可使用 --preset 或 --url / --kaggle。")
        return

    _run_url_import(urls, alias_prefix=args.alias_prefix)
    _run_kaggle_import(kaggle_slugs, alias_prefix=args.alias_prefix)

    if args.scan_raw:
        added, skipped, notices = scan_raw_datasets(DATA_DIR)
        print(f"\nraw扫描完成：新增 {added}，跳过 {skipped}")
        for row in notices[:5]:
            print("提示:", row)

    if int(args.sync_images) > 0:
        ok, msg, paths = sync_dataset_samples_to_vision(DATA_DIR, SAMPLE_DIR, selected_ids=None, max_images=int(args.sync_images))
        print(f"\n样本同步：{msg}")
        if ok:
            for p in paths[:10]:
                print("新增样本:", p.name)


if __name__ == "__main__":
    main()
