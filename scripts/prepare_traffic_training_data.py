from __future__ import annotations

import json
import io
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


RAW_DIR = Path("data/city_data/vision_datasets/raw")
TRAIN_ROOT = Path("data/city_data/vision_datasets/training")
VISDRONE_OUT = TRAIN_ROOT / "visdrone_traffic_yolo"
COCO_OUT = TRAIN_ROOT / "coco128_raw"

# VisDrone 类别映射 -> 交通检测6类
# 1:pedestrian,2:people,3:bicycle,4:car,5:van,6:truck,7:tricycle,8:awning-tricycle,9:bus,10:motor
CLASS_MAP = {
    1: 0,  # 行人
    2: 0,  # 行人群
    3: 1,  # 自行车
    4: 2,  # 小汽车
    5: 2,  # 货车/面包等归并为小汽车类
    6: 3,  # 货车
    7: 5,  # 摩托/三轮归并
    8: 5,
    9: 4,  # 公交车
    10: 5,  # 摩托车
}

NAMES = ["行人", "自行车", "小汽车", "货车", "公交车", "摩托车"]


def _ensure_dirs() -> None:
    (VISDRONE_OUT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (VISDRONE_OUT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (VISDRONE_OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (VISDRONE_OUT / "labels" / "val").mkdir(parents=True, exist_ok=True)
    COCO_OUT.mkdir(parents=True, exist_ok=True)


def _convert_visdrone_ann(ann_text: str, width: int, height: int) -> list[str]:
    rows: list[str] = []
    if width <= 1 or height <= 1:
        return rows
    for raw in ann_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        seg = [x.strip() for x in line.split(",")]
        if len(seg) < 6:
            continue
        try:
            x = float(seg[0])
            y = float(seg[1])
            w = float(seg[2])
            h = float(seg[3])
            cls_id = int(float(seg[5]))
        except Exception:
            continue
        if w <= 1 or h <= 1:
            continue
        if cls_id not in CLASS_MAP:
            continue
        yolo_cls = CLASS_MAP[cls_id]
        xc = (x + w / 2.0) / float(width)
        yc = (y + h / 2.0) / float(height)
        wn = w / float(width)
        hn = h / float(height)
        if wn <= 0 or hn <= 0:
            continue
        if xc <= 0 or yc <= 0 or xc >= 1 or yc >= 1:
            continue
        rows.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return rows


def _process_visdrone_zip(zip_path: Path, split: str) -> dict[str, int]:
    out_img = VISDRONE_OUT / "images" / split
    out_lab = VISDRONE_OUT / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    img_count = 0
    label_count = 0
    obj_count = 0
    cls_counter = {k: 0 for k in range(len(NAMES))}

    with ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        image_files = [n for n in names if "/images/" in n and n.lower().endswith((".jpg", ".jpeg", ".png"))]
        ann_files = [n for n in names if "/annotations/" in n and n.lower().endswith(".txt")]
        ann_map = {Path(x).stem: x for x in ann_files}

        total = max(1, len(image_files))
        for idx, img_name in enumerate(image_files, start=1):
            stem = Path(img_name).stem
            raw_img = zf.read(img_name)
            try:
                im = Image.open(io.BytesIO(raw_img)).convert("RGB")
            except Exception:
                continue
            w, h = im.size

            out_img_path = out_img / f"{stem}.jpg"
            im.save(out_img_path, format="JPEG", quality=92)
            img_count += 1

            ann_name = ann_map.get(stem)
            ann_text = ""
            if ann_name:
                try:
                    ann_text = zf.read(ann_name).decode("utf-8", errors="ignore")
                except Exception:
                    ann_text = ""

            yolo_rows = _convert_visdrone_ann(ann_text, width=w, height=h)
            if yolo_rows:
                (out_lab / f"{stem}.txt").write_text("\n".join(yolo_rows), encoding="utf-8")
                label_count += 1
                obj_count += len(yolo_rows)
                for row in yolo_rows:
                    try:
                        cid = int(row.split(" ", 1)[0])
                        if cid in cls_counter:
                            cls_counter[cid] += 1
                    except Exception:
                        pass
            else:
                (out_lab / f"{stem}.txt").write_text("", encoding="utf-8")

            if idx % 400 == 0 or idx == total:
                print(f"[{split}] {idx}/{total} 已处理")

    return {
        "images": img_count,
        "labels": label_count,
        "objects": obj_count,
        "cls_0": cls_counter[0],
        "cls_1": cls_counter[1],
        "cls_2": cls_counter[2],
        "cls_3": cls_counter[3],
        "cls_4": cls_counter[4],
        "cls_5": cls_counter[5],
    }


def _extract_coco128() -> None:
    zip_path = RAW_DIR / "COCO128_交通基准.zip"
    if not zip_path.exists():
        print("未检测到 COCO128_交通基准.zip，跳过COCO提取。")
        return
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(COCO_OUT)
    print("COCO128 提取完成。")


def _write_yaml() -> None:
    yaml_text = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 6\n"
        "names: ['person', 'bicycle', 'car', 'truck', 'bus', 'motorcycle']\n"
    )
    (VISDRONE_OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    train_zip = RAW_DIR / "VisDrone2019_训练集.zip"
    val_zip = RAW_DIR / "VisDrone2019_验证集.zip"
    if not train_zip.exists() or not val_zip.exists():
        raise FileNotFoundError("未找到 VisDrone 训练/验证压缩包，请先在 raw 目录确认文件。")

    print("开始转换 VisDrone 训练集...")
    train_stats = _process_visdrone_zip(train_zip, split="train")
    print("开始转换 VisDrone 验证集...")
    val_stats = _process_visdrone_zip(val_zip, split="val")
    _write_yaml()
    _extract_coco128()

    summary = {
        "train": train_stats,
        "val": val_stats,
        "classes_cn": NAMES,
        "dataset_root": str(VISDRONE_OUT.resolve()),
    }
    (VISDRONE_OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n转换完成。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n训练集目录: {VISDRONE_OUT.resolve()}")
    print(f"可直接训练命令示例: yolo detect train data={VISDRONE_OUT / 'data.yaml'} model=data/city_data/weights/yolo11n.pt epochs=50 imgsz=960 batch=8")


if __name__ == "__main__":
    main()
