from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import site
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from PIL import Image

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

RELEASE_PAGES = [
    "https://github.com/ultralytics/yolov5/releases/tag/v1.0",
    "https://github.com/ultralytics/assets/releases/tag/v0.0.0",
]

TRAFFIC_KEYWORDS = {
    "coco128",
    "voc",
    "visdrone",
    "xview",
    "wheat",
    "traffic",
    "vehicle",
    "bdd",
}

MANUAL_SOURCES: List[Dict[str, str]] = [
    {
        "数据集名称": "自动驾驶场景视觉数据集（需申请）",
        "下载链接": "https://github.com/bdd100k/bdd100k",
        "来源": "官方仓库",
        "类型": "交通检测",
        "备注": "官方页面提供申请与下载说明",
        "Kaggle标识": "",
    },
    {
        "数据集名称": "城市交通车辆检测数据集（需申请）",
        "下载链接": "https://sites.google.com/view/daweidu/projects/ua-detrac",
        "来源": "官方站点",
        "类型": "交通检测",
        "备注": "部分资源需要填写信息后下载",
        "Kaggle标识": "",
    },
    {
        "数据集名称": "Kaggle交通检测数据集（自定义标识）",
        "下载链接": "https://www.kaggle.com/datasets",
        "来源": "Kaggle",
        "类型": "交通检测",
        "备注": "在界面输入 Kaggle 标识，示例：username/dataset-name",
        "Kaggle标识": "username/dataset-name",
    },
]


def _dataset_dir(base_dir: Path) -> Path:
    return base_dir / "vision_datasets"


def _raw_dir(base_dir: Path) -> Path:
    return _dataset_dir(base_dir) / "raw"


def _export_dir(base_dir: Path) -> Path:
    return _dataset_dir(base_dir) / "exports"


def _tmp_dir(base_dir: Path) -> Path:
    return _dataset_dir(base_dir) / "tmp"


def _registry_file(base_dir: Path) -> Path:
    return _dataset_dir(base_dir) / "dataset_registry.json"


def _sample_index_file(base_dir: Path) -> Path:
    return _dataset_dir(base_dir) / "dataset_sample_index.json"


def _ensure_dirs(base_dir: Path) -> None:
    _raw_dir(base_dir).mkdir(parents=True, exist_ok=True)
    _export_dir(base_dir).mkdir(parents=True, exist_ok=True)
    _tmp_dir(base_dir).mkdir(parents=True, exist_ok=True)


def _load_registry(base_dir: Path) -> List[Dict[str, object]]:
    _ensure_dirs(base_dir)
    path = _registry_file(base_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_registry(base_dir: Path, rows: List[Dict[str, object]]) -> None:
    _ensure_dirs(base_dir)
    _registry_file(base_dir).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_sample_index(base_dir: Path) -> Dict[str, str]:
    path = _sample_index_file(base_dir)
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            return {str(k): str(v) for k, v in rows.items()}
    except Exception:
        pass
    return {}


def _save_sample_index(base_dir: Path, rows: Dict[str, str]) -> None:
    _sample_index_file(base_dir).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_url(url: str) -> str:
    return str(url or "").strip().split("#", 1)[0]


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fr:
        while True:
            chunk = fr.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dedupe_by_source(registry: List[Dict[str, object]], source_link: str = "", source_type: str = "", source_id: str = "") -> Dict[str, object] | None:
    link = _normalize_url(source_link)
    stype = str(source_type or "").strip().lower()
    sid = str(source_id or "").strip().lower()
    for row in registry:
        row_link = _normalize_url(str(row.get("来源链接", "")))
        row_type = str(row.get("来源类型", "")).strip().lower()
        row_id = str(row.get("来源标识", "")).strip().lower()
        if link and row_link and row_link == link:
            return row
        if stype and sid and row_type == stype and row_id == sid:
            return row
    return None


def _dedupe_by_hash(registry: List[Dict[str, object]], sha256: str) -> Dict[str, object] | None:
    digest = str(sha256 or "").strip().lower()
    for row in registry:
        row_digest = str(row.get("sha256", "")).strip().lower()
        if digest and row_digest and row_digest == digest:
            return row
    return None


def _safe_progress(progress_callback: Callable[[int, str], None] | None, pct: int, text: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(max(0, min(100, int(pct))), str(text))
    except Exception:
        pass


def crawl_detection_dataset_sources(timeout_s: int = 15, max_items: int = 80) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    zip_href = re.compile(r'href="([^"]+?\.zip)"', flags=re.IGNORECASE)

    for page in RELEASE_PAGES:
        try:
            resp = requests.get(page, headers=HEADERS, timeout=timeout_s)
            resp.raise_for_status()
            content = resp.text
        except Exception:
            continue

        for m in zip_href.finditer(content):
            href = html.unescape(m.group(1))
            if href.startswith("/"):
                url = f"https://github.com{href}"
            elif href.startswith("http"):
                url = href
            else:
                continue
            file_name = unquote(Path(urlparse(url).path).name)
            file_lower = file_name.lower()
            if not any(k in file_lower for k in TRAFFIC_KEYWORDS):
                continue
            rows.append(
                {
                    "数据集名称": file_name,
                    "下载链接": url,
                    "来源": "GitHub发布页",
                    "类型": "交通检测",
                    "备注": "可直接下载压缩包",
                }
            )

    rows.extend(MANUAL_SOURCES)
    if not rows:
        return pd.DataFrame(columns=["数据集名称", "下载链接", "来源", "类型", "备注", "Kaggle标识"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["下载链接"]).reset_index(drop=True)
    if len(df) > max_items:
        df = df.head(max_items).copy()
    if "Kaggle标识" not in df.columns:
        df["Kaggle标识"] = ""
    return df[["数据集名称", "下载链接", "来源", "类型", "备注", "Kaggle标识"]]


def _count_zip_contents(zip_path: Path) -> Tuple[int, int]:
    image_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    label_ext = (".txt", ".xml", ".json")
    img_count = 0
    ann_count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                low = str(name).lower()
                if low.endswith(image_ext):
                    img_count += 1
                elif low.endswith(label_ext):
                    ann_count += 1
    except Exception:
        return 0, 0
    return img_count, ann_count


def _normalize_name(name: str) -> str:
    clean = re.sub(r"[\\/:*?\"<>|]+", "_", str(name).strip())
    clean = re.sub(r"\s+", "_", clean)
    return clean or "dataset_file"


def import_dataset_from_url(
    base_dir: Path,
    url: str,
    alias: str = "",
    timeout_s: int = 120,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Tuple[bool, str, Dict[str, object] | None]:
    _ensure_dirs(base_dir)
    link = _normalize_url(url)
    if not link.startswith("http"):
        return False, "链接无效，请输入可访问的HTTP/HTTPS链接。", None

    registry = _load_registry(base_dir)
    dup = _dedupe_by_source(registry, source_link=link, source_type="URL")
    if dup is not None:
        return False, "该链接已导入过，已自动跳过重复抓取。", dup

    parsed_name = unquote(Path(urlparse(link).path).name or "")
    parsed_suffix = Path(parsed_name).suffix if parsed_name else ""
    if alias:
        base_alias = _normalize_name(alias)
        if "." not in base_alias and parsed_suffix:
            file_name = f"{base_alias}{parsed_suffix}"
        else:
            file_name = base_alias
    else:
        file_name = _normalize_name(parsed_name or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    if "." not in file_name:
        file_name = f"{file_name}.zip"
    temp_path = _tmp_dir(base_dir) / f"url_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{file_name}"
    final_path = _raw_dir(base_dir) / file_name

    sha = hashlib.sha256()
    total = 0
    first_bytes = b""
    _safe_progress(progress_callback, 5, "开始下载数据集")
    try:
        with requests.get(link, headers=HEADERS, timeout=timeout_s, stream=True) as resp:
            resp.raise_for_status()
            total_header = int(resp.headers.get("content-length", "0") or "0")
            content_type = str(resp.headers.get("content-type", "")).lower()
            with temp_path.open("wb") as fw:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    if not first_bytes:
                        first_bytes = bytes(chunk[:1024])
                    fw.write(chunk)
                    sha.update(chunk)
                    total += len(chunk)
                    if total_header > 0:
                        pct = int(10 + min(70, (total / total_header) * 70))
                        _safe_progress(progress_callback, pct, f"正在下载：{total / 1024 / 1024:.1f}MB")
            is_html = b"<html" in first_bytes.lower() or b"<!doctype" in first_bytes.lower()
            if "text/html" in content_type or is_html:
                temp_path.unlink(missing_ok=True)
                return False, "该链接返回的是网页而不是可直接下载的数据文件，请使用文件直链。", None
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return False, f"下载失败：{exc}", None

    digest = sha.hexdigest()
    dup = _dedupe_by_hash(registry, digest)
    if dup is not None:
        temp_path.unlink(missing_ok=True)
        return False, "该数据与已导入文件一致，已自动去重。", dup

    if final_path.exists():
        stem = _normalize_name(Path(file_name).stem)
        suffix = Path(file_name).suffix
        final_path = _raw_dir(base_dir) / f"{stem}_{datetime.now().strftime('%H%M%S')}{suffix}"
    temp_path.replace(final_path)
    _safe_progress(progress_callback, 85, "正在解析数据集结构")

    img_count, ann_count = _count_zip_contents(final_path) if final_path.suffix.lower() == ".zip" else (0, 0)
    rec = {
        "编号": digest[:12],
        "数据集名称": alias or Path(final_path).name,
        "来源链接": link,
        "来源类型": "URL",
        "来源标识": "",
        "本地文件": str(final_path),
        "文件大小字节": int(total),
        "sha256": digest,
        "图片文件数": int(img_count),
        "标注文件数": int(ann_count),
        "导入时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    registry.append(rec)
    _save_registry(base_dir, registry)
    _safe_progress(progress_callback, 100, "导入完成")
    return True, "导入成功。", rec


def import_dataset_from_upload(
    base_dir: Path,
    file_name: str,
    file_bytes: bytes,
    alias: str = "",
) -> Tuple[bool, str, Dict[str, object] | None]:
    _ensure_dirs(base_dir)
    if not file_bytes:
        return False, "上传内容为空，请重新选择文件。", None

    safe_name = _normalize_name(file_name or "upload_dataset.zip")
    if "." not in safe_name:
        safe_name = f"{safe_name}.zip"

    registry = _load_registry(base_dir)
    digest = hashlib.sha256(file_bytes).hexdigest()
    dup = _dedupe_by_hash(registry, digest)
    if dup is not None:
        return False, "该文件与已导入数据完全一致，已自动去重。", dup

    final_path = _raw_dir(base_dir) / safe_name
    if final_path.exists():
        stem = _normalize_name(Path(safe_name).stem)
        suffix = Path(safe_name).suffix
        final_path = _raw_dir(base_dir) / f"{stem}_{datetime.now().strftime('%H%M%S')}{suffix}"

    try:
        final_path.write_bytes(file_bytes)
    except Exception as exc:
        return False, f"保存失败：{exc}", None

    img_count, ann_count = _count_zip_contents(final_path) if final_path.suffix.lower() == ".zip" else (0, 0)
    rec = {
        "编号": digest[:12],
        "数据集名称": alias or Path(final_path).name,
        "来源链接": f"upload://{safe_name}",
        "来源类型": "上传",
        "来源标识": safe_name,
        "本地文件": str(final_path),
        "文件大小字节": int(len(file_bytes)),
        "sha256": digest,
        "图片文件数": int(img_count),
        "标注文件数": int(ann_count),
        "导入时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    registry.append(rec)
    _save_registry(base_dir, registry)
    return True, "本地文件导入成功。", rec


def load_dataset_registry_df(base_dir: Path) -> pd.DataFrame:
    rows = _load_registry(base_dir)
    if not rows:
        return pd.DataFrame(
            columns=["编号", "数据集名称", "文件大小(MB)", "图片文件数", "标注文件数", "来源类型", "导入时间", "来源链接", "本地文件"]
        )
    df = pd.DataFrame(rows)
    if "文件大小字节" in df.columns:
        df["文件大小(MB)"] = (df["文件大小字节"].astype(float) / 1024 / 1024).round(2)
    else:
        df["文件大小(MB)"] = 0.0
    cols = ["编号", "数据集名称", "文件大小(MB)", "图片文件数", "标注文件数", "来源类型", "导入时间", "来源链接", "本地文件"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df.fillna("")
    return df[cols].sort_values("导入时间", ascending=False).reset_index(drop=True)


def export_dataset_bundle(base_dir: Path, selected_ids: List[str], bundle_name: str = "目标检测数据包") -> Tuple[bool, str, Path | None]:
    ids = [str(x).strip() for x in selected_ids if str(x).strip()]
    if not ids:
        return False, "请先选择要导出的数据集。", None

    registry = _load_registry(base_dir)
    picked = [r for r in registry if str(r.get("编号", "")) in ids]
    if not picked:
        return False, "未找到可导出的数据集记录。", None

    _ensure_dirs(base_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = _normalize_name(bundle_name) or "目标检测数据包"
    out_path = _export_dir(base_dir) / f"{out_name}_{ts}.zip"

    try:
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rec in picked:
                fpath = Path(str(rec.get("本地文件", "")))
                if fpath.exists():
                    zf.write(fpath, arcname=f"datasets/{fpath.name}")
            zf.writestr("manifest.json", json.dumps(picked, ensure_ascii=False, indent=2).encode("utf-8"))
    except Exception as exc:
        return False, f"导出失败：{exc}", None

    return True, "导出成功。", out_path


def import_dataset_from_kaggle(
    base_dir: Path,
    dataset_slug: str,
    alias: str = "",
    timeout_s: int = 1800,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Tuple[bool, str, Dict[str, object] | None]:
    _ensure_dirs(base_dir)
    slug = str(dataset_slug or "").strip().strip("/")
    if "/" not in slug:
        return False, "Kaggle 标识格式应为 username/dataset-name。", None

    source_link = f"https://www.kaggle.com/datasets/{slug}"
    registry = _load_registry(base_dir)
    dup = _dedupe_by_source(registry, source_link=source_link, source_type="Kaggle", source_id=slug)
    if dup is not None:
        return False, "该 Kaggle 数据集已导入过，已自动去重。", dup

    work_tmp = _tmp_dir(base_dir) / f"kaggle_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    work_tmp.mkdir(parents=True, exist_ok=True)
    before_files = {x.name for x in work_tmp.glob("*") if x.is_file()}

    _safe_progress(progress_callback, 5, "正在调用 Kaggle 下载")
    user_kaggle_exe = (
        Path.home()
        / "AppData"
        / "Roaming"
        / "Python"
        / f"Python{sys.version_info.major}{sys.version_info.minor}"
        / "Scripts"
        / "kaggle.exe"
    )
    cmd_candidates = [
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(work_tmp), "--force"],
        [str(user_kaggle_exe), "datasets", "download", "-d", slug, "-p", str(work_tmp), "--force"],
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(work_tmp), "--force"],
    ]
    run_env = dict(os.environ)
    try:
        user_site = site.getusersitepackages()
        if user_site:
            prev = str(run_env.get("PYTHONPATH", "")).strip()
            run_env["PYTHONPATH"] = f"{user_site}{os.pathsep}{prev}" if prev else user_site
    except Exception:
        pass

    proc = None
    last_err = ""
    for cmd in cmd_candidates:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(work_tmp),
                capture_output=True,
                text=True,
                timeout=max(90, int(timeout_s)),
                shell=False,
                env=run_env,
            )
            if proc.returncode == 0:
                break
            last_err = (proc.stderr or proc.stdout or "").strip()
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            shutil.rmtree(work_tmp, ignore_errors=True)
            return False, "Kaggle 下载超时，请稍后重试。", None
        except Exception as exc:
            last_err = str(exc)
            continue

    if proc is None:
        shutil.rmtree(work_tmp, ignore_errors=True)
        return False, "未检测到 kaggle 命令，请先安装并配置 Kaggle API。", None

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or last_err or "").strip()
        shutil.rmtree(work_tmp, ignore_errors=True)
        if not err:
            err = "未知错误"
        if "No module named kaggle" in err:
            err = "当前解释器缺少 kaggle 依赖，请在运行本项目的同一环境执行：python -m pip install kaggle"
        return False, f"Kaggle 下载失败：{err}", None

    _safe_progress(progress_callback, 70, "下载完成，正在入库")
    after_files = [x for x in work_tmp.glob("*") if x.is_file() and x.name not in before_files]
    if not after_files:
        after_files = [x for x in work_tmp.glob("*") if x.is_file()]
    if not after_files:
        shutil.rmtree(work_tmp, ignore_errors=True)
        return False, "下载完成但未找到可导入文件。", None

    downloaded = max(after_files, key=lambda p: p.stat().st_size if p.exists() else 0)
    suffix = downloaded.suffix.lower()
    suggested_name = _normalize_name(alias or slug.replace("/", "_"))
    if suffix and not suggested_name.lower().endswith(suffix):
        suggested_name = f"{suggested_name}{suffix}"
    if "." not in suggested_name:
        suggested_name = f"{suggested_name}.zip"
    renamed = downloaded.with_name(suggested_name)
    try:
        downloaded.rename(renamed)
        downloaded = renamed
    except Exception:
        pass

    digest = _sha256_file(downloaded)
    dup_hash = _dedupe_by_hash(registry, digest)
    if dup_hash is not None:
        shutil.rmtree(work_tmp, ignore_errors=True)
        return False, "该文件与已导入数据重复，已自动去重。", dup_hash

    final_path = _raw_dir(base_dir) / downloaded.name
    if final_path.exists():
        stem = _normalize_name(final_path.stem)
        final_path = _raw_dir(base_dir) / f"{stem}_{datetime.now().strftime('%H%M%S')}{final_path.suffix}"
    downloaded.replace(final_path)
    shutil.rmtree(work_tmp, ignore_errors=True)

    img_count, ann_count = _count_zip_contents(final_path) if final_path.suffix.lower() == ".zip" else (0, 0)
    rec = {
        "编号": digest[:12],
        "数据集名称": alias or slug,
        "来源链接": source_link,
        "来源类型": "Kaggle",
        "来源标识": slug,
        "本地文件": str(final_path),
        "文件大小字节": int(final_path.stat().st_size if final_path.exists() else 0),
        "sha256": digest,
        "图片文件数": int(img_count),
        "标注文件数": int(ann_count),
        "导入时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    registry.append(rec)
    _save_registry(base_dir, registry)
    _safe_progress(progress_callback, 100, "Kaggle 导入完成")
    return True, "Kaggle 数据集导入成功。", rec


def scan_raw_datasets(base_dir: Path) -> Tuple[int, int, List[str]]:
    _ensure_dirs(base_dir)
    raw_dir = _raw_dir(base_dir)
    registry = _load_registry(base_dir)

    added = 0
    skipped = 0
    notices: List[str] = []

    for file_path in sorted(raw_dir.glob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix not in (".zip", ".tar", ".gz", ".tgz", ".rar", ".7z", ".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue

        try:
            digest = _sha256_file(file_path)
        except Exception:
            skipped += 1
            notices.append(f"{file_path.name} 哈希计算失败")
            continue

        if _dedupe_by_hash(registry, digest) is not None:
            skipped += 1
            continue

        img_count, ann_count = _count_zip_contents(file_path) if suffix == ".zip" else (0, 0)
        rec = {
            "编号": digest[:12],
            "数据集名称": file_path.stem,
            "来源链接": f"local://{file_path.name}",
            "来源类型": "本地扫描",
            "来源标识": file_path.name,
            "本地文件": str(file_path),
            "文件大小字节": int(file_path.stat().st_size),
            "sha256": digest,
            "图片文件数": int(img_count),
            "标注文件数": int(ann_count),
            "导入时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        registry.append(rec)
        added += 1

    if added > 0:
        _save_registry(base_dir, registry)
    return added, skipped, notices


def sync_dataset_samples_to_vision(
    base_dir: Path,
    sample_dir: Path,
    selected_ids: List[str] | None = None,
    max_images: int = 24,
) -> Tuple[bool, str, List[Path]]:
    _ensure_dirs(base_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    registry = _load_registry(base_dir)
    if not registry:
        return False, "当前没有可同步的数据集记录。", []

    picked: List[Dict[str, object]]
    ids = [str(x).strip() for x in (selected_ids or []) if str(x).strip()]
    if ids:
        picked = [r for r in registry if str(r.get("编号", "")) in ids]
    else:
        picked = sorted(registry, key=lambda r: str(r.get("导入时间", "")), reverse=True)[:3]
    if not picked:
        return False, "未找到可同步的数据集。", []

    max_images = max(1, int(max_images))
    sample_hash_index = _load_sample_index(base_dir)
    sample_hash_index = {k: v for k, v in sample_hash_index.items() if (sample_dir / v).exists()}

    existing_seq = len([p for p in sample_dir.glob("dataset_sample_*.jpg") if p.is_file()])
    seq = existing_seq + 1
    added_paths: List[Path] = []
    dedupe_skip = 0

    for row in picked:
        if len(added_paths) >= max_images:
            break
        local_path = Path(str(row.get("本地文件", "")))
        if not local_path.exists():
            continue

        rid = str(row.get("编号", "unknown"))
        handled_as_zip = False
        if local_path.suffix.lower() == ".zip":
            try:
                handled_as_zip = True
                with zipfile.ZipFile(local_path, "r") as zf:
                    img_names = [n for n in zf.namelist() if str(n).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
                    for img_name in img_names:
                        if len(added_paths) >= max_images:
                            break
                        try:
                            raw = zf.read(img_name)
                        except Exception:
                            continue
                        digest = _sha256_bytes(raw)
                        if digest in sample_hash_index:
                            dedupe_skip += 1
                            continue
                        try:
                            img = Image.open(io.BytesIO(raw)).convert("RGB")
                        except Exception:
                            continue
                        out_name = f"dataset_sample_{rid}_{seq:03d}.jpg"
                        out_path = sample_dir / out_name
                        try:
                            img.save(out_path, format="JPEG", quality=92)
                        except Exception:
                            continue
                        sample_hash_index[digest] = out_name
                        added_paths.append(out_path)
                        seq += 1
            except Exception:
                handled_as_zip = False

        if not handled_as_zip:
            try:
                raw = local_path.read_bytes()
                digest = _sha256_bytes(raw)
                if digest in sample_hash_index:
                    dedupe_skip += 1
                    continue
                img = Image.open(local_path).convert("RGB")
                out_name = f"dataset_sample_{rid}_{seq:03d}.jpg"
                out_path = sample_dir / out_name
                img.save(out_path, format="JPEG", quality=92)
                sample_hash_index[digest] = out_name
                added_paths.append(out_path)
                seq += 1
            except Exception:
                continue

    _save_sample_index(base_dir, sample_hash_index)
    if not added_paths:
        return False, f"未新增样本（去重跳过 {dedupe_skip} 张）。", []
    return True, f"样本同步完成，新增 {len(added_paths)} 张（去重跳过 {dedupe_skip} 张）。", added_paths
