from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


STATUS_ORDER = ["待开始", "进行中", "已完成", "风险阻塞"]


def _default_rows() -> List[Dict[str, object]]:
    return [
        {
            "模块编号": "M1",
            "模块名称": "数据底座与治理",
            "状态": "已完成",
            "完成度": 95,
            "优先级": 1,
            "本周目标": "保证数据导入、去重、导出、台账可复现",
            "下一步": "补充更多真实路段样本并校验标签质量",
            "负责人": "你",
        },
        {
            "模块编号": "M2",
            "模块名称": "基线模型实验室",
            "状态": "已完成",
            "完成度": 90,
            "优先级": 2,
            "本周目标": "完成随机森林/梯度提升/感知机基线训练与对比",
            "下一步": "补充交叉验证统计并固化实验报告",
            "负责人": "你",
        },
        {
            "模块编号": "M3",
            "模块名称": "深度预测与异常检测",
            "状态": "进行中",
            "完成度": 75,
            "优先级": 3,
            "本周目标": "完善时序预测与异常识别解释链路",
            "下一步": "增加未来24-48小时多场景预测对比",
            "负责人": "你",
        },
        {
            "模块编号": "M4",
            "模块名称": "策略推演与问答联动",
            "状态": "进行中",
            "完成度": 72,
            "优先级": 4,
            "本周目标": "形成参数调节-结果变化-策略建议闭环",
            "下一步": "沉淀可复用问答模板与案例库",
            "负责人": "你",
        },
        {
            "模块编号": "M5",
            "模块名称": "应急站选址优化",
            "状态": "进行中",
            "完成度": 68,
            "优先级": 5,
            "本周目标": "完成网络选址建模与基线方案对比",
            "下一步": "优化地图表达与站点解释文本",
            "负责人": "你",
        },
        {
            "模块编号": "M6",
            "模块名称": "本地训练闭环",
            "状态": "进行中",
            "完成度": 60,
            "优先级": 6,
            "本周目标": "跑通数据检查、训练命令、训练记录回看",
            "下一步": "补充训练参数模板与GPU训练稳定性验证",
            "负责人": "你",
        },
        {
            "模块编号": "M7",
            "模块名称": "视觉检测专项（暂缓）",
            "状态": "待开始",
            "完成度": 20,
            "优先级": 7,
            "本周目标": "保留接口与数据准备，不作为当前主线",
            "下一步": "后续单独开展模型调优与误检治理",
            "负责人": "你",
        },
        {
            "模块编号": "M8",
            "模块名称": "答辩演示与交付",
            "状态": "待开始",
            "完成度": 30,
            "优先级": 8,
            "本周目标": "整理演示脚本、录屏流程、项目交接材料",
            "下一步": "生成最终提交包并进行彩排",
            "负责人": "你",
        },
    ]


def _board_path(base_dir: Path) -> Path:
    path = base_dir / "data" / "city_data" / "project_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_progress_board(base_dir: Path) -> pd.DataFrame:
    path = _board_path(base_dir)
    if not path.exists():
        rows = _default_rows()
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return pd.DataFrame(rows)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("invalid progress file")
        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame(_default_rows())
    except Exception:
        df = pd.DataFrame(_default_rows())
    return _normalize_board(df)


def _normalize_board(df: pd.DataFrame) -> pd.DataFrame:
    required = ["模块编号", "模块名称", "状态", "完成度", "优先级", "本周目标", "下一步", "负责人"]
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = ""

    out["状态"] = out["状态"].astype(str).where(out["状态"].astype(str).isin(STATUS_ORDER), "待开始")
    out["完成度"] = pd.to_numeric(out["完成度"], errors="coerce").fillna(0).clip(0, 100).round(0).astype(int)
    out["优先级"] = pd.to_numeric(out["优先级"], errors="coerce").fillna(99).astype(int)
    out = out[required].sort_values(["优先级", "模块编号"]).reset_index(drop=True)
    return out


def save_progress_board(base_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_board(df)
    path = _board_path(base_dir)
    path.write_text(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_progress_kpis(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"模块数": 0, "已完成": 0, "进行中": 0, "风险阻塞": 0, "总体完成度": 0.0}

    total = int(len(df))
    done = int((df["状态"] == "已完成").sum())
    doing = int((df["状态"] == "进行中").sum())
    blocked = int((df["状态"] == "风险阻塞").sum())
    overall = float(pd.to_numeric(df["完成度"], errors="coerce").fillna(0).mean())
    return {"模块数": total, "已完成": done, "进行中": doing, "风险阻塞": blocked, "总体完成度": overall}
