# 城市交通智能中枢（交通治理 AI 项目）

本项目面向交通治理场景，支持可视化展示、模型对比、目标检测、策略推演与问答分析。

## 主要功能
- 总览驾驶舱：拥堵趋势、热力分析、风险走廊、在线热点摘要。
- 模型实验室：基础模型与高级算法评估对比。
- 深度预测：短时交通指数预测与异常波动识别。
- 视觉感知：YOLO 多版本目标检测与结果对比展示。
- 数据集管理：本地导入、URL 导入、Kaggle 导入、去重与导出。
- 应急站选址：复杂网络视角下的候选站点评估。
- 智能问答：围绕交通治理数据的解释型问答。

## 快速启动
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## GPU 启动（推荐）
```powershell
conda run -n pytorch python -m streamlit run app.py
```

## 目录结构
```text
app.py
src/
scripts/
data/city_data/
```

## 说明
- 代码可直接交接给队友继续开发。
- 大体积训练数据建议按需下载，不建议直接纳入仓库版本管理。
