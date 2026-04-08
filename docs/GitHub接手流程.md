# GitHub 接手流程

## 1. 本地仓库初始化（已完成）

```powershell
git init
```

## 2. 创建远程仓库并关联

在 GitHub 创建空仓库后，执行：

```powershell
git remote add origin <你的仓库地址>
```

## 3. 首次提交

```powershell
git add .
git commit -m "初始化：交通AI系统可交接版本"
git branch -M main
git push -u origin main
```

## 4. 团队协作规范（建议）

- 每人一个分支：`feature/xxx` 或 `fix/xxx`
- 提交粒度小：一个功能一个提交
- 合并前本地先跑：

```powershell
python -m compileall app.py src
```

## 5. 大文件处理建议

- 模型权重和原始数据不要直接进仓库
- 若必须共享大文件，使用：
  - Git LFS
  - 网盘 + 下载脚本
  - 校内共享盘

