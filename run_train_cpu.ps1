$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path

python "$repo\scripts\train_yolo_local.py" `
  --data "data/city_data/vision_datasets/training/visdrone_traffic_yolo/data.yaml" `
  --model "data/city_data/weights/yolo11n.pt" `
  --device "cpu" `
  --epochs 20 `
  --imgsz 640 `
  --batch 4 `
  --workers 2 `
  --name "traffic_yolo_cpu"
