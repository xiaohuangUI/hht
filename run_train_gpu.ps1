$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path

conda run -n pytorch python "$repo\scripts\train_yolo_local.py" `
  --data "data/city_data/vision_datasets/training/visdrone_traffic_yolo/data.yaml" `
  --model "data/city_data/weights/yolo11n.pt" `
  --device "cuda:0" `
  --epochs 80 `
  --imgsz 960 `
  --batch 8 `
  --workers 6 `
  --name "traffic_yolo11n_gpu"
