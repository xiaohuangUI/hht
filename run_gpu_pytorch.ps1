Set-Location -LiteralPath $PSScriptRoot
$env:VISION_DEVICE_DEFAULT = "cuda:0"
$env:ULTRALYTICS_CONFIG_DIR = (Join-Path $PSScriptRoot "data\city_data\ultralytics_cfg")
$env:YOLO_CONFIG_DIR = $env:ULTRALYTICS_CONFIG_DIR
conda run -n pytorch python scripts/check_gpu_env.py
conda run -n pytorch python -m streamlit run app.py
