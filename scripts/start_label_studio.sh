#!/usr/bin/env bash
# Start Label Studio (own conda env "labelstudio", Python 3.12) and open it in the browser: http://localhost:8080
# Local files are enabled for the project's recordings, so frames can be added without uploading:
#   Project -> Settings -> Cloud Storage -> Add Source Storage -> Local files ->
#   absolute path e.g. /home/elton/Desktop/camera/Camera1/Camera_robot/detection/data/label_studio/rec_.../frames
# Export for training: Export -> "YOLO" (with images) -> python3 detection/import_label_studio.py <the zip>
# Data (projects, labels, users) is kept in ~/.local/share/label-studio. Close the window / Ctrl+C to stop.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$ROOT/detection/data"
unset PYTHONPATH PYTHONHOME
LS="$HOME/miniconda3/envs/labelstudio/bin/label-studio"
[ -x "$LS" ] || { echo "Label Studio not found: conda create -n labelstudio python=3.12 && conda activate labelstudio && pip install label-studio 'sentry-sdk<3'"; exit 1; }
echo "Label Studio -> http://localhost:8080  (first start takes ~1 min; Ctrl+C or close this window to stop)"
exec "$LS" start --port 8080
