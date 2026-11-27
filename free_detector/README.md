# free_detector: license-free egg detection

The same job as the YOLO pipeline (egg/ball detection + egg X, Y, Z in `base_link`), built **only from parts
with permissive licenses**. It can go into a commercial robot without paying for a license and without
publishing the robot's code.

| Part | License | Where |
|---|---|---|
| D-FINE model code (Hugging Face `transformers`) | Apache-2.0 | only for training |
| Start weights `ustc-community/dfine-small-coco` (COCO pretrained) | Apache-2.0 | only for training |
| PyTorch / torchvision | BSD-3 | only for training |
| onnxruntime | MIT | robot |
| OpenCV, numpy | Apache-2.0 / BSD | robot |
| ROS 2 Humble, `realsense2_camera`, `cv_bridge`, `message_filters`, `tf2` | Apache-2.0 / BSD | robot |
| `egg_detect_node.py`, `train_dfine.py`, `export_onnx.py` | ours (written for this project) | robot / training |
| Images and labels (`detection/data/...`) | ours | training |

Not used here: `ultralytics` (AGPL-3.0), YOLO weights (AGPL-3.0), `yolo_ros` / `yolo_msgs` (GPL-3.0).
Isaac Sim (NVIDIA license) only generates training images and does not run on the robot.
*This table is a technical summary, not legal advice. Check the licenses again before a product release.*

## Results (27.11.2026)

| | D-FINE-S (this folder) | YOLO `egg_ball_ls` |
|---|---|---|
| Real D455 validation: eggs found | 160 / 160 | 160 / 160 |
| Real ball-only frames called "egg" | 0 / 40 | 0 / 40 |
| Isaac Sim validation: eggs found / ball → egg | 248 / 264, 4 / 28 | 252 / 264, 2 / 28 |
| CPU per image | ~100 ms | ~42 ms |
| Live: camera → Isaac Sim twin (median) | 155 ms | 156 ms |

Training: 12 epochs (~2 min each, RTX 5070), best = epoch 9 (`models/dfine_s_egg_ball/log.txt`).

## Files

| File | What it does |
|---|---|
| `train_dfine.py` | trains D-FINE-S on our egg/ball images (GPU, Isaac Sim's python) -> `models/<name>/` |
| `export_onnx.py` | trained model -> `model.onnx` + `labels.json` (the robot needs only onnxruntime) |
| `egg_detect_node.py` | ROS 2 node: colour + aligned depth -> boxes -> egg X, Y, Z; publishes `egg/point`, `egg/points`, `free/dbg_image`, `free/markers` |
| `eval_compare.py` | compares the ONNX model with the YOLO model on held-out real, Label Studio and Isaac Sim frames |
| `models/` | trained models (not in git, too big) |

## Use

```bash
cd ~/Desktop/camera/Camera1/Camera_robot
DETECTOR=free scripts/start_all.sh real      # real D455 + RViz
DETECTOR=free scripts/start_all.sh           # real D455 + Isaac Sim twin
DETECTOR=free scripts/start_all.sh sim       # Isaac Sim only
```

The node publishes the same `egg/point` / `egg/points` topics as `ros2/egg_xyz.py`, so RViz and the Isaac Sim
twin work unchanged. The image window shows `/free/dbg_image`.

## Train again (e.g. after new Label Studio labels)

```bash
cd ~/Desktop/camera/Camera1/Camera_robot
python3 detection/import_label_studio.py <export.zip>              # new labels (see the main README)
# rebuild detection/data/egg_ball_ls2_train.txt / _val.txt if needed, then:
HF_HUB_OFFLINE=1 ~/isaacsim/python.sh free_detector/train_dfine.py --epochs 30      # ~50 min, RTX 5070
~/isaacsim/python.sh free_detector/export_onnx.py
python3 free_detector/eval_compare.py
```

## Setup on a new computer

```bash
# robot / ROS python (system python 3.10): onnxruntime in its own folder, the system stays untouched
/usr/bin/python3 -m pip install --target ~/egg_free_pkgs --no-deps onnxruntime==1.20.1
# training (Isaac Sim's python already has CUDA torch + transformers >= 4.57), start weights once (~40 MB):
~/isaacsim/python.sh -c "from huggingface_hub import snapshot_download; snapshot_download('ustc-community/dfine-small-coco')"
```

## How the 3D position is computed

1. D-FINE gives boxes (no NMS needed; overlapping duplicates are dropped).
2. Anti-blink: a new object needs score >= 0.5, an object already seen stays while its score >= 0.3.
3. Depth in the middle half of the egg box; the nearest 40 % of those pixels = the egg's front surface.
4. Centre = surface + 22 mm (egg radius) along the viewing ray, using the colour camera's intrinsics.
5. Camera optical frame -> `base_link` with TF. Accepted only 20 cm - 1 m from `camera_link`, real size 2-10 cm.
