# Meshari robot — Intel RealSense D455 egg detection in 3D (real camera + Isaac Sim)

The meshari two-arm robot sees an egg on the table with an Intel RealSense **D455** in its torso, finds it in the
colour image with a **YOLO model trained on real and Isaac Sim images**, reads the **depth aligned to the colour
image** inside the box and outputs the egg's **X, Y, Z in the robot frame (`base_link`)** and its distance from
the camera. The same pipeline runs on the **real camera** and on a **simulated D455 in Isaac Sim 5.1**.

```
 D455 (real)  ── realsense2_camera ──►  /camera/camera/color/image_raw                 (640×480 rgb8, 30 Hz)
                                        /camera/camera/aligned_depth_to_color/image_raw (16UC1 mm, same pixels)
 D455 (Isaac Sim, meshari_d455.usd) ──► /sim/camera/camera/... (same topics, 32FC1 metres)
                                                │
                     yolo_ros 4.7.1 (ultralytics, use_3d)   ros2/egg_yolo.launch.py
       yolo_node ──► tracking_node (ByteTrack) ──► detect_3d_node ──► debug_node
       (egg box)        (stable id)           (depth in the box → 3D, TF → base_link)   (image + RViz markers)
                                                │  /yolo/detections_3d
                                       ros2/egg_xyz.py  ─► prints X, Y, Z + distance, publishes /egg/point
```

Measured results (details in [Results](#results)): the egg is found in **96–100 %** of real frames the model never
saw in training (score ≈ 0.95), 98 % of eggs in held-out Isaac Sim images, at **~18 Hz on the CPU**; the egg's
height comes out within **~1 cm** of the table geometry.

---

## Contents
1. [Files](#files)
2. [Coordinate frames and camera mount](#coordinate-frames-and-camera-mount)
3. [Installation (once)](#installation-once)
4. [Running — real camera](#running--real-camera)
5. [Running — Isaac Sim](#running--isaac-sim)
6. [Topics and messages](#topics-and-messages)
7. [How each part works](#how-each-part-works)
8. [The egg model: data and training](#the-egg-model-data-and-training)
9. [Measuring the camera tilt](#measuring-the-camera-tilt)
10. [Results](#results)
11. [Troubleshooting](#troubleshooting)
12. [Known limits and next steps](#known-limits-and-next-steps)

---

## Files

```
Camera_robot/
├── env.sh                         per-terminal ROS 2 environment (leaves conda, sources ROS + ~/ros2_ws, DDS profile)
├── shared/d455_model.py           camera mount on the robot: MOUNT_XYZ / MOUNT_RPY (one place)
├── ros2/
│   ├── d455.launch.py             real D455 driver (colour + aligned depth) + TFs + RViz
│   ├── egg_yolo.launch.py         yolo_ros 3D egg detection, source:=real | sim
│   └── egg_xyz.py                 prints the eggs' X, Y, Z + distance, 20 cm–1 m range + size check, /egg/point
├── isaac/
│   ├── isaac_ros2.sh              starts Isaac Sim / its python with the internal ROS 2 bridge
│   ├── meshari_d455.py            builds "Isaac SIm/meshari_d455.usd": D455 in the torso slot + ROS 2 graph
│   ├── add_test_egg.py            test table (20 cm below / 20 cm ahead of the camera) + real-size brown egg
│   ├── drop_egg.py                same table + egg, dropped with real physics (60 g, friction, bounce)
│   └── egg_twin.py                digital twin: the sim egg follows the egg the REAL camera sees (/egg/point)
├── detection/
│   ├── record_egg_yolo.py         guided recording of real colour + depth frames (18 placements)
│   ├── label_egg_yolo.py          automatic egg boxes for those recordings → YOLO dataset
│   ├── gen_egg_rgb.py             synthetic egg images from Isaac Sim (real-size eggs, real camera calibration)
│   ├── train_egg_yolo.py          YOLOv8s training on the GPU (Isaac Sim's CUDA torch)
│   ├── yolo/egg_real_sim_yolov8s.pt   ★ the model in use (real + sim)
│   ├── yolo/egg_yolov8s.pt            real-only model (start point for the real + sim training)
│   ├── yolo/yolov8s-worldv2-egg.pt    zero-shot YOLO-World "egg" (used by the auto-labeller)
│   └── data/                      recordings + datasets (git-ignored)
├── config/
│   ├── d455_color_intrinsics.json real colour calibration (640×480, fx 385.7) — used by the sim colour camera
│   ├── d455_intrinsics.json       real depth calibration (424×240, fx 217.2) — used by the sim depth camera
│   ├── sim_color_extrinsics.json  sim camera_link → colour camera (written by meshari_d455.py)
│   ├── egg_bytetrack.yaml         tracker settings for the egg
│   ├── d455.rviz                  RViz layout (point clouds, egg boxes, egg points, TF)
│   └── fastdds_shm.xml            Fast DDS profile shared by ROS terminals and Isaac Sim
├── scripts/install_ubuntu.sh      one-time install (ROS 2, RealSense, yolo_ros, Python packages)
├── scripts/start_all.sh           one command for everything (real | twin | sim), Ctrl+C stops all
├── scripts/record_demo_video.py   demo video: detection image + egg X, Y, Z
├── Isaac SIm/meshari 1.usd        the robot (never modified);  meshari_d455.usd = robot + D455 (generated)
└── Documentation/                 task description and earlier reports (PDF)
```

Models (`*.pt`), recordings (`detection/data/`) and the `.usd` scenes are **git-ignored** (too large for GitHub):
back them up separately. `meshari_d455.usd` can always be rebuilt with `isaac/meshari_d455.py`.

---

## Coordinate frames and camera mount

| Frame | Meaning |
|---|---|
| `base_link` | meshari robot base, origin on the floor. **Forward = −Y, left = +X, up = +Z** |
| `camera_link` | D455 depth origin (left imager), as in realsense-ros. x forward, y left, z up |
| `camera_depth_optical_frame`, `camera_color_optical_frame` | image frames: x right, y down, z forward. The colour camera is 5.9 cm to the right of `camera_link` |

`shared/d455_model.py`:
```python
MOUNT_XYZ = (0.1781, -0.043, 1.153)          # torso slot centre (CAD) + 47.5 mm to the left imager
MOUNT_RPY = (0.0, 0.2015855, -1.5707963)     # roll 0 (straight in the slot), pitch 11.55° down, yaw −90°
```
`ros2/d455.launch.py` publishes it as the static TF `base_link → camera_link`. The Isaac Sim build
(`meshari_d455.py --pitch-deg / --roll-deg`) and the image generator (`gen_egg_rgb.py CAM_RPY`) use the same
values as defaults — **keep all three equal**. Any value can be overridden when launching, e.g. on a test rig:
`ros2 launch ros2/d455.launch.py mount_pitch:=0.1169 mount_roll:=-0.0105` (radians).

The pitch is **measured**, not guessed — see [Measuring the camera tilt](#measuring-the-camera-tilt).
With the table 20 cm below and its near edge 20 cm in front of the camera, a tilt of ≥ 10° shows the whole table:
an egg is visible from the table edge (≈ 27 cm from the camera, the closest possible on that table) to beyond 1 m.

---

## Installation (once)

Ubuntu 22.04, ROS 2 Humble, Isaac Sim 5.1 in `~/isaacsim`, an NVIDIA GPU for Isaac Sim and training.
```bash
bash scripts/install_ubuntu.sh
```
It installs: ROS 2 Humble + `realsense2_camera`, the RealSense udev rule, CPU torch + `ultralytics==8.4.6` for
ROS's Python 3.10, and builds **yolo_ros tag 4.7.1** in `~/ros2_ws` (the newest yolo_ros is C++/ONNX without
`use_3d`; 4.7.1 is the last Python/ultralytics release). `env.sh` sources `~/ros2_ws/install/setup.bash`.

For **training on the GPU** (only needed to retrain), ultralytics goes into a separate folder for Isaac Sim's
Python, so Isaac Sim itself is not changed:
```bash
~/isaacsim/kit/python/bin/python3 -m pip install --target ~/egg_train_pkgs --no-deps "ultralytics==8.4.6" polars py-cpuinfo ultralytics-thop
```
The auto-labeller needs OpenAI CLIP once (to build the YOLO-World "egg" model); pip 22 cannot install it, so copy
the `clip/` folder of https://github.com/ultralytics/CLIP into `~/.local/lib/python3.10/site-packages/`.

Every ROS terminal starts with `cd ~/Desktop/camera/Camera1/Camera_robot && source ./env.sh`.

---

## Start everything with one command

```bash
scripts/start_all.sh            # real D455 + yolo egg detection + egg X, Y, Z + image window + Isaac Sim egg twin
scripts/start_all.sh real       # real camera only (with RViz)
scripts/start_all.sh sim        # Isaac Sim only (its D455 + test egg, yolo on /sim)
```
Or double-click **Meshari Egg Detection** on the Desktop (also in the app menu). The terminal window shows the eggs'
positions; **close it (or Ctrl+C) to stop everything**. Logs: `/tmp/meshari_egg/*.log`. The desk-test-rig values
(`MOUNT_ARGS`, `TWIN_ARGS`) are at the top of the script; on the robot set `MOUNT_ARGS=""` and
`TWIN_ARGS="--twin-real-z"`.

## Running — real camera

Four terminals (each: `source ./env.sh` first):
```bash
ros2 launch ros2/d455.launch.py                         # 1. camera + TFs + RViz
ros2 launch ros2/egg_yolo.launch.py                     # 2. yolo_ros 3D egg detection   -> /yolo/...
python3 ros2/egg_xyz.py                                 # 3. egg X, Y, Z + distance        -> /egg/point
ros2 run rqt_image_view rqt_image_view /yolo/dbg_image  # 4. camera image with the egg box
```
Terminal 3 prints, ~18 times per second:
```
egg #1 score 0.93: X +0.149 Y -0.373 Z +1.060 m in base_link,  34.4 cm from the camera
```
`#1` is the tracker id, `score` the model confidence, X/Y/Z the egg centre in `base_link` (metres), and the
distance is measured from `camera_link`. RViz shows the D455 point cloud, the 3D egg box and the egg point.

## Running — Isaac Sim

```bash
isaac/isaac_ros2.sh isaac/meshari_d455.py --run --test-egg                # Isaac Sim window: robot + D455 + table + egg
ros2 launch ros2/d455.launch.py camera:=false                            # TFs + RViz (no real camera)
ros2 launch ros2/egg_yolo.launch.py source:=sim                          # -> /sim/yolo/...
python3 ros2/egg_xyz.py --ros-args -r __ns:=/sim -p detections:=/sim/yolo/detections_3d
```
Move the egg in Isaac Sim (`/World/TestEgg`, gizmo) and watch the numbers follow. With `--drop-egg` instead of
`--test-egg` the egg falls onto the table with real physics and the detector picks it up once it lands. Or open the GUI with
`isaac/isaac_ros2.sh`, load `Isaac SIm/meshari_d455.usd`, press Play, and run `isaac/add_test_egg.py` from
Window → Script Editor. **Always start Isaac Sim through `isaac/isaac_ros2.sh`** (it sets the ROS 2 bridge up).
Real and sim can run at the same time (different namespaces); then skip `camera:=false` — the real launch already
publishes the TFs.

Rebuild the robot + camera file after changing the mount (never while it is open in Isaac Sim):
```bash
isaac/isaac_ros2.sh isaac/meshari_d455.py --headless          # --pitch-deg 11.55 --roll-deg 0 by default
```

---

## Topics and messages

| Topic (real; sim = `/sim` + same) | Type | Content |
|---|---|---|
| `/camera/camera/color/image_raw` | sensor_msgs/Image rgb8 | colour image 640×480 |
| `/camera/camera/aligned_depth_to_color/image_raw` | sensor_msgs/Image 16UC1 mm (sim: 32FC1 m) | depth, same pixels as colour |
| `/camera/camera/aligned_depth_to_color/camera_info` | sensor_msgs/CameraInfo | intrinsics of that image |
| `/camera/camera/depth/color/points` | sensor_msgs/PointCloud2 | point cloud (RViz) |
| `/yolo/detections` | yolo_msgs/DetectionArray | 2D egg boxes |
| `/yolo/tracking` | yolo_msgs/DetectionArray | + tracker ids |
| `/yolo/detections_3d` | yolo_msgs/DetectionArray | `bbox3d.center` = egg position in `base_link`, `bbox3d.size` |
| `/yolo/dbg_image` | sensor_msgs/Image | colour image with boxes and scores |
| `/yolo/dgb_bb_markers` | visualization_msgs/MarkerArray | 3D egg boxes in RViz |
| `/egg/point` | geometry_msgs/PointStamped | best egg (from `egg_xyz.py`), for the arm controller |

`yolo_msgs/Detection`: `class_name` ("egg"), `score`, `id`, `bbox` (2D), `bbox3d` (`center` Pose, `size`, `frame_id`).

---

## How each part works

### `ros2/d455.launch.py`
Starts `realsense2_camera` (`rs_launch.py`) with colour 640×480@30, depth 424×240@30 (424×240 is the profile
with the shortest minimum distance, ≈ 16 cm), `align_depth.enable:=true` (depth re-projected onto the colour
image), `enable_sync`, temporal filter and point cloud. It publishes the static TF `base_link → camera_link` from
`MOUNT_XYZ/MOUNT_RPY`; the driver publishes `camera_link → optical frames` from the factory calibration.
With `camera:=false` it publishes those optical-frame TFs itself (sim camera geometry) and does not start the driver.
Arguments: `camera`, `rviz`, `base_frame`, `mount_x/y/z/roll/pitch/yaw`.

### `ros2/egg_yolo.launch.py`
Includes yolo_ros' `yolo.launch.py` with: `model` = `detection/yolo/egg_real_sim_yolov8s.pt`, `model_type:=YOLO`,
`use_3d:=True`, `use_tracking:=True`, `target_frame:=base_link`, `threshold` 0.5, `device:=cpu`, the tracker
`config/egg_bytetrack.yaml`, best-effort QoS, and per source:

| `source:=` | colour | aligned depth / info | depth unit divisor | namespace |
|---|---|---|---|---|
| `real` (default) | `/camera/camera/color/image_raw` | `/camera/camera/aligned_depth_to_color/...` | 1000 (mm) | `yolo` |
| `sim` | `/sim/camera/camera/color/image_raw` | `/sim/camera/camera/aligned_depth_to_color/...` | 1 (m) | `sim/yolo` |

yolo_ros nodes: **yolo_node** runs the model on each colour image → **tracking_node** gives each egg a stable id
(ByteTrack) → **detect_3d_node** takes the depth pixels inside the box, back-projects them with the camera
intrinsics, and transforms the 3D box to `base_link` via TF → **debug_node** draws the image and RViz markers.
Options: `threshold:=…`, `device:=cuda:0` (needs a CUDA torch in ROS's Python), `model:=… model_type:=YOLO|World`.

### `ros2/egg_xyz.py`
Subscribes to `detections_3d`. For each egg it looks up `camera_link` in TF and computes the straight-line
distance. It keeps only eggs **20 cm – 1 m** from the camera (`min_distance_m`, `max_distance_m`) whose measured
size is egg-like, **2–10 cm** (`min_size_m`, `max_size_m`; an egg is 44 × 57 mm); this rejects e.g. a hand box.
It logs X, Y, Z + distance and publishes the highest-score egg on `egg/point`.
Parameters: `detections`, `camera_frame`, the four limits above.

### `isaac/meshari_d455.py`
Opens `Isaac SIm/meshari 1.usd` (unchanged) and saves `meshari_d455.usd` with:
* the official NVIDIA D455 asset in the torso slot, child of `base_link` (moves with the robot), mount =
  `--pitch-deg 11.55 --roll-deg 0`, yaw −90°; its physics and IMU are removed (they conflict with the robot body);
* a **depth camera** at the left imager with the real depth calibration (424×240, fx 217.2);
* the asset's **colour camera set to the real colour calibration** (640×480, fx 385.7 — the asset's own camera was
  90° wide and made objects ~18 % too small), near clip 5 cm;
* an OmniGraph (`/World/D455_ROS2`) that on Play publishes depth, point cloud, colour, colour info and
  **depth rendered by the colour camera = aligned depth**, under `/sim/camera/camera/...` with the real frame ids;
* floor + dome light if missing; writes `config/sim_color_extrinsics.json` (camera_link → colour camera).
Options: `--run` (publish after building), `--headless`, `--test-egg`, `--ns`, `--out`, `--pitch-deg`, `--roll-deg`.

### `isaac/add_test_egg.py`
Adds a table (top 20 cm below the camera, near edge 20 cm ahead of it, dark material) and a **57 × 44 mm** brown
egg lying on it (`EGG_X`, `EGG_Y`, `EGG_YAW_DEG`). Run from the Script Editor or via `--test-egg`.

### `isaac/drop_egg.py`
Builds the table + egg with `add_test_egg.py`, then makes them physical: the egg is a rigid body of **60 g** with a
convex-hull collider of the 57 × 44 mm mesh, continuous collision detection, friction 0.6 / 0.5 and restitution 0.15
(an egg barely bounces), angular damping 0.3; the table is a static collider. The egg starts `DROP_HEIGHT` (15 cm)
above the table with a random tilt, push and spin (`RANDOM_POSE`, `SEED` for a repeatable drop). Tested: it lands
in about 0.3 s, rolls a few cm and rests on its side, centre ~2 cm above the table. Use `--run --drop-egg`, or in
the GUI: stop the simulation, run it from the Script Editor, press Play (run again, stopped, for a new drop).

### `isaac/egg_twin.py` (digital twin)
`isaac/isaac_ros2.sh isaac/meshari_d455.py --run --twin`, with the real pipeline running, gives **every egg the real
D455 sees** an egg in Isaac Sim at the same place (`/egg/points` = all accepted eggs from `egg_xyz.py`, `base_link` =
Isaac's world, ~18 Hz). Up to 6 eggs; each sim egg stays paired with the same real egg (nearest position, 10 cm),
smoothed, and is hidden 1 s after its real egg disappears. Height = sim table + the egg's height above the real
table, so a lifted egg rises: `--twin-real-table <z>` sets the real table's height in `base_link` (desk test rig:
lying eggs read Z ≈ 1.050 → `1.028`); `--twin-real-z` uses the real Z as is (camera on the robot, same table).

### `isaac/isaac_ros2.sh`, `env.sh`
Isaac Sim (Python 3.11) must not see the system ROS (Python 3.10) or conda: `isaac_ros2.sh` strips both, sets
`ROS_DISTRO`, `RMW_IMPLEMENTATION`, the bridge libraries and the shared Fast DDS profile. `env.sh` does the
opposite for ROS terminals: leaves conda, sources ROS Humble and `~/ros2_ws`.

---

## The egg model: data and training

The off-the-shelf YOLO classes contain no egg; zero-shot YOLO-World "egg" scored a real egg only 0.02–0.85
depending on pose. So a YOLOv8s was trained on two sources:

**1. Real images** — `detection/record_egg_yolo.py` (camera running) guides you by text + beeps through 18 egg
placements (lying, standing, near/far, next to and on the spool, three without egg), 7 s to move the egg, 4 s
recording, hands out of view. `detection/label_egg_yolo.py` then labels them automatically: the egg does not move
during a placement, so YOLO-World is run on every frame, the egg-sized boxes (checked with the depth: 2–10 cm) that
agree are merged into one box per placement. Check `detection/data/yolo_egg/review_*.jpg`; fix mistakes with
`--egg session:place` (placement shows an egg although recorded as empty) and `--skip session:place`.
Session 0: 17 placements, 680 images (whole placements are held out for validation).

**2. Isaac Sim images** — `detection/gen_egg_rgb.py` renders 3,000 images from the robot's colour camera
(real calibration) of a table at the real position: **real-size eggs (44 × 57 mm, ±5 %)**, lying / standing /
tilted, brown / cream / white, 1–3 per image (10 % none), distractors (discs like the spool, boxes, cans, balls
that are clearly not egg-sized, cones), random table colours, mats, lighting and ±1 cm / ±1° camera jitter.
Labels come from Isaac's exact 2D boxes; each egg's size is checked from pixels + depth (median 5.1 cm).

**Training** (GPU, ~11 min for real only, ~28 min for real + sim):
```bash
PYTHONPATH=~/egg_train_pkgs ~/isaacsim/python.sh detection/train_egg_yolo.py                     # -> egg_yolov8s.pt
PYTHONPATH=~/egg_train_pkgs ~/isaacsim/python.sh detection/train_egg_yolo.py \
    --data detection/data/egg_real_sim.yaml --base detection/yolo/egg_yolov8s.pt --name egg_real_sim --epochs 40
```
Validation always uses **real** held-out placements. Strong augmentation (colour, scale, flips, mosaic, mixup)
because the real set is small.

To improve it: record another session where the robot will work (`record_egg_yolo.py`; new session numbers are
added automatically), label it, and retrain with both yaml train lists.

---

## Measuring the camera tilt

The tilt comes from the camera itself: take a few depth frames, convert the pixels on the table to 3D points with
the colour calibration, find the table plane (RANSAC), and read the camera's pitch and roll from the plane's
normal (the table is assumed level); the plane's distance is the camera height above the table. With the right
values the table comes out flat in `base_link` (checked: left / right / near / far within 1 mm).
Measured so far: desk test rig 11.55° / roll 3.25° (16 cm above the table), after straightening 6.7° / roll −0.6°.
**Re-measure every time the camera is moved**, then set `MOUNT_RPY` (and rebuild the sim) or pass
`mount_pitch:=… mount_roll:=…` to `d455.launch.py`.

---

## Results

| | Real recording (moving egg, 285 frames) | Real, egg at 23 cm | Real, egg at 65 cm | Isaac Sim held-out (400 eggs) |
|---|---|---|---|---|
| YOLO-World "egg" (zero-shot, start) | median score 0.05 | 0.05 | 0.16 | — |
| Real-only model | 95 % | 86 % | 100 % | 80 %, 141 false boxes |
| **Real + sim model (in use)** | **97 %** | **96 %** | **100 %** | **98 %, 10 false boxes** |

(% = frames where the egg is found with score ≥ 0.5.) The misses are almost all frames where a hand holds the egg.
Live: ~18 Hz on the CPU, position jitter < 1 mm for a still egg, egg height within ~1 cm of the table geometry;
in Isaac Sim the egg is placed within 1.3 cm of its true centre.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| egg_xyz prints nothing, yolo logs `"base_link" ... does not exist` | No TF: start `d455.launch.py` (or `camera:=false` for the sim) |
| egg found but height off by a few cm | camera was moved → re-measure the tilt, pass `mount_pitch/mount_roll` |
| Isaac Sim: "Could not import internal rclpy" / no `/sim` topics | start it with `isaac/isaac_ros2.sh`, not from a conda/ROS terminal |
| render-product error flood in Isaac Sim | `meshari_d455.usd` rebuilt while open — close it, rebuild, reopen |
| yolo_ros build: `canonicalize_version() ... strip_trailing_zero` | `pip install --user "packaging>=24.2"`, rebuild |
| `operator torchvision::nms does not exist` | torchvision is a CUDA build: reinstall the `+cpu` one (install script) |
| training crashes at the end in a plot (`UnboundLocalError`) | already off (`plots=False`); copy `runs/<name>/weights/best.pt` if it ever happens |
| only some frames detected, score near the threshold | zero-shot model in use, or a new place / light: record + retrain |

---

## Known limits and next steps

* The 3D point is computed from the egg surface the camera sees, so it can be **1–2 cm too close to the camera**
  along the viewing direction — check against a ruler and correct before grasping.
* Real training data is one brown egg in one room; white eggs and the robot's own table are covered only by the
  sim images — record one session on the robot (and with a white egg) and retrain.
* When the camera is mounted in the robot, measure the tilt there and update `MOUNT_RPY`.
* GPU inference in ROS needs a CUDA torch for Python 3.10 (CPU gives ~18 Hz, enough for grasping).
