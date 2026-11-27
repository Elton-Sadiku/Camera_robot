# Meshari robot — Intel RealSense D455 egg detection in 3D (real camera + Isaac Sim)

The meshari two-arm robot sees an egg on the table with an Intel RealSense **D455** in its torso, finds it in the
colour image with a **model trained on real and Isaac Sim images** (egg vs ping-pong ball), reads the **depth aligned to the colour
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
                                       ros2/egg_xyz.py  ─► prints X, Y, Z + distance, publishes /egg/point (best egg)
                                                │            and /egg/points (all eggs)
                                       isaac/egg_twin.py ─► Isaac Sim: one sim egg per real egg (digital twin)
                                                │            and /twin/egg_poses (for the delay test)

 License-free alternative (no ultralytics / yolo_ros):  DETECTOR=free scripts/start_all.sh
   free_detector/egg_detect_node.py: D-FINE (Apache-2.0, ONNX + onnxruntime) + depth → the same /egg/point(s)
```

Start everything with one command (`scripts/start_all.sh`) or the **Meshari Egg Detection (Task2)** desktop icon.

**Two folders** (`~/Desktop/camera/Camera1/`):

| Folder | What it holds | Use it for |
|---|---|---|
| `Task2/` | the latest version, ready to run: code, documents, the models in use (`egg_ball_ls_yolov8s.pt`, `free_detector/models/dfine_s_egg_ball/`), the Isaac Sim scenes | running, demos, handing over |
| `Camera_robot/` | the git repository (branch `egg_detecting`) + training data (`detection/data/`, 809 MB), older models, training runs, photos | development, retraining, commits |

Changes are made and committed in `Camera_robot/` and copied to `Task2/`.

Measured results (details in [Results](#results) and [Tests](#tests-delay-synchronisation-accuracy)): the egg is
found in **160 / 160** held-out real frames and a ping-pong ball is called "egg" in **0 / 40** ball frames; delay from
the camera image to the egg in Isaac Sim **~155 ms** (median), the sim egg sits **< 0.5 cm** from the measured real
egg; the egg's height comes out within **~1 cm** of the table geometry.

**Task status** (integration D455 → ROS 2 → Isaac Sim, validation, documentation, grasping applications): see
[Task status](#task-status).

---

## Contents
1. [Files](#files)
2. [Coordinate frames and camera mount](#coordinate-frames-and-camera-mount)
3. [Installation (once)](#installation-once)
4. [Start everything with one command](#start-everything-with-one-command)
5. [Running — real camera](#running--real-camera)
6. [Running — Isaac Sim](#running--isaac-sim) (test egg, physics drop, digital twin)
7. [Topics and messages](#topics-and-messages)
8. [How each part works](#how-each-part-works)
9. [The egg model: data and training](#the-egg-model-data-and-training) (incl. egg vs ping-pong ball)
10. [New labelled data: photos, video, Label Studio](#new-labelled-data-photos-video-label-studio)
11. [License-free detector (`free_detector/`)](#license-free-detector-free_detector)
12. [Measuring the camera tilt](#measuring-the-camera-tilt)
13. [Results](#results)
14. [Tests: delay, synchronisation, accuracy](#tests-delay-synchronisation-accuracy)
15. [Applications for manipulation and grasping](#applications-for-manipulation-and-grasping)
16. [Task status](#task-status)
17. [Demo video](#demo-video)
18. [Troubleshooting](#troubleshooting)
19. [Known limits and next steps](#known-limits-and-next-steps)

---

## Files

```
Task2/   (= Camera_robot/ in git)
├── env.sh                         per-terminal ROS 2 environment (leaves conda, sources ROS + ~/ros2_ws, DDS profile)
├── shared/d455_model.py           camera mount on the robot: MOUNT_XYZ / MOUNT_RPY (one place)
├── ros2/
│   ├── d455.launch.py             real D455 driver (colour + aligned depth) + TFs + RViz
│   ├── egg_yolo.launch.py         yolo_ros 3D egg detection, source:=real | sim
│   └── egg_xyz.py                 eggs' X, Y, Z + distance, 20 cm–1 m range + size check, /egg/point + /egg/points
├── isaac/
│   ├── isaac_ros2.sh              starts Isaac Sim / its python with the internal ROS 2 bridge
│   ├── meshari_d455.py            builds "Isaac SIm/meshari_d455.usd": D455 in the torso slot + ROS 2 graph
│   ├── add_test_egg.py            test table (20 cm below / 20 cm ahead of the camera) + real-size brown egg
│   ├── drop_egg.py                same table + egg, dropped with real physics (60 g, friction, bounce)
│   └── egg_twin.py                digital twin: one sim egg per egg the REAL camera sees (/egg/points), up to 6;
│                                  publishes /twin/egg_poses (sim eggs + camera time stamp) for the delay test
├── detection/
│   ├── record_egg_yolo.py         guided recording of real colour + depth frames (--set eggs | balls)
│   ├── label_egg_yolo.py          automatic boxes for those recordings (class 0 egg, 1 ball) → YOLO dataset
│   ├── gen_egg_rgb.py             synthetic images from Isaac Sim (real-size eggs, real calibration; --balls)
│   ├── train_egg_yolo.py          YOLOv8s training on the GPU (Isaac Sim's CUDA torch)
│   ├── import_label_studio.py     Label Studio "YOLO with Images" export → dataset (classes by name: egg 0, ball 1)
│   ├── yolo/egg_ball_ls_yolov8s.pt    ★ the model in use: egg + ball, real + sim + Label Studio frames
│   │   only in Camera_robot/ (not needed to run):
│   ├── yolo/egg_cloud_yolo26s.pt      egg-only YOLO26s trained on the Ultralytics platform (102 photos; tested, not used)
│   ├── yolo/egg_real_sim_yolov8s.pt   earlier 1-class model (real + sim)
│   ├── yolo/egg_yolov8s.pt            real-only model (start point for the real + sim training)
│   ├── yolo/yolov8s-worldv2-egg.pt    zero-shot YOLO-World "egg" (used by the auto-labeller)
│   └── data/                      recordings + datasets + egg_ball_ls.yaml and train/val lists (git-ignored)
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
├── scripts/record_raw_video.py    clean D455 video + frames for labelling (no boxes drawn)
├── scripts/start_label_studio.sh  Label Studio (own conda env "labelstudio") with local files from detection/data
├── scripts/measure_sync.py        test: delay camera → detection → Isaac Sim, sim egg vs real egg
├── scripts/measure_accuracy.py    test: position accuracy with a ruler (distances, straightness, height, jitter)
├── scripts/measure_mount.py       camera tilt + height from the table plane (depth, RANSAC) → MOUNT_RPY / launch args
├── record/take_photos.py          D455 photos with the SPACE key (640×480, clean) for labelling
├── free_detector/                 license-free detector: D-FINE (Apache-2.0) training, ONNX export, ROS node
│                                  (see free_detector/README.md)
├── Isaac SIm/meshari 1.usd        the robot (never modified);  meshari_d455.usd = robot + D455 (generated)
└── Documentation/                 Meshari_D455_Egg_Detection.pdf (team report), egg_demo_live.mp4 (demo video),
                                   applications_grasping.md, tests/ (measured test results, JSON),
                                   team_photos/ (screenshots), task description and earlier reports
```

Models (`*.pt`, `free_detector/models/`), recordings (`detection/data/`, `record/*.jpg`), training runs
(`detection/yolo/runs/`), Label Studio exports and the `.usd` scenes are **git-ignored** (too large for GitHub):
back them up separately (`Task2/` holds the ones needed to run). `meshari_d455.usd` can always be rebuilt with `isaac/meshari_d455.py`.

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
**Current state:** the camera is on the desk test rig. `scripts/start_all.sh` passes the earlier measurement
**6.7° down, roll −0.6°** (`mount_pitch:=0.1169 mount_roll:=-0.0105`), but `scripts/measure_mount.py` read
**13.5° / 1.3°, 15.8 cm** on 27.11.2026 (the camera has moved) — update `MOUNT_ARGS` before the next tests. `MOUNT_RPY`
(11.55°) is the earlier rig measurement and the robot/sim default — set both to the value measured on the robot.
With the table 20 cm below and its near edge 20 cm in front of the camera, a tilt of ≥ 10° shows the whole table:
an egg is visible from the table edge (≈ 27 cm from the camera, the closest possible on that table) to beyond 1 m.

---

## Installation (once)

Ubuntu 22.04, ROS 2 Humble, Isaac Sim 5.1 in `~/isaacsim`, an NVIDIA GPU for Isaac Sim and training.
```bash
bash scripts/install_ubuntu.sh
```
It installs: ROS 2 Humble + `realsense2_camera`, the RealSense udev rule, CPU torch + `ultralytics==8.4.220` for
ROS's Python 3.10, and builds **yolo_ros tag 4.7.1** in `~/ros2_ws` (the newest yolo_ros is C++/ONNX without
`use_3d`; 4.7.1 is the last Python/ultralytics release). `env.sh` sources `~/ros2_ws/install/setup.bash`.

For **training on the GPU** (only needed to retrain), ultralytics goes into a separate folder for Isaac Sim's
Python, so Isaac Sim itself is not changed:
```bash
~/isaacsim/kit/python/bin/python3 -m pip install --target ~/egg_train_pkgs --no-deps "ultralytics==8.4.220" polars py-cpuinfo ultralytics-thop
```
The auto-labeller needs OpenAI CLIP once (to build the YOLO-World "egg" model); pip 22 cannot install it, so copy
the `clip/` folder of https://github.com/ultralytics/CLIP into `~/.local/lib/python3.10/site-packages/`.

For the **license-free detector** (`free_detector/`): onnxruntime for ROS's Python in its own folder, and (only to
retrain) the D-FINE start weights once (~40 MB, Apache-2.0); Isaac Sim's Python already has CUDA torch + transformers:
```bash
/usr/bin/python3 -m pip install --target ~/egg_free_pkgs --no-deps onnxruntime==1.20.1
~/isaacsim/python.sh -c "from huggingface_hub import snapshot_download; snapshot_download('ustc-community/dfine-small-coco')"
```
For **Label Studio** (labelling): its own conda env, Python 3.12 (conda base is 3.14, not supported):
```bash
conda create -y -n labelstudio python=3.12 && conda run -n labelstudio pip install label-studio "sentry-sdk<3"
```

Every ROS terminal starts with `cd ~/Desktop/camera/Camera1/Camera_robot && source ./env.sh`.

---

## Start everything with one command

```bash
scripts/start_all.sh            # real D455 + yolo egg detection + egg X, Y, Z + image window + Isaac Sim egg twin
scripts/start_all.sh real       # real camera only (with RViz)
scripts/start_all.sh sim        # Isaac Sim only (its D455 + test egg, yolo on /sim)
DETECTOR=free scripts/start_all.sh            # any mode with the license-free detector (free_detector/)
MODEL=detection/yolo/<other>.pt scripts/start_all.sh real   # another model (DETECTOR=free: a model.onnx)
```
Or double-click **Meshari Egg Detection (Task2)** on the Desktop (starts `Task2/scripts/start_all.sh twin`;
the older **Meshari Egg Detection** icon starts the same from `Camera_robot/`; both also in the app menu; the first
time: right-click → *Allow Launching*). The terminal window shows the eggs' positions; **close it (or Ctrl+C) to stop everything**.
The default mode starts: real D455 (no RViz), yolo, `egg_xyz.py`, the image window and Isaac Sim with the egg twin. Logs: `/tmp/meshari_egg/*.log`. The desk-test-rig values
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
Move the egg in Isaac Sim (`/World/TestEgg`, gizmo) and watch the numbers follow. Options of `meshari_d455.py --run`:

| Option | Scene |
|---|---|
| `--test-egg` | test table + one egg lying still |
| `--drop-egg` | the egg falls onto the table with real physics; the detector picks it up once it lands |
| `--twin` | **digital twin**: one sim egg per egg the REAL camera sees (needs the real pipeline running) |
| `--twin-real-table 1.028` | twin height: z of the real table in `base_link` (desk rig: 1.028 m) |
| `--twin-real-z` | twin uses the real Z as is (camera on the robot, real table = sim table) |
 Or open the GUI with
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
| `/egg/points` | geometry_msgs/PoseArray | all accepted eggs (empty = none), used by the Isaac Sim twin |
| `/twin/egg_poses` | geometry_msgs/PoseArray | where the twin put the sim eggs, stamped with the camera image time |
| `/free/dbg_image`, `/free/markers` | Image, MarkerArray | image / RViz eggs of the license-free detector |

`yolo_msgs/Detection`: `class_name` ("egg"; "ball" with the egg-vs-ball model), `score`, `id`, `bbox` (2D), `bbox3d` (`center` Pose, `size`, `frame_id`).

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
Includes yolo_ros' `yolo.launch.py` with: `model` = `detection/yolo/egg_ball_ls_yolov8s.pt`, `model_type:=YOLO`,
`use_3d:=True`, `use_tracking:=True`, `target_frame:=base_link`, `threshold` 0.2, `device:=cpu`, the tracker
`config/egg_bytetrack.yaml`, best-effort QoS, and per source:

| `source:=` | colour | aligned depth / info | depth unit divisor | namespace |
|---|---|---|---|---|
| `real` (default) | `/camera/camera/color/image_raw` | `/camera/camera/aligned_depth_to_color/...` | 1000 (mm) | `yolo` |
| `sim` | `/sim/camera/camera/color/image_raw` | `/sim/camera/camera/aligned_depth_to_color/...` | 1 (m) | `sim/yolo` |

yolo_ros nodes: **yolo_node** runs the model on each colour image → **tracking_node** gives each egg a stable id
(ByteTrack) → **detect_3d_node** takes the depth pixels inside the box, back-projects them with the camera
intrinsics, and transforms the 3D box to `base_link` via TF → **debug_node** draws the image and RViz markers.
**Anti-blink (hysteresis):** YOLO passes boxes from score 0.2; the tracker starts an object only at ≥ 0.5
(`new_track_thresh`) but keeps an existing one down to 0.2 (`track_low_thresh`), so a score that wobbles around 0.5
no longer makes the box blink.
Options: `threshold:=…`, `device:=cuda:0` (needs a CUDA torch in ROS's Python), `model:=… model_type:=YOLO|World`.

### `ros2/egg_xyz.py`
Subscribes to `detections_3d`. For each egg it looks up `camera_link` in TF and computes the straight-line
distance. It keeps only eggs **20 cm – 1 m** from the camera (`min_distance_m`, `max_distance_m`) whose measured
size is egg-like, **2–10 cm** (`min_size_m`, `max_size_m`; an egg is 44 × 57 mm); this rejects e.g. a hand box.
Only `class_name == "egg"` counts, so a box classed "ball" is never sent to the arm. It logs X, Y, Z + distance,
publishes the highest-score egg on `egg/point` and all accepted eggs on `egg/points` (PoseArray).
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
Options: `--run` (publish after building), `--headless`, `--test-egg`, `--drop-egg`, `--twin`, `--twin-real-table`,
`--twin-real-z`, `--ns`, `--out`, `--pitch-deg`, `--roll-deg`.

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
Isaac's world, ~10–12 Hz with the CPU detector). Up to 6 eggs; each sim egg stays paired with the same real egg (nearest position, 10 cm),
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

### Egg vs ping-pong ball (done)

**Problem found live:** the model in use (one class, "egg") boxed an **orange ping-pong ball on its holder as an
egg, score 0.943**. It never saw egg-sized balls: the real data had none and the sim generator deliberately kept
balls away from egg size. **Fix: a second class, `ball`**, so the model has to decide egg *or* ball.

| Data | What | Count |
|---|---|---|
| Real eggs (session 0) | `record_egg_yolo.py`, labelled class 0 | 680 frames |
| Real balls (session 1) | `record_egg_yolo.py --set balls` (ball on its holder / mat / notebook, near & far, **no egg**); the old model's false "egg" boxes become the class-1 "ball" labels (`label_egg_yolo.py --detector`) | 320 labelled frames, 8 placements (1:5 ball out of view, 1:7 skipped) |
| Isaac Sim eggs | `gen_egg_rgb.py` (as above) | 3,000 images |
| Isaac Sim eggs + balls | `gen_egg_rgb.py --balls`: ping-pong / golf balls 38–45 mm, orange / white / yellow / egg-brown, 40 % on a small holder, labelled class 1 | 3,000 images, 4,017 eggs, 3,798 balls |

Relabel and train (classes `0: egg`, `1: ball`; first the auto-labelled data, then with the Label Studio frames):
```bash
python3 detection/label_egg_yolo.py --egg 0:15,0:16,0:17 --skip 0:13,1:7
PYTHONPATH=~/egg_train_pkgs ~/isaacsim/python.sh detection/train_egg_yolo.py \
    --data detection/data/egg_ball.yaml --base detection/yolo/egg_real_sim_yolov8s.pt --name egg_ball --epochs 35
PYTHONPATH=~/egg_train_pkgs ~/isaacsim/python.sh detection/train_egg_yolo.py \
    --data detection/data/egg_ball_ls.yaml --base <the egg_ball weights> --name egg_ball_ls --epochs 30   # model in use
```
**Model in use: `detection/yolo/egg_ball_ls_yolov8s.pt`**, trained on the lists in `detection/data/egg_ball_ls.yaml`
(the data above + hand-labelled Label Studio frames of a video with egg and ball together, ×3).
Result on held-out real frames: ball-only frames called "egg" **0 / 40** (old 1-class model: 40 / 40), eggs found
**160 / 160**; egg + ball in the same frame 3 / 14 extra boxes (old: 13 / 14).
Remaining weak case: an egg **on the red ball holder** can be classed "ball" (in the labelled video the ball was always
on the holder). Fix with data: record the egg on the holder and the ball off it, label, import, retrain.

---

## New labelled data: photos, video, Label Studio

1. **Record** (camera driver NOT running): `python3 record/take_photos.py` (a photo per SPACE key, saved in
   `record/`) or `python3 scripts/record_raw_video.py --seconds 60` (clean video + every 10th frame as JPG in
   `detection/data/label_studio/rec_<time>/`). Both save exactly the 640×480 colour image the detector sees.
2. **Label**: `scripts/start_label_studio.sh` (or the *Label Studio* desktop icon) → http://localhost:8080 → project
   with *Object Detection with Bounding Boxes*, labels exactly `egg` and `ball` → box **every** egg and **every**
   ball in every image (an unboxed ball teaches "background").
3. **Export** *YOLO with Images* (not "YOLO", not OBB) → zip.
4. **Import + train**: `python3 detection/import_label_studio.py <zip>` (maps classes by name, strips Label Studio's
   file-name prefixes, warns if an image's labels differ from an earlier import), add the images to the train list
   and run `detection/train_egg_yolo.py` (or `free_detector/train_dfine.py`).

Label Studio is Apache-2.0 (free also commercially); it runs in its own conda env `labelstudio` (Python 3.12,
`sentry-sdk<3`), not in conda base (Python 3.14 is not supported).
The Ultralytics platform (platform.ultralytics.com) can also label and train in the cloud; a model trained only
on egg photos there (`egg_cloud_yolo26s.pt`) found the eggs well but called the ping-pong ball "egg" in 40 / 40
frames — any model needs examples of what is **not** an egg.

---

## License-free detector (`free_detector/`)

`ultralytics` and the YOLO weights are **AGPL-3.0**, `yolo_ros` is **GPL-3.0**: free for research and internal use,
but a robot that is sold would need the Ultralytics Enterprise License (or the robot's code published under AGPL).
Exporting the model to ONNX/TensorRT does not change that — the weights carry the license.
`free_detector/` does the same job with permissive parts only (D-FINE-S, Apache-2.0, trained with PyTorch +
Hugging Face transformers on **our** images and labels; on the robot only onnxruntime (MIT), OpenCV, ROS 2):
```bash
DETECTOR=free scripts/start_all.sh real                 # instead of yolo_ros + egg_xyz.py, same /egg/point(s)
HF_HUB_OFFLINE=1 ~/isaacsim/python.sh free_detector/train_dfine.py --epochs 12    # retrain (~100 s per epoch)
~/isaacsim/python.sh free_detector/export_onnx.py && python3 free_detector/eval_compare.py
```
Details, license table and setup: `free_detector/README.md`. *Not legal advice — check before a product release.*

### Libraries and licenses

| Library | Version | Used for | Where | License |
|---|---|---|---|---|
| ROS 2 Humble (rclpy, tf2, cv_bridge, message_filters) | Humble | messages, transforms, images, time sync | robot | Apache-2.0 / BSD |
| librealsense + realsense2_camera (pyrealsense2) | 2.59.2 | D455 driver: colour, aligned depth, point cloud | robot | Apache-2.0 |
| yolo_ros + yolo_msgs | 4.7.1 | YOLO detection, ByteTrack, 3D from depth | robot (YOLO path) | GPL-3.0 |
| ultralytics | 8.4.220 | YOLO inference + training | robot (YOLO path), PC | AGPL-3.0 |
| PyTorch, torchvision | CPU build (ROS), CUDA 2.7 (Isaac Sim) | inference + training | robot (YOLO path), PC | BSD-3 |
| onnxruntime | 1.20.1 | D-FINE inference on the CPU | robot (free path) | MIT |
| D-FINE-S weights `ustc-community/dfine-small-coco` | — | start model of the free detector | PC (training) | Apache-2.0 |
| Hugging Face transformers | 4.57 | D-FINE training + ONNX export | PC | Apache-2.0 |
| OpenCV, numpy | 4.11, 1.26 | images, 3D math, video | robot, PC | Apache-2.0 / BSD |
| Label Studio | 1.23.2 | labelling egg / ball boxes | PC | Apache-2.0 |
| NVIDIA Isaac Sim | 5.1 | simulation, digital twin, synthetic images | PC | NVIDIA license |

---

## Measuring the camera tilt

The tilt comes from the camera itself: take a few depth frames, convert the pixels on the table to 3D points with
the colour calibration, find the table plane (RANSAC), and read the camera's pitch and roll from the plane's
normal (the table is assumed level); the plane's distance is the camera height above the table. With the right
values the table comes out flat in `base_link` (checked: left / right / near / far within 1 mm).
Measured so far: desk test rig 11.55° / roll 3.25° (16 cm above the table), after straightening 6.7° / roll −0.6°.
**Re-measure every time the camera is moved**, then set `MOUNT_RPY` (and rebuild the sim) or pass
`mount_pitch:=… mount_roll:=…` to `d455.launch.py`.
**Script:** clear the table, camera running, then `python3 scripts/measure_mount.py` → pitch, roll, height above
the table, the launch arguments and the `MOUNT_RPY` line (result also in `Documentation/tests/mount_*.json`).
Check after setting the values: a lying egg's Z must stay the same anywhere on the table.
**27.11.2026 15:59:** the desk rig measured **13.5° down, roll 1.3°, 15.8 cm** above the table (table flat to 2 mm)
— the camera has moved since the 6.7° measurement still used in `start_all.sh`: lying eggs farther away read a
higher Z. To do: update `MOUNT_ARGS`, then the ruler test.

---

## Results

| | Real recording (moving egg, 285 frames) | Real, egg at 23 cm | Real, egg at 65 cm | Isaac Sim held-out (400 eggs) |
|---|---|---|---|---|
| YOLO-World "egg" (zero-shot, start) | median score 0.05 | 0.05 | 0.16 | — |
| Real-only model | 95 % | 86 % | 100 % | 80 %, 141 false boxes |
| **Real + sim model (in use)** | **97 %** | **96 %** | **100 %** | **98 %, 10 false boxes** |

(% = frames where the egg is found with score ≥ 0.5.) The misses are almost all frames where a hand holds the egg.
Live: ~11–18 Hz on the CPU, position jitter < 1 mm for a still egg, egg height within ~1 cm of the table geometry;
in Isaac Sim the egg is placed within 1.3 cm of its true centre.

**Egg vs ball** (held-out frames, score ≥ 0.5; Isaac Sim = 150 rendered validation images):

| Model | Real eggs found | Real ball-only frames called "egg" | Sim eggs found | Sim ball-only called "egg" |
|---|---|---|---|---|
| `egg_real_sim` (1 class) | — | 40 / 40 | — | — |
| `egg_cloud_yolo26s` (Ultralytics platform, 102 egg photos) | 143 / 160 | 40 / 40 | 89 / 131 | 10 / 16 |
| **`egg_ball_ls` (in use)** | **160 / 160** | **0 / 40** | **124 / 131** | **1 / 16** |

**License-free D-FINE-S vs YOLO `egg_ball_ls`** (same held-out frames, score ≥ 0.5, `free_detector/eval_compare.py`):

| Frames | D-FINE (free) eggs found | D-FINE ball-only → "egg" | YOLO eggs found | YOLO ball-only → "egg" |
|---|---|---|---|---|
| Real D455 validation | 160 / 160 | 0 / 40 | 160 / 160 | 0 / 40 |
| Label Studio validation | 12 / 13 | 0 / 2 | 11 / 13 | 0 / 2 |
| Isaac Sim validation | 248 / 264 | 4 / 28 | 252 / 264 | 2 / 28 |
| CPU time per image | ~100 ms | | ~42 ms | |

Same accuracy on real frames; the free model is ~2× slower on the CPU but live it reaches the same end-to-end delay
(below), because the camera driver and depth processing dominate.

---

## Tests: delay, synchronisation, accuracy

**Delay and synchronisation** — `scripts/measure_sync.py` (twin running, `scripts/start_all.sh`). Every message
carries the time stamp of the camera image it came from, so receive time − stamp = delay since the picture was taken.
Measured 27.11.2026, desk rig, yolo_ros on the CPU, Isaac Sim twin on the RTX 5070, 45 s
(`Documentation/tests/sync_20261127_155257.json`):

| | median | 95 % | max |
|---|---|---|---|
| camera driver (image → ROS) | 21 ms | 26 ms | 47 ms |
| camera → egg X, Y, Z (detection + 3D) | 139 ms | 184 ms | 258 ms |
| **camera → egg moved in Isaac Sim** | **156 ms** | **227 ms** | 391 ms |
| sim egg vs real egg, X / Y (same image) | 0.0 cm | 0.5 cm | 5.9 cm |
| sim egg vs real egg, Z (spread) | 0.0 cm | 0.2 cm | 7.1 cm |

**License-free detector** (`DETECTOR=free`, 45 s, `sync_20261127_162025.json`): camera → egg X, Y, Z 142 ms,
camera → Isaac Sim **155 ms** (95 % 177 ms), sim egg vs real egg 0.0 cm, ~9 Hz. (It only keeps the newest camera
image: with a queue of 5 old images piled up and the delay was 296 ms.)

Rates: camera 30 fps, detection and twin updates ~12 Hz (the CPU detector is the bottleneck). The twin adds
~16 ms (one Isaac frame + ROS); its smoothing (`SMOOTH` 0.4) makes the sim egg trail a **moving** egg by a few cm
for ~0.2 s — the maxima above are such moments. Faster: GPU inference (CUDA torch / onnxruntime-gpu).

**Position accuracy with a ruler** — `scripts/measure_accuracy.py`: the egg is placed at ruler marks on the table
(0, 10, 20, … cm); the measured distances between the marks are compared with the ruler.
**Result: pending** (planned 28.11.2026, after updating the tilt with `measure_mount.py`). Run:
`python3 scripts/measure_accuracy.py` → type each ruler mark (0, 10, 20, 30, 40 cm) as the egg is placed there.

---

## Applications for manipulation and grasping

Full text: [`Documentation/applications_grasping.md`](Documentation/applications_grasping.md). In short:
1. **Pick-and-place of eggs** — `/egg/point` in `base_link` → arm IK (MoveIt 2) → soft gripper.
2. **Grasp rehearsal in the digital twin** — run the planned grasp on the simulated meshari with the real egg
   positions first; send only collision-free, reachable plans to the real arms.
3. **Grasp policies in Isaac Lab (sim-to-real)** — the sim camera has the real calibration and mount; physics eggs
   (`drop_egg.py`) give random training scenes; the twin gives real egg positions at run time.
4. **Sorting / selective picking** — egg vs ball today; more classes (cracked, size, colour) with more labels.
5. **Two arms, several eggs** — `/egg/points` + stable ids: each arm takes the eggs on its side.
6. **Grasp verification** — the measured egg height rises with the gripper; no rise = failed grasp → retry.
7. **Training data** — synthetic (Isaac Sim) + real (photos/video + Label Studio) for any new object.
Before grasping on the robot: egg orientation, ruler-checked centre offset, mount measured on the robot.

---

## Task status

Task: *integrate the egg model captured with the D455 into Isaac Sim through ROS 2; real-time communication between
camera and simulation with accurate object representation, spatial positioning and synchronisation; validate with
practical tests, document, and identify applications for manipulation and grasping.*

| Part | Done | Evidence | Open |
|---|---|---|---|
| Egg model D455 → ROS 2 → Isaac Sim | ✅ | `egg_twin.py`, `start_all.sh` (twin), up to 6 eggs | egg mesh is real-size, not a scan of the egg |
| Real-time communication | ✅ | ~156 ms camera → Isaac Sim, ~12 Hz | GPU inference for > 25 Hz |
| Object representation | ✅ (partly) | 57 × 44 mm egg, brown material, height above table, lift visible | egg orientation not transferred |
| Spatial positioning | ✅ (ruler test pending) | TF `base_link`, tilt measured from the table (`measure_mount.py`), sim within 1.3 cm | ruler test, update tilt (camera moved), mount on the real robot |
| Synchronisation | ✅ | image time stamps through the chain, sim egg = real egg (< 0.5 cm, 95 %) | — |
| Validation / practical tests | ✅ | detection tests (real, sim, ball), delay test, live demo video; ruler test script ready | ruler measurement |
| Documentation | ✅ | this README, `free_detector/README.md`, team report PDF, photos | — |
| Applications for grasping | ✅ | section above + `Documentation/applications_grasping.md` | — |

---

## Demo video

`Documentation/egg_demo_live.mp4` (40 s, H.264): the live yolo image with the egg box plus a bar under it with score,
distance and X, Y, Z, while the egg is moved from 19.7 to 69.8 cm, left / right and held in a hand. Record a new one
(pipeline running; `--ns /sim` for Isaac Sim):
```bash
python3 scripts/record_demo_video.py --seconds 40 --out Documentation/egg_demo_live.mp4
```

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
| a ping-pong ball / round orange object detected as an egg | a 1-class model is in use: use `egg_ball_ls_yolov8s.pt` (default) or the free detector |
| boxes blink on and off | score near the threshold: tracker hysteresis (`config/egg_bytetrack.yaml`); egg on the ball holder → more labelled data |
| Label Studio crashes at start (`send_default_pii`) | `sentry-sdk>=3` in its env: `pip install "sentry-sdk<3"` in the `labelstudio` env |
| a new model finds nothing although trained well | trained with a newer ultralytics (e.g. YOLO26 from the platform): ultralytics ≥ the version it was trained with |
| twin egg floats a few cm above the sim table | real table height off: set `--twin-real-table` (lying real egg Z − 0.022) |
| twin egg does not rise when lifted | started with an old version, or `--twin-real-z` mixed up: use `--twin-real-table` |
| desktop icon does nothing | right-click → *Allow Launching*; logs in `/tmp/meshari_egg/` |

---

## Known limits and next steps

* yolo_ros computes the 3D point from the egg surface the camera sees, so it can be **1–2 cm too close to the
  camera** along the viewing direction; `free_detector` adds the egg radius (22 mm) along the viewing ray.
* Real training data is one brown egg in one room; white eggs and the robot's own table are covered only by the
  sim images — record one session on the robot (and with a white egg) and retrain.
* **Egg vs ball:** solved for ball-only scenes (0 / 40); an egg on the red ball holder can still be classed "ball" —
  record that case, label it in Label Studio, retrain.
* **Licenses:** the default pipeline uses ultralytics (AGPL-3.0) and yolo_ros (GPL-3.0); for a sold robot use
  `free_detector/` (Apache-2.0 / MIT parts) and ship the third-party license texts.
* **Egg orientation** is not estimated (the twin egg always has the same rotation).
* When the camera is mounted in the robot, measure the tilt there and update `MOUNT_RPY`, set `MOUNT_ARGS=""` and
  `TWIN_ARGS="--twin-real-z"` in `scripts/start_all.sh` (measure with `scripts/measure_mount.py`).
* GPU inference in ROS needs a CUDA torch (or onnxruntime-gpu) for Python 3.10; the CPU gives ~10–12 Hz live,
  enough for picking still eggs.
