# D455 → 90 FPS point cloud → ROS 2 Humble → Isaac Sim 5.1 / Isaac Lab 2.3.2

```
pointcloud/d455_pointcloud.py   Part 1: D455PointCloud class (pyrealsense2 + NumPy, optional CUDA)
shared/d455_model.py            ONE camera model used by sim AND real: intrinsics, mount, noise, obs
config/d455_intrinsics.json     real calibration of D455 #324422301637 at 424×240 (fx=fy=217.18, ppx=213.47, ppy=121.53)
config/fastdds_shm.xml          Fast DDS profile needed for 90 Hz depth/cloud topics on one host
ros2/d455_node.py               ROS 2 publisher (depth 16UC1 + CameraInfo + PointCloud2)
ros2/d455.launch.py             node + static TFs (base_link → camera_link → optical)
ros2/policy_node.py             deployment: depth → same obs as sim → TorchScript policy → action
isaac/d455_cfg.py               Isaac Lab TiledCamera + obs term + stereo noise + mount randomisation
isaac/sim_publish_d455.py       Isaac Sim D455 publishing on the real topics (sim-in-the-loop test)
source/c++/                     your existing raw V4L2 viewer (main.cpp, frame.*, depth.*; unchanged)
Documentation/                  Word documents (libraries, project report); D455_How_It_Works.pdf
```

## Part 1: 90 FPS point cloud

### Stream settings

| Stream | Mode | Bandwidth | Notes |
|---|---|---|---|
| Depth (default) | **424×240 Z16 @ 90** | 18 MB/s = 147 Mbit/s | Sees from 0.16 m, needed for close grasping |
| Depth (alt) | 848×480 / 640×360 @ 90 | 73 / 41 MB/s | More points and accuracy, but nothing closer than 0.33 / 0.25 m |
| Color (if needed) | 640×360 YUYV @ 90 | 41 MB/s = 332 Mbit/s | 848×480 color tops out at 60 FPS |
| PointCloud2 over DDS | 424×240 xyz float32 | 110 MB/s | default `cloud_decimation:=1`; fine with the Fast DDS SHM profile |
| PointCloud2 at 848×480 | 848×480 xyz float32 | **439 MB/s** | Too heavy. Publish depth and deproject in the consumer, or decimate |

**USB:** use USB 3.x end to end (5 Gbit/s). On USB 2 (≈300 Mbit/s usable) 90 FPS depth is impossible. Right now your D455 enumerates at 5000M **behind an external hub** (`lsusb -t`: Bus 04 → hub → camera). Plug it directly into a root port, and use the original cable or a short certified USB 3 cable. Check with `rs-enumerate-devices | grep -i usb` (it should say 3.2).

### Near range (small robot hands) — current default: 424×240
A stereo camera's closest distance is `fx × baseline / (126 + disparity_shift)`. Lower resolution means a smaller fx, so the camera sees closer. Measured on your D455, all at 90 FPS with 0 drops:

| Setting | Sees from → to | Points | Use |
|---|---|---|---|
| 848×480 | 0.33 m → 6 m (nothing closer than ~1 m showed up in a real test) | ~270k | far / room scale |
| **424×240** (default) | **0.16 m → 6 m** (cut at 1.5 m by `FAR_LIMIT_M`) | ~63k | hands + workspace |
| 424×240 + `disparity_shift:=50` | **0.11 m → 0.41 m** (nothing further!) | close objects only | very close grasping |

Everything is set in `shared/d455_model.py` (`DEPTH_PROFILE`, `DISPARITY_SHIFT`), which the class, the ROS node, the policy node and Isaac Lab all read. Try a shift without editing the file: `ros2 launch ros2/d455.launch.py disparity_shift:=50`. After changing the resolution, re-save the calibration. Points beyond `FAR_LIMIT_M = 1.5` m (depth along the camera axis) are dropped. With the mount from the robot CAD (camera in the torso slot, 1.15 m up, tilted 5° down), the view spans from 24.5° above to 34.5° below horizontal: a 0.75 m table top is visible from ~0.58 m ahead, and the floor from ~1.7 m. The tilt is set in `MOUNT_RPY` (pitch); rebuild the sim file with `isaac/meshari_d455.py --pitch-deg <deg>` to match. The policy sees the full 424×240 image (`DECIMATION = 1`); set 2 for 212×120 and faster training. Even closer than 0.11 m needs a close-range camera such as the D405 (7–50 cm).

### Measured on your D455 (FW 5.17.3.10, USB 3.2, Ryzen AI 7 350)
| Mode | FPS | Dropped | Deproject | Capture → points |
|---|---|---|---|---|
| 848×480, ~270k valid pts | 89.8–90.0 | 0 | 1.6 ms | ~10 ms |
| decimation 2, ~67k pts | 89.9 | 0 | 0.5 ms | ~9 ms |

### What keeps it at 90 FPS
- **Exposure ≤ ~10 ms.** The frame period at 90 FPS is 11.1 ms. If auto-exposure goes longer (dim room), the sensor silently drops to 60 or 30 FPS. The class sets `auto_exposure_limit = 10000 µs`.
- **Ray-table deprojection, valid pixels first.** `rays[v,u] = ((u-ppx)/fx, (v-ppy)/fy, 1)` is computed once. Each frame then picks the valid pixels with `flatnonzero` and multiplies only those. The naive `(rays * z)[mask]` measured 11 ms on this laptop, which caused frame drops. Pass `device="cuda"` to keep the points on the GPU for torch consumers.
- **Decimation by striding** (`[::d, ::d]`) costs nothing. `rs.decimation_filter` (median) is better quality but costs about 1–2 ms. **Avoid `rs.spatial_filter`** at 90 FPS (≈10 ms). `rs.temporal_filter` is affordable: pass it in `filters=[...]`.
- **`rs.frame_queue`** decouples the USB thread. Use size 1 for the lowest latency (always the newest frame) or 2–4 to absorb jitter. Dropped frames are counted from frame-number gaps (`cam.dropped`).
- The Open3D `--view` mode renders at screen refresh, so it will report drops. That is the viewer, not the pipeline. Benchmark without `--view`.

### Setup (Ubuntu 22.04, done once, no kernel patch needed)
```bash
bash scripts/install_ubuntu.sh
```
This installs ROS 2 Humble desktop, realsense2_camera, the RealSense udev rule, and pyrealsense2 + CPU torch for the system Python 3.10. numpy stays the apt 1.21 build, as ROS Humble needs. Then, in **every** terminal you use for this project:
```bash
cd /path/to/Camera_robot
source ./env.sh
```
Replace `/path/to/Camera_robot` with the directory containing this README. You can find it with `pwd` after entering the cloned repository. `env.sh` determines its own location, so no desktop-specific path is required. If you only want to source it from elsewhere, use `source /path/to/Camera_robot/env.sh`.
`env.sh` steps out of conda in that shell (your conda setup is untouched), sources ROS, and sets the Fast DDS profile. Then:
```bash
python3 pointcloud/d455_pointcloud.py
```
```bash
ros2 launch ros2/d455.launch.py
```
Only one process can own the camera at a time: your V4L2 `./d455`, this class, `d455_node.py` or `realsense2_camera`.

## Part 2: ROS 2 Humble + Isaac Sim 5.1 / Isaac Lab 2.3.2

### Frame conventions (the #1 sim-to-real bug)
| Frame | Axes | Used by |
|---|---|---|
| `base_link`, `camera_link` (ROS body, REP-103) | x forward, y left, z up | TF tree, mount measurement |
| `camera_depth_optical_frame` (optical, REP-103) | **x right, y down, z forward** | all depth/cloud data (real and sim) |
| USD camera prim (OpenGL) | −z forward, +y up | Isaac Sim internally |

Isaac Lab handles the conversion with `OffsetCfg(convention="ros")`, which takes the quaternion **of the optical frame**. `shared.d455_model.mount_quat_ros(roll, pitch, yaw)` builds it from the same `camera_link` rpy that goes into the ROS static TF, composed with the fixed link→optical rotation (rpy −90°, 0, −90°). The mount lives in ONE place, `MOUNT_XYZ` / `MOUNT_RPY` in `shared/d455_model.py` (now taken from the meshari robot CAD: the D455 in the torso slot, `camera_link` = depth origin at (0.1781, −0.043, 1.153) m, 5° down, yaw −90° because this robot's `base_link` faces −Y; `isaac/meshari_d455.py` mounts it in Isaac Sim and publishes the same topics). The launch file, `isaac/d455_cfg.py` and `isaac/sim_publish_d455.py` all read it.

### 1. ROS 2 node
```bash
sudo apt install ros-humble-desktop ros-humble-realsense2-camera
```
```bash
source /opt/ros/humble/setup.bash && ros2 launch ros2/d455.launch.py            # mount from shared/d455_model.py; override e.g. mount_pitch:=0.35
```
- Topics: `/camera/camera/depth/{image_rect_raw,camera_info,color/points}`, all in `camera_depth_optical_frame`, stamped with capture time.
- QoS: `SensorData` (best-effort, shallow). Capture runs in its own thread, so a slow subscriber never stalls the camera.
- **Required for 90 Hz:** `export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/config/fastdds_shm.xml` in every terminal (camera node, consumers, Isaac Sim). Fast DDS's default shared-memory segment is 512 KB, smaller than one 814 KB depth frame. Measured without the profile: depth 78–83 Hz and CameraInfo 90 Hz. With it: depth, CameraInfo and cloud all 89.9 Hz, capture → subscriber 10 ms.
- Low latency: the depth image + CameraInfo (73 MB/s) is the real interface. The cloud is only built when someone subscribes. For a zero-copy C++ path, use `realsense2_camera` in a component container with intra-process comms.
- Equivalent official driver (C++, slightly lower CPU), which publishes the same topic names:
  ```bash
  ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=848x480x90 enable_color:=false pointcloud.enable:=true
  ```
  On realsense-ros older than 4.54 the parameter is `depth_module.profile`.

### 2. Simulated D455 in Isaac Lab / Isaac Sim
- **Sensor type:** a depth camera, `TiledCameraCfg` with `data_types=["distance_to_image_plane"]`, which gives z-depth like Z16. **Not RTX Lidar:** that models time-of-flight lidars (beams, rotation, returns), not active-stereo depth.
- **Intrinsics:** taken from the real `config/d455_intrinsics.json` via `PinholeCameraCfg.from_intrinsic_matrix`. Training renders at 424×240 / `DECIMATION` (now 1, so full 424×240) with `fx/d, ppx/d`, which exactly matches striding the real image `[::d, ::d]`. The renderer assumes square pixels (fy is forced to fx). The D455's fx and fy differ by less than 0.1%, so this is harmless.
- **Noise (`stereo_noise`, train only via `enable_corruption`):**
  - σ = z² · subpixel / (fx · baseline), with baseline 95 mm and subpixel ≈ 0.08: about 7 mm at 2 m and 30 mm at 4 m (inside the D455's <2% @ 4 m spec)
  - 1 mm quantisation
  - 1% random dropout and 50% dropout on depth edges
  - out-of-range → 0
- **Further randomisation:**
  - Mount pose: ±1 cm / ±1° (`D455_MOUNT_EVENT`, startup)
  - `subpixel` 0.05–0.12
  - Latency: real capture→policy is ~15–30 ms. Use `ObsTerm(history_length=...)` or run the camera at a lower `update_period` than the policy.
  - Lighting and materials: textureless and shiny surfaces cause holes on the real D455, so add dropout patches if your scenes have them.
- **Same obs code both sides:** `to_obs()` (range → 0) and the `1/z_max` scale are shared. `real_obs()` reproduces the sim term exactly from a Z16 or 32FC1 frame.

### 3. End-to-end workflow
1. **Calibrate:** `python3 pointcloud/d455_pointcloud.py --save-intrinsics config/d455_intrinsics.json`, and measure the mount → set `MOUNT_POS/RPY` and the launch args.
2. **Check real:** `ros2 launch ros2/d455.launch.py`, then in RViz set Fixed Frame = `base_link` and add PointCloud2 `/camera/camera/depth/color/points`. The floor should be flat at z = 0.
3. **Check sim matches real:** set `ISAACLAB_ROOT` to the directory where you installed Isaac Lab, then run:
   ```bash
   export ISAACLAB_ROOT=/path/to/IsaacLab
   "$ISAACLAB_ROOT/isaaclab.sh" -p "$PWD/isaac/sim_publish_d455.py" --enable_cameras
   ```
   The `--enable_cameras` argument is required with `--headless`. Set the ROS variables documented in `isaac/sim_publish_d455.py`, then compare the sim and real clouds and `camera_info` K in RViz:
   ```bash
   ros2 topic echo --once /camera/camera/depth/camera_info
   ```
4. **Train:** add `d455_camera_cfg(...)`, `D455_OBS` and `D455_MOUNT_EVENT` to your env cfg (snippet at the top of `isaac/d455_cfg.py`), then:
   ```bash
   export ISAACLAB_ROOT=/path/to/IsaacLab
   "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task <YourTask> --enable_cameras --headless
   ```
5. **Export:** `play.py --task <YourTask> --enable_cameras` writes `logs/rsl_rl/<task>/<run>/exported/policy.pt` (TorchScript).
6. **Sim-in-the-loop:** run `ros2/policy_node.py -p policy:=.../policy.pt` against step 3's simulator (32FC1). Actions should match play.py.
7. **Deploy:** the same `policy_node.py` against the real camera (16UC1), with the robot driver subscribed to `policy/action`.

### Isaac Sim ↔ ROS 2 Humble notes
- Isaac Sim 5.1 runs Python 3.11, so it **cannot** load system Humble's Python 3.10 rclpy. The bridge ships its own Humble libs at `/path/to/isaacsim/exts/isaacsim.ros2.bridge/humble/lib`. Set `ROS_DISTRO=humble`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and `FASTRTPS_DEFAULT_PROFILES_FILE`, and keep the same `ROS_DOMAIN_ID` in every terminal. Replace `/path/to/isaacsim` with the actual Isaac Sim installation directory. When stopping a headless run, make sure the `python` child process has exited too: killing `isaaclab.sh` alone can leave it publishing.
- Sim depth is `32FC1` in metres. Real depth is `16UC1` in millimetres. `policy_node.py` handles both.
- Rendering 848×480 at 90 Hz for thousands of envs is not needed. The policy rate (e.g. 30–50 Hz) only has to see the *latest* frame on the real robot, which is what keep-last-1 QoS gives you.

## Part 3: meshari robot — sim D455, real D455 side by side, egg detection (depth)

### Sim camera in the robot
`isaac/meshari_d455.py` opens `Isaac SIm/meshari 1.usd` (left unchanged), puts the D455 in the torso camera slot (5° down, real intrinsics, depth origin = left imager) and saves `Isaac SIm/meshari_d455.usd`. Open that in Isaac Sim and press Play. **Start Isaac Sim with `isaac/isaac_ros2.sh`** (sets `ROS_DISTRO` and the bridge libs). Do not rebuild the file while it is open in Isaac Sim.

| | Real D455 (`ros2/d455.launch.py`) | Sim D455 (`meshari_d455.usd`) |
|---|---|---|
| Depth | `/camera/camera/depth/image_rect_raw` 16UC1 mm | `/sim/camera/camera/depth/image_rect_raw` 32FC1 m |
| Info / cloud | `/camera/camera/depth/{camera_info,color/points}` | `/sim/camera/camera/depth/{camera_info,color/points}` |
| Frame | `camera_depth_optical_frame` | `camera_depth_optical_frame` |

Both can run at the same time. `--ns camera/camera` puts the sim on the real topic names instead (then run only one).

### Egg detector (depth only)
```bash
isaac/isaac_ros2.sh detection/gen_egg_depth.py --frames 6000     # synthetic depth + egg boxes from the sim D455
~/isaacsim/python.sh detection/train_egg_detector.py              # GPU training -> detection/models/egg_detector.ts.pt
```
Training adds fresh D455 noise (`shared.d455_model.stereo_noise`) to every batch and feeds `to_obs()/Z_RANGE[1]`, exactly what `real_obs()` gives from the real camera. Run the detector (system Python, CPU, ~23 ms/frame):
```bash
python3 ros2/egg_detector_node.py                                                                      # real D455
python3 ros2/egg_detector_node.py --ros-args -r __ns:=/sim -p depth_topic:=/sim/camera/camera/depth/image_rect_raw   # sim
```
Outputs `egg/point` (PointStamped, egg centre in the depth optical frame), `egg/detection` ([score, x0, y0, x1, y1, X, Y, Z]) and `egg/debug_image`.
