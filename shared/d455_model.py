"""One D455 model shared by Isaac Lab (training) and the real robot (deployment).

Everything that shapes the policy's depth input lives here, so sim and real cannot drift apart:
  intrinsics()     - real calibration (config/d455_intrinsics.json), optionally decimated
  mount_quat_ros() - camera mount pose -> quaternion of the *optical* frame (Isaac Lab convention="ros")
  to_obs()         - deterministic: inf/NaN/out-of-range -> 0 (what the real D455 reports as "no data")
  stereo_noise()   - stochastic, sim only: D455 stereo error model + dropouts
  real_obs()       - real Z16 frame -> exactly what the sim observation term produces

Only depends on torch + stdlib (works in Isaac Sim's Python 3.11 and ROS 2 Humble's Python 3.10).
"""
import json
import math
from pathlib import Path



import torch

INTRINSICS_FILE = Path(__file__).resolve().parent.parent / "config" / "d455_intrinsics.json"

# ---- Camera + policy-input settings. Change here only; the ROS node, policy node and Isaac Lab import them.
# Near range of a stereo camera: min_z = fx * baseline / (126 + DISPARITY_SHIFT), so LOWER resolution sees CLOSER:
#   848x480 -> 0.33 m   640x360 -> 0.25 m   424x240 -> 0.16 m   (measured fx on this D455, all at 90 FPS)
# DISPARITY_SHIFT > 0 moves the whole window closer: 424x240 + 50 -> 0.12 m .. 0.41 m (far range is lost!)
# After changing DEPTH_PROFILE, re-save the calibration:
#   python3 pointcloud/d455_pointcloud.py --save-intrinsics config/d455_intrinsics.json
DEPTH_PROFILE = (424, 240, 90)   # width, height, fps
DISPARITY_SHIFT = 0
DECIMATION = 1                   # 1 = policy sees full 424x240 (egg ~33 px at 0.3 m); 2 -> 212x120 (faster training)
FAR_LIMIT_M = 1.5                # depth beyond this is dropped (egg-grasping workspace: 0.16-1.5 m)
                                 # with the camera 1.2 m up / 23 deg down, only a thin floor strip (~1.0-1.2 m ahead) is in range

# Camera mount: camera_link pose in base_link (metres, radians; positive pitch = tilted down).
# EXAMPLE values for now - the real mount is NOT measured yet; replace them once measured. Read by ros2/d455.launch.py, isaac/d455_cfg.py and isaac/sim_publish_d455.py.
MOUNT_XYZ = (0.05, 0.0, 1.20)
MOUNT_RPY = (0.0, 0.40, 0.0)     # 0.40 rad = 23 deg down


def depth_window(fx, baseline_m=0.095, shift=DISPARITY_SHIFT, far=FAR_LIMIT_M):
    """(min_z, max_z) the stereo matcher can measure at this focal length (px) and disparity shift."""
    min_z = 0.95 * fx * baseline_m / (126 + shift)          # 5% margin below the theoretical limit
    return round(min_z, 3), round(min(far, fx * baseline_m / shift) if shift else far, 3)


_CAL = json.loads(INTRINSICS_FILE.read_text())
Z_RANGE = depth_window(_CAL["fx"], _CAL.get("baseline_m", 0.095))   # (0.155, 1.0) for 424x240, no shift


def intrinsics(decimation=DECIMATION, path=INTRINSICS_FILE):
    """Real intrinsics scaled to the decimated grid. ppx/d (not (ppx+.5)/d-.5) because the real side
    decimates by striding [::d, ::d], so sim pixel k must see the same ray as real pixel d*k."""
    i = json.loads(Path(path).read_text())
    d = decimation
    w, h = math.ceil(i["width"] / d), math.ceil(i["height"] / d)
    fx, fy, cx, cy = i["fx"] / d, i["fy"] / d, i["ppx"] / d, i["ppy"] / d
    return dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy, K=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
                baseline_m=i.get("baseline_m", 0.095), depth_scale=i.get("depth_scale", 0.001))


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


# camera_link (x fwd, y left, z up) -> *_optical_frame (x right, y down, z fwd); same as realsense-ros TF
LINK_TO_OPTICAL = (0.5, -0.5, 0.5, -0.5)  # (w, x, y, z), rpy = (-90, 0, -90) deg


def mount_quat_ros(roll=0.0, pitch=0.0, yaw=0.0):
    """Mount rpy (radians) of camera_link in the robot link -> (w,x,y,z) of the optical frame.
    Use with Isaac Lab OffsetCfg(convention="ros"); use the same rpy in the ROS static TF."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2),
                              math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2))
    q_link = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
              cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
    return _qmul(q_link, LINK_TO_OPTICAL)


def to_obs(depth_m: torch.Tensor, z_range=Z_RANGE) -> torch.Tensor:
    """(..., H, W) metres -> invalid (inf, NaN, outside z_range) set to 0, like the real sensor."""
    depth_m = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where((depth_m > z_range[0]) & (depth_m < z_range[1]), depth_m, torch.zeros_like(depth_m))


def stereo_noise(depth_m: torch.Tensor, fx: float, baseline_m=0.095, subpixel=0.08,
                 dropout=0.01, edge_rel=0.05, edge_dropout=0.5, z_range=Z_RANGE) -> torch.Tensor:
    """D455-like corruption of a clean (N, H, W) depth image in metres (sim only).

    - axial error grows with z^2:  sigma = z^2 * subpixel / (fx * baseline)   (Intel's D4xx RMS model)
      with fx = focal length *at full 848 px resolution* (pixel disparity is measured there)
    - 1 mm quantisation (Z16)
    - random pixel dropout, and extra dropout on depth edges (stereo occlusion / flying pixels)
    """
    z = depth_m
    z = z + torch.randn_like(z) * (z * z) * (subpixel / (fx * baseline_m))
    z = torch.round(z * 1000.0) / 1000.0
    # depth discontinuities: relative jump to the right/bottom neighbour
    jump = torch.zeros_like(z)
    jump[..., :, :-1] = (z[..., :, 1:] - z[..., :, :-1]).abs()
    jump[..., :-1, :] = torch.maximum(jump[..., :-1, :], (z[..., 1:, :] - z[..., :-1, :]).abs())
    edge = jump > edge_rel * z.clamp_min(1e-3)
    drop = (torch.rand_like(z) < dropout) | (edge & (torch.rand_like(z) < edge_dropout))
    return to_obs(torch.where(drop, torch.zeros_like(z), z), z_range)


def real_obs(depth_u16, depth_scale=0.001, stride=DECIMATION, z_range=Z_RANGE, device="cpu"):
    """Real full-res Z16 frame (numpy uint16 or 32FC1 metres) -> (1, H/stride, W/stride) metres,
    identical to the sim observation term. Divide by z_range[1] the same way the ObsTerm `scale` does."""
    import numpy as np
    d = np.ascontiguousarray(depth_u16[::stride, ::stride])
    z = torch.from_numpy(d.astype(np.float32) * (depth_scale if d.dtype == np.uint16 else 1.0))
    return to_obs(z.to(device), z_range)[None] / z_range[1]
