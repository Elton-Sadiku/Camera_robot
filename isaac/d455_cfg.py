"""Isaac Lab 2.3.2 config pieces for a D455 that matches the real one.

Drop into any manager-based env:

    from d455_cfg import d455_camera_cfg, D455_OBS, D455_MOUNT_EVENT

    @configclass
    class MySceneCfg(InteractiveSceneCfg):
        ...
        d455 = d455_camera_cfg("{ENV_REGEX_NS}/Robot/base/d455")

    @configclass
    class ObservationsCfg:
        @configclass
        class DepthCfg(ObsGroup):
            depth = D455_OBS
            def __post_init__(self):
                self.enable_corruption = True     # False in the PLAY cfg -> noise off
                self.concatenate_terms = False
        depth: DepthCfg = DepthCfg()

    @configclass
    class EventCfg:
        d455_mount = D455_MOUNT_EVENT

Why "distance_to_image_plane": that is z-depth, the same quantity the D455 reports in Z16.
("distance_to_camera" is ray length and would be wrong near the image edges.)
"""
import sys
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab.utils.noise import NoiseCfg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.d455_model import (DECIMATION, MOUNT_RPY, MOUNT_XYZ, Z_RANGE, intrinsics, mount_quat_ros,  # noqa: E402
                               stereo_noise, to_obs)

FULL = intrinsics(decimation=1)   # full-res fx drives the stereo error model
POLICY = intrinsics()             # what we render: 424x240 / DECIMATION (424x240 for DECIMATION=1)

# Mount of camera_link on the robot link (metres / radians), from shared.d455_model = ros2/d455.launch.py.
MOUNT_POS = MOUNT_XYZ             # (0.1781, -0.043, 1.153) - torso camera slot of the meshari robot (from CAD)
# MOUNT_RPY = (0.0, 0.087, -1.5708): 5 deg down, looking along base_link -Y (this robot's forward); positive pitch = tilt down


def d455_camera_cfg(prim_path: str, pos=MOUNT_POS, rpy=MOUNT_RPY, update_period: float = 0.0) -> TiledCameraCfg:
    """TiledCamera with the real D455's (decimated) intrinsics, mounted in ROS optical convention."""
    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=mount_quat_ros(*rpy), convention="ros"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=POLICY["K"], width=POLICY["width"], height=POLICY["height"],
            clipping_range=(0.05, 20.0),  # render wider than Z_RANGE; to_obs() applies the sensor's range
        ),
        width=POLICY["width"],
        height=POLICY["height"],
        update_period=update_period,      # 0 = every env step; 1/90 to mimic the real rate exactly
    )


# ------------------------------------------------------------------ observation
def d455_depth(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("d455")) -> torch.Tensor:
    """(N, H, W) metres, invalid = 0. Noise (if enabled) and scale are applied by the obs manager."""
    return to_obs(env.scene.sensors[sensor_cfg.name].data.output["distance_to_image_plane"][..., 0])


def _d455_noise(data: torch.Tensor, cfg: "D455NoiseCfg") -> torch.Tensor:
    return stereo_noise(data, fx=FULL["fx"], baseline_m=FULL["baseline_m"], subpixel=cfg.subpixel,
                        dropout=cfg.dropout, edge_rel=cfg.edge_rel, edge_dropout=cfg.edge_dropout)


@configclass
class D455NoiseCfg(NoiseCfg):
    """Stereo noise as a NoiseCfg, so the group's enable_corruption flag switches it on/off."""
    func = _d455_noise
    subpixel: float = 0.08     # randomise per run (0.05-0.12) to cover unit-to-unit / lighting variation
    dropout: float = 0.01
    edge_rel: float = 0.05
    edge_dropout: float = 0.5


# obs manager order: func -> noise -> clip -> scale. scale = 1/z_max mirrors shared.real_obs().
D455_OBS = ObservationTermCfg(func=d455_depth, noise=D455NoiseCfg(), clip=(0.0, Z_RANGE[1]), scale=1.0 / Z_RANGE[1])


# ------------------------------------------------------------------ domain randomisation
def randomize_d455_mount(env: ManagerBasedEnv, env_ids, sensor_cfg: SceneEntityCfg = SceneEntityCfg("d455"),
                         pos_std: float = 0.01, rot_std: float = 0.0175):
    """Perturb the camera mount (+-1 cm, +-1 deg by default) to cover calibration error.
    Use mode="startup": it perturbs the current pose, so repeated calls would accumulate."""
    cam = env.scene.sensors[sensor_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    n = len(ids)
    pos = cam.data.pos_w[ids] + torch.randn(n, 3, device=env.device) * pos_std
    r, p, y = (torch.randn(n, 3, device=env.device) * rot_std).unbind(-1)
    quat = quat_mul(cam.data.quat_w_ros[ids], quat_from_euler_xyz(r, p, y))
    cam.set_world_poses(pos, quat, env_ids=ids, convention="ros")


D455_MOUNT_EVENT = EventTermCfg(func=randomize_d455_mount, mode="startup")

__all__ = ["d455_camera_cfg", "D455_OBS", "D455NoiseCfg", "D455_MOUNT_EVENT", "DECIMATION", "Z_RANGE"]
