"""D455PointCloud - live 90 FPS point cloud from an Intel RealSense D455.

Default depth 424x240 Z16 @ 90 FPS: sees from ~0.16 m (848x480 only from ~0.33 m, because a stereo
camera's minimum distance = fx * baseline / (126 + disparity_shift) and fx grows with resolution).
disparity_shift > 0 moves the measurable window closer still, at the cost of far range.
Deprojection is a precomputed per-pixel ray table: points = rays * z, valid pixels only.
Pass device="cuda" to do it in torch on the GPU (useful when the consumer is a torch policy anyway).

    with D455PointCloud(decimation=2) as cam:
        while True:
            pts = cam.read()                 # (N,3) float32 metres, optical frame
            # or: depth, stamp_ms, n = cam.read_depth(); pts = cam.deproject(depth)

CLI:
    python d455_pointcloud.py                        # FPS / drop / timing benchmark
    python d455_pointcloud.py --view                 # Open3D viewer (renders at screen rate)
    python d455_pointcloud.py --save-intrinsics ../config/d455_intrinsics.json
"""
import json
import time

import numpy as np
import pyrealsense2 as rs


class D455PointCloud:
    def __init__(self, width=424, height=240, fps=90, decimation=1, z_range=None, disparity_shift=0, far_limit_m=6.0,
                 filters=(), queue_size=2, exposure_limit_us=10000, device=None, serial=None):
        self.decimation, self.filters, self.device = decimation, list(filters), device

        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # A frame_queue decouples the USB thread from us: size 1 = always newest (lowest latency),
        # size 2-4 = absorbs jitter without drops. keep_frames lets us hold the frame after dequeue.
        self.queue = rs.frame_queue(queue_size, keep_frames=True)
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(cfg, self.queue)

        sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()  # 0.001 -> Z16 unit is 1 mm
        # Exposure must stay below the 11.1 ms frame period, else the sensor silently falls to 60/30 FPS.
        if hasattr(rs.option, "auto_exposure_limit") and sensor.supports(rs.option.auto_exposure_limit):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            sensor.set_option(rs.option.auto_exposure_limit, exposure_limit_us)

        # Disparity shift lives in the advanced-mode depth table and survives until the camera is unplugged,
        # so always write it (also resets a shift left over from an earlier run).
        adv = rs.rs400_advanced_mode(self.profile.get_device())
        if adv.is_enabled():
            table = adv.get_depth_table()
            table.disparityShift = int(disparity_shift)
            adv.set_depth_table(table)
        elif disparity_shift:
            raise RuntimeError("disparity_shift needs advanced mode: rs.rs400_advanced_mode(dev).toggle_advanced_mode(True)")
        self.disparity_shift = int(disparity_shift)

        self.intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        # Measurable window (same formula as shared.d455_model.depth_window); anything outside is noise.
        fb = self.intrinsics.fx * 0.095
        auto = (0.95 * fb / (126 + disparity_shift), min(far_limit_m, fb / disparity_shift) if disparity_shift else far_limit_m)
        self.z_range = z_range or auto
        self._rays = self._ray_table(self.intrinsics, decimation)
        self._rx, self._ry = (np.ascontiguousarray(self._rays[..., k]).ravel() for k in (0, 1))
        self._last_number = None
        self.dropped = 0

    @staticmethod
    def _ray_table(i, d):
        """(h, w, 3) unit-depth rays for the (decimated) pixel grid, i.e. deprojection at z = 1 m."""
        u, v = np.meshgrid(np.arange(0, i.width, d, dtype=np.float32),
                           np.arange(0, i.height, d, dtype=np.float32))
        if any(abs(c) > 1e-6 for c in i.coeffs):  # D4xx depth has zero distortion; exact fallback just in case
            rays = np.array([rs.rs2_deproject_pixel_to_point(i, [x, y], 1.0)
                             for x, y in zip(u.ravel(), v.ravel())], np.float32).reshape(*u.shape, 3)
        else:
            rays = np.stack([(u - i.ppx) / i.fx, (v - i.ppy) / i.fy, np.ones_like(u)], axis=-1)
        return rays

    # ---------------------------------------------------------------- capture
    def read_depth(self, timeout_ms=1000):
        """Block for the next frame -> (depth uint16 (h,w) decimated, timestamp_ms, frame_number)."""
        f = self.queue.wait_for_frame(timeout_ms)
        f = f.as_frameset().get_depth_frame() if f.is_frameset() else f.as_depth_frame()
        for flt in self.filters:  # e.g. rs.temporal_filter(); avoid spatial_filter at 90 FPS (~10 ms)
            f = flt.process(f).as_depth_frame()
        n = f.get_frame_number()
        if self._last_number is not None and n > self._last_number + 1:
            self.dropped += n - self._last_number - 1
        self._last_number = n
        d = self.decimation
        depth = np.asanyarray(f.get_data())[::d, ::d].copy()  # copy -> frame can go back to the pool
        return depth, self._stamp_ms(f), n

    @staticmethod
    def _stamp_ms(f):
        """Capture time in host epoch ms. librealsense's global-time estimate sometimes starts ~140 ms
        off for a whole run (measured on this D455); real USB transfer is ~3-4 ms. If the estimate
        disagrees with the host arrival time by >30 ms, trust the arrival time instead."""
        ts = f.get_timestamp()
        if f.supports_frame_metadata(rs.frame_metadata_value.time_of_arrival):
            toa = f.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
            if f.get_frame_timestamp_domain() != rs.timestamp_domain.global_time or abs(toa - ts) > 30:
                return float(toa)
        return ts

    def deproject(self, depth, organized=False):
        """uint16 depth -> points in the depth optical frame (x right, y down, z forward), metres.
        organized=False: (N,3) valid points only. organized=True: (h,w,3) with NaN for invalid."""
        z = depth.astype(np.float32) * self.depth_scale
        if self.device:  # GPU path (torch). Result stays on the device.
            import torch
            if not isinstance(self._rays, torch.Tensor):
                self._rays = torch.from_numpy(self._rays).to(self.device)
            z = torch.from_numpy(z).to(self.device, non_blocking=True)
            valid = (z > self.z_range[0]) & (z < self.z_range[1])
            pts = self._rays * z[..., None]
            return torch.where(valid[..., None], pts, torch.nan) if organized else pts[valid]
        if organized:
            z[(z <= self.z_range[0]) | (z >= self.z_range[1])] = np.nan
            return self._rays * z[..., None]
        # Pick valid pixels FIRST, then multiply only those (4x faster than multiply-all + boolean mask).
        i = np.flatnonzero((z > self.z_range[0]) & (z < self.z_range[1]))
        zi = z.ravel()[i]
        pts = np.empty((len(i), 3), np.float32)
        np.multiply(self._rx[i], zi, out=pts[:, 0])
        np.multiply(self._ry[i], zi, out=pts[:, 1])
        pts[:, 2] = zi
        return pts

    def read(self, organized=False):
        return self.deproject(self.read_depth()[0], organized)

    # ---------------------------------------------------------------- utils
    def intrinsics_dict(self):
        i = self.intrinsics
        return dict(width=i.width, height=i.height, fx=i.fx, fy=i.fy, ppx=i.ppx, ppy=i.ppy,
                    model=str(i.model), coeffs=list(i.coeffs), depth_scale=self.depth_scale,
                    baseline_m=0.095, disparity_shift=self.disparity_shift, serial=self.profile.get_device().get_info(rs.camera_info.serial_number))

    def close(self):
        self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=424)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--disparity-shift", type=int, default=0, help="e.g. 50 -> ~0.12-0.41 m at 424x240")
    ap.add_argument("--far", type=float, default=6.0, help="ignore points beyond this many metres")
    ap.add_argument("--decimation", type=int, default=1)
    ap.add_argument("--device", default=None, help="e.g. cuda")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--save-intrinsics", metavar="PATH")
    a = ap.parse_args()

    with D455PointCloud(a.width, a.height, decimation=a.decimation, disparity_shift=a.disparity_shift, far_limit_m=a.far,
                        device=a.device) as cam:
        print(f"{a.width}x{a.height} @90, disparity shift {cam.disparity_shift}: "
              f"measuring {cam.z_range[0]:.2f} - {cam.z_range[1]:.2f} m")
        if a.save_intrinsics:
            with open(a.save_intrinsics, "w") as fp:
                json.dump(cam.intrinsics_dict(), fp, indent=2)
            print("saved", a.save_intrinsics, cam.intrinsics_dict())
            raise SystemExit

        vis = None
        if a.view:
            import open3d as o3d
            vis, pcd = o3d.visualization.Visualizer(), o3d.geometry.PointCloud()
            vis.create_window("D455 point cloud")

        frames, t_proc, t0 = 0, 0.0, time.perf_counter()
        while True:
            depth, stamp, n = cam.read_depth()
            t = time.perf_counter()
            pts = cam.deproject(depth)
            t_proc += time.perf_counter() - t
            frames += 1
            if vis:
                pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy() if a.device else pts)
                if frames == 1:
                    vis.add_geometry(pcd)
                vis.update_geometry(pcd)
                if not vis.poll_events():
                    break
                vis.update_renderer()
            if (dt := time.perf_counter() - t0) >= 1.0:
                print(f"{frames / dt:5.1f} FPS | {len(pts):6d} pts | deproject {1e3 * t_proc / frames:4.2f} ms "
                      f"| dropped total {cam.dropped} | latency {time.time() * 1e3 - stamp:5.1f} ms")
                frames, t_proc, t0 = 0, 0.0, time.perf_counter()
