"""Digital twin of the eggs: every egg found by the REAL D455 gets an egg in Isaac Sim at the same place.

Listens to /egg/points (geometry_msgs/PoseArray in base_link, from ros2/egg_xyz.py on the real camera: all accepted
eggs) and moves one sim egg per real egg. In meshari_d455.usd the robot's base_link sits at the world origin, so the
real coordinates are used directly. Started by:
  isaac/isaac_ros2.sh isaac/meshari_d455.py --run --twin
with the real pipeline running (ros2/d455.launch.py, ros2/egg_yolo.launch.py, ros2/egg_xyz.py).

  - Several eggs: up to MAX_EGGS sim eggs (/World/TestEgg, /World/TwinEgg1, ...). Each sim egg stays paired with the
    same real egg (nearest position, within MATCH_M), so eggs do not swap; positions are smoothed (no jitter).
  - A sim egg is hidden when its real egg has not been seen for HIDE_AFTER_S.
  - real_table_z=0.992 (default): height = sim table + the egg's height ABOVE THE REAL TABLE (a lying egg lies on the
    sim table, a lifted egg rises by the same amount). 0.992 m = the desk test rig (camera 16.1 cm above the table).
    real_table_z=None: X, Y, Z as measured (camera on the robot, real table = sim table).
"""
import math
import time

import rclpy
from geometry_msgs.msg import PoseArray
from pxr import Gf, Sdf, UsdGeom

MAX_EGGS = 6           # sim eggs available
HIDE_AFTER_S = 1.0     # a real egg not seen for this long -> its sim egg is hidden
MATCH_M = 0.10         # a detection within 10 cm of a tracked egg is the same egg
SMOOTH = 0.4           # 0..1, share of each new measurement (lower = smoother, slower)


class EggTwin:
    def __init__(self, stage, topic="/egg/points", real_table_z=0.992, egg_path="/World/TestEgg",
                 table_top=1.153 - 0.20, egg_d=0.044):
        src = stage.GetPrimAtPath(egg_path)
        if not src.IsValid():
            raise RuntimeError(f"{egg_path} not found - run isaac/add_test_egg.py first")
        layer = stage.GetEditTarget().GetLayer()
        paths = [egg_path] + [f"/World/TwinEgg{k}" for k in range(1, MAX_EGGS)]
        for p in paths[1:]:                                   # copies of the test egg (same mesh, size, material)
            if not stage.GetPrimAtPath(p).IsValid():
                Sdf.CopySpec(layer, Sdf.Path(egg_path), layer, Sdf.Path(p))
        self.slots = []
        for p in paths:
            prim = stage.GetPrimAtPath(p)
            op = next(o for o in UsdGeom.Xformable(prim).GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            UsdGeom.Imageable(prim).MakeInvisible()
            self.slots.append({"prim": prim, "op": op, "pos": None, "last": 0.0, "visible": False})
        self.real_table_z, self.sim_table_z, self.z_lying = real_table_z, table_top, table_top + egg_d / 2
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("egg_twin")
        self.node.create_subscription(PoseArray, topic, self._on_points, 10)
        mode = f"height above the real table (z {real_table_z:.3f} m)" if real_table_z is not None else "real X, Y, Z"
        print(f"[egg twin] following {topic} -> up to {MAX_EGGS} sim eggs ({mode})", flush=True)

    def _z(self, z):
        if self.real_table_z is None:
            return z
        return max(self.z_lying, self.sim_table_z + (z - self.real_table_z))   # never below a lying egg

    def _on_points(self, m):
        now = time.time()
        dets = [(q.position.x, q.position.y, self._z(q.position.z)) for q in m.poses]
        # pair each detection with the nearest tracked egg (closest pairs first), within MATCH_M
        pairs = sorted(((math.dist(d[:2], s["pos"][:2]), i, j) for i, d in enumerate(dets)
                        for j, s in enumerate(self.slots) if s["pos"] is not None and now - s["last"] < HIDE_AFTER_S),
                       key=lambda t: t[0])
        used_d, used_s = set(), set()
        for dist, i, j in pairs:
            if dist > MATCH_M or i in used_d or j in used_s:
                continue
            s = self.slots[j]
            s["pos"] = tuple(o + SMOOTH * (n - o) for o, n in zip(s["pos"], dets[i]))
            s["last"] = now
            used_d.add(i), used_s.add(j)
        # new eggs -> free slots (not seen recently)
        for i, d in enumerate(dets):
            if i in used_d:
                continue
            free = [j for j, s in enumerate(self.slots) if j not in used_s and now - s["last"] >= HIDE_AFTER_S]
            if not free:
                break
            j = free[0]
            self.slots[j].update(pos=d, last=now)
            used_s.add(j)

    def update(self):
        """Call once per Isaac frame: reads pending ROS messages, moves / shows / hides the sim eggs."""
        rclpy.spin_once(self.node, timeout_sec=0.0)
        now = time.time()
        for k, s in enumerate(self.slots):
            seen = s["pos"] is not None and now - s["last"] < HIDE_AFTER_S
            if seen:
                s["op"].Set(Gf.Vec3d(*s["pos"]))
            if seen != s["visible"]:
                UsdGeom.Imageable(s["prim"]).MakeVisible() if seen else UsdGeom.Imageable(s["prim"]).MakeInvisible()
                s["visible"] = seen
                n = sum(x["visible"] for x in self.slots)
                print(f"[egg twin] egg {k + 1} " + (f"seen at x={s['pos'][0]:.3f} y={s['pos'][1]:.3f} z={s['pos'][2]:.3f}"
                                                    if seen else "gone - hidden") + f"  ({n} egg(s) in view)", flush=True)
