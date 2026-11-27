# Applications for robotic object manipulation and grasping

What the D455 → ROS 2 → Isaac Sim egg pipeline gives the meshari two-arm robot, and how each output can be used
for manipulation. All outputs are in the robot's own frame (`base_link`), in metres, at camera rate.

## What the pipeline delivers

| Output | Topic | Use in manipulation |
|---|---|---|
| Egg centre X, Y, Z (best egg) | `/egg/point` (PointStamped) | grasp target for one arm |
| All eggs | `/egg/points` (PoseArray) | choose which egg, plan an order, avoid the others |
| Egg vs ball class + score | detector output (`/yolo/...` or `free/...`) | pick only eggs, ignore look-alikes |
| Real size of the object (from depth) | used as a filter (2–10 cm) | gripper opening, reject wrong objects |
| Live twin in Isaac Sim | `/twin/egg_poses`, the sim scene | test arm motions on the real scene before moving the real robot |
| Same camera model in sim (calibrated intrinsics, mount) | `isaac/meshari_d455.py` | train and test grasping in simulation with the real camera view |

## Applications

### 1. Pick-and-place of eggs (first target)
Camera finds the egg → `/egg/point` → inverse kinematics for the nearest arm → approach from above → close the
gripper → place it (tray, holder). The 20 cm – 1 m range filter matches the arms' reach; the egg's height above the
table tells whether it is lying on the table or already held.
**Needed:** arm IK / motion planning (MoveIt 2), a soft gripper (eggs break at ~25–40 N), a calibrated
`MOUNT_RPY` on the real robot (`scripts/measure_mount.py`, pending), and the ruler-checked egg-centre offset.

### 2. Grasp rehearsal in the digital twin ("try in sim first")
The twin puts the real eggs into Isaac Sim at the measured positions. A planned grasp can be executed on the
simulated meshari first: collisions, joint limits and reachability are checked on the real scene; only a
successful plan is sent to the real arms. This lowers the risk of breaking eggs or hitting the table while
developing.

### 3. Training grasp policies in Isaac Lab (sim-to-real)
Because the sim camera uses the real D455 calibration and mount, a policy trained in Isaac Lab (reinforcement or
imitation learning) sees the same image geometry as on the robot. The physics drop script (`isaac/drop_egg.py`:
60 g egg, real friction and bounce) generates random egg poses for training; the twin then provides real egg
positions as the policy's input on the real robot.

### 4. Sorting and selective picking
The detector separates egg from ball (and can learn more classes from new Label Studio labels). Uses: pick only
eggs from a mixed scene, sort by class, reject foreign objects, or later classify eggs (size, colour, cracked)
with a model trained on labelled examples.

### 5. Multi-object handling with two arms
`/egg/points` gives every egg; the twin keeps each egg's identity (no swapping). The left and right arms can each
take the egg on their side, or one arm holds a tray while the other places eggs, with the twin used to check that
the arms do not collide.

### 6. Grasp verification and monitoring
After a grasp the egg's measured height rises with the gripper (the twin shows it lifting). If the egg stays on
the table after the "lift", the grasp failed and can be retried. The same signal can count eggs placed, detect a
dropped egg, or stop the arm when the egg leaves the gripper unexpectedly.

### 7. Generating training data
`detection/gen_egg_rgb.py` renders thousands of labelled images (eggs, balls, holders) with the calibrated camera;
`scripts/record_raw_video.py` / `record/take_photos.py` + Label Studio add real labelled images. The same workflow
works for any new object the robot must handle (tools, fruit, parts).

## Limits to solve before grasping on the real robot

| Limit | Effect on grasping | Fix |
|---|---|---|
| Egg orientation (which way it lies) not estimated | gripper approach angle unknown for a lying egg | fit an ellipse to the box / depth points, or a pose model |
| Egg centre from the visible surface (+22 mm offset) | 1–2 cm error along the viewing direction if the offset is off | ruler test (`scripts/measure_accuracy.py`) |
| Camera mount on the robot not yet measured | systematic offset of all positions | `scripts/measure_mount.py` on the robot |
| CPU detection (~10–25 Hz) and the twin's smoothing | sim egg lags a moving egg by a few cm | fine for static picking; for moving eggs use the raw `/egg/points` |
| Ultralytics/yolo_ros licences (AGPL/GPL) | a sold robot would need a licence | use `free_detector/` (Apache-2.0 / MIT parts only) |
