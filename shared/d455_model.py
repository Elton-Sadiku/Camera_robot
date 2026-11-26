"""Where the D455 sits on the meshari robot - one place, read by the real camera's TF and the Isaac Sim build.

camera_link pose in base_link (metres, radians; ROS roll-pitch-yaw, positive pitch = tilted DOWN).
  - meshari base_link: FORWARD = -Y, LEFT = +X, UP = +Z, origin on the floor -> yaw = -90 deg.
  - camera_link = the D455 depth origin (left imager), as in realsense-ros: torso camera slot centre
    (0.1306, -0.043, 1.153) from the CAD (Isaac SIm/meshari 1.usd) + 47.5 mm to the robot's left.
  - pitch 11.55 deg down: measured on the real D455 by fitting the table plane in its depth image.
    Roll 0: in the robot's slot the camera sits straight.
    Re-measure whenever the camera is moved (README, "Measuring the camera tilt"); a test rig can override
    at launch, e.g.  ros2 launch ros2/d455.launch.py mount_pitch:=0.1169 mount_roll:=-0.0105

Used by ros2/d455.launch.py (static TF base_link -> camera_link). isaac/meshari_d455.py (--pitch-deg / --roll-deg)
and detection/gen_egg_rgb.py (CAM_RPY) use the same numbers as defaults - keep them equal.
"""
MOUNT_XYZ = (0.1781, -0.043, 1.153)
MOUNT_RPY = (0.0, 0.2015855, -1.5707963)   # roll 0, pitch 11.55 deg down, yaw -90 deg -> looking along base_link -Y
