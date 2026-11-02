# Hand-Eye Calibration

RUSSAgent uses [IFL-CAMP/easy_handeye](https://github.com/IFL-CAMP/easy_handeye), a hand-eye calibration package for ROS 1. It is installed by [`scripts/setup.sh`](../scripts/setup.sh). See the [upstream usage guide](https://github.com/IFL-CAMP/easy_handeye#usage) for calibration and TF publishing instructions.

The released setup uses an Azure Kinect mounted on the robot's end effector (eye-in-hand), with an ArUco marker fixed relative to the robot base. Calibration estimates the transform from `iiwa_link_ee` to `camera_base`; the camera driver supplies the remaining camera-frame transforms.

## Calibration workflow

Start from the repository root after completing the setup steps in the [quick start](../README.md#quick-start).

1. Detect ArUco marker:
   ```bash
   cd OrchestratorRUSS && ./run_aruco_detection.sh
   ```
2. Copy pose library (edit for your robot if needed):
   ```bash
   cp config/calib_poses.yaml.example config/calib_poses.yaml
   ```
   The example intentionally contains an empty `poses` list. Teach collision-free poses for your installation before running automatic sampling.
3. Run calibration:
   ```bash
   ./run_handeye_calibration.sh
   ```
   In RQt easy_handeye: sample poses → Compute → Save.

4. Save the result to `~/.ros/easy_handeye/iiwa_azure_eih_eye_on_hand.yaml`.

5. Bringup for scanning (run from the repository root):
   ```bash
   source workspace/devel/setup.bash
   roslaunch OrchestratorRUSS/launch/bringup_moveit_camera_publish_eye_on_hand.launch
   ```
Verify all taught poses are collision-free before automatic sampling.

## Configuration examples and hardware adaptation

- [Calibration launch](../OrchestratorRUSS/launch/handeye_calibrate_eye_on_hand.launch): integrates the robot, camera, ArUco detection, and easy_handeye.
- [Calibration result template](../OrchestratorRUSS/config/iiwa_azure_eih_eye_on_hand.yaml.example): shows the YAML structure and frame names. Its identity transform is a placeholder; use the result computed for your own installation.
- [Sampling pose template](../OrchestratorRUSS/config/calib_poses.yaml.example): provides the pose-library format. Populate it with poses taught on your own robot.

For a different robot or camera, adapt the robot driver and MoveIt configuration, the frame names and camera topics in the calibration and bringup launches, and the sampling poses. Recalibrate for the actual mounting arrangement. The scanning tools also use iiwa-specific control interfaces; those require adaptation when changing robot platforms. See [hardware setup](hardware.md).

**Offset note:** The open-source calibration launch uses `aruco_marker_frame` directly. The previous marker-thickness compensation frame was removed. Existing results computed with `aruco_marker_plane` are not compatible; recalibrate before use.
