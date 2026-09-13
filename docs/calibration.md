# Hand-Eye Calibration

Uses easy_handeye installed by `scripts/setup.sh` with the RUSSAgent calibration launch.

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

5. Bringup for scanning:
   ```bash
   source workspace/devel/setup.bash
   roslaunch OrchestratorRUSS/launch/bringup_moveit_camera_publish_eye_on_hand.launch
   ```
Verify all taught poses are collision-free before automatic sampling.

**Offset note:** The open-source calibration launch uses `aruco_marker_frame` directly. The previous marker-thickness compensation frame was removed. Existing results computed with `aruco_marker_plane` are not compatible; recalibrate before use.
