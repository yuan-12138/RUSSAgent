#!/usr/bin/env bash
# Launch eye-on-hand calibration and mirror the saved matrix into OrchestratorRUSS/config/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/russagent_env.sh"
CONFIG_DIR="${SCRIPT_DIR}/config"
WORKSPACE_SETUP="${RUSSAGENT_WS}/devel/setup.bash"
ARUCO_CONFIG_IO="${SCRIPT_DIR}/scripts/aruco_config_io.py"

ARUCO_CONFIG="${CONFIG_DIR}/aruco_marker.yaml"
CALIB_POSES="${CONFIG_DIR}/calib_poses.yaml"
NAMESPACE_PREFIX="${NAMESPACE_PREFIX:-iiwa_azure_eih}"

# Lightweight camera defaults to avoid USB-bandwidth "Failed to poll cameras".
# rgb_camera_link TF comes from the calibration blob, so a low depth mode/fps is fine.
DEPTH_MODE="${DEPTH_MODE:-NFOV_2X2BINNED}"
COLOR_RESOLUTION="${COLOR_RESOLUTION:-720P}"
FPS="${FPS:-15}"
CALIB_NAMESPACE="${NAMESPACE_PREFIX}_eye_on_hand"
SOURCE_CALIB="${HOME}/.ros/easy_handeye/${CALIB_NAMESPACE}.yaml"
DEST_CALIB="${CONFIG_DIR}/${CALIB_NAMESPACE}.yaml"

WATCHER_PID=""
LAST_COPIED_MTIME=""

usage() {
  cat <<EOF
Usage:
  ./run_handeye_calibration.sh [options] [-- extra roslaunch args]

Options:
  --aruco-config PATH   ArUco marker yaml (default: config/aruco_marker.yaml)
  --poses PATH          Robot pose library for auto-sampling (default: config/calib_poses.yaml)
  --namespace-prefix P  easy_handeye namespace prefix (default: iiwa_azure_eih)
  --clean-camera        Stop any running Azure Kinect driver before launch
  --with-pointcloud     Also generate point cloud (default: off, not needed)
  -h, --help            Show this help

Environment overrides (lightweight camera defaults to avoid USB overload):
  DEPTH_MODE        default NFOV_2X2BINNED  (lightest depth that keeps TF valid)
  COLOR_RESOLUTION  default 720P
  FPS               default 15

Notes:
  Depth is always enabled. The Azure Kinect derives rgb_camera_link from the
  depth calibration; disabling depth yields an invalid TF quaternion and breaks
  the camera_base -> rgb_camera_link chain that ArUco/easy_handeye rely on.
  If you still hit 'Failed to poll cameras', lower the load further, e.g.:
    FPS=5 ./run_handeye_calibration.sh --clean-camera
  and make sure the Kinect is on a dedicated USB 3.0 port (not a hub).

Workflow:
  1. Reads marker_id and marker_size from config/aruco_marker.yaml
     (run ./run_aruco_detection.sh first if that file is missing).
  2. Starts handeye_calibrate_eye_on_hand.launch.
  3. In the RQt easy_handeye GUI: take samples, compute, then click Save.
  4. Whenever the calibration yaml is saved, it is copied to:
     ${CONFIG_DIR}/

Examples:
  ./run_handeye_calibration.sh
  ./run_handeye_calibration.sh -- auto_sample:=true
EOF
}

read_config_key() {
  python3 "${ARUCO_CONFIG_IO}" --config "${ARUCO_CONFIG}" --key "$1"
}

copy_calibration_if_updated() {
  if [[ ! -f "${SOURCE_CALIB}" ]]; then
    return 0
  fi

  local mtime
  mtime="$(stat -c %Y "${SOURCE_CALIB}")"
  if [[ "${mtime}" == "${LAST_COPIED_MTIME}" ]]; then
    return 0
  fi

  mkdir -p "${CONFIG_DIR}"
  cp -a "${SOURCE_CALIB}" "${DEST_CALIB}"
  LAST_COPIED_MTIME="${mtime}"
  echo "Copied hand-eye calibration to ${DEST_CALIB}"
}

watch_calibration_copy() {
  while true; do
    copy_calibration_if_updated || true
    sleep 2
  done
}

cleanup() {
  if [[ -n "${WATCHER_PID}" ]] && kill -0 "${WATCHER_PID}" 2>/dev/null; then
    kill "${WATCHER_PID}" 2>/dev/null || true
    wait "${WATCHER_PID}" 2>/dev/null || true
  fi
  copy_calibration_if_updated || true
}
trap cleanup EXIT INT TERM

CLEAN_CAMERA=0
WITH_POINTCLOUD=0
LAUNCH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aruco-config)
      ARUCO_CONFIG="$2"
      shift 2
      ;;
    --poses)
      CALIB_POSES="$2"
      shift 2
      ;;
    --namespace-prefix)
      NAMESPACE_PREFIX="$2"
      CALIB_NAMESPACE="${NAMESPACE_PREFIX}_eye_on_hand"
      SOURCE_CALIB="${HOME}/.ros/easy_handeye/${CALIB_NAMESPACE}.yaml"
      DEST_CALIB="${CONFIG_DIR}/${CALIB_NAMESPACE}.yaml"
      shift 2
      ;;
    --clean-camera)
      CLEAN_CAMERA=1
      shift
      ;;
    --with-pointcloud)
      WITH_POINTCLOUD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      LAUNCH_ARGS+=("$@")
      break
      ;;
    *)
      LAUNCH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${WORKSPACE_SETUP}" ]]; then
  echo "Workspace setup not found: ${WORKSPACE_SETUP}" >&2
  echo "Build the RUSSAgent workspace first." >&2
  exit 1
fi

if [[ ! -f "${ARUCO_CONFIG}" ]]; then
  echo "ArUco config not found: ${ARUCO_CONFIG}" >&2
  echo "Run ./run_aruco_detection.sh first to detect and save the marker settings." >&2
  exit 1
fi

if [[ ! -f "${CALIB_POSES}" ]]; then
  echo "Pose library not found: ${CALIB_POSES}" >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_SETUP}"

kinect_driver_running() {
  pgrep -f 'azure_kinect_ros_driver|lib/azure_kinect_ros_driver/node' >/dev/null 2>&1
}

stop_kinect_driver() {
  if ! kinect_driver_running; then
    return 0
  fi
  echo "Stopping existing Azure Kinect driver..."
  pkill -f 'lib/azure_kinect_ros_driver/node' 2>/dev/null || true
  pkill -f 'azure_kinect_ros_driver' 2>/dev/null || true
  sleep 2
}

preflight_camera() {
  if [[ "${CLEAN_CAMERA}" -eq 1 ]]; then
    stop_kinect_driver
    return 0
  fi

  if kinect_driver_running || rostopic list 2>/dev/null | grep -q '^/rgb/image_raw$'; then
    echo "WARNING: Azure Kinect driver or /rgb/image_raw already active." >&2
    echo "A second driver causes 'Failed to start cameras' (USB busy)." >&2
    echo "Stop the other session (Ctrl+C) or re-run with: --clean-camera" >&2
    exit 1
  fi
}

preflight_camera

MARKER_ID="$(read_config_key marker_id)"
MARKER_SIZE="$(read_config_key marker_size)"

# Depth must stay enabled: rgb_camera_link/imu_link TF are derived from the
# depth calibration; disabling depth produces invalid (0.5,0,0,0) quaternions.
# Point cloud is off by default (not needed for calibration, lowers CPU/USB load).
CAMERA_ARGS=(
  depth_enabled:=true
  depth_mode:="${DEPTH_MODE}"
  color_resolution:="${COLOR_RESOLUTION}"
  fps:="${FPS}"
  point_cloud:=false
  rgb_point_cloud:=false
)
if [[ "${WITH_POINTCLOUD}" -eq 1 ]]; then
  CAMERA_ARGS=(
    depth_enabled:=true
    depth_mode:="${DEPTH_MODE}"
    color_resolution:="${COLOR_RESOLUTION}"
    fps:="${FPS}"
    point_cloud:=true
    rgb_point_cloud:=true
  )
fi

echo "ArUco config: ${ARUCO_CONFIG}"
echo "  marker_id=${MARKER_ID}, marker_size=${MARKER_SIZE}"
echo "Camera: depth_mode=${DEPTH_MODE}, color=${COLOR_RESOLUTION}, fps=${FPS}, point_cloud=$([[ ${WITH_POINTCLOUD} -eq 1 ]] && echo on || echo off)"
echo "Pose library: ${CALIB_POSES}"
echo "Calibration will be mirrored to: ${DEST_CALIB}"
echo
echo "In RQt easy_handeye: sample poses, compute calibration, then click Save."
echo "Press Ctrl+C here when finished."
echo

watch_calibration_copy &
WATCHER_PID=$!

roslaunch "${RUSSAGENT_ORCH}/launch/handeye_calibrate_eye_on_hand.launch" \
  marker_id:="${MARKER_ID}" \
  marker_size:="${MARKER_SIZE}" \
  yaml_path:="${CALIB_POSES}" \
  namespace_prefix:="${NAMESPACE_PREFIX}" \
  "${CAMERA_ARGS[@]}" \
  "${LAUNCH_ARGS[@]}"
