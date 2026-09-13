#!/usr/bin/env bash
# Start Azure Kinect RGB stream and detect ArUco marker IDs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/russagent_env.sh"
WORKSPACE_SETUP="${RUSSAGENT_WS}/devel/setup.bash"

if [[ ! -f "${WORKSPACE_SETUP}" ]]; then
  echo "Workspace setup not found: ${WORKSPACE_SETUP}" >&2
  echo "Build the RUSSAgent workspace first." >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_SETUP}"

MODE="once"
EXTRA_ARGS=()
CAMERA_PID=""

cleanup() {
  if [[ -n "${CAMERA_PID}" ]] && kill -0 "${CAMERA_PID}" 2>/dev/null; then
    kill "${CAMERA_PID}" 2>/dev/null || true
    wait "${CAMERA_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

usage() {
  cat <<'EOF'
Usage:
  ./run_aruco_detection.sh [options] [-- extra python args]

Options:
  --continuous   Keep running until you press q in the preview window
  --no-camera    Do not start the Kinect driver (assume it is already running)
  -h, --help     Show this help

Examples:
  ./run_aruco_detection.sh
  ./run_aruco_detection.sh -- --dictionary DICT_ARUCO_ORIGINAL --save marker.jpg
  ./run_aruco_detection.sh --continuous
EOF
}

START_CAMERA=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --continuous)
      MODE="continuous"
      shift
      ;;
    --no-camera)
      START_CAMERA=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${START_CAMERA}" -eq 1 ]]; then
  echo "Starting Azure Kinect driver..."
  roslaunch azure_kinect_ros_driver driver.launch \
    color_enabled:=true \
    depth_enabled:=false \
    point_cloud:=false \
    rgb_point_cloud:=false \
    color_resolution:=720P \
    fps:=30 \
    required:=true &
  CAMERA_PID=$!

  echo "Waiting for camera topics..."
  for _ in $(seq 1 30); do
    if rostopic list 2>/dev/null | grep -q '^/rgb/image_raw$'; then
      break
    fi
    sleep 1
  done
fi

CONFIG_FILE="${SCRIPT_DIR}/config/aruco_marker.yaml"

PY_ARGS=()
if [[ "${MODE}" == "once" ]]; then
  PY_ARGS+=(--once)
fi
PY_ARGS+=(--config-out "${CONFIG_FILE}")
PY_ARGS+=("${EXTRA_ARGS[@]}")

python3 "${SCRIPT_DIR}/scripts/capture_aruco_from_azure_kinect.py" "${PY_ARGS[@]}"
