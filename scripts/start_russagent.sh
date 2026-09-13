#!/usr/bin/env bash
# Start the complete RUSSAgent stack: robot, Azure Kinect, hand-eye TF, and LLM agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/russagent_env.sh"

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
WS_SETUP="${RUSSAGENT_WS}/devel/setup.bash"
CALIBRATION_FILE="${RUSSAGENT_CALIBRATION_FILE:-$HOME/.ros/easy_handeye/iiwa_azure_eih_eye_on_hand.yaml}"
CAPTURE_POSES_FILE="${RUSSAGENT_CAPTURE_POSES_FILE:-$HOME/.ros/autonomous_scan_agent/capture_poses.yaml}"
BRINGUP_LAUNCH="${RUSSAGENT_ORCH}/launch/bringup_moveit_camera_publish_eye_on_hand.launch"
BRINGUP_PID=""

cleanup() {
  if [[ -n "$BRINGUP_PID" ]] && kill -0 "$BRINGUP_PID" 2>/dev/null; then
    kill "$BRINGUP_PID" 2>/dev/null || true
    wait "$BRINGUP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[error] Missing required directory: $1" >&2
    exit 1
  fi
}

require_file "$ROS_SETUP"
require_file "$CONDA_SH"
require_file "$WS_SETUP"
require_file "$CALIBRATION_FILE"
require_file "$CAPTURE_POSES_FILE"
require_file "$BRINGUP_LAUNCH"
require_dir "${RUSSAGENT_WS}/models/SKEL"
require_dir "${RUSSAGENT_WS}/src/easy_handeye"
require_dir "${RUSSAGENT_WS}/src/Azure_Kinect_ROS_Driver"

CLIFF_DATA="${RUSSAGENT_ASA_PKG}/src/autonomous_scan_agent/tools/cliff_repo/data"
require_file "${CLIFF_DATA}/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"
require_file "${CLIFF_DATA}/ckpt/yolov3.weights"
require_file "${CLIFF_DATA}/smpl_mean_params.npz"
require_file "${CLIFF_DATA}/smpl/SMPL_NEUTRAL.pkl"
require_file "${RUSSAGENT_WS}/models/SKEL/data/skel/skel_male.pkl"

if [[ "${RUSSAGENT_ENABLE_ROBOT:-0}" != "1" ]]; then
  echo "RUSSAgent will control a real KUKA IIWA robot."
  echo "Confirm that calibration, collision checking, low-speed validation, and emergency stop are ready."
  read -r -p "Type START to continue: " confirmation
  if [[ "$confirmation" != "START" ]]; then
    echo "Cancelled."
    exit 1
  fi
  export RUSSAGENT_ENABLE_ROBOT=1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate Russ_agent
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$WS_SETUP"

if [[ -z "${LLM_API_BASE:-}" ]]; then
  read -r -p "OpenAI-compatible API endpoint: " LLM_API_BASE
fi
if [[ -z "${LLM_CHAT_MODEL:-}" ]]; then
  read -r -p "Model name: " LLM_CHAT_MODEL
fi
if [[ -z "$LLM_API_BASE" || -z "$LLM_CHAT_MODEL" ]]; then
  echo "[error] API endpoint and model name are required." >&2
  exit 1
fi
export LLM_API_BASE
export LLM_CHAT_MODEL
export RUSSAGENT_CALIBRATION_FILE="$CALIBRATION_FILE"

echo "[start] Robot, Azure Kinect, MoveIt, and hand-eye TF"
roslaunch "$BRINGUP_LAUNCH" calibration_file:="$CALIBRATION_FILE" &
BRINGUP_PID=$!

echo "[wait] Waiting for MoveIt and Azure Kinect topics..."
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$BRINGUP_PID" 2>/dev/null; then
    echo "[error] Bringup process exited before becoming ready." >&2
    exit 1
  fi
  if rosnode list 2>/dev/null | grep -q "/move_group" \
     && rostopic list 2>/dev/null | grep -q "^/rgb/image_raw$"; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "[error] Bringup did not become ready within 120 seconds." >&2
  exit 1
fi

echo "[start] RUSSAgent"
bash "${RUSSAGENT_ORCH}/scripts/start_agent.sh" "$@"
