#!/usr/bin/env bash
# Install external source dependencies, create the Python environment, and build the ROS workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/russagent_env.sh"

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] Missing required file: $1" >&2
    exit 1
  fi
}

clone_at_commit() {
  local name="$1"
  local url="$2"
  local path="$3"
  local commit="$4"

  if [[ -d "${path}/.git" ]]; then
    echo "[ok] ${name}: ${path}"
    return
  fi
  if [[ -e "$path" ]]; then
    echo "[error] ${path} exists but is not a Git checkout." >&2
    echo "        Move it away, then rerun this script." >&2
    exit 1
  fi

  echo "[clone] ${name}"
  git clone "$url" "$path"
  git -C "$path" checkout "$commit"
}

apply_dependency_patch() {
  local name="$1"
  local path="$2"
  local patch="$3"

  if git -C "$path" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "[ok] ${name} patch already applied"
  elif git -C "$path" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$path" apply "$patch"
    echo "[patch] ${name}"
  elif ! git -C "$path" diff --quiet; then
    echo "[warn] ${name} has local modifications; keeping them unchanged"
  else
    echo "[error] ${name} patch does not apply to the tested commit" >&2
    exit 1
  fi
}

require_file "$ROS_SETUP"
require_file "$CONDA_SH"

clone_at_commit \
  "SKEL" \
  "https://github.com/MarilynKeller/SKEL.git" \
  "${RUSSAGENT_WS}/models/SKEL" \
  "c32cf16"

clone_at_commit \
  "easy_handeye" \
  "https://github.com/IFL-CAMP/easy_handeye.git" \
  "${RUSSAGENT_WS}/src/easy_handeye" \
  "ffcc43d"

clone_at_commit \
  "Azure Kinect ROS Driver" \
  "https://github.com/microsoft/Azure_Kinect_ROS_Driver.git" \
  "${RUSSAGENT_WS}/src/Azure_Kinect_ROS_Driver" \
  "62c7406"

apply_dependency_patch \
  "SKEL headless integration" \
  "${RUSSAGENT_WS}/models/SKEL" \
  "${SCRIPT_DIR}/patches/skel-headless.patch"

apply_dependency_patch \
  "Azure Kinect SDK version compatibility" \
  "${RUSSAGENT_WS}/src/Azure_Kinect_ROS_Driver" \
  "${SCRIPT_DIR}/patches/azure-kinect-version.patch"

YOLO_DIR="${RUSSAGENT_ASA_PKG}/src/autonomous_scan_agent/tools/cliff_repo/lib/pytorch_yolo_v3_master"
clone_at_commit \
  "pytorch-yolo-v3" \
  "https://github.com/ayooshkathuria/pytorch-yolo-v3.git" \
  "$YOLO_DIR" \
  "fbb4ef9"

# shellcheck disable=SC1090
source "$CONDA_SH"
if conda env list | awk '{print $1}' | grep -qx "Russ_agent"; then
  conda env update -n Russ_agent -f "${RUSSAGENT_ROOT}/environment.yml" --prune
else
  conda env create -f "${RUSSAGENT_ROOT}/environment.yml"
fi

# shellcheck disable=SC1090
source "$ROS_SETUP"
cd "$RUSSAGENT_WS"
catkin build

echo
echo "Source dependencies and ROS workspace are ready."
echo "Download the model files and complete calibration as described in README.md."
