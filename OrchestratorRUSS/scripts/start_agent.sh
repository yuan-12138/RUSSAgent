#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/russagent_env.sh"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
WS_SETUP="${RUSSAGENT_WS}/devel/setup.bash"
AGENT_PY="${SCRIPT_DIR}/run_agent.py"
if [[ -f "$CONDA_SH" ]]; then source "$CONDA_SH"; conda activate Russ_agent 2>/dev/null || true; fi
source "$ROS_SETUP" 2>/dev/null || echo "[warn] ROS not found"
source "$WS_SETUP" 2>/dev/null || echo "[warn] run catkin build"
: "${LLM_API_BASE:?Set LLM_API_BASE to your OpenAI-compatible API endpoint.}"
: "${LLM_CHAT_MODEL:?Set LLM_CHAT_MODEL to the model name served by your endpoint.}"
export LLM_API_BASE
export LLM_CHAT_MODEL
export RUSSAGENT_ENABLE_ROBOT="${RUSSAGENT_ENABLE_ROBOT:-0}"
export ASA_PROJECTION_PYTHON="${ASA_PROJECTION_PYTHON:-$(which python)}"
mkdir -p "${RUSSAGENT_ROS_DIR}"
exec python "$AGENT_PY" "$@"
