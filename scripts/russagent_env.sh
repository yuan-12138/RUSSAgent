#!/usr/bin/env bash
# Source from other scripts: source "$(dirname "$0")/russagent_env.sh"
if [[ -z "${RUSSAGENT_ROOT:-}" ]]; then
  if [[ -n "${BASH_VERSION:-}" ]]; then
    _RUSSAGENT_ENV_SOURCE="${BASH_SOURCE[0]}"
  elif [[ -n "${ZSH_VERSION:-}" ]]; then
    _RUSSAGENT_ENV_SOURCE="${(%):-%N}"
  else
    _RUSSAGENT_ENV_SOURCE="$0"
  fi
  _SCRIPT_DIR="$(cd "$(dirname "${_RUSSAGENT_ENV_SOURCE}")" && pwd)"
  export RUSSAGENT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
  unset _RUSSAGENT_ENV_SOURCE
fi
export RUSSAGENT_WS="${RUSSAGENT_ROOT}/workspace"
export RUSSAGENT_ASA_PKG="${RUSSAGENT_WS}/src/autonomous_scan_agent"
export RUSSAGENT_ORCH="${RUSSAGENT_ROOT}/OrchestratorRUSS"
export RUSSAGENT_CONFIG="${RUSSAGENT_ORCH}/config"
export RUSSAGENT_ROS_DIR="${HOME}/.ros/russagent"
export RUSSAGENT_PATH_PREVIEW="${RUSSAGENT_ROS_DIR}/path_preview1.yaml"
