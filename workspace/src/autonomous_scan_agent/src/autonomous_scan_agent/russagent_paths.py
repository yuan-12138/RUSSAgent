"""Resolve RUSSAgent repository paths from RUSSAGENT_ROOT or infer from package location."""
from __future__ import annotations

import os


def repo_root() -> str:
    env = os.environ.get("RUSSAGENT_ROOT", "").strip()
    if env:
        return os.path.abspath(env)
    # .../workspace/src/autonomous_scan_agent/src/autonomous_scan_agent/russagent_paths.py
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))


def workspace_root() -> str:
    return os.path.join(repo_root(), "workspace")


def asa_pkg_root() -> str:
    return os.path.join(workspace_root(), "src", "autonomous_scan_agent")


def orchestrator_root() -> str:
    return os.path.join(repo_root(), "OrchestratorRUSS")


def ros_state_dir() -> str:
    return os.environ.get("RUSSAGENT_ROS_DIR", os.path.join(os.path.expanduser("~"), ".ros", "russagent"))


def path_preview_yaml() -> str:
    return os.environ.get("RUSSAGENT_PATH_PREVIEW", os.path.join(ros_state_dir(), "path_preview1.yaml"))
