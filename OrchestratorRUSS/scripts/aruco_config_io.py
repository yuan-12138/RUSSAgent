#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read/write ArUco marker config used by hand-eye calibration launch files."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_MARKER_SIZE_M = 0.05


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "aruco_marker.yaml"


def _parse_simple_yaml(text: str) -> Dict[str, object]:
    data: Dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not value:
            continue
        if value.lower() in ("true", "false"):
            data[key] = value.lower() == "true"
        else:
            try:
                if "." in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except ValueError:
                data[key] = value
    return data


def load_config(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError("ArUco config not found: {}".format(path))
    return _parse_simple_yaml(path.read_text(encoding="utf-8"))


def save_config(
    path: Path,
    marker_id: int,
    dictionary: str,
    marker_size_m: float = DEFAULT_MARKER_SIZE_M,
    all_detections: Optional[List[Dict[str, object]]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# Auto-generated ArUco marker settings for hand-eye calibration.",
        "# Used by run_handeye_calibration.sh and easy_handeye launch files.",
        "# Re-run ./run_aruco_detection.sh to refresh marker_id after changing markers.",
        "marker_id: {}".format(marker_id),
        "dictionary: {}".format(dictionary),
        "marker_size: {}".format(marker_size_m),
        "detected_at: {}".format(timestamp),
    ]

    if all_detections:
        lines.append("# Other detections in the same frame:")
        for det in all_detections:
            ids = det.get("ids", [])
            if not ids:
                continue
            id_list = ", ".join(str(int(x)) for x in ids)
            lines.append("#   {} -> [{}]".format(det["dictionary"], id_list))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def primary_detection(detections: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not detections:
        return None
    for det in detections:
        ids = det.get("ids", [])
        if ids:
            return {
                "marker_id": int(ids[0]),
                "dictionary": str(det["dictionary"]),
                "ids": [int(x) for x in ids],
            }
    return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read saved ArUco marker config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to aruco_marker.yaml",
    )
    parser.add_argument(
        "--key",
        default="",
        help="Print one field, e.g. marker_id or dictionary",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Run ./run_aruco_detection.sh first to detect and save the marker ID.",
            file=sys.stderr,
        )
        return 1

    if args.key:
        if args.key not in config:
            print("Key '{}' not found in {}".format(args.key, args.config), file=sys.stderr)
            return 1
        print(config[args.key])
        return 0

    for key in ("marker_id", "dictionary", "marker_size"):
        if key in config:
            print("{}: {}".format(key, config[key]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
