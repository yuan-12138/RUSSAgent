#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture Azure Kinect images and detect visible ArUco marker IDs.

Requires a sourced ROS Noetic workspace and a running Azure Kinect driver.
Use ``--once`` for one snapshot or run without it for a continuous preview.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from aruco_config_io import (
    default_config_path,
    primary_detection,
    save_config,
)

DICT_CHOICES: Dict[str, int] = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_MIP_36h12": cv2.aruco.DICT_ARUCO_MIP_36h12,
}

DEFAULT_TRY_ORDER: List[str] = [
    "DICT_ARUCO_ORIGINAL",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_5X5_1000",
    "DICT_4X4_1000",
    "DICT_7X7_1000",
    "DICT_ARUCO_MIP_36h12",
    "DICT_6X6_100",
    "DICT_5X5_250",
    "DICT_4X4_250",
]


def _make_detector(dict_id: int):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary)
    return dictionary


def _detect(gray: np.ndarray, detector) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _rejected = detector.detectMarkers(gray)
        return corners, ids
    corners, ids, _rejected = cv2.aruco.detectMarkers(gray, detector)
    return corners, ids


def detect_markers(
    bgr: np.ndarray,
    dict_names: List[str],
) -> List[Dict[str, object]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    results: List[Dict[str, object]] = []

    for name in dict_names:
        if name not in DICT_CHOICES:
            continue
        detector = _make_detector(DICT_CHOICES[name])
        corners, ids = _detect(gray, detector)
        if ids is None or len(ids) == 0:
            continue
        flat_ids = [int(x) for x in ids.flatten().tolist()]
        results.append(
            {
                "dictionary": name,
                "ids": flat_ids,
                "corners": corners,
                "marker_ids": ids,
            }
        )
    return results


def draw_detections(
    bgr: np.ndarray,
    detections: List[Dict[str, object]],
) -> np.ndarray:
    out = bgr.copy()
    y = 28
    if not detections:
        cv2.putText(
            out,
            "No ArUco markers detected",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    for det in detections:
        corners = det["corners"]
        ids = det["marker_ids"]
        dict_name = str(det["dictionary"])
        flat_ids = det["ids"]
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        label = "{}: {}".format(dict_name, flat_ids)
        cv2.putText(
            out,
            label,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return out


def print_detections(detections: List[Dict[str, object]]) -> None:
    if not detections:
        print("No ArUco markers detected in the captured image.")
        return

    print("Detected ArUco markers:")
    for det in detections:
        print("  dictionary={}  ids={}".format(det["dictionary"], det["ids"]))

    primary = detections[0]
    if len(primary["ids"]) == 1:
        print(
            "Primary marker: id={} (dictionary={})".format(
                primary["ids"][0], primary["dictionary"]
            )
        )


class AzureKinectImageSource:
    def __init__(self, rgb_topic: str, timeout_sec: float) -> None:
        self.bridge = CvBridge()
        self.rgb_topic = rgb_topic
        self.timeout_sec = timeout_sec
        self._latest_bgr: Optional[np.ndarray] = None
        self._sub = rospy.Subscriber(rgb_topic, Image, self._on_image, queue_size=1)

    def _on_image(self, msg: Image) -> None:
        try:
            self._latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "cv_bridge failed: %s", str(exc))

    def wait_for_frame(self) -> np.ndarray:
        deadline = time.time() + self.timeout_sec
        rate = rospy.Rate(30.0)
        while not rospy.is_shutdown():
            if self._latest_bgr is not None:
                return self._latest_bgr.copy()
            if time.time() > deadline:
                raise RuntimeError(
                    "No image received on {} within {:.1f}s. "
                    "Is the Azure Kinect driver running?".format(
                        self.rgb_topic, self.timeout_sec
                    )
                )
            rate.sleep()
        raise RuntimeError("ROS shutdown before an image was received.")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Azure Kinect RGB frames and detect ArUco marker IDs."
    )
    parser.add_argument(
        "--rgb-topic",
        default="/rgb/image_raw",
        help="ROS image topic from azure_kinect_ros_driver (default: /rgb/image_raw)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the first camera frame (default: 15)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one frame, print results, then exit",
    )
    parser.add_argument(
        "--dictionary",
        default="",
        help="Only use one ArUco dictionary, e.g. DICT_ARUCO_ORIGINAL",
    )
    parser.add_argument(
        "--try-all-dictionaries",
        action="store_true",
        help="Try every predefined OpenCV ArUco dictionary (slower)",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Path to save the annotated image (e.g. output.jpg)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print detection results as JSON on stdout",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV preview window",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=default_config_path(),
        help="Save detected marker_id/dictionary for hand-eye calibration (default: config/aruco_marker.yaml)",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Do not write the calibration config file",
    )
    return parser.parse_args(argv)


def _maybe_save_config(
    args: argparse.Namespace,
    detections: List[Dict[str, object]],
) -> None:
    if args.no_config:
        return

    primary = primary_detection(detections)
    if primary is None:
        print("No marker detected; calibration config not updated.")
        return

    save_config(
        args.config_out,
        marker_id=primary["marker_id"],
        dictionary=primary["dictionary"],
        all_detections=[
            {"dictionary": d["dictionary"], "ids": d["ids"]} for d in detections
        ],
    )
    print(
        "Saved calibration config to {} (marker_id={}, dictionary={})".format(
            args.config_out, primary["marker_id"], primary["dictionary"]
        )
    )


def resolve_dictionary_list(args: argparse.Namespace) -> List[str]:
    if args.dictionary:
        if args.dictionary not in DICT_CHOICES:
            valid = ", ".join(sorted(DICT_CHOICES.keys())[:5])
            raise ValueError(
                "Unknown dictionary '{}'. Examples: {}".format(args.dictionary, valid)
            )
        return [args.dictionary]
    if args.try_all_dictionaries:
        return list(DICT_CHOICES.keys())
    return list(DEFAULT_TRY_ORDER)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dict_names = resolve_dictionary_list(args)

    rospy.init_node("capture_aruco_from_azure_kinect", anonymous=True)
    source = AzureKinectImageSource(args.rgb_topic, args.timeout)

    rospy.loginfo("Waiting for camera frames on %s", args.rgb_topic)

    show_gui = not args.no_gui
    window_name = "Azure Kinect ArUco Detection"

    try:
        if args.once:
            frame = source.wait_for_frame()
            detections = detect_markers(frame, dict_names)
            annotated = draw_detections(frame, detections)

            if args.json:
                payload = [
                    {"dictionary": d["dictionary"], "ids": d["ids"]} for d in detections
                ]
                print(json.dumps(payload, indent=2))
            else:
                print_detections(detections)

            if args.save:
                cv2.imwrite(args.save, annotated)
                print("Saved annotated image to {}".format(args.save))

            _maybe_save_config(args, detections)

            if show_gui:
                cv2.imshow(window_name, annotated)
                print("Press any key in the preview window to exit.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return 0

        while not rospy.is_shutdown():
            frame = source.wait_for_frame()
            detections = detect_markers(frame, dict_names)
            annotated = draw_detections(frame, detections)

            if show_gui:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    print_detections(detections)
                    if args.save:
                        cv2.imwrite(args.save, annotated)
                        print("Saved annotated image to {}".format(args.save))
                    _maybe_save_config(args, detections)
            else:
                print_detections(detections)
                time.sleep(0.5)

        if show_gui:
            cv2.destroyAllWindows()
        return 0

    except (RuntimeError, ValueError) as exc:
        rospy.logerr("%s", str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
