#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from difflib import get_close_matches
from typing import Any, Dict

import rospy

from .tool_base import BaseTool
from ..reasoning_tts import speak_guidance_sync

_PENDING_EXECUTE_PARAM = "/autonomous_scan_agent/point0_pending_execute"
_PENDING_CONTACT_PARAM = "/autonomous_scan_agent/point0_pending_contact"
_PENDING_YAML_PARAM = "/autonomous_scan_agent/point0_pending_yaml"


def _set_guard(pending_execute: bool = False, pending_contact: bool = False) -> None:
    try:
        rospy.set_param(_PENDING_EXECUTE_PARAM, bool(pending_execute))
        rospy.set_param(_PENDING_CONTACT_PARAM, bool(pending_contact))
    except Exception:
        pass


def _get_guard() -> Dict[str, Any]:
    try:
        return {
            "pending_execute": bool(rospy.get_param(_PENDING_EXECUTE_PARAM, False)),
            "pending_contact": bool(rospy.get_param(_PENDING_CONTACT_PARAM, False)),
            "pending_yaml": str(rospy.get_param(_PENDING_YAML_PARAM, "")),
        }
    except Exception:
        return {"pending_execute": False, "pending_contact": False, "pending_yaml": ""}


class ContactQueryTool(BaseTool):
    """
    Ask operator to assess probe contact quality at current pose.
    Normalizes free-text into {good, poor, none}.
    """

    def __init__(self):
        super().__init__(
            name="contact_query_tool",
            description="Ask operator to assess contact quality (good/poor/none). Returns normalized contact_quality.",
        )

    def execute(self, prompt: str = "Contact quality? (good/poor/none)", **kwargs) -> Dict[str, Any]:
        _ = kwargs
        p = (prompt or "Contact quality? (good/poor/none)").strip()

        # Tool-level guardrail (v3-style): do not allow contact query before executing point0.
        g = _get_guard()
        if g.get("pending_execute") is True:
            msg = "must_execute_point0_next"
            hint = 'Point0 was exported but not executed. Call path_execute_server_tool(path_id="~/.ros/path_point0.yaml") first, then ask contact.'
            try:
                import rospy  # already imported, but keep safe
                rospy.logwarn("[contact_query_tool] %s: %s", msg, hint)
            except Exception:
                pass
            return {
                "status": "failed",
                "message": msg,
                "hint": hint,
                "pending_yaml": g.get("pending_yaml", ""),
            }
        try:
            print(f"\n[contact_query_tool] {p}", flush=True)
            speak_guidance_sync(
                "Please check probe contact quality on the ultrasound image. "
                "Type good if contact is good, poor if contact is weak, or none if there is no contact."
            )
            ans = input("> ").strip()
        except Exception as e:
            return {"status": "error", "message": str(e), "contact_quality": "unknown", "answer": None}

        q = self._normalize(ans)
        if q not in ("good", "poor", "none"):
            return {
                "status": "need_user",
                "message": "unrecognized_contact",
                "answer": ans,
                "contact_quality": "unknown",
                "hint": "Please answer one of: good / poor / none (free-text is ok, e.g., 'its good', 'not so good', 'no contact').",
            }
        # Contact answered -> clear "pending contact" so execute can proceed / correction can happen.
        _set_guard(pending_execute=False, pending_contact=False)
        return {"status": "success", "answer": ans, "contact_quality": q}

    def _normalize(self, text: str) -> str:
        s = (text or "").strip().lower()
        if not s:
            return "unknown"

        # explicit tokens - expanded
        if any(x in s for x in ["none", "no contact", "no touch", "black", "miss", "lost", "gap"]):
            return "none"
        if any(x in s for x in ["poor", "bad", "not good", "not so good", "a little", "weak", "light"]):
            return "poor"
        if any(x in s for x in ["good", "perfect", "great", "nice", "ok", "fine", "well", "excellent"]):
            return "good"

        # Chinese common - expanded
        if any(x in text for x in ["没接触", "无接触", "没碰到", "黑的", "空的", "没贴上"]):
            return "none"
        if any(x in text for x in ["不好", "差", "不太好", "一般", "有点差", "轻", "虚"]):
            return "poor"
        if any(x in text for x in ["好", "很好", "可以", "完美", "不错", "行", "贴上了"]):
            return "good"

        # fuzzy token match
        tokens = re.findall(r"[a-z]+", s)
        vocab = ["good", "poor", "none"]
        for t in tokens:
            m = get_close_matches(t, vocab, n=1, cutoff=0.75)
            if m:
                return m[0]
        return "unknown"

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": []}

