#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict
import time
import rospy

from .tool_base import BaseTool


class OperatorIOTool(BaseTool):
    """
    Tool0：操作者交互输入/观测采集（文本版“口头指令”与超声反馈）。
    现在支持：
    1. 提问并等待回答 (expect_response=True)
    2. 仅说话/提示 (expect_response=False)
    3. 说话后等待几秒 (sleep_sec > 0)
    """

    def __init__(self):
        super().__init__(
            name="operator_io_tool",
            description="Ask operator for observation or give instructions. Can wait for input or sleep.",
        )

    def execute(self, prompt: str = "", say: str = "", **kwargs) -> Dict[str, Any]:
        # 兼容参数：say 或 prompt
        text = say if say else prompt
        audience = str(kwargs.get("audience", "operator")).strip().lower()
        if audience not in ("operator", "patient"):
            audience = "operator"
        
        # 打印提示
        if text:
            if audience == "patient":
                print("\n" + "=" * 60)
                print("PATIENT INSTRUCTION (please say this to the patient):")
                print(text)
                print("=" * 60)
            else:
                print(f"\n[operator_io_tool] {text}")
        
        # 是否需要等待用户输入？默认 True 如果没有指定 sleep
        expect_response = bool(kwargs.get("expect_response", True))
        sleep_sec = float(kwargs.get("sleep_sec", 0.0))

        # 如果指定了 sleep_sec，通常不需要用户输入（除非显式要求）
        # 逻辑：如果有 sleep，默认不等待输入，除非强制 expect_response=True
        if "sleep_sec" in kwargs and "expect_response" not in kwargs:
            expect_response = False

        # For patient-facing instructions, ALWAYS require an explicit 'ok' confirmation.
        if audience == "patient":
            expect_response = True

        ans = ""
        try:
            if expect_response:
                if text:
                    if audience == "patient":
                        print("When the patient has completed it, type 'ok'.")
                    else:
                        print("Please type 'ok' when completed, or type your reply.")
                ans = input("> ").strip()
            elif sleep_sec > 0:
                rospy.loginfo(f"[operator_io_tool] Sleeping for {sleep_sec}s...")
                time.sleep(sleep_sec)
        except Exception as e:
            return {"status": "failed", "message": str(e), "answer": None, "nl_observation": f"operator_io_tool execution failed: {e}"}
            
        # Enrich the output for LLM readability (keep existing fields for compatibility)
        meaning = "User provided input." if ans else "No input provided."
        if "task" in (text or "").lower() and ans:
            meaning = f"TASK IDENTIFIED: {ans}"
        if "hold" in (text or "").lower() and ("ok" in (ans or "").lower() or "done" in (ans or "").lower()):
            meaning = "BREATH HOLD CONFIRMED."

        # Match sim_nl_standalone OperatorIOTool nl_observation as closely as possible.
        if (ans or "").strip().lower() == "ok" and (text or "").strip():
            if audience == "patient":
                nl_obs = f"Clinician confirmed patient instruction completed: {text}"
            else:
                nl_obs = f"Instruction confirmed: {text}"
        elif ans:
            nl_obs = f"Operator replied: '{ans}'."
        else:
            nl_obs = "Operator interaction completed."

        return {
            "status": "success",
            "answer": ans,
            "interpretation": meaning,
            "nl_observation": nl_obs,
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Text to display to operator (alias of say)"},
            "say": {"type": "string", "description": "Text to display to operator"},
            "audience": {
                "type": "string",
                "enum": ["operator", "patient"],
                "description": "Who is the instruction for (operator or patient).",
            },
            "expect_response": {"type": "boolean", "description": "Wait for user input?"},
            "sleep_sec": {"type": "number", "description": "Time to sleep in seconds (optional)"},
        },
        "required": [],
    }
        

