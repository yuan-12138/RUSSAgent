#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_agent.py

Natural Language Driven RAG Agent for Ultrasound Scanning.

Refactored to:
1. Use a purely natural language handbook (txt).
2. Use RAG (TF-IDF/Cosine) to retrieve handbook guidance and relevant tools.
3. Eliminate structured 'env_state' in favor of natural language observations.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import rospy
import numpy as np
from datetime import datetime
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.environ.get("RUSSAGENT_ROOT") or os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
_HAMLYN_ASA_PKG = os.path.join(_REPO_ROOT, "workspace", "src", "autonomous_scan_agent")
PKG_SRC = os.path.join(_HAMLYN_ASA_PKG, "src")
if PKG_SRC not in sys.path:
    sys.path.insert(0, PKG_SRC)

from autonomous_scan_agent.russagent_paths import asa_pkg_root, orchestrator_root, repo_root, path_preview_yaml  # noqa: E402

from autonomous_scan_agent.llm_client import LlamaClient
from autonomous_scan_agent.llm_interface import ToolRegistry
from autonomous_scan_agent.reasoning_tts import speak_reasoning_async


_PROMPTS_DIR = os.path.join(_HAMLYN_ASA_PKG, "prompts")

# Resolve the project-owned path execution server.
_HAMLYN_PATH_EXEC_SERVER = os.path.join(repo_root(), "workspace", "src", "image_processing", "scripts", "path_execute_server.py")


def _setup_run_log_tee() -> Optional[str]:
    """
    Record all terminal output (stdout+stderr) for each run.
    Writes logs under: OrchestratorRUSS/logs
    Never overwrites: filename includes timestamp + pid; uses open(..., 'x') for safety.
    """
    out_dir = os.path.join(orchestrator_root(), "logs")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    base = f"run_agent_{ts}_pid{pid}.log"
    path = os.path.join(out_dir, base)

    f = None
    try:
        # 'x' => fail if exists (no overwrite)
        f = open(path, "x", encoding="utf-8")
    except FileExistsError:
        # Extremely unlikely, but keep safe.
        for i in range(1, 1000):
            alt = os.path.join(out_dir, base.replace(".log", f"_{i}.log"))
            try:
                f = open(alt, "x", encoding="utf-8")
                path = alt
                break
            except FileExistsError:
                continue
    except Exception:
        return None
    if f is None:
        return None

    class _Tee:
        def __init__(self, stream, file_obj):
            self._stream = stream
            self._file = file_obj

        def write(self, s):
            try:
                self._stream.write(s)
            except Exception:
                pass
            try:
                self._file.write(s)
            except Exception:
                pass

        def flush(self):
            try:
                self._stream.flush()
            except Exception:
                pass
            try:
                self._file.flush()
            except Exception:
                pass

        def isatty(self):
            try:
                return bool(getattr(self._stream, "isatty", lambda: False)())
            except Exception:
                return False

    # Redirect both stdout and stderr
    sys.stdout = _Tee(sys.stdout, f)  # type: ignore
    sys.stderr = _Tee(sys.stderr, f)  # type: ignore
    try:
        print(f"[agent] log recording enabled: {path}")
    except Exception:
        pass
    return path


def _resolve_prompt_paths() -> Tuple[str, str]:
    """
    Returns (api_catalog_path, handbook_path).
    NOTE: reuse mode removed; always use the real catalog.
    """
    return (
        os.path.join(_PROMPTS_DIR, "russagent_api_catalog.json"),
        os.path.join(_PROMPTS_DIR, "russagent_handbook.txt"),
    )


class SimpleRAG:
    """
    A simple RAG system using TF-IDF and Cosine Similarity.
    """
    def __init__(self, handbook_text: str, api_catalog: List[Dict]):
        if not SKLEARN_AVAILABLE:
            rospy.logwarn("sklearn not found. RAG will fallback to keyword matching (degraded performance).")
        
        self.handbook_chunks = self._chunk_handbook(handbook_text)
        self.tool_chunks, self.tools = self._chunk_api(api_catalog)
        
        self.vectorizer = None
        self.hb_vectors = None
        self.tool_vectors = None
        
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            # Fit on all corpus
            all_text = self.handbook_chunks + self.tool_chunks
            self.vectorizer.fit(all_text)
            
            self.hb_vectors = self.vectorizer.transform(self.handbook_chunks)
            self.tool_vectors = self.vectorizer.transform(self.tool_chunks)

    def _chunk_handbook(self, text: str) -> List[str]:
        # Split by double newlines to get paragraphs
        chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        return chunks

    def _chunk_api(self, catalog: List[Dict]) -> Tuple[List[str], List[Dict]]:
        chunks = []
        tools = []
        for t in catalog:
            name = t.get("name", "")
            desc = t.get("description", "")
            # Create a rich representation for embedding
            text = f"Tool: {name}. Description: {desc}."
            chunks.append(text)
            tools.append(t)
        return chunks, tools

    def retrieve_handbook(self, query: str, k: int = 2) -> List[str]:
        if not SKLEARN_AVAILABLE:
            return self.handbook_chunks[:k] # Fallback
        
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.hb_vectors).flatten()
        indices = np.argsort(sims)[::-1][:k]
        return [self.handbook_chunks[i] for i in indices]

    def retrieve_tools(self, query: str, k: int = 5) -> List[Dict]:
        if not SKLEARN_AVAILABLE:
            return self.tools[:k] # Fallback

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tool_vectors).flatten()
        indices = np.argsort(sims)[::-1][:k]
        return [self.tools[i] for i in indices]


BASE_SYSTEM_PROMPT = """You are an autonomous ultrasound scanning agent.
Use the Handbook as guidance, but you may make non-linear decisions based on the Current Observation and History.

Critical behavior:
- Choose the most appropriate NEXT tool based on the Current Observation + History + Handbook.
- Do NOT repeat a step that is already completed unless the Current Observation explicitly indicates failure or the need to redo it.
- Only call operator_io_tool when human confirmation/input is truly needed (task selection, patient instruction confirmation, clinician judgment).
- Once the scanning task has been identified, do NOT ask to confirm it again. Proceed to acquisition/projection/Point0/execute/reset as the workflow specifies.
Output format (IMPORTANT):
- First, output EXACTLY ONE plain-text line starting with: Reasoning: <short reasoning>
- Then, output EXACTLY ONE valid JSON object (no markdown, no backticks, no extra text).
IMPORTANT JSON formatting rules:
- It MUST be a single valid JSON object (no extra braces, no trailing commas).
{
  "type": "tool_call",
  "tool_name": "name_of_tool",
  "tool_args": { ... }
}
"""

ACTION_KEYS = ["type", "tool_name", "tool_args", "thought", "status", "next", "message"]

def _extract_reasoning_block_from_llm_text(text: str) -> str:
    """
    Extract the FULL reasoning block from LLM output, without adding/modifying content.
    
    We record everything from the first line that starts with one of:
      Reasoning: / REASONING: / THOUGHT:
    up to (but not including) the first JSON object start '{' (if present).
    """
    if not text:
        return ""
    
    # Find the first reasoning marker line.
    m = re.search(r"^\s*(Reasoning|REASONING|THOUGHT)\s*:\s*", text, flags=re.MULTILINE)
    if not m:
        return ""
    
    start = m.start()  # include the marker line itself ("Reasoning:" / "THOUGHT:")
    tail = text[start:]
    
    # Stop before the first JSON object if present (we do NOT want to include action JSON in reasoning).
    brace_idx = tail.find("{")
    block = tail[:brace_idx] if brace_idx >= 0 else tail
    return block.strip()

def _safe_json_extract(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    s = re.sub(r"```json", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```", "", s)
    s = s.strip()
    try:
        return json.loads(s)
    except:
        # Simple retry for common errors
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return {}

def _extract_sticky_task_anchor_from_history(history: List[str]) -> str:
    """
    SIM-style sticky task anchor:
    Scan history (latest first) and extract:
      "Task identified by operator: <task_name>."
    """
    sticky_task_anchor = ""
    for h in reversed(history[-10:] if history else []):
        if not h:
            continue
        h_low = h.lower()
        if "task identified by operator:" in h_low:
            m = re.search(r"task identified by operator:\s*([^.\n]+)", h, re.IGNORECASE)
            if m:
                task_name = m.group(1).strip()
                sticky_task_anchor = f"Task identified by operator: {task_name}."
            else:
                idx = h_low.find("task identified by operator:")
                rest = h[idx + len("task identified by operator:"):].strip()
                dot_idx = rest.find(".")
                nl_idx = rest.find("\n")
                end_idx = (
                    min(dot_idx, nl_idx)
                    if dot_idx >= 0 and nl_idx >= 0
                    else (dot_idx if dot_idx >= 0 else (nl_idx if nl_idx >= 0 else len(rest)))
                )
                task_name = rest[:end_idx].strip() if end_idx > 0 else rest.strip()
                if task_name:
                    sticky_task_anchor = f"Task identified by operator: {task_name}."
            if sticky_task_anchor:
                break
    return sticky_task_anchor

def _normalize_action(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: d.get(k) for k in ACTION_KEYS}
    out["type"] = out.get("type") or "tool_call"
    out["tool_name"] = str(out.get("tool_name") or "").strip()
    out["tool_args"] = out.get("tool_args") or {}
    return out


def _schema_keys_by_tool(api_catalog: List[Dict[str, Any]]) -> Dict[str, set]:
    """
    Build tool_name -> allowed arg keys from api_catalog[*].args_schema.
    """
    out: Dict[str, set] = {}
    for t in api_catalog or []:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        schema = t.get("args_schema") or {}
        if isinstance(schema, dict):
            out[name] = set([str(k) for k in schema.keys()])
    return out


def _filter_tool_args(tool_name: str, tool_args: Dict[str, Any], allowed_keys_map: Dict[str, set]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Strip unknown args not present in api_catalog args_schema for this tool.
    Returns (filtered_args, removed_keys).
    """
    name = str(tool_name or "").strip()
    args = dict(tool_args or {}) if isinstance(tool_args, dict) else {}
    allowed = allowed_keys_map.get(name)
    if not allowed:
        return args, []
    removed: List[str] = []
    for k in list(args.keys()):
        if k not in allowed:
            removed.append(k)
            args.pop(k, None)
    return args, removed

def _strip_thought_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove any "thought-like" keys from the LLM action object.
    (We want reasoning printed BEFORE JSON, and JSON should be tool_call-only.)
    """
    banned = {"thought", "think", "reasoning", "analysis", "cot"}
    out: Dict[str, Any] = {}
    for k, v in (obj or {}).items():
        if str(k).lower() in banned:
            continue
        out[k] = v
    return out

@dataclass
class RealState:
    last_observation_text: str = "Start of task. No actions taken yet."
    history: List[str] = field(default_factory=list)
    last_tool_result: Dict = field(default_factory=dict)
def _real_robot_enabled() -> bool:
    return os.environ.get("RUSSAGENT_ENABLE_ROBOT", "0").strip().lower() in ("1", "true", "yes")


class RealToolEnv:
    def __init__(self):
        # Register tools (broad set, let RAG filter)
        include_tools = [
            "operator_io_tool", 
            "acquire_trajectory_tool",
            "reset_to_capture_pose_tool", "ribline_projection_tool", 
            "path_execute_server_tool",
            "point0_auto_verify_tool",
            "post_scan_adjust_tool",
        ]

        self.registry = ToolRegistry(include_tools=include_tools)
        self.state = RealState()
        self.robot_enabled = _real_robot_enabled()
        if self.robot_enabled:
            self._ensure_path_execute_server_running()
        else:
            rospy.logwarn("[SAFETY] Robot execution disabled. Set RUSSAGENT_ENABLE_ROBOT=1 only after validation.")

    def _ensure_path_execute_server_running(self):
        # Restart on each agent session so code/param updates (e.g. Tool6 point0 sequence) are picked up.
        try:
            nodes = subprocess.check_output(["rosnode", "list"]).decode().split()
            if "/path_execute_server" in nodes:
                subprocess.run(["rosnode", "kill", "/path_execute_server"], check=False)
                time.sleep(1.0)
        except Exception:
            pass
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    _HAMLYN_PATH_EXEC_SERVER,
                    "_auto_configure_position:=false",
                    "_auto_configure_impedance:=true",
                    "_revert_to_position_on_finish:=false",
                    "_use_position_for_pre_stage:=false",
                    "_enable_pre_stage:=true",
                    "_pre_stage_pause:=0.0",
                    "_cancel_on_new:=false",
                    "_ignore_z_error:=true",
                    "_error_position_tolerance_xy:=0.02",
                    "_lookahead_distance_xy:=0.02",
                ]
            )
        except Exception:
            pass

    def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.robot_enabled and tool_name != "operator_io_tool":
            return {
                "status": "blocked",
                "message": "robot_execution_disabled",
                "nl_observation": "Robot action blocked by the RUSSAGENT_ENABLE_ROBOT safety gate.",
            }
        res = self.registry.execute_tool(tool_name, **(tool_args or {}))

        # Guard against projection/YAML race:
        # ribline_projection_tool spawns a node and may return before path_preview1.yaml is fully written.
        # point0_auto_verify_tool will fail if it loads an empty/partial YAML (frames empty).
        if tool_name == "ribline_projection_tool" and isinstance(res, dict) and str(res.get("status") or "").lower() == "success":
            if res.get("trajectory_validated"):
                pass  # Tool2 already waited for YAML and operator confirmation.
            else:
                out_yaml = str(res.get("output_yaml") or path_preview_yaml())

                def _yaml_has_frames(p: str) -> bool:
                    if not YAML_AVAILABLE:
                        return False
                    try:
                        if not os.path.exists(p):
                            return False
                        with open(p, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f) or {}
                        frames = data.get("frames", None)
                        return isinstance(frames, list) and (len(frames) > 0)
                    except Exception:
                        return False

                t0 = time.time()
                timeout = float(os.environ.get("REAL_YAML_READY_TIMEOUT_SEC", "20.0"))
                while (time.time() - t0) < timeout:
                    if _yaml_has_frames(out_yaml):
                        break
                    time.sleep(0.2)
                else:
                    # If still not ready, mark as failed so the agent retries projection instead of proceeding to point0.
                    res = dict(res)
                    res["status"] = "failed"
                    res["message"] = "projection_yaml_not_ready"
                    res["nl_observation"] = f"Projection finished but YAML is not ready (frames empty): {out_yaml}. Please retry ribline_projection_tool."
        # SIM-style: do NOT store structured task_kind/task_side in state.
        # Instead, inject a sticky NL anchor into nl_observation so the runner can recover it from history.
        if tool_name == "operator_io_tool":
            try:
                ans_raw = str((res or {}).get("answer") or "").strip()
                ans = ans_raw.lower()

                task_summary = ""
                # If your operator tool already provides a structured summary, prefer it.
                # (SIM tool uses task_summary sometimes.)
                if isinstance(res, dict) and str(res.get("task_summary") or "").strip():
                    task_summary = str(res.get("task_summary")).strip()
                else:
                    # Lightweight parse from answer text (no state stored)
                    if "gallbladder" in ans:
                        task_summary = "gallbladder"
                    elif "kidney" in ans:
                        # keep side words if present
                        if "left" in ans or "左" in ans:
                            task_summary = "kidney (left)"
                        elif "right" in ans or "右" in ans:
                            task_summary = "kidney (right)"
                        else:
                            task_summary = "kidney"
                    elif "spine" in ans:
                        task_summary = "spine"

                if task_summary:
                    anchor = f"Task identified by operator: {task_summary}."
                    # Append anchor to existing nl_observation (or create if missing)
                    old_nl = str((res or {}).get("nl_observation") or "").strip()
                    if old_nl:
                        res["nl_observation"] = f"{old_nl} {anchor}"
                    else:
                        res["nl_observation"] = anchor

                    # Optional: keep ROS param if your other nodes rely on it (safe to keep).
                    # If you truly want SIM-like behavior, you can delete this block.
                    try:
                        kind = task_summary.split()[0].strip()  # "gallbladder"/"kidney"/"spine"
                        rospy.set_param("/autonomous_scan_agent/task_kind", kind)
                    except Exception:
                        pass
            except Exception:
                pass
        # Update State with Natural Language Observation
        nl_obs = res.get("nl_observation", f"{tool_name} executed.")
        self.state.last_observation_text = nl_obs
        self.state.history.append(f"Action: {tool_name} -> Result: {nl_obs}")
        self.state.history = self.state.history[-10:] # Keep last 10
        self.state.last_tool_result = res
        return res

def _load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main() -> int:
    if not _real_robot_enabled():
        print("[SAFETY] Refusing to start real-robot agent. Set RUSSAGENT_ENABLE_ROBOT=1 after simulation and low-speed validation.", file=sys.stderr)
        return 2
    _setup_run_log_tee()
    rospy.init_node("russagent", anonymous=False)
    env = RealToolEnv()
    
    catalog_path, handbook_path = _resolve_prompt_paths()
    
    # Load RAG Data
    try:
        handbook_text = _load_text(handbook_path)
        api_catalog = _load_json(catalog_path).get("tools", [])
        rag = SimpleRAG(handbook_text, api_catalog)
    except Exception as e:
        rospy.logerr(f"Failed to load RAG data: {e}")
        return 1
    allowed_keys_map = _schema_keys_by_tool(api_catalog)

    llm = LlamaClient()
    max_steps = int(rospy.get_param("~max_steps", 50))
    valid_tools_all = os.environ.get("REAL_VALID_TOOLS_ALL", "0").strip().lower() in ("1", "true", "yes", "y")
    api_k = int(os.environ.get("REAL_API_K", "6"))

    rospy.loginfo("[agent] RAG Agent Started.")

    for step in range(max_steps):
        if rospy.is_shutdown():
            break

        # 1. Retrieve Context
        obs_text = env.state.last_observation_text

        # SIM-style sticky anchor recovered from history
        sticky_task_anchor = _extract_sticky_task_anchor_from_history(env.state.history)
        sticky = f" {sticky_task_anchor}" if sticky_task_anchor else ""

        # include a small amount of history text to reduce step regression
        hist_k = 5
        recent_hist = env.state.history[-hist_k:] if env.state.history else []
        hist_text = " ".join(recent_hist)

        # Query for handbook retrieval (inject sticky)
        query_text = f"{obs_text} {hist_text}{sticky}"
        relevant_handbook = rag.retrieve_handbook(query_text, k=4)

        # Tool query (obs + hist + handbook + sticky)
        hb_context = " ".join(relevant_handbook)
        tool_query = f"{obs_text} {hist_text} {hb_context}{sticky}"
        relevant_tools = rag.retrieve_tools(tool_query, k=max(1, api_k))

        if valid_tools_all:
            allowed_names = [t.get("name") for t in api_catalog if t.get("name")]
        else:
            allowed_names = [t.get("name") for t in relevant_tools if t.get("name")]

        # 2. Construct Prompt
        # Important: if we allow all tools, ALSO provide full API defs so LLM sees the exact args_schema
        # (otherwise it may hallucinate fields like "input_type").
        api_for_prompt = api_catalog if valid_tools_all else relevant_tools
        tool_defs = [
            json.dumps({k: v for k, v in t.items() if k in ("name", "description", "args_schema")}, ensure_ascii=False)
            for t in (api_for_prompt or [])
        ]

        prompt = (
            f"Current Observation: {obs_text}\n\n"
            + (f"Sticky Context: {sticky_task_anchor}\n\n" if sticky_task_anchor else "")
            + f"History (Last {len(recent_hist)}): {json.dumps(recent_hist, ensure_ascii=False)}\n\n"
            + "Relevant Handbook (RHR):\n" + "\n---\n".join(relevant_handbook) + "\n\n"
            + "Relevant APIs (UAR):\n" + "\n".join(tool_defs) + "\n\n"
            + f"Instructions: Based on the Observation and Handbook, pick the next tool. "
            + f"Valid tools are: {json.dumps(allowed_names)}. "
            + "IMPORTANT: tool_args MUST ONLY use keys defined in that tool's args_schema. Do NOT add extra keys.\n"
            + "Output exactly ONE JSON object and NOTHING else."
        )
        # 3. LLM Call
        messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        
        debug = bool(rospy.get_param("~llm_debug", False))
        if debug:
            rospy.loginfo(f"PROMPT:\n{prompt}")

        # 3.5 LLM call with same-step retries
        last_resp = ""
        action: Dict[str, Any] = {}
        for attempt in range(4):
            resp = llm.chat(messages, max_tokens=512, temperature=0.0)
            last_resp = resp
            cand = _normalize_action(_safe_json_extract(resp))
            # Minimal validation: must be tool_call/finish; tool_name must be allowed when tool_call.
            t = cand.get("type")
            tool_name_cand = str(cand.get("tool_name") or "").strip()
            if t == "finish":
                action = cand
                break
            if t != "tool_call":
                messages.append({"role": "user", "content": "INVALID ACTION. type must be tool_call or finish. Output JSON only."})
                continue
            if (not tool_name_cand) or (tool_name_cand not in allowed_names):
                messages.append({"role": "user", "content": f"INVALID TOOL. tool_name must be one of: {json.dumps(allowed_names)}. Output JSON only."})
                continue
            if not isinstance(cand.get("tool_args"), dict):
                messages.append({"role": "user", "content": "INVALID tool_args. tool_args must be a JSON object. Output JSON only."})
                continue
            action = cand
            break
        if not action:
            rospy.logerr("[agent] No valid action after retries. Last resp (head):\n%s", (last_resp or "")[:2500])
            continue
        
        if action["type"] == "finish":
            rospy.loginfo("[agent] LLM decided to finish.")
            return 0
            
        tool_name = action.get("tool_name")
        if not tool_name or tool_name not in allowed_names:
            # Fallback: if LLM hallucinates, warn and maybe try again or just fail step
            rospy.logwarn(f"[agent] Invalid tool selected: {tool_name}. Allowed: {allowed_names}")
            # Optional: Could implement a retry loop here
            continue
        tool_args = action.get("tool_args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        # Strip hallucinated args not in args_schema (e.g., input_type)
        tool_args, removed = _filter_tool_args(str(tool_name), tool_args, allowed_keys_map)
        if removed:
            rospy.logwarn("[agent] stripped unknown tool_args keys for %s: %s", str(tool_name), ",".join(sorted(removed)))
        # --- SIM-style terminal output ---
        # 1) Print reasoning block BEFORE JSON (if present)
        reasoning_block = _extract_reasoning_block_from_llm_text(last_resp or "").strip()
        if reasoning_block:
            # Keep the model's own "Reasoning:" line as-is (can be multi-line).
            print(reasoning_block)
            speak_reasoning_async(reasoning_block)

        # 2) Print ONE tool_call JSON object ONLY (no thought field)
        action_clean = _strip_thought_fields(action)
        action_json = {
            "type": str(action_clean.get("type") or "tool_call"),
            "tool_name": str(tool_name or "").strip(),
            "tool_args": tool_args if isinstance(tool_args, dict) else {},
        }
        print(json.dumps(action_json, ensure_ascii=False, indent=2))
        
        res = env.execute(tool_name, tool_args)
        if isinstance(res, dict) and str(res.get('status') or '').lower() not in ('success','ok'):
            rospy.logwarn("[agent] %s failed. args=%s message=%s nl_observation=%s", tool_name, json.dumps(tool_args, ensure_ascii=False), str(res.get('message') or ''), str(res.get('nl_observation') or ''))

        # Stop only when reset indicates finish=true (default). This keeps sim/real consistent and
        # allows future multi-segment workflows to use finish=false for intermediate resets.
        if tool_name == "reset_to_capture_pose_tool" and isinstance(res, dict) and str(res.get("status") or "").lower() == "success":
            if bool(res.get("finish", True)):
                rospy.loginfo("[agent] Reset finished (finish=true); task complete. Exiting.")
                return 0
            rospy.loginfo("[agent] Reset finished (finish=false); session continues.")
        
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
