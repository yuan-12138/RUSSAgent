#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential English TTS queue for LLM reasoning and operator guidance.

Terminal text is always printed by the caller; this module only handles optional audio.
Utterances are played one after another (FIFO); a new line never cuts off the previous one.

Environment:
  REAL_TTS_ENABLED=1          Enable speech (default on; set 0 to disable)
  REAL_TTS_ENGINE=edge        edge | pyttsx3 | espeak (default edge — much clearer than espeak)
  REAL_TTS_VOICE=en-US-JennyNeural   edge-tts voice id
  REAL_TTS_MAX_CHARS=500      Truncate very long reasoning for speech
  REAL_TTS_LEAD_MS=1000       Leading silence before each clip (ffplay cold-start fix)
"""

from __future__ import annotations

import asyncio
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Optional

_REASONING_PREFIX_RE = re.compile(r"^\s*(Reasoning|REASONING|THOUGHT)\s*:\s*", flags=re.IGNORECASE | re.MULTILINE)

_play_lock = threading.Lock()
_play_proc: Optional[subprocess.Popen] = None

_speech_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()


@dataclass
class _TtsJob:
    text: str
    done: Optional[threading.Event] = None


def clean_reasoning_for_speech(text: str, *, max_chars: int = 500) -> str:
    """Strip Reasoning:/THOUGHT: markers; normalize text for natural TTS (audio only)."""
    if not text:
        return ""
    s = str(text).strip()
    s = _REASONING_PREFIX_RE.sub("", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars > 0 and len(s) > max_chars:
        s = s[: max_chars - 3].rstrip() + "..."
    return s


def is_tts_enabled() -> bool:
    raw = os.environ.get("REAL_TTS_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "no", "n", "off")


def _engine_name() -> str:
    return (os.environ.get("REAL_TTS_ENGINE", "edge") or "edge").strip().lower()


def _max_chars() -> int:
    try:
        return int(os.environ.get("REAL_TTS_MAX_CHARS", "500"))
    except Exception:
        return 500


def _lead_ms() -> int:
    """Silence prepended via ffplay adelay so the audio device can open before speech."""
    try:
        return max(0, int(os.environ.get("REAL_TTS_LEAD_MS", "1000")))
    except Exception:
        return 1000


def _stop_playback_locked() -> None:
    global _play_proc
    proc = _play_proc
    _play_proc = None
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=0.5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _play_audio_file(path: str) -> None:
    """Play an audio file with ffplay and wait until done (single-threaded worker only)."""
    global _play_proc
    ffplay = shutil.which("ffplay")
    if not ffplay:
        raise RuntimeError("ffplay not found (install ffmpeg: sudo apt install ffmpeg)")
    cmd = [
        ffplay,
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "quiet",
        "-vn",
    ]
    lead_ms = _lead_ms()
    if lead_ms > 0:
        # ffplay spawns fresh each utterance; ALSA/Pulse needs ~200-400ms to open.
        # adelay emits silence first so the device is ready before speech starts.
        cmd.extend(["-af", f"adelay={lead_ms}|{lead_ms}"])
    cmd.append(path)
    with _play_lock:
        _play_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _play_proc.wait()
        finally:
            if _play_proc is not None and _play_proc.poll() is None:
                _stop_playback_locked()
            else:
                _play_proc = None


def _speak_edge(text: str) -> None:
    import edge_tts  # type: ignore

    voice = (os.environ.get("REAL_TTS_VOICE", "en-US-JennyNeural") or "en-US-JennyNeural").strip()

    async def _run() -> str:
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="reasoning_tts_")
        os.close(fd)
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)
        return path

    mp3_path = asyncio.run(_run())
    try:
        _play_audio_file(mp3_path)
    finally:
        try:
            os.remove(mp3_path)
        except Exception:
            pass


def _speak_pyttsx3(text: str) -> None:
    import pyttsx3  # type: ignore

    engine = pyttsx3.init()
    try:
        voices = engine.getProperty("voices") or []
        chosen = None
        for voice in voices:
            vid = str(getattr(voice, "id", "") or "").lower()
            vname = str(getattr(voice, "name", "") or "").lower()
            if "en-gb" in vid or "english (great britain)" in vname:
                chosen = voice.id
                break
        if chosen is None:
            for voice in voices:
                vid = str(getattr(voice, "id", "") or "").lower()
                if vid.startswith("en") or "english" in vid:
                    chosen = voice.id
                    break
        if chosen:
            engine.setProperty("voice", chosen)
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass


def _speak_espeak(text: str) -> None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise RuntimeError("espeak-ng/espeak not found")
    subprocess.run([exe, "-v", "en-gb", "-s", "160", "-a", "100", text], check=False)


def _speak_sync(text: str) -> None:
    engine = _engine_name()
    errors: List[str] = []
    attempts = [engine] if engine != "auto" else ["edge", "pyttsx3", "espeak"]
    for attempt in attempts:
        try:
            if attempt == "edge":
                _speak_edge(text)
            elif attempt == "pyttsx3":
                _speak_pyttsx3(text)
            elif attempt == "espeak":
                _speak_espeak(text)
            else:
                continue
            return
        except Exception as exc:
            errors.append(f"{attempt}: {exc}")
    if errors:
        print("[reasoning_tts] speech failed:", "; ".join(errors), flush=True)


def _worker_loop() -> None:
    while True:
        job = _speech_queue.get()
        try:
            if job is None:
                return
            _speak_sync(job.text)
        except Exception as exc:
            print(f"[reasoning_tts] worker error: {exc}", flush=True)
        finally:
            if job is not None and job.done is not None:
                job.done.set()
            _speech_queue.task_done()


def _ensure_worker() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(target=_worker_loop, name="reasoning_tts", daemon=True)
        _worker_thread.start()


def _enqueue(text: str, *, block: bool) -> None:
    if not is_tts_enabled():
        return
    clean = clean_reasoning_for_speech(text, max_chars=_max_chars())
    if not clean:
        return
    done = threading.Event() if block else None
    _speech_queue.put(_TtsJob(text=clean, done=done))
    _ensure_worker()
    if block and done is not None:
        done.wait()


def speak_reasoning_async(text: str) -> None:
    """
    Queue reasoning text for English TTS (non-blocking).
    Waits behind any in-progress or queued utterances.
    """
    _enqueue(text, block=False)


def speak_guidance_sync(text: str) -> None:
    """
    Queue operator guidance and block until this utterance finishes playing.
    Does not interrupt earlier queued speech.
    """
    _enqueue(text, block=True)
