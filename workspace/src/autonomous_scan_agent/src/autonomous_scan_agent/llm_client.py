#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
from typing import Any, Dict, List, Optional, Sequence

import requests

class LlamaClient:
    """
    统一 LLM Client：
    - 默认：使用用户配置的 OpenAI-compatible HTTP endpoint 进行 chat + embeddings
    - 可选：若未配置 HTTP endpoint，才回退到本地 llama_cpp GGUF

    远端配置（环境变量优先）：
    - LLM_API_BASE: API endpoint
    - LLM_CHAT_MODEL: endpoint 提供的模型名称
    - LLM_EMBED_MODEL: 可选的 embedding 模型名称
    - LLM_API_KEY: 可选 API key
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 8192,
        api_base: Optional[str] = None,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_sec: float = 60.0,
    ):
        self.timeout_sec = float(timeout_sec)

        # --- Prefer remote OpenAI-compatible endpoint ---
        env_api_base = os.environ.get("LLM_API_BASE", "").strip()
        self.api_base = (api_base or env_api_base).strip() or None
        self.api_key = (api_key or os.environ.get("LLM_API_KEY", "").strip()) or ""
        self.chat_model = (chat_model or os.environ.get("LLM_CHAT_MODEL", "").strip()) or None
        self.embed_model = (embed_model or os.environ.get("LLM_EMBED_MODEL", "").strip()) or "text-embedding-nomic-embed-text-v1.5"

        self._backend = "remote" if self.api_base else "local"
        self._llama = None

        if self._backend == "remote" and not self.chat_model:
            raise ValueError("Set LLM_CHAT_MODEL to the model name served by the configured endpoint.")

        if self._backend == "local":
            # Local llama_cpp fallback (only when remote not configured)
            if model_path is None:
                model_path = os.path.expanduser(os.path.join(workspace_root(), 'models', 'llama-3-8b-instruct-q4_k_m.gguf'))
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    "Either provide a valid local model_path or set LLM_API_BASE to use a remote OpenAI-compatible endpoint."
                )
            try:
                from llama_cpp import Llama  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "llama_cpp is not available. Install llama_cpp OR set LLM_API_BASE to use a remote endpoint."
                ) from e

            print(f"Loading local model from {model_path} with n_ctx={n_ctx}...")
            self._llama = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)

    def chat(self, messages, max_tokens=512, temperature=0.7):
        """
        messages: list of dicts, e.g. [{"role": "user", "content": "..."}]
        """
        if self._backend == "remote":
            url = self.api_base.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload: Dict[str, Any] = {
                "model": self.chat_model,
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
            # Optional: force JSON object output for servers that support OpenAI response_format.
            # Enable with env: LLM_FORCE_JSON=1
            if os.environ.get("LLM_FORCE_JSON", "").strip() in ("1", "true", "True"):
                # Different OpenAI-compatible servers have different accepted enums.
                # LM Studio (Jan 2026) accepts response_format.type in {"text","json_schema"}.
                # We intentionally do NOT send json_schema here (would require supplying schema).
                # Instead we rely on prompt + robust JSON extraction.
                payload["response_format"] = {"type": "text"}
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            # If server doesn't support response_format, retry once without it.
            if r.status_code >= 400 and "response_format" in payload:
                payload.pop("response_format", None)
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            if r.status_code >= 400:
                # Help diagnose LM Studio / OpenAI-compatible 400s (invalid model name, bad payload, etc.)
                try:
                    print("[llm_client] HTTP", r.status_code, "body(head):", (r.text or "")[:500])
                except Exception:
                    pass
            r.raise_for_status()
            obj = r.json()
            return obj["choices"][0]["message"]["content"]

        # local
        assert self._llama is not None
        response = self._llama.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
        return response["choices"][0]["message"]["content"]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """
        生成 embedding 向量（用于 cosine similarity 检索，对应论文公式(2)）。
        - remote: OpenAI-compatible /v1/embeddings
        - local: llama_cpp embedding
        """
        if texts is None:
            return []
        inputs = [str(t) for t in texts]

        if self._backend == "remote":
            url = self.api_base.rstrip("/") + "/embeddings"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload: Dict[str, Any] = {"model": self.embed_model, "input": inputs}
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            r.raise_for_status()
            obj = r.json()
            data = obj.get("data", []) if isinstance(obj, dict) else []
            return [d.get("embedding", []) for d in data]

        # local llama_cpp embedding
        assert self._llama is not None
        if hasattr(self._llama, "create_embedding"):
            out: Any = self._llama.create_embedding(input=inputs)
            data = out.get("data", []) if isinstance(out, dict) else []
            return [d.get("embedding", []) for d in data]
        if hasattr(self._llama, "embed"):
            vecs: List[List[float]] = []
            for t in inputs:
                v = self._llama.embed(t)
                vecs.append(list(v) if v is not None else [])
            return vecs
        raise RuntimeError("Local llama_cpp embedding API not available in this environment")

    def list_models(self) -> List[str]:
        """
        仅 remote 可用：返回 /v1/models 下的模型 id 列表。
        """
        if self._backend != "remote":
            return []
        url = self.api_base.rstrip("/") + "/models"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = requests.get(url, headers=headers, timeout=self.timeout_sec)
        r.raise_for_status()
        obj = r.json()
        out: List[str] = []
        for item in obj.get("data", []) if isinstance(obj, dict) else []:
            mid = item.get("id")
            if mid:
                out.append(mid)
        return out

if __name__ == "__main__":
    # Quick smoke test (remote preferred when LLM_API_BASE is set)
    c = LlamaClient()
    try:
        ms = c.list_models()
        if ms:
            print("models:", ms[:5])
    except Exception as e:
        print("list_models failed:", e)
    print(c.chat([{"role": "user", "content": "Reply only: OK"}], max_tokens=8, temperature=0.2))
