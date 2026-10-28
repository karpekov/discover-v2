"""Gemini client with a content-addressed, file-based response cache.

Every call is keyed by sha256(model, system prompt, user prompt, schema, generation
settings). Responses are stored as one JSON file per call so they can be inspected,
diffed and committed. With ``mode="offline"`` a cache miss raises instead of calling
the API, which guarantees a paper re-run uses exactly the cached LLM outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from pydantic import BaseModel

from .config import DEFAULT_LLM_MODEL, LLM_CACHE_DIR

T = TypeVar("T", bound=BaseModel)
CacheMode = Literal["use", "refresh", "offline"]


class LLMCacheMiss(RuntimeError):
    pass


class GeminiClient:
    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        cache_dir: Path = LLM_CACHE_DIR,
        cache_mode: CacheMode = "use",
        thinking_level: Optional[str] = "high",
        temperature: float = 1.0,
        max_retries: int = 4,
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_mode = cache_mode
        self.thinking_level = thinking_level
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
            warnings.filterwarnings("ignore", category=FutureWarning, module="google")
            from google import genai

            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to call Gemini.")
            self._client = genai.Client(api_key=key)
        return self._client

    def _key(self, tag: str, system: str, user: str, schema_json: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system,
                "user": user,
                "schema": schema_json,
                "thinking_level": self.thinking_level,
                "temperature": self.temperature,
            },
            sort_keys=True,
        )
        return f"{tag}_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _call(self, system: str, user: str, schema: Optional[Type[BaseModel]]) -> str:
        from google.genai import types

        cfg = dict(system_instruction=system, temperature=self.temperature)
        if self.thinking_level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._get_client().models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(**cfg),
                )
                text = (resp.text or "").strip()
                if not text:
                    raise RuntimeError("Empty response from Gemini")
                if schema is not None:
                    schema.model_validate_json(text)
                return text
            except Exception as e:  # noqa: BLE001 — retry transient API / validation errors
                last_err = e
                if getattr(e, "code", None) in (400, 401, 402, 403, 404):
                    raise RuntimeError(f"Gemini call failed (not retryable): {e}") from e
                wait = 2 ** attempt * 5
                print(f"[llm] {self.model} attempt {attempt + 1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts") from last_err

    def _generate(self, tag: str, system: str, user: str, schema: Optional[Type[BaseModel]]) -> tuple[str, dict]:
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True) if schema else ""
        key = self._key(tag, system, user, schema_json)
        path = self._cache_path(key)

        if self.cache_mode != "refresh" and path.exists():
            entry = json.loads(path.read_text())
            return entry["response"], {"cache_key": key, "cache_hit": True, "model": entry["model"]}
        if self.cache_mode == "offline":
            raise LLMCacheMiss(f"No cached LLM response for {key} (offline mode)")

        print(f"[llm] calling {self.model} ({tag}) …")
        t0 = time.time()
        text = self._call(system, user, schema)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "model": self.model,
            "tag": tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "latency_s": round(time.time() - t0, 2),
            "settings": {"thinking_level": self.thinking_level, "temperature": self.temperature},
            "system": system,
            "user": user,
            "response": text,
        }
        path.write_text(json.dumps(entry, indent=2, ensure_ascii=False))
        return text, {"cache_key": key, "cache_hit": False, "model": self.model}

    def generate_json(self, tag: str, system: str, user: str, schema: Type[T]) -> tuple[T, dict]:
        text, info = self._generate(tag, system, user, schema)
        return schema.model_validate_json(text), info

    def generate_text(self, tag: str, system: str, user: str) -> tuple[str, dict]:
        return self._generate(tag, system, user, None)
