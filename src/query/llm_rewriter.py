"""
LLM-powered query rewriter.

Takes a short, vague user query (e.g. "give me all sedentary activities")
and rewrites it into one or more retrieval-optimised sentences matched to
the embedding space learned during CLIP alignment training.

Three rewrite modes
-------------------
single        One rich, descriptive sentence covering all variants.
              Best for a single broad sweep of the space.

multi_location  One sentence per relevant room / sensor combination.
                e.g. "person sitting in the living room armchair" and
                     "person at the office desk" as separate queries.
                Lets FAISS cast a narrow but precise net per location.

multi_wording   Several paraphrases of the same concept in different
                surface forms.  Useful when the user's intent is clear
                but the embedding space may be sensitive to phrasing.

All modes return List[str].  The caller decides how to use the list
(single search vs. one search per sentence + merge).

Supported backends
------------------
- Gemini (google-generativeai)  ← default
- Any OpenAI-compatible API
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal, Optional

RewriteMode = Literal["single", "multi_location", "multi_wording"]

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

_METADATA_PATH = Path(__file__).resolve().parents[2] / "metadata" / "casas_metadata.json"
_PROJECT_ROOT  = Path(__file__).resolve().parents[2]

# Default captions path pattern: data/processed/casas/{home}/FD_60/train_captions_baseline.json
def _default_captions_path(home: str) -> Path:
    return _PROJECT_ROOT / "data" / "processed" / "casas" / home / "FD_60" / "train_captions_baseline.json"


_SENSOR_TYPE_DESCRIPTIONS = {
    "M": "motion sensor (detects presence / movement)",
    "D": "door/contact sensor (detects open/close events)",
    "T": "temperature sensor",
    "I": "infrared sensor",
    "AD": "motion sensor",
    "MA": "motion sensor",
}


def _load_example_captions(
    captions_path: Path | None,
    n: int = 6,
    seed: int = 0,
) -> list[str]:
    """
    Load n random captions from a train_captions_*.json file.
    Returns an empty list if the file doesn't exist or can't be parsed.
    """
    if captions_path is None or not captions_path.exists():
        return []
    try:
        import random
        rng = random.Random(seed)
        with open(captions_path) as f:
            data = json.load(f)
        # Handle both list and {"captions": [...]} formats
        items = data if isinstance(data, list) else data.get("captions", [])
        # Collect non-empty first captions
        pool = [
            item["captions"][0]
            for item in items
            if item.get("captions") and item["captions"][0].strip()
        ]
        return rng.sample(pool, min(n, len(pool)))
    except Exception:
        return []


def _load_home_context(home: str, metadata_path: Path = _METADATA_PATH) -> dict:
    if not metadata_path.exists():
        return {}
    with open(metadata_path) as f:
        meta = json.load(f)
    home_data = meta.get(home, {})
    return {
        "sensor_locations": home_data.get("sensor_location", {}),
        "sensor_details": home_data.get("sensor_details", {}),
        # activity_descriptions kept for internal use only — NOT passed to LLM
        "activity_descriptions": home_data.get("label_to_text_sourish", {}),
        "num_residents": home_data.get("num_residents", 1),
        "name": home_data.get("name", home),
    }


def _build_sensor_context(home_context: dict) -> str:
    """
    Build a sensor inventory string that contains ONLY physical sensor
    information — no activity labels or category names.
    """
    locations = home_context.get("sensor_locations", {})
    details = home_context.get("sensor_details", {})

    # Group sensors by room
    by_room: dict[str, list[str]] = {}
    for sensor_id, room in sorted(locations.items()):
        room = room.replace("_", " ")
        by_room.setdefault(room, []).append(sensor_id)

    lines = []
    for room, sensor_ids in sorted(by_room.items()):
        lines.append(f"  {room}:")
        for sid in sorted(sensor_ids):
            # Sensor type from prefix
            prefix = "".join(c for c in sid if c.isalpha())
            type_desc = _SENSOR_TYPE_DESCRIPTIONS.get(prefix, "sensor")
            detail = details.get(sid)
            if detail:
                lines.append(f"    {sid}  [{type_desc}]  — {detail}")
            else:
                lines.append(f"    {sid}  [{type_desc}]")
    return "\n".join(lines)


def _build_system_prompt(
    home_context: dict,
    mode: RewriteMode,
    example_captions: list[str] | None = None,
) -> str:
    sensor_context = _build_sensor_context(home_context)

    examples_block = ""
    if example_captions:
        formatted = "\n".join(f'  - "{c}"' for c in example_captions)
        examples_block = f"""
## Example sentences from the training data
These are real captions describing sensor sequences in this home.
Use them to understand the expected sentence style and vocabulary.
Do NOT copy them — they are style references only.
{formatted}
"""

    header = f"""You are a query-rewriting assistant for a smart-home sensor retrieval system.

Your job is to translate a user's natural-language activity description into
one or more retrieval sentences that will be matched against embedded sensor
sequences via cosine similarity.

## Available sensors (by room)
{sensor_context}
{examples_block}

## Sensor behaviour reference
- Motion sensors fire when a person is present or moving nearby.
  A long sequence of motion events in one location suggests the person
  stayed there for an extended period (stationary or slow-moving).
- Door/contact sensors fire on open or close events.
- Absence of motion in a room means the person is not there.

## Reasoning step (mandatory)
Before writing any sentences you MUST first reason:
1. Which sensors / rooms are relevant to the requested activity?
2. What sensor *pattern* would this activity produce?
   (e.g. sustained motion in one room, no door events, low transition rate)
3. Is this activity actually detectable from the sensors listed above?
   If NO — explain why and output an empty list [].

## Output constraints (apply to the OUTPUT sentences only — not to your reasoning)
- You MUST interpret the user's query and understand what they mean, even if
  they use high-level or category-like words (e.g. "sedentary", "cooking",
  "night wandering"). These are the *intent*, not forbidden words.
- The OUTPUT sentences must describe only observable physical behaviour:
  presence in a room, sensor firings, duration, transitions between rooms.
- Do NOT use abstract category names or dataset labels *in the output sentences*
  (e.g. avoid "sedentary activity" as a phrase in the sentence itself — instead
  describe what the sensors would see: sustained presence, minimal transitions).
- Write in third-person present tense.
- Use room names and sensor detail descriptions (e.g. "armchair", "working desk")
  from the list above. Do NOT use sensor IDs (e.g. M003, D001) in the output
  sentences — sensor IDs may appear in your reasoning but never in the sentences.
"""

    if mode == "single":
        return header + """
## Output format
After your reasoning, output ONE broad, inclusive sentence that:
- Mentions ALL relevant rooms and specific sensor positions identified in
  your reasoning, even if the activity can occur in multiple locations.
- Deliberately covers every plausible variant of the requested activity
  so the embedding can match a wide range of sequences.
- A sentence covering multiple rooms and sensor patterns is correct and
  expected — do not refuse because the activity spans more than one location.
- Contains NO activity label names.

A useful sentence looks like:
  "A person remains present near the [sensor details] in the [rooms],
   with sustained motion sensor activations and few room transitions."

Format your response as:
REASONING: <your reasoning here>
SENTENCE: <the single retrieval sentence>
"""

    if mode == "multi_location":
        return header + """
## Output format
After your reasoning, output one SHORT sentence per relevant room or
sensor location where this activity could occur.

Each sentence must:
- Be anchored to a specific room / sensor detail from the list above.
- Describe what the sensors in that location would observe.
- Contain NO activity label names.
- Preserve ALL temporal, frequency, and contextual qualifiers from the user's
  query (e.g. time of day, day of week, duration, recurrence). If the user says
  "during the night", every sentence must reflect that time-of-day context.

Format your response as:
REASONING: <your reasoning here>
SENTENCES: ["sentence for location A", "sentence for location B", ...]

Return a valid JSON array for SENTENCES. No extra text after the array.
"""

    if mode == "multi_wording":
        return header + """
## Output format
After your reasoning, output 4–6 PARAPHRASES of the same observable behaviour.
If the user's query covers multiple locations or sub-behaviours, produce
paraphrases that together span all of them.

Each paraphrase emphasises a different aspect:
  [1] focus on the room(s) and specific sensor positions
  [2] focus on the duration / temporal pattern (long dwell, few transitions)
  [3] focus on the absence of movement between rooms
  [4] focus on time of day if relevant
  [5+] any other salient sensor-level variation

Rules:
- If the query is broad (e.g. "sedentary"), produce paraphrases that cover
  ALL the relevant locations identified in your reasoning — do not refuse
  because the concept spans more than one location.
- Each sentence describes sensor-observable behaviour only (no label names).

Format your response as:
REASONING: <your reasoning here>
SENTENCES: ["paraphrase 1", "paraphrase 2", ...]

Return a valid JSON array for SENTENCES. No extra text after the array.
"""

    raise ValueError(f"Unknown mode: {mode!r}")


def _parse_response(raw: str, mode: RewriteMode) -> tuple[str, list[str]]:
    """
    Parse the structured LLM response into (reasoning, sentences).

    Expected format:
        REASONING: <text>
        SENTENCE: <one sentence>        ← single mode
        SENTENCES: ["...", "..."]       ← multi modes
    """
    raw = raw.strip()

    # Extract reasoning block
    reasoning = ""
    m = re.search(r"REASONING\s*:\s*(.*?)(?=\nSENTENCE[S]?\s*:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()

    if mode == "single":
        m = re.search(r"SENTENCE\s*:\s*(.+)", raw, re.IGNORECASE)
        if m:
            return reasoning, [m.group(1).strip()]
        # Fallback: last non-empty line
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return reasoning, [lines[-1]] if lines else [raw]

    # multi modes: find SENTENCES: [...]
    m = re.search(r"SENTENCES\s*:\s*(\[.*?\])", raw, re.DOTALL | re.IGNORECASE)
    if m:
        json_str = m.group(1)
    else:
        # Fallback: find first [...] block anywhere in the text
        m = re.search(r"(\[.*?\])", raw, re.DOTALL)
        json_str = m.group(1) if m else "[]"

    # Strip markdown fences inside the extracted block just in case
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str.strip())
    json_str = re.sub(r"\s*```$", "", json_str)

    try:
        parsed = json.loads(json_str)
        sentences = [str(s).strip() for s in parsed if str(s).strip()]
    except json.JSONDecodeError:
        # Last-resort: split by newlines
        sentences = [l.strip().lstrip("-•").strip() for l in raw.splitlines() if l.strip()]

    return reasoning, sentences


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


class _GeminiBackend:
    MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str, model: str | None = None):
        # Suppress gRPC/absl noise before the import triggers them
        import os as _os, warnings as _warnings
        _os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
        _os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")
        _warnings.filterwarnings(
            "ignore",
            message=".*Python version.*end of life.*",
            category=FutureWarning,
            module="google",
        )
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required. Install: pip install google-generativeai"
            ) from e
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = model or self.MODEL

    def call(self, user_query: str, system_prompt: str) -> str:
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
        )
        return model.generate_content(user_query).text.strip()

    @property
    def model_id(self) -> str:
        return f"gemini/{self.model_name}"


class _OpenAICompatibleBackend:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai is required. Install: pip install openai") from e
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model

    def call(self, user_query: str, system_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    @property
    def model_id(self) -> str:
        return f"openai/{self.model_name}"


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class LLMRewriter:
    """
    Rewrites user queries into retrieval-optimised sentences.

    Args:
        home:           Dataset identifier ("milan", "aruba", etc.)
        backend:        "gemini" (default) or "openai"
        api_key:        API key. Falls back to GEMINI_API_KEY / GOOGLE_API_KEY /
                        OPENAI_API_KEY env vars.
        model:          Override the default model name.
        base_url:       For OpenAI-compatible endpoints (Ollama, vLLM, etc.)
        metadata_path:  Override path to casas_metadata.json.
        captions_path:  Path to a train_captions_*.json file whose captions will
                        be shown as style examples in the system prompt.
                        Defaults to data/processed/casas/{home}/FD_60/train_captions_baseline.json.
                        Pass False to disable examples entirely.
        n_examples:     Number of random captions to sample (default 6).
    """

    def __init__(
        self,
        home: str = "milan",
        backend: str = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        metadata_path: Optional[str | Path] = None,
        captions_path: Optional[str | Path | bool] = None,
        n_examples: int = 6,
    ):
        self.home = home
        meta_path = Path(metadata_path) if metadata_path else _METADATA_PATH
        self.home_context = _load_home_context(home, meta_path)

        # Resolve captions path
        if captions_path is False:
            resolved_captions: Path | None = None
        elif captions_path is None:
            resolved_captions = _default_captions_path(home)
        else:
            resolved_captions = Path(captions_path)

        self._example_captions = _load_example_captions(resolved_captions, n=n_examples)
        if self._example_captions:
            src = resolved_captions.name if resolved_captions else "?"
            print(f"[LLMRewriter] Loaded {len(self._example_captions)} style examples from {src}")
        else:
            print("[LLMRewriter] No style examples loaded (captions file not found or disabled)")

        if backend == "gemini":
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                raise ValueError(
                    "Gemini API key required. Pass api_key= or set GEMINI_API_KEY / GOOGLE_API_KEY."
                )
            self._backend = _GeminiBackend(api_key=key, model=model)

        elif backend in ("openai", "openai_compatible"):
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise ValueError("OpenAI API key required. Pass api_key= or set OPENAI_API_KEY.")
            if not model:
                raise ValueError("model= is required for the openai backend.")
            self._backend = _OpenAICompatibleBackend(api_key=key, model=model, base_url=base_url)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Choose 'gemini' or 'openai'.")

    # ------------------------------------------------------------------

    def rewrite(
        self, user_query: str, mode: RewriteMode = "single"
    ) -> tuple[str, list[str]]:
        """
        Rewrite a user query into one or more retrieval-optimised sentences.

        Args:
            user_query: The raw user question.
            mode:       "single"         → 1 rich sentence
                        "multi_location" → one sentence per relevant location
                        "multi_wording"  → several paraphrases of the concept

        Returns:
            (reasoning: str, sentences: list[str])
            reasoning  — the LLM's chain-of-thought about sensor detectability
            sentences  — retrieval-ready sentences (empty list if not detectable)
        """
        system_prompt = _build_system_prompt(self.home_context, mode, self._example_captions)
        raw = self._backend.call(user_query, system_prompt)
        reasoning, sentences = _parse_response(raw, mode)
        return reasoning, sentences

    @property
    def model_id(self) -> str:
        return self._backend.model_id
