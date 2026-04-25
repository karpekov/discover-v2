"""
LLM-powered query rewriter.

Takes a short, vague user query (e.g. "give me all sedentary activities")
and rewrites it into one or more retrieval-optimised sentences matched to
the embedding space learned during CLIP alignment training.

Rewrite modes
-------------
auto  ← default
              The LLM decides its own expansion strategy based on the query.
              Core insight: cosine-similarity retrieval handles OR poorly —
              "room A or room B" averages the embeddings. Separate sentences
              per room (or per time window) each retrieve their target precisely.
              Defaults to 1–3 sentences; only expands further when the activity
              has genuinely distinct sensor locations or sub-actions. Cap: 6.

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

Use ``max_subqueries`` on ``LLMRewriter.rewrite`` to cap how many sentences
are produced (defaults: 6 for auto, 8 for multi_location, 6 for multi_wording;
single is always 1).

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

RewriteMode = Literal["auto", "single", "multi_location", "multi_wording"]

# Hard cap on sub-queries (prompt + post-parse truncation).
_MAX_SUBQUERIES_CAP = 20


def resolve_max_subqueries(mode: RewriteMode, max_subqueries: int | None) -> int:
    """
    Effective max number of retrieval sentences for this mode.

    ``max_subqueries`` overrides mode defaults when set (clamped 1–20).
    ``single`` always resolves to 1.
    ``auto`` defaults to 6 (the prompt instructs the LLM to use as few as needed, typically 1–3).
    """
    if mode == "single":
        return 1
    if max_subqueries is not None:
        return max(1, min(int(max_subqueries), _MAX_SUBQUERIES_CAP))
    if mode == "auto":
        return 6
    if mode == "multi_wording":
        return 6
    if mode == "multi_location":
        return 8
    return 6

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
    strip_temporal: bool = False,
    max_subqueries: int = 6,
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

## Precision over coverage — only include what the sensors can confirm
Only anchor a retrieval sentence to a location if the sensor detail description
for that location **directly and unambiguously** supports the queried activity.
Do NOT extrapolate to rooms or sensors where another activity could equally
explain the same sensor pattern.

**Activity-specific anchoring rule (critical):**
For activities tied to a specific piece of equipment or behaviour (e.g. "watching TV",
"exercising", "cooking"), ONLY use sensors whose detail descriptions explicitly name
that equipment or action.
- Do NOT use generic sedentary sensors (armchair, couch, bed) as evidence of a
  specific activity like "watching TV" — sustained presence in a bedroom is sleeping
  or resting, not TV watching, unless the sensor is explicitly labelled as a TV location.
- A bed sensor confirms sleep/rest, NOT watching TV in bed.
- An armchair sensor confirms sedentary presence, NOT TV watching, unless the
  sensor description explicitly mentions a TV (e.g. "TV armchair", "TV room armchair").
- If the sensor inventory has no sensor explicitly associated with the queried activity,
  say so in your REASONING and produce only the sentences that CAN be grounded.

Examples of this principle:
- For a "watching TV" query: ONLY include sensors explicitly described as "TV armchair",
  "TV room", "couch facing TV", etc. Do NOT include bedroom bed sensors or generic
  armchairs unless the description names them as TV-watching locations.
- For a general sedentary / resting query: sensors labelled armchair, couch, sofa,
  chair, desk, or bed are valid. Do NOT include kitchens or bathrooms.
- For a cooking query: only include sensors near the stove, counter, or oven.
  Do NOT include a hallway sensor just because the person passed through.
- For a sleep query: only include bedroom bed sensors, not the living room couch
  unless it is explicitly described as a sleeping location.

If a room or sensor lacks a detail description that directly supports the activity,
leave it out. Fewer, precise sentences are far better than many uncertain ones.
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
        temporal_rule = (
            "- Do NOT embed time-of-day or day-of-week constraints in output sentences\n"
            "  (e.g. do not add \"at night\" or \"on weekdays\"). Preserve duration and\n"
            "  recurrence qualifiers only. Time and day filtering is handled separately\n"
            "  by a rule-based post-processor applied after retrieval."
        ) if strip_temporal else (
            "- Preserve ALL temporal, frequency, and contextual qualifiers from the user's\n"
            "  query (e.g. time of day, day of week, duration, recurrence). If the user says\n"
            "  \"during the night\", every sentence must reflect that time-of-day context."
        )
        n = max_subqueries
        return header + f"""
## Output format
After your reasoning, output **exactly as many sentences as you can, up to {n}** — one SHORT
sentence per **distinct** room or sensor-detail anchor from the inventory above where this
activity could **plausibly** occur.

**Coverage rule (critical):** Walk through the sensor list by room. For each room that could
host this activity, emit one sentence (different rooms → different sentences). **Do not stop
after only a few rooms** if more rooms in the inventory are still plausible and you have not yet
reached {n} sentences. **Aim to use all {n} slots** whenever the home has at least {n} distinct
plausible locations for the query (e.g. broad queries like "sedentary activity" often apply to
many rooms: living room, bedrooms, office, kitchen seating, hallways with chairs, etc.).

Only output **fewer than {n}** when every remaining room in the inventory is clearly irrelevant,
or when fewer than {n} distinct plausible locations exist in this home.

Never output more than {n} sentences.

Each sentence must:
- Be anchored to a specific room / sensor detail from the list above.
- Describe what the sensors in that location would observe.
- Contain NO activity label names.
{temporal_rule}

Format your response as:
REASONING: <your reasoning here>
SENTENCES: ["sentence for location A", "sentence for location B", ...]

Return a valid JSON array for SENTENCES with at most {n} strings. No extra text after the array.
"""

    if mode == "multi_wording":
        tod_paraphrase = (
            "  [4] focus on sensor transition patterns between rooms"
        ) if strip_temporal else (
            "  [4] focus on time of day if relevant"
        )
        strip_note = (
            "\n- Do NOT include time-of-day or day-of-week constraints in the output sentences."
            "\n  These are filtered separately."
        ) if strip_temporal else ""
        n = max_subqueries
        return header + f"""
## Output format
After your reasoning, output **as many distinct paraphrases as you can, up to {n}** (at least 1).
**Aim to use all {n} slots** by varying wording, sensor emphasis, and (if the query spans
multiple rooms) which room or sub-location you stress — do not stop early after only a handful
if more distinct paraphrases are still meaningful. Never output more than {n} sentences.

If the user's query covers multiple locations or sub-behaviours, distribute paraphrases across
them until you reach {n} or genuinely run out of non-redundant angles.

Each paraphrase should emphasise a different aspect where possible (use as many of these as fit, up to {n}):
  [1] focus on the room(s) and specific sensor positions
  [2] focus on the duration / temporal pattern (long dwell, few transitions)
  [3] focus on the absence of movement between rooms
{tod_paraphrase}
  [5+] any other salient sensor-level variation

Rules:
- If the query is broad (e.g. "sedentary"), produce paraphrases that cover
  ALL the relevant locations identified in your reasoning — do not refuse
  because the concept spans more than one location.
- Each sentence describes sensor-observable behaviour only (no label names).{strip_note}

Format your response as:
REASONING: <your reasoning here>
SENTENCES: ["paraphrase 1", "paraphrase 2", ...]

Return a valid JSON array for SENTENCES with at most {n} strings. No extra text after the array.
"""

    if mode == "auto":
        n = max_subqueries
        return header + f"""
## Your task: adaptive query expansion

Your goal is to produce a set of retrieval sentences that together give the best possible
coverage of the user's query.

The retrieval system matches each sentence independently to embedded sensor sequences via
cosine similarity. This has an important implication:

  A sentence containing "room A or room B" does NOT retrieve both well — the embedding
  averages them out. Separate sentences for room A and for room B each retrieve their
  target precisely. The same applies to time of day: "morning or evening" is better
  expressed as two sentences, one per time window.

Therefore: whenever the user's concept naturally splits along a meaningful axis — location,
time of day, or sub-behaviour — produce one sentence per variant. When no meaningful split
exists, produce a few paraphrases that vary in wording or sensor emphasis.

## Decision guidance  (not rigid rules — use your judgment)

Paraphrase a few variants of the same idea when:
  - The activity is inherently mobile or multi-room by nature (walking, wandering, transitions
    between rooms). In this case the whole-home trajectory is the signal, not one room.
  - The activity is tied to a single specific sensor or location.
  - Location or time-of-day variants would be redundant or sensor-indistinguishable.

Split by location — one sentence per distinct plausible room or sensor cluster — when:
  - A stationary or sedentary activity could occur in several different rooms.
  - The sensor signature would look meaningfully different across those rooms.

Split by time of day — one sentence per distinct time window — only when:
  - The query has a strong temporal qualifier ("at night", "morning routine", "after dinner").
  - The activity looks or feels meaningfully different at different times of day.
  - Skip this split if the activity occurs uniformly throughout the day.

Combine strategies freely. For example: split by location AND add a few paraphrases of
each, or produce separate morning/evening variants AND anchor each to a specific room.

## Mandatory: preserve explicit temporal qualifiers

If the user's query contains an explicit time-of-day or day-of-week qualifier
(e.g. "at night", "in the morning", "on weekends", "after dinner", "nighttime hours"),
you MUST embed that qualifier in EVERY output sentence without exception.
These are not optional context — they are hard retrieval filters.
Do NOT drop or soften them (e.g. do not change "at night" to "during the evening or night").
If you are splitting by location, each location-specific sentence still carries the temporal qualifier.
If you are paraphrasing, every paraphrase still carries the temporal qualifier.

## Sentence count

Default to the minimum number of sentences that give complete coverage — usually 1–3.
Only produce more sentences when the activity has genuinely distinct variants that
a single sentence cannot cover:
  - Multiple separate sensor locations where the same activity occurs but sensors differ
    (e.g. "sitting in the living room armchair" vs. "sitting at the bedroom chair") →
    one sentence per location.
  - Distinct sub-actions each with a different sensor signature (e.g. opening the fridge
    vs. standing at the counter for a cooking query) → one sentence per sub-action.
  - A strong temporal split where the query explicitly names different time windows and
    the sensor pattern differs meaningfully between them.

Do NOT add sentences for:
  - Minor paraphrases or synonyms of the same idea.
  - The same location described with slightly different wording.
  - Filling up the budget just because slots remain.

Hard cap: never exceed {n} sentences. Prefer 1–3 unless the activity clearly warrants more.

## Output format

REASONING: Explain (1) what physical sensor pattern the query corresponds to,
           (2) what expansion strategy you chose and why — which axis you split on,
           or why paraphrasing suffices, (3) anything in the query that cannot be
           detected from the available sensors.

SENTENCES: ["sentence 1", "sentence 2", ...]

Return a valid JSON array for SENTENCES (1–{n} strings). No extra text after the array.
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

    def call(self, user_query: str, system_prompt: str, *, max_subqueries: int = 1) -> str:
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
        )
        # Scale output budget with requested sentence count (reasoning + JSON array)
        max_out = min(8192, 512 + max(1, max_subqueries) * 200)
        cfg = self._genai.GenerationConfig(
            max_output_tokens=max_out,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
        resp = model.generate_content(user_query, generation_config=cfg)
        return (resp.text or "").strip()

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

    def call(self, user_query: str, system_prompt: str, *, max_subqueries: int = 1) -> str:
        max_tok = min(4096, 256 + max(1, max_subqueries) * 160)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_tokens=max_tok,
        )
        return (response.choices[0].message.content or "").strip()

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
        self,
        user_query: str,
        mode: RewriteMode = "auto",
        strip_temporal: bool = False,
        max_subqueries: int | None = None,
    ) -> tuple[str, list[str]]:
        """
        Rewrite a user query into one or more retrieval-optimised sentences.

        Args:
            user_query:     The raw user question.
            mode:           "auto" (default) | "single" | "multi_location" | "multi_wording"
                            "auto" lets the LLM decide whether to paraphrase, split by
                            location, split by time of day, or combine strategies.
            strip_temporal: When True, instruct the LLM to omit time-of-day and
                            day-of-week constraints from output sentences (they
                            will be applied via a separate rule-based filter).
                            Has no effect in "auto" mode (the LLM decides).
            max_subqueries: Max retrieval sentences (default: 6 for auto,
                            8 for multi_location, 6 for multi_wording). Ignored
                            for single (always 1).

        Returns:
            (reasoning: str, sentences: list[str])
        """
        n = resolve_max_subqueries(mode, max_subqueries)
        system_prompt = _build_system_prompt(
            self.home_context, mode, self._example_captions,
            strip_temporal=strip_temporal,
            max_subqueries=n,
        )
        uq = user_query
        if mode == "auto":
            uq = (
                f"{user_query}\n\n"
                f"[Produce the minimum sentences needed — default to 1–3. "
                f"Only use more (up to {n}) when the activity has genuinely distinct "
                f"sensor locations or sub-actions that cannot be captured in one sentence. "
                f"Do not paraphrase the same idea in multiple ways.]"
            )
        elif mode == "multi_location":
            uq = (
                f"{user_query}\n\n"
                f"[Retrieval target: up to {n} sentences — one per distinct plausible room "
                f"from the sensor inventory; use all {n} slots when that many locations apply.]"
            )
        elif mode == "multi_wording":
            uq = (
                f"{user_query}\n\n"
                f"[Retrieval target: up to {n} distinct paraphrases; use all {n} slots when "
                f"non-redundant angles remain.]"
            )
        raw = self._backend.call(uq, system_prompt, max_subqueries=n)
        reasoning, sentences = _parse_response(raw, mode)
        if mode != "single" and len(sentences) > n:
            sentences = sentences[:n]
        return reasoning, sentences

    @property
    def model_id(self) -> str:
        return self._backend.model_id
