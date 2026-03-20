# LLM Query Rewriter

## Overview

`src/query/` adds an LLM intermediary layer between the user and the FAISS-based
sensor-embedding retrieval system.

```
user query
    │
    ▼
QueryCache ──(hit)──────────────────────────────────────────┐
    │ (miss)                                                  │
    ▼                                                         │
LLMRewriter                                                   │
  • Gemini / OpenAI-compatible backend                        │
  • System prompt grounded in house sensor layout             │
  • Rewrites vague queries into descriptive sentences         │
    matching the CLIP training caption style                  │
    │                                                         │
    ├──(store in cache)                                       │
    │                                                         │
    ▼                                                         │
rewritten sentence ◄─────────────────────────────────────────┘
    │
    ▼
SmartHomeRetrieval.query()
  • text encoder → embedding
  • FAISS IndexFlatIP search
  • returns top-k sensor sequences
```

---

## Module layout

```
src/query/
├── __init__.py        exports SmartQuery, LLMRewriter, QueryCache
├── llm_rewriter.py    Gemini / OpenAI backends + system-prompt builder
├── query_cache.py     SQLite cache  (data/query_cache/queries.db)
└── smart_query.py     Orchestrator + CLI entry point

data/query_cache/
└── queries.db         Auto-created SQLite database
```

---

## Why LLM rewriting?

The text encoder was trained on rich, style-specific captions like:

> "Sedentary activity takes place when a person sits quietly in the living room
>  armchair or office chair or dining table with minimal movement for extended periods"

A short user query like `"sedentary activities"` lands in a very different region of
the embedding space.  The LLM rewrites it into the same descriptive style, grounded
in the concrete room names and sensor layout of the target home — dramatically
improving retrieval precision.

---

## Quick start

### Python

```python
from query import SmartQuery

sq = SmartQuery.from_checkpoint(
    checkpoint_path="trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt",
    test_data_path="data/processed/casas/milan/FD_60/test.json",
    vocab_path="data/processed/casas/milan/FD_60/vocab.json",
    home="milan",
    gemini_api_key="YOUR_KEY",   # or set GEMINI_API_KEY env var
)

out = sq.query("give me all sedentary activities", top_k=10)
# out["original_query"]   → "give me all sedentary activities"
# out["rewritten_query"]  → "Sedentary activity takes place when ..."
# out["cache_hit"]        → False (first call)
# out["results"]          → list of dicts from SmartHomeRetrieval
```

### CLI — single query

```bash
conda activate discover-v2-env
python src/query/smart_query.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \
    --test_data  data/processed/casas/milan/FD_60/test.json \
    --vocab      data/processed/casas/milan/FD_60/vocab.json \
    --home       milan \
    --query      "give me all sedentary activities" \
    --top_k      10
```

### CLI — interactive mode

```bash
python src/query/smart_query.py \
    --checkpoint ... --test_data ... --vocab ... --home milan
# Enter queries at the prompt; type 'cache' for stats, 'quit' to exit.
```

### CLI — rewrite only (no model loading)

```bash
python src/query/smart_query.py --rewrite_only \
    --home milan \
    --query "cooking at night"
# Original : cooking at night
# Rewritten: Kitchen Activity takes place when a person ...
```

### Cache management

```bash
# Show statistics
python src/query/smart_query.py --cache_stats

# List all cached rewrites for milan
python src/query/smart_query.py --list_cache --home milan
```

---

## LLM backends

| Backend | Install | Key env var | `--llm_backend` |
|---------|---------|-------------|-----------------|
| Gemini (default) | `pip install google-generativeai` | `GEMINI_API_KEY` | `gemini` |
| OpenAI / compatible | `pip install openai` | `OPENAI_API_KEY` | `openai` |

For local models (Ollama, vLLM, LM Studio) use `--llm_backend openai` with
`--llm_model <name>` and set `base_url` in code.

---

## Cache

The SQLite database at `data/query_cache/queries.db` stores:

| Column | Description |
|--------|-------------|
| `original_query` | Raw user input |
| `home` | Dataset name (cache namespace) |
| `rewritten_query` | LLM output |
| `model_used` | Model that produced the rewrite |
| `created_at` | ISO timestamp |
| `hit_count` | Number of times reused without an LLM call |

The unique key is `(original_query, home)` — so the same phrasing can have
different rewrites for Milan vs Aruba (different room layouts).

---

## Extending

To add a new LLM backend, implement a class with:
```python
class MyBackend:
    def rewrite(self, user_query: str, system_prompt: str) -> str: ...
    @property
    def model_id(self) -> str: ...
```
and register it in `LLMRewriter.__init__`.
