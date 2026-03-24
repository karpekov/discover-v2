# LLM Query Rewriter

## Overview

`src/query/` adds an LLM intermediary layer between the user and the FAISS-based
sensor-embedding retrieval system.  The user writes a natural-language question;
the LLM rewrites it into one or more retrieval-optimised sentences; FAISS returns
the matching sensor sequences.

```
user query
    │
    ▼
QueryCache (rewrite) ──(hit)────────────────────────────────────┐
    │ (miss)                                                      │
    ▼                                                             │
LLMRewriter                                                       │
  • Gemini 2.5 Flash (default) / OpenAI-compatible               │
  • Sensor-only system prompt (no activity labels)               │
  • Mandatory reasoning: is this detectable from sensors?        │
  • Rewrites into 1–6 retrieval sentences                        │
  • Style examples sampled from training captions                │
    │                                                             │
    ├──(store in rewrite cache)                                   │
    ▼                                                             │
rewritten sentences ◄────────────────────────────────────────────┘
    │
    ▼
QueryCache (results) ──(hit)────────────────────────────────────┐
    │ (miss)                                                      │
    ▼                                                             │
FAISS retrieval  (one search per sentence → merge by best score) │
  • text encoder → MLP projection → L2 normalise                 │
  • IndexFlatIP cosine similarity                                 │
  • top-k mode  OR  threshold mode (range_search)                │
    │                                                             │
    ├──(store in result cache)                                    │
    ▼                                                             │
ranked results ◄─────────────────────────────────────────────────┘
  • top 10 printed to terminal
  • full set in returned dict + SQLite cache
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
├── queries.db                        SQLite database (rewrite + result cache)
└── merged/
    ├── data_<hash>.json              Merged split files (auto-generated)
    └── captions_<style>_<hash>.json  Merged caption files (auto-generated)
```

Merged files in `data/query_cache/merged/` are content-addressed by an mtime hash
of their source files — stale entries are never used, fresh ones are always reused.

---

## Why LLM rewriting?

The text encoder was trained on rich, style-specific captions like:

> "On wednesday in the afternoon, resident was active for about 5 minutes in
>  living room, with sustained presence near the armchair."

A short user query like `"sedentary activities"` lands in a very different region
of the embedding space.  The LLM rewrites it into the same descriptive style,
grounded in the concrete sensor layout of the target home.

---

## Rewrite modes

| Mode | LLM output | FAISS searches | Best for |
|------|-----------|---------------|----------|
| `single` | 1 broad sentence covering all variants | 1 | Quick broad sweep |
| `multi_location` | 1 sentence per relevant room/sensor position | N | Precise, location-anchored retrieval |
| `multi_wording` | 4–6 paraphrases of the same concept | N | Robust to embedding sensitivity to phrasing |

For multi modes, results from all sub-queries are merged by best cosine score per
sample.  Each result shows which sub-query matched it and all individual scores.

---

## Retrieval modes

| Mode | Flag | Behaviour |
|------|------|-----------|
| Top-k | `--top_k N` | Returns the N highest-scoring samples |
| Threshold | `--threshold 0.10` | Returns ALL samples with score ≥ threshold (no cap) |

Threshold mode uses FAISS `range_search()` — exact, no approximation.

Score interpretation (cosine similarity, range −1 to 1):

| Score | Quality |
|-------|---------|
| > 0.20 | Excellent |
| 0.10–0.20 | Decent |
| < 0.05 | Very poor |

Terminal output is always capped at **10 results**.  The full set is available in
the returned dict (`out["results"]`) and the SQLite result cache.

---

## System prompt design

The LLM receives **only**:
- Sensor IDs grouped by room, with type (motion / door / temperature) and
  physical detail description (e.g. "armchair", "working desk", "entrance door")
- N random style examples sampled from the actual training captions (default: 6)
- Sensor behaviour reference (what each sensor type physically detects)

The LLM does **not** see:
- Activity labels or category names (`label_to_text_sourish`)
- Home name or number of residents
- Any other dataset metadata

The prompt enforces a mandatory **reasoning step** before generating sentences:
1. Which sensors / rooms are relevant?
2. What sensor pattern would this activity produce?
3. Is it actually detectable from available sensors?  If NO → output `[]`

Output sentences must describe only observable physical behaviour — no label names,
no sensor IDs (M003 etc.), no dataset-specific terms.

---

## Two-level SQLite cache

Both tables live in `data/query_cache/queries.db`.

### `query_cache` — rewrite cache

| Column | Description |
|--------|-------------|
| `original_query` | Raw user input |
| `home` | Dataset name |
| `rewrite_mode` | `single` / `multi_location` / `multi_wording` |
| `sentences` | JSON list of rewritten sentences |
| `model_used` | LLM that produced the rewrite |
| `created_at` | ISO timestamp |
| `hit_count` | Times reused without an LLM call |

Unique key: `(original_query, home, rewrite_mode)`

### `result_cache` — retrieval result cache

| Column | Description |
|--------|-------------|
| `original_query` | Raw user input |
| `home` | Dataset name |
| `rewrite_mode` | Rewrite mode used |
| `checkpoint` | Checkpoint filename (model identifier) |
| `top_k` | Top-k used (`-1` for threshold mode) |
| `results_json` | Full ranked result list as JSON |
| `created_at` | ISO timestamp |
| `hit_count` | Times served from cache |

Unique key: `(original_query, home, rewrite_mode, checkpoint, top_k)`

Cache bypass flags: `--force_rewrite` (LLM), `--force_retrieve` (FAISS).

---

## Data loading

Two factory methods are available depending on how much data you want to index.

### Single split (test only)

```python
sq = SmartQuery.from_checkpoint(
    checkpoint_path="trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt",
    test_data_path="data/processed/casas/milan/FD_60/test.json",
    vocab_path="data/processed/casas/milan/FD_60/vocab.json",
    home="milan",
)
```

### Full dataset — all splits merged (recommended for longitudinal analysis)

```python
sq = SmartQuery.from_data_dir(
    checkpoint_path="trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt",
    data_dir="data/processed/casas/milan/FD_60",
    home="milan",
    # splits=["train", "val", "test"]  ← default
    # caption_style="baseline"         ← default
)
```

`from_data_dir` merges train + val + test (42 829 samples for Milan FD_60),
writes the combined files to `data/query_cache/merged/` and reuses them on
subsequent runs (invalidated automatically when source files change).

---

## Quick start

### Python

```python
from query import SmartQuery

# Full dataset
sq = SmartQuery.from_data_dir(
    checkpoint_path="trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt",
    data_dir="data/processed/casas/milan/FD_60",
    home="milan",
    # API key from GOOGLE_API_KEY / GEMINI_API_KEY env var, or pass explicitly
)

# Top-k mode
out = sq.query("give me all sedentary activities", mode="multi_location", top_k=20)

# Threshold mode — retrieve ALL samples above a similarity floor (no cap)
out = sq.query("give me all sedentary activities", mode="multi_location", threshold=0.10)

# out keys:
#   original_query, rewrite_mode, reasoning, sentences,
#   rewrite_cache_hit, result_cache_hit, model_used, results
print(f"{len(out['results'])} results retrieved")
# Terminal always shows at most 10; full set is in out["results"] and the cache
```

### CLI — full dataset, threshold mode

```bash
conda activate discover-v2-env
python src/query/smart_query.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \
    --data_dir   data/processed/casas/milan/FD_60 \
    --home       milan \
    --query      "give me all sedentary activities" \
    --mode       multi_location \
    --threshold  0.10
```

### CLI — single split, top-k

```bash
python src/query/smart_query.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \
    --test_data  data/processed/casas/milan/FD_60/test.json \
    --vocab      data/processed/casas/milan/FD_60/vocab.json \
    --home       milan \
    --query      "give me all sedentary activities" \
    --mode       multi_location \
    --top_k      20
```

### CLI — rewrite only (no model loading)

```bash
python src/query/smart_query.py --rewrite_only \
    --home milan --mode multi_location \
    --query "give me all sedentary activities"
# Shows LLM reasoning + generated sentences; no FAISS involved
```

### Cache management

```bash
# Statistics for both rewrite and result cache
python src/query/smart_query.py --cache_stats

# List all cached rewrites for milan
python src/query/smart_query.py --list_cache --home milan

# Force fresh LLM rewrite (overwrites rewrite cache)
python src/query/smart_query.py ... --force_rewrite

# Force fresh FAISS retrieval (overwrites result cache)
python src/query/smart_query.py ... --force_retrieve
```

### Style examples from training data

```bash
# Custom captions file
python src/query/smart_query.py ... \
    --llm_captions_path data/processed/casas/milan/FD_60/train_captions_rb_and_llm.json \
    --llm_n_examples 10

# Disable examples entirely
python src/query/smart_query.py ... --llm_captions_path none
```

Default: 6 random captions from `data/processed/casas/{home}/FD_60/train_captions_baseline.json`.

---

## LLM backends

| Backend | Install | Key env var | `--llm_backend` |
|---------|---------|-------------|-----------------|
| Gemini 2.5 Flash (default) | `pip install google-generativeai` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini` |
| OpenAI / compatible | `pip install openai` | `OPENAI_API_KEY` | `openai` |

For local models (Ollama, vLLM, LM Studio) use `--llm_backend openai` with
`--llm_model <name>` and `base_url=` in code.

---

## Extending

To add a new LLM backend implement:
```python
class MyBackend:
    def call(self, user_query: str, system_prompt: str) -> str: ...
    @property
    def model_id(self) -> str: ...
```
and register it in `LLMRewriter.__init__`.
