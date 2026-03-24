# Longitudinal Analysis Module

## Overview

`src/evals/longitudinal/` provides a pipeline that runs an LLM-rewritten natural-language query through FAISS retrieval, then analyses **when** the matched sensor sequences occurred over calendar time.

```
user query
    │
    ▼
SmartQuery (LLM rewrite + FAISS)     ← see docs/QUERY_REWRITER.md
    │  results with start_time
    ▼
LongitudinalAnalyzer
  • parse timestamps from labels.start_time
  • aggregate into daily / weekly buckets
  • compute moving average + ±1σ variability band
  • KDE sensor heatmap on floor plan
  • write summaries + raw results JSON
    │
    ▼
results/long/{dataset_name}/{dataset_split}/{safe_query}/
  longitudinal.png   – time-series plot
  heatmap.png        – sensor activation density on floor plan
  scores.png         – retrieval score distribution
  summary.md         – structured markdown report
  summary.txt        – plain-text report
  results.json       – full raw retrieval output (for interactive tool)
```

---

## Module layout

```
src/evals/longitudinal/
├── __init__.py               exports LongitudinalAnalyzer
└── longitudinal_analysis.py  core class + CLI entry point
```

---

## Output files

| File | Description |
|------|-------------|
| `longitudinal.png` | Top panel: daily/weekly matched-sample counts (bars) with moving-average trend line (orange) and ±1σ variability band. Bottom strip: average cosine similarity per bucket as a colour heatmap. |
| `heatmap.png` | KDE density heatmap on the house floor plan. Each activated sensor is annotated with its ID and activation count. |
| `scores.png` | Histogram of retrieval cosine similarity scores with mean and median lines. |
| `summary.md` | Full markdown report: rewritten sentences, LLM reasoning, statistics table, activity label / room / time-of-day / day-of-week distributions. |
| `summary.txt` | Same content in plain-text format. |
| `results.json` | Complete `SmartQuery.query()` output (all results, sentences, reasoning). Intended as the data source for the future interactive browser tool. |

---

## Quick start

### Python

```python
from query import SmartQuery
from evals.longitudinal import LongitudinalAnalyzer

sq = SmartQuery.from_data_dir(
    checkpoint_path="trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt",
    data_dir="data/processed/casas/milan/FD_60",
    home="milan",
)

analyzer = LongitudinalAnalyzer(sq, dataset_name="milan", dataset_split="FD_60")

out_dir = analyzer.analyze(
    query="morning kitchen routines",
    mode="multi_location",    # single | multi_location | multi_wording
    threshold=0.10,           # None → use top_k instead
    time_window="day",        # "day" | "week"
    ma_window=7,              # moving-average width
)
print(f"Outputs in: {out_dir}")
```

### CLI — full dataset, threshold mode

```bash
conda activate discover-v2-env
python src/evals/longitudinal/longitudinal_analysis.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \
    --data_dir   data/processed/casas/milan/FD_60 \
    --home       milan \
    --split      FD_60 \
    --query      "morning kitchen routines" \
    --threshold  0.10
```

### CLI — weekly aggregation with 4-week moving average

```bash
python src/evals/longitudinal/longitudinal_analysis.py \
    --checkpoint ... \
    --data_dir   data/processed/casas/milan/FD_60 \
    --home milan --split FD_60 \
    --query "sedentary activities in living room" \
    --threshold 0.08 \
    --time_window week \
    --ma_window 4
```

### CLI — single split, top-k mode

```bash
python src/evals/longitudinal/longitudinal_analysis.py \
    --checkpoint trained_models/milan/.../best_model.pt \
    --test_data  data/processed/casas/milan/FD_60/test.json \
    --vocab      data/processed/casas/milan/FD_60/vocab.json \
    --home milan --split FD_60 \
    --query "night wandering" \
    --top_k 100
```

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Model checkpoint `.pt` |
| `--data_dir` | — | Dataset directory (train+val+test merged). Supersedes `--test_data`. |
| `--test_data` | — | Single split JSON (requires `--vocab`). |
| `--vocab` | — | `vocab.json` path (required with `--test_data`). |
| `--home` | `milan` | Dataset name for LLM rewriter and cache. |
| `--split` | `FD_60` | Split label used in output path. |
| `--query` | required | Natural-language query. |
| `--mode` | `multi_location` | Rewrite mode. |
| `--threshold` | `0.10` | Cosine similarity floor (`None` → top-k). |
| `--top_k` | `200` | Max results when threshold is unset. |
| `--max_samples` | `50 000` | Max samples loaded into FAISS. |
| `--time_window` | `day` | Aggregation bucket: `day` or `week`. |
| `--ma_window` | `7` | Moving-average / rolling-std window width. |
| `--output_base` | `results/long` | Root output directory. |
| `--force_rewrite` | false | Bypass LLM rewrite cache. |
| `--force_retrieve` | false | Bypass FAISS result cache. |

---

## Output directory structure

```
results/long/
└── {dataset_name}/          e.g. milan
    └── {dataset_split}/     e.g. FD_60
        └── {safe_query}/    e.g. morning_kitchen_routines
            ├── longitudinal.png
            ├── heatmap.png
            ├── scores.png
            ├── summary.md
            ├── summary.txt
            └── results.json
```

The `safe_query` folder name is derived from the query string (lowercase, non-alphanumeric chars removed, spaces → underscores, capped at 60 chars).

---

## Timestamp source

Timestamps are extracted from `result['labels']['start_time']`, which is populated by `SmartHomeRetrieval._get_caption_and_labels()` from `sample['metadata']['start_time']` in the processed data JSON.  All CASAS datasets (milan, aruba, cairo, kyoto) have this field.

---

## Sensor heatmap

Sensor activation coordinates come from `metadata/casas_metadata.json` (`sensor_coordinates` dict per dataset).  Only `ON` / `OPEN` events are counted.  KDE bandwidth is fixed at 0.15 (gaussian_kde).  Falls back to scatter plot if fewer than 3 activations are found.

---

## Next step: interactive browser tool

`results.json` stores the complete serialised `SmartQuery` output and is designed as the data source for a future interactive browser dashboard that will display all the same statistics dynamically.
