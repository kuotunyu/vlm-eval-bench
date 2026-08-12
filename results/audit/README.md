# Archived 2026-07-10 implementation evidence

This directory preserves the original implementation results: 12 deterministic
gzip JSONL files, four models × three tasks, and 2,000 final rows.
`run_manifest.json` records source hashes, row counts, frozen aggregates,
sample/config hashes, and deduplication semantics. These scores are historical
evidence and were not overwritten by the corrected release.

The original human-readable values remain in
[`../archived_leaderboard.md`](../archived_leaderboard.md). The corrected
scoring release is separate under [`../corrected/`](../corrected/).

## Privacy boundary

Schema version 2 contains only opaque IDs, model/task, original computed score,
token accounting, cost, latency/cache state, and final error. It excludes
dataset images, questions, prompts, references, model predictions, provider
responses, credentials, and cache data.

## Verify

```bash
uv sync --locked
uv run python scripts/verify_audit.py
```

The archived verifier validates hashes, coverage, strict schemas, row counts,
cost/usage/latency/error statistics, and the archived leaderboard. DocVQA and
ChartQA original means are recomputed from original public row scores. CORD and
all reference-dependent intervals are provenance-checked.

For the corrected release and old/new comparisons, run:

```bash
uv run python scripts/verify_corrected.py
```
