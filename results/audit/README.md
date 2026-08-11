# Published result audit pack

This directory contains the 2,000 final rows behind the four complete model
runs. Each model/task pair is a deterministic gzip JSONL file; `run_manifest.json`
records file hashes, source hashes, sample-manifest/config hashes, row counts,
aggregate values, and the exact deduplication rule.

To keep redistribution within a conservative license boundary, the pack contains
only opaque sample IDs, model outputs, already-computed scores, usage/cost/latency,
and errors. It deliberately does **not** redistribute dataset images, questions,
prompts, or references. Dataset attribution and licenses are linked from the root
README.

From a fresh clone, verify every file hash, row count, public statistic, and the
score/cost/token values printed in the committed leaderboard:

```bash
uv sync --locked                 # API/core dependencies only; no local GPU extra
uv run python scripts/verify_audit.py
```

DocVQA ANLS and ChartQA relaxed accuracy are means of the published per-row
scores, so the verifier recomputes them independently. CORD's published metric is
corpus-level micro F1 with a reference-dependent bootstrap interval. Because the
references are intentionally absent, the verifier checks that value against the
aggregate recorded during export and its raw-source SHA-256 provenance; it does
not claim to independently rescore CORD.
