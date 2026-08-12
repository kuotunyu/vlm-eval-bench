# Corrected offline rescore pack

This directory contains the corrected scoring release for the unchanged
2026-07-10 predictions: 12 deterministic gzip JSONL files, four models × three
tasks, and 2,000 latest rows. `rescore_manifest.json` records every artifact
hash, private-source SHA-256 provenance, physical/latest row counts, corrected
aggregates and confidence intervals, and archived-to-corrected deltas. It also
names out-of-scope disabled-model files found beside the complete matrix
without reading, hashing, or publishing their contents.

## Privacy boundary

Each public row contains only an opaque sample ID, model/task IDs, corrected
scalar score, token accounting, cost, latency/cache state, and final error.
It contains no dataset image, question, prompt, reference, prediction, raw
provider response, or credential.

## Verify

```bash
uv sync --locked
uv run python scripts/verify_corrected.py
```

The verifier:

- validates exact 4×3 coverage, SHA-256, strict schemas, row counts, and IDs;
- compares every corrected row with its archived counterpart and permits only
  `score` to differ;
- independently recomputes DocVQA/ChartQA mean scores and seeded bootstrap
  intervals from sanitized rows;
- verifies CORD scores and aggregate are unchanged, because the CORD metric
  implementation did not change;
- recomputes costs, tokens, latency, error rates, old/new deltas, and public
  README/leaderboard arithmetic.

CORD corpus F1 and its interval remain reference-dependent. Their values are
publicly provenance-verified but cannot be independently derived from a pack
that intentionally excludes references and model answers.
