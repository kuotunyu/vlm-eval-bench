# Published result audit pack

This directory contains 2,000 final rows for the four complete archived model
runs. Twelve deterministic gzip JSONL files cover every model/task pair.
`run_manifest.json` records hashes, row counts, aggregate provenance, the exact
historical configuration snapshot, sample-manifest hashes, and the
deduplication rule.

## Privacy boundary

Schema version 2 contains only:

- opaque sample ID, model, task, and already-computed per-row score;
- input/output token counts and their accounting source;
- cost, uncached/cached latency state, and final error state.

It excludes dataset images, questions, prompts, references, model predictions,
raw provider responses, credentials, and cache data. Strict schema validation
rejects extra fields and non-finite or invalid numeric values.

## Verify

From the repository root:

```bash
uv sync --locked
uv run python scripts/verify_audit.py
```

The verifier checks:

- all gzip, sample-manifest, configuration, and published-artifact hashes;
- exact model/task coverage, unique sample IDs, and row counts;
- per-row field types and ranges;
- independently recomputed DocVQA/ChartQA row-score means;
- cost, usage, latency, terminal error, and valid-JSON claims;
- score intervals and numeric claims in the leaderboard and root README.

## Recomputability boundary

DocVQA and ChartQA published means are independently recomputed from the
already-computed row scores. The pack intentionally cannot rerun metric
matching because predictions and references are absent. CORD's corpus-level
micro F1, per-field values, and all bootstrap intervals require private
predictions/references, so the verifier checks their frozen aggregate
provenance rather than claiming an independent rescore.

Maintainers who lawfully retain the ignored inputs can rebuild this pack with
`uv run python scripts/build_audit_pack.py`. A rebuild must undergo a privacy
review before publication.
