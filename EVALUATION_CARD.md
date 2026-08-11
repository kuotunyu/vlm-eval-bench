# Evaluation card: archived 2026-07-10 run

## Status and intended use

This is an auditable historical comparison produced by a local evaluation
harness. It is useful for examining the measured effect of a receipt-specific
QLoRA adapter and for testing the repository's evaluation infrastructure. It is
not a universal model ranking, production SLA, current pricing comparison, or
official benchmark submission.

The public repository was reconstructed with a clean lineage from reviewed
content at source commit `91e4fee4d1ef63046e4ee1375332c538a8b1f2cd`.
No evaluation was rerun during publication hardening.

## Evaluation matrix

| Dimension | Frozen value |
|---|---|
| Run date | 2026-07-10 |
| Seed | 3407 |
| Bootstrap | percentile, 2,000 iterations |
| DocVQA | validation split, 200 fixed indices |
| ChartQA | test split, 100 human + 100 machine fixed indices |
| CORD-v2 | test split, all 100 rows |
| Local execution | RTX 4090, WSL2, 4-bit, batch size 1 |
| Decoding | temperature 0 / greedy; provider reasoning minimized where supported |
| Image policy | EXIF transpose, RGB, max side 1,280 px, LANCZOS, JPEG quality 90 |

Exact indices are committed under `data/samples/`. The historical configuration
is frozen at `results/audit/evaluation_config.yaml` and SHA-256-bound from the
audit manifest. The root `config.yaml` is for future runs and may intentionally
differ.

## Models

- `qwen3vl-8b-base`: local 4-bit Qwen3-VL-8B baseline.
- `qwen3vl-8b-receipt-qlora`: receipt adapter trained on CORD-v2 training data.
- `gemini-3.1-flash-lite`: complete API run.
- `gpt-5.4-mini`: complete API run.

Disabled or incomplete provider configurations are excluded from every public
comparison.

## Metric semantics

The archived labels are retained for provenance, but two implementations had
edge-case deviations from the official references:

| Metric | Archived-run behavior | Current future-run behavior |
|---|---|---|
| DocVQA ANLS | lowercased and collapsed whitespace; similarity at exactly 0.5 was retained | lowercases with edge trimming only; similarity must be strictly greater than 0.5 |
| ChartQA relaxed accuracy | normalized currency punctuation/commas and treated numeric edge cases more permissively | follows Pix2Struct reference behavior: percentage conversion, 5% relative tolerance for nonzero targets, otherwise case-insensitive exact text |
| CORD-v2 F1 | corpus-level field micro F1 with line-item matching | unchanged |

The current behavior is regression-tested against the
[Pix2Struct metric reference](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py)
and the [DocVQA challenge definition](https://www.docvqa.org/challenges/2020).
Because public artifacts exclude predictions and references, the archived
scores cannot be safely rescored here. They are disclosed as historical
implementation results rather than silently changed.

## Uncertainty and aggregation

Each task score carries a frozen 95% percentile-bootstrap interval. DocVQA and
ChartQA use means of per-sample scores. CORD uses reference-dependent
corpus-level micro F1 and a reference-dependent bootstrap. The unweighted
three-task arithmetic mean is displayed only as a navigation aid; averaging
unlike metrics does not produce a universal quality scale.

The adapter's archived deltas versus its base model were:

| Task | Delta |
|---|---:|
| CORD-v2 | +0.181 |
| DocVQA | -0.015 |
| ChartQA | +0.005 |

CORD is in-domain because the adapter was trained on its training split.
DocVQA and ChartQA are limited out-of-domain checks, not comprehensive tests of
catastrophic forgetting.

## Cost, usage, latency, and reliability

API costs were reconstructed from provider-returned usage using the frozen
configuration prices, not reconciled to invoices. The historical OpenAI config
used a separately priced, inferred image-token share. Current model
documentation describes images as input tokens, so the archived value remains
a historical configured-price estimate; no offline correction was applied.

Local dollars are imputed from inference wall time × $0.35/hour. They are not
API charges. Local tokenizer counts are not provider-billed usage. Local
batch-1 latency excludes network time, while API latency includes round trips;
the two categories are not directly comparable.

The 0% error figures cover final deduplicated rows. Retried transient failures
are not represented. The source hashes document ignored raw prediction files,
but those private files are neither required nor available in the public pack.

## Public evidence and independent checks

| Claim/artifact | Publicly recomputable? | Boundary |
|---|---|---|
| File integrity and coverage | Yes | SHA-256, exact schemas, counts, unique IDs |
| DocVQA/ChartQA published means | Yes | recomputed from published row scores, not re-matched from answers |
| CORD aggregate and field F1 | No; provenance-checked | references/predictions intentionally excluded |
| Bootstrap intervals | No; provenance-checked | requires reference-dependent aggregate inputs |
| Cost, token, latency, final errors | Yes | recomputed from audit rows |
| README/leaderboard tables and charts | Integrity-checked | hashes bind published artifacts to the manifest |

Run `uv run python scripts/verify_audit.py` for numeric and provenance checks
and `uv run python scripts/verify_release.py` for the public-tree privacy scan.

## Limitations

- Small fixed samples may not represent full datasets or production traffic.
- Results are tied to the recorded model identifiers, prompts, image pipeline,
  and provider behavior at run time; API aliases can change.
- The public pack prioritizes privacy/licensing over independent rescoring.
- No invoice evidence or raw retry-attempt log is published.
- Dataset licenses and provider terms are external to the MIT-licensed harness.
