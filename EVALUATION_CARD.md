# Evaluation card: corrected rescore of the 2026-07-10 run

## Status

This release is an offline corrected rescore of previously recorded model
outputs. It did not run a model, access a dataset, call a provider, or change
inference/accounting evidence. It is useful for auditing the measured effect
of a receipt-specific QLoRA adapter and the harness's scoring behavior. It is
not a universal model ranking, production SLA, current pricing comparison, or
official benchmark submission.

The public repository's clean lineage originated from reviewed content at
source commit `91e4fee4d1ef63046e4ee1375332c538a8b1f2cd`. The corrected rescore
uses private JSONL files whose SHA-256 values already appear in the archived
manifest; no private path or content is published.

## Frozen evaluation matrix

| Dimension | Value |
|---|---|
| Inference date | 2026-07-10 |
| Rescore type | scoring-only; predictions unchanged |
| Seed | 3407 |
| Bootstrap | percentile, 2,000 iterations |
| DocVQA | validation, 200 fixed indices |
| ChartQA | test, 100 human + 100 machine fixed indices |
| CORD-v2 | test, all 100 rows |
| Local execution | RTX 4090, WSL2, 4-bit, batch size 1 |
| Image policy | EXIF transpose, RGB, max side 1,280 px, LANCZOS, JPEG quality 90 |

Exact indices are in `data/samples/`; the historical run configuration is
SHA-256-bound at `results/audit/evaluation_config.yaml`.

## Corrected metric semantics

| Metric | Archived implementation | Corrected implementation |
|---|---|---|
| DocVQA ANLS | lowercased and collapsed internal whitespace; retained similarity exactly 0.5 | case-insensitive but space-sensitive; normalized distance must be strictly below 0.5 |
| ChartQA relaxed accuracy | stripped `%` without converting to a fraction; removed currency/thousands punctuation; normalized whitespace | Pix2Struct behavior: `%` converts to a fraction, 5% tolerance applies to nonzero numeric targets, otherwise case-insensitive exact text |
| CORD-v2 F1 | corpus-level field micro F1 with line-item alignment | unchanged |

The implementation and reference tests follow the
[DocVQA challenge definition](https://www.docvqa.org/challenges/2020) and
[Pix2Struct source](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py).

## What changed

| Model | DocVQA delta | ChartQA delta | CORD delta |
|---|---:|---:|---:|
| qwen3vl-8b-base | -0.0050 | 0.0000 | 0.0000 |
| qwen3vl-8b-receipt-qlora | -0.0026 | -0.0050 | 0.0000 |
| gemini-3.1-flash-lite | -0.0026 | -0.2200 | 0.0000 |
| gpt-5.4-mini | -0.0075 | -0.2600 | 0.0000 |

The API ChartQA changes are driven mainly by percentage conversion and by
retaining currency/thousands punctuation. They are not provider regressions:
the predictions are byte-for-byte the same private evidence. DocVQA changed
only 2–4 rows per model. CORD did not change at all.

Under corrected scoring, the adapter's deltas versus the same base model are
CORD +0.1811, DocVQA -0.0128, and ChartQA 0.0000. The conclusion remains a
large receipt-domain benefit with no evidence of broad improvement.

## Aggregation and uncertainty

DocVQA and ChartQA are means of corrected per-sample scores; their 95%
intervals are deterministically recomputed from the public corrected rows.
CORD is reference-dependent corpus micro F1, and its bootstrap resamples the
private reference/prediction structures. The corrected release therefore
proves CORD remained identical but does not claim an independent public
rescore. The three-task arithmetic mean is a navigation aid only.

## Evidence flow

1. The CLI verifies each private source SHA-256 against the archived manifest.
2. Last-row-wins deduplication retains first-seen sample order.
3. Every file must have its exact full-run row count, model/task identity, and
   the same sample-ID set as the archived public pack.
4. Only scores are recomputed; usage, cost, latency, cache, and errors pass
   through unchanged.
5. A strict serializer emits only privacy-safe corrected rows.
6. The public verifier compares corrected and archived packs row by row and
   validates all published claims.

Disabled Gemini 2.5 evidence was not included: its model/task matrix was
incomplete and ended in terminal quota errors. The pipeline fails closed on
any missing or partial enabled model/task matrix.

## Public recomputability

| Claim | Publicly recomputable? | Boundary |
|---|---|---|
| Corrected file integrity/coverage | Yes | hashes, strict schema, row counts, unique IDs |
| Accounting unchanged | Yes | row-by-row comparison with archived pack |
| DocVQA/ChartQA mean and CI | Yes | corrected scalar rows plus frozen seed/iterations |
| CORD aggregate, fields, CI | No; provenance-verified unchanged | references/predictions intentionally excluded |
| Historical/corrected deltas | Yes | both manifests and corrected rows |
| README/leaderboard arithmetic | Yes | checked by `verify_corrected.py` |

## Cost, latency, and reliability

API dollars remain historical configured-price estimates based on
provider-returned usage, not invoice evidence. Local dollars remain an imputed
RTX 4090 rental equivalent. Local batch-1 latency excludes network time and is
not comparable with API round trips. The 0% error rate covers terminal latest
rows, not transient attempts that were retried.

## Limitations

- Fixed samples may not represent full datasets or production traffic.
- The public boundary prevents independent answer matching and CORD rescoring.
- API aliases and behavior may have changed since inference time.
- No new models, datasets, or inference runs were added during correction.
