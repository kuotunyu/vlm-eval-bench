# vlm-eval-bench — archived leaderboard

Run date 2026-07-10 · seed 3407 · DocVQA n=200 · ChartQA n=200 · CORD n=100 ·
greedy decoding · fixed samples · percentile bootstrap n=2,000.

The score labels describe the historical implementations used by this run.
DocVQA/ChartQA edge cases did not exactly match current official reference
semantics; see [`../EVALUATION_CARD.md`](../EVALUATION_CARD.md). Values below
have not been silently rescored.

## Scores

| Model | DocVQA ANLS | ChartQA RelaxedAcc | CORD-v2 F1 | Avg |
|---|---|---|---|---|
| qwen3vl-8b-base † | 0.937 [0.905, 0.964] | 0.830 [0.775, 0.880] | 0.742 [0.704, 0.772] | 0.836 |
| qwen3vl-8b-receipt-qlora † | 0.921 [0.887, 0.952] | 0.835 [0.780, 0.885] | 0.923 [0.891, 0.950] | 0.893 |
| gemini-3.1-flash-lite | 0.882 [0.840, 0.921] | 0.595 [0.525, 0.665] | 0.870 [0.846, 0.893] | 0.782 |
| gpt-5.4-mini | 0.862 [0.817, 0.903] | 0.550 [0.480, 0.620] | 0.819 [0.789, 0.847] | 0.744 |

ChartQA provenance-only split aggregates: qwen3vl-8b-base human=0.71,
machine=0.95; qwen3vl-8b-receipt-qlora human=0.72, machine=0.95;
gemini-3.1-flash-lite human=0.65, machine=0.54; gpt-5.4-mini human=0.60,
machine=0.50.

† Local model on RTX 4090 under WSL2, 4-bit, batch size 1. `Avg` is an
unweighted arithmetic mean used only for navigation across non-equivalent
metrics.

## Cost

| Model | Total $ | $/100 questions | Input tokens | Output tokens |
|---|---|---|---|---|
| qwen3vl-8b-base † | 0.1207 † | 0.0241 † | 436,373 | 19,455 |
| qwen3vl-8b-receipt-qlora † | 0.1194 † | 0.0239 † | 436,373 | 13,371 |
| gemini-3.1-flash-lite | 0.1731 | 0.0346 | 565,626 | 21,149 |
| gpt-5.4-mini | 0.4810 | 0.0962 | 527,260 | 13,151 |

† Local dollars are imputed as measured inference time × $0.35/hour; they are
not API bills. API dollars are configured-price estimates reconstructed from
provider-returned usage, not invoice-reconciled charges. The historical OpenAI
configuration separately priced an inferred image-token share; the value is
retained as run provenance, not represented as current pricing.

## Latency (seconds per question, uncached calls only)

| Model | Task | Mean | p50 | p95 |
|---|---|---|---|---|
| qwen3vl-8b-base † | docvqa | 0.85 | 0.55 | 1.02 |
| qwen3vl-8b-base † | chartqa | 0.32 | 0.32 | 0.49 |
| qwen3vl-8b-base † | cord | 9.53 | 7.56 | 23.17 |
| qwen3vl-8b-receipt-qlora † | docvqa | 0.95 | 0.74 | 1.54 |
| qwen3vl-8b-receipt-qlora † | chartqa | 0.52 | 0.50 | 0.76 |
| qwen3vl-8b-receipt-qlora † | cord | 9.34 | 7.56 | 21.53 |
| gemini-3.1-flash-lite | docvqa | 1.26 | 1.17 | 1.65 |
| gemini-3.1-flash-lite | chartqa | 1.07 | 1.03 | 1.32 |
| gemini-3.1-flash-lite | cord | 1.71 | 1.60 | 2.41 |
| gpt-5.4-mini | docvqa | 1.52 | 1.24 | 3.08 |
| gpt-5.4-mini | chartqa | 1.25 | 1.08 | 2.77 |
| gpt-5.4-mini | cord | 1.63 | 1.47 | 2.62 |

Local latency is batch-1 inference without network time and is not comparable
with API round trips.

## Reliability

| Model | Error rate | CORD valid-JSON rate |
|---|---|---|
| qwen3vl-8b-base | 0.00% | 100.00% |
| qwen3vl-8b-receipt-qlora | 0.00% | 100.00% |
| gemini-3.1-flash-lite | 0.00% | 100.00% |
| gpt-5.4-mini | 0.00% | 100.00% |

Error rate covers final deduplicated rows only. Transient retry attempts are
not retained in this statistic.

## Analysis

The adapter's measured change against the same base was CORD +0.181, DocVQA
-0.015, and ChartQA +0.005. This is consistent with a narrow receipt-domain
adaptation, not broad improvement. Fixed sample sizes and metric caveats rule
out generalizing these results to production traffic or unrelated benchmarks.

Among the two complete API runs, Gemini had the higher cross-task arithmetic
mean and lower reconstructed cost. That comparison is limited to these fixed
samples and frozen configurations; it is not a current price/performance claim.

## Charts

![Scores by task with archived confidence intervals](charts/scores_by_task.png)

![Uncached latency](charts/latency_p50_p95.png)

![CORD field F1 for the two local models](charts/cord_f1_breakdown.png)
