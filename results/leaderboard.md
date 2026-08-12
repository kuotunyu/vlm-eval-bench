# vlm-eval-bench — corrected offline rescore

The values below rescore the unchanged 2026-07-10 predictions with corrected
DocVQA and ChartQA metric semantics. No model inference, API request, dataset
access, prompt, response, usage, cost, or latency measurement was repeated.

Seed 3407 · DocVQA n=200 · ChartQA n=200 · CORD n=100 · fixed samples ·
percentile bootstrap n=2,000. The original implementation results remain in
the [archived leaderboard](archived_leaderboard.md).

## Corrected scores

| Model | DocVQA ANLS | ChartQA RelaxedAcc | CORD-v2 F1 | Avg |
|---|---|---|---|---|
| qwen3vl-8b-base † | 0.9316 [0.8967, 0.9608] | 0.8300 [0.7750, 0.8800] | 0.7423 [0.7037, 0.7719] | 0.8346 |
| qwen3vl-8b-receipt-qlora † | 0.9188 [0.8823, 0.9507] | 0.8300 [0.7750, 0.8800] | 0.9234 [0.8915, 0.9500] | 0.8907 |
| gemini-3.1-flash-lite | 0.8795 [0.8361, 0.9180] | 0.3750 [0.3150, 0.4400] | 0.8701 [0.8460, 0.8928] | 0.7082 |
| gpt-5.4-mini | 0.8546 [0.8069, 0.8974] | 0.2900 [0.2300, 0.3550] | 0.8193 [0.7892, 0.8469] | 0.6546 |

Corrected ChartQA split aggregates, retained as private-evidence-derived
provenance: qwen3vl-8b-base human=0.71, machine=0.95;
qwen3vl-8b-receipt-qlora human=0.71, machine=0.95;
gemini-3.1-flash-lite human=0.41, machine=0.34; gpt-5.4-mini human=0.32,
machine=0.26.

† Local model on RTX 4090 under WSL2, 4-bit, batch size 1. `Avg` is an
unweighted arithmetic mean used only to navigate non-equivalent metrics.

## Historical vs corrected

| Model / task | Archived | Corrected | Delta | Changed rows |
|---|---:|---:|---:|---:|
| qwen3vl-8b-base / docvqa | 0.9366 | 0.9316 | -0.0050 | 3 |
| qwen3vl-8b-base / chartqa | 0.8300 | 0.8300 | 0.0000 | 0 |
| qwen3vl-8b-base / cord | 0.7423 | 0.7423 | 0.0000 | 0 |
| qwen3vl-8b-receipt-qlora / docvqa | 0.9214 | 0.9188 | -0.0026 | 2 |
| qwen3vl-8b-receipt-qlora / chartqa | 0.8350 | 0.8300 | -0.0050 | 1 |
| qwen3vl-8b-receipt-qlora / cord | 0.9234 | 0.9234 | 0.0000 | 0 |
| gemini-3.1-flash-lite / docvqa | 0.8821 | 0.8795 | -0.0026 | 2 |
| gemini-3.1-flash-lite / chartqa | 0.5950 | 0.3750 | -0.2200 | 46 |
| gemini-3.1-flash-lite / cord | 0.8701 | 0.8701 | 0.0000 | 0 |
| gpt-5.4-mini / docvqa | 0.8621 | 0.8546 | -0.0075 | 4 |
| gpt-5.4-mini / chartqa | 0.5500 | 0.2900 | -0.2600 | 52 |
| gpt-5.4-mini / cord | 0.8193 | 0.8193 | 0.0000 | 0 |

The changed-row counts identify scoring decisions, not new answers. Most API
ChartQA changes came from the reference behavior that converts a trailing
percentage to a fraction instead of merely stripping `%`; currency/thousands
punctuation also remains significant. DocVQA changes come from preserving
internal spaces and rejecting normalized distance exactly 0.5.

## Cost

| Model | Total $ | $/100 questions | Input tokens | Output tokens |
|---|---|---|---|---|
| qwen3vl-8b-base † | 0.1207 † | 0.0241 † | 436,373 | 19,455 |
| qwen3vl-8b-receipt-qlora † | 0.1194 † | 0.0239 † | 436,373 | 13,371 |
| gemini-3.1-flash-lite | 0.1731 | 0.0346 | 565,626 | 21,149 |
| gpt-5.4-mini | 0.4810 | 0.0962 | 527,260 | 13,151 |

Costs and usage are unchanged historical accounting. Local dollars are
imputed as measured inference time × $0.35/hour, not bills. API dollars are
configured-price estimates reconstructed from provider-returned usage, not
invoice-reconciled charges. The historical OpenAI configuration separately
priced an inferred image-token share; this is retained as provenance rather
than presented as current pricing.

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

Local latency is batch-1 inference without network time and cannot be compared
directly with API round trips.

## Reliability

| Model | Error rate | CORD valid-JSON rate |
|---|---|---|
| qwen3vl-8b-base | 0.00% | 100.00% |
| qwen3vl-8b-receipt-qlora | 0.00% | 100.00% |
| gemini-3.1-flash-lite | 0.00% | 100.00% |
| gpt-5.4-mini | 0.00% | 100.00% |

Error rate covers final deduplicated rows only. Transient retry attempts are
not represented.

## Interpretation

Under corrected scoring, the receipt adapter changes CORD by **+0.181**,
DocVQA by **-0.013**, and ChartQA by **0.000** versus the same 8B base. The
large, narrow CORD improvement remains the strongest result; CORD is in-domain
because the adapter was trained on its training split. DocVQA and ChartQA are
limited out-of-domain checks.

The API ChartQA correction is a metric-semantic correction, not evidence that
the providers changed or that new calls performed worse. All comparisons use
the same frozen predictions. Fixed sample sizes and the arithmetic-mean caveat
remain in force.
