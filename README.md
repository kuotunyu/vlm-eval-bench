# vlm-eval-bench

An auditable, config-driven harness for comparing local and API vision-language
models on document understanding. It keeps sampling, image preparation,
decoding, retries, caching, cost gates, and reporting under one reproducible
interface.

## Results snapshot

Archived run: 2026-07-10 · seed 3407 · DocVQA n=200 · ChartQA n=200 · CORD-v2 n=100.

| Model | DocVQA ANLS | ChartQA RelaxedAcc | CORD-v2 F1 | Avg |
|---|---:|---:|---:|---:|
| **qwen3vl-8b-receipt-qlora** (local, fine-tuned) | 0.921 | 0.835 | **0.923** | **0.893** |
| qwen3vl-8b-base (local) | **0.937** | 0.830 | 0.742 | 0.836 |
| gemini-3.1-flash-lite | 0.882 | 0.595 | 0.870 | 0.782 |
| gpt-5.4-mini | 0.862 | 0.550 | 0.819 | 0.744 |

The receipt QLoRA changed CORD by **+0.181**, DocVQA by **-0.015**, and
ChartQA by **+0.005** versus the same 8B base model. This supports a strong,
narrow in-domain gain—not a claim of universal improvement. The arithmetic
mean is only a navigation aid because the three task metrics are not
interchangeable.

These are archived observations, not fresh rerun results. The historical
DocVQA and ChartQA implementations differed in edge cases from their official
reference semantics; current code fixes those behaviors for future runs, but
the archived scores were not rewritten without source predictions and
references. See the [evaluation card](EVALUATION_CARD.md) before comparing
these numbers with another leaderboard.

![Scores by task](results/charts/scores_by_task.png)

The [full leaderboard](results/leaderboard.md) adds frozen 95% intervals,
cost, latency, and reliability tables without publishing questions,
references, or model answers.

## Audit the published evidence

The committed [audit pack](results/audit/README.md) contains 2,000
deduplicated, privacy-reduced score/accounting rows. No API key, dataset image,
question, prompt, reference, raw response, or model prediction is included.
From a fresh clone, without an API key, GPU, model, or dataset download:

```bash
uv sync --locked
uv run python scripts/verify_audit.py
uv run python scripts/verify_release.py
```

The first command checks every audit hash and recomputes all public row counts,
DocVQA/ChartQA row-score means, cost, usage, latency, and terminal error
statistics. It also validates the README and leaderboard claims. CORD's
reference-dependent corpus F1 and all bootstrap intervals are provenance-checked,
not falsely described as independently rescored.

## What the harness controls

- Committed, seeded sample-index manifests give every model the same rows.
- One image pipeline applies EXIF transpose, RGB conversion, a 1,280 px maximum
  side, LANCZOS resizing, and JPEG quality 90; the prepared-image digest is
  part of the response-cache key.
- Temperature-zero decoding and shared answer cleanup are configured across
  providers, with provider-specific reasoning minimized where supported.
- Per-provider concurrency/RPM limits, retries, resume-safe JSONL output, and a
  paid-API cost gate are explicit. `--no-cache` estimates a fully uncached run.
- Only complete model/task matrices enter the leaderboard. Reporting sorts
  deduplicated rows canonically before seeded bootstrap operations.

## Models and tasks

| Model | Provider | Archived-run role |
|---|---|---|
| `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` | local RTX 4090 | 4-bit base |
| receipt QLoRA adapter | local RTX 4090 | CORD-v2-trained comparison |
| `gemini-3.1-flash-lite` | Google | complete API run |
| `gpt-5.4-mini` | OpenAI | complete API run |

| Task | Dataset/split | Archived sample | Metric label |
|---|---|---:|---|
| DocVQA | `lmms-lab/DocVQA`, validation | 200 | ANLS |
| ChartQA | `HuggingFaceM4/ChartQA`, test | 100 human + 100 machine | relaxed accuracy |
| CORD-v2 | `naver-clova-ix/cord-v2`, test | all 100 | field-level micro F1 |

The sample IDs and indices are in [`data/samples/`](data/samples/). Dataset
content is intentionally absent. Follow each dataset card and upstream terms
before downloading or redistributing data.

## Run new evaluations

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked
Copy-Item .env.example .env       # PowerShell; fill only providers you use

uv run python run.py --scale mini --dry-run
uv run python run.py --scale mini
uv run python run.py --scale full
uv run python report.py
```

On POSIX shells, use `cp .env.example .env`. API keys and Hugging Face tokens
remain in the ignored `.env`. Local inference is optional and deliberately not
installed by the locked audit path:

```bash
uv sync --locked --extra local
```

Before any paid run, independently verify provider pricing and model
availability; comments in `config.yaml` are estimates, not a billing promise.
The cost cap applies to estimated/observed paid API charges. Local dollars are
separately imputed from measured inference time at the configured GPU rental
rate.

## Cost and latency caveats

API dollar values in the archived report were reconstructed from
provider-returned usage using the frozen pricing configuration; they were not
reconciled to invoices. The historical OpenAI configuration priced an inferred
image-token share separately. Current OpenAI documentation treats images as
input tokens for the selected model, so the old dollar total remains a
configuration-derived historical estimate rather than a corrected charge.

Local cost is an imputed RTX 4090 rental equivalent, not a payment. Local
batch-1 latency excludes network time and is not comparable with API round-trip
latency. The reported 0% error rate is for final deduplicated rows; it does not
measure transient attempts that were retried.

## Repository entry points

- `run.py`: evaluate configured models and write ignored private predictions.
- `report.py`: generate a sanitized aggregate report from private predictions.
- `scripts/verify_audit.py`: validate the committed evidence and numeric claims.
- `scripts/verify_release.py`: scan the public tree for private files, secrets,
  workstation paths, oversized files, and disallowed result content.
- `scripts/verify_distribution.py`: inspect built wheel/sdist membership and
  enforce the public packaging boundary.
- `scripts/build_audit_pack.py`: maintainer-only export from retained private
  predictions.

These are repository scripts, not installed console commands.

## Upstream references and attribution

- [DocVQA challenge and ANLS](https://www.docvqa.org/challenges/2020)
- [ChartQA reference implementation](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py)
- [DocVQA dataset card](https://huggingface.co/datasets/lmms-lab/DocVQA)
- [ChartQA dataset card](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
- [CORD-v2 dataset card](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- CORD schema/comparator adapted from the MIT-licensed receipt-extraction
  project identified in the source comments and license notices.

Project code is released under the [MIT License](LICENSE). Dataset licenses and
provider terms remain separate.
