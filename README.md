# vlm-eval-bench

An auditable, config-driven harness for comparing local and API vision-language
models on document understanding. It controls sampling, image preparation,
decoding, retries, caching, cost gates, and reporting under one reproducible
interface.

## Corrected results

Offline rescore of the unchanged 2026-07-10 predictions · seed 3407 · DocVQA
n=200 · ChartQA n=200 · CORD-v2 n=100.

| Model | DocVQA ANLS | ChartQA RelaxedAcc | CORD-v2 F1 | Avg |
|---|---:|---:|---:|---:|
| **qwen3vl-8b-receipt-qlora** (local, fine-tuned) | 0.9188 [0.8823, 0.9507] | 0.8300 [0.7750, 0.8800] | **0.9234 [0.8915, 0.9500]** | **0.8907** |
| qwen3vl-8b-base (local) | **0.9316 [0.8967, 0.9608]** | 0.8300 [0.7750, 0.8800] | 0.7423 [0.7037, 0.7719] | 0.8346 |
| gemini-3.1-flash-lite | 0.8795 [0.8361, 0.9180] | 0.3750 [0.3150, 0.4400] | 0.8701 [0.8460, 0.8928] | 0.7082 |
| gpt-5.4-mini | 0.8546 [0.8069, 0.8974] | 0.2900 [0.2300, 0.3550] | 0.8193 [0.7892, 0.8469] | 0.6546 |

The receipt QLoRA changed CORD by **+0.181**, DocVQA by **-0.013**, and
ChartQA by **0.000** versus the same 8B base model. This is a strong, narrow
in-domain gain—not a universal quality claim. CORD is the adapter's training
domain; DocVQA and ChartQA are limited out-of-domain checks. The arithmetic
mean is only a navigation aid across non-equivalent metrics.

This release did **not** rerun inference. It rescored the same frozen private
predictions with space-sensitive DocVQA ANLS and Pix2Struct-compatible ChartQA
relaxed correctness. The original implementation values are preserved in the
[archived leaderboard](results/archived_leaderboard.md); the
[corrected leaderboard](results/leaderboard.md) contains every old/new delta.

## Verify without GPU, data, or APIs

The archived and corrected packs each contain 2,000 privacy-reduced rows. They
exclude API keys, dataset images, questions, prompts, references, model
answers, raw predictions, and provider responses.

```bash
uv sync --locked
uv run python scripts/verify_audit.py
uv run python scripts/verify_corrected.py
uv run python scripts/verify_release.py
```

The corrected verifier checks every hash and row, proves accounting is
unchanged from the archived pack, independently recomputes DocVQA/ChartQA
means and confidence intervals, validates all old/new deltas, and checks the
README and leaderboard arithmetic. CORD remains reference-dependent, so its
unchanged aggregate and interval are provenance-verified rather than falsely
described as independently rescored.

## Why the scores changed

- DocVQA answers are case-insensitive but space-sensitive, and normalized
  Levenshtein distance must be strictly below 0.5. The old implementation
  collapsed internal whitespace and accepted the equality boundary.
- ChartQA converts trailing percentages to fractions before applying 5%
  relative tolerance to a nonzero target. The old implementation stripped `%`
  without scaling and also removed currency/thousands punctuation.
- CORD's corpus-level field micro F1 implementation did not change; every CORD
  aggregate and per-row score is identical.

The API ChartQA drops are therefore scoring corrections—not new provider
behavior or new model calls. See the [evaluation card](EVALUATION_CARD.md) for
the complete evidence boundary.

## Models and tasks

| Model | Provider | Frozen-run role |
|---|---|---|
| `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` | local RTX 4090 | 4-bit base |
| receipt QLoRA adapter | local RTX 4090 | CORD-v2-trained comparison |
| `gemini-3.1-flash-lite` | Google | complete API run |
| `gpt-5.4-mini` | OpenAI | complete API run |

| Task | Dataset/split | Frozen sample | Corrected metric |
|---|---|---:|---|
| DocVQA | `lmms-lab/DocVQA`, validation | 200 | ANLS |
| ChartQA | `HuggingFaceM4/ChartQA`, test | 100 human + 100 machine | relaxed accuracy |
| CORD-v2 | `naver-clova-ix/cord-v2`, test | all 100 | field-level micro F1 |

Only sample indices and opaque IDs are public under [`data/samples/`](data/samples/).
Follow upstream dataset terms before downloading or redistributing content.

## Harness controls

- Committed seed-3407 index manifests give every model the same rows.
- One image pipeline applies EXIF transpose, RGB conversion, a 1,280 px maximum
  side, LANCZOS resizing, and JPEG quality 90; prepared-image SHA-256 is part
  of the cache key.
- Temperature-zero decoding and common answer cleanup are configured across
  providers, with provider-specific reasoning minimized where supported.
- Per-provider concurrency/RPM limits, retries, resume-safe JSONL, and a paid
  API cost gate are explicit. `--no-cache` estimates a fully uncached run.
- Only a complete model/task matrix enters public comparisons.

## Run a new evaluation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked
Copy-Item .env.example .env       # PowerShell; fill only providers you use

uv run python run.py --scale mini --dry-run
uv run python run.py --scale mini
uv run python run.py --scale full
uv run python report.py
```

On POSIX shells, use `cp .env.example .env`. Local GPU inference is optional
and deliberately absent from the locked verification path:

```bash
uv sync --locked --extra local
```

Before a paid run, verify current model availability, pricing, quotas, and
image accounting. Config comments are estimates, not billing promises.

## Maintainer-only offline rescore

The rescoring CLI reads private evidence in place, disables dotenv loading,
requires a complete matrix, verifies archived source hashes and sample IDs,
uses deterministic latest-row semantics, and writes only sanitized artifacts
to a separate empty directory:

```bash
uv run python scripts/recompute_private_run.py \
  --input-dir <PRIVATE_PATH> \
  --output-dir <ISOLATED_EMPTY_PATH>
```

Review the isolated output before copying it into a public tree. Never point
`--output-dir` at the private input or an existing nonempty directory.

## Cost and latency caveats

All accounting is inherited unchanged from the historical inference run. API
dollars were reconstructed from provider-returned usage and frozen config
prices, not reconciled to invoices. The historical OpenAI configuration used
a separately priced inferred image-token share; it remains provenance, not a
current price claim.

Local dollars are an imputed RTX 4090 rental equivalent, not a payment. Local
batch-1 latency excludes network time and is not comparable with API
round-trip latency. The 0% error rate covers final deduplicated rows, not
transient attempts that were retried.

## Repository entry points

- `run.py`: run configured inference into ignored private predictions.
- `report.py`: render aggregate reports from private predictions.
- `scripts/recompute_private_run.py`: isolated, offline corrected rescore.
- `scripts/verify_audit.py`: verify archived evidence.
- `scripts/verify_corrected.py`: verify corrected evidence and public claims.
- `scripts/verify_release.py`: scan the public tree for secrets/private data.
- `scripts/verify_distribution.py`: inspect wheel/sdist membership.

These are repository scripts, not installed console commands.

## Upstream references and attribution

- [DocVQA challenge and ANLS](https://www.docvqa.org/challenges/2020)
- [Pix2Struct metric reference](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py)
- [DocVQA dataset card](https://huggingface.co/datasets/lmms-lab/DocVQA)
- [ChartQA dataset card](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
- [CORD-v2 dataset card](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

Project code is released under the [MIT License](LICENSE). Dataset licenses and
provider terms remain separate.
