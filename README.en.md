# vlm-eval-bench: Document Understanding VLM Evaluation & Auditable Benchmark

[![CI](https://github.com/kuotunyu/vlm-eval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/vlm-eval-bench/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-4--bit-7C3AED)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F.svg)](LICENSE)

[繁體中文](README.md)

An auditable, config-driven harness for evaluating local and API vision-language models on document understanding tasks: controlling sampling, deterministic image preparation, greedy decoding, automated retries, SQLite caching, and paid API cost gates under one reproducible interface, with a 2,000-row privacy-reduced evidence pack and a full offline verification chain.

---

## System Architecture & Pipeline

### 1. Vision-Language Model Evaluation & Audit Pipeline

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph DataStage ["Phase 1: Data Engineering & Standardized Preprocessing"]
        direction LR
        Dataset[("Multimodal Benchmarks<br/>(DocVQA · ChartQA · CORD-v2)")] --> Prep["Deterministic Image Pipeline<br/>(EXIF transpose · 1280px · LANCZOS)"] --> Cache[("Cache Identification Key<br/>(Prepared Image SHA-256)")]
    end

    subgraph ExecStage ["Phase 2: Multi-Target Execution & Cost Guard"]
        direction LR
        Cache --> Models["Local & API Model Execution<br/>(Qwen3-VL 8B · Gemini · GPT)"] --> Safe{"Rate & Cost Limits<br/>(RPM/Concurrency · Paid Cost Cap)"} --> Raw[("Resume-safe Records<br/>(Append-only JSONL)")]
    end

    subgraph EvalStage ["Phase 3: Rigorous Rescoring & Verification"]
        direction LR
        Raw --> Rescore["Standard Metric Rescoring<br/>(ANLS · RelaxedAcc · Micro F1)"] --> AuditPack[("Privacy-Reduced Audit Pack<br/>(2,000 public evidence rows)")] --> Gate{"3-Stage Verification Gates<br/>(Audit · Corrected · Release)"}
    end

    DataStage --> ExecStage --> EvalStage

    classDef srcStyle fill:#e7f5ff,stroke:#1971c2,stroke-width:2px,color:#212529
    classDef procStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#212529
    classDef condStyle fill:#fff9db,stroke:#f59f00,stroke-width:2px,color:#212529
    classDef evalStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class Dataset,Cache,Raw,AuditPack srcStyle
    class Prep,Models,Rescore procStyle
    class Safe,Gate condStyle

    style DataStage fill:#f8f9fa,stroke:#1971c2,stroke-width:2px,color:#1971c2,stroke-dasharray: 4 4
    style ExecStage fill:#faf5ff,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2,stroke-dasharray: 4 4
    style EvalStage fill:#f4fbf7,stroke:#0ca678,stroke-width:2px,color:#0ca678,stroke-dasharray: 4 4
```

### 2. Evaluation System Architecture & Gate Protection

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph AdapterStage ["Phase 1: Model Adapters & Execution Targets"]
        direction LR
        M1[("Local RTX 4090<br/>(Qwen3-VL Base / QLoRA)")]
        M2[("Google Gemini API<br/>(gemini-3.1-flash-lite)")]
        M3[("OpenAI API<br/>(gpt-5.4-mini)")]
    end

    subgraph CoreStage ["Phase 2: Core Execution Engine & Cost Guard"]
        direction LR
        Runner["Config-driven Runner<br/>(Auto retries · Greedy decode)"] --> CacheDB[("SQLite Cache DB<br/>(Prevents redundant spend)")] --> CostGuard{"Cost & Concurrency Guard<br/>(Hard cap manual confirmation)"}
    end

    subgraph PublicStage ["Phase 3: Public Delivery & Full-Chain Audit"]
        direction LR
        V1["verify_audit.py<br/>(Archived evidence check)"] & V2["verify_corrected.py<br/>(Rescore & Delta check)"] & V3["verify_release.py<br/>(Secrets & Privacy scan)"] --> Release(["Public Release Candidate<br/>(Verifiable within stated scope)"])
    end

    M1 & M2 & M3 --> Runner
    CostGuard --> V1 & V2 & V3

    classDef srcStyle fill:#e7f5ff,stroke:#1971c2,stroke-width:2px,color:#212529
    classDef procStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#212529
    classDef condStyle fill:#fff9db,stroke:#f59f00,stroke-width:2px,color:#212529
    classDef safeStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class M1,M2,M3,CacheDB srcStyle
    class Runner,V1,V2,V3 procStyle
    class CostGuard condStyle
    class Release safeStyle

    style AdapterStage fill:#f8f9fa,stroke:#1971c2,stroke-width:2px,color:#1971c2,stroke-dasharray: 4 4
    style CoreStage fill:#faf5ff,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2,stroke-dasharray: 4 4
    style PublicStage fill:#f4fbf7,stroke:#0ca678,stroke-width:2px,color:#0ca678,stroke-dasharray: 4 4
```

---

## Corrected results

Offline rescore of the unchanged 2026-07-10 predictions · seed 3407 · DocVQA n=200 · ChartQA n=200 · CORD-v2 n=100:

| Model | DocVQA ANLS | ChartQA RelaxedAcc | CORD-v2 F1 | Avg |
|---|---:|---:|---:|---:|
| **qwen3vl-8b-receipt-qlora** (local, fine-tuned) | 0.9188 [0.8823, 0.9507] | 0.8300 [0.7750, 0.8800] | **0.9234 [0.8915, 0.9500]** | **0.8907** |
| qwen3vl-8b-base (local) | **0.9316 [0.8967, 0.9608]** | 0.8300 [0.7750, 0.8800] | 0.7423 [0.7037, 0.7719] | 0.8346 |
| gemini-3.1-flash-lite | 0.8795 [0.8361, 0.9180] | 0.3750 [0.3150, 0.4400] | 0.8701 [0.8460, 0.8928] | 0.7082 |
| gpt-5.4-mini | 0.8546 [0.8069, 0.8974] | 0.2900 [0.2300, 0.3550] | 0.8193 [0.7892, 0.8469] | 0.6546 |

Key takeaways:
- **Specialized Adaptation:** Versus the same 8B base model, the receipt-specific QLoRA shows an observed in-domain gain on its training domain, CORD (**+0.181** F1 — an observed delta, not a formally tested paired-significance result). DocVQA (-0.013) and ChartQA (0.000) are two limited out-of-domain checks only, not evidence of broad capability improvement.
- **Rigorous Metric Correction:** The release did not rerun inference. It rescored the same frozen predictions under a corrected metric contract (space-sensitive DocVQA ANLS; Pix2Struct-compatible ChartQA relaxed accuracy); the CORD-v2 F1 algorithm is unchanged. Historical values remain preserved in the [archived leaderboard](results/archived_leaderboard.md), and delta analysis is documented in the [corrected leaderboard](results/leaderboard.md).

---

## Models and Tasks

| Model | Provider / Environment | Benchmark Role |
|---|---|---|
| `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` | Local RTX 4090 | 4-bit baseline |
| receipt QLoRA adapter | Local RTX 4090 | CORD-v2 fine-tuned comparison |
| `gemini-3.1-flash-lite` | Google Cloud API | Cloud lightweight multimodal reference |
| `gpt-5.4-mini` | OpenAI API | Cloud small multimodal reference |

| Task | Dataset / Split | Sample Size | Metric |
|---|---|---:|---|
| DocVQA | `lmms-lab/DocVQA`, validation | 200 | ANLS (Average Normalized Levenshtein Similarity) |
| ChartQA | `HuggingFaceM4/ChartQA`, test | 100 human + 100 synthetic | Relaxed Accuracy (5% tolerance) |
| CORD-v2 | `naver-clova-ix/cord-v2`, test | all 100 | Field-level Micro F1 |

---

## Verify and Reproduce

Verify claims offline (CPU-only, no GPUs, datasets, or API calls required):

```bash
uv sync --locked
uv run python scripts/verify_audit.py
uv run python scripts/verify_corrected.py
uv run python scripts/verify_release.py
```

`verify_corrected.py` verifies all SHA-256 chains across 2,000 privacy-reduced rows, independently recomputes DocVQA and ChartQA confidence intervals, and confirms leaderboard integrity.

**Public/private evidence boundary:** The public pack carries only scalar fields — sample ID, score, usage, cost, latency, and error status. It excludes questions, images, references, model answers, raw predictions, and provider responses. Only integer indices are public under `data/samples/`. CORD-v2's references and predictions are intentionally excluded, so its F1 can only be **provenance-verified** (hash and row-count match against the archived pack) — not independently rescored in public. DocVQA and ChartQA means and confidence intervals are the part that is independently recomputable.

---

## Why the Scores Changed

- **DocVQA:** The old scorer collapsed internal whitespace and still passed when the normalized Levenshtein distance was exactly 0.5. The corrected metric contract is case-insensitive but space-sensitive, and requires the distance to be strictly below 0.5.
- **ChartQA:** The old scorer stripped `%` without converting it to a fraction, and also removed currency/thousands punctuation. The corrected behavior follows Pix2Struct: `%` converts to a fraction before a 5% relative tolerance is applied, and non-numeric targets use case-insensitive exact text.
- **CORD-v2:** The scoring algorithm (corpus-level field micro F1) is unchanged; every CORD aggregate and per-row score is identical to the archived pack.

The API-side ChartQA drops are a scoring correction, **not** a change in provider behavior or a new model call — every comparison uses the same frozen predictions. See [EVALUATION_CARD.md](EVALUATION_CARD.md) for the complete evidence boundary.

---

## Method & Engineering Controls

- **Committed Sampling Seeds:** Seed 3407 index manifests ensure exact input alignment across all tested models.
- **Deterministic Image Pipeline:** EXIF transposition, RGB conversion, 1,280 px max edge, LANCZOS resizing, and JPEG 90; processed image SHA-256 is part of the cache key.
- **Greedy Decoding:** Zero-temperature sampling minimizes provider-side variance.
- **Cost & Concurrency Controls:** Enforced RPM/concurrency bounds, resume-safe JSONL, and a hard API cost gate.

---

## Cost and Latency Caveats

- All cost/usage/latency accounting is **inherited unchanged** from the original inference run, not recomputed. API dollars are estimated from provider-returned usage against the price configured at the time, **not reconciled to invoices**; the historical OpenAI config separately priced an inferred image-token share, retained only as provenance, not a current price.
- Provider prices, quotas, and model aliases in the config are estimates, **not a current billing promise** — reconfirm pricing and quota before any new paid run.
- Local dollars are an imputed RTX 4090 rental equivalent (not an actual payment). Local batch-1 latency **excludes network time** and is **not directly comparable** with API round-trip latency.
- The 0% error rate covers final deduplicated rows only; it does not represent transient failures that were retried.

---

## Data and licensing

- Code: [MIT License](LICENSE). Third-party dataset licenses and each API provider's terms are **separate** and do not transfer with this project's license — confirm them yourself before downloading or using dataset content.
- Full evidence boundaries and the publicly recomputable claim scope are detailed in [EVALUATION_CARD.md](EVALUATION_CARD.md).
