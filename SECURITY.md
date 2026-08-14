# Security policy

## Supported versions

This project is a research evaluation harness, not a hosted service. Security
fixes are applied to the `main` branch. Tagged releases are immutable evidence
snapshots and are not patched in place; a fix ships as a new release.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/kuotunyu/vlm-eval-bench/security/advisories/new)
on this repository. Do not open a public issue for a security report.

Please do not include real API keys, credentials, or private dataset content in
a report. A redacted reproduction is sufficient.

Expect an initial response within 14 days. This is a personal, unfunded
research project, so there is no paid bounty and no guaranteed remediation
timeline.

## What is in scope

- Credential handling: leakage of API keys or tokens through logs, caches,
  error messages, committed files, or built distributions.
- The privacy boundary of published evidence: any path by which dataset
  questions, images, references, model answers, raw predictions, or provider
  responses could be recovered from the public repository or its release
  artifacts.
- The release and audit verifiers (`scripts/verify_release.py`,
  `scripts/verify_audit.py`, `scripts/verify_corrected.py`): any way to make a
  tree or evidence pack pass verification while carrying private or
  inconsistent content.
- Unsafe file handling in the harness, such as archive extraction or path
  traversal.

## What is out of scope

- Vulnerabilities in upstream datasets, model weights, or third-party provider
  APIs. Report those to their maintainers.
- Model output quality, benchmark scores, or disagreement with the published
  metric semantics. These are correctness and methodology questions — open a
  normal issue instead.
- Cost, quota, or rate-limit behavior of paid provider accounts.
