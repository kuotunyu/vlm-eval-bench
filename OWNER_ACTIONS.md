# Owner actions and feature freeze

The corrected local release candidate is complete without additional model or
data work. Feature freeze is now appropriate: do not add models, datasets,
fine-tuning, API evaluation, or broader product features as part of this
release.

## Publication status

The repository was published at
<https://github.com/kuotunyu/vlm-eval-bench> on 2026-08-14. Tag `v0.1.0` and
its GitHub release point at the audited commit; `main` is protected against
force-pushes and deletion and requires the `quality` check.

Publication added repository metadata only. No model, dataset, prediction,
metric, or accounting evidence was changed, and no inference or provider call
was performed.

## v0.1.1: claim and governance closeout

`v0.1.0`'s content (the audited evidence tree) is unchanged. `v0.1.1` is a
documentation- and governance-only follow-up:

- README.md/README.en.md: calibrated language that had drifted into
  overclaiming (e.g. "significant gain" / "high anti-degradation stability" /
  "unbiased rescore" / "100% reproducible leaderboard" / "de-identified"), and
  restored condensed evidence-boundary, "why scores changed", and cost/latency
  caveat sections that existed at v0.1.0 but were dropped by a later README
  redesign. No score, claim boundary, or evidence file changed.
- GitHub governance: Projects disabled, Dependabot vulnerability alerts
  enabled, delete-branch-on-merge enabled, branch protection admin enforcement
  turned on for `main`.
- `pyproject.toml`/`uv.lock`/`CITATION.cff` version bumped to `0.1.1`; no
  dependency versions changed.

## History rewrite (2026-08-14)

Every commit hash in this repository changed once, after `v0.1.1` shipped.
`LICENSE` and this file both carried a legal name in earlier history; it was
removed with `git filter-branch` across all commits and both tags. No content
other than that name changed anywhere — authors, committers, dates, commit
messages, and every other file are identical to before the rewrite. No model,
dataset, prediction, metric, or accounting evidence changed.

Old commit hashes (including the original `v0.1.0` and `v0.1.1` commits) no
longer resolve; a fresh clone or `git fetch --all --tags --prune` picks up the
rewritten history. `v0.1.0` and `v0.1.1` keep their tag names, now pointing at
the rewritten equivalents. This needed one force-push to `main`; branch
protection's force-push restriction was disabled only for that push and
restored immediately after.

## Attribution

This project is authored and copyrighted by a single party, `kuotunyu`.
`LICENSE`, `CITATION.cff`, and `pyproject.toml` all use that identity, and every
commit is authored by it. Keep new files consistent with it.

## Remaining owner decisions

1. Recheck provider model aliases, prices, quotas, and token accounting before
   any future paid run; such a run must be published as a new evaluation.
2. Pursue an official benchmark submission only with dataset-authorized
   evaluation inputs and the corrected metrics. Do not replace either the
   archived or corrected 2026-07-10 evidence.
