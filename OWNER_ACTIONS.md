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

Tag `v0.1.0` (commit `049ef34`) is unchanged and still points at the audited
evidence tree. `v0.1.1` is a documentation- and governance-only follow-up:

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
