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
