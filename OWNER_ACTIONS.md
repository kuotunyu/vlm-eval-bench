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

## Remaining owner decisions

1. `LICENSE` carries two copyright lines, a legal name and `kuotunyu`. If these
   name one person, collapse them into a single line; if they name separate
   parties, keep both. Only the owner can make this determination, so it was
   left untouched.
2. `CITATION.cff` credits the author by the `kuotunyu` handle, mirroring
   `pyproject.toml`. Replace it with a preferred legal name and ORCID if the
   work should be cited that way.
3. Recheck provider model aliases, prices, quotas, and token accounting before
   any future paid run; such a run must be published as a new evaluation.
4. Pursue an official benchmark submission only with dataset-authorized
   evaluation inputs and the corrected metrics. Do not replace either the
   archived or corrected 2026-07-10 evidence.
