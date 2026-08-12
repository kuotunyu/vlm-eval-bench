# Owner actions and feature freeze

The corrected local release candidate is complete without additional model or
data work. Feature freeze is now appropriate: do not add models, datasets,
fine-tuning, API evaluation, or broader product features as part of this
release.

Before external publication, the owner may choose to:

1. Select a public repository URL and security-reporting contact. No remote,
   push, pull request, tag, hosted release, or remote mutation was performed.
2. Recheck provider model aliases, prices, quotas, and token accounting before
   any future paid run; such a run must be published as a new evaluation.
3. Pursue an official benchmark submission only with dataset-authorized
   evaluation inputs and the corrected metrics. Do not replace either the
   archived or corrected 2026-07-10 evidence.
