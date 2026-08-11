"""CORD-v2 receipt extraction, scored with the vendored field-level micro F1.

The prompt is the exact string the QLoRA adapter was trained and originally
evaluated with. The test split has only 100 rows, so full scale runs all of
them in dataset order and mini is the first 20.
"""

from __future__ import annotations

import json

from vlmeval.metrics.bootstrap import bootstrap_ci
from vlmeval.tasks.base import BaseTask, Sample
from vlmeval.vendored.metrics import compute_metrics
from vlmeval.vendored.schema import PROMPT, gt_to_target, tolerant_json_parse


class CordTask(BaseTask):
    def select_indices(self, ds) -> list[int]:
        return list(range(len(ds)))  # all 100, dataset order; mini = prefix

    def make_sample(self, row: dict, index: int) -> Sample:
        gt_parse = json.loads(row["ground_truth"])["gt_parse"]
        target, _info = gt_to_target(gt_parse)
        return Sample(
            sample_id=f"cord_{index:04d}",
            image_jpeg=self._prepare(row["image"]),
            prompt=PROMPT,
            reference=target,
        )

    def score_one(self, pred_text: str, reference: dict) -> dict:
        parsed = tolerant_json_parse(pred_text)
        m = compute_metrics([{"gt": reference, "parsed": parsed}])
        return {
            "score": m["overall"]["f1"],
            "parsed": parsed,
            "parsed_ok": parsed is not None,
        }

    def aggregate(self, rows: list[dict]) -> dict:
        """Micro F1 over the whole set (not the mean of per-sample F1)."""
        cord_rows = [{"gt": r["reference"], "parsed": r.get("aux", {}).get("parsed")} for r in rows]
        m = compute_metrics(cord_rows)

        def micro_f1(sub) -> float:
            return compute_metrics(sub)["overall"]["f1"]

        ci = bootstrap_ci(
            cord_rows, micro_f1, n_boot=self.run_cfg.bootstrap_iters, seed=self.run_cfg.seed
        )
        n_err = sum(1 for r in rows if r.get("error"))
        return {
            "metric": self.cfg.metric,
            "n": len(rows),
            "score": m["overall"]["f1"],
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "error_rate": round(n_err / len(rows), 4) if rows else 0.0,
            "valid_json_rate": m["valid_json_rate"],
            "total_exact_match": m["total_exact_match"],
            "fields": m["fields"],
        }
