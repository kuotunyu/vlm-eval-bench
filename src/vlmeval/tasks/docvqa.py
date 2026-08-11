"""DocVQA (lmms-lab/DocVQA, validation split) scored with ANLS."""

from __future__ import annotations

from vlmeval.metrics.anls import anls_score
from vlmeval.tasks.base import BaseTask, Sample, clean_answer

PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


class DocVQATask(BaseTask):
    def make_sample(self, row: dict, index: int) -> Sample:
        return Sample(
            sample_id=f"docvqa_{row['questionId']}",
            image_jpeg=self._prepare(row["image"]),
            prompt=row["question"] + PROMPT_SUFFIX,
            reference=list(row["answers"]),
        )

    def score_one(self, pred_text: str, reference: list[str]) -> dict:
        pred = clean_answer(pred_text)
        return {"score": anls_score(pred, reference), "pred_clean": pred}
