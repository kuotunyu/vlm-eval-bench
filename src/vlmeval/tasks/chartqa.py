"""ChartQA (HuggingFaceM4/ChartQA, test split) scored with relaxed accuracy.

Sampling is stratified 50/50 over human- vs machine-generated questions and
interleaved, so every prefix (including the mini scale) stays balanced.
"""

from __future__ import annotations

import random

from vlmeval.metrics.relaxed_acc import relaxed_correct
from vlmeval.tasks.base import BaseTask, Sample, clean_answer
from vlmeval.tasks.docvqa import PROMPT_SUFFIX


def _source_name(value, ds) -> str:
    """human_or_machine may be a ClassLabel int or a plain string."""
    if isinstance(value, str):
        return value
    feature = ds.features.get("human_or_machine") if hasattr(ds, "features") else None
    if feature is not None and hasattr(feature, "names"):
        return feature.names[value]
    return "human" if value == 0 else "machine"


class ChartQATask(BaseTask):
    def select_indices(self, ds) -> list[int]:
        rng = random.Random(self.run_cfg.seed)
        by_source: dict[str, list[int]] = {"human": [], "machine": []}
        for i, v in enumerate(ds["human_or_machine"]):
            by_source[_source_name(v, ds)].append(i)
        half = self.cfg.n_full // 2
        picked_h = rng.sample(by_source["human"], half)
        picked_m = rng.sample(by_source["machine"], half)
        interleaved: list[int] = []
        for h, m in zip(picked_h, picked_m):
            interleaved += [h, m]
        return interleaved

    def make_sample(self, row: dict, index: int) -> Sample:
        label = row["label"]
        reference = str(label[0]) if isinstance(label, list) else str(label)
        ds_for_names = getattr(self, "_ds_for_names", None)
        return Sample(
            sample_id=f"chartqa_{index}",
            image_jpeg=self._prepare(row["image"]),
            prompt=row["query"] + PROMPT_SUFFIX,
            reference=reference,
            meta={"source": _source_name(row["human_or_machine"], ds_for_names)},
        )

    def load_samples(self, scale: str):
        # make the dataset visible to make_sample for ClassLabel name lookup
        ds = self.load_dataset()
        self._ds_for_names = ds
        indices = self.get_or_create_indices(ds)
        n = self.cfg.n_for(scale)
        return [self.make_sample(ds[i], i) for i in indices[:n]]

    def score_one(self, pred_text: str, reference: str) -> dict:
        pred = clean_answer(pred_text)
        return {"score": 1.0 if relaxed_correct(pred, reference) else 0.0, "pred_clean": pred}

    def aggregate(self, rows: list[dict]) -> dict:
        out = super().aggregate(rows)
        for source in ("human", "machine"):
            sub = [r["score"] for r in rows if r.get("meta", {}).get("source") == source]
            out[f"{source}_accuracy"] = round(sum(sub) / len(sub), 4) if sub else None
        return out
