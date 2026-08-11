# Vendored from the sibling project vlm-receipt-extractor (src/metrics.py).
# Single change vs upstream: the import below targets vlmeval.vendored.schema
# instead of src.schema.
"""Field-level metrics for receipt extraction predictions.

Scoring model (micro-averaged over the dataset):

- Scalar fields (subtotal/discount/service/tax/total): TP when GT and
  prediction are both non-null and equal (numeric comparison); FP when the
  prediction is non-null but spurious or wrong; FN when a non-null GT value
  is missed or wrong. A wrong value counts as both FP and FN.
- Items: predicted items are aligned to GT items by name similarity
  (difflib ratio >= 0.5, best pairs first, one-to-one). Within an aligned
  pair each of name/count/unit_price/price is scored like a scalar field
  (names compare exactly after normalization). Unaligned predicted items
  contribute FPs, unaligned GT items FNs, for every non-null field.
- Samples with unparseable output contribute FNs for all non-null GT fields.
- total_exact_match: share of samples whose predicted total equals GT total
  (both-null counts as a match).
"""

from __future__ import annotations

from difflib import SequenceMatcher

from vlmeval.vendored.schema import ITEM_FIELDS, SCALAR_FIELDS, coerce_to_schema, normalize_name

ALIGN_THRESHOLD = 0.5


def values_equal(gt, pred):
    if gt is None or pred is None:
        return False
    if isinstance(gt, str) or isinstance(pred, str):
        return normalize_name(gt) == normalize_name(pred)
    try:
        return abs(float(gt) - float(pred)) < 1e-6
    except (TypeError, ValueError):
        return False


class Tally:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def score(self, gt, pred):
        if gt is None and pred is None:
            return
        if gt is None:
            self.fp += 1
        elif pred is None:
            self.fn += 1
        elif values_equal(gt, pred):
            self.tp += 1
        else:
            self.fp += 1
            self.fn += 1

    def add_missing_gt(self, gt):
        if gt is not None:
            self.fn += 1

    def add_spurious(self, pred):
        if pred is not None:
            self.fp += 1

    def merge(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def metrics(self):
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": self.tp + self.fn,
        }


def _field(item, key):
    v = item.get(key)
    if key == "name":
        v = v or None  # empty name counts as absent
    return v


def align_items(gt_items, pred_items):
    """One-to-one greedy alignment by name similarity, best pairs first."""
    pairs = []
    for gi, g in enumerate(gt_items):
        gname = normalize_name(g.get("name"))
        for pi, p in enumerate(pred_items):
            ratio = SequenceMatcher(None, gname, normalize_name(p.get("name"))).ratio()
            if ratio >= ALIGN_THRESHOLD:
                pairs.append((ratio, gi, pi))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_g, used_p, matches = set(), set(), []
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi))
    return matches, used_g, used_p


def compute_metrics(rows):
    """Compute metrics over rows of {"gt": target_dict, "parsed": dict|None}."""
    tallies = {f: Tally() for f in SCALAR_FIELDS}
    for k in ITEM_FIELDS:
        tallies[f"items.{k}"] = Tally()

    n = len(rows)
    n_valid = 0
    n_total_em = 0

    for row in rows:
        gt = row["gt"]
        gt_items = gt.get("items") or []
        pred = coerce_to_schema(row.get("parsed"))

        if pred is None:
            for f in SCALAR_FIELDS:
                tallies[f].add_missing_gt(gt.get(f))
            for it in gt_items:
                for k in ITEM_FIELDS:
                    tallies[f"items.{k}"].add_missing_gt(_field(it, k))
            continue

        n_valid += 1
        for f in SCALAR_FIELDS:
            tallies[f].score(gt.get(f), pred.get(f))
        if values_equal(gt.get("total"), pred.get("total")) or (
            gt.get("total") is None and pred.get("total") is None
        ):
            n_total_em += 1

        pred_items = pred["items"]
        matches, used_g, used_p = align_items(gt_items, pred_items)
        for gi, pi in matches:
            for k in ITEM_FIELDS:
                tallies[f"items.{k}"].score(_field(gt_items[gi], k), _field(pred_items[pi], k))
        for gi, it in enumerate(gt_items):
            if gi not in used_g:
                for k in ITEM_FIELDS:
                    tallies[f"items.{k}"].add_missing_gt(_field(it, k))
        for pi, it in enumerate(pred_items):
            if pi not in used_p:
                for k in ITEM_FIELDS:
                    tallies[f"items.{k}"].add_spurious(_field(it, k))

    overall = Tally()
    for t in tallies.values():
        overall.merge(t)

    return {
        "n": n,
        "valid_json_rate": round(n_valid / n, 4) if n else 0.0,
        "total_exact_match": round(n_total_em / n, 4) if n else 0.0,
        "overall": overall.metrics(),
        "fields": {name: t.metrics() for name, t in tallies.items()},
    }


def format_metrics_table(metrics):
    """Render a metrics dict as an aligned plain-text table."""
    lines = [
        f"n={metrics['n']}  valid_json_rate={metrics['valid_json_rate']:.2%}  "
        f"total_exact_match={metrics['total_exact_match']:.2%}",
        f"{'field':<16}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}",
    ]
    rows = list(metrics["fields"].items()) + [("overall", metrics["overall"])]
    for name, m in rows:
        lines.append(
            f"{name:<16}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}{m['support']:>9}"
        )
    return "\n".join(lines)
