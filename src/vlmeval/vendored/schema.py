# Vendored from the sibling project vlm-receipt-extractor (src/schema.py),
# https://huggingface.co/steven0226/vlm-receipt-extractor - copied verbatim so this
# harness scores CORD-v2 with exactly the same schema/prompt/normalizers.
"""Target schema, prompt, and conversion utilities for vlm-receipt-extractor.

Defines the fixed output JSON schema (aligned with CORD-v2 gt_parse), the
instruction prompt shared by training and evaluation, normalization helpers
for money/count values, the gt_parse -> target-schema converter, and a
tolerant JSON parser for model outputs.
"""

from __future__ import annotations

import json
import re

SCALAR_FIELDS = ["subtotal", "discount", "service", "tax", "total"]
ITEM_FIELDS = ["name", "count", "unit_price", "price"]

# One prompt shared by the zero-shot baseline, the training targets and the
# fine-tuned evaluation, so the before/after comparison is apples-to-apples.
PROMPT = (
    "Extract all information from this receipt image and return it as a JSON "
    "object with exactly this structure:\n"
    '{"items": [{"name": <string>, "count": <integer or null>, '
    '"unit_price": <number or null>, "price": <number or null>}], '
    '"subtotal": <number or null>, "discount": <number or null>, '
    '"service": <number or null>, "tax": <number or null>, '
    '"total": <number or null>}\n'
    "Rules: amounts must be plain numbers without currency symbols or "
    "thousands separators; use null for values not present on the receipt; "
    "output only the JSON object with no explanation and no markdown."
)


def build_messages(image=None):
    """Build the single-turn chat messages for one sample.

    With image=None the image stays a placeholder (for apply_chat_template at
    inference time); pass a PIL image to embed it (for training samples).
    """
    image_part = {"type": "image"} if image is None else {"type": "image", "image": image}
    return [{"role": "user", "content": [image_part, {"type": "text", "text": PROMPT}]}]


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _as_text(v):
    """Coerce a CORD field into a clean string (fields are occasionally lists)."""
    if v is None:
        return None
    if isinstance(v, list):
        parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
        return " ".join(parts) or None
    s = str(v).strip()
    return s or None


def parse_money(v):
    """Parse a money value into int/float, tolerating Indonesian formats.

    Handles thousands separators ("25,000" / "25.000" / "1.500.000"), decimal
    tails ("10,5"), currency prefixes ("Rp 25.000") and negatives ("-2,000",
    "(2,000)"). Returns None when no number is present.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, list):
        for x in v:
            n = parse_money(x)
            if n is not None:
                return n
        return None
    s = str(v).strip()
    if not s:
        return None
    negative = s.startswith("-") or s.endswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[^\d.,]", "", s)
    if not re.search(r"\d", s):
        return None
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", s):
        n = float(re.sub(r"[.,]", "", s))  # thousands separators only
    elif re.fullmatch(r"\d+[.,]\d{1,2}", s):
        n = float(s.replace(",", "."))  # decimal tail
    else:
        digits = re.sub(r"[.,]", "", s)  # plain or messy -> keep digits
        if not digits:
            return None
        n = float(digits)
    if negative:
        n = -n
    return int(n) if n == int(n) else n


def parse_count(v):
    """Parse a count like "2", "x2", "2x" into int; None when absent."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, list):
        for x in v:
            n = parse_count(x)
            if n is not None:
                return n
        return None
    digits = re.sub(r"[^\d]", "", str(v))
    return int(digits) if digits else None


def normalize_name(s):
    """Normalize an item name for comparison: casefold + collapsed spaces."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def _group_dict(gt_parse, key, anomalies):
    g = gt_parse.get(key)
    if g is None:
        return {}
    if isinstance(g, list):
        anomalies.append(f"{key}: list group merged")
        merged = {}
        for part in g:
            if isinstance(part, dict):
                merged.update(part)
        return merged
    if not isinstance(g, dict):
        anomalies.append(f"{key}: unexpected type {type(g).__name__}")
        return {}
    return g


def gt_to_target(gt_parse):
    """Convert a CORD-v2 gt_parse dict into the fixed target schema.

    menu.sub sub-items are flattened into top-level items. Returns
    (target, info) where info = {"anomalies": [...], "sub_items": int}
    for data-quality statistics.
    """
    anomalies = []
    items = []
    sub_items = 0

    def add_item(d, origin):
        if not isinstance(d, dict):
            anomalies.append(f"{origin}: non-dict item skipped")
            return
        name = _as_text(d.get("nm"))
        if name is None:
            anomalies.append(f"{origin}: item without name")
        items.append(
            {
                "name": name or "",
                "count": parse_count(d.get("cnt")),
                "unit_price": parse_money(d.get("unitprice")),
                "price": parse_money(d.get("price")),
            }
        )

    for m in _as_list(gt_parse.get("menu")):
        if not isinstance(m, dict):
            anomalies.append("menu: non-dict entry skipped")
            continue
        add_item(m, "menu")
        for sub in _as_list(m.get("sub")):
            add_item(sub, "menu.sub")
            sub_items += 1

    sub_total = _group_dict(gt_parse, "sub_total", anomalies)
    total = _group_dict(gt_parse, "total", anomalies)
    target = {
        "items": items,
        "subtotal": parse_money(sub_total.get("subtotal_price")),
        "discount": parse_money(sub_total.get("discount_price")),
        "service": parse_money(sub_total.get("service_price")),
        "tax": parse_money(sub_total.get("tax_price")),
        "total": parse_money(total.get("total_price")),
    }
    return target, {"anomalies": anomalies, "sub_items": sub_items}


def canonical_json(target):
    """Serialize a target dict as the canonical compact training string."""
    ordered = {"items": [{k: it.get(k) for k in ITEM_FIELDS} for it in target.get("items", [])]}
    for k in SCALAR_FIELDS:
        ordered[k] = target.get(k)
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def tolerant_json_parse(text):
    """Extract and parse the first JSON object from model output.

    Strips markdown code fences, then scans for the first balanced {...}
    block (string-aware). Returns a dict, or None when nothing parses.
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(t[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def coerce_to_schema(parsed):
    """Coerce an arbitrary parsed dict into the target schema for scoring.

    Missing keys become null/[]; numeric strings are re-parsed with the same
    money/count normalizers so formatting differences don't mask correct
    values. Returns None for non-dict input.
    """
    if not isinstance(parsed, dict):
        return None
    out = {"items": []}
    raw_items = parsed.get("items")
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if isinstance(raw_items, list):
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            out["items"].append(
                {
                    "name": _as_text(it.get("name")) or "",
                    "count": parse_count(it.get("count")),
                    "unit_price": parse_money(it.get("unit_price")),
                    "price": parse_money(it.get("price")),
                }
            )
    for k in SCALAR_FIELDS:
        out[k] = parse_money(parsed.get(k))
    return out
