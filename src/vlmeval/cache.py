"""SQLite response cache — reruns never re-hit APIs.

Cache key = sha256 over the canonical JSON of (model_id, task, sample_id,
prompt hash, prepared-image hash, generation params incl. the image-preparation
policy). Errors are never written, so failed samples are retried on the next run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from vlmeval.models.base import ModelResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
  cache_key     TEXT PRIMARY KEY,
  model_id      TEXT NOT NULL,
  task          TEXT NOT NULL,
  sample_id     TEXT NOT NULL,
  prompt_hash   TEXT NOT NULL,
  gen_params    TEXT NOT NULL,
  response_text TEXT NOT NULL,
  input_tokens  INTEGER,
  output_tokens INTEGER,
  usage_source  TEXT,
  latency_s     REAL,
  cost_usd      REAL,
  created_at    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_responses_model_task ON responses(model_id, task);
"""


@dataclass(frozen=True)
class CachedRow:
    response_text: str
    input_tokens: int | None
    output_tokens: int | None
    usage_source: str
    latency_s: float
    cost_usd: float | None


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        # DELETE journal: WAL is unreliable on WSL /mnt/c (9p/DrvFs locking)
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    @staticmethod
    def make_key(
        model_id: str,
        task: str,
        sample_id: str,
        prompt: str,
        image_jpeg: bytes,
        gen_params: dict,
    ) -> str:
        payload = json.dumps(
            {
                "m": model_id,
                "t": task,
                "s": sample_id,
                "p": _sha256(prompt),
                "i": _sha256_bytes(image_jpeg),
                "g": gen_params,
            },
            sort_keys=True,
        )
        return _sha256(payload)

    def get(self, key: str) -> CachedRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response_text, input_tokens, output_tokens, usage_source, latency_s, cost_usd"
                " FROM responses WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CachedRow(*row)

    def has(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM responses WHERE cache_key = ?", (key,)
            ).fetchone()
        return row is not None

    def put(
        self,
        key: str,
        model_id: str,
        task: str,
        sample_id: str,
        prompt: str,
        gen_params: dict,
        resp: "ModelResponse",
    ) -> None:
        if resp.error is not None:
            raise ValueError("refusing to cache an error response")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses"
                " (cache_key, model_id, task, sample_id, prompt_hash, gen_params,"
                "  response_text, input_tokens, output_tokens, usage_source, latency_s, cost_usd)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    model_id,
                    task,
                    sample_id,
                    _sha256(prompt),
                    json.dumps(gen_params, sort_keys=True),
                    resp.text,
                    resp.usage.input_tokens,
                    resp.usage.output_tokens,
                    resp.usage.source,
                    resp.latency_s,
                    resp.cost_usd,
                ),
            )
            self._conn.commit()

    def stats(self) -> dict[tuple[str, str], int]:
        """Row counts per (model_id, task) — used by the pre-run estimate."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT model_id, task, COUNT(*) FROM responses GROUP BY model_id, task"
            ).fetchall()
        return {(m, t): c for m, t, c in rows}

    def close(self) -> None:
        self._conn.close()
