"""SQLite persistence with an in-memory numpy index for fast cosine search."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 3,
    created REAL NOT NULL,
    last_access REAL NOT NULL,
    speaker TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    session TEXT NOT NULL DEFAULT '',
    meta TEXT NOT NULL DEFAULT '{}',
    embedding BLOB,
    sig TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    platform TEXT NOT NULL DEFAULT '',
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    messages INTEGER NOT NULL DEFAULT 0,
    sessions INTEGER NOT NULL DEFAULT 0,
    last_session TEXT NOT NULL DEFAULT '',
    support REAL NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started REAL NOT NULL,
    ended REAL,
    summary TEXT NOT NULL DEFAULT '',
    diary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass
class MemoryRecord:
    id: int
    kind: str
    text: str
    importance: float
    created: float
    last_access: float
    speaker: str = ""
    subject: str = ""
    session: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class UserProfile:
    name: str
    platform: str
    first_seen: float
    last_seen: float
    messages: int
    sessions: int
    last_session: str
    support: float
    notes: str


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._lock = threading.RLock()
        self._ids: list[int] = []
        self._pos: dict[int, int] = {}
        self._buf: Optional[np.ndarray] = None  # growable (capacity, dim) buffer
        self._sig = ""
        self._load_index()

    # ------------------------------------------------------------------ index
    @property
    def _matrix(self) -> Optional[np.ndarray]:
        return None if self._buf is None else self._buf[: len(self._ids)]

    def _append_vector(self, mid: int, vec: np.ndarray) -> bool:
        if self._buf is not None and self._buf.shape[1] != vec.shape[0]:
            return False
        if self._buf is None:
            self._buf = np.zeros((64, vec.shape[0]), dtype=np.float32)
        elif len(self._ids) >= self._buf.shape[0]:
            grown = np.zeros((self._buf.shape[0] * 2, self._buf.shape[1]), dtype=np.float32)
            grown[: self._buf.shape[0]] = self._buf
            self._buf = grown
        self._buf[len(self._ids)] = vec
        self._pos[mid] = len(self._ids)
        self._ids.append(mid)
        return True

    def _load_index(self) -> None:
        rows = self.db.execute("SELECT id, embedding, sig FROM memories WHERE embedding IS NOT NULL ORDER BY id").fetchall()
        self._ids, self._pos, self._buf = [], {}, None
        sigs = {r["sig"] for r in rows}
        self._sig = sigs.pop() if len(sigs) == 1 else ""
        dims = {len(r["embedding"]) for r in rows}
        if len(dims) > 1:
            self._sig = ""
            return  # mixed embeddings: re-embed before searching
        for r in rows:
            self._append_vector(r["id"], np.frombuffer(r["embedding"], dtype=np.float32))

    @property
    def index_signature(self) -> str:
        return self._sig

    def needs_reembed(self, signature: str) -> bool:
        total = self.count()
        return total > 0 and (self._sig != signature or len(self._ids) != total)

    def rows_for_reembed(self) -> list[tuple[int, str]]:
        return [(r["id"], r["text"]) for r in self.db.execute("SELECT id, text FROM memories ORDER BY id")]

    def set_embeddings(self, ids: list[int], matrix: np.ndarray, signature: str) -> None:
        with self._lock, self.db:
            for mid, vec in zip(ids, matrix):
                self.db.execute(
                    "UPDATE memories SET embedding = ?, sig = ? WHERE id = ?",
                    (np.asarray(vec, dtype=np.float32).tobytes(), signature, mid),
                )
        self._load_index()

    # ------------------------------------------------------------------ memories
    def add(
        self,
        kind: str,
        text: str,
        embedding: Optional[np.ndarray],
        *,
        importance: float = 3.0,
        speaker: str = "",
        subject: str = "",
        session: str = "",
        meta: Optional[dict] = None,
        signature: str = "",
        ts: Optional[float] = None,
    ) -> int:
        ts = ts or time.time()
        blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None
        with self._lock, self.db:
            cur = self.db.execute(
                "INSERT INTO memories (kind, text, importance, created, last_access, speaker, subject, session, meta, embedding, sig)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (kind, text, float(importance), ts, ts, speaker, subject.lower(), session,
                 json.dumps(meta or {}), blob, signature),
            )
            mid = int(cur.lastrowid)
            if embedding is not None and self._append_vector(mid, np.asarray(embedding, dtype=np.float32)):
                self._sig = signature if len(self._ids) == 1 or self._sig == signature else ""
        return mid

    def _record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], kind=row["kind"], text=row["text"], importance=row["importance"],
            created=row["created"], last_access=row["last_access"], speaker=row["speaker"],
            subject=row["subject"], session=row["session"], meta=json.loads(row["meta"] or "{}"),
        )

    def get_many(self, ids: Iterable[int]) -> dict[int, MemoryRecord]:
        ids = list(ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(f"SELECT * FROM memories WHERE id IN ({marks})", ids).fetchall()
        return {r["id"]: self._record(r) for r in rows}

    def vector(self, mid: int) -> Optional[np.ndarray]:
        idx = self._pos.get(mid)
        return None if idx is None or self._buf is None else self._buf[idx]

    def search(self, query: np.ndarray, k: int = 20, kinds: Optional[Iterable[str]] = None) -> list[tuple[MemoryRecord, float]]:
        matrix = self._matrix
        if matrix is None or not self._ids or query.shape[0] != matrix.shape[1]:
            return []
        sims = matrix @ query.astype(np.float32)
        order = np.argsort(-sims)
        kinds = set(kinds) if kinds else None
        wanted = [self._ids[i] for i in order[: max(k * 3, k)]]
        records = self.get_many(wanted)
        out = []
        for i in order[: max(k * 3, k)]:
            rec = records.get(self._ids[i])
            if rec is None or (kinds and rec.kind not in kinds):
                continue
            out.append((rec, float(sims[i])))
            if len(out) >= k:
                break
        return out

    def by_subject(self, subject: str, limit: int = 5, kinds: Iterable[str] = ("fact",)) -> list[MemoryRecord]:
        kinds = list(kinds)
        marks = ",".join("?" * len(kinds))
        rows = self.db.execute(
            f"SELECT * FROM memories WHERE subject = ? AND kind IN ({marks}) ORDER BY importance DESC, created DESC LIMIT ?",
            (subject.lower(), *kinds, limit),
        ).fetchall()
        return [self._record(r) for r in rows]

    def recent(self, kind: Optional[str] = None, limit: int = 30, session: Optional[str] = None) -> list[MemoryRecord]:
        query, args = "SELECT * FROM memories WHERE 1=1", []
        if kind:
            query += " AND kind = ?"
            args.append(kind)
        if session:
            query += " AND session = ?"
            args.append(session)
        rows = self.db.execute(query + " ORDER BY created DESC LIMIT ?", (*args, limit)).fetchall()
        return [self._record(r) for r in rows]

    def touch(self, ids: Iterable[int], ts: Optional[float] = None) -> None:
        ids = list(ids)
        if not ids:
            return
        with self._lock, self.db:
            self.db.executemany("UPDATE memories SET last_access = ? WHERE id = ?", [(ts or time.time(), i) for i in ids])

    def delete_subject(self, subject: str) -> int:
        with self._lock, self.db:
            cur = self.db.execute("DELETE FROM memories WHERE subject = ?", (subject.lower(),))
            self.db.execute("DELETE FROM users WHERE lower(name) = ?", (subject.lower(),))
        self._load_index()
        return cur.rowcount

    def count(self, kind: Optional[str] = None) -> int:
        if kind:
            return self.db.execute("SELECT COUNT(*) FROM memories WHERE kind = ?", (kind,)).fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # ------------------------------------------------------------------ users
    def seen_user(self, name: str, platform: str, session: str, ts: Optional[float] = None, support: float = 0.0) -> tuple[UserProfile, bool]:
        ts = ts or time.time()
        key = name.lower()
        with self._lock, self.db:
            row = self.db.execute("SELECT * FROM users WHERE name = ?", (key,)).fetchone()
            is_new = row is None
            if is_new:
                self.db.execute(
                    "INSERT INTO users (name, platform, first_seen, last_seen, messages, sessions, last_session, support)"
                    " VALUES (?, ?, ?, ?, 1, 1, ?, ?)",
                    (key, platform, ts, ts, session, support),
                )
            else:
                new_session = row["last_session"] != session
                self.db.execute(
                    "UPDATE users SET last_seen = ?, messages = messages + 1, sessions = sessions + ?,"
                    " last_session = ?, support = support + ? WHERE name = ?",
                    (ts, 1 if new_session else 0, session, support, key),
                )
        return self.user(name), is_new

    def user(self, name: str) -> Optional[UserProfile]:
        row = self.db.execute("SELECT * FROM users WHERE name = ?", (name.lower(),)).fetchone()
        return UserProfile(**dict(row)) if row else None

    def set_user_notes(self, name: str, notes: str) -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE users SET notes = ? WHERE name = ?", (notes, name.lower()))

    def top_users(self, limit: int = 10) -> list[UserProfile]:
        rows = self.db.execute("SELECT * FROM users ORDER BY messages DESC LIMIT ?", (limit,)).fetchall()
        return [UserProfile(**dict(r)) for r in rows]

    # ------------------------------------------------------------------ sessions & kv
    def start_session(self, session: str, ts: Optional[float] = None) -> None:
        with self._lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO sessions (id, started) VALUES (?, ?)", (session, ts or time.time()))

    def end_session(self, session: str, summary: str, diary: str, ts: Optional[float] = None) -> None:
        with self._lock, self.db:
            self.db.execute(
                "UPDATE sessions SET ended = ?, summary = ?, diary = ? WHERE id = ?",
                (ts or time.time(), summary, diary, session),
            )

    def last_sessions(self, limit: int = 3) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM sessions WHERE ended IS NOT NULL ORDER BY started DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def kv_get(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        with self._lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))

    def stats(self) -> dict:
        kinds = {r["kind"]: r["n"] for r in self.db.execute("SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind")}
        users = self.db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        sessions = self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {"memories": sum(kinds.values()), "by_kind": kinds, "users": users, "sessions": sessions}

    def close(self) -> None:
        self.db.close()
