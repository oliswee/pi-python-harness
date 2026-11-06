from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_SCHEMA_VERSION = 1
_ACTIVE = "valid_from <= ? AND valid_to IS NULL AND revoked_at IS NULL AND tombstone = 0"


class MemoryConflict(RuntimeError):
    """A requested revision or memory store is no longer current."""


@dataclass(frozen=True, order=True)
class MemoryRef:
    memory_id: str
    revision: int


@dataclass(frozen=True)
class MemoryRecord:
    ref: MemoryRef
    scope: str
    kind: str
    content: str
    provenance: str
    created_at: str
    valid_from: str
    valid_to: str | None
    superseded_by: int | None
    revoked_at: str | None
    tombstone: bool
    source_run_id: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _record(row: sqlite3.Row) -> MemoryRecord:
    values = dict(row)
    ref = MemoryRef(values.pop("memory_id"), values.pop("revision"))
    values["tombstone"] = bool(values["tombstone"])
    return MemoryRecord(ref=ref, **values)


def _validate_text(value: str, field: str, max_bytes: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field} must contain between 1 and {max_bytes} UTF-8 bytes of nonblank text")


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or not 1 <= limit <= 16:
        raise ValueError("Memory recall limit must be between 1 and 16")


class MemoryStore:
    """SQLite authority for scoped, versioned memories."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction(initialize=True) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                conn.execute("CREATE TABLE memory_store (store_id TEXT PRIMARY KEY)")
                conn.execute("INSERT INTO memory_store VALUES (?)", (uuid4().hex,))
                conn.execute("""
                    CREATE TABLE memory (
                        scope TEXT NOT NULL,
                        memory_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        kind TEXT NOT NULL,
                        content TEXT NOT NULL,
                        provenance TEXT NOT NULL,
                        source_run_id TEXT,
                        created_at TEXT NOT NULL,
                        valid_from TEXT NOT NULL,
                        valid_to TEXT,
                        superseded_by INTEGER,
                        revoked_at TEXT,
                        tombstone INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (scope, memory_id, revision)
                    )
                """)
                conn.execute("""
                    CREATE INDEX memory_recall ON memory (scope, created_at DESC, memory_id, revision)
                    WHERE valid_to IS NULL AND revoked_at IS NULL AND tombstone = 0
                """)
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION:
                raise ValueError(f"Unsupported memory database version {version}")
            self.store_id = conn.execute("SELECT store_id FROM memory_store").fetchone()[0]

    @contextmanager
    def _transaction(self, *, initialize: bool = False):
        mode = "rwc" if initialize else "rw"
        location = f"{self.path.as_uri()}?mode={mode}"
        with closing(sqlite3.connect(location, uri=True, timeout=5)) as conn, conn:
            conn.row_factory = sqlite3.Row
            # Serialize writes and context assembly across connections/processes.
            conn.execute("BEGIN IMMEDIATE")
            if not initialize and conn.execute("SELECT store_id FROM memory_store").fetchone()[0] != self.store_id:
                raise MemoryConflict("Memory store identity changed")
            yield conn

    def create(
        self, scope: str, *, kind: str, content: str, provenance: str, source_run_id: str | None = None,
    ) -> MemoryRecord:
        _validate_text(scope, "scope", 256)
        _validate_text(kind, "kind", 64)
        self._validate_entry(content, provenance, source_run_id)
        with self._transaction() as conn:
            return self._insert(conn, scope, MemoryRef(uuid4().hex, 1), kind, content, provenance, source_run_id, _now())

    def supersede(
        self, scope: str, ref: MemoryRef, *, content: str, provenance: str, source_run_id: str | None = None,
    ) -> MemoryRecord:
        self._validate_entry(content, provenance, source_run_id)
        with self._transaction() as conn:
            now = _now()
            old = self._active_record(conn, scope, ref, now)
            replacement = MemoryRef(ref.memory_id, ref.revision + 1)
            conn.execute(
                "UPDATE memory SET valid_to = ?, superseded_by = ? WHERE scope = ? AND memory_id = ? AND revision = ?",
                (now, replacement.revision, scope, ref.memory_id, ref.revision),
            )
            return self._insert(conn, scope, replacement, old.kind, content, provenance, source_run_id, now)

    def invalidate(self, scope: str, ref: MemoryRef) -> None:
        with self._transaction() as conn:
            now = _now()
            self._active_record(conn, scope, ref, now)
            conn.execute(
                "UPDATE memory SET valid_to = ? WHERE scope = ? AND memory_id = ? AND revision = ?",
                (now, scope, ref.memory_id, ref.revision),
            )

    def revoke(self, scope: str, memory_id: str) -> None:
        with self._transaction() as conn:
            now = _now()
            changed = conn.execute("""
                UPDATE memory SET revoked_at = COALESCE(revoked_at, ?),
                                  valid_to = COALESCE(valid_to, ?), tombstone = 1
                WHERE scope = ? AND memory_id = ?
            """, (now, now, scope, memory_id))
            if changed.rowcount == 0:
                raise KeyError(memory_id)

    def history(self, scope: str, memory_id: str) -> list[MemoryRecord]:
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM memory WHERE scope = ? AND memory_id = ? ORDER BY revision", (scope, memory_id),
            )
            return [_record(row) for row in rows]

    def recall(self, scope: str, *, limit: int = 16) -> list[MemoryRecord]:
        with self.assembly(scope, set(), limit=limit) as records:
            return records

    @contextmanager
    def assembly(self, scope: str, required: set[MemoryRef], *, limit: int):
        _validate_text(scope, "scope", 256)
        _validate_limit(limit)
        with self._transaction() as conn:
            now = _now()
            self._require_current(conn, scope, required, now)
            rows = conn.execute(
                f"SELECT * FROM memory WHERE scope = ? AND {_ACTIVE} "
                "ORDER BY created_at DESC, memory_id, revision DESC LIMIT ?",
                (scope, now, limit),
            )
            yield [_record(row) for row in rows]

    def require_current(self, scope: str, refs: set[MemoryRef]) -> None:
        with self._transaction() as conn:
            self._require_current(conn, scope, refs, _now())

    def _require_current(self, conn, scope: str, refs: set[MemoryRef], now: str) -> None:
        for ref in refs:
            self._active_record(conn, scope, ref, now)

    @staticmethod
    def _active_record(conn, scope: str, ref: MemoryRef, now: str) -> MemoryRecord:
        row = conn.execute(
            f"SELECT * FROM memory WHERE scope = ? AND memory_id = ? AND revision = ? AND {_ACTIVE}",
            (scope, ref.memory_id, ref.revision, now),
        ).fetchone()
        if row is None:
            raise MemoryConflict("Memory revision is no longer active in this scope")
        return _record(row)

    @staticmethod
    def _validate_entry(content: str, provenance: str, source_run_id: str | None) -> None:
        _validate_text(content, "content", 2048)
        _validate_text(provenance, "provenance", 256)
        if source_run_id is not None:
            _validate_text(source_run_id, "source_run_id", 256)

    @staticmethod
    def _insert(conn, scope, ref, kind, content, provenance, source_run_id, now) -> MemoryRecord:
        conn.execute("""
            INSERT INTO memory (scope, memory_id, revision, kind, content, provenance, source_run_id, created_at, valid_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (scope, ref.memory_id, ref.revision, kind, content, provenance, source_run_id, now, now))
        return MemoryRecord(ref, scope, kind, content, provenance, now, now, None, None, None, False, source_run_id)


@dataclass(frozen=True)
class MemoryScope:
    store: MemoryStore
    scope: str
    limit: int = 16

    def __post_init__(self) -> None:
        _validate_text(self.scope, "scope", 256)
        _validate_limit(self.limit)

    def binding(self) -> dict[str, str]:
        return {"store_id": self.store.store_id, "path": str(self.store.path), "scope": self.scope}
