from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_SCHEMA_VERSION = 2
_ACTIVE = (
    "valid_from <= ? AND (expires_at IS NULL OR expires_at > ?) "
    "AND valid_to IS NULL AND revoked_at IS NULL AND tombstone = 0"
)


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
    expires_at: str | None = None


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


def _search_terms(query: str) -> str:
    if not isinstance(query, str) or len(query.encode("utf-8")) > 4096:
        raise ValueError("Memory query must be text of at most 4096 UTF-8 bytes")
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.lower())))[:32]
    return " OR ".join(f'"{term}"' for term in terms)


def _ensure_search_index(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT 1 FROM sqlite_schema WHERE name = 'memory_fts'").fetchone() is not None:
        return
    try:
        conn.execute("CREATE VIRTUAL TABLE memory_fts USING fts5(content, content='memory', content_rowid='rowid')")
    except sqlite3.OperationalError as error:
        if "no such module: fts5" in str(error):
            raise RuntimeError("Ranked memory recall requires SQLite FTS5") from error
        raise
    conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
    # Memory revisions are append-only. Validity changes are filtered on the authority table.
    conn.execute("""
        CREATE TRIGGER memory_fts_insert AFTER INSERT ON memory BEGIN
            INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
        END
    """)


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
                        expires_at TEXT,
                        PRIMARY KEY (scope, memory_id, revision)
                    )
                """)
                conn.execute("""
                    CREATE INDEX memory_recall ON memory (scope, created_at DESC, memory_id, revision)
                    WHERE valid_to IS NULL AND revoked_at IS NULL AND tombstone = 0
                """)
            elif version == 1:
                conn.execute("ALTER TABLE memory ADD COLUMN expires_at TEXT")
            elif version != _SCHEMA_VERSION:
                raise ValueError(f"Unsupported memory database version {version}")
            if version != _SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
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
        expires_at: datetime | None = None,
    ) -> MemoryRecord:
        _validate_text(scope, "scope", 256)
        _validate_text(kind, "kind", 64)
        self._validate_entry(content, provenance, source_run_id)
        if expires_at is not None and (not isinstance(expires_at, datetime) or expires_at.utcoffset() is None):
            raise ValueError("Memory expiry requires a timezone-aware datetime")
        deadline = expires_at.astimezone(timezone.utc).isoformat(timespec="microseconds") if expires_at is not None else None
        with self._transaction() as conn:
            return self._insert(
                conn, scope, MemoryRef(uuid4().hex, 1), kind, content, provenance, source_run_id, _now(), deadline,
            )

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
            return self._insert(conn, scope, replacement, old.kind, content, provenance, source_run_id, now, old.expires_at)

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

    def recall(self, scope: str, *, limit: int = 16, query: str | None = None) -> list[MemoryRecord]:
        with self.assembly(scope, set(), limit=limit, query=query) as records:
            return records

    @contextmanager
    def assembly(self, scope: str, required: set[MemoryRef], *, limit: int, query: str | None = None):
        _validate_text(scope, "scope", 256)
        _validate_limit(limit)
        terms = _search_terms(query) if query is not None else None
        with self._transaction() as conn:
            now = _now()
            self._require_current(conn, scope, required, now)
            if terms == "":
                yield []
                return
            if terms is None:
                rows = conn.execute(
                    f"SELECT * FROM memory WHERE scope = ? AND {_ACTIVE} "
                    "ORDER BY created_at DESC, memory_id, revision DESC LIMIT ?",
                    (scope, now, now, limit),
                )
            else:
                _ensure_search_index(conn)
                rows = conn.execute(
                    f"SELECT memory.* FROM memory_fts JOIN memory ON memory.rowid = memory_fts.rowid "
                    f"WHERE memory_fts MATCH ? AND scope = ? AND {_ACTIVE} "
                    "ORDER BY rank, created_at DESC, memory_id, revision DESC LIMIT ?",
                    (terms, scope, now, now, limit),
                )
            yield [_record(row) for row in rows]

    def require_current(self, scope: str, refs: set[MemoryRef]) -> None:
        with self.guard_current(scope, refs):
            pass

    @contextmanager
    def guard_current(self, scope: str, refs: set[MemoryRef]):
        with self._transaction() as conn:
            self._require_current(conn, scope, refs, _now())
            yield

    def _require_current(self, conn, scope: str, refs: set[MemoryRef], now: str) -> None:
        for ref in refs:
            self._active_record(conn, scope, ref, now)

    @staticmethod
    def _active_record(conn, scope: str, ref: MemoryRef, now: str) -> MemoryRecord:
        row = conn.execute(
            f"SELECT * FROM memory WHERE scope = ? AND memory_id = ? AND revision = ? AND {_ACTIVE}",
            (scope, ref.memory_id, ref.revision, now, now),
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
    def _insert(conn, scope, ref, kind, content, provenance, source_run_id, now, expires_at) -> MemoryRecord:
        conn.execute("""
            INSERT INTO memory (
                scope, memory_id, revision, kind, content, provenance, source_run_id, created_at, valid_from, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (scope, ref.memory_id, ref.revision, kind, content, provenance, source_run_id, now, now, expires_at))
        return MemoryRecord(ref, scope, kind, content, provenance, now, now, None, None, None, False, source_run_id, expires_at)


@dataclass(frozen=True)
class MemoryScope:
    store: MemoryStore
    scope: str
    limit: int = 16
    query: str | None = None

    def __post_init__(self) -> None:
        _validate_text(self.scope, "scope", 256)
        _validate_limit(self.limit)
        if self.query is not None:
            _search_terms(self.query)

    def binding(self) -> dict[str, str | int]:
        binding = {"store_id": self.store.store_id, "path": str(self.store.path), "scope": self.scope}
        if self.query is not None:
            binding.update(query_sha256=hashlib.sha256(self.query.encode("utf-8")).hexdigest(), query_limit=self.limit)
        return binding
