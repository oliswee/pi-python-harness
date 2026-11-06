from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import aclosing, contextmanager
from pathlib import Path
from uuid import uuid4

from pi_agent.types import AgentToolResult
from pi_ai.types import AssistantMessage, EventDone, EventError, ToolCall

from .memory_store import MemoryRef

VERSION = 1


class JournalError(RuntimeError):
    """Durable state cannot safely be opened or advanced."""


def _encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def unique_tool_calls(message: AssistantMessage, known_ids) -> list[ToolCall]:
    calls = [item for item in message.content if isinstance(item, ToolCall)]
    ids = {call.id for call in calls}
    if len(ids) != len(calls) or ids.intersection(known_ids):
        raise ValueError("Duplicate tool call ID")
    return calls


def _seal(record: dict) -> dict:
    return {**record, "checksum": digest(record)}


def _check(record: dict) -> None:
    if record["checksum"] != digest({key: value for key, value in record.items() if key != "checksum"}):
        raise JournalError("Journal checksum mismatch")
    if record["version"] != VERSION:
        raise JournalError("Unsupported journal version")


def _lock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        raise JournalError("Cannot acquire the run journal lock") from error


class RunJournal:
    """One logical run's model replies and tool operations."""

    @classmethod
    @contextmanager
    def open(cls, directory: Path, request: dict, *, tool_policy_version: int | None = None):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "writer.lock").open("a+b") as lock:
            lock.seek(0)
            _lock(lock)
            journal = cls(directory)
            try:
                journal._load()
                if journal._state["request"] is None:
                    initial = {**request, "run_id": uuid4().hex}
                    if tool_policy_version is not None:
                        initial["tool_policy_version"] = tool_policy_version
                    journal.append("request", initial)
                elif any(journal._state["request"].get(key) != value for key, value in request.items()):
                    raise JournalError("Request does not match the existing run")
                if tool_policy_version is not None and not journal.uncertain_call_ids:
                    journal._bind_tool_policy(tool_policy_version)
                yield journal
            finally:
                journal._closed = True

    def __init__(self, directory: Path) -> None:
        self.path = directory / "journal.jsonl"
        self.checkpoint = directory / "checkpoint.json"
        self._state: dict = {"request": None, "replies": [], "operations": {}}
        self._seq = 0
        self._head: str | None = None
        self._closed = False
        self.error: JournalError | None = None

    @property
    def run_id(self) -> str:
        return self._state["request"]["run_id"]

    @property
    def memory_refs(self) -> set[MemoryRef]:
        return {MemoryRef(*item) for item in self._state["request"].get("memory_refs", [])}

    @property
    def active_compaction(self) -> dict | None:
        return self._state["request"].get("active_compaction")

    def record_memory_refs(self, refs: set[MemoryRef]) -> None:
        self.append("memory_recalled", {"refs": [[ref.memory_id, ref.revision] for ref in sorted(refs)]})

    @property
    def uncertain_call_ids(self) -> tuple[str, ...]:
        return tuple(
            call_id for call_id, op in self._state["operations"].items()
            if op["state"] == "started" and not op["read_only"]
        )

    def check_active(self) -> None:
        if self.error is not None:
            raise self.error
        if self._closed:
            raise JournalError("Run journal is closed")
        if self.uncertain_call_ids:
            raise JournalError("A write has an uncertain outcome")

    def _bind_tool_policy(self, version: int) -> None:
        existing = self._state["request"].get("tool_policy_version")
        if existing == version:
            return
        if existing is not None:
            raise JournalError("Run journal uses a different tool policy version")
        if any(op["state"] == "succeeded" and op["read_only"] for op in self._state["operations"].values()):
            raise JournalError("Legacy journal has read results without tool policy provenance")
        self.append("tool_policy", {"version": version})

    def replay_stream(self, provider):
        reply_index = 0

        async def stream(model, context, options=None):
            nonlocal reply_index
            self.check_active()
            replies = self._state["replies"]
            if reply_index < len(replies):
                message = AssistantMessage.model_validate(replies[reply_index])
                reply_index += 1
                yield EventDone(type="done", reason=message.stop_reason, message=message)
                return

            async with aclosing(provider(model, context, options)) as events:
                async for event in events:
                    if isinstance(event, EventDone):
                        self.append("reply", {"message": event.message.model_dump(mode="json")})
                        reply_index += 1
                    yield event
                    if isinstance(event, (EventDone, EventError)):
                        return
            raise RuntimeError("Provider stream ended without a terminal event")

        return stream

    def saved_result(self, call_id: str, name: str) -> AgentToolResult | None:
        self.check_active()
        op = self._operation(call_id, name)
        if op["state"] == "succeeded":
            return AgentToolResult.model_validate(op["result"])
        if op["state"] in {"denied", "failed"}:
            raise RuntimeError(op["error"])
        return None

    def start_tool(self, call_id: str, name: str, *, read_only: bool) -> None:
        self.check_active()
        op = self._operation(call_id, name)
        if op["state"] == "planned":
            self.append("approved", {"call_id": call_id, "read_only": read_only})
        elif op["read_only"] is not read_only:
            raise JournalError("Tool risk class differs from the journal")
        self.append("started", {"call_id": call_id})

    def _operation(self, call_id: str, name: str) -> dict:
        op = self._state["operations"].get(call_id)
        if op is None or op["name"] != name:
            raise JournalError("Tool invocation has no matching durable plan")
        return op

    def append(self, kind: str, data: dict) -> None:
        if self.error is not None:
            raise self.error
        if self._closed:
            raise JournalError("Run journal is closed")
        try:
            record = _seal({
                "version": VERSION, "seq": self._seq + 1, "previous": self._head, "kind": kind, "data": data,
            })
            self._apply(kind, data)
            with self.path.open("ab") as handle:
                handle.write(_encode(record) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, ValueError, TypeError, JournalError) as error:
            self.error = JournalError(f"Cannot append journal record {kind}: {error}")
            raise self.error from error
        self._seq, self._head = record["seq"], record["checksum"]

    def finish(self, status: str) -> None:
        self.append("finished", {"status": status, "state_checksum": digest(self._state)})

    def _load(self) -> None:
        if not self.path.exists():
            if self.checkpoint.exists():
                raise JournalError("Run checkpoint exists but its journal is missing")
            return
        records = []
        try:
            with self.path.open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        raise JournalError("Journal has an incomplete final record")
                    record = json.loads(line)
                    _check(record)
                    if record["seq"] != self._seq + 1 or record["previous"] != self._head:
                        raise JournalError("Journal sequence or hash chain is broken")
                    records.append(record)
                    self._seq, self._head = record["seq"], record["checksum"]

            if not records:
                raise JournalError("Existing run journal is empty")
            start = self._load_checkpoint(records)
            for record in records[start:]:
                self._apply(record["kind"], record["data"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise JournalError("Cannot read durable run journal") from error

    def _apply(self, kind: str, data: dict) -> None:
        operations = self._state["operations"]
        if self._state["request"] is None:
            if kind == "request":
                self._state["request"] = data
                return
            raise JournalError("Journal is missing its initial request")
        if kind == "tool_policy" and "tool_policy_version" not in self._state["request"]:
            self._state["request"] = {**self._state["request"], "tool_policy_version": data["version"]}
            return
        if kind == "memory_recalled" and self._state["request"].get("memory") is not None:
            refs = self._state["request"].get("memory_refs", []) + data["refs"]
            self._state["request"] = {**self._state["request"], "memory_refs": refs}
            return
        if kind in {"compaction_staged", "compaction_committed"}:
            self._apply_compaction(kind, data)
            return
        if kind == "reply":
            message = AssistantMessage.model_validate(data["message"])
            if message.stop_reason not in {"toolUse", "stop", "length"}:
                raise JournalError("Unexpected durable model stop reason")
            calls = unique_tool_calls(message, operations)
            operations.update({call.id: {"state": "planned", "name": call.name} for call in calls})
            self._state["replies"].append(data["message"])
            return
        if kind == "finished" and data["status"] in {
            "ok", "error", "aborted", "uncertain", "memory_changed", "context_limit",
        }:
            if data["state_checksum"] == digest(self._state):
                return
            raise JournalError("Finished record does not match journal state")

        op = operations.get(data.get("call_id"))
        if op is not None:
            state = op["state"]
            if kind == "approved" and state == "planned":
                op.update(state=kind, read_only=data["read_only"])
                return
            if kind == "denied" and (state in {"planned", "approved"} or (state == "started" and op["read_only"])):
                op.update(state=kind, error=data["error"])
                return
            if kind == "started" and (state == "approved" or (state == "started" and op["read_only"])):
                op["state"] = kind
                return
            if state == "started" and kind == "succeeded":
                AgentToolResult.model_validate(data["result"])
                op.update(state=kind, result=data["result"])
                return
            if state == "started" and kind == "failed" and op["read_only"]:
                op.update(state=kind, error=data["error"])
                return
        raise JournalError(f"Invalid journal transition: {kind}")

    def _apply_compaction(self, kind: str, data: dict) -> None:
        request = self._state["request"]
        if request.get("context_policy") is None:
            raise JournalError("Compaction requires a bound context policy")
        if kind == "compaction_staged":
            view = data["view"]
            current = self.active_compaction
            if view["version"] != (current["version"] if current is not None else 0) + 1:
                raise JournalError("Invalid compaction version")
            self._state["request"] = {**request, "staged_compaction": view}
            return
        staged = request.get("staged_compaction")
        if staged is None or digest(staged) != data["view_checksum"]:
            raise JournalError("Compaction commit does not match the staged view")
        updated = {**request, "active_compaction": staged}
        updated.pop("staged_compaction")
        self._state["request"] = updated

    def _load_checkpoint(self, records: list[dict]) -> int:
        try:
            snapshot = json.loads(self.checkpoint.read_bytes())
            _check(snapshot)
            seq = snapshot["seq"]
            head, state = snapshot["head"], snapshot["state"]
            if not isinstance(seq, int) or seq <= 0 or set(state) != set(self._state):
                return 0
        except (OSError, ValueError, KeyError, TypeError, JournalError):
            return 0

        if seq > len(records) or head != records[seq - 1]["checksum"]:
            raise JournalError("Run checkpoint is inconsistent with its journal prefix")
        anchor = records[seq - 1]
        if anchor["kind"] != "finished" or anchor["data"]["state_checksum"] != digest(state):
            return 0
        self._state = state
        return seq

    def save_checkpoint(self) -> None:
        snapshot = _seal({
            "version": VERSION, "seq": self._seq, "head": self._head, "state": self._state,
        })
        with tempfile.NamedTemporaryFile(dir=self.checkpoint.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(_encode(snapshot) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                handle.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(self.checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
