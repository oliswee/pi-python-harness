from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from pi_agent.types import AgentEvent


class RunTrace:
    """Project Pi events onto an allowlist of diagnostic metadata."""

    def __init__(self, directory: Path, model_id: str, run_id: str | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        attempt_id = uuid4().hex
        self.run_id = run_id or attempt_id
        self.session_id = run_id or uuid4().hex
        self.path = directory / f"{attempt_id}.jsonl"
        self._model_id = model_id
        self._turn = 0
        self._step = 0
        self._started = monotonic()
        self._pending_tools: dict[str, tuple[int, float]] = {}
        self._records: list[dict[str, str | int | None]] = []
        self._record("run_start", "started")

    def observe(self, event: AgentEvent) -> None:
        if event.type == "turn_start":
            self._turn += 1
            self._record("turn_start", "started")
        elif event.type == "tool_execution_start":
            self._step += 1
            self._pending_tools[event.tool_call_id] = (self._step, monotonic())
            self._record("tool_start", "started", call_id=event.tool_call_id, step=self._step)
        elif event.type == "tool_execution_end":
            step, started = self._pending_tools.pop(event.tool_call_id)
            self._record(
                "tool_end",
                "error" if event.is_error else "ok",
                call_id=event.tool_call_id,
                step=step,
                duration_ms=round((monotonic() - started) * 1000),
                error_category="tool_error" if event.is_error else None,
            )

    def finish(self, status: str, error_category: str | None) -> None:
        self._record(
            "run_end", status,
            duration_ms=round((monotonic() - self._started) * 1000),
            error_category=error_category,
        )
        with self.path.open("x", encoding="utf-8", newline="\n") as output:
            for record in self._records:
                output.write(json.dumps(record) + "\n")

    def _record(
        self,
        event_type: str,
        status: str,
        *,
        call_id: str | None = None,
        step: int | None = None,
        duration_ms: int | None = None,
        error_category: str | None = None,
    ) -> None:
        self._records.append({
            "run_id": self.run_id,
            "session_id": self.session_id,
            "turn_id": self._turn or None,
            "step_id": step,
            "tool_call_id": call_id,
            "event_type": event_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "model_id": self._model_id,
            "input_tokens_est": None,
            "output_tokens_est": None,
            "status": status,
            "error_category": error_category,
        })
