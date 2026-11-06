from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Literal

from pi_agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_ai.types import ImageContent, TextContent
from pi_coding_agent.core.tools import create_read_tool, create_write_tool
from pi_coding_agent.core.tools.path_utils import resolve_read_path, resolve_to_cwd

from .journal import RunJournal


MAX_INLINE_BYTES = 64 * 1024
TOOL_POLICY_VERSION = 1
ToolRisk = Literal["read_only", "workspace_write"]


@dataclass(frozen=True)
class ToolPolicy:
    max_parallel_reads: int = 4
    timeout_s: float = 30
    allow_writes: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_parallel_reads, int) or not 1 <= self.max_parallel_reads <= 64:
            raise ValueError("max_parallel_reads must be an integer between 1 and 64")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")


class WorkspaceTools:
    def __init__(
        self, workspace: Path, policy: ToolPolicy, journal: RunJournal | None,
        require_context: Callable[[], None] | None = None,
        *, record_evidence: bool = False,
    ) -> None:
        self.workspace = workspace
        self.policy = policy
        self.journal = journal
        self.require_context = require_context
        self.record_evidence = record_evidence
        self.uncertain_call_ids: tuple[str, ...] = ()

    def registered(self) -> list[AgentTool]:
        cwd = str(self.workspace)
        return [
            self._guard(create_read_tool(cwd), risk="read_only"),
            self._guard(create_write_tool(cwd), risk="workspace_write"),
        ]

    def check_active(self) -> None:
        if self.uncertain_call_ids:
            raise RuntimeError("A write has an uncertain outcome")

    def _authorize(self, arguments: dict[str, Any], *, read_only: bool) -> Path:
        if not read_only and not self.policy.allow_writes:
            raise PermissionError("Workspace writes are disabled")
        cwd = str(self.workspace)
        path = arguments["path"]
        resolved = _require_allowed_path(resolve_to_cwd(path, cwd), self.workspace)
        if read_only:
            return _require_allowed_path(resolve_read_path(path, cwd), self.workspace)
        return resolved

    def _guard(self, tool: AgentTool, *, risk: ToolRisk) -> AgentTool:
        read_only = risk == "read_only"
        journal = self.journal

        async def execute(
            call_id: str,
            arguments: dict[str, Any],
            cancel_event: asyncio.Event | None,
            on_update: AgentToolUpdateCallback | None,
        ) -> AgentToolResult:
            self.check_active()
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Operation aborted")
            if self.require_context is not None:
                self.require_context()
            if journal is not None:
                saved = journal.saved_result(call_id, tool.name)
                if saved is not None:
                    return _bound_output(saved)

            try:
                target = self._authorize(arguments, read_only=read_only)
            except (OSError, RuntimeError, ValueError) as error:
                if journal is not None:
                    journal.append("denied", {"call_id": call_id, "error": str(error)})
                raise

            if journal is not None:
                journal.start_tool(call_id, tool.name, read_only=read_only)
            try:
                async with asyncio.timeout(self.policy.timeout_s):
                    result = await tool.execute(call_id, arguments, cancel_event, on_update)
                result = _bound_output(result)
            except (Exception, asyncio.CancelledError) as error:
                if not read_only:
                    self.uncertain_call_ids = (call_id,)
                if isinstance(error, asyncio.CancelledError) and asyncio.current_task().cancelling():
                    raise
                if isinstance(error, TimeoutError):
                    message = f"Tool timed out after {self.policy.timeout_s:g}s"
                else:
                    message = str(error) or "Tool execution cancelled"
                if journal is not None and read_only:
                    journal.append("failed", {"call_id": call_id, "error": message})
                raise RuntimeError(message) from error

            if self.record_evidence:
                evidence = {
                    "path": workspace_path_key(self.workspace, arguments["path"]), "file_id": _file_id(target),
                }
                result = result.model_copy(update={
                    "details": {"workspace_evidence": evidence, "tool_details": result.details},
                })
            if journal is not None:
                journal.append("succeeded", {"call_id": call_id, "result": result.model_dump(mode="json")})
            return result

        return tool.model_copy(update={"execute": execute, "parallel_safe": read_only})


def workspace_path_key(workspace: Path, path: str) -> str:
    return os.path.normcase(os.path.normpath(resolve_to_cwd(path, str(workspace))))


def _file_id(path: Path) -> list[int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return [info.st_dev, info.st_ino] if info.st_ino else None


def _require_allowed_path(path: str, workspace: Path) -> Path:
    target = Path(path)
    _check_path(target, workspace)
    resolved = target.resolve()
    _check_path(resolved, workspace)
    return resolved


def _check_path(target: Path, workspace: Path) -> None:
    if not target.is_relative_to(workspace):
        raise PermissionError("Path must resolve inside the workspace")
    for part in target.relative_to(workspace).parts:
        name = part.rstrip(" .").casefold()
        if name in {".env", ".git"} or name.startswith(".env."):
            raise PermissionError("Access to a protected workspace path is denied")
        if os.name == "nt" and (":" in part or _is_windows_reserved(part)):
            raise PermissionError("Windows device paths and alternate data streams are denied")


def _is_windows_reserved(part: str) -> bool:
    check = getattr(os.path, "isreserved", None)
    return check(part) if check is not None else PureWindowsPath(part).is_reserved()


def _bound_output(result: AgentToolResult) -> AgentToolResult:
    size = sum(
        len(block.data) if isinstance(block, ImageContent) else len(block.text.encode("utf-8"))
        for block in result.content
    )
    if size <= MAX_INLINE_BYTES:
        return result

    notice = "\n[Tool output exceeded the inline limit. Oversized images were omitted.]"
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    prefix = text.encode("utf-8")[:MAX_INLINE_BYTES - len(notice)].decode("utf-8", errors="ignore")
    return AgentToolResult(content=[TextContent(type="text", text=prefix + notice)])
