from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_coding_agent.core.tools import create_read_tool, create_write_tool
from pi_coding_agent.core.tools.path_utils import resolve_read_path, resolve_to_cwd

from .journal import RunJournal


def create_workspace_tools(workspace: Path, journal: RunJournal | None = None) -> list[AgentTool]:
    cwd = str(workspace)
    return [
        _guard(create_read_tool(cwd), workspace, journal, read_only=True),
        _guard(create_write_tool(cwd), workspace, journal, read_only=False),
    ]


def _require_inside(path: str, workspace: Path) -> None:
    target = Path(path)
    if not target.resolve().is_relative_to(workspace):
        raise PermissionError("Path must resolve inside the workspace")


def _guard(tool: AgentTool, workspace: Path, journal: RunJournal | None, *, read_only: bool) -> AgentTool:
    cwd = str(workspace)

    async def execute(
        call_id: str,
        arguments: dict[str, Any],
        cancel_event: asyncio.Event | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")
        if journal is not None:
            saved = journal.saved_result(call_id, tool.name)
            if saved is not None:
                return saved

        try:
            path = arguments["path"]
            _require_inside(resolve_to_cwd(path, cwd), workspace)
            if read_only:
                _require_inside(resolve_read_path(path, cwd), workspace)
        except (OSError, RuntimeError, ValueError) as error:
            if journal is not None:
                journal.append("denied", {"call_id": call_id, "error": str(error)})
            raise

        if journal is not None:
            journal.start_tool(call_id, tool.name, read_only=read_only)
        try:
            result = await tool.execute(call_id, arguments, cancel_event, on_update)
        except Exception as error:
            if journal is not None and read_only:
                journal.append("failed", {"call_id": call_id, "error": str(error)})
            raise

        if journal is not None:
            journal.append("succeeded", {"call_id": call_id, "result": result.model_dump(mode="json")})
        return result

    return tool.model_copy(update={"execute": execute})
