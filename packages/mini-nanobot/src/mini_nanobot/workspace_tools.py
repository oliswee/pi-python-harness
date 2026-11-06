from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_coding_agent.core.tools import create_read_tool, create_write_tool
from pi_coding_agent.core.tools.path_utils import resolve_read_path, resolve_to_cwd


def create_workspace_tools(workspace: Path) -> list[AgentTool]:
    cwd = str(workspace)
    return [
        _guard(create_read_tool(cwd), workspace),
        _guard(create_write_tool(cwd), workspace),
    ]


def _require_inside(path: str, workspace: Path) -> None:
    target = Path(path)
    if not target.resolve().is_relative_to(workspace):
        raise PermissionError("Path must resolve inside the workspace")


def _guard(tool: AgentTool, workspace: Path) -> AgentTool:
    cwd = str(workspace)

    async def execute(
        call_id: str,
        arguments: dict[str, Any],
        cancel_event: asyncio.Event | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        path = arguments["path"]
        _require_inside(resolve_to_cwd(path, cwd), workspace)
        if tool.name == "read":
            _require_inside(resolve_read_path(path, cwd), workspace)
        return await tool.execute(call_id, arguments, cancel_event, on_update)

    return tool.model_copy(update={"execute": execute})
