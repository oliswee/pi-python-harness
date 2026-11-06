from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pi_agent import Agent, AgentOptions
from pi_agent.types import StreamFn
from pi_ai.types import AssistantMessage, Model, TextContent

from .trace import RunTrace
from .workspace_tools import create_workspace_tools

RunStatus = Literal["ok", "error", "aborted"]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    text: str
    trace_path: Path


async def run(
    prompt: str,
    *,
    workspace: Path,
    trace_dir: Path,
    model: Model,
    stream_fn: StreamFn,
) -> RunResult:
    """Run one prompt through Pi with an explicitly supplied model stream."""
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise NotADirectoryError(workspace)

    trace = RunTrace(trace_dir, model.id)
    agent = Agent(AgentOptions(stream_fn=_guard_stream(stream_fn), session_id=trace.session_id))
    agent.set_model(model)
    agent.set_tools(create_workspace_tools(workspace))
    agent.subscribe(trace.observe)

    prompt_task = asyncio.create_task(agent.prompt(prompt))
    try:
        await asyncio.shield(prompt_task)
    except asyncio.CancelledError:
        # Let Pi signal its detached producer before the prompt listener exits.
        agent.abort()
        prompt_task.cancel()
        await asyncio.gather(prompt_task, return_exceptions=True)
        trace.finish("aborted", "cancelled")
        raise

    final = agent.state.messages[-1]
    status: RunStatus = "error"
    error_category = "agent_error"
    text = ""
    if isinstance(final, AssistantMessage):
        if final.stop_reason == "aborted":
            status, error_category = "aborted", "cancelled"
        elif final.stop_reason == "stop" and agent.state.error is None:
            status, error_category = "ok", None
            text = "".join(item.text for item in final.content if isinstance(item, TextContent))

    trace.finish(status, error_category)
    return RunResult(trace.run_id, status, text, trace.path)


def _guard_stream(stream_fn: StreamFn) -> StreamFn:
    async def stream(model, context, options=None):
        signal = options.signal if options else None
        if signal and signal.is_set():
            raise RuntimeError("Run cancelled")

        try:
            async with aclosing(stream_fn(model, context, options)) as events:
                async for event in events:
                    if signal and signal.is_set():
                        raise RuntimeError("Run cancelled")
                    yield event
        except asyncio.CancelledError as error:
            # Pi's producer only forwards Exception subclasses to its listener.
            raise RuntimeError("Provider stream cancelled") from error

    return stream
