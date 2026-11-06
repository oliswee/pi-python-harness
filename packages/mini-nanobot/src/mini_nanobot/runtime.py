from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pi_agent import Agent, AgentOptions
from pi_agent.types import StreamFn
from pi_ai.types import AssistantMessage, Model, TextContent

from .journal import RunJournal, digest
from .trace import RunTrace
from .workspace_tools import create_workspace_tools

RunStatus = Literal["ok", "error", "aborted", "uncertain"]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    text: str
    trace_path: Path
    uncertain_call_ids: tuple[str, ...] = ()


async def run(
    prompt: str,
    *,
    workspace: Path,
    trace_dir: Path,
    model: Model,
    stream_fn: StreamFn,
    state_dir: Path | None = None,
) -> RunResult:
    """Run one prompt, reopening the same logical run when state_dir is supplied."""
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise NotADirectoryError(workspace)

    stream = _guard_stream(stream_fn)
    if state_dir is None:
        return await _run(prompt, workspace, trace_dir, model, stream)
    state_dir = state_dir.resolve()
    if state_dir.is_relative_to(workspace):
        raise ValueError("Durable state directory must be outside the tool workspace")
    request = {"prompt": digest(prompt), "workspace": str(workspace), "model": digest(model.model_dump(mode="json"))}
    with RunJournal.open(state_dir, request) as journal:
        return await _run(prompt, workspace, trace_dir, model, journal.replay_stream(stream), journal)


async def _run(
    prompt: str,
    workspace: Path,
    trace_dir: Path,
    model: Model,
    stream_fn: StreamFn,
    journal: RunJournal | None = None,
) -> RunResult:
    trace = RunTrace(trace_dir, model.id, journal.run_id if journal else None)
    if journal and journal.uncertain_call_ids:
        trace.finish("uncertain", "uncertain_write")
        return RunResult(trace.run_id, "uncertain", "", trace.path, journal.uncertain_call_ids)

    agent = Agent(AgentOptions(stream_fn=stream_fn, session_id=trace.session_id))
    agent.set_model(model)
    agent.set_tools(create_workspace_tools(workspace, journal))
    agent.subscribe(trace.observe)

    prompt_task = asyncio.create_task(agent.prompt(prompt))
    try:
        await asyncio.shield(prompt_task)
    except asyncio.CancelledError:
        # Let Pi signal its detached producer before the prompt listener exits.
        agent.abort()
        prompt_task.cancel()
        await asyncio.gather(prompt_task, return_exceptions=True)
        if journal and journal.error is None:
            journal.finish("aborted")
        trace.finish("aborted", "cancelled")
        raise

    final = agent.state.messages[-1] if agent.state.messages else None
    status: RunStatus = "error"
    error_category = "agent_error"
    text = ""
    if isinstance(final, AssistantMessage):
        if final.stop_reason == "aborted":
            status, error_category = "aborted", "cancelled"
        elif final.stop_reason == "stop" and agent.state.error is None:
            status, error_category = "ok", None
            text = "".join(item.text for item in final.content if isinstance(item, TextContent))

    uncertain = ()
    if journal is not None:
        if journal.error is not None:
            raise journal.error
        uncertain = journal.uncertain_call_ids
        if uncertain:
            status, error_category, text = "uncertain", "uncertain_write", ""
        journal.finish(status)
        journal.save_checkpoint()

    trace.finish(status, error_category)
    return RunResult(trace.run_id, status, text, trace.path, uncertain)


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
