from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pi_agent import Agent, AgentOptions
from pi_agent.types import StreamFn
from pi_ai.types import AssistantMessage, EventDone, EventError, Model, TextContent

from .context_budget import ContextPolicy, RunBudget
from .journal import RunJournal, digest, unique_tool_calls
from .memory_context import RunMemory
from .memory_store import MemoryScope
from .trace import RunTrace
from .workspace_tools import TOOL_POLICY_VERSION, ToolPolicy, WorkspaceTools

RunStatus = Literal["ok", "error", "aborted", "uncertain", "memory_changed", "context_limit"]


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
    tool_policy: ToolPolicy | None = None,
    memory: MemoryScope | None = None,
    context_policy: ContextPolicy | None = None,
    system_prompt: str = "",
) -> RunResult:
    """Run one prompt, reopening the same logical run when state_dir is supplied."""
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise NotADirectoryError(workspace)
    if memory is not None and memory.store.path.resolve().is_relative_to(workspace):
        raise ValueError("Memory database must be outside the tool workspace")

    policy = tool_policy if tool_policy is not None else ToolPolicy()
    if state_dir is None:
        return await _run(prompt, workspace, trace_dir, model, stream_fn, policy, memory, context_policy, system_prompt)
    state_dir = state_dir.resolve()
    if state_dir.is_relative_to(workspace):
        raise ValueError("Durable state directory must be outside the tool workspace")
    request = {
        "prompt": digest(prompt), "workspace": str(workspace), "model": digest(model.model_dump(mode="json")),
        "memory": memory.binding() if memory is not None else None,
        "context_policy": context_policy.binding() if context_policy is not None else None,
    }
    if system_prompt:
        request["system_prompt"] = digest(system_prompt)
    with RunJournal.open(state_dir, request, tool_policy_version=TOOL_POLICY_VERSION) as journal:
        return await _run(
            prompt, workspace, trace_dir, model, stream_fn, policy, memory, context_policy, system_prompt, journal,
        )


async def _run(
    prompt: str,
    workspace: Path,
    trace_dir: Path,
    model: Model,
    stream_fn: StreamFn,
    policy: ToolPolicy,
    memory: MemoryScope | None,
    context_policy: ContextPolicy | None,
    system_prompt: str,
    journal: RunJournal | None = None,
) -> RunResult:
    trace = RunTrace(trace_dir, model.id, journal.run_id if journal else None)
    if journal and journal.uncertain_call_ids:
        trace.finish("uncertain", "uncertain_write")
        return RunResult(trace.run_id, "uncertain", "", trace.path, journal.uncertain_call_ids)

    run_memory = RunMemory(memory, journal) if memory is not None else None
    if run_memory is not None and not run_memory.current():
        trace.finish("memory_changed", "stale_memory")
        return RunResult(trace.run_id, "memory_changed", "", trace.path)

    tools = WorkspaceTools(
        workspace, policy, journal, run_memory.require_current if run_memory else None,
        record_evidence=context_policy is not None,
    )
    budget = RunBudget(context_policy, workspace, trace, journal, run_memory) if context_policy is not None else None
    if budget is not None:
        stream_fn = budget.wrap(stream_fn)
    if run_memory is not None:
        stream_fn = run_memory.wrap(stream_fn)
    stream = _guard_stream(stream_fn, tools.check_active, track_call_ids=journal is None)
    if journal is not None:
        stream = journal.replay_stream(stream)
    agent = Agent(AgentOptions(
        stream_fn=stream, session_id=trace.session_id, read_parallelism=policy.max_parallel_reads,
        tool_error_max_bytes=4096,
    ))
    agent.set_model(model)
    agent.set_system_prompt(system_prompt)
    agent.set_tools(tools.registered())
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

    uncertain = tools.uncertain_call_ids
    if journal is not None:
        if journal.error is not None:
            raise journal.error
        uncertain = journal.uncertain_call_ids
    if uncertain:
        status, error_category, text = "uncertain", "uncertain_write", ""
    elif run_memory is not None and not run_memory.current():
        status, error_category, text = "memory_changed", "stale_memory", ""
    elif budget is not None and budget.blocked:
        status, error_category, text = "context_limit", "context_budget", ""
    if journal is not None:
        journal.finish(status)
        journal.save_checkpoint()

    trace.finish(status, error_category)
    return RunResult(trace.run_id, status, text, trace.path, uncertain)


def _guard_stream(stream_fn: StreamFn, check_active: Callable[[], None], *, track_call_ids: bool) -> StreamFn:
    known_ids: set[str] = set()

    async def stream(model, context, options=None):
        check_active()
        signal = options.signal if options else None
        if signal and signal.is_set():
            raise RuntimeError("Run cancelled")

        try:
            async with aclosing(stream_fn(model, context, options)) as events:
                async for event in events:
                    if signal and signal.is_set():
                        raise RuntimeError("Run cancelled")
                    if track_call_ids and isinstance(event, EventDone):
                        known_ids.update(call.id for call in unique_tool_calls(event.message, known_ids))
                    if isinstance(event, EventError) and event.error.stop_reason != event.reason:
                        event = event.model_copy(update={
                            "error": event.error.model_copy(update={"stop_reason": event.reason}),
                        })
                    yield event
                    if isinstance(event, (EventDone, EventError)):
                        return
        except asyncio.CancelledError as error:
            # Pi's producer only forwards Exception subclasses to its listener.
            raise RuntimeError("Provider stream cancelled") from error
        raise RuntimeError("Provider stream ended without a terminal event")

    return stream
