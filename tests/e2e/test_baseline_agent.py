import asyncio
import json
from datetime import datetime, timezone

import pytest

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall, ToolResultMessage, Usage


def scripted_stream(*steps):
    responses = iter(steps)

    async def stream(model, context, options=None):
        content = next(responses)(context)
        stop = "toolUse" if isinstance(content[0], ToolCall) else "stop"
        message = AssistantMessage(
            role="assistant", content=content, api=model.api, provider=model.provider,
            model=model.id, usage=Usage(), stop_reason=stop, timestamp=0,
        )
        yield EventDone(type="done", reason=stop, message=message)

    return stream


def call(call_id, name, **arguments):
    return lambda _: [ToolCall(type="toolCall", id=call_id, name=name, arguments=arguments)]


def tool_result(context, call_id, *, is_error=False):
    result = context.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.tool_call_id == call_id
    assert result.is_error is is_error
    return result.content[0].text


async def run_fixture(tmp_path, workspace, *steps):
    from mini_nanobot import run

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    return await run(
        "private fixture prompt",
        workspace=workspace,
        trace_dir=tmp_path / "traces",
        model=model,
        stream_fn=scripted_stream(*steps),
    )


async def test_reads_fixture_and_writes_verified_result(tmp_path):
    (tmp_path / "numbers.txt").write_text("2\n3\n5\n", encoding="utf-8")

    def write_sum(context):
        contents = tool_result(context, "read-1")
        assert contents == "2\n3\n5\n"
        return call("write-1", "write", path="result.txt", content=str(sum(map(int, contents.splitlines()))))(context)

    def finish(context):
        assert "result.txt" in tool_result(context, "write-1")
        return [TextContent(type="text", text="10")]

    result = await run_fixture(tmp_path, tmp_path, call("read-1", "read", path="numbers.txt"), write_sum, finish)

    assert result.text == "10"
    assert result.status == "ok"
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "10"
    raw_trace = result.trace_path.read_text(encoding="utf-8")
    for private_value in ("private fixture prompt", "numbers.txt", "result.txt", "2\\n3\\n5"):
        assert private_value not in raw_trace
    records = [json.loads(line) for line in raw_trace.splitlines()]
    assert records[0]["event_type"] == "run_start"
    assert records[-1]["event_type"] == "run_end"
    assert records[-1]["status"] == "ok"
    assert {record["run_id"] for record in records} == {result.run_id}
    assert len({record["session_id"] for record in records}) == 1
    for record in records:
        assert datetime.fromisoformat(record["timestamp_utc"]).tzinfo == timezone.utc
        assert record["model_id"] == "claude-3-5-sonnet-20241022"
        assert record["input_tokens_est"] is None
        assert record["output_tokens_est"] is None
    tools = [record for record in records if record["tool_call_id"]]
    assert [(record["event_type"], record["tool_call_id"]) for record in tools] == [
        ("tool_start", "read-1"), ("tool_end", "read-1"),
        ("tool_start", "write-1"), ("tool_end", "write-1"),
    ]
    for start, end in (tools[:2], tools[2:]):
        assert start["turn_id"] == end["turn_id"]
        assert start["step_id"] == end["step_id"]
        assert end["status"] == "ok"
        assert end["duration_ms"] >= 0
    assert tools[0]["step_id"] != tools[2]["step_id"]


async def test_invalid_write_arguments_return_paired_error(tmp_path):
    target = tmp_path / "protected.txt"
    target.write_text("keep", encoding="utf-8")

    def finish(context):
        error = tool_result(context, "invalid-1", is_error=True)
        assert "Validation failed" in error and "content" in error
        return [TextContent(type="text", text="Invalid write")]

    result = await run_fixture(tmp_path, tmp_path, call("invalid-1", "write", path=target.name), finish)

    assert result.text == "Invalid write"
    assert target.read_text(encoding="utf-8") == "keep"
    records = [json.loads(line) for line in result.trace_path.read_text(encoding="utf-8").splitlines()]
    start, end = [record for record in records if record["tool_call_id"] == "invalid-1"]
    assert start["step_id"] == end["step_id"]
    assert end["status"] == "error"
    assert end["error_category"] == "tool_error"


@pytest.mark.parametrize("tool", ["read", "write"])
async def test_file_tools_deny_paths_outside_workspace(tmp_path, tool):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    protected = tmp_path / "protected.txt"
    protected.write_text("outside content", encoding="utf-8")
    args = {"path": "@../protected.txt"}
    if tool == "write":
        args["content"] = "overwrite"

    def finish(context):
        assert "outside content" not in tool_result(context, "escape-1", is_error=True)
        return [TextContent(type="text", text="Path denied")]

    result = await run_fixture(tmp_path, workspace, call("escape-1", tool, **args), finish)

    assert result.text == "Path denied"
    assert protected.read_text(encoding="utf-8") == "outside content"


async def test_model_failure_is_visible_without_leaking_its_message(tmp_path):
    def fail(_):
        raise RuntimeError("private provider error")

    result = await run_fixture(tmp_path, tmp_path, fail)

    assert result.status == "error"
    assert result.text == ""
    raw_trace = result.trace_path.read_text(encoding="utf-8")
    assert "private provider error" not in raw_trace
    end = json.loads(raw_trace.splitlines()[-1])
    assert end["event_type"] == "run_end"
    assert end["error_category"] == "agent_error"


async def test_cancelled_run_blocks_late_model_write(tmp_path):
    from mini_nanobot import run

    entered = asyncio.Event()
    release = asyncio.Event()
    provider_closed = asyncio.Event()
    late_write = call("late-1", "write", path="late.txt", content="late write")

    scripted = scripted_stream(late_write)
    signals = []

    async def delayed_stream(model, context, options=None):
        signals.append(options.signal)
        entered.set()
        try:
            await release.wait()
            async for event in scripted(model, context, options):
                yield event
        finally:
            provider_closed.set()

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    task = asyncio.create_task(
        run("Write a file", workspace=tmp_path, trace_dir=tmp_path, model=model, stream_fn=delayed_stream)
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    await asyncio.wait_for(provider_closed.wait(), timeout=5)

    assert len(signals) == 1 and signals[0].is_set()
    assert not (tmp_path / "late.txt").exists()
    trace, = tmp_path.glob("*.jsonl")
    end = json.loads(trace.read_text(encoding="utf-8").splitlines()[-1])
    assert end["status"] == "aborted"


async def test_provider_cancellation_terminates_run(tmp_path):
    def cancel(_):
        raise asyncio.CancelledError()

    result = await asyncio.wait_for(run_fixture(tmp_path, tmp_path, cancel), timeout=2)

    assert result.status == "error"
    end = json.loads(result.trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert end["event_type"] == "run_end"
    assert end["error_category"] == "agent_error"
