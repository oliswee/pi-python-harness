import asyncio
import base64
import os
from pathlib import Path

import pytest

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, EventError, ImageContent, TextContent, ToolCall, ToolResultMessage
from pi_coding_agent.core.tools.read import DefaultReadOperations

from mini_nanobot import run
from mini_nanobot import workspace_tools


def call(call_id, name, **arguments):
    return ToolCall(type="toolCall", id=call_id, name=name, arguments=arguments)


async def exercise(tmp_path, calls, verify, *, policy=None, durable=False):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    async def provider(model, context, options=None):
        if context.messages[-1].role == "user":
            content, reason = calls, "toolUse"
        else:
            results = [message for message in context.messages if isinstance(message, ToolResultMessage)]
            verify(results)
            content, reason = [TextContent(type="text", text="verified")], "stop"
        message = AssistantMessage(
            content=content, api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason=reason,
        )
        yield EventDone(type="done", reason=reason, message=message)

    kwargs = {"tool_policy": policy} if policy else {}
    return await run(
        "Fixture", workspace=workspace, trace_dir=tmp_path / "traces",
        model=get_model("anthropic", "claude-3-5-sonnet-20241022"), stream_fn=provider,
        state_dir=tmp_path / "state" if durable else None, **kwargs,
    )


@pytest.mark.parametrize("durable", [False, True])
async def test_concurrent_reads_preserve_plan_order_and_write_barriers(tmp_path, monkeypatch, durable):
    from mini_nanobot import ToolPolicy

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("a", "b", "c"):
        (workspace / f"{name}.txt").write_text(name, encoding="utf-8")
    read_text = DefaultReadOperations.read_text_file
    first_pair_finished = asyncio.Event()
    active = 0
    peak = 0
    completion_order = []

    async def observe(self, path):
        nonlocal active, peak
        name = Path(path).stem
        active += 1
        peak = max(peak, active)
        try:
            if name == "a":
                await asyncio.wait_for(first_pair_finished.wait(), timeout=2)
            return await read_text(self, path)
        finally:
            active -= 1
            completion_order.append(name)
            if name == "b":
                first_pair_finished.set()

    monkeypatch.setattr(DefaultReadOperations, "read_text_file", observe)
    calls = [
        call("a", "read", path="a.txt"), call("b", "read", path="b.txt"), call("c", "read", path="c.txt"),
        call("write-1", "write", path="changed.txt", content="first"),
        call("after-1", "read", path="changed.txt"),
        call("write-2", "write", path="changed.txt", content="second"),
        call("after-2", "read", path="changed.txt"),
    ]

    def verify(results):
        assert [result.tool_call_id for result in results] == [item.id for item in calls]
        assert all(not result.is_error for result in results)
        assert results[4].content[0].text == "first"
        assert results[6].content[0].text == "second"

    result = await exercise(tmp_path, calls, verify, policy=ToolPolicy(max_parallel_reads=2), durable=durable)

    assert result.status == "ok"
    assert peak == 2
    assert completion_order == ["b", "a", "c", "changed", "changed"]
    assert (workspace / "changed.txt").read_text(encoding="utf-8") == "second"


async def test_policy_denies_sensitive_paths_and_writes_before_execution(tmp_path):
    from mini_nanobot import ToolPolicy

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("private-value", encoding="utf-8")
    calls = [
        call("secret", "read", path=".env"),
        call("write", "write", path="new.txt", content="deny"),
        call("shell", "bash", command="echo blocked"),
    ]

    def verify(results):
        assert [result.tool_call_id for result in results] == ["secret", "write", "shell"]
        assert all(result.is_error for result in results)
        assert "private-value" not in results[0].content[0].text
        assert "disabled" in results[1].content[0].text
        assert "not found" in results[2].content[0].text

    result = await exercise(tmp_path, calls, verify, policy=ToolPolicy(allow_writes=False), durable=True)

    assert result.status == "ok"
    assert not (workspace / "new.txt").exists()


async def test_duplicate_ephemeral_calls_rejected_before_side_effects(tmp_path):
    calls = [call("same", "write", path=name, content="deny") for name in ("a.txt", "b.txt")]
    result = await exercise(tmp_path, calls, lambda _: None)

    assert result.status == "error"
    assert not list((tmp_path / "workspace").iterdir())


async def test_read_timeout_returns_one_error_and_other_results(tmp_path, monkeypatch):
    from mini_nanobot import ToolPolicy

    read_text = DefaultReadOperations.read_text_file
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("slow", "fast"):
        (workspace / f"{name}.txt").write_text(name, encoding="utf-8")

    async def maybe_slow(self, path):
        if Path(path).stem == "slow":
            await asyncio.Event().wait()
        return await read_text(self, path)

    monkeypatch.setattr(DefaultReadOperations, "read_text_file", maybe_slow)
    calls = [call(name, "read", path=f"{name}.txt") for name in ("slow", "fast")]

    def verify(results):
        slow, fast = results
        assert slow.tool_call_id == "slow" and slow.is_error
        assert "timed out" in slow.content[0].text
        assert fast.tool_call_id == "fast" and not fast.is_error
        assert fast.content[0].text == "fast"

    result = await exercise(tmp_path, calls, verify, policy=ToolPolicy(timeout_s=0.5))

    assert result.status == "ok"


@pytest.mark.parametrize("durable", [False, True])
async def test_timed_out_write_blocks_later_mutations(tmp_path, monkeypatch, durable):
    from mini_nanobot import ToolPolicy

    write_factory = workspace_tools.create_write_tool

    def factory(cwd):
        tool = write_factory(cwd)

        async def execute(call_id, params, cancel_event, on_update):
            result = await tool.execute(call_id, params, cancel_event, on_update)
            if call_id == "slow":
                await asyncio.Event().wait()
            return result

        return tool.model_copy(update={"execute": execute})

    monkeypatch.setattr(workspace_tools, "create_write_tool", factory)
    calls = [call(name, "write", path=f"{name}.txt", content=name) for name in ("slow", "next")]

    def unexpected_provider(_):
        pytest.fail("Model should not be called after an uncertain write")

    policy = ToolPolicy(timeout_s=0.5)
    result = await exercise(tmp_path, calls, unexpected_provider, policy=policy, durable=durable)

    assert result.status == "uncertain"
    assert result.uncertain_call_ids == ("slow",)
    assert (tmp_path / "workspace" / "slow.txt").read_text(encoding="utf-8") == "slow"
    assert not (tmp_path / "workspace" / "next.txt").exists()
    if durable:
        restarted = await exercise(tmp_path, calls, unexpected_provider, policy=policy, durable=True)
        assert restarted.status == "uncertain"
        assert not (tmp_path / "workspace" / "next.txt").exists()


async def test_large_binary_attachment_and_schema_error_are_bounded(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jcAAAAABJRU5ErkJggg=="
    )
    (workspace / "big.png").write_bytes(png + b"X" * (70 * 1024))
    calls = [
        call("image", "read", path="big.png"),
        call("invalid", "write", content="X" * (70 * 1024)),
    ]

    def verify(results):
        image, invalid = results
        assert not image.is_error and all(not isinstance(block, ImageContent) for block in image.content)
        assert "inline limit" in image.content[0].text
        assert invalid.is_error and "Validation failed" in invalid.content[0].text
        assert len(invalid.content[0].text.encode("utf-8")) <= 4096

    result = await exercise(tmp_path, calls, verify, durable=True)

    assert result.status == "ok"
    assert (tmp_path / "state" / "journal.jsonl").stat().st_size < 100 * 1024


@pytest.mark.skipif(os.name != "nt", reason="Windows normalizes trailing spaces in file names")
async def test_windows_path_alias_cannot_create_protected_env_file(tmp_path):
    calls = [call("alias", "write", path=".env ", content="denied")]

    def verify(results):
        assert results[0].is_error
        assert "protected workspace path" in results[0].content[0].text

    result = await exercise(tmp_path, calls, verify)

    assert result.status == "ok"
    assert not (tmp_path / "workspace" / ".env").exists()


async def test_cancellation_during_read_batch_prevents_queued_write(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("a", "b"):
        (workspace / f"{name}.txt").write_text(name, encoding="utf-8")
    entered = asyncio.Event()
    release = asyncio.Event()
    drained = asyncio.Event()
    active = 0
    read_text = DefaultReadOperations.read_text_file

    async def delayed_read(self, path):
        nonlocal active
        active += 1
        if active == 2:
            entered.set()
        try:
            await release.wait()
            return await read_text(self, path)
        finally:
            active -= 1
            if active == 0:
                drained.set()

    monkeypatch.setattr(DefaultReadOperations, "read_text_file", delayed_read)
    calls = [call(name, "read", path=f"{name}.txt") for name in ("a", "b")]
    calls.append(call("queued", "write", path="queued.txt", content="deny"))
    task = asyncio.create_task(exercise(tmp_path, calls, lambda _: None, durable=True))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    await asyncio.wait_for(drained.wait(), timeout=5)

    assert not (workspace / "queued.txt").exists()
    assert '"status":"aborted"' in (tmp_path / "state" / "journal.jsonl").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "nt", reason="DOS device aliases are Windows-specific")
async def test_windows_device_name_is_denied_before_resolution(tmp_path, monkeypatch):
    resolve = type(tmp_path).resolve

    def check_resolution(path, *args, **kwargs):
        assert path.name != "NUL.txt", "Device name must be rejected before resolving it"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "resolve", check_resolution)

    def verify(results):
        assert results[0].is_error
        assert "device paths" in results[0].content[0].text

    result = await exercise(tmp_path, [call("device", "write", path="NUL.txt", content="deny")], verify)

    assert result.status == "ok"


async def test_error_terminal_cannot_dispatch_a_tool(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    invocation = 0

    async def provider(model, context, options=None):
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            content = [call("same", "write", path="test.txt", content="first")]
            message = AssistantMessage(
                content=content, api=model.api, provider=model.provider, model=model.id,
                timestamp=0, stop_reason="toolUse",
            )
            yield EventDone(type="done", reason="toolUse", message=message)
        elif invocation == 2:
            content = [call("same", "write", path="test.txt", content="duplicate")]
            message = AssistantMessage(
                content=content, api=model.api, provider=model.provider, model=model.id,
                timestamp=0, stop_reason="toolUse",
            )
            yield EventError(type="error", reason="error", error=message)
        else:
            message = AssistantMessage(
                content=[TextContent(type="text", text="wrong")],
                api=model.api, provider=model.provider, model=model.id, timestamp=0,
            )
            yield EventDone(type="done", reason="stop", message=message)

    result = await run(
        "Fixture", workspace=workspace, trace_dir=tmp_path / "traces", model=model, stream_fn=provider,
    )

    assert result.status == "error"
    assert invocation == 2
    assert (workspace / "test.txt").read_text(encoding="utf-8") == "first"


async def test_legacy_read_cannot_feed_a_new_provider_turn(tmp_path):
    from mini_nanobot.journal import JournalError, RunJournal, digest

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    request = {
        "workspace": str(workspace.resolve()), "prompt": digest("Fixture"), "model": digest(model.model_dump(mode="json")),
    }
    with RunJournal.open(tmp_path / "state", request) as legacy:
        message = AssistantMessage(
            content=[call("secret", "read", path=".env")],
            api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason="toolUse",
        )
        legacy.append("reply", {"message": message.model_dump(mode="json")})
        legacy.start_tool("secret", "read", read_only=True)
        legacy.append("succeeded", {"call_id": "secret", "result": {
            "content": [{"type": "text", "text": "private-value"}],
        }})

    with pytest.raises(JournalError, match="read results without tool policy"):
        await exercise(tmp_path, [], lambda _: None, durable=True)


async def test_legacy_journal_without_reads_adopts_current_policy(tmp_path):
    from mini_nanobot.journal import RunJournal, digest

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "public.txt").write_text("public", encoding="utf-8")
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    request = {
        "workspace": str(workspace.resolve()), "prompt": digest("Fixture"), "model": digest(model.model_dump(mode="json")),
    }
    with RunJournal.open(tmp_path / "state", request) as legacy:
        legacy_id = legacy.run_id
        legacy.finish("error")
        legacy.save_checkpoint()

    def verify(results):
        assert results[0].content[0].text == "public"

    first = await exercise(tmp_path, [call("public", "read", path="public.txt")], verify, durable=True)
    second = await exercise(tmp_path, [], verify, durable=True)

    assert first.status == second.status == "ok"
    assert first.run_id == second.run_id == legacy_id
