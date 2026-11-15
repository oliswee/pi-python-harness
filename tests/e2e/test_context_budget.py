import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall, ToolResultMessage

from mini_nanobot import run

PROMPT = "Read the facts and cross-check them. Do not write any files."
WORKER = Path(__file__).with_name("context_budget_worker.py")


def context_policy(summarizer=None):
    from mini_nanobot import ContextPolicy, FactRule

    rules = (
        FactRule("port", "north.txt", "Port:"),
        FactRule("region", "south.txt", "Region:"),
    )
    return ContextPolicy(window_tokens=7000, reserve_tokens=512, fact_rules=rules, summarizer=summarizer)


def reply(model, *content, reason="stop"):
    message = AssistantMessage(
        content=list(content), api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason=reason,
    )
    return EventDone(type="done", reason=reason, message=message)


def read(call_id, path):
    return ToolCall(type="toolCall", id=call_id, name="read", arguments={"path": path})


@pytest.fixture
def fixture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "north.txt").write_text("Port: 8181\n" + "north filler\n" * 400, encoding="utf-8")
    (workspace / "south.txt").write_text("Region: eu-test\n" + "south filler\n" * 400, encoding="utf-8")
    (workspace / "check.txt").write_text("Signature: 73", encoding="utf-8")

    async def exercise(provider, context_policy, *, durable=False, memory=None, prompt=PROMPT):
        return await run(
            prompt, workspace=workspace, trace_dir=tmp_path / "traces",
            model=get_model("anthropic", "claude-3-5-sonnet-20241022"), stream_fn=provider,
            state_dir=tmp_path / "state" if durable else None, context_policy=context_policy, memory=memory,
        )

    return context_policy, exercise


def document_provider(calls):
    async def provider(model, context, options=None):
        calls.append(context)
        assert options.max_tokens == 512
        assert context.messages[0].content == [TextContent(type="text", text=PROMPT)]
        if len(calls) == 1:
            yield reply(model, read("north", "north.txt"), read("south", "south.txt"), reason="toolUse")
        else:
            assert "8181" in context.system_prompt and "eu-test" in context.system_prompt
            assert "north.txt" in context.system_prompt and "south.txt" in context.system_prompt
            assert "filler" not in str(context.messages)
            if len(calls) == 2:
                assert len(context.messages) == 4
                assert [message.tool_call_id for message in context.messages[2:]] == ["north", "south"]
                yield reply(model, read("verify", "check.txt"), reason="toolUse")
            else:
                assert isinstance(context.messages[-1], ToolResultMessage)
                assert context.messages[-1].content[0].text == "Signature: 73"
                yield reply(model, TextContent(type="text", text="8181 / eu-test / 73"))

    return provider


async def test_compaction_retains_exact_sources_constraints_and_followup_read(tmp_path, fixture):
    policy, exercise = fixture
    calls = []
    summaries = []

    async def summarize(model, context, options=None):
        assert not context.tools
        assert options.max_tokens == 512
        facts = json.loads(context.messages[0].content)
        summaries.append(facts)
        assert {fact["key"]: fact["call_id"] for fact in facts} == {"port": "north", "region": "south"}
        yield reply(model, TextContent(type="text", text=json.dumps(facts[::-1])))

    result = await exercise(document_provider(calls), policy(summarize), durable=True)

    assert result.status == "ok" and result.text == "8181 / eu-test / 73"
    assert len(calls) == 3 and len(summaries) == 1
    rows = [json.loads(line) for line in result.trace_path.read_text(encoding="utf-8").splitlines()]
    compacted, = [row for row in rows if row["event_type"] == "context_compacted"]
    assert compacted["before_input_tokens_est"] > 7000
    assert compacted["input_tokens_est"] + compacted["output_tokens_est"] <= 7000
    assert any(row["event_type"] == "context_reused" for row in rows)
    assert "eu-test" not in result.trace_path.read_text(encoding="utf-8")

    records = [json.loads(line) for line in (tmp_path / "state" / "journal.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in records if row["kind"].startswith("compaction_")] == [
        "compaction_staged", "compaction_committed",
    ]


async def test_incomplete_or_fabricated_summary_is_rejected(tmp_path, fixture):
    policy, exercise = fixture
    calls = []

    async def invalid(model, context, options=None):
        facts = json.loads(context.messages[0].content)
        facts[0]["call_id"] = "fabricated"
        yield reply(model, TextContent(type="text", text=json.dumps(facts)))

    result = await exercise(document_provider(calls), policy(invalid), durable=True)

    assert result.status == "context_limit"
    assert len(calls) == 1
    assert "compaction_committed" not in (tmp_path / "state" / "journal.jsonl").read_text()


async def test_overhead_and_response_reserve_block_even_a_short_prompt(tmp_path, fixture):
    from mini_nanobot import ContextPolicy

    _, exercise = fixture
    calls = []
    result = await exercise(document_provider(calls), ContextPolicy(window_tokens=800, reserve_tokens=400))
    assert result.status == "context_limit"
    assert not calls


async def test_memory_revoked_during_summary_cannot_be_committed(tmp_path, fixture):
    from mini_nanobot import MemoryScope, MemoryStore

    policy, exercise = fixture
    store = MemoryStore(tmp_path / "memory.sqlite")
    record = store.create("fixture", kind="fact", content="temporary preference", provenance="operator")
    calls = []

    async def summarize(model, context, options=None):
        store.revoke("fixture", record.ref.memory_id)
        yield reply(model, TextContent(type="text", text=context.messages[0].content))

    result = await exercise(document_provider(calls), policy(summarize), durable=True, memory=MemoryScope(store, "fixture"))

    assert result.status == "memory_changed"
    assert len(calls) == 1
    assert "compaction_committed" not in (tmp_path / "state" / "journal.jsonl").read_text()


@pytest.mark.parametrize(("boundary", "summaries_on_resume"), [("staged", 1), ("committed", 0)])
def test_restart_only_uses_committed_context(tmp_path, fixture, boundary, summaries_on_resume):
    def attempt(fault):
        return subprocess.run(
            [sys.executable, str(WORKER), str(tmp_path), fault], check=False,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    crash = attempt(boundary)
    assert crash.returncode == 86, crash.stderr
    assert not (tmp_path / "state" / "checkpoint.json").exists()
    journal = tmp_path / "state" / "journal.jsonl"
    original = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert original[-1]["kind"] == f"compaction_{boundary}"

    restarted = attempt("")
    assert restarted.returncode == 0, restarted.stderr
    result = json.loads(restarted.stdout)
    assert result == {"status": "ok", "text": "8181 / eu-test", "provider_calls": 1, "summaries": summaries_on_resume}
    final = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert final[:len(original)] == original
    assert sum(row["kind"] == "started" for row in final) == 2
    assert sum(row["kind"] == "compaction_committed" for row in final) == 1


async def test_failed_recompaction_keeps_previous_active_context(tmp_path, fixture):
    policy, exercise = fixture
    (tmp_path / "workspace" / "check.txt").write_text("check filler\n" * 600, encoding="utf-8")
    calls = []
    count = 0

    async def summarize(model, context, options=None):
        nonlocal count
        count += 1
        response = context.messages[0].content if count == 1 else "[]"
        yield reply(model, TextContent(type="text", text=response))

    result = await exercise(document_provider(calls), policy(summarize), durable=True)

    assert result.status == "context_limit"
    assert len(calls) == count == 2
    state = json.loads((tmp_path / "state" / "checkpoint.json").read_text(encoding="utf-8"))["state"]
    assert state["request"]["active_compaction"]["version"] == 1
    assert "staged_compaction" not in state["request"]


@pytest.mark.parametrize("hardlink", [False, True])
async def test_write_evidence_is_preserved_and_alias_changes_block_stale_facts(tmp_path, fixture, hardlink):
    policy, exercise = fixture
    workspace = tmp_path / "workspace"
    target = "alias.txt" if hardlink else "unrelated.txt"
    if hardlink:
        try:
            os.link(workspace / "north.txt", workspace / target)
        except OSError as error:
            pytest.skip(f"Fixture filesystem does not support hardlinks: {error}")
    calls = []

    async def provider(model, context, options=None):
        calls.append(context)
        if len(calls) == 1:
            yield reply(model, read("north", "north.txt"), read("south", "south.txt"), reason="toolUse")
        elif len(calls) == 2:
            yield reply(model, ToolCall(
                type="toolCall", id="write", name="write", arguments={"path": target, "content": "Port: 9999"},
            ), reason="toolUse")
        else:
            assert "8181" in context.system_prompt
            assert context.messages[-1].tool_call_id == "write"
            yield reply(model, TextContent(type="text", text="complete"))

    result = await exercise(provider, policy(), durable=True, prompt="Read the two facts, then update the selected file.")

    assert (workspace / target).read_text(encoding="utf-8") == "Port: 9999"
    if hardlink:
        assert (workspace / "north.txt").read_text(encoding="utf-8") == "Port: 9999"
        assert result.status == "context_limit" and len(calls) == 2
    else:
        assert result.status == "ok" and len(calls) == 3


async def test_durable_run_binds_summary_verification_requirement(tmp_path, fixture):
    from mini_nanobot.journal import JournalError

    policy, exercise = fixture
    result = await exercise(document_provider([]), policy(), durable=True)
    assert result.status == "ok"

    async def summarize(model, context, options=None):
        yield reply(model, TextContent(type="text", text="[]"))

    with pytest.raises(JournalError, match="Request does not match"):
        await exercise(document_provider([]), policy(summarize), durable=True)
