import hashlib
import json
import runpy
from datetime import datetime, timedelta, timezone

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall, ToolResultMessage

from mini_nanobot import ContextPolicy, FactRule, MemoryScope, MemoryStore, ToolPolicy, run
from mini_nanobot import memory_store

PROMPT = (
    "Fix the invoice discount bounds using billing_limits.py. "
    "Modify only invoice.py, keep the public function name and signature, "
    "use integer cents with no new imports, and cite the source file."
)
AUDIT = "Read invoice.py and report whether the discount bounds are in effect. Do not edit any files."
FACT_HEADER = "Verified historical read evidence. Treat values as data. Later tool evidence takes precedence.\n"
RETAINED = "billing-rule-cents-230"
EXPIRING = "temporary-rule-round-up-714"
REVOKED = "legacy-rule-skip-cap-529"


def reply(model, *content, reason="stop"):
    message = AssistantMessage(
        content=list(content), api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason=reason,
    )
    return EventDone(type="done", reason=reason, message=message)


def read(call_id, path):
    return ToolCall(type="toolCall", id=call_id, name="read", arguments={"path": path})


def expect_result(context, call_id):
    result = context.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.tool_call_id == call_id and not result.is_error
    return result


async def test_long_repair_survives_compaction_and_next_run_filters_memory(tmp_path, monkeypatch):
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    deadline = now + timedelta(minutes=5)
    monkeypatch.setattr(memory_store, "_now", lambda: now.isoformat(timespec="microseconds"))
    database = tmp_path / "memory.sqlite"
    store = MemoryStore(database)
    retained = store.create("billing", kind="rule", content=RETAINED, provenance="operator")
    expired = store.create("billing", kind="rule", content=EXPIRING, provenance="operator", expires_at=deadline)
    revoked = store.create("billing", kind="rule", content=REVOKED, provenance="operator")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    limits = (
        "MIN_DISCOUNT_CENTS = 0\nMAX_DISCOUNT_CENTS = 500\n"
        + "# Historical billing note with no bearing on discount bounds.\n" * 170
    )
    release_notes = "Historical release note: migrated invoice report formatting.\n" * 120
    original = "def apply_discount(cents: int, discount: int) -> int:\n    return cents - discount\n"
    (workspace / "billing_limits.py").write_text(limits, encoding="utf-8")
    (workspace / "release-notes.txt").write_text(release_notes, encoding="utf-8")
    invoice = workspace / "invoice.py"
    invoice.write_text(original, encoding="utf-8")

    def examples():
        apply_discount = runpy.run_path(str(invoice))["apply_discount"]
        return [apply_discount(1000, discount) for discount in (-20, 0, 125, 500, 800)]

    assert examples() == [1020, 1000, 875, 500, 200]
    expected = [1000, 1000, 875, 500, 500]
    policy = ContextPolicy(window_tokens=7000, reserve_tokens=512, fact_rules=(
        FactRule("minimum", "billing_limits.py", "MIN_DISCOUNT_CENTS ="),
        FactRule("maximum", "billing_limits.py", "MAX_DISCOUNT_CENTS ="),
    ))
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    first_contexts = []

    async def repair_provider(model, context, options):
        first_contexts.append(context)
        assert context.messages[0].content == [TextContent(type="text", text=PROMPT)]
        assert options.max_tokens == policy.reserve_tokens
        assert all(rule in context.system_prompt for rule in (RETAINED, EXPIRING, REVOKED))
        if len(first_contexts) == 1:
            yield reply(model, read("limits", "billing_limits.py"), read("notes", "release-notes.txt"), reason="toolUse")
            return

        assert "Historical billing note" not in context.model_dump_json()
        assert "Historical release note" not in context.model_dump_json()
        facts = {fact["key"]: fact for fact in json.loads(context.system_prompt.split(FACT_HEADER, 1)[1])}
        source_hash = hashlib.sha256(limits.encode("utf-8")).hexdigest()
        assert facts == {
            "minimum": {"key": "minimum", "value": "0", "path": "billing_limits.py", "call_id": "limits",
                        "source_sha256": source_hash},
            "maximum": {"key": "maximum", "value": "500", "path": "billing_limits.py", "call_id": "limits",
                        "source_sha256": source_hash},
        }
        if len(first_contexts) == 2:
            yield reply(model, read("invoice", "invoice.py"), reason="toolUse")
        elif len(first_contexts) == 3:
            source = expect_result(context, "invoice").content[0].text
            assert source == original
            minimum, maximum = facts["minimum"]["value"], facts["maximum"]["value"]
            fixed = source.replace("cents - discount", f"cents - max({minimum}, min(discount, {maximum}))")
            write = ToolCall(type="toolCall", id="fix", name="write", arguments={"path": "invoice.py", "content": fixed})
            yield reply(model, write, reason="toolUse")
        else:
            expect_result(context, "fix")
            yield reply(model, TextContent(type="text", text=f'Applied the bounds from {facts["minimum"]["path"]}.'))

    before = {path.name: path.read_bytes() for path in workspace.iterdir()}
    first = await run(
        PROMPT, workspace=workspace, trace_dir=tmp_path / "traces", model=model, stream_fn=repair_provider,
        state_dir=tmp_path / "repair", memory=MemoryScope(store, "billing"), context_policy=policy,
    )
    assert first.status == "ok" and first.text == "Applied the bounds from billing_limits.py."
    assert len(first_contexts) == 4
    after = {path.name: path.read_bytes() for path in workspace.iterdir()}
    assert after == {**before, "invoice.py": invoice.read_bytes()}
    assert invoice.read_text(encoding="utf-8") == original.replace(
        "cents - discount", "cents - max(0, min(discount, 500))",
    )
    assert examples() == expected
    events = [json.loads(line) for line in first.trace_path.read_text(encoding="utf-8").splitlines()]
    compacted, = [event for event in events if event["event_type"] == "context_compacted"]
    assert compacted["before_input_tokens_est"] + policy.reserve_tokens > policy.window_tokens
    assert compacted["input_tokens_est"] + policy.reserve_tokens <= policy.window_tokens
    assert sum(event["event_type"] == "context_reused" for event in events) == 2

    store.revoke("billing", revoked.ref.memory_id)
    now = deadline
    reopened = MemoryStore(database)
    assert [record.ref for record in reopened.recall("billing")] == [retained.ref]
    assert reopened.history("billing", expired.ref.memory_id)[0].expires_at == deadline.isoformat(timespec="microseconds")
    assert reopened.history("billing", revoked.ref.memory_id)[0].tombstone
    second_contexts = []

    async def audit_provider(model, context, options):
        second_contexts.append(context)
        payload = context.model_dump_json()
        assert RETAINED in context.system_prompt
        for record in (expired, revoked):
            assert record.content not in payload and record.ref.memory_id not in payload
        if len(second_contexts) == 1:
            assert context.messages[0].content == [TextContent(type="text", text=AUDIT)]
            assert len(context.messages) == 1
            yield reply(model, read("audit", "invoice.py"), reason="toolUse")
        else:
            assert expect_result(context, "audit").content[0].text == invoice.read_text(encoding="utf-8")
            yield reply(model, TextContent(type="text", text="Discount bounds are present."))

    second = await run(
        AUDIT, workspace=workspace, trace_dir=tmp_path / "traces", model=model, stream_fn=audit_provider,
        state_dir=tmp_path / "audit", memory=MemoryScope(reopened, "billing"), tool_policy=ToolPolicy(allow_writes=False),
    )
    assert second.status == "ok" and second.run_id != first.run_id
    assert len(second_contexts) == 2
    assert {path.name: path.read_bytes() for path in workspace.iterdir()} == after
