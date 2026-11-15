import json

import pytest

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall

from mini_nanobot import ContextPolicy, run


def reply(model, content, reason="stop"):
    message = AssistantMessage(
        content=[content], api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason=reason,
    )
    return EventDone(type="done", reason=reason, message=message)


@pytest.mark.parametrize("forged", [False, True])
async def test_selector_verifies_reread_and_persisted_source(tmp_path, forged):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "billing.py").write_text(
        "# notes\n" * 210 + "MAX_CENTS = 500\n" + "# more notes\n" * 340, encoding="utf-8",
    )
    invoice = workspace / "invoice.py"
    invoice.write_text("def discount(cents):\n    return cents\n", encoding="utf-8")
    prompt = "Fix invoice.py to honor the billing limit. Keep the public API unchanged."
    quote = "MAX_CENTS = 500"
    selection = {
        "call_id": "source", "path": "billing.py", "line_start": 211, "line_end": 211,
        "symbol": "MAX_CENTS", "quote": quote,
    }
    calls, selections = [], []

    async def select(model, ctx, options):
        assert not ctx.tools
        payload = json.loads(ctx.messages[0].content)
        assert payload["task"] == prompt
        selections.append(payload)
        if len(selections) == 1:
            result = {"read_more": {"call_id": "source", "line_start": 211, "line_end": 211}}
        else:
            assert payload["sources"][0]["lines"] == [[211, quote]]
            result = [{**selection, "quote": "MAX_CENTS = 999" if forged else quote}]
        payload = json.dumps(result)
        yield reply(model, TextContent(type="text", text=f"```json\n{payload}\n```"))

    async def provider(model, ctx, options):
        calls.append(ctx)
        assert ctx.messages[0].content[0].text == prompt
        if len(calls) == 1:
            call = ToolCall(type="toolCall", id="source", name="read", arguments={"path": "billing.py"})
            yield reply(model, call, "toolUse")
        elif len(calls) == 2:
            [fact] = json.loads(ctx.system_prompt.splitlines()[-1])
            assert fact["quote"] == quote and fact["path"] == "billing.py"
            assert '"source_sha256":' in ctx.system_prompt
            assert len(ctx.messages) == 1
            maximum = int(fact["quote"].split("=")[1])
            call = ToolCall(type="toolCall", id="fix", name="write", arguments={
                "path": "invoice.py", "content": f"def discount(cents):\n    return min(cents, {maximum})\n",
            })
            yield reply(model, call, "toolUse")
        else:
            assert quote in ctx.system_prompt and ctx.messages[-1].tool_call_id == "fix"
            yield reply(model, TextContent(type="text", text="billing.py:211"))

    result = await run(
        prompt, workspace=workspace, trace_dir=tmp_path / "traces",
        model=get_model("anthropic", "claude-3-5-sonnet-20241022"), stream_fn=provider,
        state_dir=tmp_path / "state",
        context_policy=ContextPolicy(window_tokens=5000, reserve_tokens=512, evidence_selector=select),
    )
    journal = (tmp_path / "state" / "journal.jsonl").read_text(encoding="utf-8")
    assert len(selections) == 2
    if forged:
        assert result.status == "context_limit" and len(calls) == 1
        assert "compaction_committed" not in journal
        assert invoice.read_text(encoding="utf-8") == "def discount(cents):\n    return cents\n"
    else:
        assert result.status == "ok" and result.text == "billing.py:211" and len(calls) == 3
        assert invoice.read_text(encoding="utf-8") == "def discount(cents):\n    return min(cents, 500)\n"
        assert '"event_type": "context_reused"' in result.trace_path.read_text(encoding="utf-8")
        assert "compaction_committed" in journal
