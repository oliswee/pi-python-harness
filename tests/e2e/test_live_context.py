import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, EventError, TextContent, ToolCall, Usage


@pytest.mark.parametrize(("case_name", "fixed"), [
    ("billing", "cents - max(0, min(discount, 500))"),
    ("retry", "min(45, base * attempt)"),
    ("inventory", "max(2, min(quantity, 24))"),
])
@pytest.mark.parametrize(("compression", "layout"), [
    ("full", "stable-first"), ("fixed", "stable-first"),
    ("selected", "stable-first"), ("selected", "dynamic-first"),
])
async def test_repair_comparison_runs_each_arm_and_verifies_preconditions(
    tmp_path, monkeypatch, compression, layout, case_name, fixed,
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    bench = importlib.import_module("live_context")
    case = bench.CASES[case_name]
    finished = []
    selection_calls = 0

    async def provider(model, context, options):
        nonlocal selection_calls
        reason = "stop"
        if not context.tools:
            selection_calls += 1
            payload = json.loads(context.messages[0].content)
            source = next(src for src in payload["sources"] if src["path"] == "policy.py")
            proposals = [
                {"call_id": source["call_id"], "path": "policy.py", "line_start": line, "line_end": line,
                 "symbol": quote.split()[0], "quote": quote}
                for line, quote in source["lines"]
            ]
            content = TextContent(type="text", text=json.dumps(proposals))
        else:
            step = len(finished)
            finished.append(context.system_prompt)
            if layout == "stable-first":
                assert context.system_prompt.startswith(bench.SYSTEM_PROMPT)
            else:
                assert context.system_prompt.startswith("Audit sequence")
            if step == 0:
                content = [
                    ToolCall(type="toolCall", id=f"read-{index}", name="read", arguments={"path": path})
                    for index, path in enumerate(("policy.py", "history.txt", "logic.py"))
                ]
                reason = "toolUse"
            elif step == 1:
                if compression != "full":
                    assert '"path":"history.txt"' in context.system_prompt
                    assert '"path":"logic.py"' in context.system_prompt and case.original in context.system_prompt
                    assert "Diagnostic 045" not in context.system_prompt
                content = ToolCall(type="toolCall", id="fix", name="write", arguments={
                    "path": "logic.py", "content": f"def {case.signature}:\n    return {fixed}\n",
                })
                reason = "toolUse"
            else:
                if compression != "full":
                    assert f"    return {case.original}\\n" not in context.system_prompt
                content = TextContent(type="text", text="Capped delay using policy.py.")
        usage = Usage(input=9, output=3, cache_read=12, total_tokens=24)
        message = AssistantMessage(
            content=content if isinstance(content, list) else [content], api=model.api, provider=model.provider, model=model.id,
            timestamp=0, stop_reason=reason, usage=usage,
        )
        yield EventDone(type="done", reason=reason, message=message)

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    result = await bench.run_case(case, model, provider, tmp_path / "trial", compression, layout)
    assert result["repair_verified"] and result["status"] == "ok"
    assert result["compactions"] == int(compression != "full")
    assert len(finished) == 3 and selection_calls == int(compression == "selected")
    assert result["cache_hit_ratio"] == pytest.approx(12 / 21)
    assert result["total_tokens"] == (3 + selection_calls) * 24
    assert not bench.verify_source("def retry_delay(base, attempt):\n    return __import__('os')\n", case)


async def test_comparison_reports_http_status_and_stops_on_first_credential_failure(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    bench = importlib.import_module("live_context")
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    calls = 0

    async def rejected(model, context, options):
        nonlocal calls
        calls += 1
        message = AssistantMessage(
            content=[], api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason="error",
            error_message="Error code: 401 - authentication denied for mock-secret",
        )
        yield EventError(type="error", reason="error", error=message)

    args = SimpleNamespace(output=tmp_path / "comparison", trials=1, case=None)
    assert await bench.compare(args, model, rejected) == 1
    assert calls == 1
    raw = (args.output / "report.json").read_text(encoding="utf-8")
    assert "mock-secret" not in raw
    report = json.loads(raw)
    assert report["stopped_on_provider_http_status"] == 401
    assert report["results"][0]["calls"][0]["provider_http_status"] == 401
