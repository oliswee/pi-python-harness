"""Subprocess fixture that terminates at durable execution boundaries."""

import asyncio
import json
import os
import sys
from pathlib import Path

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall

from mini_nanobot import run


def main():
    from mini_nanobot.journal import RunJournal

    root, fault, tool, prompt = sys.argv[1:]
    root = Path(root)
    append = RunJournal.append
    checkpoint = RunJournal.save_checkpoint

    def injected_append(self, kind, data):
        if (fault, kind) in {
            ("before_intent", "reply"), ("before_execution", "started"), ("after_effect", "succeeded"),
        }:
            os._exit(86)
        append(self, kind, data)
        if fault == "after_result" and kind == "succeeded":
            os._exit(86)

    def injected_checkpoint(self):
        if fault == "before_checkpoint":
            os._exit(86)
        return checkpoint(self)

    RunJournal.append = injected_append
    RunJournal.save_checkpoint = injected_checkpoint
    calls = 0

    async def provider(model, context, options=None):
        nonlocal calls
        calls += 1
        previous = context.messages[-1]
        if previous.role == "user":
            arguments = {"path": "effect.txt"}
            if tool == "write":
                arguments["content"] = "effect from tool"
            content = [ToolCall(type="toolCall", id="operation-1", name=tool, arguments=arguments)]
            if fault == "duplicate_id":
                content.append(content[0])
            reason = "toolUse"
        else:
            assert previous.role == "toolResult" and previous.tool_call_id == "operation-1"
            assert not previous.is_error
            if fault == "provider_error":
                raise RuntimeError("fixture provider interrupted")
            text = previous.content[0].text if tool == "read" else "Complete"
            content = [TextContent(type="text", text=text)]
            reason = "stop"
        message = AssistantMessage(
            content=content, api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason=reason,
        )
        yield EventDone(type="done", reason=reason, message=message)

    result = asyncio.run(run(
        prompt, workspace=root / "workspace", trace_dir=root / "traces",
        model=get_model("anthropic", "claude-3-5-sonnet-20241022"), stream_fn=provider, state_dir=root / "state",
    ))
    print(json.dumps({
        "status": result.status, "text": result.text, "run_id": result.run_id,
        "provider_calls": calls, "uncertain_call_ids": result.uncertain_call_ids, "trace_path": str(result.trace_path),
    }))


if __name__ == "__main__":
    main()
