"""Terminate a durable run immediately after a compaction transition."""

import asyncio
import json
import os
import sys
from pathlib import Path

from pi_ai import get_model
from pi_ai.types import TextContent

from mini_nanobot import run
from mini_nanobot.journal import RunJournal
from test_context_budget import PROMPT, context_policy, read, reply


async def main():
    directory, fault = sys.argv[1:]
    directory = Path(directory)
    original_append = RunJournal.append

    def injected(self, kind, data):
        original_append(self, kind, data)
        if fault and kind == f"compaction_{fault}":
            os._exit(86)

    RunJournal.append = injected
    calls = summaries = 0

    async def provider(model, context, options=None):
        nonlocal calls
        calls += 1
        if not context.system_prompt:
            yield reply(model, read("north", "north.txt"), read("south", "south.txt"), reason="toolUse")
        else:
            assert "8181" in context.system_prompt and "eu-test" in context.system_prompt
            assert len(context.messages) == 4
            assert [message.tool_call_id for message in context.messages[2:]] == ["north", "south"]
            yield reply(model, TextContent(type="text", text="8181 / eu-test"))

    async def summarize(model, context, options=None):
        nonlocal summaries
        summaries += 1
        yield reply(model, TextContent(type="text", text=context.messages[0].content))

    result = await run(
        PROMPT, workspace=directory / "workspace", trace_dir=directory / "traces",
        model=get_model("anthropic", "claude-3-5-sonnet-20241022"), stream_fn=provider,
        state_dir=directory / "state", context_policy=context_policy(summarize),
    )
    print(json.dumps({"status": result.status, "text": result.text, "provider_calls": calls, "summaries": summaries}))


if __name__ == "__main__":
    asyncio.run(main())
