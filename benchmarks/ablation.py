"""Compare four independently controlled runtime features with scripted providers."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, EventError, TextContent, ToolCall, ToolResultMessage

from mini_nanobot import ContextPolicy, FactRule, MemoryScope, MemoryStore, ToolPolicy, run

from evaluate import git, write_json

MODEL = get_model("anthropic", "claude-3-5-sonnet-20241022")


def reply(model, *parts, reason="stop"):
    message = AssistantMessage(
        content=list(parts), api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason=reason,
    )
    return EventDone(type="done", reason=reason, message=message)


def call(call_id, name, **arguments):
    return ToolCall(type="toolCall", id=call_id, name=name, arguments=arguments)


async def execute(directory, provider, **options):
    workspace = directory / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return await run(
        "Run the seeded evaluation task.", workspace=workspace, trace_dir=directory / "traces",
        model=MODEL, stream_fn=provider, **options,
    )


async def authorization(directory, enabled):
    async def provider(model, context, options=None):
        last = context.messages[-1]
        if last.role == "user":
            yield reply(model, call("w", "write", path="output.txt", content="changed"), reason="toolUse")
        else:
            yield reply(model, TextContent(type="text", text="denied" if last.is_error else "wrote"))

    result = await execute(directory, provider, tool_policy=ToolPolicy(allow_writes=not enabled))
    target = directory / "workspace" / "output.txt"
    effect = target.exists()
    denied = result.status == "ok" and result.text == "denied" and not effect
    permitted = (
        result.status == "ok" and result.text == "wrote" and effect and target.read_text(encoding="utf-8") == "changed"
    )
    return {"verified": denied if enabled else permitted, "task_success": denied, "invalid_writes": int(effect)}


async def durability(directory, enabled):
    calls = 0

    async def provider(model, context, options=None):
        nonlocal calls
        calls += 1
        if context.messages[-1].role == "user":
            yield reply(model, call("w", "write", path="output.txt", content="original effect"), reason="toolUse")
        else:
            yield reply(model, TextContent(type="text", text="complete"))

    state = directory / "state" if enabled else None
    first = await execute(directory, provider, state_dir=state)
    output = directory / "workspace" / "output.txt"
    if first.status != "ok" or output.read_text(encoding="utf-8") != "original effect":
        raise RuntimeError("Initial write fixture failed")
    output.write_text("operator sentinel", encoding="utf-8")
    second = await execute(directory, provider, state_dir=state)
    final_text = output.read_text(encoding="utf-8")
    preserved = second.status == "ok" and second.text == "complete" and final_text == "operator sentinel" and calls == 2
    repeated = second.status == "ok" and second.text == "complete" and final_text == "original effect" and calls == 4
    return {
        "verified": preserved if enabled else repeated, "task_success": preserved,
        "duplicate_writes": int(final_text != "operator sentinel"), "provider_calls": calls,
    }


async def memory(directory, enabled):
    store = MemoryStore(directory / "memory.sqlite")
    old = store.create("fixture", kind="fact", content="revoked-002", provenance="seed")
    store.create("fixture", kind="fact", content="retained-001", provenance="seed")
    store.revoke("fixture", old.ref.memory_id)
    retained = revoked = False

    async def provider(model, context, options=None):
        nonlocal retained, revoked
        system = context.system_prompt or ""
        retained, revoked = "retained-001" in system, "revoked-002" in system
        yield reply(model, TextContent(type="text", text="observed"))

    result = await execute(directory, provider, memory=MemoryScope(store, "fixture") if enabled else None)
    success = result.status == "ok" and result.text == "observed"
    recalled = success and retained and not revoked
    empty = success and not retained and not revoked
    return {
        "verified": recalled if enabled else empty, "task_success": recalled,
        "retained": retained, "revoked_injections": int(revoked),
    }


async def compaction(directory, enabled):
    workspace = directory / "workspace"
    workspace.mkdir(parents=True)
    for filename, value in (("a.txt", "101"), ("b.txt", "202")):
        (workspace / filename).write_text(f"Value: {value}\n" + "filler line\n" * 450, encoding="utf-8")
    observed = 0
    rejected = False
    ceiling = 7000
    reserve = 512

    async def provider(model, context, options=None):
        nonlocal observed, rejected
        if context.messages[-1].role == "user" and not context.system_prompt:
            yield reply(model, call("a", "read", path="a.txt"), call("b", "read", path="b.txt"), reason="toolUse")
            return
        messages = [item.model_dump(mode="json", exclude={"timestamp", "usage", "details"}) for item in context.messages]
        schemas = [item.model_dump(mode="json", include={"name", "description", "parameters"}) for item in context.tools]
        payload = {"model": model.id, "system": context.system_prompt, "messages": messages, "tools": schemas}
        observed = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        observed += reserve + 32 + 8 * (len(messages) + len(schemas))
        if observed > ceiling:
            rejected = True
            error = AssistantMessage(
                content=[], api=model.api, provider=model.provider, model=model.id,
                timestamp=0, stop_reason="error", error_message="Fixture context limit",
            )
            yield EventError(type="error", reason="error", error=error)
        elif "101" in context.system_prompt and "202" in context.system_prompt:
            yield reply(model, TextContent(type="text", text="101/202"))

    policy = ContextPolicy(
        window_tokens=ceiling, reserve_tokens=reserve,
        fact_rules=(FactRule("a", "a.txt", "Value:"), FactRule("b", "b.txt", "Value:")),
    )
    result = await execute(directory, provider, context_policy=policy if enabled else None)
    fitted = result.status == "ok" and result.text == "101/202" and 0 < observed <= ceiling
    overflowed = result.status == "error" and rejected and observed > ceiling
    return {
        "verified": fitted if enabled else overflowed, "task_success": fitted,
        "request_with_reserve_est": observed, "window_est": ceiling, "runtime_status": result.status,
    }


async def evaluate(output: Path, pairs: int) -> int:
    revision, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for pair in range(1, pairs + 1):
        for case in (authorization, durability, memory, compaction):
            modes = (True, False) if pair % 2 else (False, True)
            for order, enabled in enumerate(modes, 1):
                mode = "on" if enabled else "off"
                fixture = output / f"pair-{pair:02}" / f"{case.__name__}-{mode}"
                started = time.perf_counter()
                result = await case(fixture, enabled)
                rows.append({
                    "pair": pair, "order": order, "feature": case.__name__, "mode": mode,
                    "duration_s": round(time.perf_counter() - started, 6), **result,
                })
    write_json(output / "ablations.json", {
        "revision": revision, "dirty": dirty, "pairs": pairs, "python": platform.python_version(),
        "platform": platform.platform(), "provider": "deterministic fixture", "results": rows,
    })
    passed = all(row["verified"] for row in rows)
    print(f"{len(rows)} paired observations. Expected differences {'verified' if passed else 'failed'}.")
    return int(not passed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    return asyncio.run(evaluate(args.output.resolve(), args.pairs))


if __name__ == "__main__":
    raise SystemExit(main())
