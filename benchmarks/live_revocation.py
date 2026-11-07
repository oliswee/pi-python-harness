"""Opt-in Gemini verification of eight post-revocation provider requests."""

import argparse
import asyncio
import json
import os
from contextlib import aclosing
from pathlib import Path

from pi_ai import get_model, stream_simple
from pi_ai.types import EventDone

from mini_nanobot import MemoryScope, MemoryStore, ToolPolicy, run

from evaluate import git, write_json

RETAINED = "retained-canary-230"
REVOKED = "revoked-canary-529"


async def verify(output, model, key):
    revision = git("rev-parse", "HEAD")
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    store = MemoryStore(output / "memory.sqlite")
    removed = store.create("probe", kind="fact", content=REVOKED, provenance="synthetic probe")
    store.create("probe", kind="fact", content=RETAINED, provenance="synthetic probe")
    store.revoke("probe", removed.ref.memory_id)
    rows = []

    for index in range(1, 9):
        record = {"probe": index, "dispatched": False, "retained_in_request": False, "revoked_in_request": False}

        async def provider(model, context, options):
            outbound = context.model_copy(update={"tools": []})
            payload = outbound.model_dump_json()
            record["retained_in_request"] = RETAINED in payload
            record["revoked_in_request"] = REVOKED in payload
            if not record["retained_in_request"] or record["revoked_in_request"]:
                raise RuntimeError("Context verification failed before dispatch")
            settings = options.model_copy(update={"api_key": key, "max_tokens": 96})
            record["dispatched"] = True
            async with aclosing(stream_simple(model, outbound, settings)) as events:
                async for event in events:
                    if isinstance(event, EventDone):
                        record["usage"] = event.message.usage.model_dump(mode="json")
                    yield event

        try:
            result = await asyncio.wait_for(run(
                "Return the retained canary from the supplied memory as a single line.",
                workspace=workspace, trace_dir=output / "traces", model=model, stream_fn=provider,
                memory=MemoryScope(store, "probe"), tool_policy=ToolPolicy(allow_writes=False),
            ), timeout=45)
            record.update(
                status=result.status, run_id=result.run_id,
                retained_in_answer=RETAINED in result.text, revoked_in_answer=REVOKED in result.text,
            )
        except TimeoutError:
            record["status"] = "timeout"
        rows.append(record)
        with (output / "requests.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if record["status"] != "ok":
            break

    passed = len(rows) == 8 and all(
        row["status"] == "ok" and row["dispatched"] and row["retained_in_request"] and not row["revoked_in_request"]
        for row in rows
    )
    write_json(output / "report.json", {
        "revision": revision, "provider": "google", "model": model.id, "passed": passed, "requests": rows,
    })
    print(f"{len(rows)}/8 probes recorded. Context verification {'passed' if passed else 'incomplete'}.")
    return int(not passed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="A Gemini model ID present in the pinned Pi registry.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not key.strip():
        parser.error("Set an authorized Gemini API key in GEMINI_API_KEY or GOOGLE_API_KEY before running live probes")
    model = get_model("google", args.model)
    if model is None:
        parser.error("The Gemini model ID is absent from the pinned Pi registry")
    return asyncio.run(verify(args.output.resolve(), model, key))


if __name__ == "__main__":
    raise SystemExit(main())
