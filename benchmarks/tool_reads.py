"""Paired serial/parallel benchmark using the normal Pi runtime entry point."""

import argparse
import asyncio
import csv
import hashlib
import platform
import random
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall, ToolResultMessage
from pi_coding_agent.core.tools.read import DefaultReadOperations

from mini_nanobot import ToolPolicy, run

DOC_COUNT = 34


async def measure(workspace, trace_dir, model, expected, concurrency):
    read_started = read_elapsed = 0.0
    verified_digest = ""

    async def provider(model, context, options=None):
        nonlocal read_started, read_elapsed, verified_digest
        if context.messages[-1].role == "user":
            content = [
                ToolCall(type="toolCall", id=name, name="read", arguments={"path": name})
                for name in expected
            ]
            reason = "toolUse"
            read_started = time.perf_counter()
        else:
            read_elapsed = time.perf_counter() - read_started
            results = [message for message in context.messages if isinstance(message, ToolResultMessage)]
            actual = [(message.tool_call_id, message.content[0].text) for message in results if not message.is_error]
            if actual != list(expected.items()):
                raise ValueError("Incorrect read contents, order, or call IDs")
            verified_digest = hashlib.sha256(repr(actual).encode()).hexdigest()
            content = [TextContent(type="text", text=verified_digest)]
            reason = "stop"
        reply = AssistantMessage(
            content=content, api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason=reason,
        )
        yield EventDone(type="done", reason=reason, message=reply)

    started = time.perf_counter()
    result = await run(
        "Read the seeded documents in order", workspace=workspace, trace_dir=trace_dir,
        model=model, stream_fn=provider, tool_policy=ToolPolicy(max_parallel_reads=concurrency),
    )
    elapsed = time.perf_counter() - started
    if result.status != "ok" or not verified_digest or result.text != verified_digest:
        raise RuntimeError("Read benchmark verification failed")
    return {
        "read_ms": round(read_elapsed * 1000, 3), "total_ms": round(elapsed * 1000, 3),
        "verified_docs": DOC_COUNT, "result_sha256": verified_digest,
    }


async def benchmark(args):
    rng = random.Random(20261106)
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    original = DefaultReadOperations.read_text_file

    async def delayed_read(self, path):
        await asyncio.sleep(args.delay_ms / 1000)
        return await original(self, path)

    rows = []
    with tempfile.TemporaryDirectory(prefix="mini-nanobot-bench-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        workspace.mkdir()
        expected = {
            f"doc-{index:02}.txt": f"Document {index}\n" + "\n".join(
                f"Record {line:03}: {rng.getrandbits(96):024x}" for line in range(128)
            )
            for index in range(DOC_COUNT)
        }
        for name, text in expected.items():
            (workspace / name).write_text(text, encoding="utf-8")

        for name in expected:
            (workspace / name).read_bytes()

        read_fn = delayed_read if args.delay_ms else original
        with patch.object(DefaultReadOperations, "read_text_file", read_fn):
            # One warmup per mode. Every measured pair sees the same warm fixture.
            for concurrency in (1, args.parallelism):
                await measure(workspace, root / "traces", model, expected, concurrency)
            for pair in range(1, args.pairs + 1):
                modes = [("serial", 1), ("parallel", args.parallelism)]
                if pair % 2 == 0:
                    modes.reverse()
                for order, (mode, concurrency) in enumerate(modes, start=1):
                    metrics = await measure(workspace, root / "traces", model, expected, concurrency)
                    rows.append({
                        "pair": pair, "order": order, "mode": mode, "concurrency": concurrency,
                        "delay_ms": args.delay_ms, "cache": "warm", **metrics,
                    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--delay-ms", type=float, default=20, help="Synthetic read latency; 0 uses native I/O.")
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 2 or not 2 <= args.parallelism <= 64 or not 0 <= args.delay_ms < 1000:
        parser.error("use at least two pairs, parallelism 2–64, and a delay between 0 and 1000ms")

    rows = asyncio.run(benchmark(args))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{platform.platform()} / Python {platform.python_version()} / {DOC_COUNT} verified docs")
    for mode in ("serial", "parallel"):
        print(mode, end="")
        for phase in ("read_ms", "total_ms"):
            values = [row[phase] for row in rows if row["mode"] == mode]
            p95 = statistics.quantiles(values, n=20, method="inclusive")[-1]
            print(f" {phase} p50={statistics.median(values):.2f}, p95={p95:.2f}", end="")
        print()
    phase_pairs = zip(
        (row["read_ms"] for row in rows if row["mode"] == "serial"),
        (row["read_ms"] for row in rows if row["mode"] == "parallel"),
        strict=True,
    )
    savings = [100 * (serial - parallel) / serial for serial, parallel in phase_pairs]
    print(f"Median paired read-phase reduction: {statistics.median(savings):.1f}%")


if __name__ == "__main__":
    main()
