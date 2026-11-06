# M3 tool policy and bounded reads

Mini-Nanobot uses Pi's tool schemas and executor. The registered capabilities determine which tools can run concurrently. The runtime exposes workspace `read` and `write` tools.

| Capability | Rule |
| --- | --- |
| Read | May overlap with adjacent reads in the same model response, up to the configured limit. A timeout produces one paired error. |
| Workspace write | Runs alone. The previous read batch drains before the write starts, and the following batch waits for the write. A timeout or exception after the write starts stops further operations and returns `uncertain`. |
| Shell or external effects | Outside the registered capability set. Pi reports unregistered tools as unavailable. |

```python
from mini_nanobot import ToolPolicy, run

result = await run(
    prompt, workspace=workspace, trace_dir=trace_dir, model=model, stream_fn=provider,
    tool_policy=ToolPolicy(max_parallel_reads=4, timeout_s=30, allow_writes=False),
)
```

The default permits writes with four concurrent reads and a 30-second execution timeout. Set `max_parallel_reads=1` to serialize all tools. Other Pi callers retain serial execution by default.

Pi validates each call against the registered schema. Before new execution, the runtime checks the write setting and resolves the target against the workspace. It denies `.git`, `.env`, and `.env.*` path components, including case and trailing-dot or space variants, plus Windows device names and alternate data streams. These checks assume a trusted workspace and a stable filesystem.

Call IDs are unique within one logical run. Mini-Nanobot rejects a reply containing duplicates before any of its calls execute. In durable mode, the journal binds a run to the tool policy version. Legacy runs with successful reads require their original M2 runtime, because those journals lack the policy provenance needed to replay their contents safely. Legacy runs without successful reads can adopt the current policy. An unresolved write continues to return `uncertain` on restart.

Pi reports execution events in completion order and delivers tool results to the model in the original call order. Mini-Nanobot retains Pi's text pagination and caps aggregate inline success content at 64 KiB and error text at 4 KiB. Oversized binary attachments are omitted. The durable journal retains assistant tool arguments to support replay, so it can contain sensitive input.

Tool timeouts bound the await of the tool. A filesystem call already submitted to an OS thread may still settle after a timeout or cancellation. A write timeout stops the logical run and the M2 journal blocks automatic replay of that write. The guarantees apply per logical run on a trusted filesystem.

## Reproduce the read benchmark

```powershell
uv run --locked python benchmarks/tool_reads.py --pairs 8 --delay-ms 20 --csv benchmarks/results/m3-controlled.csv
uv run --locked python benchmarks/tool_reads.py --pairs 8 --delay-ms 0 --csv benchmarks/results/m3-native.csv
```

Each pair runs the same seeded 34-file workload with one and four readers. The order alternates by pair. Files are pre-read and each mode runs one warmup. The fixture checks every result's call ID, text, and ordering before writing a sample. The controlled run adds 20 ms before each actual file read. The native run uses warm local file I/O. Both use an offline provider and the ephemeral runtime to measure scheduling without journal syncs. The CSV files contain the read phase, total wall time, and the verified result digest for every sample.

Measured on Windows 11 with Python 3.13.13. Each cell lists p50 and p95 in milliseconds. The paired reduction is the median of the eight within-pair read-phase reductions.

| Workload | Serial read p50 / p95 | Parallel read p50 / p95 | Serial total p50 / p95 | Parallel total p50 / p95 | Paired read reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Controlled 20 ms delay](../benchmarks/results/m3-controlled.csv) | 1135.76 / 1138.52 | 371.10 / 374.91 | 1147.38 / 1154.70 | 380.27 / 387.42 | 67.3% |
| [Native warm-cache I/O](../benchmarks/results/m3-native.csv) | 86.00 / 89.93 | 75.90 / 99.52 | 90.80 / 95.00 | 80.61 / 105.73 | 11.7% |

The native run improved the paired median while increasing the read-phase p95. These eight pairs describe this fixture and machine.

All 14 M3 policy cases passed on this setup. The full offline suite had 632 passes and the same 18 upstream failures recorded in the [M1 audit](baseline-gap-audit.md).
