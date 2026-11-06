# M2 durable execution and recovery

Mini-Nanobot can reopen a run through the same Pi adapter used in M1. One logical run owns one state directory. A normal call without a state directory still creates an independent, ephemeral run.

```python
result = await run(
    prompt,
    workspace=workspace,
    trace_dir=trace_dir,
    model=model,
    stream_fn=provider,
    state_dir=state_dir,
)
```

Use a state directory outside the tool workspace. Call `run` again with the same state directory, workspace, prompt, and model configuration to recover. It returns the same `run_id` with a separate diagnostic trace for each attempt. An `uncertain` result includes the affected IDs in `uncertain_call_ids`. It blocks execution and leaves the workspace available for operator inspection.

## Tool transitions

| State | Journal evidence | Recovery behavior |
| --- | --- | --- |
| Planned | A checksummed assistant reply contains the tool call and its ID. | Replay the reply into Pi and run the schema and path checks. |
| Approved | The validated call passed path authorization. | Reauthorize the path before recording the start. |
| Started, read | The operation start is durable and the tool is read-only. | Retry the read and record the outcome. The new read can observe a changed workspace. |
| Started, write | The operation start is durable and there is no committed result. | Return `uncertain` before calling the provider or another tool. This also covers a write that raises after starting. |
| Succeeded | The paired tool result is durable. | Return the saved result to Pi without invoking the underlying tool. |
| Denied or failed read | A path denial or read failure is durable. | Return the saved error. Pi handles schema errors deterministically before invoking the tool wrapper. |

Pi's normal loop handles both live and replayed replies. The durable adapter saves each terminal model reply before giving it to Pi, so recovery retains the original tool-call IDs. It rejects duplicate IDs before delivering the reply. The tool wrapper commits the result before making it visible to the next model turn. The OS writer lock permits one active process per state directory.

## Journal and checkpoint

- `journal.jsonl` is authoritative. Records have a schema version, contiguous sequence, predecessor checksum, and SHA-256 checksum. A record is flushed and synced before the associated execution step advances.
- `checkpoint.json` is a derived snapshot installed with an atomic file replacement. To use it, recovery validates the journal's hash chain, locates the snapshot's prefix anchor, and checks the snapshot state against the digest in that authoritative finished record. Recovery then folds any subsequent journal records. A damaged snapshot can be rebuilt from the log. A missing or inconsistent journal prefix blocks recovery.
- A recorded terminal status distinguishes an explicit cancellation, a provider error, and a completed attempt. A process exit can leave an unfinished prefix. The tool state in that prefix governs the next attempt.
- The journal and checkpoint contain assistant replies, tool arguments, and results. Keep the state directory private. The separate JSONL trace contains diagnostic metadata.

These guarantees apply to process termination on a trusted local filesystem. A valid suffix removed from storage without a surviving checkpoint anchor is outside the detection guarantee. The protocol leaves interrupted writes unresolved until an operator inspects the workspace.

## Crash-window evidence

The fixtures run Pi in a subprocess and use `os._exit(86)` at each boundary. The parent test starts a fresh process against the same state directory.

| Termination point | Expected recovery |
| --- | --- |
| Before durable intent | There is no operation to replay. A fresh model turn may produce the call. |
| After durable approval, before execution | Pi replays the plan, then starts and performs the write once. |
| After the write, before the result record | The restarted process reports `uncertain`. A sentinel installed by the test remains intact. |
| After the result record, before checkpoint | Pi uses the saved tool result and continues the model turn. The sentinel remains intact. |

Other cases cover read retry, checkpoint repair, an older checkpoint with a journal suffix, corrupt and truncated journals, request mismatch, writer exclusion, and duplicate call IDs.

```powershell
uv run --locked pytest -q tests/e2e/test_durable_run.py
```

On Windows with CPython 3.13.13, all 14 M2 cases passed. The full offline suite had 618 passes and the same 18 upstream failures recorded in the [M1 audit](baseline-gap-audit.md).
