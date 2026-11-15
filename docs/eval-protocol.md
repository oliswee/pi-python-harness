# M6 evaluation protocol

The case manifest names the deterministic verifier and pytest selectors for the original 12 cases and the later B13 repair case. Every trial uses a fresh subprocess. Pytest allocates an independent temporary workspace for each verifier. The fixtures create their inputs locally and use scripted providers. [Current case manifest](../benchmarks/cases.json)

```powershell
uv sync --locked --all-packages --python 3.13
uv run --locked python benchmarks/evaluate.py --trials 3 --output evaluation-m6
uv run --locked python benchmarks/ablation.py --pairs 3 --output ablations-m6
```

Use fresh output paths. Keep the run output private unless you know the fixture data is synthetic. Each evaluation run saves the source revision, dirty flag, manifest and lock hashes, platform, and case outcomes. Every trial includes a pytest log, phase timings, captured scripted assistant replies, and available journal replies and diagnostic traces. Phase details identify failed or skipped verifiers. The check requires at least ten passing case IDs across all three trials, and reports unsupported fixtures separately.

Case success means every registered verifier passed. Tool pairing counts come from attempt trace events. A replayed event pair can describe returning a cached result. The filesystem and journal assertions are the evidence for actual side effects. For crash windows, each fixture deliberately terminates a child and then verifies a fresh process against the same state. The trace can be absent for the killed attempt.

The frozen Windows run passed **12/12 cases in each of three trials**, with 102 tool start events, 102 result events, and zero unpaired events in the captured traces. The separate ablation run verified all 24 observations across three pairs for each of four controls. The revision of each run is recorded in its report. [Case report](../evidence/m6/report.json) · [Paired controls](../evidence/m6/ablations.json) · [Trial artifact archive](../evidence/m6/trial-artifacts.zip)

## Paired controls

| Feature | On | Off | Verifier |
| --- | --- | --- | --- |
| Write authorization | Writes disabled | Writes permitted | The same attempted write leaves the protected fixture untouched. Other path controls stay enabled. |
| Durability | Reopen the same state directory | Run again without a journal | An operator sentinel survives a repeated invocation only in durable mode. Crash recovery is exercised separately in B03 and B04. |
| Memory | Inject the current scoped SQLite records | Run without memory | The retained canary appears and the revoked canary stays absent. |
| Compaction | Enforce budget with verified read facts | Send the full view to the same scripted window | The request fits and both exact values are available. |
| Scheduling | Four readers | One reader | Compare paired latency while verifying all 34 ordered outputs. [M3 timing protocol](tool-policy.md) |

The M6 ablation script alternates mode order and isolates each feature and mode in a new fixture directory. Each ON and OFF observation must demonstrate its own expected behavior. These scripted trials measure the specified behaviors. Live model success rates, provider charges and statistical confidence intervals are outside the scope of these small deterministic samples. M5 and the compaction ablation report request estimates separately from provider-reported usage.

## Long-context repair and memory expiry

B13 reads oversized billing source and historical notes, then uses the compacted view to repair an invoice discount function. The verifier checks the original user constraints and the exact source path, tool call ID, and SHA-256 in the provider context. It executes five cases against the Python function before and after the repair and checks that only the intended file changed. An independent read-only run reopens the SQLite database at the expiry boundary and checks both provider requests for retained, expired, and revoked rules.

```powershell
uv run --locked python benchmarks/evaluate.py --case B13 --trials 3 --output evaluation-context-repair
```

B13 uses a scripted provider. Its trace reports request estimates using the default UTF-8 byte estimator. [Repair verifier](../tests/e2e/test_long_context_repair.py)

## Optional real-provider probe

With an authorized credential in `DEEPSEEK_API_KEY`, run the following. The script defaults to `deepseek-flash` with thinking disabled. Use `--model` to select another accessible DeepSeek model. The eight probes use billable API calls. [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)

```powershell
uv run --locked python benchmarks/live_revocation.py --provider deepseek --output live-m4-deepseek
```

For Gemini, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` and use an accessible Gemini model ID from the pinned registry.

```powershell
uv run --locked python benchmarks/live_revocation.py --model <registered-model-id> --output live-m4
```

The probe checks retained and revoked synthetic canaries in each actual outbound context, disables provider tools, and records response flags and provider usage. The evidence addresses the request assembly boundary. It does not establish provider-side deletion.

## DeepSeek compaction and cache comparison

`benchmarks/live_context.py` compares full context, fixed fact rules, and model-selected source excerpts with a stable system prompt first. It adds a fourth condition that moves the same changing metadata to the start of the system message. Each condition uses a fresh local workspace and the same task, model, and response reserve. It rotates the order across cases. Server cache state is observable through returned usage but is not controlled by the runner.

```powershell
uv run --locked python benchmarks/live_context.py --trials 1 --output live-context
```

Set `DEEPSEEK_API_KEY` in the process environment. The suite runs billing, retry, and inventory fixtures. Use `--case retry` for a smaller run. One trial supports up to eight model calls for each condition and case. The verifier checks file scope, the source citation, and three arithmetic cases with a restricted Python expression. It records total provider token usage including selector requests, cache hit ratio for agent requests with cache telemetry, time to first streamed text or tool output, wall time, and repair outcome. [DeepSeek cache fields](https://api-docs.deepseek.com/api/create-chat-completion/)

The first live trial at `f95c0fd` verified 3 of 12 repairs, all with full context. Five conditions repeatedly read the same files after compaction, which had discarded the read-progress record for those batches. The other four conditions reached the context limit before sending a compacted request. DeepSeek reported zero cache hits across the measured agent requests. [Initial live report](../evidence/m8/live-context-initial-report.json)
