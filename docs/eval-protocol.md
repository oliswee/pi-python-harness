# M6 evaluation protocol

The [case manifest](../benchmarks/cases.json) names the deterministic verifier and pytest selectors for 12 cases across eight capabilities. Every trial uses a fresh subprocess. Pytest allocates an independent temporary workspace for each verifier. The fixtures create their inputs locally and use scripted providers with fixed outputs.

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

## Optional real-provider probe

With an authorized Gemini credential in `GEMINI_API_KEY` or `GOOGLE_API_KEY`, choose an accessible Gemini model ID from the pinned registry and run the following. The probe can send up to eight paid requests.

```powershell
uv run --locked python benchmarks/live_revocation.py --model <registered-model-id> --output live-m4
```

The probe checks retained and revoked synthetic canaries in each actual outbound context, disables provider tools, and records response flags and provider usage. The evidence addresses the request assembly boundary. It does not establish provider-side deletion.
