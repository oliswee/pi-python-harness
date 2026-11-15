# M5 request budget and verified compaction

Enable this policy per run. A durable run binds the window, response reserve, fact rules, tokenizer ID, and whether summary verification is enabled.

```python
from mini_nanobot import ContextPolicy, FactRule, run

result = await run(
    prompt, workspace=workspace, trace_dir=trace_dir, model=model, stream_fn=provider,
    state_dir=state_dir,
    context_policy=ContextPolicy(
        window_tokens=7000,
        reserve_tokens=512,
        fact_rules=(
            FactRule("port", "north.txt", "Port:"),
            FactRule("region", "south.txt", "Region:"),
        ),
    ),
)
```

## Budget boundary

The estimate includes the model identifier, system prompt with recalled memory and any active snapshot, every retained message and tool result, and the registered tool schemas. The runtime adds a framing allowance and reserves the configured response tokens. It also passes that response limit to the provider. The effective window is capped by the model's declared context window.

The default counts UTF-8 bytes of a normalized JSON request plus the framing allowance. This conservative text approximation can overcount and differs from provider accounting for protocol framing and images. For provider-specific estimation, supply a `token_counter` and a stable `tokenizer_id` identifying its behavior. The budget is an estimate at the Pi provider boundary.

## Verified context view

Compaction replaces only a contiguous prefix of complete read-only tool turns in the provider's view. The original user message and every write or mixed turn remain in the request. The underlying Pi transcript and tool journal remain the source of recovery.

Each fact rule selects one nonempty value by path and line prefix from a successful read result. The snapshot carries the exact value, path, call ID, and SHA-256 digest of the source text. Missing, ambiguous, oversized, or newer conflicting evidence blocks compaction. In the trusted workspace, recorded file identities detect subsequent writes through aliases such as hardlinks. A successful write with unknown identity invalidates prior read evidence conservatively. The original prompt retains the caller's operation constraints.

With no summarizer, the verified facts form an extractive summary. An optional `summarizer` stream receives the fact array in its own budgeted request without tools. It must return JSON with every exact value and citation. Its response can reorder entries. The runtime admits only the verified source-backed fields into the compacted view.

### Model-selected source excerpts

Set `ContextPolicy(evidence_selector=provider)` in place of `fact_rules` to admit source excerpts selected by a model. The selector receives the original task and a bounded preview of the latest successful text reads in the compactable prefix. It can request one more line range from an offered tool result. That lookup uses the captured tool output and adds no filesystem operation. Selectors and optional summary verification use tool-free requests subject to the same input and output budget.

The runtime requires the exact quote, file path, call ID, inclusive line range, and a symbol present in the quote before computing the excerpt SHA-256. It checks for newer conflicting read or write evidence before activation and each reuse. Invalid proposals or evidence that exceeds the budget leave compaction uncommitted. Configure a timeout on a live provider, and use `read` with an explicit offset and limit when a source exceeds the file tool's output limit.

## Durability and invalidation

A durable run syncs a staged view, then syncs a commit that names the exact staged checksum. The journal and the existing derived checkpoint retain the active version. Restart after staging rebuilds the candidate, and restart after commit reuses the active view once its source prefix and current evidence validate. A failed candidate preserves the prior active version and returns `context_limit` if the request still exceeds the budget.

With M4 memory enabled, context preparation and compaction activation check the SQLite authority. A revision revoked during summary generation blocks activation and returns `memory_changed`. Diagnostic trace rows record the input estimate, response reserve, and pre-compaction estimate without including fact content.

## Reproduce

```powershell
uv run --locked pytest -q tests/e2e/test_context_budget.py
```

The fixture reads two long files, compacts them, and verifies both exact values, their source paths, the unchanged user instruction, and a new read's exact result. Separate subprocess fixtures terminate the process after staging and after commit. Recovery checks that the original reads remain paired and are reused from the journal.

In the seeded fixture, the estimated request plus response reserve fell from **13,738 to 2,268** after compaction, an **83.5%** reduction for that request. The optional summary was a separate request with an estimate plus reserve of 1,101. These figures use the default byte estimator. [Captured request estimates](../evidence/m5/context-budget.jsonl)

All 10 M5 cases passed. A first full run encountered a Windows directory rename error in an upstream migration test. The migration suite passed when rerun, and the full rerun in a fresh temporary directory had 650 passes and the same 18 upstream failures recorded in the [M1 audit](baseline-gap-audit.md).
