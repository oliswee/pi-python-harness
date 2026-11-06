# Mini-Nanobot M1 baseline audit

Mini-Nanobot uses the Python Pi agent loop as its execution engine. The project's engineering work belongs at the boundaries where model output becomes a tool side effect or where persisted state is assembled back into context. Those boundaries support measurable recovery, tool governance, memory revocation and compaction correctness.

## Baseline selection

| Candidate | Evidence | Decision |
| --- | --- | --- |
| [Fork at `5e41e26552d2e8ed29b7934f126bae0ee8031761`](https://github.com/oliswee/pi-python-harness/commit/5e41e26552d2e8ed29b7934f126bae0ee8031761) | Locked install and CLI help work on Windows. All 19 upstream Agent tests pass, including three mock E2E tests. The broader offline suite has 18 upstream failures. | Pin this revision for the local M1 implementation. |
| [Fork main at `95c8dc1ffa260bc0080c00c1f66e19844e8f53a6`](https://github.com/oliswee/pi-python-harness/commit/95c8dc1ffa260bc0080c00c1f66e19844e8f53a6) | The lock omits new workspace packages. After re-resolution, `pi --help` and test collection fail on the missing `pi_agent.harness.env` import. `AgentHarness.prompt`, `resume` and `compact` are placeholders at this revision. | Keep as a reference for later upstream evaluation. |
| [vamsi/pi](https://github.com/vamsi/pi) | GitHub shows a checked-in MIT license and a Python agent/CLI workspace. Its APIs and local test suite are still unevaluated. | Fallback candidate if the pinned port fails the release or runtime gate. |

The pinned port's Chinese README declares MIT and links to `LICENSE`. The pinned tree has no matching license file. Confirm the port's licensing and attribution before distributing the derivative. [Pinned README](https://github.com/oliswee/pi-python-harness/blob/5e41e26552d2e8ed29b7934f126bae0ee8031761/README_CN.md)

## Feature ownership

| Feature | Pinned Pi implementation | M1 evidence and extension point |
| --- | --- | --- |
| Model streaming and agent loop | `packages/agent/src/pi_agent/agent.py`, `packages/agent/src/pi_agent/agent_loop.py` | The runtime injects `AgentOptions.stream_fn` and calls `Agent.prompt`. The offline fixture exercises model → tool → model through that loop. |
| Schema validation and tool result pairing | `packages/ai/src/pi_ai/utils/validation.py`, `packages/agent/src/pi_agent/agent_loop.py` | Pi rejects a write missing its required content and returns an error with the same call ID. The adapter subscribes to Pi's tool events to correlate trace records. |
| Workspace file operations | `packages/coding-agent/src/pi_coding_agent/core/tools/{read,write,path_utils}.py` | M1 wraps Pi's read and write tools with resolved-path containment checks. The integration tests verify read/write side effects and reject parent-directory traversal. The checks assume a trusted local workspace with a stable filesystem. |
| Transcript, resume and abort | `packages/coding-agent/src/pi_coding_agent/core/{session_manager,agent_session}.py`, `packages/agent/src/pi_agent/agent.py` | Pi provides conversation history and continuation APIs. The M1 adapter runs a fresh session per invocation. For M2, put the operation-intent and completion records at the tool invocation boundary so recovery can distinguish a completed write from an uncertain one. |
| Tool ordering | `packages/agent/src/pi_agent/agent_loop.py` | The pinned tool loop executes sequentially. M3 can introduce bounded concurrency for tools explicitly classified as read-only, with deterministic result pairing. |
| Context compaction | `packages/coding-agent/src/pi_coding_agent/core/compaction/compaction.py` | Pi supplies estimation and summary generation. M5 should verify whole-request budgeting and promote a new context version only after key facts and provenance pass validation. |
| Cross-session memory and extensions | `packages/coding-agent/src/pi_coding_agent/core/skills.py`, `packages/coding-agent/src/pi_coding_agent/core/extensions/` | Pi includes a Skills loader and extension hooks. M4 needs a separate authority for cross-session records and revocation checks at context assembly. MCP remains an inventory item. |

M1 JSONL records contain run, session, turn, step and tool-call identifiers, UTC timestamps, elapsed times and status. The trace allowlist omits prompts, arguments, tool results and raw errors. Token estimate fields are null until a budget implementation exists. Trace records are written when the run terminates and serve diagnostic purposes.

Caller cancellation signals the Pi loop and discards late model events. An in-flight provider call relies on the provider honoring cancellation for prompt I/O shutdown. Any side effect already in progress remains subject to the recovery contract planned for M2.

## Reproduce on Windows

Verified with CPython 3.13.13 and uv 0.13.0 from the workspace root.

```powershell
uv sync --locked --all-packages
uv run --locked --package pi-coding-agent pi --help
uv run --locked pytest -q tests/e2e/test_baseline_agent.py packages/agent/tests
```

The provider smoke test supplies a deterministic stream and needs no API credentials. It verifies the text read from a seeded fixture, the contents of a file written by the agent, and the pairing and redaction in the saved trace. Separate cases exercise caller cancellation and a provider stream that cancels itself.

For the full offline comparison, run the same command on the pinned commit and on the M1 branch with live provider credentials absent.

```powershell
uv run --locked pytest -q --ignore=packages/ai/tests/test_live_gemini.py --timeout=20 --tb=short
```

The pinned suite had **597 passed and 18 failed**. With the seven M1 cases, the same suite had **604 passed and the same 18 failed**. The upstream failures comprise ten AgentSession or coding-agent E2E cases hitting the Anthropic credential check, five subprocess cases relying on commands unavailable as executables on this Windows setup, two cases importing `readline`, and one provider test asserting older schema-validation error text.

The first offline slice is reproducible. The broader suite and upstream license still need resolution before treating this as a release baseline.
