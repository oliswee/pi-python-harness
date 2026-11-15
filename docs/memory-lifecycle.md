# M4 scoped memory

Mini-Nanobot can inject active memory records from a caller-selected scope into a model request. SQLite holds the authoritative revisions. A scope is a caller-selected namespace, such as a user or project identifier.

```python
from pathlib import Path

from mini_nanobot import MemoryScope, MemoryStore, run

store = MemoryStore(Path("../private-state/memory.sqlite"))
record = store.create(
    "project-a",
    kind="preference",
    content="Use pytest for this project",
    provenance="confirmed by the user",
)
result = await run(
    prompt, workspace=workspace, trace_dir=trace_dir, model=model, stream_fn=provider,
    memory=MemoryScope(store, "project-a"),
)

replacement = store.supersede(
    "project-a", record.ref, content="Use unittest for this project", provenance="user correction",
)
store.revoke("project-a", replacement.ref.memory_id)
```

Keep the database outside the tool workspace. A run with a durable state directory is bound to its original database identity, path, and scope. A run can opt in to memory when it starts. Model tools cannot create memory records through the harness.

| Operation | Effect |
| --- | --- |
| `create` | Creates a new identity with revision 1, kind, provenance, an optional source run ID, and an optional timezone-aware `expires_at` datetime. |
| `supersede` | Closes the expected active revision and creates its replacement in one transaction. The replacement inherits the expiry. A stale edit raises `MemoryConflict`. |
| `invalidate` | Ends the validity of the expected active revision. |
| `revoke` | Tombstones every revision of an identity within its scope. Repeating revocation preserves the original revocation timestamp. |
| `recall` | Selects active rows in one scope, most recent first. The limit is between 1 and 16. |
| `history` | Lists retained revisions for caller-side inspection, including tombstones. |

Opening an existing version 1 database preserves its identity and history, and adds the nullable expiry field. The store normalizes expiry to UTC and treats a revision as inactive at its exact deadline. Expiry applies both to fresh recall and to the revision checks for an existing run.

## Request and recovery boundaries

The harness validates previously used revision IDs, selects current candidates, and assembles the provider context within a SQLite transaction. This transaction serializes with revocation and ends before model I/O. A request assembled after revocation commits cannot include the revoked record through recall. A request already in flight may have received the old record.

Each run tracks the revisions it supplied to the model. Durable runs sync their revision IDs to the journal before sending the associated context. If an earlier revision becomes inactive, the run returns `memory_changed` before replaying stale history or giving an affected model response back to the caller. It also validates before starting a tool. Previously started operations are covered by the [M2 recovery contract](recovery-invariants.md).

This rule conservatively stops the whole run because a historical answer or tool argument can paraphrase a memory record. Start a fresh run to assemble context from the current scope. When M5 adds compaction, summaries and key facts can use the same revision dependency check.

Memory content can remain in the SQLite history, old model replies, tool outputs, or other artifacts after revocation. The guarantee is about fresh context assembly and replay against the current database. Storage deletion, provider-side retention, and rollback to a backup with the same database identity fall outside that guarantee.

## Verification

```powershell
uv run --locked pytest -q tests/e2e/test_memory_lifecycle.py
```

The deterministic suite covers the lifecycle across database reopen, scope isolation, a concurrent revoke during assembly, rejection of stale model tool plans, and durable history checks. The mock provider captures eight newly assembled requests after revocation and verifies the retained, revoked, and other-scope contents separately. The fixtures use local SQLite and an injected offline model stream.

All eight M4 cases passed. The full offline suite had 640 passes and the same 18 upstream failures recorded in the [M1 audit](baseline-gap-audit.md).
