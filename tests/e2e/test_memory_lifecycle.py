import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import pytest

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall

from mini_nanobot import run


def create(store, scope="project", content="old setting"):
    return store.create(scope, kind="preference", content=content, provenance="operator")


def reply(model, *parts, reason="stop"):
    message = AssistantMessage(
        content=list(parts), api=model.api, provider=model.provider, model=model.id, timestamp=0, stop_reason=reason,
    )
    return EventDone(type="done", reason=reason, message=message)


async def exercise(tmp_path, provider, *, memory=None, durable=False):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return await run(
        "Fixture", workspace=workspace, trace_dir=tmp_path / "traces",
        model=get_model("anthropic", "claude-3-5-sonnet-20241022"), stream_fn=provider,
        state_dir=tmp_path / "state" if durable else None, memory=memory,
    )


def test_revision_lifecycle_is_persistent_and_scoped(tmp_path):
    from mini_nanobot import MemoryConflict, MemoryStore

    db = tmp_path / "memory.sqlite"
    store = MemoryStore(db)
    original = create(store)
    other = create(store, "other", "private to other scope")
    assert original.ref.revision == 1 and original.provenance == "operator"
    assert store.recall("project") == [original]

    replacement = store.supersede(
        "project", original.ref, content="new setting", provenance="user correction", source_run_id="run-2",
    )
    assert replacement.ref.memory_id == original.ref.memory_id
    assert replacement.ref.revision == 2 and replacement.source_run_id == "run-2"
    with pytest.raises(MemoryConflict):
        store.supersede("project", original.ref, content="stale edit", provenance="operator")
    with pytest.raises(MemoryConflict):
        store.invalidate("other", replacement.ref)

    restarted = MemoryStore(db)
    assert restarted.recall("project") == [replacement]
    assert restarted.recall("other") == [other]
    restarted.invalidate("project", replacement.ref)
    assert store.recall("project") == []
    store.revoke("project", replacement.ref.memory_id)
    store.revoke("project", replacement.ref.memory_id)
    assert [record.ref.revision for record in store.history("project", original.ref.memory_id)] == [1, 2]
    old, new = store.history("project", original.ref.memory_id)
    assert old.superseded_by == 2 and old.valid_to is not None
    assert new.valid_to is not None
    assert all(record.tombstone and record.revoked_at for record in (old, new))
    with pytest.raises(MemoryConflict):
        store.supersede("project", replacement.ref, content="resurrect", provenance="operator")


def test_revoke_serializes_against_context_assembly(tmp_path):
    from mini_nanobot import MemoryStore

    db = tmp_path / "memory.sqlite"
    store, revoker = MemoryStore(db), MemoryStore(db)
    record = create(store)
    started = threading.Event()

    def revoke():
        started.set()
        revoker.revoke("project", record.ref.memory_id)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.assembly("project", set(), limit=8) as current:
            assert current == [record]
            pending = pool.submit(revoke)
            assert started.wait(2)
            with pytest.raises(FutureTimeout):
                pending.result(timeout=0.1)
        pending.result(timeout=5)

    with store.assembly("project", set(), limit=8) as current:
        assert current == []


def test_stale_store_handle_cannot_mutate_replacement_database(tmp_path):
    from mini_nanobot import MemoryConflict, MemoryStore

    store = MemoryStore(tmp_path / "memory.sqlite")
    replacement = MemoryStore(tmp_path / "replacement.sqlite")
    retained = create(replacement, content="belongs to replacement")
    replacement.path.replace(store.path)

    with pytest.raises(MemoryConflict, match="identity"):
        create(store, content="must not be inserted")

    assert MemoryStore(store.path).recall("project") == [retained]


async def test_eight_new_requests_exclude_revoked_revision_after_restart(tmp_path):
    from mini_nanobot import MemoryScope, MemoryStore

    db = tmp_path / "memory.sqlite"
    store = MemoryStore(db)
    removed = create(store, content="revocable-key-529")
    create(store, content="retained-key-230")
    create(store, scope="other", content="other-scope-859")
    contexts = []

    async def provider(model, context, options=None):
        contexts.append(context.model_dump(mode="json", exclude={"tools"}))
        yield reply(model, TextContent(type="text", text="ok"))

    first = await exercise(tmp_path, provider, memory=MemoryScope(store, "project"))
    assert first.status == "ok"
    assert "revocable-key-529" in contexts[0]["system_prompt"]
    assert "other-scope-859" not in contexts[0]["system_prompt"]
    store.revoke("project", removed.ref.memory_id)

    for _ in range(8):
        scope = MemoryScope(MemoryStore(db), "project")
        result = await exercise(tmp_path, provider, memory=scope)
        assert result.status == "ok"
        assert "revocable-key-529" not in result.trace_path.read_text(encoding="utf-8")

    assert len(contexts) == 9
    for context in contexts[1:]:
        assert "revocable-key-529" not in json.dumps(context)
        assert "retained-key-230" in context["system_prompt"]
        assert "other-scope-859" not in context["system_prompt"]


@pytest.mark.parametrize("durable", [False, True])
async def test_revocation_while_model_is_running_blocks_its_tool_plan(tmp_path, durable):
    from mini_nanobot import MemoryScope, MemoryStore

    store = MemoryStore(tmp_path / "memory.sqlite")
    record = create(store)
    calls = 0

    async def provider(model, context, options=None):
        nonlocal calls
        calls += 1
        assert "old setting" in context.system_prompt
        await asyncio.to_thread(store.revoke, "project", record.ref.memory_id)
        yield reply(model, ToolCall(
            type="toolCall", id="late", name="write", arguments={"path": "late.txt", "content": "deny"},
        ), reason="toolUse")

    result = await exercise(tmp_path, provider, memory=MemoryScope(store, "project"), durable=durable)

    assert result.status == "memory_changed" and result.text == ""
    assert calls == 1
    assert not (tmp_path / "workspace" / "late.txt").exists()


async def test_durable_history_is_blocked_after_supersession(tmp_path):
    from mini_nanobot import MemoryScope, MemoryStore
    from mini_nanobot.journal import JournalError

    db = tmp_path / "memory.sqlite"
    store = MemoryStore(db)
    record = create(store, content="original injected fact")

    async def provider(model, context, options=None):
        assert "original injected fact" in context.system_prompt
        yield reply(model, TextContent(type="text", text="paraphrase of old fact"))

    first = await exercise(tmp_path, provider, memory=MemoryScope(store, "project"), durable=True)
    assert first.text == "paraphrase of old fact"

    store.supersede("project", record.ref, content="corrected fact", provenance="operator")
    invoked = False

    async def unexpected_provider(model, context, options=None):
        nonlocal invoked
        invoked = True
        yield reply(model, TextContent(type="text", text="unexpected"))

    reopened = MemoryScope(MemoryStore(db), "project")
    result = await exercise(tmp_path, unexpected_provider, memory=reopened, durable=True)
    assert result.status == "memory_changed"
    assert result.run_id == first.run_id and result.text == ""
    assert not invoked
    with pytest.raises(JournalError, match="Request does not match"):
        await exercise(tmp_path, unexpected_provider, durable=True)
    with pytest.raises(JournalError, match="Request does not match"):
        await exercise(tmp_path, unexpected_provider, memory=MemoryScope(store, "other"), durable=True)


async def test_replaced_empty_store_blocks_planned_write_on_recovery(tmp_path):
    from mini_nanobot import MemoryScope, MemoryStore
    from mini_nanobot.journal import RunJournal, digest
    from mini_nanobot.workspace_tools import TOOL_POLICY_VERSION

    store = MemoryStore(tmp_path / "memory.sqlite")
    scope = MemoryScope(store, "project")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    request = {
        "workspace": str(workspace.resolve()), "prompt": digest("Fixture"),
        "model": digest(model.model_dump(mode="json")), "memory": scope.binding(),
    }
    planned = reply(model, ToolCall(
        type="toolCall", id="planned", name="write", arguments={"path": "late.txt", "content": "deny"},
    ), reason="toolUse")
    with RunJournal.open(tmp_path / "state", request, tool_policy_version=TOOL_POLICY_VERSION) as journal:
        journal.append("reply", {"message": planned.message.model_dump(mode="json")})

    replacement = MemoryStore(tmp_path / "replacement.sqlite")
    replacement.path.replace(store.path)

    async def provider(model, context, options=None):
        yield reply(model, TextContent(type="text", text="unexpected"))

    restarted = await exercise(tmp_path, provider, memory=scope, durable=True)

    assert restarted.status == "memory_changed"
    assert not (workspace / "late.txt").exists()
