from datetime import datetime, timedelta, timezone

from pi_ai import get_model
from pi_ai.types import AssistantMessage, EventDone, TextContent

from mini_nanobot import MemoryScope, MemoryStore, run
from mini_nanobot import memory_store


async def test_ranked_recall_filters_history_in_both_independent_runs(tmp_path, monkeypatch):
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(memory_store, "_now", lambda: now.isoformat(timespec="microseconds"))
    db = tmp_path / "memory.sqlite"
    store = MemoryStore(db)

    def add(content, scope="shop", **kwargs):
        nonlocal now
        now += timedelta(microseconds=1)
        return store.create(scope, kind="rule", content=content, provenance="fixture", **kwargs)

    relevant = add("invoice.py discount: retain cents")
    old = add("invoice.py discount: legacy cap")
    temporary = add("invoice.py discount: temporary override", expires_at=now + timedelta(minutes=1))
    removed = add("invoice.py discount: never round down")
    unrelated = add("shipping labels use uppercase")
    foreign = add("invoice.py discount: foreign account", scope="other")
    assert store.recall("shop", limit=1) == [unrelated]
    query = 'invoice.py "discount" OR ()'
    assert relevant in store.recall("shop", query=query)
    assert store.recall("shop", query="unmatched storage keyword") == []
    assert store.recall("shop", query='(): * " …') == []

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    contexts = []
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    async def provider(model, context, options):
        contexts.append(context.model_dump_json())
        message = AssistantMessage(
            content=[TextContent(type="text", text="ok")], api=model.api, provider=model.provider,
            model=model.id, timestamp=0, stop_reason="stop",
        )
        yield EventDone(type="done", reason="stop", message=message)

    async def perform(store, state):
        result = await run(
            "Review the invoice discount rules.", workspace=workspace, trace_dir=tmp_path / "traces",
            state_dir=tmp_path / state, model=model, stream_fn=provider,
            memory=MemoryScope(store, "shop", query=query),
        )
        assert result.status == "ok"
        return result

    first = await perform(store, "first")
    assert all(record.content in contexts[0] for record in (relevant, old, temporary, removed))
    assert all(record.content not in contexts[0] for record in (unrelated, foreign))

    updated = store.supersede("shop", old.ref, content="invoice.py discount: current cap", provenance="fixture")
    store.revoke("shop", removed.ref.memory_id)
    now += timedelta(minutes=1)
    reopened = MemoryStore(db)
    assert set(reopened.recall("shop", query=query)) == {relevant, updated}
    second = await perform(reopened, "second")
    assert first.run_id != second.run_id and len(contexts) == 2
    assert all(record.content in contexts[1] for record in (updated, relevant))
    assert all(record.content not in contexts[1] for record in (old, temporary, removed, unrelated, foreign))
