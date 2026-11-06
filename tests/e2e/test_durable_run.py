import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).with_name("durable_run_worker.py")


def attempt(tmp_path, fault="", *, tool="write", prompt="Run fixture"):
    return subprocess.run(
        [sys.executable, str(WORKER), str(tmp_path), fault, tool, prompt],
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )


def outcome(process):
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def events(tmp_path):
    path = tmp_path / "state" / "journal.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.mark.parametrize(
    ("window", "last_event", "resume_status", "calls"),
    [
        ("before_intent", "request", "ok", 2),
        ("before_execution", "approved", "ok", 1),
        ("after_effect", "started", "uncertain", 0),
        ("after_result", "succeeded", "ok", 1),
    ],
)
def test_restart_after_process_crash(tmp_path, workspace, window, last_event, resume_status, calls):
    crash = attempt(tmp_path, window)
    assert crash.returncode == 86, crash.stderr
    history = events(tmp_path)
    assert history[-1]["kind"] == last_event
    assert not (tmp_path / "state" / "checkpoint.json").exists()

    target = workspace / "effect.txt"
    if window in {"after_effect", "after_result"}:
        assert target.read_text(encoding="utf-8") == "effect from tool"
        target.write_text("operator sentinel", encoding="utf-8")
    else:
        assert not target.exists()

    result = outcome(attempt(tmp_path))
    assert result["status"] == resume_status
    assert result["provider_calls"] == calls
    assert result["uncertain_call_ids"] == (["operation-1"] if resume_status == "uncertain" else [])
    assert target.read_text(encoding="utf-8") == (
        "operator sentinel" if window in {"after_effect", "after_result"} else "effect from tool"
    )
    final_history = events(tmp_path)
    assert final_history[:len(history)] == history
    assert [record["seq"] for record in final_history] == list(range(1, len(final_history) + 1))
    assert sum(record["kind"] == "started" for record in final_history) == 1


def test_interrupted_read_can_retry(tmp_path, workspace):
    target = workspace / "effect.txt"
    target.write_text("original", encoding="utf-8")
    crash = attempt(tmp_path, "after_effect", tool="read")
    assert crash.returncode == 86, crash.stderr
    target.write_text("updated", encoding="utf-8")

    result = outcome(attempt(tmp_path, tool="read"))

    assert result["status"] == "ok"
    assert result["text"] == "updated"
    assert result["provider_calls"] == 1
    assert sum(record["kind"] == "started" for record in events(tmp_path)) == 2


def test_corrupt_checkpoint_replays_journal_and_preserves_completed_write(tmp_path, workspace):
    first = outcome(attempt(tmp_path))
    checkpoint = tmp_path / "state" / "checkpoint.json"
    checkpoint.write_text('{"checksum":"bad"}', encoding="utf-8")
    target = workspace / "effect.txt"
    target.write_text("operator sentinel", encoding="utf-8")

    restarted = outcome(attempt(tmp_path))

    assert restarted["status"] == "ok"
    assert restarted["run_id"] == first["run_id"]
    assert restarted["provider_calls"] == 0
    assert restarted["trace_path"] != first["trace_path"]
    assert target.read_text(encoding="utf-8") == "operator sentinel"
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["checksum"] != "bad"


def test_corrupt_authoritative_record_fails_closed_even_with_checkpoint(tmp_path, workspace):
    outcome(attempt(tmp_path))
    journal = tmp_path / "state" / "journal.jsonl"
    journal.write_bytes(journal.read_bytes().replace(b"effect from tool", b"tampered content"))
    target = workspace / "effect.txt"
    target.write_text("operator sentinel", encoding="utf-8")

    process = attempt(tmp_path)

    assert process.returncode != 0
    assert "checksum" in process.stderr.lower()
    assert target.read_text(encoding="utf-8") == "operator sentinel"


def test_recovery_rejects_a_different_request(tmp_path, workspace):
    outcome(attempt(tmp_path))
    process = attempt(tmp_path, prompt="Different prompt")
    assert process.returncode != 0
    assert "request does not match" in process.stderr.lower()


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_checkpoint_ahead_of_journal_fails_closed(tmp_path, workspace, damage):
    outcome(attempt(tmp_path))
    journal = tmp_path / "state" / "journal.jsonl"
    if damage == "missing":
        journal.unlink()
    else:
        # Keep a valid checksummed prefix that predates the tool execution.
        journal.write_bytes(journal.read_bytes().splitlines(keepends=True)[0])
    target = workspace / "effect.txt"
    target.write_text("operator sentinel", encoding="utf-8")

    restarted = attempt(tmp_path)

    assert restarted.returncode != 0
    assert "journal" in restarted.stderr.lower()
    assert target.read_text(encoding="utf-8") == "operator sentinel"


def test_concurrent_process_cannot_open_same_run(tmp_path, workspace):
    from mini_nanobot.journal import RunJournal

    with RunJournal.open(tmp_path / "state", {"owner": "active writer"}):
        rejected = attempt(tmp_path)

    assert rejected.returncode != 0
    assert "cannot acquire the run journal lock" in rejected.stderr.lower()
    assert not (workspace / "effect.txt").exists()


def test_duplicate_call_ids_rejected_before_any_effect(tmp_path, workspace):
    rejected = attempt(tmp_path, "duplicate_id")

    assert rejected.returncode != 0
    assert "duplicate tool call id" in rejected.stderr.lower()
    assert events(tmp_path)[-1]["kind"] == "request"
    assert not (workspace / "effect.txt").exists()


def test_self_consistent_checkpoint_cannot_reclassify_completed_write(tmp_path, workspace):
    outcome(attempt(tmp_path))
    checkpoint = tmp_path / "state" / "checkpoint.json"
    snapshot = json.loads(checkpoint.read_text(encoding="utf-8"))
    snapshot.pop("checksum")
    snapshot["state"]["operations"]["operation-1"] = {"state": "planned", "name": "write"}
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot["checksum"] = hashlib.sha256(payload).hexdigest()
    checkpoint.write_text(json.dumps(snapshot), encoding="utf-8")
    target = workspace / "effect.txt"
    target.write_text("operator sentinel", encoding="utf-8")

    restarted = outcome(attempt(tmp_path))

    assert restarted["status"] == "ok"
    assert restarted["provider_calls"] == 0
    assert target.read_text(encoding="utf-8") == "operator sentinel"


def test_replays_records_after_an_older_valid_checkpoint(tmp_path, workspace):
    first = outcome(attempt(tmp_path, "provider_error"))
    assert first["status"] == "error"
    checkpoint = tmp_path / "state" / "checkpoint.json"
    old_checkpoint = checkpoint.read_bytes()

    crash = attempt(tmp_path, "before_checkpoint")
    assert crash.returncode == 86, crash.stderr
    assert checkpoint.read_bytes() == old_checkpoint
    assert events(tmp_path)[-1]["data"]["status"] == "ok"
    target = workspace / "effect.txt"
    target.write_text("operator sentinel", encoding="utf-8")

    restarted = outcome(attempt(tmp_path))

    assert restarted["status"] == "ok"
    assert restarted["provider_calls"] == 0
    assert target.read_text(encoding="utf-8") == "operator sentinel"
