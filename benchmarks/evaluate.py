"""Run the scripted Mini-Nanobot case manifest in independent pytest processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "cases.json"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def collect_artifacts(fixture: Path | None, responses: list[dict]) -> dict:
    traces, journal_replies, terminal_statuses = [], [], Counter()
    starts = ends = unpaired = 0
    if fixture is not None:
        for path in sorted(fixture.rglob("*.jsonl")):
            if path.parent.name == "traces":
                rows = read_jsonl(path)
                traces.append({"file": path.relative_to(fixture).as_posix(), "records": rows})
                began = Counter(row["tool_call_id"] for row in rows if row["event_type"] == "tool_start")
                finished = Counter(row["tool_call_id"] for row in rows if row["event_type"] == "tool_end")
                starts += began.total()
                ends += finished.total()
                unpaired += (began - finished).total() + (finished - began).total()
                terminal_statuses.update(row["status"] for row in rows if row["event_type"] == "run_end")
            elif path.name == "journal.jsonl":
                journal_replies.extend(
                    row["data"]["message"] for row in read_jsonl(path) if row["kind"] == "reply"
                )
    return {
        "responses": responses, "durable_replies": journal_replies, "traces": traces,
        "metrics": {
            "tool_starts": starts, "tool_results": ends, "unpaired_events": unpaired,
            "terminal_statuses": dict(terminal_statuses),
        },
    }


def worker(output: Path, nodes: list[str]) -> int:
    import pytest
    from pi_agent import Agent
    from pi_ai.types import AssistantMessage

    class Capture:
        def __init__(self):
            self.reports = {}
            self.fixtures = {}

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            yield
            if call.when == "call":
                self.fixtures[item.nodeid] = item.funcargs.get("tmp_path")

        def pytest_runtest_logreport(self, report):
            observation = self.reports.setdefault(report.nodeid, {"phases": {}})
            phase = {"status": report.outcome, "duration_s": round(report.duration, 6)}
            if report.failed or report.skipped:
                phase["detail"] = report.longreprtext
            observation["phases"][report.when] = phase

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_protocol(self, item, nextitem):
            responses = []
            original = Agent._emit

            def emit(agent, event):
                if event.type == "message_end" and isinstance(event.message, AssistantMessage):
                    responses.append(event.message.model_dump(mode="json"))
                original(agent, event)

            with patch.object(Agent, "_emit", emit):
                yield
            observation = self.reports.setdefault(item.nodeid, {"phases": {}})
            payload = collect_artifacts(self.fixtures.get(item.nodeid), responses)
            artifact = hashlib.sha256(item.nodeid.encode()).hexdigest()[:16] + ".json"
            write_json(output / artifact, payload)
            observation.update(artifact=artifact, **payload["metrics"])

    capture = Capture()
    code = pytest.main(
        ["-q", "--tb=short", "--timeout=50", f"--basetemp={output / 'fixtures'}", *nodes], plugins=[capture],
    )
    write_json(output / "observations.json", capture.reports)
    return int(code)


def summarize_case(case: dict, observations: dict, trial: int, directory: Path) -> dict:
    tests = {node: observations.get(node, {"phases": {}}) for node in case["nodes"]}
    phases = [phase for test in tests.values() for phase in test["phases"].values()]
    outcomes = {phase["status"] for phase in phases}
    if "failed" in outcomes or any(not test["phases"] for test in tests.values()):
        status = "failed"
    elif "skipped" in outcomes:
        status = "skipped"
    elif all(test["phases"].get("call", {}).get("status") == "passed" for test in tests.values()):
        status = "passed"
    else:
        status = "failed"
    terminals = Counter()
    for test in tests.values():
        terminals.update(test.get("terminal_statuses", {}))
    return {
        "id": case["id"], "trial": trial, "status": status,
        "duration_s": round(sum(phase["duration_s"] for phase in phases), 6),
        "tool_starts": sum(test.get("tool_starts", 0) for test in tests.values()),
        "tool_results": sum(test.get("tool_results", 0) for test in tests.values()),
        "unpaired_events": sum(test.get("unpaired_events", 0) for test in tests.values()),
        "terminal_statuses": dict(terminals),
        "artifacts": [f"{directory.name}/{test['artifact']}" for test in tests.values() if "artifact" in test],
    }


def evaluate(output: Path, selected: list[str], trials: int) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {case["id"] for case in manifest}
    if set(selected) - known:
        raise ValueError(f"Unknown benchmark case: {', '.join(sorted(set(selected) - known))}")
    cases = [case for case in manifest if not selected or case["id"] in selected]
    nodes = list(dict.fromkeys(node for case in cases for node in case["nodes"]))
    metadata = {
        "revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "platform": platform.platform(), "python": platform.python_version(),
        "provider": "deterministic fixture", "trials": trials, "case_ids": [case["id"] for case in cases],
    }
    output.mkdir(parents=True, exist_ok=False)
    rows, worker_errors = [], []
    for trial in range(1, trials + 1):
        directory = output / f"trial-{trial:02}"
        directory.mkdir()
        started = time.perf_counter()
        try:
            with (directory / "pytest.log").open("wb") as log:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory), *nodes],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=900,
                )
            if process.returncode != 0:
                worker_errors.append({"trial": trial, "code": process.returncode})
        except subprocess.TimeoutExpired:
            worker_errors.append({"trial": trial, "code": "timeout"})
        path = directory / "observations.json"
        observations = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        rows.extend(summarize_case(case, observations, trial, directory) for case in cases)
        if not observations:
            worker_errors.append({"trial": trial, "code": "missing_observations"})
        print(f"Trial {trial}/{trials} finished in {time.perf_counter() - started:.2f}s", flush=True)

    passed = sum(
        all(row["status"] == "passed" for row in rows if row["id"] == case["id"]) for case in cases
    )
    summary = {
        **metadata, "passed_cases": passed, "outcomes": dict(Counter(row["status"] for row in rows)),
        "worker_errors": worker_errors, "results": rows,
    }
    write_json(output / "report.json", summary)
    print(f"{passed}/{len(cases)} cases passed every trial. Report: {output / 'report.json'}")
    return int(bool(worker_errors) or any(row["status"] == "failed" for row in rows) or passed < min(10, len(cases)))


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(Path(sys.argv[2]), sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    try:
        return evaluate(args.output.resolve(), args.case, args.trials)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
