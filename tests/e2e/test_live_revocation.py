import importlib
import json
import sys
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from pi_ai.providers import openai_completions


def test_deepseek_revocation_probe_checks_the_serialized_request(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    probe = importlib.import_module("live_revocation")
    output = tmp_path / "live"
    monkeypatch.setattr(sys, "argv", ["live_revocation.py", "--provider", "deepseek", "--output", str(output)])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock-provider-key")

    requests = []
    chunk = {
        "id": "mock-completion",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-flash",
    }

    def handle(request):
        requests.append(request)
        cache_fields = (
            {"prompt_cache_hit_tokens": 12, "prompt_cache_miss_tokens": 8}
            if len(requests) <= 4 else {"prompt_tokens_details": {"cached_tokens": 12}}
        )
        messages = [
            {**chunk, "choices": [{
                "index": 0, "delta": {"content": probe.RETAINED}, "finish_reason": None,
            }], "usage": None},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {
                "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, **cache_fields,
            }},
        ]
        stream = "".join(f"data: {json.dumps(message)}\n\n" for message in messages)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, text=stream + "data: [DONE]\n\n",
        )

    def client(**kwargs):
        return AsyncOpenAI(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), **kwargs)

    monkeypatch.setattr(openai_completions._openai, "AsyncOpenAI", client)

    assert probe.main() == 0
    assert len(requests) == 8
    for request in requests:
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["Authorization"] == "Bearer mock-provider-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-flash"
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["max_tokens"] == 96
        assert payload["stream_options"] == {"include_usage": True}
        assert "tools" not in payload
        messages = json.dumps(payload["messages"])
        assert probe.RETAINED in messages
        assert probe.REVOKED not in messages

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["provider"] == "deepseek"
    assert report["passed"]
    assert len(report["requests"]) == 8
    assert all(row["usage"]["total_tokens"] == 25 for row in report["requests"])
    assert all(row["usage"]["input"] == 8 and row["usage"]["cache_read"] == 12 for row in report["requests"])
