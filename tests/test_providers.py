import json

import httpx
import pytest

from judgment_gates.providers import call, measured_successes, model_latencies


def test_gateway_uses_decision_shape_and_retains_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-token")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(403, json={"error": {"type": "no_providers_available"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = call("typesafe-ai/jev", {"x": [1, 0, 0, 0, 0]}, "loss", tmp_path, client=client)
    assert str(requests[0].url) == "https://ai-gateway.vercel.sh/v1/evaluate"
    assert result["request"]["questions"]["veto"]["type"] == "boolean"
    assert "messages" not in result["request"]
    assert result["raw_response"]["error"]["type"] == "no_providers_available"
    assert "test-only-token" not in next(tmp_path.iterdir()).read_text()


def test_gateway_reads_boolean_probability_and_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-token")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "answers": {"veto": {"type": "boolean", "probability": 0.23}},
                    "usage": {"inputTokens": 50},
                    "providerMetadata": {"gateway": {"cost": "0.00001"}},
                },
            )
        )
    ) as client:
        result = call("typesafe-ai/jev", {"x": [1, 0, 0, 0, 0]}, "neutral", tmp_path, client=client)
    assert result["answer"] == {"p_toxic": 0.23, "regime": "calm"}
    assert result["usage"]["inputTokens"] == 50
    assert model_latencies([result], ["typesafe-ai/jev"])["typesafe-ai/jev"][
        "cost_per_million_usd"
    ] == pytest.approx(10)


@pytest.mark.parametrize("model", ["z-ai/glm-5.3-flash", "deepseek-ai/deepseek-v4.1-flash"])
def test_nim_documented_reasoning_controls(model, tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-only-token")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json={"choices": [{"message": {"content": '{"p_toxic":0.2,"regime":"calm"}'}}]}
            )
        )
    ) as client:
        result = call(model, {"x": [1, 0, 0, 0, 0]}, "neutral", tmp_path, client=client)
    assert result["request"]["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in result["request"]
    if "glm" in model:
        assert result["request"]["max_tokens"] == 8192
    else:
        assert result["request"]["thinking"] == {"type": "disabled"}


def test_timeout_records_every_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-only-token")
    monkeypatch.setattr("judgment_gates.providers.time.sleep", lambda _: None)

    def timeout(request):
        raise httpx.ReadTimeout("simulated timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        result = call(
            "openai/gpt-oss-20b", {"x": [1, 0, 0, 0, 0]}, "neutral", tmp_path, client=client
        )
    assert result["status"] == "failed"
    assert len(result["attempts"]) == 3
    assert all(a["error"] == "ReadTimeout" for a in result["attempts"])


def test_cached_rows_do_not_inflate_measured_latency_sample():
    first = {"request_id": "a", "measured_ns": 1, "latency_ms": 10, "status": "ok"}
    cached = {**first, "cache_hit": True}
    independent = {**first, "request_id": "b", "latency_ms": 30}
    failed = {**first, "request_id": "c", "status": "failed"}
    assert [r["latency_ms"] for r in measured_successes([first, cached, independent, failed])] == [
        10,
        30,
    ]


def test_nemotron_uses_supported_reasoning_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-only-token")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json={"choices": [{"message": {"content": '{"p_toxic":0.2,"regime":"calm"}'}}]}
            )
        )
    ) as client:
        result = call(
            "nvidia/nemotron-3-ultra-550b-a55b",
            {"x": [1, 0, 0, 0, 0]},
            "neutral",
            tmp_path,
            client=client,
        )
    body = result["request"]
    assert body["reasoning_effort"] == "low"
    assert "reasoning_budget" not in body


def test_analysis_rejects_calls_from_a_different_cohort(tmp_path):
    from judgment_gates.study import model_analysis

    cohort = tmp_path / "cohort.json"
    calls = tmp_path / "calls.json"
    cohort.write_text(json.dumps({"markets": {"binance/BTC": {"states": [{"ns": 10}]}}}))
    calls.write_text(
        json.dumps({"provenance": {"config": {}}, "calls": [{"state_index": 0, "state_ns": 20}]})
    )
    with pytest.raises(ValueError, match="call state/cohort timestamp mismatch"):
        model_analysis(cohort, calls, tmp_path / "analysis.json")


def test_resume_preserves_failure_cutoff_without_acquiring_slot(tmp_path, monkeypatch):
    from judgment_gates import providers

    path = tmp_path / "calls.json"
    path.write_text(
        json.dumps(
            {
                "calls": [],
                "stopped_models": {"z-ai/glm-5.3-flash": "three consecutive failed judgments"},
            }
        )
    )

    def forbidden():
        raise AssertionError("finished model must not acquire a slot or make requests")

    monkeypatch.setattr(providers, "nim_slot", forbidden)
    result = providers.measure_models([{"ns": 1}], ["z-ai/glm-5.3-flash"], path, tmp_path, count=1)
    assert result["calls"] == []
    assert result["stopped_models"]["z-ai/glm-5.3-flash"] == "three consecutive failed judgments"


def test_resume_counts_prior_consecutive_failures(tmp_path, monkeypatch):
    import contextlib

    from judgment_gates import providers

    model = "z-ai/glm-5.3-flash"
    path = tmp_path / "calls.json"
    path.write_text(
        json.dumps(
            {
                "calls": [
                    {"model": model, "state_index": i, "context": "neutral", "status": "failed"}
                    for i in range(2)
                ]
            }
        )
    )
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(providers, "nim_slot", contextlib.nullcontext)
    monkeypatch.setattr(providers, "call", lambda *a, **k: {"model": model, "status": "failed"})
    result = providers.measure_models(
        [{"ns": i} for i in range(5)], [model], path, tmp_path, count=5, contexts=["neutral"]
    )
    assert len(result["calls"]) == 3
    assert "three consecutive" in result["stopped_models"][model]
