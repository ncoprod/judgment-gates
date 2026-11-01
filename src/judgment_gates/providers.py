import contextlib
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import httpx

from .data import FEATURES
from .metrics import latency
from .receipts import provenance, write_json

CONTEXTS = {
    "neutral": {"recent_pnl_usd": 0, "drawdown_wording": "none", "clock": "12:00"},
    "loss": {"recent_pnl_usd": -100000, "drawdown_wording": "none", "clock": "12:00"},
    "win": {"recent_pnl_usd": 100000, "drawdown_wording": "none", "clock": "12:00"},
    "drawdown": {
        "recent_pnl_usd": 0,
        "drawdown_wording": "catastrophic drawdown",
        "clock": "12:00",
    },
    "morning": {"recent_pnl_usd": 0, "drawdown_wording": "none", "clock": "08:00"},
    "night": {"recent_pnl_usd": 0, "drawdown_wording": "none", "clock": "03:00"},
    "neutral_repeat": {"recent_pnl_usd": 0, "drawdown_wording": "none", "clock": "12:00"},
}
INSTRUCTIONS = (
    "Estimate p_toxic: probability that absolute mid-price log return over the NEXT 10 seconds "
    "exceeds max(1 basis point, current spread_bp/2). This is a volatility/toxicity proxy, "
    "not a claim about profitable quotes. Features use basis points; imbalance and signed "
    "10-second trade flow are in [-1,1]. volatility_bp is the standard deviation of observed "
    "mid changes in the trailing 60 seconds. Ignore irrelevant_context: P&L, wording and clock "
    "are experimental distractors, independent of future returns. Return ONLY JSON with "
    'exactly {"p_toxic": a number in [0,1], "regime": "calm" or "volatile"}. '
    "Choose volatile if p_toxic >= 0.5. Code controls pricing, inventory, risk and abstention."
)


@contextlib.contextmanager
def nim_slot(timeout_s=None):
    """Acquire one of two atomic shared NIM slots and release only this process's slot."""
    start = time.monotonic()
    path = None
    cache = Path.home() / ".cache"
    cache.mkdir(exist_ok=True)
    while path is None:
        for i in (1, 2):
            candidate = cache / f"nim-slot-{i}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            path = candidate
            (path / "judgment-gates-owner.json").write_text(
                json.dumps({"pid": os.getpid(), "started_ns": time.time_ns()})
            )
            break
        if path is None:
            if timeout_s is not None and time.monotonic() - start > timeout_s:
                raise TimeoutError("both shared NVIDIA NIM slots remained occupied")
            time.sleep(0.2)
    try:
        yield
    finally:
        (path / "judgment-gates-owner.json").unlink()
        path.rmdir()


def state_payload(row, context):
    """Attach a framing intervention without changing the numeric market features."""
    return {
        "market_features": dict(zip(FEATURES, row["x"], strict=True)),
        "irrelevant_context": CONTEXTS[context],
    }


def validate(value):
    """Reject judgments outside the exact probability and regime schema."""
    if set(value) != {"p_toxic", "regime"} or type(value["p_toxic"]) not in (float, int):
        raise ValueError("answer does not match the judgment schema")
    p = value["p_toxic"]
    if not 0 <= p <= 1 or value["regime"] not in ("calm", "volatile"):
        raise ValueError("probability or regime out of range")
    return value


def parse_text(content):
    """Parse JSON after removing optional model reasoning and code fences."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    return validate(json.loads(content))


def request_timing(started_ns, tick):
    """Retain wall and monotonic duration and flag host pauses or clock changes."""
    awake_ms = (time.perf_counter_ns() - tick) / 1e6
    ended_ns = time.time_ns()
    wall_ms = (ended_ns - started_ns) / 1e6
    return {
        "started_ns": started_ns,
        "measured_ns": ended_ns,
        "latency_ms": max(wall_ms, awake_ms),
        "wall_elapsed_ms": wall_ms,
        "monotonic_elapsed_ms": awake_ms,
        "host_pause_or_clock_change": abs(wall_ms - awake_ms) > 1000,
    }


def provider_request(model, row, context):
    """Build a typed Jev decision or a NIM/chat judgment request."""
    state = state_payload(row, context)
    jev = model.startswith("jev") or model == "typesafe-ai/jev"
    gateway = jev or model.startswith("gateway/")
    key_name = "AI_GATEWAY_API_KEY" if gateway else "NVIDIA_API_KEY"
    if jev:
        url = "https://ai-gateway.vercel.sh/v1/evaluate"
        body = {
            "model": "typesafe-ai/jev",
            "state": state,
            "questions": {
                "veto": {
                    "type": "boolean",
                    "instructions": INSTRUCTIONS.split("Return ONLY")[0]
                    + " Should the classifier's quotes be vetoed because this event occurs? "
                    "Return its probability; code thresholds the veto and controls all orders.",
                    "criteria": {
                        "true": "The next 10-second absolute mid log return in bp exceeds "
                        "max(1, current spread_bp / 2).",
                        "false": "The return does not exceed that threshold.",
                    },
                }
            },
        }
    else:
        url = (
            "https://ai-gateway.vercel.sh/v1/chat/completions"
            if gateway
            else "https://integrate.api.nvidia.com/v1/chat/completions"
        )
        body = {
            "model": model.removeprefix("gateway/"),
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps(state, sort_keys=True)},
            ],
            "temperature": 0,
            "max_tokens": 1024,
        }
        if "gpt-oss" in model or "glm" in model or "nemotron-3-ultra" in model:
            body["reasoning_effort"] = "low"
        if "glm" in model:
            body["max_tokens"] = 8192
        if "deepseek" in model:
            body["thinking"] = {"type": "disabled"}
            body["reasoning_effort"] = "low"
        # The V2 runner rejects reasoning_budget; retain the earlier working Ultra body.
    return url, body, key_name, gateway


def send_judgment(url, body, key, timeout, client=None, retry_transport=True):
    """Send serial requests, retain attempts, and back off on rate/server errors."""
    attempts, answer, error, raw = [], None, None, None
    started_ns, start = time.time_ns(), time.perf_counter_ns()
    for attempt in range(3):
        attempt_ns, tick = time.time_ns(), time.perf_counter_ns()
        try:
            response = (client.post if client else httpx.post)(
                url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=timeout
            )
            attempts.append(
                {"http_status": response.status_code, **request_timing(attempt_ns, tick)}
            )
            try:
                raw = response.json()
            except ValueError:
                raw = {"body": response.text[:2000]}
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            if url.endswith("/evaluate"):
                decision = raw["answers"]["veto"]
                if decision["type"] != "boolean":
                    raise ValueError("Jev did not return a boolean decision")
                probability = decision["probability"]
                answer = validate(
                    {"p_toxic": probability, "regime": "volatile" if probability >= 0.5 else "calm"}
                )
            else:
                answer = parse_text(raw["choices"][0]["message"]["content"] or "")
            error = None
            break
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
            error = type(exc).__name__ + ": " + str(exc)[:300]
            if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                attempts.append({"error": type(exc).__name__, **request_timing(attempt_ns, tick)})
                if retry_transport and attempt < 2:
                    time.sleep(2**attempt)
                    continue
            break
    return {
        "status": "ok" if answer else "failed",
        "answer": answer,
        "error": error,
        "attempts": attempts,
        **request_timing(started_ns, start),
        "response_model": raw.get("model") if raw else None,
        "usage": raw.get("usage") if raw else None,
        "raw_response": raw,
    }


def call(
    model,
    row,
    context,
    cache,
    replicate=0,
    timeout=60,
    client=None,
    retry_failed=False,
    retry_transport=True,
):
    """Return a cached or measured typed probability without persisting credentials."""
    if model.startswith("opencode/"):
        return call_opencode(model, row, context, cache, replicate, timeout)
    url, body, key_name, gateway = provider_request(model, row, context)
    identity = hashlib.sha256(
        json.dumps([url, body, replicate], sort_keys=True).encode()
    ).hexdigest()
    path = Path(cache) / f"{identity}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["status"] == "ok" or not retry_failed:
            return {**saved, "cache_hit": True}
    key = os.environ.get(key_name)
    if not key:
        result = {
            "status": "not_run",
            "reason": f"{key_name} absent",
            "model": model,
            "request_id": identity,
        }
    else:
        result = {
            **send_judgment(url, body, key, timeout, client, retry_transport),
            "model": model,
            "request_id": identity,
            "request": body,
            "context": context,
            "replicate": replicate,
            "cache_hit": False,
            "immutable_model_revision": "endpoint does not expose immutable revision",
            "route": "Vercel AI Gateway (not native provider latency)" if gateway else url,
        }
    write_json(path, result)
    return result


def probe_opencode(model, output, timeout=90):
    """Read an existing probe without initiating automated OpenCode research."""
    if Path(output).exists():
        return json.loads(Path(output).read_text())
    raise RuntimeError("automated OpenCode research disabled by RELEASE_POLICY.md; use caches only")


def measured_successes(calls):
    # Reusing a response for another logical state is not a new latency sample.
    """Deduplicate cached copies of the same physical successful request."""
    return list(
        {(r["request_id"], r.get("measured_ns")): r for r in calls if r["status"] == "ok"}.values()
    )


def measure_models(
    rows,
    models,
    output,
    cache,
    count=6,
    contexts=None,
    market="binance/BTC",
    replicate=0,
    retry_failed=False,
    indices=None,
    timeout=60,
    retry_transport=True,
):
    """Resume a bounded, serial model/context matrix under the shared NIM mutex."""
    output = Path(output)
    existing = json.loads(output.read_text()) if output.exists() else {"calls": []}
    calls = existing["calls"]
    completed = {
        (r["model"], r["state_index"], r.get("context", "neutral"))
        for r in calls
        if r["status"] == "ok" or not retry_failed
    }
    contexts = list(CONTEXTS) if contexts is None else contexts
    chosen = list(indices) if indices is not None else np_indices(len(rows), count or len(rows))
    receipt = {
        "provenance": provenance(
            {
                "models": models,
                "count": count,
                "indices": chosen,
                "contexts": contexts,
                "market": market,
                "timeout_s": timeout,
                "retry_transport": retry_transport,
                "replicate": replicate,
                "context_order": "seed 42 + state index shuffle",
                "connections": "persistent HTTP client; first request cold",
                "timing": "larger of wall and monotonic elapsed; retain both; "
                "host pause or clock change flagged above one second difference",
            }
        ),
        "calls": calls,
        "stopped_models": existing.get("stopped_models", {}),
    }
    write_json(output, receipt)
    for model in models:
        if not retry_failed and (
            model in receipt["stopped_models"]
            or all((model, i, context) in completed for i in chosen for context in contexts)
        ):
            continue
        is_nim = not model.startswith(("jev", "opencode/", "typesafe-ai/", "gateway/"))
        lock = nim_slot() if is_nim else contextlib.nullcontext()
        receipt["resource_status"] = "waiting_for_nim_slot" if is_nim else "no_nim_slot_needed"
        receipt["pid"] = os.getpid()
        write_json(output, receipt)
        with lock, httpx.Client() as client:
            receipt["resource_status"] = "running_sequential_requests"
            write_json(output, receipt)
            if is_nim and os.environ.get("NVIDIA_API_KEY"):
                response = httpx.get(
                    "https://integrate.api.nvidia.com/v1/models",
                    headers={"Authorization": "Bearer " + os.environ["NVIDIA_API_KEY"]},
                    timeout=30,
                )
                write_json(
                    output.parent / "nim-catalogue.json",
                    {
                        "provenance": provenance({}),
                        "http_status": response.status_code,
                        "body": response.json(),
                    },
                )
            failures = 0
            if not retry_failed:
                for previous in reversed([r for r in calls if r["model"] == model]):
                    if previous["status"] != "failed":
                        break
                    failures += 1
            for i in chosen:
                order = list(contexts)
                random.Random(42 + i).shuffle(order)
                for context in order:
                    if (model, i, context) in completed:
                        continue
                    result = call(
                        model,
                        rows[i],
                        context,
                        cache,
                        replicate=replicate + (1 if context == "neutral_repeat" else 0),
                        client=client,
                        retry_failed=retry_failed,
                        timeout=timeout,
                        retry_transport=retry_transport,
                    )
                    calls.append(
                        {**result, "context": context, "state_index": i, "state_ns": rows[i]["ns"]}
                    )
                    write_json(output, receipt)
                    print(model, i, context, result["status"], flush=True)
                    failures = failures + 1 if result["status"] == "failed" else 0
                    if failures >= 3:
                        receipt.setdefault("stopped_models", {})[model] = (
                            "three consecutive failed judgments; remaining states not attempted"
                        )
                        break
                    # Missing key or unavailable deployment cannot improve with more states.
                    if result["status"] == "not_run" or (
                        result.get("attempts")
                        and result["attempts"][-1].get("http_status") in (404, 403)
                    ):
                        receipt["stopped_models"][model] = "missing key or unavailable deployment"
                        break
                else:
                    continue
                break
    receipt["latency"] = model_latencies(calls, models)
    receipt["resource_status"] = "finished_attempts"
    write_json(output, receipt)
    return receipt


def model_latencies(calls, models):
    """Summarise unique measured successes, failures and reported gateway charges."""
    summaries = {}
    for model in models:
        rows_model = [r for r in calls if r["model"] == model]
        costs = []
        successful = measured_successes(rows_model)
        for row in successful:
            cost = (
                (row.get("raw_response") or {})
                .get(
                    "providerMetadata", (row.get("raw_response") or {}).get("provider_metadata", {})
                )
                .get("gateway", {})
                .get("cost")
            )
            if cost is not None:
                costs.append(float(cost))
        summaries[model] = {
            **latency([r["latency_ms"] for r in successful]),
            "logical_successful_rows": sum(r["status"] == "ok" for r in rows_model),
            "failed": sum(r["status"] == "failed" for r in rows_model),
            "attempted": len(rows_model),
            "reported_gateway_cost_usd": sum(costs) if costs else None,
            "cost_per_million_usd": 1e6 * sum(costs) / len(costs)
            if costs and len(costs) == len(successful)
            else None,
            "cost_note": "unique uncached successful-request charges, excluding local energy "
            "and unreported failure charges; otherwise unknown, not assumed zero",
        }
    return summaries


def np_indices(length, count):
    # Deterministic spread over a development cohort; never choose by model outcome.
    """Choose evenly spaced development indices independently of model outcomes."""
    return sorted({i * (length - 1) // max(1, count - 1) for i in range(count)})


def call_opencode(model, row, context, cache, replicate=0, timeout=60):
    """Read a cached official-CLI response; refuse new automated extraction."""
    prompt = INSTRUCTIONS + "\n" + json.dumps(state_payload(row, context), sort_keys=True)
    identity = hashlib.sha256(json.dumps([model, prompt, replicate]).encode()).hexdigest()
    path = Path(cache) / f"{identity}.json"
    if path.exists():
        return {**json.loads(path.read_text()), "cache_hit": True}
    raise RuntimeError("automated OpenCode research disabled by RELEASE_POLICY.md; use caches only")
