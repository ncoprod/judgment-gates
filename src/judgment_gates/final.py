"""One-shot, checkpoint-bound analysis; no fitting or cross-outage labels."""

import json
import math
import tempfile
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np

from .ledger import verified_rows
from .metrics import performance
from .receipts import digest, provenance, write_json
from .registration import engine_names, registered_policy

HOUR = 3_600_000_000_000


def aggregate_ci(blocks, day=False, alpha=0.05):
    """Bootstrap means from hour/day aggregate sums and counts."""
    grouped = defaultdict(lambda: [0.0, 0])
    for hour, (total, count) in sorted(blocks.items()):
        key = hour // 24 if day else hour
        grouped[key][0] += total
        grouped[key][1] += count
    sums = np.array([v[0] for v in grouped.values()])
    counts = np.array([v[1] for v in grouped.values()])
    result = {
        "mean": float(sums.sum() / counts.sum()) if counts.sum() else None,
        "n": int(counts.sum()),
        "blocks": len(counts),
        "ci": None,
        "block": "day" if day else "hour",
        "alpha": alpha,
    }
    if len(counts) < 5:
        return {**result, "reason": "fewer than five UTC blocks"}
    indices = np.random.default_rng(42).integers(len(counts), size=(2000, len(counts)))
    draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist()
    return {**result, "ci": [lo, hi], "half_width": (hi - lo) / 2}


def add(blocks, ns, value):
    """Accumulate one timestamped observation into its UTC-hour block."""
    blocks[ns // HOUR][0] += value
    blocks[ns // HOUR][1] += 1


def validated_checkpoint(policy_path, directory, end):
    """Bind the captured ledger tail to registered policy and source identities."""
    policy, commit = registered_policy(policy_path)
    directory = Path(directory)
    state = json.loads((directory / "state.json").read_text())
    sha = digest(policy_path)
    if state["policy_sha256"] != sha or state["updated_ns"] < end + 62e9:
        raise ValueError("wrong policy or checkpoint does not cover the scoring tail")
    count, anchor, starts, last_ns = 0, "0" * 64, 0, -1
    for row in verified_rows(directory / "ledger.jsonl", limit=state["ledger_rows"]):
        count, anchor = row["sequence"] + 1, row["hash"]
        event = row["event"]
        if event["ns"] < last_ns:
            raise ValueError("backward ledger clock; cannot define calendar coverage")
        last_ns = event["ns"]
        if event["type"] == "start":
            starts += 1
            if event["policy_sha256"] != sha or event["frozen_commit"] != commit:
                raise ValueError("ledger start does not match registered policy/code")
    if not starts or not count or (count, anchor) != (state["ledger_rows"], state["ledger_hash"]):
        raise ValueError("missing ledger/start or truncated checkpoint prefix")
    return policy, commit, state


def new_engine(latency_path):
    """Initialise final-analysis counters, pending horizons and latency storage."""
    return {
        "labels": deque(),
        "fills": {s: deque() for s in (1, 10, 60)},
        "markouts": {s: defaultdict(lambda: [0.0, 0]) for s in (1, 10, 60)},
        "censored": {str(s): defaultdict(int) for s in (10, 1, 60)},
        "label_censored": defaultdict(int),
        "brier": defaultdict(lambda: [0.0, 0]),
        "bins": np.zeros((10, 3)),
        "hourly": {},
        "coverage": defaultdict(int),
        "last_slot": None,
        "first_ns": None,
        "last_ns": None,
        "last": None,
        "session": None,
        "baseline": None,
        "fill_count": 0,
        "filled_volume": 0.0,
        "submitted_volume": 0.0,
        "fees": 0.0,
        "spread": 0.0,
        "peak": 0.0,
        "drawdown": 0.0,
        "latency_path": latency_path,
    }


def censor(e, reason):
    """Discard pending labels and markouts at a continuity break."""
    e["label_censored"][reason] += len(e["labels"])
    e["labels"].clear()
    for seconds, pending in e["fills"].items():
        e["censored"][str(seconds)][reason] += len(pending)
        pending.clear()


def observe(e, event, start, end, cadence):
    """Update one engine from an ordered event without bridging gaps."""
    ns, kind = event["ns"], event["type"]
    session = event.get("session")
    if session is None:
        raise ValueError("decision/fill/submission missing continuity identity")
    if e["session"] is not None and session != e["session"]:
        censor(e, "gap_or_restart")
    e["session"] = session
    if kind == "submission":
        if start <= ns < end:
            e["submitted_volume"] += event["volume"]
        return
    if kind == "fill":
        if start <= ns < end:
            e["fill_count"] += 1
            e["filled_volume"] += event["size"]
            e["fees"] += event["fee"]
            e["spread"] += event["side"] * event["size"] * (event["mid"] - event["price"])
            for pending in e["fills"].values():
                pending.append(event)
        return
    if ns < start:
        e["baseline"] = event["pnl"]
        return
    while e["labels"] and e["labels"][0]["ns"] + 10e9 <= ns:
        past = e["labels"].popleft()
        if ns - past["ns"] > 12e9:
            e["label_censored"]["horizon_tolerance"] += 1
            continue
        y = int(abs(1e4 * math.log(event["mid"] / past["mid"])) > max(1, past["features"][0] / 2))
        p = past["p_toxic"]
        add(e["brier"], past["ns"], (p - y) ** 2)
        e["bins"][min(int(p * 10), 9)] += [1, p, y]
    for seconds, pending in e["fills"].items():
        while pending and pending[0]["ns"] + seconds * 1e9 <= ns:
            fill = pending.popleft()
            if ns - fill["ns"] > (seconds + 2) * 1e9:
                e["censored"][str(seconds)]["horizon_tolerance"] += 1
                continue
            add(
                e["markouts"][seconds],
                fill["ns"],
                fill["side"] * 1e4 * (event["mid"] - fill["mid"]) / fill["mid"],
            )
    if ns >= end:
        return
    e["labels"].append(event)
    e["first_ns"] = e["first_ns"] if e["first_ns"] is not None else ns
    e["last_ns"], e["last"] = ns, event
    slot = ns // cadence
    if slot != e["last_slot"]:
        e["coverage"][ns // HOUR] += 1
        e["last_slot"] = slot
    with e["latency_path"].open("ab") as f:
        np.asarray([event["latency_ms"]], dtype="float64").tofile(f)
    if e["baseline"] is not None:
        pnl = event["pnl"] - e["baseline"]
        e["hourly"][ns // HOUR] = pnl
        e["peak"] = max(e["peak"], pnl)
        e["drawdown"] = max(e["drawdown"], e["peak"] - pnl)


def coverage(e, start, end, cadence):
    """Enumerate registered-hour and day coverage, including missing boundaries."""
    hours = {}
    for hour in range(start // HOUR, (end - 1) // HOUR + 1):
        expected = len(
            range(
                max(start, hour * HOUR) // cadence, (min(end, (hour + 1) * HOUR) - 1) // cadence + 1
            )
        )
        hours[str(hour)] = {"expected": expected, "observed": e["coverage"].get(hour, 0)}
    days = defaultdict(lambda: [0, 0])
    for h, v in hours.items():
        days[int(h) // 24][0] += v["observed"]
        days[int(h) // 24][1] += v["expected"]
    complete = sum(
        expected * cadence == 24 * HOUR and observed >= 0.9 * expected
        for observed, expected in days.values()
    )
    edges = (
        e["first_ns"] is not None
        and e["first_ns"] <= start + 2 * cadence
        and e["last_ns"] >= end - 2 * cadence
    )
    eligible = bool(
        edges
        and e["baseline"] is not None
        and all(v["observed"] >= 0.9 * v["expected"] for v in hours.values())
    )
    return {
        "hours": hours,
        "edges_observed": bool(edges),
        "complete_days_90pct": complete,
        "risk_eligible": eligible,
        "precision_eligible": eligible and complete >= 5,
    }


def latency_summary(path):
    """Compute exact empirical quantiles from disk-backed decision latencies."""
    if not path.exists() or not path.stat().st_size:
        return {"n": 0, "p50_ms": None, "p99_ms": None}
    # Disk-backed exact quantiles avoid retaining months of Python floats in RAM.
    values = np.memmap(path, dtype="float64", mode="r+")
    p50, p99 = np.quantile(values, [0.5, 0.99], overwrite_input=True).tolist()
    return {"n": len(values), "p50_ms": p50, "p99_ms": p99}


def summarize(e, policy, market, start, end, sharpes):
    """Build coverage-qualified calibration, execution and risk diagnostics."""
    censor(e, "horizon_end")
    cov = coverage(e, start, end, int(policy["cadence_s"] * 1e9))
    equity = [{"ns": h * HOUR, "pnl": p} for h, p in sorted(e["hourly"].items())]
    ratios = performance(
        equity if cov["risk_eligible"] else [],
        market["risk"]["capital"],
        policy["conservative_trial_count"],
        sharpes,
    )
    if not cov["risk_eligible"]:
        ratios["reason"] = "missing baseline, boundary coverage, or 90% hourly decision coverage"
    n = int(e["bins"][:, 0].sum())
    pnl = equity[-1]["pnl"] if equity else None
    last = e["last"]
    flatten = (
        abs(last["inventory"] * last["mid"]) * market["risk"]["taker_fee_bp"] / 1e4
        if last
        else None
    )
    return {
        "status": "ok" if cov["risk_eligible"] else "insufficient_data",
        "coverage": cov,
        "brier_hour": aggregate_ci(e["brier"]),
        "brier_day": aggregate_ci(e["brier"], True),
        "label_censoring": dict(e["label_censored"]),
        "reliability": [
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "n": int(k),
                "probability": float(p / k) if k else None,
                "frequency": float(y / k) if k else None,
            }
            for i, (k, p, y) in enumerate(e["bins"])
        ],
        "ece": float(sum(abs(p - y) for k, p, y in e["bins"]) / n) if n else None,
        "pnl_usd": pnl,
        "spread_capture_usd": e["spread"],
        "fees_usd": e["fees"],
        "inventory_residual_usd": pnl - e["spread"] + e["fees"] if pnl is not None else None,
        "flatten_fee_usd": flatten,
        "pnl_after_flatten_fee_usd": pnl - flatten
        if pnl is not None and flatten is not None
        else None,
        "max_drawdown_usd": e["drawdown"],
        "fill_events": e["fill_count"],
        "submitted_volume": e["submitted_volume"],
        "filled_volume": e["filled_volume"],
        "fill_rate": e["filled_volume"] / e["submitted_volume"] if e["submitted_volume"] else None,
        "latency": latency_summary(e["latency_path"]),
        "performance": ratios,
        "markouts_bp": {
            s: {
                "hour": aggregate_ci(b),
                "day": aggregate_ci(b, True),
                "censoring": dict(e["censored"][str(s)]),
            }
            for s, b in e["markouts"].items()
        },
    }


def evaluate(policy_path, directory, output):
    """Reserve and score the registered prospective cohort once after its end."""
    policy = json.loads(Path(policy_path).read_text())
    start, end = [
        int(datetime.fromisoformat(policy[k]).timestamp() * 1e9) for k in ("test_start", "test_end")
    ]
    if time.time_ns() < end + 62e9:
        raise ValueError("prospective test has not ended; daily scoring is provisional")
    policy, commit, state = validated_checkpoint(policy_path, directory, end)
    if Path(output).exists():
        raise FileExistsError("one-shot result exists")
    reservation = Path(directory) / ("final-" + digest(policy_path) + ".started")
    with reservation.open("x") as f:
        f.write(
            json.dumps(
                {
                    "policy_sha256": digest(policy_path),
                    "frozen_commit": commit,
                    "checkpoint": state,
                    "output": str(output),
                }
            )
        )
    result = {
        "provenance": provenance(
            {
                "policy_sha256": digest(policy_path),
                "start_ns": start,
                "end_ns": end,
                "frozen_commit": commit,
            }
        ),
        "checkpoint": state,
        "trial_count": policy["conservative_trial_count"],
        "engines": {},
        "paired_markout": {},
    }
    with tempfile.TemporaryDirectory(prefix="jg-final-") as tmp:
        engines = {
            name: new_engine(Path(tmp) / f"{i}.bin") for i, name in enumerate(engine_names(policy))
        }
        for row in verified_rows(Path(directory) / "ledger.jsonl", limit=state["ledger_rows"]):
            event = row["event"]
            if event["ns"] >= end + 62e9:
                continue
            if event["type"] in ("gap", "start"):
                for name, e in engines.items():
                    if event["type"] == "start" or name.startswith(event["market"] + "/"):
                        censor(e, "gap_or_restart")
                continue
            if event["type"] in ("decision", "fill", "submission"):
                if event["engine"] not in engines:
                    raise ValueError("unregistered engine in ledger")
                observe(engines[event["engine"]], event, start, end, int(policy["cadence_s"] * 1e9))
        sharpes = []
        for e in engines.values():
            cov = coverage(e, start, end, int(policy["cadence_s"] * 1e9))
            if cov["risk_eligible"]:
                score = performance(
                    [{"ns": h * HOUR, "pnl": p} for h, p in sorted(e["hourly"].items())],
                    10000,
                    policy["conservative_trial_count"],
                )
                if score["sharpe"] is not None:
                    sharpes.append(score["sharpe"] / math.sqrt(24 * 365))
        for name, e in engines.items():
            market = policy["markets"]["/".join(name.split("/")[:2])]
            result["engines"][name] = summarize(e, policy, market, start, end, sharpes)
            if "/classifier/" in name:
                other = engines.get(name.replace("/classifier/", "/rules/"))
                if other is None:
                    continue
                a, b = e["markouts"][10], other["markouts"][10]
                paired = {
                    h: [a[h][0] / a[h][1] - b[h][0] / b[h][1], 1]
                    for h in a.keys() & b.keys()
                    if a[h][1] and b[h][1]
                }
                primary = name.endswith("/pessimistic")
                alpha = 0.05 / len(policy["markets"]) if primary else 0.05
                result["paired_markout"][name] = {
                    "family": "primary" if primary else "secondary_descriptive",
                    "multiplicity": "Bonferroni across markets" if primary else "none; exploratory",
                    "hour": aggregate_ci(paired, alpha=alpha),
                    "day": aggregate_ci(paired, True, alpha=alpha),
                    "estimand": (
                        "equal-weight matched-hour mean markout difference, classifier minus rules"
                    ),
                    "coverage_eligible": (
                        result["engines"][name]["coverage"]["precision_eligible"]
                        and coverage(other, start, end, int(policy["cadence_s"] * 1e9))[
                            "precision_eligible"
                        ]
                    ),
                }
    result["status"] = (
        "ok"
        if all(e["status"] == "ok" for e in result["engines"].values())
        else "insufficient_data"
    )
    write_json(output, result)
    return result
