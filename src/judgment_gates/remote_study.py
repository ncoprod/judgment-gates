"""Dense remote replay; refuse incomplete matrices rather than invent judgments."""

import json
from pathlib import Path

import numpy as np

from .data import chronological_splits, raw_events
from .gates import benchmark_local, classifier, fit_classifier, fit_stack, select_abstention
from .metrics import calibration, performance
from .receipts import digest, provenance, write_json
from .study import run_replay


def minute_indices(rows, indices):
    """Select the first eligible state at each minute-cadence boundary."""
    selected, next_ns = [], 0
    for i in indices:
        if rows[i]["ns"] >= next_ns:
            selected.append(i)
            next_ns = rows[i]["ns"] + 60_000_000_000
    return selected


def regime_mask(rows, predictions, delays, local, scheduled=None):
    """Apply remote regimes only after their measured arrival and before expiry."""
    ready, probability, next_call, pending = 0, 1.0, 0, None
    observed_ns = 0
    masked = []
    calls = 0
    for i, row in enumerate(rows):
        if pending is not None and row["ns"] >= ready:
            probability, observed_ns = pending
            pending = None
        eligible = scheduled is None or i in scheduled
        if eligible and row["ns"] >= next_call and pending is None:
            ready = row["ns"] + int(delays[i] * 1e6)
            pending = predictions[i], row["ns"]
            next_call = row["ns"] + 60_000_000_000
            calls += 1
        stale = row["ns"] - observed_ns > 120_000_000_000
        masked.append(max(local[i], 1.0 if stale else probability))
    return np.array(masked), calls


def evaluate(root, date, cohort_path, calls_path, output, market="binance/BTC", regime_only=False):
    """Replay validation-fitted combinations only with a complete common prediction matrix."""
    rows = json.loads(Path(cohort_path).read_text())["markets"][market]["states"]
    calls = json.loads(Path(calls_path).read_text())["calls"]
    train, val, pilot = chronological_splits(rows)
    model = fit_classifier(rows, train)
    timing = benchmark_local(rows, model)
    indices = val + pilot
    local = np.array([classifier(rows[i]["x"], model) for i in indices])
    y = np.array([rows[i]["y"] for i in indices])
    receipt = {
        "provenance": provenance(
            {
                "cohort_sha256": digest(cohort_path),
                "calls_sha256": digest(calls_path),
                "market": market,
                "kind": "development replay",
                "regime_only": regime_only,
            }
        ),
        "models": {},
        "trial_count": 0,
        "local_latency": timing,
    }
    for name in sorted({r["model"] for r in calls}):
        answers = {
            r["state_index"]: r
            for r in calls
            if r["model"] == name and r["context"] == "neutral" and r["status"] == "ok"
        }
        required = minute_indices(rows, indices) if regime_only else indices
        missing = [i for i in required if i not in answers]
        if missing:
            receipt["models"][name] = {
                "status": "not_run",
                "missing_predictions": len(missing),
                "reason": "missing required neutral validation/pilot predictions",
            }
            continue
        remote = np.array([answers[i]["answer"]["p_toxic"] if i in answers else 1 for i in indices])
        delays = np.array([answers[i]["latency_ms"] if i in answers else 0 for i in indices])
        n = len(val)
        w, stack_trials = (0, []) if regime_only else fit_stack(y[:n], local[:n], remote[:n])
        bands = []
        for band in (() if regime_only else (0.05, 0.1, 0.2, 0.3, 0.5)):
            p = np.where(abs(local[:n] - 0.5) <= band, remote[:n], local[:n])
            bands.append({"band": band, "brier": float(np.mean((p - y[:n]) ** 2))})
        band = min(bands, key=lambda r: r["brier"])["band"] if bands else 0
        uncertain = abs(local - 0.5) <= band
        # Local timing is separately measured, not substituted with a vendor claim.
        samples = timing["classifier"]["samples_ms"]
        local_delays = np.array([samples[i % len(samples)] for i in range(len(indices))])
        scheduled = {j for j, i in enumerate(indices) if i in required} if regime_only else None
        regime, regime_calls = regime_mask(
            [rows[i] for i in indices], remote, delays, local, scheduled
        )
        variants = {
            "remote": (remote, delays),
            "stack": (
                w * local + (1 - w) * remote,
                local_delays if w == 1 else delays + (local_delays if w > 0 else 0),
            ),
            "cascade": (
                np.where(uncertain, remote, local),
                local_delays + np.where(uncertain, delays, 0),
            ),
            "minute_regime": (regime, local_delays),
        }
        if regime_only:
            variants = {"minute_regime": variants["minute_regime"]}
        outcomes = {}
        venue, symbol = market.split("/")
        events = list(raw_events(root, venue, symbol, date))
        for variant, (p, milliseconds) in variants.items():
            threshold, trials = select_abstention(y[:n], p[:n])
            receipt["trial_count"] += len(trials) * 2
            for abstain in (False, True):
                selected = threshold if abstain else 0.0
                replay = run_replay(
                    events,
                    venue,
                    symbol,
                    [rows[i] for i in pilot],
                    p[n:],
                    milliseconds[n:].tolist(),
                    selected,
                )
                outcomes[variant + ("+abstain" if abstain else "")] = {
                    "calibration": calibration(y[n:], p[n:]),
                    "coverage": float((np.maximum(p[n:], 1 - p[n:]) >= selected).mean()),
                    "threshold": selected,
                    "replay": replay,
                }
        receipt["trial_count"] += len(stack_trials) + len(bands)
        receipt["models"][name] = {
            "status": "ok",
            "stack_trials": stack_trials,
            "cascade_trials": bands,
            "regime_calls": regime_calls,
            "required_indices": required,
            "outcomes": outcomes,
            "cost_note": "report actual cached API charges separately from logical cascade calls",
        }
        write_json(output, receipt)
    for model in receipt["models"].values():
        for outcome in model.get("outcomes", {}).values():
            for scenario in outcome["replay"].values():
                scenario["performance"] = performance(
                    scenario["equity"], 10000, receipt["trial_count"]
                )
    write_json(output, receipt)
    return receipt
