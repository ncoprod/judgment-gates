import json
import time
from pathlib import Path

import numpy as np

from .data import chronological_splits, label_states, raw_events, states
from .engine import Replay, Risk
from .gates import benchmark_local, classifier, fit_classifier, fit_stack, rules, select_abstention
from .metrics import block_ci, calibration, latency, markouts, paired_state_ci, performance
from .providers import CONTEXTS, measured_successes
from .receipts import digest, provenance, write_json


def prepare(root, date, output):
    """Build labelled development states with input and source provenance."""
    result = {
        "provenance": provenance(
            {"root": str(root), "date": date, "cadence_s": 1, "label_horizon_s": 10}
        ),
        "markets": {},
    }
    for venue in ("binance", "hyperliquid"):
        for symbol in ("BTC", "ETH"):
            start = time.perf_counter()
            rows = list(states(raw_events(root, venue, symbol, date), venue))
            labelled = label_states(rows)
            key = f"{venue}/{symbol}"
            result["markets"][key] = {
                "states": labelled,
                "n": len(labelled),
                "first_ns": rows[0]["ns"] if rows else None,
                "last_ns": rows[-1]["ns"] if rows else None,
                "duration_s": (rows[-1]["ns"] - rows[0]["ns"]) / 1e9 if rows else 0,
                "preparation_s": time.perf_counter() - start,
            }
            write_json(output, result)
            print(key, "states", len(labelled), flush=True)
    manifest = Path(root) / "manifest.json"
    if manifest.exists():
        result["inputs"] = json.loads(manifest.read_text())
    write_json(output, result)
    return result


def run_replay(events, venue, symbol, rows, predictions, delays, threshold):
    """Replay fixed probabilities under optimistic and pessimistic queue fills."""
    risk = Risk(
        size=0.001 if symbol == "BTC" else 0.01,
        inventory_limit=0.005 if symbol == "BTC" else 0.05,
        tick=0.1 if venue == "binance" or symbol == "ETH" else 1.0,
    )
    decision_map = {r["ns"]: i for i, r in enumerate(rows)}
    result = {}
    for pessimistic in (False, True):
        replay = Replay(venue, risk, pessimistic)
        for event in events:
            if event["ns"] > rows[-1]["label_end_ns"]:
                break
            x = replay.event(event)
            i = decision_map.get(event["ns"])
            if i is not None and x is not None:
                replay.decide(
                    event["ns"],
                    rows[i]["x"],
                    float(predictions[i]),
                    delays[i % len(delays)],
                    threshold,
                )
                replay.mark(event["ns"])
        replay.mark(rows[-1]["label_end_ns"])
        summary = replay.summary()
        summary["decomposition_residual_usd"] = (
            summary["pnl_usd"]
            - summary["spread_capture_usd"]
            - summary["inventory_usd"]
            + summary["fees_usd"]
        )
        summary["markouts_bp"] = markouts(replay.fills, rows)
        summary["adverse_selection_note"] = (
            "10-second signed future move is diagnostic, overlaps inventory P&L; do not add twice"
        )
        summary["fills"] = replay.fills
        summary["equity"] = replay.equity
        result["pessimistic" if pessimistic else "optimistic"] = summary
    return result


def walk_forward(rows, train):
    """Evaluate expanding fits wholly inside the purged training period."""
    folds = []
    for fraction in (0.5, 0.75):
        end = int(len(train) * fraction)
        boundary = rows[train[end]]["ns"]
        past = [i for i in train[:end] if rows[i]["label_end_ns"] < boundary]
        future = train[end : min(len(train), end + max(20, len(train) // 8))]
        try:
            fold_model = fit_classifier(rows, past)
            folds.append(
                {
                    "fit_end_ns": boundary,
                    "n": len(future),
                    **calibration(
                        [rows[i]["y"] for i in future],
                        [classifier(rows[i]["x"], fold_model) for i in future],
                    ),
                }
            )
        except ValueError as exc:
            folds.append({"status": "not_run", "reason": str(exc)})
    return folds


def local_outcomes(events, venue, symbol, trial_rows, predictions, y, validation, pilot, timings):
    """Apply validation thresholds and replay each fixed local gate variant."""
    all_predictions = predictions
    trial_count = 0
    outcomes = {}
    for name, predictions in all_predictions.items():
        threshold, attempts = select_abstention(y[validation], predictions[validation])
        trial_count += len(attempts)
        for abstain in (False, True):
            title = name + ("+abstain" if abstain else "")
            selected_threshold = threshold if abstain else 0.0
            p = predictions[pilot]
            score = calibration(y[pilot], p)
            loss_ci = block_ci((p - y[pilot]) ** 2, [r["ns"] for r in trial_rows])
            replay = run_replay(
                events,
                venue,
                symbol,
                trial_rows,
                p,
                timings[name]["samples_ms"],
                selected_threshold,
            )
            accepted = np.maximum(p, 1 - p) >= selected_threshold
            outcomes[title] = {
                "threshold": selected_threshold,
                "calibration": score,
                "brier_ci": loss_ci,
                "coverage": float(accepted.mean()),
                "selective_calibration": calibration(y[pilot][accepted], p[accepted])
                if accepted.any()
                else None,
                "replay": replay,
            }
    return outcomes, trial_count


def local_study(root, date, cohort_path, output):
    """Fit on training, select on validation and retain all exploratory pilot variants."""
    cohort = json.loads(Path(cohort_path).read_text())
    receipt = {
        "provenance": provenance(
            {
                "root": str(root),
                "date": date,
                "cohort": str(cohort_path),
                "cohort_sha256": digest(cohort_path),
                "split": "50/25/25 chronological, purged labels",
                "kind": "exploratory pilot",
            }
        ),
        "markets": {},
        "trial_count": 0,
    }
    for key, market in cohort["markets"].items():
        venue, symbol = key.split("/")
        rows = market["states"]
        try:
            train, validation, pilot = chronological_splits(rows)
            model = fit_classifier(rows, train)
        except ValueError as exc:
            receipt["markets"][key] = {"status": "not_run", "reason": str(exc)}
            write_json(output, receipt)
            continue
        timings = benchmark_local(rows, model)
        all_predictions = {
            "rules": np.array([rules(r["x"]) for r in rows]),
            "classifier": np.array([classifier(r["x"], model) for r in rows]),
        }
        # Expanding walk-forward diagnostics stay wholly inside the training period.
        folds = walk_forward(rows, train)
        y = np.array([r["y"] for r in rows])
        trial_rows = [rows[i] for i in pilot]
        events = list(raw_events(root, venue, symbol, date))
        # Reconstruct past books, but reset cash/quotes at the first pilot decision.
        first = trial_rows[0]["ns"]
        outcomes, trial_count = local_outcomes(
            events, venue, symbol, trial_rows, all_predictions, y, validation, pilot, timings
        )
        receipt["trial_count"] += trial_count
        paired = (all_predictions["classifier"][pilot] - y[pilot]) ** 2 - (
            all_predictions["rules"][pilot] - y[pilot]
        ) ** 2
        receipt["markets"][key] = {
            "status": "ok",
            "model": model,
            "splits": {
                "train": len(train),
                "validation": len(validation),
                "pilot": len(pilot),
                "pilot_start_ns": first,
                "validation_end_ns": max(rows[i]["label_end_ns"] for i in validation),
            },
            "walk_forward": folds,
            "latency": timings,
            "outcomes": outcomes,
            "classifier_minus_rules_brier": block_ci(paired, [r["ns"] for r in trial_rows]),
            "abstention_candidates": {
                name: select_abstention(y[validation], p[validation])[1]
                for name, p in all_predictions.items()
            },
            "local_tilt": {
                "probability_shift": 0.0,
                "decision_flip_fraction": 0.0,
                "contexts": list(CONTEXTS),
                "n_states": len(pilot),
                "note": "structural invariance: irrelevant context excluded from numeric features",
            },
        }
        write_json(output, receipt)
        print(key, "pilot", len(pilot), "done", flush=True)
    for market in receipt["markets"].values():
        if market["status"] != "ok":
            continue
        all_sr = []
        for outcome in market["outcomes"].values():
            for replay in outcome["replay"].values():
                score = performance(
                    replay["equity"], replay["risk"]["capital"], receipt["trial_count"]
                )
                if score["sharpe"] is not None:
                    all_sr.append(score["sharpe"] / np.sqrt(24 * 365))
        for outcome in market["outcomes"].values():
            for replay in outcome["replay"].values():
                replay["performance"] = performance(
                    replay["equity"], replay["risk"]["capital"], receipt["trial_count"], all_sr
                )
    write_json(output, receipt)
    return receipt


def tilt_interval(values, timestamps):
    """Keep UTC-block inference and descriptive paired-state bootstraps distinct."""
    return {**block_ci(values, timestamps), "paired_state_bootstrap": paired_state_ci(values)}


def model_analysis(cohort_path, calls_path, output):
    """Score cached probabilities and paired framing shifts with qualified intervals."""
    cohort = json.loads(Path(cohort_path).read_text())
    calls = json.loads(Path(calls_path).read_text())
    market = calls["provenance"]["config"].get("market", "binance/BTC")
    rows = cohort["markets"][market]["states"]
    for row in calls["calls"]:
        index = row["state_index"]
        if not 0 <= index < len(rows) or rows[index]["ns"] != row["state_ns"]:
            raise ValueError("call state/cohort timestamp mismatch")
    train, validation, pilot = chronological_splits(rows)
    local = fit_classifier(rows, train)
    result = {
        "provenance": provenance(
            {
                "cohort_sha256": digest(cohort_path),
                "calls_sha256": digest(calls_path),
                "market": market,
            }
        ),
        "models": {},
    }
    for model in sorted({r["model"] for r in calls["calls"]}):
        good = [r for r in calls["calls"] if r["model"] == model and r["status"] == "ok"]
        baseline = {r["state_index"]: r for r in good if r["context"] == "neutral"}
        entry = {
            "status": "ok" if baseline else "not_run",
            "neutral_n": len(baseline),
            "latency": {
                **latency([r["latency_ms"] for r in measured_successes(good)]),
                "logical_successful_rows": len(good),
                "failed": sum(
                    r["model"] == model and r["status"] == "failed" for r in calls["calls"]
                ),
                "note": "unique physical successful requests; repeated cache entries count once",
            },
            "tilt": {},
        }
        if baseline:
            entry["calibration"] = calibration(
                [rows[i]["y"] for i in baseline],
                [r["answer"]["p_toxic"] for r in baseline.values()],
            )
            entry["coupling"] = coupling(rows, local, baseline, validation, pilot)
            for context in CONTEXTS:
                changed = list(
                    {
                        r["state_index"]: r
                        for r in good
                        if r["context"] == context and r["state_index"] in baseline
                    }.values()
                )
                shift = [
                    r["answer"]["p_toxic"] - baseline[r["state_index"]]["answer"]["p_toxic"]
                    for r in changed
                ]
                flips = [
                    int(
                        (r["answer"]["p_toxic"] < 0.5)
                        != (baseline[r["state_index"]]["answer"]["p_toxic"] < 0.5)
                    )
                    for r in changed
                ]
                stamps = [r["state_ns"] for r in changed]
                entry["tilt"][context] = {
                    "signed_probability_shift": tilt_interval(shift, stamps),
                    "absolute_probability_shift": tilt_interval(np.abs(shift), stamps),
                    "decision_flip_fraction": tilt_interval(flips, stamps),
                }
        else:
            entry["reasons"] = sorted(
                {
                    r.get("reason") or r.get("error") or "failed"
                    for r in calls["calls"]
                    if r["model"] == model
                }
            )
        result["models"][model] = entry
    write_json(output, result)
    return result


def coupling(rows, local, baseline, validation, pilot):
    """Fit stacks, uncertainty routing and abstention only on common validation rows."""
    val = [i for i in validation if i in baseline]
    test = [i for i in pilot if i in baseline]
    if len(val) < 10 or len(test) < 10:
        return {
            "status": "not_run",
            "reason": "need ten common validation and pilot predictions",
            "validation_n": len(val),
            "pilot_n": len(test),
        }
    indices = val + test
    y = np.array([rows[i]["y"] for i in indices])
    first = np.array([classifier(rows[i]["x"], local) for i in indices])
    second = np.array([baseline[i]["answer"]["p_toxic"] for i in indices])
    n = len(val)
    weight, stack_attempts = fit_stack(y[:n], first[:n], second[:n])
    candidates = []
    for band in (0.05, 0.1, 0.2, 0.3, 0.5):
        uncertain = abs(first - 0.5) <= band
        combined = np.where(uncertain, second, first)
        candidates.append({"band": band, "brier": float(np.mean((combined[:n] - y[:n]) ** 2))})
    band = min(candidates, key=lambda c: c["brier"])["band"]
    use_remote = abs(first - 0.5) <= band
    variants = {
        "classifier": first,
        "remote": second,
        "stack": weight * first + (1 - weight) * second,
        "cascade": np.where(use_remote, second, first),
    }
    outcomes = {}
    for name, predictions in variants.items():
        threshold, attempts = select_abstention(y[:n], predictions[:n])
        for abstain in (False, True):
            selected = threshold if abstain else 0.0
            p = predictions[n:]
            accepted = np.maximum(p, 1 - p) >= selected
            outcomes[name + ("+abstain" if abstain else "")] = {
                "threshold": selected,
                "coverage": float(accepted.mean()),
                "calibration": calibration(y[n:], p),
                "selective_calibration": calibration(y[n:][accepted], p[accepted])
                if accepted.any()
                else None,
                "brier_ci": block_ci((p - y[n:]) ** 2, [rows[i]["ns"] for i in test]),
                "threshold_trials": attempts,
            }
    return {
        "status": "ok",
        "validation_n": n,
        "pilot_n": len(test),
        "outcomes": outcomes,
        "stack_weight_classifier": weight,
        "stack_trials": stack_attempts,
        "cascade_band": band,
        "cascade_trials": candidates,
        "cascade_remote_fraction": float(use_remote[n:].mean()),
        "trial_count": len(stack_attempts) + len(candidates) + 5 * len(variants),
        "note": "sampled calibration experiment; no dense per-second remote replay claimed",
    }
