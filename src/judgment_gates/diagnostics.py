"""Read-only forward monitoring; never scores or reserves the held-out final test."""

import argparse
import json
import math
import subprocess
from pathlib import Path

from .ledger import verified_rows
from .receipts import digest, provenance, write_json

FIELDS = ("fill_events", "notional_usd", "spread_capture_usd", "fees_usd", "submitted_orders")


def new_engine():
    """Keep running fill accounts and a separate last-mark snapshot."""
    return {
        **dict.fromkeys(FIELDS, 0),
        "cash": 0.0,
        "inventory": 0.0,
        "decisions": 0,
        "quotes_present": 0,
        "confidence_accepted": 0,
        "marked": None,
    }


def observe(engine, event, threshold):
    """Reconcile fill cash flows independently and freeze totals at each P&L mark."""
    if event["type"] == "submission":
        engine["submitted_orders"] += event["orders"]
    elif event["type"] == "fill":
        notional = event["price"] * event["size"]
        engine["cash"] -= event["side"] * notional + event["fee"]
        engine["inventory"] += event["side"] * event["size"]
        engine["fill_events"] += 1
        engine["notional_usd"] += notional
        engine["spread_capture_usd"] += (
            event["side"] * event["size"] * (event["mid"] - event["price"])
        )
        engine["fees_usd"] += event["fee"]
    elif event["type"] == "decision":
        for field in ("cash", "inventory"):
            if not math.isclose(engine[field], event[field], rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"fill/decision {field} mismatch")
        if not math.isclose(
            event["pnl"], event["cash"] + event["inventory"] * event["mid"], abs_tol=1e-6
        ):
            raise ValueError("cash/inventory P&L mismatch")
        engine["decisions"] += 1
        engine["quotes_present"] += bool(event["quotes"])
        engine["confidence_accepted"] += max(event["p_toxic"], 1 - event["p_toxic"]) >= threshold
        engine["marked"] = {k: engine[k] for k in FIELDS}
        engine["marked"].update(
            ns=event["ns"],
            pnl_usd=event["pnl"],
            inventory_residual_usd=event["pnl"] - engine["spread_capture_usd"] + engine["fees_usd"],
        )


def summarize(engines):
    """Aggregate markets without summing cumulative daily balances or unmarked fills."""
    totals = {}
    for name, engine in engines.items():
        variant = "/".join(name.split("/")[2:])
        row = totals.setdefault(
            variant,
            {
                **dict.fromkeys(FIELDS, 0),
                "pnl_usd": 0.0,
                "inventory_residual_usd": 0.0,
                "markets_marked": 0,
                "markets_expected": 0,
                "decisions": 0,
                "quotes_present": 0,
                "confidence_accepted": 0,
                "unmarked_fill_events": 0,
                "mark_ns": [],
            },
        )
        row["markets_expected"] += 1
        for k in ("decisions", "quotes_present", "confidence_accepted"):
            row[k] += engine[k]
        marked = engine["marked"]
        row["unmarked_fill_events"] += engine["fill_events"] - (
            marked["fill_events"] if marked else 0
        )
        if marked:
            row["markets_marked"] += 1
            row["mark_ns"].append(marked["ns"])
            for k in (*FIELDS, "pnl_usd", "inventory_residual_usd"):
                row[k] += marked[k]
    for row in totals.values():
        amount = row["notional_usd"]
        row["spread_bp_per_filled_notional"] = (
            1e4 * row["spread_capture_usd"] / amount if amount else None
        )
        row["fee_bp_per_filled_notional"] = 1e4 * row["fees_usd"] / amount if amount else None
        row["mark_ns_range"] = (
            [min(row["mark_ns"]), max(row["mark_ns"])] if row["mark_ns"] else None
        )
        del row["mark_ns"]
        if row["markets_marked"] != row["markets_expected"]:
            row["pnl_usd"] = row["inventory_residual_usd"] = None
    return totals


def main():
    """Audit one immutable live checkpoint and publish descriptive monitoring only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("runs/forward-v4"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/forward-diagnostics.json"))
    args = parser.parse_args()
    state = json.loads(args.checkpoint.read_text())
    policy_path = Path("configs/forward-v4.json")
    policy = json.loads(policy_path.read_text())
    assert state["policy_sha256"] == digest(policy_path)
    frozen = subprocess.check_output(["git", "rev-parse", policy["tag"]], text=True).strip()
    engines = {
        f"{market}/{gate}/{fill}": new_engine()
        for market, config in policy["markets"].items()
        for gate in config["gates"]
        for fill in ("optimistic", "pessimistic")
    }
    starts, count, tail = [], 0, "0" * 64
    for row in verified_rows(args.directory / "ledger.jsonl", state["ledger_rows"]):
        count, tail, event = row["sequence"] + 1, row["hash"], row["event"]
        if event["type"] == "start":
            assert event["policy_sha256"] == state["policy_sha256"]
            assert event["frozen_commit"] == frozen
            starts.append({k: event[k] for k in ("ns", "frozen_commit", "policy_sha256")})
        if event.get("engine") in engines:
            market, symbol, gate, _ = event["engine"].split("/")
            observe(
                engines[event["engine"]],
                event,
                policy["markets"][market + "/" + symbol]["gates"][gate]["threshold"],
            )
    assert (count, tail) == (state["ledger_rows"], state["ledger_hash"]) and starts
    result = {
        "provenance": provenance(
            {
                "checkpoint_sha256": digest(args.checkpoint),
                "policy_sha256": digest(policy_path),
                "directory": str(args.directory),
            }
        ),
        "scope": "descriptive cumulative monitoring from commissioning; no final test or retuning",
        "paper_only": True,
        "ledger_rows": count,
        "ledger_hash": tail,
        "checkpoint_ns": state["updated_ns"],
        "starts": starts,
        "variants": summarize(engines),
        "activity_note": "No-abstention retains toxicity veto and hard risk; "
        "quote-presence observations are not uptime or independent trades.",
        "accounting_note": "Latest marked P&L = fill-mid spread capture - maker fees + inventory "
        "residual. Trailing unmarked fills excluded. Markouts overlap inventory; never add twice.",
    }
    write_json(args.output, result)
    print(f"Verified {count} rows; {len(result['variants'])} paper variants reconciled")


if __name__ == "__main__":
    main()
