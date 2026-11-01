import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

from .engine import Replay, Risk
from .gates import classifier, rules
from .ledger import Ledger, compress_segment, verified_rows
from .receipts import digest, provenance, write_json
from .registration import registered_policy


def freeze(pilot_path, output):
    """Persist the fitted local policy and preregistered prospective window."""
    pilot = json.loads(Path(pilot_path).read_text())
    policy = {
        "provenance": provenance({"pilot_sha256": digest(pilot_path)}),
        "paper_only": True,
        "tag": "forward-v4",
        "cadence_s": 1,
        "test_start": "2026-10-30T00:00:00+00:00",
        "test_end": "2027-01-28T00:00:00+00:00",
        "markets": {},
        "conservative_trial_count": json.loads(Path("results/trials.json").read_text())[
            "conservative_trial_count"
        ],
    }
    for key, result in pilot["markets"].items():
        venue, symbol = key.split("/")
        risk = Risk(
            size=0.001 if symbol == "BTC" else 0.01,
            inventory_limit=0.005 if symbol == "BTC" else 0.05,
            tick=0.1 if venue == "binance" or symbol == "ETH" else 1.0,
        )
        entry = {"risk": asdict(risk), "gates": {"rules": {"threshold": 0.0}}}
        if result["status"] == "ok":
            entry["model"] = result["model"]
            for name in ("rules", "classifier"):
                entry["gates"][name] = {"threshold": 0.0}
                entry["gates"][name + "+abstain"] = {
                    "threshold": result["outcomes"][name + "+abstain"]["threshold"]
                }
        policy["markets"][key] = entry
    write_json(output, policy)
    return policy


class Paper:
    def __init__(self, policy_path, directory):
        self.directory = Path(directory)
        self.policy, tag = registered_policy(policy_path, running=True)
        if self.policy.get("paper_only") is not True:
            raise ValueError("only paper policies are supported")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = self.directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        anchor = (state["ledger_rows"], state["ledger_hash"]) if state else None
        self.ledger = Ledger(self.directory / "ledger.jsonl", anchor)
        self.engines, self.due, self.sessions, self.last = {}, {}, {}, {}
        self.fills_seen, self.events = {}, 0
        for key, market in self.policy["markets"].items():
            for gate in market["gates"]:
                for pessimistic in (False, True):
                    name = key + "/" + gate + ("/pessimistic" if pessimistic else "/optimistic")
                    self.engines[name] = Replay(
                        key.split("/")[0], Risk(**market["risk"]), pessimistic
                    )
                    self.fills_seen[name] = 0
        # Cash/inventory persist; pending quotes never survive a process outage.
        if state:
            if state["policy_sha256"] != digest(policy_path):
                raise ValueError("policy changed; use a separate forward cohort directory")
            for name, balances in state["balances"].items():
                engine = self.engines[name]
                engine.cash, engine.inventory = balances["cash"], balances["inventory"]
            for row in verified_rows(self.ledger.path, anchor=anchor):
                event = row["event"]
                if event["type"] == "decision":
                    engine = self.engines[event["engine"]]
                    engine.cash, engine.inventory = event["cash"], event["inventory"]
                elif event["type"] == "fill":
                    engine = self.engines[event["engine"]]
                    engine.cash -= event["side"] * event["price"] * event["size"] + event["fee"]
                    engine.inventory += event["side"] * event["size"]
        self.policy_sha = digest(policy_path)
        self.ledger.append(
            {
                "type": "start",
                "ns": time.time_ns(),
                "pid": os.getpid(),
                "policy_sha256": self.policy_sha,
                "frozen_commit": tag,
                "provenance": provenance({"paper_only": True}),
                "note": "restart clears outstanding quotes; gaps are not simulated fills",
            }
        )
        self.checkpoint()

    def checkpoint(self):
        """Flush a durable ledger tail and recoverable paper balances."""
        write_json(
            self.directory / "state.json",
            {
                "updated_ns": time.time_ns(),
                "policy_sha256": self.policy_sha,
                "ledger_hash": self.ledger.previous,
                "ledger_rows": self.ledger.count,
                "events": self.events,
                "last_events_ns": self.last,
                "balances": {
                    name: {"cash": e.cash, "inventory": e.inventory}
                    for name, e in self.engines.items()
                },
            },
        )

    def emit(self, key, stream, data, session=None):
        """Feed a public-stream event through the frozen paper engines."""
        ns = time.time_ns()
        if session:
            self.sessions[key] = session
        if key not in self.sessions:
            return
        if stream != "clock":
            self.last[key] = ns
            self.events += 1
        event = {
            "ns": ns,
            "session": self.sessions[key],
            "stream": stream,
            "data": data,
            "applied": True,
        }
        market = self.policy["markets"][key]
        should_decide = ns >= self.due.get(key, 0)
        for name, engine in self.engines.items():
            if not name.startswith(key + "/"):
                continue
            submitted = engine.stats["submitted"]
            x = engine.event(event)
            count = engine.stats["submitted"] - submitted
            if count:
                self.ledger.append(
                    {
                        "type": "submission",
                        "engine": name,
                        "ns": ns,
                        "orders": count,
                        "volume": count * engine.risk.size,
                        "session": self.sessions[key],
                    }
                )
            for fill in engine.fills[self.fills_seen[name] :]:
                self.ledger.append({"type": "fill", "engine": name, **fill})
            self.fills_seen[name] = len(engine.fills)
            # Fills are already durably logged; keep process memory bounded.
            if len(engine.fills) > 100:
                engine.fills.clear()
                self.fills_seen[name] = 0
            if should_decide and x is not None and engine.book.fresh(ns, engine.venue):
                gate = name.split("/")[2]
                start = time.perf_counter_ns()
                p = classifier(x, market["model"]) if gate.startswith("classifier") else rules(x)
                delay = (time.perf_counter_ns() - start) / 1e6
                engine.decide(ns, x, p, delay, market["gates"][gate]["threshold"])
                engine.mark(ns)
                self.ledger.append(
                    {
                        "type": "decision",
                        "engine": name,
                        "ns": ns,
                        "session": self.sessions[key],
                        "features": x,
                        "p_toxic": p,
                        "latency_ms": delay,
                        "pending": engine.pending,
                        "quotes": [asdict(q) for q in engine.quotes],
                        "inventory": engine.inventory,
                        "cash": engine.cash,
                        "mid": engine.book.mid,
                        "pnl": engine.equity[-1]["pnl"],
                    }
                )
                engine.equity.clear()
        if should_decide:
            step = int(self.policy["cadence_s"] * 1e9)
            self.due[key] = (ns // step + 1) * step

    def gap(self, key, reason):
        """Record a continuity break and cancel simulated outstanding exposure."""
        self.ledger.append({"type": "gap", "market": key, "reason": reason, "ns": time.time_ns()})
        for name, engine in self.engines.items():
            if name.startswith(key + "/"):
                engine.quotes, engine.pending = [], None
                engine.book.valid = False
        self.checkpoint()


async def binance_depth(paper, symbol):
    """Consume public Binance depth updates with snapshot bridging."""
    key = f"binance/{symbol}"
    url = f"wss://fstream.binance.com/public/stream?streams={symbol.lower()}usdt@depth@100ms"
    while True:
        try:
            async with connect(url, max_queue=4096, open_timeout=20) as ws:
                async with httpx.AsyncClient(timeout=20) as c:
                    snapshot = (
                        await c.get(
                            "https://fapi.binance.com/fapi/v1/depth",
                            params={"symbol": symbol + "USDT", "limit": 1000},
                        )
                    ).json()
                update = snapshot["lastUpdateId"]
                paper.emit(key, "snapshot", snapshot, uuid.uuid4().hex)
                bridged = False
                while True:
                    message = json.loads(await asyncio.wait_for(ws.recv(), 30))["data"]
                    if message["u"] < update or (bridged and message["u"] <= update):
                        continue
                    if (not bridged and not message["U"] <= update <= message["u"]) or (
                        bridged and message["pu"] != update
                    ):
                        raise ValueError("sequence_gap")
                    update, bridged = message["u"], True
                    paper.emit(key, "depthUpdate", message)
                    representative = next(
                        e for name, e in paper.engines.items() if name.startswith(key + "/")
                    )
                    if not representative.book.valid:
                        raise ValueError("incomplete_snapshot_coverage")
        except (OSError, TimeoutError, ValueError, KeyError) as exc:
            paper.gap(key, type(exc).__name__ + ": " + str(exc)[:120])
        except Exception as exc:
            paper.gap(key, type(exc).__name__)
        await asyncio.sleep(5)


async def binance_trades(paper, symbol):
    """Consume public Binance trades without exchange account access."""
    key = f"binance/{symbol}"
    url = f"wss://fstream.binance.com/market/stream?streams={symbol.lower()}usdt@aggTrade"
    while True:
        try:
            async with connect(url, open_timeout=20) as ws:
                while True:
                    message = json.loads(await asyncio.wait_for(ws.recv(), 30))["data"]
                    paper.emit(key, "aggTrade", message)
        except Exception as exc:
            paper.gap(key, "trade_stream_" + type(exc).__name__)
        await asyncio.sleep(5)


async def hyperliquid(paper):
    """Consume public Hyperliquid books and trades with reconnect handling."""
    while True:
        try:
            async with connect("wss://api.hyperliquid.xyz/ws", open_timeout=20) as ws:
                session = uuid.uuid4().hex
                for symbol in ("BTC", "ETH"):
                    for kind in ("l2Book", "trades"):
                        await ws.send(
                            json.dumps(
                                {
                                    "method": "subscribe",
                                    "subscription": {"type": kind, "coin": symbol},
                                }
                            )
                        )
                while True:
                    message = json.loads(await asyncio.wait_for(ws.recv(), 30))
                    kind, data = message.get("channel"), message.get("data")
                    if kind == "l2Book":
                        paper.emit("hyperliquid/" + data["coin"], kind, data, session)
                    elif kind == "trades":
                        for symbol in ("BTC", "ETH"):
                            selected = [t for t in data if t["coin"] == symbol]
                            if selected:
                                paper.emit("hyperliquid/" + symbol, kind, selected)
        except Exception as exc:
            for symbol in ("BTC", "ETH"):
                paper.gap("hyperliquid/" + symbol, type(exc).__name__)
        await asyncio.sleep(5)


async def live(policy, directory):
    """Run the frozen paper engines on public streams only."""
    paper = Paper(policy, directory)

    async def heartbeat():
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        while True:
            await asyncio.sleep(10)
            paper.ledger.append(
                {
                    "type": "heartbeat",
                    "ns": time.time_ns(),
                    "events": paper.events,
                    "last_events_ns": paper.last,
                }
            )
            paper.checkpoint()
            now = datetime.now(UTC).strftime("%Y-%m-%d")
            if now != date:
                segment = paper.ledger.rotate(date)
                date = now
                if segment:
                    asyncio.create_task(asyncio.to_thread(compress_segment, segment))

    async def clock():
        while True:
            await asyncio.sleep(0.1)
            for key in paper.policy["markets"]:
                if time.time_ns() >= paper.due.get(key, 0):
                    paper.emit(key, "clock", {})

    await asyncio.gather(
        hyperliquid(paper),
        heartbeat(),
        clock(),
        *[
            function(paper, symbol)
            for symbol in ("BTC", "ETH")
            for function in (binance_depth, binance_trades)
        ],
    )


def score(directory, output, readme=None):
    """Verify a checkpoint prefix and update cumulative daily paper receipts."""
    directory = Path(directory)
    # Scoring coordinates with the event loop through an immutable checkpoint prefix.
    state = json.loads((directory / "state.json").read_text())
    saved = json.loads(Path(output).read_text()) if Path(output).exists() else None
    anchor = (saved["ledger_rows"], saved["ledger_hash"]) if saved else None
    count, previous = anchor if anchor else (0, "0" * 64)
    days = saved["days"] if saved else {}
    for row in verified_rows(directory / "ledger.jsonl", state["ledger_rows"], anchor):
        previous, count = row["hash"], row["sequence"] + 1
        e = row["event"]
        if e["type"] not in ("decision", "fill", "gap"):
            continue
        date = datetime.fromtimestamp(e["ns"] / 1e9, UTC).strftime("%Y-%m-%d")
        item = days.setdefault(date, {"decisions": 0, "fills": 0, "gaps": 0, "engines": {}})
        item[{"decision": "decisions", "fill": "fills", "gap": "gaps"}[e["type"]]] += 1
        if e["type"] == "decision":
            item["engines"][e["engine"]] = e["pnl"]
    if previous != state["ledger_hash"] or count != state["ledger_rows"]:
        raise ValueError("checkpoint prefix does not match its anchor")
    latest = {}
    for day in sorted(days):
        latest.update(days[day]["engines"])
    totals = {}
    for name, pnl in latest.items():
        gate = "/".join(name.split("/")[2:])
        item = totals.setdefault(gate, {"pnl_usd": 0.0, "markets": 0})
        item["pnl_usd"] += pnl
        item["markets"] += 1
    result = {
        "provenance": provenance({"directory": str(directory)}),
        "ledger_hash": previous,
        "ledger_rows": count,
        "checkpoint_ns": state["updated_ns"],
        "days": days,
        "cumulative_gate_totals": totals,
        "paper_only": True,
        "verification": "incremental from prior receipt anchor; final scorer audits full chain",
    }
    write_json(output, result)
    if readme:
        table = "| UTC day | Variant decisions | Fill events | Gap events |\n|---|---:|---:|---:|\n"
        for date, day in sorted(days.items()):
            table += f"| {date} | {day['decisions']} | {day['fills']} | {day['gaps']} |\n"
        table += "\n| Gate / fill scenario | Markets observed | Cumulative paper P&L (USD) |\n"
        table += "|---|---:|---:|\n"
        for name, row in sorted(totals.items()):
            table += f"| {name} | {row['markets']} | {row['pnl_usd']:.8f} |\n"
        table += "\nP&L begins at commissioning, before the prospective test window.\n"
        text = Path(readme).read_text()
        before, rest = text.split("<!-- DAILY:START -->")
        _, after = rest.split("<!-- DAILY:END -->")
        link = os.path.relpath(Path(output).resolve(), Path(readme).resolve().parent)
        Path(readme).write_text(
            before
            + "<!-- DAILY:START -->\n"
            + table
            + f"\nSource: [{link}]({link}). Provisional; frozen policy is never retuned.\n"
            + "<!-- DAILY:END -->"
            + after
        )
    return result
