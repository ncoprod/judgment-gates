import json
import math
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .receipts import digest, provenance, write_json

FEATURES = ["spread_bp", "imbalance", "flow", "return_1s_bp", "volatility_bp"]
HORIZON_NS = 10_000_000_000


def files(root, venue, symbol, date, kind="raw"):
    """List recorder partitions for one venue, symbol, date and event kind."""
    base = Path(root) / venue / symbol / date
    result = []
    for hour in range(24):
        suffix = "" if kind == "raw" else ".top20"
        canonical = base / f"{hour:02d}{suffix}.parquet"
        result.extend(
            [canonical]
            if canonical.exists()
            else sorted((base / f".{hour:02d}.{kind}.parts").glob("*.parquet"))
        )
    return result


def snapshot(root, target, date):
    """Copy immutable published files once; exclude experiments and in-flight writes."""
    target = Path(target)
    if target.exists():
        raise ValueError("snapshot exists; choose a new output to preserve the old cohort")
    entries = []
    for venue in ("binance", "hyperliquid"):
        for symbol in ("BTC", "ETH"):
            for kind in ("raw", "top20"):
                for path in files(root, venue, symbol, date, kind):
                    dest = target / path.relative_to(root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copyfile(path, dest)
                    except FileNotFoundError as exc:
                        raise RuntimeError("compaction raced snapshot; discard and retry") from exc
                    entries.append(
                        {
                            "path": str(path.relative_to(root)),
                            "sha256": digest(dest),
                            "bytes": dest.stat().st_size,
                        }
                    )
    receipt = {"provenance": provenance({"root": str(root), "date": date}), "files": entries}
    write_json(target / "manifest.json", receipt)
    return receipt


def raw_events(root, venue, symbol, date):
    """Replay recorded arrivals in availability order across partitions."""
    def rows(path):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            yield from batch.to_pylist()

    # Files are recorder processing order; buffered events can have older receive times.
    last = -1
    for path in files(root, venue, symbol, date):
        for row in rows(path):
            ns = max(row["recv_ns"], row["processed_ns"])
            if ns < last:
                raise ValueError("non-monotonic event availability; do not silently reorder")
            last = ns
            yield {**row, "ns": ns, "data": json.loads(row["payload"])}


class Book:
    def __init__(self):
        self.bids, self.asks = {}, {}
        self.session = None
        self.update_id = None
        self.bridged = False
        self.ns = 0
        self.valid = False
        self.best_bid = self.best_ask = None
        self.bid_floor = self.ask_ceiling = None
        self.source_ns = 0
        self.transaction_ns = None

    def apply(self, event):
        """Apply a snapshot or diff while enforcing venue book continuity."""
        if event["session"] != self.session:
            self.__init__()
            self.session = event["session"]
        data, stream = event["data"], event["stream"]
        if stream in ("snapshot", "checkpoint"):
            self.bids = {float(p): float(q) for p, q in data["bids"]}
            self.asks = {float(p): float(q) for p, q in data["asks"]}
            self.update_id = data["lastUpdateId"]
            self.bid_floor, self.ask_ceiling = min(self.bids), max(self.asks)
            self.bridged = data.get("bridged", False)
            self.valid = self.bridged
        elif stream == "l2Book":
            self.bids = {float(x["px"]): float(x["sz"]) for x in data["levels"][0]}
            self.asks = {float(x["px"]): float(x["sz"]) for x in data["levels"][1]}
            self.valid = True
        elif stream == "depthUpdate":
            if not event["applied"]:
                if self.update_id is None or data["u"] > self.update_id:
                    self.valid = False
                return
            if (
                self.update_id is None
                or (self.bridged and data["pu"] != self.update_id)
                or (not self.bridged and not data["U"] <= self.update_id <= data["u"])
            ):
                self.valid = False
                return
            for side, changes in ((self.bids, data["b"]), (self.asks, data["a"])):
                for p, q in changes:
                    p, q = float(p), float(q)
                    if q:
                        side[p] = q
                    else:
                        side.pop(p, None)
            self.update_id, self.bridged, self.valid = data["u"], True, True
        else:
            return
        transaction_ms = data.get("T", data.get("time"))
        self.transaction_ns = int(transaction_ms * 1e6) if transaction_ms is not None else None
        self.ns = event["ns"]
        self.source_ns = int(data.get("E", data.get("time", self.ns / 1e6)) * 1e6)
        self.valid = self.valid and bool(self.bids and self.asks)
        if self.valid:
            self.best_bid, self.best_ask = max(self.bids), min(self.asks)
            if self.best_bid >= self.best_ask:
                self.valid = False
            if self.bid_floor is not None and (
                self.best_bid <= self.bid_floor or self.best_ask >= self.ask_ceiling
            ):
                self.valid = False

    @property
    def mid(self):
        """Return the current best-bid/ask midpoint when both sides exist."""
        return (self.best_bid + self.best_ask) / 2

    def fresh(self, ns, venue):
        """Check venue-specific book age and continuity."""
        ttl = 2e9 if venue == "binance" else 12e9
        return self.valid and 0 <= ns - self.ns <= ttl


def trades(event, venue):
    """Yield trade price, size, aggressor, id and exchange transaction time."""
    if event["stream"] == "aggTrade":
        d = event["data"]
        yield (
            str(d["a"]),
            float(d["p"]),
            float(d["q"]),
            -1 if d["m"] else 1,
            int(d["T"] * 1e6) if "T" in d else None,
        )
    elif event["stream"] == "trades":
        data = event["data"]
        for d in data if isinstance(data, list) else []:
            yield (
                str(d.get("tid", d.get("hash"))),
                float(d["px"]),
                float(d["sz"]),
                (1 if d["side"] == "B" else -1),
                int(d["time"] * 1e6) if "time" in d else None,
            )


class Features:
    def __init__(self):
        self.mids, self.flow = deque(), deque()
        self.cached = None
        self.cached_ns = -1_000_000_000

    def observe(self, book, ns, signed_trades):
        """Update lagged market features using only previously observed events."""
        self.flow.extend((ns, p * q * side) for p, q, side in signed_trades)
        while self.flow and self.flow[0][0] < ns - 10e9:
            self.flow.popleft()
        if not book.valid:
            return None
        if ns - self.cached_ns < 1_000_000_000:
            return self.cached
        mid = book.mid
        previous = next((p for t, p in reversed(self.mids) if t <= ns - 1e9), mid)
        if not self.mids or ns > self.mids[-1][0]:
            self.mids.append((ns, mid))
        while self.mids and self.mids[0][0] < ns - 60e9:
            self.mids.popleft()
        bids, asks = sorted(book.bids, reverse=True)[:5], sorted(book.asks)[:5]
        b, a = sum(book.bids[p] for p in bids), sum(book.asks[p] for p in asks)
        flow = sum(v for _, v in self.flow)
        volume = sum(abs(v) for _, v in self.flow)
        changes = [
            1e4 * math.log(y[1] / x[1])
            for x, y in zip(self.mids, list(self.mids)[1:], strict=False)
        ]
        self.cached_ns = ns
        self.cached = [
            1e4 * (asks[0] - bids[0]) / mid,
            (b - a) / (b + a),
            flow / volume if volume else 0.0,
            1e4 * math.log(mid / previous),
            float(np.std(changes)) if changes else 0.0,
        ]
        return self.cached


def states(events, venue, cadence_ns=1_000_000_000):
    """Sample causally available features at the requested cadence."""
    book, features, seen = Book(), Features(), {}
    due = 0
    for event in events:
        if book.session is not None and book.session != event["session"]:
            features = Features()
        book.apply(event)
        ts = []
        for tid, p, q, side, _transaction_ns in trades(event, venue):
            if tid not in seen:
                seen[tid] = None
                if len(seen) > 10000:
                    seen.pop(next(iter(seen)))
                ts.append((p, q, side))
        x = features.observe(book, event["ns"], ts)
        if x is not None and book.fresh(event["ns"], venue) and event["ns"] >= due:
            due = event["ns"] + cadence_ns
            yield {
                "ns": event["ns"],
                "book_ns": book.ns,
                "session": book.session,
                "mid": book.mid,
                "x": x,
            }


def label_states(rows, horizon_ns=HORIZON_NS):
    """Future mid is a target only; it never enters the feature matrix."""
    times = np.array([r["ns"] for r in rows])
    result = []
    for row in rows:
        i = int(np.searchsorted(times, row["ns"] + horizon_ns))
        if i == len(rows) or times[i] - row["ns"] - horizon_ns > 2e9:
            continue
        future = rows[i]
        if future["session"] != row["session"]:
            continue
        move = 1e4 * math.log(future["mid"] / row["mid"])
        result.append(
            {
                **row,
                "label_end_ns": future["ns"],
                "move_bp": move,
                "y": int(abs(move) > max(1.0, row["x"][0] / 2)),
            }
        )
    return result


def chronological_splits(rows):
    """Create chronological train, validation and pilot indices with horizon purges."""
    if len(rows) < 90:
        raise ValueError("need at least 90 labelled states for an exploratory split")
    times = np.array([r["ns"] for r in rows])
    a, b = times[len(rows) // 2], times[len(rows) * 3 // 4]
    train = [i for i, r in enumerate(rows) if r["label_end_ns"] < a]
    validation = [i for i, r in enumerate(rows) if a <= r["ns"] and r["label_end_ns"] < b]
    pilot = [i for i, r in enumerate(rows) if r["ns"] >= b]
    if min(map(len, (train, validation, pilot))) < 10:
        raise ValueError("label purge leaves fewer than ten observations in a split")
    return train, validation, pilot
