import math
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from .data import Book, Features, trades


@dataclass
class Risk:
    size: float = 0.001
    inventory_limit: float = 0.005
    maker_fee_bp: float = 2.0
    taker_fee_bp: float = 5.0
    capital: float = 10_000.0
    tick: float = 0.1
    gamma: float = 0.1
    kappa: float = 10_000.0
    horizon_s: float = 60.0
    max_state_age_s: float = 15.0


@dataclass
class Quote:
    side: int
    price: float
    remaining: float
    ahead: float
    joined_exchange_ns: int | None = None

    def trade(self, price, volume, aggressor, pessimistic):
        """Consume eligible aggressor volume through queue ahead and remaining size."""
        through = price < self.price if self.side == 1 else price > self.price
        touches = price <= self.price if self.side == 1 else price >= self.price
        if aggressor != -self.side or not (through if pessimistic else touches):
            return 0.0
        consumed = min(self.ahead, volume)
        self.ahead -= consumed
        filled = min(self.remaining, max(0.0, volume - consumed))
        self.remaining -= filled
        return filled


def round_tick(value, tick, rounding):
    """Round a price with decimal arithmetic at the venue tick."""
    step = Decimal(str(tick))
    return float((Decimal(str(value)) / step).to_integral_value(rounding=rounding) * step)


def prices(book, inventory, volatility_bp, risk):
    """A–S in relative-return units, then round outward and enforce post-only."""
    mid = book.mid
    variance = (volatility_bp / 1e4) ** 2
    reservation = mid * (1 - (inventory / risk.size) * risk.gamma * variance * risk.horizon_s)
    width = mid * (
        risk.gamma * variance * risk.horizon_s
        + 2 / risk.gamma * math.log1p(risk.gamma / risk.kappa)
    )
    # Decimal ticks round-trip to the same float keys as exchange price strings.
    bid = round_tick(min(reservation - width / 2, max(book.bids)), risk.tick, ROUND_FLOOR)
    ask = round_tick(max(reservation + width / 2, min(book.asks)), risk.tick, ROUND_CEILING)
    return bid, ask


class Replay:
    def __init__(self, venue, risk=None, pessimistic=True):
        self.venue, self.risk, self.pessimistic = venue, risk or Risk(), pessimistic
        self.book, self.features = Book(), Features()
        self.quotes, self.pending, self.fills, self.equity = [], None, [], []
        self.inventory = self.cash = self.spread_capture = self.inventory_pnl = self.fees = 0.0
        self.previous_mid = None
        self.exchange_bound = None
        self.seen = {}
        self.stats = {
            "submitted": 0,
            "rejected_post_only": 0,
            "stale_holds": 0,
            "decisions": 0,
            "busy_holds": 0,
            "unknown_queue_holds": 0,
            "pre_entry_trades": 0,
            "unknown_trade_order": 0,
        }

    def advance(self, ns):
        """Activate delayed quotes with post-only, staleness and inventory checks."""
        if self.pending is None or ns < self.pending[0]:
            return
        _, state_ns, bid, ask = self.pending
        self.pending, self.quotes = None, []
        if not self.book.fresh(ns, self.venue) or ns - state_ns > self.risk.max_state_age_s * 1e9:
            self.stats["stale_holds"] += 1
            return
        for side, price in ((1, bid), (-1, ask)):
            if price is None or abs(self.inventory + side * self.risk.size) > (
                self.risk.inventory_limit + 1e-12
            ):
                continue
            if (side == 1 and price >= min(self.book.asks)) or (
                side == -1 and price <= max(self.book.bids)
            ):
                self.stats["rejected_post_only"] += 1
                continue
            levels = self.book.bids if side == 1 else self.book.asks
            boundary = self.book.bid_floor if side == 1 else self.book.ask_ceiling
            if boundary is None:
                boundary = min(levels) if side == 1 else max(levels)
            if (side == 1 and price < boundary) or (side == -1 and price > boundary):
                self.stats["unknown_queue_holds"] += 1
                continue
            self.quotes.append(
                Quote(side, price, self.risk.size, levels.get(price, 0.0), self.exchange_bound)
            )
            self.stats["submitted"] += 1

    def event(self, event):
        """Apply one arrival while enforcing continuity and exchange-time fill eligibility."""
        ns = event["ns"]
        session_changed = self.book.session is not None and event["session"] != self.book.session
        if session_changed:
            self.quotes, self.pending = [], None
            self.features = Features()
            self.exchange_bound = None
        # Arrivals see the previously observed book and exchange-time bound.
        self.advance(ns)
        self.book.apply(event)
        if self.book.transaction_ns is not None:
            self.exchange_bound = max(self.exchange_bound or 0, self.book.transaction_ns)
        fresh = self.book.fresh(ns, self.venue)
        if not fresh:
            self.quotes = []  # hard risk hold: no simulated exposure on an unknown book
        if fresh:
            if self.previous_mid is not None:
                self.inventory_pnl += self.inventory * (self.book.mid - self.previous_mid)
            self.previous_mid = self.book.mid
        ts = []
        for tid, p, q, aggressor, transaction_ns in trades(event, self.venue):
            if tid in self.seen:
                continue
            self.seen[tid] = None
            if len(self.seen) > 10000:
                self.seen.pop(next(iter(self.seen)))
            ts.append((p, q, aggressor))
            for quote in self.quotes if fresh else []:
                # Compare exchange clocks only. Equality and missing bounds cannot
                # establish ordering, so neither consumes queue nor produces fills.
                if transaction_ns is None or quote.joined_exchange_ns is None:
                    self.stats["unknown_trade_order"] += 1
                    continue
                if transaction_ns <= quote.joined_exchange_ns:
                    self.stats["pre_entry_trades"] += 1
                    continue
                qty = quote.trade(p, q, aggressor, self.pessimistic)
                if qty:
                    fee = qty * quote.price * self.risk.maker_fee_bp / 1e4
                    self.inventory += quote.side * qty
                    self.cash -= quote.side * qty * quote.price + fee
                    self.fees += fee
                    self.spread_capture += quote.side * qty * (self.book.mid - quote.price)
                    self.fills.append(
                        {
                            "ns": ns,
                            "side": quote.side,
                            "price": quote.price,
                            "size": qty,
                            "mid": self.book.mid,
                            "fee": fee,
                            "session": self.book.session,
                            "trade_exchange_ns": transaction_ns,
                            "joined_exchange_ns": quote.joined_exchange_ns,
                        }
                    )
            if transaction_ns is not None:
                self.exchange_bound = max(self.exchange_bound or 0, transaction_ns)
            self.quotes = [quote for quote in self.quotes if quote.remaining > 1e-12]
        return self.features.observe(self.book, ns, ts)

    def decide(self, ns, x, p_toxic, latency_ms, abstention=0.0):
        """Schedule a latency-delayed quote or veto while preserving pending exposure."""
        if self.pending is not None:
            self.stats["busy_holds"] += 1
            return
        if not self.book.fresh(ns, self.venue):
            self.stats["stale_holds"] += 1
            return
        self.stats["decisions"] += 1
        quote = p_toxic is not None and p_toxic < 0.5 and max(p_toxic, 1 - p_toxic) >= abstention
        bid, ask = prices(self.book, self.inventory, x[4], self.risk) if quote else (None, None)
        self.pending = (ns + int(latency_ms * 1e6), ns, bid, ask)

    def mark(self, ns):
        """Record marked equity at the current observed mid."""
        if self.book.valid:
            self.equity.append({"ns": ns, "pnl": self.cash + self.inventory * self.book.mid})

    def summary(self):
        """Reconcile cash, inventory, fees and execution diagnostics."""
        mid = self.book.mid if self.book.valid else self.previous_mid
        pnl = self.cash + self.inventory * mid if mid is not None else None
        # Flatten is hypothetical, charged taker fee; inventory is not actually traded.
        flatten_fee = abs(self.inventory * mid) * self.risk.taker_fee_bp / 1e4 if mid else 0.0
        return {
            **self.stats,
            "risk": asdict(self.risk),
            "pnl_usd": pnl,
            "flatten_fee_usd": flatten_fee,
            "pnl_after_flatten_fee_usd": pnl - flatten_fee if pnl is not None else None,
            "spread_capture_usd": self.spread_capture,
            "inventory_usd": self.inventory_pnl,
            "fees_usd": self.fees,
            "inventory_base": self.inventory,
            "fill_events": len(self.fills),
            "fill_rate": (
                sum(f["size"] for f in self.fills) / (self.stats["submitted"] * self.risk.size)
                if self.stats["submitted"]
                else 0.0
            ),
        }
