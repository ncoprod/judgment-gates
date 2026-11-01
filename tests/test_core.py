import json

import numpy as np
import pytest

from judgment_gates.data import Book, chronological_splits, label_states, states
from judgment_gates.engine import Quote, Replay, Risk
from judgment_gates.ledger import Ledger, compress_segment, verified_rows, verify
from judgment_gates.metrics import block_ci, calibration, performance, risk_intervals
from judgment_gates.providers import parse_text, request_timing, state_payload
from judgment_gates.remote_study import minute_indices, regime_mask


def event(ns, stream="l2Book", data=None):
    return {
        "ns": ns,
        "session": "s",
        "stream": stream,
        "applied": True,
        "data": data or {"levels": [[{"px": "99", "sz": "2"}], [{"px": "101", "sz": "3"}]]},
    }


def test_queue_strictly_exceeds_ahead_and_partial_fill():
    quote = Quote(1, 99, 1, 2)
    assert quote.trade(99, 2, -1, False) == 0
    assert quote.trade(99, 0.4, -1, False) == pytest.approx(0.4)
    assert quote.trade(99, 10, -1, False) == pytest.approx(0.6)
    assert quote.remaining == 0


def test_pessimistic_requires_trade_through_and_correct_aggressor():
    quote = Quote(-1, 101, 1, 2)
    assert quote.trade(101, 99, 1, True) == 0
    assert quote.trade(102, 99, -1, True) == 0
    assert quote.trade(102, 3, 1, True) == 1


def test_cancel_replace_delay_old_quote_exposed():
    replay = Replay("hyperliquid")
    replay.event(event(1_000_000_000))
    replay.quotes = [Quote(1, 99, 0.001, 0, 0)]
    replay.decide(1_000_000_000, [2, 0, 0, 0, 0], 0.9, 500)
    replay.event(
        event(
            1_200_000_000, "trades", [{"tid": 1, "px": "98", "sz": "1", "side": "A", "time": 1200}]
        )
    )
    assert len(replay.fills) == 1
    replay.advance(1_500_000_000)
    assert replay.quotes == []


def test_post_only_checked_on_arrival_inventory_limit_and_stale_hold():
    replay = Replay("hyperliquid", Risk(size=1, inventory_limit=1))
    replay.event(event(1_000_000_000))
    replay.pending = (2_000_000_000, 1_000_000_000, 102, 98)
    replay.advance(2_000_000_000)
    assert replay.stats["rejected_post_only"] == 2
    replay.inventory = 1
    replay.pending = (3_000_000_000, 1_000_000_000, 99, 101)
    replay.advance(3_000_000_000)
    assert [q.side for q in replay.quotes] == [-1]
    replay.pending = (30_000_000_000, 1_000_000_000, 99, 101)
    replay.advance(30_000_000_000)
    assert replay.quotes == []


def test_prefix_features_do_not_depend_on_future():
    prefix = [event(i * 1_000_000_000) for i in range(1, 10)]
    changed = event(
        10_000_000_000, data={"levels": [[{"px": "900", "sz": "100"}], [{"px": "901", "sz": "1"}]]}
    )
    assert (
        list(states(iter(prefix), "hyperliquid"))
        == list(states(iter(prefix + [changed]), "hyperliquid"))[:-1]
    )


def test_labels_and_split_purge():
    rows = [
        {"ns": i * 1_000_000_000, "mid": 100 + i / 1000, "x": [1, 0, 0, 0, 0], "session": "s"}
        for i in range(200)
    ]
    labelled = label_states(rows)
    train, val, test = chronological_splits(labelled)
    assert max(labelled[i]["label_end_ns"] for i in train) < min(labelled[i]["ns"] for i in val)
    assert max(labelled[i]["label_end_ns"] for i in val) < min(labelled[i]["ns"] for i in test)
    assert all(r["label_end_ns"] >= r["ns"] + 10_000_000_000 for r in labelled)


def test_broken_chain_fails_closed():
    book = Book()
    book.apply(
        event(
            1,
            "checkpoint",
            {"bids": [["99", "1"]], "asks": [["101", "1"]], "lastUpdateId": 5, "bridged": True},
        )
    )
    book.apply(event(2, "depthUpdate", {"pu": 4, "U": 6, "u": 7, "b": [], "a": []}))
    assert not book.valid


def test_ledger_resume_tamper_and_partial_tail(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append({"quote": 1})
    Ledger(path).append({"fill": 2})
    assert verify(path)[1] == 2
    original = path.read_text()
    path.write_text(original.replace('"fill":2', '"fill":3'))
    with pytest.raises(ValueError, match="corruption"):
        verify(path)
    path.write_text(original + '{"event":')
    with pytest.raises(json.JSONDecodeError):
        Ledger(path)


def test_calibration_exact_and_probability_one_bin():
    result = calibration([0, 1], [0, 1])
    assert result["brier"] == result["ece"] == 0
    assert result["reliability"][-1]["n"] == 1
    assert calibration([0, 1], [0.5, 0.5])["brier"] == 0.25
    with pytest.raises(ValueError):
        calibration([0], [np.nan])


def test_bootstrap_refuses_one_hour_pseudoreplication():
    result = block_ci([0, 1, 2], [1, 2, 3])
    assert result["ci"] is None and result["blocks"] == 1
    result = block_ci([1] * 6, [i * 3_600_000_000_000 for i in range(6)])
    assert result["ci"] == [1.0, 1.0]


def test_typed_answer_and_market_state_invariance():
    assert parse_text('{"p_toxic":0.2,"regime":"calm"}')["p_toxic"] == 0.2
    with pytest.raises(ValueError):
        parse_text('{"p_toxic":true,"regime":"calm"}')
    row = {"x": [1, 0.2, 0.3, -1, 0.1]}
    assert (
        state_payload(row, "loss")["market_features"]
        == state_payload(row, "win")["market_features"]
    )


def test_rotated_compressed_ledger_has_same_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"quote": 1})
    segment = ledger.rotate("2026-10-29")
    ledger.append({"fill": 2})
    before = verify(path)
    compress_segment(segment)
    assert verify(path) == before == (ledger.previous, 2)
    assert Ledger(path).count == 2


def test_staleness_uses_local_elapsed_time_not_unsynchronised_exchange_clock():
    book = Book()
    data = event(1_000_000_000)["data"]
    data["time"] = 6500
    book.apply(event(1_000_000_000, data=data))
    assert book.fresh(1_500_000_000, "binance")
    assert not book.fresh(4_000_000_000, "binance")


def test_slow_regime_only_applies_after_answer_arrives():
    rows = [{"ns": i * 1_000_000_000} for i in range(4)]
    p, calls = regime_mask(rows, [0.1] * 4, [1500] * 4, [0.2] * 4)
    assert p.tolist() == [1.0, 1.0, 0.2, 0.2]
    assert calls == 1


def test_minute_schedule_skips_busy_calls_without_unscheduled_predictions():
    rows = [{"ns": i * 1_000_000_000} for i in range(181)]
    selected = minute_indices(rows, list(range(len(rows))))
    assert selected == [0, 60, 120, 180]
    p, calls = regime_mask(rows, [0.1] * 181, [70000] * 181, [0.2] * 181, set(selected))
    assert calls == 2  # The scheduled calls at 60 and 180 seconds are still busy.
    assert p[69] == 1
    assert p[70] == 0.2
    assert p[121] == 1  # The old answer expires while the second answer is pending.


def test_risk_ratios_do_not_compress_missing_hours():
    equity = [{"ns": i * 3_600_000_000_000, "pnl": float(i + i % 3)} for i in range(25) if i != 12]
    score = performance(equity, 10000, 10)
    assert score["missing_hours"] == 1
    assert score["sharpe"] is None and score["sortino"] is None and score["dsr"] is None
    assert score["risk_intervals"] is None


@pytest.mark.parametrize("wall_ms,expected", [(500000, 500000), (-500000, 2000)])
def test_request_latency_accounts_for_pause_and_clock_change(monkeypatch, wall_ms, expected):
    monkeypatch.setattr("judgment_gates.providers.time.time_ns", lambda: int(wall_ms * 1e6))
    monkeypatch.setattr("judgment_gates.providers.time.perf_counter_ns", lambda: 2_000_000_000)
    timing = request_timing(0, 0)
    assert timing["latency_ms"] == expected
    assert timing["host_pause_or_clock_change"]
    assert timing["wall_elapsed_ms"] == wall_ms


def test_day_bootstrap_preserves_zero_daily_return_and_reports_undefined_ratios():
    hours = np.arange(120)
    balanced = np.tile([-0.01, 0.01], 60)
    intervals = risk_intervals(balanced, hours, 10000, "day", repetitions=100)
    assert intervals["pnl_usd"] == [0, 0]
    assert intervals["sharpe"] == [0, 0]
    positive = risk_intervals(np.ones(120), hours, 10000, "day", repetitions=100)
    assert positive["sortino"] is None
    assert positive["undefined_resamples"]["sortino"] == 100


def test_inventory_decomposition_charges_only_inventory_held_before_price_move():
    replay = Replay("hyperliquid", Risk(maker_fee_bp=0))
    replay.event(event(1_000_000_000))
    replay.quotes = [Quote(1, 99, 0.001, 0, 0)]
    replay.event(
        event(
            2_000_000_000, data={"levels": [[{"px": "100", "sz": "2"}], [{"px": "102", "sz": "3"}]]}
        )
    )
    replay.event(
        event(
            3_000_000_000, "trades", [{"tid": 2, "px": "98", "sz": "1", "side": "A", "time": 1200}]
        )
    )
    replay.event(
        event(
            4_000_000_000, data={"levels": [[{"px": "101", "sz": "2"}], [{"px": "103", "sz": "3"}]]}
        )
    )
    result = replay.summary()
    assert result["pnl_usd"] == pytest.approx(
        result["spread_capture_usd"] + result["inventory_usd"] - result["fees_usd"]
    )


def test_checkpoint_resume_verifies_containing_segment_and_tail(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"old": 1})
    compress_segment(ledger.rotate("2026-10-28"))
    ledger.append({"current": 2})
    anchor = ledger.count, ledger.previous
    ledger.append({"tail": 3})
    assert [r["event"] for r in verified_rows(ledger.path, anchor=anchor)] == [{"tail": 3}]
    assert verify(ledger.path, anchor) == verify(ledger.path)
    ledger.path.write_text(ledger.path.read_text().replace('"current":2', '"current":9'))
    with pytest.raises(ValueError, match="corruption|anchor"):
        verify(ledger.path, anchor)


def test_decimal_tick_prices_join_displayed_queue_exactly():
    replay = Replay("hyperliquid", Risk(size=1, inventory_limit=2, tick=0.1))
    x = replay.event(
        event(
            1_000_000_000,
            data={"levels": [[{"px": "99.1", "sz": "2"}], [{"px": "101.1", "sz": "3"}]]},
        )
    )
    replay.decide(1_000_000_000, x, 0.1, 0)
    replay.advance(1_000_000_000)
    assert [(q.price, q.ahead) for q in replay.quotes] == [(99.1, 2), (101.1, 3)]
    replay.event(
        event(1_100_000_000, "trades", [{"tid": 1, "px": "101.1", "sz": "3", "side": "B"}])
    )
    assert replay.fills == []


def test_quote_beyond_finite_known_depth_is_held():
    replay = Replay("hyperliquid")
    replay.event(event(1_000_000_000))
    replay.pending = (1_000_000_000, 1_000_000_000, 98, 102)
    replay.advance(1_000_000_000)
    assert replay.quotes == []
    assert replay.stats["unknown_queue_holds"] == 2


@pytest.mark.parametrize("stamp", [900, 1000, None])
def test_historical_or_ambiguous_trade_cannot_consume_new_queue(stamp):
    replay = Replay("binance", pessimistic=False)
    book = {
        "bids": [["99", "2"], ["98", "2"]],
        "asks": [["101", "3"], ["102", "3"]],
        "lastUpdateId": 5,
        "bridged": True,
        "T": 1000,
    }
    replay.event(event(1_000_000_000, "checkpoint", book))
    replay.pending = (1_050_000_000, 1_000_000_000, 99, None)
    trade = {"a": 1, "p": "99", "q": "3", "m": True}
    if stamp is not None:
        trade["T"] = stamp
    replay.event(event(1_100_000_000, "aggTrade", trade))
    assert replay.fills == []
    assert replay.quotes[0].ahead == 2
    replay.event(event(1_200_000_000, "aggTrade", {**trade, "a": 2, "T": 1001}))
    assert len(replay.fills) == 1
    assert replay.fills[0]["joined_exchange_ns"] == 1_000_000_000


def test_reconnect_discards_quotes_before_delayed_cross_stream_trade():
    replay = Replay("hyperliquid", pessimistic=False)
    replay.event(event(1_000_000_000))
    replay.quotes = [Quote(1, 99, 0.001, 0, 0)]
    trade = event(
        1_100_000_000, "trades", [{"tid": 1, "px": "98", "sz": "1", "side": "A", "time": 900}]
    )
    trade["session"] = "reconnected"
    replay.event(trade)
    assert replay.quotes == replay.fills == []


def test_retained_variant_dispersion_never_becomes_canonical_dsr():
    rng = np.random.default_rng(42)
    equity = [
        {"ns": i * 3_600_000_000_000, "pnl": p}
        for i, p in enumerate(np.cumsum(rng.normal(1, 2, 30)))
    ]
    result = performance(equity, 10000, 100, [-0.1, 0.1])
    assert result["dsr"] is None
    assert result["dsr_sensitivity_retained_variants"] is not None


def test_quote_bound_includes_previously_observed_other_stream_trades():
    replay = Replay("hyperliquid", pessimistic=False)
    replay.event(
        event(
            1_000_000_000,
            data={"time": 1000, "levels": [[{"px": "99", "sz": "2"}], [{"px": "101", "sz": "3"}]]},
        )
    )
    replay.event(
        event(
            1_010_000_000, "trades", [{"tid": 1, "px": "99", "sz": "1", "side": "A", "time": 1050}]
        )
    )
    replay.pending = (1_020_000_000, 1_000_000_000, 99, None)
    replay.event(
        event(
            1_100_000_000, "trades", [{"tid": 2, "px": "99", "sz": "3", "side": "A", "time": 1025}]
        )
    )
    assert replay.fills == []
    assert replay.quotes[0].joined_exchange_ns == 1_050_000_000
    assert replay.quotes[0].ahead == 2


def test_paired_state_bootstrap_keeps_pair_units_and_withholds_singleton_ci():
    from judgment_gates.metrics import paired_state_ci
    from judgment_gates.study import tilt_interval

    summary = tilt_interval([-1, 1], [1, 2])
    assert summary["ci"] is None  # A state bootstrap must not impersonate a UTC-block CI.
    paired = summary["paired_state_bootstrap"]
    assert paired["ci"] == [-1, 1]
    assert paired["unit"] == "paired market state"
    assert paired == paired_state_ci([-1, 1])
    assert paired_state_ci([0.2])["ci"] is None
