import pytest

from judgment_gates.diagnostics import new_engine, observe, summarize


def decision(**updates):
    return {
        "type": "decision",
        "ns": 1,
        "cash": 0,
        "inventory": 0,
        "mid": 100,
        "pnl": 0,
        "p_toxic": 0.4,
        "quotes": [],
        **updates,
    }


def test_fill_accounts_reconcile_and_exclude_trailing_fills():
    engine = new_engine()
    fill = {"type": "fill", "side": 1, "price": 99, "size": 1, "fee": 0.02, "mid": 100}
    observe(engine, fill, 0)
    observe(engine, decision(cash=-99.02, inventory=1, mid=98, pnl=-1.02), 0)
    observe(engine, fill, 0)
    result = summarize({"binance/BTC/rules/pessimistic": engine})["rules/pessimistic"]
    assert result["fill_events"] == result["unmarked_fill_events"] == 1
    assert result["spread_capture_usd"] == 1
    assert result["fees_usd"] == 0.02
    assert result["inventory_residual_usd"] == pytest.approx(-2)
    assert result["pnl_usd"] == -1.02


def test_inactive_abstention_is_separate_from_missing_market():
    idle = new_engine()
    observe(idle, decision(), 1.01)
    result = summarize({"binance/BTC/classifier+abstain/pessimistic": idle})
    row = result["classifier+abstain/pessimistic"]
    assert row["pnl_usd"] == row["fill_events"] == row["confidence_accepted"] == 0
    assert row["spread_bp_per_filled_notional"] is None
    assert (
        summarize({"binance/BTC/rules/pessimistic": new_engine()})["rules/pessimistic"]["pnl_usd"]
        is None
    )


def test_unexplained_cash_movement_is_rejected():
    with pytest.raises(ValueError, match="cash mismatch"):
        observe(new_engine(), decision(cash=1, pnl=1), 0)
