import json

import pytest

from judgment_gates.forward import Paper, score
from judgment_gates.ledger import Ledger


@pytest.fixture(autouse=True)
def isolate_receipts(monkeypatch):
    monkeypatch.setattr("judgment_gates.forward.provenance", lambda *a, **k: {})
    monkeypatch.setattr(
        "judgment_gates.forward.registered_policy",
        lambda path, **kwargs: (json.loads(path.read_text()), "abc"),
    )


def policy(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "paper_only": True,
                "cadence_s": 1,
                "markets": {"hyperliquid/ETH": {"risk": {}, "gates": {"rules": {"threshold": 0}}}},
            }
        )
    )
    return path


def test_checkpoint_tail_recovery_and_scoring(tmp_path, monkeypatch):
    path = policy(tmp_path)
    directory = tmp_path / "forward"
    paper = Paper(path, directory)
    paper.emit(
        "hyperliquid/ETH",
        "l2Book",
        {"levels": [[{"px": "99", "sz": "2"}], [{"px": "101", "sz": "3"}]]},
        "s",
    )
    paper.checkpoint()
    paper.ledger.append(
        {
            "type": "fill",
            "engine": "hyperliquid/ETH/rules/optimistic",
            "ns": 1,
            "side": 1,
            "price": 99,
            "size": 0.001,
            "fee": 0.0001,
        }
    )
    paper.lock.close()
    recovered = Paper(path, directory)
    engine = recovered.engines["hyperliquid/ETH/rules/optimistic"]
    assert engine.inventory == 0.001
    assert engine.cash == pytest.approx(-0.0991)
    assert not engine.quotes
    result = score(directory, tmp_path / "score.json")
    assert result["ledger_rows"] == recovered.ledger.count
    again = score(directory, tmp_path / "score.json")
    assert again["ledger_rows"] == result["ledger_rows"]
    assert again["days"] == result["days"]
    recovered.lock.close()


def test_checkpoint_detects_truncation(tmp_path, monkeypatch):
    path = policy(tmp_path)
    directory = tmp_path / "forward"
    paper = Paper(path, directory)
    paper.checkpoint()
    paper.lock.close()
    (directory / "ledger.jsonl").write_text("")
    with pytest.raises(ValueError, match="anchor"):
        Paper(path, directory)


def test_single_writer_lock_and_changed_policy(tmp_path, monkeypatch):
    path = policy(tmp_path)
    directory = tmp_path / "forward"
    paper = Paper(path, directory)
    with pytest.raises(BlockingIOError):
        Paper(path, directory)
    paper.lock.close()
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="policy changed"):
        Paper(path, directory)


def test_scorer_ignores_live_uncheckpointed_tail(tmp_path):
    directory = tmp_path / "forward"
    directory.mkdir()
    ledger = Ledger(directory / "ledger.jsonl")
    ledger.append({"type": "start", "ns": 1})
    (directory / "state.json").write_text(
        json.dumps({"ledger_hash": ledger.previous, "ledger_rows": ledger.count, "updated_ns": 1})
    )
    with ledger.path.open("a") as f:
        f.write('{"incomplete":')
    assert score(directory, tmp_path / "score.json")["ledger_rows"] == 1
