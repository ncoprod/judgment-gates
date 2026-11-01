import json
import subprocess

import pytest

from judgment_gates.final import evaluate
from judgment_gates.ledger import Ledger
from judgment_gates.receipts import digest, write_json
from judgment_gates.registration import engine_names

S = 1_000_000_000


@pytest.fixture
def cohort(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    policy = {
        "tag": "frozen",
        "test_start": "2020-01-01T00:00:00+00:00",
        "test_end": "2020-01-01T00:01:00+00:00",
        "cadence_s": 1,
        "conservative_trial_count": 100,
        "markets": {
            "binance/BTC": {
                "gates": {"rules": {}, "classifier": {}},
                "risk": {"capital": 10000, "taker_fee_bp": 5},
            }
        },
    }
    path = root / "policy.json"
    write_json(path, policy)

    def git(*args):
        return (
            subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

    git("init", "-q")
    git("add", "policy.json")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "fixture")
    git("tag", "frozen")
    directory = root / "cohort"
    directory.mkdir()
    ledger = Ledger(directory / "ledger.jsonl")
    t = 1577836800 * S
    ledger.append(
        {
            "type": "start",
            "ns": t - S,
            "policy_sha256": digest(path),
            "frozen_commit": git("rev-parse", "HEAD"),
        }
    )

    def decision(name, ns, session="s"):
        return {
            "type": "decision",
            "ns": ns,
            "engine": name,
            "session": session,
            "features": [1, 0, 0, 0, 0],
            "p_toxic": 0.8,
            "mid": 100,
            "pnl": 0,
            "inventory": 0.001,
            "latency_ms": 0.01,
        }

    def checkpoint():
        write_json(
            directory / "state.json",
            {
                "updated_ns": t + 123 * S,
                "policy_sha256": digest(path),
                "ledger_rows": ledger.count,
                "ledger_hash": ledger.previous,
            },
        )

    return path, directory, ledger, t, engine_names(policy), decision, checkpoint


def test_wrong_policy_missing_ledger_and_prefix_truncation_fail_closed(cohort):
    path, directory, ledger, t, names, decision, checkpoint = cohort
    checkpoint()
    with pytest.raises(FileNotFoundError):
        evaluate(path, directory / "absent", directory / "missing.json")
    ledger.append({"type": "start", "ns": t, "policy_sha256": "WRONG", "frozen_commit": "WRONG"})
    checkpoint()
    with pytest.raises(ValueError, match="policy/code"):
        evaluate(path, directory, directory / "wrong.json")
    ledger.path.write_text(ledger.path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="truncated"):
        evaluate(path, directory, directory / "truncated.json")


def test_reservation_is_cohort_scoped_and_missing_engines_are_explicit(cohort):
    path, directory, ledger, t, names, decision, checkpoint = cohort
    checkpoint()
    result = evaluate(path, directory, directory / "first.json")
    assert result["status"] == "insufficient_data"
    assert set(result["engines"]) == set(names)
    assert all(e["coverage"]["hours"] for e in result["engines"].values())
    with pytest.raises(FileExistsError):
        evaluate(path, directory, directory / "alternate.json")


@pytest.mark.parametrize("interruption", ["gap", "start", "session"])
def test_short_outages_censor_labels_and_markouts(cohort, interruption):
    path, directory, ledger, t, names, decision, checkpoint = cohort
    name = names[0]
    ledger.append(decision(name, t + S))
    ledger.append(
        {
            "type": "fill",
            "engine": name,
            "ns": t + S,
            "session": "s",
            "fee": 0.01,
            "side": 1,
            "size": 0.001,
            "mid": 100,
            "price": 99,
        }
    )
    if interruption == "gap":
        ledger.append({"type": "gap", "market": "binance/BTC", "ns": t + 5 * S})
    if interruption == "start":
        ledger.append(
            {**json.loads(ledger.path.read_text().splitlines()[0])["event"], "ns": t + 5 * S}
        )
    ledger.append(decision(name, t + 12 * S, "new" if interruption == "session" else "s"))
    checkpoint()
    e = evaluate(path, directory, directory / "out.json")["engines"][name]
    assert e["brier_hour"]["n"] == 0
    assert e["label_censoring"]["gap_or_restart"] == 1
    assert e["markouts_bp"][10]["censoring"]["gap_or_restart"] == 1
    assert e["markouts_bp"][10]["hour"]["n"] == 0


def test_full_coverage_metrics_and_primary_family(cohort):
    path, directory, ledger, t, names, decision, checkpoint = cohort
    for name in names:
        ledger.append(decision(name, t - S))
    for seconds in range(123):
        for name in names:
            ledger.append(decision(name, t + seconds * S))
    checkpoint()
    result = evaluate(path, directory, directory / "out.json")
    assert result["status"] == "ok"
    e = result["engines"][names[0]]
    assert e["brier_hour"]["n"] == 60
    assert e["latency"]["p50_ms"] == 0.01
    assert e["flatten_fee_usd"] == pytest.approx(0.00005)
    assert e["performance"]["dsr"] is None
    for name, paired in result["paired_markout"].items():
        assert paired["family"] == (
            "primary" if name.endswith("pessimistic") else "secondary_descriptive"
        )
        assert paired["hour"]["block"] == "hour"
        assert paired["day"]["block"] == "day"
        assert paired["coverage_eligible"] is False


def export_module():
    """Load the standalone publication script for boundary tests."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_public", "scripts/export_public.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_export_contains_paper_and_only_aggregate_results(tmp_path):
    from pathlib import Path

    if not Path("results/review-offline.json").exists():
        pytest.skip("publication integration requires the private source checkout")
    module = export_module()
    target = tmp_path / "export"
    manifest = module.export(target)
    names = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    assert {n for n in names if n.startswith("results/")} == set(module.RESULTS)
    assert {"paper/main.pdf", "paper/main.tex", "paper/refs.bib"} <= names
    assert not any(n.startswith(("configs/", ".git/", "runs/")) for n in names)
    assert not any(n.endswith("-calls.json") for n in names)
    assert "REVIEW_ASTRA.md" not in names
    assert "LICENSE" in names and "src/judgment_gates/engine.py" in manifest["files"]
    assert not (target / ".git").exists()
    assert manifest["checked_local_links"] > 0
    forbidden = {
        "raw_response",
        "request",
        "calls",
        "samples_ms",
        "states",
        "features",
        "fills",
        "equity",
        "coefficients",
        "git_status",
    }

    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            assert "/Users/" not in value

    for name in module.RESULTS:
        check(json.loads((target / name).read_text()))
    with pytest.raises(FileExistsError):
        module.export(target)


def test_public_projection_drops_unrecognised_and_individual_fields():
    module = export_module()
    row = {
        "status": "ok",
        "raw_response": "private output",
        "future_field": "private",
        "latency": {"p50_ms": 12, "samples_ms": [10, 14]},
        "tilt": {
            "drawdown": {
                "signed_probability_shift": {
                    "mean": 0.1,
                    "observations": [0.1],
                    "paired_state_bootstrap": {"ci": [0, 0.2], "states": ["private"]},
                }
            }
        },
    }
    result = module.model_summary(row)
    assert result["latency"] == {"p50_ms": 12}
    assert "raw_response" not in result and "future_field" not in result
    assert result["tilt"]["drawdown"]["signed_probability_shift"] == {
        "mean": 0.1,
        "paired_state_bootstrap": {"ci": [0, 0.2]},
    }


def test_public_links_must_exist_and_stay_inside_export(tmp_path):
    module = export_module()
    root = tmp_path / "export"
    root.mkdir()
    readme = root / "README.md"
    readme.write_text("[missing](paper/main.pdf)")
    with pytest.raises(ValueError):
        module.check_links(root)
    (root / "paper").mkdir()
    (root / "paper/main.pdf").write_bytes(b"fixture")
    assert module.check_links(root) == 1
    outside = tmp_path / "private.txt"
    outside.write_text("fixture")
    readme.write_text("[escape](../private.txt)")
    with pytest.raises(ValueError):
        module.check_links(root)
    readme.write_text("[external](https://example.org/terms)")
    (root / "paper/main.tex").write_text(r"\input{generated/missing}")
    with pytest.raises(ValueError):
        module.check_links(root)
