"""Export committed code, the paper and explicitly projected aggregate measurements."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src/", "tests/", "paper/generated/")
FILES = {
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "RELEASE_POLICY.md",
    "THIRD_PARTY_NOTICES.md",
    "scripts/export_public.py",
    "scripts/daily_score.py",
    "paper/main.tex",
    "paper/main.pdf",
    "paper/refs.bib",
    "paper/README.md",
}
RESULTS = (
    "results/research-summary.json",
    "results/forward-diagnostics.json",
    "results/framing-summary.json",
    "results/licenses.json",
    "results/tilt-model-licenses.json",
)
LATENCY = (
    "n",
    "attempted",
    "failed",
    "logical_successful_rows",
    "p50_ms",
    "p99_ms",
    "cost_per_million_usd",
)
INTERVAL = ("mean", "n", "block", "blocks", "ci", "reason")
BOOTSTRAP = ("n", "ci", "alpha", "repetitions", "seed", "unit", "scope", "reason")
REPLAY = (
    "decisions",
    "fill_events",
    "fill_rate",
    "spread_capture_usd",
    "inventory_usd",
    "fees_usd",
    "pnl_usd",
    "pnl_after_flatten_fee_usd",
    "flatten_fee_usd",
    "decomposition_residual_usd",
    "stale_holds",
    "busy_holds",
    "rejected_post_only",
    "unknown_queue_holds",
    "pre_entry_trades",
    "unknown_trade_order",
)
FORWARD = (
    "confidence_accepted",
    "decisions",
    "fee_bp_per_filled_notional",
    "fees_usd",
    "fill_events",
    "inventory_residual_usd",
    "markets_expected",
    "markets_marked",
    "notional_usd",
    "pnl_usd",
    "quotes_present",
    "spread_bp_per_filled_notional",
    "spread_capture_usd",
    "submitted_orders",
    "unmarked_fill_events",
)


def pick(row, fields):
    """Project named fields; unknown future metadata never enters a publication."""
    return {name: row[name] for name in fields if name in row}


def calibration(row):
    """Keep computed calibration statistics and aggregate reliability bins only."""
    return {
        **pick(row, ("n", "brier", "ece", "prevalence")),
        "reliability": [
            pick(b, ("n", "lower", "upper", "probability", "frequency"))
            for b in row.get("reliability", [])
        ],
    }


def interval(row):
    """Keep interval estimates and their statistical scope, without observations."""
    result = pick(row, INTERVAL)
    if "paired_state_bootstrap" in row:
        result["paired_state_bootstrap"] = pick(row["paired_state_bootstrap"], BOOTSTRAP)
    return result


def model_summary(row):
    """Exclude individual timings, prompts, outputs, errors and local provenance."""
    result = {
        **pick(row, ("status", "neutral_n")),
        "latency": pick(row.get("latency", {}), LATENCY),
    }
    if "calibration" in row:
        result["calibration"] = calibration(row["calibration"])
    result["tilt"] = {
        context: {
            metric: interval(values[metric])
            for metric in (
                "signed_probability_shift",
                "absolute_probability_shift",
                "decision_flip_fraction",
            )
            if metric in values
        }
        for context, values in row.get("tilt", {}).items()
        if context in ("drawdown", "loss", "win", "morning", "night", "neutral_repeat")
    }
    return result


def publication_results(read, commit):
    """Construct five statistical publications from a fixed private source commit."""
    inputs = {}

    def source(name):
        raw = read(name)
        inputs[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    offline = source("results/review-offline.json")
    pilot = source("results/review-pilot.json")
    cohort = source("results/expanded-cohort.json")
    summary = {
        "markets": {},
        "remote_latency": {
            name: {"failed": row["failed"], "latency": pick(row["latency"], LATENCY)}
            for name, row in offline["remote"].items()
        },
    }
    for name, row in pilot["markets"].items():
        summary["markets"][name] = {
            "labels": cohort["markets"][name]["n"],
            "splits": pick(row["splits"], ("train", "validation", "pilot")),
            "isotonic_labels": row["model"]["calibration_n"],
            "local": {
                gate: {
                    "calibration": calibration(values["calibration"]),
                    "latency": pick(values["latency"], LATENCY),
                }
                for gate, values in offline["markets"][name].items()
            },
            "classifier_minus_rules_brier": interval(row["classifier_minus_rules_brier"]),
            "replay": {
                gate: {
                    scenario: {
                        **pick(values, REPLAY),
                        "markouts_bp": {
                            horizon: {
                                **pick(blocks, ("censored", "signed_future_move_usd")),
                                **{block: interval(blocks[block]) for block in ("hour", "day")},
                            }
                            for horizon, blocks in values["markouts_bp"].items()
                        },
                    }
                    for scenario, values in outcome["replay"].items()
                }
                for gate, outcome in row["outcomes"].items()
            },
        }
    summary["book_verification"] = pick(
        source("results/expanded-verify.json"), ("compared", "error_count", "passed")
    )
    summary["code_verification"] = pick(
        source("results/verification.json"), ("passed", "tests_passed")
    )
    deadlines = source("results/deadlines.json")
    summary["deadlines"] = {
        group: {
            name: {
                deadline: pick(row, (*INTERVAL, "attempted", "met"))
                for deadline, row in values.items()
            }
            for name, values in deadlines[group].items()
        }
        for group in ("local", "remote")
    }
    summary["markout_precision"] = pick(
        source("results/evidence-status.json")["markout_precision"],
        (
            "cases",
            "max_day_blocks",
            "min_day_blocks",
            "required_test_days",
            "scope",
            "supported_day_CIs",
            "target_half_width_bp",
            "target_status",
            "test_half_width_bp",
            "reason",
        ),
    )
    probe = source("results/jev-evaluate-probe.json")
    summary["jev_preflight"] = {
        **pick(probe, ("model", "route", "status", "latency_ms")),
        "http_status": probe["attempts"][-1]["http_status"],
    }
    forward = source("results/forward-diagnostics.json")
    forward = {
        **pick(
            forward,
            (
                "checkpoint_ns",
                "ledger_hash",
                "ledger_rows",
                "paper_only",
                "scope",
                "activity_note",
                "accounting_note",
            ),
        ),
        "variants": {name: pick(row, FORWARD) for name, row in forward["variants"].items()},
    }
    tilt = {
        name: {
            "models": {
                model: model_summary(row)
                for model, row in source(f"results/{name}.json")["models"].items()
            }
        }
        for name in ("opencode-analysis", "nim-low-analysis", "nim-tilt-analysis")
    }
    licences = {}
    for name in ("licenses", "tilt-model-licenses"):
        raw = source(f"results/{name}.json")
        licences[f"results/{name}.json"] = {
            group: {
                model: pick(
                    row,
                    (
                        "id",
                        "licence",
                        "licence_name",
                        "licence_url",
                        "source",
                        "revision",
                        "licence_sha256",
                    ),
                )
                for model, row in models.items()
            }
            for group, models in raw.items()
            if group in ("models", "dependencies")
        }
    provenance = {
        "source_commit": commit,
        "source_sha256": inputs,
        "command": "uv run python scripts/export_public.py TARGET",
        "seed": 42,
        "model_revision": "frozen local fits; hosted aliases not immutable",
        "scope": "author-computed aggregates; no observations, responses or personal metadata",
    }
    return {
        name: {"provenance": provenance, **values}
        for name, values in {
            RESULTS[0]: summary,
            RESULTS[1]: forward,
            RESULTS[2]: tilt,
            **licences,
        }.items()
    }


def check_links(target):
    """Reject missing or escaping local Markdown and LaTeX artifact references."""
    target = Path(target).resolve()
    checked = 0
    for path in [*target.rglob("*.md"), *target.rglob("*.tex")]:
        content = path.read_text()
        references = [(url, path.parent, "") for url in re.findall(r"\]\(([^\s)]+)", content)]
        if path.suffix == ".tex":
            for command, url in re.findall(
                r"\\(input|includegraphics|bibliography)(?:\[[^\]]*\])?\{([^}]+)\}", content
            ):
                references.append(
                    (
                        url,
                        path.parent,
                        {"input": ".tex", "includegraphics": ".pdf", "bibliography": ".bib"}[
                            command
                        ],
                    )
                )
            references += [
                (url, path.parent, "") for url in re.findall(r"\\href\{([^}]+)\}", content)
            ]
            references += [
                (url, target, "")
                for url in re.findall(r"\\path\{([^}]+)\}", content)
                if not url.startswith("/") and Path(url).suffix
            ]
        for url, base, suffix in references:
            parts = urlsplit(url.strip("<>"))
            if parts.scheme or parts.netloc or not parts.path:
                continue
            destination = (base / unquote(parts.path)).resolve()
            if suffix and not destination.suffix:
                destination = destination.with_suffix(suffix)
            if not destination.is_relative_to(target) or not destination.exists():
                raise ValueError(f"missing/escaping local link in {path.relative_to(target)}")
            checked += 1
    return checked


def export(target):
    """Export committed allowlisted files and aggregate fields into a fresh directory."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    def read(name):
        return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)

    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True
    ).splitlines()
    for name in names:
        if name not in FILES and not name.startswith(DIRECTORIES):
            continue
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(read(name))
    publications = publication_results(read, commit)
    for name, values in publications.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    summary = publications[RESULTS[0]]
    latency = summary["markets"]["binance/BTC"]["local"]["classifier"]["latency"]
    framed = publications[RESULTS[2]]["nim-tilt-analysis"]["models"].values()
    valid = sum(row["latency"]["n"] for row in framed)
    failed = sum(row["latency"]["failed"] for row in framed)
    (target / "README.md").write_text(
        "# judgment-gates\n\n"
        "[Paper (PDF)](paper/main.pdf) · [TeX and build instructions](paper/README.md)\n\n"
        f"**TL;DR:** [{summary['book_verification']['compared']:,} book comparisons, "
        f"{summary['book_verification']['error_count']} mismatches]"
        "(results/research-summary.json); prospective results remain pending.<br>\n"
        f"Local classifier: [{latency['p50_ms'] * 1000:.3f}/"
        f"{latency['p99_ms'] * 1000:.3f} µs p50/p99](results/research-summary.json); "
        "speed has not established economic value.<br>\n"
        f"[Framing extension: {valid}/{valid + failed} valid judgments]"
        "(results/framing-summary.json); tiny, conditional CIs do not establish invariance.\n\n"
        "Paper-only market-making research: narrow judgment gates around deterministic pricing "
        "and hard risk limits. The pilot does not establish profitable trading or emotion-free "
        "model judgment; independent-day results remain pending.\n\n"
        "Published evidence: [calibration, replay, latency and verifier aggregates]"
        "(results/research-summary.json), [forward activity and P&L decomposition]"
        "(results/forward-diagnostics.json), [all framing studies and descriptive CIs]"
        "(results/framing-summary.json). Missing CIs and failed routes are retained.\n\n"
        "```mermaid\nflowchart LR\n  D[Private recordings and provider calls] --> "
        "A[Author-computed aggregates]\n  A --> P[Public paper and result files]\n"
        "  C[Public code and synthetic tests] --> P\n```\n\n"
        "Install uv, then run `uv sync --frozen` and `uv run pytest -q`. "
        "[Original code](LICENSE) is MIT, Nicolas Coquelet; "
        "[third-party terms](THIRD_PARTY_NOTICES.md) apply separately.\n\n"
        "## Publication boundary\n\n"
        "Excluded: raw Binance/Hyperliquid books and trades, row-level features/labels, "
        "individual simulated events, trained policies, raw LLM call transcripts "
        "(`*-calls.json`), reasoning, caches, per-call timing tables, secrets, "
        "personal metadata and private Git history. Public endpoints and model-weight licences "
        "do not establish permission to redistribute exchange data or hosted outputs. "
        "See the [release decision and official exchange/provider terms](RELEASE_POLICY.md).\n\n"
        "These are our measured counts/timings and computed statistics, not a redistribution "
        "of provider-generated content. Raw-data replay and per-call rescoring require the "
        "private inputs; the public files cannot recreate them. "
        "[EXPORT.json](EXPORT.json) pins the private source commit and published file hashes.\n"
    )
    generated = target / "paper/generated"
    (generated / "manifest.json").write_text(
        json.dumps(
            {
                "scope": "published aggregates and precomputed tables/figures; "
                "raw inputs remain private",
                "source_commit": commit,
                "private_generation_manifest_sha256": hashlib.sha256(
                    read("paper/generated/manifest.json")
                ).hexdigest(),
                "inputs": {
                    name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                    for name in RESULTS
                },
                "outputs": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(generated.iterdir())
                    if p.name != "manifest.json"
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = {
        "source_commit": commit,
        "scope": "code, paper and projected aggregates",
        "files": {
            str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(target.rglob("*"))
            if p.is_file()
        },
    }
    (target / "EXPORT.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["checked_local_links"] = check_links(target)
    (target / "EXPORT.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    result = export(args.target)
    print(
        f"Exported {len(result['files'])} files; "
        f"{result['checked_local_links']} local links resolve"
    )
