import argparse
import asyncio
import json
from pathlib import Path

from .data import snapshot
from .final import evaluate
from .forward import freeze, live, score
from .providers import CONTEXTS, measure_models, probe_opencode
from .remote_study import evaluate as remote_evaluate
from .study import local_study, model_analysis, prepare


def main():
    """Dispatch reproducible data, model, replay and paper-only forward commands."""
    parser = argparse.ArgumentParser(description="Paper-only judgment gate research")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("snapshot", "prepare", "pilot"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--root", type=Path, default=Path.home() / "data/lob")
        cmd.add_argument("--date", default="2026-10-29")
        cmd.add_argument("--output", type=Path, required=True)
        if name == "pilot":
            cmd.add_argument("--cohort", type=Path, required=True)
    models = sub.add_parser("models")
    models.add_argument("--cohort", type=Path, required=True)
    models.add_argument("--market", default="binance/BTC")
    models.add_argument(
        "--models",
        nargs="+",
        default=[
            "typesafe-ai/jev",
            "z-ai/glm-5.3-flash",
            "deepseek-ai/deepseek-v4.1-flash",
            "openai/gpt-oss-20b",
        ],
    )
    models.add_argument("--count", type=int, default=6)
    models.add_argument("--replicate", type=int, default=0)
    models.add_argument("--retry-failed", action="store_true")
    models.add_argument(
        "--contexts",
        nargs="+",
        choices=list(CONTEXTS),
    )
    models.add_argument("--output", type=Path, required=True)
    models.add_argument("--cache", type=Path, default=Path("runs/cache"))
    probe = sub.add_parser("opencode-probe")
    probe.add_argument("--model", default="opencode/jev-1.13-free")
    probe.add_argument("--output", type=Path, required=True)
    analysis = sub.add_parser("analyse-models")
    analysis.add_argument("--cohort", type=Path, required=True)
    analysis.add_argument("--calls", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--pilot", type=Path, required=True)
    frozen.add_argument("--output", type=Path, default=Path("configs/forward-v4.json"))
    forward = sub.add_parser("forward")
    forward.add_argument("--policy", type=Path, default=Path("configs/forward-v4.json"))
    forward.add_argument("--directory", type=Path, default=Path("runs/forward-v4"))
    scorer = sub.add_parser("score")
    scorer.add_argument("--directory", type=Path, default=Path("runs/forward-v4"))
    scorer.add_argument("--output", type=Path, default=Path("results/forward-daily.json"))
    scorer.add_argument("--readme", type=Path)
    final = sub.add_parser("final")
    final.add_argument("--policy", type=Path, default=Path("configs/forward-v4.json"))
    final.add_argument("--directory", type=Path, default=Path("runs/forward-v4"))
    final.add_argument("--output", type=Path, default=Path("results/final.json"))
    remote = sub.add_parser("remote-pilot")
    remote.add_argument("--root", type=Path, required=True)
    remote.add_argument("--date", default="2026-10-29")
    remote.add_argument("--cohort", type=Path, required=True)
    remote.add_argument("--calls", type=Path, required=True)
    remote.add_argument("--market", default="binance/BTC")
    remote.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.root, args.output, args.date)
    elif args.command == "prepare":
        prepare(args.root, args.date, args.output)
    elif args.command == "pilot":
        local_study(args.root, args.date, args.cohort, args.output)
    elif args.command == "models":
        rows = json.loads(args.cohort.read_text())["markets"][args.market]["states"]
        measure_models(
            rows,
            args.models,
            args.output,
            args.cache,
            args.count,
            args.contexts,
            args.market,
            args.replicate,
            args.retry_failed,
        )
    elif args.command == "opencode-probe":
        probe_opencode(args.model, args.output)
    elif args.command == "analyse-models":
        model_analysis(args.cohort, args.calls, args.output)
    elif args.command == "freeze":
        freeze(args.pilot, args.output)
    elif args.command == "forward":
        asyncio.run(live(args.policy, args.directory))
    elif args.command == "score":
        score(args.directory, args.output, args.readme)
    elif args.command == "final":
        evaluate(args.policy, args.directory, args.output)
    elif args.command == "remote-pilot":
        remote_evaluate(args.root, args.date, args.cohort, args.calls, args.output, args.market)


if __name__ == "__main__":
    main()
