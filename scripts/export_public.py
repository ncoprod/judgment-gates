"""Export original code and synthetic tests, never the private Git history or results."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("src/", "tests/")
FILES = {
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "scripts/export_public.py",
    "scripts/daily_score.py",
}


def export(target):
    """Copy only committed original code and synthetic tests into a fresh directory."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    names = (
        subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT)
        .decode()
        .splitlines()
    )
    manifest = {
        "source_commit": commit,
        "scope": "original code and synthetic tests only",
        "files": {},
    }
    for name in names:
        if name not in FILES and not name.startswith(ALLOWED):
            continue
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["files"][name] = hashlib.sha256(content).hexdigest()
    (target / "README.md").write_text(
        "# judgment-gates: code and synthetic tests\n\n"
        "MIT, Nicolas Coquelet. Paper-only market-making research tools.\n"
        "Install uv, then run `uv sync --frozen` and `uv run pytest -q`.\n\n"
        "This clean export includes no original Git history, empirical results, trained policies, "
        "API transcripts, exchange data or paper. It cannot reproduce private empirical claims. "
        "Dependencies retain their upstream licences; no model weights are included.\n"
    )
    (target / "EXPORT.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    export(args.target)
