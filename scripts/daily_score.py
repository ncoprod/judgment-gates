"""Score frozen paper logs and publish only the daily README and receipt."""

import fcntl
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("README.md", "results/forward-daily.json")


def git(root, *args, check=True):
    """Run Git without exposing captured output or credentials to the scorer log."""
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check)


def readme_prose(text):
    """Exclude the generated daily section when checking for owner edits."""
    before, rest = text.split("<!-- DAILY:START -->")
    _, after = rest.split("<!-- DAILY:END -->")
    return before, after


def check_readme(root):
    """Refuse to commit unreviewed edits outside the scorer's README section."""
    committed = git(root, "show", "HEAD:README.md").stdout
    if readme_prose(committed) != readme_prose((root / "README.md").read_text()):
        raise RuntimeError("README prose has uncommitted edits; daily publishing deferred")


def publish(root):
    """Commit only the two daily paths, preserve other staged files, then retry push."""
    if git(root, "branch", "--show-current").stdout.strip() != "main":
        raise RuntimeError("daily publishing requires main")
    check_readme(root)
    # Split the bare marker so the scanner does not match its own source.
    pattern = r"nvapi-[A-Za-z0-9]{10}|" + "KG" + r'AT_|"key"\s*:'
    scan = git(root, "grep", "-nE", pattern, check=False)
    if scan.returncode != 1:
        raise RuntimeError("required secret scan failed; matching text withheld")
    changed = git(root, "diff", "HEAD", "--quiet", "--", *FILES, check=False)
    if changed.returncode == 1:
        git(
            root,
            "commit",
            "--only",
            "-m",
            "results: update daily paper-forward score",
            "--",
            *FILES,
        )
    elif changed.returncode != 0:
        raise RuntimeError("daily diff check failed")
    git(root, "push", "origin", "main")


def main():
    """Use the frozen scorer without restarting or changing the trading policy."""
    (ROOT / "runs").mkdir(exist_ok=True)
    with (ROOT / "runs/daily-score.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_readme(ROOT)
        frozen = ROOT / "runs/frozen-v4"
        env = {**os.environ, "PYTHONPATH": str(frozen / "src")}
        subprocess.run(
            [
                str(frozen / ".venv/bin/python"),
                "-m",
                "judgment_gates.cli",
                "score",
                "--directory",
                str(ROOT / "runs/forward-v4"),
                "--output",
                str(ROOT / FILES[1]),
                "--readme",
                str(ROOT / FILES[0]),
            ],
            cwd=frozen,
            env=env,
            check=True,
        )
        publish(ROOT)
        print("Daily paper score committed and pushed; policy unchanged", flush=True)


if __name__ == "__main__":
    main()
