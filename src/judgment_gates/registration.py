"""Bind a policy's bytes to an immutable Git tag, without trusting receipt claims."""

import json
import subprocess
from pathlib import Path


def registered_policy(path, running=False):
    """Verify policy bytes against the immutable tag and optionally the running source."""
    path = Path(path).resolve()
    root = Path(
        subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    policy = json.loads(path.read_text())

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    commit = git("rev-parse", "--verify", policy["tag"] + "^{commit}").decode().strip()
    if git("show", f"{commit}:{path.relative_to(root)}") != path.read_bytes():
        raise ValueError("policy bytes differ from frozen tag")
    if running:
        if git("rev-parse", "HEAD").decode().strip() != commit:
            raise ValueError("run from the frozen tag worktree")
        if git("diff", commit, "--", "src", "uv.lock", "pyproject.toml"):
            raise ValueError("frozen runtime source was modified")
    return policy, commit


def engine_names(policy):
    """List every registered market, gate and fill-scenario engine."""
    return [
        f"{market}/{gate}/{fill}"
        for market, entry in policy["markets"].items()
        for gate in entry["gates"]
        for fill in ("optimistic", "pessimistic")
    ]
