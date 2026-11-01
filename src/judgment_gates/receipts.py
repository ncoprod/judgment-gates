import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    """Hash a file incrementally with SHA-256."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(config, seed=42, model_revision=None):
    """Capture the command, configuration, source hashes and producing Git state."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    sources = sorted(Path("src").rglob("*.py")) + sorted(Path("scripts").rglob("*.py"))
    sources += [Path("pyproject.toml"), Path("uv.lock")]
    return {
        "command": shlex.join(sys.argv),
        "config": config,
        "seed": seed,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_sha256": {str(p): digest(p) for p in sources},
        "model_revision": model_revision,
        "created_ns": time.time_ns(),
        "python": sys.version,
        "platform": platform.platform(),
    }


def write_json(path, value):
    """Atomically write a finite JSON receipt and flush it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
