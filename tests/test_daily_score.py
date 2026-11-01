"""Check publishing boundaries using local Git repositories, never GitHub writes."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("daily_score", Path("scripts/daily_score.py"))
daily_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily_score)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def repository(tmp_path):
    remote, root = tmp_path / "remote.git", tmp_path / "source"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "results").mkdir()
    (root / "README.md").write_text("Prose\n<!-- DAILY:START -->\nold\n<!-- DAILY:END -->\n")
    (root / "results/forward-daily.json").write_text("{}\n")
    (root / "unrelated.py").write_text("old\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    git(root, "remote", "add", "origin", str(remote))
    return root, remote


def test_daily_publisher_preserves_unrelated_staged_changes_and_retries_push(tmp_path):
    root, remote = repository(tmp_path)
    (root / "unrelated.py").write_text("pending\n")
    git(root, "add", "unrelated.py")
    (root / "README.md").write_text("Prose\n<!-- DAILY:START -->\nnew\n<!-- DAILY:END -->\n")
    (root / "results/forward-daily.json").write_text('{"rows": 1}\n')
    daily_score.publish(root)
    assert set(
        git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    ) == {"README.md", "results/forward-daily.json"}
    assert git(root, "diff", "--cached", "--name-only") == "unrelated.py"
    assert git(remote, "rev-parse", "main") == git(root, "rev-parse", "HEAD")
    head = git(root, "rev-parse", "HEAD")
    daily_score.publish(root)
    assert git(root, "rev-parse", "HEAD") == head


def test_daily_publisher_refuses_owner_prose_and_other_branches(tmp_path):
    root, _ = repository(tmp_path)
    (root / "README.md").write_text("Owner edit\n<!-- DAILY:START -->\nold\n<!-- DAILY:END -->\n")
    with pytest.raises(RuntimeError, match="prose"):
        daily_score.publish(root)
    git(root, "checkout", "-b", "work")
    with pytest.raises(RuntimeError, match="requires main"):
        daily_score.publish(root)
