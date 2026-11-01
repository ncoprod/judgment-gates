import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path


def canonical(value):
    """Encode a stable JSON representation for hash chaining."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verified_rows(path, limit=None, anchor=None):
    """Yield a verified ledger prefix, optionally continuing from a trusted anchor."""
    previous, count = "0" * 64, 0
    path = Path(path)
    segments = {}
    for segment in (path.parent / "ledger-parts").glob("*.jsonl*"):
        if segment.name.endswith(".tmp"):
            continue
        segments[segment.name.removesuffix(".gz")] = segment
    paths = [segments[k] for k in sorted(segments)] + ([path] if path.exists() else [])
    anchored = anchor is None or anchor[0] == 0
    if not anchored:
        # Closed prefixes were already verified when the checkpoint was written.
        # Recheck its containing segment and tail; a full audit still starts at zero.
        first = 0
        for i, segment in enumerate(paths):
            start = int(segment.name.split("-", 1)[0]) if segment != path else None
            if start is None:
                with path.open() as f:
                    line = f.readline()
                start = json.loads(line)["sequence"] if line else anchor[0]
            if start <= anchor[0] - 1:
                first = i
        paths = paths[first:]
        previous, count = None, None
    for segment in paths:
        opener = gzip.open if segment.suffix == ".gz" else open
        try:
            f = opener(segment, "rt")
        except FileNotFoundError:
            f = gzip.open(str(segment) + ".gz", "rt")
        with f:
            for line in f:
                if limit is not None and anchored and count >= limit:
                    return
                row = json.loads(line)
                if count is None:
                    previous, count = row["previous"], row["sequence"]
                expected = hashlib.sha256(
                    canonical({"previous": previous, "sequence": count, "event": row["event"]})
                ).hexdigest()
                if (
                    row["previous"] != previous
                    or row["sequence"] != count
                    or row["hash"] != expected
                ):
                    raise ValueError(f"ledger corruption at row {count}")
                previous, count = row["hash"], count + 1
                if not anchored:
                    if count == anchor[0]:
                        if previous != anchor[1]:
                            raise ValueError("checkpoint anchor changed")
                        anchored = True
                    continue
                yield row
    if not anchored:
        raise ValueError("checkpoint anchor missing")


def verify(path, anchor=None):
    """Verify the ledger and return its row count and tail hash."""
    count, previous = anchor if anchor else (0, "0" * 64)
    for row in verified_rows(path, anchor=anchor):
        previous, count = row["hash"], row["sequence"] + 1
    return previous, count


class Ledger:
    """One writer; use the forward process lock. Hashes detect edits, not truncation."""

    def __init__(self, path, anchor=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous, self.count = verify(self.path, anchor)
        self.segment_start = self.count
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open() as f:
                self.segment_start = json.loads(f.readline())["sequence"]

    def rotate(self, date):
        """Rotate the active ledger segment at a UTC-day boundary."""
        if not self.path.exists() or not self.path.stat().st_size:
            return None
        directory = self.path.parent / "ledger-parts"
        directory.mkdir(exist_ok=True)
        target = directory / f"{self.segment_start:015d}-{date}.jsonl"
        if target.exists() or Path(str(target) + ".gz").exists():
            raise ValueError("ledger segment already exists")
        self.path.replace(target)
        self.segment_start = self.count
        return target

    def append(self, event):
        """Append and flush one event with its predecessor hash."""
        row = {"previous": self.previous, "sequence": self.count, "event": event}
        row["hash"] = hashlib.sha256(canonical(row)).hexdigest()
        with self.path.open("ab") as f:
            f.write(canonical(row) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        self.previous, self.count = row["hash"], self.count + 1
        return row


def compress_segment(path):
    """Compress an immutable segment while preserving its uncompressed content."""
    target = Path(str(path) + ".gz")
    temp = Path(str(target) + ".tmp")
    with Path(path).open("rb") as source, gzip.open(temp, "wb", compresslevel=6) as dest:
        shutil.copyfileobj(source, dest)
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    temp.replace(target)
    Path(path).unlink()
