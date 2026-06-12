"""Create a dated backup zip for local unsynced bot data."""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path


DEFAULT_EXTENSIONS = {
    ".db", ".sqlite", ".sqlite3",
    ".pkl", ".pickle",
    ".json", ".jsonl",
    ".bin", ".pt", ".safetensors", ".gguf",
    ".toml",
}


def iter_backup_files(source: Path, include_logs: bool = False):
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        parts = {p.lower() for p in path.parts}
        if "backups" in parts:
            continue
        if path.suffix.lower() == ".log" and not include_logs:
            continue
        if path.suffix.lower() in DEFAULT_EXTENSIONS:
            yield path


def prune_old_backups(dest: Path, keep: int) -> None:
    if keep <= 0:
        return
    backups = sorted(dest.glob("nolifechatter-unsynced-*.zip"),
                     key=lambda p: p.stat().st_mtime,
                     reverse=True)
    for old in backups[keep:]:
        old.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zip data/unsynced into a dated local backup and keep the newest N archives.")
    parser.add_argument("--source", default="data/unsynced",
                        help="directory to back up")
    parser.add_argument("--dest", default="_private/backups",
                        help="directory where backup zips are written")
    parser.add_argument("--keep", type=int, default=3,
                        help="number of backup zips to keep")
    parser.add_argument("--include-logs", action="store_true",
                        help="include .log files too")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be backed up without writing a zip")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    if not source.exists():
        parser.error(f"source does not exist: {source}")

    files = list(iter_backup_files(source, include_logs=args.include_logs))
    if args.dry_run:
        for path in files:
            print(path)
        print(f"{len(files)} files")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = dest / f"nolifechatter-unsynced-{stamp}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, path.relative_to(source.parent))

    prune_old_backups(dest, args.keep)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out} ({len(files)} files, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
