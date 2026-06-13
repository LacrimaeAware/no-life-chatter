"""Audit live command modules for import/handler/doc health.

This does not call Twitch or run command handlers. It catches command import
failures one-by-one so a single broken module cannot hide the rest of the
surface.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMMAND_RE = re.compile(r"`~([a-zA-Z0-9_]+)")


def _doc_commands(path: Path) -> set[str]:
    found = set()
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Planned But Not Live"):
            break
        for match in COMMAND_RE.finditer(line):
            found.add(match.group(1).lower())
    return found


def _module_files(commands_dir: Path) -> list[Path]:
    return sorted(
        path for path in commands_dir.glob("*.py")
        if not path.name.startswith("__")
    )


def audit(commands_dir: Path, doc: Path) -> dict:
    documented = _doc_commands(doc)
    rows = []
    for path in _module_files(commands_dir):
        name = path.stem
        row = {
            "command": name,
            "documented": name in documented,
            "imports": False,
            "has_handler": False,
            "async_handler": False,
            "has_description": False,
            "status": "ok",
            "error": "",
        }
        try:
            module = importlib.import_module(f"commands.{name}")
            row["imports"] = True
            handler = getattr(module, f"handle_{name}", None)
            row["has_handler"] = handler is not None
            row["async_handler"] = bool(handler and inspect.iscoroutinefunction(handler))
            row["has_description"] = bool((getattr(module, "description", "") or "").strip())
            if not row["has_handler"]:
                row["status"] = "fail"
                row["error"] = f"missing handle_{name}"
            elif not row["async_handler"]:
                row["status"] = "warn"
                row["error"] = "handler is not async"
            elif not row["has_description"] or not row["documented"]:
                row["status"] = "warn"
                missing = []
                if not row["has_description"]:
                    missing.append("description")
                if not row["documented"]:
                    missing.append("docs")
                row["error"] = "missing " + ", ".join(missing)
        except Exception as exc:
            row["status"] = "fail"
            row["error"] = repr(exc)
            row["traceback"] = traceback.format_exc()
        rows.append(row)

    live = {row["command"] for row in rows}
    extra_docs = sorted(documented - live)
    return {
        "commands": len(rows),
        "ok": sum(1 for row in rows if row["status"] == "ok"),
        "warn": sum(1 for row in rows if row["status"] == "warn"),
        "fail": sum(1 for row in rows if row["status"] == "fail"),
        "extra_docs": extra_docs,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands-dir", default="commands")
    parser.add_argument("--doc", default="docs/COMMANDS.md")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = audit(Path(args.commands_dir), Path(args.doc))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"commands={report['commands']} ok={report['ok']} "
            f"warn={report['warn']} fail={report['fail']}"
        )
        for row in report["rows"]:
            if row["status"] != "ok":
                print(f"{row['status'].upper():4} ~{row['command']}: {row['error']}")
        if report["extra_docs"]:
            print("extra docs:", ", ".join("~" + name for name in report["extra_docs"]))

    return 1 if report["fail"] or report["extra_docs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
