"""Check that the public command bible mentions every live command module."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

COMMAND_RE = re.compile(r"`~([a-zA-Z0-9_]+)")


def command_modules(commands_dir: Path) -> set[str]:
    return {
        path.stem
        for path in commands_dir.glob("*.py")
        if not path.name.startswith("__")
    }


def documented_commands(doc: Path) -> set[str]:
    found: set[str] = set()
    for line in doc.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Planned But Not Live"):
            break
        for match in COMMAND_RE.finditer(line):
            found.add(match.group(1).lower())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify docs/COMMANDS.md is in sync with commands/*.py.")
    parser.add_argument("--doc", default="docs/COMMANDS.md", help="command doc path")
    parser.add_argument("--commands-dir", default="commands", help="commands directory")
    args = parser.parse_args()

    modules = command_modules(Path(args.commands_dir))
    documented = documented_commands(Path(args.doc))
    missing = sorted(modules - documented)
    extra = sorted(documented - modules)

    if missing:
        print("Missing from command doc:")
        for name in missing:
            print(f"  ~{name}")
    if extra:
        print("Documented but no commands/*.py module exists:")
        for name in extra:
            print(f"  ~{name}")
    if missing or extra:
        return 1
    print(f"Command doc is in sync ({len(modules)} commands).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
