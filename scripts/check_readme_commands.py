"""Check that README's command table mentions every command module."""

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


def readme_commands(readme: Path) -> set[str]:
    found: set[str] = set()
    in_table = False
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Command |"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("|"):
            continue
        for match in COMMAND_RE.finditer(line):
            found.add(match.group(1).lower())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify README's command table is in sync with commands/*.py.")
    parser.add_argument("--readme", default="README.md", help="README path")
    parser.add_argument("--commands-dir", default="commands", help="commands directory")
    args = parser.parse_args()

    modules = command_modules(Path(args.commands_dir))
    documented = readme_commands(Path(args.readme))
    missing = sorted(modules - documented)
    extra = sorted(documented - modules)

    if missing:
        print("Missing from README command table:")
        for name in missing:
            print(f"  ~{name}")
    if extra:
        print("Documented but no commands/*.py module exists:")
        for name in extra:
            print(f"  ~{name}")
    if missing or extra:
        return 1
    print(f"README command table is in sync ({len(modules)} commands).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
