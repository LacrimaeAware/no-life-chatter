"""Print generated artifact status without rebuilding anything.

Use this before trusting semantic/persona commands after alias/filter/embedding
changes:

    python scripts/artifact_status.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import artifact_status  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    rows = artifact_status.status_rows()
    print(artifact_status.format_table(rows))
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
