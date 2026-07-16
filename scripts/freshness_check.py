"""Run a repo-level freshness check without mutating local state.

This is meant as the "before bed / after break" command:

    python scripts/freshness_check.py

It wraps the command/doc audits, generated artifact status, docs layout sanity,
latest maintenance-log status, and git dirtiness. Warnings are printed for stale
artifacts or dirty worktrees; hard failures are reserved for broken command
imports/docs or docs layout regressions.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_DOCS = {
    "docs/GROUND_TRUTH.md",
    "docs/README.md",
    "docs/STATE_OF_OPERATION.md",
    "docs/ROADMAP.md",
    "docs/COMMANDS.md",
    "docs/CHAT_ARCHIVE.md",
    "docs/CHAT_PERSONALITY_RESEARCH.md",
    "docs/RESEARCH_TO_APPLIED.md",
    "docs/PERSONALITY_SYSTEM_DESIGN.md",
    "docs/INVESTIGATION_LOG.md",
    "docs/GENERATE_AND_BOT_MODES.md",
    "docs/FINE_TUNING.md",
    "docs/IDEA_BANK.md",
}

ARCHIVED_DOCS = {
    "docs/archive/README.md",
    "docs/archive/HANDOFF.md",
    "docs/archive/NEXT_WORK_RANKING_2026-06-13.md",
    "docs/archive/PERSONA_BOT_ROADMAP.md",
    "docs/archive/PROJECT_AUDIT_2026-06-13.md",
    "docs/archive/PROJECT_AUDIT_2026-07-15.md",
    "docs/archive/REORG_PLAN.md",
    "docs/archive/STATE_OF_ART_AND_REMAINING_WORK_2026-06-13.md",
}

LEGACY_ROOT_DOCS = {
    "docs/HANDOFF.md",
    "docs/NEXT_WORK_RANKING_2026-06-13.md",
    "docs/PERSONA_BOT_ROADMAP.md",
    "docs/PROJECT_AUDIT_2026-06-13.md",
    "docs/REORG_PLAN.md",
    "docs/STATE_OF_ART_AND_REMAINING_WORK_2026-06-13.md",
}

LEGACY_LINK_TEXT = (
    "docs/HANDOFF.md",
    "docs/NEXT_WORK_RANKING_2026-06-13.md",
    "docs/PERSONA_BOT_ROADMAP.md",
    "docs/PROJECT_AUDIT_2026-06-13.md",
    "docs/REORG_PLAN.md",
    "docs/STATE_OF_ART_AND_REMAINING_WORK_2026-06-13.md",
    "(HANDOFF.md)",
    "(NEXT_WORK_RANKING_2026-06-13.md)",
    "(PERSONA_BOT_ROADMAP.md)",
    "(PROJECT_AUDIT_2026-06-13.md)",
    "(REORG_PLAN.md)",
    "(STATE_OF_ART_AND_REMAINING_WORK_2026-06-13.md)",
)

MAINTENANCE_SCRIPT_RE = (
    r"(?:rebuild_persona_artifacts|build_iq_v2|build_user_profiles|"
    r"build_emote_semantics)\.py"
)
MAINTENANCE_LOG_GLOBS = (
    "rebuild_persona_artifacts*.log",
    "iq-v5-*.out.log",
    "user-profiles-*.out.log",
    "emote-semantics-*.out.log",
)


@dataclass
class Step:
    name: str
    status: str
    details: str = ""
    hard_fail: bool = False


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )


def _tail(path: Path, lines: int = 8) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _fmt_output(result: subprocess.CompletedProcess[str], max_lines: int = 12) -> str:
    combined = (result.stdout or "").strip()
    if result.stderr:
        combined = (combined + "\n" + result.stderr.strip()).strip()
    if not combined:
        return f"exit={result.returncode}"
    lines = combined.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return "\n".join(lines)


def docs_layout_step() -> Step:
    missing = sorted(path for path in ACTIVE_DOCS | ARCHIVED_DOCS if not (ROOT / path).exists())
    legacy_present = sorted(path for path in LEGACY_ROOT_DOCS if (ROOT / path).exists())
    stale_links: list[str] = []
    for path_text in sorted(ACTIVE_DOCS | {"README.md"}):
        path = ROOT / path_text
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in LEGACY_LINK_TEXT:
            if needle in text:
                stale_links.append(f"{path_text}: {needle}")

    problems = []
    if missing:
        problems.append("missing: " + ", ".join(missing))
    if legacy_present:
        problems.append("legacy docs still active: " + ", ".join(legacy_present))
    if stale_links:
        problems.append("active docs still link legacy roots: " + "; ".join(stale_links))
    if problems:
        return Step("docs layout", "FAIL", "\n".join(problems), hard_fail=True)
    return Step("docs layout", "OK", "active docs and archive layout are coherent")


def command_audit_step() -> Step:
    result = _run([sys.executable, "scripts/audit_commands.py"])
    status = "OK" if result.returncode == 0 else "FAIL"
    return Step("command audit", status, _fmt_output(result), hard_fail=result.returncode != 0)


def command_doc_step() -> Step:
    result = _run([sys.executable, "scripts/check_readme_commands.py"])
    status = "OK" if result.returncode == 0 else "FAIL"
    return Step("command bible sync", status, _fmt_output(result), hard_fail=result.returncode != 0)


def artifact_status_step() -> Step:
    result = _run([sys.executable, "scripts/artifact_status.py"], timeout=60)
    status = "OK" if result.returncode == 0 else "WARN"
    return Step("artifact status", status, _fmt_output(result, max_lines=20))


def ground_truth_step() -> Step:
    # Re-verify the numbers recorded in docs/GROUND_TRUTH.md against live state.
    # DRIFT is a WARN, not a hard fail: a legitimate artifact rebuild moves the
    # numbers and should prompt a re-baseline, not block the worktree.
    result = _run([sys.executable, "scripts/ground_truth_check.py"], timeout=90)
    status = "OK" if result.returncode == 0 else "WARN"
    return Step("ground truth", status, _fmt_output(result, max_lines=14))


def _rebuild_pids() -> list[str]:
    if os.name == "nt":
        command = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -match '^python(w)?\\.exe$' -and "
            f"$_.CommandLine -match '{MAINTENANCE_SCRIPT_RE}' }} | "
            "ForEach-Object { $_.ProcessId }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=20,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=20,
    )
    if result.returncode != 0:
        return []
    pids = []
    for line in result.stdout.splitlines():
        if re.search(MAINTENANCE_SCRIPT_RE, line) and "freshness_check.py" not in line:
            pids.append(line.strip().split(maxsplit=1)[0])
    return pids


def rebuild_log_step() -> Step:
    stdout_logs = []
    for pattern in MAINTENANCE_LOG_GLOBS:
        stdout_logs.extend(
            path for path in (ROOT / "data/unsynced").glob(pattern)
            if not path.name.endswith(".err.log")
        )
    latest = max(stdout_logs, key=lambda path: path.stat().st_mtime) if stdout_logs else None
    pids = _rebuild_pids()
    if not latest:
        detail = "no rebuild log found under data/unsynced"
        if pids:
            detail += f"; running pids={','.join(pids)}"
        return Step("latest maintenance", "WARN", detail)

    err_name = (
        latest.name.replace(".out.log", ".err.log")
        if latest.name.endswith(".out.log")
        else latest.name.replace(".log", ".err.log")
    )
    err = latest.with_name(err_name)
    details = [f"log={latest.relative_to(ROOT)}"]
    if pids:
        details.append(f"running pids={','.join(pids)}")
    else:
        details.append("no rebuild process currently detected")
    tail = _tail(latest)
    if tail:
        details.append("tail:\n" + tail)
    err_tail = _tail(err, lines=5)
    if err_tail:
        details.append("stderr tail:\n" + err_tail)
    completed = (
        "exit_code=0" in (tail or "")
        or "text-IQ rows ->" in (tail or "")
        or "emote context-vectors ->" in (tail or "")
    )
    status = "OK" if pids or completed else "WARN"
    return Step("latest maintenance", status, "\n".join(details))


def git_status_step() -> Step:
    result = _run(["git", "status", "--short", "--branch"], timeout=30)
    if result.returncode != 0:
        return Step("git status", "WARN", _fmt_output(result))
    lines = (result.stdout or "").splitlines()
    dirty = [line for line in lines[1:] if line.strip()]
    status = "WARN" if dirty else "OK"
    details = result.stdout.strip() or "clean"
    return Step("git status", status, details)


def main() -> int:
    steps = [
        docs_layout_step(),
        command_audit_step(),
        command_doc_step(),
        artifact_status_step(),
        ground_truth_step(),
        rebuild_log_step(),
        git_status_step(),
    ]

    for step in steps:
        print(f"[{step.status}] {step.name}")
        if step.details:
            for line in step.details.splitlines():
                print(f"  {line}")

    if any(step.hard_fail for step in steps):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
