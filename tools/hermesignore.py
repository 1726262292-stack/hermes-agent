#!/usr/bin/env python3
"""Project-level agent file-access ignore file (``.hermesignore``).

Inspired by Cursor's ``.cursorignore``: a gitignore-syntax file committed at
the workspace root that blocks the agent's FILE TOOLS (``read_file``,
``write_file``, ``patch``, ``search_files``) from touching matched paths.
Unlike the hardcoded sensitive-path denylist in ``tools/file_tools.py``,
``.hermesignore`` is repo-scoped: a team can commit one file and every Hermes
session working in that checkout honors it.

Syntax (parsed with ``pathspec``'s gitwildmatch — real gitignore semantics):

* ``#`` comments and blank lines are skipped
* ``*`` and ``**`` globs (``*.pem``, ``docs/**/*.md``)
* trailing-slash directory patterns (``secrets/``)
* bare names match files AND directory subtrees (``build`` blocks ``build/x``)
* ``!`` negation, last match wins (``*.env`` then ``!example.env``)

Discovery walks UP from the resolved terminal cwd (and from the target file's
own directory, so worktree/subdir sessions still resolve the repo root) to the
git root — the first directory containing ``.hermesignore``, stopping at the
directory that contains ``.git`` or at the filesystem root. Parsed specs are
cached per ``(path, mtime)`` so the file is re-read only when it changes.

Scope and limitations (same tradeoff Cursor documents for ``.cursorignore``):
this gates the in-process FILE TOOLS ONLY. The ``terminal`` tool runs shell
commands as the same OS user and can still ``cat`` a matched file, and MCP
servers perform their own I/O. Defense-in-depth and a clear stop signal for
the model — not a sandbox for a hostile agent.

Kill switch: ``security.hermesignore_enabled`` (default ``true``) in
``~/.hermes/config.yaml``.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

IGNORE_FILENAME = ".hermesignore"

# Bound the walk-up so a pathological cwd (or symlink loop survivor) cannot
# spin forever. 64 levels is far deeper than any real checkout.
_MAX_WALK_UP = 64

# path -> (mtime_ns, size, PathSpec | None). ``None`` spec means the file
# existed but parsed to zero usable patterns — cached to skip re-parsing.
_spec_cache: dict[str, tuple[int, int, object | None]] = {}
_cache_lock = threading.Lock()


def _enabled() -> bool:
    """Return the ``security.hermesignore_enabled`` toggle (default True).

    Fails OPEN on config errors: this is a project-convenience guard, not a
    security boundary, and a broken config should not lock the agent out of
    every repo that happens to contain a ``.hermesignore``.
    """
    try:
        from hermes_cli.config import load_config

        security = load_config().get("security") or {}
        return bool(security.get("hermesignore_enabled", True))
    except Exception:
        return True


def _load_spec(ignore_path: str):
    """Parse *ignore_path* into a ``pathspec.PathSpec``, cached by (mtime, size).

    Returns ``None`` when the file vanished, is unreadable, or contains no
    usable patterns.
    """
    try:
        st = os.stat(ignore_path)
    except OSError:
        with _cache_lock:
            _spec_cache.pop(ignore_path, None)
        return None

    key = (st.st_mtime_ns, st.st_size)
    with _cache_lock:
        cached = _spec_cache.get(ignore_path)
        if cached is not None and (cached[0], cached[1]) == key:
            return cached[2]

    try:
        import pathspec

        with open(ignore_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        # The parser itself skips comments/blank lines; pre-filtering just
        # lets us cache "no usable patterns" as None.
        usable = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        # GitIgnoreSpec implements real gitignore semantics (negation with
        # last-match-wins, dir patterns); fall back to the older gitwildmatch
        # factory for pathspec versions that predate it.
        if not usable:
            spec = None
        elif hasattr(pathspec, "GitIgnoreSpec"):
            spec = pathspec.GitIgnoreSpec.from_lines(lines)
        else:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except Exception:
        logger.debug("failed to parse %s", ignore_path, exc_info=True)
        spec = None

    with _cache_lock:
        _spec_cache[ignore_path] = (st.st_mtime_ns, st.st_size, spec)
    return spec


def _find_ignore_file(start_dir: str) -> str | None:
    """Walk up from *start_dir* to the git root looking for ``.hermesignore``.

    Returns the ignore file's absolute path, or ``None``. The walk stops
    after checking the first directory that contains ``.git`` (the workspace
    root for our purposes) or on reaching the filesystem root.
    """
    try:
        current = Path(start_dir).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not current.is_dir():
        current = current.parent

    for _ in range(_MAX_WALK_UP):
        candidate = current / IGNORE_FILENAME
        try:
            if candidate.is_file():
                return str(candidate)
            at_git_root = (current / ".git").exists()
        except OSError:
            return None
        if at_git_root:
            return None  # Reached the repo root without finding one.
        parent = current.parent
        if parent == current:
            return None  # Filesystem root.
        current = parent
    return None


def _candidate_start_dirs(resolved_path: str, task_id: str = "default") -> list[str]:
    """Start points for ignore-file discovery, most authoritative first.

    1. The resolved terminal cwd (``_authoritative_workspace_root`` — the
       directory the agent is actually working in, e.g. a git worktree).
    2. The target file's own parent directory — covers absolute-path access
       into a repo the terminal has not cd'ed into yet.
    """
    dirs: list[str] = []
    try:
        from tools.file_tools import _authoritative_workspace_root

        root = _authoritative_workspace_root(task_id)
    except Exception:
        root = None
    if root:
        dirs.append(root)
    else:
        dirs.append(os.getcwd())
    try:
        parent = str(Path(resolved_path).parent)
    except (OSError, ValueError):
        parent = None
    if parent and parent not in dirs:
        dirs.append(parent)
    return dirs


def check_hermesignore(resolved_path: str, task_id: str = "default") -> str | None:
    """Return an error string when *resolved_path* is blocked by ``.hermesignore``.

    *resolved_path* must already be resolved against the task's cwd (callers
    in ``tools/file_tools.py`` pass the output of ``_resolve_path_for_task``).
    Returns ``None`` when no ignore file applies, the path does not match, or
    the ``security.hermesignore_enabled`` kill switch is off.
    """
    if not _enabled():
        return None

    seen_ignore_files: set[str] = set()
    for start_dir in _candidate_start_dirs(resolved_path, task_id):
        ignore_path = _find_ignore_file(start_dir)
        if not ignore_path or ignore_path in seen_ignore_files:
            continue
        seen_ignore_files.add(ignore_path)

        spec = _load_spec(ignore_path)
        if spec is None:
            continue

        root = Path(ignore_path).parent
        try:
            rel = Path(resolved_path).resolve().relative_to(root)
        except (ValueError, OSError, RuntimeError):
            continue  # Target is outside this ignore file's tree.
        rel_posix = rel.as_posix()
        if rel_posix in (".", ""):
            continue
        # Never let the ignore file hide itself — the agent should always be
        # able to read the policy that is blocking it.
        if rel_posix == IGNORE_FILENAME:
            continue
        try:
            # Directory targets (e.g. a search root) must be tested with a
            # trailing slash too — gitignore dir patterns like ``secrets/``
            # only match the slash-suffixed form.
            candidates = [rel_posix]
            try:
                if Path(resolved_path).is_dir():
                    candidates.append(rel_posix + "/")
            except OSError:
                pass
            if any(spec.match_file(c) for c in candidates):
                return (
                    f"Access to '{resolved_path}' is blocked by .hermesignore "
                    f"({ignore_path}). This project excludes the path from agent "
                    "file access. Note: .hermesignore gates file tools only — "
                    "it does not restrict terminal commands or MCP servers."
                )
        except Exception:
            logger.debug("hermesignore match failed for %s", resolved_path, exc_info=True)
            return None
    return None
