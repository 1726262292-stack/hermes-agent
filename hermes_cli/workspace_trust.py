"""Per-workspace trust store.

Ported from superagent-ai/grok-cli (src/utils/workspace-trust.ts): grok-cli
records a per-workspace sandbox decision on first run, keyed by the realpath
of the working directory, in a versioned JSON file under the user's config
home. The Hermes port keeps the same store shape (versioned schema, tolerant
loader, 0600 perms) but maps the decision onto Hermes' approval posture:

- A workspace marked **untrusted** forces the terminal approval mode to
  ``manual`` for the session via a one-way latch in ``tools/approval.py``
  (:func:`tools.approval.enforce_untrusted_workspace`). The latch only ever
  *tightens* — a trusted workspace never weakens the configured mode, and an
  untrusted one can never end up looser than the profile config.
- Decisions are keyed by ``os.path.realpath`` of the workspace **root**: the
  enclosing git repository root when inside one, else the directory itself.
  This mirrors grok-cli's realpath keying while making every subdirectory of
  a repo share one decision.
- Session-only decisions ("s" at the prompt) are kept in-process and never
  written to disk, matching grok-cli's ``remember: false`` path.

The prompt itself is opt-in via ``security.workspace_trust_prompt`` (default
false) and only runs in the plain interactive CLI path — TUI and gateway
sessions are unaffected.

Store location: ``get_hermes_home()/workspace-trust.json`` (0600).
Schema: ``{"version": 1, "workspaces": {"<realpath>": {"trusted": bool,
"decidedAt": "<iso8601>"}}}``. Malformed files and malformed entries are
silently ignored (fresh empty store), matching grok-cli's tolerant loader.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

WORKSPACE_TRUST_FILENAME = "workspace-trust.json"

# In-process, session-only decisions (never persisted). Keyed like the store.
_SESSION_DECISIONS: Dict[str, bool] = {}


def get_workspace_trust_path() -> Path:
    """Return the trust store path under the profile-aware Hermes home."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / WORKSPACE_TRUST_FILENAME


def _find_git_root(start: Path) -> Optional[Path]:
    """Walk *start* and its parents looking for a ``.git`` directory.

    Same pattern as ``agent/prompt_builder.py``. Returns the directory
    containing ``.git``, or ``None`` when the filesystem root is reached.
    """
    try:
        current = start.resolve()
    except OSError:
        return None
    for parent in [current, *current.parents]:
        try:
            if (parent / ".git").exists():
                return parent
        except OSError:
            continue
    return None


def get_workspace_trust_key(path: Optional[str] = None) -> str:
    """Return the store key for *path* (default: cwd).

    The key is the realpath of the workspace root — the enclosing git repo
    root when inside one, else the directory itself.
    """
    base = Path(path or os.getcwd())
    root = _find_git_root(base) or base
    return os.path.realpath(str(root))


def _normalize_entry(value: object) -> Optional[dict]:
    """Return a normalized ``{"trusted", "decidedAt"}`` entry or ``None``."""
    if not isinstance(value, dict):
        return None
    trusted = value.get("trusted")
    if not isinstance(trusted, bool):
        return None
    decided_at = value.get("decidedAt")
    return {
        "trusted": trusted,
        "decidedAt": decided_at if isinstance(decided_at, str) else "",
    }


def load_workspace_trust_store(store_path: Optional[Path] = None) -> dict:
    """Load the trust store, tolerating a missing or corrupt file.

    Malformed JSON yields a fresh empty store; malformed individual entries
    are dropped while valid siblings are kept (grok-cli parity).
    """
    path = store_path or get_workspace_trust_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": 1, "workspaces": {}}
    raw_workspaces = raw.get("workspaces") if isinstance(raw, dict) else None
    workspaces: Dict[str, dict] = {}
    if isinstance(raw_workspaces, dict):
        for workspace, entry in raw_workspaces.items():
            normalized = _normalize_entry(entry)
            if normalized is not None and isinstance(workspace, str):
                workspaces[workspace] = normalized
    return {"version": 1, "workspaces": workspaces}


def _save_store(store: dict, store_path: Optional[Path] = None) -> None:
    """Write the store with owner-only permissions (0600 file, 0700 dir)."""
    path = store_path or get_workspace_trust_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    os.replace(str(tmp), str(path))
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass


def get_workspace_trust(path: Optional[str] = None) -> Optional[str]:
    """Return ``'trusted'``, ``'untrusted'``, or ``None`` for a workspace.

    Session-only (in-process) decisions take precedence over persisted ones.
    """
    key = get_workspace_trust_key(path)
    if key in _SESSION_DECISIONS:
        return "trusted" if _SESSION_DECISIONS[key] else "untrusted"
    entry = load_workspace_trust_store()["workspaces"].get(key)
    if entry is None:
        return None
    return "trusted" if entry["trusted"] else "untrusted"


def set_workspace_trust(path: Optional[str], trusted: bool, remember: bool = True) -> str:
    """Record a trust decision for a workspace. Returns the store key used.

    ``remember=False`` keeps the decision in-process only (session-only),
    never touching the on-disk store.
    """
    key = get_workspace_trust_key(path)
    if not remember:
        _SESSION_DECISIONS[key] = trusted
        return key
    store = load_workspace_trust_store()
    store["workspaces"][key] = {
        "trusted": trusted,
        "decidedAt": datetime.now(timezone.utc).isoformat(),
    }
    _save_store(store)
    return key


def remove_workspace_trust(path: Optional[str] = None) -> bool:
    """Forget the persisted decision for a workspace. Returns True if removed."""
    key = get_workspace_trust_key(path)
    _SESSION_DECISIONS.pop(key, None)
    store = load_workspace_trust_store()
    if key not in store["workspaces"]:
        return False
    del store["workspaces"][key]
    _save_store(store)
    return True


# ---------------------------------------------------------------------------
# Interactive CLI startup hook
# ---------------------------------------------------------------------------

def _workspace_trust_prompt_enabled() -> bool:
    """Return the opt-in ``security.workspace_trust_prompt`` config flag."""
    try:
        from hermes_cli.config import load_config_readonly

        security = load_config_readonly().get("security", {}) or {}
        return bool(security.get("workspace_trust_prompt", False))
    except Exception:
        return False


def maybe_prompt_workspace_trust() -> None:
    """First-run workspace trust gate for the plain interactive CLI.

    Called from ``cmd_chat`` before the CLI is constructed. No-op unless
    ``security.workspace_trust_prompt`` is true (opt-in — default behavior is
    unchanged). When the current workspace has a recorded **untrusted**
    decision, arms the tighten-only manual-approvals latch. When there is no
    recorded decision (and the workspace is not the user's home directory),
    prompts::

        Trust this workspace? [y]es / [n]o (restricted) / [s]ession-only

    - y: trusted, persisted — configured approval mode applies as-is.
    - n: untrusted, persisted — approvals forced to manual this session and
      every future session in this workspace (tighten-only).
    - s: trusted for this process only; nothing is written to disk.

    A declined/EOF prompt is treated as session-only untrusted: approvals are
    forced to manual for this run but nothing is persisted.
    """
    if not _workspace_trust_prompt_enabled():
        return

    key = get_workspace_trust_key()
    if key == os.path.realpath(os.path.expanduser("~")):
        # Never prompt for the home directory itself — it isn't a project
        # workspace, and marking $HOME untrusted would catch every shell.
        return

    decision = get_workspace_trust()
    if decision == "trusted":
        return
    if decision == "untrusted":
        from tools.approval import enforce_untrusted_workspace

        enforce_untrusted_workspace()
        print("⚠ Untrusted workspace — approvals forced to manual for this session.")
        return

    try:
        if not (os.isatty(0) and os.isatty(1)):
            return
    except OSError:
        return

    print(f"\nNew workspace: {key}")
    try:
        answer = input("Trust this workspace? [y]es / [n]o (restricted) / [s]ession-only: ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    normalized = answer.strip().lower()

    if normalized in {"y", "yes"}:
        set_workspace_trust(None, trusted=True, remember=True)
        return
    if normalized in {"s", "session"}:
        set_workspace_trust(None, trusted=True, remember=False)
        return

    # "n", empty, or anything else: restricted. Persist only an explicit no.
    remember = normalized in {"n", "no"}
    set_workspace_trust(None, trusted=False, remember=remember)
    from tools.approval import enforce_untrusted_workspace

    enforce_untrusted_workspace()
    print("⚠ Workspace restricted — approvals forced to manual for this session.")


# ---------------------------------------------------------------------------
# ``hermes trust`` subcommand
# ---------------------------------------------------------------------------

def cmd_trust(args) -> None:
    """Entry point for ``hermes trust [list|set|remove]``."""
    action = getattr(args, "trust_command", None) or "list"

    if action == "list":
        store = load_workspace_trust_store()
        workspaces = store["workspaces"]
        if not workspaces:
            print("No workspace trust decisions recorded.")
            return
        for workspace in sorted(workspaces):
            entry = workspaces[workspace]
            label = "trusted" if entry["trusted"] else "untrusted"
            decided = entry.get("decidedAt") or "unknown"
            print(f"{label:>9}  {workspace}  ({decided})")
        return

    if action == "set":
        path = getattr(args, "path", None) or os.getcwd()
        if not os.path.isdir(path):
            print(f"Not a directory: {path}")
            raise SystemExit(1)
        trusted = bool(getattr(args, "trusted", False))
        key = set_workspace_trust(path, trusted=trusted, remember=True)
        print(f"Marked {'trusted' if trusted else 'untrusted'}: {key}")
        return

    if action == "remove":
        path = getattr(args, "path", None) or os.getcwd()
        key = get_workspace_trust_key(path)
        if remove_workspace_trust(path):
            print(f"Removed trust decision for {key}")
        else:
            print(f"No trust decision recorded for {key}")
        return

    print(f"Unknown trust action: {action}")
    raise SystemExit(1)
