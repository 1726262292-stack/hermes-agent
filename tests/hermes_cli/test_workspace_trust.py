"""Tests for the per-workspace trust store (hermes_cli/workspace_trust.py).

Ported from superagent-ai/grok-cli src/utils/workspace-trust.ts.
"""

import json
import os
import stat

import pytest

from hermes_cli import workspace_trust as wt


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point the trust store at a temp HERMES_HOME and clear session state."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(wt, "_SESSION_DECISIONS", {})
    yield home


def _store_path():
    return wt.get_workspace_trust_path()


# ---------------------------------------------------------------------------
# Store round-trip + permissions
# ---------------------------------------------------------------------------

def test_roundtrip_persisted_decision(tmp_path):
    ws = tmp_path / "project"
    ws.mkdir()
    assert wt.get_workspace_trust(str(ws)) is None

    key = wt.set_workspace_trust(str(ws), trusted=True, remember=True)
    assert key == os.path.realpath(str(ws))
    assert wt.get_workspace_trust(str(ws)) == "trusted"

    wt.set_workspace_trust(str(ws), trusted=False, remember=True)
    assert wt.get_workspace_trust(str(ws)) == "untrusted"

    # On-disk schema is versioned and entry carries a decidedAt timestamp.
    raw = json.loads(_store_path().read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["workspaces"][key]["trusted"] is False
    assert raw["workspaces"][key]["decidedAt"]


def test_store_file_written_with_0600_perms(tmp_path):
    ws = tmp_path / "project"
    ws.mkdir()
    wt.set_workspace_trust(str(ws), trusted=True, remember=True)
    mode = stat.S_IMODE(os.stat(_store_path()).st_mode)
    assert mode == 0o600


def test_remove_workspace_trust(tmp_path):
    ws = tmp_path / "project"
    ws.mkdir()
    wt.set_workspace_trust(str(ws), trusted=True, remember=True)
    assert wt.remove_workspace_trust(str(ws)) is True
    assert wt.get_workspace_trust(str(ws)) is None
    assert wt.remove_workspace_trust(str(ws)) is False


# ---------------------------------------------------------------------------
# Tolerant loader
# ---------------------------------------------------------------------------

def test_corrupt_store_file_yields_empty_store(tmp_path):
    _store_path().write_text("{not json!!", encoding="utf-8")
    assert wt.load_workspace_trust_store() == {"version": 1, "workspaces": {}}
    # And a fresh decision can still be written over the corrupt file.
    ws = tmp_path / "project"
    ws.mkdir()
    wt.set_workspace_trust(str(ws), trusted=True, remember=True)
    assert wt.get_workspace_trust(str(ws)) == "trusted"


def test_malformed_entries_dropped_valid_entries_kept():
    _store_path().write_text(json.dumps({
        "version": 1,
        "workspaces": {
            "/good": {"trusted": True, "decidedAt": "2026-01-01T00:00:00Z"},
            "/bad-type": {"trusted": "yes"},
            "/not-a-dict": 42,
            "/missing-decided": {"trusted": False},
        },
    }), encoding="utf-8")
    store = wt.load_workspace_trust_store()
    assert set(store["workspaces"]) == {"/good", "/missing-decided"}
    assert store["workspaces"]["/missing-decided"]["decidedAt"] == ""


# ---------------------------------------------------------------------------
# Keying: realpath + git root
# ---------------------------------------------------------------------------

def test_key_resolves_symlinks(tmp_path):
    real = tmp_path / "real-project"
    real.mkdir()
    link = tmp_path / "link-project"
    link.symlink_to(real)
    assert wt.get_workspace_trust_key(str(link)) == os.path.realpath(str(real))
    wt.set_workspace_trust(str(link), trusted=False, remember=True)
    assert wt.get_workspace_trust(str(real)) == "untrusted"


def test_key_uses_git_root_for_subdirectories(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    assert wt.get_workspace_trust_key(str(sub)) == os.path.realpath(str(repo))
    wt.set_workspace_trust(str(sub), trusted=True, remember=True)
    assert wt.get_workspace_trust(str(repo)) == "trusted"


def test_key_falls_back_to_dir_outside_git(tmp_path):
    ws = tmp_path / "no-repo"
    ws.mkdir()
    assert wt.get_workspace_trust_key(str(ws)) == os.path.realpath(str(ws))


# ---------------------------------------------------------------------------
# Session-only decisions
# ---------------------------------------------------------------------------

def test_session_only_decision_not_persisted(tmp_path):
    ws = tmp_path / "project"
    ws.mkdir()
    wt.set_workspace_trust(str(ws), trusted=True, remember=False)
    assert wt.get_workspace_trust(str(ws)) == "trusted"
    assert not _store_path().exists()


def test_session_decision_overrides_persisted(tmp_path):
    ws = tmp_path / "project"
    ws.mkdir()
    wt.set_workspace_trust(str(ws), trusted=True, remember=True)
    wt.set_workspace_trust(str(ws), trusted=False, remember=False)
    assert wt.get_workspace_trust(str(ws)) == "untrusted"
    # Disk still says trusted — session state never leaked to the store.
    raw = json.loads(_store_path().read_text(encoding="utf-8"))
    key = wt.get_workspace_trust_key(str(ws))
    assert raw["workspaces"][key]["trusted"] is True


# ---------------------------------------------------------------------------
# Tighten-only approval interaction
# ---------------------------------------------------------------------------

def test_untrusted_latch_forces_manual_over_looser_modes(monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_UNTRUSTED_WORKSPACE_LATCH", False)
    monkeypatch.setattr(approval, "_get_approval_config", lambda: {"mode": "off"})
    assert approval._get_approval_mode() == "off"

    approval.enforce_untrusted_workspace()
    assert approval._get_approval_mode() == "manual"

    # Tighten-only: even a smart config reads as manual once latched.
    monkeypatch.setattr(approval, "_get_approval_config", lambda: {"mode": "smart"})
    assert approval._get_approval_mode() == "manual"


def test_latch_untouched_leaves_configured_mode(monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_UNTRUSTED_WORKSPACE_LATCH", False)
    monkeypatch.setattr(approval, "_get_approval_config", lambda: {"mode": "smart"})
    assert approval._get_approval_mode() == "smart"


# ---------------------------------------------------------------------------
# Startup gate behavior
# ---------------------------------------------------------------------------

def test_prompt_disabled_by_default_is_noop(tmp_path, monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_UNTRUSTED_WORKSPACE_LATCH", False)
    monkeypatch.setattr(wt, "_workspace_trust_prompt_enabled", lambda: False)
    called = []
    monkeypatch.setattr("builtins.input", lambda *_: called.append(1) or "y")
    wt.maybe_prompt_workspace_trust()
    assert called == []
    assert approval._UNTRUSTED_WORKSPACE_LATCH is False


def test_recorded_untrusted_workspace_arms_latch(tmp_path, monkeypatch):
    from tools import approval

    ws = tmp_path / "project"
    ws.mkdir()
    monkeypatch.setattr(approval, "_UNTRUSTED_WORKSPACE_LATCH", False)
    monkeypatch.setattr(wt, "_workspace_trust_prompt_enabled", lambda: True)
    monkeypatch.chdir(ws)
    wt.set_workspace_trust(str(ws), trusted=False, remember=True)

    wt.maybe_prompt_workspace_trust()
    assert approval._UNTRUSTED_WORKSPACE_LATCH is True
    assert approval._get_approval_mode() == "manual"


def test_home_directory_never_prompted(monkeypatch):
    monkeypatch.setattr(wt, "_workspace_trust_prompt_enabled", lambda: True)
    monkeypatch.setattr(
        wt, "get_workspace_trust_key",
        lambda path=None: os.path.realpath(os.path.expanduser("~")),
    )
    called = []
    monkeypatch.setattr("builtins.input", lambda *_: called.append(1) or "y")
    wt.maybe_prompt_workspace_trust()
    assert called == []
