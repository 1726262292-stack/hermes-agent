"""Tests for project-level `.hermesignore` agent file-access blocking.

`.hermesignore` is a gitignore-syntax file at the workspace root that blocks
the agent's file tools (read_file / write_file / patch / search_files) from
touching matched paths. See ``tools/hermesignore.py``.
"""

import json
import os
import time
from pathlib import Path

import pytest

import tools.file_tools as file_tools
import tools.hermesignore as hermesignore
from tools.hermesignore import check_hermesignore


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Force the guard on and clear the spec cache around every test."""
    monkeypatch.setattr(hermesignore, "_enabled", lambda: True)
    hermesignore._spec_cache.clear()
    yield
    hermesignore._spec_cache.clear()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake workspace root (with .git) that discovery resolves to."""
    root = tmp_path.resolve() / "repo"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        file_tools, "_authoritative_workspace_root", lambda task_id="default": str(root)
    )
    return root


def _write_ignore(root: Path, text: str) -> Path:
    ignore = root / ".hermesignore"
    ignore.write_text(text, encoding="utf-8")
    return ignore


class TestMatching:
    def test_no_ignore_file_allows_everything(self, repo):
        target = repo / "secrets.txt"
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is None

    def test_glob_pattern_blocks_matching_file(self, repo):
        _write_ignore(repo, "*.pem\n")
        target = repo / "server.pem"
        target.write_text("x", encoding="utf-8")
        err = check_hermesignore(str(target))
        assert err is not None
        assert ".hermesignore" in err

    def test_non_matching_file_allowed(self, repo):
        _write_ignore(repo, "*.pem\n")
        target = repo / "readme.md"
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is None

    def test_comments_and_blank_lines_skipped(self, repo):
        _write_ignore(repo, "# a comment\n\n   \n# *.md would match if not a comment\n")
        target = repo / "notes.md"
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is None

    def test_trailing_slash_directory_pattern(self, repo):
        _write_ignore(repo, "secrets/\n")
        target = repo / "secrets" / "key.txt"
        target.parent.mkdir()
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is not None

    def test_bare_name_blocks_directory_subtree(self, repo):
        _write_ignore(repo, "build\n")
        target = repo / "build" / "out" / "artifact.bin"
        target.parent.mkdir(parents=True)
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is not None

    def test_double_star_glob(self, repo):
        _write_ignore(repo, "docs/**/*.md\n")
        target = repo / "docs" / "internal" / "plan.md"
        target.parent.mkdir(parents=True)
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is not None

    def test_negation_last_match_wins(self, repo):
        _write_ignore(repo, "*.env\n!example.env\n")
        blocked = repo / "prod.env"
        allowed = repo / "example.env"
        blocked.write_text("x", encoding="utf-8")
        allowed.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(blocked)) is not None
        assert check_hermesignore(str(allowed)) is None

    def test_ignore_file_never_hides_itself(self, repo):
        ignore = _write_ignore(repo, "*\n")
        assert check_hermesignore(str(ignore)) is None

    def test_path_outside_workspace_not_affected(self, repo, tmp_path):
        _write_ignore(repo, "*.txt\n")
        outside = tmp_path.resolve() / "elsewhere.txt"
        outside.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(outside)) is None

    def test_discovery_from_target_parent_without_workspace_root(
        self, tmp_path, monkeypatch
    ):
        """Absolute-path access into a repo the terminal never cd'ed into."""
        monkeypatch.setattr(
            file_tools, "_authoritative_workspace_root", lambda task_id="default": None
        )
        monkeypatch.chdir(tmp_path)
        other = tmp_path.resolve() / "other-repo"
        (other / ".git").mkdir(parents=True)
        _write_ignore(other, "*.key\n")
        target = other / "signing.key"
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is not None

    def test_walk_stops_at_git_root(self, tmp_path, monkeypatch):
        """An ignore file ABOVE the git root must not leak into the repo."""
        outer = tmp_path.resolve()
        _write_ignore(outer, "*.txt\n")
        inner = outer / "project"
        (inner / ".git").mkdir(parents=True)
        monkeypatch.setattr(
            file_tools, "_authoritative_workspace_root", lambda task_id="default": str(inner)
        )
        target = inner / "notes.txt"
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is None


class TestKillSwitchAndCache:
    def test_disabled_via_config_allows_everything(self, repo, monkeypatch):
        _write_ignore(repo, "*.pem\n")
        target = repo / "server.pem"
        target.write_text("x", encoding="utf-8")
        monkeypatch.setattr(hermesignore, "_enabled", lambda: False)
        assert check_hermesignore(str(target)) is None

    def test_enabled_reads_security_config(self, monkeypatch):
        monkeypatch.undo()  # drop the autouse _enabled override for this test
        import tools.hermesignore as hi

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"security": {"hermesignore_enabled": False}},
        )
        assert hi._enabled() is False
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
        assert hi._enabled() is True

    def test_cache_invalidated_on_mtime_change(self, repo):
        ignore = _write_ignore(repo, "*.pem\n")
        target = repo / "server.pem"
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is not None

        # Rewrite the ignore file to no longer match; bump mtime explicitly so
        # the (mtime_ns, size) cache key changes even on coarse filesystems.
        ignore.write_text("*.nothing\n", encoding="utf-8")
        future = time.time() + 10
        os.utime(ignore, (future, future))
        assert check_hermesignore(str(target)) is None

    def test_empty_ignore_file_blocks_nothing(self, repo):
        _write_ignore(repo, "# only comments\n\n")
        target = repo / "anything.txt"
        target.write_text("x", encoding="utf-8")
        assert check_hermesignore(str(target)) is None


class TestFileToolChokePoints:
    def test_read_file_tool_refuses_ignored_path(self, repo):
        _write_ignore(repo, "vault/**\n")
        target = repo / "vault" / "creds.txt"
        target.parent.mkdir()
        target.write_text("top secret", encoding="utf-8")
        result = json.loads(file_tools.read_file_tool(str(target)))
        assert ".hermesignore" in result.get("error", "")
        assert "top secret" not in json.dumps(result)

    def test_write_choke_point_refuses_ignored_path(self, repo):
        _write_ignore(repo, "generated/\n")
        target = repo / "generated" / "out.txt"
        target.parent.mkdir()
        err = file_tools._check_sensitive_path(str(target))
        assert err is not None and ".hermesignore" in err

    def test_search_tool_refuses_ignored_root(self, repo):
        _write_ignore(repo, "vault/\n")
        vault = repo / "vault"
        vault.mkdir()
        (vault / "creds.txt").write_text("top secret", encoding="utf-8")
        result = json.loads(
            file_tools.search_tool("secret", target="content", path=str(vault))
        )
        assert ".hermesignore" in result.get("error", "")

    def test_read_file_tool_allows_unmatched_path(self, repo):
        _write_ignore(repo, "vault/**\n")
        target = repo / "open.txt"
        target.write_text("hello world", encoding="utf-8")
        result = json.loads(file_tools.read_file_tool(str(target)))
        assert "error" not in result
        assert "hello world" in result.get("content", "")

    def test_wrapper_fails_open_on_module_error(self, repo, monkeypatch):
        _write_ignore(repo, "*.pem\n")
        target = repo / "server.pem"
        target.write_text("x", encoding="utf-8")
        monkeypatch.setattr(
            hermesignore,
            "check_hermesignore",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert file_tools._check_hermesignore_path(str(target)) is None
