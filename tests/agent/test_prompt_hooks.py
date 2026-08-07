"""Tests for prompt hooks — LLM-evaluated natural-language tool policies.

Prompt hooks are ``hooks: pre_tool_call:`` entries carrying ``prompt:``
instead of ``command:``.  The policy text is evaluated against the pending
tool call by the auxiliary LLM (``auxiliary.prompt_hooks``); a non-allow
verdict blocks the tool call with the canonical block shape.

The auxiliary client is mocked throughout — no network calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import agent.auxiliary_client as auxiliary_client
from agent import shell_hooks


# ── helpers ───────────────────────────────────────────────────────────────


def _llm_response(content):
    """Build the minimal shape ``call_llm`` returns."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def _mock_llm(monkeypatch, content=None, exc=None, capture=None):
    def fake_call_llm(**kwargs):
        if capture is not None:
            capture.append(kwargs)
        if exc is not None:
            raise exc
        return _llm_response(content)

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)


def _prompt_spec(**overrides):
    policy = overrides.pop("prompt", "Only allow read-only file operations")
    defaults = dict(
        event="pre_tool_call",
        command=f"{shell_hooks.PROMPT_COMMAND_PREFIX}{policy}",
        prompt=policy,
    )
    defaults.update(overrides)
    return shell_hooks.ShellHookSpec(**defaults)


@pytest.fixture(autouse=True)
def _reset_registration_state():
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()


# ── config parsing ────────────────────────────────────────────────────────


class TestParsePromptEntry:
    def test_valid_prompt_entry(self):
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [
                {
                    "prompt": "Only allow read-only file operations",
                    "matcher": "terminal",
                    "timeout": 30,
                },
            ],
        })
        assert len(specs) == 1
        spec = specs[0]
        assert spec.is_prompt_hook
        assert spec.prompt == "Only allow read-only file operations"
        assert spec.command == (
            shell_hooks.PROMPT_COMMAND_PREFIX
            + "Only allow read-only file operations"
        )
        assert spec.matcher == "terminal"
        assert spec.timeout == 30
        assert spec.fail_closed is False
        assert spec.model is None

    def test_command_and_prompt_mutually_exclusive(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            specs = shell_hooks._parse_hooks_block({
                "pre_tool_call": [
                    {"prompt": "no writes", "command": "/tmp/hook.sh"},
                ],
            })
        assert specs == []
        assert any(
            "mutually exclusive" in r.getMessage() for r in caplog.records
        )

    def test_prompt_hook_rejected_on_non_pre_tool_call_events(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            specs = shell_hooks._parse_hooks_block({
                "post_tool_call": [{"prompt": "no writes"}],
            })
        assert specs == []
        assert any(
            "only supported" in r.getMessage() for r in caplog.records
        )

    def test_model_override_and_fail_closed_parsed(self):
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [
                {
                    "prompt": "no destructive commands",
                    "model": "some-fast-model",
                    "fail_closed": True,
                },
            ],
        })
        assert len(specs) == 1
        assert specs[0].model == "some-fast-model"
        assert specs[0].fail_closed is True

    def test_non_bool_fail_closed_defaults_open(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            specs = shell_hooks._parse_hooks_block({
                "pre_tool_call": [
                    {"prompt": "no writes", "fail_closed": "yes"},
                ],
            })
        assert len(specs) == 1
        assert specs[0].fail_closed is False

    def test_shell_entries_still_parse(self):
        """Regression: the shell-hook path is untouched."""
        specs = shell_hooks._parse_hooks_block({
            "pre_tool_call": [
                {"command": "/tmp/hook.sh", "matcher": "terminal"},
            ],
        })
        assert len(specs) == 1
        assert not specs[0].is_prompt_hook
        assert specs[0].command == "/tmp/hook.sh"


# ── verdict parsing ───────────────────────────────────────────────────────


class TestParsePromptVerdict:
    def test_strict_json(self):
        assert shell_hooks._parse_prompt_verdict('{"ok": true}') == {"ok": True}

    def test_fenced_json(self):
        v = shell_hooks._parse_prompt_verdict(
            '```json\n{"ok": false, "reason": "writes file"}\n```',
        )
        assert v == {"ok": False, "reason": "writes file"}

    def test_embedded_json_blob(self):
        v = shell_hooks._parse_prompt_verdict(
            'Sure! Here is my verdict: {"ok": false, "reason": "nope"} hope that helps',
        )
        assert v == {"ok": False, "reason": "nope"}

    def test_garbage_returns_none(self):
        assert shell_hooks._parse_prompt_verdict("") is None
        assert shell_hooks._parse_prompt_verdict("not json") is None
        assert shell_hooks._parse_prompt_verdict('{"ok": "yes"}') is None
        assert shell_hooks._parse_prompt_verdict('["ok"]') is None


# ── evaluation ────────────────────────────────────────────────────────────


class TestEvaluatePromptHook:
    def test_allow_verdict_returns_none(self, monkeypatch):
        _mock_llm(monkeypatch, content='{"ok": true}')
        spec = _prompt_spec()
        result = shell_hooks._evaluate_prompt_hook(
            spec, {"tool_name": "terminal", "args": {"command": "ls"}},
        )
        assert result is None

    def test_deny_verdict_returns_canonical_block(self, monkeypatch):
        _mock_llm(
            monkeypatch,
            content='{"ok": false, "reason": "rm -rf is not read-only"}',
        )
        spec = _prompt_spec()
        result = shell_hooks._evaluate_prompt_hook(
            spec, {"tool_name": "terminal", "args": {"command": "rm -rf /"}},
        )
        assert result == {
            "action": "block",
            "message": "rm -rf is not read-only",
        }

    def test_deny_without_reason_gets_default_message(self, monkeypatch):
        _mock_llm(monkeypatch, content='{"ok": false}')
        spec = _prompt_spec()
        result = shell_hooks._evaluate_prompt_hook(
            spec, {"tool_name": "terminal", "args": {}},
        )
        assert result["action"] == "block"
        assert spec.prompt in result["message"]

    def test_llm_exception_fails_open_by_default(self, monkeypatch, caplog):
        import logging
        _mock_llm(monkeypatch, exc=RuntimeError("provider down"))
        spec = _prompt_spec()
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            result = shell_hooks._evaluate_prompt_hook(
                spec, {"tool_name": "terminal", "args": {}},
            )
        assert result is None
        assert any("fail-open" in r.getMessage() for r in caplog.records)

    def test_unparseable_verdict_fails_open_by_default(self, monkeypatch, caplog):
        import logging
        _mock_llm(monkeypatch, content="I think that seems fine to me!")
        spec = _prompt_spec()
        with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
            result = shell_hooks._evaluate_prompt_hook(
                spec, {"tool_name": "terminal", "args": {}},
            )
        assert result is None
        assert any("fail-open" in r.getMessage() for r in caplog.records)

    def test_llm_exception_blocks_when_fail_closed(self, monkeypatch):
        _mock_llm(monkeypatch, exc=RuntimeError("provider down"))
        spec = _prompt_spec(fail_closed=True)
        result = shell_hooks._evaluate_prompt_hook(
            spec, {"tool_name": "terminal", "args": {}},
        )
        assert result is not None
        assert result["action"] == "block"
        assert spec.prompt in result["message"]

    def test_unparseable_verdict_blocks_when_fail_closed(self, monkeypatch):
        _mock_llm(monkeypatch, content="garbage")
        spec = _prompt_spec(fail_closed=True)
        result = shell_hooks._evaluate_prompt_hook(
            spec, {"tool_name": "terminal", "args": {}},
        )
        assert result is not None
        assert result["action"] == "block"

    def test_llm_call_shape(self, monkeypatch):
        """The aux call must target the prompt_hooks task, honor the
        per-entry model override and timeout, and embed policy + tool
        call JSON in the user message."""
        calls = []
        _mock_llm(monkeypatch, content='{"ok": true}', capture=calls)
        spec = _prompt_spec(model="tiny-model", timeout=17)
        shell_hooks._evaluate_prompt_hook(
            spec, {"tool_name": "terminal", "args": {"command": "cat x"}},
        )
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["task"] == "prompt_hooks"
        assert kwargs["model"] == "tiny-model"
        assert kwargs["timeout"] == 17.0
        user_msg = kwargs["messages"][1]["content"]
        assert spec.prompt in user_msg
        assert '"tool_name": "terminal"' in user_msg
        assert "cat x" in user_msg

    def test_oversized_tool_args_truncated(self, monkeypatch):
        calls = []
        _mock_llm(monkeypatch, content='{"ok": true}', capture=calls)
        spec = _prompt_spec()
        shell_hooks._evaluate_prompt_hook(
            spec, {"tool_name": "terminal", "args": {"command": "x" * 100_000}},
        )
        user_msg = calls[0]["messages"][1]["content"]
        assert len(user_msg.encode("utf-8")) < 20_000
        assert "…[truncated]" in user_msg


# ── callback wiring ───────────────────────────────────────────────────────


class TestPromptHookCallback:
    def test_matcher_filters_before_llm_call(self, monkeypatch):
        calls = []
        _mock_llm(monkeypatch, content='{"ok": false, "reason": "no"}', capture=calls)
        spec = _prompt_spec(matcher="terminal")
        cb = shell_hooks._make_callback(spec)
        assert cb(tool_name="web_search", args={}) is None
        assert calls == []
        assert cb(tool_name="terminal", args={}) == {
            "action": "block", "message": "no",
        }
        assert len(calls) == 1

    def test_registration_via_config_uses_prompt_command_key(
        self, monkeypatch, tmp_path,
    ):
        """End-to-end: a prompt hook registers through register_from_config
        using the prompt:<policy> command string for the allowlist key, and
        the registered callback blocks per the LLM verdict."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
        _mock_llm(monkeypatch, content='{"ok": false, "reason": "blocked!"}')

        policy = "Never allow network access"
        cfg = {
            "hooks": {
                "pre_tool_call": [{"prompt": policy, "matcher": "terminal"}],
            },
        }
        registered = shell_hooks.register_from_config(cfg, accept_hooks=True)
        assert len(registered) == 1
        spec = registered[0]
        assert spec.is_prompt_hook

        # Allowlist recorded under the synthesised prompt:<policy> command.
        entry = shell_hooks.allowlist_entry_for(
            "pre_tool_call", f"{shell_hooks.PROMPT_COMMAND_PREFIX}{policy}",
        )
        assert entry is not None

        # The wired callback produces the canonical block shape.
        from hermes_cli.plugins import get_plugin_manager
        manager = get_plugin_manager()
        callbacks = manager._hooks.get("pre_tool_call", [])
        results = [
            cb(tool_name="terminal", args={"command": "curl example.com"})
            for cb in callbacks
        ]
        assert {"action": "block", "message": "blocked!"} in results

        # Clean up the manager hook we appended.
        manager._hooks["pre_tool_call"] = [
            cb for cb in manager._hooks.get("pre_tool_call", [])
            if cb not in callbacks or results[callbacks.index(cb)] is None
        ]
