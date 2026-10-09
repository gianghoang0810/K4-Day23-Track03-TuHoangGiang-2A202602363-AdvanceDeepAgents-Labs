import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from agents import REPORT_PATH, SOURCES_PATH
from research import build_prompt, main, save_outputs, slugify, summarize


def test_slugify():
    assert slugify("survey about world model") == "survey-about-world-model"
    assert slugify("../../x") == "x"
    assert slugify("") == "topic"
    assert slugify("   ") == "topic"
    long_topic = "a" * 100
    assert len(slugify(long_topic)) <= 60


def test_build_prompt():
    prompt = build_prompt("world models")
    assert "Topic: world models" in prompt
    assert "Today's date:" in prompt
    assert "write_todos" in prompt


def test_summarize():
    class DummyMessage:
        def __init__(self, tool_calls, usage_metadata):
            self.tool_calls = tool_calls
            self.usage_metadata = usage_metadata

    messages = [
        DummyMessage([{"name": "task"}, {"name": "write_file"}], {"input_tokens": 100, "output_tokens": 50}),
        DummyMessage([{"name": "task"}], {"input_tokens": 200, "output_tokens": 80}),
        {"role": "assistant", "content": "hello"},  # message without tool calls
    ]
    summary = summarize(messages, 12.345, "test-model")
    assert summary["model"] == "test-model"
    assert summary["elapsed_s"] == 12.3
    assert summary["subagent_calls"] == 2
    assert summary["tool_calls"] == {"task": 2, "write_file": 1}
    assert summary["tokens"] == {"input": 300, "output": 130}


def test_save_outputs_failures(tmp_path, monkeypatch):
    class FakeBackend:
        pass

    backend = FakeBackend()

    # Report missing
    monkeypatch.setattr("research.download", lambda b, paths: {REPORT_PATH: None, SOURCES_PATH: b"[]"})
    with pytest.raises(RuntimeError, match="report.md missing"):
        save_outputs(backend, "topic", [], 1.0, "m", reports_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 0

    # Sources invalid json
    monkeypatch.setattr("research.download", lambda b, paths: {REPORT_PATH: b"# Title\n", SOURCES_PATH: b"not json"})
    with pytest.raises(RuntimeError, match="sources.json is not valid JSON"):
        save_outputs(backend, "topic", [], 1.0, "m", reports_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 0

    # Sources empty
    monkeypatch.setattr("research.download", lambda b, paths: {REPORT_PATH: b"# Title\n", SOURCES_PATH: b"[]"})
    with pytest.raises(RuntimeError, match="sources.json must be a non-empty JSON list"):
        save_outputs(backend, "topic", [], 1.0, "m", reports_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 0


def test_save_outputs_success(tmp_path, monkeypatch):
    class FakeBackend:
        pass

    backend = FakeBackend()
    rep_bytes = b"# Survey Title\n## TL;DR\n- Point [1]\n"
    src_list = [
        {"n": 1, "url": "https://arxiv.org/abs/2401.0001", "source": "arxiv"},
        {"n": 2, "url": "https://example.com/web", "source": "web"},
    ]
    src_bytes = json.dumps(src_list).encode("utf-8")

    monkeypatch.setattr("research.download", lambda b, paths: {REPORT_PATH: rep_bytes, SOURCES_PATH: src_bytes})
    path = save_outputs(backend, "survey about world model", [], 5.5, "test-model", reports_dir=tmp_path)

    slug = "survey-about-world-model"
    assert path == str(tmp_path / f"{slug}.md")
    assert (tmp_path / f"{slug}.md").read_bytes() == rep_bytes
    assert (tmp_path / f"{slug}.sources.json").read_bytes() == src_bytes

    meta = json.loads((tmp_path / f"{slug}.meta.json").read_text(encoding="utf-8"))
    assert meta["topic"] == "survey about world model"
    assert meta["source_families"] == ["arxiv", "web"]
    assert meta["n_sources"] == 2


def test_main_empty_topic(monkeypatch):
    def fail_sandbox():
        raise AssertionError("Sandbox should not be opened")

    monkeypatch.setattr("research.open_sandbox", fail_sandbox)
    assert main("") == 2
    assert main("   ") == 2
