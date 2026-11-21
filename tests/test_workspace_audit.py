"""Report/CLI contracts; the live Playwright audit checks the actual interface."""
from __future__ import annotations

import json
from typing import Any

import pytest

from tools import workspace_ui_audit as audit


def test_failure_count_includes_browser_errors_and_does_not_count_observations() -> None:
    checks: list[dict[str, Any]] = []
    audit.record(checks, "default-four", False, {"actual": 11})
    audit.record(checks, "mode", True, {"inputs": 25})
    report = {"scenarios": {"phone-dark": {"checks": checks, "page_errors": ["Uncaught TypeError"]}}}
    assert audit.failure_count(report) == 2
    assert checks[0]["details"] == {"actual": 11}


@pytest.mark.parametrize("report_only,expected", [(False, 1), (True, 0)])
def test_cli_records_failure_and_always_stops_server(monkeypatch: Any, tmp_path: Any,
                                                   report_only: bool, expected: int) -> None:
    stopped: list[Any] = []

    class Server:
        def terminate(self) -> None:
            stopped.append("terminate")

        def wait(self, timeout: int) -> None:
            stopped.append(timeout)

    monkeypatch.setattr(audit.subprocess, "Popen", lambda *_args, **_kwargs: Server())
    monkeypatch.setattr(audit, "_free_port", lambda: 12345)
    monkeypatch.setattr(audit, "_wait_for", lambda _url: None)
    report = {"scenarios": {"desktop-light": {"checks": [
        {"name": "fresh-default-four", "passed": False, "details": 11}], "page_errors": []}}}
    monkeypatch.setattr(audit, "audit", lambda _url, _screenshots: report)
    path = tmp_path / "nested" / "audit.json"
    args = ["--report", str(path)]
    if report_only:
        args.append("--report-only")
    assert audit.main(args) == expected
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert stopped == ["terminate", 20]


def test_cli_stops_server_when_readiness_fails(monkeypatch: Any, tmp_path: Any) -> None:
    stopped: list[Any] = []

    class Server:
        def terminate(self) -> None:
            stopped.append("terminate")

        def wait(self, timeout: int) -> None:
            stopped.append(timeout)

    def unavailable(url: str) -> None:
        raise SystemExit("Workbench did not start")

    monkeypatch.setattr(audit.subprocess, "Popen", lambda *_args, **_kwargs: Server())
    monkeypatch.setattr(audit, "_free_port", lambda: 12345)
    monkeypatch.setattr(audit, "_wait_for", unavailable)
    with pytest.raises(SystemExit, match="Workbench did not start"):
        audit.main(["--report", str(tmp_path / "audit.json")])
    assert stopped == ["terminate", 20]

def test_persistent_project_ignores_only_dom_ids_and_render_revision() -> None:
    project = {"configuration": {"thickness": 1.25}, "ui": {"dashboard": {
        "revision": 4, "panels": [{"id": 8, "primary": "mass", "ranges": {"xaxis": [1, 2]},
                                  "visibility": {"line": False}}]}}}
    before = json.dumps(project)
    restored = json.loads(before)
    restored["ui"]["dashboard"]["revision"] = 0
    restored["ui"]["dashboard"]["panels"][0]["id"] = 1
    assert audit.persistent_project(restored) == audit.persistent_project(project)
    for section, field, value in [
        ("configuration", "thickness", 2),
        ("panel", "primary", "linear"),
        ("panel", "ranges", {"xaxis": [3, 4]}),
        ("panel", "visibility", {"line": True}),
    ]:
        changed = json.loads(json.dumps(restored))
        target = changed["configuration"] if section == "configuration" else changed["ui"]["dashboard"]["panels"][0]
        target[field] = value
        assert audit.persistent_project(changed) != audit.persistent_project(project)
    assert json.dumps(project) == before
