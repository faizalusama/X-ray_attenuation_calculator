"""Batch output remains machine-readable; invalid files fail without a traceback."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "launch.py"), *map(str, arguments)],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)


def read_finite_json(text):
    def invalid_constant(value):
        pytest.fail(f"Non-finite JSON number in batch output: {value}")

    return json.loads(text, parse_constant=invalid_constant)


def test_batch_example_has_full_input_and_finite_outputs(tmp_path):
    output = tmp_path / "result.json"
    process = run_cli("--calculate", ROOT / "examples/borosilicate-multilayer.json",
                      "--output", output)
    assert process.returncode == 0, process.stderr
    assert process.stdout == ""
    result = read_finite_json(output.read_text(encoding="utf-8"))
    assert 0 < result["reference"]["transmission"] < 1
    assert len(result["configuration"]["layers"]) == 2
    assert len(result["provenance"]["configuration_sha256"]) == 64
    assert result["spectrum"] is not None
    assert result["uncertainty"] is not None


def test_batch_accepts_bare_configuration_and_stdout(tmp_path):
    example = json.loads((ROOT / "examples/borosilicate-multilayer.json").read_text())
    configuration = example["configuration"]
    configuration["uncertainty"]["enabled"] = False
    source = tmp_path / "configuration.json"
    source.write_text(json.dumps(configuration), encoding="utf-8-sig")
    process = run_cli("--calculate", source)
    assert process.returncode == 0, process.stderr
    result = read_finite_json(process.stdout)
    assert result["configuration"] == configuration


@pytest.mark.parametrize("document", ["[]", "null", "{broken", '{"ignored": NaN}',
                                     '{"configuration": {"energy": {"min_keV": 900}}}'])
def test_bad_input_is_a_clear_cli_error(tmp_path, document):
    source = tmp_path / "invalid.json"
    source.write_text(document, encoding="utf-8")
    process = run_cli("--calculate", source)
    assert process.returncode == 2
    assert process.stdout == ""
    assert "Calculation failed:" in process.stderr
    assert "Traceback" not in process.stderr


def test_missing_file_is_a_clear_cli_error(tmp_path):
    process = run_cli("--calculate", tmp_path / "missing.json")
    assert process.returncode == 2
    assert "Calculation failed:" in process.stderr
    assert "Traceback" not in process.stderr


def test_output_flag_requires_batch_mode(tmp_path):
    process = run_cli("--output", tmp_path / "unused.json", "--no-browser")
    assert process.returncode == 2
    assert "--output requires --calculate" in process.stderr
