"""Regression: save_csv must write CSV bytes verbatim (no text-mode \n -> \r\n
translation on Windows). GH issue lakshaychhaba0001-sys/openbench-ai#129."""
from __future__ import annotations

from pathlib import Path

from openbench import DatasetLoader, EvaluationConfig, Evaluator, ModelResponseSet, ResponseItem
from openbench.reporting import ReportGenerator

ROOT = Path(__file__).resolve().parents[1]


def _result():
    ds = DatasetLoader.load(str(ROOT / "datasets" / "mathematics" / "sample.jsonl"))
    rs = ModelResponseSet("m", "1", [ResponseItem(id="mathematics_014", response="-9")])
    return Evaluator(config=EvaluationConfig(metric_names=["exact_match"])).evaluate(ds, rs)


def test_save_csv_writes_exact_bytes(tmp_path, monkeypatch):
    """save_csv writes exactly what to_csv produced, with no newline translation."""
    result = _result()
    expected = ReportGenerator.to_csv(result).encode("utf-8")

    # Emulate Windows text mode: open(newline=None) turns every \n into \r\n.
    real_write_text = Path.write_text

    def windows_write_text(self, data, encoding=None, errors=None, newline=None):
        # Path.write_text passes newline straight to open(); with newline=None
        # Python's TextIOWrapper translates "\n" -> os.linesep on Windows.
        if newline is None:
            data = data.replace("\n", "\r\n")
        return real_write_text(self, data, encoding=encoding, errors=errors, newline="")

    monkeypatch.setattr(Path, "write_text", windows_write_text)

    out = ReportGenerator.save_csv(result, tmp_path / "r.csv")
    raw = out.read_bytes()
    assert raw == expected, (raw[:80], expected[:80])


def test_save_csv_passes_newline_empty(tmp_path, monkeypatch):
    """save_csv explicitly disables newline translation."""
    result = _result()
    captured = {}
    real_write_text = Path.write_text

    def spy(self, data, encoding=None, errors=None, newline=None):
        captured["newline"] = newline
        captured["data"] = data
        return real_write_text(self, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", spy)
    ReportGenerator.save_csv(result, tmp_path / "r.csv")
    assert captured["newline"] == ""
