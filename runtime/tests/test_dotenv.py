"""The .env autoloader (runtime/_dotenv.py): parsing rules, and the
precedence guarantee that a real environment variable is never clobbered
by the file."""

import os

from runtime import _dotenv


def _write(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text)
    return p


def test_parses_the_documented_forms(tmp_path, monkeypatch):
    monkeypatch.delenv("LAMP_T_PLAIN", raising=False)
    for name in ("LAMP_T_EXPORT", "LAMP_T_QUOTED", "LAMP_T_SQ",
                 "LAMP_T_HASH", "LAMP_T_EMPTY", "LAMP_T_EQ"):
        monkeypatch.delenv(name, raising=False)
    p = _write(tmp_path, "\n".join([
        "# a comment",
        "",
        "LAMP_T_PLAIN=abc123",
        "export LAMP_T_EXPORT=xyz",
        'LAMP_T_QUOTED="spaced value"',
        "LAMP_T_SQ='single'",
        "LAMP_T_HASH=key#notacomment",      # no inline-comment stripping
        "LAMP_T_EMPTY=",
        "LAMP_T_EQ=a=b=c",                  # only the first = splits
        "not a kv line",                    # skipped, does not raise
        "=novalue",                         # skipped
    ]))
    applied = _dotenv.load(p)

    assert os.environ["LAMP_T_PLAIN"] == "abc123"
    assert os.environ["LAMP_T_EXPORT"] == "xyz"
    assert os.environ["LAMP_T_QUOTED"] == "spaced value"
    assert os.environ["LAMP_T_SQ"] == "single"
    assert os.environ["LAMP_T_HASH"] == "key#notacomment"
    assert os.environ["LAMP_T_EMPTY"] == ""
    assert os.environ["LAMP_T_EQ"] == "a=b=c"
    assert "not a kv line" not in applied and "" not in applied

    for name in applied:                    # leave the process env clean
        monkeypatch.delenv(name, raising=False)


def test_real_env_wins_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMP_T_PRECEDENCE", "from-real-env")
    p = _write(tmp_path, "LAMP_T_PRECEDENCE=from-file\n")

    applied = _dotenv.load(p)
    assert os.environ["LAMP_T_PRECEDENCE"] == "from-real-env"
    assert "LAMP_T_PRECEDENCE" not in applied

    _dotenv.load(p, override=True)          # explicit opt-in still works
    assert os.environ["LAMP_T_PRECEDENCE"] == "from-file"


def test_missing_file_is_not_an_error(tmp_path):
    assert _dotenv.load(tmp_path / "definitely-absent") == []


def test_importing_runtime_config_loads_dotenv():
    """The wiring that makes `--env-file .env` unnecessary: importing
    runtime.config must have pulled _dotenv in as a side effect."""
    import sys

    import runtime.config  # noqa: F401
    assert "runtime._dotenv" in sys.modules
