"""
.env autoloader (import for the side effect, like _paths).

The runtime's secret (GEMINI_API_KEY) and its optional endpoint override
(MOTION_SERVICE_URL) live in the gitignored .env at the repo root. uv can
inject those with `--env-file .env`, but that flag has to be retyped on
every invocation, and forgetting it fails late and confusingly: the run
starts, the lamp breathes, and the missing key only surfaces as a dialogue
error seconds into the first turn. Importing this module loads .env into
os.environ instead, so a plain `uv run runtime/main.py --converse` works.

Real environment variables always win -- a name already present in
os.environ is left untouched -- so `--env-file`, an exported shell
variable, and CI-injected secrets all still override the file, and this
module only ever fills in what nothing else supplied.

Format: `KEY=value` per line, `#` comment lines and blanks skipped, an
optional `export ` prefix allowed, and matching surrounding quotes
stripped. Values are taken literally to the end of the line (no inline
`#` comment stripping), so a key containing `#` survives intact.
Malformed lines are skipped rather than raising: a stray line in a
personal .env should not take down the robot at startup.
"""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def load(path=ENV_FILE, override=False):
    """Load `path` into os.environ. Returns the names actually set."""
    try:
        text = pathlib.Path(path).read_text()
    except (OSError, UnicodeDecodeError):
        return []                      # absent .env is the normal case

    applied = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue                   # not a KEY=value line
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied


APPLIED = load()
