"""Windows one-folder launcher for the separately built stablevqa.exe."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    executable = root / "stablevqa.exe"
    if not executable.is_file():
        raise SystemExit("stablevqa.exe is missing from this standalone bundle.")
    return subprocess.run([str(executable), *sys.argv[1:]], cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
