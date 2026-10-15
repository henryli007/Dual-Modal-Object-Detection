from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


def load_toml(path: str | Path) -> dict[str, Any]:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))

