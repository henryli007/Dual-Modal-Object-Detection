from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
DOCS_ROOT = PROJECT_ROOT / "docs"
LOG_ROOT = PROJECT_ROOT / "logs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"


def ensure_project_dirs() -> None:
    for path in [
        DATA_ROOT / "raw",
        DATA_ROOT / "interim",
        DATA_ROOT / "processed",
        DATA_ROOT / "external",
        DATA_ROOT / "submissions",
        LOG_ROOT,
        OUTPUT_ROOT / "figures",
        OUTPUT_ROOT / "models",
        OUTPUT_ROOT / "reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)
