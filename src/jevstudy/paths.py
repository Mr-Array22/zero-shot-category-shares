"""Project locations. Set JEV_PROJECT_ROOT to override (e.g. for a non-editable install)."""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("JEV_PROJECT_ROOT", PACKAGE_DIR.parents[1])).resolve()

DATA_DIR = PROJECT_ROOT / "data"
FROZEN_DATA = DATA_DIR / "frozen"            # frozen study/request inputs copied into each new run
RAW_ARCHIVE = DATA_DIR / "raw" / "kaggle-selener-v1.zip"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFINITIONS_DIR = PROJECT_ROOT / "definitions"
PREREGISTRATION = PROJECT_ROOT / "docs" / "PREREGISTRATION.md"
ENV_FILE = PROJECT_ROOT / ".env"
