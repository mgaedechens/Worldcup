"""Download raw international football datasets.

Source: https://github.com/martj42/international_results
This dataset is public, license-friendly, and requires no authentication, which
makes the whole pipeline reproducible: anyone who clones the repo can regenerate
``data/raw/`` by running this module.

We deliberately pull from the raw GitHub URLs (not Kaggle) so the download has no
login/API-key dependency.

Run:
    python -m src.data.download
"""

from __future__ import annotations

from pathlib import Path

import requests

# Resolve project root from this file's location (src/data/download.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Base of the canonical dataset repository (main branch).
_BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"

# Files we need. ``results.csv`` is the core dataset (one row per international match).
# ``shootouts.csv`` lets us recover the true winner of matches decided on penalties,
# which matters for knockout simulation later.
DATASETS: dict[str, str] = {
    "results.csv": f"{_BASE_URL}/results.csv",
    "shootouts.csv": f"{_BASE_URL}/shootouts.csv",
}

_TIMEOUT_SECONDS = 30


def download_file(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Download a single file to ``dest``.

    Idempotent by default: if the file already exists we skip the network call,
    so re-running the pipeline is cheap. Pass ``overwrite=True`` to force a refresh.
    """
    if dest.exists() and not overwrite:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size:,} bytes)")
        return dest

    print(f"[get ] {url}")
    response = requests.get(url, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"[ok  ] {dest.name} ({len(response.content):,} bytes)")
    return dest


def download_all(*, overwrite: bool = False) -> list[Path]:
    """Download every dataset declared in ``DATASETS`` into ``data/raw/``."""
    paths = []
    for filename, url in DATASETS.items():
        paths.append(download_file(url, RAW_DIR / filename, overwrite=overwrite))
    return paths


if __name__ == "__main__":
    download_all()
