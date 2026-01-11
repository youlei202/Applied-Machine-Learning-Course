"""Small helper utilities used across the course notebooks.

The goal of this module is to keep notebooks clean and focused on concepts.

Notes
-----
- Functions are intentionally lightweight and dependency-minimal.
- Notebooks typically add `./src` to `sys.path` so importing works even without
  installing the package.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Find the repository root by walking upwards.

    The function looks for a `pyproject.toml` file (preferred) or a `slides/`
    folder (fallback) to identify the project root.

    Parameters
    ----------
    start:
        Directory to start searching from. Defaults to current working directory.

    Returns
    -------
    pathlib.Path
        The detected repository root. If no marker is found, returns `start`.
    """

    start = (start or Path.cwd()).resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists() and (p / "src").exists():
            return p
        if (p / "slides").exists() and (p / "notebooks").exists():
            return p
    return start


def ensure_dir(path: Path) -> Path:
    """Create a directory if it doesn't exist and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    # Torch is optional in this course repo.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_data_from_google_drive(url: str, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV from a Google Drive *share link*.

    This helper converts a link of the form:
        https://drive.google.com/file/d/<FILE_ID>/view?... 
    into the direct-download link:
        https://drive.google.com/uc?id=<FILE_ID>

    Parameters
    ----------
    url:
        Google Drive share link.
    read_csv_kwargs:
        Extra kwargs forwarded to `pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Parsed CSV.
    """

    file_id = url.split("/")[-2]
    url_processed = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url_processed, **read_csv_kwargs)


def chdir_to_repo_root() -> Path:
    """Change the current working directory to the repository root.

    Returns
    -------
    pathlib.Path
        The repository root.
    """

    root = find_repo_root()
    os.chdir(root)
    return root
