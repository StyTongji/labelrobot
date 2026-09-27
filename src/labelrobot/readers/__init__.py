from __future__ import annotations

import os
from pathlib import Path

from .base import DatasetReader, load_json
from .v2 import V2Reader
from .v3 import V3Reader


def open_dataset(root: str | Path) -> DatasetReader:
    source = Path(root).expanduser().resolve(strict=True)
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Not a LeRobot dataset: missing {info_path}")
    version = str(load_json(info_path).get("codebase_version", ""))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    if version.startswith("v3") or version.startswith("3"):
        return V3Reader(source)
    if version.startswith("v2") or version.startswith("2"):
        return V2Reader(source)
    raise ValueError(f"Unsupported LeRobot dataset version: {version or 'unknown'}")


__all__ = ["DatasetReader", "open_dataset"]
