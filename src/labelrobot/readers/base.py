from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pyarrow as pa

from labelrobot.models import DatasetInfo, EpisodeIndex, EpisodeInfo, MediaSpan


def dataset_fingerprint(root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted((root / "meta").rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            hasher.update(relative.encode())
            hasher.update(str(stat.st_size).encode())
            hasher.update(path.read_bytes())
    for directory in (root / "data", root / "videos"):
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    stat = path.stat()
                    hasher.update(path.relative_to(root).as_posix().encode())
                    hasher.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return hasher.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class DatasetReader(ABC):
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve(strict=True)

    @abstractmethod
    def describe(self) -> DatasetInfo: ...

    @abstractmethod
    def list_episodes(self, offset: int = 0, limit: int = 100) -> list[EpisodeInfo]: ...

    @abstractmethod
    def get_episode_index(self, episode_index: int) -> EpisodeIndex: ...

    @abstractmethod
    def read_rows(
        self, episode_index: int, start: int, stop: int, columns: list[str] | None = None
    ) -> pa.Table: ...

    @abstractmethod
    def resolve_media(self, episode_index: int, camera_key: str) -> list[MediaSpan]: ...

    def timestamp(self, episode_index: int, frame_index: int) -> float:
        index = self.get_episode_index(episode_index)
        try:
            position = index.frame_index.index(frame_index)
        except ValueError as error:
            raise ValueError(f"Frame {frame_index} is not in episode {episode_index}") from error
        return index.timestamp_s[position]

    def validate_frame_range(self, episode_index: int, start: int | None, end: int | None) -> None:
        length = self.get_episode_index(episode_index).length
        if start is None or not 0 <= start < length:
            raise ValueError(f"start_frame must be in [0, {length})")
        if end is not None and not start < end <= length:
            raise ValueError(f"end_frame_exclusive must be in ({start}, {length}]")
