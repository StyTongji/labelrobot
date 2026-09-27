from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from labelrobot.models import DatasetInfo, EpisodeIndex, EpisodeInfo, MediaSpan

from .base import DatasetReader, dataset_fingerprint, load_json


def _jsonlines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class V2Reader(DatasetReader):
    def __init__(self, root: str | Path):
        super().__init__(root)
        self.info = load_json(self.root / "meta" / "info.json")
        episode_rows = _jsonlines(self.root / "meta" / "episodes.jsonl")
        if not episode_rows:
            raise ValueError("LeRobot v2 dataset requires meta/episodes.jsonl")
        self._episode_rows = {int(row["episode_index"]): row for row in episode_rows}
        self.tasks = {
            int(row["task_index"]): str(row["task"])
            for row in _jsonlines(self.root / "meta" / "tasks.jsonl")
        }
        self.camera_keys = [
            key
            for key, feature in self.info.get("features", {}).items()
            if feature.get("dtype") in {"video", "image"}
        ]
        self._dataset_version = dataset_fingerprint(self.root)

    def describe(self) -> DatasetInfo:
        return DatasetInfo(
            root=self.root,
            dataset_id=str(self.info.get("repo_id") or self.root.name),
            dataset_version=self._dataset_version,
            format_version=str(self.info.get("codebase_version", "v2.1")),
            fps=float(self.info["fps"]),
            features=self.info.get("features", {}),
            camera_keys=self.camera_keys,
            total_episodes=int(self.info.get("total_episodes", len(self._episode_rows))),
            total_frames=int(self.info.get("total_frames", sum(int(row["length"]) for row in self._episode_rows.values()))),
        )

    def list_episodes(self, offset: int = 0, limit: int = 100) -> list[EpisodeInfo]:
        result = []
        for episode_index in sorted(self._episode_rows)[offset : offset + limit]:
            row = self._episode_rows[episode_index]
            tasks = list(row.get("tasks") or [])
            result.append(EpisodeInfo(episode_index, int(row["length"]), tasks))
        return result

    def _path(self, episode_index: int, video_key: str | None = None) -> Path:
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        template = self.info["video_path" if video_key else "data_path"]
        relative = template.format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=video_key,
            chunk_index=chunk,
            file_index=episode_index,
        )
        return (self.root / relative).resolve(strict=True)

    def _episode_table(self, episode_index: int, columns: list[str] | None = None) -> pa.Table:
        path = self._path(episode_index)
        table = pq.read_table(path, columns=columns)
        if "episode_index" in table.column_names:
            table = table.filter(pc.equal(table["episode_index"], episode_index))
        return table

    def get_episode_index(self, episode_index: int) -> EpisodeIndex:
        if episode_index not in self._episode_rows:
            raise KeyError(f"Unknown episode {episode_index}")
        names = set(pq.read_schema(self._path(episode_index)).names)
        columns = [column for column in ("frame_index", "index", "timestamp") if column in names]
        table = self._episode_table(episode_index, columns).sort_by("frame_index")
        frames = [int(value) for value in table["frame_index"].to_pylist()]
        if frames != list(range(len(frames))):
            raise ValueError(f"Episode {episode_index} has invalid frame_index sequence")
        timestamps = [float(value) for value in table["timestamp"].to_pylist()]
        if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError(f"Episode {episode_index} has non-monotonic timestamps")
        sources = (
            [int(value) for value in table["index"].to_pylist()]
            if "index" in table.column_names
            else frames.copy()
        )
        return EpisodeIndex(episode_index, frames, sources, timestamps)

    def read_rows(
        self, episode_index: int, start: int, stop: int, columns: list[str] | None = None
    ) -> pa.Table:
        self.validate_frame_range(episode_index, start, stop)
        selected = None if columns is None else list(dict.fromkeys([*columns, "frame_index"]))
        table = self._episode_table(episode_index, selected)
        table = table.filter(
            pc.and_(
                pc.greater_equal(table["frame_index"], start),
                pc.less(table["frame_index"], stop),
            )
        ).sort_by("frame_index")
        return table if columns is None else table.select(columns)

    def resolve_media(self, episode_index: int, camera_key: str) -> list[MediaSpan]:
        if camera_key not in self.camera_keys:
            raise KeyError(f"Unknown camera {camera_key}")
        if self.info["features"][camera_key].get("dtype") == "image":
            return []
        path = self._path(episode_index, camera_key)
        resource = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        duration = self._episode_rows[episode_index]["length"] / float(self.info["fps"])
        return [MediaSpan(resource, camera_key, path, 0.0, duration)]
