from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from labelrobot.models import DatasetInfo, EpisodeIndex, EpisodeInfo, MediaSpan

from .base import DatasetReader, dataset_fingerprint, load_json


class V3Reader(DatasetReader):
    def __init__(self, root: str | Path):
        super().__init__(root)
        self.info = load_json(self.root / "meta" / "info.json")
        episode_paths = sorted((self.root / "meta" / "episodes").glob("*/*.parquet"))
        data_paths = sorted((self.root / "data").glob("*/*.parquet"))
        if not episode_paths or not data_paths:
            raise ValueError("LeRobot v3 dataset requires meta/episodes and data Parquet files")
        self.episodes = pads.dataset(episode_paths, format="parquet").to_table()
        self.data = pads.dataset(data_paths, format="parquet")
        self._episode_rows = {
            int(row["episode_index"]): row for row in self.episodes.to_pylist()
        }
        features = self.info.get("features", {})
        self.camera_keys = [
            key for key, feature in features.items() if feature.get("dtype") in {"video", "image"}
        ]
        self.tasks = self._load_tasks()
        self._dataset_version = dataset_fingerprint(self.root)

    def _load_tasks(self) -> dict[int, str]:
        path = self.root / "meta" / "tasks.parquet"
        if not path.exists():
            return {}
        table = pq.read_table(path)
        result: dict[int, str] = {}
        for row in table.to_pylist():
            index = row.get("task_index")
            task = row.get("task")
            if index is not None and task is not None:
                result[int(index)] = str(task)
        return result

    def describe(self) -> DatasetInfo:
        return DatasetInfo(
            root=self.root,
            dataset_id=str(self.info.get("repo_id") or self.root.name),
            dataset_version=self._dataset_version,
            format_version=str(self.info.get("codebase_version", "v3.0")),
            fps=float(self.info["fps"]),
            features=self.info.get("features", {}),
            camera_keys=self.camera_keys,
            total_episodes=int(self.info.get("total_episodes", len(self._episode_rows))),
            total_frames=int(self.info.get("total_frames", self.data.count_rows())),
        )

    def list_episodes(self, offset: int = 0, limit: int = 100) -> list[EpisodeInfo]:
        result = []
        for episode_index in sorted(self._episode_rows)[offset : offset + limit]:
            row = self._episode_rows[episode_index]
            tasks = row.get("tasks") or []
            if not tasks and row.get("task_index") is not None:
                task = self.tasks.get(int(row["task_index"]))
                tasks = [task] if task else []
            result.append(EpisodeInfo(episode_index, int(row["length"]), list(tasks)))
        return result

    def _episode_table(self, episode_index: int, columns: list[str] | None = None) -> pa.Table:
        if episode_index not in self._episode_rows:
            raise KeyError(f"Unknown episode {episode_index}")
        available = set(self.data.schema.names)
        requested = None if columns is None else list(dict.fromkeys(columns))
        if requested is not None:
            missing = set(requested) - available
            if missing:
                raise KeyError(f"Unknown feature columns: {sorted(missing)}")
        return self.data.to_table(
            columns=requested,
            filter=pads.field("episode_index") == episode_index,
        )

    def get_episode_index(self, episode_index: int) -> EpisodeIndex:
        table = self._episode_table(
            episode_index, ["frame_index", "index", "timestamp", "episode_index"]
        ).sort_by("frame_index")
        frames = [int(value) for value in table["frame_index"].to_pylist()]
        expected = list(range(len(frames)))
        if frames != expected:
            raise ValueError(f"Episode {episode_index} has invalid frame_index sequence")
        timestamps = [float(value) for value in table["timestamp"].to_pylist()]
        if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError(f"Episode {episode_index} has non-monotonic timestamps")
        metadata_length = int(self._episode_rows[episode_index]["length"])
        if len(frames) != metadata_length:
            raise ValueError(
                f"Episode {episode_index} metadata length {metadata_length} != data length {len(frames)}"
            )
        return EpisodeIndex(
            episode_index,
            frames,
            [int(value) for value in table["index"].to_pylist()],
            timestamps,
        )

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
        feature = self.info["features"][camera_key]
        row = self._episode_rows[episode_index]
        if feature.get("dtype") == "image":
            return []
        chunk = int(row[f"videos/{camera_key}/chunk_index"])
        file_index = int(row[f"videos/{camera_key}/file_index"])
        template = self.info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        relative = template.format(video_key=camera_key, chunk_index=chunk, file_index=file_index)
        path = (self.root / relative).resolve(strict=True)
        resource = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        return [
            MediaSpan(
                resource,
                camera_key,
                path,
                float(row[f"videos/{camera_key}/from_timestamp"]),
                float(row[f"videos/{camera_key}/to_timestamp"]),
            )
        ]
