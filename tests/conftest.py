from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def v3_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "robot-data"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 7,
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["joint", "gripper"]},
            "action": {"dtype": "float32", "shape": [1], "names": ["joint"]},
            "observation.images.front": {"dtype": "video", "shape": [16, 16, 3], "names": ["height", "width", "channels"]},
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1], "length": [3, 4],
                "dataset_from_index": [0, 3], "dataset_to_index": [3, 7],
                "data/chunk_index": [0, 0], "data/file_index": [0, 0],
                "videos/observation.images.front/chunk_index": [0, 0],
                "videos/observation.images.front/file_index": [0, 0],
                "videos/observation.images.front/from_timestamp": [0.0, 0.3],
                "videos/observation.images.front/to_timestamp": [0.3, 0.7],
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "index": list(range(7)), "episode_index": [0] * 3 + [1] * 4,
                "frame_index": [0, 1, 2, 0, 1, 2, 3],
                "timestamp": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.3],
                "observation.state": [[index, index + 0.5] for index in range(7)],
                "action": [[index] for index in range(7)],
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    video = root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"placeholder")
    return root


@pytest.fixture
def v2_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "robot-v2"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1", "fps": 20, "total_episodes": 1, "total_frames": 2,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"action": {"dtype": "float32", "shape": [1], "names": ["joint"]}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 2, "tasks": ["move"]}) + "\n", encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "move"}) + "\n", encoding="utf-8")
    pq.write_table(pa.table({"index": [0, 1], "episode_index": [0, 0], "frame_index": [0, 1], "timestamp": [0.0, 0.05], "action": [[1.0], [2.0]]}), root / "data" / "chunk-000" / "episode_000000.parquet")
    return root
