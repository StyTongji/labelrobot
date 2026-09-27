from __future__ import annotations

import hashlib
from pathlib import Path

from labelrobot.readers import open_dataset


def digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def test_v3_shared_shard_and_source_is_read_only(v3_dataset: Path):
    before = digest(v3_dataset)
    reader = open_dataset(v3_dataset)
    assert [episode.length for episode in reader.list_episodes()] == [3, 4]
    index = reader.get_episode_index(1)
    assert index.frame_index == [0, 1, 2, 3]
    assert index.source_index == [3, 4, 5, 6]
    assert reader.read_rows(1, 1, 3, ["action"]).to_pylist() == [{"action": [4]}, {"action": [5]}]
    assert reader.resolve_media(1, "observation.images.front")[0].from_timestamp_s == 0.3
    assert digest(v3_dataset) == before


def test_v2_reader(v2_dataset: Path):
    reader = open_dataset(v2_dataset)
    assert reader.describe().format_version == "v2.1"
    assert reader.get_episode_index(0).timestamp_s == [0.0, 0.05]
    assert reader.read_rows(0, 0, 2, ["action"]).to_pylist() == [{"action": [1.0]}, {"action": [2.0]}]
