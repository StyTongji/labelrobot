from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .resolver import SnapshotResolver


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Base sample {name} must be scalar")
        return int(value.item())
    return int(value)


class LabelRobotDatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        base: torch.utils.data.Dataset,
        annotation_path: str | Path,
        snapshot_id: str | None = None,
    ):
        self.base = base
        self.annotation_path = Path(annotation_path)
        self.snapshot_id = snapshot_id or self._latest_snapshot()
        self._resolver: SnapshotResolver | None = None

    def _latest_snapshot(self) -> str:
        snapshots = self.annotation_path / "snapshots"
        choices = sorted(path.name for path in snapshots.iterdir() if (path / "info.json").exists())
        if not choices:
            raise FileNotFoundError("No complete LabelRobot snapshot found")
        return choices[-1]

    @property
    def resolver(self) -> SnapshotResolver:
        if self._resolver is None:
            self._resolver = SnapshotResolver(self.annotation_path / "snapshots" / self.snapshot_id)
        return self._resolver

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_resolver"] = None
        return state

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        try:
            episode_index = _as_int(sample["episode_index"], "episode_index")
            frame_index = _as_int(sample["frame_index"], "frame_index")
        except KeyError as error:
            raise ValueError("Base dataset samples must contain episode_index and frame_index") from error
        timestamp = float(sample["timestamp"].item() if isinstance(sample.get("timestamp"), torch.Tensor) else sample.get("timestamp", frame_index))
        annotations = self.resolver.get_frame_annotations(episode_index, frame_index)
        path = self.resolver.get_subtask_path(episode_index, frame_index)
        subtask_vocabulary = list(self.resolver.tools.get("subtask", [])) + list(
            self.resolver.tools.get("recovery", [])
        )
        leaf = path[-1] if path else None
        sample["annotation.subtask"] = torch.tensor(
            subtask_vocabulary.index(leaf) if leaf in subtask_vocabulary else -1, dtype=torch.int64
        )
        sample["annotation.subtask_valid"] = torch.tensor(leaf in subtask_vocabulary)
        event_vocabulary = list(self.resolver.tools.get("observed_event", []))
        event_labels = {
            annotation["label"]
            for annotation in annotations
            if annotation["kind"] == "observed_event" and annotation["label"] is not None
        }
        event_vector = torch.zeros(len(event_vocabulary), dtype=torch.bool)
        for label in event_labels:
            if label in event_vocabulary:
                event_vector[event_vocabulary.index(label)] = True
        sample["annotation.observed_event"] = event_vector
        sample["annotation.observed_event_valid"] = torch.tensor(bool(event_labels))
        progress = self.resolver.progress(episode_index, frame_index, timestamp)
        sample["annotation.progress"] = torch.tensor(progress or 0.0, dtype=torch.float32)
        sample["annotation.progress_valid"] = torch.tensor(progress is not None)
        evaluation = next(
            (annotation for annotation in annotations if annotation["kind"] == "episode_evaluation"), None
        )
        score = evaluation["score"] if evaluation else None
        success = evaluation["success"] if evaluation else None
        sample["annotation.episode_score"] = torch.tensor(score if score is not None else -1, dtype=torch.int64)
        sample["annotation.episode_score_valid"] = torch.tensor(score is not None)
        sample["annotation.episode_success"] = torch.tensor(bool(success))
        sample["annotation.episode_success_valid"] = torch.tensor(success is not None)
        return sample

    def get_episode_annotations(self, episode_index: int):
        return self.resolver.get_episode_annotations(episode_index)

    def get_frame_annotations(self, episode_index: int, frame_index: int):
        return self.resolver.get_frame_annotations(episode_index, frame_index)

    def get_segments(self, episode_index: int):
        return self.resolver.get_segments(episode_index)

    def get_events(self, episode_index: int):
        return self.resolver.get_events(episode_index)

    def get_subtask_path(self, episode_index: int, frame_index: int):
        return self.resolver.get_subtask_path(episode_index, frame_index)

    def get_transition_targets(self, episode_index: int, start_frame: int, end_frame_exclusive: int):
        return self.resolver.get_transition_targets(episode_index, start_frame, end_frame_exclusive)
