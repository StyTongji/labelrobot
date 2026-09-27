from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


class SnapshotResolver:
    def __init__(self, snapshot_path: str | Path):
        self.path = Path(snapshot_path)
        self.info = json.loads((self.path / "info.json").read_text(encoding="utf-8"))
        self.tools = json.loads((self.path / "tools.json").read_text(encoding="utf-8"))
        self.annotations = []
        for row in pq.read_table(self.path / "annotations.parquet").to_pylist():
            row["attributes"] = json.loads(row.pop("attributes_json"))
            self.annotations.append(row)
        self.links = pq.read_table(self.path / "links.parquet").to_pylist()
        self._by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._by_id: dict[str, dict[str, Any]] = {}
        for annotation in self.annotations:
            self._by_episode[int(annotation["episode_index"])].append(annotation)
            self._by_id[annotation["id"]] = annotation

    def get_episode_annotations(self, episode_index: int) -> list[dict[str, Any]]:
        return list(self._by_episode.get(episode_index, ()))

    def get_segments(self, episode_index: int) -> list[dict[str, Any]]:
        return [
            annotation
            for annotation in self.get_episode_annotations(episode_index)
            if annotation["scope"] == "segment"
        ]

    def get_events(self, episode_index: int) -> list[dict[str, Any]]:
        return [
            annotation
            for annotation in self.get_episode_annotations(episode_index)
            if annotation["kind"] == "observed_event"
        ]

    def get_frame_annotations(self, episode_index: int, frame_index: int) -> list[dict[str, Any]]:
        result = []
        for annotation in self.get_episode_annotations(episode_index):
            if annotation["scope"] == "episode":
                result.append(annotation)
            elif annotation["scope"] == "segment":
                if annotation["start_frame"] <= frame_index < annotation["end_frame_exclusive"]:
                    result.append(annotation)
            elif annotation["start_frame"] == frame_index:
                result.append(annotation)
        return result

    def get_subtask_path(self, episode_index: int, frame_index: int) -> list[str]:
        active = {
            annotation["id"]: annotation
            for annotation in self.get_frame_annotations(episode_index, frame_index)
            if annotation["kind"] in {"subtask", "recovery"}
        }
        if not active:
            return []
        leaf = next(
            (
                annotation
                for annotation in active.values()
                if not any(other.get("parent_id") == annotation["id"] for other in active.values())
            ),
            None,
        )
        if leaf is None:
            return []
        path = []
        cursor = leaf
        while cursor is not None:
            path.append(cursor.get("label") or cursor["id"])
            cursor = active.get(cursor.get("parent_id"))
        return list(reversed(path))

    def progress(self, episode_index: int, frame_index: int, timestamp_s: float) -> float | None:
        anchors = sorted(
            (
                annotation
                for annotation in self.get_episode_annotations(episode_index)
                if annotation["kind"] == "progress"
            ),
            key=lambda annotation: annotation["start_frame"],
        )
        if not anchors or frame_index < anchors[0]["start_frame"] or frame_index > anchors[-1]["start_frame"]:
            return None
        if len(anchors) == 1:
            return anchors[0]["value_float"] if frame_index == anchors[0]["start_frame"] else None
        times = np.asarray([annotation["timestamp_s"] for annotation in anchors], dtype=np.float64)
        values = np.asarray([annotation["value_float"] for annotation in anchors], dtype=np.float64)
        return float(np.interp(timestamp_s, times, values))

    def get_transition_targets(
        self, episode_index: int, start_frame: int, end_frame_exclusive: int
    ) -> dict[str, Any]:
        selected = []
        selected_ids = set()
        for annotation in self.get_episode_annotations(episode_index):
            if annotation["kind"] not in {"expected_outcome", "observed_event", "verification"}:
                continue
            event_end = annotation["end_frame_exclusive"] or annotation["start_frame"] + 1
            if annotation["start_frame"] < end_frame_exclusive and start_frame < event_end:
                selected.append(annotation)
                selected_ids.add(annotation["id"])
        links = [
            link
            for link in self.links
            if link["source_id"] in selected_ids or link["target_id"] in selected_ids
        ]
        return {"annotations": selected, "links": links}
