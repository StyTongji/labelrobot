from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

AnnotationKind = Literal[
    "subtask", "recovery", "observed_event", "expected_outcome", "verification", "progress", "episode_evaluation"
]
AnnotationScope = Literal["point", "segment", "frame_value", "episode"]


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    dataset_id: str
    dataset_version: str
    format_version: str
    fps: float
    features: dict[str, Any]
    camera_keys: list[str]
    total_episodes: int
    total_frames: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["root"] = str(self.root)
        return value


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int
    tasks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EpisodeIndex:
    episode_index: int
    frame_index: list[int]
    source_index: list[int]
    timestamp_s: list[float]

    @property
    def length(self) -> int:
        return len(self.frame_index)


@dataclass(frozen=True)
class MediaSpan:
    resource_id: str
    camera_key: str
    path: Path
    from_timestamp_s: float
    to_timestamp_s: float | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        return value


@dataclass
class Annotation:
    id: str
    dataset_id: str
    dataset_version: str
    episode_index: int
    kind: AnnotationKind
    scope: AnnotationScope
    start_frame: int | None = None
    end_frame_exclusive: int | None = None
    timestamp_s: float | None = None
    end_timestamp_s: float | None = None
    parent_id: str | None = None
    camera_key: str | None = None
    label: str | None = None
    value_float: float | None = None
    score: int | None = None
    success: bool | None = None
    description: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Annotation":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: item for key, item in value.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Link:
    source_id: str
    relation: Literal["verifies_expected", "supported_by"]
    target_id: str
