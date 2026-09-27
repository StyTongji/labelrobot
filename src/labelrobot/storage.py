from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from labelrobot.models import Annotation, Link
from labelrobot.readers.base import DatasetReader


ANNOTATION_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("dataset_id", pa.string()),
        ("dataset_version", pa.string()),
        ("episode_index", pa.int64()),
        ("kind", pa.string()),
        ("scope", pa.string()),
        ("start_frame", pa.int64()),
        ("end_frame_exclusive", pa.int64()),
        ("timestamp_s", pa.float64()),
        ("end_timestamp_s", pa.float64()),
        ("parent_id", pa.string()),
        ("camera_key", pa.string()),
        ("label", pa.string()),
        ("value_float", pa.float64()),
        ("score", pa.int64()),
        ("success", pa.bool_()),
        ("description", pa.string()),
        ("attributes_json", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
LINK_SCHEMA = pa.schema([("source_id", pa.string()), ("relation", pa.string()), ("target_id", pa.string())])


def _now() -> str:
    return datetime.now(UTC).isoformat()


def default_sidecar(source: Path) -> Path:
    return source.with_name(f"{source.name}.labelrobot")


def validate_sidecar_path(source: Path, sidecar: Path) -> None:
    source = source.resolve()
    sidecar = sidecar.expanduser().resolve()
    if sidecar == source or source in sidecar.parents:
        raise ValueError("Annotation path must be outside the source dataset")


class RevisionConflict(ValueError):
    pass


class AnnotationStore:
    def __init__(self, path: str | Path, reader: DatasetReader):
        self.path = Path(path).expanduser().resolve()
        validate_sidecar_path(reader.root, self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.reader = reader
        self.db_path = self.path / "project.sqlite"
        self.tools_path = self.path / "tools.json"
        if not self.tools_path.exists():
            shutil.copyfile(files("labelrobot.assets").joinpath("default_tools.json"), self.tools_path)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        self._initialize_project()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS project (
              singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
              dataset_id TEXT NOT NULL,
              dataset_version TEXT NOT NULL,
              annotation_revision INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS annotations (
              id TEXT PRIMARY KEY,
              dataset_id TEXT NOT NULL,
              dataset_version TEXT NOT NULL,
              episode_index INTEGER NOT NULL,
              kind TEXT NOT NULL,
              scope TEXT NOT NULL,
              start_frame INTEGER,
              end_frame_exclusive INTEGER,
              timestamp_s REAL,
              end_timestamp_s REAL,
              parent_id TEXT REFERENCES annotations(id) ON DELETE RESTRICT,
              camera_key TEXT,
              label TEXT,
              value_float REAL,
              score INTEGER,
              success INTEGER,
              description TEXT,
              attributes_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS links (
              source_id TEXT NOT NULL REFERENCES annotations(id) ON DELETE RESTRICT,
              relation TEXT NOT NULL,
              target_id TEXT NOT NULL REFERENCES annotations(id) ON DELETE RESTRICT,
              PRIMARY KEY (source_id, relation, target_id)
            );
            CREATE TABLE IF NOT EXISTS applied_commands (
              command_id TEXT PRIMARY KEY,
              resulting_revision INTEGER NOT NULL,
              applied_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS annotations_episode_frame
              ON annotations(episode_index, start_frame);
            CREATE INDEX IF NOT EXISTS annotations_parent
              ON annotations(parent_id);
            """
        )
        self.connection.commit()

    def _initialize_project(self) -> None:
        source = self.reader.describe()
        row = self.connection.execute("SELECT * FROM project WHERE singleton=1").fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO project VALUES (1, ?, ?, 0, ?)",
                (str(uuid.uuid4()), source.dataset_version, _now()),
            )
            self.connection.commit()
        elif row["dataset_version"] != source.dataset_version:
            raise ValueError("Source dataset changed since this annotation project was created")

    @property
    def project(self) -> dict[str, Any]:
        return dict(self.connection.execute("SELECT * FROM project WHERE singleton=1").fetchone())

    @property
    def revision(self) -> int:
        return int(self.project["annotation_revision"])

    def list_annotations(self, episode_index: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM annotations"
        parameters: tuple[Any, ...] = ()
        if episode_index is not None:
            query += " WHERE episode_index=?"
            parameters = (episode_index,)
        query += " ORDER BY episode_index, COALESCE(start_frame, -1), id"
        return [self._decode(row) for row in self.connection.execute(query, parameters)]

    def list_links(self, episode_index: int | None = None) -> list[dict[str, Any]]:
        if episode_index is None:
            rows = self.connection.execute("SELECT * FROM links ORDER BY source_id, relation, target_id")
        else:
            rows = self.connection.execute(
                """SELECT links.* FROM links JOIN annotations ON annotations.id=links.source_id
                   WHERE annotations.episode_index=? ORDER BY source_id, relation, target_id""",
                (episode_index,),
            )
        return [dict(row) for row in rows]

    def apply_command(
        self,
        command_id: str,
        expected_revision: int,
        operations: list[dict[str, Any]],
    ) -> int:
        existing = self.connection.execute(
            "SELECT resulting_revision FROM applied_commands WHERE command_id=?", (command_id,)
        ).fetchone()
        if existing is not None:
            return int(existing[0])
        if expected_revision != self.revision:
            raise RevisionConflict(f"Expected revision {expected_revision}, current revision is {self.revision}")
        with self.connection:
            for operation in operations:
                action = operation.get("action")
                if action == "upsert":
                    annotation = Annotation.from_dict(operation["annotation"])
                    self._upsert(annotation)
                    if "links" in operation:
                        self.connection.execute("DELETE FROM links WHERE source_id=?", (annotation.id,))
                        for link in operation["links"]:
                            self._insert_link(Link(**link))
                elif action == "delete":
                    self._delete(str(operation["id"]))
                else:
                    raise ValueError(f"Unsupported command action: {action}")
            revision = expected_revision + 1
            self.connection.execute(
                "UPDATE project SET annotation_revision=? WHERE singleton=1", (revision,)
            )
            self.connection.execute(
                "INSERT INTO applied_commands VALUES (?, ?, ?)", (command_id, revision, _now())
            )
        return revision

    def _upsert(self, annotation: Annotation) -> None:
        project = self.project
        annotation.dataset_id = project["dataset_id"]
        annotation.dataset_version = project["dataset_version"]
        existing = self.connection.execute("SELECT created_at FROM annotations WHERE id=?", (annotation.id,)).fetchone()
        now = _now()
        annotation.created_at = existing[0] if existing else now
        annotation.updated_at = now
        self._validate(annotation)
        values = annotation.to_dict()
        values["attributes_json"] = json.dumps(values.pop("attributes"), ensure_ascii=False, sort_keys=True)
        columns = list(values)
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "id")
        placeholders = ", ".join("?" for _ in columns)
        self.connection.execute(
            f"INSERT INTO annotations ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {assignments}",
            tuple(values[column] for column in columns),
        )

    def _validate(self, annotation: Annotation) -> None:
        info = self.reader.describe()
        if annotation.episode_index < 0 or annotation.episode_index >= info.total_episodes:
            raise ValueError("Unknown episode_index")
        if annotation.camera_key is not None and annotation.camera_key not in info.camera_keys:
            raise ValueError("Unknown camera_key")
        expected_scope = {
            "subtask": "segment",
            "recovery": "segment",
            "observed_event": annotation.scope,
            "expected_outcome": "point",
            "verification": "point",
            "progress": "frame_value",
            "episode_evaluation": "episode",
        }[annotation.kind]
        if annotation.scope != expected_scope or (
            annotation.kind == "observed_event" and annotation.scope not in {"point", "segment"}
        ):
            raise ValueError(f"Invalid scope {annotation.scope} for {annotation.kind}")
        if annotation.scope == "episode":
            if annotation.start_frame is not None or annotation.end_frame_exclusive is not None:
                raise ValueError("Episode annotations cannot have frame bounds")
        else:
            self.reader.validate_frame_range(
                annotation.episode_index,
                annotation.start_frame,
                annotation.end_frame_exclusive if annotation.scope == "segment" else None,
            )
            if annotation.scope != "segment" and annotation.end_frame_exclusive is not None:
                raise ValueError("Only segment annotations have end_frame_exclusive")
        if annotation.kind == "progress" and (
            annotation.value_float is None or not 0 <= annotation.value_float <= 1
        ):
            raise ValueError("Progress value must be in [0, 1]")
        if annotation.kind == "episode_evaluation" and annotation.score is not None and not 0 <= annotation.score <= 5:
            raise ValueError("Episode score must be in [0, 5]")
        if annotation.kind == "episode_evaluation" and self.connection.execute(
            "SELECT 1 FROM annotations WHERE episode_index=? AND kind='episode_evaluation' AND id<>?",
            (annotation.episode_index, annotation.id),
        ).fetchone():
            raise ValueError("An episode can have only one evaluation")
        if annotation.kind == "progress" and self.connection.execute(
            "SELECT 1 FROM annotations WHERE episode_index=? AND kind='progress' AND start_frame=? AND id<>?",
            (annotation.episode_index, annotation.start_frame, annotation.id),
        ).fetchone():
            raise ValueError("A frame can have only one progress anchor")
        if annotation.kind == "expected_outcome":
            attributes = annotation.attributes
            if attributes.get("provenance", "retrospective") != "retrospective":
                raise ValueError("MVP expected outcomes must use retrospective provenance")
            window_start = int(attributes.get("window_start_frame", annotation.start_frame))
            window_end = int(attributes.get("window_end_frame_exclusive", window_start + 1))
            self.reader.validate_frame_range(annotation.episode_index, window_start, window_end)
        self._validate_parent(annotation)
        if annotation.kind == "verification":
            result = annotation.attributes.get("result")
            if result not in {"met", "not_met", "uncertain"}:
                raise ValueError("Verification result must be met, not_met, or uncertain")

    def _validate_parent(self, annotation: Annotation) -> None:
        if annotation.parent_id is None:
            if annotation.kind in {"subtask", "recovery"}:
                self._validate_sibling_overlap(annotation, None)
        else:
            parent = self.connection.execute("SELECT * FROM annotations WHERE id=?", (annotation.parent_id,)).fetchone()
            if parent is None:
                raise ValueError("parent_id does not exist")
            if annotation.parent_id == annotation.id:
                raise ValueError("An annotation cannot parent itself")
            if int(parent["episode_index"]) != annotation.episode_index:
                raise ValueError("Parent must be in the same episode")
            if parent["kind"] not in {"subtask", "recovery"}:
                raise ValueError("Parent must be a subtask or recovery segment")
            cursor = parent
            while cursor["parent_id"] is not None:
                if cursor["parent_id"] == annotation.id:
                    raise ValueError("Subtask hierarchy cannot contain cycles")
                cursor = self.connection.execute(
                    "SELECT * FROM annotations WHERE id=?", (cursor["parent_id"],)
                ).fetchone()
            if annotation.kind in {"subtask", "recovery"}:
                if not (
                    int(parent["start_frame"]) <= int(annotation.start_frame)
                    and int(annotation.end_frame_exclusive) <= int(parent["end_frame_exclusive"])
                ):
                    raise ValueError("Child segment must be contained by its parent")
                self._validate_sibling_overlap(annotation, annotation.parent_id)
            if annotation.kind not in {"subtask", "recovery"} and annotation.start_frame is not None:
                if not int(parent["start_frame"]) <= annotation.start_frame < int(parent["end_frame_exclusive"]):
                    raise ValueError("Child event must occur inside its parent subtask")
        children = self.connection.execute(
            "SELECT start_frame, end_frame_exclusive FROM annotations WHERE parent_id=? AND id<>? AND kind IN ('subtask','recovery')",
            (annotation.id, annotation.id),
        )
        for child in children:
            if not (
                annotation.start_frame <= child["start_frame"]
                and child["end_frame_exclusive"] <= annotation.end_frame_exclusive
            ):
                raise ValueError("Updated parent bounds would exclude a child")

    def _validate_sibling_overlap(self, annotation: Annotation, parent_id: str | None) -> None:
        rows = self.connection.execute(
            """SELECT id, start_frame, end_frame_exclusive FROM annotations
               WHERE episode_index=? AND kind IN ('subtask','recovery') AND id<>?
               AND parent_id IS ?""",
            (annotation.episode_index, annotation.id, parent_id),
        )
        for row in rows:
            if annotation.start_frame < row["end_frame_exclusive"] and row["start_frame"] < annotation.end_frame_exclusive:
                raise ValueError(f"Segment overlaps sibling {row['id']}")

    def _insert_link(self, link: Link) -> None:
        source = self.connection.execute("SELECT * FROM annotations WHERE id=?", (link.source_id,)).fetchone()
        target = self.connection.execute("SELECT * FROM annotations WHERE id=?", (link.target_id,)).fetchone()
        if source is None or target is None:
            raise ValueError("Link source and target must exist")
        if source["kind"] != "verification":
            raise ValueError("Only verification annotations create links")
        if source["episode_index"] != target["episode_index"]:
            raise ValueError("Linked annotations must be in the same episode")
        expected_kind = "expected_outcome" if link.relation == "verifies_expected" else "observed_event"
        if target["kind"] != expected_kind:
            raise ValueError(f"{link.relation} must target {expected_kind}")
        target_end = target["end_frame_exclusive"] - 1 if target["end_frame_exclusive"] else target["start_frame"]
        if source["start_frame"] < target_end:
            raise ValueError("Verification cannot precede its evidence")
        if link.relation == "verifies_expected":
            other = self.connection.execute(
                """SELECT 1 FROM links JOIN annotations ON annotations.id=links.source_id
                   WHERE links.relation='verifies_expected' AND links.target_id=? AND links.source_id<>?""",
                (link.target_id, link.source_id),
            ).fetchone()
            if other:
                raise ValueError("An expected outcome can have only one current verification")
        self.connection.execute(
            "INSERT OR IGNORE INTO links VALUES (?, ?, ?)",
            (link.source_id, link.relation, link.target_id),
        )
        self._validate_verification_window(link.source_id)

    def _validate_verification_window(self, verification_id: str) -> None:
        expected = self.connection.execute(
            """SELECT annotations.* FROM links JOIN annotations ON annotations.id=links.target_id
               WHERE links.source_id=? AND links.relation='verifies_expected'""",
            (verification_id,),
        ).fetchone()
        if expected is None:
            return
        attributes = json.loads(expected["attributes_json"])
        start = int(attributes.get("window_start_frame", expected["start_frame"]))
        end = int(attributes.get("window_end_frame_exclusive", start + 1))
        events = self.connection.execute(
            """SELECT annotations.* FROM links JOIN annotations ON annotations.id=links.target_id
               WHERE links.source_id=? AND links.relation='supported_by'""",
            (verification_id,),
        )
        for event in events:
            event_end = int(event["end_frame_exclusive"] or event["start_frame"] + 1)
            if not (start <= int(event["start_frame"]) and event_end <= end):
                raise ValueError("Observed evidence must be inside the expected outcome window")

    def _delete(self, annotation_id: str) -> None:
        children = self.connection.execute("SELECT 1 FROM annotations WHERE parent_id=?", (annotation_id,)).fetchone()
        links = self.connection.execute(
            "SELECT 1 FROM links WHERE source_id=? OR target_id=?", (annotation_id, annotation_id)
        ).fetchone()
        if children or links:
            raise ValueError("Cannot delete an annotation that is still referenced")
        if self.connection.execute("DELETE FROM annotations WHERE id=?", (annotation_id,)).rowcount == 0:
            raise KeyError(f"Unknown annotation {annotation_id}")

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["attributes"] = json.loads(value.pop("attributes_json"))
        if value["success"] is not None:
            value["success"] = bool(value["success"])
        return value

    def create_snapshot(self, snapshot_id: str | None = None) -> str:
        snapshot_id = snapshot_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self.path / "snapshots" / snapshot_id
        if destination.exists():
            raise FileExistsError(f"Snapshot already exists: {snapshot_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=destination.parent))
        try:
            with self.connection:
                revision = self.revision
                annotations = self.list_annotations()
                links = self.list_links()
            annotation_rows = []
            for item in annotations:
                row = dict(item)
                row["attributes_json"] = json.dumps(row.pop("attributes"), ensure_ascii=False, sort_keys=True)
                annotation_rows.append(row)
            pq.write_table(
                pa.Table.from_pylist(annotation_rows, schema=ANNOTATION_SCHEMA),
                temporary / "annotations.parquet",
                compression="zstd",
            )
            pq.write_table(
                pa.Table.from_pylist(links, schema=LINK_SCHEMA),
                temporary / "links.parquet",
                compression="zstd",
            )
            tools = self.tools_path.read_bytes()
            (temporary / "tools.json").write_bytes(tools)
            info = {
                "snapshot_id": snapshot_id,
                "annotation_revision": revision,
                "dataset_id": self.project["dataset_id"],
                "dataset_version": self.project["dataset_version"],
                "tools_sha256": hashlib.sha256(tools).hexdigest(),
                "annotation_count": len(annotation_rows),
                "link_count": len(links),
                "created_at": _now(),
            }
            (temporary / "info.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.rename(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return snapshot_id
