from __future__ import annotations

from pathlib import Path

import pytest

from labelrobot.models import Annotation
from labelrobot.readers import open_dataset
from labelrobot.storage import AnnotationStore


def annotation(store: AnnotationStore, **values) -> dict:
    project = store.project
    defaults = dict(
        id="root", dataset_id=project["dataset_id"], dataset_version=project["dataset_version"],
        episode_index=0, kind="subtask", scope="segment", start_frame=0,
        end_frame_exclusive=3, parent_id=None, camera_key=None, label="pick", attributes={},
    )
    defaults.update(values)
    return Annotation(**defaults).to_dict()


def test_hierarchy_overlap_and_parent_bounds(v3_dataset: Path, tmp_path: Path):
    store = AnnotationStore(tmp_path / "annotations", open_dataset(v3_dataset))
    revision = store.apply_command("1", 0, [{"action": "upsert", "annotation": annotation(store)}])
    revision = store.apply_command("2", revision, [{"action": "upsert", "annotation": annotation(store, id="child", parent_id="root", start_frame=0, end_frame_exclusive=2, label="grasp")}])
    with pytest.raises(ValueError, match="overlaps sibling"):
        store.apply_command("3", revision, [{"action": "upsert", "annotation": annotation(store, id="sibling", parent_id="root", start_frame=1, end_frame_exclusive=3)}])
    with pytest.raises(ValueError, match="exclude a child"):
        store.apply_command("4", revision, [{"action": "upsert", "annotation": annotation(store, start_frame=1, end_frame_exclusive=3)}])
    with pytest.raises(ValueError, match="still referenced"):
        store.apply_command("5", revision, [{"action": "delete", "id": "root"}])


def test_event_links_snapshot_and_idempotency(v3_dataset: Path, tmp_path: Path):
    store = AnnotationStore(tmp_path / "annotations", open_dataset(v3_dataset))
    expected = annotation(store, id="expect", kind="expected_outcome", scope="point", start_frame=0, end_frame_exclusive=None, label="secured", attributes={"provenance": "retrospective", "window_start_frame": 0, "window_end_frame_exclusive": 3})
    event = annotation(store, id="event", kind="observed_event", scope="point", start_frame=1, end_frame_exclusive=None, label="slip")
    verification = annotation(store, id="verify", kind="verification", scope="point", start_frame=2, end_frame_exclusive=None, label=None, attributes={"result": "not_met"})
    operations = [
        {"action": "upsert", "annotation": expected},
        {"action": "upsert", "annotation": event},
        {"action": "upsert", "annotation": verification, "links": [
            {"source_id": "verify", "relation": "verifies_expected", "target_id": "expect"},
            {"source_id": "verify", "relation": "supported_by", "target_id": "event"},
        ]},
    ]
    assert store.apply_command("events", 0, operations) == 1
    assert store.apply_command("events", 0, operations) == 1
    snapshot = store.create_snapshot("test")
    assert snapshot == "test"
    assert (tmp_path / "annotations" / "snapshots" / "test" / "annotations.parquet").exists()
