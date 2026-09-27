from __future__ import annotations

from pathlib import Path

import torch

from labelrobot.models import Annotation
from labelrobot.readers import open_dataset
from labelrobot.storage import AnnotationStore
from labelrobot.training import LabelRobotDatasetWrapper


class Base(torch.utils.data.Dataset):
    def __len__(self): return 3
    def __getitem__(self, index):
        return {"episode_index": torch.tensor(0), "frame_index": torch.tensor(index), "timestamp": torch.tensor(index / 10)}


def test_wrapper_exposes_current_labels_but_not_future_targets(v3_dataset: Path, tmp_path: Path):
    sidecar = tmp_path / "annotations"
    store = AnnotationStore(sidecar, open_dataset(v3_dataset))
    project = store.project
    def item(**values):
        defaults = dict(dataset_id=project["dataset_id"], dataset_version=project["dataset_version"], episode_index=0, parent_id=None, camera_key=None, label=None, attributes={})
        defaults.update(values)
        return Annotation(**defaults).to_dict()
    store.apply_command("labels", 0, [
        {"action": "upsert", "annotation": item(id="task", kind="subtask", scope="segment", start_frame=0, end_frame_exclusive=3, label="grasp")},
        {"action": "upsert", "annotation": item(id="future", kind="observed_event", scope="point", start_frame=2, end_frame_exclusive=None, label="object_slip")},
    ])
    store.create_snapshot("fixed")
    wrapped = LabelRobotDatasetWrapper(Base(), sidecar, "fixed")
    sample = wrapped[0]
    assert sample["annotation.subtask_valid"].item()
    assert not sample["annotation.observed_event_valid"].item()
    targets = wrapped.get_transition_targets(0, 0, 3)
    assert [value["id"] for value in targets["annotations"]] == ["future"]
