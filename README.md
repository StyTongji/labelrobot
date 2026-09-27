# LabelRobot

**English** | [简体中文](README.zh-CN.md)

LabelRobot is a local, read-only trajectory viewer and annotation layer for LeRobot datasets. It stores annotations beside the source dataset, exports immutable Parquet snapshots, and exposes labels through a PyTorch dataset wrapper.

## Run

```bash
pip install -e .
python -m labelrobot /path/to/dataset
```

The default annotation directory is `/path/to/dataset.labelrobot`. The source dataset is never written.

Useful commands:

```bash
python -m labelrobot /path/to/dataset --no-browser
python -m labelrobot snapshot /path/to/dataset
```

The browser UI supports episode inspection, exact frame stepping, hierarchical subtasks, observed events, expected outcomes, verification links, and episode evaluation.

## Training

```python
from labelrobot.training import LabelRobotDatasetWrapper

wrapped = LabelRobotDatasetWrapper(base, "/path/to/dataset.labelrobot")
sample = wrapped[0]
path = wrapped.get_subtask_path(episode_index=0, frame_index=42)
```

Future events and verification results are available only through `get_transition_targets`; they are not injected into current-frame samples.
