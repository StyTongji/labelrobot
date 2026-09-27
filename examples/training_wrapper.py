"""Minimal training integration example.

Set DATASET_ROOT before running this example.
"""

import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from labelrobot.training import LabelRobotDatasetWrapper

root = Path(os.environ["DATASET_ROOT"])
base = LeRobotDataset(root.name, root=root)
dataset = LabelRobotDatasetWrapper(base, root.with_name(f"{root.name}.labelrobot"))
batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
print(batch["annotation.subtask"], batch["annotation.subtask_valid"])
