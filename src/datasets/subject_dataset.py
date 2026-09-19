"""Dataset-agnostic subject-level dataset for full 10-hour PPG recordings.

A single class that handles MESA, CFS and HomePAP via DatasetType, used for
evaluation. It returns subject IDs alongside each recording so the
Benchmarker can track per-subject metrics. The trainers use the per-dataset
classes in full_night_dataset.py instead."""

import os

import h5py
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from common.constants import WINDOWS_PER_SUBJECT
from datasets.dataset_type import DatasetType
from datasets.test_subjects import TEST_SUBJECTS


class SubjectDataset(Dataset):
    """(ppg, labels, subject_id) for a single subject's full recording.

    ppg: (1, 1_228_800) - full night at 34.133 Hz
    labels: (1200,) - one per 30-second epoch
    """

    def __init__(
        self,
        data_path,
        split="train",
        transform=None,
        seed=42,
        dataset_name=DatasetType.MESA,
    ):
        self.split = split
        self.transform = transform
        self.seed = seed
        self.dataset_name = dataset_name

        self.ppg_file_path = data_path["ppg"]
        self.index_file_path = data_path["index"]

        for path in [self.ppg_file_path, self.index_file_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        self._prepare_subjects()

    def _prepare_subjects(self):
        """Split at subject level to prevent data leakage."""
        with h5py.File(self.index_file_path, "r") as f:
            all_subjects = list(f["subjects"].keys())
            # Only keep subjects with exactly 1200 epochs (full 10-hour recordings),
            # since the models expect fixed-length input
            valid_subjects = [
                s
                for s in all_subjects
                if f[f"subjects/{s}"].attrs["n_windows"] == WINDOWS_PER_SUBJECT
            ]

        fixed = TEST_SUBJECTS.get(self.dataset_name, [])
        test_subjects = [s for s in fixed if s in valid_subjects]
        train_val = [s for s in valid_subjects if s not in test_subjects]
        train_subjects, val_subjects = train_test_split(
            train_val, test_size=0.2, random_state=self.seed
        )

        splits = {"train": train_subjects, "val": val_subjects, "test": test_subjects}
        self.subjects = splits[self.split]

        # Map subject -> row offset in the flat H5 arrays (ppg, labels)
        self.subject_start_indices = {}
        with h5py.File(self.index_file_path, "r") as f:
            for subj in self.subjects:
                indices = f[f"subjects/{subj}/window_indices"][:]
                if len(indices) == WINDOWS_PER_SUBJECT:
                    self.subject_start_indices[subj] = indices[0]

        print(f"{self.split} set: {len(self.subjects)} subjects")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        start = self.subject_start_indices[subject_id]
        end = start + WINDOWS_PER_SUBJECT

        with h5py.File(self.ppg_file_path, "r") as f:
            ppg_epochs = f["ppg"][start:end]
            labels = f["labels"][start:end]

        # Flatten (1200, 1024) epoch windows into one continuous PPG signal
        ppg_continuous = ppg_epochs.reshape(-1)

        if self.transform:
            ppg_continuous = self.transform(ppg_continuous)

        # (1, total_samples) - channel dim first for conv layers
        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
        labels_tensor = torch.LongTensor(labels)

        return ppg_tensor, labels_tensor, subject_id


def get_subject_test_loader(
    data_path,
    batch_size=1,
    num_workers=4,
    dataset_name=DatasetType.MESA,
    pin_memory=True,
):
    """Test-only DataLoader yielding full subjects with subject IDs."""
    test_dataset = SubjectDataset(
        data_path=data_path,
        split="test",
        dataset_name=dataset_name,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_subject_dataloaders(
    data_path,
    batch_size=1,
    num_workers=4,
    seed=42,
    pin_memory=True,
    dataset_name=DatasetType.MESA,
):
    """Create train/val/test dataloaders that yield full subjects."""
    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = SubjectDataset(
            data_path=data_path,
            split=split,
            seed=seed,
            dataset_name=dataset_name,
        )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )
