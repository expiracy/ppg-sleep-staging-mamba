"""Windowed dataset that splits 10-hour PPG recordings into smaller
fixed-duration windows. The benchmarker uses it to score the Mamba models on
less than a full night."""

import os

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from common.constants import SAMPLES_PER_WINDOW, WINDOW_DURATION, WINDOWS_PER_SUBJECT
from datasets.dataset_type import DatasetType
from datasets.test_subjects import RANDOM_TEST_SIZE, TEST_SUBJECTS


class WindowedSleepDataset(Dataset):
    """Yields fixed-duration windows from full-night PPG recordings.

    Windows are indexed by (subject_id, start_epoch). A per-subject cache
    avoids repeated H5 reads when consecutive windows belong to the same
    subject (the common case with sequential iteration).
    """

    def __init__(
        self,
        data_path,
        split="train",
        window_duration_minutes=5,
        overlap_ratio=0.0,
        transform=None,
        seed=42,
        use_fixed_test_set=True,
        dataset_name=DatasetType.MESA,
        pad=True,
    ):
        self.split = split
        self.window_duration_minutes = window_duration_minutes
        self.overlap_ratio = overlap_ratio
        self.transform = transform
        self.seed = seed
        self.use_fixed_test_set = use_fixed_test_set
        self.dataset_name = dataset_name
        self.pad = pad

        # Window and stride lengths, in epochs and in samples
        self.epochs_per_window = int(window_duration_minutes * 60 / WINDOW_DURATION)
        self.samples_per_window = self.epochs_per_window * SAMPLES_PER_WINDOW

        overlap_epochs = int(self.epochs_per_window * overlap_ratio)
        self.stride_epochs = self.epochs_per_window - overlap_epochs
        self.stride_samples = self.stride_epochs * SAMPLES_PER_WINDOW

        # How many complete windows fit within the recording
        full_windows = (
            WINDOWS_PER_SUBJECT - self.epochs_per_window
        ) // self.stride_epochs + 1
        self.windows_per_subject = full_windows

        # If padding is enabled, add one partial window for any leftover epochs
        if self.pad:
            next_start = full_windows * self.stride_epochs
            if next_start < WINDOWS_PER_SUBJECT:
                self.windows_per_subject = full_windows + 1

        reduction = (WINDOWS_PER_SUBJECT * SAMPLES_PER_WINDOW) / self.samples_per_window
        print(
            f"Windowed: {window_duration_minutes} min ({self.epochs_per_window} epochs), "
            f"stride {self.stride_epochs}, {self.windows_per_subject} windows/subject, "
            f"{reduction:.0f}x reduction"
        )

        self.ppg_file_path = data_path["ppg"]
        self.index_file_path = data_path["index"]

        # Lazy H5 handle, opened on first access
        self._ppg_h5 = None

        for path in [self.ppg_file_path, self.index_file_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        self._prepare_subjects()
        self._build_window_index()

        # One-subject cache to avoid per-window H5 reads
        self._cached_subject_id = None
        self._cached_ppg = None
        self._cached_labels = None

    def _load_subject(self, subject_id):
        """Read a full subject into the cache (one H5 read per subject)."""
        start = self.subject_start_indices[subject_id]
        end = start + WINDOWS_PER_SUBJECT

        if self._ppg_h5 is None:
            self._ppg_h5 = h5py.File(self.ppg_file_path, "r")
        self._cached_ppg = self._ppg_h5["ppg"][start:end]
        self._cached_labels = self._ppg_h5["labels"][start:end]

        self._cached_subject_id = subject_id

    def _prepare_subjects(self):
        """Split subjects into train/val/test sets.

        Split at subject level to prevent data leakage: all windows from
        a subject go to the same split.
        """
        with h5py.File(self.index_file_path, "r") as f:
            all_subjects = list(f["subjects"].keys())

            # Only keep subjects with exactly 1200 epochs (full 10-hour recordings)
            valid_subjects = []
            for subj in all_subjects:
                n_windows = f[f"subjects/{subj}"].attrs["n_windows"]
                if n_windows == WINDOWS_PER_SUBJECT:
                    valid_subjects.append(subj)

        if self.use_fixed_test_set:
            fixed_subjects = TEST_SUBJECTS.get(self.dataset_name, [])
            test_subjects = [s for s in fixed_subjects if s in valid_subjects]
            train_val_subjects = [s for s in valid_subjects if s not in test_subjects]

            train_subjects, val_subjects = train_test_split(
                train_val_subjects, test_size=0.2, random_state=self.seed
            )
        else:
            random_test_size = RANDOM_TEST_SIZE.get(self.dataset_name, 0.2)
            train_subjects, test_subjects = train_test_split(
                valid_subjects, test_size=random_test_size, random_state=self.seed
            )
            train_subjects, val_subjects = train_test_split(
                train_subjects, test_size=0.2, random_state=self.seed
            )

        if self.split == "train":
            self.subjects = train_subjects
        elif self.split == "val":
            self.subjects = val_subjects
        else:
            self.subjects = test_subjects

        total = len(self.subjects) * self.windows_per_subject
        print(
            f"{self.split} set: {len(self.subjects)} subjects, "
            f"{self.windows_per_subject} windows each, {total} total"
        )

        # Map subject -> row offset in the flat H5 arrays
        self.subject_start_indices = {}
        with h5py.File(self.index_file_path, "r") as f:
            for subj in self.subjects:
                indices = f[f"subjects/{subj}/window_indices"][:]
                if len(indices) == WINDOWS_PER_SUBJECT:
                    self.subject_start_indices[subj] = indices[0]

    def _build_window_index(self):
        """Map dataset index -> (subject_id, window_start_epoch)."""
        self.window_index = []
        for subject_id in self.subjects:
            for window_idx in range(self.windows_per_subject):
                start_epoch = window_idx * self.stride_epochs
                self.window_index.append((subject_id, start_epoch))

    def __del__(self):
        if self._ppg_h5 is not None:
            self._ppg_h5.close()

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        subject_id, start_epoch = self.window_index[idx]
        end_epoch = start_epoch + self.epochs_per_window

        # Avoid re-reading H5 when consecutive windows belong to the same subject
        if subject_id != self._cached_subject_id:
            self._load_subject(subject_id)

        if end_epoch <= WINDOWS_PER_SUBJECT:
            # Full window - slice directly from cache
            ppg_epochs = self._cached_ppg[start_epoch:end_epoch]
            labels = self._cached_labels[start_epoch:end_epoch]
        else:
            # Partial window at end of recording - zero-pad PPG,
            # fill padded labels with -1 so they can be ignored in loss
            available_ppg = self._cached_ppg[start_epoch:]
            pad_epochs = self.epochs_per_window - len(available_ppg)
            ppg_epochs = np.pad(
                available_ppg, ((0, pad_epochs), (0, 0)), mode="constant"
            )

            available_labels = self._cached_labels[start_epoch:]
            labels = np.concatenate(
                [
                    available_labels,
                    np.full(pad_epochs, -1, dtype=available_labels.dtype),
                ]
            )

        # Flatten epoch windows into a continuous signal
        ppg_continuous = ppg_epochs.reshape(-1)

        if self.transform:
            ppg_continuous = self.transform(ppg_continuous)

        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
        labels_tensor = torch.LongTensor(labels)

        return ppg_tensor, labels_tensor, subject_id


def get_windowed_dataloaders(
    data_path,
    batch_size=4,
    num_workers=4,
    window_duration_minutes=5,
    overlap_ratio=0.1,
    use_fixed_test_set=True,
    pin_memory=True,
    dataset_name=DatasetType.MESA,
    pad=True,
):
    """Create train/val/test dataloaders with windowed data."""
    common = dict(
        data_path=data_path,
        window_duration_minutes=window_duration_minutes,
        overlap_ratio=overlap_ratio,
        use_fixed_test_set=use_fixed_test_set,
        dataset_name=dataset_name,
        pad=pad,
    )

    train_dataset = WindowedSleepDataset(split="train", **common)
    val_dataset = WindowedSleepDataset(split="val", **common)
    test_dataset = WindowedSleepDataset(split="test", **common)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


def get_windowed_test_loader(
    data_path,
    batch_size=None,
    num_workers=4,
    window_duration_minutes=3,
    dataset_name=DatasetType.MESA,
    pin_memory=True,
    pad=True,
):
    """Test-only DataLoader with non-overlapping windows and subject IDs,
    so the Benchmarker can reassemble per-subject predictions.

    If batch_size is None, defaults to one subject's worth of windows.
    """
    test_dataset = WindowedSleepDataset(
        data_path=data_path,
        split="test",
        window_duration_minutes=window_duration_minutes,
        overlap_ratio=0.0,
        use_fixed_test_set=True,
        dataset_name=dataset_name,
        pad=pad,
    )

    if batch_size is None:
        batch_size = test_dataset.windows_per_subject

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    from common.paths import MESA_DATA_PATHS

    data_paths = MESA_DATA_PATHS

    for window_minutes in [3, 5, 10]:
        print(f"\nTesting {window_minutes}-minute windows")

        train_loader, val_loader, test_loader, *datasets = get_windowed_dataloaders(
            data_path=data_paths,
            batch_size=4,
            num_workers=0,
            window_duration_minutes=window_minutes,
            overlap_ratio=0.1,
        )

        for i, (ppg, labels, subject_ids) in enumerate(train_loader):
            print(f"Batch {i + 1}: PPG {ppg.shape}, Labels {labels.shape}")
            if i >= 2:
                break

        print(
            f"Batches: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test"
        )
