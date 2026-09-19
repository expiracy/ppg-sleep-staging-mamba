"""Full-night (10-hour) PPG datasets for sleep stage classification.

MESA is used for training and in-domain evaluation, CFS and HomePAP for transfer
learning and out-of-domain evaluation. Each item is one subject's
complete recording: a (1, 1_228_800) PPG tensor and 1200 epoch labels.

The MESA dataset and the fixed SleepPPG-Net test split are sourced from
https://github.com/DavyWJW/sleep-staging-models/. CFS and HomePAP are extensions.
"""

import os

import h5py
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from common.constants import SAMPLES_PER_WINDOW, WINDOWS_PER_SUBJECT
from common.paths import INDEX_DATA_PATH, PPG_DATA_PATH
from datasets.test_subjects import (
    CFS_TEST_SUBJECTS,
    HOMEPAP_TEST_SUBJECTS,
    SLEEPPPG_TEST_SUBJECTS,
)


class PPGOnlyDataset(Dataset):
    """MESA full-night PPG dataset.

    With use_sleepppg_test_set=True the test split is the fixed 204-subject
    SleepPPG-Net test set and the remaining subjects are split 80/20 into
    train/val with the given seed."""

    def __init__(
        self,
        data_path,
        split="train",
        transform=None,
        seed=42,
        use_sleepppg_test_set=True,
    ):
        self.split = split
        self.transform = transform
        self.seed = seed
        self.use_sleepppg_test_set = use_sleepppg_test_set

        self.windows_per_subject = WINDOWS_PER_SUBJECT
        self.samples_per_window = SAMPLES_PER_WINDOW

        if isinstance(data_path, dict):
            self.ppg_file_path = data_path["ppg"]
            self.index_file_path = data_path["index"]
        else:
            self.ppg_file_path = (
                os.path.join(data_path, "mesa_ppg_with_labels.h5")
                if data_path
                else str(PPG_DATA_PATH)
            )
            self.index_file_path = (
                os.path.join(data_path, "mesa_subject_index.h5")
                if data_path
                else str(INDEX_DATA_PATH)
            )

        if not os.path.exists(self.ppg_file_path):
            raise FileNotFoundError(f"PPG file not found: {self.ppg_file_path}")
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"Index file not found: {self.index_file_path}")

        print(f"Loading PPG data from: {self.ppg_file_path}")

        self._prepare_subjects()

    def _prepare_subjects(self):
        with h5py.File(self.index_file_path, "r") as f:
            all_subjects = list(f["subjects"].keys())

            valid_subjects = []
            for subj in all_subjects:
                n_windows = f[f"subjects/{subj}"].attrs["n_windows"]
                if n_windows == self.windows_per_subject:
                    valid_subjects.append(subj)

        if self.use_sleepppg_test_set:
            test_subjects = [s for s in SLEEPPPG_TEST_SUBJECTS if s in valid_subjects]
            train_val_subjects = [s for s in valid_subjects if s not in test_subjects]

            print("Using SleepPPG-Net test set split:")
            print(f"  Test subjects: {len(test_subjects)}")
            print(f"  Train+Val subjects: {len(train_val_subjects)}")

            train_subjects, val_subjects = train_test_split(
                train_val_subjects, test_size=0.2, random_state=self.seed
            )
        else:
            train_subjects, test_subjects = train_test_split(
                valid_subjects, test_size=0.2, random_state=self.seed
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

        print(f"{self.split} set: {len(self.subjects)} subjects")

        self.subject_indices = {}
        with h5py.File(self.index_file_path, "r") as f:
            for subj in self.subjects:
                indices = f[f"subjects/{subj}/window_indices"][:]
                if len(indices) == self.windows_per_subject:
                    self.subject_indices[subj] = indices[0]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        start_idx = self.subject_indices[subject_id]

        with h5py.File(self.ppg_file_path, "r") as f:
            ppg_windows = f["ppg"][start_idx : start_idx + self.windows_per_subject]
            labels = f["labels"][start_idx : start_idx + self.windows_per_subject]

        ppg_continuous = ppg_windows.reshape(-1)

        if self.transform:
            ppg_continuous = self.transform(ppg_continuous)

        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
        labels_tensor = torch.LongTensor(labels)

        return ppg_tensor, labels_tensor, subject_id


def get_dataloaders(
    data_path,
    batch_size=1,
    num_workers=0,
    use_sleepppg_test_set=True,
):
    """Train/val/test dataloaders over full MESA recordings.

    Returns (train_loader, val_loader, test_loader,
             train_dataset, val_dataset, test_dataset).
    """
    train_dataset = PPGOnlyDataset(
        data_path, split="train", use_sleepppg_test_set=use_sleepppg_test_set
    )
    val_dataset = PPGOnlyDataset(
        data_path, split="val", use_sleepppg_test_set=use_sleepppg_test_set
    )
    test_dataset = PPGOnlyDataset(
        data_path, split="test", use_sleepppg_test_set=use_sleepppg_test_set
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


def verify_test_set():
    from common.paths import MESA_DATA_PATHS

    data_paths = MESA_DATA_PATHS

    test_dataset = PPGOnlyDataset(data_paths, split="test", use_sleepppg_test_set=True)

    print("\nTest set verification:")
    print(f"Number of test subjects: {len(test_dataset.subjects)}")
    print(f"Test subjects (first 10): {test_dataset.subjects[:10]}")

    matched = set(test_dataset.subjects) == set(SLEEPPPG_TEST_SUBJECTS)
    print(f"Matches SleepPPG-Net test set: {matched}")

    if not matched:
        missing = set(SLEEPPPG_TEST_SUBJECTS) - set(test_dataset.subjects)
        extra = set(test_dataset.subjects) - set(SLEEPPPG_TEST_SUBJECTS)
        if missing:
            print(f"Missing subjects: {missing}")
        if extra:
            print(f"Extra subjects: {extra}")


class CFSDataset(Dataset):
    """
    CFS (Cleveland Family Study) Dataset for PPG-based sleep staging.

    Uses fixed test set (50% of subjects) defined in CFS_TEST_SUBJECTS for
    consistent evaluation across runs.
    """

    def __init__(
        self, data_path, split="train", transform=None, seed=42, use_fixed_test_set=True
    ):
        self.split = split
        self.transform = transform
        self.seed = seed
        self.use_fixed_test_set = use_fixed_test_set

        self.windows_per_subject = WINDOWS_PER_SUBJECT
        self.samples_per_window = SAMPLES_PER_WINDOW

        if isinstance(data_path, dict):
            self.ppg_file_path = data_path["ppg"]
            self.index_file_path = data_path["index"]
        else:
            self.ppg_file_path = os.path.join(data_path, "cfs_ppg_with_labels.h5")
            self.index_file_path = os.path.join(data_path, "cfs_subject_index.h5")

        if not os.path.exists(self.ppg_file_path):
            raise FileNotFoundError(f"CFS PPG file not found: {self.ppg_file_path}")
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"CFS Index file not found: {self.index_file_path}")

        print(f"Loading CFS PPG data from: {self.ppg_file_path}")

        self._prepare_subjects()

    def _prepare_subjects(self):
        with h5py.File(self.index_file_path, "r") as f:
            all_subjects = list(f["subjects"].keys())

            valid_subjects = []
            for subj in all_subjects:
                n_windows = f[f"subjects/{subj}"].attrs["n_windows"]
                if n_windows == self.windows_per_subject:
                    valid_subjects.append(subj)

        if self.use_fixed_test_set:
            test_subjects = [s for s in CFS_TEST_SUBJECTS if s in valid_subjects]
            train_val_subjects = [s for s in valid_subjects if s not in test_subjects]

            print("Using fixed CFS test set split:")
            print(f"  Test subjects: {len(test_subjects)}")
            print(f"  Train+Val subjects: {len(train_val_subjects)}")

            train_subjects, val_subjects = train_test_split(
                train_val_subjects, test_size=0.2, random_state=self.seed
            )
        else:
            train_subjects, test_subjects = train_test_split(
                valid_subjects, test_size=0.5, random_state=self.seed
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

        print(f"{self.split} set: {len(self.subjects)} subjects")

        # Map subject -> row offset in the flat H5 arrays
        self.subject_indices = {}
        with h5py.File(self.index_file_path, "r") as f:
            for subj in self.subjects:
                indices = f[f"subjects/{subj}/window_indices"][:]
                if len(indices) == self.windows_per_subject:
                    self.subject_indices[subj] = indices[0]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        start_idx = self.subject_indices[subject_id]

        with h5py.File(self.ppg_file_path, "r") as f:
            ppg_windows = f["ppg"][start_idx : start_idx + self.windows_per_subject]
            labels = f["labels"][start_idx : start_idx + self.windows_per_subject]

        ppg_continuous = ppg_windows.reshape(-1)

        if self.transform:
            ppg_continuous = self.transform(ppg_continuous)

        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
        labels_tensor = torch.LongTensor(labels)

        return ppg_tensor, labels_tensor


class HomePAPDataset(Dataset):
    """
    HomePAP Dataset for PPG-based sleep staging.

    Uses fixed test set (50% of subjects) defined in HOMEPAP_TEST_SUBJECTS for
    consistent evaluation across runs.
    """

    def __init__(
        self, data_path, split="train", transform=None, seed=42, use_fixed_test_set=True
    ):
        self.split = split
        self.transform = transform
        self.seed = seed
        self.use_fixed_test_set = use_fixed_test_set

        self.windows_per_subject = WINDOWS_PER_SUBJECT
        self.samples_per_window = SAMPLES_PER_WINDOW

        if isinstance(data_path, dict):
            self.ppg_file_path = data_path["ppg"]
            self.index_file_path = data_path["index"]
        else:
            self.ppg_file_path = os.path.join(data_path, "homepap_ppg_with_labels.h5")
            self.index_file_path = os.path.join(data_path, "homepap_subject_index.h5")

        if not os.path.exists(self.ppg_file_path):
            raise FileNotFoundError(f"HomePAP PPG file not found: {self.ppg_file_path}")
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(
                f"HomePAP Index file not found: {self.index_file_path}"
            )

        print(f"Loading HomePAP PPG data from: {self.ppg_file_path}")

        self._prepare_subjects()

    def _prepare_subjects(self):
        with h5py.File(self.index_file_path, "r") as f:
            all_subjects = list(f["subjects"].keys())

            valid_subjects = []
            for subj in all_subjects:
                n_windows = f[f"subjects/{subj}"].attrs["n_windows"]
                if n_windows == self.windows_per_subject:
                    valid_subjects.append(subj)

        if self.use_fixed_test_set:
            test_subjects = [s for s in HOMEPAP_TEST_SUBJECTS if s in valid_subjects]
            train_val_subjects = [s for s in valid_subjects if s not in test_subjects]

            print("Using fixed HomePAP test set split:")
            print(f"  Test subjects: {len(test_subjects)}")
            print(f"  Train+Val subjects: {len(train_val_subjects)}")

            train_subjects, val_subjects = train_test_split(
                train_val_subjects, test_size=0.2, random_state=self.seed
            )
        else:
            train_subjects, test_subjects = train_test_split(
                valid_subjects, test_size=0.5, random_state=self.seed
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

        print(f"{self.split} set: {len(self.subjects)} subjects")

        # Map subject -> row offset in the flat H5 arrays
        self.subject_indices = {}
        with h5py.File(self.index_file_path, "r") as f:
            for subj in self.subjects:
                indices = f[f"subjects/{subj}/window_indices"][:]
                if len(indices) == self.windows_per_subject:
                    self.subject_indices[subj] = indices[0]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        start_idx = self.subject_indices[subject_id]

        with h5py.File(self.ppg_file_path, "r") as f:
            ppg_windows = f["ppg"][start_idx : start_idx + self.windows_per_subject]
            labels = f["labels"][start_idx : start_idx + self.windows_per_subject]

        ppg_continuous = ppg_windows.reshape(-1)

        if self.transform:
            ppg_continuous = self.transform(ppg_continuous)

        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
        labels_tensor = torch.LongTensor(labels)

        return ppg_tensor, labels_tensor


def get_transfer_dataloaders(
    dataset_name, data_path, batch_size=1, num_workers=0, seed=42
):
    """Train/test dataloaders for transfer learning on CFS or HomePAP.

    Uses ALL non-test subjects (train + val combined) for training
    and the fixed test set (50% of subjects) for evaluation.

    Returns (train_loader, test_loader, train_dataset, test_dataset).
    """
    DATASET_CLASSES = {
        "cfs": CFSDataset,
        "homepap": HomePAPDataset,
    }

    if dataset_name not in DATASET_CLASSES:
        available = ", ".join(DATASET_CLASSES.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    DatasetClass = DATASET_CLASSES[dataset_name]

    # Fixed test set
    test_dataset = DatasetClass(
        data_path, split="test", use_fixed_test_set=True, seed=seed
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Combine train + val subjects into a single training set
    train_dataset_full = DatasetClass(
        data_path, split="train", use_fixed_test_set=True, seed=seed
    )
    val_dataset_full = DatasetClass(
        data_path, split="val", use_fixed_test_set=True, seed=seed
    )

    all_train_subjects = train_dataset_full.subjects + val_dataset_full.subjects
    all_subject_indices = {
        **train_dataset_full.subject_indices,
        **val_dataset_full.subject_indices,
    }

    train_dataset = DatasetClass(
        data_path, split="train", use_fixed_test_set=True, seed=seed
    )
    train_dataset.subjects = all_train_subjects
    train_dataset.subject_indices = {
        s: all_subject_indices[s] for s in all_train_subjects
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"{dataset_name.upper()} Transfer Learning Setup:")
    print(f"  Training subjects: {len(all_train_subjects)}")
    print(f"  Fixed test subjects: {len(test_dataset.subjects)}")

    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == "__main__":
    verify_test_set()
