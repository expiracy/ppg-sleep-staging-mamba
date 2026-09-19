"""
MESA Data Extractor

Extracts PPG signals and sleep-stage annotations from MESA polysomnography
recordings into the HDF5 layout described in the README.

Adapted from https://github.com/DavyWJW/sleep-staging-models and shares the
preprocessing pipeline in preprocessing.py with the CFS and HomePAP extractors.

    python src/data_extractors/extract_mesa_data.py
    python src/data_extractors/extract_mesa_data.py --list-signals
"""

import argparse
import os

import h5py
import numpy as np
import pyedflib
from tqdm import tqdm

from common.constants import (
    SAMPLES_PER_WINDOW,
    TARGET_FS,
    WINDOW_DURATION,
    WINDOWS_PER_SUBJECT,
)
from data_extractors.preprocessing import (
    expand_labels_to_windows,
    pad_or_truncate,
    parse_sleep_stages_xml,
    print_signal_labels,
    process_ppg,
)


class MESADataExtractor:
    def __init__(self, mesa_path, output_path):
        self.mesa_path = mesa_path
        self.output_path = output_path
        self.target_fs = TARGET_FS
        self.window_duration = WINDOW_DURATION
        self.samples_per_window = SAMPLES_PER_WINDOW
        self.target_windows = WINDOWS_PER_SUBJECT
        self.target_length = self.target_windows * self.samples_per_window

        self.edf_dir = os.path.join(mesa_path, "polysomnography", "edfs")
        self.xml_dir = os.path.join(
            mesa_path, "polysomnography", "annotations-events-nsrr"
        )

        os.makedirs(output_path, exist_ok=True)

    def extract_ppg_from_edf(self, edf_file):
        """Read the PPG channel ("Pleth" in MESA). Returns (signal, fs) or (None, None)."""
        try:
            f = pyedflib.EdfReader(edf_file)
            signal_labels = f.getSignalLabels()

            ppg_idx = None
            for idx, label in enumerate(signal_labels):
                label_lower = label.lower()
                if "pleth" in label_lower or "ppg" in label_lower:
                    ppg_idx = idx
                    break

            if ppg_idx is None:
                print(f"No PPG signal found in {os.path.basename(edf_file)}")
                f.close()
                return None, None

            ppg_signal = f.readSignal(ppg_idx)
            ppg_fs = f.getSampleFrequency(ppg_idx)
            f.close()

            return ppg_signal, ppg_fs

        except Exception as e:
            print(f"Error reading {os.path.basename(edf_file)}: {e}")
            return None, None

    def process_subject(self, edf_file, xml_file):
        """Extract PPG, preprocess, parse sleep stages, and reshape into
        fixed-length windows for one subject."""
        ppg, ppg_fs = self.extract_ppg_from_edf(edf_file)

        if ppg is None:
            return None

        ppg_processed = process_ppg(ppg, ppg_fs)
        ppg_final = pad_or_truncate(ppg_processed, self.target_length)

        # A recording with no usable annotation file keeps its PPG, all unscored
        if os.path.exists(xml_file):
            df_stages = parse_sleep_stages_xml(xml_file)
            if df_stages is not None:
                total_duration = len(ppg_processed) / self.target_fs
                labels = expand_labels_to_windows(df_stages, total_duration)
                labels_final = pad_or_truncate(
                    labels, self.target_windows, fill_value=-1
                )
            else:
                labels_final = np.full(self.target_windows, -1, dtype=int)
        else:
            labels_final = np.full(self.target_windows, -1, dtype=int)

        ppg_windows = ppg_final.reshape(self.target_windows, self.samples_per_window)
        return ppg_windows, labels_final

    def process_all_subjects(self, subject_list=None):
        """Extract every recording and write the HDF5 files. Returns the number of
        windows written. `subject_list` limits the run to filenames containing one
        of the given IDs."""
        edf_files = sorted(f for f in os.listdir(self.edf_dir) if f.endswith(".edf"))

        if subject_list:
            edf_files = [
                f for f in edf_files if any(subj in f for subj in subject_list)
            ]

        all_ppg_windows = []
        all_labels = []
        subject_ids = []
        failed_subjects = 0

        for edf_filename in tqdm(edf_files, desc="Processing MESA subjects"):
            edf_path = os.path.join(self.edf_dir, edf_filename)

            # mesa-sleep-0001.edf -> 0001
            subject_id = edf_filename.split("-")[2].split(".")[0]

            xml_filename = edf_filename.replace(".edf", "-nsrr.xml")
            xml_path = os.path.join(self.xml_dir, xml_filename)

            try:
                result = self.process_subject(edf_path, xml_path)

                if result is not None:
                    ppg_windows, labels = result
                    all_ppg_windows.append(ppg_windows)
                    all_labels.append(labels)
                    subject_ids.extend([subject_id] * len(ppg_windows))
                else:
                    failed_subjects += 1
            except Exception as e:
                print(f"Error processing {edf_filename}: {e}")
                failed_subjects += 1
                continue

        print("\nProcessing summary:")
        print(f"  Failed subjects: {failed_subjects}")
        print(f"  Total subjects processed: {len(all_ppg_windows)}")

        if all_ppg_windows:
            all_ppg_windows = np.vstack(all_ppg_windows)
            all_labels = np.concatenate(all_labels)
            subject_ids = np.array(subject_ids)

            self.save_data(all_ppg_windows, all_labels, subject_ids)

            return len(all_ppg_windows)

        return 0

    def save_data(self, ppg_windows, labels, subject_ids):
        """Save the PPG windows with labels and the per-subject index as HDF5."""
        ppg_file = os.path.join(self.output_path, "mesa_ppg_with_labels.h5")
        print(f"\nSaving PPG data to {ppg_file}...")
        with h5py.File(ppg_file, "w") as f:
            f.create_dataset(
                "ppg",
                data=ppg_windows,
                compression="gzip",
                chunks=(100, self.samples_per_window),
            )
            f.create_dataset("labels", data=labels, compression="gzip")
            f.create_dataset(
                "subject_ids", data=subject_ids.astype("S10"), compression="gzip"
            )

            f.attrs["sampling_rate"] = self.target_fs
            f.attrs["window_duration"] = self.window_duration
            f.attrs["samples_per_window"] = self.samples_per_window
            f.attrs["total_windows"] = len(ppg_windows)
            f.attrs["total_subjects"] = len(np.unique(subject_ids))

        index_file = os.path.join(self.output_path, "mesa_subject_index.h5")
        print(f"Creating index file {index_file}...")
        with h5py.File(index_file, "w") as f:
            unique_subjects = np.unique(subject_ids)
            subject_group = f.create_group("subjects")

            for subj in unique_subjects:
                subj_str = subj.decode() if isinstance(subj, bytes) else str(subj)
                indices = np.where(subject_ids == subj)[0]

                subj_group = subject_group.create_group(subj_str)
                subj_group.create_dataset("window_indices", data=indices)
                subj_group.attrs["n_windows"] = len(indices)

            f.attrs["total_subjects"] = len(unique_subjects)
            f.attrs["total_windows"] = len(ppg_windows)

        self.save_statistics(ppg_windows, labels, subject_ids)

    def save_statistics(self, ppg_windows, labels, subject_ids):
        """Save dataset statistics as .npy and a plain-text summary."""
        valid_labels = labels[labels != -1]

        stats = {
            "dataset": "MESA",
            "total_windows": len(ppg_windows),
            "total_subjects": len(np.unique(subject_ids)),
            "ppg_shape": ppg_windows.shape,
            "valid_labels": len(valid_labels),
            "label_distribution": dict(
                zip(*np.unique(valid_labels, return_counts=True))
            )
            if len(valid_labels) > 0
            else {},
            "sampling_rate": self.target_fs,
            "window_duration": self.window_duration,
            "samples_per_window": self.samples_per_window,
            "file_structure": {
                "mesa_ppg_with_labels.h5": ["ppg", "labels", "subject_ids"],
                "mesa_subject_index.h5": ["subjects/{subject_id}/window_indices"],
            },
        }

        stats_file = os.path.join(self.output_path, "data_stats.npy")
        np.save(stats_file, stats)

        stats_txt = os.path.join(self.output_path, "data_stats.txt")
        with open(stats_txt, "w") as f:
            f.write("MESA Sleep Data Processing Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write("Dataset: MESA (Multi-Ethnic Study of Atherosclerosis)\n")
            f.write("Data type: MESA Sleep PSG recordings\n\n")
            f.write(f"Total windows: {stats['total_windows']}\n")
            f.write(f"Total subjects: {stats['total_subjects']}\n")
            f.write(f"Valid labels: {stats['valid_labels']}\n")
            f.write("\nLabel distribution:\n")
            stage_names = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
            for label, count in stats["label_distribution"].items():
                f.write(f"  {stage_names.get(label, f'Stage{label}')}: {count}\n")
            f.write(f"\nTarget sampling rate: {stats['sampling_rate']:.3f} Hz\n")
            f.write(f"Window duration: {stats['window_duration']} seconds\n")
            f.write(f"Samples per window: {stats['samples_per_window']}\n")

        print(f"Statistics saved to {stats_file} and {stats_txt}")


def main():
    from common.paths import DATA_DIR_MESA, DATA_DIR_MESA_PROCESSED

    parser = argparse.ArgumentParser(description="Extract MESA recordings into HDF5")
    parser.add_argument(
        "--list-signals",
        action="store_true",
        help="Print the channel labels found in the EDF files and exit",
    )
    args = parser.parse_args()

    print(f"MESA data path: {DATA_DIR_MESA}")
    print(f"Output path: {DATA_DIR_MESA_PROCESSED}")

    extractor = MESADataExtractor(DATA_DIR_MESA, DATA_DIR_MESA_PROCESSED)
    if args.list_signals:
        print_signal_labels(extractor.edf_dir)
        return

    n_windows = extractor.process_all_subjects()
    print(f"\nDone, {n_windows} windows written")


if __name__ == "__main__":
    main()
