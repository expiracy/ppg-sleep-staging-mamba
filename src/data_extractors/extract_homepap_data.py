"""
HomePAP Data Extractor

Extracts PPG signals and sleep-stage annotations from the HomePAP full in-lab
polysomnography recordings into the HDF5 layout described in the README.
HomePAP mixes PSG systems, so recordings with a low sampling rate or a
saturated PPG are left out.

Shares the preprocessing pipeline in preprocessing.py with the MESA and CFS
extractors.

    python src/data_extractors/extract_homepap_data.py
    python src/data_extractors/extract_homepap_data.py --list-signals
"""

import argparse
import os
from collections import Counter

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


class HomePAPDataExtractor:
    def __init__(
        self,
        homepap_path,
        output_path,
        min_ppg_fs=35,
        max_clip_pct=1.0,
        clip_range_frac=0.01,
    ):
        """Two quality filters sit on top of the shared pipeline.

        min_ppg_fs: recordings sampled below this are dropped rather than
            upsampled (some HomePAP PPG is 16-25 Hz, most is 64-256 Hz).
        max_clip_pct: recordings with more than this percentage of samples
            within clip_range_frac of the signal floor or ceiling are dropped
            as saturated.
        """
        self.homepap_path = homepap_path
        self.output_path = output_path
        self.max_clip_pct = max_clip_pct
        self.clip_range_frac = clip_range_frac
        self.target_fs = TARGET_FS
        # Never upsample, so the threshold can not sit below the target rate
        self.min_ppg_fs = max(min_ppg_fs, self.target_fs + 0.1)
        self.window_duration = WINDOW_DURATION
        self.samples_per_window = SAMPLES_PER_WINDOW
        self.target_windows = WINDOWS_PER_SUBJECT
        self.target_length = self.target_windows * self.samples_per_window

        self.edf_dir = os.path.join(
            homepap_path, "polysomnography", "edfs", "lab", "full"
        )
        self.xml_dir = os.path.join(
            homepap_path, "polysomnography", "annotations-events-nsrr", "lab", "full"
        )

        os.makedirs(output_path, exist_ok=True)
        self.processing_summary = {}

    def extract_ppg_from_edf(self, edf_file):
        """Read the PPG channel ("Pleth", "PLETH" or "TcCO2 Pleth" in HomePAP, at
        16-256 Hz). Returns (signal, fs) or (None, None)."""
        try:
            f = pyedflib.EdfReader(edf_file)
            signal_labels = f.getSignalLabels()

            # Tried in order, most specific first
            ppg_idx = None
            ppg_priorities = [
                lambda x: x == "pleth",
                lambda x: "pleth" in x and "tcco2" not in x,
                lambda x: "pleth" in x,  # falls back to TcCO2 Pleth
                lambda x: "ppg" in x,
            ]

            for priority_func in ppg_priorities:
                for idx, label in enumerate(signal_labels):
                    if priority_func(label.lower()):
                        fs = f.getSampleFrequency(idx)
                        if fs >= 10:  # skip low-rate channels like pulse rate
                            ppg_idx = idx
                            break
                if ppg_idx is not None:
                    break

            if ppg_idx is None:
                print(f"No PPG signal found in {os.path.basename(edf_file)}")
                f.close()
                return None, None

            ppg_signal = f.readSignal(ppg_idx)
            ppg_fs = f.getSampleFrequency(ppg_idx)

            # Many HomePAP files declare a physical range of -32768 to 32767, which
            # is the digital range, so the channel was never calibrated. Rescale
            # those to [-1, 1] and drop the flat ones.
            ppg_phys_min = f.getPhysicalMinimum(ppg_idx)
            ppg_phys_max = f.getPhysicalMaximum(ppg_idx)

            if abs(ppg_phys_min - (-32768)) < 1 and abs(ppg_phys_max - 32767) < 1:
                ppg_min = np.min(ppg_signal)
                ppg_max = np.max(ppg_signal)
                ppg_range = ppg_max - ppg_min

                if ppg_range > 0:
                    ppg_signal = 2.0 * (ppg_signal - ppg_min) / ppg_range - 1.0
                else:
                    print(f"Flat PPG signal in {os.path.basename(edf_file)}")
                    f.close()
                    return None, None

            f.close()

            return ppg_signal, ppg_fs

        except Exception as e:
            print(f"Error reading {os.path.basename(edf_file)}: {e}")
            return None, None

    def compute_clipping_stats(self, signal_data):
        """Percentage of samples sitting near the signal ceiling and near its floor.
        A saturated sensor piles samples up at one or both."""
        max_val = float(np.max(signal_data))
        min_val = float(np.min(signal_data))
        value_range = max_val - min_val

        if value_range == 0:
            return 100.0, 100.0

        near_ceil = (
            np.mean(signal_data > max_val - self.clip_range_frac * value_range) * 100
        )
        near_floor = (
            np.mean(signal_data < min_val + self.clip_range_frac * value_range) * 100
        )
        return float(near_ceil), float(near_floor)

    def process_subject(self, edf_file, xml_file):
        """Extract PPG, check for clipping, preprocess, parse sleep stages, and
        reshape into fixed-length windows for one subject.

        Returns a dict whose "status" is "ok", "no_ppg" or "clipped", since the
        caller counts each reason for dropping a recording.
        """
        ppg, ppg_fs = self.extract_ppg_from_edf(edf_file)

        if ppg is None:
            return {"status": "no_ppg"}

        # Checked on the raw signal, before filtering smooths the flat tops
        pct_ceil, pct_floor = self.compute_clipping_stats(ppg)
        if max(pct_ceil, pct_floor) > self.max_clip_pct:
            return {
                "status": "clipped",
                "clip_ceil_pct": pct_ceil,
                "clip_floor_pct": pct_floor,
            }

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
        return {
            "status": "ok",
            "ppg_windows": ppg_windows,
            "labels": labels_final,
            "clip_ceil_pct": pct_ceil,
            "clip_floor_pct": pct_floor,
        }

    def process_all_subjects(self, subject_list=None):
        """Extract every recording that passes the filters and write the HDF5 files.
        Returns the number of windows written. `subject_list` limits the run to
        filenames containing one of the given IDs."""
        edf_files = sorted(f for f in os.listdir(self.edf_dir) if f.endswith(".edf"))

        if subject_list:
            edf_files = [
                f for f in edf_files if any(subj in f for subj in subject_list)
            ]

        all_ppg_windows = []
        all_labels = []
        subject_ids = []
        ppg_sampling_rates = []  # rate of each recording before resampling
        failed_subjects = 0
        no_ppg_subjects = 0
        low_fs_subjects = 0
        clipped_subjects = 0

        for edf_filename in tqdm(edf_files, desc="Processing HomePAP subjects"):
            edf_path = os.path.join(self.edf_dir, edf_filename)

            # homepap-lab-full-1600001.edf -> 1600001
            subject_id = edf_filename.split("-")[-1].replace(".edf", "")

            xml_filename = edf_filename.replace(".edf", "-nsrr.xml")
            xml_path = os.path.join(self.xml_dir, xml_filename)

            try:
                # Read the PPG rate from the header first, so a low-rate recording
                # is skipped without loading its signal
                f = pyedflib.EdfReader(edf_path)
                signal_labels = f.getSignalLabels()
                original_ppg_fs = None
                for idx, label in enumerate(signal_labels):
                    if "pleth" in label.lower() or "ppg" in label.lower():
                        original_ppg_fs = f.getSampleFrequency(idx)
                        break
                f.close()

                if original_ppg_fs is None:
                    no_ppg_subjects += 1
                    continue

                if original_ppg_fs < self.min_ppg_fs:
                    low_fs_subjects += 1
                    continue

                result = self.process_subject(edf_path, xml_path)

                if result["status"] == "no_ppg":
                    no_ppg_subjects += 1
                elif result["status"] == "clipped":
                    clipped_subjects += 1
                else:
                    ppg_windows = result["ppg_windows"]
                    all_ppg_windows.append(ppg_windows)
                    all_labels.append(result["labels"])
                    subject_ids.extend([subject_id] * len(ppg_windows))
                    ppg_sampling_rates.extend([original_ppg_fs] * len(ppg_windows))
            except Exception as e:
                print(f"Error processing {edf_filename}: {e}")
                failed_subjects += 1
                continue

        print("\nProcessing summary:")
        print(f"  Subjects without PPG: {no_ppg_subjects}")
        print(f"  Subjects skipped (fs < {self.min_ppg_fs} Hz): {low_fs_subjects}")
        print(
            f"  Subjects skipped (clipping > {self.max_clip_pct}%): {clipped_subjects}"
        )
        print(f"  Failed subjects: {failed_subjects}")
        print(f"  Total subjects processed: {len(all_ppg_windows)}")

        self.processing_summary = {
            "no_ppg_subjects": no_ppg_subjects,
            "low_fs_subjects": low_fs_subjects,
            "clipped_subjects": clipped_subjects,
            "failed_subjects": failed_subjects,
            "min_ppg_fs": float(self.min_ppg_fs),
            "max_clip_pct": float(self.max_clip_pct),
            "clip_range_frac": float(self.clip_range_frac),
        }

        if all_ppg_windows:
            all_ppg_windows = np.vstack(all_ppg_windows)
            all_labels = np.concatenate(all_labels)
            subject_ids = np.array(subject_ids)
            ppg_sampling_rates = np.array(ppg_sampling_rates)

            self.save_data(all_ppg_windows, all_labels, subject_ids, ppg_sampling_rates)

            return len(all_ppg_windows)

        return 0

    def save_data(self, ppg_windows, labels, subject_ids, ppg_sampling_rates):
        """Save the PPG windows with labels and the per-subject index as HDF5. Same
        layout as MESA and CFS, plus each recording's original sampling rate."""
        ppg_file = os.path.join(self.output_path, "homepap_ppg_with_labels.h5")
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
            f.create_dataset(
                "original_ppg_fs", data=ppg_sampling_rates, compression="gzip"
            )

            f.attrs["sampling_rate"] = self.target_fs
            f.attrs["window_duration"] = self.window_duration
            f.attrs["samples_per_window"] = self.samples_per_window
            f.attrs["total_windows"] = len(ppg_windows)
            f.attrs["total_subjects"] = len(np.unique(subject_ids))

        index_file = os.path.join(self.output_path, "homepap_subject_index.h5")
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
                subj_group.attrs["original_ppg_fs"] = float(
                    ppg_sampling_rates[indices[0]]
                )

            f.attrs["total_subjects"] = len(unique_subjects)
            f.attrs["total_windows"] = len(ppg_windows)

        self.save_statistics(ppg_windows, labels, subject_ids, ppg_sampling_rates)

    def save_statistics(self, ppg_windows, labels, subject_ids, ppg_sampling_rates):
        """Save dataset statistics as .npy and a plain-text summary."""
        valid_labels = labels[labels != -1]
        unique_subjects = np.unique(subject_ids)

        # One rate per subject, taken from the subject's first window
        first_rows = np.array(
            [np.where(subject_ids == s)[0][0] for s in unique_subjects]
        )
        ppg_fs_distribution = Counter(ppg_sampling_rates[first_rows])

        stats = {
            "dataset": "HomePAP",
            "total_windows": len(ppg_windows),
            "total_subjects": len(unique_subjects),
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
            "ppg_fs_distribution": dict(ppg_fs_distribution),
            "file_structure": {
                "homepap_ppg_with_labels.h5": [
                    "ppg",
                    "labels",
                    "subject_ids",
                    "original_ppg_fs",
                ],
                "homepap_subject_index.h5": ["subjects/{subject_id}/window_indices"],
            },
        }

        if self.processing_summary:
            stats["processing_summary"] = dict(self.processing_summary)

        stats_file = os.path.join(self.output_path, "data_stats.npy")
        np.save(stats_file, stats)

        stats_txt = os.path.join(self.output_path, "data_stats.txt")
        with open(stats_txt, "w") as f:
            f.write("HomePAP Sleep Data Processing Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write("Dataset: HomePAP (Home Positive Airway Pressure)\n")
            f.write("Data type: Full in-lab PSG recordings\n\n")
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
            f.write("\nOriginal PPG sampling rate distribution (by subject):\n")
            for fs, count in sorted(stats["ppg_fs_distribution"].items()):
                f.write(f"  {fs} Hz: {count} subjects\n")

        print(f"Statistics saved to {stats_file} and {stats_txt}")


def main():
    from common.paths import DATA_DIR_HOMEPAP, DATA_DIR_HOMEPAP_PROCESSED

    parser = argparse.ArgumentParser(description="Extract HomePAP recordings into HDF5")
    parser.add_argument(
        "--list-signals",
        action="store_true",
        help="Print the channel labels found in the EDF files and exit",
    )
    args = parser.parse_args()

    print(f"HomePAP data path: {DATA_DIR_HOMEPAP}")
    print(f"Output path: {DATA_DIR_HOMEPAP_PROCESSED}")

    extractor = HomePAPDataExtractor(DATA_DIR_HOMEPAP, DATA_DIR_HOMEPAP_PROCESSED)
    if args.list_signals:
        print_signal_labels(extractor.edf_dir)
        return

    n_windows = extractor.process_all_subjects()
    print(f"\nDone, {n_windows} windows written")


if __name__ == "__main__":
    main()
