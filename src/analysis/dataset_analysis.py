"""Look at the extracted HDF5 datasets: file layout, stage counts and one night's signal.

Prints the structure of the files each extractor wrote, how many subjects and scored
epochs they hold, and how a night divides between the four stages, then draws whichever
figures you ask for.

Examples (run from the repository root with PYTHONPATH=src):

    # Layout and tables for every dataset that has been extracted
    python src/analysis/dataset_analysis.py

    # Every figure, written to ./figures
    python src/analysis/dataset_analysis.py --save-dir figures

    # A minute of one subject's waveform
    python src/analysis/dataset_analysis.py --plot waveform --subject 1234
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.constants import (
    SAMPLES_PER_WINDOW,
    STAGES,
    TARGET_FS,
    WINDOW_DURATION,
    WINDOWS_PER_SUBJECT,
)
from common.paths import CFS_DATA_PATHS, HOMEPAP_DATA_PATHS, MESA_DATA_PATHS
from common.plotting import apply_plot_style, finish
from datasets.dataset_type import DatasetType

DATA_PATHS = {
    DatasetType.MESA: MESA_DATA_PATHS,
    DatasetType.CFS: CFS_DATA_PATHS,
    DatasetType.HOMEPAP: HOMEPAP_DATA_PATHS,
}


def describe_ppg_file(path):
    """Print the arrays and file attributes in a <dataset>_ppg_with_labels.h5."""
    with h5py.File(path, "r") as f:
        print(Path(path).name)
        for key, value in f.attrs.items():
            print(f"  attrs[{key!r}] = {value}")
        for name, dset in f.items():
            print(f"  {name}: shape {dset.shape}, dtype {dset.dtype}")


def describe_index_file(path, n_subjects=2):
    """Print the subject groups in a <dataset>_subject_index.h5."""
    with h5py.File(path, "r") as f:
        print(Path(path).name)
        for key, value in f.attrs.items():
            print(f"  attrs[{key!r}] = {value}")
        subjects = list(f["subjects"])
        print(f"  subjects: {len(subjects)} groups, first few {subjects[:5]}")
        for subject in subjects[:n_subjects]:
            group = f[f"subjects/{subject}"]
            indices = group["window_indices"]
            print(
                f"    {subject}: n_windows={group.attrs['n_windows']}, "
                f"window_indices {indices.shape} starting at row {indices[0]}"
            )


def _full_night_starts(index_path):
    """First row of every subject with a full 10-hour recording.

    Shorter recordings are dropped, which is the rule the dataset classes apply
    before feeding a model.
    """
    with h5py.File(index_path, "r") as f:
        return {
            subject: group["window_indices"][0]
            for subject, group in f["subjects"].items()
            if group.attrs["n_windows"] == WINDOWS_PER_SUBJECT
        }


def load_stage_counts(paths_by_dataset):
    """Epoch counts per stage for every full-night subject, one row each.

    Epochs the technician did not score carry label -1 and are left out of the stage
    columns.
    """
    rows = []
    for dataset, paths in paths_by_dataset.items():
        starts = _full_night_starts(paths["index"])
        with h5py.File(paths["ppg"], "r") as f:
            labels = f["labels"][:]

        for subject, start in starts.items():
            night = labels[start : start + WINDOWS_PER_SUBJECT]
            row = {
                "dataset": dataset.value,
                "subject": subject,
                "scored_epochs": int((night != -1).sum()),
            }
            for stage, name in enumerate(STAGES):
                row[name] = int((night == stage).sum())
            rows.append(row)

    return pd.DataFrame(rows)


def load_segment(paths, subject_id=None, start_epoch=300, n_epochs=2):
    """A few epochs of one subject's waveform, one row per sample.

    Each row carries its time in seconds, the PPG value, and the epoch and scored
    stage it falls in. The subject defaults to the first with a full recording.
    """
    starts = _full_night_starts(paths["index"])
    if subject_id is None:
        subject_id = next(iter(starts))
    elif subject_id not in starts:
        raise SystemExit(
            f"Subject {subject_id!r} has no full recording here. "
            f"First few available: {list(starts)[:5]}"
        )

    first = starts[subject_id] + start_epoch
    with h5py.File(paths["ppg"], "r") as f:
        ppg = f["ppg"][first : first + n_epochs]
        labels = f["labels"][first : first + n_epochs]

    epochs = np.arange(start_epoch, start_epoch + n_epochs)
    return pd.DataFrame(
        {
            "subject": subject_id,
            "second": np.arange(n_epochs * SAMPLES_PER_WINDOW) / TARGET_FS,
            "ppg": ppg.reshape(-1),
            "epoch": np.repeat(epochs, SAMPLES_PER_WINDOW),
            "stage": np.repeat(labels, SAMPLES_PER_WINDOW),
        }
    )


def summarise(counts):
    """One row per dataset: subject count, scored epochs and time per stage.

    Takes the frame returned by `load_stage_counts`. The percentages are over all
    scored epochs in the dataset, the minutes are what an average night spends in
    that stage.
    """
    rows = []
    for name, df in counts.groupby("dataset", sort=False):
        totals = df[STAGES].sum()
        row = {
            "dataset": name,
            "subjects": len(df),
            "scored_epochs": int(df.scored_epochs.sum()),
            "scored_hours_per_night": df.scored_epochs.mean() * WINDOW_DURATION / 3600,
        }
        for stage in STAGES:
            row[f"{stage} %"] = totals[stage] / totals.sum() * 100
            row[f"{stage} min"] = df[stage].mean() * WINDOW_DURATION / 60
        rows.append(row)

    return pd.DataFrame(rows)


def plot_stage_distribution(df, *, title=None, save_path=None):
    """Grouped bars of the share of scored epochs in each stage, per dataset.

    Takes the frame returned by `summarise`.
    """
    datasets = DatasetType.ordered(set(df.dataset))
    positions = np.arange(len(STAGES))
    width = 0.8 / len(datasets)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, dataset in enumerate(datasets):
        row = df[df.dataset == dataset].iloc[0]
        shares = [row[f"{stage} %"] for stage in STAGES]
        offset = (i - (len(datasets) - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            shares,
            width,
            label=dataset,
            color=DatasetType(dataset).colour,
        )
        for bar, share in zip(bars, shares):
            ax.annotate(
                f"{share:.1f}",
                (bar.get_x() + bar.get_width() / 2, share),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
            )

    ax.set_xticks(positions, STAGES)
    ax.set_ylabel("Share of scored epochs (%)")
    ax.legend()
    ax.set_title(title)

    finish(fig, save_path)


def plot_ppg_waveform(df, *, title=None, save_path=None):
    """The normalised waveform over a few epochs, annotated with the scored stage.

    Takes the frame returned by `load_segment`. Epoch boundaries are dotted so the
    30-second scoring grid is visible.
    """
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(df.second, df.ppg, color="black")

    for _, group in df.groupby("epoch"):
        boundary = group.second.iloc[0]
        ax.axvline(boundary, color="black", linestyle=":")
        stage = int(group.stage.iloc[0])
        ax.annotate(
            STAGES[stage] if stage >= 0 else "unscored",
            (boundary + WINDOW_DURATION / 2, df.ppg.max()),
            ha="center",
            va="bottom",
        )

    ax.margins(y=0.15)  # headroom for the stage labels
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalised PPG")
    ax.set_title(title)

    finish(fig, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the extracted HDF5 datasets and plot what is in them"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[d.value for d in DatasetType],
        choices=[d.value for d in DatasetType],
        help="Datasets to look at (missing processed data is skipped)",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject ID for the waveform segment, defaults to the first one",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=300,
        help="First epoch of the waveform segment (default: 300, about 2.5 hours in)",
    )
    parser.add_argument(
        "--segment-epochs",
        type=int,
        default=2,
        help="Epochs to draw in the waveform segment (default: 2, one minute)",
    )
    parser.add_argument(
        "--plot",
        default="all",
        choices=["all", "none", "stages", "waveform"],
        help="Figure to draw, or all or none (default: all)",
    )
    parser.add_argument(
        "--save-dir", default=None, help="Write PNGs here instead of opening windows"
    )
    parser.add_argument("--csv", default=None, help="Write the summary table here")
    args = parser.parse_args()

    apply_plot_style()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def out(name):
        return save_dir / name if save_dir else None

    available = {}
    for name in args.datasets:
        dataset = DatasetType(name)
        paths = DATA_PATHS[dataset]
        if all(Path(paths[key]).exists() for key in ("ppg", "index")):
            available[dataset] = paths
        else:
            print(f"{name}: no processed data at {paths['ppg']}, skipping")

    if not available:
        raise SystemExit("No extracted datasets found, check DATA_DIR")

    for dataset, paths in available.items():
        print(f"\n{dataset.value}")
        describe_ppg_file(paths["ppg"])
        describe_index_file(paths["index"])

    counts = load_stage_counts(available)
    summary = summarise(counts)
    print()
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(2).to_string(index=False))

    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")

    if args.plot in ("all", "stages"):
        plot_stage_distribution(
            summary,
            title="Sleep stage distribution",
            save_path=out("stage_distribution.png"),
        )

    if args.plot in ("all", "waveform"):
        # The segment is per subject, so take the first dataset asked for
        dataset, paths = next(iter(available.items()))
        segment = load_segment(
            paths, args.subject, args.start_epoch, args.segment_epochs
        )
        plot_ppg_waveform(
            segment,
            title=f"{dataset.value} subject {segment.subject.iloc[0]}",
            save_path=out("ppg_waveform.png"),
        )

    if save_dir:
        print(f"Wrote figures to {save_dir}")


if __name__ == "__main__":
    main()
