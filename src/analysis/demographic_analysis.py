"""Describe who is in each cohort: age, sex, race and apnoea severity.

Reads the NSRR dataset CSV for each cohort, keeps the subjects that survived
extraction, prints the cohort tables and draws whichever figures you ask for. A cohort
missing its CSV or its extracted H5 index is skipped.

Examples (run from the repository root with PYTHONPATH=src):

    # The cohort table and the composition tables
    python src/analysis/demographic_analysis.py

    # Every figure, written to ./figures
    python src/analysis/demographic_analysis.py --save-dir figures

    # Apnoea severity for one cohort
    python src/analysis/demographic_analysis.py --datasets MESA --plot ahi
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.constants import WINDOWS_PER_SUBJECT
from common.demographics import (
    AHI_BINS,
    AHI_LABELS,
    COMMON_RACES,
    DATASET_CONFIGS,
    RACE_MAPS,
    SEX_LABELS,
    get_processed_subject_ids,
    load_demographics,
)
from common.plotting import THICK_LINE_WIDTH, apply_plot_style, finish
from datasets.dataset_type import DatasetType
from datasets.test_subjects import TEST_SUBJECTS

SEXES = list(SEX_LABELS.values())

# Sex, race and apnoea severity have no enum to take colours from
SEX_COLOURS = {"Female": "#f4b6b6", "Male": "#a6c8e8"}
RACE_COLOURS = {
    "White": "#a3c4f3",
    "Black": "#f4b6b6",
    "Asian": "#b5e6b5",
    "Hispanic": "#f0d6a4",
    "Other": "#d5c8e6",
}
AHI_COLOURS = {
    "Normal": "#b2e2cc",
    "Mild": "#fcd5b5",
    "Moderate": "#f2b8d1",
    "Severe": "#f5e6c8",
}


def load_cohorts(names):
    """Demographics of every extracted subject, one row each, tagged with `dataset`.

    Sex and race are mapped to the labels shared across cohorts, and `ahi_band` adds
    the clinical apnoea severity band.
    """
    frames = []
    for name in names:
        cfg = DATASET_CONFIGS[DatasetType(name)]
        if not Path(cfg["csv_path"]).exists():
            print(f"{name}: no demographics CSV at {cfg['csv_path']}, skipping")
            continue
        if not Path(cfg["index_h5_path"]).exists():
            print(f"{name}: no processed data at {cfg['index_h5_path']}, skipping")
            continue

        df = load_demographics(DatasetType(name))
        frames.append(
            df.assign(
                dataset=name,
                sex=df.sex.map(SEX_LABELS),
                race=df.race.map(RACE_MAPS[name]),
                ahi_band=pd.cut(df.ahi, bins=AHI_BINS, labels=AHI_LABELS),
            )
        )

    if not frames:
        raise SystemExit("No cohorts found, check DATA_DIR")
    return pd.concat(frames, ignore_index=True)


def load_label_coverage(dataset):
    """How many of a cohort's epoch slots carry a stage.

    Every subject occupies a fixed 1,200 epoch block, so the scored share is how much
    of that block the recordings cover.
    """
    cfg = DATASET_CONFIGS[dataset]
    subject_ids = get_processed_subject_ids(cfg["index_h5_path"])

    with h5py.File(cfg["index_h5_path"], "r") as f:
        starts = {s: f[f"subjects/{s}/window_indices"][0] for s in subject_ids}
    with h5py.File(cfg["ppg_h5_path"], "r") as f:
        labels = f["labels"][:]

    blocks = np.concatenate(
        [labels[starts[s] : starts[s] + WINDOWS_PER_SUBJECT] for s in subject_ids]
    )
    # -1 marks padding and unscored epochs
    return len(blocks), int(np.sum(blocks != -1))


def summarise(people):
    """One row per cohort: size, split, age, sex, apnoea severity and label coverage.

    Test counts come from the fixed subject lists in datasets.test_subjects. `ahi_n`
    counts the subjects with an AHI value, since some cohorts leave it blank.
    """
    rows = []
    for name, df in people.groupby("dataset", sort=False):
        dataset = DatasetType(name)
        extracted = set(
            get_processed_subject_ids(DATASET_CONFIGS[dataset]["index_h5_path"])
        )
        n_test = len([s for s in TEST_SUBJECTS[dataset] if s in extracted])
        slots, scored = load_label_coverage(dataset)

        rows.append(
            {
                "dataset": name,
                "subjects": len(df),
                "train_val": len(df) - n_test,
                "test": n_test,
                "age": f"{df.age.mean():.1f} +/- {df.age.std():.1f}",
                "female_pct": df.sex.eq("Female").mean() * 100,
                "ahi": f"{df.ahi.mean():.1f} +/- {df.ahi.std():.1f}",
                "ahi_n": int(df.ahi.notna().sum()),
                "ppg_rate_hz": DATASET_CONFIGS[dataset]["ppg_rate"],
                "scored_epochs_pct": scored / slots * 100,
            }
        )

    return pd.DataFrame(rows)


def summarise_shares(people, column, categories):
    """Share and count of each category per cohort, over subjects with a value.

    Used for sex, race and apnoea severity band. Columns are `<category> %` and
    `<category> n`.
    """
    rows = []
    for name, df in people.groupby("dataset", sort=False):
        counts = df[column].value_counts().reindex(categories, fill_value=0)
        total = df[column].notna().sum()
        row = {"dataset": name}
        for category in categories:
            row[f"{category} %"] = counts[category] / total * 100 if total else 0
            row[f"{category} n"] = int(counts[category])
        rows.append(row)

    return pd.DataFrame(rows)


def _grouped_bars(ax, df, categories, colours):
    """Bars per cohort, one per category, annotated with share and count."""
    datasets = DatasetType.ordered(set(df.dataset))
    x = np.arange(len(datasets))
    width = 0.8 / len(categories)

    for i, category in enumerate(categories):
        offset = (i - (len(categories) - 1) / 2) * width
        shares = [df.loc[df.dataset == d, f"{category} %"].iloc[0] for d in datasets]
        counts = [df.loc[df.dataset == d, f"{category} n"].iloc[0] for d in datasets]
        bars = ax.bar(
            x + offset, shares, width, label=category, color=colours[category]
        )
        for bar, share, count in zip(bars, shares, counts):
            if count:
                ax.annotate(
                    f"{share:.1f}\nn={count}",
                    (bar.get_x() + bar.get_width() / 2, share),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                )

    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of subjects (%)")
    ax.legend()


def plot_age_distribution(df, *, title=None, save_path=None):
    """Violins of the age spread per cohort, with the median marked.

    Takes the frame returned by `load_cohorts`.
    """
    datasets = DatasetType.ordered(set(df.dataset))
    positions = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for position, dataset in zip(positions, datasets):
        ages = df.loc[df.dataset == dataset, "age"].dropna()
        parts = ax.violinplot(
            [ages], positions=[position], widths=0.5, showmedians=True, showextrema=True
        )
        for body in parts["bodies"]:
            body.set_facecolor(DatasetType(dataset).colour)
            body.set_alpha(1.0)
        for key in ("cmins", "cmaxes", "cbars", "cmedians"):
            parts[key].set_color("black")
        parts["cmedians"].set_linewidth(THICK_LINE_WIDTH)

    ax.set_xticks(positions, datasets)
    ax.set_ylabel("Age (years)")
    ax.set_title(title)

    finish(fig, save_path)


def plot_sex_distribution(df, *, title=None, save_path=None):
    """Grouped bars of the sex split per cohort. Takes `summarise_shares` on sex."""
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    _grouped_bars(ax, df, SEXES, SEX_COLOURS)
    ax.set_title(title)

    finish(fig, save_path)


def plot_race_distribution(df, *, title=None, save_path=None):
    """Stacked bars of racial composition per cohort. Takes `summarise_shares` on race.

    Races absent from every cohort are left out, and only segments worth at least 5%
    are labelled, since thinner ones have no room for the text.
    """
    datasets = DatasetType.ordered(set(df.dataset))
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bottoms = np.zeros(len(datasets))
    for race in COMMON_RACES:
        shares = np.array(
            [df.loc[df.dataset == d, f"{race} %"].iloc[0] for d in datasets]
        )
        if not shares.sum():
            continue
        ax.bar(x, shares, 0.6, bottom=bottoms, label=race, color=RACE_COLOURS[race])
        for position, bottom, share in zip(x, bottoms, shares):
            if share >= 5:
                ax.annotate(
                    f"{share:.1f}",
                    (position, bottom + share / 2),
                    ha="center",
                    va="center",
                )
        bottoms += shares

    ax.set_xticks(x, datasets)
    ax.set_ylabel("Share of subjects (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title(title)

    finish(fig, save_path)


def plot_ahi_distribution(df, *, title=None, save_path=None):
    """Grouped bars of apnoea severity per cohort. Takes `summarise_shares` on ahi_band."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _grouped_bars(ax, df, AHI_LABELS, AHI_COLOURS)
    ax.set_xlabel("AHI severity band, events per hour")
    ax.set_title(title)

    finish(fig, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Summarise and plot the demographics of the extracted cohorts"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[d.value for d in DatasetType],
        choices=[d.value for d in DatasetType],
        help="Cohorts to look at (missing CSVs or processed data are skipped)",
    )
    parser.add_argument(
        "--plot",
        default="all",
        choices=["all", "none", "age", "sex", "race", "ahi"],
        help="Figure to draw, or all or none (default: all)",
    )
    parser.add_argument(
        "--save-dir", default=None, help="Write PNGs here instead of opening windows"
    )
    parser.add_argument("--csv", default=None, help="Write the cohort table here")
    args = parser.parse_args()

    apply_plot_style()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def out(name):
        return save_dir / name if save_dir else None

    people = load_cohorts(args.datasets)
    summary = summarise(people)
    sex = summarise_shares(people, "sex", SEXES)
    race = summarise_shares(people, "race", COMMON_RACES)
    ahi = summarise_shares(people, "ahi_band", AHI_LABELS)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nCohorts")
        print(summary.round(1).to_string(index=False))
        print("\nSex")
        print(sex.round(1).to_string(index=False))
        print("\nRace")
        print(race.round(1).to_string(index=False))
        print("\nApnoea severity")
        print(ahi.round(1).to_string(index=False))

    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")

    if args.plot in ("all", "age"):
        plot_age_distribution(
            people, title="Age by cohort", save_path=out("age_distribution.png")
        )

    if args.plot in ("all", "sex"):
        plot_sex_distribution(
            sex, title="Sex by cohort", save_path=out("sex_distribution.png")
        )

    if args.plot in ("all", "race"):
        plot_race_distribution(
            race,
            title="Racial composition by cohort",
            save_path=out("race_distribution.png"),
        )

    if args.plot in ("all", "ahi"):
        plot_ahi_distribution(
            ahi,
            title="Apnoea severity by cohort",
            save_path=out("ahi_distribution.png"),
        )

    if save_dir:
        print(f"Wrote figures to {save_dir}")


if __name__ == "__main__":
    main()
