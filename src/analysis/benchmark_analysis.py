"""Read the Parquet files written by model_benchmarker.py, then table and plot them.

Prints a model comparison, per-stage recall and F1, per-subject kappa quartiles, the
VIKOR ranking and the transfer learning gains, then draws the matching figures. The
README's Benchmarking section lists every column these files hold.

Examples (run from the repository root with PYTHONPATH=src):

    # Newest benchmark run, tables and every figure
    python src/analysis/benchmark_analysis.py

    # Every row the run holds, before any slicing
    python src/analysis/benchmark_analysis.py --show --plot none

    # A particular run, figures written to ./figures
    python src/analysis/benchmark_analysis.py --benchmark-dir "$OUTPUT_DIR/benchmarks/<run>" \
        --save-dir figures

    # One figure, for the CFS fine-tuned checkpoints
    python src/analysis/benchmark_analysis.py --plot vikor --dataset CFS --variant CFS-TL

    # Confusion matrix for a single model
    python src/analysis/benchmark_analysis.py --plot confusion --model "SS-M[192, Shallow](base)"
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from common.constants import CHECKPOINT_VARIANTS, STAGES
from common.paths import BENCHMARKS_DIR
from common.plotting import (
    FALLBACK_COLOUR,
    LINE_WIDTH,
    THICK_LINE_WIDTH,
    apply_plot_style,
    finish,
    model_colour,
    plot_confusion_matrix,
)
from datasets.dataset_type import DatasetType
from models.model_type import ModelType

# Scalar columns printed by --show
SHOW_COLUMNS = [
    "model_name_short",
    "variant",
    "dataset",
    "kappa_overall",
    "kappa_median",
    "accuracy_overall",
    "f1_macro",
    "throughput_samples_sec",
    "vram_delta_mb",
    "static_model_size_mb",
]

# VIKOR as the paper sets it up: half the weight on agreement, the other half split
# evenly across the three costs. Benefit criteria are better high, cost criteria low.
VIKOR_WEIGHTS = {
    "kappa_overall": 0.5,
    "throughput_samples_sec": 1 / 6,
    "vram_delta_mb": 1 / 6,
    "static_model_size_mb": 1 / 6,
}
VIKOR_CRITERIA = {
    "kappa_overall": "benefit",
    "throughput_samples_sec": "benefit",
    "vram_delta_mb": "cost",
    "static_model_size_mb": "cost",
}
# Each VIKOR score: the panel title, then what it is and which way is better
SCORE_LABELS = {
    "Q": ("Q (Compromise Score)", "S and R blended, rescaled 0 to 1, lower is better"),
    "S": ("S (Group Utility)", "total weighted gap from the ideal, lower is better"),
    "R": ("R (Individual Regret)", "largest gap on any one criterion, lower is better"),
}
CRITERION_LABELS = {
    "kappa_overall": "Cohen's κ",
    "throughput_samples_sec": "Throughput",
    "vram_delta_mb": "VRAM",
    "static_model_size_mb": "Model size",
}
# Criteria have no enum to take colours from. These stay clear of the ModelType
# colours, since the VIKOR figure shows both side by side.
CRITERION_COLOURS = {
    "kappa_overall": "#b39ddb",
    "throughput_samples_sec": "#fff176",
    "vram_delta_mb": "#bcaaa4",
    "static_model_size_mb": "#80deea",
}
MODEL_TYPE_LABELS = {
    ModelType.SINGLE_STREAM_MAMBA.value: "SS-M, single stream",
    ModelType.DUAL_STREAM_MAMBA.value: "DS-M, dual stream",
    ModelType.PPG_ONLY.value: "SleepPPG-Net, baseline",
    ModelType.PPG_UNFILTERED.value: "DS-CA, baseline",
}
STRIPE_COLOUR = "#f0f0f0"

# The transfer learning pairs: the MESA weights scored on a cohort against the
# checkpoint fine-tuned on that cohort. MESA itself has no fine-tuned variant.
TRANSFER_PAIRS = [
    ("MESA", "k_base_mesa", None),
    ("CFS", "k_base_cfs", "k_cfs_tl"),
    ("HomePAP", "k_base_hp", "k_hp_tl"),
]


def load_benchmarks(benchmark_dir):
    """Read the full-night results of a benchmark directory."""
    path = Path(benchmark_dir) / "results_full.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No results_full.parquet in {benchmark_dir}")
    return pd.read_parquet(path)


def newest_benchmark(benchmarks_dir):
    """The most recent benchmark directory holding full-night results.

    Names end in the run's timestamp, with any --name before it, so directories are
    ordered on the timestamp alone.
    """
    runs = [
        d
        for d in benchmarks_dir.glob("benchmark_*")
        if (d / "results_full.parquet").exists()
    ]
    if not runs:
        raise SystemExit(f"No benchmark with results_full.parquet in {benchmarks_dir}")
    return max(runs, key=lambda d: d.name[-len("YYYYmmdd_HHMMSS") :])


def summarise_models(df):
    """One row per model: agreement, cost and size, best agreement first.

    A sweep can hold several runs of the same configuration, so rows sharing a short
    name are averaged and counted in the `runs` column.
    """
    return (
        df.groupby(["model_name_short", "model_type"], as_index=False)
        .agg(
            kappa=("kappa_overall", "mean"),
            accuracy=("accuracy_overall", "mean"),
            f1_macro=("f1_macro", "mean"),
            throughput=("throughput_samples_sec", "mean"),
            vram_delta_mb=("vram_delta_mb", "mean"),
            size_mb=("static_model_size_mb", "mean"),
            runs=("kappa_overall", "size"),
        )
        .sort_values("kappa", ascending=False)
        .reset_index(drop=True)
    )


def summarise_stages(df):
    """Per-stage recall and F1 for each model, averaged over repeated runs."""
    rows = []
    for name, group in df.groupby("model_name_short"):
        recall = np.mean([list(r) for r in group.recall_per_class], axis=0)
        f1 = np.mean([list(f) for f in group.f1_per_class], axis=0)
        row = {"model_name_short": name}
        for i, stage in enumerate(STAGES):
            row[f"{stage} recall"] = recall[i]
            row[f"{stage} F1"] = f1[i]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("model_name_short").reset_index(drop=True)


def summarise_subject_spread(df):
    """Quartiles of each model's per-subject kappa, pooled over repeated runs.

    `scores` counts the subject values behind each row, so a configuration trained
    twice contributes twice.
    """
    rows = []
    for name, group in df.groupby("model_name_short"):
        scores = np.concatenate(
            [np.asarray(values, dtype=float) for values in group.kappa_per_subject]
        )
        rows.append(
            {
                "model_name_short": name,
                "runs": len(group),
                "scores": len(scores),
                "min": np.nanmin(scores),
                "q1": np.nanpercentile(scores, 25),
                "median": np.nanmedian(scores),
                "q3": np.nanpercentile(scores, 75),
                "max": np.nanmax(scores),
            }
        )

    return pd.DataFrame(rows).set_index("model_name_short")


def confusion_counts(df):
    """Epoch counts of scored stage (rows) against predicted stage (columns).

    Sums the confusion matrices of every row in `df`, so one row gives one model and
    several rows pool them.
    """
    counts = sum(
        np.array([list(row) for row in cm], dtype=int)
        for cm in df.confusion_matrix_overall
    )
    return pd.DataFrame(counts, index=STAGES, columns=STAGES)


def vikor(df, weights=VIKOR_WEIGHTS, criteria=VIKOR_CRITERIA, v=0.5):
    """Rank models on agreement against cost. Lower `vikor_Q` is better.

    Each model's regret on a criterion is its weighted, normalised distance from the
    best model on that criterion, kept as a `<criterion>_regret` column. S sums a
    model's regrets, R keeps its worst one, and Q blends the two with `v` as the
    weight on S. Every score is relative to the set being ranked, so adding or
    removing a model moves them all. Repeated runs of a configuration are averaged
    first.
    """
    columns = list(weights)
    data = df.groupby(["model_name_short", "model_type"], as_index=False)[
        columns
    ].mean()

    weight_vec = np.array([weights[c] for c in columns], dtype=float)
    weight_vec /= weight_vec.sum()
    is_benefit = np.array([criteria[c] == "benefit" for c in columns])

    matrix = data[columns].to_numpy(dtype=float)
    f_best = np.where(is_benefit, matrix.max(axis=0), matrix.min(axis=0))
    f_worst = np.where(is_benefit, matrix.min(axis=0), matrix.max(axis=0))
    spread = f_best - f_worst
    spread[spread == 0] = 1  # a criterion every model ties on contributes nothing
    regret = weight_vec * (f_best - matrix) / spread

    S = regret.sum(axis=1)
    R = regret.max(axis=1)
    S_range = (S.max() - S.min()) or 1.0
    R_range = (R.max() - R.min()) or 1.0
    Q = v * (S - S.min()) / S_range + (1 - v) * (R - R.min()) / R_range

    result = data[["model_name_short", "model_type"]].copy()
    result["vikor_S"] = S
    result["vikor_R"] = R
    result["vikor_Q"] = Q
    for j, column in enumerate(columns):
        result[f"{column}_regret"] = regret[:, j]
    result = result.sort_values("vikor_Q").reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result


def _strip_variant(model_name_short):
    """SS-M[192, Shallow](base) -> SS-M[192, Shallow]."""
    return re.sub(r"\([^()]*\)$", "", model_name_short)


def _mean_kappa(df, model_name, variant, dataset):
    matched = df[
        (df.model_name == model_name)
        & (df.variant == variant)
        & (df.dataset == dataset)
    ].kappa_overall
    return float(matched.mean()) if not matched.empty else np.nan


def _nanmean(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def summarise_domain_generalisation(df):
    """Transfer learning gain and best cross-dataset agreement, one row per model.

    `g_cfs` and `g_hp` are how much fine-tuning lifts kappa on CFS and HomePAP over the
    MESA-trained weights scored there, and `g_target` is their mean. `kappa_best`
    averages, over the three datasets, the best kappa any variant reached on each.
    Variants are matched by `model_name`, since the short name carries the variant.
    """
    rows = []
    for model_name, group in df.groupby("model_name"):
        row = {
            "model_name_short": _strip_variant(group.model_name_short.iloc[0]),
            "model_type": group.model_type.iloc[0],
            "k_base_mesa": _mean_kappa(df, model_name, "Base", "MESA"),
            "k_base_cfs": _mean_kappa(df, model_name, "Base", "CFS"),
            "k_base_hp": _mean_kappa(df, model_name, "Base", "HomePAP"),
            "k_cfs_tl": _mean_kappa(df, model_name, "CFS-TL", "CFS"),
            "k_hp_tl": _mean_kappa(df, model_name, "HP-TL", "HomePAP"),
        }
        row["g_cfs"] = row["k_cfs_tl"] - row["k_base_cfs"]
        row["g_hp"] = row["k_hp_tl"] - row["k_base_hp"]
        row["g_target"] = _nanmean([row["g_cfs"], row["g_hp"]])
        row["kappa_best"] = _nanmean(
            [
                group[group.dataset == dataset].kappa_overall.max()
                for dataset in ("MESA", "CFS", "HomePAP")
            ]
        )
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("g_target", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _model_type_handles(model_types):
    """Legend patches for the model families present, in ModelType order."""
    known = [t.value for t in ModelType if t.value in model_types]
    handles = [
        Patch(facecolor=model_colour(t), label=MODEL_TYPE_LABELS[t]) for t in known
    ]
    if set(model_types) - set(known):
        handles.append(Patch(facecolor=FALLBACK_COLOUR, label="Other architecture"))
    return handles


def _criterion_handles(columns):
    """Legend patches for the criteria named, in the order given."""
    return [
        Patch(facecolor=CRITERION_COLOURS[c], label=CRITERION_LABELS[c])
        for c in columns
    ]


def plot_vikor_ranking(df, *, title=None, save_path=None):
    """VIKOR per model: the Q that ranks them, then the S and R behind it.

    Takes the frame returned by `vikor`, and draws its first row, the best Q, at the
    top. Q bars are coloured by model family. S stacks each model's regrets by
    criterion, and R is the single worst regret, coloured by the criterion
    responsible.
    """
    columns = list(VIKOR_WEIGHTS)
    regret = df[[f"{c}_regret" for c in columns]].to_numpy()
    worst = regret.argmax(axis=1)
    positions = np.arange(len(df))

    fig, (ax_q, ax_s, ax_r) = plt.subplots(
        1, 3, figsize=(14, len(df) * 0.4 + 1.5), sharey=True
    )

    ax_q.barh(positions, df.vikor_Q, color=[model_colour(t) for t in df.model_type])

    left = np.zeros(len(df))
    for j, column in enumerate(columns):
        ax_s.barh(positions, regret[:, j], left=left, color=CRITERION_COLOURS[column])
        left += regret[:, j]

    ax_r.barh(
        positions,
        df.vikor_R,
        color=[CRITERION_COLOURS[columns[i]] for i in worst],
    )

    for ax, score in ((ax_q, "Q"), (ax_s, "S"), (ax_r, "R")):
        values = df[f"vikor_{score}"]
        for y, value in zip(positions, values):
            ax.annotate(
                f"{value:.2f}",
                (value, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
            )

        ax.set_xlim(0, values.max() * 1.2 or 1.0)  # a lone model has zero regret
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        panel_title, description = SCORE_LABELS[score]
        ax.set_xlabel(description)
        ax.set_title(panel_title)

    ax_q.set_yticks(positions, df.model_name_short)
    ax_q.invert_yaxis()

    # A legend on the panel it explains. The shortest bars are at the top, so the
    # upper right of each panel is clear.
    ax_q.legend(
        handles=_model_type_handles(set(df.model_type)),
        loc="upper right",
        title="Q: architecture",
    )
    ax_s.legend(
        handles=_criterion_handles(columns),
        loc="upper right",
        title="S: criterion",
    )
    # R shows one criterion per model, so its legend covers only those that turned up
    ax_r.legend(
        handles=_criterion_handles([c for i, c in enumerate(columns) if i in worst]),
        loc="upper right",
        title="R: worst criterion",
    )
    if title:  # an empty suptitle still takes up space
        fig.suptitle(title)

    finish(fig, save_path)


def plot_domain_generalisation(df, *, title=None, save_path=None):
    """Dumbbells from the MESA weights to the fine-tuned ones, per cohort.

    Takes the frame returned by `summarise_domain_generalisation`, and draws its first
    row, the largest transfer gain, at the top. Each model row has a dot for MESA and,
    for CFS and HomePAP, a triangle at the MESA-trained kappa joined to a dot at the
    fine-tuned kappa.
    """
    data = df.reset_index(drop=True)
    positions = np.arange(len(data))
    slot = 0.25  # vertical room for each cohort within a model's row

    fig, ax = plt.subplots(figsize=(6.4, len(data) * 0.4 + 1.5))
    for y in positions[::2]:
        ax.axhspan(y - 0.5, y + 0.5, color=STRIPE_COLOUR, linewidth=0, zorder=0)

    marker = {"s": 50, "zorder": 2, "edgecolors": "black", "linewidths": LINE_WIDTH}
    for i, row in data.iterrows():
        for k, (dataset, base_col, tl_col) in enumerate(TRANSFER_PAIRS):
            y = positions[i] + (k - 1) * slot
            colour = DatasetType(dataset).colour
            short = "HP" if dataset == "HomePAP" else dataset
            first = i == 0  # legend entries once, and only for cohorts with data

            if tl_col is None:
                if not np.isnan(row[base_col]):
                    ax.scatter(
                        row[base_col],
                        y,
                        facecolors=colour,
                        label=f"Base on {short}" if first else None,
                        **marker,
                    )
                continue

            if np.isnan(row[base_col]) and np.isnan(row[tl_col]):
                continue
            ax.plot(
                [row[base_col], row[tl_col]],
                [y, y],
                color="black",
                linewidth=THICK_LINE_WIDTH,
                zorder=1,
            )
            ax.scatter(
                row[base_col],
                y,
                marker="^",
                facecolors=colour,
                label=f"Base on {short}" if first else None,
                **marker,
            )
            ax.scatter(
                row[tl_col],
                y,
                facecolors=colour,
                label=f"{short}-TL on {short}" if first else None,
                **marker,
            )

    kappas = data[[pair[1] for pair in TRANSFER_PAIRS] + ["k_cfs_tl", "k_hp_tl"]]
    low, high = np.nanmin(kappas.values), np.nanmax(kappas.values)
    pad = max((high - low) * 0.05, 1e-6)
    ax.set_xlim(low - pad, high + pad)

    for i, row in data.iterrows():
        ax.annotate(
            f"$G_{{target}}$ = {row.g_target:.3f}\n"
            f"$\\kappa_{{best}}$ = {row.kappa_best:.3f}",
            (high + pad, positions[i]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            clip_on=False,
        )

    ax.set_yticks(positions, data.model_name_short)
    ax.set_ylim(positions[0] - 0.5, positions[-1] + 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's κ")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)

    finish(fig, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Table and plot the Parquet results written by model_benchmarker.py"
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="A benchmark_<name>_<timestamp> directory, defaults to the newest one",
    )
    parser.add_argument(
        "--dataset",
        default="MESA",
        choices=[d.value for d in DatasetType],
        help="Dataset to report (default: MESA)",
    )
    parser.add_argument(
        "--variant",
        default="Base",
        choices=list(CHECKPOINT_VARIANTS),
        help="Checkpoint variant: Base, CFS-TL or HP-TL (default: Base)",
    )
    parser.add_argument(
        "--model", default=None, help="model_name_short for the confusion matrix"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print every row the run holds before slicing, one line each",
    )
    parser.add_argument(
        "--plot",
        default="all",
        choices=["all", "none", "confusion", "vikor", "domain"],
        help="Figure to draw, or all or none (default: all)",
    )
    parser.add_argument(
        "--save-dir", default=None, help="Write PNGs here instead of opening windows"
    )
    parser.add_argument("--csv", default=None, help="Write the model table here")
    args = parser.parse_args()

    apply_plot_style()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def out(name):
        return save_dir / name if save_dir else None

    if args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir)
    else:
        benchmark_dir = newest_benchmark(BENCHMARKS_DIR)
        print(f"Using the newest run: {benchmark_dir}")

    df = load_benchmarks(benchmark_dir)

    if args.show:
        with pd.option_context(
            "display.width", 250, "display.max_columns", None, "display.max_rows", None
        ):
            print(f"\n{len(df)} rows in {benchmark_dir.name}")
            print(df[SHOW_COLUMNS].round(3).to_string(index=False))

    # One dataset and variant, the usual comparison view
    full = df[(df.dataset == args.dataset) & (df.variant == args.variant)]
    if full.empty:
        raise SystemExit(
            f"No rows for dataset={args.dataset} variant={args.variant}. "
            f"Available: {sorted(df.dataset.unique())} x {sorted(df.variant.unique())}"
        )

    # The transfer learning metrics compare variants across datasets, so use every row
    transfers = df.variant.nunique() > 1 or df.dataset.nunique() > 1

    models = summarise_models(full)
    ranking = vikor(full)
    label = f"{args.dataset}, {args.variant}"
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\n{label}, full-night inference")
        print(models.round(3).to_string(index=False))
        print()
        print(summarise_stages(full).round(3).to_string(index=False))
        print()
        print(summarise_subject_spread(full).round(3).to_string())
        print(f"\nVIKOR ranking ({label}), lower Q is better")
        print(ranking.round(3).to_string(index=False))
        if transfers:
            print("\nDomain generalisation, every dataset and variant")
            print(summarise_domain_generalisation(df).round(3).to_string(index=False))
        else:
            print(
                "\nOne dataset and variant only, so no domain generalisation to report"
            )

    if args.csv:
        models.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")

    if args.plot in ("all", "confusion"):
        rows = full if args.model is None else full[full.model_name_short == args.model]
        if rows.empty:
            raise SystemExit(f"No model matching {args.model!r}")
        name = args.model or f"{len(rows)} models pooled"
        plot_confusion_matrix(
            confusion_counts(rows),
            title=f"{name}\n{label}",
            save_path=out("confusion_matrix.png"),
        )

    if args.plot in ("all", "vikor"):
        plot_vikor_ranking(
            ranking,
            title=f"VIKOR ranking ({label})",
            save_path=out("vikor_ranking.png"),
        )

    if args.plot in ("all", "domain"):
        if transfers:
            plot_domain_generalisation(
                summarise_domain_generalisation(df),
                title="Domain generalisation across MESA, CFS and HomePAP",
                save_path=out("domain_generalisation.png"),
            )
        else:
            print("Skipping the domain plot, it needs more than one dataset or variant")

    if save_dir:
        print(f"Wrote figures to {save_dir}")


if __name__ == "__main__":
    main()
