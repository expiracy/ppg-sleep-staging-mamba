"""Look at what a model does: its size and speed, and how it stages one night.

With `--config` it builds an architecture with random weights and runs it on random
input, which is the quickest check that the CUDA install works. With `--model-dir` it
loads a trained run, stages a subject from the matching test set and scores the
predictions against the PSG reference.

CUDA only, the Mamba kernels have no CPU path.

Examples (run from the repository root with PYTHONPATH=src):

    # Install check: random weights on a random 10-hour night
    python src/analysis/model_analysis.py --config configs/ssm_shallow_192.yaml

    # A trained run on the first MESA test subject
    python src/analysis/model_analysis.py --model-dir "$OUTPUT_DIR/models/<run_dir>"

    # A named subject, scored with the CFS fine-tuned weights
    python src/analysis/model_analysis.py --model-dir <run_dir> --dataset CFS \
        --variant CFS-TL --subject <subject_id> --save-dir figures
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common.constants import (
    CHECKPOINT_VARIANTS,
    SAMPLES_PER_WINDOW,
    STAGES,
    WINDOW_DURATION,
    WINDOWS_PER_SUBJECT,
)
from common.paths import CFS_DATA_PATHS, HOMEPAP_DATA_PATHS, MESA_DATA_PATHS
from common.plotting import (
    apply_plot_style,
    finish,
    model_colour,
    plot_confusion_matrix,
)
from datasets.dataset_type import DatasetType
from datasets.subject_dataset import SubjectDataset
from models.dual_stream_mamba import DualStreamMamba
from models.model_type import ModelType
from models.ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
from models.single_stream_mamba import SingleStreamMamba
from models.sleep_ppg_net import SleepPPGNet
from models.sleep_staging_model import SleepStagingModel

DATA_PATHS = {
    DatasetType.MESA: MESA_DATA_PATHS,
    DatasetType.CFS: CFS_DATA_PATHS,
    DatasetType.HOMEPAP: HOMEPAP_DATA_PATHS,
}

# Config file names start with the model prefix: ssm_*, dsm_*, or the baseline's name
PREFIX_TO_MODEL = {
    "ssm": (SingleStreamMamba, ModelType.SINGLE_STREAM_MAMBA),
    "dsm": (DualStreamMamba, ModelType.DUAL_STREAM_MAMBA),
    "sleepppgnet": (SleepPPGNet, ModelType.PPG_ONLY),
    "dsca": (PPGUnfilteredCrossAttention, ModelType.PPG_UNFILTERED),
}


def build_from_config(config_path):
    """Instantiate the architecture a training config describes, with random weights.

    Training configs carry no `model_type`, so the class comes from the file name.
    """
    path = Path(config_path)
    with open(path) as f:
        config = yaml.safe_load(f)

    prefix = path.stem.split("_")[0]
    if prefix not in PREFIX_TO_MODEL:
        raise SystemExit(
            f"Cannot infer the model type from '{path.name}', "
            f"config names must start with one of {sorted(PREFIX_TO_MODEL)}"
        )

    model_cls, model_type = PREFIX_TO_MODEL[prefix]
    return model_cls.from_config(config), model_type


def load_from_run_dir(model_dir, variant, checkpoint):
    """Load one checkpoint of a trained run, ready for inference."""
    wrapper = SleepStagingModel.from_model_dir(model_dir, {variant: checkpoint})
    if variant not in wrapper.variant_names:
        raise SystemExit(f"No {checkpoint} in {Path(model_dir) / 'checkpoints'}")

    wrapper.load(variant).eval()
    return wrapper.model, wrapper.model_type


def report_model(model):
    """Print the model's names, parameter count and static size."""
    n_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_bytes += sum(b.numel() * b.element_size() for b in model.buffers())

    print(f"Model:      {model.get_name()}")
    print(f"Short name: {model.get_short_name()}")
    print(f"Parameters: {n_params:,}  ({size_bytes / 1024 / 1024:.1f} MB)")


def stage_probabilities(model, ppg):
    """Run one timed forward pass and return (probabilities, seconds).

    The first pass compiles the CUDA kernels, so it is thrown away.
    """
    with torch.no_grad():
        model(ppg)
        torch.cuda.synchronize()
        start = time.perf_counter()
        probs = model(ppg)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return probs, elapsed


def load_test_subject(dataset_type, subject_id=None):
    """One subject from a dataset's fixed test split, the first by default."""
    dataset = SubjectDataset(
        data_path=DATA_PATHS[dataset_type], split="test", dataset_name=dataset_type
    )

    if subject_id is None:
        index = 0
    elif subject_id in dataset.subjects:
        index = dataset.subjects.index(subject_id)
    else:
        raise SystemExit(
            f"Subject {subject_id!r} is not in the {dataset_type.value} test split. "
            f"First few available: {dataset.subjects[:5]}"
        )

    return dataset[index]


def score(df):
    """Epoch count, accuracy and Cohen's kappa over the scored epochs."""
    scored = df[df.reference != -1]
    return {
        "epochs": len(scored),
        "accuracy": (scored.predicted == scored.reference).mean() * 100,
        "kappa": cohen_kappa_score(scored.reference, scored.predicted),
    }


def summarise_stages(df):
    """Minutes per stage, predicted against reference, for one night."""
    scored = df[df.reference != -1]
    rows = []
    for stage, name in enumerate(STAGES):
        rows.append(
            {
                "stage": name,
                "reference_min": (scored.reference == stage).sum()
                * WINDOW_DURATION
                / 60,
                "predicted_min": (scored.predicted == stage).sum()
                * WINDOW_DURATION
                / 60,
                "recall": (
                    (scored.predicted[scored.reference == stage] == stage).mean() * 100
                    if (scored.reference == stage).any()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def confusion_counts(df):
    """Epoch counts of scored stage (rows) against predicted stage (columns)."""
    scored = df[df.reference != -1]
    counts = confusion_matrix(
        scored.reference, scored.predicted, labels=range(len(STAGES))
    )
    return pd.DataFrame(counts, index=STAGES, columns=STAGES)


def plot_hypnograms(df, *, title=None, save_path=None):
    """Reference and predicted hypnograms, one panel each, Wake at the top.

    Takes the frame returned by `predict_subject`. Unscored reference epochs are left
    as gaps rather than drawn at Wake.
    """
    panels = [
        ("Reference (PSG)", df.reference.where(df.reference != -1), "black"),
        ("Predicted", df.predicted, model_colour(df.model_type.iloc[0])),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True, sharey=True)
    for ax, (label, stages, colour) in zip(axes, panels):
        ax.step(df.hour, stages, where="post", color=colour)
        ax.set_yticks(range(len(STAGES)), STAGES)
        ax.set_ylabel(label)

    # The axes share a y-axis, so inverting one inverts both
    axes[0].invert_yaxis()
    axes[-1].set_xlabel("Time (hours)")
    if title:  # an empty suptitle still takes up space
        fig.suptitle(title)

    finish(fig, save_path)


def predict_subject(model, model_type, dataset_type, subject_id=None):
    """Stage one test subject and return a frame of one row per 30-second epoch."""
    ppg, labels, subject = load_test_subject(dataset_type, subject_id)
    probs, elapsed = stage_probabilities(model, ppg.unsqueeze(0).to("cuda"))

    predicted = probs.argmax(dim=1).squeeze(0).cpu().numpy()
    reference = labels.numpy()
    samples = ppg.shape[-1]
    print(f"Subject:    {subject} ({samples:,} samples)")
    print(f"Output:     {tuple(probs.shape)}  (batch, stage probabilities, epochs)")
    print(
        f"Inference:  {elapsed * 1000:.1f} ms, {samples / elapsed / 1e6:.1f} M samples/s"
    )

    return pd.DataFrame(
        {
            "subject": subject,
            "model_type": model_type.value,
            "hour": np.arange(len(reference)) * WINDOW_DURATION / 3600,
            "reference": reference,
            "predicted": predicted,
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Report a model's size and speed, and stage one night with it"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Training config YAML, gives random weights")
    source.add_argument(
        "--model-dir", help="Trained run directory with config.yaml and checkpoints/"
    )
    parser.add_argument(
        "--dataset",
        default="MESA",
        choices=[d.value for d in DatasetType],
        help="Dataset whose test set to stage (default: MESA)",
    )
    parser.add_argument(
        "--variant",
        default="Base",
        choices=list(CHECKPOINT_VARIANTS),
        help="Checkpoint variant: Base, CFS-TL or HP-TL (default: Base)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename in the run's checkpoints/ directory "
        "(default: the variant's own)",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject ID to stage, defaults to the first test subject",
    )
    parser.add_argument(
        "--plot",
        default="all",
        choices=["all", "none", "hypnogram", "confusion"],
        help="Figure to draw, or all or none (default: all)",
    )
    parser.add_argument(
        "--save-dir", default=None, help="Write PNGs here instead of opening windows"
    )
    parser.add_argument(
        "--csv", default=None, help="Write the per-stage table here (with --model-dir)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required, mamba_ssm has no CPU kernels")

    apply_plot_style()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def out(name):
        return save_dir / name if save_dir else None

    if args.config:
        model, model_type = build_from_config(args.config)
        model = model.to("cuda").eval()
        report_model(model)

        # Random weights say nothing about agreement, so this is size and speed only
        ppg = torch.randn(1, 1, WINDOWS_PER_SUBJECT * SAMPLES_PER_WINDOW, device="cuda")
        probs, elapsed = stage_probabilities(model, ppg)
        samples = ppg.shape[-1]
        print(f"Input:      random PPG, a full night ({samples:,} samples)")
        print(f"Output:     {tuple(probs.shape)}  (batch, stage probabilities, epochs)")
        print(
            f"Inference:  {elapsed * 1000:.1f} ms, "
            f"{samples / elapsed / 1e6:.1f} M samples/s"
        )

        counts = torch.bincount(probs.argmax(dim=1).squeeze(0).cpu(), minlength=4)
        summary = ", ".join(f"{s}: {int(c)}" for s, c in zip(STAGES, counts))
        print(f"Predicted:  {summary}")
        return

    checkpoint = args.checkpoint or CHECKPOINT_VARIANTS[args.variant]
    model, model_type = load_from_run_dir(args.model_dir, args.variant, checkpoint)
    report_model(model)

    dataset_type = DatasetType(args.dataset)
    df = predict_subject(model, model_type, dataset_type, args.subject)
    metrics = score(df)
    print(
        f"Agreement:  kappa {metrics['kappa']:.3f}, "
        f"accuracy {metrics['accuracy']:.1f}% over {metrics['epochs']} scored epochs"
    )
    print()
    stages = summarise_stages(df)
    print(stages.round(1).to_string(index=False))

    if args.csv:
        stages.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")

    subject = df.subject.iloc[0]
    label = f"{model.get_short_name()}, {args.dataset} subject {subject}"

    if args.plot in ("all", "hypnogram"):
        plot_hypnograms(
            df,
            title=f"{label} (κ = {metrics['kappa']:.3f})",
            save_path=out("hypnograms.png"),
        )

    if args.plot in ("all", "confusion"):
        plot_confusion_matrix(
            confusion_counts(df), title=label, save_path=out("confusion_matrix.png")
        )

    if save_dir:
        print(f"Wrote figures to {save_dir}")


if __name__ == "__main__":
    main()
