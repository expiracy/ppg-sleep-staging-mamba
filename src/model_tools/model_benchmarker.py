"""Benchmark trained models: accuracy, kappa, F1, throughput and VRAM on the
fixed test sets, for full-night inference and, with --windows, shorter windows."""

import gc
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


def results_to_dataframe(results):
    """Build a DataFrame from result dicts, skipping Nones and auto-numbering model_id."""
    rows = []
    model_num = 1
    for result in results:
        if result is None:
            continue
        row = result.copy()
        if row.get("model_id") is None:
            row["model_id"] = model_num
        model_num += 1
        rows.append(row)
    return pd.DataFrame(rows)


class CUDAMemoryMonitor:
    """CUDA VRAM monitor. Tracks peak and delta memory via PyTorch."""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.baseline_vram = 0

    def start(self):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize(self.device_id)
        # Clear the high-water mark so we only measure this run's peak
        torch.cuda.reset_peak_memory_stats(self.device_id)
        self.baseline_vram = torch.cuda.memory_allocated(self.device_id) / 1024 / 1024

    def stop(self):
        if not torch.cuda.is_available():
            return {"peak_mb": 0, "delta_mb": 0}

        torch.cuda.synchronize(self.device_id)
        max_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1024 / 1024

        return {
            "peak_mb": max_allocated,
            "delta_mb": max_allocated - self.baseline_vram,
        }


class Benchmarker:
    """Benchmarker for sleep staging models on PPG-only data.

    Performs warmup passes, then runs the full test set multiple times to
    collect stable timing, VRAM usage, accuracy, and Cohen's kappa metrics.
    Requires CUDA.
    """

    def __init__(self, warmup_runs=3, repetitions=3):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for benchmarking. No CUDA device available."
            )

        self.device = "cuda"
        self.warmup_runs = warmup_runs
        self.repetitions = repetitions

    def run(self, sleep_staging_model, data_loader, verbose=True):
        """Benchmark a single loaded model on a single data loader.

        Model must already be loaded (model.load(variant) called).
        Does NOT call load/unload - caller manages model lifecycle.
        """
        model_instance = sleep_staging_model.model  # raises if not loaded
        model_instance.eval()
        model_instance.to(self.device)

        if verbose:
            print(f"\nBenchmarking {sleep_staging_model.get_name()}")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Warmup once before all repetitions so CUDA kernels are compiled
        # and cached, giving stable timing from the first timed pass
        first_batch = next(iter(data_loader))
        test_data = first_batch[0]

        if verbose:
            print(f"  Warmup ({self.warmup_runs} passes)")

        with torch.no_grad():
            for _ in range(self.warmup_runs):
                warmup_data = test_data.to(self.device)
                _ = model_instance(warmup_data)
                del warmup_data

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # Run multiple passes and average metrics for more stable results
        rep_results = []
        for rep in range(self.repetitions):
            if verbose:
                print(f"  Pass {rep + 1}/{self.repetitions}")
            pass_start = time.perf_counter()
            result = self._single_run(model_instance, data_loader, verbose=verbose)
            pass_time = time.perf_counter() - pass_start
            if verbose:
                print(f"    Pass time: {pass_time:.2f}s")
            rep_results.append(result)

        if verbose:
            print("  Measuring model size")
        model_size_mb = sleep_staging_model.get_static_size_mb()

        merged = self._aggregate_results(rep_results)

        merged["static_model_size_mb"] = model_size_mb
        merged["checkpoint_file"] = str(sleep_staging_model.checkpoint_file)
        merged["model_id"] = sleep_staging_model.model_id
        merged["model_name"] = sleep_staging_model.get_name(include_model_id=False)
        merged["model_name_short"] = sleep_staging_model.get_short_name()
        merged["model_type"] = sleep_staging_model.model_type.value
        merged["variant"] = sleep_staging_model.variant

        gc.collect()
        torch.cuda.empty_cache()

        return merged

    def run_all(
        self,
        models,
        data_loaders,
        verbose=True,
        save_dir=None,
        window_minutes=-1,
    ):
        """Benchmark every model x variant x dataset combination.

        models: list of SleepStagingModel. data_loaders: dict of DatasetType
        to DataLoader. Handles the load/unload lifecycle. Iteration order:
        for model -> for variant -> load -> for dataset -> run -> unload
        (loads each variant once, tests all datasets, then unloads)
        """
        self._validate_unique_names(models)
        rows = []

        for model in models:
            for variant_name in model.variant_names:
                model.load(variant_name)
                for ds_name, loader in data_loaders.items():
                    result = self.run(model, loader, verbose=verbose)
                    result["dataset"] = ds_name.value
                    result["window_minutes"] = window_minutes
                    rows.append(result)
                model.unload()

        results = results_to_dataframe(rows)

        if verbose:
            n_variants = sum(len(m.variant_names) for m in models)
            n = n_variants * len(data_loaders)
            print(f"\nDone, {n} benchmark runs")

        if save_dir:
            self._save_results(results, save_dir)

        return results

    def _single_run(self, model_instance, data_loader, verbose=False):
        """Run a single benchmark pass (no warmup, no load/unload).

        Expects each batch to be a 3-tuple (data, labels, subject_ids).
        Predictions are grouped by subject_id so per-subject metrics are
        correct even when multiple windows from the same subject appear
        across different batches (windowed mode).
        """
        num_items = len(data_loader)
        monitor = CUDAMemoryMonitor(device_id=0)

        # Group predictions by subject so we can compute per-subject kappa/accuracy.
        # In windowed mode, a single subject's epochs get split across many batches,
        # so we can't just compute metrics per-batch.
        subject_preds = defaultdict(list)
        subject_labels = defaultdict(list)
        total_samples = 0
        batch_times = []

        gc.collect()
        torch.cuda.empty_cache()

        monitor.start()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                test_data, labels, subject_ids = batch[0], batch[1], batch[2]
                test_data = test_data.to(self.device)

                # Timed inference
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = model_instance(test_data)
                torch.cuda.synchronize()
                end = time.perf_counter()

                elapsed_ms = (end - start) * 1000
                batch_times.append(end - start)

                if verbose:
                    print(f"    Batch {i + 1}/{num_items} ({elapsed_ms:.1f} ms)")

                # Move to CPU before argmax to keep GPU memory free for next batch
                outputs_cpu = outputs.cpu()
                predictions = torch.argmax(outputs_cpu, dim=1)

                for j, sid in enumerate(subject_ids):
                    subject_preds[sid].append(predictions[j])
                    subject_labels[sid].append(labels[j])

                # Throughput counts raw PPG samples, input is (batch, 1, samples)
                total_samples += test_data.shape[0] * test_data.shape[2]

                del test_data, labels, outputs, predictions

        torch.cuda.synchronize()
        memory_results = monitor.stop()

        # Accuracy and Cohen's kappa per subject, then aggregated
        per_subject_kappas = []
        per_subject_accuracies = []
        all_preds_list = []
        all_labels_list = []

        for sid in subject_preds:
            preds_cat = torch.cat(subject_preds[sid]).flatten().numpy()
            labels_cat = torch.cat(subject_labels[sid]).flatten().numpy()

            # -1 marks unscored epochs and the zero-padding on short recordings,
            # neither counts towards the metrics
            valid_mask = labels_cat != -1
            preds_cat = preds_cat[valid_mask]
            labels_cat = labels_cat[valid_mask]

            if len(preds_cat) == 0:
                continue

            all_preds_list.append(preds_cat)
            all_labels_list.append(labels_cat)

            per_subject_accuracies.append((preds_cat == labels_cat).mean() * 100)
            try:
                per_subject_kappas.append(cohen_kappa_score(labels_cat, preds_cat))
            except ValueError:
                per_subject_kappas.append(float("nan"))

        # Overall metrics across all subjects (pooled predictions)
        if not all_preds_list:
            raise ValueError("No valid (non -1) labels found in any subject")

        preds_np = np.concatenate(all_preds_list)
        labels_np = np.concatenate(all_labels_list)
        accuracy = (preds_np == labels_np).mean() * 100
        kappa = cohen_kappa_score(labels_np, preds_np)
        f1_macro = f1_score(labels_np, preds_np, average="macro")

        # Per-class precision, recall, F1 from the confusion matrix.
        # labels=[0,1,2,3] maps to Wake, Light, Deep, REM
        cm = confusion_matrix(labels_np, preds_np, labels=[0, 1, 2, 3])
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        for i in range(4):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_class_precision.append(p)
            per_class_recall.append(r)
            per_class_f1.append(f)

        # Quartiles of the per-subject scores. Kappa is NaN for a subject scored as a
        # single stage, which the nan-aware versions skip.
        median_kappa = float(np.nanmedian(per_subject_kappas))
        q1_kappa = float(np.nanpercentile(per_subject_kappas, 25))
        q3_kappa = float(np.nanpercentile(per_subject_kappas, 75))

        median_accuracy = np.median(per_subject_accuracies)
        q1_accuracy = np.percentile(per_subject_accuracies, 25)
        q3_accuracy = np.percentile(per_subject_accuracies, 75)

        del subject_preds, subject_labels, all_preds_list, all_labels_list
        gc.collect()
        torch.cuda.empty_cache()

        total_time = sum(batch_times)
        throughput_samples = total_samples / total_time if total_time > 0 else 0

        results = {
            "accuracy_overall": accuracy,
            "accuracy_median": median_accuracy,
            "accuracy_q1": q1_accuracy,
            "accuracy_q3": q3_accuracy,
            "accuracy_per_subject": per_subject_accuracies,
            "kappa_overall": kappa,
            "kappa_median": median_kappa,
            "kappa_q1": q1_kappa,
            "kappa_q3": q3_kappa,
            "kappa_per_subject": per_subject_kappas,
            "f1_macro": f1_macro,
            "confusion_matrix_overall": cm.tolist(),
            "precision_per_class": per_class_precision,
            "recall_per_class": per_class_recall,
            "f1_per_class": per_class_f1,
            "throughput_samples_sec": throughput_samples,
            "vram_delta_mb": memory_results["delta_mb"],
        }

        return results

    def _aggregate_results(self, rep_results):
        """Mean of numeric metrics across repetitions.

        Non-numeric fields (confusion matrices, per-subject lists) are taken
        from the first run since averaging them wouldn't be meaningful.
        """
        first = rep_results[0]
        merged = {}

        for key in first:
            val = first[key]
            if isinstance(val, (int, float)):
                values = [r[key] for r in rep_results if key in r]
                if isinstance(val, int):
                    merged[key] = int(np.mean(values))
                else:
                    merged[key] = float(np.mean(values))
            else:
                # Strings, lists, None - take from first run
                merged[key] = val

        return merged

    @staticmethod
    def validate_models(sleep_staging_models):
        """Load and unload every variant to verify checkpoints are valid."""
        total = len(sleep_staging_models)
        for i, sleep_staging_model in enumerate(sleep_staging_models, 1):
            if sleep_staging_model is None:
                continue
            print(f"Validating ({i}/{total})")
            for variant in sleep_staging_model.variant_names:
                try:
                    sleep_staging_model.load(variant)
                    sleep_staging_model.unload()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to validate model ({i}/{total}): "
                        f"{sleep_staging_model.get_name()} ({variant})"
                    ) from e

    @staticmethod
    def _validate_unique_names(models):
        """Check that no two model x variant combinations share the same name."""
        seen = []
        for model in models:
            for variant in model.variant_names:
                key = (model.get_name(include_model_id=True), variant)
                seen.append(key)
        dupes = [k for k in seen if seen.count(k) > 1]
        if dupes:
            raise ValueError(f"Duplicate model+variant combinations: {set(dupes)}")

    def _save_results(self, results, save_dir):
        """Save combined and per-dataset Parquet files."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        results.to_parquet(save_dir / "results.parquet", index=False)

        for ds_name in results["dataset"].unique():
            ds_dir = save_dir / ds_name
            ds_dir.mkdir(exist_ok=True)
            filtered = results[results["dataset"] == ds_name].reset_index(drop=True)
            filtered.to_parquet(ds_dir / "results.parquet", index=False)

        print(f"Saved to {save_dir}")


def _discover_model_dirs(model_dirs, parent_dir):
    """Resolve the model directories to benchmark from the CLI arguments."""
    dirs = []
    if parent_dir:
        parent = Path(parent_dir)
        dirs.extend(
            p
            for p in sorted(parent.iterdir())
            if p.is_dir() and not p.name.startswith(".")
        )
    if model_dirs:
        dirs.extend(Path(p) for p in model_dirs)

    valid = []
    for d in dirs:
        has_config = (d / "config.yaml").exists() or (d / "config.json").exists()
        if has_config and (d / "checkpoints").is_dir():
            valid.append(d)
        else:
            print(f"Skipping {d}: no config.yaml or checkpoints/ directory")
    return valid


def main():
    import argparse

    from common.paths import (
        BENCHMARKS_DIR,
        MESA_DATA_PATHS,
        CFS_DATA_PATHS,
        HOMEPAP_DATA_PATHS,
        make_benchmark_name,
    )
    from common.constants import CHECKPOINT_VARIANTS
    from models.sleep_staging_model import SleepStagingModel
    from datasets.dataset_type import DatasetType
    from datasets.subject_dataset import get_subject_test_loader
    from datasets.windowed_dataset import get_windowed_test_loader
    from models.model_type import ModelType

    full_night_only = (ModelType.PPG_ONLY, ModelType.PPG_UNFILTERED)

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark trained models on the fixed test sets: accuracy, kappa, "
            "F1, throughput and VRAM, on full nights and optionally shorter windows."
        )
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=None,
        help="Model output directories (each with config.yaml and checkpoints/)",
    )
    parser.add_argument(
        "--models-parent-dir",
        default=None,
        help="Benchmark every model directory found directly under this folder",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MESA", "CFS", "HomePAP"],
        choices=["MESA", "CFS", "HomePAP"],
        help="Datasets to evaluate on (missing processed data is skipped)",
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        type=int,
        default=[],
        help="Also run windowed inference at these lengths in minutes (Mamba models "
        "only, the baselines take a full night)",
    )
    parser.add_argument(
        "--skip-full-night",
        action="store_true",
        help="Skip the full-night (10-hour) benchmark",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--name",
        default=None,
        help="Optional label for the benchmark output directory",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.model_dirs and not args.models_parent_dir:
        parser.error("Provide --model-dirs and/or --models-parent-dir")

    model_dirs = _discover_model_dirs(args.model_dirs, args.models_parent_dir)
    if not model_dirs:
        parser.error("No valid model directories found")

    all_paths = {
        DatasetType.MESA: MESA_DATA_PATHS,
        DatasetType.CFS: CFS_DATA_PATHS,
        DatasetType.HOMEPAP: HOMEPAP_DATA_PATHS,
    }
    dataset_paths = {}
    for ds in (DatasetType(name) for name in args.datasets):
        paths = all_paths[ds]
        if all(Path(paths[k]).exists() for k in ("ppg", "index")):
            dataset_paths[ds] = paths
        else:
            print(f"Skipping {ds.value}: processed data not found at {paths['ppg']}")
    if not dataset_paths:
        parser.error("No datasets with processed data available")

    save_dir = BENCHMARKS_DIR / make_benchmark_name(args.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {save_dir}")

    print(f"Found {len(model_dirs)} model directories:")
    for p in model_dirs:
        print(f"  {p.name}")

    # One variant per checkpoint file present in checkpoints/
    sleep_staging_models = SleepStagingModel.from_model_dirs(
        model_dirs,
        variant_to_checkpoint_filename=CHECKPOINT_VARIANTS,
    )
    print(f"{len(sleep_staging_models)} models loaded")

    benchmarker = Benchmarker(
        warmup_runs=args.warmup_runs, repetitions=args.repetitions
    )
    benchmarker.validate_models(sleep_staging_models)

    if not args.skip_full_night:
        print("Running full-night benchmarks")
        loaders = {
            ds: get_subject_test_loader(
                paths, num_workers=args.num_workers, dataset_name=ds
            )
            for ds, paths in dataset_paths.items()
        }
        results = benchmarker.run_all(
            sleep_staging_models, data_loaders=loaders, verbose=args.verbose
        )
        results.to_parquet(save_dir / "results_full.parquet", index=False)
        print(f"Saved {len(results)} rows to {save_dir / 'results_full.parquet'}")
        print(results.to_string())

    # The baselines reshape to 1,200 epochs inside their forward pass, so only the
    # Mamba models can run on a shorter window
    windowed_models = [
        m for m in sleep_staging_models if m.model_type not in full_night_only
    ]
    if args.windows and len(windowed_models) < len(sleep_staging_models):
        print("Skipping the baselines for windowed inference, they need a full night")
    if not windowed_models:
        return

    for window_minutes in args.windows:
        print(f"Running {window_minutes}-minute window benchmarks")
        loaders = {
            ds: get_windowed_test_loader(
                paths,
                batch_size=None,
                num_workers=args.num_workers,
                dataset_name=ds,
                window_duration_minutes=window_minutes,
            )
            for ds, paths in dataset_paths.items()
        }
        results = benchmarker.run_all(
            windowed_models,
            data_loaders=loaders,
            verbose=args.verbose,
            window_minutes=window_minutes,
        )
        out = save_dir / f"results_{window_minutes}min.parquet"
        results.to_parquet(out, index=False)
        print(f"Saved {len(results)} rows to {out}")
        print(results.to_string())


if __name__ == "__main__":
    main()
