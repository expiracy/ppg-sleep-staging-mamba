"""Transfer learning trainer for sleep staging models.

Fine-tunes a pretrained model on a target dataset (CFS / HomePAP).
Trains on all non-test subjects and evaluates on the fixed test set.
"""

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from tqdm import tqdm

from common.gpu_utils import setup_gpu_memory_limit
from common.paths import CFS_DATA_PATHS, HOMEPAP_DATA_PATHS
from datasets.full_night_dataset import get_transfer_dataloaders
from models.sleep_staging_model import SleepStagingModel

DATASET_PATHS = {
    "cfs": CFS_DATA_PATHS,
    "homepap": HOMEPAP_DATA_PATHS,
}


class TransferLearningTrainer:
    """Fine-tunes a pretrained sleep staging model on a target dataset."""

    def __init__(
        self,
        model_dir,
        output_filename="best_model_transfer_learning.pth",
        source_checkpoint=None,
        dataset_name="target",
        epochs=5,
        learning_rate=1e-4,
        memory_fraction=0.9,
        use_amp=False,
    ):
        self.model_dir = Path(model_dir)
        self.output_filename = output_filename
        self.source_checkpoint = source_checkpoint
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_amp = use_amp

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            setup_gpu_memory_limit(fraction=memory_fraction)

        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.use_amp and self.device.type == "cuda"
            else None
        )

        # Outputs live inside the pretrained run directory
        self.checkpoint_dir = self.model_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir = self.model_dir / "transfer_results"
        self.results_dir.mkdir(exist_ok=True)

        self.config = {
            "pretrained_model_dir": str(model_dir),
            "source_checkpoint": source_checkpoint,
            "target_dataset": dataset_name,
            "output_filename": output_filename,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "model_type": "transfer_learning",
        }

        # Reuse the pretrained run's weight decay so fine-tuning matches its regularisation
        self.pretrained_weight_decay = 1e-5
        pretrained_config_path = self.model_dir / "config.yaml"
        if not pretrained_config_path.exists():
            pretrained_config_path = self.model_dir / "config.json"

        if pretrained_config_path.exists():
            with open(pretrained_config_path, "r") as f:
                if pretrained_config_path.suffix == ".yaml":
                    pretrained_config = yaml.safe_load(f)
                else:
                    pretrained_config = json.load(f)
            self.config["pretrained_config"] = pretrained_config
            self.config["model"] = pretrained_config.get("model", {})
            self.config["source_model_type"] = pretrained_config.get(
                "model_type", "unknown"
            )
            training_cfg = pretrained_config.get("training", {})
            self.pretrained_weight_decay = training_cfg.get("weight_decay", 1e-5)

        with open(self.results_dir / "transfer_config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"Output: {self.checkpoint_dir / self.output_filename}")

    def load_pretrained_model(self):
        checkpoint_filename = self.source_checkpoint or "best_model.pth"
        model = SleepStagingModel.from_model_dir(
            model_dir=str(self.model_dir),
            variant_to_checkpoint_filename={"MESA": checkpoint_filename},
        )
        model.load("MESA")
        return model

    def calculate_class_weights(self, train_loader):
        """Estimate inverse-frequency class weights from the first 30 batches."""
        all_labels = []
        max_batches = 30

        for i, (ppg, labels) in enumerate(train_loader):
            mask = labels != -1
            if mask.any():
                all_labels.extend(labels[mask].numpy().flatten())
            if i + 1 >= max_batches:
                break

        label_counts = Counter(all_labels)
        total = sum(label_counts.values())

        weights = torch.zeros(4)
        for label in range(4):
            if label_counts[label] > 0:
                weights[label] = total / (4 * label_counts[label])

        return weights.to(self.device)

    def train_epoch(
        self,
        sleep_staging_model,
        train_loader,
        criterion,
        optimizer,
        epoch,
        scheduler=None,
    ):
        """Train for one epoch with NaN detection and recovery."""
        sleep_staging_model.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        nan_count = 0
        max_nan_batches = 5

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (ppg, labels) in enumerate(progress_bar):
            ppg = ppg.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda", enabled=self.use_amp and self.device.type == "cuda"
            ):
                # (batch, classes, epochs) -> (batch, epochs, classes) so the
                # epoch and batch dims can be flattened together for the loss
                outputs = sleep_staging_model(ppg).permute(0, 2, 1)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1)
                ).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(
                    f"\n  NaN/Inf loss at epoch {epoch}, batch {batch_idx} "
                    f"({nan_count}/{max_nan_batches})"
                )
                optimizer.zero_grad(set_to_none=True)
                if nan_count >= max_nan_batches:
                    print(f"  Aborting epoch {epoch}: too many NaN batches")
                    break
                continue

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    sleep_staging_model.model.parameters(), max_norm=0.5
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    sleep_staging_model.model.parameters(), max_norm=0.5
                )
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            mask = labels != -1
            if mask.any():
                masked_outputs = outputs[mask]
                masked_labels = labels[mask]
                _, predicted = masked_outputs.max(1)
                total += masked_labels.numel()
                correct += predicted.eq(masked_labels).sum().item()
                running_loss += loss.item() * masked_labels.numel()

            if total > 0:
                progress_bar.set_postfix(
                    {"loss": running_loss / total, "acc": 100.0 * correct / total}
                )

            if batch_idx % 5 == 0:
                gc.collect()
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.empty_cache()

        epoch_loss = running_loss / total if total > 0 else float("nan")
        epoch_acc = 100.0 * correct / total if total > 0 else 0

        if nan_count > 0:
            print(f"  Epoch {epoch}: {nan_count} NaN batches skipped")

        return epoch_loss, epoch_acc

    def evaluate(self, model, dataloader, criterion):
        """Evaluate model and return overall + per-subject metrics."""
        model.eval()

        running_loss = 0
        all_preds = []
        all_labels_list = []
        subject_predictions = defaultdict(list)
        subject_labels = defaultdict(list)

        dataset = dataloader.dataset
        has_subjects = hasattr(dataset, "subjects")

        with torch.no_grad():
            for batch_idx, (ppg, labels) in enumerate(
                tqdm(dataloader, desc="Evaluating")
            ):
                ppg = ppg.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast(
                    "cuda", enabled=self.use_amp and self.device.type == "cuda"
                ):
                    outputs = model(ppg).permute(0, 2, 1)
                    loss = criterion(
                        outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1)
                    ).mean()

                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    sample_idx = batch_idx * dataloader.batch_size + i

                    if has_subjects and sample_idx < len(dataset.subjects):
                        subject_id = dataset.subjects[sample_idx]
                    else:
                        subject_id = f"subject_{sample_idx}"

                    mask = labels[i] != -1
                    if mask.any():
                        patient_outputs = outputs[i][mask]
                        patient_labels_i = labels[i][mask]
                        _, predicted = patient_outputs.max(1)

                        subject_predictions[subject_id].extend(predicted.cpu().numpy())
                        subject_labels[subject_id].extend(
                            patient_labels_i.cpu().numpy()
                        )
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels_list.extend(patient_labels_i.cpu().numpy())
                        running_loss += loss.item() * patient_labels_i.numel()

        # Per-subject metrics
        subject_kappas = {}
        subject_accuracies = {}
        subject_f1s = {}
        for subject_id in subject_predictions:
            preds = np.array(subject_predictions[subject_id])
            lbls = np.array(subject_labels[subject_id])
            subject_accuracies[subject_id] = float(np.mean(preds == lbls))
            if len(np.unique(lbls)) > 1:
                subject_kappas[subject_id] = float(cohen_kappa_score(lbls, preds))
            else:
                subject_kappas[subject_id] = 0.0
            subject_f1s[subject_id] = float(
                f1_score(lbls, preds, average="weighted", zero_division=0)
            )

        # Overall metrics
        epoch_loss = running_loss / len(all_labels_list) if all_labels_list else 0
        overall_accuracy = (
            float(np.mean(np.array(all_preds) == np.array(all_labels_list)))
            if all_labels_list
            else 0
        )
        overall_kappa = (
            float(cohen_kappa_score(all_labels_list, all_preds))
            if all_labels_list
            else 0
        )
        overall_f1 = (
            float(
                f1_score(
                    all_labels_list, all_preds, average="weighted", zero_division=0
                )
            )
            if all_labels_list
            else 0
        )
        median_kappa = (
            float(np.median(list(subject_kappas.values()))) if subject_kappas else 0
        )
        median_accuracy = (
            float(np.median(list(subject_accuracies.values())))
            if subject_accuracies
            else 0
        )
        median_f1 = float(np.median(list(subject_f1s.values()))) if subject_f1s else 0
        cm = confusion_matrix(all_labels_list, all_preds, labels=range(4))

        return {
            "loss": epoch_loss,
            "overall_accuracy": overall_accuracy,
            "overall_kappa": overall_kappa,
            "overall_f1": overall_f1,
            "median_accuracy": median_accuracy,
            "median_kappa": median_kappa,
            "median_f1": median_f1,
            "subject_kappas": subject_kappas,
            "subject_accuracies": subject_accuracies,
            "subject_f1s": subject_f1s,
            "all_preds": all_preds,
            "all_labels": all_labels_list,
            "confusion_matrix": cm,
        }

    def plot_confusion_matrix(self, cm, phase="test"):
        plt.figure(figsize=(10, 8))

        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

        sns.heatmap(
            cm_percent,
            annot=annotations,
            fmt="",
            cmap="Blues",
            xticklabels=["Wake", "Light", "Deep", "REM"],
            yticklabels=["Wake", "Light", "Deep", "REM"],
        )
        plt.title(f"Confusion Matrix ({phase})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        save_path = self.results_dir / f"confusion_matrix_{phase}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def run(self, train_loader, test_loader):
        """Run transfer learning: train on full training set, evaluate on test set."""
        print(
            f"\nTraining: "
            f"{len(train_loader)} train batches, {len(test_loader)} test batches"
        )

        model = self.load_pretrained_model()
        model.to(self.device)
        inner = model.model

        # Weighted loss for training (compensates for class imbalance), unweighted for evaluation
        class_weights = self.calculate_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=-1, reduction="none"
        )
        val_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

        optimizer = torch.optim.AdamW(
            inner.parameters(),
            lr=self.learning_rate,
            weight_decay=self.pretrained_weight_decay,
        )

        # Linear warmup (~1 epoch or 20%) then cosine decay
        total_steps = len(train_loader) * self.epochs
        warmup_steps = min(len(train_loader), total_steps // 5)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch, scheduler=scheduler
            )
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            if np.isnan(train_loss):
                print(f"NaN loss at epoch {epoch}, stopping training early")
                break

        test_metrics = self.evaluate(model, test_loader, val_criterion)
        print(
            f"Test: "
            f"Kappa={test_metrics['overall_kappa']:.4f}, "
            f"Acc={test_metrics['overall_accuracy']:.4f}, "
            f"F1={test_metrics['overall_f1']:.4f}"
        )

        self.plot_confusion_matrix(test_metrics["confusion_matrix"], "test")

        checkpoint = {
            "model_state_dict": model.model.state_dict(),
            "test_metrics": {
                k: v
                for k, v in test_metrics.items()
                if k not in ["all_preds", "all_labels", "confusion_matrix"]
            },
            "config": self.config,
        }
        torch.save(checkpoint, self.checkpoint_dir / self.output_filename)

        result = {
            "n_train": len(train_loader.dataset),
            "n_test": len(test_loader.dataset),
            "overall_accuracy": test_metrics["overall_accuracy"],
            "overall_kappa": test_metrics["overall_kappa"],
            "overall_f1": test_metrics["overall_f1"],
            "median_accuracy": test_metrics["median_accuracy"],
            "median_kappa": test_metrics["median_kappa"],
            "median_f1": test_metrics["median_f1"],
            "subject_kappas": test_metrics["subject_kappas"],
        }

        del optimizer, criterion, val_criterion
        model.unload()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def save_results(self, result):
        subject_kappas = result["subject_kappas"]
        kappa_values = list(subject_kappas.values())
        median_kappa = float(np.median(kappa_values))
        mean_kappa = float(np.mean(kappa_values))

        print(
            f"\n{len(kappa_values)} subjects: "
            f"median kappa={median_kappa:.4f}, "
            f"mean kappa={mean_kappa:.4f} +/- {np.std(kappa_values):.4f}"
        )

        final_results = {
            "pretrained_model": str(self.model_dir),
            "target_dataset": self.dataset_name,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "total_subjects": len(subject_kappas),
            "overall_kappa": result["overall_kappa"],
            "overall_accuracy": result["overall_accuracy"],
            "overall_f1": result["overall_f1"],
            "median_kappa": median_kappa,
            "mean_kappa": mean_kappa,
            "std_kappa": float(np.std(kappa_values)),
            "median_accuracy": result["median_accuracy"],
            "median_f1": result["median_f1"],
            "all_subject_kappas": subject_kappas,
        }

        with open(self.results_dir / "transfer_learning_results.json", "w") as f:
            json.dump(final_results, f, indent=2)

        with open(self.results_dir / "summary.txt", "w") as f:
            f.write(f"Transfer Learning Results: {self.dataset_name}\n\n")
            f.write(f"Pretrained model: {self.model_dir.name}\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Learning rate: {self.learning_rate}\n")
            f.write(f"Train subjects: {result['n_train']}\n")
            f.write(f"Test subjects: {result['n_test']}\n\n")
            f.write("Per-subject Kappa:\n")
            f.write(f"  Median: {median_kappa:.4f}\n")
            f.write(f"  Mean: {mean_kappa:.4f} +/- {np.std(kappa_values):.4f}\n")
            f.write(
                f"  Range: [{np.min(kappa_values):.4f}, {np.max(kappa_values):.4f}]\n\n"
            )
            f.write(f"Overall Kappa: {result['overall_kappa']:.4f}\n")
            f.write(f"Overall Accuracy: {result['overall_accuracy']:.4f}\n")
            f.write(f"Overall F1: {result['overall_f1']:.4f}\n")

        self._plot_kappa_distribution(kappa_values)
        print(f"Results saved to: {self.results_dir}")

        return final_results

    def _plot_kappa_distribution(self, kappa_values):
        plt.figure(figsize=(10, 6))
        plt.hist(kappa_values, bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(
            np.median(kappa_values),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(kappa_values):.3f}",
        )
        plt.axvline(
            np.mean(kappa_values),
            color="green",
            linestyle="--",
            label=f"Mean: {np.mean(kappa_values):.3f}",
        )
        plt.xlabel("Cohen's Kappa")
        plt.ylabel("Number of Subjects")
        plt.title(f"Per-Subject Kappa ({self.dataset_name})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(self.results_dir / "kappa_distribution.png", dpi=150)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a trained run on CFS or HomePAP",
    )

    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PATHS))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Defaults to best_model_<dataset>.pth, the name the benchmarker looks for",
    )
    parser.add_argument(
        "--source-checkpoint",
        default=None,
        help="Load this checkpoint instead of best_model.pth (for chaining).",
    )

    return parser.parse_args()


def validate_args(args):
    data_paths = DATASET_PATHS[args.dataset]
    for key in ("ppg", "index"):
        if not Path(data_paths[key]).exists():
            print(f"Error: {key} file not found: {data_paths[key]}")
            sys.exit(1)
    args.data_paths = data_paths
    if args.output_filename is None:
        args.output_filename = f"best_model_{args.dataset}.pth"

    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    if (
        not (model_path / "config.yaml").exists()
        and not (model_path / "config.json").exists()
    ):
        print(f"Error: No config file found in: {args.model_dir}")
        sys.exit(1)

    return args


def main():
    args = validate_args(parse_args())

    print(
        f"Transfer learning: {args.dataset} on {Path(args.model_dir).name}, "
        f"{args.epochs} epochs, lr={args.learning_rate}"
    )

    trainer = TransferLearningTrainer(
        model_dir=args.model_dir,
        output_filename=args.output_filename,
        source_checkpoint=args.source_checkpoint,
        dataset_name=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        memory_fraction=args.memory_fraction,
        use_amp=args.amp,
    )

    train_loader, test_loader, _, _ = get_transfer_dataloaders(
        dataset_name=args.dataset,
        data_path=args.data_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    result = trainer.run(train_loader=train_loader, test_loader=test_loader)
    trainer.save_results(result)

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
