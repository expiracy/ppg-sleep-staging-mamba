"""Training loop shared by every model trainer.

Trains on full-night MESA recordings with early stopping on validation kappa,
then evaluates the best checkpoint on the fixed test set. Each run writes a
timestamped directory under the config's output.save_dir:

    config.yaml                  resolved config plus model_type
    checkpoints/best_model.pth   best validation kappa
    checkpoints/last_model.pth   most recent epoch, used by --resume
    logs/                        TensorBoard scalars
    results/                     test metrics, classification report, plots
"""

import argparse
import gc
import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.gpu_utils import setup_gpu_memory_limit
from common.paths import make_run_name, resolve_config_paths
from datasets.full_night_dataset import get_dataloaders

STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]


class SleepStagingTrainer:
    """Trains one model class described by a config.

    model_cls must provide from_config(config) and get_name(). model_type is
    the ModelType member written into the run's config so the run can be
    reloaded later by SleepStagingModel.
    """

    def __init__(self, config, model_cls, model_type, run_id=None, resume_from=None):
        self.config = config
        self.model_cls = model_cls
        self.model_type = model_type
        self.run_id = run_id
        self.resume_from = resume_from

        # The Mamba kernels are CUDA-only, so there is no CPU fallback
        if not torch.cuda.is_available():
            raise RuntimeError("Training requires a CUDA device")
        setup_gpu_memory_limit(config.get("gpu", {}).get("memory_fraction", 0.8))
        self.device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(self.device)}")

        self.setup_directories()

        self.writer = SummaryWriter(self.log_dir)

        self.use_amp = config.get("use_amp", True)
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using mixed precision training")

        config["model_type"] = model_type.value
        if not resume_from:
            with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def setup_directories(self):
        if self.resume_from:
            self.output_dir = self.resume_from
            print(f"Resuming from: {self.output_dir}")
        else:
            self.output_dir = os.path.join(
                self.config["output"]["save_dir"],
                make_run_name(self.model_type.value, self.run_id),
            )

        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.results_dir = os.path.join(self.output_dir, "results")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def create_model(self):
        model = self.model_cls.from_config(self.config)
        print(f"Model name: {model.get_name()}")
        return model.to(self.device)

    def calculate_class_weights(self, train_dataset):
        """Inverse-frequency class weights estimated from the first 50 subjects."""
        print("Calculating class weights...")

        all_labels = []
        sample_size = min(len(train_dataset), 50)

        for idx in tqdm(range(sample_size), desc="Sampling labels"):
            _, labels, *_ = train_dataset[idx]

            # -1 marks unscored epochs and the padding on short recordings
            mask = labels != -1
            if mask.any():
                all_labels.extend(labels[mask].numpy())

        label_counts = Counter(all_labels)
        total_samples = sum(label_counts.values())

        class_weights = torch.zeros(4)
        for label in range(4):
            if label_counts[label] > 0:
                class_weights[label] = total_samples / (4 * label_counts[label])

        class_weights = class_weights.to(self.device)

        print(f"Class distribution: {dict(label_counts)}")
        print(f"Class weights: {class_weights.cpu().numpy()}")

        return class_weights

    def _step_loss(self, model, ppg, labels, criterion):
        """Forward pass and loss. Returns (outputs as (B, T, C), loss)."""
        with autocast("cuda", enabled=self.use_amp):
            outputs = model(ppg).permute(0, 2, 1)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
        return outputs, loss

    def train_epoch(self, model, dataloader, optimizer, criterion, scheduler, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (ppg, labels, *_) in enumerate(progress_bar):
            ppg = ppg.to(self.device)
            labels = labels.to(self.device)

            outputs, loss = self._step_loss(model, ppg, labels, criterion)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            # Leave unscored and padding epochs out of the running accuracy
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
                    {
                        "loss": running_loss / total,
                        "acc": 100.0 * correct / total,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            if batch_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = 100.0 * correct / total if total > 0 else 0

        return epoch_loss, epoch_acc

    def evaluate(self, model, dataloader, criterion):
        """Overall (pooled) and per-subject (median) accuracy, kappa and F1."""
        model.eval()

        running_loss = 0
        all_preds = []
        all_labels = []
        subject_predictions = defaultdict(list)
        subject_labels = defaultdict(list)

        with torch.no_grad():
            for ppg, labels, subject_ids in tqdm(dataloader, desc="Validation"):
                ppg = ppg.to(self.device)
                labels = labels.to(self.device)

                outputs, loss = self._step_loss(model, ppg, labels, criterion)

                for i in range(outputs.shape[0]):
                    mask = labels[i] != -1
                    if not mask.any():
                        continue

                    _, predicted = outputs[i][mask].max(1)
                    predicted = predicted.cpu().numpy()
                    true = labels[i][mask].cpu().numpy()

                    subject_predictions[subject_ids[i]].extend(predicted)
                    subject_labels[subject_ids[i]].extend(true)
                    all_preds.extend(predicted)
                    all_labels.extend(true)
                    running_loss += loss.item() * len(true)

                gc.collect()
                torch.cuda.empty_cache()

        subject_accuracies = []
        subject_kappas = []
        subject_f1s = []

        for subject_id, preds in subject_predictions.items():
            true = subject_labels[subject_id]
            subject_accuracies.append(np.mean(np.array(preds) == np.array(true)))

            # Kappa is undefined when a subject has a single label class
            if len(np.unique(true)) > 1:
                subject_kappas.append(cohen_kappa_score(true, preds))

            subject_f1s.append(
                f1_score(true, preds, average="weighted", zero_division=0)
            )

        epoch_loss = running_loss / len(all_labels) if all_labels else 0
        overall_accuracy = (
            np.mean(np.array(all_preds) == np.array(all_labels)) if all_labels else 0
        )
        overall_kappa = cohen_kappa_score(all_labels, all_preds) if all_labels else 0
        overall_f1 = (
            f1_score(all_labels, all_preds, average="weighted", zero_division=0)
            if all_labels
            else 0
        )

        median_accuracy = np.median(subject_accuracies) if subject_accuracies else 0
        median_kappa = np.median(subject_kappas) if subject_kappas else 0
        median_f1 = np.median(subject_f1s) if subject_f1s else 0

        if subject_kappas:
            print(
                f"Kappa (n={len(subject_kappas)}): "
                f"min={np.min(subject_kappas):.4f} "
                f"25%={np.percentile(subject_kappas, 25):.4f} "
                f"med={median_kappa:.4f} "
                f"75%={np.percentile(subject_kappas, 75):.4f} "
                f"max={np.max(subject_kappas):.4f}"
            )

        cm = confusion_matrix(all_labels, all_preds, labels=range(len(STAGE_NAMES)))

        return {
            "loss": epoch_loss,
            "overall_accuracy": overall_accuracy,
            "overall_kappa": overall_kappa,
            "overall_f1": overall_f1,
            "median_accuracy": median_accuracy,
            "median_kappa": median_kappa,
            "median_f1": median_f1,
            "all_preds": all_preds,
            "all_labels": all_labels,
            "per_class_metrics": self.calculate_per_class_metrics(cm),
            "subject_kappas": subject_kappas,
            "subject_accuracies": subject_accuracies,
            "subject_f1s": subject_f1s,
            "confusion_matrix": cm,
        }

    def calculate_per_class_metrics(self, cm):
        n_classes = cm.shape[0]
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)

        for i in range(n_classes):
            if cm[:, i].sum() > 0:
                precision[i] = cm[i, i] / cm[:, i].sum()  # column = predicted as i

            if cm[i, :].sum() > 0:
                recall[i] = cm[i, i] / cm[i, :].sum()  # row = truly class i

            if precision[i] + recall[i] > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

        return {"precision": precision, "recall": recall, "f1": f1}

    def save_checkpoint(
        self, model, optimizer, epoch, val_metrics, is_best, scheduler, best_kappa
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_overall_kappa": best_kappa,
            "best_median_kappa": val_metrics["median_kappa"],
            "val_acc": val_metrics["overall_accuracy"],
            "val_f1": val_metrics["overall_f1"],
            "config": self.config,
        }
        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, "last_model.pth"))

        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, "best_model.pth"))
            print(f"Saved best model (kappa: {val_metrics['overall_kappa']:.4f})")

    def plot_confusion_matrix(self, cm, epoch, phase="val"):
        plt.figure(figsize=(10, 8))

        # Row-normalise so each cell shows where a true class ends up
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
            xticklabels=STAGE_NAMES,
            yticklabels=STAGE_NAMES,
        )
        plt.title(f"Confusion Matrix - Epoch {epoch} ({phase})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        save_path = os.path.join(
            self.results_dir, f"confusion_matrix_{phase}_epoch_{epoch}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_training_curves(
        self, train_losses, val_losses, val_overall_kappas, val_median_kappas
    ):
        epochs = range(1, len(train_losses) + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(epochs, train_losses, "b-", label="Train Loss")
        ax1.plot(epochs, val_losses, "r-", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, val_overall_kappas, "g-", label="Val Overall Kappa")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Kappa")
        ax2.set_title("Validation Overall Kappa")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(epochs, val_median_kappas, "m-", label="Val Median Kappa")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Kappa")
        ax3.set_title("Validation Median Kappa")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.plot(epochs, val_overall_kappas, "g-", label="Overall Kappa")
        ax4.plot(epochs, val_median_kappas, "m-", label="Median Kappa")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Kappa")
        ax4.set_title("Overall vs Median Kappa")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_curves.png"), dpi=300)
        plt.close()

    def _print_metrics(self, label, metrics):
        print(
            f"{label} Overall - Acc: {metrics['overall_accuracy']:.4f}, "
            f"Kappa: {metrics['overall_kappa']:.4f}, F1: {metrics['overall_f1']:.4f}"
        )
        print(
            f"{label} Median  - Acc: {metrics['median_accuracy']:.4f}, "
            f"Kappa: {metrics['median_kappa']:.4f}, F1: {metrics['median_f1']:.4f}"
        )
        per_class = metrics["per_class_metrics"]
        for i, name in enumerate(STAGE_NAMES):
            print(
                f"  {name}: P={per_class['precision'][i]:.3f}, "
                f"R={per_class['recall'][i]:.3f}, F1={per_class['f1'][i]:.3f}"
            )

    def _log_metrics(self, epoch, train_loss, train_acc, val_metrics, lr):
        self.writer.add_scalar("Train/Loss", train_loss, epoch)
        self.writer.add_scalar("Train/Acc", train_acc, epoch)
        self.writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
        scalar_tags = {
            "overall_accuracy": "Val/Overall_Accuracy",
            "overall_kappa": "Val/Overall_Kappa",
            "overall_f1": "Val/Overall_F1",
            "median_accuracy": "Val/Median_Accuracy",
            "median_kappa": "Val/Median_Kappa",
            "median_f1": "Val/Median_F1",
        }
        for key, tag in scalar_tags.items():
            self.writer.add_scalar(tag, val_metrics[key], epoch)
        self.writer.add_scalar("LR", lr, epoch)

        per_class = val_metrics["per_class_metrics"]
        for i, name in enumerate(STAGE_NAMES):
            self.writer.add_scalar(
                f"Val/Precision_{name}", per_class["precision"][i], epoch
            )
            self.writer.add_scalar(f"Val/Recall_{name}", per_class["recall"][i], epoch)
            self.writer.add_scalar(f"Val/F1_{name}", per_class["f1"][i], epoch)

    def _load_resume_checkpoint(self, model, optimizer, scheduler):
        """Restore last_model.pth if resuming. Returns (start_epoch, best_kappa)."""
        checkpoint_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}, starting fresh")
            return 1, 0

        print(f"Loading checkpoint from {checkpoint_path}")
        # weights_only=False because the checkpoint carries optimizer state and config
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_kappa = checkpoint.get("best_overall_kappa", float("-inf"))
        print(f"Resuming from epoch {start_epoch}, best_kappa: {best_kappa:.4f}")
        return start_epoch, best_kappa

    def train(self):
        print(f"Model type: {self.config['model_type']}")
        print("Config:")
        print(json.dumps(self.config, indent=4))
        print(f"Output directory: {self.output_dir}")
        if self.run_id is not None:
            print(f"Run {self.run_id}/{self.config['training'].get('num_runs', 1)}")

        data_paths = {
            "ppg": self.config["data"]["ppg_file"],
            "index": self.config["data"]["index_file"],
        }
        train_loader, val_loader, test_loader, train_dataset, _, _ = get_dataloaders(
            data_paths,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            use_sleepppg_test_set=self.config["training"].get(
                "use_sleepppg_test_set", True
            ),
        )

        model = self.create_model()
        print(f"Model created on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        class_weights = self.calculate_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        num_epochs = self.config["training"].get(
            "num_epochs", self.config["training"].get("epochs", 50)
        )
        # One-cycle schedule: the rate rises on a cosine over the first 10% of steps,
        # then falls on a cosine for the rest
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["training"]["learning_rate"],
            total_steps=len(train_loader) * num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        # Kappa can start below 0, and epoch 1 must still write best_model.pth
        start_epoch, best_kappa = 1, float("-inf")
        if self.resume_from:
            start_epoch, best_kappa = self._load_resume_checkpoint(
                model, optimizer, scheduler
            )

        patience_counter = 0
        train_losses = []
        val_losses = []
        val_overall_kappas = []
        val_median_kappas = []

        for epoch in range(start_epoch, num_epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{num_epochs}, LR: {lr:.6f}")

            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )
            train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            val_metrics = self.evaluate(model, val_loader, criterion)
            val_losses.append(val_metrics["loss"])
            val_overall_kappas.append(val_metrics["overall_kappa"])
            val_median_kappas.append(val_metrics["median_kappa"])

            print(f"Val Loss: {val_metrics['loss']:.4f}")
            self._print_metrics("Val", val_metrics)
            self._log_metrics(epoch, train_loss, train_acc, val_metrics, lr)

            is_best = val_metrics["overall_kappa"] > best_kappa
            if is_best:
                best_kappa = val_metrics["overall_kappa"]
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(
                model, optimizer, epoch, val_metrics, is_best, scheduler, best_kappa
            )

            if epoch % 10 == 0:
                self.plot_confusion_matrix(
                    val_metrics["confusion_matrix"], epoch, "val"
                )

            if patience_counter >= self.config["training"]["patience"]:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        print("Testing best model")
        checkpoint = torch.load(
            os.path.join(self.checkpoint_dir, "best_model.pth"), weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self.evaluate(model, test_loader, criterion)
        self._print_metrics("Test", test_metrics)
        self.plot_confusion_matrix(test_metrics["confusion_matrix"], "final", "test")

        results = {
            "model_name": model.get_name(),
            "model_config": self.config.get("model", {}),
            "test_overall_accuracy": float(test_metrics["overall_accuracy"]),
            "test_overall_kappa": float(test_metrics["overall_kappa"]),
            "test_overall_f1": float(test_metrics["overall_f1"]),
            "test_median_accuracy": float(test_metrics["median_accuracy"]),
            "test_median_kappa": float(test_metrics["median_kappa"]),
            "test_median_f1": float(test_metrics["median_f1"]),
            "best_val_kappa": float(best_kappa),
            "per_class_precision": test_metrics["per_class_metrics"][
                "precision"
            ].tolist(),
            "per_class_recall": test_metrics["per_class_metrics"]["recall"].tolist(),
            "per_class_f1": test_metrics["per_class_metrics"]["f1"].tolist(),
        }

        with open(os.path.join(self.results_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        report = classification_report(
            test_metrics["all_labels"],
            test_metrics["all_preds"],
            labels=range(len(STAGE_NAMES)),
            target_names=STAGE_NAMES,
        )
        with open(
            os.path.join(self.results_dir, "classification_report.txt"), "w"
        ) as f:
            f.write(report)
        print(f"Classification Report:\n{report}")

        self.plot_training_curves(
            train_losses, val_losses, val_overall_kappas, val_median_kappas
        )

        self.writer.close()

        return results


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (not required when resuming)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model output directory to resume from",
    )
    return parser


def load_config(args, parser):
    """Load the YAML/JSON config, preferring the one saved in a resumed run."""
    if not args.resume and not args.config:
        parser.error("--config is required when not resuming")

    config_path = args.config
    if args.resume:
        saved_config_path = os.path.join(args.resume, "config.yaml")
        if os.path.exists(saved_config_path):
            print(f"Loading saved config from checkpoint: {saved_config_path}")
            config_path = saved_config_path
        elif config_path is None:
            parser.error(f"No config.yaml in {args.resume}, pass --config as well")
        else:
            print(
                f"No saved config found at {saved_config_path}, using provided config file"
            )

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    return resolve_config_paths(config)


def run_training(args, config, model_cls, model_type):
    """Train `args.runs` seeds in sequence and print a summary when there are several."""
    all_results = []

    for run_id in range(1, args.runs + 1):
        print(f"Starting Run {run_id}/{args.runs}")

        torch.manual_seed(args.seed + run_id)
        np.random.seed(args.seed + run_id)
        torch.cuda.manual_seed(args.seed + run_id)

        trainer = SleepStagingTrainer(
            config,
            model_cls,
            model_type,
            run_id=run_id if not args.resume else None,
            resume_from=args.resume,
        )
        all_results.append(trainer.train())

        gc.collect()
        torch.cuda.empty_cache()

    if args.runs > 1:
        print(f"Summary of {args.runs} runs")
        for metric in (
            "test_overall_kappa",
            "test_overall_accuracy",
            "test_overall_f1",
            "test_median_kappa",
            "test_median_accuracy",
            "test_median_f1",
        ):
            values = [r[metric] for r in all_results]
            print(
                f"{metric}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, "
                f"Min={np.min(values):.4f}, Max={np.max(values):.4f}"
            )

    return all_results
