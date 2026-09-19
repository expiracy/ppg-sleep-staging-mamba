"""Loads a trained run directory and swaps checkpoint variants in and out
of a single model instance."""

import json
from pathlib import Path

import torch
import yaml

from models.dual_stream_mamba import DualStreamMamba
from models.model_type import ModelType
from models.ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
from models.single_stream_mamba import SingleStreamMamba
from models.sleep_ppg_net import SleepPPGNet

_MODEL_CLASSES = {
    ModelType.PPG_ONLY: SleepPPGNet,
    ModelType.PPG_UNFILTERED: PPGUnfilteredCrossAttention,
    ModelType.SINGLE_STREAM_MAMBA: SingleStreamMamba,
    ModelType.DUAL_STREAM_MAMBA: DualStreamMamba,
}


class SleepStagingModel:
    """Wrapper for sleep staging models with explicit load/unload.

    One instance per model directory. Holds a dict of variant names to
    checkpoint paths. load(variant) swaps weights into a single shared
    model instance.
    """

    def __init__(
        self,
        config,
        variant_to_checkpoint_file=None,
    ):
        self.config = config
        self.config_file = None
        self.model_type = ModelType(self.config["model_type"])
        self.model_id = config.get("model_id", -1)
        self.device = "cuda"
        self._model = self._build_model()

        if variant_to_checkpoint_file is not None:
            self._variant_checkpoints = {
                k: Path(v) for k, v in variant_to_checkpoint_file.items()
            }
        else:
            self._variant_checkpoints = {}
        self._loaded_variant = None

    @property
    def variant_names(self):
        return list(self._variant_checkpoints.keys())

    @property
    def variant(self):
        return self._loaded_variant

    @property
    def checkpoint_file(self):
        if self._loaded_variant is None:
            return None
        return self._variant_checkpoints.get(self._loaded_variant)

    def get_name(self, include_variant=False, include_model_id=False):
        name = self._model.get_name()
        if include_model_id and self.model_id is not None and self.model_id != -1:
            name = f"({self.model_id}) {name}"
        if include_variant and self._loaded_variant:
            name = f"{name} ({self._loaded_variant})"
        return name

    def get_short_name(self, include_variant=True, include_model_id=False):
        short = self._model.get_short_name()
        if include_model_id and self.model_id is not None and self.model_id != -1:
            short = f"({self.model_id}) {short}"
        if include_variant and self._loaded_variant:
            v = "base" if self._loaded_variant == "Base" else self._loaded_variant
            short = f"{short}({v})"
        return short

    def load(self, variant):
        """Load variant weights into the model.

        Auto-unloads any currently loaded variant first.
        """
        if variant not in self._variant_checkpoints:
            raise KeyError(
                f"Unknown variant '{variant}'. Available: {self.variant_names}"
            )
        if self._loaded_variant is not None:
            self.unload()

        checkpoint_path = self._variant_checkpoints[variant]
        print(f"Loading model: {self._model.get_name()} ({variant})")
        # weights_only=False because checkpoints include optimizer state and metadata
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self.device)
        self._loaded_variant = variant
        return self

    def unload(self):
        """Reset model to uninitialized weights and free GPU memory."""
        if self._loaded_variant is not None:
            # Delete and clear cache before rebuilding to avoid holding two copies
            del self._model
            torch.cuda.empty_cache()
            self._model = self._build_model()
            self._loaded_variant = None
        return self

    def is_loaded(self):
        return self._loaded_variant is not None

    @classmethod
    def from_model_dir(
        cls,
        model_dir,
        variant_to_checkpoint_filename=None,
    ):
        """Load from a model output directory (config.yaml + checkpoints/).

        Reads model_id from config (defaults to -1 if missing).
        Defaults variant_to_checkpoint_filename to {"MESA": "best_model.pth"}.
        Only includes checkpoints that exist on disk.
        """
        model_dir = Path(model_dir)

        config_file = model_dir / "config.yaml"
        if not config_file.exists():
            config_file = model_dir / "config.json"
            if not config_file.exists():
                raise FileNotFoundError(f"No config file found in {model_dir}")

        with open(config_file, "r") as f:
            if config_file.suffix == ".yaml":
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        config.setdefault("model_id", -1)

        if variant_to_checkpoint_filename is None:
            variant_to_checkpoint_filename = {"MESA": "best_model.pth"}

        checkpoint_dir = model_dir / "checkpoints"
        resolved = {}
        for variant_name, filename in variant_to_checkpoint_filename.items():
            full_path = checkpoint_dir / filename
            if full_path.exists():
                resolved[variant_name] = str(full_path)

        instance = cls(config=config, variant_to_checkpoint_file=resolved)
        instance.config_file = config_file.resolve()

        if resolved:
            print(instance.get_name(include_model_id=True))

        return instance

    def to(self, device):
        self.device = device
        self._model = self._model.to(device)
        return self

    def eval(self):
        if self._loaded_variant is None:
            raise RuntimeError("Model not loaded. Call load(variant) first.")
        self._model.eval()
        return self

    @property
    def model(self):
        if self._loaded_variant is None:
            raise RuntimeError("Model not loaded. Call load(variant) first.")
        return self._model

    def __call__(self, *args, **kwargs):
        if self._loaded_variant is None:
            raise RuntimeError("Model not loaded. Call load(variant) first.")
        return self._model(*args, **kwargs)

    def get_static_size_mb(self):
        total_size = sum(p.numel() * p.element_size() for p in self._model.parameters())
        total_size += sum(b.numel() * b.element_size() for b in self._model.buffers())
        return total_size / 1024 / 1024

    def _build_model(self):
        model_cls = _MODEL_CLASSES.get(self.model_type)
        if model_cls is None:
            raise NotImplementedError(f"Model type {self.model_type} not supported.")
        return model_cls.from_config(self.config)

    @classmethod
    def from_model_dirs(
        cls,
        model_dirs,
        variant_to_checkpoint_filename=None,
    ):
        """Load from multiple model directories.

        Returns one SleepStagingModel per directory. Each has all existing
        checkpoints from the map as variants.
        """
        if variant_to_checkpoint_filename is None:
            variant_to_checkpoint_filename = {"MESA": "best_model.pth"}

        models = []
        for dir_path in model_dirs:
            print()
            print(dir_path)
            model = cls.from_model_dir(
                model_dir=dir_path,
                variant_to_checkpoint_filename=variant_to_checkpoint_filename,
            )
            models.append(model)
        return models
