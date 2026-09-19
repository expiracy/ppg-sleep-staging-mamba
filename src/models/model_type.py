from enum import Enum


class ModelType(Enum):
    """Consistent identifiers for each model architecture."""

    PPG_ONLY = "ppg_only"
    PPG_UNFILTERED = "ppg_unfiltered"
    SINGLE_STREAM_MAMBA = "single_stream_mamba"
    DUAL_STREAM_MAMBA = "dual_stream_mamba"

    @property
    def colour(self):
        return {
            ModelType.PPG_ONLY: "#e67e22",
            ModelType.PPG_UNFILTERED: "#e74c3c",
            ModelType.SINGLE_STREAM_MAMBA: "#27ae60",
            ModelType.DUAL_STREAM_MAMBA: "#2980b9",
        }[self]
