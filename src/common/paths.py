import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
MODELS_DIR = OUTPUT_DIR / "models"
BENCHMARKS_DIR = OUTPUT_DIR / "benchmarks"

# Raw data base directories
DATA_DIR_MESA = DATA_DIR / "mesa"
DATA_DIR_CFS = DATA_DIR / "cfs"
DATA_DIR_HOMEPAP = DATA_DIR / "homepap"

# Demographics CSVs, downloaded from NSRR alongside the recordings
MESA_DEMO_CSV = DATA_DIR_MESA / "datasets" / "mesa-sleep-dataset-0.8.0.csv"
CFS_DEMO_CSV = DATA_DIR_CFS / "datasets" / "cfs-visit5-dataset-0.7.0.csv"
HOMEPAP_DEMO_CSV = DATA_DIR_HOMEPAP / "datasets" / "homepap-baseline-dataset-0.2.0.csv"

# MESA processed data paths
DATA_DIR_MESA_PROCESSED = DATA_DIR / "mesa_processed"
PPG_DATA_PATH = DATA_DIR_MESA_PROCESSED / "mesa_ppg_with_labels.h5"
INDEX_DATA_PATH = DATA_DIR_MESA_PROCESSED / "mesa_subject_index.h5"

# CFS processed data paths
DATA_DIR_CFS_PROCESSED = DATA_DIR / "cfs_processed"
CFS_PPG_DATA_PATH = DATA_DIR_CFS_PROCESSED / "cfs_ppg_with_labels.h5"
CFS_INDEX_DATA_PATH = DATA_DIR_CFS_PROCESSED / "cfs_subject_index.h5"

# HomePAP processed data paths
DATA_DIR_HOMEPAP_PROCESSED = DATA_DIR / "homepap_processed"
HOMEPAP_PPG_DATA_PATH = DATA_DIR_HOMEPAP_PROCESSED / "homepap_ppg_with_labels.h5"
HOMEPAP_INDEX_DATA_PATH = DATA_DIR_HOMEPAP_PROCESSED / "homepap_subject_index.h5"

# Path dicts in the form the dataset classes expect
MESA_DATA_PATHS = {
    "ppg": str(PPG_DATA_PATH),
    "index": str(INDEX_DATA_PATH),
}

CFS_DATA_PATHS = {
    "ppg": str(CFS_PPG_DATA_PATH),
    "index": str(CFS_INDEX_DATA_PATH),
}

HOMEPAP_DATA_PATHS = {
    "ppg": str(HOMEPAP_PPG_DATA_PATH),
    "index": str(HOMEPAP_INDEX_DATA_PATH),
}


def make_run_name(prefix, run_id=None):
    """Generate a timestamped run directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{timestamp}"
    if run_id is not None:
        name += f"_run{run_id}"
    return name


def make_benchmark_name(name=None):
    """Generate a timestamped benchmark directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        return f"benchmark_{name}_{timestamp}"
    return f"benchmark_{timestamp}"


def resolve_config_paths(config):
    """Expand ${ENV_VAR} placeholders in config path fields."""
    config = deepcopy(config)
    for key in ("ppg_file", "index_file"):
        if key in config.get("data", {}):
            config["data"][key] = os.path.expandvars(config["data"][key])
    if "save_dir" in config.get("output", {}):
        config["output"]["save_dir"] = os.path.expandvars(config["output"]["save_dir"])
    return config
