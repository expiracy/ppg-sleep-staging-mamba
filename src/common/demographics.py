"""Subject demographics for the three NSRR cohorts.

Each cohort ships a dataset CSV with its own column names and coding for age, sex,
race, AHI and BMI. This module maps them onto one set of column names and keeps only
the subjects that survived extraction.
"""

import h5py
import pandas as pd

from common.paths import (
    CFS_DEMO_CSV,
    CFS_INDEX_DATA_PATH,
    CFS_PPG_DATA_PATH,
    HOMEPAP_DEMO_CSV,
    HOMEPAP_INDEX_DATA_PATH,
    HOMEPAP_PPG_DATA_PATH,
    INDEX_DATA_PATH,
    MESA_DEMO_CSV,
    PPG_DATA_PATH,
)
from datasets.dataset_type import DatasetType

AGE_BINS = [0, 50, 60, 70, 80, 120]
AGE_LABELS = ["<50", "50-59", "60-69", "70-79", "80+"]

BMI_BINS = [0, 18.5, 25, 30, 35, 100]
BMI_LABELS = ["Under", "Normal", "Over", "Obese I", "Obese II+"]

# AHI cut-offs for apnoea severity, events per hour, as used clinically
AHI_BINS = [-1, 5, 15, 30, 200]
AHI_LABELS = ["Normal", "Mild", "Moderate", "Severe"]

SEX_LABELS = {0: "Female", 1: "Male"}

# Each cohort codes race differently, so the numeric codes are mapped onto a shared
# set of labels. Keys are DatasetType.value strings.
RACE_MAPS = {
    "MESA": {1: "White", 2: "Asian", 3: "Black", 4: "Hispanic"},
    "CFS": {1: "White", 2: "Black", 3: "Other"},
    "HomePAP": {
        1: "White",
        2: "Other",
        3: "Black",
        4: "Asian",
        5: "Other",
        6: "Other",
        7: "Other",
    },
}
COMMON_RACES = ["White", "Black", "Asian", "Hispanic", "Other"]

DATASET_CONFIGS = {
    DatasetType.MESA: {
        "csv_path": MESA_DEMO_CSV,
        "index_h5_path": INDEX_DATA_PATH,
        "ppg_h5_path": PPG_DATA_PATH,
        "ppg_rate": "256",
        "subject_id_col": "mesaid",
        "age_col": "sleepage5c",
        "sex_col": "gender1",
        "race_col": "race1c",
        "ahi_col": "ahi_a0h3",
        "bmi_col": "bmi5c",
        # MESA IDs are zero padded to four digits in the recording filenames
        "id_formatter": lambda x: str(int(x)).zfill(4),
    },
    DatasetType.CFS: {
        "csv_path": CFS_DEMO_CSV,
        "index_h5_path": CFS_INDEX_DATA_PATH,
        "ppg_h5_path": CFS_PPG_DATA_PATH,
        "ppg_rate": "128",
        "subject_id_col": "nsrrid",
        "age_col": "age",
        "sex_col": "sex",
        "race_col": "race",
        "ahi_col": "ahi",
        "bmi_col": "bmi",
        "id_formatter": lambda x: str(int(x)),
    },
    DatasetType.HOMEPAP: {
        "csv_path": HOMEPAP_DEMO_CSV,
        "index_h5_path": HOMEPAP_INDEX_DATA_PATH,
        "ppg_h5_path": HOMEPAP_PPG_DATA_PATH,
        # HomePAP was recorded across sites at several sampling rates
        "ppg_rate": "64-256",
        "subject_id_col": "nsrrid",
        "age_col": "age",
        "sex_col": "gender",
        "race_col": "race7",
        "ahi_col": "ahi",
        "bmi_col": "bmi",
        "id_formatter": lambda x: str(int(x)),
    },
}


def get_processed_subject_ids(index_h5_path):
    """The subject IDs that made it through extraction, read from the H5 index."""
    with h5py.File(index_h5_path, "r") as f:
        return list(f["subjects"].keys())


def load_demographics(dataset_type, include_bmi=False):
    """Demographics for every extracted subject in a dataset.

    Returns a frame with the columns subject_id, age, sex, race, ahi, and bmi when
    `include_bmi` is set. A column missing from the cohort's CSV is left out.
    """
    cfg = DATASET_CONFIGS[dataset_type]
    valid_ids = set(get_processed_subject_ids(cfg["index_h5_path"]))

    # The NSRR CSVs are wide and mix types within a column, so no type inference
    df = pd.read_csv(cfg["csv_path"], low_memory=False)
    df["subject_id"] = df[cfg["subject_id_col"]].apply(cfg["id_formatter"])
    df = df[df["subject_id"].isin(valid_ids)]

    col_map = {
        cfg["age_col"]: "age",
        cfg["sex_col"]: "sex",
        cfg["race_col"]: "race",
        cfg["ahi_col"]: "ahi",
    }
    if include_bmi:
        col_map[cfg["bmi_col"]] = "bmi"

    existing = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    canonical = ["age", "sex", "race", "ahi"]
    if include_bmi:
        canonical.append("bmi")
    keep = [c for c in ["subject_id"] + canonical if c in df.columns]
    return df[keep].reset_index(drop=True)
