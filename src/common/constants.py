# Signal constants shared by the data extractors, datasets and models
TARGET_FS = 34.133333333  # Hz, 1024 / 30 rounded, so a 30 s epoch holds 1024 samples
SAMPLES_PER_WINDOW = 1024  # samples per 30 s epoch
WINDOW_DURATION = 30  # seconds per epoch
WINDOWS_PER_SUBJECT = 1200  # 10 hours, recordings are zero-padded or truncated
ADULT_AGE_THRESHOLD = 18  # years, minimum age for inclusion in CFS
STAGES = ["Wake", "Light", "Deep", "REM"]  # labels 0 to 3, -1 marks an unscored epoch

# Checkpoint file for each trained variant, named as in every results file
CHECKPOINT_VARIANTS = {
    "Base": "best_model.pth",
    "CFS-TL": "best_model_cfs.pth",
    "HP-TL": "best_model_homepap.pth",
}
