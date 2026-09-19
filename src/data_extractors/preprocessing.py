"""
Shared preprocessing for PPG signals and NSRR sleep-stage annotations.

Some functions were sourced from https://github.com/DavyWJW/sleep-staging-models/
"""

import os
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
import pyedflib
from scipy import signal as sp_signal
from tqdm import tqdm

from common.constants import (
    TARGET_FS,
    WINDOW_DURATION,
)


def process_ppg(raw_ppg, original_fs, target_fs=TARGET_FS):
    """Cheby-II lowpass at 8 Hz (order 8, 40 dB stopband), resample to
    target_fs, clip at +/-3 sigma, z-score."""
    # Lowpass filter to remove high-frequency noise above the PPG band
    nyq = 0.5 * original_fs
    cutoff = 8 / nyq
    if cutoff >= 1:  # guard for low sample rates where 8 Hz exceeds Nyquist
        cutoff = 0.99
    sos = sp_signal.cheby2(N=8, rs=40, Wn=cutoff, btype="lowpass", output="sos")
    filtered = sp_signal.sosfiltfilt(sos, raw_ppg)

    # Resample to a uniform rate by linear interpolation
    duration = len(filtered) / original_fs
    n_out = int(duration * target_fs)
    old_idx = np.linspace(0, len(filtered) - 1, len(filtered))
    new_idx = np.linspace(0, len(filtered) - 1, n_out)
    resampled = np.interp(new_idx, old_idx, filtered)

    # Clip outliers beyond 3 sigma, then z-score. The epsilon avoids dividing by
    # zero on a flat signal.
    mu, sigma = resampled.mean(), resampled.std()
    clipped = np.clip(resampled, mu - 3 * sigma, mu + 3 * sigma)
    return (clipped - clipped.mean()) / (clipped.std() + 1e-12)


def pad_or_truncate(arr, target_length, fill_value=0):
    """Pad with fill_value or truncate array to target_length."""
    if len(arr) >= target_length:
        return arr[:target_length]
    padding = np.full(target_length - len(arr), fill_value, dtype=arr.dtype)
    return np.concatenate([arr, padding])


def parse_sleep_stages_xml(xml_path):
    """Parse NSRR XML annotation file into a DataFrame of scored stages."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        scored = root.find(".//ScoredEvents")
        if scored is None:
            return None
        rows = []
        for ev in scored.iter("ScoredEvent"):
            etype = ev.find("EventType")
            if etype is None or etype.text != "Stages|Stages":
                continue
            concept = ev.find("EventConcept").text
            start = float(ev.find("Start").text)
            dur = float(ev.find("Duration").text)
            # Map stages to our 4 class scheme
            if "Wake" in concept:
                stage = 0
            elif "Stage 1" in concept or "Stage 2" in concept:
                stage = 1
            elif "Stage 3" in concept or "Stage 4" in concept:
                stage = 2
            elif "REM" in concept:
                stage = 3
            else:
                continue  # skip unscored or movement epochs
            rows.append({"Start": start, "Duration": dur, "Stage": stage})
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def expand_labels_to_windows(df_stages, total_duration):
    """Convert stage events into a per-30s-window label array."""
    n_win = int(np.ceil(total_duration / WINDOW_DURATION))
    labels = np.full(n_win, -1, dtype=int)  # -1 marks unlabelled windows
    for _, row in df_stages.iterrows():
        s = int(row["Start"] // WINDOW_DURATION)
        e = int((row["Start"] + row["Duration"]) // WINDOW_DURATION)
        labels[s : min(e, n_win)] = row["Stage"]
    return labels


def print_signal_labels(edf_dir):
    """Print the channel labels in a folder of EDF files, and the sampling rates of
    the PPG candidates.

    NSRR cohorts name the PPG channel differently (Pleth, PlethWV, TcCO2 Pleth), so
    this is the first thing to check when an extractor reports no PPG signal.
    """
    edf_files = sorted(f for f in os.listdir(edf_dir) if f.endswith(".edf"))
    all_labels = []
    ppg_labels = []
    ppg_rates = []

    for edf_file in tqdm(edf_files, desc="Reading EDF headers"):
        try:
            f = pyedflib.EdfReader(os.path.join(edf_dir, edf_file))
        except Exception:
            continue
        labels = f.getSignalLabels()
        all_labels.extend(labels)
        for idx, label in enumerate(labels):
            if "pleth" in label.lower() or "ppg" in label.lower():
                ppg_labels.append(label)
                ppg_rates.append(f.getSampleFrequency(idx))
        f.close()

    print("\nMost common signal labels:")
    for label, count in Counter(all_labels).most_common(30):
        print(f"  {label}: {count}")

    print("\nPPG candidates:")
    for label, count in Counter(ppg_labels).most_common():
        print(f"  {label}: {count}")

    print("\nPPG candidate sampling rates:")
    for rate, count in sorted(Counter(ppg_rates).items()):
        print(f"  {rate} Hz: {count} files")
