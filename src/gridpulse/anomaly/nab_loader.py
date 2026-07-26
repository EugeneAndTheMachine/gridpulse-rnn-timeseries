"""Load Numenta Anomaly Benchmark dataset."""
import json
import urllib.request
import pandas as pd
from pathlib import Path
from gridpulse.utils.paths import RAW_DIR
from gridpulse.utils.logger import logger


NAB_ROOT = RAW_DIR / "nab"
NAB_LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"


def download_nab_labels(dest: Path | None = None, force: bool = False) -> Path:
    """Download NAB combined_windows.json labels from the numenta/NAB repo."""
    dest = dest or (NAB_ROOT / "labels" / "combined_windows.json")
    if dest.exists() and not force:
        logger.info(f"NAB labels already exist at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading NAB labels from {NAB_LABELS_URL}")
    urllib.request.urlretrieve(NAB_LABELS_URL, dest)
    logger.info(f"Saved to {dest}")
    return dest


def list_nab_files() -> list[Path]:
    """List all NAB CSV files."""
    return sorted(NAB_ROOT.rglob("*.csv"))


def load_nab_series(csv_path: Path) -> pd.DataFrame:
    """Load 1 NAB series."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_nab_labels(labels_path: Path | None = None) -> dict:
    """Load anomaly labels dict."""
    if labels_path is None:
        # NAB labels thường ở labels/combined_windows.json
        labels_path = NAB_ROOT / "labels" / "combined_windows.json"

    if not labels_path.exists():
        logger.warning(f"Labels not found at {labels_path}")
        return {}

    with open(labels_path) as f:
        return json.load(f)


def get_anomaly_windows(
    labels: dict, series_name: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (start, end) timestamps marked anomalous."""
    key = None
    for k in labels.keys():
        if k.endswith(series_name) or series_name in k:
            key = k
            break

    if key is None:
        return []

    windows = []
    for window in labels[key]:
        start = pd.to_datetime(window[0])
        end = pd.to_datetime(window[1])
        windows.append((start, end))
    return windows