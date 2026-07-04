"""Download ETT-small dataset from GitHub."""
import requests
from pathlib import Path
from gridpulse.utils.paths import RAW_DIR
from gridpulse.utils.logger import logger

ETT_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
)
ETT_FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]


def download_ett(output_dir: Path | None = None) -> None:
    output_dir = output_dir or RAW_DIR / "ett"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in ETT_FILES:
        filepath = output_dir / filename
        if filepath.exists():
            logger.info(f"Already exists: {filepath}")
            continue

        url = f"{ETT_BASE_URL}/{filename}"
        logger.info(f"Downloading {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        filepath.write_text(response.text)
        logger.info(f"Saved to {filepath}")


if __name__ == "__main__":
    download_ett()