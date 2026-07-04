"""Download UCI Air Quality dataset."""
import zipfile
import io
import requests
from pathlib import Path
from gridpulse.utils.paths import RAW_DIR
from gridpulse.utils.logger import logger

AIR_QUALITY_URL = (
    "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
)


def download_air_quality(output_dir: Path | None = None) -> None:
    output_dir = output_dir or RAW_DIR / "air_quality_uci"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "AirQualityUCI.csv"
    if csv_path.exists():
        logger.info(f"Already exists: {csv_path}")
        return

    logger.info(f"Downloading UCI Air Quality from {AIR_QUALITY_URL}")
    response = requests.get(AIR_QUALITY_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(output_dir)
    logger.info(f"Extracted to {output_dir}")


if __name__ == "__main__":
    download_air_quality()