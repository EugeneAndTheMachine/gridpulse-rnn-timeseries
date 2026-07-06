"""Test data download and validation."""
from gridpulse.utils.paths import RAW_DIR


def test_ett_files_exist():
    """After make download-data, ETT files should exist."""
    ett_dir = RAW_DIR / "ett"
    assert (ett_dir / "ETTh1.csv").exists()


def test_ett_shape():
    """ETTh1 should have 8 columns and ~17420 rows."""
    import pandas as pd
    df = pd.read_csv(RAW_DIR / "ett" / "ETTh1.csv")
    assert df.shape[1] == 8
    assert 17000 < df.shape[0] < 18000
