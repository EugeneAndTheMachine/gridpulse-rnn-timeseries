from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[3]

# Data directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# Model directories
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTS_DIR = MODELS_DIR / "exports"

# Config directory
CONFIGS_DIR = ROOT / "configs"