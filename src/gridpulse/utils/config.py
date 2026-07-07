from pathlib import Path
from typing import Any
import yaml

from gridpulse.utils.paths import CONFIGS_DIR

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = CONFIGS_DIR / path

    with open(path) as f:
        return yaml.safe_load(f)
    
def load_training_config(name: str = "forecasting_default") -> dict:
    """Load a training configuration file."""
    return load_config(f"training/{name}.yaml")

def load_model_config(name: str) -> dict:
    return load_config(f"models/{name}.yaml")