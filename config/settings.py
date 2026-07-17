# هذا الملف يقرأ ملف YAML.
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"


def load_config():

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:

        config = yaml.safe_load(file)

    return config


config = load_config()