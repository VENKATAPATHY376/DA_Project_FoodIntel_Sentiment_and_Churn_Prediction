import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PAGES = ROOT / "pages"

for p in [DATA_RAW, DATA_PROC, MODELS, REPORTS, PAGES]:
    p.mkdir(parents=True, exist_ok=True)

def get_raw_file(name):
    return DATA_RAW / name

def get_proc_file(name):
    return DATA_PROC / name

def get_model_file(name):
    return MODELS / name

def get_report_file(name):
    return REPORTS / name
