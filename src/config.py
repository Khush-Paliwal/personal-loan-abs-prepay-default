"""Paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The 1.7 GB raw file is shared with the credit-default-underwriting project; override
# with LC_RAW_CSV if it lives elsewhere.
RAW_CSV = Path(os.environ.get("LC_RAW_CSV", ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"))
PROCESSED = ROOT / "data" / "processed"
LOANS = PROCESSED / "loans.parquet"
MARKET_RATES = PROCESSED / "market_rates.parquet"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
for p in (PROCESSED, RESULTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

TRAIN_VINTAGES = [2012, 2013]      # fit the hazard model on fully observed lives
TEST_VINTAGES = [2014, 2015]       # project these pools out-of-time, compare to actual
TERM = 36
