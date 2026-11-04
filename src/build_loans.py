"""Step 1 - loan-level table with life-of-loan outcomes.

Reads the public Lending Club accepted-loans file (2.26M loans) with pyarrow, keeps
36-month individual loans issued 2012-2015 (all scheduled to mature before the
Mar-2019 snapshot, so every loan's full life is observed) and derives:

* event: prepaid in full early, defaulted (charged off), or ran to maturity
* event month on book (MOB) and the balance at the event
* realized principal and net losses on defaults
* the lender's going rate for the same sub-grade in every calendar month, used later
  to measure each borrower's refinancing incentive
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv

sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_CSV, LOANS, MARKET_RATES, RESULTS  # noqa: E402

COLS = ["id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade",
        "sub_grade", "emp_length", "home_ownership", "annual_inc", "verification_status",
        "issue_d", "loan_status", "purpose", "dti", "delinq_2yrs", "fico_range_low",
        "fico_range_high", "inq_last_6mths", "revol_util", "open_acc", "total_acc",
        "total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee",
        "last_pymnt_d", "application_type", "addr_state"]


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def main() -> None:
    table = pcsv.read_csv(
        RAW_CSV,
        convert_options=pcsv.ConvertOptions(
            include_columns=COLS, column_types={c: pa.string() for c in COLS}),
        parse_options=pcsv.ParseOptions(newlines_in_values=True),
    )
    df = table.to_pandas()
    del table
    n_raw = len(df)
    df = df[df.id.str.fullmatch(r"\d+", na=False)]
    df["issue_dt"] = pd.to_datetime(df.issue_d, format="%b-%Y", errors="coerce")
    df["term_m"] = df.term.str.extract(r"(\d+)").astype(float)

    # going rate by sub-grade x issue month, from ALL 36-month loans 2012-2018
    t36 = df[(df.term_m == 36) & df.issue_dt.notna()].copy()
    t36["int_rate"] = t36.int_rate.astype(float)
    market = (t36.groupby(["sub_grade", "issue_dt"]).int_rate.median()
              .rename("market_rate").reset_index())
    market.to_parquet(MARKET_RATES, index=False)

    df = df[(df.term_m == 36) & (df.application_type == "Individual")
            & df.issue_dt.dt.year.between(2012, 2015)
            & df.loan_status.isin(["Fully Paid", "Charged Off", "Default"])].copy()
    num = ["loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti",
           "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
           "revol_util", "open_acc", "total_acc", "total_rec_prncp", "total_rec_int",
           "recoveries", "collection_recovery_fee"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fico"] = (df.fico_range_low + df.fico_range_high) / 2
    df["last_pay_dt"] = pd.to_datetime(df.last_pymnt_d, format="%b-%Y", errors="coerce")
    df["mob_last_pay"] = months_between(df.last_pay_dt, df.issue_dt).fillna(0).clip(lower=0)

    bad = df.loan_status.isin(["Charged Off", "Default"])
    early = (~bad) & (df.mob_last_pay <= df.term_m - 2)
    df["event"] = np.where(bad, "default", np.where(early, "prepay", "matured"))
    # default timing = first payment never made (month after the last payment)
    df["event_mob"] = np.where(bad, (df.mob_last_pay + 1).clip(1, 36),
                               np.where(early, df.mob_last_pay.clip(1, 36), 36)).astype(int)

    r = df.int_rate / 1200
    n = df.term_m

    def sched_bal(t):
        return df.funded_amnt * ((1 + r) ** n - (1 + r) ** t) / ((1 + r) ** n - 1)

    df["bal_at_event"] = np.where(
        df.event == "prepay", sched_bal(df.event_mob - 1),
        np.where(df.event == "default", (df.funded_amnt - df.total_rec_prncp).clip(lower=0), 0.0))
    df["net_loss"] = np.where(
        df.event == "default",
        (df.funded_amnt - df.total_rec_prncp - df.recoveries + df.collection_recovery_fee).clip(lower=0),
        0.0)
    df["vintage"] = df.issue_dt.dt.year
    df["sub_grade_rank"] = (df.sub_grade.str[0].map(ord) - 65) * 5 + df.sub_grade.str[1].astype(int)
    df["emp_years"] = pd.to_numeric(df.emp_length.str.extract(r"(\d+)")[0], errors="coerce")
    df.loc[df.emp_length.str.startswith("<", na=False), "emp_years"] = 0
    keep = ["id", "vintage", "issue_dt", "funded_amnt", "int_rate", "installment", "grade",
            "sub_grade", "sub_grade_rank", "fico", "dti", "annual_inc", "emp_years",
            "home_ownership", "verification_status", "purpose", "delinq_2yrs",
            "inq_last_6mths", "revol_util", "open_acc", "total_acc", "loan_status", "event",
            "event_mob", "bal_at_event", "net_loss", "recoveries", "total_rec_prncp"]
    out = df[keep].reset_index(drop=True)
    out.to_parquet(LOANS, index=False)
    summary = out.groupby("vintage").agg(
        loans=("id", "size"), balance_musd=("funded_amnt", lambda s: s.sum() / 1e6),
        prepay_share=("event", lambda s: (s == "prepay").mean()),
        default_share=("event", lambda s: (s == "default").mean()),
        cnl_pct=("net_loss", "sum")).reset_index()
    summary["cnl_pct"] = summary.cnl_pct / (summary.balance_musd * 1e6)
    summary.to_csv(RESULTS / "vintage_summary.csv", index=False)
    print(f"raw rows {n_raw:,}; sample {len(out):,}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
