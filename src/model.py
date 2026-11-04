"""Step 2 - competing-risks prepayment/default model, pool projection and capital.

1. Static-pool curves: CPR, CDR and cumulative net loss (CNL) by vintage and month
   on book (MOB) - the standard ABS collateral view.
2. Discrete-time competing-risks hazard model: a multinomial logit on loan-months with
   three outcomes (stay current, prepay in full, default) and one indicator per month
   on book, so the seasoning curve is not forced into buckets.
3. Two out-of-time folds. Fold 1 (train 2012-13 -> project 2014) is used to choose the
   model specification; fold 2 (train 2012-14 -> project 2015) is the untouched test.
   Each pool is projected (a) at origination and (b) after 12 months of observed
   performance with an actual-to-expected (A/E) update - the way a surveillance
   analyst re-projects a seasoning ABS pool. A naive benchmark (the training vintages'
   average SMM/MDR curves, A/E-updated the same way) is run alongside.
4. Basel IRB "other retail" capital for the 2015 pool at origination and a default
   stress sensitivity for lifetime CNL.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from config import FIGURES, LOANS, MARKET_RATES, RESULTS, TERM  # noqa: E402

SEED = 42
FOLDS = [("validation", [2012, 2013], 2014), ("test", [2012, 2013, 2014], 2015)]
UPDATE_MOB = 12
TRAIN_FRAC = 0.4          # loans sampled (whole lives) for fitting, to fit in laptop RAM
PURPOSES = ["credit_card", "home_improvement", "other", "major_purchase", "small_business",
            "car", "medical"]                      # vs. debt_consolidation
HOMES = ["RENT", "OWN"]                            # vs. MORTGAGE
VERIF = ["Source Verified", "Not Verified"]        # vs. Verified
PALETTE = {2012: "#9bb7e0", 2013: "#5b8fd6", 2014: "#1f5fbf", 2015: "#0b2f6b"}
CONT = ["fico", "dti", "log_inc", "log_amt", "int_rate", "emp_years", "revol_util", "inq6",
        "delinq2", "refi_incentive", "refi_pos"]


# ----------------------------------------------------------------------------- helpers
def sched_balance(funded, rate_pct, t):
    r = rate_pct / 1200.0
    return funded * ((1 + r) ** TERM - (1 + r) ** t) / ((1 + r) ** TERM - 1)


def annualize(m):
    return 1 - (1 - np.clip(m, 0, 1)) ** 12


def market_grid(market):
    """Going rate for each sub-grade x calendar month, forward-filled over gaps."""
    grid = market.pivot(index="issue_dt", columns="sub_grade", values="market_rate").sort_index()
    full = pd.date_range(grid.index.min(), "2019-03-01", freq="MS")
    return grid.reindex(full).ffill().bfill()


def static_block(loans):
    f = {
        "fico": loans.fico.values,
        "dti": loans.dti.clip(0, 45).values,
        "log_inc": np.log1p(loans.annual_inc.clip(lower=1000)).values,
        "log_amt": np.log(loans.funded_amnt).values,
        "int_rate": loans.int_rate.values,
        "emp_years": loans.emp_years.fillna(5).values,
        "emp_missing": loans.emp_years.isna().values,
        "revol_util": loans.revol_util.fillna(55).clip(0, 150).values,
        "inq6": loans.inq_last_6mths.fillna(0).clip(0, 8).values,
        "delinq2": loans.delinq_2yrs.fillna(0).clip(0, 6).values,
    }
    for p in PURPOSES:
        f[f"purpose_{p}"] = (loans.purpose == p).values
    for h in HOMES:
        f[f"home_{h}"] = (loans.home_ownership == h).values
    for v in VERIF:
        f[f"verif_{v.replace(' ', '_')}"] = (loans.verification_status == v).values
    return pd.DataFrame(f).astype("float32")


class Design:
    """Builds the loan-month design matrix with a fixed column order."""

    def __init__(self, mkt, use_incentive):
        self.mkt, self.use_incentive = mkt, use_incentive
        self.cols = None

    def matrix(self, loans, stat, idx, mob):
        S = stat.to_numpy()[idx]
        names = list(stat.columns)
        blocks = [S]
        cal = pd.DatetimeIndex(loans.issue_dt.values[idx].astype("datetime64[M]")
                               + mob.astype("timedelta64[M]"))
        if self.use_incentive:
            ri = self.mkt.index.get_indexer(cal)
            ci = self.mkt.columns.get_indexer(loans.sub_grade.values[idx])
            assert (ri >= 0).all() and (ci >= 0).all()
            inc = loans.int_rate.values[idx] - self.mkt.to_numpy()[ri, ci]
            blocks.append(np.c_[np.clip(inc, -6, 10), np.clip(inc, 0, 10)])
            names += ["refi_incentive", "refi_pos"]
        blocks.append((mob[:, None] == np.arange(2, TERM)[None, :]))      # MOB 1 = reference
        names += [f"mob_{m:02d}" for m in range(2, TERM)]
        blocks.append((cal.month.values[:, None] == np.arange(2, 13)[None, :]))  # Jan = ref
        names += [f"cal_m{m:02d}" for m in range(2, 13)]
        X = np.hstack([b.astype("float32") for b in blocks])
        self.cols = names
        return X, cal


def expand(loans, full_life):
    """Loan-month index arrays. Observed panel stops at the event month."""
    last = np.full(len(loans), TERM - 1) if full_life else np.minimum(loans.event_mob.values, TERM - 1)
    idx = np.repeat(np.arange(len(loans)), last)
    mob = np.concatenate([np.arange(1, k + 1) for k in last]).astype(int)
    return idx, mob


class Hazard:
    def __init__(self, design):
        self.d = design

    def fit(self, loans):
        stat = static_block(loans)
        idx, mob = expand(loans, full_life=False)
        X, cal = self.d.matrix(loans, stat, idx, mob)
        hit = mob == loans.event_mob.values[idx]
        ev = loans.event.values[idx]
        y = np.where(hit & (ev == "prepay"), 1, np.where(hit & (ev == "default"), 2, 0))
        cont = [i for i, c in enumerate(self.d.cols) if c in CONT]
        self.mu = np.zeros(X.shape[1], dtype="float32")
        self.sd = np.ones(X.shape[1], dtype="float32")
        self.mu[cont] = X[:, cont].mean(0)
        self.sd[cont] = X[:, cont].std(0) + 1e-6
        X -= self.mu
        X /= self.sd
        self.est = LogisticRegression(C=1.0, max_iter=500, random_state=SEED).fit(X, y)
        self.n_rows = len(y)
        self.season = pd.Series(y == 1).groupby(cal.month).mean()
        return self

    def hazards(self, loans, chunk=20000):
        """Monthly prepay / default hazards for MOB 1..35 for every loan."""
        hp = np.zeros((len(loans), TERM - 1), dtype="float32")
        hd = np.zeros_like(hp)
        stat = static_block(loans)
        for s in range(0, len(loans), chunk):
            L = loans.iloc[s:s + chunk]
            idx, mob = expand(L, full_life=True)
            X, _ = self.d.matrix(L, stat.iloc[s:s + chunk].reset_index(drop=True), idx, mob)
            p = self.est.predict_proba((X - self.mu) / self.sd)
            hp[s:s + chunk] = p[:, 1].reshape(len(L), TERM - 1)
            hd[s:s + chunk] = p[:, 2].reshape(len(L), TERM - 1)
        return hp, hd

    def relative_coefs(self):
        c = pd.DataFrame(self.est.coef_.T, index=self.d.cols, columns=["stay", "prepay", "default"])
        return pd.DataFrame({"prepay_vs_stay": c.prepay - c.stay, "default_vs_stay": c.default - c.stay})


def sched_matrix(loans):
    t = np.arange(1, TERM)
    return sched_balance(loans.funded_amnt.values[:, None], loans.int_rate.values[:, None], t[None, :] - 1)


def project(loans, hp, hd, lgd, start=1, alive=None, mult_p=1.0, mult_d=1.0):
    """Expected pool behaviour for MOB start..35 given hazards (optionally scaled)."""
    hp = hp[:, start - 1:] * mult_p
    hd = np.clip(hd[:, start - 1:] * mult_d, 0, 0.999)
    hp = np.minimum(hp, 1 - hd)
    s0 = np.ones(len(loans)) if alive is None else alive.astype(float)
    surv_prev = s0[:, None] * np.cumprod(np.hstack([np.ones((len(loans), 1)), 1 - hp - hd]), 1)[:, :-1]
    bal = sched_matrix(loans)[:, start - 1:]
    c = pd.DataFrame({"mob": np.arange(start, TERM),
                      "balance": (surv_prev * bal).sum(0),
                      "prepaid": (surv_prev * hp * bal).sum(0),
                      "defaulted": (surv_prev * hd * bal).sum(0)})
    c["net_loss"] = c.defaulted * lgd
    c["smm"], c["mdr"] = c.prepaid / c.balance, c.defaulted / c.balance
    c["cpr"], c["cdr"] = annualize(c.smm), annualize(c.mdr)
    life_def = (surv_prev * hd).sum(1)
    life_pre = (surv_prev * hp).sum(1)
    return c, life_def, life_pre


def actual_curves(loans):
    sched = sched_matrix(loans)
    ev, emob = loans.event.values, loans.event_mob.values
    rows = []
    for t in range(1, TERM):
        alive = emob >= t
        rows.append({"mob": t, "balance": sched[alive, t - 1].sum(),
                     "prepaid": sched[alive & (emob == t) & (ev == "prepay"), t - 1].sum(),
                     "defaulted": sched[alive & (emob == t) & (ev == "default"), t - 1].sum(),
                     "net_loss": loans.net_loss.values[(emob == t) & (ev == "default")].sum()})
    c = pd.DataFrame(rows)
    c["smm"], c["mdr"] = c.prepaid / c.balance, c.defaulted / c.balance
    c["cpr"], c["cdr"] = annualize(c.smm), annualize(c.mdr)
    c["cnl"] = c.net_loss.cumsum() / loans.funded_amnt.sum()
    return c


def naive_project(loans, smm, mdr, lgd, start=1, alive_bal=None, mult_p=1.0, mult_d=1.0):
    """Apply average SMM/MDR curves to the pool's scheduled balance path."""
    sched_frac = (sched_matrix(loans).sum(0) / loans.funded_amnt.sum())
    b0 = loans.funded_amnt.sum() if alive_bal is None else alive_bal
    rows, surv = [], 1.0
    for t in range(start, TERM):
        bal = b0 * surv * sched_frac[t - 1] / sched_frac[start - 1]
        p, d = smm[t] * mult_p, mdr[t] * mult_d
        rows.append({"mob": t, "balance": bal, "prepaid": bal * p, "defaulted": bal * d})
        surv *= (1 - p - d)
    c = pd.DataFrame(rows)
    c["net_loss"] = c.defaulted * lgd
    c["smm"], c["mdr"] = c.prepaid / c.balance, c.defaulted / c.balance
    c["cpr"], c["cdr"] = annualize(c.smm), annualize(c.mdr)
    return c


def finish_cnl(c, prior_loss, orig):
    c = c.copy()
    c["cnl"] = (prior_loss + c.net_loss.cumsum()) / orig
    return c


def wal_years(balance_by_mob):
    bal = np.append(balance_by_mob, 0.0)
    principal = bal[:-1] - bal[1:]
    t = np.arange(1, len(bal)).astype(float)
    t[-1] = TERM
    return float((principal * t).sum() / principal.sum() / 12)


def irb_retail_k(pd_, lgd):
    pd_ = np.clip(pd_, 0.0003, 0.999)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    r = 0.03 * w + 0.16 * (1 - w)
    return lgd * (norm.cdf((norm.ppf(pd_) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)) - pd_)


def mae_pp(a, b, col, start):
    a, b = a.set_index("mob")[col], b.set_index("mob")[col]
    idx = [m for m in a.index if m >= start and m in b.index]
    return float((a[idx] - b[idx]).abs().mean() * 100)


# ------------------------------------------------------------------------------- run
def run_fold(loans, mkt, train_years, test_year, use_incentive):
    train = loans[loans.vintage.isin(train_years)]
    tr_def = train[train.event == "default"]
    lgd = float(tr_def.net_loss.sum() / tr_def.bal_at_event.sum())
    fit_loans = train.sample(frac=TRAIN_FRAC, random_state=SEED).reset_index(drop=True)
    hz = Hazard(Design(mkt, use_incentive)).fit(fit_loans)
    pool = loans[loans.vintage == test_year].reset_index(drop=True)
    hp, hd = hz.hazards(pool)
    act = actual_curves(pool)
    orig = pool.funded_amnt.sum()
    hist = pd.concat([actual_curves(train[train.vintage == v]) for v in train_years])
    hist = hist.groupby("mob")[["smm", "mdr"]].mean()

    # (a) at origination
    m0, life_def, life_pre = project(pool, hp, hd, lgd)
    m0 = finish_cnl(m0, 0.0, orig)
    n0 = finish_cnl(naive_project(pool, hist.smm, hist.mdr, lgd), 0.0, orig)

    # (b) after UPDATE_MOB months: A/E ratios on observed months, project the survivors
    k = UPDATE_MOB
    ev, emob = pool.event.values, pool.event_mob.values
    obs_p = ((ev == "prepay") & (emob <= k)).sum()
    obs_d = ((ev == "default") & (emob <= k)).sum()
    sp = np.cumprod(np.hstack([np.ones((len(pool), 1)), 1 - hp - hd]), 1)[:, :k]
    ae_p = float(obs_p / (sp * hp[:, :k]).sum())
    ae_d = float(obs_d / (sp * hd[:, :k]).sum())
    alive = emob > k
    m1, _, _ = project(pool, hp, hd, lgd, start=k + 1, alive=alive, mult_p=ae_p, mult_d=ae_d)
    prior_loss = pool.net_loss.values[(ev == "default") & (emob <= k)].sum()
    m1 = finish_cnl(m1, prior_loss, orig)
    a_k = act[act.mob <= k]
    nae_p = float(a_k.prepaid.sum() / (a_k.balance * hist.smm[a_k.mob].values).sum())
    nae_d = float(a_k.defaulted.sum() / (a_k.balance * hist.mdr[a_k.mob].values).sum())
    alive_bal = act.loc[act.mob == k + 1, "balance"].iloc[0]
    n1 = finish_cnl(naive_project(pool, hist.smm, hist.mdr, lgd, start=k + 1, alive_bal=alive_bal,
                                  mult_p=nae_p, mult_d=nae_d), prior_loss, orig)

    def scores(c, start):
        return {"cpr_mae_pp": mae_pp(act, c, "cpr", start), "cdr_mae_pp": mae_pp(act, c, "cdr", start),
                "lifetime_cnl": float(c.cnl.iloc[-1]),
                "cnl_error_pp": float((c.cnl.iloc[-1] - act.cnl.iloc[-1]) * 100)}

    res = {"train": "-".join(map(str, train_years)), "test_vintage": test_year,
           "use_incentive": use_incentive, "train_loan_months": hz.n_rows, "lgd": lgd,
           "actual_cnl": float(act.cnl.iloc[-1]), "actual_avg_cpr": float(act.cpr.mean()),
           "actual_avg_cdr": float(act.cdr.mean()),
           "model_at_origination": scores(m0, 1), "naive_at_origination": scores(n0, 1),
           f"model_updated_m{k}": {**scores(m1, k + 1), "ae_prepay": ae_p, "ae_default": ae_d},
           f"naive_updated_m{k}": {**scores(n1, k + 1), "ae_prepay": nae_p, "ae_default": nae_d},
           "auc_lifetime_default_model": float(roc_auc_score(ev == "default", life_def)),
           "auc_lifetime_default_subgrade": float(roc_auc_score(ev == "default", pool.sub_grade_rank)),
           "auc_lifetime_prepay_model": float(roc_auc_score(ev == "prepay", life_pre)),
           "wal_actual_yrs": wal_years(act.balance.values), "wal_model_orig_yrs": wal_years(m0.balance.values)}
    curves = {"actual": act, "model_orig": m0, "naive_orig": n0, "model_upd": m1, "naive_upd": n1}
    return res, curves, hz, (pool, hp, hd, lgd)


def main():
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9.5})
    loans = pd.read_parquet(LOANS)
    mkt = market_grid(pd.read_parquet(MARKET_RATES))
    out = {}

    # static-pool curves for every vintage
    allc = []
    for v in sorted(loans.vintage.unique()):
        c = actual_curves(loans[loans.vintage == v])
        c["vintage"] = v
        allc.append(c)
    allc = pd.concat(allc)
    allc.to_csv(RESULTS / "actual_pool_curves.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for v, g in allc.groupby("vintage"):
        for ax, col in zip(axes, ["cpr", "cdr", "cnl"]):
            ax.plot(g.mob, g[col] * 100, color=PALETTE[v], label=str(v))
    for ax, t in zip(axes, ["CPR (%, annualized prepayment)", "CDR (%, annualized default)",
                            "Cumulative net loss (% of original balance)"]):
        ax.set(title=t, xlabel="Months on book")
    axes[0].legend(title="Vintage", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "static_pool_curves.png")
    plt.close(fig)
    out["vintage_lifetime"] = (allc.groupby("vintage").apply(lambda g: pd.Series({
        "lifetime_cnl": g.cnl.iloc[-1], "avg_cpr": g.cpr.mean(), "avg_cdr": g.cdr.mean()}))
        .round(4).reset_index().to_dict(orient="records"))

    # model selection on the validation fold only (with vs. without refi incentive)
    val = {}
    for inc in (True, False):
        r, *_ = run_fold(loans, mkt, FOLDS[0][1], FOLDS[0][2], inc)
        val[inc] = r
        print(f"validation 2014, incentive={inc}: "
              f"orig CPR MAE {r['model_at_origination']['cpr_mae_pp']:.2f}pp, "
              f"upd CPR MAE {r['model_updated_m12']['cpr_mae_pp']:.2f}pp, "
              f"upd CNL err {r['model_updated_m12']['cnl_error_pp']:.2f}pp")
    score = {k: v["model_updated_m12"]["cpr_mae_pp"] + v["model_updated_m12"]["cdr_mae_pp"]
             for k, v in val.items()}
    use_inc = min(score, key=score.get)
    out["validation_2014"] = {str(k): v for k, v in val.items()}
    out["chosen_use_incentive"] = use_inc
    print("chosen use_incentive =", use_inc)

    # untouched test fold
    res, curves, hz, (pool, hp, hd, lgd) = run_fold(loans, mkt, FOLDS[1][1], FOLDS[1][2], use_inc)
    out["test_2015"] = res
    print(json.dumps(res, indent=2, default=float))
    rel = hz.relative_coefs()
    rel.round(4).to_csv(RESULTS / "hazard_coefficients.csv")
    sd = pd.Series(hz.sd, index=hz.d.cols)
    if use_inc:
        per_pp = (rel.loc["refi_incentive", "prepay_vs_stay"] / sd["refi_incentive"]
                  + rel.loc["refi_pos", "prepay_vs_stay"] / sd["refi_pos"])
        out["prepay_odds_per_1pp_incentive"] = float(np.exp(per_pp))
    out["default_odds_per_20pt_fico"] = float(np.exp(20 * rel.loc["fico", "default_vs_stay"] / sd["fico"]))
    out["prepay_seasonality_index"] = (hz.season / hz.season.mean()).round(3).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    style = {"actual": ("Actual", "#111111", "-", 1.9), "model_orig": ("Model at origination", "#9bb7e0", "-", 1.2),
             "model_upd": ("Model, A/E-updated at month 12", "#1f5fbf", "-", 1.6),
             "naive_upd": ("Naive curve, A/E-updated at month 12", "#e07b00", "--", 1.2)}
    for ax, col, lab in zip(axes, ["cpr", "cdr", "cnl"], ["CPR %", "CDR %", "Cumulative net loss %"]):
        for key, (name, color, ls, lw) in style.items():
            c = curves[key]
            ax.plot(c.mob, c[col] * 100, color=color, ls=ls, lw=lw, label=name)
        ax.axvline(UPDATE_MOB + 0.5, color="#bbbbbb", ls=":")
        ax.set(title=f"2015 vintage (out-of-time) - {lab}", xlabel="Months on book")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "oot_projection_2015.png")
    plt.close(fig)
    for key, c in curves.items():
        c.to_csv(RESULTS / f"curve_2015_{key}.csv", index=False)

    # Basel IRB retail capital at origination (12-month PD from the hazards)
    sp = np.cumprod(np.hstack([np.ones((len(pool), 1)), 1 - hp - hd]), 1)[:, :12]
    pd12 = (sp * hd[:, :12]).sum(1)
    k_ = irb_retail_k(pd12, lgd)
    ead = pool.funded_amnt.values
    cap = pd.DataFrame({"grade": pool.grade, "ead": ead, "pd12": pd12, "k": k_})
    by_grade = cap.groupby("grade").apply(lambda g: pd.Series({
        "loans": len(g), "ead_musd": g.ead.sum() / 1e6, "avg_pd12": np.average(g.pd12, weights=g.ead),
        "capital_k_pct": np.average(g.k, weights=g.ead) * 100}))
    by_grade["rwa_density_pct"] = by_grade.capital_k_pct * 12.5
    by_grade.round(3).to_csv(RESULTS / "basel_irb_capital_by_grade_2015.csv")
    out["basel"] = {"pool_pd12": float(np.average(pd12, weights=ead)), "lgd": lgd,
                    "capital_pct_of_ead": float(np.average(k_, weights=ead) * 100),
                    "rwa_density_pct": float(np.average(k_, weights=ead) * 1250),
                    "observed_12m_default_rate": float(((pool.event == "default") & (pool.event_mob <= 12)).mean())}

    # default-stress sensitivity of lifetime CNL (at origination)
    stress = []
    for m in [1.0, 1.5, 2.0, 3.0]:
        c, *_ = project(pool, hp, hd, lgd, mult_d=m)
        stress.append({"default_multiplier": m, "projected_cnl": float(c.net_loss.sum() / ead.sum())})
    out["cnl_stress_2015"] = stress
    out["cnl_actual_2015"] = res["actual_cnl"]
    pd.DataFrame(stress).round(4).to_csv(RESULTS / "cnl_stress_2015.csv", index=False)

    # drivers figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    s = pd.Series(out["prepay_seasonality_index"]).sort_index()
    axes[0].bar(s.index, s.values, color=["#1f5fbf" if v == s.max() else "#9bb7e0" for v in s.values])
    axes[0].axhline(1, color="#777", ls=":")
    axes[0].set(title="Prepayment seasonality (1.0 = average month)", xlabel="Calendar month",
                xticks=range(1, 13))
    mobc = rel.loc[[c for c in rel.index if c.startswith("mob_")]]
    axes[1].plot(range(2, TERM), np.exp(mobc.prepay_vs_stay.values), color="#1f5fbf", label="Prepay")
    axes[1].plot(range(2, TERM), np.exp(mobc.default_vs_stay.values), color="#c0392b", label="Default")
    axes[1].axhline(1, color="#777", ls=":")
    axes[1].set(title="Hazard odds by loan age (vs. month 1)", xlabel="Months on book", yscale="log")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "drivers.png")
    plt.close(fig)

    print(by_grade.round(3).to_string())
    print(json.dumps({k: out[k] for k in ("basel", "cnl_stress_2015", "prepay_seasonality_index")},
                     indent=1, default=float))
    print({k: out.get(k) for k in ("prepay_odds_per_1pp_incentive", "default_odds_per_20pt_fico")})
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
