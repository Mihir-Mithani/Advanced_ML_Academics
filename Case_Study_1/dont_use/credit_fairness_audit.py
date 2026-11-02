# %% [markdown]
# # Algorithmic Fairness & Risk Assessment in High-Stakes Lending
# **Home Credit Default Risk** — ML Ethics & Risk Consulting audit
#
# | Part | Content |
# |---|---|
# | **A** | Baseline pipeline, imbalance handling, ROC-AUC / PR-AUC, 5:1 business cost matrix |
# | **B** | Protected-attribute proxies, Disparate Impact Ratio, Equalized Odds, proxy / redlining analysis, mitigation and the profit-vs-fairness trade-off |
# | **C** | SHAP (global + local), counterfactual recourse, automated Adverse Action Notice |
#
# **How to run:** set `DATA_DIR` in the config cell below (or the `HC_DATA_DIR` environment variable) to the folder that contains
# `application_train.csv` (and optionally `bureau.csv`). If no data is found the notebook falls back to a *synthetic* dataset with the same
# schema so the code can be tested end-to-end — **synthetic numbers must never be reported as findings.**
# All tables / figures are written to `outputs/` and a machine-readable `outputs/results_summary.json` is produced at the end.
# *Runtime:* the full 307k-row file with bureau features should take on the order of 15-30 minutes on a laptop; set `SAMPLE_N = 60000` for a quick smoke test.
#
# **Conventions.** Positive class (`TARGET = 1`) = default. The model outputs a probability of default (PD). A client is **rejected** (flagged) when `PD >= threshold`, otherwise **approved**.
# *False Negative* = a defaulter that was approved (cost **5**); *False Positive* = a good client that was rejected (cost **1**).

# %%
# ----------------------------- imports -----------------------------
import os, json, warnings, itertools, textwrap, time
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", lambda v: f"{v:,.4f}")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

# %%
# ----------------------------- CONFIG -----------------------------
SEED = 42
FN_COST, FP_COST = 5.0, 1.0      # business cost matrix: approving a defaulter costs 5x rejecting a good client
MIN_GROUP_N = 100                # groups smaller than this (in the test split) are reported but excluded from DIR / gap summaries
USE_BUREAU = True                # aggregate bureau.csv if present (optional table)
FORCE_SYNTHETIC = False          # True -> always use synthetic data (for testing the code)
SYNTH_N = 30000                  # size of the synthetic fallback
SYNTH_BIAS = 1.0                 # strength of built-in group effects in the synthetic data (only for code testing)
SAMPLE_N = None                  # e.g. 60000 -> quick smoke-test on a random subsample of the real data
N_BOOT = 300                     # bootstrap resamples for DIR confidence intervals

# folder containing application_train.csv. Edit this line, or export HC_DATA_DIR=/path/to/home-credit-default-risk
DATA_DIR = Path(os.environ.get("HC_DATA_DIR", "home-credit-default-risk"))
_candidates = [DATA_DIR, Path("dataset/home-credit-default-risk"), Path("../dataset/home-credit-default-risk"),
               Path("../home-credit-default-risk"), Path("data/home-credit-default-risk"), Path("data")]

OUT = Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)
RESULTS = {}                     # everything reportable ends up here -> results_summary.json


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def save_table(df, name):
    df.to_csv(TAB / f"{name}.csv")
    return df

# %% [markdown]
# ## 0. Data loading
# `application_train.csv` is the only labelled file (`application_test.csv` has no `TARGET`), so I create my own stratified
# **60 % train / 20 % validation / 20 % test** split. Validation is used for early stopping, threshold selection and
# group-threshold learning; **test is touched only for final reporting.**

# %%
def make_synthetic(n=30000, seed=SEED, bias=1.0):
    """Synthetic data with the Home Credit schema (subset). ONLY for testing the code path."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["F", "M"], n, p=[0.66, 0.34]); male = (gender == "M").astype(float)
    age = np.clip(rng.normal(43, 12, n), 21, 69)
    region = rng.choice([1, 2, 3], n, p=[0.12, 0.74, 0.14])
    q = rng.normal(0, 1, n) - 0.20 * bias * (region == 3) + 0.10 * bias * (region == 1)          # latent creditworthiness
    inc_type = np.where(age > 58, rng.choice(["Pensioner", "Working", "State servant"], n, p=[.65, .25, .10]),
                        rng.choice(["Working", "Commercial associate", "State servant", "Pensioner", "Unemployed", "Student", "Businessman"],
                                   n, p=[.58, .26, .10, .02, .02, .017, .003]))
    no_job = np.isin(inc_type, ["Pensioner", "Unemployed", "Student"])
    yrs = np.minimum(rng.exponential(6, n), age - 18)
    days_emp = np.where(no_job, 365243, -(yrs * 365.25))
    occ = np.empty(n, dtype=object)
    mm = male == 1
    occ[mm] = rng.choice(["Laborers", "Drivers", "Core staff", "Managers", "Security staff", "Sales staff"], mm.sum(), p=[.32, .22, .14, .13, .12, .07])
    occ[~mm] = rng.choice(["Sales staff", "Core staff", "Accountants", "Medicine staff", "Cooking staff", "Laborers", "Managers"], (~mm).sum(), p=[.28, .24, .12, .12, .10, .08, .06])
    occ[no_job & (rng.random(n) < .5)] = np.nan
    org = rng.choice(["Business Entity Type 3", "Self-employed", "Other", "Medicine", "Government", "School", "Construction", "Trade: type 7"], n)
    org = np.where(no_job, "XNA", org)
    housing = rng.choice(["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"], n,
                         p=np.array([.886, .048, .036, .016, .0087, .0057]) / np.sum([.886, .048, .036, .016, .0087, .0057]))
    housing[(age < 28) & (rng.random(n) < .2)] = "With parents"
    income = np.exp(rng.normal(11.85, .55, n) + .12 * male - .15 * (region == 3) + .15 * (region == 1) + .1 * q) * np.where(inc_type == "Pensioner", .7, 1)
    credit = np.round(income * np.exp(rng.normal(np.log(3.2), .5, n)), -2)
    annuity = credit * rng.uniform(.035, .085, n)
    goods = np.where(rng.random(n) < .001, np.nan, credit * rng.uniform(.85, 1.0, n))
    e = lambda: np.clip(.5 + .15 * (q + rng.normal(0, .7, n)), .01, .99)
    e1, e2, e3 = e(), e(), e()
    e1[rng.random(n) < .56] = np.nan; e2[rng.random(n) < .002] = np.nan; e3[rng.random(n) < .20] = np.nan
    base = -0.85 * q + .35 * (np.log(credit / income) - np.log(3.2)) + .35 * bias * male + .25 * bias * (region == 3) - .15 * bias * (region == 1) \
           + .25 * (age < 30) - .2 * (age > 55) + .3 * (inc_type == "Unemployed") + rng.normal(0, .3, n)
    lo, hi = -8, 2
    for _ in range(50):
        b0 = (lo + hi) / 2
        if (1 / (1 + np.exp(-(b0 + base)))).mean() > .081: hi = b0
        else: lo = b0
    target = (rng.random(n) < 1 / (1 + np.exp(-(b0 + base)))).astype(int)
    car = rng.random(n) < np.where(mm, .45, .30)
    children = np.minimum(rng.poisson(.4, n), 6)
    fam = rng.choice(["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], n, p=[.64, .15, .10, .065, .045])
    gender_col = gender.astype(object); gender_col[rng.choice(n, 3, replace=False)] = "XNA"
    pop = np.where(region == 1, rng.uniform(.03, .07, n), np.where(region == 2, rng.uniform(.008, .035, n), rng.uniform(.0003, .012, n)))
    reg_city = np.clip(region + np.where(rng.random(n) < .1, rng.choice([-1, 1], n), 0), 1, 3)
    app = pd.DataFrame({
        "SK_ID_CURR": 100001 + np.arange(n), "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[.9, .1]),
        "CODE_GENDER": gender_col, "FLAG_OWN_CAR": np.where(car, "Y", "N"), "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[.7, .3]),
        "CNT_CHILDREN": children, "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods,
        "NAME_INCOME_TYPE": inc_type,
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], n, p=np.array([.7115, .244, .033, .0115, .0005]) / 1.0005),
        "NAME_FAMILY_STATUS": fam, "NAME_HOUSING_TYPE": housing, "REGION_POPULATION_RELATIVE": pop,
        "DAYS_BIRTH": -(age * 365.25).astype(int), "DAYS_EMPLOYED": days_emp, "DAYS_REGISTRATION": -rng.uniform(0, 20000, n),
        "DAYS_ID_PUBLISH": -rng.integers(0, 6500, n), "OWN_CAR_AGE": np.where(car, rng.integers(0, 20, n), np.nan),
        "OCCUPATION_TYPE": occ, "CNT_FAM_MEMBERS": children + rng.choice([1, 2], n, p=[.3, .7]),
        "REGION_RATING_CLIENT": region, "REGION_RATING_CLIENT_W_CITY": reg_city, "ORGANIZATION_TYPE": org,
        "EXT_SOURCE_1": e1, "EXT_SOURCE_2": e2, "EXT_SOURCE_3": e3,
        "DAYS_LAST_PHONE_CHANGE": -rng.exponential(900, n).astype(int),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(n) < .13, np.nan, rng.poisson(np.clip(1.9 - .3 * q, .1, None))),
        "FLAG_DOCUMENT_3": rng.choice([0, 1], n, p=[.29, .71]),
    })
    nb = np.where(rng.random(n) < .15, 0, rng.poisson(3, n))
    ids = np.repeat(app["SK_ID_CURR"].values, nb); qrep = np.repeat(q, nb); m = len(ids)
    csum = np.exp(rng.normal(11.5, 1, m)); active = rng.random(m) < .4
    frac = np.clip(rng.beta(2, 3, m) - .1 * qrep, 0, 1)
    bureau = pd.DataFrame({
        "SK_ID_CURR": ids, "CREDIT_ACTIVE": np.where(active, "Active", "Closed"), "DAYS_CREDIT": -rng.integers(30, 3000, m),
        "CREDIT_DAY_OVERDUE": np.where(rng.random(m) < .03, rng.integers(1, 60, m), 0),
        "AMT_CREDIT_SUM": csum, "AMT_CREDIT_SUM_DEBT": np.where(active, csum * frac, 0.0),
        "AMT_CREDIT_SUM_OVERDUE": np.where(rng.random(m) < .02, csum * .05, 0.0)})
    return app, bureau


def find_data_dir():
    for c in _candidates:
        if (c / "application_train.csv").exists():
            return c
    return None


BUREAU_COLS = ["SK_ID_CURR", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE"]
real_dir = None if FORCE_SYNTHETIC else find_data_dir()
if real_dir is not None:
    print(f"Using REAL data from: {real_dir.resolve()}")
    app = pd.read_csv(real_dir / "application_train.csv")
    app_test_path = real_dir / "application_test.csv"
    bureau = pd.read_csv(real_dir / "bureau.csv", usecols=BUREAU_COLS) if (USE_BUREAU and (real_dir / "bureau.csv").exists()) else None
    DATA_SOURCE = "real"
else:
    print("!" * 90 + "\n!! Real data NOT found -> using SYNTHETIC data. Do not report these numbers as findings. !!\n" + "!" * 90)
    app, bureau = make_synthetic(SYNTH_N, bias=SYNTH_BIAS)
    app_test_path = None
    DATA_SOURCE = "synthetic"
if SAMPLE_N and len(app) > SAMPLE_N:
    app = app.sample(SAMPLE_N, random_state=SEED).reset_index(drop=True)
print(f"application_train: {app.shape}, default rate = {app['TARGET'].mean():.4f}, bureau rows = {0 if bureau is None else len(bureau):,}")
RESULTS["data"] = dict(source=DATA_SOURCE, n_rows=int(len(app)), default_rate=float(app["TARGET"].mean()), bureau_used=bureau is not None)

# %% [markdown]
# ## 1. Feature engineering & preprocessing
# * `DAYS_EMPLOYED == 365243` is a known placeholder (pensioners / unemployed) → set to `NaN` and keep an indicator.
# * Ratio features (`CREDIT_INCOME_PCT`, `ANNUITY_INCOME_PCT`, `CREDIT_TERM_MONTHS`, …) are built in **one reusable function** (`add_ratios`) so the
#   counterfactual search in Part C can change raw amounts and recompute every dependent feature consistently.
# * Optional bureau aggregates (credit utilisation = `BUREAU_DEBT_RATIO`, overdue balances, number of accounts).
# * **Categorical encoding:** LightGBM uses native categorical splits; for Logistic Regression I compare **One-Hot** vs **Target Encoding** (cross-fitted to avoid target leakage).

# %%
def aggregate_bureau(b):
    b = b.copy()
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    g = b.groupby("SK_ID_CURR")
    agg = pd.DataFrame({
        "BUREAU_COUNT": g.size(), "BUREAU_ACTIVE_COUNT": g["IS_ACTIVE"].sum(),
        "BUREAU_DAYS_CREDIT_MEAN": g["DAYS_CREDIT"].mean(), "BUREAU_CREDIT_SUM": g["AMT_CREDIT_SUM"].sum(),
        "BUREAU_DEBT_SUM": g["AMT_CREDIT_SUM_DEBT"].sum(), "BUREAU_OVERDUE_SUM": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "BUREAU_MAX_DAY_OVERDUE": g["CREDIT_DAY_OVERDUE"].max()})
    return agg.reset_index()


def clean_raw(df):
    df = df.copy()
    if "DAYS_EMPLOYED" in df:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    if "CODE_GENDER" in df:
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    return df


def add_ratios(df):
    """Derived features. Safe to call repeatedly on modified raw amounts (used by the counterfactual search)."""
    df = df.copy()
    inc = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_INCOME_PCT"] = df["AMT_CREDIT"] / inc
    df["ANNUITY_INCOME_PCT"] = df["AMT_ANNUITY"] / inc
    df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["GOODS_CREDIT_PCT"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = inc / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["DAYS_EMPLOYED_PCT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]          # age-derived -> treated as a sensitive-derived feature
    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df]
    df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
    if "BUREAU_DEBT_SUM" in df:
        df["BUREAU_DEBT_RATIO"] = df["BUREAU_DEBT_SUM"] / (df["BUREAU_CREDIT_SUM"] + 1.0)   # credit utilisation proxy
    return df.replace([np.inf, -np.inf], np.nan)


def build_features(app_df, bureau_agg=None):
    df = app_df.copy()
    if bureau_agg is not None:
        df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
        df["BUREAU_COUNT"] = df["BUREAU_COUNT"].fillna(0)               # 0 = thin file (no bureau history)
        df["BUREAU_ACTIVE_COUNT"] = df["BUREAU_ACTIVE_COUNT"].fillna(0)
    return add_ratios(clean_raw(df))


BUREAU_AGG = aggregate_bureau(bureau) if bureau is not None else None
df_all = build_features(app, BUREAU_AGG)
y_all = df_all["TARGET"].to_numpy().astype(int)

X_full = df_all.drop(columns=["TARGET", "SK_ID_CURR"])
CAT_COLS = [c for c in X_full.columns if not pd.api.types.is_numeric_dtype(X_full[c])]
NUM_COLS = [c for c in X_full.columns if c not in CAT_COLS]
for c in CAT_COLS:
    X_full[c] = X_full[c].astype(object).where(X_full[c].notna(), np.nan).astype("category")
CATEGORIES = {c: X_full[c].cat.categories for c in CAT_COLS}      # fixed category levels (also used to score application_test)
ALL_FEATS = list(X_full.columns)
print(f"{len(ALL_FEATS)} features  ({len(NUM_COLS)} numeric, {len(CAT_COLS)} categorical)")

# ---- stratified 60/20/20 split ----
idx = np.arange(len(X_full))
idx_tr, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y_all, random_state=SEED)
idx_va, idx_te = train_test_split(idx_tmp, test_size=0.5, stratify=y_all[idx_tmp], random_state=SEED)
y_tr, y_va, y_te = y_all[idx_tr], y_all[idx_va], y_all[idx_te]
print(f"train {len(idx_tr):,} | valid {len(idx_va):,} | test {len(idx_te):,} | default rate {y_tr.mean():.4f} / {y_va.mean():.4f} / {y_te.mean():.4f}")

# %% [markdown]
# ## 2. Sensitive attributes (defined from proxies available in the data)
# The dataset has no race / ethnicity field. I use the proxies suggested in the brief, and I keep them **outside the feature matrix logic**: the audit
# frame is always built from the raw columns, even for models that do not use them as inputs.
#
# | Audit attribute | Definition | Why it matters |
# |---|---|---|
# | `GENDER` | `CODE_GENDER` (F / M; `XNA` excluded) | Protected class (ECOA / Reg B) |
# | `AGE_GROUP` | `<30`, `30-44`, `45-59`, `60+` | Age is protected (elderly in particular) |
# | `REGION_TIER` | `REGION_RATING_CLIENT` 1 (best) – 3 (worst) | Geographic / redlining proxy for low-income postal regions |
# | `INCOME_CATEGORY` | Salaried / business vs. non-traditional (pensioner, unemployed, student, maternity leave) | "Non-traditional borrowers" |
# | `HOUSING_GROUP` | Own / co-op house vs. renting, with parents, municipal, office | Housing type proxy |
#
# **Direct sensitive columns** (`DIRECT`): gender, age and everything derived from age, marital status.
# **Candidate proxy columns** (`PROXY`): region ratings / population density, housing type, occupation, organisation type.

# %%
def build_audit_frame(df):
    a = pd.DataFrame(index=df.index)
    a["GENDER"] = df["CODE_GENDER"].map({"F": "Female", "M": "Male"})
    a["AGE_GROUP"] = pd.cut(df["AGE_YEARS"], [0, 30, 45, 60, 200], labels=["<30", "30-44", "45-59", "60+"], right=False).astype(object)
    a["REGION_TIER"] = df["REGION_RATING_CLIENT"].map({1: "Region 1 (best)", 2: "Region 2", 3: "Region 3 (worst)"})
    trad = df["NAME_INCOME_TYPE"].isin(["Working", "State servant", "Commercial associate", "Businessman"])
    a["INCOME_CATEGORY"] = np.where(trad, "Traditional (salaried/business)", "Non-traditional")
    own = df["NAME_HOUSING_TYPE"].isin(["House / apartment", "Co-op apartment"])
    a["HOUSING_GROUP"] = np.where(own, "Own / standard housing", "Renting / parents / municipal / office")
    return a


ATTRS = ["GENDER", "AGE_GROUP", "REGION_TIER", "INCOME_CATEGORY", "HOUSING_GROUP"]
AUD = build_audit_frame(df_all)
AUD_tr, AUD_va, AUD_te = (AUD.iloc[i].reset_index(drop=True) for i in (idx_tr, idx_va, idx_te))

DIRECT = [c for c in ["CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS", "DAYS_EMPLOYED_PCT", "NAME_FAMILY_STATUS"] if c in ALL_FEATS]
PROXY = [c for c in ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "REGION_POPULATION_RELATIVE", "NAME_HOUSING_TYPE",
                     "OCCUPATION_TYPE", "ORGANIZATION_TYPE"] if c in ALL_FEATS]

# Quick EDA: raw default rate and group size per audit attribute (full labelled data)
eda = pd.concat({a: pd.DataFrame({"n": AUD[a].value_counts(), "default_rate": pd.Series(y_all, index=AUD.index).groupby(AUD[a]).mean()}) for a in ATTRS})
save_table(eda, "eda_group_default_rates"); eda

# %% [markdown]
# # PART A — Baseline modelling & business-aware evaluation
# Four candidates are compared, covering the encoding and class-imbalance choices requested:
#
# | Model | Categorical encoding | Imbalance handling |
# |---|---|---|
# | `LogReg_OHE` | One-Hot | `class_weight="balanced"` |
# | `LogReg_TargetEnc` | Cross-fitted Target Encoding | `class_weight="balanced"` |
# | `LGBM_plain` | native categorical | none (rely on **cost-optimal threshold**) |
# | `LGBM_spw` | native categorical | `scale_pos_weight = n_neg / n_pos` |
#
# Accuracy is ignored (a model that approves everybody is ~92 % "accurate"). I report **ROC-AUC**, **PR-AUC** (average precision; the base rate is the
# no-skill baseline), and the **total business cost** at the cost-minimising threshold, which is chosen **on the validation split only**.
# With calibrated probabilities the theoretical optimum is to reject when `PD > FP_COST / (FP_COST + FN_COST) = 1/6 ≈ 0.167`.

# %%
def best_cost_threshold(y, p, fn_cost=FN_COST, fp_cost=FP_COST):
    """Exact cost-minimising threshold: reject the k riskiest applicants, choose k minimising cost. Returns (threshold, cost)."""
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    fn = ys.sum() - np.cumsum(ys)                 # defaulters still approved after rejecting top-k
    fp = np.cumsum(1 - ys)                        # good clients rejected
    cost = fn_cost * fn + fp_cost * fp
    k = int(np.argmin(cost))
    if fn_cost * ys.sum() <= cost[k]:             # reject nobody
        return np.inf, fn_cost * ys.sum()
    return float(ps[k]), float(cost[k])


def confusion(y, flag):
    flag = np.asarray(flag).astype(bool); y = np.asarray(y).astype(bool)
    return dict(TP=int((flag & y).sum()), FP=int((flag & ~y).sum()), FN=int((~flag & y).sum()), TN=int((~flag & ~y).sum()))


def evaluate(name, y_va, p_va, y_te, p_te):
    thr, _ = best_cost_threshold(y_va, p_va)
    flag = p_te >= thr
    c = confusion(y_te, flag)
    cost = FN_COST * c["FN"] + FP_COST * c["FP"]
    base_all = FN_COST * y_te.sum(); base_none = FP_COST * (len(y_te) - y_te.sum())
    return dict(model=name, ROC_AUC=roc_auc_score(y_te, p_te), PR_AUC=average_precision_score(y_te, p_te), threshold=thr,
                approval_rate=1 - flag.mean(), default_recall=c["TP"] / max(c["TP"] + c["FN"], 1), precision=c["TP"] / max(c["TP"] + c["FP"], 1),
                FN=c["FN"], FP=c["FP"], cost_per_1000=1000 * cost / len(y_te), saving_vs_approve_all=1 - cost / base_all,
                saving_vs_reject_all=1 - cost / base_none)


# ---- Logistic regression pipelines ----
X_lr = X_full.copy()
for c in CAT_COLS:
    X_lr[c] = X_lr[c].astype(object)


def make_lr(encoder):
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    if encoder == "ohe":
        cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("enc", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005))])
    else:
        cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("enc", TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=SEED)), ("sc", StandardScaler())])
    ct = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)], sparse_threshold=0.3)
    return Pipeline([("prep", ct), ("clf", LogisticRegression(C=0.05, max_iter=400, class_weight="balanced"))])


# ---- LightGBM ----
def fit_lgb(cols, sample_weight=None, scale_pos_weight=1.0, **over):
    params = dict(n_estimators=3000, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.5, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1, scale_pos_weight=scale_pos_weight)
    params.update(over)
    m = lgb.LGBMClassifier(**params)
    m.fit(X_full.iloc[idx_tr][cols], y_tr, sample_weight=sample_weight,
          eval_set=[(X_full.iloc[idx_va][cols], y_va)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def predict(m, idx_, cols):
    return m.predict_proba(X_full.iloc[idx_][cols])[:, 1]


PROBS, MODELS, rows = {}, {}, []
t0 = time.time()
for name, enc in (("LogReg_OHE", "ohe"), ("LogReg_TargetEnc", "te")):
    pipe = make_lr(enc).fit(X_lr.iloc[idx_tr], y_tr)
    PROBS[name] = (pipe.predict_proba(X_lr.iloc[idx_va])[:, 1], pipe.predict_proba(X_lr.iloc[idx_te])[:, 1])
    print(f"{name} done ({time.time()-t0:.0f}s)")
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
for name, w in (("LGBM_plain", 1.0), ("LGBM_spw", spw)):
    m = fit_lgb(ALL_FEATS, scale_pos_weight=w)
    MODELS[name] = (m, w)
    PROBS[name] = (predict(m, idx_va, ALL_FEATS), predict(m, idx_te, ALL_FEATS))
    print(f"{name} done, {m.best_iteration_} trees ({time.time()-t0:.0f}s)")

for name, (pv, pt) in PROBS.items():
    rows.append(evaluate(name, y_va, pv, y_te, pt))
model_table = pd.DataFrame(rows).set_index("model")
save_table(model_table, "partA_model_comparison")
base_rows = pd.DataFrame({"approve everybody": {"cost_per_1000": 1000 * FN_COST * y_te.sum() / len(y_te)},
                          "reject everybody": {"cost_per_1000": 1000 * FP_COST * (len(y_te) - y_te.sum()) / len(y_te)}}).T
print(f"No-skill PR-AUC (base rate): {y_te.mean():.4f}")
display(model_table.round(4)); display(base_rows.round(1))
RESULTS["partA_models"] = model_table.round(5).reset_index().to_dict("records")
RESULTS["partA_trivial_cost_per_1000"] = base_rows["cost_per_1000"].round(2).to_dict()

# %%
# Main model = LightGBM variant with the lowest *validation* cost (selection never looks at test)
val_cost = {n: best_cost_threshold(y_va, PROBS[n][0])[1] for n in ("LGBM_plain", "LGBM_spw")}
MAIN_NAME = min(val_cost, key=val_cost.get)
MAIN_SPW = MODELS[MAIN_NAME][1]
print("validation cost:", {k: round(v) for k, v in val_cost.items()}, "->", MAIN_NAME)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for name, (pv, pt) in PROBS.items():
    fpr, tpr, _ = roc_curve(y_te, pt); ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, pt):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, pt); ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y_te, pt):.3f})")
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC curve (test)", xlabel="FPR", ylabel="TPR"); ax[0].legend(fontsize=8)
ax[1].axhline(y_te.mean(), color="k", ls=":"); ax[1].set(title="Precision-Recall curve (test)", xlabel="Recall", ylabel="Precision"); ax[1].legend(fontsize=8)
# cost as a function of reject rate for the main model
pt_main = PROBS[MAIN_NAME][1]; order = np.argsort(-pt_main); ys = y_te[order]
k = np.arange(len(ys) + 1)
cost_curve = FN_COST * (ys.sum() - np.concatenate([[0], np.cumsum(ys)])) + FP_COST * np.concatenate([[0], np.cumsum(1 - ys)])
ax[2].plot(k / len(ys), 1000 * cost_curve / len(ys)); ax[2].set(title=f"Business cost vs. reject rate ({MAIN_NAME})", xlabel="share of applicants rejected", ylabel="cost per 1,000 applicants")
ax[2].axvline(1 - model_table.loc[MAIN_NAME, "approval_rate"], color="r", ls="--", label="validation-chosen threshold"); ax[2].legend(fontsize=8)
savefig("partA_curves")

# %% [markdown]
# **Reading the table.** Compare `cost_per_1000` with the trivial policies (approve everyone / reject everyone). A good model has to beat both.
# `saving_vs_approve_all` is the share of default-loss avoided relative to the current "no model" world.
# Next: the audit. The main model is called **M0 (baseline)**.

# %% [markdown]
# # PART B — Fairness audit
# For each attribute: approval rate, **Disparate Impact Ratio** = approval rate of group / approval rate of the *most-approved* group (Four-Fifths Rule threshold 0.8),
# and **Equalized Odds** quantities: TPR = share of true defaulters rejected, FPR = share of good clients rejected.
# Bootstrap 95 % intervals on the minimum DIR show whether a failure could be sampling noise.
#
# > **Caution when reading DIR.** When approval rates are high (~85-95 %), the four-fifths rule on *approvals* is very lenient: even a group rejected twice as often can still have DIR > 0.8.
# > I therefore also report `rejection_RR` = group rejection rate / reference-group rejection rate (relative risk of being turned down).

# %%
def group_report(y, flag, groups, min_n=MIN_GROUP_N):
    d = pd.DataFrame({"y": np.asarray(y), "flag": np.asarray(flag).astype(float), "g": pd.Series(groups).astype(object).to_numpy()}).dropna(subset=["g"])
    rows = []
    for g, s in d.groupby("g"):
        pos, neg = s[s.y == 1], s[s.y == 0]
        rows.append(dict(group=g, n=len(s), default_rate=s.y.mean(), approval_rate=1 - s.flag.mean(), rejection_rate=s.flag.mean(),
                         TPR=pos.flag.mean() if len(pos) else np.nan, FPR=neg.flag.mean() if len(neg) else np.nan))
    out = pd.DataFrame(rows).set_index("group")
    ok = out["n"] >= min_n
    ref = out.loc[ok, "approval_rate"].idxmax()
    out["DIR"] = out["approval_rate"] / out.loc[ref, "approval_rate"]
    out["rejection_RR"] = out["rejection_rate"] / max(out.loc[ref, "rejection_rate"], 1e-9)   # relative risk of rejection vs reference
    out["reference"] = out.index == ref
    out["four_fifths_pass"] = out["DIR"] >= 0.8
    out["small_group"] = ~ok
    return out


def bootstrap_min_dir(flag, groups, B=N_BOOT, min_n=MIN_GROUP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(groups).astype(object))
    m = codes >= 0; codes, f = codes[m], np.asarray(flag, dtype=float)[m]
    keep = np.where(np.bincount(codes, minlength=len(uniq)) >= min_n)[0]
    out = []
    for _ in range(B):
        i = rng.integers(0, len(codes), len(codes)); c = codes[i]
        tot = np.bincount(c, minlength=len(uniq)); rej = np.bincount(c, weights=f[i], minlength=len(uniq))
        appr = 1 - rej[keep] / np.maximum(tot[keep], 1)
        out.append(appr.min() / appr.max())
    return tuple(np.percentile(out, [2.5, 97.5]))


def audit_summary(y, flag, aud, boot=False):
    rows = []
    for a in ATTRS:
        rep = group_report(y, flag, aud[a]); ok = rep[~rep.small_group]
        r = dict(attribute=a, reference_group=ok.index[ok.reference][0], worst_group=ok["DIR"].idxmin(), min_DIR=ok["DIR"].min(),
                 passes_80=bool(ok["DIR"].min() >= 0.8), max_rejection_RR=ok["rejection_RR"].max(), TPR_gap=ok["TPR"].max() - ok["TPR"].min(), FPR_gap=ok["FPR"].max() - ok["FPR"].min())
        if boot:
            r["DIR_CI_low"], r["DIR_CI_high"] = bootstrap_min_dir(flag, aud[a])
        rows.append(r)
    return pd.DataFrame(rows).set_index("attribute")


thr_main, _ = best_cost_threshold(y_va, PROBS[MAIN_NAME][0])
flag_main = PROBS[MAIN_NAME][1] >= thr_main
print(f"M0 = {MAIN_NAME}: threshold {thr_main:.4f}, test approval rate {1 - flag_main.mean():.3f}")

audit_M0 = audit_summary(y_te, flag_main, AUD_te, boot=True)
save_table(audit_M0, "partB_audit_summary_M0")
display(audit_M0.round(3))
group_tables = {a: group_report(y_te, flag_main, AUD_te[a]) for a in ATTRS}
for a, t in group_tables.items():
    print(f"\n=== {a} ===")
    display(t.drop(columns=["reference"]).round(3))
    save_table(t, f"partB_groups_M0_{a}")
RESULTS["partB_audit_M0"] = audit_M0.round(4).reset_index().to_dict("records")
RESULTS["partB_groups_M0"] = {a: t.round(4).reset_index().to_dict("records") for a, t in group_tables.items()}

# %%
def plot_fairness(group_tables, title_suffix="", fname="partB_fairness_M0"):
    fig, axes = plt.subplots(2, len(ATTRS), figsize=(4.2 * len(ATTRS), 7.5))
    for j, a in enumerate(ATTRS):
        t = group_tables[a]
        cols = ["#2a9d8f" if p else "#e76f51" for p in t["four_fifths_pass"]]
        axes[0, j].bar(range(len(t)), t["approval_rate"], color=cols)
        axes[0, j].axhline(0.8 * t.loc[t.reference, "approval_rate"].iloc[0], color="k", ls="--", lw=1, label="80% of reference")
        axes[0, j].set_xticks(range(len(t))); axes[0, j].set_xticklabels([str(i)[:16] for i in t.index], rotation=35, ha="right", fontsize=8)
        axes[0, j].set_title(f"{a}\napproval rate" + title_suffix, fontsize=10); axes[0, j].set_ylim(0, 1)
        w = 0.38; x = np.arange(len(t))
        axes[1, j].bar(x - w / 2, t["TPR"], w, label="TPR (defaulters rejected)", color="#264653")
        axes[1, j].bar(x + w / 2, t["FPR"], w, label="FPR (good clients rejected)", color="#e9c46a")
        axes[1, j].set_xticks(x); axes[1, j].set_xticklabels([str(i)[:16] for i in t.index], rotation=35, ha="right", fontsize=8)
        axes[1, j].set_title("Equalized-odds components", fontsize=10)
    axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
    savefig(fname)


plot_fairness(group_tables)

# %% [markdown]
# ## B3. Proxy discrimination & redlining analysis
# Removing `CODE_GENDER` ("fairness through unawareness") does not remove gender information if other columns encode it. Four tests:
#
# 1. **Proxy predictability** – train a model to predict membership of the *most-disadvantaged* group from features that *exclude* the attribute's own defining columns. High AUC ⇒ the remaining features leak the attribute.
# 2. **Ablation** – retrain without `DIRECT` columns (M1) and without `DIRECT + PROXY` columns (M2); compare AUC, cost and DIR.
# 3. **Redlining view of region** – default rate, mean predicted PD, approval, and FPR (creditworthy clients rejected) by region tier; calibration by group.
# 4. **Contribution of sensitive / proxy features by group** (SHAP, Part C).

# %%
DEFINING = {"GENDER": ["CODE_GENDER"], "AGE_GROUP": ["DAYS_BIRTH", "AGE_YEARS", "DAYS_EMPLOYED_PCT"],
            "REGION_TIER": ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"], "INCOME_CATEGORY": ["NAME_INCOME_TYPE", "DAYS_EMPLOYED_ANOM"],
            "HOUSING_GROUP": ["NAME_HOUSING_TYPE"]}
rows, proxy_top = [], {}
sub = np.random.default_rng(SEED).choice(len(X_full), size=min(len(X_full), 80000), replace=False)
for a in ATTRS:
    worst = audit_M0.loc[a, "worst_group"]
    tgt = (AUD[a].iloc[sub] == worst).astype(int).to_numpy()
    cols = [c for c in ALL_FEATS if c not in DEFINING[a]]
    Xs_, ys_ = X_full.iloc[sub][cols], tgt
    i_a, i_b = train_test_split(np.arange(len(sub)), test_size=0.3, stratify=ys_, random_state=SEED)
    pm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, colsample_bytree=0.6, random_state=SEED, n_jobs=-1, verbose=-1, importance_type="gain")
    pm.fit(Xs_.iloc[i_a], ys_[i_a])
    auc = roc_auc_score(ys_[i_b], pm.predict_proba(Xs_.iloc[i_b])[:, 1])
    imp = pd.Series(pm.feature_importances_, index=cols).sort_values(ascending=False)
    proxy_top[a] = (imp / imp.sum()).head(5).round(3).to_dict()
    rows.append(dict(attribute=a, group_predicted=worst, group_share=ys_.mean(), AUC_from_other_features=auc, top_leaking_features=", ".join(list(imp.index[:4]))))
proxy_pred = pd.DataFrame(rows).set_index("attribute")
save_table(proxy_pred, "partB_proxy_predictability"); display(proxy_pred.round(3))
RESULTS["partB_proxy_predictability"] = proxy_pred.round(4).reset_index().to_dict("records")

# %%
# ---- Model variants: M0 baseline, M1 unaware (drop direct), M2 unaware + drop proxies, M3 reweighed (+unaware) ----
def reweighing_weights(y, g):
    """Kamiran & Calders (2012): w(g,y) = P(g) P(y) / P(g,y), makes group and label statistically independent in the training data."""
    d = pd.DataFrame({"y": y, "g": pd.Series(g).astype(object).fillna("NA").to_numpy()})
    n = len(d)
    pg = d.groupby("g")["y"].transform("size") / n
    py = d.groupby("y")["g"].transform("size") / n
    pgy = d.groupby(["g", "y"])["y"].transform("size") / n
    return (pg * py / pgy).to_numpy()


VARIANTS = {}


def register(name, model, cols):
    pv, pt = predict(model, idx_va, cols), predict(model, idx_te, cols)
    thr, _ = best_cost_threshold(y_va, pv)
    VARIANTS[name] = dict(model=model, cols=cols, p_va=pv, p_te=pt, thr=thr, flag_te=pt >= thr)


register("M0_baseline", MODELS[MAIN_NAME][0], ALL_FEATS)
cols_M1 = [c for c in ALL_FEATS if c not in DIRECT]
cols_M2 = [c for c in ALL_FEATS if c not in DIRECT + PROXY]
register("M1_unaware", fit_lgb(cols_M1, scale_pos_weight=MAIN_SPW), cols_M1)
register("M2_no_proxies", fit_lgb(cols_M2, scale_pos_weight=MAIN_SPW), cols_M2)
WORST_ATTR = audit_M0["min_DIR"].idxmin()
w_rw = reweighing_weights(y_tr, AUD_tr[WORST_ATTR]) * np.where(y_tr == 1, MAIN_SPW, 1.0)
print(f"Reweighing on attribute with the lowest baseline DIR: {WORST_ATTR}")
register("M3_reweighed", fit_lgb(cols_M1, sample_weight=w_rw), cols_M1)     # reweighed training, direct attributes removed

rows = []
for n, v in VARIANTS.items():
    ev = evaluate(n, y_va, v["p_va"], y_te, v["p_te"]); s = audit_summary(y_te, v["flag_te"], AUD_te)
    r = dict(variant=n, ROC_AUC=ev["ROC_AUC"], PR_AUC=ev["PR_AUC"], cost_per_1000=ev["cost_per_1000"], approval_rate=ev["approval_rate"])
    for a in ATTRS:
        r[f"minDIR_{a}"] = s.loc[a, "min_DIR"]
    r["max_TPR_gap"] = s["TPR_gap"].max(); r["max_FPR_gap"] = s["FPR_gap"].max(); r["attrs_failing_80"] = int((~s["passes_80"]).sum())
    rows.append(r)
variant_table = pd.DataFrame(rows).set_index("variant")
save_table(variant_table, "partB_variant_comparison"); display(variant_table.round(3))
RESULTS["partB_variants"] = variant_table.round(4).reset_index().to_dict("records")

# %%
# ---- Redlining view: region tier ----
reg = group_tables["REGION_TIER"][["n", "default_rate", "approval_rate", "DIR", "TPR", "FPR"]].copy()
p0 = VARIANTS["M0_baseline"]["p_te"]
reg["mean_predicted_PD"] = pd.Series(p0).groupby(AUD_te["REGION_TIER"].to_numpy()).mean()
reg["calibration_ratio (pred/obs)"] = reg["mean_predicted_PD"] / reg["default_rate"]
save_table(reg, "partB_redlining_region"); display(reg.round(3))

cal = {}
for a in ("GENDER", "REGION_TIER", "HOUSING_GROUP", "AGE_GROUP"):
    d = pd.DataFrame({"pd": p0, "y": y_te, "g": AUD_te[a].to_numpy()}).dropna()
    t = d.groupby("g").agg(n=("y", "size"), mean_PD=("pd", "mean"), observed=("y", "mean")); t["pred/obs"] = t["mean_PD"] / t["observed"]
    cal[a] = t
print("Calibration by group (ratio > 1 = group is over-penalised relative to realised default):")
display(pd.concat(cal).round(3))
save_table(pd.concat(cal), "partB_calibration_by_group")
RESULTS["partB_redlining_region"] = reg.round(4).reset_index().to_dict("records")

# %% [markdown]
# ## B4. Profitability vs. fairness trade-off
# *Post-processing* (group-specific thresholds learned on **validation**): for attribute `A` and interpolation weight `alpha in [0,1]`,
# `threshold_g(alpha) = (1-alpha) * t_global + alpha * t_g`, where `t_g` makes group *g*'s reject rate equal the global reject rate (→ equal approval rates, DIR ≈ 1).
# `alpha = 0` is the cost-optimal baseline; `alpha = 1` is demographic parity. Each curve shows the **price of fairness** in business-cost units.
#
# > **Legal caveat.** Using the protected attribute itself to set thresholds is *disparate treatment* for gender / age / marital status under ECOA / Reg B (US) and similar laws elsewhere.
# > For these attributes the curve quantifies the trade-off for the board but is **not** a deployable remedy; the deployable levers are M1–M3 (removing attributes, proxies, reweighing).
# > Group-aware thresholds on non-protected attributes (e.g. income category) may be lawful; seek counsel.

# %%
def group_thresholds(p_ref, groups_ref, t0):
    r0 = (p_ref >= t0).mean()
    g_arr = pd.Series(groups_ref).astype(object).to_numpy()
    out = {}
    for g in pd.unique(g_arr[~pd.isna(g_arr)]):
        pg = p_ref[g_arr == g]
        out[g] = np.quantile(pg, 1 - r0) if len(pg) >= 20 else t0
    return out


def tradeoff_curve(variant="M0_baseline", alphas=np.linspace(0, 1, 21)):
    v = VARIANTS[variant]; curves = {}
    for a in ATTRS:
        tg = group_thresholds(v["p_va"], AUD_va[a], v["thr"])
        g_te = AUD_te[a].astype(object).to_numpy()
        tg_row = np.array([tg.get(g, v["thr"]) for g in g_te])
        rows = []
        for al in alphas:
            thr_row = (1 - al) * v["thr"] + al * tg_row
            flag = v["p_te"] >= thr_row
            c = confusion(y_te, flag); cost = (FN_COST * c["FN"] + FP_COST * c["FP"]) / len(y_te) * 1000
            rep = group_report(y_te, flag, AUD_te[a]); ok = rep[~rep.small_group]
            rows.append(dict(alpha=al, cost_per_1000=cost, min_DIR=ok["DIR"].min(), approval_rate=1 - flag.mean(),
                             TPR_gap=ok["TPR"].max() - ok["TPR"].min(), FPR_gap=ok["FPR"].max() - ok["FPR"].min()))
        curves[a] = pd.DataFrame(rows)
    return curves


curves = tradeoff_curve()
fig, axes = plt.subplots(1, len(ATTRS), figsize=(4.2 * len(ATTRS), 4.2), sharey=False)
for ax, a in zip(axes, ATTRS):
    cv = curves[a]
    ax.plot(cv["min_DIR"], cv["cost_per_1000"], "-o", ms=3, label="group-threshold post-processing (M0)")
    for n, mk in (("M1_unaware", "s"), ("M2_no_proxies", "^"), ("M3_reweighed", "D")):
        ax.scatter(variant_table.loc[n, f"minDIR_{a}"], variant_table.loc[n, "cost_per_1000"], marker=mk, s=60, label=n, zorder=3)
    ax.scatter(variant_table.loc["M0_baseline", f"minDIR_{a}"], variant_table.loc["M0_baseline", "cost_per_1000"], c="k", s=70, marker="*", label="M0 baseline", zorder=4)
    ax.axvline(0.8, color="r", ls="--", lw=1); ax.set(title=a, xlabel="minimum DIR (test)", ylabel="business cost / 1,000 applicants")
axes[0].legend(fontsize=7)
fig.suptitle("Profitability vs. fairness trade-off (red dashed = four-fifths rule)", y=1.03)
savefig("partB_tradeoff_curves")

summ = []
for a in ATTRS:
    cv = curves[a]; base = cv.iloc[0]
    r = dict(attribute=a, baseline_minDIR=base["min_DIR"], baseline_cost_per_1000=base["cost_per_1000"])
    for tgt in (0.8, 0.9, 0.95):
        okc = cv[cv["min_DIR"] >= tgt]
        r[f"extra_cost_pct_for_DIR>={tgt}"] = 0.0 if base["min_DIR"] >= tgt else (100 * (okc.iloc[0]["cost_per_1000"] / base["cost_per_1000"] - 1) if len(okc) else np.nan)
    summ.append(r)
price_of_fairness = pd.DataFrame(summ).set_index("attribute")
print("Extra business cost (%) of reaching a DIR target via group thresholds (0 = already met, NaN = not reachable; small negatives are test-set noise):")
save_table(price_of_fairness, "partB_price_of_fairness"); display(price_of_fairness.round(3))
RESULTS["partB_price_of_fairness"] = price_of_fairness.round(4).reset_index().to_dict("records")
RESULTS["partB_tradeoff_curves"] = {a: curves[a].round(4).to_dict("records") for a in ATTRS}

# %% [markdown]
# # PART C — Interpretation & Explainable AI
# **Which model is explained?** A lender cannot lawfully use (or cite as a reason) gender, age or marital status. So Part C explains the
# **deployment candidate** = the model *without* direct protected attributes (M1 / M2 / M3) that has the lowest validation business cost.
# The *baseline* is also explained globally, purely as an audit tool: does it lean on sensitive or proxy features?

# %%
def compute_shap(model, X):
    """TreeSHAP values in log-odds space + expected value. Falls back to LightGBM's built-in TreeSHAP if the shap API changes."""
    try:
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(X)
        if isinstance(sv, list): sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3: sv = sv[..., -1]
        base = float(np.ravel(ex.expected_value)[-1])
    except Exception as e:
        print("shap.TreeExplainer failed -> using LightGBM pred_contrib:", e)
        c = model.booster_.predict(X, pred_contrib=True); sv, base = c[:, :-1], float(c[0, -1])
    return sv, base


def plot_frame(X):
    """Numeric frame for SHAP colouring: categorical -> integer codes (-1 -> NaN)."""
    Xp = X.copy()
    for c in Xp.columns:
        if isinstance(Xp[c].dtype, pd.CategoricalDtype):
            codes = Xp[c].cat.codes.astype(float); Xp[c] = codes.where(codes >= 0, np.nan)
    return Xp


deployable = {n: best_cost_threshold(y_va, VARIANTS[n]["p_va"])[1] for n in ("M1_unaware", "M2_no_proxies", "M3_reweighed")}
DEPLOY = min(deployable, key=deployable.get)
D = VARIANTS[DEPLOY]
print("Deployable candidates (validation cost):", {k: round(v) for k, v in deployable.items()}, "->", DEPLOY)
RESULTS["deployment_model"] = DEPLOY

rng = np.random.default_rng(SEED)
samp = rng.choice(len(idx_te), size=min(5000, len(idx_te)), replace=False)      # positions within the test split


def global_shap(variant, fname):
    v = VARIANTS[variant]; Xs = X_full.iloc[idx_te[samp]][v["cols"]]
    sv, base = compute_shap(v["model"], Xs)
    raw = v["model"].predict(Xs, raw_score=True)
    print(f"[{variant}] SHAP additivity check, max |base + sum(phi) - logit| = {np.abs(base + sv.sum(1) - raw).max():.2e}")
    Xp = plot_frame(Xs)
    shap.summary_plot(sv, Xp, max_display=20, show=False); plt.title(f"SHAP summary (beeswarm) — {variant}"); savefig(f"{fname}_beeswarm")
    shap.summary_plot(sv, Xp, plot_type="bar", max_display=20, show=False); plt.title(f"Mean |SHAP| — {variant}"); savefig(f"{fname}_bar")
    imp = pd.Series(np.abs(sv).mean(0), index=v["cols"]).sort_values(ascending=False)
    return sv, base, Xs, imp


sv0, base0, Xs0, imp0 = global_shap("M0_baseline", "partC_shap_M0")
svD, baseD, XsD, impD = global_shap(DEPLOY, "partC_shap_deploy")
imp_table = pd.concat({"M0_baseline": imp0.head(15), DEPLOY: impD.head(15)}, axis=1)
save_table(imp_table, "partC_shap_top15"); display(imp_table.round(4))
RESULTS["partC_shap_top15"] = {"M0_baseline": imp0.head(15).round(5).to_dict(), DEPLOY: impD.head(15).round(5).to_dict()}

# %%
# Audit link back to Part B: how much do sensitive / proxy features push each group's risk score in the BASELINE?
sens_cols = [c for c in DIRECT + PROXY if c in Xs0.columns]
net = pd.Series(sv0[:, [list(Xs0.columns).index(c) for c in sens_cols]].sum(1))
share = np.abs(sv0[:, [list(Xs0.columns).index(c) for c in sens_cols]]).sum() / np.abs(sv0).sum()
print(f"Sensitive + proxy features carry {share:.1%} of the baseline's total |SHAP| mass: {sens_cols}")
rows = []
for a in ATTRS:
    t = net.groupby(AUD_te[a].iloc[samp].to_numpy()).agg(["mean", "size"])
    for g, r in t.iterrows():
        rows.append(dict(attribute=a, group=g, n=int(r["size"]), mean_net_SHAP_sensitive_and_proxy_features=r["mean"]))
shap_by_group = pd.DataFrame(rows).set_index(["attribute", "group"])
save_table(shap_by_group, "partC_shap_sensitive_by_group"); display(shap_by_group.round(4))
RESULTS["partC_sensitive_shap_share"] = float(share)
RESULTS["partC_sensitive_shap_by_group"] = shap_by_group.round(5).reset_index().to_dict("records")

# %% [markdown]
# ## C2. Local explanations for borderline applicants (deployment model)
# Borderline = predicted PD closest to the decision threshold, so a small change flips the decision. Waterfall plots show how each feature moves the log-odds of default from the portfolio average.

# %%
p_te_D, thrD = D["p_te"], D["thr"]
rej_pos = np.where(p_te_D >= thrD)[0]; app_pos = np.where(p_te_D < thrD)[0]
border = list(rej_pos[np.argsort(p_te_D[rej_pos])][:2]) + list(app_pos[np.argsort(-p_te_D[app_pos])][:1])
labels = ["REJECTED (just above threshold)", "REJECTED (2nd closest)", "APPROVED (just below threshold)"]
for k, (pos, lab) in enumerate(zip(border, labels)):
    row = X_full.iloc[[idx_te[pos]]][D["cols"]]
    sv_i, b_i = compute_shap(D["model"], row)
    disp = np.array([("" if pd.isna(v) else (f"{v:,.3g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v))) for v in row.iloc[0].tolist()], dtype=object)
    try:
        shap.plots.waterfall(shap.Explanation(values=sv_i[0], base_values=b_i, data=disp, feature_names=D["cols"]), max_display=12, show=False)
    except Exception:
        shap.plots.waterfall(shap.Explanation(values=sv_i[0], base_values=b_i, data=plot_frame(row).iloc[0].to_numpy(), feature_names=D["cols"]), max_display=12, show=False)
    plt.title(f"{lab}: PD = {p_te_D[pos]:.3f} (threshold {thrD:.3f})", fontsize=10); savefig(f"partC_local_{k+1}")

# %% [markdown]
# ## C3. Counterfactual analysis — what is the *minimum* change that flips a rejection?
# Only **non-sensitive, actionable** levers are changed; income, age, gender, region etc. are held fixed:
#
# | Lever | Change | Recomputed features |
# |---|---|---|
# | Smaller loan | cut `AMT_CREDIT`, `AMT_GOODS_PRICE` and `AMT_ANNUITY` by *r* (same term) | credit/income, annuity/income, … |
# | Longer term | additionally cut the instalment `AMT_ANNUITY` by *s* (→ more months) | term, annuity/income |
# | Lower utilisation | pay down bureau debt by *u* (only if bureau data is used) | `BUREAU_DEBT_RATIO` |
#
# The search is an exhaustive grid over (*r*, *s*, *u*); "effort" = *r* + *s* + *u* (sum of fractional changes). For each rejected applicant I report the cheapest combination whose PD falls below the threshold, plus the best **single-lever** (loan-size-only) fix.
# Grids are capped (loan cut ≤ 80 %) – an applicant needing more is reported as *no feasible recourse*.

# %%
LOAN_CUTS = np.round(np.arange(0, 0.801, 0.05), 3)
TERM_CUTS = np.array([0, .1, .2, .3, .4])
UTIL_CUTS = np.array([0, .2, .4, .6, .8]) if "BUREAU_DEBT_SUM" in ALL_FEATS else np.array([0.0])


def counterfactual_search(rows, model, cols, thr):
    combos = np.array(list(itertools.product(LOAN_CUTS, TERM_CUTS, UTIL_CUTS)))
    G = len(combos)
    rep = rows.loc[rows.index.repeat(G)].reset_index(drop=True)
    app_id = np.repeat(rows.index.to_numpy(), G)
    r, s, u = (np.tile(combos[:, j], len(rows)) for j in range(3))
    rep["AMT_CREDIT"] = rep["AMT_CREDIT"].to_numpy() * (1 - r)
    rep["AMT_GOODS_PRICE"] = rep["AMT_GOODS_PRICE"].to_numpy() * (1 - r)
    rep["AMT_ANNUITY"] = rep["AMT_ANNUITY"].to_numpy() * (1 - r) * (1 - s)
    if "BUREAU_DEBT_SUM" in rep:
        rep["BUREAU_DEBT_SUM"] = rep["BUREAU_DEBT_SUM"].to_numpy() * (1 - u)
    rep = add_ratios(rep)
    for c in CAT_COLS:                                           # keep category levels intact after copies
        rep[c] = pd.Categorical(rep[c].astype(object), categories=CATEGORIES[c])
    pnew = model.predict_proba(rep[cols])[:, 1]
    res = pd.DataFrame({"app_id": app_id, "loan_cut": r, "term_cut": s, "util_cut": u, "pd_new": pnew,
                        "credit_new": rep["AMT_CREDIT"].to_numpy(), "annuity_new": rep["AMT_ANNUITY"].to_numpy()})
    res["effort"] = res["loan_cut"] + res["term_cut"] + res["util_cut"]
    res["flipped"] = res["pd_new"] < thr
    best = res[res.flipped].sort_values(["app_id", "effort", "loan_cut"]).groupby("app_id").head(1).set_index("app_id")
    single = res[res.flipped & (res.term_cut == 0) & (res.util_cut == 0)].sort_values(["app_id", "loan_cut"]).groupby("app_id").head(1).set_index("app_id")
    return best, single


N_CF = min(150, len(rej_pos))
cf_pos = np.random.default_rng(SEED).choice(rej_pos, size=N_CF, replace=False)
cf_rows = X_full.iloc[idx_te[cf_pos]]
cf_best, cf_single = counterfactual_search(cf_rows, D["model"], D["cols"], thrD)
cf = pd.DataFrame(index=cf_rows.index)
cf["pd_orig"] = p_te_D[cf_pos]; cf["credit_orig"] = cf_rows["AMT_CREDIT"].to_numpy()
cf = cf.join(cf_best[["loan_cut", "term_cut", "util_cut", "effort", "pd_new", "credit_new"]]).join(cf_single["loan_cut"].rename("loan_cut_only"))
cf["feasible"] = cf["effort"].notna(); cf["feasible_loan_only"] = cf["loan_cut_only"].notna()
for a in ("GENDER", "REGION_TIER", "AGE_GROUP", "INCOME_CATEGORY"):
    cf[a] = AUD_te[a].iloc[cf_pos].to_numpy()
save_table(cf, "partC_counterfactuals")

cf_summary = pd.Series({
    "rejected_profiles_tested": N_CF,
    "flippable_with_any_lever_combination": cf["feasible"].mean(),
    "flippable_by_loan_cut_alone": cf["feasible_loan_only"].mean(),
    "median_loan_cut_alone_(feasible)": cf["loan_cut_only"].median(),
    "median_loan_cut_in_best_combo": cf["loan_cut"].median(),
    "share_needing_term_extension": (cf["term_cut"] > 0).sum() / max(cf["feasible"].sum(), 1),
    "share_needing_utilisation_paydown": (cf["util_cut"] > 0).sum() / max(cf["feasible"].sum(), 1),
    "median_effort": cf["effort"].median()})
display(cf_summary.round(3).to_frame("value"))
RESULTS["partC_counterfactual_summary"] = {k: (None if pd.isna(v) else float(v)) for k, v in cf_summary.items()}

# Recourse fairness: is the *burden* of reversing a rejection equal across groups?
rec = pd.concat({a: cf.groupby(a).agg(n=("feasible", "size"), feasible_rate=("feasible", "mean"), median_loan_cut=("loan_cut", "median"), median_effort=("effort", "median")) for a in ("GENDER", "REGION_TIER", "AGE_GROUP", "INCOME_CATEGORY")})
save_table(rec, "partC_recourse_by_group"); display(rec.round(3))
RESULTS["partC_recourse_by_group"] = rec.round(4).reset_index().to_dict("records")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
cf["loan_cut_only"].dropna().mul(100).plot.hist(bins=16, ax=ax[0], color="#264653"); ax[0].set(title="Loan-size cut that flips rejection (loan lever only)", xlabel="% reduction in requested amount")
cf["effort"].dropna().plot.hist(bins=16, ax=ax[1], color="#e76f51"); ax[1].set(title="Effort of cheapest combined counterfactual", xlabel="sum of fractional changes")
savefig("partC_counterfactual_hist")

# %% [markdown]
# ## C4. Regulatory output — automated Adverse Action Notice
# The notice lists the **top 3 actionable reasons** taken from the applicant's own SHAP values (largest positive push towards default).
# Only features in `REASONS` can be cited; direct protected attributes and geographic / occupational proxies are **never** cited. If any such feature is among the applicant's
# largest drivers it is logged as an internal compliance flag (that would indicate the deployed model should not use it).
# Reasons that share a headline (e.g. the three external scores) are merged. The counterfactual result is added as "what could change this outcome".
# *This is a template; wording, lender identity, CRA disclosures and timeframes require legal / compliance review.*

# %%
APPROVED_MED = X_full.iloc[idx_te[app_pos]].median(numeric_only=True)
fmt_money = lambda v: f"{v:,.0f}"
REASONS = {}
def _reg(features, headline, detail):
    for f in features:
        REASONS[f] = (headline, detail)

_reg(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN"], "External credit score is below the level we require",
     lambda r: f"Your combined external score is {r['EXT_SOURCE_MEAN']:.2f}; applicants we approve have a median of {APPROVED_MED['EXT_SOURCE_MEAN']:.2f}. It can improve with on-time payments across your credit accounts.")
_reg(["CREDIT_INCOME_PCT", "AMT_CREDIT", "AMT_GOODS_PRICE"], "Requested loan amount is high relative to income",
     lambda r: f"You requested {fmt_money(r['AMT_CREDIT'])}, which is {r['CREDIT_INCOME_PCT']:.1f}x your annual income (median for approved applicants: {APPROVED_MED['CREDIT_INCOME_PCT']:.1f}x).")
_reg(["ANNUITY_INCOME_PCT", "AMT_ANNUITY", "CREDIT_TERM_MONTHS"], "Repayment instalment is high relative to income",
     lambda r: f"The scheduled annual repayment is {r['ANNUITY_INCOME_PCT']:.1%} of your stated income (median for approved applicants: {APPROVED_MED['ANNUITY_INCOME_PCT']:.1%}).")
_reg(["GOODS_CREDIT_PCT"], "Little or no down payment on the purchase",
     lambda r: f"The loan finances {r['GOODS_CREDIT_PCT']:.0%} of the goods price; a larger down payment lowers risk.")
_reg(["AMT_INCOME_TOTAL", "INCOME_PER_PERSON"], "Stated income is low relative to the credit requested",
     lambda r: f"Stated annual income: {fmt_money(r['AMT_INCOME_TOTAL'])}.")
_reg(["DAYS_EMPLOYED", "DAYS_EMPLOYED_ANOM"], "Short or no recorded length of current employment",
     lambda r: ("No current employment recorded." if pd.isna(r["DAYS_EMPLOYED"]) else f"Recorded time in current employment: {-r['DAYS_EMPLOYED'] / 365.25:.1f} years."))
_reg(["BUREAU_DEBT_RATIO", "BUREAU_DEBT_SUM"], "High utilisation of existing credit",
     lambda r: f"About {r['BUREAU_DEBT_RATIO']:.0%} of your existing credit balances are still outstanding. Paying these down improves your profile.")
_reg(["BUREAU_OVERDUE_SUM", "BUREAU_MAX_DAY_OVERDUE"], "Overdue amounts or late payments on other credit accounts",
     lambda r: f"Overdue balance recorded on other accounts: {fmt_money(r['BUREAU_OVERDUE_SUM'])}.")
_reg(["BUREAU_ACTIVE_COUNT", "BUREAU_COUNT"], "Number of open credit accounts / limited credit history",
     lambda r: f"Open credit accounts: {int(r['BUREAU_ACTIVE_COUNT'])}; accounts on file: {int(r['BUREAU_COUNT'])}.")
_reg([c for c in ALL_FEATS if c.startswith("AMT_REQ_CREDIT_BUREAU")], "Number of recent credit enquiries",
     lambda r: "Several recent applications for credit were recorded; fewer enquiries over the next months can help.")
FORBIDDEN = set(DIRECT + PROXY)
assert not (FORBIDDEN & set(REASONS)), "a protected attribute or proxy is citable as a reason"


def adverse_action_notice(pos, lender="[LENDER NAME]", top_k=3):
    """pos = position inside the TEST split. Returns (notice_text, compliance_log)."""
    row = X_full.iloc[[idx_te[pos]]]
    sv_i, _ = compute_shap(D["model"], row[D["cols"]])
    contrib = pd.Series(sv_i[0], index=D["cols"]).sort_values(ascending=False)
    contrib = contrib[contrib > 0]
    flagged = [f for f in contrib.index[:5] if f in FORBIDDEN]
    reasons, seen = [], set()
    for f in contrib.index:
        if f not in REASONS: continue
        head, det = REASONS[f]
        if head in seen: continue
        seen.add(head)
        try: detail = det(row.iloc[0])
        except Exception: detail = ""
        reasons.append((head, detail, f, float(contrib[f])))
        if len(reasons) == top_k: break
    r0 = row.iloc[0]
    sug = ""
    cfr = cf_best.loc[row.index[0]] if row.index[0] in cf_best.index else None
    if cfr is None:                                              # compute on the fly for this applicant
        b, _s = counterfactual_search(row, D["model"], D["cols"], thrD)
        cfr = b.iloc[0] if len(b) else None
    if cfr is not None:
        parts = []
        if cfr["loan_cut"] > 0: parts.append(f"reducing the requested amount by about {cfr['loan_cut']:.0%} (to roughly {fmt_money(cfr['credit_new'])})")
        if cfr["term_cut"] > 0: parts.append(f"spreading repayments over a longer term (instalment about {cfr['term_cut']:.0%} lower)")
        if cfr["util_cut"] > 0: parts.append(f"reducing your outstanding balances on other credit by about {cfr['util_cut']:.0%}")
        sug = ("Our analysis indicates that " + " and ".join(parts) + " would likely have resulted in approval. This is indicative only and not a commitment to lend.") if parts \
            else "Our analysis indicates a small change in your application could result in approval."
    else:
        sug = "No modification of the loan amount or term within the ranges we tested would have changed this decision; improving the factors above over time may help."
    L = [f"ADVERSE ACTION NOTICE", f"{lender}", f"Date: {date.today():%d %B %Y}",
         f"Application reference: {df_all['SK_ID_CURR'].iloc[idx_te[pos]]}", "",
         "Decision: your application for credit was NOT APPROVED at this time.",
         f"Amount requested: {fmt_money(r0['AMT_CREDIT'])}   |   Annual income stated: {fmt_money(r0['AMT_INCOME_TOTAL'])}", "",
         "PRINCIPAL REASONS (the factors that most increased the assessed risk of non-repayment):"]
    for i, (h, d, f, v) in enumerate(reasons, 1):
        L += [textwrap.fill(f"{i}. {h}.", 92, subsequent_indent="   ")] + ([textwrap.fill(d, 92, initial_indent="   ", subsequent_indent="   ")] if d else [])
    L += ["", "WHAT COULD CHANGE THIS OUTCOME:", textwrap.fill(sug, 92), "",
          "YOUR RIGHTS:",
          textwrap.fill("This decision was made with the help of an automated scoring model. You may request a review by a member of our credit team and may provide additional information. "
                        "If information from a consumer reporting agency was used, you have the right to obtain a free copy of your report within 60 days and to dispute its accuracy: [CRA NAME / ADDRESS / PHONE]. "
                        "Our decision is based only on the credit-related factors listed above. We do not consider sex, age, marital status, race, religion, national origin, or receipt of public assistance in deciding whether to extend credit. "
                        "[ECOA / REGULATOR STATEMENT AND ADDRESS - REQUIRES LEGAL REVIEW]", 92), "",
          f"Contact: [PHONE / EMAIL / ADDRESS]"]
    log = dict(position=int(pos), pd=float(p_te_D[pos]), threshold=float(thrD), top_reason_features=[r[2] for r in reasons], forbidden_in_top5_drivers=flagged)
    return "\n".join(L), log


feas = cf[cf["feasible"]]
if len(feas):                                                    # demo applicant: feasible recourse with median effort
    k_idx = (feas["effort"] - feas["effort"].median()).abs().idxmin()
    notice_pos = int(cf_pos[list(cf.index).index(k_idx)])
else:
    notice_pos = int(border[0])
notice, log = adverse_action_notice(notice_pos)
print(notice); print("\nINTERNAL COMPLIANCE LOG:", log)
(OUT / "adverse_action_notice_example.txt").write_text(notice)
RESULTS["partC_notice_log"] = log

# %% [markdown]
# ## D. Wrap-up: score `application_test.csv` (optional) and export summary

# %%
if app_test_path is not None and Path(app_test_path).exists():
    at = pd.read_csv(app_test_path)
    built = build_features(at, BUREAU_AGG)
    Xt = built.drop(columns=["SK_ID_CURR"])
    for c in CAT_COLS:
        Xt[c] = pd.Categorical(Xt[c].astype(object), categories=CATEGORIES[c])
    Xt = Xt.reindex(columns=ALL_FEATS)
    pdt = D["model"].predict_proba(Xt[D["cols"]])[:, 1]
    scored = pd.DataFrame({"SK_ID_CURR": at["SK_ID_CURR"], "PD": pdt, "DECISION": np.where(pdt >= thrD, "REJECT", "APPROVE")})
    scored.to_csv(OUT / "application_test_scored.csv", index=False)
    print(f"Scored {len(scored):,} unlabelled applications with {DEPLOY}; reject rate {np.mean(pdt >= thrD):.3f}")
else:
    print("application_test.csv not available (or synthetic mode) - skipping.")

summary_rows = []
for n, v in VARIANTS.items():
    s = audit_summary(y_te, v["flag_te"], AUD_te)
    summary_rows.append(dict(variant=n, attrs_failing_80=int((~s["passes_80"]).sum()), worst_min_DIR=s["min_DIR"].min()))
RESULTS["final_variant_summary"] = summary_rows
RESULTS["config"] = dict(FN_COST=FN_COST, FP_COST=FP_COST, MIN_GROUP_N=MIN_GROUP_N, main_model=MAIN_NAME, threshold_M0=float(thr_main),
                         deployment_threshold=float(thrD), split="60/20/20 stratified", seed=SEED, n_test=int(len(idx_te)))


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


(OUT / "results_summary.json").write_text(json.dumps(RESULTS, indent=2, default=_json_default))
print("Saved: outputs/results_summary.json, outputs/tables/*.csv, outputs/figures/*.png")
print("DATA SOURCE:", DATA_SOURCE.upper(), "" if DATA_SOURCE == "real" else "  <-- synthetic: numbers are NOT findings")
