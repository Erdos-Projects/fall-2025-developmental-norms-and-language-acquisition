# %% [markdown]
# # Logistic Regression Runner (section-by-section)
# 
# 1) Set paths & params  
# 2) Load data  
# 3) Define preprocessing & utilities  
# 4) (Optional) Cross-validation on TRAIN only  
# 5) Train + evaluate **Baseline**  
# 6) Train + evaluate **Enhanced**  
# 7) Save artifacts  
# 

# %%
# === 1) Imports & parameters ===
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, GroupKFold, GridSearchCV
import joblib

import warnings, re
# Silence the setuptools deprecation noise emitted when sklearn/joblib spawn workers
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
)

# ---- Paths: EDIT THESE ----
TRAIN_PATH = Path('data/processed/train_subset_l1-en_l2-es_plus.parquet')
DEV_PATH   = Path('data/processed/dev_subset_l1-en_l2-es_plus.parquet')
TEST_PATH  = Path('data/processed/test_subset_l1-en_l2-es_plus.parquet')

# ---- Output directory ----
OUTDIR = Path('./outputs_notebook')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- Modeling params ----
TARGET_PRECISION = 0.30
USE_CV = True      # set True to enable CV on TRAIN
CV_SPLITS = 5
GROUP_COL = None    # e.g., 'user' for GroupKFold (set to None to use StratifiedKFold)

# ---- Feature configuration ----
LABEL_COL = "token_wrong"

# Baseline
BASE_CAT_COLS = ["token_pos", "token_morph", "token_edges"]  # try both with & without token
BASE_NUM_COLS = ["time", "days"]  



# Enhanced: add category (cat) + extra nums
ENH_CAT_COLS = BASE_CAT_COLS + ["category"]
ENH_NUM_COLS = BASE_NUM_COLS + ["median_aoa", "levdistfrac", "src_freq_lm", "dest_freq_lm"]


print('Params set. Edit paths and flags above as needed.')

# %%
# All candidate categorical columns (strings / enums / IDs)
CAT_COLS_ALL = [
    "token",            # surface form (high-card; try with/without)
    "token_pos",
    "token_morph",
    "token_dep_label",
    "token_edges",
    "countries",
    "category",
    "uni_lemma",
    "l1",
    "l2",
    "client",           # usually exclude (meta)
    "session",          # usually exclude (meta)
    "format",           # usually exclude (meta)
    "prompt",           # often noisy; usually exclude
    "clean_translation",# possible leakage; usually exclude
    "user",             # for GroupKFold only; not as a feature
    "token_id"          # identifier; usually exclude
]

# All candidate numeric columns (continuous / counts)
NUM_COLS_ALL = [
    "time",
    "days",
    "growth_rate",
    "median_aoa",
    "levdist",
    "levdistfrac",
    "aoa_lookup",
    "src_freq_lm",
    "dest_freq_lm",
    "word_count",
    "word_freq",
    "token_count",
    "token_freq",
    "block_id"          # identifier-like; usually exclude
    # NOTE: "token_wrong" is the LABEL; do not include as a feature
]


# %%
# === 2) Load data ===
def _read_parquet(p: str | Path) -> pd.DataFrame:
    p = str(p)
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return pd.read_parquet(p, engine="fastparquet")

df_train = _read_parquet(TRAIN_PATH)
df_dev   = _read_parquet(DEV_PATH)
df_test  = _read_parquet(TEST_PATH)

print('Shapes -> TRAIN:', df_train.shape, '| DEV:', df_dev.shape, '| TEST:', df_test.shape)
df_train.head()

# %%
# === 3) Utilities ===

def ensure_columns_and_cast(frames, cat_cols, label_col: str, fill_missing=True):
    for frame in frames:
        if fill_missing:
            for c in cat_cols:
                if c not in frame.columns:
                    frame[c] = pd.Series(pd.NA, index=frame.index)
        for c in (c for c in cat_cols if c in frame.columns):
            frame[c] = frame[c].astype("string")
        if label_col in frame.columns:
            frame[label_col] = frame[label_col].astype(int)

def intersect_or_warn(df: pd.DataFrame, cols: list[str], tag: str) -> list[str]:
    have = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in have]
    if missing:
        print(f"[{tag}] warn: missing columns ignored: {missing}")
    return have

from sklearn.preprocessing import FunctionTransformer

# Put this near your other constants:
LOG1P_COLS = ["time", "days"]  # only these get log1p

def build_pipeline(num_cols_present: list[str], cat_cols_present: list[str], random_state: int = 42) -> Pipeline:
    # Route numeric columns: some get log1p, the rest go linear
    log_cols = [c for c in LOG1P_COLS if c in num_cols_present]
    lin_cols = [c for c in num_cols_present if c not in log_cols]

    num_log = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),  # safe for zeros: log(1+x)
        ("scaler", StandardScaler()),
    ])
    num_lin = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer([
        ("num_log",  num_log,  log_cols),  # time/days → log1p → scale
        ("num_lin",  num_lin,  lin_cols),  # other numerics (if any) → scale
        ("cat",      cat_pipeline, cat_cols_present),
    ], sparse_threshold=1.0)  # keep sparse

    clf = LogisticRegression(
        solver="saga", penalty="l2", C=0.2, max_iter=2000, tol=2e-3,
        class_weight="balanced", random_state=random_state, n_jobs=1
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def fit_with_optional_cv(pipe: Pipeline, X_tr, y_tr, use_cv: bool, cv_splits: int, groups: pd.Series | None):
    if not use_cv:
        pipe.fit(X_tr, y_tr)
        return pipe, None
    param_grid = [
    {"clf__penalty": ["l2"], "clf__C": [0.05, 0.2]},                 # 2 points
    {"clf__penalty": ["elasticnet"], "clf__l1_ratio": [0.2], "clf__C": [0.1]},  # 1 point
]
    scorer = "average_precision"
    if groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
        gs = GridSearchCV(pipe, param_grid=param_grid, scoring=scorer, cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe, param_grid=param_grid, scoring=scorer, cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
    print(f"[CV] best params: {gs.best_params_}")
    print(f"[CV] best {scorer}: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs

def report_convergence(pipe):
    clf = pipe.named_steps["clf"]
    # n_iter_ is an array (one per class); if any hit max_iter, treat as non-converged
    iters = np.atleast_1d(clf.n_iter_)
    hit_cap = np.any(iters >= clf.max_iter)
    status = "NOT converged" if hit_cap else "converged"
    print(f"[logreg] {status}: iterations per class = {iters.tolist()} / max_iter={clf.max_iter}, tol={clf.tol}")

def choose_threshold_at_precision(y_true: np.ndarray, p: np.ndarray, target_precision: float) -> float:
    prec, rec, thresh = precision_recall_curve(y_true, p)
    # indices of thresholds where precision meets/ exceeds target
    meets = np.where(prec[:-1] >= target_precision)[0]  # align with thresh length
    if len(meets) == 0:
        print(f"[DEV] warn: could not reach precision {target_precision:.0%}; using 0.50")
        return 0.50
    # choose the threshold that gives the highest recall among those that meet precision
    best_idx = meets[np.argmax(rec[meets])]
    return float(thresh[best_idx])


def extract_feature_names(pipe: Pipeline, num_cols_present: list[str], cat_cols_present: list[str]) -> np.ndarray:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    if len(cat_cols_present):
        ohe: OneHotEncoder = pre.named_transformers_["cat"]["ohe"]
        cat_names = ohe.get_feature_names_out(cat_cols_present)
    else:
        cat_names = np.array([])
    num_names = np.array(num_cols_present)
    return np.concatenate([num_names, cat_names])

# Cast categoricals & label
ensure_columns_and_cast((df_train, df_dev, df_test), ENH_CAT_COLS, LABEL_COL, fill_missing=True)
print("Utilities ready; categoricals cast to string and label to int.")

# %%
# Baseline checks
need = set(BASE_CAT_COLS) | set(BASE_NUM_COLS) | {LABEL_COL}
assert not (need - set(df_train.columns)), f"Missing in TRAIN: {need - set(df_train.columns)}"
ensure_columns_and_cast((df_train, df_dev, df_test), BASE_CAT_COLS, LABEL_COL, fill_missing=False)

# Enhanced checks
need_enh = set(ENH_CAT_COLS) | set(ENH_NUM_COLS) | {LABEL_COL}
assert not (need_enh - set(df_train.columns)), f"Missing in TRAIN: {need_enh - set(df_train.columns)}"
ensure_columns_and_cast((df_train, df_dev, df_test), ENH_CAT_COLS, LABEL_COL, fill_missing=False)

# %%
# === 4) ===
def run_once(run_name: str,
             df_train, df_dev, df_test, label_col,
             num_cols, cat_cols,
             outdir: Path,
             target_precision: float,
             use_cv: bool, cv_splits: int,
             group_col: str | None,
             evaluate_test: bool = True):

    # --- local helper: robust feature names from fitted pipeline ---
    def _get_feature_names(pipe: Pipeline) -> np.ndarray:
        pre: ColumnTransformer = pipe.named_steps["pre"]
        # try native if available
        try:
            return np.asarray(pre.get_feature_names_out())
        except Exception:
            names = []
            for name, trans, cols in pre.transformers_:
                if name in ("remainder",) or trans == "drop":
                    continue
                # categorical: OHE names
                if name == "cat":
                    try:
                        ohe: OneHotEncoder = trans.named_steps["ohe"]
                        names.extend(ohe.get_feature_names_out(cols))
                        continue
                    except Exception:
                        pass
                # numeric branches commonly used in our utils
                if name == "num_log":
                    for c in cols: names.append(f"{c}__log1p")
                    continue
                if name == "num_lin":
                    for c in cols: names.append(str(c))
                    continue
                if name == "num_bins":
                    # KBinsDiscretizer(n_bins=5) assumed
                    for c in cols:
                        for k in range(5):
                            names.append(f"{c}__bin_{k}")
                    continue
                # fallback: just echo raw column names
                for c in cols: names.append(str(c))
            return np.asarray(names)

    # --- column presence checks / splits ---
    present_num = intersect_or_warn(df_train, num_cols, f"{run_name}/TRAIN(num)")
    present_cat = intersect_or_warn(df_train, cat_cols, f"{run_name}/TRAIN(cat)")
    needed = present_num + present_cat + [label_col]
    have = intersect_or_warn(df_train, needed, f"{run_name}/TRAIN(cols)")
    if label_col not in have:
        raise ValueError(f"[{run_name}] label column '{label_col}' missing from TRAIN!")

    train_df = df_train[have].copy()
    dev_df   = df_dev[[c for c in have if c in df_dev.columns]].copy()
    test_df  = df_test[[c for c in have if c in df_test.columns]].copy()

    X_tr, y_tr = train_df.drop(columns=[label_col]), train_df[label_col]
    X_dev, y_dev = dev_df.drop(columns=[label_col]), dev_df[label_col]
    X_te, y_te = test_df.drop(columns=[label_col]), test_df[label_col]

    # --- build & fit ---
    pipe = build_pipeline(present_num, present_cat, random_state=42)

    groups = None
    if group_col is not None and group_col in df_train.columns:
        groups = df_train[group_col]
        print(f"[{run_name}] Using GroupKFold with group_col='{group_col}'")
    elif group_col is not None:
        print(f"[{run_name}] warn: group_col '{group_col}' not found; using StratifiedKFold")

    pipe, gs = fit_with_optional_cv(pipe, X_tr, y_tr, use_cv, cv_splits, groups)
    report_convergence(pipe)

    # --- DEV threshold selection ---
    p_dev = pipe.predict_proba(X_dev)[:, 1]
    dev_auc = roc_auc_score(y_dev, p_dev)
    dev_ap = average_precision_score(y_dev, p_dev)
    thresh = choose_threshold_at_precision(y_dev.values, p_dev, target_precision)
    dev_pred = (p_dev > thresh).astype(int)
    dev_report = classification_report(y_dev, dev_pred, output_dict=True)
    dev_cm = confusion_matrix(y_dev, dev_pred).tolist()

    # --- save DEV artifacts ---
    outdir.mkdir(parents=True, exist_ok=True)
    dev_metrics = {
        "run": run_name,
        "target_precision": target_precision,
        "dev": {
            "auc": float(dev_auc),
            "average_precision": float(dev_ap),
            "threshold": float(thresh),
            "classification_report": dev_report,
            "confusion_matrix": dev_cm,
        }
    }
    (outdir / f"{run_name}_metrics_DEV.json").write_text(json.dumps(dev_metrics, indent=2))
    joblib.dump(pipe, outdir / f"{run_name}_pipeline.joblib")
    (outdir / f"{run_name}_threshold.json").write_text(json.dumps({"threshold": float(thresh)}, indent=2))

    # --- coefficients (robust to transforms; length-safe) ---
    coef = pipe.named_steps["clf"].coef_.ravel()
    feature_names = _get_feature_names(pipe)
    if len(feature_names) != len(coef):
        print(f"[warn] feature_names ({len(feature_names)}) != coef length ({len(coef)}). Using generic names.")
        feature_names = np.asarray([f"f_{i}" for i in range(len(coef))])

    coef_df = (
        pd.DataFrame({"feature": feature_names, "coef": coef})
          .assign(abs_coef=lambda d: d["coef"].abs())
          .sort_values("abs_coef", ascending=False)
          .drop(columns="abs_coef")
    )
    coef_df.to_csv(outdir / f"{run_name}_coefficients.csv", index=False)

    # --- pretty DEV summary ---
    print("\n" + "="*72)
    print(f"{run_name.upper()} — DEV ONLY")
    print(f"AUC: {dev_auc:.4f} | AP: {dev_ap:.4f} | chosen threshold: {thresh:.4f}")
    print(pd.DataFrame(dev_report).round(3).to_string())
    print("Confusion Matrix [DEV] (rows=true, cols=pred):", dev_cm)
    print("="*72 + "\n")

    if not evaluate_test:
        return {"run": run_name, "pipe": pipe, "threshold": float(thresh), "X_te": X_te, "y_te": y_te, "outdir": outdir}

    # --- TEST evaluation (optional) ---
    p_te = pipe.predict_proba(X_te)[:, 1]
    te_auc = roc_auc_score(y_te, p_te)
    te_ap  = average_precision_score(y_te, p_te)
    te_pred = (p_te > thresh).astype(int)
    te_report = classification_report(y_te, te_pred, output_dict=True)
    te_cm = confusion_matrix(y_te, te_pred).tolist()

    full_metrics = {
        **dev_metrics,
        "test": {
            "auc": float(te_auc),
            "average_precision": float(te_ap),
            "threshold_used": float(thresh),
            "classification_report": te_report,
            "confusion_matrix": te_cm,
        }
    }
    (outdir / f"{run_name}_metrics.json").write_text(json.dumps(full_metrics, indent=2))

    print("-"*72)
    print(f"{run_name.upper()} — TEST")
    print(f"AUC: {te_auc:.4f} | AP: {te_ap:.4f} | threshold_used: {thresh:.4f}")
    print(pd.DataFrame(te_report).round(3).to_string())
    print("Confusion Matrix [TEST] (rows=true, cols=pred):", te_cm)
    print("="*72 + "\n")

    return {"run": run_name, "pipe": pipe, "threshold": float(thresh), "X_te": X_te, "y_te": y_te, "outdir": outdir}


# %%
# === Evaluate TEST later ===
def evaluate_test_only(artifacts: dict):
    run_name = artifacts["run"]
    pipe = artifacts["pipe"]
    thresh = artifacts["threshold"]
    X_te, y_te = artifacts["X_te"], artifacts["y_te"]
    outdir = artifacts["outdir"]

    p_te = pipe.predict_proba(X_te)[:, 1]
    te_auc = roc_auc_score(y_te, p_te)
    te_ap  = average_precision_score(y_te, p_te)
    te_pred = (p_te > thresh).astype(int)
    te_report = classification_report(y_te, te_pred, output_dict=True)
    te_cm = confusion_matrix(y_te, te_pred).tolist()

    # Save TEST-only metrics
    test_metrics = {
        "run": run_name,
        "test": {
            "auc": float(te_auc),
            "average_precision": float(te_ap),
            "threshold_used": float(thresh),
            "classification_report": te_report,
            "confusion_matrix": te_cm,
        }
    }
    (outdir / f"{run_name}_metrics_TEST.json").write_text(json.dumps(test_metrics, indent=2))

    print("\n" + "-"*72)
    print(f"{run_name.upper()} — TEST (run later)")
    print(f"AUC: {te_auc:.4f} | AP: {te_ap:.4f} | threshold_used: {thresh:.4f}")
    print(pd.DataFrame(te_report).round(3).to_string())
    print("Confusion Matrix [TEST] (rows=true, cols=pred):", te_cm)
    print("="*72 + "\n")


# %%
# Baseline 
baseline_art = run_once(
    run_name="baseline",
    df_train=df_train, df_dev=df_dev, df_test=df_test,
    label_col=LABEL_COL,
    num_cols=BASE_NUM_COLS,
    cat_cols=BASE_CAT_COLS,
    outdir=OUTDIR,
    target_precision=TARGET_PRECISION,
    use_cv=USE_CV, cv_splits=CV_SPLITS,
    group_col=GROUP_COL,
    evaluate_test=True
)

# Enhanced 
enhanced_art = run_once(
    run_name="enhanced",
    df_train=df_train, df_dev=df_dev, df_test=df_test,
    label_col=LABEL_COL,
    num_cols=ENH_NUM_COLS,
    cat_cols=ENH_CAT_COLS,
    outdir=OUTDIR,
    target_precision=TARGET_PRECISION,
    use_cv=USE_CV, cv_splits=CV_SPLITS,
    group_col=GROUP_COL,
    evaluate_test=True
)



