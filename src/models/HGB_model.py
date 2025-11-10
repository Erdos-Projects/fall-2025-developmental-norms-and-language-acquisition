# %%
# === HGB: Non-linear model (tree-based) ===
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, classification_report, confusion_matrix
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, classification_report, confusion_matrix)



# ---- Paths: EDIT THESE ----
#TRAIN_PATH = Path('data/processed/complete_data/train_subset_l1-en_l2-es_plus.parquet')
#DEV_PATH   = Path('data/processed/complete_data/dev_subset_l1-en_l2-es_plus.parquet')
#TEST_PATH  = Path('data/processed/complete_data/test_subset_l1-en_l2-es_plus.parquet')

TRAIN_PATH = Path('data/processed/train_subset_l1-en_l2-es_plus.parquet')
DEV_PATH   = Path('data/processed/dev_subset_l1-en_l2-es_plus.parquet')
TEST_PATH  = Path('data/processed/test_subset_l1-en_l2-es_plus.parquet')

# ---- Output directory ----
OUTDIR = Path('./Gradient_boosting_outputs_notebook')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Core knobs
TARGET_PRECISION = 0.30     # adjust as needed (e.g., 0.27, 0.33)
LABEL_COL = "token_wrong"   # keep consistent with your splits



# %%
# === Prep: load data & define features for the tree run ===
import pandas as pd

# 1) Load the splits (uses the paths you set earlier)
df_train = pd.read_parquet(TRAIN_PATH)
df_dev   = pd.read_parquet(DEV_PATH)
df_test  = pd.read_parquet(TEST_PATH)

# 2) Make sure the label exists and is int
LABEL_COL = "token_wrong"  # keep in sync
for _df in (df_train, df_dev, df_test):
    if LABEL_COL not in _df.columns:
        raise ValueError(f"Label '{LABEL_COL}' missing from one of the splits")
    _df[LABEL_COL] = _df[LABEL_COL].astype(int)



# %%
# === Features ===

BASE_CAT_COLS = ["token_pos", "token_morph", "token_edges"]  # try both with & without token
BASE_NUM_COLS = ["time", "days"]  # basic numerics that usually help


ENH_CAT_COLS = BASE_CAT_COLS + ["category"]
ENH_NUM_COLS = BASE_NUM_COLS + ["median_aoa_p", "levdistfrac", "src_freq_lm", "dest_freq_lm"]



# %%


def choose_threshold_at_precision_max_recall(y_true, p, target_precision: float) -> float:
    prec, rec, thr = precision_recall_curve(y_true, p)
    ok = np.where(prec[:-1] >= target_precision)[0]  # align with thr length
    if len(ok) == 0:
        print(f"[DEV] warn: cannot reach precision {target_precision:.0%}; using 0.50")
        return 0.50
    best_ix = ok[np.argmax(rec[ok])]
    return float(thr[best_ix])

def evaluate_on_dev(best_pipe, df_dev, label_col, num_cols, cat_cols, target_precision: float):
    X_dev = df_dev[cat_cols + num_cols]
    y_dev = df_dev[label_col].astype(int)

    p_dev = best_pipe.predict_proba(X_dev)[:, 1]
    dev_auc = roc_auc_score(y_dev, p_dev)
    dev_ap  = average_precision_score(y_dev, p_dev)
    thr     = choose_threshold_at_precision_max_recall(y_dev.values, p_dev, target_precision)

    yhat = (p_dev > thr).astype(int)
    rep  = classification_report(y_dev, yhat, output_dict=True)
    cm   = confusion_matrix(y_dev, yhat).tolist()

    print("\n=== DEV (threshold picked here) ===")
    print(f"AUC={dev_auc:.4f} | AP={dev_ap:.4f} | targetP={target_precision:.2f} | chosen thr={thr:.4f}")
    print(pd.DataFrame(rep).round(3).to_string())
    print("Confusion Matrix [DEV]:", cm)
    return thr


# %%
# --- Build HGB pipeline using OrdinalEncoder for cats  ---
# Note: OrdinalEncoder keeps things DENSE but compact, much smaller than one-hot for huge vocab.
def build_hgb_pipeline(num_cols_present: list[str], cat_cols_present: list[str]) -> Pipeline:
    # numeric branch: median -> log1p on time/days -> scale
    LOG1P_COLS = ["time", "days"]
    log_cols = [c for c in LOG1P_COLS if c in num_cols_present]
    lin_cols = [c for c in num_cols_present if c not in log_cols]

    num_log = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    num_lin = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # categorical branch: ordinal-encode with a safe unknown_value
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    pre = ColumnTransformer([
        ("num_log", num_log, log_cols),
        ("num_lin", num_lin, lin_cols),
        ("cat",     cat,     cat_cols_present),
    ], remainder="drop")  # dense on purpose for HGB

    # A fast HGB config (class imbalance aware)
    gb = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=None,           # let it grow shallow trees as needed
        max_iter=300,             # early-ish but strong
        max_leaf_nodes=31,        # small trees generalize well
        min_samples_leaf=50,      # regularization
        l2_regularization=0.0,
        validation_fraction=None, # we have a separate DEV, so no internal split
        early_stopping=False,
        class_weight="balanced",
        random_state=42
    )
    return Pipeline([("pre", pre), ("clf", gb)])


# %%
# --- Runner mirroring your logistic run_once (DEV threshold, optional TEST) ---
def run_tree_once(run_name: str,
                  df_train, df_dev, df_test, label_col,
                  num_cols, cat_cols,
                  outdir: Path,
                  target_precision: float,
                  evaluate_test: bool = True):
    # keep only columns we need
    present_num = [c for c in num_cols if c in df_train.columns]
    present_cat = [c for c in cat_cols if c in df_train.columns]
    needed = present_num + present_cat + [label_col]
    for colset_name, df, tag in (("TRAIN", df_train, "train"), ("DEV", df_dev, "dev"), ("TEST", df_test, "test")):
        missing = [c for c in needed if c not in df.columns]
        if missing and colset_name != "TEST":  # allow TEST to miss and just skip test later
            raise ValueError(f"[{run_name}] {colset_name} missing columns: {missing}")

    train_df = df_train[needed].copy()
    dev_df   = df_dev[[c for c in needed if c in df_dev.columns]].copy()
    test_df  = df_test[[c for c in needed if c in df_test.columns]].copy()

    X_tr, y_tr = train_df.drop(columns=[label_col]), train_df[label_col].astype(int)
    X_dev, y_dev = dev_df.drop(columns=[label_col]), dev_df[label_col].astype(int)

    pipe_tree = build_hgb_pipeline(present_num, present_cat)

    # Fit on TRAIN only
    pipe_tree.fit(X_tr, y_tr)

    # DEV eval + threshold
    p_dev = pipe_tree.predict_proba(X_dev)[:, 1]
    dev_auc = roc_auc_score(y_dev, p_dev)
    dev_ap  = average_precision_score(y_dev, p_dev)
    thresh  = choose_threshold_at_precision_max_recall(y_dev.values, p_dev, target_precision)
    yhat_dev = (p_dev > thresh).astype(int)
    dev_rep = classification_report(y_dev, yhat_dev, output_dict=True)
    dev_cm  = confusion_matrix(y_dev, yhat_dev).tolist()

    # Save DEV artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    dev_metrics = {
        "run": run_name,
        "model": "HistGradientBoostingClassifier",
        "target_precision": float(target_precision),
        "dev": {
            "auc": float(dev_auc),
            "average_precision": float(dev_ap),
            "threshold": float(thresh),
            "classification_report": dev_rep,
            "confusion_matrix": dev_cm,
        },
        "features": {"num": present_num, "cat": present_cat}
    }
    (outdir / f"{run_name}_tree_metrics_DEV.json").write_text(json.dumps(dev_metrics, indent=2))
    import joblib; joblib.dump(pipe_tree, outdir / f"{run_name}_tree_pipeline.joblib")
    (outdir / f"{run_name}_tree_threshold.json").write_text(json.dumps({"threshold": float(thresh)}, indent=2))

    print("\n" + "="*72)
    print(f"{run_name.upper()} (HGB) — DEV ONLY")
    print(f"AUC: {dev_auc:.4f} | AP: {dev_ap:.4f} | chosen threshold: {thresh:.4f}")
    print(pd.DataFrame(dev_rep).round(3).to_string())
    print("Confusion Matrix [DEV] (rows=true, cols=pred):", dev_cm)
    print("="*72 + "\n")

    if not evaluate_test or test_df.empty:
        return {"run": run_name, "pipe": pipe_tree, "threshold": float(thresh)}

    # TEST eval (optional)
    X_te, y_te = test_df.drop(columns=[label_col]), test_df[label_col].astype(int)
    p_te = pipe_tree.predict_proba(X_te)[:, 1]
    te_auc = roc_auc_score(y_te, p_te)
    te_ap  = average_precision_score(y_te, p_te)
    yhat_te = (p_te > thresh).astype(int)
    te_rep = classification_report(y_te, yhat_te, output_dict=True)
    te_cm  = confusion_matrix(y_te, yhat_te).tolist()

    full_metrics = {**dev_metrics, "test": {
        "auc": float(te_auc),
        "average_precision": float(te_ap),
        "threshold_used": float(thresh),
        "classification_report": te_rep,
        "confusion_matrix": te_cm,
    }}
    (outdir / f"{run_name}_tree_metrics.json").write_text(json.dumps(full_metrics, indent=2))

    print("-"*72)
    print(f"{run_name.upper()} (HGB) — TEST")
    print(f"AUC: {te_auc:.4f} | AP: {te_ap:.4f} | threshold_used: {thresh:.4f}")
    print(pd.DataFrame(te_rep).round(3).to_string())
    print("Confusion Matrix [TEST] (rows=true, cols=pred):", te_cm)
    print("="*72 + "\n")

    return {"run": run_name, "pipe": pipe_tree, "threshold": float(thresh)}


# %%
tree_art = run_tree_once(
    run_name="hgb_base",
    df_train=df_train, df_dev=df_dev, df_test=df_test,
    label_col=LABEL_COL,
    num_cols=BASE_NUM_COLS,
    cat_cols=BASE_CAT_COLS,
    outdir=OUTDIR,
    target_precision=TARGET_PRECISION,
    evaluate_test=True   # start DEV-only for speed; flip True when happy
)


# %%
tree_art_enh = run_tree_once(
    run_name="hgb_enh",
    df_train=df_train, df_dev=df_dev, df_test=df_test,
    label_col=LABEL_COL,
    num_cols=ENH_NUM_COLS,
    cat_cols=ENH_CAT_COLS,
    outdir=OUTDIR,
    target_precision=TARGET_PRECISION,
    evaluate_test=True   # start DEV-only for speed; flip True when happy
)
