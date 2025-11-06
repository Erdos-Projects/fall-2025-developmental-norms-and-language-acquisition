# %%

import os, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, pointbiserialr
import statsmodels.api as sm
import statsmodels.formula.api as smf

sns.set_theme()

# ----------------------- CONFIG -----------------------
PARQUET_PATH = "data/processed/train_slam_with_features2.parquet"   # <- set if running as a cell
CSV_PATH     = None                        # optional fallback
SAVE_DIR     = "figs"
SAMPLE_ATTEMPTS = 20000    # for attempt-level pairplot speed
MIN_ATTEMPTS_PER_WORD = 100
TOP_K_POS = 8
INCLUDE_GROWTH = True      # set False to drop growth_rate from GLM

os.makedirs(SAVE_DIR, exist_ok=True)


# %%
# ----------------------- LOAD -------------------------
def load_table(parquet_path=None, csv_path=None):
    if parquet_path and os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Provide a valid PARQUET_PATH or CSV_PATH.")
    return df

try:
    df
except NameError:
    df = load_table(PARQUET_PATH, CSV_PATH)

print("[info] df shape:", df.shape)
print("[info] columns:", list(df.columns))

# %%
# ------------------- BASIC HYGIENE --------------------
# Coerce types
if "token_wrong" in df.columns:
    df["token_wrong"] = df["token_wrong"].astype(int)

for col in ["median_aoa","growth_rate","days","time"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Key columns
KEY_WORD = "uni_lemma" if "uni_lemma" in df.columns else "token"
has_pos  = "token_pos" in df.columns

# %%
# ---------------- WORD-LEVEL AGGREGATION --------------
agg = (df.groupby(KEY_WORD, as_index=False)
         .agg(n=("token_wrong","size"),
              errors=("token_wrong","sum"),
              error_rate=("token_wrong","mean")))

# Representative POS per word (mode)
if has_pos:
    pos_mode = (df.groupby(KEY_WORD)["token_pos"]
                  .agg(lambda s: s.value_counts(dropna=False).idxmax())
                  .reset_index()
                  .rename(columns={"token_pos":"token_pos_mode"}))
    agg = agg.merge(pos_mode, on=KEY_WORD, how="left")

# Attach AoA / growth from the same file (first non-null per word)
for c in ["median_aoa","growth_rate"]:
    if c in df.columns:
        tmp = df[[KEY_WORD, c]].dropna().drop_duplicates(subset=[KEY_WORD])
        agg = agg.merge(tmp, on=KEY_WORD, how="left")

# Filter for stability
agg = agg[(agg["n"] >= MIN_ATTEMPTS_PER_WORD) & (agg["median_aoa"].notna())].copy()
agg["err_rate"] = agg["errors"]/agg["n"]
print(f"[info] word-level rows after filter: {len(agg)}")

# %%
# ---------------- WORD-LEVEL CORRELATIONS -------------
def corr_report(x, y, label="Overall"):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = pd.DataFrame({"x":x, "y":y}).dropna()
    if len(m) < 3:
        print(f"{label:>12} | not enough data")
        return
    rho, p_rho = spearmanr(m["x"], m["y"])
    r,   p_r   = pearsonr(m["x"], m["y"])
    print(f"{label:>12} | Spearman ρ={rho:.3f} (p={p_rho:.2e}) | Pearson r={r:.3f} (p={p_r:.2e})")

corr_report(agg["median_aoa"], agg["error_rate"], "Overall (word-level)")

# Correlation heatmap (numeric only)
num_cols = [c for c in ["median_aoa","growth_rate","n","err_rate"] if c in agg.columns]
if len(num_cols) >= 2:
    corr = agg[num_cols].corr(method="spearman")
    plt.figure(figsize=(4.5,4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", square=True, cbar_kws={'shrink': .75})
    plt.title("Word-level Spearman correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "corr_heatmap_wordlevel.png"), dpi=160)
    plt.close()
    print("[save] corr_heatmap_wordlevel.png")

# %%
# ---------------- AoA vs ERROR SCATTER ----------------
plt.figure(figsize=(6.6,5))
ax = sns.regplot(data=agg, x="median_aoa", y="error_rate", lowess=True, scatter=False)
plt.scatter(agg["median_aoa"], agg["error_rate"],
            s=np.clip(agg["n"]/5, 10, 150), alpha=0.45, edgecolor="none")
plt.xlabel("Age of Acquisition (median_aoa)")
plt.ylabel("Error rate (per word)")
plt.title(f"AoA vs Error Rate (size ∝ attempts, n≥{MIN_ATTEMPTS_PER_WORD})")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "aoa_vs_error_scatter.png"), dpi=160)
plt.close()
print("[save] aoa_vs_error_scatter.png")


# %%
# ------------- STRATIFIED BY POS (top-K) --------------
if "token_pos_mode" in agg.columns:
    top_pos = agg["token_pos_mode"].value_counts().head(TOP_K_POS).index.tolist()
    sub = agg[agg["token_pos_mode"].isin(top_pos)].copy()

    print("\n[POS] correlations within POS:")
    for pos in top_pos:
        g = sub[sub["token_pos_mode"] == pos]
        if len(g) >= 8:
            corr_report(g["median_aoa"], g["error_rate"], pos)

    g = sns.lmplot(
        data=sub, x="median_aoa", y="error_rate",
        hue="token_pos_mode", col="token_pos_mode", col_wrap=4,
        scatter_kws=dict(s=20, alpha=0.5),
        line_kws=dict(), lowess=True, height=3, aspect=1.05, legend=False
    )
    g.set_axis_labels("Age of Acquisition (median_aoa)", "Error rate")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("AoA vs Error by POS (LOWESS)")
    g.savefig(os.path.join(SAVE_DIR, "aoa_vs_error_by_pos.png"), dpi=160)
    plt.close(g.fig)
    print("[save] aoa_vs_error_by_pos.png")

# %%
# ------ WEIGHTED BINOMIAL GLM (word-level counts) -----
formula = "err_rate ~ median_aoa"
if INCLUDE_GROWTH and "growth_rate" in agg.columns and agg["growth_rate"].notna().any():
    formula += " + growth_rate"
if "token_pos_mode" in agg.columns:
    formula += " + C(token_pos_mode)"

glm_data = agg.dropna(subset=["median_aoa", "err_rate", "n"]).copy()
glm_data = glm_data[glm_data["n"] > 0]
m = smf.glm(formula, data=glm_data, family=sm.families.Binomial(),
            freq_weights=glm_data["n"]).fit()
print("\n[GLM] Weighted binomial on word counts")
print("Formula:", formula)
print(m.summary())


