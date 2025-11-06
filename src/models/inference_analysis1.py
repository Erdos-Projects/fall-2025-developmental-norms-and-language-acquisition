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
PARQUET_PATH = "data/processed/complete_data/train_subset_l1-en_l2-es_plus.parquet"
# CSV_PATH     = args.csv
# SAMPLE_ATTEMPTS = args.sample
# SAVE_DIR     = args.save_dir
# os.makedirs(SAVE_DIR, exist_ok=True)

# %%
# ===================== LOAD =====================
def load_table(parquet_path=None, csv_path=None):
    if parquet_path and os.path.exists(parquet_path):
        print(f"[load] parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)
    if csv_path and os.path.exists(csv_path):
        print(f"[load] csv: {csv_path}")
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Provide a valid --parquet or --csv path.")

df = load_table(PARQUET_PATH, CSV_PATH)
df = df[df['median_aoa'] <= 50].copy()
print("\n=== DATA LOADED ===")
print("Shape:", df.shape)
print("Columns:", ", ".join(map(str, df.columns)))

# %%
# ===================== SUBSET BY L1/L2 =====================

import os
import pandas as pd

out_dir = "data/processed"
os.makedirs(out_dir, exist_ok=True)

# count how many rows per (l1,l2)
pair_counts = df[['l1','l2']].value_counts()

for (l1_val, l2_val), n in pair_counts.items():
    # skip tiny groups if you want a floor (optional)
    if n < 1000:
        continue

    df_sub = df[(df['l1'] == l1_val) & (df['l2'] == l2_val)].copy()

    parquet_path = f"{out_dir}/test_subset_l1-{l1_val}_l2-{l2_val}.parquet"
    csv_path     = f"{out_dir}/test_subset_l1-{l1_val}_l2-{l2_val}.csv"

    df_sub.to_parquet(parquet_path, index=False)
    df_sub.to_csv(csv_path, index=False)

    print(f"Saved {len(df_sub)} rows for l1={l1_val}, l2={l2_val} -> {parquet_path}")



# %%
# ===================== HYGIENE =====================
# Coerce dtypes for the features we’ll plot
num_feats_all = ["median_aoa", "growth_rate", "days", "time"]
for c in num_feats_all:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "token_wrong" in df.columns:
    df["token_wrong"] = df["token_wrong"].astype(int)
else:
    raise ValueError("Column 'token_wrong' is required for pair plots.")

# Which features exist in this table?
feats = [c for c in num_feats_all if c in df.columns]
if not feats:
    raise ValueError("None of the expected numeric features found: median_aoa, growth_rate, days, time")

# Optional hues for group heterogeneity
opt_hues = [c for c in ["token_pos", "l1", "l2"] if c in df.columns]

# Attempt-level working frame
keep_cols = feats + ["token_wrong"] + opt_hues
d = df[keep_cols].dropna(subset=feats + ["token_wrong"]).copy()

# Downsample for speed/clarity
if len(d) > SAMPLE_ATTEMPTS:
    d = d.sample(SAMPLE_ATTEMPTS, random_state=42).reset_index(drop=True)
print(f"[info] attempt-level rows used: {len(d):,}")


# %%
# ===================== QUICK STATS =====================
print("\n=== QUICK CORRELATIONS (attempt-level) ===")
# Feature vs token_wrong (point-biserial & Spearman)
for c in feats:
    x = pd.to_numeric(d[c], errors="coerce")
    y = d["token_wrong"].astype(int)
    ok = (~x.isna()) & (~y.isna())
    if ok.sum() >= 3:
        r_pb, p_pb = pointbiserialr(y[ok], x[ok])
        rho, p_rho = spearmanr(x[ok], y[ok])
        print(f"{c:12s} | point-biserial r={r_pb: .3f} (p={p_pb:.2e}) | Spearman ρ={rho: .3f} (p={p_rho:.2e})")

# Collinearity among numeric features (Spearman)
if len(feats) >= 2:
    corr_s = d[feats].corr(method="spearman")
    print("\nNumeric Spearman correlations among features:")
    print(corr_s.round(3).to_string())

# %%

# ===================== PAIR PLOT (hue = token_wrong) =====================
print("\n[plot] Pair plot (hue=token_wrong) …")
g = sns.PairGrid(d, vars=feats, hue="token_wrong", corner=True, diag_sharey=False)

# Diagonal: histograms (distribution checks)
g.map_diag(sns.histplot, bins=30, edgecolor=None, alpha=0.9)

# Lower: scatter + LOWESS line (signal, nonlinearity/thresholds)
def scatter_lowess(x, y, color, **kws):
    sns.scatterplot(x=x, y=y, alpha=0.25, s=12, edgecolor="none", linewidth=0)
    sns.regplot(x=x, y=y, scatter=False, lowess=True, color=color, line_kws={"lw":1.25})

g.map_lower(scatter_lowess)
g.add_legend(title="token_wrong")
g.fig.suptitle("Attempt-level Pair Plot (hue = token_wrong)\nLOWESS highlights nonlinearity/thresholds", y=1.02)

out1 = os.path.join(SAVE_DIR, "pairplot_attempts_token_wrong.png")
g.savefig(out1, dpi=180, bbox_inches="tight")
if args.show: plt.show()
plt.close(g.fig)
print("[save]", out1)



# %%
# ===================== PAIR PLOT (hue = token_wrong) =====================
print("\n[plot] Pair plot (hue=token_wrong) …")

# 1) Make sure hue is categorical (not numeric) so legend/palette behave
d['token_wrong_cat'] = d['token_wrong'].astype('category')

# 2) Define the lower-triangle mapper (scatter + LOWESS)
def scatter_lowess(x, y, color, **kws):
    # points
    sns.scatterplot(x=x, y=y, alpha=0.25, s=12, edgecolor="none", linewidth=0)
    # smooth (LOWESS)
    sns.regplot(x=x, y=y, scatter=False, lowess=True, color=color, line_kws={"lw": 1.25})

# 3) Try PairGrid with corner=True (newer seaborn). Fallback if not available.
try:
    g = sns.PairGrid(
        d,
        vars=feats,
        hue="token_wrong_cat",
        corner=True,          # <- may not exist on older seaborn
        diag_sharey=False
    )
    g.map_diag(sns.histplot, bins=30, edgecolor=None, alpha=0.9)
    g.map_lower(scatter_lowess)
    g.add_legend(title="token_wrong")
    g.fig.suptitle(
        "Attempt-level Pair Plot (hue = token_wrong)\nLOWESS highlights nonlinearity/thresholds",
        y=1.02
    )
    out1 = os.path.join(SAVE_DIR, "pairplot_attempts_token_wrong.png")
    g.savefig(out1, dpi=180, bbox_inches="tight")
    if args.show: plt.show()
    plt.close(g.fig)
    print("[save]", out1)

except Exception as e:
    print("[warn] PairGrid with corner=True failed:", e)
    print("[info] Falling back to sns.pairplot (no LOWESS)…")
    g2 = sns.pairplot(
        d[feats + ['token_wrong_cat']],
        hue="token_wrong_cat",
        diag_kind="hist",
        plot_kws=dict(alpha=0.25, s=12, edgecolor="none"),
        corner=True if 'corner' in sns.pairplot.__code__.co_varnames else False
    )
    g2.fig.suptitle("Attempt-level Pair Plot (hue = token_wrong)", y=1.02)
    out2 = os.path.join(SAVE_DIR, "pairplot_attempts_token_wrong_fallback.png")
    g2.savefig(out2, dpi=180, bbox_inches="tight")
    if args.show: plt.show()
    plt.close(g2.fig)
    print("[save]", out2)


# %%
# --- Attempt-level pair plot ------------------------------------
feats = [c for c in ["median_aoa","growth_rate","days","time"] if c in df.columns]
d = df[feats + ["token_wrong","token_pos","l1","l2"]].dropna(subset=feats+["token_wrong"]).copy()
d["token_wrong_cat"] = d["token_wrong"].astype("category")
if len(d) > SAMPLE_ATTEMPTS:
    d = d.sample(SAMPLE_ATTEMPTS, random_state=42)

# quick console stats
from scipy.stats import pointbiserialr, spearmanr
print("\n[attempt-level] feature ↔ token_wrong")
for c in feats:
    r_pb, p_pb = pointbiserialr(d["token_wrong"], d[c])
    rho, p_rho = spearmanr(d[c], d["token_wrong"])
    print(f"{c:12s} | point-biserial r={r_pb: .3f} (p={p_pb:.2e}) | Spearman ρ={rho: .3f} (p={p_rho:.2e})")

# pair plot with LOWESS in lower triangle
g = sns.PairGrid(d, vars=feats, hue="token_wrong_cat", corner=True, diag_sharey=False)
g.map_diag(sns.histplot, bins=30, edgecolor=None, alpha=0.9)
def scatter_lowess(x, y, color, **kws):
    sns.scatterplot(x=x, y=y, alpha=0.25, s=12, edgecolor="none")
    sns.regplot(x=x, y=y, scatter=False, lowess=True, color=color, line_kws={"lw":1.25})
g.map_lower(scatter_lowess)
g.add_legend(title="token_wrong")
g.fig.suptitle("Attempt-level Pair Plot (hue = token_wrong)", y=1.02)
g.savefig(os.path.join(SAVE_DIR, "pairplot_attempts_token_wrong.png"), dpi=180, bbox_inches="tight")
plt.close(g.fig)


# %%

# --- Word-level pair plot ---------------------------------------
KEY = "uni_lemma" if "uni_lemma" in df.columns else "token"
agg = (df.groupby(KEY, as_index=False)
         .agg(n=("token_wrong","size"),
              errors=("token_wrong","sum"),
              error_rate=("token_wrong","mean")))
# POS mode (optional)
if "token_pos" in df.columns:
    pos_mode = (df.groupby(KEY)["token_pos"].agg(lambda s: s.value_counts(dropna=False).idxmax()).reset_index()
                .rename(columns={"token_pos":"token_pos_mode"}))
    agg = agg.merge(pos_mode, on=KEY, how="left")
# bring AoA/growth
for c in ["median_aoa","growth_rate"]:
    if c in df.columns:
        agg = agg.merge(df[[KEY,c]].dropna().drop_duplicates(subset=[KEY]), on=KEY, how="left")

agg = agg[(agg["n"] >= MIN_ATTEMPTS_PER_WORD) & (agg["median_aoa"].notna())].copy()

# Feature–feature correlations at word level
print("\n[word-level] Spearman among numeric:")
num_cols = [c for c in ["median_aoa","growth_rate","n","error_rate"] if c in agg.columns]
print(agg[num_cols].corr(method="spearman").round(3))


# plain word-level pair plot
sns.pairplot(agg[num_cols], diag_kind="hist", corner=True,
             plot_kws=dict(alpha=0.6, s=20, edgecolor="none"))
plt.suptitle("Word-level Pair Plot", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pairplot_wordlevel.png"), dpi=180)
plt.close()

# word-level by POS (top 8)
if "token_pos_mode" in agg.columns:
    top_pos = agg["token_pos_mode"].value_counts().head(TOP_K_POS).index
    sub = agg[agg["token_pos_mode"].isin(top_pos)].copy()
    sns.pairplot(sub[num_cols + ["token_pos_mode"]],
                 hue="token_pos_mode", diag_kind="hist", corner=True,
                 plot_kws=dict(alpha=0.6, s=20, edgecolor="none"))
    plt.suptitle("Word-level Pair Plot (hue = POS)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "pairplot_wordlevel_pos.png"), dpi=180)
    plt.close()

# %%

w = agg['n'].to_numpy()
x = agg['median_aoa'].to_numpy()
y = agg['error_rate'].to_numpy()

xbar = np.average(x, weights=w)
ybar = np.average(y, weights=w)
r_w = np.sum(w * (x - xbar) * (y - ybar)) / np.sqrt(np.sum(w * (x - xbar)**2) * np.sum(w * (y - ybar)**2))
print(f"Weighted Pearson (AoA vs error_rate): r_w = {r_w:.3f}")


# %%
a = agg.dropna(subset=['median_aoa','n','errors']).copy()
a['err_rate'] = a['errors'] / a['n']

# Base model
m = smf.glm('err_rate ~ median_aoa', data=a,
            family=sm.families.Binomial(), freq_weights=a['n']).fit()
print(m.summary())

# Add POS (baseline difficulty differences)
if 'token_pos_mode' in a.columns:
    m_pos = smf.glm('err_rate ~ median_aoa + C(token_pos_mode)', data=a,
                    family=sm.families.Binomial(), freq_weights=a['n']).fit()
    print(m_pos.summary())


# %%
if 'token_pos_mode' in a.columns:
    m_int = smf.glm('err_rate ~ median_aoa * C(token_pos_mode)', data=a,
                    family=sm.families.Binomial(), freq_weights=a['n']).fit()
    print(m_int.summary())


# %%
a['log_n'] = np.log(a['n'])
m_expo = smf.glm('err_rate ~ median_aoa + log_n', data=a,
                 family=sm.families.Binomial(), freq_weights=a['n']).fit()
print(m_expo.summary())



