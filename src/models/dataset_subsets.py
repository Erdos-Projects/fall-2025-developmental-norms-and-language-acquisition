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
#PARQUET_PATH = "data/processed/train_slam_with_features2.parquet"   # <- set if running as a cell
PARQUET_PATH = "data/processed/train_slam_with_features.parquet"   # <- set if running as a cell
CSV_PATH     = None                        # optional fallback
SAVE_DIR     = "figs"
SAMPLE_ATTEMPTS = 20000    # for attempt-level pairplot speed
MIN_ATTEMPTS_PER_WORD = 100
TOP_K_POS = 8
INCLUDE_GROWTH = True      # set False to drop growth_rate from GLM

os.makedirs(SAVE_DIR, exist_ok=True)


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
#df = df[df['median_aoa'] <= 50].copy()
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

splits = ["train", "dev", "test"]

for split in splits:
    for (l1_val, l2_val), n in pair_counts.items():
        # skip tiny groups if you want a floor (optional)
        if n < 1000:
            continue

        df_sub = df[(df['l1'] == l1_val) & (df['l2'] == l2_val)].copy()

        #parquet_path = f"{out_dir}/complete_data/{split}_subset_l1-{l1_val}_l2-{l2_val}.parquet"
        parquet_path = f"{out_dir}/{split}_subset_l1-{l1_val}_l2-{l2_val}.parquet"
        #csv_path     = f"{out_dir}/complete_data/{split}_subset_l1-{l1_val}_l2-{l2_val}.csv"
        csv_path     = f"{out_dir}/{split}_subset_l1-{l1_val}_l2-{l2_val}.csv"

        df_sub.to_parquet(parquet_path, index=False)
        df_sub.to_csv(csv_path, index=False)

        print(f"Saved {len(df_sub)} rows for {split}, l1={l1_val}, l2={l2_val} -> {parquet_path}")
