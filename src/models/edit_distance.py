# %%
# build_word_features.py
# - Cleans rows with missing/invalid tokens
# - Adds dataset word frequency per root (attempt frequency)
# - NYU-style cognateness features: translate(root) -> clean -> Levenshtein ->  wordfreq
# - (Optional) prompt-based edit distance per attempt
# - Merges results back and writes an enriched parquet

import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
import editdistance
from wordfreq import word_frequency

# If you want NYU-style translation-based cognateness:
USE_TRANSLATION = True
SRC_LANG = "es"   # src
DEST_LANG = "en"  # dest

# If you want attempt-level prompt-distance too:
ADD_PROMPT_DISTANCE = False       # set True to compute min distance to any word in the prompt
PROMPT_COL = "prompt"


# %%

splits = ["train", "dev", "test"]

for split in splits:

    # ----------------------- PATHS -----------------------
    IN_PARQUET  = f"data/processed/{split}_subset_l1-en_l2-es.parquet"
    OUT_PARQUET = f"data/processed/{split}_subset_l1-en_l2-es_plus.parquet"
    OUT_TXT     = f"{SRC_LANG}_{DEST_LANG}_rootwordfeats.txt"  # NYU-style text  (one line per root)


    # Which column represents the root word; 'uni_lemma' by default.
    ROOT_COL = "token"            # or "token" if you prefer surface forms
    CLEAN_SKIP_APOSTROPHE = True      #  NYU: skip translation for words with apostrophes
    API_SLEEP_SEC = 0.20              # be gentle to the translation API

    # %%

    # ----------------------- LOAD -----------------------
    if not os.path.exists(IN_PARQUET):
        raise FileNotFoundError(f"Input parquet not found: {IN_PARQUET}")

    df = pd.read_parquet(IN_PARQUET)

    if ROOT_COL not in df.columns:
        raise ValueError(f"Column '{ROOT_COL}' not found in dataframe. Available: {list(df.columns)}")
    if "token" not in df.columns:
        raise ValueError("Expected a 'token' column in the dataframe.")


    # %%
    #df.head(20)
    #df.tail()
    df.sample(20)
    df.info()

    # %%
    df.describe()
    df.token.value_counts(normalize = True)

    # %%
    # ------------------ CLEAN TOKENS/ROOTS ------------------
    # Accept alphabetic words (Latin letters) with optional internal hyphen/apostrophe.
    WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*")

    def clean_word(x):
        """Lowercase, strip; return first 'wordy' token or NaN."""
        if not isinstance(x, str):
            return np.nan
        s = x.strip().lower()
        if not s or s in {"nan", "none", "<unk>", "<null>"}:
            return np.nan
        m = WORD_RE.findall(s)
        return m[0] if m else np.nan

    before = len(df)
    df["token_clean"] = df["token"].apply(clean_word)
    df["root_clean"]  = df[ROOT_COL].apply(clean_word)

    # Drop rows with unusable token
    df = df[df["token_clean"].notna()].copy()
    dropped = before - len(df)
    print(f"[clean] dropped rows without a usable token: {dropped} (from {before} → {len(df)})")

    # Fill root with token if missing
    df["token"] = df["token_clean"]
    df[ROOT_COL] = df["root_clean"].fillna(df["token"])
    df.drop(columns=["token_clean", "root_clean"], inplace=True)

    # Normalize prompt if we’ll use it
    if ADD_PROMPT_DISTANCE:
        if PROMPT_COL not in df.columns:
            raise ValueError(f"ADD_PROMPT_DISTANCE=True but '{PROMPT_COL}' column is missing.")
        df[PROMPT_COL] = df[PROMPT_COL].astype(str)

    # %%
    # ====== SAMPLE AFTER CLEANING (pilot) ======
    RNG_SEED = 42
    N_PILOT  = 50           # how many unique items to keep
    SAMPLE_BY = "token"       # choose: "root" or "token"

    if SAMPLE_BY == "token":
        uniq = df["token"].dropna().astype(str).str.lower().unique()
        keep = pd.Series(uniq).sample(min(N_PILOT, len(uniq)), random_state=RNG_SEED).tolist()
        df = df[df["token"].astype(str).str.lower().isin(keep)].copy()
        print(f"[sample] kept {len(set(keep))} unique TOKENS → rows now {len(df):,}")

    elif SAMPLE_BY == "root":
        uniq = df[ROOT_COL].dropna().astype(str).str.lower().unique()
        keep = pd.Series(uniq).sample(min(N_PILOT, len(uniq)), random_state=RNG_SEED).tolist()
        df = df[df[ROOT_COL].astype(str).str.lower().isin(keep)].copy()
        print(f"[sample] kept {len(set(keep))} unique ROOTS ({ROOT_COL}) → rows now {len(df):,}")

    # %%
    aoa_df = (
        df[['uni_lemma', 'median_aoa_p']]
        .dropna(subset=['uni_lemma', 'median_aoa_p'])
        .drop_duplicates(subset='uni_lemma')
        .rename(columns={'uni_lemma': 'Word'})
    )
    print("aoa_df columns:", aoa_df.columns.tolist())
    print(aoa_df.head())


    # %%
    # ------------- NYU-STYLE TRANSLATION FEATURES -------------
    if USE_TRANSLATION:
        try:
            from googletrans import Translator
            translator = Translator()
        except Exception as e:
            raise RuntimeError(
                "googletrans not available. Install with: pip install googletrans==4.0.0-rc1"
            ) from e

        # Translation cache
        trans_cache = {}

        FILLER_1 = {'to', 'i', 'we', 'you', 'they', 'he', 'she', 'the', "i'll"}

        def translate_clean(word: str) -> str:
            """Translate word SRC->DEST, then Pam-style clean: unidecode, lower, drop filler."""
            if word in trans_cache:
                return trans_cache[word]
            if CLEAN_SKIP_APOSTROPHE and "'" in word:
                # mimic NYU's skip; keep as-is
                trans_cache[word] = word
                return word
            # translate (retry once on failure)
            try:
                time.sleep(API_SLEEP_SEC)
                t = translator.translate(word, src=SRC_LANG, dest=DEST_LANG).text
            except Exception:
                time.sleep(0.5)
                try:
                    t = translator.translate(word, src=SRC_LANG, dest=DEST_LANG).text
                except Exception:
                    t = word  # fallback: keep original
            ct = unidecode(t).lower().strip()
            parts = ct.split()
            if len(parts) == 2 and parts[0] in FILLER_1:
                ct = parts[1]
            elif len(parts) > 2 and parts[1] == "will":
                ct = " ".join(parts[2:])
            trans_cache[word] = ct
            return ct

        # Unique roots to process (lowercased already by clean_word)
        roots = df[ROOT_COL].dropna().astype(str).str.lower().unique().tolist()
        print(f"[translation] unique roots to process: {len(roots)}")

        rows = []
        for root in tqdm(roots, desc="Translating & computing cognateness"):
            r = root.strip().lower()
            trans = translate_clean(r)
            # Levenshtein distance between root and cleaned translation
            ld = editdistance.eval(r, trans)
            denom = max(len(r), len(trans)) or 1
            ldfrac = ld / denom
            # AoA lookup (if translating TO English, look up AoA on translation; else on root)
            if aoa_df is not None:
                aoa_key = trans if DEST_LANG.lower() == "en" else r
                hit = aoa_df.loc[aoa_df["Word"] == aoa_key]
                aoa_val = float(hit["median_aoa_p"].iloc[0]) if not hit.empty else np.nan
            else:
                aoa_val = np.nan
            # wordfreq in src/dest languages (LM frequencies)
            src_freq_lm  = word_frequency(r, SRC_LANG)
            dest_freq_lm = word_frequency(trans, DEST_LANG)

            rows.append({
                ROOT_COL: r,
                "clean_translation": trans,
                "levdist": ld,
                "levdistfrac": ldfrac,
                "aoa_lookup": aoa_val,
                "src_freq_lm": src_freq_lm,
                "dest_freq_lm": dest_freq_lm,
            })

        pam_features = pd.DataFrame(rows).drop_duplicates(subset=[ROOT_COL])
        print(f"[translation] built pam_features: {pam_features.shape}")

        # Write NYU-style text file (root, clean_trans, src_freq, levdist, levdistfrac, aoa)
        with open(OUT_TXT, "w", encoding="utf-8") as fp:
            for r in pam_features.itertuples(index=False):
                line = ",".join([
                    getattr(r, ROOT_COL),
                    r.clean_translation,
                    f"{r.src_freq_lm}",
                    f"{r.levdist}",
                    f"{r.levdistfrac:.4f}",
                    f"" if np.isnan(r.aoa_lookup) else f"{r.aoa_lookup}",
                ])
                fp.write(line + "\n")
        print(f"[save] Pam-style root features → {OUT_TXT}")

        # Merge NYU-style features back to all rows by ROOT_COL
        df = df.merge(pam_features, on=ROOT_COL, how="left")



    # %%
    cols = [ROOT_COL, "clean_translation","levdist","levdistfrac","src_freq_lm","dest_freq_lm"]
    print(df[cols].sample(10, random_state=0).to_string(index=False))

    # rough distributions
    print(df["levdistfrac"].describe(percentiles=[.1,.5,.9]))
    print(df[["src_freq_lm","dest_freq_lm"]].describe())


    # %%
    # ------------- OPTIONAL: PROMPT-BASED EDIT DISTANCE (+nearest word, collision-safe merge) -------------
    if ADD_PROMPT_DISTANCE:
        WORD_SPLIT_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")

        def tokenize(text: str):
            return WORD_SPLIT_RE.findall(text.lower()) if isinstance(text, str) else []

        # Prefer rapidfuzz (faster); fall back to editdistance
        try:
            from rapidfuzz.distance import Levenshtein as Lev
            def lev(a, b): return Lev.distance(a, b)
        except Exception:
            def lev(a, b): return editdistance.eval(a, b)

        # --- Build de-duplicated (token, prompt) pairs as strings
        keys = ["token", PROMPT_COL]
        pairs = df[keys].copy()
        pairs["token"] = pairs["token"].astype(str)
        pairs[PROMPT_COL] = pairs[PROMPT_COL].astype(str)
        pairs = pairs.drop_duplicates().reset_index(drop=True)

        def nearest_word_and_distance(tok: str, prompt: str):
            """Return (min_distance, best_word, best_index) for tok vs words in prompt."""
            t = (tok or "").lower().strip()
            if not t:
                return 0, "", -1
            words = tokenize(prompt)
            if not words:
                return len(t), "", -1
            if t in words:
                i = words.index(t)
                return 0, t, i
            dists = [lev(t, w) for w in words]
            i_min = int(np.argmin(dists))
            return dists[i_min], words[i_min], i_min

        from tqdm import tqdm
        tqdm.pandas(desc="Computing prompt-based edit distance + nearest word")
        pairs[["edit_distance_prompt",
               "prompt_nearest_word",
               "prompt_nearest_index"]] = pairs.progress_apply(
            lambda r: pd.Series(nearest_word_and_distance(r["token"], r[PROMPT_COL])),
            axis=1
        )

        # Normalized distance
        pairs["edit_distance_prompt_norm"] = (
            pairs["edit_distance_prompt"] / pairs["token"].str.len().clip(lower=1)
        )

        # --- Collision-safe merge back to df
        # 1) make sure LHS keys are strings
        df["token"] = df["token"].astype(str)
        df[PROMPT_COL] = df[PROMPT_COL].astype(str)

        feature_cols = [c for c in pairs.columns if c not in keys]
        # 2) if df already has *_pair names from a previous run, drop them to avoid suffix clashes
        SUFFIX = "_pair"
        would_conflict = [c + SUFFIX for c in feature_cols if (c + SUFFIX) in df.columns]
        if would_conflict:
            print("[prompt] dropping prior suffixed columns to avoid clash:", would_conflict)
            df.drop(columns=would_conflict, inplace=True)

        # 3) merge with explicit suffix, then coalesce any overlaps (keep existing, fill from _pair)
        merged = df.merge(pairs, on=keys, how="left", suffixes=("", SUFFIX))

        # coalesce overlapping feature names
        for c in feature_cols:
            c_new = c + SUFFIX
            if c_new in merged.columns:
                if c in merged.columns:
                    merged[c] = merged[c].fillna(merged[c_new])
                    merged.drop(columns=c_new, inplace=True)
                else:
                    merged.rename(columns={c_new: c}, inplace=True)

        df = merged
        print("[prompt] ensured columns present:",
              ["edit_distance_prompt", "edit_distance_prompt_norm",
               "prompt_nearest_word", "prompt_nearest_index"])


    # %%
    # ------------- DATASET-INTERNAL FREQUENCY (per ROOT_COL) -------------
    # Requires: df loaded; ROOT_COL defined (e.g., ROOT_COL = "uni_lemma")

    if ROOT_COL not in df.columns:
        raise ValueError(f"ROOT_COL '{ROOT_COL}' not found. Available: {list(df.columns)}")

    # Build fresh frequency table
    freq = (df.groupby(ROOT_COL, as_index=False)
              .size()
              .rename(columns={"size": "word_count"}))
    freq["word_freq"] = freq["word_count"] / len(df)

    # Drop pre-existing columns to avoid merge suffix clashes (idempotent behavior)
    for c in ["word_count", "word_freq"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Merge back on ROOT_COL
    df = df.merge(freq, on=ROOT_COL, how="left")

    # (Optional) quick sanity checks
    print(f"[freq:{ROOT_COL}] unique entries: {len(freq)}; "
          f"min/max count: {freq['word_count'].min()} / {freq['word_count'].max()}")
    print(f"[freq:{ROOT_COL}] sum of word_freq (should be ~1 across rows if you sum per-appearances): "
          f"{freq['word_freq'].sum():.4f}")


    # %%
    # ------------- DATASET-INTERNAL FREQUENCY (per surface token) -------------
    if "token" in df.columns:
        freq_tok = (df.groupby("token", as_index=False)
                      .size()
                      .rename(columns={"size": "token_count"}))
        freq_tok["token_freq"] = freq_tok["token_count"] / len(df)

        for c in ["token_count", "token_freq"]:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

        df = df.merge(freq_tok, on="token", how="left")
        print(f"[freq:token] unique tokens: {len(freq_tok)}; "
              f"min/max: {freq_tok['token_count'].min()} / {freq_tok['token_count'].max()}")


    # %%
    # ----------------------- PEEK (rich) -----------------------
    # core id/keys
    peek_cols = [ROOT_COL]


    # translation/cognateness features (include optional best_trans_token if you added it)
    if USE_TRANSLATION:
        peek_cols += [
            "clean_translation",     # best_trans_token optional
            "levdist", "levdistfrac",
            "src_freq_lm", "dest_freq_lm",
        ]

    # prompt-distance features (now with nearest word + index)
    if ADD_PROMPT_DISTANCE:
        peek_cols += [
            "edit_distance_prompt", "edit_distance_prompt_norm",
            "prompt_nearest_word", "prompt_nearest_index"
        ]


    # dataset-internal frequencies (lemma + token, if present)
    peek_cols += ["token_count", "token_freq"]

    # keep only those that exist
    missing = [c for c in peek_cols if c not in df.columns]
    if missing:
        print("[peek] (info) these columns aren't in df and will be skipped:", missing)

    peek_cols = [c for c in peek_cols if c in df.columns]

    # nicer console view
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df[peek_cols].head(25).to_string(index=False))



    # %%
    # ----------------------- SAVE -----------------------
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[done] wrote enriched parquet → {OUT_PARQUET}")
