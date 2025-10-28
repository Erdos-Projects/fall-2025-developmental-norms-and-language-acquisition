import pandas as pd

tokens_slam = pd.read_parquet("data/processed/tokens_slam.parquet")
tokens_wordbank_en = pd.read_csv("data/processed/wordbank_en_logistic_fits.csv")
tokens_wordbank_es = pd.read_csv("data/processed/wordbank_es_logistic_fits.csv")
tokens_wordbank = pd.concat([tokens_wordbank_en, tokens_wordbank_es], ignore_index=True)
tokens_langtab = pd.read_csv("data/processed/language_translation_table.csv")
# TODO: Check if this is correct
tokens_wordbank = tokens_wordbank.drop(columns=["token"])
# Rename token column in wordbank to uni_lemma
# tokens_wordbank = tokens_wordbank.rename(columns={"token": "uni_lemma"})
# TODO: Force uniqueness, fix later
tokens_wordbank = tokens_wordbank.drop_duplicates(subset=["uni_lemma", "l1"])

# Reshape tokens_langtab to long format
tokens_langtab_long = tokens_langtab.melt(
    id_vars=["uni_lemma"],
    var_name="l2",
    value_name="token"
)

# Relabel French, English, Spanish language codes
lang_code_map = {
    "French": "fr",
    "English": "en",
    "Spanish": "es"
}
tokens_langtab_long['l2'] = tokens_langtab_long['l2'].map(lang_code_map)

# Remove astrisk from tokens
tokens_langtab_long['token'] = tokens_langtab_long['token'].str.replace('*', '', regex=False)

# Split tokens where token column contains multiple tokens separated by ; or /
tokens_langtab_long = tokens_langtab_long.assign(
    token=tokens_langtab_long['token'].str.split(r'[;/]')
).explode('token').reset_index(drop=True)

# Strip whitespace from tokens
tokens_langtab_long['token'] = tokens_langtab_long['token'].str.strip()
# Drop complete duplicates
tokens_langtab_long = tokens_langtab_long.drop_duplicates(ignore_index=True)

# Disambiguate
ambigious_tokens = tokens_langtab_long.loc[
    (tokens_langtab_long['token'].str.contains("(", regex=False, na=False)) &
    (tokens_langtab_long['uni_lemma'].isna() == False),
    :
].rename(columns={"token": "langtab_token"})

ambigious_tokens['token'] = ambigious_tokens['langtab_token'].str.split('(', expand=True)[0].str.strip()
ambigious_tokens = ambigious_tokens.drop_duplicates(subset=["token", "l2"])

ambigious_slam_list = []
for split in ["train", "test", "dev"]:
    slam_path = f"data/processed/{split}_slam.parquet"
    slam_dat = pd.read_parquet(slam_path)
    # Create an exercise id column with everything but the last two characters in token_id
    slam_dat["exercise_id"] = slam_dat["token_id"].str[:-2]

    merged_dat = pd.merge(
        slam_dat,
        ambigious_tokens,
        on=["token", "l2"],
        how="inner",
        validate="many_to_one"
    )

    # Keep only the columns: exercise_id and langtab_token
    merged_dat = merged_dat[["exercise_id", "langtab_token"]].drop_duplicates(ignore_index=True)
    merged_dat = merged_dat.rename(columns={"langtab_token": "ambigious_token"})
    merged_dat['exercise_count'] = merged_dat.groupby("exercise_id").cumcount()+1
    # Reshape to wide on exercie_id
    merged_dat = merged_dat.pivot(
        index="exercise_id",
        columns="exercise_count",
        values="ambigious_token"
    ).reset_index()

    append_dat = pd.merge(
        slam_dat,
        merged_dat,
        on="exercise_id",
        how="inner",
        validate="many_to_one"
    )
    ambigious_slam_list.append(append_dat)

ambigious_slam = pd.concat(ambigious_slam_list, ignore_index=True, axis=0)

ambigious_slam = ambigious_slam.to_parquet("data/processed/ambigious_slam.parquet", index=False)

