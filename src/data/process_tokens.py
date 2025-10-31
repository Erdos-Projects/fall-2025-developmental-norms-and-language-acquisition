import pandas as pd

def main():
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

    # Add uni_lemma to tokens_slam by merging with tokens_langtab_long on 'token' and 'l2'
    tokens_slam_unilemma = pd.merge(
        tokens_langtab_long,
        tokens_slam,
        on=['token', 'l2'],
        how='left',
        validate="many_to_one"
    )
    # Drop cases without uni_lemma
    # TODO: Check
    tokens_slam_unilemma = tokens_slam_unilemma.loc[
        tokens_slam_unilemma['uni_lemma'].isna() == False, :
    ]
    tokens_slam_unilemma = tokens_slam_unilemma.loc[
        tokens_slam_unilemma['l1'].isna() == False, :
    ]

    # TODO: Makes sense?
    tokens_slam_unilemma = tokens_slam_unilemma.loc[tokens_slam_unilemma['token'].isna() == False, :]
    # TODO: Fix?
    tokens_slam_unilemma = tokens_slam_unilemma.loc[tokens_slam_unilemma[["token", "l2", "l1"]].duplicated() == False, :]

    # Merge the two datasets on 'token', 'l2', and 'l1'
    tokens_features = pd.merge(
        tokens_slam_unilemma,
        tokens_wordbank,
        left_on=["uni_lemma" , "l1"],
        right_on=["uni_lemma" , "l1"],
        how="inner",
        validate="many_to_one"
    )

    # Export the merged dataset with token features
    tokens_features.to_parquet(
        "data/processed/tokens_features.parquet",
        index=False
    )
    # Describe data before exporting
    print(f"Wrote {len(tokens_features)} rows to data/processed/tokens_features.parquet")

if __name__ == "__main__":
    main()

