import pandas as pd

def main():
    tokens_slam = pd.read_parquet("data/processed/tokens_slam.parquet")

    # # Simulate another token dataset with some overlapping and some unique tokens
    # tokens_mapped = pd.DataFrame({
    #     "token": ["Je", "suis", "sûr", "Ce"],
    #     "l2": ["fr", "fr", "fr", "fr"],
    #     "l1": ["en", "en", "en", "en"],
    #     "median_aoa": [18, 20, 22, 24],
    #     "growth_aoa": [0.5, 0.6, 0.7, 0.8]
    # })
    tokens_wordbank = pd.read_csv("data/processed/wordbank_en_logistc_fits.csv")
    tokens_langtab = pd.read_csv("data/processed/language_translation_table.csv")
    # Rename token column in wordbank to uni_lemma
    tokens_wordbank = tokens_wordbank.rename(columns={"token": "uni_lemma"})

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

    # Merge tokens_wordbank with tokens_langtab_long on 'uni_lemma'
    tokens_mapped = pd.merge(
        tokens_langtab_long,
        tokens_wordbank,
        on="uni_lemma",
        how="inner",
        validate="many_to_one"
    )
    # TODO: Makes sense?
    tokens_mapped = tokens_mapped.loc[tokens_mapped['token'].isna() == False, :]
    # TODO: Fix?
    tokens_mapped = tokens_mapped.loc[tokens_mapped[["token", "l2", "l1"]].duplicated() == False, :]

    # Merge the two datasets on 'token', 'l2', and 'l1'
    tokens_features = pd.merge(
        tokens_slam,
        tokens_mapped,
        on=["token", "l2", "l1"],
        how="left",
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
