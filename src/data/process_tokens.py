import pandas as pd

def main():
    tokens_slam = pd.read_parquet("data/processed/tokens_slam.parquet")

    # Simulate another token dataset with some overlapping and some unique tokens
    tokens_wordbank = pd.DataFrame({
        "token": ["hello", "world", "foo", "bar"],
        "l2": ["en", "en", "es", "es"],
        "l1": ["fr", "fr", "en", "en"],
        "median_aoa": [18, 20, 22, 24],
        "growth_aoa": [0.5, 0.6, 0.7, 0.8]
    })

    # Merge the two datasets on 'token', 'l2', and 'l1'
    tokens_features = pd.merge(
        tokens_slam,
        tokens_wordbank,
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
