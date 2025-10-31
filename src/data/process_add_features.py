# Add token features back to the SLAM train, test, and dev datasets
import pandas as pd

def main():
    # Merge the two datasets on 'token', 'l2', and 'l1'
    tokens_features = pd.read_parquet("data/processed/tokens_features.parquet")

    for split in ["train", "test", "dev"]:
        slam_path = f"data/processed/{split}_slam.parquet"
        slam_dat = pd.read_parquet(slam_path)

        # Merge SLAM data with token features
        merged_dat = pd.merge(
            slam_dat,
            tokens_features,
            on=["token", "l2", "l1"],
            how="left",
            validate="many_to_one"
        )

        # Export the merged dataset
        out_path = f"data/processed/{split}_slam_with_features.parquet"
        merged_dat.to_parquet(out_path, index=False)
        print(f"Wrote {len(merged_dat)} rows to {out_path}")

if __name__ == "__main__":
    main()
