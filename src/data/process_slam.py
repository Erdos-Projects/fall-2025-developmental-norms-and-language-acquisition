import re
from pathlib import Path
import pandas as pd

# TODO: Add correctness from key/test

def parse_slam(input_path: Path) -> pd.DataFrame:
    """Process a single dataset in SLAM format into a pandas DataFrame.

    Keyword arguments:
    input_path: path to the SLAM formatted text file
    returns: a pandas.DataFrame with the parsed data
    """

    # Unzip and read the file content
    text = input_path.read_text(encoding="utf-8")

    # Split content blocks by one or more blank lines
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    list_of_blocks = []

    # Loop through each block and parse its content into a dictionary
    for i, block in enumerate(blocks, 1):
        # Incrementally create a block ID
        block_dict = {"block_id": i}
        # Loop through each line in the block
        for line in block.split("\n"):
            # Check for prompt line (note that listen doesn't have a prompt)
            if line.strip()[:8].lower() == "# prompt":
                prompt_text = line.split(":", 1)[1].strip()
                block_dict["prompt"] = prompt_text
            # Check for metadata lines
            elif line.strip().startswith("# user:"):
                # Split the meta lines into key-value pairs based on ':'
                meta_matches = re.findall(r"(\w+):(\S+)", line)
                for k, v in meta_matches:
                    block_dict[k.lower()] = v
            else:
                # Process token lines
                parts = re.split(r"\s{2,}", line.strip())
                token_record = {
                    "token_id": parts[0],
                    "token": parts[1],
                    "token_pos": parts[2],
                    "token_morph": parts[3],
                    "token_dep_label": parts[4],
                    "token_edges": parts[5],
                }
                if len(parts) > 6:
                    token_record["token_wrong"] = parts[6]
                block_dict.setdefault("tokens", []).append(token_record)

        # Append the processed block dictionary to the list
        list_of_blocks.append(block_dict)

    dat = pd.json_normalize(
        data=list_of_blocks,
        record_path="tokens",
        meta=[
            "block_id",
            "prompt",
            "user",
            "countries",
            "days",
            "client",
            "session",
            "format",
            "time",
        ],
        errors="ignore",
    )

    # Check whether input_path is dev or test/if so, merge in the key file
    if input_path.name in ['dev', 'test']:
        key_dat = pd.read_csv(
            input_path.parent / (input_path.name + ".key"),
            sep=" ",
            header=None,
            names=["token_id", "token_wrong"]
        )
        dat = dat.merge(
            key_dat, on="token_id",
            how="left",
            validate="one_to_one"
        )

    # Add the language column based on the filename
    dat["l2"], dat["l1"] = input_path.name.split(".", 1)[0].split("_")

    # Handle data types
    dat["days"] = pd.to_numeric(dat["days"], errors="coerce")
    dat["time"] = pd.to_numeric(dat["time"].replace("null", ""), errors="coerce")
    if "token_wrong" in dat.columns:
        dat["token_wrong"] = dat["token_wrong"].astype(int)

    return dat

def concat_and_write(paths, out_path: Path, parse_fn=parse_slam) -> pd.DataFrame:
    """Parse a list of SLAM files, concatenate them, write to parquet, and return the DataFrame.

    paths: iterable of Path objects pointing to SLAM input files
    out_path: Path where the combined parquet will be written
    parse_fn: callable that accepts a Path and returns a pandas.DataFrame (defaults to parse_slam)
    """
    dats = [parse_fn(p) for p in paths]
    all_dats = pd.concat(dats, ignore_index=True)
    all_dats.to_parquet(out_path, index=False)
    print(f"Wrote {len(all_dats)} rows to {out_path}")
    return None

def main():
    train_path = [
        Path("data/raw/data_fr_en/fr_en.slam.20190204.train"),
        Path("data/raw/data_en_es/en_es.slam.20190204.train"),
        Path("data/raw/data_es_en/es_en.slam.20190204.train"),
    ]
    test_path = [
        Path("data/raw/data_fr_en/fr_en.slam.20190204.test"),
        Path("data/raw/data_en_es/en_es.slam.20190204.test"),
        Path("data/raw/data_es_en/es_en.slam.20190204.test"),
    ]
    dev_path = [
        Path("data/raw/data_fr_en/fr_en.slam.20190204.dev"),
        Path("data/raw/data_en_es/en_es.slam.20190204.dev"),
        Path("data/raw/data_es_en/es_en.slam.20190204.dev"),
    ]
    # Concatenate and write train/test/dev datasets
    concat_and_write(train_path, Path("data/processed/train_slam.parquet"))
    concat_and_write(test_path, Path("data/processed/test_slam.parquet"))
    concat_and_write(dev_path, Path("data/processed/dev_slam.parquet"))

if __name__ == "__main__":
    main()


