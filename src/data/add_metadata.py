import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

# TODO: Check the names
slam_schema = DataFrameSchema({
    "token_id": Column(pa.Int, description="Unique identifier for each token"),
    "token": Column(pa.String, nullable=False, description="The token text"),
    "token_pos": Column(pa.String, nullable=True, description="Part-of-speech tag for the token"),
    "token_morph": Column(pa.String, nullable=True, description="Morphological features of the token"),
    "token_dep_label": Column(pa.String, nullable=True, description="Dependency label for the token"),
    "token_edges": Column(pa.String, nullable=True, description="Edges associated with the token in dependency graph"),
    "token_correct": Column(pa.Bool, nullable=True, description="Whether the token is correct (True/False)"),
    "block_id": Column(pa.Int, description="Sequential identifier for the block within each file"),
    "prompt": Column(pa.String, nullable=True, description="Prompt text"),
    "prompt_length": Column(pa.Int, Check.ge(0), nullable=True, description="Length of the prompt"),
    "token_length": Column(pa.Int, Check.ge(0), nullable=True, description="Length of the token"),
    "user": Column(pa.String, nullable=True, description="User identifier"),
    "countries": Column(pa.String, nullable=True, description="Countries associated with the user"),
    "days": Column(pa.Int, Check.ge(0), nullable=True, description="Number of days"),
    "client": Column(pa.String, nullable=True, description="Client identifier"),
    "session": Column(pa.String, nullable=True, description="Session identifier"),
    "format": Column(pa.String, nullable=True, description="Format of the data"),
    "time": Column(pa.DateTime, nullable=True, description="Timestamp of the token"),
    "language": Column(pa.String, nullable=True, description="Language of the token"),
})


def validate_slam_data(df):
    """Validate the DataFrame against the SLAM schema."""
    return slam_schema.validate(df)


def export_slam_schema_markdown(schema: DataFrameSchema):
    """Export the given DataFrameSchema to a markdown file called data_inventory.md."""
    out_path = "data_inventory.md"
    with open(out_path, "w") as f:
        f.write(f"# SLAM Data Schema\n\n")
        f.write(f"| Column Name     | Data Type | Nullable | Description |\n")
        f.write(f"|-----------------|-----------|----------|-------------|\n")
        for col_name, col in schema.columns.items():
            data_type = col.dtype
            nullable = "Yes" if col.nullable else "No"
            description = col.description if col.description else ""
            f.write(f"| {col_name} | {data_type} | {nullable} | {description} |\n")

if __name__ == "__main__":
    export_slam_schema_markdown(slam_schema)
    print("Exported SLAM schema to data_inventory.md")
