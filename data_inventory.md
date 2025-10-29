# SLAM Data Schema

| Column Name     | Data Type | Nullable | Description |
|-----------------|-----------|----------|-------------|
| token_id | str | No | Unique identifier for each token |
| token | str | No | The token text |
| token_pos | str | Yes | Part-of-speech tag for the token |
| token_morph | str | Yes | Morphological features of the token |
| token_dep_label | str | Yes | Dependency label for the token |
| token_edges | str | Yes | Edges associated with the token in dependency graph |
| token_wrong | int64 | Yes | Whether the token is wrong (1/0) |
| block_id | int64 | No | Sequential identifier for the block within each file |
| prompt | str | Yes | Prompt text |
| user | str | Yes | User identifier |
| countries | str | Yes | Countries associated with the user |
| days | float64 | Yes | Days (fraction) relative to first session |
| client | str | Yes | Client identifier |
| session | str | Yes | Session identifier |
| format | str | Yes | Format of the data |
| time | float64 | Yes | Time taken for the token in seconds |
| l2 | str | Yes | Second language of the user |
| l1 | str | Yes | First language of the user |
