# SLAM Data Schema

| Column Name     | Data Type | Nullable | Description |
|-----------------|-----------|----------|-------------|
| token_id | int64 | No | Unique identifier for each token |
| token | str | No | The token text |
| token_pos | str | Yes | Part-of-speech tag for the token |
| token_morph | str | Yes | Morphological features of the token |
| token_dep_label | str | Yes | Dependency label for the token |
| token_edges | str | Yes | Edges associated with the token in dependency graph |
| token_correct | bool | Yes | Whether the token is correct (True/False) |
| block_id | int64 | No | Sequential identifier for the block within each file |
| prompt | str | Yes | Prompt text |
| prompt_length | int64 | Yes | Length of the prompt |
| token_length | int64 | Yes | Length of the token |
| user | str | Yes | User identifier |
| countries | str | Yes | Countries associated with the user |
| days | int64 | Yes | Number of days |
| client | str | Yes | Client identifier |
| session | str | Yes | Session identifier |
| format | str | Yes | Format of the data |
| time | datetime64[ns] | Yes | Timestamp of the token |
| language | str | Yes | Language of the token |
