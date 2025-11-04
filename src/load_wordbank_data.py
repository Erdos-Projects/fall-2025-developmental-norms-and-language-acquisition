import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import math

def clean_token_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new column 'token_clean' by copying the 'token' column,
    converting all entries to lowercase, and removing all occurrences of '*', '.', '!', and '¡'.

    Args:
        df (pd.DataFrame): The input DataFrame which must contain a 'token' column.

    Returns:
        pd.DataFrame: The DataFrame with the new 'token_clean' column added.
    """

    # Convert to lowercase and remove all asterisks in one chained operation
    df['token_clean'] = df['token'].str.lower().str.replace(r'[\*!¡\.]', '', regex=True)
    return df


def add_lemma_data(df_main, df_lemmas):
    """
    Merges the 'uni_lemma' column from the lookup table into the main DataFrame
    and reorders the columns so 'uni_lemma' is the third column (after item_id 
    and item_definition).

    Args:
        df_main (pd.DataFrame): The main DataFrame (e.g., df_WS or df_WG).
        df_lemmas (pd.DataFrame): The lemma mapping table (e.g., uni_lemmas_WS or uni_lemmas_WG).
            This file should have 2 columns: 'item_definition' and 'uni_lemma'.

    Returns:
        pd.DataFrame: The updated DataFrame with the new column.
    """

    # Perform a Left Merge
    # - 'left': df_main (the original acquisition data)
    # - 'right': df_lemmas (the table containing the uni_lemma to add)
    # - 'on': 'item_definition' (the common key)
    # - 'how': 'left' (ensures all rows from df_WS are kept, even if no match is found)
    df_main = pd.merge(
        df_main, 
        df_lemmas, 
        on='item_definition', 
        how='left'
    )

    # Print a concise summary instead of dumping the whole DataFrame/Series
    if 'inventory' in df_main.columns:
        try:
            inv_vals = [str(v) for v in pd.unique(df_main['inventory'])]
        except Exception:
            inv_vals = ['<unknown>']
    else:
        inv_vals = ['<unknown>']

    # print(f"--- inventory: {', '.join(inv_vals)} — DataFrame updated ---")
    # print(f"Columns now: {df_main.columns.tolist()}")
    return df_main

def split_row(df, old_token, new_tokens, new_lemmas):
    """
    Splits a single row in the DataFrame into two new rows based on new item definitions
    and assigned uni_lemma values. This is used for cases where a single Wordbank item
    maps to multiple concepts (e.g., "inside/in").

    Args:
        df (pd.DataFrame): The original DataFrame containing the row to be split.
        old_token (str): The 'token' value of the row to be split.
        new_tokens (tuple): A tuple containing the two new 'token' 
            values (e.g., ('inside', 'in')).
        new_lemmas (tuple): A tuple containing the two new 'uni_lemma' values corresponding 
            to the new token.

    Returns:
        pd.DataFrame: The updated DataFrame with the original row removed and the 
            two split rows added.
    """

    # --- 1. Isolate the row and create copies ---
    # Isolate the target row
    original_row = df[df['token'] == old_token].copy()

    # If nothing matched, return original df unchanged
    if original_row.empty:
        print(f"split_row: no rows found for '{old_token}'; no changes made.")
        return df

    # Assert at most one match to avoid ambiguous behavior
    assert original_row.shape[0] <= 1, f"split_row: expected at most one match for '{old_token}', found {original_row.shape[0]}"

    # Create new row 1 (deepcopy is a safe practice)
    new_def1, new_def2 = new_tokens
    new_lemma1, new_lemma2 = new_lemmas
    new_row1 = original_row.copy()
    new_row1['token'] = new_def1
    new_row1['uni_lemma'] = new_lemma1
    # Create new row 2
    new_row2 = original_row.copy()
    new_row2['token'] = new_def2
    new_row2['uni_lemma'] = new_lemma2

    # --- 2. Remove the original row from the main DataFrame ---

    # Filter out the old row.
    df = df[df['token'] != old_token].copy()
    # --- 3. Concatenate the filtered DF with the two new rows ---

    # Combine the filtered data with the two new rows
    df = pd.concat(
        [df, new_row1, new_row2],
        ignore_index=True  # Optional: resets the index
    )
    n_matched = original_row.shape[0]
    # print(f"split_row: split '{old_item_definition}' into '{new_def1}' and '{new_def2}' ({n_matched} rows expanded).")
    return df

def autosplit_slashes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits rows containing '/' in 'token_clean' into multiple rows using 
    pandas.explode.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame expanded based on '/' delimiters.
    """
    if df.empty:
        return df.copy()

    # Separate rows that need splitting from those that don't
    mask_slash = df['token_clean'].str.contains('/')
    df_slash_split = df[mask_slash].copy()
    df_non_slash = df[~mask_slash].copy()

    # 1b. Apply split and explode
    if not df_slash_split.empty:
        df_slash_split['token_clean'] = df_slash_split['token_clean'].str.split('/')
        df_slash_split = df_slash_split.explode('token_clean')

    # Recombine and return
    return pd.concat([df_non_slash, df_slash_split], ignore_index=True)


def autosplit_se(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits rows ending in '(se)' into two new rows: 
    1. One with the '(se)' removed (e.g., 'quedar').
    2. One with '(se)' replaced by 'se' (e.g., 'quedarse').

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame expanded based on the '(se)' pattern.
    """
    if df.empty:
        return df.copy()

    # Identify rows ending in (se). Use regex to ensure correct matching.
    mask_se = df['token_clean'].str.endswith(r'(se)', na=False)
    df_se_split = df[mask_se].copy()
    df_se_kept = df[~mask_se].copy()

    if not df_se_split.empty:
        # Create the first set: '(se)' is removed (e.g., 'quedar')
        df_se_1 = df_se_split.copy()
        # Use regex to replace (se) at the end of the string with an empty string
        df_se_1['token_clean'] = df_se_1['token_clean'].str.replace(r'\(se\)$', '', regex=True)

        # Create the second set: '(se)' is replaced by 'se' (e.g., 'quedarse')
        df_se_2 = df_se_split.copy()
        # Use regex to replace (se) at the end of the string with 'se'
        df_se_2['token_clean'] = df_se_2['token_clean'].str.replace(r'\(se\)$', 'se', regex=True)

        # Combine the rows that were kept, the first split set, and the second split set
        return pd.concat([df_se_kept, df_se_1, df_se_2], ignore_index=True)
    else:
        return df.copy()


def autosplit_rows(df: pd.DataFrame, exceptions=None) -> pd.DataFrame:
    """
    Orchestrates the row expansion process by handling exceptions and applying 
    slash and '(se)' splitting sequentially.
    
    Args:
        df (pd.DataFrame): The input DataFrame. Must contain 'token_clean' and 'uni_lemma' columns.
        exceptions (list, optional): A list of 'uni_lemma' strings to exclude from processing. 
                                     Defaults to None (no exceptions).

    Returns:
        pd.DataFrame: The fully expanded DataFrame.
    """

    if exceptions is None:
        exceptions = []   
    df_result = df.copy()
    
    # --- Step 0: Separate Exceptions ---
    # The change is here: check against 'uni_lemma' instead of 'token_clean'
    mask_exception = df_result['uni_lemma'].isin(exceptions)
    df_exceptions = df_result[mask_exception].copy()
    df_to_process = df_result[~mask_exception].copy()
    
    if df_to_process.empty:
        return df_exceptions.copy()

    # --- Step 1: Handle '/' Splitting ---
    df_step1_result = autosplit_slashes(df_to_process)
    
    # --- Step 2: Handle '(se)' Splitting ---
    df_step2_result = autosplit_se(df_step1_result)

    # --- Step 3: Final Recombination ---
    # Add the original exception rows back to the result
    df_final = pd.concat([df_step2_result, df_exceptions], ignore_index=True)
    
    return df_final

def remove_parentheticals(df):
    """
    Removes parenthetical content (including the preceding space) from the 
    'token_clean' column (e.g., 'pescado (food_drink)' becomes 'pescado').
    
    NOTE: This only removes parentheticals that are preceded by at least one space.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with cleaned 'token_clean' entries.
    """
    # Regex targets mandatory whitespace (\s+), literal opening parenthesis (\(), 
    # any characters inside ([^)]*), and literal closing parenthesis (\)).
    df['token_clean'] = df['token_clean'].str.replace(r'\s+\([^)]*\)', '', regex=True)
    return df


# This is the top-level function to read in and clean the Wordbank data
def read_and_clean_wordbank_data(path_to_wordbank_data, path_to_uni_lemma_data,
                                 lang, inventory, measure,
                                 autosplit=True, autosplit_exceptions=["child's own name"],
                                 items_to_split=None, cols_to_drop=None):
    '''
    Reads in the raw Wordbank data and uni_lemma mapping, performs initial cleaning,
    standardizes column names, and adds metadata (language, inventory, measure).

    Args:
        path_to_wordbank_data (str): The file path to the raw Wordbank data CSV 
            (containing item metadata and age-group data).
        path_to_uni_lemma_data (str): The file path to the uni_lemma mapping CSV 
            (must contain 'item_definition' and 'uni_lemma').
        lang (str): The language code (e.g., 'English', 'Spanish').
        inventory (str): The inventory type ('WG' for Words & Gestures or 'WS' for 
            Words & Sentences).
        measure (str): The specific measure used ('produces', 'understands', etc.).
        autosplit: Boolean specifying whether to automatically split rows in which 'token'
            contains '/' or '(se)'
        autosplit_exceptions: a list of strings to specify rows to avoid autosplitting.
            The string is passed to the 'uni_lemma' field.
        items_to_split (list of dict, optional): List of dictionaries defining rows 
            to split. Each dict must contain keys: 'old_token', 
            'new_tokens' (tuple), and 'new_lemmas' (tuple). Defaults to None.
        cols_to_drop (list of str, optional): A list of column names to remove from 
            the DataFrame (e.g., ['category']). Defaults to None.

    Returns:
        pd.DataFrame: The cleaned and updated Wordbank DataFrame with core columns 
            reordered to: ['item_id', 'l1', 'inventory', 'measure', 'uni_lemma', 'token'], 
            followed by the data columns.
    '''
    # Read in the Wordbank word data
    df = pd.read_csv(path_to_wordbank_data)
    # Read in the uni_lemma mapping data
    uni_lemmas = pd.read_csv(path_to_uni_lemma_data)
    # Add the uni_lemma column to the main dataframe and reorder columns
    df = add_lemma_data(df, uni_lemmas)
    # Rename
    df = df.rename(columns={'item_definition': 'token'})
    # Create token_clean column
    df = clean_token_entries(df)
    # Remove unnecessary columns
    if 'downloaded' in df.columns:
        df.drop(columns=['downloaded'], inplace=True)
    df['l1'] = lang
    df['inventory'] = inventory
    df['measure'] = measure
    # Mark which rows have non-unique uni_lemmas
    df['synonym'] = df.duplicated(subset=['uni_lemma'], keep=False)
    # Split any rows as specified
    if items_to_split:
        for item in items_to_split:
            old_token = item['old_token']
            new_tokens = item['new_tokens']
            new_lemmas = item['new_lemmas']
            df = split_row(df, old_token, new_tokens, new_lemmas)
    # Remove parenthetical clarifications in token_clean column
    df = remove_parentheticals(df)
    # Auto-split rows where token_clean has '/' or '(se)'
    if autosplit:
        df = autosplit_rows(df, exceptions=autosplit_exceptions)
    # Drop any specified columns
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # Reorder Columns
    # Define the desired core columns in order
    core_cols = ['item_id', 'l1', 'inventory', 'measure', 'uni_lemma', 'token', 'token_clean', 'synonym']
    # Filter core_cols to only include those present in the main DataFrame
    present_core_cols = [col for col in core_cols if col in df.columns]
    # Get all columns that are NOT the core columns, maintaining their original order
    original_cols = df.columns.tolist()
    other_cols = [col for col in original_cols if col not in present_core_cols]
    # Create and apply the final desired column order
    new_column_order = present_core_cols + other_cols
    df = df[new_column_order]
    return df