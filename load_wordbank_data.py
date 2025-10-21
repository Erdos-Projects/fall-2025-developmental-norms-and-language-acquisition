import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import math

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

def split_row(df, old_item_definition, new_item_definitions, new_lemmas):
    """
    Splits a row in the DataFrame into two new rows based on new item definitions.
    Args:
        df (pd.DataFrame): The original DataFrame containing the row to be split.
        old_item_definition (str): The item_definition value of the row to be split.
        new_item_definitions (tuple): A tuple containing the two new item_definition values.
    Returns:
        pd.DataFrame: The updated DataFrame with the split rows.
    E.g., split the "inside/in" row into two rows, one for "inside" and one for "in".
    """

    # --- 1. Isolate the row and create copies ---
    # Isolate the target row
    original_row = df[df['item_definition'] == old_item_definition].copy()

    # If nothing matched, return original df unchanged
    if original_row.empty:
        # print(f"split_row: no rows found for '{old_item_definition}'; no changes made.")
        return df

    # Assert at most one match to avoid ambiguous behavior
    assert original_row.shape[0] <= 1, f"split_row: expected at most one match for '{old_item_definition}', found {original_row.shape[0]}"

    # Create new row 1 (deepcopy is a safe practice)
    new_def1, new_def2 = new_item_definitions
    new_lemma1, new_lemma2 = new_lemmas
    new_row1 = original_row.copy()
    new_row1['item_definition'] = new_def1
    new_row1['uni_lemma'] = new_lemma1
    # Create new row 2
    new_row2 = original_row.copy()
    new_row2['item_definition'] = new_def2
    new_row2['uni_lemma'] = new_lemma2

    # --- 2. Remove the original row from the main DataFrame ---

    # Filter out the old row.
    df = df[df['item_definition'] != old_item_definition].copy()
    # --- 3. Concatenate the filtered DF with the two new rows ---

    # Combine the filtered data with the two new rows
    df = pd.concat(
        [df, new_row1, new_row2],
        ignore_index=True  # Optional: resets the index
    )
    n_matched = original_row.shape[0]
    # print(f"split_row: split '{old_item_definition}' into '{new_def1}' and '{new_def2}' ({n_matched} rows expanded).")
    return df

# This is the top-level function to read in and clean the Wordbank data
def read_and_clean_wordbank_data(path_to_wordbank_data, path_to_uni_lemma_data,
                                 inventory, measure,
                                 items_to_split=None, cols_to_drop=None):
    '''
    Reads in the Wordbank data and the uni_lemma mapping data, cleans the data by removing
        unnecessary columns, adds the uni_lemma column, and reorders the columns.
        Args:
            path_to_wordbank_data (str): The file path to the Wordbank data CSV.
                This file should have metadata columns item_id, item_definition, category
                and data for each age group.
            path_to_uni_lemma_data (str): The file path to the uni_lemma mapping CSV.
                This file should have 2 columns: 'item_definition' and 'uni_lemma'.
            inventory (str): The inventory type ('WG' or 'WS').
        Returns:
            pd.DataFrame: The cleaned and updated Wordbank DataFrame.
    '''
    # Read in the Wordbank word data
    df = pd.read_csv(path_to_wordbank_data)
    # Remove unnecessary columns
    if 'downloaded' in df.columns:
        df.drop(columns=['downloaded'], inplace=True)
    df['inventory'] = inventory
    df['measure'] = measure
    # Read in the uni_lemma mapping data
    uni_lemmas = pd.read_csv(path_to_uni_lemma_data)
    # Add the uni_lemma column to the main dataframe and reorder columns
    df = add_lemma_data(df, uni_lemmas)
    # Split any rows as specified
    if items_to_split:
        for item in items_to_split:
            old_item_definition = item['old_item_definition']
            new_item_definitions = item['new_item_definitions']
            new_lemmas = item['new_lemmas']
            df = split_row(df, old_item_definition, new_item_definitions, new_lemmas)
    # Drop any specified columns
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    # Reorder Columns
    # Define the desired core columns in order
    core_cols = ['item_id', 'inventory', 'measure', 'uni_lemma', 'item_definition']
    # Filter core_cols to only include those present in the main DataFrame
    present_core_cols = [col for col in core_cols if col in df.columns]
    # Get all columns that are NOT the core columns, maintaining their original order
    original_cols = df.columns.tolist()
    other_cols = [col for col in original_cols if col not in present_core_cols]
    # Create and apply the final desired column order
    new_column_order = present_core_cols + other_cols
    df = df[new_column_order]
    return df