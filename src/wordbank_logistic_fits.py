import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import math
import hashlib
from typing import List, Tuple, Dict, Any, Union

def sigmoid(age, k, x0):
    """
    2-parameter sigmoid function used for logistic curve fitting.
    
    Args:
        age (float/np.array): The age in months (x-axis value).
        k (float): The growth rate (steepness) of the curve.
        x0 (float): The inflection point, corresponding to median Age of Acquisition (median_aoa).
        
    Returns:
        float/np.array: The fitted proportion (y-axis value).
    """
    return 1 / (1 + np.exp(-k * (age - x0)))

# --- 2. Helper Functions ---

def get_age_columns(df: pd.DataFrame) -> List[Union[int, str]]:
    """
    Helper to identify columns representing age/proportion data.
    Assumes age columns are columns that are either numeric types or whose names are purely digits.
    """
    return [col for col in df.columns if isinstance(col, (int, float)) or str(col).isdigit()]

def row_to_df_for_fit(row_data):
    """
    Transforms a single wide-format row (representing one item/uni_lemma) into a clean, 
    long-format DataFrame suitable for curve fitting.
    
    Args:
        row_data (pd.Series or pd.DataFrame): A single row of wide-format Wordbank data.
            It must contain columns named with digits (age in months) and optionally 
            metadata columns like 'inventory', 'measure', and 'uni_lemma'.

    Returns:
        pd.DataFrame: A long-format DataFrame with columns ['Age', 'Proportion Acquired', 
            'inventory', 'measure', 'uni_lemma']. Ages are ensured to be integers and sorted.
    """
    if isinstance(row_data, pd.Series):
        df_row = row_data.to_frame().T
    else:
        df_row = row_data

    # Detect age columns: prefer column names that are purely numeric (e.g., '8','16',...)
    # This avoids accidentally treating metadata columns like 'inventory' or 'measure'
    # as age columns when melting.
    age_cols = get_age_columns(df_row)
    if not age_cols:
        # Fallback: exclude common metadata columns if no numeric-named columns are found.
        EXCLUDE_COLS = ['item_id', 'uni_lemma', 'token', 'token_clean', 'category', 'inventory', 'measure']
        age_cols = [c for c in df_row.columns if c not in EXCLUDE_COLS]

    proportions_wide = df_row[age_cols]
    
    row_df = pd.melt(
        proportions_wide,
        value_vars=age_cols,
        var_name='Age',
        value_name='Proportion Acquired'
    )
    
    row_df = row_df.dropna(subset=['Proportion Acquired'])
    row_df['Age'] = row_df['Age'].astype(int)
    row_df = row_df.sort_values(by='Age')
    
    # Preserve 'inventory' and 'measure' (and other metadata)
    METADATA_COLS = ['inventory', 'measure', 'uni_lemma', 'token', 'token_clean'] 
    for meta_col in METADATA_COLS:
        if meta_col in df_row.columns and meta_col not in row_df.columns:
            meta_val = df_row.iloc[0].get(meta_col)
            # Ensure the value is correctly propagated to all long rows
            row_df[meta_col] = meta_val 

    return row_df.reset_index(drop=True)

def calculate_sigmoid_params(df_combined):
    """
    Fits the 2-parameter sigmoid curve (logistic regression) to the long-format data. 
    
    Args:
        df_combined (pd.DataFrame): Long-format DataFrame containing acquisition data. 
            Must contain columns 'Age' (x-values) and 'Proportion Acquired' (y-values).
            
    Returns:
        pd.Series: A Series with keys 'growth_rate' (k) and 'median_aoa' (x0). 
            Returns NaN for both if the fitting optimization fails (RuntimeError).
    """
    X = df_combined['Age'].values
    Y = df_combined['Proportion Acquired'].values
    p0 = [0.5, X.mean() if X.size > 0 else 20] 

    try:
        popt, pcov = curve_fit(sigmoid, X, Y, p0=p0, maxfev=5000)
        return pd.Series(
            {'growth_rate': popt[0], 'median_aoa': popt[1]}
        )
    except RuntimeError:
        return pd.Series({'growth_rate': np.nan, 'median_aoa': np.nan})

# --- Plotting Function ---

def plot_acquisition_curve(ax, word, df_data, k_fit, x0_fit, colors=[]):
    """
    Generates a scatter plot of the raw data, overlays the fitted logistic curve,
    and adds median AoA and 50% lines onto the provided Axes (ax) object.
    
    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw the plot onto.
        word (str): The word (token) used for the plot title.
        df_data (pd.DataFrame): Long-format DataFrame containing the raw acquisition data. 
            Must contain 'Age', 'Proportion Acquired', 'inventory', and 'measure' columns.
        k_fit (float): The fitted growth rate (k) for the sigmoid curve.
        x0_fit (float): The fitted median Age of Acquisition (x0) for the sigmoid curve.
        
    Returns:
        None: The function modifies the passed Matplotlib Axes object in place.
    """
    
    # Assume columns are lowercase 'inventory' and 'measure' as requested.
    assert ('inventory' in df_data.columns) and ('measure' in df_data.columns)
 
    label_col = '_inv_measure'
    df_data[label_col] = (
        df_data['inventory'].fillna('').astype(str) + ' | ' + df_data['measure'].fillna('').astype(str)
    )

    # Preserve order of first appearance
    label_vals = list(dict.fromkeys(df_data[label_col].dropna().astype(str).tolist()))

    if colors:
        CUSTOM_COLORS = colors
    else:
        # Define custom 3-color palette (Blue, Orange, Purple)
        CUSTOM_COLORS = ['#0000FF', '#FFA500', '#9e1cd6']
    
    # Create a stable mapping from label string -> color using a hashed index into
    # a larger palette. This guarantees different label strings map to different
    # colors deterministically.
    if len(label_vals) <= len(CUSTOM_COLORS):
        # Use specific colors for 3 or fewer sources
        palette = {
            lbl: CUSTOM_COLORS[i] 
            for i, lbl in enumerate(label_vals)
        }
    else:
        # Fallback to the deterministic hashed palette for >3 sources
        big_palette = sns.color_palette('tab20', n_colors=20).as_hex()
        # Note: We rely on the first 3 colors of tab20 being distinct from the custom ones above
        
        def label_to_color(lbl):
            # Using 1-byte hash for simplicity and speed
            h = hashlib.md5(lbl.encode('utf8')).digest()
            idx = h[0] % len(big_palette)
            return big_palette[idx]

        palette = {lbl: label_to_color(lbl) for lbl in label_vals}

    sns.scatterplot(
        data=df_data,
        x='Age',
        y='Proportion Acquired',
        hue=label_col,
        palette=palette,
        hue_order=label_vals,
        s=40, # Smaller points for better visibility in a grid
        edgecolor='black',
        alpha=0.7,
        zorder=3,
        ax=ax # Pass the axis object to seaborn
    )

    # --- Generate and Plot Fitted Curve (Requirement 3) ---
    x_range = np.linspace(df_data['Age'].min() - 5, df_data['Age'].max() + 5, 100)
    y_fitted = sigmoid(x_range, k_fit, x0_fit)
    
    ax.plot(
        x_range, 
        y_fitted, 
        color='green', 
        linewidth=1.5, 
        label=f'Fitted Curve (k={k_fit:.2f})'
    )

    # --- Plot Vertical median_aoa Line (Requirement 4) ---
    ax.axvline(
        x=x0_fit,
        color='green',
        linestyle='--',
        linewidth=1,
        label=f'median_aoa ({x0_fit:.1f} mos)',
        alpha=0.7
    )
    
    # --- Plot Horizontal 50% Acquisition Line (Requirement 5) ---
    ax.axhline(
        y=0.5, 
        color='red', 
        linestyle='--', 
        linewidth=1, 
        label='50% Threshold',
        alpha=0.7
    )

    # --- Customization ---
    ax.set_title(f'{word}', fontsize=10)
    ax.set_xlabel('Age (Months)', fontsize=8)
    ax.set_ylabel('Proportion Acquired', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(df_data['Age'].min() - 2, df_data['Age'].max() + 2)
    ax.grid(axis='both', linestyle=':', alpha=0.5)
    
    # Remove the legend from each subplot to keep the grid clean
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def compute_curve_fits(dfs, match_cols=['uni_lemma', 'token_clean']):
    """
    Compute sigmoid fits for every row in the primary DataFrame (dfs[0]) and combine
    matching rows from other DataFrames in the list `dfs` using `match_col`.

    This function identifies and returns a separate DataFrame for primary rows that 
    were fitted using only their own data and are 'ambiguous'—meaning the uni_lemma 
    was present in an auxiliary data frame but the token did not strictly match (uni_lemma, token_clean).

    Args:
        dfs (list[pd.DataFrame]): list of DataFrames where dfs[0] is the primary (base) DF.
        match_col (str): column name used to match rows across DataFrames (default 'uni_lemma').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - The main fitted DataFrame (df_curve_fits).
            - A DataFrame of primary rows that are considered 'ambiguous'.
    """
    if not isinstance(dfs, (list, tuple)) or len(dfs) == 0:
        raise ValueError('dfs must be a non-empty list of DataFrames')

    primary = dfs[0]
    others = dfs[1:]

    # Convert match_cols to a list of keys for consistent handling
    if isinstance(match_cols, str):
        match_keys = [match_cols]
    elif isinstance(match_cols, (list, tuple)):
        match_keys = list(match_cols)
    else:
        raise TypeError("match_cols must be a string or a list/tuple of strings.")

    # Build lookup dicts for each other DF mapping match value -> row series (STRICT MATCH)
    other_dicts = []
    # Build sets of all uni_lemmas present in each auxiliary DF (AMBIGUITY CHECK)
    uni_lemma_sets = [] 
    for odf in others:
        # 1. Strict Match Dict (for combining data)
        if all(col in odf.columns for col in match_keys):
            odf_indexed = odf.set_index(match_keys)
            other_dicts.append(odf_indexed.T.to_dict('series'))
        else:
            other_dicts.append({})
        # 2. Uni_Lemma Set (for ambiguity check)
        uni_lemma_sets.append(set(odf['uni_lemma'].unique()))

    # Nested function used by apply()
    def combined_logistic_regression(primary_row):
        # Extract the key value(s) from the primary row to use for strict lookup
        if len(match_keys) == 1:
            match_val = primary_row.get(match_keys[0])
        else:
            # For a multi-index, the key is a tuple of values
            match_val = tuple(primary_row.get(k) for k in match_keys)
            
        row_primary_long = row_to_df_for_fit(primary_row)

        # Collect matching rows from other sources (STRICT MATCH: uni_lemma + token)
        combined_parts = [row_primary_long]
        strict_match_found = False
        for odict in other_dicts:
            if match_val in odict:
                other_series = odict[match_val]
                other_long = row_to_df_for_fit(other_series)
                combined_parts.append(other_long)
                strict_match_found = True # At least one strict match was found
        
        # Flag if the fit was performed only with the primary row's data
        only_self_fit = not strict_match_found

        # --- AMBIGUOUS LOGIC ---
        ambiguous_fit = False
        # Only check for ambiguity if no strict match was found
        if only_self_fit:
            primary_uni_lemma = primary_row.get('uni_lemma')
            
            # Check for uni_lemma presence in any of the auxiliary sets
            # An item is AMBIGUOUS if:
            # 1. No strict (uni_lemma + token) match was found (only_self_fit is True)
            # 2. BUT, the uni_lemma *was* found in an auxiliary DF
            uni_lemma_match_found = any(primary_uni_lemma in ul_set for ul_set in uni_lemma_sets)
            
            if uni_lemma_match_found:
                ambiguous_fit = True
        
        row_df_combined = pd.concat(combined_parts, ignore_index=True)

        # Fit the curve
        fit_params = calculate_sigmoid_params(row_df_combined)

        # Attach plot data and status flag
        fit_params['__plot_data__'] = row_df_combined
        fit_params['__is_ambiguous__'] = ambiguous_fit # New flag for internal tracking
        # We don't need __only_self_fit__ anymore, only the final ambiguity status
        return fit_params

    # Run the fit across primary rows
    results = primary.apply(combined_logistic_regression, axis=1)

    # Determine all columns needed for the results DataFrame
    cols_to_keep = list(set(match_keys + ['token', 'token_clean', 'l1', 'category']))
    
    # Build the results DataFrame
    df_curve_fits = primary[cols_to_keep].copy()
    
    # Add fit results and internal flag
    df_curve_fits['growth_rate'] = results['growth_rate']
    df_curve_fits['median_aoa'] = results['median_aoa']
    df_curve_fits['__plot_data__'] = results['__plot_data__']
    df_curve_fits['__is_ambiguous__'] = results['__is_ambiguous__']
    
    # 5. Identify and create the ambiguous rows DataFrame
    df_ambiguous = df_curve_fits[df_curve_fits['__is_ambiguous__']].copy()

    # 6. Printing about ambiguous rows
    if not df_ambiguous.empty:
        print(f"\n--- Warning: {len(df_ambiguous)} primary rows were fitted using only their own data but had uni_lemma matches in auxiliary DFs. ---")
        print(df_ambiguous[match_keys + ['growth_rate', 'median_aoa']].to_string(index=True))
        print("------------------------------------------------------------------------------------------------")

    # 7. Clean up the flag column before return (it's for internal use/logging)
    df_curve_fits.drop(columns=['__is_ambiguous__'], inplace=True)
    df_ambiguous.drop(columns=['__is_ambiguous__'], inplace=True, errors='ignore')

    return df_curve_fits, df_ambiguous


def plot_curve_fits(df_curve_fits, cols=6, figsize_scale=(3.5, 3), colors=[]):
    """
    Plot the curve fits stored in df_curve_fits produced by compute_curve_fits.

    df_curve_fits must contain columns: 'uni_lemma', 'token', 'token_clean', 'growth_rate', 'median_aoa', '__plot_data__'.
    """
    valid_fits = df_curve_fits[~pd.isna(df_curve_fits['growth_rate'])]
    valid_fits = valid_fits.sort_values(by='token_clean') # sort so plots are alphabetically ordered
    num_plots = len(valid_fits)
    COLS = cols
    ROWS = math.ceil(num_plots / COLS)

    # Set the overall figure size (adjust as needed for readability)
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * figsize_scale[0], ROWS * figsize_scale[1]))

    # Flatten the axes array for simplified, reliable indexing
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    else:
        axes = axes.ravel()

    plot_index = 0
    for index, row in valid_fits.iterrows():
        word = row['token_clean'] # title of plot is token_clean
        k_fit = row['growth_rate']
        x0_fit = row['median_aoa']
        df_plot_data = row['__plot_data__']

        # Use the simple 1D index to access the correct subplot
        ax = axes[plot_index]

        # Plot the curve using the current axis
        plot_acquisition_curve(ax, word, df_plot_data, k_fit, x0_fit, colors=colors)

        plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, ROWS * COLS):
        axes[i].axis('off')

    # Add a title for the entire figure and adjust layout
    fig.suptitle('Combined Acquisition Curve Fits', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust rect to make space for suptitle
    plt.show()

def combine_measures(df_curve_fits_produces: pd.DataFrame, df_curve_fits_understands: pd.DataFrame) -> pd.DataFrame:
    """
    Combines two curve-fit DataFrames by merging the growth_rate and median_aoa
    columns from the 'understands' DataFrame into the 'produces' DataFrame.

    The columns from the 'understands' DataFrame are renamed to '_u' suffixes.

    Args:
        df_curve_fits_produces: The primary DataFrame (left side of merge).
        df_curve_fits_understands: The secondary DataFrame (right side of merge, source of new columns).

    Returns:
        pd.DataFrame: The combined DataFrame with new columns: 
                      'growth_rate_u' and 'median_aoa_u'.
    """

    # Rename columns in a copy of the 'produces' DataFrame to include the '_p' suffix.
    df_produces_renamed = df_curve_fits_produces.copy()
    df_produces_renamed.rename(columns={
        'growth_rate': 'growth_rate_p',
        'median_aoa': 'median_aoa_p'
    }, inplace=True)

    # Select key column and the two value columns we need.
    df_understands_subset = df_curve_fits_understands[['uni_lemma', 'growth_rate', 'median_aoa']].copy()

    # Rename the columns to to include the '_u' suffix.
    df_understands_subset.rename(columns={
        'growth_rate': 'growth_rate_u',
        'median_aoa': 'median_aoa_u'
    }, inplace=True)

    # Use a LEFT MERGE to join the two DataFrames on the 'uni_lemma' column.
    # 'how=left' ensures all rows from df_curve_fits_produces are kept.
    df_curve_fits = df_produces_renamed.merge(
        df_understands_subset,
        on='uni_lemma',  # The common column used for matching
        how='left'       # Keep all rows from the left DF (produces)
    )
    return df_curve_fits

def compute_and_export_curve_fits(path_to_write, dfs_p, dfs_u, match_cols=['uni_lemma', 'token_clean']):
    """
    Computes logistic curve fits and exports the results to a CSV file.

    Args:
        path_to_write (str): The file path where the resulting CSV will be saved.
        dfs_p (list[pd.DataFrame]): List of DataFrames passed to compute_curve_fits whose measure is Produces.
        dfs_u (list[pd.DataFrame]): List of DataFrames passed to compute_curve_fits whose measure is Understands.
        match_col (str): The column used to match items across the input DataFrames (default 'uni_lemma').

    Returns:
        str: A message confirming the file was written, including the file path.
    """
    df_curve_fits_p, df_ambiguous_p = compute_curve_fits(dfs_p, match_cols=match_cols)
    df_curve_fits_u, df_ambiguous_u = compute_curve_fits(dfs_u, match_cols=match_cols)
    df_curve_fits_p.drop(columns=['__plot_data__'], inplace=True)
    df_curve_fits_u.drop(columns=['__plot_data__'], inplace=True)
    df_curve_fits = combine_measures(df_curve_fits_p, df_curve_fits_u)
    df_curve_fits.to_csv(path_to_write, index=False)
    return f"Curve fits written to {path_to_write}"