import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import math
import hashlib

# 2-parameter sigmoid function for logistic fits
def sigmoid(age, k, x0):
    """
    k: Growth rate.
    x0: Inflection point / median_aoa.
    """
    return 1 / (1 + np.exp(-k * (age - x0)))

# --- 2. Helper Functions (Reused from previous steps) ---

def row_to_df_for_fit(row_data):
    """
    Transforms a single wide-format row into a clean long-format DataFrame.
    """
    if isinstance(row_data, pd.Series):
        df_row = row_data.to_frame().T
    else:
        df_row = row_data

    # Detect age columns: prefer column names that are purely numeric (e.g., '8','16',...)
    # This avoids accidentally treating metadata columns like 'inventory' or 'measure'
    # as age columns when melting.
    age_cols = [c for c in df_row.columns if str(c).isdigit()]
    if not age_cols:
        # Fallback: exclude common metadata columns if no numeric-named columns are found.
        EXCLUDE_COLS = ['item_id', 'uni_lemma', 'item_definition', 'category', 'inventory', 'measure']
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
    METADATA_COLS = ['inventory', 'measure', 'uni_lemma'] 
    for meta_col in METADATA_COLS:
        if meta_col in df_row.columns and meta_col not in row_df.columns:
            meta_val = df_row.iloc[0].get(meta_col)
            # Ensure the value is correctly propagated to all long rows
            row_df[meta_col] = meta_val 

    return row_df.reset_index(drop=True)

def calculate_sigmoid_params(df_combined):
    """
    Fits the sigmoid curve to the combined long-format data. 
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

def plot_acquisition_curve(ax, word, df_data, k_fit, x0_fit):
    """
    Generates a scatter plot of the raw data, overlays the fitted logistic curve,
    and adds median AoA and 50% lines onto the provided Axes (ax) object.
    """
    
    # Assume columns are lowercase 'inventory' and 'measure' as requested.
    assert ('inventory' in df_data.columns) and ('measure' in df_data.columns)
 
    label_col = '_inv_measure'
    df_data[label_col] = (
        df_data['inventory'].fillna('').astype(str) + ' | ' + df_data['measure'].fillna('').astype(str)
    )

    # Preserve order of first appearance
    label_vals = list(dict.fromkeys(df_data[label_col].dropna().astype(str).tolist()))

    # Create a stable mapping from label string -> color using a hashed index into
    # a larger palette. This guarantees different label strings map to different
    # colors deterministically.
    # Define custom 3-color palette (Blue, Orange, Purple)
    #CUSTOM_COLORS = ['#0000FF', '#FFA500', '#9467bd']
    CUSTOM_COLORS = ['#0000FF', '#FFA500', '#9e1cd6']
    
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


def compute_curve_fits(dfs, match_col='uni_lemma'):
    """
    Compute sigmoid fits for every row in the primary DataFrame (dfs[0]) and combine
    matching rows from other DataFrames in the list `dfs` using `match_col`.

    Args:
        dfs (list[pd.DataFrame]): list of DataFrames where dfs[0] is the primary (base) DF.
        match_col (str): column name used to match rows across DataFrames (default 'uni_lemma').

    Returns:
        pd.DataFrame: columns [match_col, 'growth_rate', 'median_aoa', '__plot_data__']
    """
    if not isinstance(dfs, (list, tuple)) or len(dfs) == 0:
        raise ValueError('dfs must be a non-empty list of DataFrames')

    primary = dfs[0]
    others = dfs[1:]

    # Build lookup dicts for each other DF mapping match value -> row series
    other_dicts = []
    for odf in others:
        if match_col not in odf.columns:
            other_dicts.append({})
        else:
            other_dicts.append(odf.set_index(match_col).T.to_dict('series'))

    # Nested function used by apply()
    def combined_logistic_regression(primary_row):
        match_val = primary_row.get(match_col)
        row_primary_long = row_to_df_for_fit(primary_row)

        # Collect matching rows from other sources; rely on each input DF's
        # existing 'inventory' column instead of creating synthetic tags.
        combined_parts = [row_primary_long]
        for odict in other_dicts:
            if match_val in odict:
                other_series = odict[match_val]
                other_long = row_to_df_for_fit(other_series)
                combined_parts.append(other_long)

        row_df_combined = pd.concat(combined_parts, ignore_index=True)

        # Fit the curve
        fit_params = calculate_sigmoid_params(row_df_combined)

        # Attach plot data
        fit_params['__plot_data__'] = row_df_combined
        return fit_params

    # Run the fit across primary rows
    results = primary.apply(combined_logistic_regression, axis=1)

    # Build the results DataFrame keyed by the matching column
    df_curve_fits = primary[[match_col]].copy()
    for col in ['token', 'l1', 'category']:
        df_curve_fits[col] = primary[col]
    df_curve_fits['growth_rate'] = results['growth_rate']
    df_curve_fits['median_aoa'] = results['median_aoa']
    df_curve_fits['__plot_data__'] = results['__plot_data__']

    return df_curve_fits


def plot_curve_fits(df_curve_fits, cols=6, figsize_scale=(3.5, 3)):
    """
    Plot the curve fits stored in df_curve_fits produced by compute_curve_fits.

    df_curve_fits must contain columns: 'uni_lemma', 'token', 'growth_rate', 'median_aoa', '__plot_data__'.
    """
    valid_fits = df_curve_fits[~pd.isna(df_curve_fits['growth_rate'])]
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
        word = row['token'] # title of plot is token
        k_fit = row['growth_rate']
        x0_fit = row['median_aoa']
        df_plot_data = row['__plot_data__']

        # Use the simple 1D index to access the correct subplot
        ax = axes[plot_index]

        # Plot the curve using the current axis
        plot_acquisition_curve(ax, word, df_plot_data, k_fit, x0_fit)

        plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, ROWS * COLS):
        axes[i].axis('off')

    # Add a title for the entire figure and adjust layout
    fig.suptitle('Combined Acquisition Curve Fits', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust rect to make space for suptitle
    plt.show()

def compute_and_export_curve_fits(path_to_write, dfs, match_col='uni_lemma'):
    df_curve_fits = compute_curve_fits(dfs, match_col=match_col)
    df_for_export = df_curve_fits.drop(columns=['__plot_data__'])
    df_for_export.to_csv(path_to_write, index=False)
    return f"Curve fits written to {path_to_write}"