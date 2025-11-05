from load_wordbank_data import read_and_clean_wordbank_data
from wordbank_logistic_fits import compute_curve_fits, plot_curve_fits, compute_and_export_curve_fits
# create WS, WG, WG_U tables using load_wordbank_data.py functions

df_WS = read_and_clean_wordbank_data('data/raw/Wordbank/wordbank_data_WS_Produces_es_MX.csv',
                                        'data/raw/Wordbank/uni_lemmas_WS_es_MX.csv',
                                        lang='es', inventory='WS', measure='Produces',
                                        cols_to_drop=['30'] # drop 30 month outlier
                                      )
df_WG = read_and_clean_wordbank_data('data/raw/Wordbank/wordbank_data_WG_Produces_es_MX.csv',
                                        'data/raw/Wordbank/uni_lemmas_WG_es_MX.csv',
                                        lang='es', inventory='WG', measure='Produces',
                                      )
df_WG_U = read_and_clean_wordbank_data('data/raw/Wordbank/wordbank_data_WG_Understands_es_MX.csv',
                                        'data/raw/Wordbank/uni_lemmas_WG_es_MX.csv',
                                        lang='es', inventory='WG', measure='Understands',
                                      )

# fits for Produces data
dfs = [df_WS, df_WG]
df_curve_fits_produces, df_ambiguous_produces = compute_curve_fits(dfs)

# fits for Understands data
dfs_U = [df_WG_U]
df_curve_fits_understands, df_ambiguous_understands = compute_curve_fits(dfs_U)

# export data to csv
compute_and_export_curve_fits('data/processed/wordbank_es_logistic_fits.csv', [df_WS, df_WG], [df_WG_U])