from load_wordbank_data import read_and_clean_wordbank_data
from wordbank_logistic_fits import compute_curve_fits, plot_curve_fits, compute_and_export_curve_fits

# create WS, WG, WG_U tables using load_wordbank_data.py functions
split_dict_WS = {'old_token':'inside/in',
              'new_tokens':['inside', 'in'],
              'new_lemmas':['inside', 'in']
              }
df_WS = read_and_clean_wordbank_data('data/raw/Wordbank/wordbank_data_WS_Produces_en_US.csv',
                                        'data/raw/Wordbank/uni_lemmas_WS_en_US.csv',
                                        lang='en', inventory='WS', measure='Produces',
                                        items_to_split=[split_dict_WS], cols_to_drop=['28'] # drop 28 month outlier
                                      )
df_WG = read_and_clean_wordbank_data('data/raw/Wordbank/wordbank_data_WG_Produces_en_US.csv',
                                        'data/raw/Wordbank/uni_lemmas_WG_en_US.csv',
                                        lang='en', inventory='WG', measure='Produces',
                                      )

df_WG_U = read_and_clean_wordbank_data('data/raw/Wordbank/wordbank_data_WG_Understands_en_US.csv',
                                        'data/raw/Wordbank/uni_lemmas_WG_en_US.csv',
                                        lang='en', inventory='WG', measure='Understands',
                                      )

# fits for Produces data
dfs = [df_WS, df_WG]
df_curve_fits_produces, df_ambiguous_produces = compute_curve_fits(dfs)

# fits for Understands data
dfs_U = [df_WG_U]
df_curve_fits_understands, df_ambiguous_understands = compute_curve_fits(dfs_U)

# export data to csv
compute_and_export_curve_fits('wordbank_en_logistic_fits.csv', [df_WS, df_WG], [df_WG_U])