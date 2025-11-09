# Create logistic fits for Wordbank data
python "src/data/make_wordbank_fits_en.py"
python "src/data/make_wordbank_fits_es.py"
# Load SLAM data and add features from Wordbank
python "src/data/process_slam.py"
python "src/data/process_tokens.py"
python "src/data/process_add_features.py"
python "src/data/add_metadata.py"
# Subset and add more engineered features to data
python "src/models/dataset_subsets.py"
python "src/models/edit_distance.py"
# Inferential analysis
python "src/models/inference_final.py"
# Fit predictive models
python "src/models/LG_model.py"
python "src/models/HGB_model.py"
