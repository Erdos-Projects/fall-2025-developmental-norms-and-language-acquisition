# Create logistic fits for Wordbank data
printf '%.0s=' {1..80}; echo ""
echo "Making logistic fits for Wordbank data"
printf '%.0s=' {1..80}; echo ""

python "src/data/make_wordbank_fits_en.py"
python "src/data/make_wordbank_fits_es.py"

# Load SLAM data and add features from Wordbank
printf '%.0s=' {1..80}; echo ""
echo "Processing SLAM data and adding Wordbank features"
printf '%.0s=' {1..80}; echo ""

python "src/data/process_slam.py"
python "src/data/process_tokens.py"
python "src/data/process_add_features.py"
python "src/data/add_metadata.py"

# Subset and add more engineered features to data
printf '%.0s=' {1..80}; echo ""
echo "Adding edit distance and word frequency features"
printf '%.0s=' {1..80}; echo ""

python "src/models/dataset_subsets.py"
python "src/models/edit_distance.py"

# Inferential analysis
printf '%.0s=' {1..80}; echo ""
echo "Performing inferential analysis"
printf '%.0s=' {1..80}; echo ""

python "src/models/inference_final.py"

# Fit predictive models
printf '%.0s=' {1..80}; echo ""
echo "Fitting predictive models:"
echo "Logistic and Histogram-based Gradient Boosting"
printf '%.0s=' {1..80}; echo ""

python "src/models/LG_model.py"
python "src/models/HGB_model.py"

printf '%.0s=' {1..80}; echo ""
echo "DONE"
printf '%.0s=' {1..80}; echo ""
