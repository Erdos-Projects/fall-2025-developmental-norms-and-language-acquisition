# This script extracts specific vocabulary and uni_lemmas from the Wordbank database
# for American English (WG/WS) and Mexican Spanish (WG/WS), 
# in order to add a uni_lemmas column to the by-word
# Wordbank data to facilitate aggregation and translation

# --- Dependency Loading and Setup ---
library(wordbankr)
library(dplyr)
library(tidyr)

# In case of connection errors, make sure you're using the updated wordbankr library
# devtools::install("~/github/wordbankr")


# --- Function for Data Extraction and Export ---

# Helper function to fetch data, clean it, and export to a named CSV file.
fetch_clean_and_export <- function(language_name, form_name, filename) {
  message(paste("Processing:", language_name, form_name, "->", filename))
  
  # 1. Data Retrieval
  data <- wordbankr::get_item_data(
    language = language_name, 
    form = form_name
  )
  
  # 2. Data Cleaning and Selection
  # Select only the required columns and clean up the data.
  cleaned_data <- data %>%
    select(uni_lemma, item_definition) %>%
    filter(!is.na(uni_lemma) & !is.na(item_definition)) %>%
    distinct()
  
  # 3. EXPORT
  write.csv(cleaned_data, filename, row.names = FALSE)
}

# --- 4. Execution for all four requested datasets ---

# 4.1 American English Words and Gestures (WG)
fetch_clean_and_export(
  language_name = "English (American)", 
  form_name = "WG", 
  filename = "uni_lemmas_WG_en_US.csv"
)

# 4.2 American English Words and Sentences (WS)
fetch_clean_and_export(
  language_name = "English (American)", 
  form_name = "WS", 
  filename = "uni_lemmas_WS_en_US.csv"
)

# 4.3 Mexican Spanish Words and Gestures (WG)
fetch_clean_and_export(
  language_name = "Spanish (Mexican)", 
  form_name = "WG", 
  filename = "uni_lemmas_WG_es_MX.csv"
)

# 4.4 Mexican Spanish Words and Sentences (WS)
fetch_clean_and_export(
  language_name = "Spanish (Mexican)", 
  form_name = "WS", 
  filename = "uni_lemmas_WS_es_MX.csv"
)

# --- 5. FINAL STATUS ---

cat("\n--- CSV Export Status ---\n")
cat("Successfully extracted and exported four uni_lemma files:\n")
cat(" - uni_lemmas_WG_en_US.csv\n")
cat(" - uni_lemmas_WS_en_US.csv\n")
cat(" - uni_lemmas_WG_es_MX.csv\n")
cat(" - uni_lemmas_WS_es_MX.csv\n")
cat("Columns for all files: uni_lemma, item_definition\n")
