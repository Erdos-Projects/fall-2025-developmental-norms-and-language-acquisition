# This script retrieves and consolidates vocabulary lists from both WG and WS
# instruments for a standard dialect of English, Spanish, and French,
# creating a cross-language translation mapping table using the uni_lemma concept ID.

# --- Dependency Loading and Setup ---
library(wordbankr)
library(dplyr)
library(tidyr)

insts <- get_instruments()
# If the above breaks, make sure you're using the updated wordbankr library

# --- 1. Consolidated Vocabulary Retrieval (WG + WS) ---

# Helper function to fetch both WS and WG, combine, and clean up.
# This ensures maximum vocabulary coverage per language.
fetch_and_consolidate <- function(language_name, simple_label) {
  message(paste("Fetching:", language_name, "(WS & WG)"))
  
  # Fetch WS form
  ws_data <- wordbankr::get_item_data(language = language_name, form = "WS")
  
  # Fetch WG form
  wg_data <- tryCatch(
    wordbankr::get_item_data(language = language_name, form = "WG"),
    error = function(e) {
      # Return empty data frame if WG form is not available/fails to load
      data.frame(item_definition = character(), uni_lemma = character())
    }
  )

  # Combine WS and WG, and retain only unique uni_lemma/item_definition pairs
  combined_data <- bind_rows(ws_data, wg_data) %>%
    mutate(language_id = simple_label) %>%
    # Ensure every uni_lemma and word combination is unique, consolidating the lists
    distinct(language_id, item_definition, uni_lemma) %>%
    select(language_id, item_definition, uni_lemma)
  
  return(combined_data)
}

# 1.1 English (American) - The chosen pivot
english_instruments <- fetch_and_consolidate(
  language_name = "English (American)",
  simple_label = "English"
)

# 1.2 Spanish (Mexican) - The chosen representative dialect
spanish_instruments <- fetch_and_consolidate(
  language_name = "Spanish (Mexican)",
  simple_label = "Spanish"
)

# 1.3 French (Standard) - The chosen representative dialect
french_instruments <- fetch_and_consolidate(
  language_name = "French (French)",
  simple_label = "French"
)

# --- 2. Create the English Translation Pivot Table ---

# 2.1 Create the American English pivot table: uni_lemma -> English Word
english_pivot <- english_instruments %>%
  # Rename item_definition to clarify it is the standard English word
  rename(english_word = item_definition) %>%
  # FIX 1 REVISION: Instead of dropping words, collapse all English words for a concept
  # into a single semicolon-separated string. This ensures data integrity and prevents the join warning.
  group_by(uni_lemma) %>%
  dplyr::summarise(
      english_word = paste(unique(english_word), collapse = "; "),
      .groups = 'drop'
  ) %>%
  distinct(uni_lemma, english_word)

# 2.2 Combine ALL instruments into a single long table
all_instruments_long <- bind_rows(
  english_instruments,
  spanish_instruments,
  french_instruments
)

# 2.3 Join the full list back to the English pivot to get the translation equivalent
# This creates the final translation table.
cross_language_pivot <- all_instruments_long %>%
  # Join all items to the English pivot on the uni_lemma concept
  left_join(english_pivot, by = "uni_lemma") %>%
  # Clean up and order columns
  rename(source_language = language_id,
         source_word = item_definition) %>%
  select(uni_lemma, source_language, source_word, english_word) %>%
  # Remove rows where the source word or English word is missing
  filter(!is.na(source_word) & !is.na(english_word)) %>%
  # Pivot the table wider to show all foreign words alongside the English word
  pivot_wider(
      id_cols = c(uni_lemma, english_word),
      names_from = source_language,
      values_from = source_word,
      names_prefix = "word_",
      # FIX 2: Collapse multiple source words (e.g., "chupete", "chupón") for the same concept 
      # into a single, semicolon-separated string. This resolves the list-col warning.
      values_fn = ~paste(unique(.), collapse = "; ")
  ) %>%
  # Drop the redundant word_English column
  select(-word_English) %>%
  # *** NEW STEP: Final Column Renaming for User Clarity ***
  dplyr::rename(
      English = english_word,
      Spanish = word_Spanish,
      French = word_French
  )

# --- 3. EXPORT MAPPING FOR PYTHON/PANDAS ---

# Export the final translation pivot table
write.csv(cross_language_pivot, "language_translation_table.csv", row.names = FALSE)

cat("\n--- CSV Export Status ---\n")
cat("Successfully transformed and exported the cross-language translation pivot table.\n")
cat("File: 'cross_language_translation_pivot.csv'\n")
cat("Columns: uni_lemma, English, Spanish, French.\n")
