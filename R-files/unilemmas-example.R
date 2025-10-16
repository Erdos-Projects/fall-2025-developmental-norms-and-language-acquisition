# This script demonstrates how the 'uni_lemma' column provides dialectic mappings,
# specifically comparing American and British English vocabulary forms.

library(wordbankr)
library(dplyr)

# Get all instrument metadata for filtering
insts <- wordbankr::get_instruments()
english_dialects <- insts %>%
  filter(grepl("English", language)) %>%
  select(language, form) %>%
  distinct()

english_dialects

# --- 1. Retrieve Instrument Vocabulary Item Lists ---

# American English (WS Form)
american_instruments <- wordbankr::get_item_data(language = "English (American)", form = "WS") %>%
  mutate(dialect = "American")

# British English (Oxford CDI Form)
british_instruments <- wordbankr::get_item_data(language = "English (British)", form = "Oxford CDI") %>%
  mutate(dialect = "British (Oxford CDI)")

# Combine the two datasets of ITEMS (vocabulary words) for comparison
combined_instruments <- bind_rows(american_instruments, british_instruments)

# --- 2. Find Explicit Dialectal Mappings ---
# We look for the known dialectal example: "diaper" (American) vs. "nappy" (British)
diaper_unilemma <- combined_instruments %>%
  filter(uni_lemma == "diaper") %>%
  arrange(uni_lemma, dialect)

diaper_output <- diaper_unilemma %>%
      select(dialect, item_definition, uni_lemma, category)

diaper_output

truck_unilemma <- combined_instruments %>%
  filter(uni_lemma == "truck") %>%
  arrange(uni_lemma, dialect)

truck_output <- truck_unilemma %>%
      select(dialect, item_definition, uni_lemma, category)

truck_output
