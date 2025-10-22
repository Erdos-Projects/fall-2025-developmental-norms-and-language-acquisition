# --- 1. ESTABLISH THE CONNECTION (Run this first!) ---

# Load the correct driver package
library(DBI)
library(RMariaDB)

#--------------------------------------------------------------
# NOTE ON CREDENTIALS:
# Database settings are loaded from environment variables (e.g., from a .Renviron file).
# For public, read-only credentials see : https://wordbank.stanford.edu/db_args/
# -------------------------------------------------------------

# Define the connection arguments by reading environment variables
readRenviron(".Renviron") # NOTE: Must run in the same directory as the .Renviron file
db_host <- Sys.getenv("WORDBANK_HOST")
db_port <- Sys.getenv("WORDBANK_PORT")
db_user <- Sys.getenv("WORDBANK_USER")
db_pass <- Sys.getenv("WORDBANK_PASS")
db_name <- "wordbank"
# Create the new, fresh connection object
wordbank_con <- dbConnect(
  RMariaDB::MariaDB(),
  host = db_host,
  port = db_port,
  user = db_user,
  password = db_pass,
  dbname = db_name,
  ssl.mode = "disabled" # Bypasses the SSL negotiation error
)

# --- 2. QUERY THE DATA (Run this immediately after connection)

library(dplyr)
# Use dbReadTable for the small instrument metadata table
instruments_df <- dbReadTable(wordbank_con, "common_instrument")

# Print the result (this is your instruments list!)
print("--- common_instrument Table Head ---")
print(head(instruments_df))

# Use tbl() for large tables and query with dplyr
# (Assuming the connection is still valid)
english_ws_table <- tbl(wordbank_con, "instruments_english_american_ws")

# Example: Look at the first 10 rows
data_preview <- english_ws_table %>%
  head(10) %>%
  collect() # Collect pulls the data into R memory

print("--- instruments_english_american_ws Table Preview ---")
print(data_preview)

# NOTE: The connection object 'wordbank_con' must be active!
# If you disconnected, re-run the dbConnect block first.

instrument_id_df <- tbl(wordbank_con, "common_instrument") %>% 
  filter(language == "English (American)", form == "WS") %>% 
  select(id) %>% 
  collect()

# Safely extract the single ID value
instrument_id <- instrument_id_df$id[1]
print(paste("Found Instrument ID:", instrument_id))


# --- 2. Query the common_item table using the ID ---

# Filter the common_item table by the foreign key 'instrument_id'
item_metadata_df <- tbl(wordbank_con, "common_item") %>%
  filter(instrument_id == !!instrument_id) %>% # Filter by the ID we found
  arrange(id) %>% 
  collect()

print("--- Item Metadata (The Words) Preview ---")

# Select only the most relevant columns for easier viewing:
print(
  item_metadata_df %>%
    select(item_id, item_definition, english_gloss) %>%
    head(10)
)

# --- 3. DISCONNECT ---
dbDisconnect(wordbank_con)
