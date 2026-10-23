
# ============================================================
# Script: 01_load_and_filter.R
# Purpose:
#   - Load ADNIMERGE dataset
#   - Lock prediction timepoint to baseline (VISCODE == 'bl')
#   - Define binary classification task: CN vs AD
#   - Remove identifiers and obvious leakage variables
# Output:
#   - Clean baseline dataframe: adni_baseline
# ============================================================

# Step 1: Clean workspace
rm(list = ls())

# Step 2: Load required packages
library(tidyverse)

# Step 3: Load raw ADNIMERGE data
adni_raw <- utils::read.csv(
  file = "data/ADNIMERGE_18Nov2025.csv",
  stringsAsFactors = FALSE
)

# Step 4: Filter to baseline visits only
adni_bl <- adni_raw %>%
  filter(VISCODE == "bl")

# Step 5: Keep only CN vs AD diagnoses
adni_bl <- adni_bl %>%
  filter(DX_bl %in% c("CN", "AD"))

# Step 6: Define target variable
# CN = negative class, AD = positive class
adni_bl$DX_bl <- factor(adni_bl$DX_bl, levels = c("CN", "AD"))

# Step 7: Drop identifiers and obvious leakage columns
id_cols <- c(
  "RID", "PTID", "VISCODE", "VISCODE2",
  "EXAMDATE", "DX", "DXCHANGE"
)

adni_baseline <- adni_bl %>%
  select(-any_of(id_cols))

# Step 8: Basic sanity checks
cat("Baseline dataset dimensions:\n")
print(dim(adni_baseline))

cat("\nClass distribution (CN vs AD):\n")
print(table(adni_baseline$DX_bl))
