
# ============================================================
# Script: 03_preprocess_functions.R
# Purpose:
#   Provide CV-safe preprocessing utilities for repeated 80/20 evaluation.
#
# Key rule:
#   Any preprocessing that "learns" from data (dropping by missingness, imputing,
#   scaling, encoding) must be FIT on TRAIN ONLY and then APPLIED to TEST.
#
# What this script does:
#   - Feature tiering helpers (Tier 0 / 1 / 2)
#   - Duplicate baseline column cleanup (prefer *_bl)
#   - Train-fitted missingness dropper
#   - Train-fitted imputers (median for numeric, mode for categorical)
#   - Train-fitted scalers (z-score)
#   - Train-fitted one-hot encoder via model.matrix()
#
# This file defines functions only (no execution).
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
})

# Utility: prefer *_bl columns
# ADNIMERGE often has both X and X_bl. For baseline-only modeling, X_bl is the clean choice.
# If both exist, we drop X and keep X_bl.
prefer_bl_columns <- function(df, target_col = "DX_bl") {
  cols <- names(df)
  bl_cols <- cols[str_ends(cols, "_bl")]
  
  # map base name -> _bl name
  base_names <- str_remove(bl_cols, "_bl$")
  non_bl_dupes <- base_names[base_names %in% cols]
  
  drop_cols <- setdiff(non_bl_dupes, target_col)
  df %>% select(-any_of(drop_cols))
}

# 1) Feature tiering
# Tier 0: conservative "non-trivial" baseline predictors (demographics / genetics / basic clinical).
# Tier 1: add cognition / questionnaires (diagnosis-adjacent in practice).
# Tier 2: add strong biomarkers like amyloid PET summaries (often sparse subcohorts).
#
# NOTE: Tiering is implemented via EXCLUSION patterns. This keeps the logic transparent and editable.
tier_exclude_patterns <- function(tier = 0) {
  # Always exclude obvious PET amyloid biomarkers (very sparse, subcohort-specific)
  pet_patterns <- c("^PIB(_bl)?$", "^FBB(_bl)?$", "^AV45(_bl)?$")
  
  # Common cognition / functional assessment patterns (diagnosis-adjacent)
  cog_patterns <- c(
    "^MMSE(_bl)?$", "^CDR.*", "^ADAS.*", "^FAQ(_bl)?$",
    "^RAVLT.*", "^TRAIL.*", "^DIGIT.*",
    "^Ecog.*", "^MOCA(_bl)?$", "^LDELTOTAL.*",
    "^BNT.*", "^CLOCK.*", "^NPI.*"
  )
  
  if (tier == 0) return(c(pet_patterns, cog_patterns))
  if (tier == 1) return(pet_patterns) # allow cognition, exclude PET
  if (tier == 2) return(character(0)) # allow all (still train-only preprocessing)
  stop("tier must be 0, 1, or 2")
}

apply_tier_filter <- function(df, tier = 0, target_col = "DX_bl", extra_exclude = NULL) {
  patterns <- c(tier_exclude_patterns(tier), extra_exclude)
  keep <- names(df)
  
  # Never drop target
  keep <- setdiff(keep, character(0))
  
  # Drop columns matching any exclude regex
  if (length(patterns) > 0) {
    drop <- keep[map_lgl(keep, ~ any(str_detect(.x, patterns)))]
    drop <- setdiff(drop, target_col)
    df <- df %>% select(-any_of(drop))
  }
  df
}

# Train-fitted missingness dropper
fit_missingness_dropper <- function(train_df, target_col = "DX_bl", threshold = 0.30) {
  miss_frac <- map_dbl(train_df, ~ mean(is.na(.x)))
  drop_cols <- names(miss_frac[miss_frac > threshold])
  drop_cols <- setdiff(drop_cols, target_col)
  
  list(threshold = threshold, drop_cols = drop_cols)
}

apply_missingness_dropper <- function(df, dropper) {
  df %>% select(-any_of(dropper$drop_cols))
}

# Train-fitted imputer (median for numeric, mode for categorical)
mode_value <- function(x, missing_label = "MISSING") {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(missing_label)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

fit_imputer <- function(train_df, target_col = "DX_bl") {
  x <- train_df %>% select(-all_of(target_col))
  
  num_cols <- names(x)[map_lgl(x, is.numeric)]
  cat_cols <- setdiff(names(x), num_cols)
  
  medians <- map_dbl(x[num_cols], ~ {
    m <- median(.x, na.rm = TRUE)
    if (!is.finite(m) || is.na(m)) 0 else m
  })
  modes <- map(x[cat_cols], ~ mode_value(.x, missing_label = "MISSING"))
  
  list(num_cols = num_cols, cat_cols = cat_cols, medians = medians, modes = modes, target_col = target_col)
}

apply_imputer <- function(df, imputer) {
  target_col <- imputer$target_col
  y <- df[[target_col]]
  x <- df %>% select(-all_of(target_col))
  
  # numeric: median
  for (col in imputer$num_cols) {
    if (!col %in% names(x)) next
    m <- imputer$medians[[col]]
    x[[col]] <- ifelse(is.na(x[[col]]), m, x[[col]])
  }
  
  # categorical: mode (stored as character/factor)
  for (col in imputer$cat_cols) {
    if (!col %in% names(x)) next
    m <- imputer$modes[[col]]
    x[[col]] <- ifelse(is.na(x[[col]]), m, x[[col]])
  }
  
  out <- bind_cols(x, tibble(!!target_col := y))
  out
}

# Align factor levels in test to training to avoid new unseen levels breaking encoding.
align_factor_levels <- function(test_df, train_df, target_col = "DX_bl") {
  tr_x <- train_df %>% select(-all_of(target_col))
  te_x <- test_df %>% select(-all_of(target_col))
  
  for (col in intersect(names(tr_x), names(te_x))) {
    if (is.character(tr_x[[col]]) || is.factor(tr_x[[col]])) {
      tr_levels <- sort(unique(na.omit(as.character(tr_x[[col]]))))
      # Add an explicit bucket for unseen categories
      tr_levels <- unique(c(tr_levels, "OTHER"))
      te_x[[col]] <- as.character(te_x[[col]])
      te_x[[col]][is.na(te_x[[col]])] <- "MISSING"
      te_x[[col]][!te_x[[col]] %in% tr_levels] <- "OTHER"
      te_x[[col]] <- factor(te_x[[col]], levels = tr_levels)
    }
  }
  
  bind_cols(te_x, tibble(!!target_col := test_df[[target_col]]))
}

# Train-fitted scaler (z-score for numeric predictors)
fit_scaler <- function(train_df, target_col = "DX_bl") {
  x <- train_df %>% select(-all_of(target_col))
  num_cols <- names(x)[map_lgl(x, is.numeric)]
  
  means <- map_dbl(x[num_cols], mean, na.rm = TRUE)
  sds <- map_dbl(x[num_cols], sd, na.rm = TRUE)
  sds[sds == 0 | !is.finite(sds)] <- 1  # safety
  
  list(num_cols = num_cols, means = means, sds = sds, target_col = target_col)
}

apply_scaler <- function(df, scaler) {
  target_col <- scaler$target_col
  y <- df[[target_col]]
  x <- df %>% select(-all_of(target_col))
  
  for (col in scaler$num_cols) {
    if (!col %in% names(x)) next
    x[[col]] <- (x[[col]] - scaler$means[[col]]) / scaler$sds[[col]]
  }
  
  bind_cols(x, tibble(!!target_col := y))
}

# Train-fitted one-hot encoder (model.matrix)
fit_encoder <- function(train_df, target_col = "DX_bl") {
  # model.matrix handles numeric + factor/character predictors
  x <- train_df %>% select(-all_of(target_col))
  
  # Convert characters to factors
  x <- x %>% mutate(across(where(is.character), as.factor))
  
  # Store factor levels from training data
  train_levels <- list()
  for (col in names(x)) {
    if (is.factor(x[[col]])) {
      train_levels[[col]] <- levels(x[[col]])
    } else {
      train_levels[[col]] <- NULL
    }
  }
  
  formula <- as.formula("~ .")
  
  mm <- model.matrix(formula, data = x, na.action = na.pass)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  mm[is.na(mm)] <- 0
  
  list(
    formula = formula,
    colnames = colnames(mm),
    train_levels = train_levels,
    target_col = target_col
  )
}
  
apply_encoder <- function(df, encoder) {
  target_col <- encoder$target_col
  y <- df[[target_col]]
  x <- df %>% select(-all_of(target_col))
  
  # Convert characters to factors and align factor levels to training
  for (col in names(x)) {
    if (is.character(x[[col]])) x[[col]] <- as.factor(x[[col]])
    if (is.factor(x[[col]]) && !is.null(encoder$train_levels[[col]])) {
      x[[col]] <- factor(as.character(x[[col]]), levels = encoder$train_levels[[col]])
    }
  }
  
  mm <- model.matrix(encoder$formula, data = x, na.action = na.pass)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  mm[is.na(mm)] <- 0
  
  # Ensure same columns as training (add missing cols as 0; drop extras)
  missing_cols <- setdiff(encoder$colnames, colnames(mm))
  if (length(missing_cols) > 0) {
    zeros <- matrix(0, nrow = nrow(mm), ncol = length(missing_cols))
    colnames(zeros) <- missing_cols
    mm <- cbind(mm, zeros)
  }
  extra_cols <- setdiff(colnames(mm), encoder$colnames)
  if (length(extra_cols) > 0) {
    mm <- mm[, setdiff(colnames(mm), extra_cols), drop = FALSE]
  }
  mm <- mm[, encoder$colnames, drop = FALSE]
  
  list(x = mm, y = y)
}

# Full train/test preprocessing pipeline (Tier-aware, train-fitted)
preprocess_train_test <- function(train_df,
                                  test_df,
                                  target_col = "DX_bl",
                                  tier = 0,
                                  missing_threshold = 0.30,
                                  extra_exclude = NULL) {
  
  # Prefer baseline columns
  train_df <- prefer_bl_columns(train_df, target_col)
  test_df  <- prefer_bl_columns(test_df, target_col)
  
  # Apply tier filter (structure only; no learning)
  train_df <- apply_tier_filter(train_df, tier = tier, target_col = target_col, extra_exclude = extra_exclude)
  test_df  <- apply_tier_filter(test_df,  tier = tier, target_col = target_col, extra_exclude = extra_exclude)
  
  # Fit missingness dropper on train; apply to both
  dropper <- fit_missingness_dropper(train_df, target_col = target_col, threshold = missing_threshold)
  train_df <- apply_missingness_dropper(train_df, dropper)
  test_df  <- apply_missingness_dropper(test_df,  dropper)
  
  # Align factor levels (test to train) before imputation/encoding
  test_df <- align_factor_levels(test_df, train_df, target_col = target_col)
  
  # Fit imputer on train; apply to both
  imputer <- fit_imputer(train_df, target_col = target_col)
  train_df <- apply_imputer(train_df, imputer)
  test_df  <- apply_imputer(test_df,  imputer)
  
  # Fit scaler on train; apply to both
  scaler <- fit_scaler(train_df, target_col = target_col)
  train_df <- apply_scaler(train_df, scaler)
  test_df  <- apply_scaler(test_df,  scaler)
  
  # Fit encoder on train; apply to both
  encoder <- fit_encoder(train_df, target_col = target_col)
  tr <- apply_encoder(train_df, encoder)
  te <- apply_encoder(test_df,  encoder)
  
  list(
    x_train = tr$x,
    y_train = tr$y,
    x_test  = te$x,
    y_test  = te$y,
    artifacts = list(dropper = dropper, imputer = imputer, scaler = scaler, encoder = encoder),
    kept_features = colnames(tr$x)
  )
}
