# ============================================================
# Script: 06_knn_tier2C_multi_k_connected_PATCHED_v2.R
# Purpose: Tier 2C (Cognition-only) multi-k kNN with connected-line plots
# - Tier 2C can degenerate to ZERO predictor columns after train-only filtering.
# - This breaks model.matrix(~ . , data = x) with:
#     "." in formula and no 'data' argument
# - We DO NOT edit scripts/03_preprocess_functions.R on disk.
# - Instead, we source it, then OVERRIDE fit_encoder/apply_encoder in-memory
#   to safely handle ncol(x)==0 by creating a single constant dummy feature.
#
# Parallel: OFF (robust). (used earlier to boost processing on my old i7)
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(class)     # knn()
})

# Output dirs
dir.create("results", showWarnings = FALSE)
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# Load baseline data
e1 <- new.env(parent = globalenv())
source("scripts/01_load_and_filter.R", local = e1)
stopifnot(exists("adni_baseline", envir = e1))
adni_baseline <- get("adni_baseline", envir = e1)

# Load preprocessing utilities
source("scripts/03_preprocess_functions.R")

# override encoder to handle zero predictors
fit_encoder <- function(train_df, target_col = "DX_bl") {
  x <- train_df %>% select(-all_of(target_col))
  x <- x %>% mutate(across(where(is.character), as.factor))
  
  # If no predictors survive (Tier 2C edge case), create a constant dummy feature.
  if (ncol(x) == 0) {
    mm <- matrix(0, nrow = nrow(train_df), ncol = 1)
    colnames(mm) <- ".dummy_zero"
    return(list(
      target_col = target_col,
      colnames = colnames(mm),
      formula = as.formula("~ 1"),
      train_levels = list()
    ))
  }
  
  formula <- as.formula("~ .")
  mm <- stats::model.matrix(formula, data = as.data.frame(x), na.action = na.pass)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  mm[is.na(mm)] <- 0
  
  list(
    target_col = target_col,
    colnames = colnames(mm),
    formula = formula,
    train_levels = map(x, ~ if (is.factor(.x)) levels(.x) else NULL)
  )
}

apply_encoder <- function(df, encoder) {
  stopifnot(!is.null(encoder$target_col))
  y <- df[[encoder$target_col]]
  
  x <- df %>% select(-all_of(encoder$target_col))
  
  # If training collapsed to dummy feature, mirror it for test
  if (identical(encoder$colnames, ".dummy_zero")) {
    mm <- matrix(0, nrow = nrow(df), ncol = 1)
    colnames(mm) <- ".dummy_zero"
    return(list(x = mm, y = y))
  }
  
  # Convert characters to factors and align factor levels to training
  for (col in names(x)) {
    if (is.character(x[[col]])) x[[col]] <- as.factor(x[[col]])
    if (is.factor(x[[col]]) && !is.null(encoder$train_levels[[col]])) {
      x[[col]] <- factor(as.character(x[[col]]), levels = encoder$train_levels[[col]])
    }
  }
  
  # If test also has no predictors (rare), create zeros for training columns
  if (ncol(x) == 0) {
    mm <- matrix(0, nrow = nrow(df), ncol = length(encoder$colnames))
    colnames(mm) <- encoder$colnames
    return(list(x = mm, y = y))
  }
  
  mm <- stats::model.matrix(encoder$formula, data = as.data.frame(x), na.action = na.pass)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  mm[is.na(mm)] <- 0
  
  # Ensure same columns as training (add missing cols as 0; drop extras)
  missing_cols <- setdiff(encoder$colnames, colnames(mm))
  if (length(missing_cols) > 0) {
    for (mc in missing_cols) mm <- cbind(mm, setNames(rep(0, nrow(mm)), mc))
  }
  extra_cols <- setdiff(colnames(mm), encoder$colnames)
  if (length(extra_cols) > 0) mm <- mm[, setdiff(colnames(mm), extra_cols), drop = FALSE]
  
  # Reorder to training order
  mm <- mm[, encoder$colnames, drop = FALSE]
  
  list(x = mm, y = y)
}

# Parameters 
set.seed(42)
target_col <- "DX_bl"
positive_class <- "AD"
R <- 100
train_frac <- 0.80
missing_threshold <- 0.30
k_grid <- c(1, 3, 5, 7, 9, 11, 15, 21, 27)

use_parallel <- FALSE  # intentional; patched pipeline is robust either way

# Tier 2C feature set (cognition-only)
tier2c_cols <- c(
  target_col,
  "MMSE","CDRSB","FAQ",
  "ADAS11","ADAS13","ADASQ4",
  "RAVLT_immediate","RAVLT_learning","RAVLT_forgetting","RAVLT_perc_forgetting",
  "LDELTOTAL"
)

missing_cols <- setdiff(tier2c_cols, names(adni_baseline))
if (length(missing_cols) > 0) {
  warning("Some Tier 2C columns were not found and will be skipped: ",
          paste(missing_cols, collapse = ", "))
}

tier2c_cols_present <- intersect(tier2c_cols, names(adni_baseline))

adni_tier2c <- adni_baseline %>%
  select(all_of(tier2c_cols_present))

cat("Tier 2C dataset dims (pre-preprocess):\n")
print(dim(adni_tier2c))
cat("Tier 2C class distribution:\n")
print(table(adni_tier2c[[target_col]]))

# Helpers
confusion_counts <- function(y_true, y_pred, pos = "AD") {
  y_true <- factor(y_true, levels = c("CN", "AD"))
  y_pred <- factor(y_pred, levels = c("CN", "AD"))
  c(
    TP = sum(y_true == pos & y_pred == pos),
    FP = sum(y_true != pos & y_pred == pos),
    TN = sum(y_true != pos & y_pred != pos),
    FN = sum(y_true == pos & y_pred != pos)
  )
}

stratified_split <- function(y, train_frac = 0.8) {
  y <- factor(y, levels = c("CN","AD"))
  idx_CN <- which(y == "CN")
  idx_AD <- which(y == "AD")
  tr <- c(
    sample(idx_CN, floor(length(idx_CN) * train_frac)),
    sample(idx_AD, floor(length(idx_AD) * train_frac))
  )
  list(train_idx = sort(tr), test_idx = setdiff(seq_along(y), tr))
}

select_best_k_from_summary <- function(summary_tbl) {
  summary_tbl %>%
    arrange(FN_sum, desc(specificity), desc(k)) %>%
    slice(1) %>%
    pull(k)
}

# Main: repeated splits, multi-k (Tier 2C)
cat("\n=== Tier 2C: Repeated stratified 80/20, multi-k kNN (PATCH v2) ===\n")
cat("Repeats:", R, " | k grid:", paste(k_grid, collapse = ","), "\n")
cat("Parallel:", use_parallel, "\n\n")

run_one_repeat_all_k <- function(r) {
  split <- stratified_split(adni_tier2c[[target_col]], train_frac = train_frac)
  train_df <- adni_tier2c[split$train_idx, , drop = FALSE]
  test_df  <- adni_tier2c[split$test_idx,  , drop = FALSE]
  
  prep <- preprocess_train_test(
    train_df = train_df,
    test_df  = test_df,
    target_col = target_col,
    tier = 0,
    missing_threshold = missing_threshold
  )
  
  map_dfr(k_grid, function(k) {
    pred <- class::knn(train = prep$x_train, test = prep$x_test, cl = prep$y_train, k = k)
    cnt <- confusion_counts(prep$y_test, pred, pos = positive_class)
    
    tibble(
      rep = r,
      k = k,
      TP = as.numeric(cnt["TP"]),
      FP = as.numeric(cnt["FP"]),
      TN = as.numeric(cnt["TN"]),
      FN = as.numeric(cnt["FN"])
    )
  })
}

res_df <- bind_rows(lapply(1:R, run_one_repeat_all_k))

write.csv(res_df, "results/tables/knn_tier2C_multi_k_per_repeat.csv", row.names = FALSE)

# Summary per k
summary_k <- res_df %>%
  group_by(k) %>%
  summarise(
    TP_sum = sum(TP), FP_sum = sum(FP), TN_sum = sum(TN), FN_sum = sum(FN),
    sensitivity = TP_sum / (TP_sum + FN_sum),
    specificity = TN_sum / (TN_sum + FP_sum),
    ppv = TP_sum / (TP_sum + FP_sum),
    npv = TN_sum / (TN_sum + FN_sum),
    accuracy = (TP_sum + TN_sum) / (TP_sum + TN_sum + FP_sum + FN_sum),
    .groups = "drop"
  ) %>%
  mutate(k = as.numeric(k)) %>%
  arrange(k)

write.csv(summary_k, "results/tables/knn_tier2C_multi_k_summary.csv", row.names = FALSE)

k_star <- select_best_k_from_summary(summary_k)
write.csv(tibble(k_star = k_star), "results/tables/knn_tier2C_selected_k_star.csv", row.names = FALSE)

cat("Selected k* (Tier 2C) by FN-minimization:", k_star, "\n\n")
cat("Summary table (Tier 2C):\n")
print(summary_k)
# Plots: points + connecting lines
p_fn <- ggplot(summary_k, aes(x = k, y = FN_sum, group = 1)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  labs(
    title = "Tier 2C: Total False Negatives vs k",
    subtitle = paste0("Selected k* = ", k_star, " (min FN; tie-break spec, then larger k)"),
    x = "k",
    y = "FN (sum over repeats)"
  )
ggsave("results/plots/tier2C_FN_vs_k.png", p_fn, width = 7, height = 5, dpi = 300)

p_sens <- ggplot(summary_k, aes(x = k, y = sensitivity, group = 1)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  labs(
    title = "Tier 2C: Sensitivity vs k",
    subtitle = paste0("Selected k* = ", k_star),
    x = "k",
    y = "Sensitivity"
  )
ggsave("results/plots/tier2C_sensitivity_vs_k.png", p_sens, width = 7, height = 5, dpi = 300)

p_spec <- ggplot(summary_k, aes(x = k, y = specificity, group = 1)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  labs(
    title = "Tier 2C: Specificity vs k",
    subtitle = paste0("Selected k* = ", k_star),
    x = "k",
    y = "Specificity"
  )
ggsave("results/plots/tier2C_specificity_vs_k.png", p_spec, width = 7, height = 5, dpi = 300)

# Confusion matrix for selected k*
sel <- res_df %>% filter(k == k_star)
cm_sum <- sel %>% summarise(TP = sum(TP), FP = sum(FP), TN = sum(TN), FN = sum(FN)) %>% as.list()
cm_vec <- unlist(cm_sum)

cm_plot_df <- tibble(
  True = rep(c("AD","CN"), each = 2),
  Pred = rep(c("AD","CN"), times = 2),
  Count = c(cm_vec["TP"], cm_vec["FN"], cm_vec["FP"], cm_vec["TN"])
)

p_cm <- ggplot(cm_plot_df, aes(x = Pred, y = True, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count)) +
  labs(
    title = "Tier 2C: Confusion Matrix (Aggregate, Selected k*)",
    subtitle = paste0("k* = ", k_star),
    x = "Predicted",
    y = "True"
  )
ggsave("results/plots/tier2C_confusion_matrix_selected_k.png", p_cm, width = 6, height = 5, dpi = 300)

cat("\nScript 06 (Tier 2C) PATCH v2 complete\n")
