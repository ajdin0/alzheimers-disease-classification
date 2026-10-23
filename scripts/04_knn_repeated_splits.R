
# ============================================================
# Script: 04_knn_stageB_multi_k.R
# Purpose (SENSITIVITY-FIRST):
#   - Skip unstable inner-CV (Stage A) because CV metrics became NA (folds with 0 AD).
#   - Instead: run repeated stratified 80/20 splits for MULTIPLE fixed k values.
#   - Select k by minimizing FN / maximizing sensitivity (tie-break by specificity, then larger k).
#
# Outputs:
#   - results/tables/knn_stageB_multi_k_summary_tier0.csv
#   - results/tables/knn_stageB_multi_k_per_repeat_tier0.csv
#   - results/plots/stageB_multi_k_FN_vs_k_tier0.png
#   - results/plots/stageB_multi_k_sens_vs_k_tier0.png
#   - results/plots/stageB_multi_k_spec_vs_k_tier0.png
#   - results/plots/stageB_multi_k_selected_confusion_matrix_tier0.png
#
# Notes:
#   - Euclidean distance via class::knn() baseline.
#   - Minkowski/Manhattan sensitivity analysis can be added later by swapping the backend.
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

# Load preprocessing functions
source("scripts/03_preprocess_functions.R")

# Parameters (LOCKED)
set.seed(42)

R <- 100
train_frac <- 0.80
tier <- 0
missing_threshold <- 0.30

# Multi-k grid (keep modest for speed; expand later if needed)
k_grid <- c(1, 3, 5, 7, 9, 11, 15, 21, 27)

positive_class <- "AD"

# Parallelization toggle (CPU)
use_parallel <- TRUE
n_cores <- max(1, parallel::detectCores() - 1)

# Helpers
confusion_counts <- function(y_true, y_pred, pos = "AD") {
  y_true <- factor(y_true, levels = c("CN", "AD"))
  y_pred <- factor(y_pred, levels = c("CN", "AD"))
  TP <- sum(y_true == pos & y_pred == pos)
  FN <- sum(y_true == pos & y_pred != pos)
  TN <- sum(y_true != pos & y_pred != pos)
  FP <- sum(y_true != pos & y_pred == pos)
  c(TP = TP, FP = FP, TN = TN, FN = FN)
}

metrics_from_counts <- function(cnt) {
  TP <- cnt["TP"]; FP <- cnt["FP"]; TN <- cnt["TN"]; FN <- cnt["FN"]
  sens <- if ((TP + FN) > 0) TP / (TP + FN) else NA_real_
  spec <- if ((TN + FP) > 0) TN / (TN + FP) else NA_real_
  ppv  <- if ((TP + FP) > 0) TP / (TP + FP) else NA_real_
  npv  <- if ((TN + FN) > 0) TN / (TN + FN) else NA_real_
  acc  <- (TP + TN) / (TP + TN + FP + FN)
  c(sensitivity = sens, specificity = spec, ppv = ppv, npv = npv, accuracy = acc)
}

stratified_split <- function(y, train_frac = 0.8) {
  y <- factor(y, levels = c("CN", "AD"))
  idx_CN <- which(y == "CN")
  idx_AD <- which(y == "AD")
  
  n_tr_CN <- floor(length(idx_CN) * train_frac)
  n_tr_AD <- floor(length(idx_AD) * train_frac)
  
  tr_CN <- sample(idx_CN, n_tr_CN)
  tr_AD <- sample(idx_AD, n_tr_AD)
  
  train_idx <- sort(c(tr_CN, tr_AD))
  test_idx  <- setdiff(seq_along(y), train_idx)
  
  list(train_idx = train_idx, test_idx = test_idx)
}

# Selection rule: minimize FN (equiv maximize sensitivity when totals comparable),
# tie-break by higher specificity, then larger k for smoother boundary.
select_best_k_from_summary <- function(summary_tbl) {
  summary_tbl %>%
    arrange(FN_sum, desc(specificity), desc(k)) %>%
    slice(1) %>%
    pull(k)
}

# Main: repeated splits, multi-k
cat("=== Stage B (Multi-k): Repeated stratified 80/20 ===\n")
cat("Repeats:", R, " | k grid:", paste(k_grid, collapse = ","), "\n")
cat("Parallel:", use_parallel, "| cores:", ifelse(use_parallel, n_cores, 1), "\n\n")

run_one_repeat_all_k <- function(r) {
  split <- stratified_split(adni_baseline$DX_bl, train_frac = train_frac)
  train_df <- adni_baseline[split$train_idx, , drop = FALSE]
  test_df  <- adni_baseline[split$test_idx,  , drop = FALSE]
  
  prep <- preprocess_train_test(
    train_df = train_df,
    test_df  = test_df,
    tier = tier,
    missing_threshold = missing_threshold
  )
  
  # For each k, predict + metrics
  map_dfr(k_grid, function(k) {
    pred <- class::knn(train = prep$x_train, test = prep$x_test, cl = prep$y_train, k = k)
    cnt <- confusion_counts(prep$y_test, pred, pos = positive_class)
    met <- metrics_from_counts(cnt)
    
    tibble(
      rep = r,
      k = k,
      TP = as.numeric(cnt["TP"]),
      FP = as.numeric(cnt["FP"]),
      TN = as.numeric(cnt["TN"]),
      FN = as.numeric(cnt["FN"]),
      sensitivity = as.numeric(met["sensitivity"]),
      specificity = as.numeric(met["specificity"]),
      ppv = as.numeric(met["ppv"]),
      npv = as.numeric(met["npv"]),
      accuracy = as.numeric(met["accuracy"])
    )
  })
}

if (use_parallel && n_cores > 1) {
  cat("Parallel ON | cores =", n_cores, "\n")
  cl <- parallel::makeCluster(n_cores)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  
  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages({ library(tidyverse); library(class) })
    source("scripts/03_preprocess_functions.R")
    NULL
  })
  
  parallel::clusterExport(
    cl,
    varlist = c("adni_baseline", "train_frac", "tier", "missing_threshold", "k_grid",
                "positive_class", "stratified_split", "confusion_counts",
                "metrics_from_counts", "run_one_repeat_all_k"),
    envir = environment()
  )
  
  res_list <- parallel::parLapply(cl, 1:R, run_one_repeat_all_k)
  res_df <- bind_rows(res_list)
  
} else {
  cat("Parallel OFF\n")
  res_df <- bind_rows(lapply(1:R, run_one_repeat_all_k))
}

write.csv(res_df, "results/tables/knn_stageB_multi_k_per_repeat_tier0.csv", row.names = FALSE)

# Summaries per k
summary_k <- res_df %>%
  group_by(k) %>%
  summarise(
    TP_sum = sum(TP), FP_sum = sum(FP), TN_sum = sum(TN), FN_sum = sum(FN),
    # Robust metrics computed from aggregated counts (avoid NaN from mean() on empty)
    sensitivity = TP_sum / (TP_sum + FN_sum),
    specificity = TN_sum / (TN_sum + FP_sum),
    ppv = TP_sum / (TP_sum + FP_sum),
    npv = TN_sum / (TN_sum + FN_sum),
    accuracy = (TP_sum + TN_sum) / (TP_sum + TN_sum + FP_sum + FN_sum),
    .groups = "drop"
  ) %>%
  arrange(k)

write.csv(summary_k, "results/tables/knn_stageB_multi_k_summary_tier0.csv", row.names = FALSE)

k_star <- select_best_k_from_summary(summary_k)
write.csv(tibble(k_star = k_star), "results/tables/knn_stageB_selected_k_star_tier0.csv", row.names = FALSE)

cat("Selected k* by FN-minimization:", k_star, "\n\n")

# Plots: FN vs k, sensitivity vs k, specificity vs k
p_fn <- ggplot(summary_k, aes(x = k, y = FN_sum)) +
  geom_line() + geom_point() +
  labs(
    title = "Stage B (Multi-k): Total False Negatives vs k (Tier 0)",
    subtitle = paste0("Selected k* = ", k_star, " (min FN; tie-break spec, then larger k)"),
    x = "k",
    y = "FN (sum over repeats)"
  )
ggsave("results/plots/stageB_multi_k_FN_vs_k_tier0.png", p_fn, width = 7, height = 5, dpi = 300)

p_sens <- ggplot(summary_k, aes(x = k, y = sensitivity)) +
  geom_line() + geom_point() +
  labs(
    title = "Stage B (Multi-k): Mean Sensitivity vs k (Tier 0)",
    subtitle = paste0("Selected k* = ", k_star),
    x = "k",
    y = "Mean sensitivity (across repeats)"
  )
ggsave("results/plots/stageB_multi_k_sens_vs_k_tier0.png", p_sens, width = 7, height = 5, dpi = 300)

p_spec <- ggplot(summary_k, aes(x = k, y = specificity)) +
  geom_line() + geom_point() +
  labs(
    title = "Stage B (Multi-k): Mean Specificity vs k (Tier 0)",
    subtitle = paste0("Selected k* = ", k_star),
    x = "k",
    y = "Mean specificity (across repeats)"
  )
ggsave("results/plots/stageB_multi_k_spec_vs_k_tier0.png", p_spec, width = 7, height = 5, dpi = 300)

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
    title = "Selected k*: Aggregate Confusion Matrix (Sum over repeats, Tier 0)",
    subtitle = paste0("k* = ", k_star),
    x = "Predicted",
    y = "True"
  )
ggsave("results/plots/stageB_multi_k_selected_confusion_matrix_tier0.png", p_cm, width = 6, height = 5, dpi = 300)

cat("=== Script 04 (Multi-k Stage B) complete ===\n")
cat("Saved outputs:\n")
cat("- results/tables/knn_stageB_multi_k_per_repeat_tier0.csv\n")
cat("- results/tables/knn_stageB_multi_k_summary_tier0.csv\n")
cat("- results/tables/knn_stageB_selected_k_star_tier0.csv\n")
cat("- results/plots/stageB_multi_k_FN_vs_k_tier0.png\n")
cat("- results/plots/stageB_multi_k_sens_vs_k_tier0.png\n")
cat("- results/plots/stageB_multi_k_spec_vs_k_tier0.png\n")
cat("- results/plots/stageB_multi_k_selected_confusion_matrix_tier0.png\n\n")

cat("Summary table:\n")
print(summary_k)
