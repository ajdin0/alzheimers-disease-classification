# ============================================================
# Script: 05_knn_tier1_multi_k.R
# Purpose (Tier 1 Feature Expansion, Sensitivity-First):
#   - Build Tier 1 feature set: demographics + APOE4 + baseline imaging biomarkers
#   - Evaluate kNN over multiple k with repeated stratified 80/20 splits
#   - Select k* by minimizing FN (tie-break: higher specificity, then larger k)
#   - Save plots/tables under results/
#
# Tier 1 INCLUDED:
#   Demographics: AGE, PTGENDER, PTEDUCAT, PTETHCAT, PTRACCAT, PTMARRY
#   Genetics: APOE4
#   Imaging biomarkers (baseline): Ventricles_bl, Hippocampus_bl, WholeBrain_bl, Entorhinal_bl,
#                                Fusiform_bl, MidTemp_bl, ICV_bl
#   PET (baseline): FDG_bl, AV45_bl, FBB_bl
#
# Tier 1 EXCLUDED (saved for Tier 2): cognition/functional tests (MMSE/ADAS/CDRSB/FAQ/MOCA/RAVLT/Ecog/mPACC/etc.)
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

# Load baseline data (from Script 01)
e1 <- new.env(parent = globalenv())
source("scripts/01_load_and_filter.R", local = e1)
stopifnot(exists("adni_baseline", envir = e1))
adni_baseline <- get("adni_baseline", envir = e1)

# Load preprocessing functions (train-only fit)
source("scripts/03_preprocess_functions.R")

# Parameters (LOCKED)
set.seed(42)

R <- 100
train_frac <- 0.80
missing_threshold <- 0.30

# Multi-k grid (expand later if needed)
k_grid <- c(1, 3, 5, 7, 9, 11, 15, 21, 27)

positive_class <- "AD"

# Parallelization (CPU)
use_parallel <- TRUE
n_cores <- max(1, parallel::detectCores() - 1)

# Tier 1 feature set
tier1_cols <- c(
  # Target
  "DX_bl",
  # Demographics
  "AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY",
  # Genetics
  "APOE4",
  # Baseline MRI volumetrics
  "Ventricles_bl", "Hippocampus_bl", "WholeBrain_bl", "Entorhinal_bl",
  "Fusiform_bl", "MidTemp_bl", "ICV_bl",
  # Baseline PET summaries
  "FDG_bl", "AV45_bl", "FBB_bl"
)

missing_cols <- setdiff(tier1_cols, names(adni_baseline))
if (length(missing_cols) > 0) {
  warning("Some Tier 1 columns were not found and will be skipped: ",
          paste(missing_cols, collapse = ", "))
}
tier1_cols_present <- intersect(tier1_cols, names(adni_baseline))

adni_tier1 <- adni_baseline %>%
  select(all_of(tier1_cols_present))

stopifnot("DX_bl" %in% names(adni_tier1))

cat("Tier 1 dataset dims (pre-preprocess):\n")
print(dim(adni_tier1))
cat("Tier 1 class distribution:\n")
print(table(adni_tier1$DX_bl))

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

select_best_k_from_summary <- function(summary_tbl) {
  summary_tbl %>%
    arrange(FN_sum, desc(specificity), desc(k)) %>%
    slice(1) %>%
    pull(k)
}

# Main: repeated splits, multi-k (Tier 1)
cat("\n=== Tier 1: Repeated stratified 80/20, multi-k kNN ===\n")
cat("Repeats:", R, " | k grid:", paste(k_grid, collapse = ","), "\n")
cat("Parallel:", use_parallel, "| cores:", ifelse(use_parallel, n_cores, 1), "\n\n")

run_one_repeat_all_k <- function(r) {
  split <- stratified_split(adni_tier1$DX_bl, train_frac = train_frac)
  train_df <- adni_tier1[split$train_idx, , drop = FALSE]
  test_df  <- adni_tier1[split$test_idx,  , drop = FALSE]

  # Use tier=0 inside preprocess, because we already control tier by column selection here.
  prep <- preprocess_train_test(
    train_df = train_df,
    test_df  = test_df,
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
    varlist = c("adni_tier1", "train_frac", "missing_threshold", "k_grid", "positive_class",
                "stratified_split", "confusion_counts", "run_one_repeat_all_k"),
    envir = environment()
  )

  res_list <- parallel::parLapply(cl, 1:R, run_one_repeat_all_k)
  res_df <- bind_rows(res_list)

} else {
  cat("Parallel OFF\n")
  res_df <- bind_rows(lapply(1:R, run_one_repeat_all_k))
}

write.csv(res_df, "results/tables/knn_tier1_multi_k_per_repeat.csv", row.names = FALSE)

# Summary per k (aggregate counts -> robust metrics)
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
  arrange(k)

write.csv(summary_k, "results/tables/knn_tier1_multi_k_summary.csv", row.names = FALSE)

k_star <- select_best_k_from_summary(summary_k)
write.csv(tibble(k_star = k_star), "results/tables/knn_tier1_selected_k_star.csv", row.names = FALSE)

cat("Selected k* (Tier 1) by FN-minimization:", k_star, "\n\n")
cat("Summary table (Tier 1):\n")
print(summary_k)

# Plots
p_fn <- ggplot(summary_k, aes(x = k, y = FN_sum)) +
  geom_line() + geom_point() +
  labs(
    title = "Tier 1: Total False Negatives vs k",
    subtitle = paste0("Selected k* = ", k_star, " (min FN; tie-break spec, then larger k)"),
    x = "k",
    y = "FN (sum over repeats)"
  )
ggsave("results/plots/tier1_FN_vs_k.png", p_fn, width = 7, height = 5, dpi = 300)

p_sens <- ggplot(summary_k, aes(x = k, y = sensitivity)) +
  geom_line() + geom_point() +
  labs(
    title = "Tier 1: Sensitivity vs k",
    subtitle = paste0("Selected k* = ", k_star),
    x = "k",
    y = "Sensitivity"
  )
ggsave("results/plots/tier1_sensitivity_vs_k.png", p_sens, width = 7, height = 5, dpi = 300)

p_spec <- ggplot(summary_k, aes(x = k, y = specificity)) +
  geom_line() + geom_point() +
  labs(
    title = "Tier 1: Specificity vs k",
    subtitle = paste0("Selected k* = ", k_star),
    x = "k",
    y = "Specificity"
  )
ggsave("results/plots/tier1_specificity_vs_k.png", p_spec, width = 7, height = 5, dpi = 300)

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
    title = "Tier 1: Confusion Matrix (Aggregate, Selected k*)",
    subtitle = paste0("k* = ", k_star),
    x = "Predicted",
    y = "True"
  )
ggsave("results/plots/tier1_confusion_matrix_selected_k.png", p_cm, width = 6, height = 5, dpi = 300)

# quick Tier 0 vs Tier 1 comparison (if Tier 0 summary exists)

tier0_path <- "results/tables/knn_stageB_multi_k_summary_tier0.csv"
if (file.exists(tier0_path)) {
  tier0 <- read.csv(tier0_path)
  tier0$Tier <- "Tier0"
  tier1 <- summary_k
  tier1$Tier <- "Tier1"

  comp <- bind_rows(
    tier0 %>% transmute(Tier, k = k, FN_sum = FN_sum, sensitivity = sensitivity, specificity = specificity, accuracy = accuracy),
    tier1 %>% transmute(Tier, k = k, FN_sum = FN_sum, sensitivity = sensitivity, specificity = specificity, accuracy = accuracy)
  )

  write.csv(comp, "results/tables/tier0_vs_tier1_comparison.csv", row.names = FALSE)

  cat("\nWrote comparison file: results/tables/tier0_vs_tier1_comparison.csv\n")
} else {
  cat("\nTier 0 summary not found at:", tier0_path, "\n")
  cat("Skipping Tier0 vs Tier1 comparison file.\n")
}

cat("\n=== Script 05 complete ===\n")
cat("Saved outputs:\n")
cat("- results/tables/knn_tier1_multi_k_per_repeat.csv\n")
cat("- results/tables/knn_tier1_multi_k_summary.csv\n")
cat("- results/tables/knn_tier1_selected_k_star.csv\n")
cat("- results/plots/tier1_FN_vs_k.png\n")
cat("- results/plots/tier1_sensitivity_vs_k.png\n")
cat("- results/plots/tier1_specificity_vs_k.png\n")
cat("- results/plots/tier1_confusion_matrix_selected_k.png\n")
cat("results/tables/tier0_vs_tier1_comparison.csv\n")
