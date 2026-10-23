# ============================================================
# Script: 06_knn_tier2A_multi_k_connected_FIX.R
# Purpose: Tier 2A (Conservative Cognition) with POINTS + CONNECTING LINES
# Explicitly pass named args to preprocess_train_test() to avoid any
#      accidental positional matching of target_col.
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(class)   # knn()
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

# Parameters 
set.seed(42)

target_col <- "DX_bl"
positive_class <- "AD"

R <- 100
train_frac <- 0.80
missing_threshold <- 0.30

k_grid <- c(1, 3, 5, 7, 9, 11, 15, 21, 27)

# Tier 2A feature set (Tier 1 + conservative cognition)
tier2a_cols <- c(
  target_col,
  # Demographics
  "AGE","PTGENDER","PTEDUCAT","PTETHCAT","PTRACCAT","PTMARRY",
  # Genetics
  "APOE4",
  # Baseline MRI volumetrics
  "Ventricles_bl","Hippocampus_bl","WholeBrain_bl","Entorhinal_bl",
  "Fusiform_bl","MidTemp_bl","ICV_bl",
  # Baseline PET summaries
  "FDG_bl","AV45_bl","FBB_bl",
  # Conservative cognition / function
  "MMSE","CDRSB","FAQ"
)

missing_cols <- setdiff(tier2a_cols, names(adni_baseline))
if (length(missing_cols) > 0) {
  warning("Some Tier 2A columns were not found and will be skipped: ",
          paste(missing_cols, collapse = ", "))
}

tier2a_cols_present <- intersect(tier2a_cols, names(adni_baseline))

adni_tier2a <- adni_baseline %>%
  select(all_of(tier2a_cols_present))

cat("Tier 2A dataset dims (pre-preprocess):\n")
print(dim(adni_tier2a))
cat("Tier 2A class distribution:\n")
print(table(adni_tier2a[[target_col]]))

# Helpers
confusion_counts <- function(y_true, y_pred, pos = "AD") {
  y_true <- factor(y_true, levels = c("CN","AD"))
  y_pred <- factor(y_pred, levels = c("CN","AD"))
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

select_best_k <- function(summary_tbl) {
  summary_tbl %>%
    arrange(FN_sum, desc(specificity), desc(k)) %>%
    slice(1) %>%
    pull(k)
}

# Main: repeated splits, multi-k
res_df <- map_dfr(1:R, function(r) {
  sp <- stratified_split(adni_tier2a[[target_col]], train_frac = train_frac)
  train_df <- adni_tier2a[sp$train_idx, , drop = FALSE]
  test_df  <- adni_tier2a[sp$test_idx,  , drop = FALSE]
  
  # use named args to prevent target_col confusion
  prep <- preprocess_train_test(
    train_df = train_df,
    test_df  = test_df,
    target_col = target_col,
    tier = 0,
    missing_threshold = missing_threshold
  )
  
  map_dfr(k_grid, function(k) {
    pred <- knn(train = prep$x_train, test = prep$x_test, cl = prep$y_train, k = k)
    cnt <- confusion_counts(prep$y_test, pred, pos = positive_class)
    
    tibble(
      rep = r, k = k,
      TP = as.numeric(cnt["TP"]),
      FP = as.numeric(cnt["FP"]),
      TN = as.numeric(cnt["TN"]),
      FN = as.numeric(cnt["FN"])
    )
  })
})

write.csv(res_df, "results/tables/knn_tier2A_multi_k_per_repeat.csv", row.names = FALSE)

summary_k <- res_df %>%
  group_by(k) %>%
  summarise(
    TP_sum = sum(TP), FP_sum = sum(FP),
    TN_sum = sum(TN), FN_sum = sum(FN),
    sensitivity = TP_sum/(TP_sum + FN_sum),
    specificity = TN_sum/(TN_sum + FP_sum),
    ppv = TP_sum/(TP_sum + FP_sum),
    npv = TN_sum/(TN_sum + FN_sum),
    accuracy = (TP_sum + TN_sum)/(TP_sum + TN_sum + FP_sum + FN_sum),
    .groups = "drop"
  ) %>%
  arrange(k)

write.csv(summary_k, "results/tables/knn_tier2A_multi_k_summary.csv", row.names = FALSE)

k_star <- select_best_k(summary_k)
write.csv(tibble(k_star = k_star), "results/tables/knn_tier2A_selected_k_star.csv", row.names = FALSE)

cat("Selected k* (Tier 2A):", k_star, "\n")

# Plots
summary_k <- summary_k %>% mutate(k = as.numeric(k)) %>% arrange(k)

p_fn <- ggplot(summary_k, aes(x = k, y = FN_sum, group = 1)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  labs(
    title = "Tier 2A: Total False Negatives vs k",
    subtitle = paste0("Selected k* = ", k_star, " (min FN; tie-break spec, then larger k)"),
    x = "k", y = "FN (sum over repeats)"
  )
ggsave("results/plots/tier2A_FN_vs_k.png", p_fn, width = 7, height = 5, dpi = 300)

p_sens <- ggplot(summary_k, aes(x = k, y = sensitivity, group = 1)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  labs(
    title = "Tier 2A: Sensitivity vs k",
    subtitle = paste0("Selected k* = ", k_star),
    x = "k", y = "Sensitivity"
  )
ggsave("results/plots/tier2A_sensitivity_vs_k.png", p_sens, width = 7, height = 5, dpi = 300)

p_spec <- ggplot(summary_k, aes(x = k, y = specificity, group = 1)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  labs(
    title = "Tier 2A: Specificity vs k",
    subtitle = paste0("Selected k* = ", k_star),
    x = "k", y = "Specificity"
  )
ggsave("results/plots/tier2A_specificity_vs_k.png", p_spec, width = 7, height = 5, dpi = 300)

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
    title = "Tier 2A: Confusion Matrix (Aggregate, Selected k*)",
    subtitle = paste0("k* = ", k_star),
    x = "Predicted", y = "True"
  )
ggsave("results/plots/tier2A_confusion_matrix_selected_k.png", p_cm, width = 6, height = 5, dpi = 300)

cat("\nScript 06 Tier 2A (connected plots,) complete\n")
