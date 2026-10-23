# ============================================================
# Script: 07_lasso_logistic_tier1.R
# Purpose (Phase C - Model Comparison):
#   LASSO logistic regression (glmnet) on Tier 1 features.
#   - Repeated stratified 80/20 splits
#   - Train-only preprocessing (impute + encode + scale)
#   - Train-only hyperparameter selection for lambda*
#     (FN-minimization at threshold=0.5; tie-break specificity; then larger lambda)
#   - Save per-repeat results, summary table, selected lambda*
#   - Save scientifically useful plots
#
#   - This script is intentionally aligned with our kNN scripts.
#   - Phase E (threshold analysis) can be a follow-up; here we keep a fixed 0.5.
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmnet)
})

# ---- Output dirs ----
dir.create("results", showWarnings = FALSE)
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# Load baseline data (from Script 01)
e1 <- new.env(parent = globalenv())
source("scripts/01_load_and_filter.R", local = e1)
stopifnot(exists("adni_baseline", envir = e1))
adni_baseline <- get("adni_baseline", envir = e1)

# load preprocessing functions (train-only fit) 
source("scripts/03_preprocess_functions.R")

# Parameters
set.seed(42)

R <- 100
train_frac <- 0.80
missing_threshold <- 0.30

positive_class <- "AD"
threshold <- 0.50  # fixed decision threshold (Phase E can tune this later)

# glmnet CV folds (train-only)
cv_folds <- 5

# Tier 1 feature set (same as Script 05)
# keep this identical to Tier 1 kNN for fair comparison.
tier1_cols <- c(
  "DX_bl",
  # demographics
  "AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY",
  # genetics
  "APOE4",
  # MRI volumes
  "Ventricles_bl", "Hippocampus_bl", "WholeBrain_bl", "Entorhinal_bl",
  "Fusiform_bl", "MidTemp_bl", "ICV",
  # PET biomarkers (baseline)
  "FDG", "AV45", "FBB"
)

missing_cols <- setdiff(tier1_cols, names(adni_baseline))
if (length(missing_cols) > 0) {
  warning("Some Tier 1 columns were not found and will be skipped: ",
          paste(missing_cols, collapse = ", "))
}

tier1_cols_present <- intersect(tier1_cols, names(adni_baseline))
adni_tier1 <- adni_baseline %>% select(all_of(tier1_cols_present))
stopifnot("DX_bl" %in% names(adni_tier1))

cat("Tier 1 dataset dims (pre-preprocess):\n")
print(dim(adni_tier1))
cat("Tier 1 class distribution:\n")
print(table(adni_tier1$DX_bl))

# Helpers
confusion_counts <- function(y_true, y_pred, positive = "AD") {
  y_true <- factor(y_true, levels = c("CN", "AD"))
  y_pred <- factor(y_pred, levels = c("CN", "AD"))

  TP <- sum(y_true == positive & y_pred == positive)
  TN <- sum(y_true != positive & y_pred != positive)
  FP <- sum(y_true != positive & y_pred == positive)
  FN <- sum(y_true == positive & y_pred != positive)

  tibble(TP = TP, FP = FP, TN = TN, FN = FN)
}

metrics_from_counts <- function(TP, FP, TN, FN) {
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), NA_real_)
  ppv <- ifelse((TP + FP) > 0, TP / (TP + FP), NaN)  # NaN when TP=FP=0 (expected)
  npv <- ifelse((TN + FN) > 0, TN / (TN + FN), NA_real_)
  accuracy <- (TP + TN) / (TP + TN + FP + FN)

  tibble(
    sensitivity = sensitivity,
    specificity = specificity,
    ppv = ppv,
    npv = npv,
    accuracy = accuracy
  )
}

stratified_split <- function(y, train_frac = 0.8) {
  y <- factor(y, levels = c("CN", "AD"))
  idx_CN <- which(y == "CN")
  idx_AD <- which(y == "AD")

  n_train_CN <- floor(length(idx_CN) * train_frac)
  n_train_AD <- floor(length(idx_AD) * train_frac)

  train_idx <- c(sample(idx_CN, n_train_CN), sample(idx_AD, n_train_AD))
  test_idx  <- setdiff(seq_along(y), train_idx)

  list(train_idx = train_idx, test_idx = test_idx)
}

select_best_lambda_from_train <- function(lambda_tbl) {
  # sensitivity-first goal via FN minimization (equiv when #positives fixed)
  # tie-break: higher specificity, then larger lambda (simpler model)
  lambda_tbl %>%
    arrange(FN, desc(specificity), desc(lambda)) %>%
    slice(1) %>%
    pull(lambda)
}

# Main: repeated splits, train-only lambda selection
cat("\n=== Tier 1: Repeated stratified 80/20, LASSO Logistic (glmnet) ===\n")
cat("Repeats:", R, "| train_frac:", train_frac, "| threshold:", threshold, "| cv_folds:", cv_folds, "\n\n")

run_one_repeat <- function(r) {
  sp <- stratified_split(adni_tier1$DX_bl, train_frac = train_frac)
  train_df <- adni_tier1[sp$train_idx, , drop = FALSE]
  test_df  <- adni_tier1[sp$test_idx,  , drop = FALSE]

  # ---- Train-only preprocessing ----
  pre <- preprocess_train_test(
    train_df = train_df,
    test_df  = test_df,
    target_col = "DX_bl",
    missing_threshold = missing_threshold
  )

  x_train <- as.matrix(pre$x_train)
  x_test  <- as.matrix(pre$x_test)
  y_train <- factor(pre$y_train, levels = c("CN", "AD"))
  y_test  <- factor(pre$y_test,  levels = c("CN", "AD"))

  y_train_bin <- ifelse(y_train == positive_class, 1, 0)

  # CV fit on TRAIN only 
  cvfit <- cv.glmnet(
    x = x_train,
    y = y_train_bin,
    family = "binomial",
    alpha = 1,
    nfolds = cv_folds,
    standardize = FALSE
  )

  lambdas <- cvfit$lambda

  # Choose lambda* by FN-minimization on TRAIN at fixed threshold
  train_tbl <- map_dfr(lambdas, function(lam) {
    p_tr <- as.numeric(predict(cvfit$glmnet.fit, newx = x_train, s = lam, type = "response"))
    pred_tr <- ifelse(p_tr >= threshold, positive_class, "CN")
    cc <- confusion_counts(y_train, pred_tr, positive = positive_class)
    met <- metrics_from_counts(cc$TP, cc$FP, cc$TN, cc$FN)

    tibble(
      lambda = lam,
      TP = cc$TP, FP = cc$FP, TN = cc$TN, FN = cc$FN,
      sensitivity = met$sensitivity,
      specificity = met$specificity
    )
  })

  lambda_star <- select_best_lambda_from_train(train_tbl)

  # Evaluate on TEST using lambda*
  p_te <- as.numeric(predict(cvfit$glmnet.fit, newx = x_test, s = lambda_star, type = "response"))
  pred_te <- ifelse(p_te >= threshold, positive_class, "CN")

  cc_te <- confusion_counts(y_test, pred_te, positive = positive_class)
  met_te <- metrics_from_counts(cc_te$TP, cc_te$FP, cc_te$TN, cc_te$FN)

  tibble(
    repeat_id = r,
    lambda_star = lambda_star,
    TP = cc_te$TP,
    FP = cc_te$FP,
    TN = cc_te$TN,
    FN = cc_te$FN,
    sensitivity = met_te$sensitivity,
    specificity = met_te$specificity,
    ppv = met_te$ppv,
    npv = met_te$npv,
    accuracy = met_te$accuracy
  )
}

# Run repeats (single-core; glmnet is fast and this keeps debugging simple)
res_df <- map_dfr(1:R, run_one_repeat)

# Save per-repeat
write.csv(res_df, "results/tables/lasso_tier1_per_repeat.csv", row.names = FALSE)

# Aggregate confusion counts across repeats -> robust metrics
summary_tbl <- res_df %>%
  summarise(
    repeats = n(),
    lambda_star_median = median(lambda_star),
    lambda_star_mean   = mean(lambda_star),
    TP_sum = sum(TP), FP_sum = sum(FP), TN_sum = sum(TN), FN_sum = sum(FN)
  ) %>%
  mutate(
    sensitivity = TP_sum / (TP_sum + FN_sum),
    specificity = TN_sum / (TN_sum + FP_sum),
    ppv = TP_sum / (TP_sum + FP_sum),
    npv = TN_sum / (TN_sum + FN_sum),
    accuracy = (TP_sum + TN_sum) / (TP_sum + TN_sum + FP_sum + FN_sum)
  )

write.csv(summary_tbl, "results/tables/lasso_tier1_summary.csv", row.names = FALSE)

# Choose a single representative lambda* for plotting/CM (median of per-repeat selections)
lambda_star_rep <- summary_tbl$lambda_star_median
write.csv(tibble(lambda_star_rep = lambda_star_rep), "results/tables/lasso_tier1_selected_lambda_star.csv", row.names = FALSE)

cat("Selected representative lambda* (median over repeats):", lambda_star_rep, "\n")
cat("Summary (aggregate counts):\n")
print(summary_tbl)

# Plots 
# 1) Distribution of chosen lambda* across repeats
p_lambda_hist <- ggplot(res_df, aes(x = lambda_star)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Tier 1 LASSO Logistic: Distribution of selected lambda* (over repeats)",
    x = "lambda*", y = "count"
  )

ggsave("results/plots/tier1_lasso_lambda_star_hist.png", p_lambda_hist, width = 7, height = 5, dpi = 300)

# 2) Confusion matrix heatmap using aggregate counts across repeats
cm <- summary_tbl %>% select(TP_sum, FP_sum, TN_sum, FN_sum) %>% as.list()
cm_plot_df <- tibble(
  True = rep(c("AD", "CN"), each = 2),
  Pred = rep(c("AD", "CN"), times = 2),
  Count = c(cm$TP_sum, cm$FN_sum, cm$FP_sum, cm$TN_sum)
)

p_cm <- ggplot(cm_plot_df, aes(x = Pred, y = True, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count)) +
  labs(
    title = paste0("Tier 1 LASSO Logistic: Confusion Matrix (Aggregate over ", summary_tbl$repeats, " repeats)")
  )

ggsave("results/plots/tier1_lasso_confusion_matrix.png", p_cm, width = 6, height = 5, dpi = 300)

# 3) Coefficients at representative lambda* (top beta)
#    Fit a single model on FULL Tier1 data purely for interpretability (NOT evaluation!)
#    This is for the report's coefficient table/plot; do not confuse with test performance. //Adil
pre_full <- preprocess_train_test(
  train_df = adni_tier1,
  test_df  = adni_tier1,
  target_col = "DX_bl",
  missing_threshold = missing_threshold
)

x_full <- as.matrix(pre_full$x_train)
y_full <- factor(pre_full$y_train, levels = c("CN", "AD"))
y_full_bin <- ifelse(y_full == positive_class, 1, 0)

fit_full <- glmnet(
  x = x_full,
  y = y_full_bin,
  family = "binomial",
  alpha = 1,
  standardize = FALSE
)

beta <- as.matrix(coef(fit_full, s = lambda_star_rep))
coef_df <- tibble(
  feature = rownames(beta),
  beta = as.numeric(beta)
) %>%
  filter(feature != "(Intercept)") %>%
  mutate(abs_beta = abs(beta)) %>%
  arrange(desc(abs_beta)) %>%
  slice(1:15)

write.csv(coef_df, "results/tables/tier1_lasso_top_coefficients.csv", row.names = FALSE)

p_coef <- ggplot(coef_df, aes(x = reorder(feature, abs_beta), y = beta)) +
  geom_col() +
  coord_flip() +
  labs(
    title = paste0("Tier 1 LASSO Logistic: Top 15 coefficients at lambda* = ", signif(lambda_star_rep, 3)),
    x = NULL, y = "coefficient (beta)"
  )

ggsave("results/plots/tier1_lasso_top_coefficients.png", p_coef, width = 7, height = 6, dpi = 300)

cat("\nScript 07 complete\n")
cat("Saved outputs:\n")
cat("- results/tables/lasso_tier1_per_repeat.csv\n")
cat("- results/tables/lasso_tier1_summary.csv\n")
cat("- results/tables/lasso_tier1_selected_lambda_star.csv\n")
cat("- results/tables/tier1_lasso_top_coefficients.csv\n")
cat("- results/plots/tier1_lasso_lambda_star_hist.png\n")
cat("- results/plots/tier1_lasso_confusion_matrix.png\n")
cat("- results/plots/tier1_lasso_top_coefficients.png\n")
