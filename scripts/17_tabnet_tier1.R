# ============================================================
# Script: 17_tabnet_tier1.R
# Purpose:
#   Deep Learning model: TabNet on Tier 1 ADNIMERGE features
#   - Same CN vs AD setup
#   - Same train-only preprocessing pipeline
#   - Repeated stratified 80/20 splits
#   - Default threshold and FN-optimized threshold
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(tabnet)
  library(torch)
  library(pROC)
})

dir.create("results", showWarnings = FALSE)
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# Load data safely, because 01_load_and_filter.R clears workspace
e1 <- new.env(parent = globalenv())
source("scripts/01_load_and_filter.R", local = e1)
adni_baseline <- get("adni_baseline", envir = e1)

source("scripts/03_preprocess_functions.R")

set.seed(42)

# First test with 5. If it works, change to 100.
R <- 100

train_frac <- 0.80
missing_threshold <- 0.30

positive_class <- "AD"
negative_class <- "CN"

tier1_cols <- c(
  "DX_bl",
  "AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY",
  "APOE4",
  "Ventricles_bl", "Hippocampus_bl", "WholeBrain_bl", "Entorhinal_bl",
  "Fusiform_bl", "MidTemp_bl", "ICV_bl",
  "FDG_bl", "AV45_bl", "FBB_bl"
)

cols <- intersect(tier1_cols, names(adni_baseline))
d <- adni_baseline %>% select(all_of(cols))

strat_split <- function(y, train_frac = 0.8) {
  y <- factor(y, levels = c(negative_class, positive_class))
  idx_neg <- which(y == negative_class)
  idx_pos <- which(y == positive_class)
  
  n_tr_neg <- floor(length(idx_neg) * train_frac)
  n_tr_pos <- floor(length(idx_pos) * train_frac)
  
  tr <- sort(c(sample(idx_neg, n_tr_neg), sample(idx_pos, n_tr_pos)))
  te <- setdiff(seq_along(y), tr)
  
  list(train = tr, test = te)
}

conf_counts <- function(y_true, y_pred) {
  y_true <- factor(y_true, levels = c(negative_class, positive_class))
  y_pred <- factor(y_pred, levels = c(negative_class, positive_class))
  
  TP <- sum(y_true == positive_class & y_pred == positive_class)
  FN <- sum(y_true == positive_class & y_pred == negative_class)
  FP <- sum(y_true == negative_class & y_pred == positive_class)
  TN <- sum(y_true == negative_class & y_pred == negative_class)
  
  c(TP = TP, FP = FP, TN = TN, FN = FN)
}

metrics_from_counts <- function(TP, FP, TN, FN) {
  acc <- (TP + TN) / (TP + FP + TN + FN)
  sens <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
  spec <- ifelse((TN + FP) > 0, TN / (TN + FP), NA_real_)
  bal_acc <- 0.5 * (sens + spec)
  prec <- ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_)
  f1 <- ifelse(is.finite(prec) && is.finite(sens) && (prec + sens) > 0,
               2 * prec * sens / (prec + sens), NA_real_)
  
  tibble(
    accuracy = acc,
    balanced_accuracy = bal_acc,
    sensitivity = sens,
    specificity = spec,
    f1 = f1
  )
}

pick_threshold_train <- function(y_true, score_ad, grid = NULL) {
  y_true <- factor(y_true, levels = c(negative_class, positive_class))
  
  if (is.null(grid)) {
    grid <- sort(unique(score_ad))
    grid <- unique(c(0, grid, 1))
  }
  
  res <- purrr::map_dfr(grid, function(th) {
    y_pred <- ifelse(score_ad >= th, positive_class, negative_class)
    cc <- conf_counts(y_true, y_pred)
    met <- metrics_from_counts(cc["TP"], cc["FP"], cc["TN"], cc["FN"])
    
    tibble(
      threshold = th,
      FN = cc["FN"],
      specificity = met$specificity
    )
  })
  
  res %>%
    arrange(FN, desc(specificity), desc(threshold)) %>%
    slice(1) %>%
    pull(threshold)
}

get_ad_prob <- function(pred_obj) {
  pred_df <- as.data.frame(pred_obj)
  
  possible_names <- c(
    ".pred_AD", "AD", "prob_AD", "Pr.AD.", "Pr..AD.",
    "class_AD", "X.AD"
  )
  
  hit <- intersect(possible_names, names(pred_df))
  if (length(hit) > 0) {
    return(as.numeric(pred_df[[hit[1]]]))
  }
  
  numeric_cols <- names(pred_df)[sapply(pred_df, is.numeric)]
  
  if (length(numeric_cols) == 1) {
    return(as.numeric(pred_df[[numeric_cols[1]]]))
  }
  
  if (length(numeric_cols) >= 2) {
    # Usually binary probability columns are ordered by factor levels: CN, AD.
    return(as.numeric(pred_df[[numeric_cols[2]]]))
  }
  
  stop("Could not extract AD probability from TabNet prediction output.")
}

summarise_model <- function(df) {
  cc <- conf_counts(df$y_true, df$y_pred)
  met <- metrics_from_counts(cc["TP"], cc["FP"], cc["TN"], cc["FN"])
  
  y <- factor(df$y_true, levels = c(negative_class, positive_class))
  s <- as.numeric(df$score)
  
  aucv <- NA_real_
  if (nlevels(droplevels(y)) == 2 && length(unique(s[is.finite(s)])) >= 2) {
    roc_obj <- pROC::roc(
      y, s,
      levels = c(negative_class, positive_class),
      direction = "<",
      quiet = TRUE
    )
    aucv <- as.numeric(pROC::auc(roc_obj))
  }
  
  tibble(
    TP = cc["TP"],
    FP = cc["FP"],
    TN = cc["TN"],
    FN = cc["FN"],
    accuracy = met$accuracy,
    balanced_accuracy = met$balanced_accuracy,
    sensitivity = met$sensitivity,
    specificity = met$specificity,
    f1 = met$f1,
    auc = aucv
  )
}

all_pred_default <- list()
all_pred_tuned <- list()
history_summary <- list()

cat("\n=== Script 17: TabNet Tier1 ===\n")
cat("Repeats:", R, "| train_frac:", train_frac, "\n\n")

for (r in seq_len(R)) {
  cat(sprintf("TabNet progress: %d/%d\n", r, R))
  flush.console()
  
  set.seed(42 + r)
  torch::torch_manual_seed(42 + r)
  
  sp <- strat_split(d$DX_bl, train_frac = train_frac)
  tr_df <- d[sp$train, , drop = FALSE]
  te_df <- d[sp$test, , drop = FALSE]
  
  pp <- preprocess_train_test(
    train_df = tr_df,
    test_df = te_df,
    target_col = "DX_bl",
    missing_threshold = missing_threshold
  )
  
  xtr <- as.data.frame(pp$x_train)
  xte <- as.data.frame(pp$x_test)
  
  ytr_factor <- factor(pp$y_train, levels = c(negative_class, positive_class))
  yte_factor <- factor(pp$y_test, levels = c(negative_class, positive_class))
  
  cfg <- tabnet_config(
    epochs = 100,
    valid_split = 0.20,
    batch_size = 64,
    virtual_batch_size = 32,
    decision_width = 16,
    attention_width = 16,
    num_steps = 3,
    num_shared = 2,
    num_independent = 2,
    learn_rate = 0.01,
    penalty = 0.001,
    early_stopping_patience = 10,
    checkpoint_epochs = 0,
    verbose = FALSE,
    device = "cpu",
    skip_importance = TRUE
  )
  
  fit <- tabnet_fit(
    x = xtr,
    y = ytr_factor,
    config = cfg
  )
  
  pred_te_prob <- predict(fit, xte, type = "prob")
  pred_tr_prob <- predict(fit, xtr, type = "prob")
  
  score_te <- get_ad_prob(pred_te_prob)
  score_tr <- get_ad_prob(pred_tr_prob)
  
  th_tabnet <- pick_threshold_train(ytr_factor, score_tr)
  
  pred_default <- ifelse(score_te >= 0.5, positive_class, negative_class)
  pred_tuned <- ifelse(score_te >= th_tabnet, positive_class, negative_class)
  
  all_pred_default[[length(all_pred_default) + 1]] <- tibble(
    tier = "Tier1",
    repeat_id = r,
    model = "TabNet",
    y_true = as.character(yte_factor),
    y_pred = as.character(pred_default),
    score = as.numeric(score_te),
    hyper = "TabNet(dw=16, aw=16, steps=3), thr=0.5",
    threshold = 0.5
  )
  
  all_pred_tuned[[length(all_pred_tuned) + 1]] <- tibble(
    tier = "Tier1",
    repeat_id = r,
    model = "TabNet",
    y_true = as.character(yte_factor),
    y_pred = as.character(pred_tuned),
    score = as.numeric(score_te),
    hyper = "TabNet(dw=16, aw=16, steps=3), thr=train-optimized",
    threshold = as.numeric(th_tabnet)
  )
  
  history_summary[[length(history_summary) + 1]] <- tibble(
    repeat_id = r,
    selected_threshold = as.numeric(th_tabnet),
    n_features = ncol(xtr)
  )
}

pred_default <- bind_rows(all_pred_default)
pred_tuned <- bind_rows(all_pred_tuned)

metrics_default <- pred_default %>%
  group_by(tier, model) %>%
  group_modify(~ summarise_model(.x)) %>%
  ungroup() %>%
  mutate(threshold_policy = "Default (0.5)")

metrics_tuned <- pred_tuned %>%
  group_by(tier, model) %>%
  group_modify(~ summarise_model(.x)) %>%
  ungroup() %>%
  mutate(threshold_policy = "Train-optimized (min FN; tie-break spec; then higher thr)")

metrics_all <- bind_rows(metrics_default, metrics_tuned)

write.csv(
  pred_default,
  "results/tables/tabnet_tier1_predictions_default.csv",
  row.names = FALSE
)

write.csv(
  pred_tuned,
  "results/tables/tabnet_tier1_predictions_tuned.csv",
  row.names = FALSE
)

write.csv(
  metrics_all,
  "results/tables/tabnet_tier1_metrics_default_vs_tuned.csv",
  row.names = FALSE
)

write.csv(
  bind_rows(history_summary),
  "results/tables/tabnet_tier1_training_history_summary.csv",
  row.names = FALSE
)

cat("\nTabNet Tier1 complete.\n")
cat("Saved:\n")
cat("- results/tables/tabnet_tier1_predictions_default.csv\n")
cat("- results/tables/tabnet_tier1_predictions_tuned.csv\n")
cat("- results/tables/tabnet_tier1_metrics_default_vs_tuned.csv\n")
cat("- results/tables/tabnet_tier1_training_history_summary.csv\n\n")

print(metrics_all)