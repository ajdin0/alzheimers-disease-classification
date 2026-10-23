# ============================================================
# Script: 11_compare_models_roc_tier1_tier2b.R
# Purpose:
#     - one ROC overlay figure per tier (Tier 1 and Tier 2B)
#     - performance table (accuracy, bal acc, sens, spec, F1, AUC)
#     - includes optimal hyperparameters: k* (kNN), lambda* (LASSO), RF params
#
# Notes:
#   - Repeated stratified 80/20 splits, pooled test predictions across repeats
#   - Leakage-safe preprocessing via preprocess_train_test() from Script 03
# ============================================================

rm(list=ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(class)
  library(glmnet)
  library(randomForest)
  library(pROC)
})

# Output dirs
dir.create("results", showWarnings = FALSE)
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# Load baseline data + preprocessing
e1 <- new.env(parent = globalenv())
source("scripts/01_load_and_filter.R", local = e1)
adni_baseline <- get("adni_baseline", envir = e1)

source("scripts/03_preprocess_functions.R")
source("scripts/10_metrics_utils.R")

set.seed(42)

# Locked evaluation params
R <- 100
train_frac <- 0.80
missing_threshold <- 0.30
target_col <- "DX_bl"

k_star <- 9        # Tier 1 / Tier 2A
rf_ntree <- 500

# LASSO: choose lambda via CV on each repeat's TRAIN set
lasso_nfolds <- 5

# ============================================================
# Tier column definitions
# ============================================================
tier1_cols <- c(
  "DX_bl",
  "AGE","PTGENDER","PTEDUCAT","PTETHCAT","PTRACCAT","PTMARRY",
  "APOE4",
  "Ventricles_bl","Hippocampus_bl","WholeBrain_bl","Entorhinal_bl","Fusiform_bl","MidTemp_bl","ICV_bl",
  "FDG_bl","AV45_bl","FBB_bl"
)

tier2b_cols <- c(
  tier1_cols,
  "CDRSB","MMSE","FAQ",
  "ADAS11","ADAS13","ADASQ4",
  "RAVLT_immediate","RAVLT_learning","RAVLT_forgetting","RAVLT_perc_forgetting",
  "LDELTOTAL"
)
# Stratified split helper
strat_split <- function(y, train_frac=0.8) {
  y <- factor(y, levels=c("CN","AD"))
  idx_CN <- which(y=="CN"); idx_AD <- which(y=="AD")
  n_tr_CN <- floor(length(idx_CN)*train_frac)
  n_tr_AD <- floor(length(idx_AD)*train_frac)
  tr <- sort(c(sample(idx_CN, n_tr_CN), sample(idx_AD, n_tr_AD)))
  te <- setdiff(seq_along(y), tr)
  list(train=tr, test=te)
}

# One repeat runner for a given tier
# Returns pooled test rows: y_true, y_pred, score, model, tier, repeat
run_repeat_for_tier <- function(df, cols, tier_name, r) {
  d <- df %>% select(all_of(intersect(cols, names(df))))
  stopifnot(target_col %in% names(d))
  d <- prefer_bl_columns(d, target_col = target_col)
  
  sp <- strat_split(d[[target_col]], train_frac = train_frac)
  tr_df <- d[sp$train, , drop=FALSE]
  te_df <- d[sp$test,  , drop=FALSE]
  
  pp <- preprocess_train_test(
    train_df = tr_df,
    test_df  = te_df,
    target_col = target_col,
    missing_threshold = missing_threshold
  )
  
  xtr <- pp$x_train; ytr <- pp$y_train
  xte <- pp$x_test;  yte <- pp$y_test
  
  # kNN (score via prob=TRUE -> P(AD))
  pred_knn <- class::knn(train=xtr, test=xte, cl=ytr, k=k_star, prob=TRUE)
  p_win <- attr(pred_knn, "prob")
  score_knn <- ifelse(pred_knn=="AD", p_win, 1 - p_win)
  
  out_knn <- tibble(
    tier = tier_name, repeat_id = r, model = "kNN",
    y_true = yte,
    y_pred = factor(pred_knn, levels=c("CN","AD")),
    score = as.numeric(score_knn)
  )
  
  # LASSO
  ytr01 <- ifelse(ytr=="AD", 1, 0)
  cvfit <- cv.glmnet(xtr, ytr01, family="binomial", alpha=1, nfolds=lasso_nfolds)
  score_lasso <- as.numeric(predict(cvfit, newx=xte, s="lambda.min", type="response"))
  pred_lasso <- factor(ifelse(score_lasso >= 0.5, "AD", "CN"), levels=c("CN","AD"))
  
  out_lasso <- tibble(
    tier = tier_name, repeat_id = r, model = "LASSO-LogReg",
    y_true = yte,
    y_pred = pred_lasso,
    score = score_lasso,
    hyper = paste0("lambda.min=", signif(cvfit$lambda.min, 4))
  )
  
  # Random Forest 
  rf <- randomForest(x=xtr, y=ytr, ntree=rf_ntree)
  prob_rf <- as.numeric(predict(rf, newdata=xte, type="prob")[,"AD"])
  pred_rf <- factor(ifelse(prob_rf >= 0.5, "AD", "CN"), levels=c("CN","AD"))
  
  out_rf <- tibble(
    tier = tier_name, repeat_id = r, model = "RandomForest",
    y_true = yte,
    y_pred = pred_rf,
    score = prob_rf,
    hyper = paste0("ntree=", rf_ntree)
  )
  
  bind_rows(
    out_knn %>% mutate(hyper = paste0("k=", k_star)),
    out_lasso,
    out_rf
  )
}

# Run all repeats for Tier 1 and Tier 2B
run_all <- function(cols, tier_name) {
  cat("\nRunning:", tier_name, " | R=", R, "\n")
  out_list <- vector("list", R)
  for (r in 1:R) {
    out_list[[r]] <- tryCatch(
      run_repeat_for_tier(adni_baseline, cols, tier_name, r),
      error = function(e) {
        warning(sprintf("Repeat %d failed (%s): %s", r, tier_name, conditionMessage(e)))
        tibble()
      }
    )
    if (r %% 10 == 0) cat("  completed", r, "of", R, "\n")
  }
  bind_rows(out_list)
}

pred_t1  <- run_all(tier1_cols,  "Tier1")
pred_t2b <- run_all(tier2b_cols, "Tier2B")
pred_all <- bind_rows(pred_t1, pred_t2b)

write.csv(pred_all, "results/tables/predictions_pooled_tier1_tier2b.csv", row.names=FALSE)

# metrics table per tier x model
summarise_model <- function(df_sub) {
  cc <- conf_counts(df_sub$y_true, df_sub$y_pred)
  met <- metrics_from_counts(cc$TP, cc$FP, cc$TN, cc$FN)
  ra <- roc_auc(df_sub$y_true, df_sub$score)
  
  tibble(
    TP = cc$TP, FP = cc$FP, TN = cc$TN, FN = cc$FN,
    accuracy = met$accuracy,
    balanced_accuracy = met$balanced_accuracy,
    sensitivity = met$sensitivity,
    specificity = met$specificity,
    f1 = met$f1,
    auc = ra$auc
  )
}

metrics_tbl <- pred_all %>%
  group_by(tier, model) %>%
  group_modify(~ summarise_model(.x)) %>%
  ungroup()

# Add hyperparameter summary (simple, deterministic)
hyper_tbl <- tibble(
  model = c("kNN", "LASSO-LogReg", "RandomForest"),
  hyperparameters = c(
    paste0("k*=", k_star, " (selected sensitivity-first in Phase A)"),
    "lambda=min per-repeat CV (cv.glmnet); threshold=0.5",
    paste0("ntree=", rf_ntree, " (default mtry; threshold=0.5)")
  )
)

metrics_tbl <- metrics_tbl %>%
  left_join(hyper_tbl, by="model") %>%
  relocate(hyperparameters, .after = model)

write.csv(metrics_tbl, "results/tables/model_comparison_metrics_tier1_tier2b.csv", row.names=FALSE)

cat("\nSaved comparison table:\n- results/tables/model_comparison_metrics_tier1_tier2b.csv\n")

# ROC overlay plots (one per tier)
plot_roc_overlay <- function(pred_df, tier_name, out_path) {
  # Build ROC objects for each model (skip any model where ROC cannot be computed)
  model_list <- sort(unique(pred_df$model))
  
  dfs <- list()
  for (m in model_list) {
    dfm <- pred_df %>% filter(model == m)
    
    # Guard: need both classes present and non-constant score
    y <- factor(dfm$y_true, levels=c("CN","AD"))
    s <- dfm$score
    
    if (nlevels(droplevels(y)) < 2) {
      warning(sprintf("Skipping ROC for %s (%s): only one class present in pooled data.", m, tier_name))
      next
    }
    if (length(unique(s[is.finite(s)])) < 2) {
      warning(sprintf("Skipping ROC for %s (%s): score is constant/degenerate.", m, tier_name))
      next
    }
    
    ra <- tryCatch(
      roc_auc(y, s),
      error = function(e) {
        warning(sprintf("Skipping ROC for %s (%s): %s", m, tier_name, conditionMessage(e)))
        NULL
      }
    )
    if (is.null(ra)) next
    
    auc_val <- ra$auc
    if (!is.finite(auc_val)) {
      warning(sprintf("Skipping ROC for %s (%s): AUC not finite.", m, tier_name))
      next
    }
    
    dfroc <- roc_to_df(ra$roc, m) %>%
      mutate(model_lab = paste0(m, " (AUC=", sprintf("%.3f", auc_val), ")"))
    
    dfs[[length(dfs) + 1]] <- dfroc
  }
  
  roc_df <- bind_rows(dfs)
  
  if (nrow(roc_df) == 0) {
    stop(sprintf("No valid ROC curves could be computed for %s. Check pooled predictions.", tier_name))
  }
  
  p <- ggplot(roc_df, aes(x=fpr, y=tpr, color=model_lab)) +
    geom_line(linewidth=1) +
    geom_abline(slope=1, intercept=0, linetype="dashed") +
    coord_equal() +
    labs(
      title = paste0("ROC Overlay: ", tier_name),
      subtitle = "Pooled test predictions across repeated stratified splits",
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)",
      color = "Model"
    )
  
  ggsave(out_path, p, width=7.5, height=6, dpi=300)
  cat("Saved ROC overlay:", out_path, "\n")
}

plot_roc_overlay(pred_all %>% filter(tier=="Tier1"),  "Tier 1",  "results/plots/roc_overlay_tier1.png")
plot_roc_overlay(pred_all %>% filter(tier=="Tier2B"), "Tier 2B", "results/plots/roc_overlay_tier2b.png")

cat("\nScript 11 complete\n")
cat("Saved:\n")
cat("- results/tables/predictions_pooled_tier1_tier2b.csv\n")
cat("- results/tables/model_comparison_metrics_tier1_tier2b.csv\n")
cat("- results/plots/roc_overlay_tier1.png\n")
cat("- results/plots/roc_overlay_tier2b.png\n")
