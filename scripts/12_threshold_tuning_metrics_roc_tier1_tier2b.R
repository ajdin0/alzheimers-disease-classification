# ============================================================
# Script: 12_threshold_tuning_metrics_roc_tier1_tier2b_v2.R
# Purpose:
#   Tier 1 + Tier 2B | kNN + LASSO-LogReg + RandomForest
#   - Repeated stratified 80/20 splits (R)
#   - Train-only preprocessing (no leakage)
#   - Two threshold policies:
#       (1) Default: 0.5 for prob models; kNN vote
#       (2) FN-Optimized: choose threshold on TRAIN to minimize FN, then apply to TEST
#   - Metrics table per tier x model x policy:
#       Accuracy, Balanced Accuracy, Sensitivity, Specificity, AUC, F1
#   - ROC overlays per tier (from pooled TEST scores; threshold-independent)
#   - Threshold distribution plots (FN-optimized thresholds)
#
# Fixes vs v1:
#   - live progress (%) with repeat counter
#   - hardened ROC/threshold plotting to avoid "Unknown column: model" warnings
# ============================================================

rm(list=ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(class)         # knn()
  library(glmnet)        # LASSO logistic
  library(randomForest)  # RF
  library(pROC)          # ROC/AUC
})

# Output dirs
dir.create("results", showWarnings = FALSE)
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# Load data + preprocessing
e1 <- new.env(parent = globalenv())
source("scripts/01_load_and_filter.R", local = e1)
stopifnot(exists("adni_baseline", envir = e1))
adni_baseline <- get("adni_baseline", envir = e1)

source("scripts/03_preprocess_functions.R")

# Parameters (LOCKED unless intentionally changed)
set.seed(42)

R <- 100
train_frac <- 0.80
missing_threshold <- 0.30

k_fixed <- 9
rf_ntree <- 500

positive_class <- "AD"
negative_class <- "CN"

# Tier column definitions 
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

tiers <- list(Tier1 = tier1_cols, Tier2B = tier2b_cols)
# Helpers
strat_split <- function(y, train_frac=0.8) {
  y <- factor(y, levels=c(negative_class, positive_class))
  idx_neg <- which(y == negative_class)
  idx_pos <- which(y == positive_class)
  
  n_tr_neg <- floor(length(idx_neg) * train_frac)
  n_tr_pos <- floor(length(idx_pos) * train_frac)
  
  tr <- sort(c(sample(idx_neg, n_tr_neg), sample(idx_pos, n_tr_pos)))
  te <- setdiff(seq_along(y), tr)
  list(train = tr, test = te)
}

conf_counts <- function(y_true, y_pred, positive=positive_class) {
  y_true <- factor(y_true, levels=c(negative_class, positive_class))
  y_pred <- factor(y_pred, levels=c(negative_class, positive_class))
  
  TP <- sum(y_true == positive & y_pred == positive)
  FN <- sum(y_true == positive & y_pred != positive)
  FP <- sum(y_true != positive & y_pred == positive)
  TN <- sum(y_true != positive & y_pred != positive)
  
  c(TP=TP, FP=FP, TN=TN, FN=FN)
}

metrics_from_counts <- function(TP, FP, TN, FN) {
  acc <- (TP+TN)/(TP+FP+TN+FN)
  sens <- ifelse((TP+FN) > 0, TP/(TP+FN), NA_real_)
  spec <- ifelse((TN+FP) > 0, TN/(TN+FP), NA_real_)
  bal_acc <- 0.5*(sens + spec)
  
  prec <- ifelse((TP+FP) > 0, TP/(TP+FP), NA_real_)
  f1 <- ifelse(is.finite(prec) && is.finite(sens) && (prec+sens) > 0, 2*prec*sens/(prec+sens), NA_real_)
  
  tibble(
    accuracy = acc,
    balanced_accuracy = bal_acc,
    sensitivity = sens,
    specificity = spec,
    f1 = f1
  )
}

# Threshold selection on TRAIN:
#   1) minimize FN
#   2) tie-break: maximize specificity
#   3) tie-break: pick larger threshold
pick_threshold_train <- function(y_true, score_ad, grid=NULL) {
  y_true <- factor(y_true, levels=c(negative_class, positive_class))
  
  if (is.null(grid)) {
    grid <- sort(unique(score_ad))
    grid <- unique(c(0, grid, 1))
  }
  
  res <- purrr::map_dfr(grid, function(th) {
    y_pred <- ifelse(score_ad >= th, positive_class, negative_class)
    cc <- conf_counts(y_true, y_pred, positive=positive_class)
    met <- metrics_from_counts(cc["TP"], cc["FP"], cc["TN"], cc["FN"])
    tibble(threshold=th, FN=cc["FN"], specificity=met$specificity)
  })
  
  res %>%
    arrange(FN, desc(specificity), desc(threshold)) %>%
    slice(1) %>%
    pull(threshold)
}

roc_df_from_pROC <- function(roc_obj, label) {
  tibble(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    label = label
  )
}

plot_roc_overlay <- function(pred_df, tier_label, out_png) {
  stopifnot(all(c("y_true","score","model") %in% names(pred_df)))
  
  pred_df <- pred_df %>%
    mutate(
      y_true = factor(y_true, levels=c(negative_class, positive_class)),
      score  = as.numeric(score),
      model  = as.character(model)
    )
  
  roc_df <- pred_df %>%
    group_by(model) %>%
    group_map(~{
      dfm <- .x
      if (nlevels(droplevels(dfm$y_true)) < 2) return(NULL)
      if (length(unique(dfm$score[is.finite(dfm$score)])) < 2) return(NULL)
      
      r <- pROC::roc(dfm$y_true, dfm$score,
                     levels=c(negative_class, positive_class),
                     direction="<", quiet=TRUE)
      aucv <- as.numeric(pROC::auc(r))
      roc_df_from_pROC(r, label=paste0(unique(dfm$model), " (AUC=", sprintf("%.3f", aucv), ")"))
    }) %>%
    bind_rows()
  
  if (nrow(roc_df) == 0) {
    warning(sprintf("No valid ROC curves for %s. Skipping ROC plot.", tier_label))
    return(invisible(NULL))
  }
  
  p <- ggplot(roc_df, aes(x=fpr, y=tpr, color=label)) +
    geom_line(linewidth=1) +
    geom_abline(intercept=0, slope=1, linetype="dashed") +
    labs(
      title = paste0("ROC Overlay: ", tier_label),
      subtitle = "Pooled test predictions across repeated stratified splits",
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)",
      color = "Model"
    ) +
    theme(legend.position="right")
  
  ggsave(out_png, p, width=8.5, height=6.5, dpi=300)
  cat("Saved ROC overlay:", out_png, "\n")
}

# Main 
all_pred_default <- list()
all_pred_tuned   <- list()

cat("\n=== Script 12 v2: Threshold tuning + metrics + ROC ===\n")
cat("Repeats:", R, "| train_frac:", train_frac, "| k_fixed:", k_fixed, "| rf_ntree:", rf_ntree, "\n\n")

for (tier_name in names(tiers)) {
  
  cat("Running:", tier_name, " | R=", R, "\n")
  
  cols <- intersect(tiers[[tier_name]], names(adni_baseline))
  d <- adni_baseline %>% select(all_of(cols))
  stopifnot("DX_bl" %in% names(d))
  
  for (r in seq_len(R)) {
    
    # live progress (%)
    if (r == 1 || r %% 5 == 0 || r == R) {
      pct <- round(100 * r / R)
      cat(sprintf("  %s progress: %3d/%3d  (%3d%%)\n", tier_name, r, R, pct))
      flush.console()
    }
    
    sp <- strat_split(d$DX_bl, train_frac=train_frac)
    tr_df <- d[sp$train, , drop=FALSE]
    te_df <- d[sp$test,  , drop=FALSE]
    
    pp <- preprocess_train_test(
      train_df = tr_df,
      test_df  = te_df,
      target_col = "DX_bl",
      missing_threshold = missing_threshold
    )
    
    xtr <- pp$x_train; ytr <- factor(pp$y_train, levels=c(negative_class, positive_class))
    xte <- pp$x_test;  yte <- factor(pp$y_test,  levels=c(negative_class, positive_class))
    
    # ---- 1) kNN (score = P(AD) from winning-class prob) ----
    pred_knn_te <- class::knn(train=xtr, test=xte, cl=ytr, k=k_fixed, prob=TRUE)
    p_win_te <- attr(pred_knn_te, "prob")
    score_knn_te <- ifelse(pred_knn_te==positive_class, p_win_te, 1-p_win_te)
    
    pred_knn_tr <- class::knn(train=xtr, test=xtr, cl=ytr, k=k_fixed, prob=TRUE)
    p_win_tr <- attr(pred_knn_tr, "prob")
    score_knn_tr <- ifelse(pred_knn_tr==positive_class, p_win_tr, 1-p_win_tr)
    
    th_knn <- pick_threshold_train(ytr, score_knn_tr)
    pred_knn_tuned_te <- ifelse(score_knn_te >= th_knn, positive_class, negative_class)
    
    all_pred_default[[length(all_pred_default)+1]] <- tibble(
      tier=tier_name, repeat_id=r, model="kNN",
      y_true=as.character(yte), y_pred=as.character(pred_knn_te),
      score=as.numeric(score_knn_te),
      hyper=paste0("k=", k_fixed),
      threshold=NA_real_
    )
    all_pred_tuned[[length(all_pred_tuned)+1]] <- tibble(
      tier=tier_name, repeat_id=r, model="kNN",
      y_true=as.character(yte), y_pred=as.character(pred_knn_tuned_te),
      score=as.numeric(score_knn_te),
      hyper=paste0("k=", k_fixed, ", thr=train-optimized"),
      threshold=as.numeric(th_knn)
    )
    
    # ---- 2) LASSO logistic ----
    ytr01 <- ifelse(ytr==positive_class, 1, 0)
    cvfit <- cv.glmnet(as.matrix(xtr), ytr01, family="binomial",
                       alpha=1, nfolds=5, standardize=FALSE)
    
    score_lasso_te <- as.numeric(predict(cvfit, newx=as.matrix(xte), s="lambda.min", type="response"))
    score_lasso_tr <- as.numeric(predict(cvfit, newx=as.matrix(xtr), s="lambda.min", type="response"))
    
    th_lasso <- pick_threshold_train(ytr, score_lasso_tr)
    
    pred_lasso_default_te <- ifelse(score_lasso_te >= 0.5, positive_class, negative_class)
    pred_lasso_tuned_te   <- ifelse(score_lasso_te >= th_lasso, positive_class, negative_class)
    
    all_pred_default[[length(all_pred_default)+1]] <- tibble(
      tier=tier_name, repeat_id=r, model="LASSO-LogReg",
      y_true=as.character(yte), y_pred=as.character(pred_lasso_default_te),
      score=as.numeric(score_lasso_te),
      hyper="lambda.min (cv.glmnet), thr=0.5",
      threshold=0.5
    )
    all_pred_tuned[[length(all_pred_tuned)+1]] <- tibble(
      tier=tier_name, repeat_id=r, model="LASSO-LogReg",
      y_true=as.character(yte), y_pred=as.character(pred_lasso_tuned_te),
      score=as.numeric(score_lasso_te),
      hyper="lambda.min (cv.glmnet), thr=train-optimized",
      threshold=as.numeric(th_lasso)
    )
    
    # ---- 3) Random Forest ----
    rf <- randomForest(x=xtr, y=ytr, ntree=rf_ntree)
    
    score_rf_te <- as.numeric(predict(rf, newdata=xte, type="prob")[, positive_class])
    score_rf_tr <- as.numeric(predict(rf, newdata=xtr, type="prob")[, positive_class])
    
    th_rf <- pick_threshold_train(ytr, score_rf_tr)
    
    pred_rf_default_te <- ifelse(score_rf_te >= 0.5, positive_class, negative_class)
    pred_rf_tuned_te   <- ifelse(score_rf_te >= th_rf, positive_class, negative_class)
    
    all_pred_default[[length(all_pred_default)+1]] <- tibble(
      tier=tier_name, repeat_id=r, model="RandomForest",
      y_true=as.character(yte), y_pred=as.character(pred_rf_default_te),
      score=as.numeric(score_rf_te),
      hyper=paste0("ntree=", rf_ntree, ", thr=0.5"),
      threshold=0.5
    )
    all_pred_tuned[[length(all_pred_tuned)+1]] <- tibble(
      tier=tier_name, repeat_id=r, model="RandomForest",
      y_true=as.character(yte), y_pred=as.character(pred_rf_tuned_te),
      score=as.numeric(score_rf_te),
      hyper=paste0("ntree=", rf_ntree, ", thr=train-optimized"),
      threshold=as.numeric(th_rf)
    )
  }
  
  cat("  completed ", tier_name, "\n\n")
}

pred_default <- bind_rows(all_pred_default) %>% as_tibble()
pred_tuned   <- bind_rows(all_pred_tuned)   %>% as_tibble()

stopifnot(all(c("tier","repeat_id","model","y_true","y_pred","score") %in% names(pred_default)))
stopifnot(all(c("tier","repeat_id","model","y_true","y_pred","score") %in% names(pred_tuned)))

write.csv(pred_default, "results/tables/predictions_default_threshold_tier1_tier2b.csv", row.names=FALSE)
write.csv(pred_tuned,   "results/tables/predictions_fn_optimized_threshold_tier1_tier2b.csv", row.names=FALSE)

#  metrics (includes AUC from scores)
summarise_model <- function(df) {
  cc <- conf_counts(df$y_true, df$y_pred, positive=positive_class)
  met <- metrics_from_counts(cc["TP"], cc["FP"], cc["TN"], cc["FN"])
  
  y <- factor(df$y_true, levels=c(negative_class, positive_class))
  s <- as.numeric(df$score)
  
  aucv <- NA_real_
  if (nlevels(droplevels(y)) == 2 && length(unique(s[is.finite(s)])) >= 2) {
    roc_obj <- pROC::roc(y, s, levels=c(negative_class, positive_class), direction="<", quiet=TRUE)
    aucv <- as.numeric(pROC::auc(roc_obj))
  }
  
  tibble(
    TP = cc["TP"], FP = cc["FP"], TN = cc["TN"], FN = cc["FN"],
    accuracy = met$accuracy,
    balanced_accuracy = met$balanced_accuracy,
    sensitivity = met$sensitivity,
    specificity = met$specificity,
    f1 = met$f1,
    auc = aucv
  )
}

metrics_default <- pred_default %>%
  group_by(tier, model) %>%
  group_modify(~ summarise_model(.x)) %>%
  ungroup() %>%
  mutate(threshold_policy = "Default (0.5 for prob models; kNN vote)")

metrics_tuned <- pred_tuned %>%
  group_by(tier, model) %>%
  group_modify(~ summarise_model(.x)) %>%
  ungroup() %>%
  mutate(threshold_policy = "Train-optimized (min FN; tie-break spec; then higher thr)")

metrics_all <- bind_rows(metrics_default, metrics_tuned) %>%
  arrange(tier, model, threshold_policy)

write.csv(metrics_all, "results/tables/model_comparison_metrics_default_vs_tuned.csv", row.names=FALSE)

# ROC overlays 
plot_roc_overlay(pred_default %>% filter(tier=="Tier1"),  "Tier 1",  "results/plots/roc_overlay_tier1.png")
plot_roc_overlay(pred_default %>% filter(tier=="Tier2B"), "Tier 2B", "results/plots/roc_overlay_tier2b.png")

# Threshold stability plot (tuned)
thr_tbl <- pred_tuned %>%
  mutate(model = as.character(model)) %>%
  filter(is.finite(threshold)) %>%
  group_by(tier, model, repeat_id) %>%
  summarise(threshold = threshold[1], .groups="drop")

if (nrow(thr_tbl) > 0 && all(c("tier","model","threshold") %in% names(thr_tbl))) {
  p_thr <- ggplot(thr_tbl, aes(x=threshold)) +
    geom_histogram(bins=30) +
    facet_grid(tier ~ model, scales="free_y") +
    labs(
      title = "Train-optimized decision thresholds (distribution across repeats)",
      subtitle = "Chosen to minimize FN on TRAIN; applied to TEST",
      x = "threshold", y = "count"
    )
  ggsave("results/plots/threshold_distributions_tier1_tier2b.png", p_thr, width=10, height=6, dpi=300)
  cat("Saved threshold plot: results/plots/threshold_distributions_tier1_tier2b.png\n")
} else {
  cat("No finite tuned thresholds found; skipping threshold distribution plot.\n")
}

cat("\nScript 12 v2 complete\n")
cat("Saved:\n")
cat("- results/tables/predictions_default_threshold_tier1_tier2b.csv\n")
cat("- results/tables/predictions_fn_optimized_threshold_tier1_tier2b.csv\n")
cat("- results/tables/model_comparison_metrics_default_vs_tuned.csv\n")
cat("- results/plots/roc_overlay_tier1.png\n")
cat("- results/plots/roc_overlay_tier2b.png\n")
cat("- results/plots/threshold_distributions_tier1_tier2b.png (if thresholds exist)\n")
