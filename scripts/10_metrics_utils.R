# ============================================================
# Script: 10_metrics_utils.R
# Purpose:
#   Utility functions for classification metrics + ROC/AUC
#   used in Phase C comparison (Tier 1 and Tier 2B).
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(pROC)
})

# Confusion matrix counts: assumes factors with levels c("CN","AD")
conf_counts <- function(y_true, y_pred) {
  y_true <- factor(y_true, levels = c("CN","AD"))
  y_pred <- factor(y_pred, levels = c("CN","AD"))

  TP <- sum(y_true=="AD" & y_pred=="AD", na.rm=TRUE)
  FN <- sum(y_true=="AD" & y_pred=="CN", na.rm=TRUE)
  TN <- sum(y_true=="CN" & y_pred=="CN", na.rm=TRUE)
  FP <- sum(y_true=="CN" & y_pred=="AD", na.rm=TRUE)

  tibble(TP=TP, FP=FP, TN=TN, FN=FN)
}

# Metrics from confusion counts
metrics_from_counts <- function(TP, FP, TN, FN) {
  acc  <- (TP + TN) / (TP + TN + FP + FN)
  sens <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
  spec <- ifelse((TN + FP) > 0, TN / (TN + FP), NA_real_)
  bal_acc <- mean(c(sens, spec), na.rm = TRUE)
  prec <- ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_)
  f1   <- ifelse(is.finite(prec) && is.finite(sens) && (prec + sens) > 0, 2*prec*sens/(prec + sens), NA_real_)

  tibble(
    accuracy = acc,
    balanced_accuracy = bal_acc,
    sensitivity = sens,
    specificity = spec,
    precision = prec,
    f1 = f1
  )
}

# ROC + AUC given y_true factor (CN/AD) and score = P(AD) in [0,1]
roc_auc <- function(y_true, score) {
  y_true <- factor(y_true, levels=c("CN","AD"))
  roc_obj <- pROC::roc(response=y_true, predictor=score, levels=c("CN","AD"),
                       direction="<", quiet=TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  list(roc = roc_obj, auc = auc_val)
}

# Create an overlay ROC plot dataframe from list of roc objects
roc_to_df <- function(roc_obj, model_name) {
  tibble(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    model = model_name
  )
}
