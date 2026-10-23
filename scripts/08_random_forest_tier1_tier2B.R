# ============================================================
# Script: 08_random_forest_tier1_tier2B.R
# Phase C: Model comparison
# Purpose:
#   - Random Forest classification on Tier 1 and Tier 2B
#   - Repeated stratified 80/20 splits
#   - Train-only preprocessing (no leakage)
#   - Sensitivity-first evaluation
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(randomForest)
})

# Parameters
R <- 100
train_frac <- 0.80
missing_threshold <- 0.30
positive_class <- "AD"

set.seed(42)

# Helpers
confusion_counts <- function(y_true, y_pred, positive = "AD") {
  y_true <- factor(y_true, levels = c("CN","AD"))
  y_pred <- factor(y_pred, levels = c("CN","AD"))

  TP <- sum(y_true == positive & y_pred == positive)
  FN <- sum(y_true == positive & y_pred != positive)
  FP <- sum(y_true != positive & y_pred == positive)
  TN <- sum(y_true != positive & y_pred != positive)

  tibble(TP=TP, FP=FP, TN=TN, FN=FN)
}

metrics_from_counts <- function(TP, FP, TN, FN) {
  sensitivity <- ifelse((TP+FN)>0, TP/(TP+FN), NA)
  specificity <- ifelse((TN+FP)>0, TN/(TN+FP), NA)
  ppv <- ifelse((TP+FP)>0, TP/(TP+FP), NA)
  npv <- ifelse((TN+FN)>0, TN/(TN+FN), NA)
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  tibble(sensitivity, specificity, ppv, npv, accuracy)
}

stratified_split <- function(y, train_frac=0.8) {
  y <- factor(y, levels=c("CN","AD"))
  idx_CN <- which(y=="CN")
  idx_AD <- which(y=="AD")
  tr_CN <- sample(idx_CN, floor(train_frac*length(idx_CN)))
  tr_AD <- sample(idx_AD, floor(train_frac*length(idx_AD)))
  train_idx <- c(tr_CN, tr_AD)
  test_idx <- setdiff(seq_along(y), train_idx)
  list(train=train_idx, test=test_idx)
}

# Load data & preprocessing
e1 <- new.env(parent=globalenv())
source("scripts/01_load_and_filter.R", local=e1)
adni_baseline <- get("adni_baseline", envir=e1)

source("scripts/03_preprocess_functions.R")

# Tier definitions
tier1_cols <- c(
  "DX_bl","AGE","PTGENDER","PTEDUCAT","PTETHCAT","PTRACCAT","PTMARRY",
  "APOE4","Ventricles_bl","Hippocampus_bl","WholeBrain_bl","Entorhinal_bl",
  "Fusiform_bl","MidTemp_bl","ICV_bl","FDG_bl","AV45_bl","FBB_bl"
)

tier2B_cols <- c(
  tier1_cols,
  "MMSE","CDRSB","ADAS11","ADAS13",
  "RAVLT_immediate","RAVLT_learning","RAVLT_forgetting",
  "RAVLT_perc_forgetting","LDELTOTAL"
)

tiers <- list(
  Tier1 = tier1_cols,
  Tier2B = tier2B_cols
)

dir.create("results/tables", recursive=TRUE, showWarnings=FALSE)
dir.create("results/plots", recursive=TRUE, showWarnings=FALSE)

# Main
all_results <- list()

for (tier_name in names(tiers)) {

  message("Running Random Forest for ", tier_name)

  cols <- intersect(tiers[[tier_name]], names(adni_baseline))
  data_tier <- adni_baseline %>% select(all_of(cols))

  res_list <- vector("list", R)

  for (r in seq_len(R)) {
    split <- stratified_split(data_tier$DX_bl, train_frac)
    train_df <- data_tier[split$train, , drop=FALSE]
    test_df  <- data_tier[split$test,  , drop=FALSE]

    prep <- preprocess_train_test(
      train_df = train_df,
      test_df  = test_df,
      target_col = "DX_bl",
      tier = 0,
      missing_threshold = missing_threshold
    )

    rf_fit <- randomForest(
      x = prep$x_train,
      y = factor(prep$y_train, levels=c("CN","AD")),
      ntree = 500
    )

    y_pred <- predict(rf_fit, prep$x_test)

    counts <- confusion_counts(prep$y_test, y_pred, positive_class)
    res_list[[r]] <- counts
  }

  res_df <- bind_rows(res_list) %>% mutate(Tier=tier_name)
  summary <- res_df %>%
    summarise(
      TP_sum=sum(TP), FP_sum=sum(FP),
      TN_sum=sum(TN), FN_sum=sum(FN)
    ) %>%
    bind_cols(metrics_from_counts(.$TP_sum,.$FP_sum,.$TN_sum,.$FN_sum)) %>%
    mutate(Tier=tier_name)

  write.csv(res_df,
            file=paste0("results/tables/rf_",tolower(tier_name),"_per_repeat.csv"),
            row.names=FALSE)

  write.csv(summary,
            file=paste0("results/tables/rf_",tolower(tier_name),"_summary.csv"),
            row.names=FALSE)

  all_results[[tier_name]] <- summary
}

message("Script 08 complete")
