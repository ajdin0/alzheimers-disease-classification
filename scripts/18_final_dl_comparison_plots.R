# ============================================================
# Script: 18_final_dl_comparison_plots.R
# Purpose:
#   Final plots for classical ML vs deep learning comparison
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(pROC)
})

dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)

positive_class <- "AD"
negative_class <- "CN"

# -----------------------------
# 1. Default-threshold metric plot
# -----------------------------

default_only <- read.csv("results/tables/final_default_threshold_model_comparison.csv")

default_long <- default_only %>%
  select(model, accuracy, balanced_accuracy, sensitivity, specificity, f1, auc) %>%
  pivot_longer(
    cols = -model,
    names_to = "metric",
    values_to = "value"
  )

p_metrics <- ggplot(default_long, aes(x = reorder(model, value), y = value)) +
  geom_col() +
  facet_wrap(~ metric, scales = "free_x") +
  coord_flip() +
  labs(
    title = "Model comparison using default threshold",
    subtitle = "Classical ML vs deep learning models on Tier 1 ADNIMERGE features",
    x = "Model",
    y = "Metric value"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "results/plots/final_default_metric_comparison.png",
  p_metrics,
  width = 11,
  height = 7,
  dpi = 300
)

cat("Saved: results/plots/final_default_metric_comparison.png\n")

# -----------------------------
# 2. False negative comparison plot
# -----------------------------

p_fn <- ggplot(default_only, aes(x = reorder(model, FN), y = FN)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "False negatives by model",
    subtitle = "Lower false negatives are preferred for AD screening",
    x = "Model",
    y = "False negatives"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "results/plots/final_false_negative_comparison.png",
  p_fn,
  width = 8,
  height = 5,
  dpi = 300
)

cat("Saved: results/plots/final_false_negative_comparison.png\n")

# -----------------------------
# 3. ROC overlay for all six models
# -----------------------------

old_pred <- read.csv("results/tables/predictions_default_threshold_tier1_tier2b.csv") %>%
  filter(tier == "Tier1")

mlp_pred <- read.csv("results/tables/mlp_tier1_predictions_default.csv")
cnn_pred <- read.csv("results/tables/cnn1d_tier1_predictions_default.csv")
tabnet_pred <- read.csv("results/tables/tabnet_tier1_predictions_default.csv")

all_pred <- bind_rows(
  old_pred,
  mlp_pred,
  cnn_pred,
  tabnet_pred
) %>%
  mutate(
    y_true = factor(y_true, levels = c(negative_class, positive_class)),
    score = as.numeric(score),
    model = as.character(model)
  )

roc_df <- all_pred %>%
  group_by(model) %>%
  group_map(~ {
    model_name <- .y$model[[1]]
    
    r <- pROC::roc(
      .x$y_true,
      .x$score,
      levels = c(negative_class, positive_class),
      direction = "<",
      quiet = TRUE
    )
    
    aucv <- as.numeric(pROC::auc(r))
    
    tibble(
      fpr = 1 - r$specificities,
      tpr = r$sensitivities,
      label = paste0(model_name, " (AUC=", sprintf("%.3f", aucv), ")")
    )
  }) %>%
  bind_rows()

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, color = label)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "ROC comparison of classical ML and deep learning models",
    subtitle = "Pooled test predictions across 100 repeated stratified splits",
    x = "False Positive Rate",
    y = "True Positive Rate / Sensitivity",
    color = "Model"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "results/plots/final_roc_classical_vs_dl_tier1.png",
  p_roc,
  width = 9,
  height = 6,
  dpi = 300
)

cat("Saved: results/plots/final_roc_classical_vs_dl_tier1.png\n")

cat("\nFinal comparison plots complete.\n")