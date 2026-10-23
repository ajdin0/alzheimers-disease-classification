# ============================================================
# Script: 13_heatmaps_metrics_tier1_tier2b.R
# Purpose:
#   Produce heatmaps from Script 12 outputs.
#   - Heatmaps are normalized per-metric within a panel so different metric
#     scales (e.g., AUC vs accuracy) remain visually comparable.
#   - Delta heatmaps show raw differences (tuned - default) per metric.
# ============================================================

rm(list=ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
})

# paths
in_path <- file.path("results", "tables", "model_comparison_metrics_default_vs_tuned.csv")
stopifnot(file.exists(in_path))

dir.create(file.path("results","plots"), recursive=TRUE, showWarnings=FALSE)

# load 
m <- read.csv(in_path) %>% as_tibble()

# required columns
req <- c("tier","model","threshold_policy","accuracy","balanced_accuracy","sensitivity","specificity","auc","f1")
stopifnot(all(req %in% names(m)))

# keep only metrics 
m_long <- m %>%
  select(tier, model, threshold_policy,
         accuracy, balanced_accuracy, sensitivity, specificity, auc, f1) %>%
  pivot_longer(
    cols = c(accuracy, balanced_accuracy, sensitivity, specificity, auc, f1),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    tier = factor(tier, levels=c("Tier1","Tier2B")),
    model = factor(model, levels=c("kNN","LASSO-LogReg","RandomForest")),
    policy = case_when(
      str_detect(threshold_policy, "^Default") ~ "Default",
      str_detect(threshold_policy, "^Train-optimized") ~ "Tuned(FN-min)",
      TRUE ~ "Other"
    ),
    metric = factor(metric, levels=c("accuracy","balanced_accuracy","sensitivity","specificity","f1","auc"))
  )

# helper: per-panel normalization
norm_within_panel <- function(df) {
  df %>%
    group_by(metric) %>%
    mutate(
      # clamp to [0,1] after rescaling within this panel
      value_norm = ifelse(all(is.na(value)), NA_real_,
                          rescale(value, to=c(0,1), from=range(value, na.rm=TRUE)))
    ) %>%
    ungroup()
}

plot_heatmap <- function(df_panel, title, out_png) {
  # df_panel expects: model, metric, value_norm, value
  p <- ggplot(df_panel, aes(x=metric, y=model, fill=value_norm)) +
    geom_tile(color="white", linewidth=0.5) +
    geom_text(aes(label=sprintf("%.3f", value)), size=3) +
    scale_fill_continuous(na.value="grey90") +
    labs(title=title, x=NULL, y=NULL, fill="Scaled\n(0–1)") +
    theme_minimal(base_size=12) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle=30, hjust=1),
      plot.title = element_text(face="bold")
    )

  ggsave(out_png, p, width=9, height=3.8, dpi=300)
  cat("Saved:", out_png, "\n")
}

for (t in c("Tier1","Tier2B")) {
  for (pcy in c("Default","Tuned(FN-min)")) {
    dfp <- m_long %>%
      filter(tier == t, policy == pcy) %>%
      norm_within_panel()

    out_png <- file.path("results","plots",
                         paste0("heatmap_metrics_", tolower(t), "_",
                                ifelse(pcy=="Default","default","tuned"), ".png"))

    plot_heatmap(
      dfp,
      title = paste0("Model Metrics Heatmap — ", t, " — ", pcy),
      out_png = out_png
    )
  }
}

# Delta heatmaps per tier x model x metric
delta <- m_long %>%
  select(tier, model, policy, metric, value) %>%
  pivot_wider(names_from=policy, values_from=value) %>%
  mutate(delta = `Tuned(FN-min)` - Default) %>%
  select(tier, model, metric, delta)

plot_delta_heatmap <- function(df_panel, title, out_png) {
  # normalize by symmetric max abs within this panel for visuals
  maxabs <- max(abs(df_panel$delta), na.rm=TRUE)
  df_panel <- df_panel %>%
    mutate(delta_norm = ifelse(is.finite(maxabs) && maxabs>0, delta/maxabs, 0))

  p <- ggplot(df_panel, aes(x=metric, y=model, fill=delta_norm)) +
    geom_tile(color="white", linewidth=0.5) +
    geom_text(aes(label=sprintf("%+.3f", delta)), size=3) +
    scale_fill_continuous(limits=c(-1,1), oob=squish, na.value="grey90") +
    labs(title=title, x=NULL, y=NULL, fill="Δ scaled\n(-1..1)") +
    theme_minimal(base_size=12) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle=30, hjust=1),
      plot.title = element_text(face="bold")
    )

  ggsave(out_png, p, width=9, height=3.8, dpi=300)
  cat("Saved:", out_png, "\n")
}

for (t in c("Tier1","Tier2B")) {
  dfp <- delta %>% filter(tier == t)

  out_png <- file.path("results","plots",
                       paste0("heatmap_metrics_delta_", tolower(t), "_tuned_minus_default.png"))

  plot_delta_heatmap(
    dfp,
    title = paste0("delta Metrics Heatmap (Tuned − Default) — ", t),
    out_png = out_png
  )
}

cat("\nScript 13 complete.\n")
