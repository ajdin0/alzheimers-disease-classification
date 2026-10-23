
# ============================================================
# Script: 02_exploration_visualization.R  (PATCHED v2)
# Fixes:
#   1) Source Script 01 into a local env whose parent is globalenv()
#      so attached-package symbols resolve.
#   2) Before PCA with scale.=TRUE, drop zero-variance columns
#
# Purpose (EXPLORATION ONLY):
#   - Load baseline-locked dataset (adni_baseline) from Script 01
#   - Summarize missingness (per-feature + overall)
#   - Plot class balance
#   - Create an UNSUPERVISED PCA view (diagnostic only)
#
# IMPORTANT:
#   - No modeling preprocessing here.
# ============================================================

rm(list = ls())

library(tidyverse)

dir.create("results", showWarnings = FALSE)
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# Load adni_baseline safely from Script 01
e <- new.env(parent = globalenv())
source("scripts/01_load_and_filter.R", local = e)

stopifnot(exists("adni_baseline", envir = e))
adni_baseline <- get("adni_baseline", envir = e)

n_rows <- nrow(adni_baseline)
n_cols <- ncol(adni_baseline)
overall_na <- mean(is.na(adni_baseline))

cat("Exploration dataset summary:\n")
cat("Rows:", n_rows, " | Cols:", n_cols, "\n")
cat("Overall NA fraction:", round(overall_na, 4), "\n\n")

# Class balance 
stopifnot("DX_bl" %in% names(adni_baseline))

class_tbl <- adni_baseline %>%
  count(DX_bl, name = "n") %>%
  mutate(prop = n / sum(n))

write.csv(class_tbl, "results/tables/class_balance.csv", row.names = FALSE)

p_class <- ggplot(class_tbl, aes(x = DX_bl, y = n)) +
  geom_col() +
  labs(
    title = "Class Balance (Baseline CN vs AD)",
    x = "Diagnosis (DX_bl)",
    y = "Count"
  )

ggsave("results/plots/class_balance.png", p_class, width = 7, height = 5, dpi = 300)

# Missingness per feature
miss_tbl <- tibble(
  feature = names(adni_baseline),
  missing_frac = map_dbl(adni_baseline, ~ mean(is.na(.x))),
  missing_count = map_int(adni_baseline, ~ sum(is.na(.x))),
  n = n_rows
) %>%
  arrange(desc(missing_frac))

write.csv(miss_tbl, "results/tables/missingness_by_feature.csv", row.names = FALSE)

miss_top20 <- miss_tbl %>%
  filter(feature != "DX_bl") %>%
  slice_head(n = 20) %>%
  mutate(feature = fct_reorder(feature, missing_frac))

p_miss <- ggplot(miss_top20, aes(x = feature, y = missing_frac)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Top 20 Features by Missingness (Exploration)",
    x = "Feature",
    y = "Missing fraction"
  )

ggsave("results/plots/missingness_top20.png", p_miss, width = 8, height = 6, dpi = 300)

# PCA diagnostic plot (UNSUPERVISED)
missing_threshold_for_pca <- 0.30

num_df <- adni_baseline %>%
  select(-DX_bl) %>%
  select(where(is.numeric))

if (ncol(num_df) < 2) {
  warning("Not enough numeric features for PCA after filtering. Skipping PCA plot.")
} else {
  num_miss_frac <- map_dbl(num_df, ~ mean(is.na(.x)))
  keep_cols <- names(num_miss_frac[num_miss_frac <= missing_threshold_for_pca])
  
  num_df_pca <- num_df %>%
    select(all_of(keep_cols))
  
  # Median imputation (exploration-only)
  num_df_pca_imp <- num_df_pca %>%
    mutate(across(everything(), ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)))
  
  # Drop zero-variance columns (PCA cannot scale constant columns)
  sds <- sapply(num_df_pca_imp, sd)
  keep_var_cols <- names(sds[is.finite(sds) & sds > 0])
  
  num_df_pca_imp <- num_df_pca_imp %>%
    select(all_of(keep_var_cols))
  
  if (ncol(num_df_pca_imp) < 2) {
    warning("Not enough non-constant numeric features for PCA after filtering. Skipping PCA plot.")
  } else {
    pca <- prcomp(num_df_pca_imp, center = TRUE, scale. = TRUE)
    
    scores <- as_tibble(pca$x[, 1:2, drop = FALSE]) %>%
      rename(PC1 = 1, PC2 = 2) %>%
      mutate(DX_bl = adni_baseline$DX_bl)
    
    var_expl <- (pca$sdev^2) / sum(pca$sdev^2)
    pc1_pct <- round(100 * var_expl[1], 1)
    pc2_pct <- round(100 * var_expl[2], 1)
    
    p_pca <- ggplot(scores, aes(x = PC1, y = PC2, color = DX_bl)) +
      geom_point(alpha = 0.7) +
      labs(
        title = "PCA (Exploration View): CN vs AD",
        subtitle = paste0("PC1: ", pc1_pct, "% variance | PC2: ", pc2_pct, "% variance"),
        x = "PC1",
        y = "PC2",
        color = "DX_bl"
      )
    
    ggsave("results/plots/pca_pc1_pc2.png", p_pca, width = 7, height = 5, dpi = 300)
    
    ev_tbl <- tibble(
      component = paste0("PC", seq_along(var_expl)),
      explained_variance = var_expl,
      explained_variance_pct = 100 * var_expl
    )
    write.csv(ev_tbl, "results/tables/pca_explained_variance.csv", row.names = FALSE)
  }
}

cat("Script 02 complete.\n")
cat("Saved outputs:\n")
cat("- results/plots/class_balance.png\n")
cat("- results/plots/missingness_top20.png\n")
cat("- results/plots/pca_pc1_pc2.png (if PCA ran)\n")
cat("- results/tables/class_balance.csv\n")
cat("- results/tables/missingness_by_feature.csv\n")
cat("- results/tables/pca_explained_variance.csv (if PCA ran)\n")
