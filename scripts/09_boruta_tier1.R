# ============================================================
# Script: 09_boruta_tier1.R
# Phase B: Feature Selection (EXPLANATORY, NOT MODEL SELECTION)
#
# Purpose:
#   - Run Boruta on Tier 1 ONLY
#   - Leakage-safe: Boruta runs on TRAIN ONLY inside repeated resampling
#   - Aggregate selection stability (Confirmed / Tentative / Rejected)
#
# Notes:
#   - This does NOT change Phase A or Phase C results.
#   - This is confirmatory/explanatory analysis to support the narrative.
#   - Keep predictors factor-friendly (no one-hot, no scaling) for interpretability.
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(Boruta)
})

# Output dirs
dir.create("results", showWarnings = FALSE)
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# Load baseline data (Script 01)
e1 <- new.env(parent = globalenv())
source("scripts/01_load_and_filter.R", local = e1)
stopifnot(exists("adni_baseline", envir = e1))
adni_baseline <- get("adni_baseline", envir = e1)

# load CV-safe preprocessing utilities (earlier from Script 03)
source("scripts/03_preprocess_functions.R")

# Parameters
set.seed(42)

R <- 100
train_frac <- 0.80
missing_threshold <- 0.30
target_col <- "DX_bl"

# Parallelization (outer repeats)
use_parallel <- TRUE
n_cores <- max(1, parallel::detectCores() - 1)

# Boruta parameters
boruta_maxRuns <- 100
boruta_doTrace <- 0
# Tier 1 feature set (aligned with Script 05)
tier1_cols <- c(
  target_col,
  # Demographics
  "AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY",
  # Genetics
  "APOE4",
  # Baseline MRI volumetrics
  "Ventricles_bl", "Hippocampus_bl", "WholeBrain_bl", "Entorhinal_bl",
  "Fusiform_bl", "MidTemp_bl", "ICV_bl",
  # Baseline PET summaries
  "FDG_bl", "AV45_bl", "FBB_bl"
)

missing_cols <- setdiff(tier1_cols, names(adni_baseline))
if (length(missing_cols) > 0) {
  warning("Some Tier 1 columns were not found and will be skipped: ",
          paste(missing_cols, collapse = ", "))
}

tier1_cols_present <- intersect(tier1_cols, names(adni_baseline))

adni_tier1 <- adni_baseline %>% select(all_of(tier1_cols_present))
stopifnot(target_col %in% names(adni_tier1))

cat("Tier 1 dataset dims (pre-Boruta pipeline):\n")
print(dim(adni_tier1))
cat("Tier 1 class distribution:\n")
print(table(adni_tier1[[target_col]]))

# Helpers
stratified_split <- function(y, train_frac = 0.8) {
  y <- factor(y, levels = c("CN", "AD"))
  idx_CN <- which(y == "CN")
  idx_AD <- which(y == "AD")

  n_tr_CN <- floor(length(idx_CN) * train_frac)
  n_tr_AD <- floor(length(idx_AD) * train_frac)

  tr_CN <- sample(idx_CN, n_tr_CN)
  tr_AD <- sample(idx_AD, n_tr_AD)

  train_idx <- sort(c(tr_CN, tr_AD))
  test_idx  <- setdiff(seq_along(y), train_idx)

  list(train_idx = train_idx, test_idx = test_idx)
}

# Train-only preprocessing for Boruta (NO scaling, NO one-hot)
prep_train_for_boruta <- function(train_df,
                                  target_col = "DX_bl",
                                  missing_threshold = 0.30) {

  train_df <- prefer_bl_columns(train_df, target_col = target_col)

  dropper <- fit_missingness_dropper(train_df, target_col = target_col, threshold = missing_threshold)
  train_df <- apply_missingness_dropper(train_df, dropper)

  imputer <- fit_imputer(train_df, target_col = target_col)
  train_df <- apply_imputer(train_df, imputer)

  x <- train_df %>%
    select(-all_of(target_col)) %>%
    mutate(across(where(is.character), as.factor))

  y <- factor(train_df[[target_col]], levels = c("CN", "AD"))

  list(x = x, y = y, artifacts = list(dropper = dropper, imputer = imputer))
}

# Run Boruta once on a given training set, return attStats (after TentativeRoughFix)
run_boruta_once <- function(train_df, rep_id) {
  pre <- prep_train_for_boruta(
    train_df = train_df,
    target_col = target_col,
    missing_threshold = missing_threshold
  )

  x <- pre$x

  # Drop numeric zero-variance
  num_cols <- names(x)[map_lgl(x, is.numeric)]
  if (length(num_cols) > 0) {
    sds <- map_dbl(x[num_cols], sd)
    drop_num <- names(sds[!is.finite(sds) | sds == 0])
    if (length(drop_num) > 0) x <- x %>% select(-any_of(drop_num))
  }

  # Drop factors with <2 levels
  fac_cols <- names(x)[map_lgl(x, is.factor)]
  if (length(fac_cols) > 0) {
    nlv <- map_int(x[fac_cols], ~ nlevels(droplevels(.x)))
    drop_fac <- names(nlv[nlv < 2])
    if (length(drop_fac) > 0) x <- x %>% select(-any_of(drop_fac))
  }

  if (ncol(x) == 0) {
    return(tibble(
      repeat_id = rep_id,
      feature = character(0),
      decision = character(0),
      meanImp = numeric(0),
      medianImp = numeric(0),
      minImp = numeric(0),
      maxImp = numeric(0),
      normHits = numeric(0)
    ))
  }

  b <- Boruta(
    x = x,
    y = pre$y,
    maxRuns = boruta_maxRuns,
    doTrace = boruta_doTrace
  )

  b <- TentativeRoughFix(b)

  st <- attStats(b) %>%
    rownames_to_column("feature") %>%
    as_tibble() %>%
    transmute(
      repeat_id = rep_id,
      feature = feature,
      decision = as.character(decision),
      meanImp = meanImp,
      medianImp = medianImp,
      minImp = minImp,
      maxImp = maxImp,
      normHits = normHits
    )

  st
}
# Main: repeated stratified splits, Boruta on TRAIN only
cat("\n=== Phase B: Boruta Tier 1 (train-only inside repeated splits) ===\n")
cat("Repeats:", R, " | train_frac:", train_frac, " | missing_threshold:", missing_threshold, "\n")
cat("Boruta maxRuns:", boruta_maxRuns, " | Parallel:", use_parallel, " | cores:", ifelse(use_parallel, n_cores, 1), "\n\n")

run_one_repeat <- function(r) {
  sp <- stratified_split(adni_tier1[[target_col]], train_frac = train_frac)
  train_df <- adni_tier1[sp$train_idx, , drop = FALSE]
  run_boruta_once(train_df = train_df, rep_id = r)
}

if (use_parallel && n_cores > 1) {
  cat("Parallel ON | cores =", n_cores, "\n")
  cl <- parallel::makeCluster(n_cores)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages({
      library(tidyverse)
      library(Boruta)
    })
    source("scripts/03_preprocess_functions.R")
    NULL
  })

  parallel::clusterExport(
    cl,
    varlist = c(
      "adni_tier1", "R", "train_frac", "missing_threshold", "target_col",
      "boruta_maxRuns", "boruta_doTrace",
      "stratified_split", "prep_train_for_boruta", "run_boruta_once", "run_one_repeat"
    ),
    envir = environment()
  )

  res_list <- parallel::parLapply(cl, 1:R, run_one_repeat)
  att_df <- bind_rows(res_list)

} else {
  cat("Parallel OFF\n")
  att_df <- bind_rows(lapply(1:R, run_one_repeat))
}

# Save per-repeat attStats
write.csv(att_df, "results/tables/boruta_tier1_attstats_per_repeat.csv", row.names = FALSE)

# Aggregate stability summary
stab <- att_df %>%
  group_by(feature) %>%
  summarise(
    n_obs = n(),
    confirmed_n = sum(decision == "Confirmed", na.rm = TRUE),
    tentative_n = sum(decision == "Tentative", na.rm = TRUE),
    rejected_n  = sum(decision == "Rejected",  na.rm = TRUE),
    confirmed_rate = confirmed_n / n_obs,
    tentative_rate = tentative_n / n_obs,
    rejected_rate  = rejected_n  / n_obs,
    medianImp_mean = mean(medianImp, na.rm = TRUE),
    medianImp_median = median(medianImp, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(confirmed_rate), desc(medianImp_mean))

write.csv(stab, "results/tables/boruta_tier1_stability_summary.csv", row.names = FALSE)

cat("\nTop features by confirmed rate:\n")
print(stab %>% select(feature, confirmed_rate, tentative_rate, rejected_rate, medianImp_mean) %>% slice_head(n = 12))

# Plots
top_n <- 20

stab_top <- stab %>%
  slice_head(n = min(top_n, nrow(stab))) %>%
  mutate(feature = fct_reorder(feature, confirmed_rate))

p_conf <- ggplot(stab_top, aes(x = feature, y = confirmed_rate)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Boruta Tier 1: Confirmed Rate (Top Features)",
    subtitle = paste0("Repeated train-only Boruta | R=", R, " | train_frac=", train_frac, " | maxRuns=", boruta_maxRuns),
    x = NULL,
    y = "Confirmed rate"
  )

ggsave("results/plots/boruta_tier1_confirmed_rate.png", p_conf, width = 8, height = 6, dpi = 300)

stab_long <- stab_top %>%
  select(feature, confirmed_rate, tentative_rate, rejected_rate) %>%
  pivot_longer(
    cols = c(confirmed_rate, tentative_rate, rejected_rate),
    names_to = "decision",
    values_to = "rate"
  ) %>%
  mutate(
    decision = recode(decision,
                      confirmed_rate = "Confirmed",
                      tentative_rate = "Tentative",
                      rejected_rate  = "Rejected")
  )

p_stack <- ggplot(stab_long, aes(x = feature, y = rate, fill = decision)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Boruta Tier 1: Decision Stability (Top Features)",
    subtitle = "Stacked proportions across repeats (Confirmed / Tentative / Rejected)",
    x = NULL,
    y = "Proportion of repeats",
    fill = "Boruta decision"
  )

ggsave("results/plots/boruta_tier1_decision_stacked.png", p_stack, width = 8.5, height = 6.5, dpi = 300)

cat("\nScript 09 complete\n")
cat("Saved outputs:\n")
cat("- results/tables/boruta_tier1_attstats_per_repeat.csv\n")
cat("- results/tables/boruta_tier1_stability_summary.csv\n")
cat("- results/plots/boruta_tier1_confirmed_rate.png\n")
cat("- results/plots/boruta_tier1_decision_stacked.png\n")
