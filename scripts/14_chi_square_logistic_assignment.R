library(dplyr)
library(readr)

adni <- read_csv("data/ADNIMERGE_18Nov2025.csv")

table(adni$DX_bl)

adni_cn_ad <- adni %>%
  filter(VISCODE == "bl") %>%
  filter(DX_bl %in% c("CN", "AD"))

table(adni_cn_ad$DX_bl)

names(adni_cn_ad)

table(adni_cn_ad$PTGENDER)
table(adni_cn_ad$PTMARRY)
table(adni_cn_ad$APOE4)
summary(adni_cn_ad$AGE)
summary(adni_cn_ad$PTEDUCAT)
table(adni_cn_ad$PTETHCAT)
table(adni_cn_ad$PTRACCAT)

table(adni_cn_ad$PTETHCAT, adni_cn_ad$DX_bl)
table(adni_cn_ad$PTRACCAT, adni_cn_ad$DX_bl)

adni_cn_ad <- adni_cn_ad %>%
  mutate(
    Age_Group = case_when(
      AGE < 70 ~ "<70",
      AGE >= 70 & AGE < 80 ~ "70-79",
      AGE >= 80 ~ "80+",
      TRUE ~ NA_character_
    ),
    
    Education_Group = case_when(
      PTEDUCAT <= 12 ~ "<=12",
      PTEDUCAT >= 13 & PTEDUCAT <= 16 ~ "13-16",
      PTEDUCAT >= 17 ~ "17+",
      TRUE ~ NA_character_
    ),
    
    Ethnicity_Group = case_when(
      PTETHCAT == "Hisp/Latino" ~ "Hisp/Latino",
      PTETHCAT == "Not Hisp/Latino" ~ "Not Hisp/Latino",
      TRUE ~ NA_character_
    ),
    
    Race_Group = case_when(
      PTRACCAT == "White" ~ "White",
      PTRACCAT != "White" & PTRACCAT != "Unknown" ~ "Non-White",
      TRUE ~ NA_character_
    )
  )

table(adni_cn_ad$Age_Group)
table(adni_cn_ad$Education_Group)
table(adni_cn_ad$Ethnicity_Group)
table(adni_cn_ad$Race_Group)

table(adni_cn_ad$Age_Group, adni_cn_ad$DX_bl)
table(adni_cn_ad$Education_Group, adni_cn_ad$DX_bl)
table(adni_cn_ad$Ethnicity_Group, adni_cn_ad$DX_bl)
table(adni_cn_ad$Race_Group, adni_cn_ad$DX_bl)



adni_cn_ad <- adni_cn_ad %>%
  mutate(
    Marital_Group = case_when(
      PTMARRY == "Unknown" ~ NA_character_,
      TRUE ~ PTMARRY
    )
  )

chisq.test(table(adni_cn_ad$PTGENDER, adni_cn_ad$DX_bl))
chisq.test(table(adni_cn_ad$APOE4, adni_cn_ad$DX_bl))
chisq.test(table(adni_cn_ad$Marital_Group, adni_cn_ad$DX_bl))
chisq.test(table(adni_cn_ad$Age_Group, adni_cn_ad$DX_bl))
chisq.test(table(adni_cn_ad$Education_Group, adni_cn_ad$DX_bl))
chisq.test(table(adni_cn_ad$Ethnicity_Group, adni_cn_ad$DX_bl))
chisq.test(table(adni_cn_ad$Race_Group, adni_cn_ad$DX_bl))



adni_cn_ad <- adni_cn_ad %>%
  mutate(
    AD_status = ifelse(DX_bl == "AD", 1, 0)
  )

logistic_data <- adni_cn_ad %>%
  select(
    AD_status,
    AGE,
    PTGENDER,
    PTEDUCAT,
    Marital_Group,
    APOE4,
    Ethnicity_Group,
    Race_Group
  ) %>%
  na.omit()

model <- glm(
  AD_status ~ AGE + PTGENDER + PTEDUCAT + Marital_Group + APOE4 + Ethnicity_Group + Race_Group,
  data = logistic_data,
  family = binomial
)

summary(model)

odds_ratios <- exp(coef(model))
odds_ratios

exp(coef(model))


logistic_data$predicted_probability <- predict(model, type = "response")

logistic_data$predicted_class <- ifelse(
  logistic_data$predicted_probability >= 0.5,
  1,
  0
)

confusion_matrix <- table(
  Actual = logistic_data$AD_status,
  Predicted = logistic_data$predicted_class
)

confusion_matrix

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
accuracy

TN <- confusion_matrix["0", "0"]
FP <- confusion_matrix["0", "1"]
FN <- confusion_matrix["1", "0"]
TP <- confusion_matrix["1", "1"]

sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)

sensitivity
specificity

drop1(model, test = "Chisq")











#Selected features vs chi-sqare tests

library(dplyr)

# 1. Verified feature sets

knn_features <- c(
  "AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY",
  "APOE4",
  "Ventricles_bl", "Hippocampus_bl", "WholeBrain_bl", "Entorhinal_bl",
  "Fusiform_bl", "MidTemp_bl", "ICV_bl",
  "FDG_bl", "AV45_bl", "FBB_bl"
)

lasso_features <- c(
  "PTRACCAT",
  "Hippocampus_bl",
  "PTMARRY",
  "PTGENDER",
  "MidTemp_bl",
  "APOE4",
  "Entorhinal_bl",
  "Ventricles_bl",
  "PTEDUCAT",
  "AGE",
  "ICV",
  "PTETHCAT",
  "WholeBrain_bl"
)

boruta_features <- c(
  "Hippocampus_bl",
  "Entorhinal_bl",
  "MidTemp_bl",
  "APOE4",
  "Ventricles_bl",
  "Fusiform_bl",
  "WholeBrain_bl",
  "ICV_bl",
  "AGE",
  "PTGENDER",
  "PTEDUCAT",
  "PTMARRY"
)

# 2. Function to prepare variable for Chi-square

prepare_chisq_variable <- function(df, feature) {
  
  if (feature == "AGE") {
    return(list(variable = df$Age_Group, tested_as = "Age_Group"))
  }
  
  if (feature == "PTEDUCAT") {
    return(list(variable = df$Education_Group, tested_as = "Education_Group"))
  }
  
  if (feature == "PTMARRY") {
    return(list(variable = df$Marital_Group, tested_as = "Marital_Group"))
  }
  
  if (feature == "PTETHCAT") {
    return(list(variable = df$Ethnicity_Group, tested_as = "Ethnicity_Group"))
  }
  
  if (feature == "PTRACCAT") {
    return(list(variable = df$Race_Group, tested_as = "Race_Group"))
  }
  
  if (feature %in% c("PTGENDER", "APOE4")) {
    return(list(variable = df[[feature]], tested_as = feature))
  }
  
  # For numeric biomarkers: convert to Low / Medium / High using tertiles
  x <- df[[feature]]
  
  grouped_x <- factor(
    dplyr::ntile(x, 3),
    levels = c(1, 2, 3),
    labels = c("Low", "Medium", "High")
  )
  
  return(list(variable = grouped_x, tested_as = paste0(feature, "_tertiles")))
}

# 3. Function to run one Chi-square test

run_feature_chisq <- function(df, feature, method_name) {
  
  prepared <- prepare_chisq_variable(df, feature)
  
  temp <- data.frame(
    DX_bl = df$DX_bl,
    test_var = prepared$variable
  ) %>%
    na.omit()
  
  tab <- table(temp$test_var, temp$DX_bl)
  
  # Skip variables that do not have enough categories after removing missing values
  if (nrow(tab) < 2 || ncol(tab) < 2) {
    return(data.frame(
      Method = method_name,
      Feature = feature,
      Tested_as = prepared$tested_as,
      N = sum(tab),
      Chi_square = NA,
      df = NA,
      p_value = NA,
      p_value_formatted = NA,
      Significant = "Not tested"
    ))
  }
  
  test <- chisq.test(tab)
  
  data.frame(
    Method = method_name,
    Feature = feature,
    Tested_as = prepared$tested_as,
    N = sum(tab),
    Chi_square = round(as.numeric(test$statistic), 3),
    df = as.numeric(test$parameter),
    p_value = test$p.value,
    p_value_formatted = ifelse(test$p.value < 0.0001, "<0.0001", round(test$p.value, 4)),
    Significant = ifelse(test$p.value < 0.05, "Yes", "No")
  )
}

# 4. Run Chi-square tests for each method

knn_chisq_results <- do.call(rbind, lapply(knn_features, function(feature) {
  run_feature_chisq(adni_cn_ad, feature, "kNN Tier 1 input features")
}))

lasso_chisq_results <- do.call(rbind, lapply(lasso_features, function(feature) {
  run_feature_chisq(adni_cn_ad, feature, "LASSO selected features")
}))

boruta_chisq_results <- do.call(rbind, lapply(boruta_features, function(feature) {
  run_feature_chisq(adni_cn_ad, feature, "Boruta/RF selected features")
}))

# 5. Combine all results

ml_feature_chisq_results <- rbind(
  knn_chisq_results,
  lasso_chisq_results,
  boruta_chisq_results
)

# 6. Print full results

ml_feature_chisq_results


