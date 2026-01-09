#!/usr/bin/env Rscript




options(warn = -1)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript predict_cox_model.R <training_data.csv> <new_patient_features.csv> <output.csv>")
}

training_file <- args[1]
new_patient_file <- args[2]
output_file <- args[3]

# Load required packages
suppressPackageStartupMessages({
  # install.packages(c("survival", "dplyr", "caret"))
  library(survival)
  library(dplyr)
  library(caret)
})

# Feature definitions - APIC signature features
features <- c('Area.Energy.var', 'Area.InvDiffMom.Skewness', 'MinorAxisLength.Energy.Prcnt90',
              'Area.DiffAvg.Prcnt10', 'X341', 'X51')

# Load training data and create training set
chaarted_data <- read.csv(training_file)
arm_B_data <- chaarted_data %>%
  filter(has_features == TRUE, ASSIGNED_TX_ARM == "B") %>%
  select(group_uid, os, dead, all_of(features)) %>%
  na.omit()

# Recreate training split with same seed
set.seed(1)
train_index <- createDataPartition(arm_B_data$dead, p = 0.5, list = FALSE)
train_data <- arm_B_data[train_index, ]

# Normalize training data and store parameters
normalize_minmax <- function(x) {
  if (max(x) == min(x)) return(x)
  return((x - min(x)) / (max(x) - min(x)))
}

normalization_params <- list()
for (feat in features) {
  normalization_params[[feat]] <- list(
    min = min(train_data[[feat]]),
    max = max(train_data[[feat]])
  )
  train_data[[feat]] <- normalize_minmax(train_data[[feat]])
}

# Fit Cox model
cox_formula <- as.formula(paste("Surv(os, dead) ~", paste(features, collapse = " + ")))
cox_model <- coxph(cox_formula, data = train_data)

# Calculate threshold (33rd percentile)
train_risks <- predict(cox_model, type = "risk", newdata = train_data)
threshold <- quantile(train_risks, 0.33)

# Load new patient data
new_patient <- read.csv(new_patient_file)

# Normalize new patient data using training parameters
for (feat in features) {
  train_min <- normalization_params[[feat]]$min
  train_max <- normalization_params[[feat]]$max
  
  if (train_max == train_min) {
    new_patient[[feat]] <- 0
  } else {
    new_patient[[feat]] <- (new_patient[[feat]] - train_min) / (train_max - train_min)
  }
}

# Predict risk score
risk_score <- predict(cox_model, type = "risk", newdata = new_patient)

# Assign risk group
risk_group <- ifelse(risk_score > threshold, "High Risk", "Low Risk")

# Create output dataframe
output <- data.frame(
  patient_id = new_patient$patient_id,
  risk_score = risk_score,
  risk_group = risk_group,
  threshold = threshold
)

# Save output
write.csv(output, output_file, row.names = FALSE)

cat("Prediction completed:\n")
cat("  Risk Score:", risk_score, "\n")
cat("  Risk Group:", risk_group, "\n")
cat("  Threshold:", threshold, "\n")
cat("  Output saved to:", output_file, "\n")