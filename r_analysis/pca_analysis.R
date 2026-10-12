#!/usr/bin/env Rscript
# ==============================================================================
# AcouMap - R PCA Baseline Model Fitter
# Computes:
#   1. Covariance matrix & eigenvalues
#   2. Principal components (prcomp)
#   3. Loadings matrix & explained variance ratios
# ==============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(stats)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0 && file.exists(args[1])) {
  raw_input <- paste(readLines(args[1], warn = FALSE), collapse = "\n")
} else {
  raw_input <- paste(readLines("stdin", warn = FALSE), collapse = "\n")
}

input_data <- fromJSON(raw_input)
feature_matrix <- as.matrix(input_data$feature_matrix)
feature_names <- as.character(input_data$feature_names)

N <- nrow(feature_matrix)
P <- ncol(feature_matrix)
colnames(feature_matrix) <- feature_names

# Compute baseline stats
mean_vec <- colMeans(feature_matrix)
std_vec <- apply(feature_matrix, 2, sd)
std_vec[is.na(std_vec) | std_vec < 1e-6] <- 1e-4

# Standardize
norm_matrix <- scale(feature_matrix, center = mean_vec, scale = std_vec)

# Covariance
cov_mat <- cov(norm_matrix)
# Regularize
lambda <- 0.1
cov_mat_reg <- (1 - lambda) * cov_mat + lambda * diag(diag(cov_mat)) + diag(1e-4, P)

# Fit PCA
pca_fit <- prcomp(norm_matrix, center = FALSE, scale. = FALSE)
var_vals <- pca_fit$sdev^2
explained_var <- var_vals / sum(var_vals)

output_payload <- list(
  status = "success",
  engine = "R stats::prcomp",
  mean_vector = round(as.numeric(mean_vec), 6),
  std_vector = round(as.numeric(std_vec), 6),
  cov_matrix = round(as.matrix(cov_mat_reg), 6),
  rotation = round(as.matrix(pca_fit$rotation), 6),
  explained_variance = round(as.numeric(explained_var), 6),
  pc1_loadings = round(as.numeric(pca_fit$rotation[, 1]), 6),
  pc2_loadings = if (ncol(pca_fit$rotation) >= 2) round(as.numeric(pca_fit$rotation[, 2]), 6) else rep(0, P),
  feature_names = feature_names
)

cat(toJSON(output_payload, auto_unbox = TRUE, digits = 6))
