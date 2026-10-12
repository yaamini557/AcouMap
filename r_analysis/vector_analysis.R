#!/usr/bin/env Rscript
# ==============================================================================
# AcouMap - Genuine R Vector and Statistical Analysis Engine
# Performs:
#   1. Feature-vector normalization (Z-score scaling relative to baseline)
#   2. Euclidean distance calculation
#   3. Regularized Mahalanobis distance calculation
#   4. Cosine vector similarity calculation
#   5. PCA 2D projection (PC1, PC2) & explained variance ratio
#   6. Feature-wise percentage deviations and Z-scores
# ==============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(stats)
  library(MASS)
})

# Read JSON input from argument file or stdin
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0 && file.exists(args[1])) {
  raw_input <- paste(readLines(args[1], warn = FALSE), collapse = "\n")
} else {
  raw_input <- paste(readLines("stdin", warn = FALSE), collapse = "\n")
}

input_data <- fromJSON(raw_input)

# Extract input payloads
# feature_matrix: (N x P matrix of current recording feature vectors)
# feature_names: vector of names
# baseline_mean: (P-length baseline mean vector)
# baseline_std: (P-length baseline standard deviation vector)
# baseline_cov: (P x P baseline covariance matrix, optional)

features_df <- as.data.frame(input_data$feature_matrix)
feature_names <- as.character(input_data$feature_names)
colnames(features_df) <- feature_names

N <- nrow(features_df)
P <- ncol(features_df)

baseline_mean <- as.numeric(input_data$baseline_mean)
baseline_std <- as.numeric(input_data$baseline_std)

# Prevent division by zero in baseline_std
baseline_std[is.na(baseline_std) | baseline_std < 1e-6] <- 1e-4
baseline_mean[is.na(baseline_mean)] <- 0.0

# ------------------------------------------------------------------------------
# 1. Vector Normalization (Standardize by baseline mean and std)
# ------------------------------------------------------------------------------
X <- as.matrix(features_df)
X_norm <- matrix(0, nrow = N, ncol = P)
colnames(X_norm) <- feature_names

for (j in 1:P) {
  X_norm[, j] <- (X[, j] - baseline_mean[j]) / baseline_std[j]
}

# ------------------------------------------------------------------------------
# 2. Euclidean Distance to Baseline
# ------------------------------------------------------------------------------
# Calculate Euclidean distance of each normalized window vector from origin (baseline)
euclidean_distances <- numeric(N)
for (i in 1:N) {
  diff_vec <- X_norm[i, ]
  euclidean_distances[i] <- sqrt(sum(diff_vec^2))
}

# ------------------------------------------------------------------------------
# 3. Regularized Mahalanobis Distance
# ------------------------------------------------------------------------------
# Retrieve or construct regularized covariance matrix
has_cov <- !is.null(input_data$baseline_cov)

if (has_cov) {
  cov_mat <- as.matrix(input_data$baseline_cov)
} else {
  # If covariance not provided, use identity on normalized vectors (uncorrelated fallback)
  cov_mat <- diag(P)
}

# Add ridge shrinkage regularization lambda * I to avoid singularity:
# Sigma_reg = (1 - lambda) * Sigma + lambda * diag(Sigma) + eps * I
lambda <- 0.15
cov_mat_norm <- cov(X_norm)
if (is.na(cov_mat_norm[1, 1]) || N < P) {
  cov_mat_norm <- diag(P)
}
cov_reg <- (1 - lambda) * cov_mat_norm + lambda * diag(diag(cov_mat_norm)) + diag(1e-4, P)

# Invert regularized covariance matrix
inv_cov <- tryCatch({
  solve(cov_reg)
}, error = function(e) {
  diag(1 / (diag(cov_reg) + 1e-4))
})

mahalanobis_distances <- numeric(N)
for (i in 1:N) {
  diff_vec <- matrix(X_norm[i, ], nrow = 1)
  # d_M^2 = diff * inv_cov * diff^T
  m_dist_sq <- as.numeric(diff_vec %*% inv_cov %*% t(diff_vec))
  mahalanobis_distances[i] <- sqrt(max(0, m_dist_sq))
}

# ------------------------------------------------------------------------------
# 4. Cosine Vector Similarity
# ------------------------------------------------------------------------------
# Similarity of raw vector against raw baseline mean
norm_baseline <- sqrt(sum(baseline_mean^2))
cosine_similarities <- numeric(N)

for (i in 1:N) {
  v <- X[i, ]
  norm_v <- sqrt(sum(v^2))
  if (norm_v > 1e-8 && norm_baseline > 1e-8) {
    sim <- sum(v * baseline_mean) / (norm_v * norm_baseline)
    # Clamp to [-1, 1]
    cosine_similarities[i] <- max(-1.0, min(1.0, sim))
  } else {
    cosine_similarities[i] <- 0.0
  }
}

# ------------------------------------------------------------------------------
# 5. Principal Component Analysis (PCA)
# ------------------------------------------------------------------------------
# Fit PCA on normalized vectors
# If baseline PCA rotation is provided, use it; otherwise compute prcomp
if (!is.null(input_data$baseline_pca_rotation)) {
  rotation <- as.matrix(input_data$baseline_pca_rotation)
  pca_coords <- X_norm %*% rotation[, 1:min(2, ncol(rotation))]
  explained_var <- as.numeric(input_data$baseline_pca_explained_var)
} else {
  # Compute local PCA
  # Combine baseline centroid (0 vector) with X_norm
  combined_for_pca <- rbind(matrix(0, nrow = 1, ncol = P), X_norm)
  pca_fit <- prcomp(combined_for_pca, scale. = FALSE, center = FALSE)
  # Project X_norm
  pca_coords <- predict(pca_fit, newdata = X_norm)[, 1:min(2, ncol(pca_fit$x))]
  var_vals <- pca_fit$sdev^2
  explained_var <- var_vals / sum(var_vals)
  rotation <- pca_fit$rotation
}

pc1_vals <- round(as.numeric(pca_coords[, 1]), 4)
pc2_vals <- if (ncol(pca_coords) >= 2) round(as.numeric(pca_coords[, 2]), 4) else rep(0, N)

# ------------------------------------------------------------------------------
# 6. Statistical Deviation & Feature-wise Percentage Deviation
# ------------------------------------------------------------------------------
# Calculate percentage deviation: (x - baseline) / |baseline| * 100%
pct_deviations <- matrix(0, nrow = N, ncol = P)
colnames(pct_deviations) <- feature_names

for (j in 1:P) {
  denom <- ifelse(abs(baseline_mean[j]) > 1e-6, abs(baseline_mean[j]), 1.0)
  pct_deviations[, j] <- round(((X[, j] - baseline_mean[j]) / denom) * 100.0, 2)
}

# Aggregate overall recording statistics
mean_current_vector <- round(colMeans(X), 4)
mean_euclidean_dist <- round(mean(euclidean_distances), 4)
mean_mahalanobis_dist <- round(mean(mahalanobis_distances), 4)
mean_similarity <- round(mean(cosine_similarities), 4)

# Format window-by-window details
windows_r_analysis <- vector("list", N)
for (i in 1:N) {
  # Find top 3 deviating features
  dev_order <- order(abs(pct_deviations[i, ]), decreasing = TRUE)
  top_deviations <- list()
  for (k in 1:min(3, P)) {
    feat_idx <- dev_order[k]
    top_deviations[[k]] <- list(
      feature = feature_names[feat_idx],
      pct_change = pct_deviations[i, feat_idx],
      z_score = round(X_norm[i, feat_idx], 3),
      current_value = round(X[i, feat_idx], 4),
      baseline_value = round(baseline_mean[feat_idx], 4)
    )
  }

  windows_r_analysis[[i]] <- list(
    window_index = i - 1,
    euclidean_distance = round(euclidean_distances[i], 4),
    mahalanobis_distance = round(mahalanobis_distances[i], 4),
    similarity_score = round(cosine_similarities[i], 4),
    pc1 = pc1_vals[i],
    pc2 = pc2_vals[i],
    normalized_vector = round(as.numeric(X_norm[i, ]), 4),
    top_deviations = top_deviations
  )
}

# Build response payload
output_payload <- list(
  status = "success",
  engine = "Rscript (R) version 4.6.1",
  vector_dimension = P,
  sample_windows_count = N,
  mean_euclidean_distance = mean_euclidean_dist,
  mean_mahalanobis_distance = mean_mahalanobis_dist,
  mean_similarity_score = mean_similarity,
  pca_explained_variance = round(as.numeric(explained_var[1:min(2, length(explained_var))]), 4),
  feature_names = feature_names,
  baseline_mean = round(baseline_mean, 4),
  current_mean = mean_current_vector,
  feature_mean_pct_deviation = as.list(round(colMeans(pct_deviations), 2)),
  windows = windows_r_analysis
)

# Output JSON to stdout
cat(toJSON(output_payload, auto_unbox = TRUE, digits = 6))
