#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(nnls)
})

baseline_env <- new.env()
new_env <- new.env()

source("rye_nnls.R", local = baseline_env)
source("rye.R", local = new_env)

fullPCA <- read.table("examples/example.eigenvec", header = FALSE, row.names = NULL)
rownames(fullPCA) <- fullPCA[, 2]
fullEigenVal <- read.table("examples/example.eigenval", header = FALSE)[, 1]
pop2group <- read.table("examples/pop2group.txt", header = TRUE, stringsAsFactors = FALSE)

referenceGroups <- pop2group$Group
names(referenceGroups) <- pop2group$Pop
fam <- as.matrix(fullPCA[, c(1, 2)])
colnames(fam) <- c("population", "id")
rownames(fam) <- fam[, "id"]

fullPCA_mat <- as.matrix(fullPCA[, 3:ncol(fullPCA)])
fullPCA_mat <- new_env$rye.scale(fullPCA_mat)

weight_full <- fullEigenVal / max(fullEigenVal)

regionFAM <- fam
regionFAM[fam[, 1] %in% names(referenceGroups), 1] <-
  referenceGroups[fam[fam[, 1] %in% names(referenceGroups), 1]]

referenceGroupsUnique <- unique(referenceGroups)
names(referenceGroupsUnique) <- referenceGroupsUnique
referencePops <- referenceGroupsUnique

alpha_vec <- rep(0.001, length(referencePops))
names(alpha_vec) <- referencePops

pcs <- 20
X_use <- fullPCA_mat[, seq(pcs)]
weight_vec <- weight_full[seq(pcs)]

run_opt <- function(env, seed = 123) {
  set.seed(seed)
  env$rye.optimize(
    X = X_use,
    fam = regionFAM,
    referencePops = referencePops,
    referenceGroups = referenceGroupsUnique,
    startSD = 0.01,
    endSD = 0.005,
    threads = 1,
    iterations = 5,
    rounds = 5,
    attempts = 4,
    weight = weight_vec,
    alpha = alpha_vec,
    optimizeWeight = TRUE,
    optimizeAlpha = TRUE
  )
}

baseline_opt <- run_opt(baseline_env, seed = 123)
new_opt <- run_opt(new_env, seed = 123)

baseline_weight <- baseline_opt[[3]]
baseline_means <- baseline_opt[[4]]
new_weight <- new_opt[[3]]
new_means <- new_opt[[4]]

baseline_time <- system.time({
  baseline_raw <- baseline_env$rye.predict(
    X = X_use,
    means = baseline_means,
    weight = baseline_weight,
    referenceGroups = NULL
  )
  baseline_groups <- baseline_env$rye.predict(
    X = X_use,
    means = baseline_means,
    weight = baseline_weight,
    referenceGroups = referenceGroupsUnique
  )
})

new_time <- system.time({
  new_raw <- new_env$rye.predict(
    X = X_use,
    means = new_means,
    weight = new_weight,
    referenceGroups = NULL,
    solver_iters = 300,
    solver_tol = 1e-6
  )
  new_groups <- new_env$rye.predict(
    X = X_use,
    means = new_means,
    weight = new_weight,
    referenceGroups = referenceGroupsUnique,
    solver_iters = 300,
    solver_tol = 1e-6
  )
})

weightedX <- X_use * matrix(weight_vec, nrow = nrow(X_use), ncol = length(weight_vec), byrow = TRUE)

pred_weight_baseline <- baseline_raw %*% baseline_means
pred_weight_new <- new_raw %*% new_means

residual_nnls <- rowSums((pred_weight_baseline - weightedX) ^ 2)
residual_new <- rowSums((pred_weight_new - weightedX) ^ 2)

diff_matrix <- baseline_groups - new_groups
l1_dist <- rowSums(abs(diff_matrix))
l2_dist <- sqrt(rowSums(diff_matrix ^ 2))

metrics <- data.frame(
  sample = rownames(X_use),
  r_nnls = residual_nnls,
  r_simplex = residual_new,
  l1 = l1_dist,
  l2 = l2_dist
)

dir.create("bench", showWarnings = FALSE)
write.table(metrics, file = "bench/metrics.tsv", sep = "\t", row.names = FALSE, quote = FALSE)

summary_lines <- c(
  sprintf("Residual mean (nnls): %.6f", mean(residual_nnls)),
  sprintf("Residual mean (fista): %.6f", mean(residual_new)),
  sprintf("Residual median diff: %.6f", median(residual_new - residual_nnls)),
  sprintf("Residual 95th pct diff: %.6f", quantile(residual_new - residual_nnls, 0.95)),
  sprintf("Mean L1 distance: %.6f", mean(l1_dist)),
  sprintf("Median L1 distance: %.6f", median(l1_dist)),
  sprintf("Mean L2 distance: %.6f", mean(l2_dist)),
  sprintf("Predict time (nnls) [s]: %.3f", baseline_time[["elapsed"]]),
  sprintf("Predict time (fista) [s]: %.3f", new_time[["elapsed"]]),
  sprintf("Predict speedup: %.2fx", baseline_time[["elapsed"]] / new_time[["elapsed"]])
)

baseline_group_means <- colMeans(baseline_groups)
new_group_means <- colMeans(new_groups)
group_diff <- abs(baseline_group_means - new_group_means)
summary_lines <- c(
  summary_lines,
  "Group mean abs diff:",
  paste(names(group_diff), sprintf("%.6f", group_diff))
)

for (grp in colnames(baseline_groups)) {
  summary_lines <- c(
    summary_lines,
    sprintf("Group %s Pearson: %.6f", grp, cor(baseline_groups[, grp], new_groups[, grp])),
    sprintf(
      "Group %s Spearman: %.6f",
      grp,
      suppressWarnings(cor(baseline_groups[, grp], new_groups[, grp], method = "spearman"))
    )
  )
}

writeLines(summary_lines, "bench/summary.txt")

cat(paste(summary_lines, collapse = "\n"), "\n")
