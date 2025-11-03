# Benchmark Notes

## Dataset & Scope
- Inputs: `examples/example.eigenvec`, `examples/example.eigenval`, and `examples/pop2group.txt` bundled with the repository.
- Individuals: All 3 399 rows present in the example eigenvector file were included. No subsampling or holdout set was used.
- Preprocessing: PCs scaled via `rye.scale`, eigenvalues converted into weights (`weight_vec = eigenval / max(eigenval)`), and population/group mappings derived from `pop2group.txt`.

## Procedure
1. Created `rye_nnls.R` (legacy solver) and the updated `rye.R` (FISTA solver).
2. Ran `bench/run_benchmarks.R` inside the `rye_env` conda environment:
   ```bash
   conda run -n rye_env --no-capture-output Rscript bench/run_benchmarks.R
   ```
   - Both scripts share identical optimization hyperparameters (`pcs=20`, `rounds=5`, `iter=5`, `attempts=4`).
   - `warm_starts` were enabled automatically inside the new solver; no manual caching beyond what `rye.predict` already performs.
   - Residuals are computed against the weight-adjusted PCA matrix (`X_use * weight_vec`), matching legacy behavior.
3. The script measures:
   - Wall-clock prediction time for NNLS vs. FISTA (single-threaded).
   - Per-individual residual sums of squares (`‖Aw - x‖²`), **L₁** (`∑|w_nnls - w_fista|`) and **L₂** (`√∑(w_nnls - w_fista)²`) distances between ancestry vectors, and solver iteration counts.
   - Group-level aggregate differences plus Pearson/Spearman correlations.

## Outputs
- `bench/metrics.tsv`: row-wise metrics for every individual.
- `bench/summary.txt`: aggregated statistics, including L₁ / L₂ summaries and speedup factor.

## Repro Tips
- Activate `rye_env` (created via `mamba create -n rye_env r-base r-hmisc r-optparse r-crayon r-nnls`).
- Ensure the repo is in a writable location; the script overwrites the two benchmark files above.
- For custom datasets, adjust the `pcs` count and file paths inside `bench/run_benchmarks.R`. Keep the NNLS snapshot (`rye_nnls.R`) untouched to preserve the baseline.
- Multithreading: the C++ FISTA kernel honours `OMP_NUM_THREADS`. For example:
  ```bash
  conda run -n rye_env OMP_NUM_THREADS=16 Rscript bench/run_benchmarks.R
  ```
  On the bundled dataset the prediction step improved modestly (≈0.21 s → 0.20 s) because only 3 399 samples are processed; larger cohorts should exhibit clearer scaling.
