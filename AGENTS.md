# Repository Guidelines

## Project Structure & Module Organization
- `rye.R` is the Rscript entry point; it orchestrates PCA scaling, reference aggregation, and NNLS ancestry estimation.
- `examples/` contains minimal PCA and population mapping inputs; duplicate or adapt these when validating changes.
- `logo/` holds branding assets; leave untouched unless design updates are coordinated with maintainers.
- Output files (`*.Q`, `*.fam`) land in the working directory using the prefix passed via `--out`/`--output`.

## Build, Test, and Development Commands
- `chmod +x rye.R` once after cloning to ensure the launcher is executable on new environments.
- `./rye.R -h` prints the optparse-driven help page and verifies required R packages load successfully.
- `./rye.R --eigenvec=examples/example.eigenvec --eigenval=examples/example.eigenval --pop2group=examples/pop2group.txt --rounds=5 --threads=2 --iter=5 --out=smoke` runs the bundled smoke test and produces reference outputs under `smoke-*`.
- `./rye.R ... --optimizer=gradient --grad-maxiter=50` switches the outer optimisation to the deterministic FISTA-based gradient loop; omit the flags to keep the legacy Gibbs annealing path.

## Coding Style & Naming Conventions
- Follow the existing 2-space indentation, `{` on the same line, and space after commas style seen throughout `rye.R`.
- Prefer lowerCamelCase for helper functions (`printDims`) and prefix exported utilities with `rye.` (`rye.scale`) to signal core API surface.
- Keep assignments with `=` for readability, and group related configuration constants at the top of the script.
- Run `styler::style_file("rye.R")` only after confirming it preserves current formatting conventions.

## Testing Guidelines
- There is no automated test harness yet; rely on the example run above and compare outputs to known ancestry proportions.
- Document any deterministic seeds or randomization knobs introduced; prefer deterministic defaults to ease review.

## Commit & Pull Request Guidelines
- Write descriptive commits in the style `feat: add NNLS warm-start` or `fix: guard eigenvector parsing`; avoid generic “update file” subjects seen in older history.
- Open PRs with: objective summary, input data used for validation, command log snippets, and follow-up tasks if accuracy trade-offs remain.
- Link related issues, request at least one reviewer familiar with PCA workflows, and attach before/after runtime metrics when optimization changes are proposed.

## Data & Configuration Tips
- Large PCA inputs should remain outside the repository; store only minimal fixtures and document acquisition steps in the PR.
- Update `examples/pop2group.txt` cautiously—downstream users rely on stable column headers (`Pop`, `Group`).
- If adding configuration flags, thread them through optparse with explicit defaults and document them in the README Quickstart section.
