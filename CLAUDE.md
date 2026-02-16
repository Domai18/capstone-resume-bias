# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Capstone research project (CDA 490) by Derrick Omai investigating how ML-based automated resume screening introduces or amplifies demographic bias. Uses matched-pair testing across multiple models to detect disparate outcomes by race and gender, following the audit study methodology of Bertrand & Mullainathan (2004).

## Technology Stack

- **Quarto** (v1.8+) for literate programming documents (.qmd)
- **Python 3** via Jupyter kernel
- **Core libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- Output format: HTML (configured in each .qmd YAML front matter)

## Common Commands

Render a single document:
```
quarto render notebooks/01_data_collection.qmd
```

Render all project documents:
```
quarto render
```

Preview with live reload:
```
quarto preview notebooks/01_data_collection.qmd
```

## Project Structure

```
data/raw/           — Original downloaded datasets (never modify)
data/processed/     — Cleaned, transformed, labeled data
notebooks/          — Numbered .qmd files (01_ through 07_), one per pipeline stage
src/                — Reusable Python modules imported by notebooks
results/figures/    — Saved plots for final report
results/tables/     — Exported statistical test results
```

## Key Conventions

- **Data immutability**: Files in `data/raw/` are never modified. All transformations happen in code and output to `data/processed/`.
- **Notebook ordering**: Numbered prefixes (01_, 02_, ...) indicate execution order and pipeline stage.
- **Three feature sets** are central to the experiment:
  1. **Baseline** — objective qualifications only (experience, skills, education)
  2. **Expanded** — adds demographic proxy variables (names, geography)
  3. **Problematic** — intentionally includes features encoding societal biases
- **Matched-pair testing**: Bias is measured by comparing model scores on resume pairs that are identical in qualifications but differ in demographic signals (names correlated with race/gender via SSA/Census data).
- **`*_files/` directories** are auto-generated Quarto build artifacts — do not edit manually.

## Statistical Methods Used

- Chi-square tests for independence between demographics and predictions
- Cohen's d for effect size measurement
- Disparate impact ratio (80% rule from employment discrimination law)
- McNemar's test for paired model comparison
- Bootstrap resampling for confidence intervals
- Paired t-tests / Wilcoxon signed-rank tests for bias mitigation evaluation

## Research Context

Synthetic hire/no-hire labels are used because real hiring outcome data is not publicly available. This is by design — it allows studying whether ML models introduce bias beyond what exists in the labeling process itself.
