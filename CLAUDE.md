# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Capstone research project (CDA 490) by Derrick Omai investigating whether excluding demographic features from ML resume screening models mitigates bias when trained on historically biased data, or whether proxy variables (names) enable reconstruction of discrimination.

**Central Research Question:**
> Does excluding demographic features from model inputs mitigate bias in outcomes when models are trained on historically biased data, or do proxy variables (names) enable the reconstruction of discrimination?

**Methodology:** Uses matched-pair testing (Bertrand & Mullainathan, 2004) with callback labels calibrated to Quillian et al. (2017) meta-analysis findings (White:Black callback ratio = 1.36).

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
- **Scope: Race only** — Gender removed from analysis. Quillian et al. (2017) provides race-specific discrimination ratios only; including gender would require a separate empirical source.
- **Three experimental regimes** are central to the experiment:
  1. **Regime A (Qualifications Only)** — skills, experience, education (3 features)
  2. **Regime B (+ Names)** — Regime A + one-hot encoded names (48 features). Tests proxy discrimination.
  3. **Regime C (+ Explicit Race)** — Regime A + race_encoded. Upper bound on discrimination.
- **Matched-pair testing**: Bias is measured by comparing model scores on resume pairs that are identical in qualifications but differ in racial signals (24 names per racial group via SSA/Census data).
- **1,102 matched pairs**: 551 original resumes × 2 racial variants (White, Black).
- **`*_files/` directories** are auto-generated Quarto build artifacts — do not edit manually.

## Statistical Methods Used

- Chi-square tests for independence between demographics and predictions
- Cohen's d for effect size measurement
- Disparate impact ratio (80% rule from employment discrimination law)
- McNemar's test for paired model comparison
- Bootstrap resampling for confidence intervals
- Paired t-tests / Wilcoxon signed-rank tests for bias mitigation evaluation

## Research Context

Callback labels are synthetically generated with embedded racial discrimination, calibrated to Quillian et al. (2017) meta-analysis findings (White:Black callback ratio = 1.36). This tests whether models trained on historically biased data can be "debiased" by excluding demographic features — a common industry practice.

## Scope Decision Log (March 2, 2026)

Gender removed from analysis. Two options for name assignment were considered:
- **Option A**: One name per race — rejected because it collapses Regime B into Regime C (name perfectly encodes race)
- **Option B**: Multiple names per race (24 per group) — selected to preserve proxy discrimination testing

This maintains methodological consistency with Quillian while simplifying the analysis to race-only comparisons.
