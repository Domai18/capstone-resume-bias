# Development Roadmap

> **⚠️ METHODOLOGY PIVOT (February 2026):** Switched from bias-free labels to Quillian-calibrated historically-biased labels. See METHODOLOGY.md for full rationale.

## Phase 1: Environment & Data Foundation ✅
- Set up Python environment with required libraries (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Find and download a resume dataset (e.g., Kaggle resume datasets)
- Load the raw data and do initial inspection: what columns exist, how many rows, what's missing
- **Deliverable**: `notebooks/data_collection.qmd`

## Phase 2: Data Preprocessing & Demographic Signals ✅
- Clean the data (handle missing values, standardize formats)
- Engineer features: years of experience, education level, skill counts
- Assign demographic signals using SSA/Census naming patterns (Bertrand & Mullainathan methodology)
- Create matched pairs: 551 resumes × 4 demographic variants = 2,204 records
- **Deliverable**: `notebooks/data_preprocessing.qmd`, `data/processed/resumes_with_features.csv`

## Phase 3: Exploratory Data Analysis ✅
- Examine distributions of features across demographic groups
- Check for existing correlations between demographic signals and qualifications
- **Deliverable**: `notebooks/eda.qmd`

## Phase 4: Quillian-Calibrated Label Generation 🔄 NEW
- Generate callback labels reflecting historical discrimination
- Calibrate γ_race to achieve White:Black callback ratio ≈ 1.36 (Quillian et al., 2017)
- Target overall callback rate ~10%
- **Deliverable**: `notebooks/04_label_generation.qmd`, `data/processed/resumes_callback_labels.csv`

## Phase 5: Model Training (Revised)
- Train Logistic Regression and Random Forest under three regimes:
  - Regime A: Qualifications only
  - Regime B: Qualifications + Names
  - Regime C: Qualifications + Explicit Demographics
- **Deliverable**: `notebooks/05_model_training.qmd`, `data/processed/trained_models_v2.pkl`

## Phase 6: Bias Testing (Matched-Pair Audit)
- Run predictions on all 2,204 matched pairs under each regime
- Calculate callback rates by demographic group
- Compute disparate impact ratios, Cohen's d, chi-square tests
- Compare regime results to assess debiasing effectiveness
- **Deliverable**: `notebooks/06_bias_testing.qmd`, `results/tables/bias_metrics.csv`

## Phase 7: Statistical Analysis & Visualization
- Formal hypothesis tests (chi-square, McNemar's)
- Bootstrap confidence intervals for callback ratios
- Visualization: callback rate bar charts, ROC curves by group, regime comparison plots
- **Deliverable**: `notebooks/07_analysis_and_viz.qmd`, `results/figures/`

## Central Research Question

> Does excluding demographic features from model inputs mitigate bias in outcomes when models are trained on historically biased data, or do proxy variables (names) enable the reconstruction of discrimination?
