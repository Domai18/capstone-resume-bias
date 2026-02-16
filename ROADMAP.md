# Development Roadmap

## Phase 1: Environment & Data Foundation
- Set up Python environment with required libraries (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Find and download a resume dataset (e.g., Kaggle resume datasets)
- Load the raw data and do initial inspection: what columns exist, how many rows, what's missing
- **Deliverable**: `notebooks/01_data_collection.qmd` that loads data and prints summary statistics

## Phase 2: Data Preprocessing & Demographic Signals
- Clean the data (handle missing values, standardize formats)
- Engineer features: years of experience, education level, skill counts, etc.
- Assign demographic signals using SSA/Census naming patterns (Bertrand & Mullainathan methodology)
- Generate synthetic hire/no-hire labels with a defined labeling strategy
- Build three feature sets: baseline (qualifications only), expanded (with proxy variables), problematic (encoding known biases)
- **Deliverable**: `notebooks/02_data_preprocessing.qmd` with clean dataset saved to `data/processed/`

## Phase 3: Exploratory Data Analysis
- Examine distributions of features across demographic groups
- Check for existing correlations between demographic signals and qualifications
- Visualize the label distribution — are hire rates balanced?
- **Deliverable**: `notebooks/03_eda.qmd` with charts and summary findings

## Phase 4: Baseline Modeling
- Train all four model types (logistic regression, random forest, gradient boosting, neural network) on the baseline feature set only (no demographic info)
- Evaluate with accuracy, precision, recall, F1, and AUC
- Establish what the models can do with purely objective features
- **Deliverable**: `notebooks/04_baseline_models.qmd` with model performance comparison table

## Phase 5: Bias Testing (Matched Pairs)
- Generate matched-pair test resumes (identical qualifications, different demographic signals)
- Run each trained model on matched pairs
- Record score differences between demographically different but equally qualified candidates
- Repeat with models trained on the expanded and problematic feature sets
- **Deliverable**: `notebooks/05_bias_testing.qmd` with raw results of matched-pair experiments

## Phase 6: Statistical Analysis
- Chi-square tests for independence between demographics and predictions
- Cohen's d for effect size of score differences
- Disparate impact ratios (80% rule threshold check)
- McNemar's test comparing model pairs
- Bootstrap confidence intervals
- Sensitivity analysis varying demographic signal assignments
- **Deliverable**: `notebooks/06_statistical_tests.qmd` with significance results

## Phase 7: Visualization & Final Report
- Stratified ROC curves (one curve per demographic group per model)
- Fairness-accuracy Pareto frontiers
- Feature importance distributions showing which features drive bias
- Compile findings into final presentation
- **Deliverable**: `notebooks/07_visualization.qmd` and polished figures in `results/figures/`
