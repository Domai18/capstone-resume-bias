# Methodology Documentation

This file documents all data processing decisions, criteria, and assumptions made throughout the project. Updated as work progresses.

---

## Phase 1: Data Collection

**Date completed:** February 9, 2026

### Resume Dataset

**Source:** [Kaggle Resume Dataset by snehaanbhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

**Original size:** 2,484 resumes across 24 job categories

**Filtration criteria:** Kept only resumes where `Category` matched one of our 5 target fields:

| Original Category | Our Field Label | Resumes Kept | Rationale |
|---|---|---|---|
| TEACHER | Education | 102 | Female-dominated (~89% female in US) — tests gender bias |
| HEALTHCARE | Healthcare | 115 | Female-dominated (~85% female nurses) — tests gender bias |
| INFORMATION-TECHNOLOGY | Technology | 120 | Male-dominated (~75% male) — tests gender bias |
| FINANCE | Finance | 118 | Male-dominated (~60% male) — tests gender bias |
| DIGITAL-MEDIA | Marketing | 96 | Closer to gender-balanced — baseline comparison |

**Total after filtering:** 551 resumes

**Columns retained:**
- `ID` — unique identifier from original dataset
- `Resume_str` — plain text of resume
- `Resume_html` — HTML formatted version
- `Category` — original category label
- `Field` — our standardized field label

**Output file:** `data/raw/resumes_filtered.csv`

---

### Job Descriptions Dataset

**Source:** Synthetically generated using Python Faker library

**Why synthetic:**
- Publicly available job description datasets had quality issues (e.g., 376 unique rows duplicated to 1.6M)
- Synthetic data gives us full control over job requirements
- Matches our resume field categories exactly

**Generation parameters:**

| Field | Job Titles | Skills Pool | Education Options | Experience Range |
|---|---|---|---|---|
| Education | Elementary School Teacher, Kindergarten Teacher, Special Education Teacher, Primary School Teacher, Early Childhood Educator | 12 education-specific skills | Bachelor's/Master's in Education | 1-10 years |
| Healthcare | Registered Nurse, ICU Nurse, Pediatric Nurse, ER Nurse, Medical-Surgical Nurse, Clinical Nurse | 13 nursing skills | BSN, ADN, MSN | 1-15 years |
| Technology | Software Engineer, Backend Developer, Full Stack Engineer, Software Developer, Application Developer, Systems Engineer | 16 technical skills | BS/MS in CS or related | 1-12 years |
| Finance | Financial Analyst, Budget Analyst, Investment Analyst, Senior Financial Analyst, Corporate Finance Analyst, FP&A Analyst | 13 finance skills | BS in Finance/Accounting/Economics, MBA | 1-10 years |
| Marketing | Marketing Manager, Brand Manager, Digital Marketing Manager, Marketing Coordinator, Content Marketing Manager, Growth Marketing Manager | 13 marketing skills | BS in Marketing/Communications/Business, MBA | 2-12 years |

**Generation logic:**
- 100 job postings per field (500 total)
- Each posting randomly selects: 1 job title, 4-7 required skills, 1 education requirement, 1 experience level
- Salary ranges calculated deterministically: `salary_min = base_salary + (years_experience × $3,000)`, `salary_max = salary_min + $15,000`. **Note:** Salary data is cosmetic only — not used in any analysis or modeling.
- Company names generated via Faker
- Random seed set to 42 for reproducibility

**Columns:**
- `job_id` — unique identifier
- `field` — career field (Education, Healthcare, Technology, Finance, Marketing)
- `job_title` — specific position title
- `company` — Faker-generated company name
- `required_skills` — comma-separated list of 4-7 skills
- `required_education` — degree requirement
- `years_experience_required` — minimum years required
- `salary_min` — lower bound of salary range
- `salary_max` — upper bound of salary range
- `job_description` — full text description

**Output file:** `data/raw/job_descriptions.csv`

---

## Phase 2: Data Preprocessing

**Date completed:** February 9, 2026

### Feature Extraction from Resume Text

Three features were extracted from each resume's raw text:

| Feature | Extraction Method | Range |
|---------|-------------------|-------|
| `skills_count` | Count of field-relevant keywords found in text | 0-15+ |
| `years_experience` | Parsed from date ranges (e.g., "2015 to 2020") in work history | 1-40 |
| `education_level` | Keyword matching for degree types | 1-5 scale |

**Education level encoding:**
- 1 = High School / No degree mentioned
- 2 = Associate's degree
- 3 = Bachelor's degree
- 4 = Master's degree
- 5 = Doctorate / PhD

**Skills keywords per field:**
- Education: teaching, classroom, curriculum, lesson plan, student, instruction, assessment, etc.
- Healthcare: patient, nursing, clinical, medical, care, medication, vital signs, HIPAA, etc.
- Technology: python, java, javascript, sql, programming, software, api, cloud, git, etc.
- Finance: financial, accounting, budget, forecast, analysis, excel, audit, GAAP, etc.
- Marketing: marketing, campaign, social media, seo, analytics, brand, content, digital, etc.

---

### Demographic Signal Assignment

**Methodology:** Following Bertrand & Mullainathan (2004), names were selected that research has shown to correlate with specific racial groups based on SSA baby name data and Census Bureau surname statistics.

**Two racial groups (gender removed per Quillian scope):**

| Race | Example Names |
|------|---------------|
| White | Brad Anderson, Greg Baker, Todd Miller, Emily Walsh, Anne Peterson, Sarah Mitchell |
| Black | Jamal Washington, Leroy Harris, Tyrone Jackson, Lakisha Washington, Tanisha Harris, Aisha Jackson |

Each resume was assigned one name randomly from the full pool (24 names per racial group, 48 total). Multiple names per race (Option B) preserves the proxy discrimination research design.

---

### Synthetic Callback Label Generation (REVISED)

> **⚠️ METHODOLOGY PIVOT (February 2026)**
>
> Original design used bias-free labels to test whether models *introduce* bias. However, this approach has limited practical relevance: if training labels contain no bias signal, models cannot learn systematic discrimination.
>
> **Revised approach:** Generate callback labels that reflect *historical hiring discrimination*, calibrated to empirical audit study findings. This tests whether models trained on biased data can be "debiased" by excluding demographic features — a common industry practice.

---

#### Rationale for Pivot

**Problem with bias-free labels:**
- If labels are generated purely from qualifications, and names are randomly assigned, there is no correlation between demographics and outcomes in the training data.
- Models cannot learn bias that doesn't exist in the training signal.
- Expected finding: minimal/no disparities across all feature sets.

**Why historically-biased labels matter:**
- Real ATS systems are trained on decades of human hiring decisions that reflect documented discrimination.
- The practical question is: *"Can companies reduce bias by removing demographic features from model inputs, even when training data reflects historical bias?"*
- This requires training labels that contain bias.

---

#### Calibration Source: Quillian et al. (2017)

**Citation:** Quillian, L., Pager, D., Hexel, O., & Midtbøen, A. H. (2017). Meta-analysis of field experiments shows no change in racial discrimination in hiring over time. *Proceedings of the National Academy of Sciences*, 114(41), 10870-10875.

**Key findings (random-effects meta-analysis, 1989–2015):**

| Comparison | Callback Advantage | 95% CI | Ratio |
|------------|-------------------|--------|-------|
| White vs Black | 36% more callbacks | 25–47% | 1.36 |
| White vs Latino | 24% more callbacks | 15–33% | 1.24 |

**Current implementation:** White:Black ratio = 1.36

**Potential extensions:** The Latino callback data (ratio ≈ 1.24) could support future multi-group analysis if Hispanic/Latino names are added.

**Why this source:**
- Meta-analysis provides robust estimate across multiple studies
- Directly measures callback rates (our target variable)
- Establishes empirically-grounded bias magnitude (not arbitrary)
- Provides confidence intervals for sensitivity analysis

---

#### Scope: Race Only (Gender Removed)

**Decision:** Focus exclusively on racial bias (White vs Black), not gender.

**Rationale:**
- Quillian et al. (2017) reports race-specific discrimination only
- Adding gender would require separate empirical source or arbitrary parameters
- Simplicity: 2-group comparison is cleaner to analyze and explain

**Implementation Change (March 2, 2026):**

Original design used 4 demographic groups (White Male, White Female, Black Male, Black Female) creating 2,204 matched pairs (551 resumes × 4 variants). This has been simplified to 2 racial groups (White, Black) creating **1,102 matched pairs** (551 resumes × 2 variants).

**Name Assignment - Option B (Multiple Names per Race):**

Two options were considered for assigning names after removing gender:
- **Option A:** One name per race (e.g., "Greg" = White, "Jamal" = Black)
- **Option B:** Multiple names per race (24 names per group, merged from original male/female lists)

**Option B was selected** because Option A would collapse Regime B into Regime C:
- With one name per race, name perfectly encodes race (1:1 mapping)
- Model trivially learns race from name — no proxy inference happening
- Defeats the purpose of testing whether names *enable* proxy discrimination

With Option B (24 names per racial group):
- Models see varied name features
- Must *learn* racial patterns embedded in callback rates
- Mirrors real-world proxy discrimination (names signal but don't perfectly encode race)

**Updated Name Lists:**

| Race | Names (24 per group) |
|------|---------------------|
| White | Brad Anderson, Greg Baker, Todd Miller, Brett Wilson, Brendan Taylor, Geoffrey Thomas, Brett Martin, Jay Garcia, Neil Johnson, Todd Williams, Matthew Davis, Connor Brown, Emily Walsh, Anne Peterson, Meredith Collins, Carrie Stewart, Kristen Murphy, Laurie Rogers, Sarah Mitchell, Jill Anderson, Allison Martin, Emily Taylor, Claire Johnson, Molly Davis |
| Black | Jamal Washington, Leroy Harris, Tyrone Jackson, Darnell Robinson, Hakim Williams, Rasheed Jones, Tremayne Brown, Kareem Davis, Terrence Moore, Jermaine Taylor, DeShawn Wilson, Marquis Thomas, Lakisha Washington, Tanisha Harris, Aisha Jackson, Ebony Robinson, Keisha Williams, Tamika Jones, Latoya Brown, Kenya Davis, Precious Moore, Imani Taylor, Aaliyah Wilson, Shanice Thomas |

---

#### Label Generation Process

**Step 1: Compute Qualification Score**

```
Q_i = w_exp × standardize(years_experience)
    + w_skill × standardize(skills_count)
    + w_edu × standardize(education_level)

Default weights: w_exp = 0.4, w_skill = 0.35, w_edu = 0.25
```

Standardization: z-score within each field to ensure comparability.

**Step 2: Convert to Baseline Callback Probability**

```
p_base = sigmoid(α + β × Q_i)
```

Where:
- α (intercept) controls overall callback rate (tuned to achieve ~10% baseline)
- β (slope) controls how much qualifications matter

**Step 3: Apply Racial Bias**

```
p_biased = sigmoid(α + β × Q_i + γ_race × is_black)
```

Where:
- γ_race is calibrated to produce White:Black callback ratio ≈ 1.36
- is_black = 1 for Black applicants, 0 for White applicants
- γ_race is negative (penalty for Black applicants)

**Step 4: Calibration Procedure**

```python
# Pseudocode for finding γ_race
target_ratio = 1.36  # Quillian et al.

for γ_race in np.linspace(-2, 0, 100):
    # Generate labels with this γ_race
    p_white = sigmoid(α + β × Q + 0)        # is_black = 0
    p_black = sigmoid(α + β × Q + γ_race)   # is_black = 1

    ratio = mean(p_white) / mean(p_black)
    if abs(ratio - target_ratio) < tolerance:
        break
```

**Step 5: Sample Callback Labels**

```
y_callback ~ Bernoulli(p_biased)
```

Each resume variant receives a stochastic callback label based on its biased probability.

---

#### Proxy Analysis: Names Only

**Decision:** Use names as the sole demographic proxy.

**Rationale:**
- Searched resume corpus for other potential proxies:
  - Professional affiliations (NSBE, NABA, SHPE): 0 matches
  - HBCUs: 19 resumes (3.4%) — too sparse
  - Divine Nine Greek organizations: 4 matches — negligible
- Names provide clean experimental isolation of proxy effects
- Directly mirrors Bertrand & Mullainathan (2004) methodology

---

#### Three Experimental Regimes

| Regime | Features | Research Question |
|--------|----------|-------------------|
| **A: Qualifications Only** | skills_count, years_experience, education_level | Does removing all demographic info eliminate learned bias? |
| **B: Qualifications + Names** | Regime A + name (one-hot encoded) | Do names act as proxies that reconstruct bias? |
| **C: Qualifications + Explicit Demographics** | Regime A + race + gender | Upper bound: how bad is explicit discrimination? |

**Hypothesis:**
- Regime A: Reduced but possibly non-zero bias (if qualifications correlate with demographics in training data due to biased labels)
- Regime B: Higher bias than A (names enable proxy discrimination)
- Regime C: Highest bias (direct demographic access)

---

#### Output Files

| File | Location | Contents |
|------|----------|----------|
| `resumes_callback_labels.csv` | `data/processed/` | 1,102 matched pairs with Quillian-calibrated callback labels |
| `calibration_results.json` | `data/processed/` | γ_race value, achieved callback ratio, overall callback rate |

---

### Matched Pair Creation

**Purpose:** Enable direct bias measurement by comparing model predictions on identical resumes with different racial signals.

**Process:**
1. Take each of the 551 original resumes
2. Duplicate it 2 times (race-only design, gender removed)
3. Assign each duplicate a name from a different racial group
4. All qualification features remain identical

**Result:** 551 × 2 = **1,102 matched resume records**

**Example matched set:**

| Original ID | Name | Race | Skills | Experience | Education | Qual Score |
|-------------|------|------|--------|------------|-----------|------------|
| 001 | Greg Baker | white | 8 | 6 | 3 | 72.4 |
| 001 | Jamal Washington | black | 8 | 6 | 3 | 72.4 |

If a model assigns different scores to these two versions, that difference is attributable solely to the racial signal (name).

---

### Feature Set Definitions (REVISED)

Three experimental regimes for comparative modeling:

| Regime | Features Included | Purpose |
|--------|-------------------|---------|
| **A: Qualifications Only** | skills_count, years_experience, education_level | Tests whether removing demographic info mitigates bias learned from biased training labels |
| **B: Qualifications + Names** | Regime A + name (one-hot encoded, 48 features) | Tests whether names act as proxies to reconstruct discrimination |
| **C: Qualifications + Explicit Race** | Regime A + race_encoded (binary) | Diagnostic upper bound on possible discrimination |

**Key differences from original design:**
1. Models are now trained on *biased* callback labels (Quillian-calibrated). The question shifts from "do models introduce bias?" to "does excluding demographic features reduce bias learned from historically biased training data?"
2. Gender removed from all regimes — race-only analysis per Quillian et al. (2017) scope.

---

### Output Files

| File | Location | Contents |
|------|----------|----------|
| `resumes_processed.csv` | `data/processed/` | 2,204 matched pairs with all features |
| `resumes_with_features.csv` | `data/processed/` | 551 original resumes with extracted features |
| `feature_sets.json` | `data/processed/` | Feature list definitions for each set |

---

## Phase 3: Exploratory Data Analysis

**Status:** Not started

---

## Phase 4: Model Training (REVISED)

**Status:** To be re-run with Quillian-calibrated labels

**Approach:**

For each experimental regime (A, B, C), train:

**Classification models** (predict `callback`):
- Logistic Regression (interpretable baseline)
- Random Forest Classifier (ensemble method, common in industry)

**Simplified from original plan:** Reduced from 4 models to 2 for clarity. Neural networks and gradient boosting added complexity without substantially different bias patterns for this dataset size.

**Training procedure:**
1. Generate Quillian-calibrated callback labels
2. Split data: 80% train, 20% test (stratified by field and demographic group)
3. Train each model under each regime
4. Save trained models for bias evaluation

**Output:**
- 6 trained models (2 models × 3 regimes)
- `trained_models_v2.pkl`

---

### Classification Metrics Explained

| Metric | What It Measures | Relevance to Bias Detection |
|--------|------------------|----------------------------|
| **Accuracy** | Percentage of correct predictions (hired + not hired) | Confirms the model works; if accuracy drops when demographics are added, the model may be overfitting to irrelevant features |
| **Precision** | Of those predicted "hire," what % were actually hired? | Different precision by demographic group = different error rates in recommendations |
| **Recall** | Of those who should be hired, what % did the model find? | **Critical for bias** — lower recall for a group means qualified candidates from that group are being missed |
| **F1 Score** | Harmonic mean of precision and recall | Different F1 by group = model performs worse for some demographics overall |
| **AUC** | How well the model separates "hire" from "not hire" across all thresholds | Different AUC by group = model fundamentally evaluates groups differently |

**How to interpret for bias:**

When these metrics are calculated separately for each demographic group in Phase 5, differences indicate bias:

- **Lower recall for Black names**: Qualified Black candidates are being overlooked at higher rates than equally qualified white candidates
- **Different precision by group**: The model's "hire" recommendations are more reliable for some groups than others
- **Different AUC by group**: The model's ability to distinguish qualified from unqualified candidates varies by demographics

**Example finding:**
> "The Random Forest classifier achieved 82% accuracy overall. However, recall for white candidates was 78% while recall for Black candidates was 65%. This 13-percentage-point gap indicates the model systematically overlooks qualified Black applicants despite identical qualifications in the matched-pair design."

---

### Regression Metrics Explained

| Metric | What It Measures | Relevance to Bias Detection |
|--------|------------------|----------------------------|
| **MAE (Mean Absolute Error)** | Average absolute difference between predicted and actual scores | Lower is better; shows typical prediction error in points |
| **RMSE (Root Mean Squared Error)** | Similar to MAE but penalizes large errors more | Sensitive to outliers; useful for detecting inconsistent predictions |
| **R²** | Proportion of variance explained by the model (0-1) | Higher is better; 0.8+ means model explains most variation in scores |

**How to interpret for bias:**

In Phase 5, we compare **predicted scores** across demographic groups for matched pairs:

- **Mean score difference**: "Black names scored 5 points lower on average"
- **Cohen's d effect size**: Standardized measure of the gap (0.2 = small, 0.5 = medium, 0.8 = large)

Regression provides more granular bias measurement than classification because it reveals the *magnitude* of bias, not just whether someone crossed the hire threshold.

---

### Output Files

| File | Location | Contents |
|------|----------|----------|
| `classification_results.csv` | `results/tables/` | Performance metrics for all 12 classification models |
| `regression_results.csv` | `results/tables/` | Performance metrics for all 12 regression models |
| `trained_models.pkl` | `data/processed/` | Saved models for bias testing in Phase 5 |

---

## Phase 5: Bias Testing (REVISED)

**Status:** Not started

**Approach: Matched-Pair Audit**

For each original resume (551 base resumes × 2 racial variants = 1,102 records):

1. **Generate predictions** under each regime:
   - Regime A: Model sees only qualifications
   - Regime B: Model sees qualifications + name
   - Regime C: Model sees qualifications + race + gender

2. **Compare predicted callback rates** across demographic groups within matched sets.

**Metrics:**

| Metric | What It Measures | Interpretation |
|--------|------------------|----------------|
| **Callback Rate by Group** | P(callback=1 \| demographic_group) | Raw disparity |
| **Disparate Impact Ratio** | min_group_rate / max_group_rate | < 0.80 = legal concern (EEOC 80% rule) |
| **White:Black Callback Ratio** | callback_rate_white / callback_rate_black | Compare to 1.36 baseline in training labels |
| **Cohen's d** | Standardized mean difference in predicted probabilities | Effect size: 0.2=small, 0.5=medium, 0.8=large |
| **Chi-square test** | Independence of predictions and demographics | Statistical significance |

**Key Comparisons:**

| Comparison | Question Answered |
|------------|-------------------|
| Regime A vs Training Labels | Does removing demographics reduce the 1.36 ratio? |
| Regime B vs Regime A | Do names reconstruct bias that was "removed"? |
| Regime C vs Regime B | How much worse is explicit demographic access? |

**Expected Findings:**
- Training labels: White:Black ratio ≈ 1.36 (by construction)
- Regime C: Ratio ≈ 1.36 (model has direct access to race)
- Regime B: Ratio between 1.0 and 1.36 (names partially reconstruct bias)
- Regime A: Ratio closer to 1.0 (but possibly > 1.0 if qualifications correlate with demographics)

---

## Phase 6: Statistical Analysis

**Status:** Not started

---

## Phase 7: Visualization & Reporting

**Status:** Not started

---

## Methods Rationale

This section explains why specific methods were chosen over alternatives.

### Machine Learning Models

| Model | Why Chosen | Why Not Alternatives |
|-------|------------|---------------------|
| **Logistic Regression** | Simple, interpretable baseline. Easy to see which features drive predictions. If bias appears here, it's clearly from the features, not model complexity. | — |
| **Random Forest** | Captures non-linear relationships. Popular in industry hiring tools. Tests whether ensemble methods amplify or reduce bias. | — |
| **Gradient Boosting** | Often the best performer on tabular data. Used in production ML systems. Tests state-of-the-art performance vs. fairness tradeoff. | — |
| **Neural Network** | Tests whether deep learning introduces different bias patterns than traditional ML. Can learn subtle correlations. | — |

**Models not used:**
- **SVM:** Less common in hiring tools, harder to interpret, doesn't add much over logistic regression for this use case.
- **Naive Bayes:** Assumes feature independence, which doesn't hold for resume data (skills correlate with experience).
- **Large Language Models:** Overkill for structured tabular data, harder to analyze for bias, and outside the scope of this study.

---

### Bias Detection: Matched Pairs Design

**Why matched pairs:**
- Directly isolates demographic effect — no confounding variables
- Based on established audit study methodology (Bertrand & Mullainathan, 2004)
- Simple to explain and defend
- Produces clear, interpretable results (e.g., "Model scored Black names 15% lower")

**Why not alternatives:**
- **Observational analysis** (comparing outcomes across groups without matching): Cannot separate demographic effect from qualification differences. If Black candidates score lower, is it bias or different qualifications? Matched pairs eliminate this ambiguity.
- **Counterfactual fairness models:** More complex, harder to implement and explain, unnecessary for a focused study.

---

### Statistical Tests

| Test | Purpose | Why This Test |
|------|---------|---------------|
| **Chi-square** | Test if demographics and hiring are independent | Standard for categorical variables. Answers: "Is there a statistically significant association?" |
| **Cohen's d** | Measure effect size | Chi-square tells you *if* there's a difference; Cohen's d tells you *how big*. Important for practical significance. |
| **Disparate Impact Ratio (80% rule)** | Legal standard for discrimination | Used in actual employment law. If one group's hire rate is <80% of another's, it's legally suspect. Directly relevant to the research question. |
| **McNemar's test** | Compare paired model predictions | Tests whether two models differ significantly on matched pairs. Appropriate for paired binary outcomes. |

**Tests not used:**
- **t-tests:** Designed for continuous outcomes; our hire decision is binary.
- **ANOVA:** For comparing 3+ group means; we're comparing proportions and paired predictions.
- **Regression with interaction terms:** Could work, but more complex to explain. Matched pairs + chi-square is cleaner.

---

### Three Experimental Regimes (REVISED)

| Regime | What It Tests |
|--------|---------------|
| **A: Qualifications only** | Can bias learned from historical data be eliminated by excluding demographic features? |
| **B: + Names** | Do names act as proxies that allow models to reconstruct discrimination? |
| **C: + Explicit race** | Upper bound: maximum discrimination when model has direct race access. |

This progression produces a clear narrative:
1. Training data contains historical bias (White:Black ratio = 1.36)
2. Regime A tests the "just remove protected attributes" debiasing strategy
3. Regime B reveals whether proxy variables undermine this strategy
4. Regime C establishes the ceiling for comparison

**Scope:** Race only (White vs Black). Gender removed per Quillian et al. (2017) scope — our calibration source reports race-specific discrimination only.

**Central Research Question:**

> Does excluding demographic features from model inputs mitigate bias in outcomes when models are trained on historically biased data, or do proxy variables (names) enable the reconstruction of discrimination?

---

### Two Prediction Targets: Scores and Hire Decisions

We use **both** numerical scores and binary hire labels:

| Target | Model Type | What It Measures |
|--------|------------|------------------|
| `qualification_score` (0-100) | Regression | Granular bias — "Black names scored 5 points lower on average" |
| `hired` (0 or 1) | Classification | Practical impact — "Black names were hired 10% less often" |

**Why both:**
- **Numerical scores** detect subtle bias that binary outcomes might miss. A score of 41 vs 39 both result in "not hired," but the 2-point gap reveals bias.
- **Binary hire labels** map to real-world decisions and allow calculation of the **disparate impact ratio (80% rule)** used in employment law.

**Bias metrics from each:**

| From Scores (Regression) | From Hire Labels (Classification) |
|--------------------------|-----------------------------------|
| Mean score difference by demographic group | Hire rate by demographic group |
| Cohen's d effect size | Disparate impact ratio |
| Score distribution plots | Chi-square test for independence |

---

### Computational Requirements

**This project requires minimal computing power.**

- Dataset size: 2,204 rows — tiny by ML standards
- Logistic Regression: Trains in <1 second
- Random Forest: A few seconds
- Gradient Boosting: A few seconds
- Neural Network: Under 1 minute (simple architecture, not deep learning)

No GPU, cloud computing, or special hardware required. A standard laptop handles everything easily.

---

## Key References

- Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg More Employable Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination. *American Economic Review*.
- **Quillian, L., Pager, D., Hexel, O., & Midtbøen, A. H. (2017). Meta-analysis of field experiments shows no change in racial discrimination in hiring over time. *Proceedings of the National Academy of Sciences*, 114(41), 10870-10875.** — Source for callback ratio calibration (1.36 White:Black ratio).
- SSA Baby Names Database: https://www.ssa.gov/oact/babynames/
- US Census Bureau Surname Data: https://www.census.gov/topics/population/genealogy/data/2010_surnames.html
