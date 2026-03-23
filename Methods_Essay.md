# Methods

## Introduction

Automated hiring systems powered by machine learning algorithms now screen millions of job applications annually, promising efficiency and objectivity in candidate evaluation. However, these systems learn from historical hiring data—data that reflects decades of documented racial discrimination in employment. A common industry response to fairness concerns is to simply remove protected attributes like race and gender from model inputs, with companies claiming this makes their algorithms unbiased. But does this strategy actually work?

This research investigates a critical question: **Does excluding demographic features from machine learning models mitigate bias when those models are trained on historically biased data, or do proxy variables—such as names that signal race—enable the reconstruction of discrimination?**

The significance of this question extends beyond academic interest. The Equal Employment Opportunity Commission's 80% rule establishes legal thresholds for discriminatory impact, and companies deploying biased hiring algorithms face both legal liability and reputational damage. Moreover, Quillian, Pager, Hexel, and Midtbøen's (2017) meta-analysis of 28 field experiments found that racial discrimination in hiring callbacks has remained constant since 1989, with White applicants receiving 36% more callbacks than equally qualified Black applicants. If automated systems simply learn and perpetuate this discrimination, they offer no improvement over human decision-making—and may even provide a veneer of objectivity that makes discrimination harder to detect and challenge.

Our thesis is this: **Removing explicit demographic variables from model inputs is insufficient to eliminate bias learned from discriminatory training data, because proxy variables such as names leak racial information that allows models to reconstruct discriminatory patterns.** We test this thesis by training machine learning models on callback labels calibrated to documented discrimination rates, then comparing bias levels across three experimental regimes: qualifications only, qualifications plus names, and qualifications plus explicit race. We adopt an individual fairness criterion—candidates with identical qualifications should receive identical callback probabilities regardless of race—and measure outcomes using disparate impact ratios aligned with EEOC legal standards.

The findings of this research have direct implications for organizations deploying automated hiring tools and for policymakers evaluating fairness claims in algorithmic decision-making.

## Data Description

This research draws upon two primary data sources. The first is the Kaggle Resume Dataset compiled by Sneha Anbhawal, containing 2,484 resumes spanning 24 professional categories. Each resume includes full text content, an HTML-formatted version, and a category label indicating professional field. This dataset provides authentic resume content with realistic variation in how candidates present qualifications, experience, and skills—essential for testing whether models can extract meaningful qualification signals from unstructured text.

The second data source consists of 500 synthetically generated job descriptions, created using Python's Faker library. While public job posting datasets exist, quality issues including excessive duplication and inconsistent formatting necessitated generating our own. Synthetic generation offered full control over job requirements, ensuring clean alignment with resume categories. Each posting includes job title, company name, required skills (4-7 per posting), education requirements, years of experience, and salary range. A random seed of 42 ensures complete reproducibility.

These data sources help answer our research question by providing realistic resume content from which qualification features can be extracted, while synthetic job descriptions establish field-specific skill benchmarks. The combination allows us to construct a controlled experimental environment where candidate qualifications are measurable and comparable.

The raw Kaggle dataset required substantial processing. From 2,484 resumes across 24 categories, we retained only those matching five target professional fields: Education (102 resumes), Healthcare (115), Technology (120), Finance (118), and Marketing (96)—yielding 551 resumes. These fields provide diversity in job types and qualification profiles while ensuring sufficient sample size. Quillian et al. (2017) found that racial discrimination persists across industries without significant variation, supporting generalizability regardless of field composition.

Following Bertrand and Mullainathan's (2004) landmark methodology, we created matched pairs by duplicating each resume twice—once with a distinctively White name (e.g., Greg Baker, Emily Walsh) and once with a distinctively Black name (e.g., Jamal Washington, Lakisha Washington). This yielded 1,102 matched records where qualification features remain identical; only names differ. Names were drawn from pools of 24 per racial group, selected based on Social Security Administration and Census Bureau data indicating strong racial associations.

The data pipeline required merging across processing stages. Feature extraction produced resumes_processed.csv containing qualification features and one-hot encoded names (48 binary columns). Callback label generation produced resumes_callback_labels.csv with Quillian-calibrated outcomes. These files merge on both resume ID and race indicator—necessary because each ID appears twice (White and Black variants), and callback labels differ by race due to embedded discrimination. Merging on ID alone would misalign labels; merging on both columns ensures correct correspondence.

During train-test splitting, we partition by original resume ID rather than individual rows, ensuring both racial variants of each resume appear together in either training or test sets. This prevents data leakage that would invalidate matched-pair analysis.

## Method

The methodological foundation rests on calibrating synthetic callback labels to empirically documented discrimination rates, then testing whether feature removal mitigates learned bias. This approach was inspired directly by two foundational studies in discrimination research.

Bertrand and Mullainathan's (2004) field experiment "Are Emily and Greg More Employable Than Lakisha and Jamal?" established the matched-pair audit methodology we employ. By sending identical resumes with racially distinctive names to real employers, they demonstrated that names alone trigger discriminatory responses—White names received 50% more callbacks than Black names. Our experimental design replicates this logic computationally: identical qualification features, different names, measuring outcome disparities.

Quillian, Pager, Hexel, and Midtbøen's (2017) meta-analysis provides our calibration target. Synthesizing 28 field experiments spanning 1989-2015 with approximately 55,000 applications, they found White applicants receive 36% more callbacks than equally qualified Black applicants (ratio = 1.36, 95% CI: 1.25-1.47). Rather than inventing arbitrary bias parameters, we calibrate to this empirically-grounded estimate.

**Why this approach over alternatives?** We considered three options:

1. **Use real hiring outcome data**: Not feasible—such data is proprietary and rarely includes demographic information due to legal sensitivity.

2. **Generate bias-free labels, test if models introduce bias**: This produces trivial null results. If training labels contain no demographic signal, models cannot learn discrimination that doesn't exist.

3. **Generate biased labels calibrated to documented discrimination, test if feature removal reduces bias**: This mirrors real-world conditions where historical hiring data reflects past discrimination, and tests the actual debiasing strategy companies claim to use.

We selected option 3 because it addresses the practically relevant question: given that training data is biased, what can be done?

**The mathematical implementation** proceeds as follows. First, we compute qualification scores by standardizing features within each field (z-scores) and computing weighted sums: 40% years of experience, 35% skills count, 25% education level. Second, we convert scores to callback probabilities using the sigmoid function: P(callback) = sigmoid(α + β × qualification_score), tuning α for approximately 10% overall callback rate. Third, we add racial bias: P(callback) = sigmoid(α + β × qualification_score + γ × is_black), where γ is a negative penalty for Black applicants. Fourth, we calibrate γ using Brent's root-finding algorithm until mean(P_white) / mean(P_black) ≈ 1.36. Finally, we sample binary labels from these probabilities using Bernoulli sampling, introducing realistic stochastic variation.

**Three experimental regimes** test our thesis:

- **Regime A (Qualifications Only)**: Skills, experience, education—no demographic information. Tests the "just remove race" debiasing claim.

- **Regime B (Qualifications + Names)**: Adds one-hot encoded names (48 features). Tests whether names serve as proxies enabling discrimination without explicit race.

- **Regime C (Qualifications + Explicit Race)**: Adds binary race indicator. Establishes upper bound on possible discrimination.

**What this approach reveals that others cannot**: By controlling bias magnitude precisely (1.36 ratio) and systematically varying available features, we isolate the causal effect of feature availability on bias transmission. Observational studies of real hiring systems cannot disentangle whether observed disparities stem from qualification differences, biased training data, model architecture, or feature selection. Our experimental design holds all factors constant except feature availability, enabling clean causal inference about the effectiveness of demographic feature removal as a debiasing strategy.

**Analysis tools**: The pipeline uses Python with pandas for data manipulation, NumPy for numerical computation, scikit-learn for machine learning (Logistic Regression and Random Forest classifiers), SciPy for statistical functions and optimization, and Matplotlib/Seaborn for visualization. Analysis is conducted in Quarto documents enabling reproducible literate programming.

**Bias measurement** employs multiple metrics: callback rate disparities by race, disparate impact ratios (with 0.80 threshold per EEOC guidelines), White-to-Black callback ratios (compared against 1.36 baseline), Cohen's d effect sizes, and chi-square tests for statistical significance. The key comparison is between regimes—whether Regime A shows less bias than Regime C, and whether Regime B falls between them—rather than absolute numbers.

## Visualizations

The visualization strategy serves both exploratory and communicative purposes. During exploratory analysis, distribution plots reveal qualification feature variation across fields and demographic groups, confirming feature extraction produces reasonable distributions without severe outliers.

Bias calibration visualizations demonstrate successful implementation of Quillian-based discrimination. Scatter plots of callback probability versus qualification score, colored by race, reveal parallel but vertically offset curves—White and Black applicants with identical qualifications receive systematically different callback probabilities.

Model comparison visualizations form the core output. Grouped bar charts display White-to-Black callback ratios across regimes, with reference lines marking training label ratio (1.36) and parity (1.00). These immediately reveal whether removing demographics reduces bias, whether names enable proxy discrimination, and how explicit race access compares. Similar charts present disparate impact ratios with 80% threshold marked. Cohen's d plots quantify effect size magnitude with conventional benchmarks (0.2 small, 0.5 medium, 0.8 large).

Summary visualizations synthesize findings across regimes and models, combining callback rates, disparate impact ratios, effect sizes, and bias reduction percentages into unified displays. All figures maintain consistent color coding—steel blue for White, coral for Black—ensuring immediate interpretability.

## References

Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg more employable than Lakisha and Jamal? A field experiment on labor market discrimination. *American Economic Review*, 94(4), 991-1013.

Quillian, L., Pager, D., Hexel, O., & Midtbøen, A. H. (2017). Meta-analysis of field experiments shows no change in racial discrimination in hiring over time. *Proceedings of the National Academy of Sciences*, 114(41), 10870-10875.
