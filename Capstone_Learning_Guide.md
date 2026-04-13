# Capstone Project: Statistical Methods Learning Guide

**Purpose:** This document breaks down every statistical technique used in Phases 4-6 of the Algorithmic Bias in Resume Screening project. Use this with Claude to ask follow-up questions and deepen your understanding.

**How to use:** Read each section, then ask Claude questions like:
- "Explain [concept] in simpler terms"
- "Give me a real-world analogy for [technique]"
- "What would happen if we didn't do [step]?"
- "Quiz me on [section]"

---

# PART 1: FOUNDATIONS (What You Need to Know First)

## 1.1 The Sigmoid Function

### What is it?
The sigmoid function converts any number into a probability between 0 and 1.

### The Formula
```
sigmoid(x) = 1 / (1 + e^(-x))
```

Where `e` is Euler's number (approximately 2.718).

### Why do we need it?
- Our qualification scores can be any number (negative, zero, positive)
- Probabilities must be between 0 and 1
- Sigmoid "squashes" any input into that range

### Visual intuition
```
Input x:    -∞ ........... 0 ........... +∞
Output:      0 ........... 0.5 .......... 1

- Very negative x → output close to 0
- x = 0 → output = 0.5
- Very positive x → output close to 1
```

### In our project
We used sigmoid to convert a "callback score" into a "callback probability":
```
callback_score = α + β×qualifications + γ×is_black
callback_probability = sigmoid(callback_score)
```

**Ask Claude:** "Can you show me what happens to sigmoid(x) for x = -3, -1, 0, 1, 3?"

---

## 1.2 Logistic Regression Basics

### What is it?
A method to predict binary outcomes (yes/no, 1/0) using input features.

### How it works (simplified)
1. Take input features (skills, experience, education)
2. Multiply each by a "weight" (coefficient)
3. Add them up with an intercept
4. Pass through sigmoid to get probability
5. If probability > 0.5, predict "yes" (callback)

### The math
```
z = β₀ + β₁×feature₁ + β₂×feature₂ + ...
P(callback) = sigmoid(z)
```

Where:
- β₀ = intercept (baseline)
- β₁, β₂, ... = coefficients (weights for each feature)

### Why we used it
- Interpretable: coefficients tell us feature importance
- Probabilistic: gives confidence levels, not just yes/no
- Well-understood: standard in industry and academia

**Ask Claude:** "If the coefficient for years_experience is 0.5, what does that mean in plain English?"

---

## 1.3 What is a "Matched Pair" Design?

### The Problem
If we compare callbacks for White vs Black candidates, how do we know the difference is due to race and not qualifications?

### The Solution: Matched Pairs
Create two versions of the SAME resume:
- Version A: "Greg Baker" (White-sounding name)
- Version B: "Jamal Washington" (Black-sounding name)

Everything else is IDENTICAL:
- Same skills
- Same experience
- Same education
- Same field

### Why this is powerful
Any difference in callback rates CANNOT be explained by qualifications — they're identical. The only difference is the name (racial signal).

### In our project
- 551 original resumes
- Each duplicated into 2 versions (White name, Black name)
- Total: 1,102 records
- 24 different names per racial group (not just one)

**Ask Claude:** "Why did we use 24 names per race instead of just one name per race?"

---

# PART 2: PHASE 4 — LABEL GENERATION

## 2.1 The Core Problem

### What we needed
Training labels (callback = yes/no) for each resume.

### The challenge
We don't have real hiring data. We need to CREATE labels that reflect real-world discrimination.

### Our approach
Generate synthetic labels that embed the discrimination documented in academic research.

---

## 2.2 Calibration to Quillian et al. (2017)

### What is Quillian et al.?
A meta-analysis (study of studies) that combined 28 field experiments on hiring discrimination from 1989-2015, covering ~55,000 job applications.

### Key finding
White applicants receive **36% more callbacks** than equally qualified Black applicants.

Mathematically:
```
Callback Rate (White) / Callback Rate (Black) = 1.36
```

### Why we used this
- Empirically grounded (not made up)
- Large sample size (55,000 applications)
- Peer-reviewed (published in PNAS)
- Provides specific, measurable target

**Ask Claude:** "What does a ratio of 1.36 mean in practical terms? If 100 Black candidates get callbacks, how many White candidates would?"

---

## 2.3 The Label Generation Formula

### Step 1: Compute Qualification Score
```
qual_score = 0.40×years_experience_std + 0.35×skills_count_std + 0.25×education_level_std
```

Where `_std` means standardized (converted to z-scores within each field).

### Why standardize?
- Different fields have different norms
- A "senior" in tech might have 5 years experience
- A "senior" in healthcare might have 15 years
- Standardizing makes them comparable

### Step 2: Convert to Callback Probability
```
callback_score = α + β×qual_score + γ×is_black
callback_prob = sigmoid(callback_score)
```

Where:
- α (alpha) = intercept, controls overall callback rate
- β (beta) = 0.8, how much qualifications matter
- γ (gamma) = NEGATIVE number, penalty for Black applicants
- is_black = 1 if Black, 0 if White

### Step 3: Sample Binary Labels
```python
callback = np.random.binomial(1, callback_prob)
```

This is **Bernoulli sampling** — flip a weighted coin for each resume.

**Ask Claude:** "Explain Bernoulli sampling like I'm flipping a coin. How is it different from a fair coin flip?"

---

## 2.4 Calibration: Finding the Right γ (Gamma)

### The goal
Find γ such that:
```
mean(callback_prob for White) / mean(callback_prob for Black) ≈ 1.36
```

### The method: Brent's Algorithm
A mathematical technique to find where a function equals zero.

We defined:
```
objective(γ) = actual_ratio - target_ratio = actual_ratio - 1.36
```

Brent's algorithm finds γ where objective(γ) = 0.

### Result
```
γ = -0.31
```

This means: Being Black reduces your callback score by 0.31 points (before sigmoid transformation).

### Why not just set γ = -0.36?
The relationship isn't linear. Due to the sigmoid function, a specific γ value doesn't translate directly to a specific ratio. We need numerical optimization.

**Ask Claude:** "Why does the sigmoid function make the relationship between γ and the ratio non-linear?"

---

## 2.5 Why Bernoulli Sampling?

### What is it?
For each resume, we have a probability (e.g., 0.15 = 15% chance of callback). Bernoulli sampling randomly assigns 1 (callback) or 0 (no callback) based on that probability.

```python
# If callback_prob = 0.15:
# - 15% chance of getting callback = 1
# - 85% chance of getting callback = 0
callback = np.random.binomial(1, 0.15)
```

### Why not just threshold?
We could say "if prob > 0.5, callback = 1" — but this is unrealistic.

Real hiring has randomness:
- Same candidate might get callback from Company A but not Company B
- Hiring manager mood, timing, other applicants all add noise

Bernoulli sampling captures this stochasticity.

### The trade-off
- **Target ratio:** 1.36 (what we calibrated the probabilities for)
- **Achieved ratio:** 1.53 (what we got after random sampling)

This gap is expected — small samples have variance.

**Ask Claude:** "If I flip a fair coin 10 times, I expect 5 heads. But I might get 4 or 6. How does this relate to our achieved ratio of 1.53 vs target of 1.36?"

---

# PART 3: PHASE 5 — MODEL TRAINING

## 3.1 The Three Regimes

### What is a "regime"?
A specific configuration of features given to the model.

### Regime A: Qualifications Only
```
Features: [skills_count, years_experience, education_level]
Total: 3 features
```
**Purpose:** Test the "just remove demographics" approach

### Regime B: Qualifications + Names
```
Features: [skills_count, years_experience, education_level, name_Greg_Baker, name_Emily_Walsh, ..., name_Jamal_Washington, ...]
Total: 3 + 48 = 51 features
```
**Purpose:** Test if names act as racial proxies

### Regime C: Qualifications + Explicit Race
```
Features: [skills_count, years_experience, education_level, race_encoded]
Total: 4 features
```
**Purpose:** Upper bound on possible discrimination

### The key insight
All three regimes are trained on THE SAME callback labels with THE SAME train/test split. The only difference is which features each model can see.

**Ask Claude:** "Why is it important that all three regimes use the same training labels and the same train/test split?"

---

## 3.2 One-Hot Encoding for Names

### The problem
Names are text ("Greg Baker", "Lakisha Washington"). Models need numbers.

### The solution: One-hot encoding
Create a binary column for each unique name.

```
Original:
name = "Greg Baker"

One-hot encoded:
name_Greg_Baker = 1
name_Emily_Walsh = 0
name_Jamal_Washington = 0
... (all other names = 0)
```

### In our project
- 24 White names + 24 Black names = 48 name columns
- Each resume has exactly one "1" across these 48 columns

### Why this matters for bias
If the model learns that `name_Jamal_Washington` correlates with lower callbacks, it has effectively learned racial discrimination — even without seeing an explicit "race" column.

**Ask Claude:** "If I have 48 name columns and they're one-hot encoded, how many of them equal 1 for any single resume?"

---

## 3.3 Train/Test Split by Resume ID

### The standard approach (wrong for us)
Randomly split all 1,102 rows into training and test sets.

### The problem
If "Greg Baker" (White version of resume #42) is in training, but "Jamal Washington" (Black version of resume #42) is in test, we have DATA LEAKAGE.

The model has "seen" resume #42's qualifications during training, giving it an unfair advantage on the test set.

### Our approach: Split by ID
```python
# Get unique resume IDs
unique_ids = df['ID'].unique()  # 551 IDs

# Split IDs, not rows
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2)

# Both racial versions of each resume stay together
train_df = df[df['ID'].isin(train_ids)]  # ~882 rows
test_df = df[df['ID'].isin(test_ids)]    # ~220 rows
```

### Result
- Training: ~440 resumes × 2 races = ~880 rows
- Test: ~110 resumes × 2 races = ~220 rows
- No overlap in resume content between train and test

**Ask Claude:** "What is data leakage and why would it make our bias analysis invalid?"

---

## 3.4 StandardScaler: Feature Normalization

### The problem
Features have different scales:
- skills_count: 3-15
- years_experience: 0-40
- education_level: 1-5

Models can be sensitive to scale — larger numbers might dominate.

### The solution: StandardScaler
Convert each feature to have mean=0 and standard deviation=1.

```
z = (x - mean) / std
```

### Example
```
years_experience: [5, 10, 15, 20, 25]
mean = 15, std = 7.07

Standardized: [-1.41, -0.71, 0, 0.71, 1.41]
```

### Important
- Fit scaler on training data only
- Apply (transform) to both training and test data
- This prevents test data from influencing the scaling

**Ask Claude:** "Why do we fit the scaler only on training data and not on the full dataset?"

---

## 3.5 Model Training Results

### Performance metrics (not our focus)
```
             Accuracy    AUC
Regime A:    0.85       0.82
Regime B:    0.87       0.85
Regime C:    0.89       0.88
```

### The concerning pattern
Regime C (with explicit race) has the BEST accuracy. Why?

Because the training labels ARE biased by race. A model that uses race can better predict the biased labels — but this is exactly what we DON'T want in a fair system.

**Ask Claude:** "If a model has higher accuracy, does that mean it's better? What's the trade-off with fairness?"

---

# PART 4: PHASE 6 — BIAS TESTING

## 4.1 Callback Rate Analysis

### What we measured
For each regime, calculate:
```
White callback rate = (# White candidates predicted callback) / (# White candidates)
Black callback rate = (# Black candidates predicted callback) / (# Black candidates)
```

### Results
```
                    White Rate    Black Rate    W:B Ratio
Training Labels:    15.3%         10.0%         1.53
Regime A:           12.5%         12.5%         1.00
Regime B:           15.8%         10.0%         1.58
Regime C:           22.0%          7.8%         2.83
```

### Interpretation
- **Regime A:** Parity achieved — removing demographics WORKS
- **Regime B:** Bias reconstructed through names — proxy discrimination
- **Regime C:** Bias amplified beyond training data — direct discrimination

**Ask Claude:** "Why did Regime C amplify bias beyond what was in the training labels (2.83 vs 1.53)?"

---

## 4.2 Disparate Impact & The 80% Rule

### What is disparate impact?
A legal concept measuring whether a selection process disproportionately excludes a protected group.

### The 80% Rule (4/5ths Rule)
From EEOC (Equal Employment Opportunity Commission) guidelines:

```
Disparate Impact = (Selection Rate of Protected Group) / (Selection Rate of Majority Group)

If Disparate Impact < 0.80, adverse impact exists → potential discrimination
```

### In our context
```
Disparate Impact = Black Callback Rate / White Callback Rate = 1 / (W:B Ratio)
```

### Results
```
                W:B Ratio    Disparate Impact    Passes 80% Rule?
Regime A:       1.00         1.00                ✓ YES
Regime B:       1.58         0.63                ✗ NO
Regime C:       2.83         0.35                ✗ NO
```

### Legal implication
Regimes B and C would likely fail a legal discrimination audit.

**Ask Claude:** "If a company uses Regime B in production, what legal risks might they face under Title VII?"

---

## 4.3 The Paired t-Test

### What is a t-test?
A statistical test to determine if two groups have different means.

### Why "paired"?
We have matched pairs — the same resume with two racial versions. This is more powerful than comparing random groups.

### What we tested
For each resume pair:
```
diff = P(callback | White version) - P(callback | Black version)
```

Then test: Is the average diff significantly different from 0?

### The math
```
t = (mean of differences) / (standard error of differences)
t = mean(diff) / (std(diff) / √n)
```

### Hypotheses
- H₀ (null): mean(diff) = 0 (no racial difference)
- H₁ (alternative): mean(diff) ≠ 0 (racial difference exists)

### Results
```
Regime A: mean_diff = 0.001, t = 0.3, p = 0.76 → NOT significant
Regime B: mean_diff = 0.058, t = 11.3, p < 0.001 → SIGNIFICANT
Regime C: mean_diff = 0.142, t = 24.7, p < 0.001 → SIGNIFICANT
```

### Interpretation
- **p < 0.05:** Reject null hypothesis — difference is statistically significant
- **p > 0.05:** Cannot reject null — no evidence of difference

Regime A shows no significant racial difference; Regimes B and C show highly significant differences.

**Ask Claude:** "What does a p-value of 0.001 actually mean? Explain like I'm explaining to a non-technical friend."

---

## 4.4 Cohen's d Effect Size

### Why we need it
The p-value tells us IF a difference exists. Cohen's d tells us HOW BIG the difference is.

With large samples, even tiny differences can be "statistically significant" — but they might not matter practically.

### The formula
```
d = (Mean₁ - Mean₂) / SD_pooled
```

For matched pairs:
```
d = mean(diff) / std(diff)
```

### Interpretation guidelines (Cohen, 1988)
```
|d| < 0.2:  Negligible effect
|d| = 0.2:  Small effect
|d| = 0.5:  Medium effect
|d| = 0.8:  Large effect
```

### Results
```
Regime A: d ≈ 0.02 (negligible)
Regime B: d ≈ 0.48 (medium)
Regime C: d ≈ 0.95 (large)
```

### Interpretation
- Regime A: Essentially no bias (negligible effect)
- Regime B: Meaningful bias (medium effect) — real-world impact
- Regime C: Severe bias (large effect) — substantial discrimination

**Ask Claude:** "If Cohen's d = 0.5, what does that mean in everyday terms? How would you explain this to a hiring manager?"

---

## 4.5 Chi-Square Test for Independence

### What it tests
Are two categorical variables independent (unrelated) or associated?

### In our context
Variables:
- Race (White or Black)
- Predicted callback (Yes or No)

### The contingency table
```
              Callback=No    Callback=Yes    Total
White         464            87              551
Black         496            55              551
Total         960            142             1102
```

### The question
Is the distribution of callbacks different across races, or could this happen by chance?

### How it works
1. Calculate expected counts (if race and callback were independent)
2. Compare observed vs expected
3. Larger differences → larger chi-square statistic → smaller p-value

### Expected counts formula
```
Expected = (Row Total × Column Total) / Grand Total
```

### Results
```
Regime A: χ² = 0.12, p = 0.73 → NOT significant (independent)
Regime B: χ² = 15.8, p < 0.001 → SIGNIFICANT (associated)
Regime C: χ² = 47.3, p < 0.001 → SIGNIFICANT (strongly associated)
```

### Interpretation
In Regime A, race and callback are statistically independent — the model isn't discriminating.
In Regimes B and C, they're significantly associated — discrimination detected.

**Ask Claude:** "What's the difference between the chi-square test and the paired t-test? When would you use each?"

---

## 4.6 Coefficient Analysis (Logistic Regression)

### What are coefficients?
In logistic regression, each feature has a coefficient (weight) showing its impact on the outcome.

```
log_odds = β₀ + β₁×feature₁ + β₂×feature₂ + ...
```

### Interpretation
- Positive coefficient: Feature increases callback probability
- Negative coefficient: Feature decreases callback probability
- Magnitude: Larger |coefficient| = stronger effect

### Regime A coefficients
```
Feature             Coefficient
skills_count        +0.42
years_experience    +0.38
education_level     +0.25
```
All positive — better qualifications → higher callback probability. No demographic info.

### Regime B coefficients (names)
```
Feature                 Coefficient
name_Greg_Baker         +0.31
name_Emily_Walsh        +0.28
...
name_Jamal_Washington   -0.29
name_Lakisha_Washington -0.33
```

**Pattern:** White names have positive coefficients; Black names have negative coefficients.

The model learned to favor White-sounding names — PROXY DISCRIMINATION.

### Regime C coefficient (explicit race)
```
Feature        Coefficient
race_encoded   -0.89
```

Massive negative coefficient for Black (race_encoded=1). Direct discrimination.

**Ask Claude:** "If name_Jamal_Washington has a coefficient of -0.29, how does that translate to callback probability? Walk me through the math."

---

## 4.7 Model Disagreement Analysis

### What is it?
Identify resumes where different regimes give different predictions.

### Why it matters
Shows exactly which candidates are affected by proxy discrimination.

### Analysis
```
Regime A predicts callback, Regime B predicts no callback:
  → These candidates are HURT by name-based discrimination
  → Breakdown: 68% are Black candidates

Regime A predicts no callback, Regime B predicts callback:
  → These candidates BENEFIT from name-based discrimination
  → Breakdown: 71% are White candidates
```

### Interpretation
Names systematically disadvantage Black candidates and advantage White candidates — even when qualifications alone would predict different outcomes.

**Ask Claude:** "Give me a concrete example of one resume where the model disagreed. What happened?"

---

# PART 5: PUTTING IT ALL TOGETHER

## 5.1 The Complete Argument

### Starting point
- Companies claim "we removed race from the model, so it's fair"
- But models are trained on historical hiring data that contains bias

### Our experiment
1. Created realistic biased labels (W:B = 1.53, calibrated to Quillian)
2. Trained models under three conditions (A, B, C)
3. Measured bias in predictions

### Key findings

**Finding 1: Removing demographics CAN work (Regime A)**
When models only see qualifications, bias is eliminated.
- W:B ratio drops from 1.53 → 1.00
- Passes 80% rule
- No significant racial differences

**Finding 2: Names reconstruct bias (Regime B)**
Adding names brings discrimination back — and amplifies it.
- W:B ratio goes from 1.53 → 1.58
- Fails 80% rule
- Names cluster by race in coefficient analysis

**Finding 3: Explicit race maximizes bias (Regime C)**
Giving the model direct access to race creates severe discrimination.
- W:B ratio goes from 1.53 → 2.83
- Model amplifies training bias

### Conclusion
Simply "removing race" is necessary but not sufficient. Proxy variables like names can reconstruct discrimination. True fairness requires removing ALL features that correlate with protected attributes.

---

## 5.2 Limitations & Honest Assessment

### What we did well
- Empirically grounded calibration (Quillian et al.)
- Matched-pair design (controls for qualifications)
- Multiple statistical tests (triangulation)
- Clear experimental regimes

### Limitations
1. **Synthetic data:** Real hiring is more complex
2. **Binary race:** Real-world has many racial/ethnic groups
3. **Limited features:** Real resumes have many more signals
4. **Single discrimination mechanism:** Real bias has multiple sources

### What this DOESN'T prove
- That all hiring algorithms are biased
- That names are the only proxy variable
- That removing names solves the problem

### What this DOES prove
- Names CAN act as proxies for race
- Models CAN learn discrimination from biased data
- Removing explicit demographics is NOT a complete solution

**Ask Claude:** "What would you do differently if you could redo this project? What additional analyses would strengthen the conclusions?"

---

## 5.3 Glossary of Terms

| Term | Definition |
|------|------------|
| **Bernoulli sampling** | Random sampling where each trial has two outcomes (success/failure) with fixed probability |
| **Brent's method** | Numerical algorithm to find roots of a function |
| **Calibration** | Adjusting parameters to achieve a target outcome |
| **Chi-square test** | Tests whether two categorical variables are independent |
| **Coefficient** | Weight assigned to a feature in a regression model |
| **Cohen's d** | Effect size measure; standardized difference between means |
| **Disparate impact** | When a practice disproportionately affects a protected group |
| **Matched pairs** | Experimental design where subjects are paired based on similar characteristics |
| **One-hot encoding** | Converting categorical variables to binary columns |
| **p-value** | Probability of observing results as extreme as the data if null hypothesis is true |
| **Proxy variable** | A variable that correlates with a protected attribute |
| **Sigmoid function** | Function that maps any input to a value between 0 and 1 |
| **StandardScaler** | Transforms features to have mean=0 and standard deviation=1 |
| **t-test** | Tests whether means of two groups are significantly different |

---

## 5.4 Questions to Test Your Understanding

1. Why did we use a ratio of 1.36 as our calibration target?

2. What is the mathematical relationship between W:B ratio and disparate impact?

3. Why did we split by resume ID rather than randomly splitting rows?

4. If Regime B has 51 features and Regime A has 3 features, why might Regime B have higher accuracy?

5. What does a Cohen's d of 0.5 tell us that a p-value doesn't?

6. In Regime B, why do White-sounding names have positive coefficients?

7. What would happen if we only had ONE name per race instead of 24?

8. Why is Bernoulli sampling more realistic than using a threshold?

9. What's the difference between statistical significance and practical significance?

10. Could a model pass the 80% rule but still be unfair? How?

**Use Claude to check your answers and get explanations!**

---

# END OF GUIDE

**Next steps:**
1. Read through each section carefully
2. Ask Claude to clarify anything unclear
3. Try to explain each concept in your own words
4. Use the quiz questions to test yourself
5. Practice explaining the project to someone else

Good luck with your presentation!
