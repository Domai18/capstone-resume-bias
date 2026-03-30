# Dr. Pardue Critique Response

This document addresses feedback received on the research methodology.

---

## Critique 1: Difficulty Isolating Feature Effects from Training Differences

**The Concern:**
> "Right now, since three separate models are being created with different inputs, it's difficult to isolate whether the differences in bias are due to the features themselves or differences in how the models were trained."

**Our Response:**

This is controlled in our experimental design. All three regimes share:

| Controlled Variable | How It's Controlled |
|---------------------|---------------------|
| Training labels | Same Quillian-calibrated biased labels |
| Training/test split | Same 80/20 split by resume ID |
| Random seed | Fixed at 42 |
| Model algorithms | Same (Logistic Regression, Random Forest) |
| Hyperparameters | Default scikit-learn settings |

The **only** variable that changes across regimes is which features the model can access:
- Regime A: 3 features (qualifications only)
- Regime B: 51 features (qualifications + 48 name dummies)
- Regime C: 4 features (qualifications + race)

Therefore, any difference in bias across regimes **is** attributable to feature availability, not training differences.

**Action Taken:** Added explicit framing in Methods Essay emphasizing this controlled design.

---

## Critique 2: Frame as Simulation Study

**The Suggestion:**
> "I'd suggest that you consider explicitly framing the project as a simulation comparing how different training inputs lead to different learned biases."

**Our Response:**

Agreed. We now frame the study as:

> "A controlled simulation study comparing how different feature configurations affect bias transmission from training labels to model predictions."

This framing:
- Acknowledges the synthetic nature of the bias injection
- Emphasizes the controlled comparison across regimes
- Clarifies we're testing a debiasing strategy, not discovering natural bias

**Action Taken:** Updated Methods Essay introduction with simulation framing.

---

## Critique 3: Compare Logistic Regression Coefficients

**The Suggestion:**
> "If you're using logistic regression as a model, you might also compare how the equations shift across the three methods to get a better idea of how the models shift."

**Our Response:**

Excellent suggestion. Coefficient analysis reveals:

| Regime | What to Examine | Expected Finding |
|--------|-----------------|------------------|
| A (quals only) | Coefficients for skills, experience, education | No demographic signal |
| B (+ names) | Coefficients for name dummy variables | Black-associated names may have negative coefficients |
| C (+ race) | Coefficient for race_encoded | Direct measure of discrimination learned |

For Regime B specifically, we can:
- Calculate average coefficient for White-associated names vs Black-associated names
- If White names have higher average coefficients, the model learned to favor them

**Action Taken:** Added coefficient analysis section to Phase 6 notebook.

---

## Critique 4: Analyze Model Disagreements

**The Suggestion:**
> "If you use the same training/testing data for each model, you can look at resumes where the models disagree and try to find what's different about them. It would give you more insight into what's driving differences across models."

**Our Response:**

This is a valuable insight. We will:

1. **Identify disagreements:** Find resumes where Regime A and Regime B produce different predictions
2. **Analyze by race:** Check if disagreements disproportionately affect Black vs White applicants
3. **Categorize:**
   - Regime A = callback, Regime B = no callback → Names hurt this candidate
   - Regime A = no callback, Regime B = callback → Names helped this candidate

**Expected Findings:**

If names act as proxies for race:
- More Black applicants in "A=yes, B=no" (names hurt them)
- More White applicants in "A=no, B=yes" (names helped them)

This analysis provides concrete examples of how proxy variables alter individual outcomes.

**Action Taken:** Added disagreement analysis section to Phase 6 notebook.

---

## Summary of Changes Made

| Critique | Response | Action |
|----------|----------|--------|
| Isolating feature effects | Design already controls for this | Clarified in Methods Essay |
| Frame as simulation | Agreed | Updated Methods Essay introduction |
| Compare LR coefficients | Excellent idea | Added to Phase 6 notebook |
| Analyze disagreements | Valuable insight | Added to Phase 6 notebook |

---

## Updated Methods Essay Language

The introduction now includes:

> "We test this thesis through a controlled simulation study comparing how different feature configurations affect bias transmission from training labels to model predictions. Crucially, all three experimental regimes—qualifications only, qualifications plus names, and qualifications plus explicit race—are trained on identical callback labels using the same training/test split and random seed. The only variable that changes is which features each model can access. This design isolates the effect of feature availability on learned bias, eliminating confounds from training differences."
