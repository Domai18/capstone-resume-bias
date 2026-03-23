# Algorithmic Bias in Resume Screening: A Simple Explanation

## The Research Question

**Can companies eliminate hiring bias by simply removing race from their algorithms, or do proxy variables like names allow discrimination to persist?**

---

## The Core Concept

We simulate what happens in real-world automated hiring systems:

1. Companies have historical hiring data reflecting past discrimination
2. They train AI models on this biased data
3. They claim removing race from the model makes it fair
4. We test whether this is actually true

---

## Where Does Bias Come From?

Bias enters through **callback labels** (who gets a callback vs. who doesn't).

We use findings from Quillian et al. (2017), a meta-analysis of 28 hiring studies:

| Applicant Race | Callback Rate | What This Means |
|----------------|---------------|-----------------|
| White | ~10% | Baseline |
| Black | ~7.4% | 36% fewer callbacks for identical qualifications |

This 1.36 ratio (White callbacks / Black callbacks) is real, documented discrimination. We embed this into our synthetic callback labels to simulate biased historical hiring data.

---

## The Experimental Design

We train machine learning models on biased data, then test three scenarios:

### Regime A: "Remove All Demographics"
| What the model sees | What's hidden |
|---------------------|---------------|
| Skills, Experience, Education | Names, Race |

**Tests the claim:** "Our algorithm is fair because we don't use race."

### Regime B: "Names Are Visible"
| What the model sees | What's hidden |
|---------------------|---------------|
| Skills, Experience, Education, Names | Explicit Race |

**Tests the concern:** "Can names like 'Jamal' vs 'Greg' reveal race indirectly?"

### Regime C: "Race Is Visible"
| What the model sees | What's hidden |
|---------------------|---------------|
| Skills, Experience, Education, Race | Nothing |

**Establishes the ceiling:** Maximum possible bias when race is directly available.

---

## The Critical Insight

**All three regimes are trained on the same biased labels.**

The models aren't programmed to be racist. They learn patterns from data. The question is:

- Regime A: With no demographic info, can the model even learn the bias?
- Regime B: Do names provide a "backdoor" that leaks racial information?
- Regime C: How bad is it when race is directly available?

---

## The Pipeline

```
PHASE 1-2: DATA PREPARATION
     │
     │   • 551 resumes from Kaggle dataset
     │   • Create matched pairs: same resume, different names
     │   • Result: 1,102 records (551 × 2 racial variants)
     │
     ▼
PHASE 4: INJECT BIAS ← This simulates historical discrimination
     │
     │   • Generate callback labels using Quillian's 1.36 ratio
     │   • White names get higher callback probability
     │   • Black names get lower callback probability
     │
     ▼
PHASE 5: TRAIN MODELS
     │
     │   • Train under Regime A (qualifications only)
     │   • Train under Regime B (qualifications + names)
     │   • Train under Regime C (qualifications + race)
     │   • All use the SAME biased labels
     │
     ▼
PHASE 6: MEASURE BIAS
     │
     │   • Generate predictions for each regime
     │   • Compare callback rates: White vs Black
     │   • Calculate bias metrics
     │
     ▼
RESULTS: Which regime reproduced the most bias?
```

---

## Expected Findings

| Regime | Expected White:Black Ratio | Interpretation |
|--------|---------------------------|----------------|
| A | Close to 1.0 | Removing demographics may reduce bias |
| B | Between A and C | Names partially reconstruct bias |
| C | Close to 1.36 | Full bias reproduced |

---

## Why This Matters

Many companies claim their hiring AI is fair because they "removed protected attributes." Our research tests whether this is true or whether seemingly neutral variables (like names) can serve as proxies that reintroduce discrimination.

**Real-world implications:**
- If Regime A shows low bias → Removing demographics works
- If Regime B shows high bias → Companies need to audit proxy variables
- Legal standard: Disparate impact ratio below 0.80 triggers EEOC scrutiny

---

## The Math (Simplified)

### How We Create Biased Labels

```
Step 1: Calculate qualification score
        Score = 40% × Experience + 35% × Skills + 25% × Education

Step 2: Convert to callback probability
        P(callback) = sigmoid(baseline + score effect + race penalty)

        For White applicants: No penalty
        For Black applicants: Negative penalty term

Step 3: Calibrate the penalty
        We need to find the exact penalty that produces a 1.36 ratio.

        The computer tries different penalty values:
        • Try -0.5 → ratio = 1.20 (too low, need more penalty)
        • Try -0.8 → ratio = 1.45 (too high, need less penalty)
        • Try -0.65 → ratio = 1.36 ✓ (found it!)

        This is automated trial-and-error that takes milliseconds.

Step 4: Generate labels
        Each applicant gets callback = Yes/No based on their probability
```

The sigmoid function ensures probabilities stay between 0% and 100%.

---

## Key Terminology

| Term | Simple Definition |
|------|-------------------|
| Matched Pair | Same resume with two different names (one White, one Black) |
| Proxy Variable | A seemingly neutral variable that reveals protected information |
| Disparate Impact | When a policy disproportionately affects one group |
| Callback Ratio | White callback rate ÷ Black callback rate |
| Regime | A specific set of features the model is allowed to see |

---

## Summary

1. **We embed real-world discrimination** into synthetic callback labels (1.36 ratio)
2. **We train models** on this biased data under three conditions
3. **We measure** how much bias each condition reproduces
4. **We answer:** Does removing demographics actually work, or do names leak bias back in?

This directly addresses a common industry practice and has implications for fair AI policy.
