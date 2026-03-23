# Common Concerns and Responses

This document addresses potential critiques of the research methodology and provides responses to each concern.

---

## Concern 0: Assigning Hiring Outcomes Based on 1.36 Ratio

**The Concern:**
> "Not sure if you're going to bias a pre-existing hiring outcome or you're assigning hiring outcome based on the 1.36 ratio. If you're planning to pursue the latter option, that doesn't sound right to me, unless implemented very masterfully."

**Our Approach:**
We ARE assigning hiring outcomes based on the 1.36 ratio.

**Our Response:**
This is intentional and methodologically sound. We are simulating historical hiring data that reflects documented discrimination. Real hiring outcome data is proprietary and rarely includes demographic information due to legal sensitivity. By calibrating synthetic labels to Quillian et al.'s (2017) empirically-documented discrimination rate, we create a controlled experimental environment that mirrors real-world conditions where training data reflects past discrimination.

The key insight: We are not claiming to discover bias. We are testing whether common debiasing strategies (removing demographics) can UNDO bias that exists in training data. This is the practically relevant question companies face.

---

## Concern 1: "You're Measuring the Bias You're Introducing"

**The Concern:**
> "You're measuring the bias you're introducing. Unless you're planning a unique randomization using some algorithm, not sure if the conclusions based on cross-examining regimes A, B, C hold."

**Our Response:**
This concern misunderstands our research goal. We are NOT trying to "discover" the 1.36 ratio—we know it's there because we put it there.

What we ARE measuring:
- **Bias transfer**: How much of the 1.36 ratio in training labels appears in model predictions?
- **Regime differences**: Does Regime A show less bias than Regime C? Does Regime B fall between them?

The interesting finding is NOT the absolute numbers, but the DIFFERENCES between regimes:
- If Regime A ≈ 1.0 → Removing demographics works
- If Regime A ≈ 1.36 → Removing demographics doesn't work
- If Regime B > Regime A → Names leak racial information

This is analogous to testing a water filter: we deliberately contaminate water (inject bias), then test whether the filter (feature removal) cleans it.

---

## Concern 2: Duplication Distorts Reality

**The Concern:**
> "Consider assigning bias to CVs without duplication. Yes, the dataset will be smaller but I feel like doubling it will distort realities too much. Maybe somewhere in between."

**Our Response:**
Matched pairs are standard methodology in discrimination research, established by Bertrand and Mullainathan (2004) in their landmark study "Are Emily and Greg More Employable Than Lakisha and Jamal?"

Why duplication is necessary:
- It isolates the demographic effect from qualification differences
- Without matched pairs, we cannot distinguish whether outcome differences stem from qualifications or discrimination
- Real audit studies send identical resumes with different names—we replicate this computationally

The "distortion" concern is addressed by our experimental design:
- We never claim the dataset represents natural resume distribution
- We explicitly state this is a controlled experiment
- The matched-pair structure is the mechanism that enables causal inference

**Acknowledgment:** We note this as a methodological choice and cite Bertrand & Mullainathan as precedent.

---

## Concern 3: Sample Size Compared to Quillian

**The Concern:**
> "Compare the number of samples in the dataset that gave the 1.36 number. Not the most serious concern, but I'd have slight reservations about the difference in the scope of your samples."

**Our Response:**

| Study | Sample Size |
|-------|-------------|
| Quillian meta-analysis | ~55,000 applications across 28 studies |
| Our study | 551 resumes × 2 variants = 1,102 records |

**Why this difference is acceptable:**
- Different goals: Quillian measured real-world discrimination; we test debiasing strategies
- We use Quillian's FINDING (1.36 ratio) as input, not trying to replicate the measurement
- The methodology is sound at smaller scale for testing feature removal effects
- Statistical power is sufficient for detecting regime differences

**Acknowledgment:** We note sample size as a limitation and suggest larger-scale replication in future work.

---

## Concern 4: Fairness is Subjective

**The Concern:**
> "Fairness itself is subjective. Seems like you wanna go for individual fairness i.e. identical resumes should produce identical outcomes. But good to define it explicitly."

**Our Response:**
We explicitly define our fairness criterion:

> **Individual Fairness**: Candidates with identical qualifications should receive identical callback probabilities regardless of race.

We also measure outcomes using:
- **Disparate Impact Ratio**: Aligned with EEOC 80% rule (legal standard)
- **Cohen's d Effect Size**: Standardized measure of practical significance
- **White-to-Black Callback Ratio**: Direct comparison to Quillian's 1.36 baseline

This explicit definition appears in our Introduction section.

---

## Concern 5: What Are You Actually Testing?

**The Concern:**
> "Properly define what you're going for. Are you testing whether removing demographic information makes hiring fairer? Or are you proposing a new 'unbiased' ML recruitment assistant?"

**Our Answer:**
We are testing whether removing demographic information makes hiring fairer (Option 5.1).

We are NOT proposing a new system. We are evaluating a common industry claim:
> "Our hiring algorithm is fair because we removed race and gender from the inputs."

Our experiment tests whether this claim holds when training data contains historical discrimination.

---

## Concern 5.1: Will Conclusions Hold for Bigger Dataset?

**The Concern:**
> "A critique can be raised about whether the conclusions will hold for a bigger dataset. You have identified possibility of proxy variables, but not sure if ~500 CVs will be enough to produce any proxy variable. Name isn't the only possible proxy variable."

**Our Response:**

**On sample size:**
- 551 resumes × 2 variants = 1,102 records is sufficient for detecting regime differences
- We acknowledge this as a limitation and suggest larger-scale replication

**On names as proxies:**
- Names are strong, well-documented racial proxies (Bertrand & Mullainathan 2004)
- With 24 names per racial group, models have sufficient signal to learn associations
- We use one-hot encoding (48 features), giving models opportunity to detect patterns

**On other proxy variables:**
- We acknowledge names are not the only possible proxy
- Other potential proxies (HBCUs, professional associations) had insufficient representation in our dataset
- Future work could explore additional proxies
- Names provide clean experimental isolation consistent with established methodology

**Key point from reviewer:**
> "Difference between the regimes are more important than the individual numbers."

We agree completely. Our analysis focuses on regime DIFFERENCES, not absolute bias levels.

---

## Concern 5.2: Adversarial Debiasing

**The Concern:**
> "If you're answering the second question, lookup General adversarial networks and adversarial debiasing. I think they will be helpful."

**Our Response:**
We are answering the first question (testing debiasing), not the second (proposing a solution). Therefore, adversarial debiasing is outside our scope.

However, this is a valid direction for future work:
- If our findings show that feature removal is insufficient, adversarial debiasing could be a next step
- GANs and adversarial approaches actively remove demographic information from learned representations
- This would address the proxy variable problem we identify

**Acknowledgment:** We note adversarial debiasing as potential future work in our limitations section.

---

## Summary: How We Address Each Concern

| Concern | Resolution |
|---------|------------|
| 0. Assigning outcomes based on ratio | Justified as simulating historical bias |
| 1. Measuring what we introduce | Focus on regime DIFFERENCES, not absolute numbers |
| 2. Duplication distorts reality | Standard methodology (Bertrand & Mullainathan 2004) |
| 3. Sample size vs Quillian | Acknowledge as limitation; methodology sound at smaller scale |
| 4. Define fairness | Explicitly state individual fairness criterion |
| 5. What are we testing? | Testing debiasing strategy, not proposing new system |
| 5.1 Bigger dataset / other proxies | Acknowledge limitations; focus on regime differences |
| 5.2 Adversarial debiasing | Outside scope; note as future work |
