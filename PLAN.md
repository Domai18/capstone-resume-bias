# PLAN.md — Capstone Project Memory

**Last Updated:** March 27, 2026
**Project:** Algorithmic Bias in Resume Screening
**Author:** Derrick Omai

---

## A) General Work Plan

**Research Question:** Does excluding demographic features from ML models mitigate bias when trained on historically biased data, or do proxy variables (names) enable reconstruction of discrimination?

**Thesis:** Removing explicit demographics is insufficient; names leak racial information allowing bias reconstruction.

**Methodology:** Matched-pair testing (Bertrand & Mullainathan) with Quillian-calibrated bias (1.36 W:B ratio).

**Scope:** Race only (White vs Black). Gender removed per Quillian scope.

---

## B) Implementation Stages

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Collection (Kaggle resumes, synthetic jobs) | ✅ Complete |
| 2 | Preprocessing (race-only, 551×2=1,102 pairs) | ✅ Complete |
| 3 | EDA | ✅ Complete |
| 3b | Methodology Pivot Documentation | ✅ Complete |
| 4 | Label Generation (Quillian 1.36 calibration) | ✅ Complete |
| 5 | Model Training (3 regimes × 2 models) | ✅ Complete |
| 6 | Bias Testing (coefficients, disagreement analysis) | ✅ Complete |
| 7 | Final Analysis & Report | 🔄 In Progress |

---

## C) Checklist

### Completed ✅
- [x] Filter resumes to 5 fields (551 total)
- [x] Remove gender, keep race-only (Option B: 24 names/race)
- [x] Create matched pairs (1,102 records)
- [x] Generate Quillian-calibrated labels (achieved 1.53 ratio)
- [x] Train Logistic Regression & Random Forest under 3 regimes
- [x] Run bias testing with coefficient & disagreement analysis
- [x] Methods Essay with introduction, thesis, citations
- [x] Common Concerns document
- [x] Dr. Pardue Critique Response
- [x] Push phases 4-6 to GitHub

### Remaining ⏳
- [ ] Write Phase 7 final analysis notebook
- [ ] Create executive summary of findings
- [ ] Final report/presentation
- [ ] Address any remaining professor feedback

---

## D) Progress

**Overall: 85%**

```
[██████████████████░░] 85%
```

| Component | Progress |
|-----------|----------|
| Data Pipeline | 100% |
| Analysis Code | 100% |
| Documentation | 80% |
| Final Report | 0% |

---

## E) Key Results (For Reference)

| Regime | W:B Ratio | Disparate Impact | Finding |
|--------|-----------|------------------|---------|
| A (quals only) | 1.0 | 1.0 ✅ | No bias — removing demographics works |
| B (+ names) | 1.59 | 0.65 ❌ | Names reconstruct bias (proxy effect) |
| C (+ race) | 2.83 | 0.35 ❌ | Direct race amplifies bias |

**Thesis Supported:** Yes. Names act as proxies enabling discrimination.

---

## F) Next Actions

1. **Create Phase 7 notebook** — Final analysis with publication-ready findings
2. **Update Methods Essay** — Insert actual results
3. **Prepare presentation** — Key findings for professor
4. **Final commit** — All remaining files to GitHub

---

## G) Key Files

| File | Purpose |
|------|---------|
| `notebooks/04_label_generation.qmd` | Quillian calibration |
| `notebooks/05_model_training.qmd` | 3 regimes training |
| `notebooks/06_bias_testing.qmd` | Bias analysis |
| `Methods_Essay.md` | Formal methodology |
| `Common_Concerns.md` | Critique responses |
| `results/tables/bias_summary.csv` | Key results |

---

## H) Instructions for Claude

**At session start:** Read this PLAN.md first to understand project state.

**At session end:** Update this file with:
- New checklist items completed
- Updated progress percentage
- New next actions
- Any key decisions made

**Command:** "Claude, continue with PLAN.md"

---

## I) Session Log

| Date | Summary |
|------|---------|
| Mar 2 | Removed gender, updated to race-only |
| Mar 27 | Ran phases 4-6, got results, pushed to GitHub |
| Mar 27 | Created PLAN.md for persistent memory |
| Mar 30 | Fixed Phase 3 EDA to align with race-only methodology |
