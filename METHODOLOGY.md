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

**Methodology:** Following Bertrand & Mullainathan (2004), names were selected that research has shown to correlate with specific racial and gender groups based on SSA baby name data and Census Bureau surname statistics.

**Four demographic groups:**

| Group | Example Names |
|-------|---------------|
| White Male | Brad Anderson, Greg Baker, Todd Miller, Geoffrey Thomas |
| White Female | Emily Walsh, Anne Peterson, Meredith Collins, Sarah Mitchell |
| Black Male | Jamal Washington, Leroy Harris, Tyrone Jackson, Darnell Robinson |
| Black Female | Lakisha Washington, Tanisha Harris, Aisha Jackson, Ebony Robinson |

Each resume was assigned one name randomly from the full pool (12 names per group, 48 total).

---

### Synthetic Hire/No-Hire Label Generation

**Key design decision:** Labels are generated based on **objective qualifications only** — no demographic information influences the hire decision. This allows us to test whether ML models introduce bias that wasn't present in the training labels.

**Qualification Score Calculation:**

```
qualification_score = exp_score + skill_score + edu_score + noise

Where:
- exp_score = min(years_experience / avg_field_requirement, 2.0) × 30 pts
- skill_score = min(skills_count / 10, 1.0) × 25 pts
- edu_score = (education_level / 5) × 15 pts
- noise = random normal(0, 5) to simulate real-world variation
```

**Why field-relative scoring matters:**

The `avg_field_requirement` is calculated by averaging `years_experience_required` across the 100 job descriptions for that field. This makes qualification scores **field-relative**:

| Field | Avg Experience Required | Why It Matters |
|-------|------------------------|----------------|
| Education | ~5.5 years | 10 years = exceptional |
| Healthcare | ~8 years | 10 years = above average |
| Technology | ~6.5 years | 10 years = senior |
| Finance | ~5.5 years | 10 years = exceptional |
| Marketing | ~7 years | 10 years = experienced |

Without field-relative scoring, fields with naturally higher experience levels would dominate the "hired" category. Using averages ensures ~40% hire rate per field.

**Hire threshold:** Top 40% of qualification scores within each field are labeled `hired = 1`.

---

### Matched Pair Creation

**Purpose:** Enable direct bias measurement by comparing model predictions on identical resumes with different demographic signals.

**Process:**
1. Take each of the 551 original resumes
2. Duplicate it 4 times
3. Assign each duplicate a name from a different demographic group
4. All qualification features remain identical

**Result:** 551 × 4 = 2,204 matched resume records

**Example matched set:**

| Original ID | Name | Demographics | Skills | Experience | Education | Qual Score |
|-------------|------|--------------|--------|------------|-----------|------------|
| 001 | Brad Anderson | white_male | 8 | 6 | 3 | 72.4 |
| 001 | Emily Walsh | white_female | 8 | 6 | 3 | 72.4 |
| 001 | Jamal Washington | black_male | 8 | 6 | 3 | 72.4 |
| 001 | Lakisha Washington | black_female | 8 | 6 | 3 | 72.4 |

If a model assigns different scores to these four versions, that difference is attributable solely to the demographic signal (name).

---

### Feature Set Definitions

Three feature sets for comparative modeling:

| Feature Set | Features Included | Purpose |
|-------------|-------------------|---------|
| **Baseline** | skills_count, years_experience, education_level | Tests model performance with objective qualifications only |
| **Expanded** | Baseline + name (one-hot encoded, 48 features) | Tests whether name as a feature introduces bias |
| **Problematic** | Baseline + race_encoded + gender_encoded | Tests explicit demographic features (intentionally biased) |

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

## Phase 4: Baseline Modeling

**Status:** Not started

---

## Phase 5: Bias Testing

**Status:** Not started

---

## Phase 6: Statistical Analysis

**Status:** Not started

---

## Phase 7: Visualization & Reporting

**Status:** Not started

---

## Key References

- Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg More Employable Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination. *American Economic Review*.
- SSA Baby Names Database: https://www.ssa.gov/oact/babynames/
- US Census Bureau Surname Data: https://www.census.gov/topics/population/genealogy/data/2010_surnames.html
