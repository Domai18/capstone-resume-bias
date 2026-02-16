# Synthetic Job Descriptions Dataset Proposal

**Author:** Derrick Omai

**Project:** Algorithmic Bias in Automated Resume Screening

---

## Purpose

This document describes the synthetic job descriptions dataset created to complement the Kaggle Resume Dataset in our study of machine learning bias in hiring systems. Real-world job description datasets available on platforms like Kaggle suffered from significant quality issues—most notably, one widely-used dataset contained only 376 unique rows duplicated to inflate the count to 1.6 million entries. To ensure data integrity and alignment with our research design, we generated a controlled synthetic dataset using Python's Faker library.

## Generation Methodology

The dataset was generated programmatically with a fixed random seed (42) to ensure reproducibility. For each of our five career fields—Education, Healthcare, Technology, Finance, and Marketing—we created 100 job postings, yielding 500 total records. Each posting was constructed by randomly selecting from field-specific pools of job titles, required skills, education requirements, and experience levels. Company names and supplementary job description text were generated using the Faker library to create realistic-looking postings.

## Field-Specific Criteria

Each career field was configured with parameters reflecting real-world hiring standards:

- **Education:** Job titles include Elementary School Teacher and Special Education Teacher; skills emphasize classroom management, curriculum development, and student assessment; experience ranges from 1-10 years; education requires a Bachelor's or Master's in Education.

- **Healthcare:** Titles include Registered Nurse and ICU Nurse; skills cover patient care, medication administration, and EMR systems; experience ranges from 1-15 years; education requires BSN, ADN, or MSN credentials.

- **Technology:** Titles include Software Engineer and Full Stack Developer; skills include Python, Java, SQL, and cloud services; experience ranges from 1-12 years; education requires a Bachelor's or Master's in Computer Science.

- **Finance:** Titles include Financial Analyst and Budget Analyst; skills emphasize financial modeling, Excel, and forecasting; experience ranges from 1-10 years; education requires degrees in Finance, Accounting, or an MBA.

- **Marketing:** Titles include Marketing Manager and Brand Manager; skills cover campaign management, SEO, and analytics; experience ranges from 2-12 years; education requires degrees in Marketing, Communications, or Business.

## Role in the Research Design

The job descriptions dataset serves one critical function: establishing field-relative qualification standards. When calculating whether a resume qualifies for hiring, we compare each candidate's years of experience against the average required experience for their field. This field-relative approach ensures that a teacher with 8 years of experience is evaluated against teaching standards, not against software engineering norms. The result is balanced hire rates (~40%) across all five career fields, preventing any single field from dominating the "hired" category.

## Dataset Access

The generated dataset is stored within the project repository at:

`data/raw/job_descriptions.csv`

The dataset contains the following columns: job_id, field, job_title, company, required_skills, required_education, years_experience_required, salary_min, salary_max, and job_description.

---

*This dataset was generated as part of a capstone research project investigating algorithmic bias in automated resume screening systems.*
