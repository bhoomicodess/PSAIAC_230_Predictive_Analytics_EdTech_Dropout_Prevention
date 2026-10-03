# Final Presentation Outline

Target length: 12-15 slides. Keep slide text brief; use the speaker notes to explain caveats. Replace team/supervisor placeholders manually. Use only genuine screenshots and verified results.

## Slide 1 - Title

- PSAIAC_230 - Predictive Analytics for EdTech Dropout Prevention
- Early warning and student support
- Team, department, supervisor, date: **confirm and fill in**

## Slide 2 - Problem Statement

> Online courses see 80–90% dropout rates. Industry needs a data science model that identifies at-risk learners early based on engagement signals and triggers personalized intervention strategies.

Speaker note: this is the supplied project statement. The current implementation uses registration and early-assessment information, not learner clickstream data.

## Slide 3 - Motivation

- Timely educator review can make support actionable.
- A probability/risk band helps prioritize review, not make an automatic student decision.
- Prediction is intended to support prevention through early intervention; it does not guarantee prevention.

## Slide 4 - Implemented Objectives

- Build a Day-60 OULAD early-warning pipeline.
- Produce dropout probabilities, risk bands, and coefficient-based model indicators.
- Support recommendations, interventions, batch prediction, dashboards, and reports.
- Protect selected routes with Django authentication and role checks.

## Slide 5 - Proposed Solution

- Separate generic CSV analysis workflow.
- OULAD early-warning workflow with persisted Logistic Regression inference.
- Educator-facing risk review and intervention tracking.
- Django application with server-side authorization.

## Slide 6 - Dataset and Methodology

- OULAD: `studentInfo`, `studentRegistration`, `studentAssessment`, `assessments`; `courses` and `vle` metadata also present.
- 32,593 learner-course-presentation records.
- Target: Withdrawn=1; Pass/Fail/Distinction=0.
- Scheduled assessment definition date through Day 60; registration and early-assessment aggregate features.
- No `studentVle.csv`; no learner clicks/login/active-day metrics.

## Slide 7 - System Architecture

```text
OULAD CSV -> Day-60 features -> model evaluation -> joblib artifacts
-> individual/batch prediction -> risk and indicators -> recommendations
-> intervention records -> reports
                         Django views/templates + auth + database
```

Mention that account/profile/intervention data is in Django's configured database; the early-warning dataset/artifacts are files.

## Slide 8 - Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- Stratified 80/20 split, `random_state=42`
- 26,074 training rows; 6,519 test rows

## Slide 9 - Model Results

| Model | Accuracy | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 82.68% | 82.04% | 84.93% |
| Decision Tree | 74.92% | 74.91% | 71.90% |
| Random Forest | 80.21% | 79.71% | 81.16% |

Speaker note: Logistic Regression was selected for this project's current experiment; these numbers are not a universal ranking or an external validation result.

## Slide 10 - Prediction and Model Indicators

- Individual and batch dropout probabilities.
- Project-defined Low `<0.30`, Medium `0.30 to <0.70`, High `>=0.70` bands.
- Logistic Regression coefficient contributions shown as higher/lower indicators.
- Indicators are not causal explanations.

## Slide 11 - Recommendations and Interventions

- Deterministic support rules use risk and early assessment fields.
- Examples: assessment participation support, learning support, monitoring/faculty follow-up.
- Educators record type, priority, notes, follow-up date, and status.
- Tracking a follow-up does not establish that it caused a change in outcome.

## Slide 12 - Web Application

Use actual screenshots: risk dashboard, individual prediction, recommendations, intervention register, model performance, reports, and batch prediction. Explain the generic upload/preprocessing/model-training workflow is separate from OULAD early-warning inference.

## Slide 13 - Authentication and RBAC

- Roles: Admin, Educator, Student.
- Server-side route decorators; mapped StudentProfile ID check for individual prediction/recommendations.
- Tests reject another learner ID and deny Student reports/model training.
- Root route is role-aware; non-Admin upload POSTs and Student institutional operations are denied server-side. Student learner options are limited to their mapped ID and another ID is rejected.
- Describe the demonstrated routes precisely; this is not a claim of a complete infrastructure security audit.

## Slide 14 - Results, Testing, and Limitations

- 62 tests passed, 0 failed in the current baseline.
- 0.50 threshold: precision 78.29%, recall 61.45%, F1 68.86%; TP=1,248, FP=346, TN=4,142, FN=783.
- Results are dataset-specific; thresholds are unvalidated; VLE learner activity is absent; scheduled-date cutoff does not independently filter submission dates.

## Slide 15 - Conclusion and Future Scope

- Prototype integrates prediction with educator support workflow.
- Predictions support, not replace, human judgment.
- Future: review student data boundaries, verify strict observation-time cutoff, collect governed learner-level activity data if permitted, validate cohort performance/thresholds, and complete live deployment checks.
- Questions
