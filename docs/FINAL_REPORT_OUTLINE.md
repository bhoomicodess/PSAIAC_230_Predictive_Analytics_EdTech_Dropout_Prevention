# PSAIAC_230 Final-Year Project Report

## Working Title

**PSAIAC_230 - Predictive Analytics for EdTech Dropout Prevention**

Use the official project description: **An early-warning and student-support platform for identifying learners at elevated dropout risk and enabling timely intervention.** Prediction supports prevention through early intervention; it does not guarantee dropout prevention.

> Submission note: this is a source-grounded report outline and content brief, not a claim that the final bound report is complete. Confirm institutional formatting, team roles, supervisor details, and citations before submission.

## Abstract (Draft)

Online learning providers need ways to identify learners who may need timely support. PSAIAC_230 is a Django web application that combines a generic CSV workflow with a separate OULAD-based early-warning workflow. The latter constructs learner-course-presentation records from registration and early assessment data, maps `Withdrawn` to a binary dropout target, evaluates Logistic Regression, Decision Tree, and Random Forest on a stratified held-out split, and persists a Logistic Regression model for individual and batch probability predictions. The application presents project-defined risk bands, model coefficient indicators, deterministic support recommendations, educator intervention tracking, dashboards, model diagnostics, and aggregate CSV reports. On the current 32,593-row OULAD-derived dataset, Logistic Regression measured 82.68% accuracy and 84.93% ROC-AUC in the current experiment. These results are dataset- and split-specific; thresholds and recommendations require domain validation, and predictions do not establish causation or guarantee prevention.

## Chapter 1 - Introduction

Describe online learning participation and the practical need to identify support needs early. Introduce prediction as a decision-support input rather than an automated decision-maker. Add verified context and literature citations during final editing; do not use uncited prevalence claims as independently established facts.

## Chapter 2 - Problem Statement and Objectives

### Problem Statement

> Online courses see 80–90% dropout rates. Industry needs a data science model that identifies at-risk learners early based on engagement signals and triggers personalized intervention strategies.

Retain this exact project statement. Clarify that the current repository does not contain learner-level VLE activity; implemented indicators are registration timing and early assessment records.

### Implemented Objectives

- Construct an OULAD-derived early-warning dataset using the available data and Day-60 scheduled-assessment cutoff.
- Estimate learner dropout probability and display project-defined risk bands.
- Display model-native feature contribution indicators.
- Generate support recommendations from risk and early-assessment features.
- Provide individual/batch prediction, risk dashboard, intervention tracking, model performance, and aggregate reports.
- Provide login, role-based route controls, and mapped learner access checks.

### Scope Exclusions

No learner clickstream modeling, causal intervention-effect measurement, institution-specific threshold validation, live deployment, or additional predictive model is claimed.

## Chapter 3 - Literature and Background

Cover early-warning systems in online learning, classification of educational outcomes, evaluation under class imbalance, probability thresholds, interpretable model indicators, and human-centered learner support. **References are not supplied in the audited repository.** Insert only sources personally reviewed and verified, including the official OULAD dataset citation and publication metadata. Do not submit placeholder references as real citations.

## Chapter 4 - Dataset and Preprocessing

### Dataset

The repository contains OULAD-derived files: `studentInfo.csv` (32,593 rows), `studentRegistration.csv` (32,593), `studentAssessment.csv` (173,912), `assessments.csv` (206), `courses.csv` (22), and `vle.csv` (6,364). The row/column counts are the checked CSV inventory at documentation time.

`vle.csv` describes VLE sites/activity types and course presentation weeks. It is **content/activity metadata**, not learner-level VLE activity. There is no `studentVle.csv`; no learner clicks, login frequency, active days, active weeks, or activity trends are computed.

### Target

The observed `final_result` is mapped as follows:

| `final_result` | `dropout` |
|---|---:|
| Withdrawn | 1 |
| Pass | 0 |
| Fail | 0 |
| Distinction | 0 |

The current generated dataset has 32,593 records: 22,437 non-dropout (68.84%) and 10,156 dropout (31.16%).

### Early-Warning Inputs

- `registration_offset_days`
- `early_assessment_count`
- `early_assessment_score_mean`
- `early_assessment_score_std`
- `early_assessment_scheduled_count`
- `early_assessment_submission_rate`
- `code_module`
- `code_presentation`

Categorical module and presentation values are one-hot encoded for model training. `id_student` remains a record key, but is excluded from the predictive feature list. `dropout` is the target; `final_result` is excluded as an input.

## Chapter 5 - Methodology

1. Read student, registration, assessment submission, and assessment-definition files.
2. Map final course outcome to the binary target.
3. Retain assessment definitions with scheduled date at or before Day 60.
4. Aggregate available early-assessment count, score mean/std, scheduled count, and submission rate per learner/module/presentation; include registration timing and course/presentation identifiers.
5. Validate target, required features, numeric values, and record uniqueness.
6. One-hot encode categorical features and fill missing transformed values using training/workflow medians.
7. Make a stratified 80/20 split with `random_state=42`; compare three classifiers.
8. Persist the selected Logistic Regression model and preprocessing metadata; use those artifacts for individual and batch predictions.

### Leakage and Cutoff Qualification

The feature builder excludes `id_student`, `final_result`, `date_unregistration`, full-course aggregates, and assessment definitions scheduled after Day 60 from model inputs. Important implementation detail: the current code filters using the assessment definition's scheduled `date`, but does not independently require each `studentAssessment.date_submitted` to be on/before Day 60. Thus, late submissions for an assessment scheduled by Day 60 can be included. Describe this limitation explicitly; do not claim a strict per-submission observation-time cutoff.

## Chapter 6 - Machine-Learning Models

The early-warning experiment evaluates Logistic Regression (`max_iter=1000`), Decision Tree (`random_state=42`), and Random Forest (100 trees, `random_state=42`). Evaluation uses the same stratified 80/20 split of 26,074 training and 6,519 held-out records.

| Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 82.68% | 82.29% | 82.68% | 82.04% | 84.93% |
| Decision Tree | 74.92% | 74.90% | 74.92% | 74.91% | 71.90% |
| Random Forest | 80.21% | 79.64% | 80.21% | 79.71% | 81.16% |

Logistic Regression was selected for this project's persisted prediction workflow because it had the highest measured accuracy and ROC-AUC in this experiment. Do not call it universally best. No convergence warning appeared in the reproduced current training run; state this only as a run-specific observation.

## Chapter 7 - System Design and Architecture

```text
OULAD files -> validation/feature engineering -> Day-60 dataset
-> model comparison -> persisted model/preprocessing artifacts
-> individual and batch risk prediction -> indicators/recommendations
-> educator interventions -> aggregate dashboard/reports

Generic CSV -> session-scoped upload state -> preprocessing -> generic model comparison

Django templates/views -> role checks -> Django database
                                   |-> StudentProfile mapping
                                   |-> Intervention records
```

Describe the Django web layer, pandas/scikit-learn ML layer, CSV/joblib data and artifact layer, Django database for accounts/profile/interventions, Django group-based access, and reporting layer. Do not describe a separate API or learner-level VLE pipeline.

## Chapter 8 - Implementation

Document the actual modules: upload/dashboard, preprocessing, generic model training, early-warning training, individual prediction, risk dashboard, batch prediction, recommendations, interventions, model performance, reports, settings, help, login/logout, and access denied. Explain that the generic CSV pipeline and OULAD early-warning pipeline are separate workflows. Add verified UI screenshots from the checklist only after capturing them from the real app.

## Chapter 9 - Results and Evaluation

Present the Chapter 6 comparison table and these held-out threshold diagnostics:

| Probability threshold | Predicted positive | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.30 | 2,016 | 70.78% | 70.26% | 70.52% |
| 0.50 | 1,594 | 78.29% | 61.45% | 68.86% |
| 0.70 | 1,377 | 81.70% | 55.39% | 66.02% |

At 0.50: TP=1,248, FP=346, TN=4,142, FN=783; ROC-AUC=84.93%. Explain that the higher threshold raises precision and lowers the positive prediction count in this run while recall falls. Threshold choice depends on institutional costs and requires domain review; none is universally optimal.

## Chapter 10 - Web Application

Explain dashboard aggregates, search/filter/sort/pagination, individual probability and indicators, CSV batch predictions/download, rule-based recommendations, intervention register/status/follow-up, model-performance diagnostics, and aggregate report/CSV download. State that the report is an aggregate current system view, not a causal outcome evaluation.

## Chapter 11 - Security and RBAC

| Role | Verified access summary |
|---|---|
| ADMIN | Generic preprocessing/training and early-warning training; also protected learner-support, performance, and reports routes. |
| EDUCATOR | Risk dashboard, individual prediction/recommendations, batch prediction, interventions, model performance, and reports; not generic model training. |
| STUDENT | Individual prediction and recommendations for the mapped `StudentProfile` ID only. Server-generated learner options are filtered to that ID; tests reject another learner ID and deny institutional operations. |

Django login and server-side route checks are the security boundary; hiding a link is not authorization. The root route now serves the administrative upload/dashboard page only to Admins, redirects Educators to the learner-risk dashboard, and redirects Students to individual prediction. Non-Admin POSTs to the root route return 403 before dataset work. Student learner-option context is filtered on the server for both individual prediction and recommendations, and direct requests for another learner remain rejected. Regression tests also cover student access denial for institutional operations and retained Admin/Educator permissions. These view-level tests do not replace institutional privacy review or a complete production security audit.

## Chapter 12 - Testing

The current suite contains 62 tests. The final baseline is 62 passed, 0 failed. Covered categories include feature/target and cutoff behavior, recommendations, batch input/prediction, individual predictions and thresholds, model performance and report downloads, login/logout, route permissions, student ID manipulation and option filtering, role-specific root/upload access, intervention creation/update/history/filtering, navigation, and generic dataset session state.

Document the commands and exact results from the final validation section. The Django test client and test database do not replace a production security review or full end-to-end browser test.

## Chapter 13 - Limitations

- Metrics are from this dataset and one fixed experiment; transfer to other cohorts is unknown.
- Current class balance is 68.84% non-dropout and 31.16% dropout; accuracy alone is insufficient.
- Risk thresholds are project-defined and not institutionally validated.
- Coefficient contributions are model indicators, not causal explanations.
- There is no `studentVle.csv` or learner-level VLE activity.
- Scheduled assessment date is cut off at Day 60, but individual submission date is not independently filtered.
- Recommendations are rule-based support suggestions; no intervention outcome or causal effect is established.
- Student access is restricted in the covered profile/recommendation views; production data protection still requires institutional review and infrastructure controls.
- Production settings are prepared, but no live deployment is evidenced; WhiteNoise is not active in middleware and static hosting needs explicit setup.

## Chapter 14 - Future Enhancements

Present only as proposed work: expand tests across every role/route and data-disclosure path; obtain learner-level activity data only with suitable governance; enforce an observation-time cutoff if required; evaluate calibration, cohort generalization, and fairness; validate thresholds with educators; and complete production database, HTTPS, and static-file deployment configuration.

## Chapter 15 - Conclusion

Conclude that the implemented prototype combines a Day-60 OULAD-derived classifier with an operational Django workflow for risk review, recommendations, intervention tracking, and reporting. Restate the current measured experiment results with their scope, human-review requirement, data limitations, and absence of guaranteed prevention or live deployment.

## Submission Items to Complete Manually

- Add institution-approved cover page, degree/program, supervisor, and submission date.
- Confirm the official team roster and each member's contribution; the README currently has names but no confirmed responsibility mapping.
- Add verified OULAD and literature references after consulting the original sources.
- Decide whether a license is required and add the approved license; none was present in the audited repository.
- Capture genuine application screenshots using `SCREENSHOT_CHECKLIST.md`.
- Conduct the institution's required privacy/security review before using real learner data.
- Record the actual live hosting provider/URL only if deployment is completed; otherwise state prototype readiness, not deployment.