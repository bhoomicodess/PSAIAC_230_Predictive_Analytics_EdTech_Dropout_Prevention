# Demonstration Script and Viva Preparation

## 5-8 Minute Demonstration

Use the demo database and approved test accounts. Do not show real learner data or put credentials in screenshots. Confirm that the early-warning artifacts exist before the presentation. Timings total approximately 7 minutes 30 seconds.

| Time | What to click | What to explain | Technical concept demonstrated |
|---|---|---|---|
| 0:00-0:25 | Open `/login/`; sign in with the approved Admin demo account. | Introduce the system as an early-warning/student-support prototype. | Django authentication and session creation. |
| 0:25-0:55 | Show the Dashboard at `/`. | Point out aggregate learner-risk and intervention summaries. Distinguish this from the separate generic CSV workflow. | Django-rendered dashboard and summaries from current data/artifacts. |
| 0:55-1:35 | Open **Student Risk Dashboard**. | Show counts and the learner table; explain filters, sorting, search, and pagination. | Persisted model inference, cached result set, server-side filtering. |
| 1:35-2:00 | Select **High** risk and, if useful, a module filter. | Explain that High is a project-defined probability band, not an institutional alert standard. | Probability-to-band mapping and query controls. |
| 2:00-2:40 | Open one synthetic learner from the results. | Explain learner-course-presentation identity and the current predicted probability. | One-record lookup and persisted model prediction. |
| 2:40-3:05 | Show higher/lower indicators on the profile. | Explain that these are Logistic Regression coefficient contributions, not causal explanations. | Model-native coefficient contribution display. |
| 3:05-3:30 | Open **Recommendations** for that learner. | Point out that suggestions use early assessment signals and risk level. | Deterministic, rule-based support logic. |
| 3:30-4:05 | Click **Record Intervention**; submit a demo support type, priority, note, and follow-up date. | Explain the probability is captured server-side and that support is an educator decision. | Validated intervention form and Django database record. |
| 4:05-4:35 | Open **Interventions**, filter by status/type, then return to the profile history. | Show pending/in-progress/completed tracking and follow-up history. | Database-backed workflow, filters, status updates, overdue calculation. |
| 4:35-5:00 | Open **Reports** and optionally download the CSV. | Explain this is an aggregate current risk/intervention snapshot, not an impact evaluation. | Aggregate reporting and CSV export. |
| 5:00-5:35 | Open **Model Performance**. | Compare the three models and explain the 0.30/0.50/0.70 precision-recall tradeoff. | Held-out metrics, ROC-AUC, threshold diagnostics. |
| 5:35-6:05 | Open **Batch Prediction**; use a prepared compatible sample CSV, then show the results/download. | Explain required learner/course keys and Day-60 numeric inputs; outcomes are not exported as model inputs. | Input validation and batched probability inference. |
| 6:05-7:25 | Sign out; sign in using a Student test account mapped to a synthetic learner. Verify root redirects to the own profile, show only the mapped learner option, then try another learner ID and the upload/model-operation URLs. | Explain route authorization and the mapped `StudentProfile` check. The other-ID request and institutional operations must be denied; the student must not receive another learner ID in page HTML. | Server-side context filtering, direct URL checks, and role-specific root behavior. |

### Before the Demo

- Verify that the configured demo users have the expected groups and StudentProfile mapping.
- Use synthetic IDs and non-sensitive notes.
- Have a small CSV containing only the required identifiers and six early-warning features ready for batch prediction.
- Avoid retraining during a short demo unless the training page itself is the focus; training evaluates and persists artifacts.
- Do not describe recommendations or interventions as guaranteed to prevent dropout.
- Do not claim learner VLE activity features; `studentVle.csv` is absent.

## Likely Viva Questions and Concise Answers

1. **What problem does the project address?**
   It provides an early-warning workflow to help educators identify learners who may need support and organize timely follow-up. It does not guarantee dropout prevention.

2. **What is the official problem statement?**
   “Online courses see 80–90% dropout rates. Industry needs a data science model that identifies at-risk learners early based on engagement signals and triggers personalized intervention strategies.” In the current implementation, the available signals are registration and assessment fields, not learner clickstream.

3. **Which dataset is used?**
   OULAD files in the repository, primarily `studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`, and `assessments.csv` for the early-warning feature builder.

4. **What does `vle.csv` contain here?**
   Course/site activity-type metadata and week ranges. It is not learner-level VLE activity.

5. **Is `studentVle.csv` available?**
   No. Therefore the system does not compute learner click counts, login frequency, active days/weeks, or VLE activity trends.

6. **How is the target created?**
   `Withdrawn` maps to 1; `Pass`, `Fail`, and `Distinction` map to 0. The target column is named `dropout`.

7. **What is the target distribution?**
   22,437 non-dropout records (68.84%) and 10,156 dropout records (31.16%), total 32,593.

8. **Why use Day 60?**
   It is the project's chosen early-warning cutoff so the experiment can evaluate information from early scheduled assessments. It is a project-defined experimental choice, not an institutionally validated point.

9. **How is the Day-60 cutoff implemented?**
   Assessment definitions are retained when `assessments.date <= 60`. The code does not separately enforce `studentAssessment.date_submitted <= 60`, so late submissions for an assessment scheduled by Day 60 may be included.

10. **Which features are used?**
    Registration offset, early assessment count, mean and standard deviation of score, scheduled assessment count, submission rate, module, and presentation. Module/presentation are one-hot encoded.

11. **Why exclude `id_student`?**
    It is an identifier used to locate the learner record, not a predictive behavior. Including it could let the classifier memorize identifiers rather than learn general patterns.

12. **Why exclude `final_result`?**
    It is used to derive the target and is only known after the course outcome; using it as an input would leak the answer.

13. **How is future unregistration data handled?**
    `date_unregistration` is not used as a feature. The current early-warning table uses `date_registration` for registration timing.

14. **Which models were compared?**
    Logistic Regression, Decision Tree, and Random Forest.

15. **How was performance evaluated?**
    A stratified 80/20 split with `random_state=42`: 26,074 training records and 6,519 held-out records. Metrics include accuracy, precision, recall, weighted F1, and ROC-AUC.

16. **Why was Logistic Regression selected?**
    It had the highest measured accuracy (82.68%) and ROC-AUC (84.93%) among the three models in this current experiment. That does not make it universally best.

17. **What does precision mean?**
    Of the records predicted as dropout-positive at a chosen classification threshold, the share that are positive in the held-out labels.

18. **What does recall mean?**
    Of the actual dropout-positive records in the held-out labels, the share the model classified as positive at that threshold.

19. **What does F1 measure?**
    The harmonic mean of precision and recall. The model comparison reports weighted F1 across the two classes.

20. **What does ROC-AUC mean?**
    Area under the ROC curve; it summarizes ranking discrimination across probability thresholds on this held-out split. It does not guarantee deployment performance.

21. **How do thresholds affect results?**
    In this experiment, raising the positive threshold from 0.30 to 0.70 raised precision from 70.78% to 81.70% and reduced recall from 70.26% to 55.39%. The choice depends on domain costs and review, not one universal optimum.

22. **What are the risk bands?**
    Low `<0.30`, Medium `0.30 to <0.70`, High `>=0.70`. These are project-defined thresholds and are not institutionally validated.

23. **What is the 0.50 confusion matrix?**
    TP=1,248, FP=346, TN=4,142, FN=783, on the held-out split.

24. **What are the model indicators?**
    The implementation calculates feature value times Logistic Regression coefficient and presents the largest positive/negative contributions. These are model indicators, not causal explanations or proof of why a learner may withdraw.

25. **How are recommendations generated?**
    Deterministic rules use risk level and early-assessment fields. Examples include submission rate below 75% and mean score below 50 triggering specific support suggestions. Recommendations are supportive prompts, not predicted intervention effects.

26. **What does intervention management store?**
    Learner/course/presentation, support type, priority, status, notes, creation time, follow-up date, and the model probability captured when the action was recorded.

27. **Can the system prove an intervention worked?**
    No. It tracks interventions and may show a later risk probability for context, but there is no causal evaluation of intervention effectiveness.

28. **What are the user roles?**
    `ADMIN`, `EDUCATOR`, and `STUDENT` Django groups. Route access is checked server-side; a StudentProfile maps a student login to a learner ID.

29. **How is student-to-student prediction access checked?**
    The individual prediction/recommendations views compare the requested learner ID with the mapped StudentProfile and reject a mismatch. Tests cover this direct-ID rejection.

30. **How does the final Student boundary work?**
    Student learner-option context is filtered to the mapped ID on both prediction and recommendation pages; direct requests for another ID are rejected. The root GET redirects to the learner profile, root POST is denied, and institutional operations are role-protected. These view checks do not replace a complete deployment security audit.

31. **What is batch prediction?**
    It validates a CSV with the identifiers and required Day-60 inputs, applies the persisted model/preprocessing metadata, and returns probabilities and risk bands. It ignores outcome columns as model inputs/outputs.

32. **What does Reports show?**
    Current aggregate risk distribution, module summaries, and intervention status counts, with a CSV summary download. It is not a research impact report.

33. **How many tests pass?**
    The current suite has 62 tests, and the verified baseline is 62 passed, 0 failed.

34. **Is the application deployed in production?**
    No live production deployment is evidenced. The repository has environment-aware settings, WSGI configuration, a health endpoint, and deployment dependencies, but host-specific HTTPS/database/static configuration remains.

35. **What does `collectstatic` prove?**
    It gathers static assets into `STATIC_ROOT`. It does not by itself configure a web server to deliver those assets; WhiteNoise is not active in the current middleware.

36. **What are the main limitations?**
    Dataset-specific results, unvalidated project thresholds, no learner VLE activity, the submission-date cutoff caveat, non-causal model indicators/recommendations, and the need for institutional privacy and production security review.
