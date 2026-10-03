"""Deterministic, rule-based support suggestions for early-warning results.

This module deliberately does not train, load, or call the prediction model.
It turns an already-created prediction context into non-causal support
suggestions using only the available early-assessment features.
"""

from math import isfinite


LOW_SUBMISSION_RATE = 0.75
WEAK_ASSESSMENT_SCORE = 50.0
REQUIRED_FIELDS = {"category", "priority", "title", "message", "reason"}


def _number(context, feature_name):
    """Read a finite numeric early-warning feature, if one is available."""
    features = context.get("early_warning_features", {})
    value = features.get(feature_name, context.get(feature_name))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _recommendation(category, priority, title, message, reason):
    return {
        "category": category,
        "priority": priority,
        "title": title,
        "message": message,
        "reason": reason,
    }


def generate_recommendations(prediction_context):
    """Return unique, supportive recommendations for a valid prediction context.

    Invalid or incomplete contexts safely produce no recommendations. The
    returned order is stable: specific assessment support, learning support,
    then risk-level follow-up or monitoring.
    """
    if not isinstance(prediction_context, dict):
        return []

    risk_level = prediction_context.get("risk_level")
    if risk_level not in {"Low", "Medium", "High"}:
        return []
    if any(
        prediction_context.get(field) in (None, "")
        for field in ("student_id", "module", "presentation")
    ):
        return []

    probability = prediction_context.get("dropout_probability")
    try:
        if not isfinite(float(probability)):
            return []
    except (TypeError, ValueError):
        return []

    # Low-risk results intentionally remain lightweight, even if individual
    # feature values might otherwise match a higher-intensity rule.
    if risk_level == "Low":
        return [
            _recommendation(
                "Progress Maintenance",
                "Low",
                "Maintain Current Progress",
                "Continue current learning progress and maintain regular assessment participation.",
                "The current model prediction is in the Low risk level, so no additional intervention alert is suggested.",
            )
        ]

    scheduled = _number(prediction_context, "early_assessment_scheduled_count")
    submission_rate = _number(
        prediction_context, "early_assessment_submission_rate"
    )
    assessment_count = _number(prediction_context, "early_assessment_count")
    assessment_score = _number(prediction_context, "early_assessment_score_mean")

    recommendations = []
    support_indicator_count = 0
    specific_priority = "High" if risk_level == "High" else "Medium"

    if (
        scheduled is not None
        and scheduled > 0
        and submission_rate is not None
        and submission_rate < LOW_SUBMISSION_RATE
    ):
        support_indicator_count += 1
        recommendations.append(
            _recommendation(
                "Assessment Support",
                specific_priority,
                "Early Assessment Participation Support",
                "Encourage completion of pending or early assessments and provide additional assessment support.",
                "The current profile shows an early assessment submission rate below 75%, an indicator associated with the model's current prediction.",
            )
        )

    if (
        assessment_count is not None
        and assessment_count > 0
        and assessment_score is not None
        and assessment_score < WEAK_ASSESSMENT_SCORE
    ):
        support_indicator_count += 1
        recommendations.append(
            _recommendation(
                "Learning Support",
                specific_priority,
                "Additional Learning Support",
                "Provide additional learning resources and guided study support.",
                "The current profile includes relatively weak early assessment performance, an indicator associated with the model's current prediction.",
            )
        )

    if risk_level == "High":
        follow_up_priority = "High" if support_indicator_count >= 2 else "Medium"
        recommendations.append(
            _recommendation(
                "Learner Follow-up",
                follow_up_priority,
                "Mentor or Faculty Follow-up",
                "Consider a mentor or faculty follow-up to understand the learner's current challenges and provide appropriate support.",
                "The current model prediction is in the High risk level; this is an application-level intervention priority, not a guaranteed outcome.",
            )
        )
    elif support_indicator_count >= 2:
        recommendations.append(
            _recommendation(
                "Learner Follow-up",
                "Medium",
                "Mentor or Faculty Follow-up",
                "Consider a mentor or faculty follow-up to understand the learner's current challenges and provide appropriate support.",
                "Multiple early assessment indicators are associated with the model's current Medium risk prediction.",
            )
        )
    elif not recommendations:
        recommendations.append(
            _recommendation(
                "Progress Monitoring",
                "Medium",
                "Continue Progress Monitoring",
                "Continue monitoring the learner's early-course progress and reassess risk as new information becomes available.",
                "The model indicates elevated risk, but the available early assessment indicators do not support a more specific suggestion.",
            )
        )

    unique_recommendations = []
    seen = set()
    for recommendation in recommendations:
        key = (recommendation["category"], recommendation["title"])
        if key not in seen:
            seen.add(key)
            unique_recommendations.append(recommendation)
    return unique_recommendations
