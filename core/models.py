from django.conf import settings
from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.utils import timezone


class StudentProfile(models.Model):
    """Map an authenticated student account to an OULAD learner ID."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="student_profile",
    )
    student_id = models.CharField(max_length=64, unique=True)

    class Meta:
        ordering = ["student_id"]

    def __str__(self):
        return f"{self.user.username} ({self.student_id})"


class Intervention(models.Model):
    """A documented support action for one learner-course presentation."""

    class InterventionType(models.TextChoices):
        ASSESSMENT_SUPPORT = "assessment_support", "Assessment Support"
        LEARNING_SUPPORT = "learning_support", "Learning Support"
        MENTOR_FOLLOW_UP = "mentor_follow_up", "Mentor / Faculty Follow-up"
        STUDY_PLAN_SUPPORT = "study_plan_support", "Study Plan Support"
        OTHER = "other", "Other"

    class Priority(models.TextChoices):
        HIGH = "High", "High"
        MEDIUM = "Medium", "Medium"
        LOW = "Low", "Low"

    class Status(models.TextChoices):
        PENDING = "Pending", "Pending"
        IN_PROGRESS = "In Progress", "In Progress"
        COMPLETED = "Completed", "Completed"
        CANCELLED = "Cancelled", "Cancelled"

    student_id = models.CharField(max_length=32)
    code_module = models.CharField(max_length=16)
    code_presentation = models.CharField(max_length=16)
    intervention_type = models.CharField(max_length=32, choices=InterventionType.choices)
    priority = models.CharField(max_length=8, choices=Priority.choices)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    follow_up_date = models.DateField(null=True, blank=True)
    risk_probability_at_intervention = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["student_id", "code_module", "code_presentation"]),
            models.Index(fields=["status", "follow_up_date"]),
        ]

    @property
    def is_overdue(self):
        return bool(
            self.follow_up_date
            and self.follow_up_date < timezone.localdate()
            and self.status in {self.Status.PENDING, self.Status.IN_PROGRESS}
        )
