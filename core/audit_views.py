
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, redirect

from core.views import role_required
from core.data_quality_audit import run_data_quality_audit


@role_required("ADMIN")
def data_quality_audit(request):
    report = None
    error = None

    if request.method == "POST":
        try:
            project_root = Path(settings.BASE_DIR)
            data_directory = project_root / "data"
            output_path = data_directory / "early_warning_audit_report.json"

            report = run_data_quality_audit(
                data_directory=data_directory,
                output_path=output_path,
            )
            messages.success(request, "Data quality audit completed.")
        except (FileNotFoundError, ValueError) as exc:
            error = (
                "The audit could not run with the current dataset. "
                f"Please check the required OULAD files and columns. Details: {exc}"
            )
        except Exception:
            error = (
                "The audit encountered an unexpected error. "
                "Please check the server terminal for details."
            )
    return render(
        request,
        "core/data_quality_audit.html",
        {
            "report": report,
            "error": error,
            "user_role": "ADMIN",
        },
    )
