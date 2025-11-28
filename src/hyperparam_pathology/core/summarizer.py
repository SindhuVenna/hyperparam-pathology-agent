# src/hyperparam_pathology/core/summarizer.py

from __future__ import annotations

from typing import Any, Dict, List

from .detectors import TrialIssue


def serialize_issue(issue: TrialIssue) -> Dict[str, Any]:
    """
    Convert TrialIssue into a plain dict that can be JSON-serialized.
    """
    return {
        "trial_id": issue.trial_id,
        "issue_type": issue.issue_type,
        "severity": issue.severity,
        "details": issue.details,
        "hyperparams": issue.hyperparams,
    }


def build_structured_summary(
    issues: List[TrialIssue],
    param_correlations: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a structured summary dict that the LLM agents will consume.

    Shape (roughly):

    {
      "meta": {
        "num_issues": ...,
        "num_trials_with_issue": ...,
        "issue_types": {...},
        "severity_counts": {...}
      },
      "issues_by_type": {
        "nan_or_inf_metric": [...examples...],
        ...
      },
      "param_correlations": {
        "lr": {...},
        ...
      }
    }
    """
    summary: Dict[str, Any] = {}

    # Basic meta stats
    num_issues = len(issues)
    issues_by_type: Dict[str, List[TrialIssue]] = {}
    severity_counts: Dict[str, int] = {}

    trial_ids_with_issues = set()

    for iss in issues:
        issues_by_type.setdefault(iss.issue_type, []).append(iss)
        severity_counts[iss.severity] = severity_counts.get(iss.severity, 0) + 1
        trial_ids_with_issues.add(iss.trial_id)

    counts_by_type = {k: len(v) for k, v in issues_by_type.items()}

    meta = {
        "num_issues": num_issues,
        "num_trials_with_issue": len(trial_ids_with_issues),
        "counts_by_type": counts_by_type,
        "severity_counts": severity_counts,
    }

    # Example issues per type (keep it small to avoid overwhelming the LLM)
    examples_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for issue_type, lst in issues_by_type.items():
        examples_by_type[issue_type] = [serialize_issue(i) for i in lst[:10]]

    summary["meta"] = meta
    summary["issues_by_type"] = examples_by_type
    summary["param_correlations"] = param_correlations

    return summary
