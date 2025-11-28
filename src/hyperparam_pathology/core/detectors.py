# src/hyperparam_pathology/core/detectors.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TrialIssue:
    """
    Represents a single 'issue' detected in a hyperparameter sweep trial.
    """
    trial_id: Any
    issue_type: str              # e.g. "nan_or_inf_metric", "overfitting_suspect"
    severity: str                # "low" | "medium" | "high"
    details: str                 # human-readable string
    hyperparams: Dict[str, Any]  # subset of row with just hyperparameters


# ---------- Helper ----------

# Columns we treat as bookkeeping, not hyperparameters
_IGNORE_COLS = {
    "trial_id",
    "status",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "epochs",
    "runtime_sec",
}


def _extract_hparams(row: pd.Series) -> Dict[str, Any]:
    return {k: row[k] for k in row.index if k not in _IGNORE_COLS}


# ---------- Detectors ----------

def detect_nan_inf_metrics(
    df: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
) -> List[TrialIssue]:
    """
    Detect trials where any metric is NaN or Inf.
    Default metric columns: common loss/acc columns if not provided.
    """
    issues: List[TrialIssue] = []

    if metric_columns is None:
        # Heuristic: look for typical metric column names
        candidate_cols = [
            "train_loss", "val_loss",
            "train_acc", "val_acc",
        ]
        metric_columns = [c for c in candidate_cols if c in df.columns]

    if not metric_columns:
        return issues

    for _, row in df.iterrows():
        for col in metric_columns:
            val = row.get(col, None)
            if pd.isna(val) or (isinstance(val, (float, int)) and not np.isfinite(val)):
                issues.append(
                    TrialIssue(
                        trial_id=row.get("trial_id"),
                        issue_type="nan_or_inf_metric",
                        severity="high",
                        details=f"{col} is {val}",
                        hyperparams=_extract_hparams(row),
                    )
                )
                # one issue per trial is enough for this detector
                break

    return issues


def detect_failed_trials(
    df: pd.DataFrame,
    completed_status: str = "completed",
) -> List[TrialIssue]:
    """
    Detect trials where status != completed (e.g., 'failed', 'crashed', etc.)
    """
    issues: List[TrialIssue] = []
    if "status" not in df.columns:
        return issues

    for _, row in df.iterrows():
        status = str(row.get("status", "")).lower()
        if status and status != completed_status.lower():
            issues.append(
                TrialIssue(
                    trial_id=row.get("trial_id"),
                    issue_type="failed_trial",
                    severity="high",
                    details=f"status='{row.get('status')}'",
                    hyperparams=_extract_hparams(row),
                )
            )

    return issues


def detect_overfitting(
    df: pd.DataFrame,
    train_col: str = "train_loss",
    val_col: str = "val_loss",
    threshold_ratio: float = 1.5,
    min_epochs: int = 5,
) -> List[TrialIssue]:
    """
    Detect overfitting based on loss: val_loss / train_loss >= threshold_ratio
    Only considers trials with at least `min_epochs` (if epochs column exists).
    """
    issues: List[TrialIssue] = []
    if train_col not in df.columns or val_col not in df.columns:
        return issues

    for _, row in df.iterrows():
        train_val = row.get(train_col)
        val_val = row.get(val_col)

        if pd.isna(train_val) or pd.isna(val_val):
            continue

        if "epochs" in df.columns:
            epochs = row.get("epochs", None)
            if pd.notna(epochs) and epochs < min_epochs:
                # too few epochs -> ignore for overfitting detection
                continue

        # loss: higher is worse, so val_loss >> train_loss = overfitting
        ratio = float(val_val) / max(float(train_val), 1e-8)

        if ratio >= threshold_ratio:
            issues.append(
                TrialIssue(
                    trial_id=row.get("trial_id"),
                    issue_type="overfitting_suspect",
                    severity="medium",
                    details=f"{val_col}/{train_col} ratio = {ratio:.2f}",
                    hyperparams=_extract_hparams(row),
                )
            )

    return issues


def detect_short_runs(
    df: pd.DataFrame,
    epoch_col: str = "epochs",
    runtime_col: str = "runtime_sec",
    quantile: float = 0.1,
) -> List[TrialIssue]:
    """
    Detect suspiciously short runs based on epochs and/or runtime.
    Uses lower `quantile` (e.g., 10%) as threshold.
    """
    issues: List[TrialIssue] = []
    if epoch_col not in df.columns and runtime_col not in df.columns:
        return issues

    # We'll consider epochs first, then runtime as a backup
    if epoch_col in df.columns:
        valid = df[epoch_col].dropna()
        if len(valid) > 0:
            epoch_threshold = float(valid.quantile(quantile))
        else:
            epoch_threshold = None
    else:
        epoch_threshold = None

    if runtime_col in df.columns:
        valid = df[runtime_col].dropna()
        if len(valid) > 0:
            runtime_threshold = float(valid.quantile(quantile))
        else:
            runtime_threshold = None
    else:
        runtime_threshold = None

    for _, row in df.iterrows():
        epochs_val = row.get(epoch_col) if epoch_col in df.columns else None
        rt_val = row.get(runtime_col) if runtime_col in df.columns else None

        short_by_epochs = (
            epoch_threshold is not None
            and pd.notna(epochs_val)
            and float(epochs_val) <= epoch_threshold
        )
        short_by_runtime = (
            runtime_threshold is not None
            and pd.notna(rt_val)
            and float(rt_val) <= runtime_threshold
        )

        if short_by_epochs or short_by_runtime:
            details_parts = []
            if short_by_epochs:
                details_parts.append(f"{epoch_col}={epochs_val} (<= {epoch_threshold:.1f})")
            if short_by_runtime:
                details_parts.append(f"{runtime_col}={rt_val} (<= {runtime_threshold:.1f})")

            issues.append(
                TrialIssue(
                    trial_id=row.get("trial_id"),
                    issue_type="short_run",
                    severity="medium",
                    details="; ".join(details_parts),
                    hyperparams=_extract_hparams(row),
                )
            )

    return issues


def detect_param_correlations(
    df: pd.DataFrame,
    issues: List[TrialIssue],
    param_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    For each hyperparameter, compute the fraction of trials with issues for:
      - value buckets (numeric)
      - each category (categorical)
    Returns a dict that the LLM can interpret to describe patterns.
    """
    summary: Dict[str, Any] = {}

    if df.empty or not issues:
        return summary

    # Work on a copy so we don't mutate caller's df
    df_tmp = df.copy()

    issue_trial_ids = {i.trial_id for i in issues}
    df_tmp["has_issue"] = df_tmp["trial_id"].isin(issue_trial_ids)

    # Decide which columns are "hyperparameters"
    if param_cols is None:
        param_cols = [c for c in df_tmp.columns if c not in _IGNORE_COLS and c != "has_issue"]

    for param in param_cols:
        col = df_tmp[param]
        # Skip all-null or constant columns
        if col.isna().all() or col.nunique(dropna=True) <= 1:
            continue

        # Numeric vs categorical handling
        if np.issubdtype(col.dtype, np.number):
            # Bin numeric column into up to 5 quantile-based buckets
            try:
                df_tmp["_bin"] = pd.qcut(
                    col,
                    q=min(5, col.nunique()),
                    duplicates="drop",
                )
                grouped = df_tmp.groupby("_bin")["has_issue"].mean().sort_values(ascending=False)
                buckets = {
                    str(interval): float(rate)
                    for interval, rate in grouped.items()
                }
                summary[param] = {
                    "type": "numeric",
                    "buckets": buckets,
                }
            except Exception:
                # fallback: treat as categorical
                grouped = df_tmp.groupby(param)["has_issue"].mean().sort_values(ascending=False)
                values = {str(k): float(v) for k, v in grouped.items()}
                summary[param] = {
                    "type": "categorical",
                    "values": values,
                }
        else:
            grouped = df_tmp.groupby(param)["has_issue"].mean().sort_values(ascending=False)
            values = {str(k): float(v) for k, v in grouped.items()}
            summary[param] = {
                "type": "categorical",
                "values": values,
            }

    # Clean up helper columns
    df_tmp.drop(columns=["has_issue"], inplace=True, errors="ignore")
    if "_bin" in df_tmp.columns:
        df_tmp.drop(columns=["_bin"], inplace=True, errors="ignore")

    return summary
