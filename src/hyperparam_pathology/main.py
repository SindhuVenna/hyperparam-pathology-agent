#!/usr/bin/env python
import os
import sys
import json
import warnings

from datetime import datetime

import pandas as pd

from hyperparam_pathology.crew import HyperparamPathologyCrew
from hyperparam_pathology.core.detectors import (
    detect_nan_inf_metrics,
    detect_overfitting,
    detect_short_runs,
    detect_param_correlations,
)
from hyperparam_pathology.core.summarizer import build_structured_summary

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def _build_raw_summary_json(csv_path: str) -> str:
    """
    Load a CSV of hyperparameter sweep results and return a JSON string
    with the structured summary that the crew will consume.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    issues = []
    issues += detect_nan_inf_metrics(df)
    issues += detect_overfitting(df)
    issues += detect_short_runs(df)

    param_corr = detect_param_correlations(df, issues)
    summary = build_structured_summary(issues, param_corr)

    return json.dumps(summary, indent=2, default=str)


def run():
    """
    Run the crew on a hyperparameter sweep CSV.
    """
    csv_path = os.getenv("HPATH_RESULTS_CSV", "results.csv")
    print(f"[INFO] Using CSV: {csv_path}")

    try:
        raw_summary_json = _build_raw_summary_json(csv_path)

        crew = HyperparamPathologyCrew()
        result = crew.crew().kickoff(inputs={"raw_summary_json": raw_summary_json})

        # Optional: print final result if you care
        print(result)

        # Do NOT return the result string, so sys.exit() sees None → exit code 0
        return 0
    except Exception as e:
        # This branch is only hit on actual failures
        raise Exception(f"An error occurred while running the crew: {e}")  


def train():
    """
    Train the crew for a given number of iterations.
    Usage (from shell):
        python -m hyperparam_pathology.main train <n_iterations> <outfile>
    """
    if len(sys.argv) < 3:
        raise Exception("Usage: train <n_iterations> <outfile>")

    csv_path = os.getenv("HPATH_RESULTS_CSV", "results.csv")
    raw_summary_json = _build_raw_summary_json(csv_path)

    try:
        crew = HyperparamPathologyCrew()
        crew.crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs={"raw_summary_json": raw_summary_json},
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    Usage:
        python -m hyperparam_pathology.main replay <task_id>
    """
    if len(sys.argv) < 2:
        raise Exception("Usage: replay <task_id>")

    try:
        crew = HyperparamPathologyCrew()
        crew.crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and return the results.
    Usage:
        python -m hyperparam_pathology.main test <n_iterations> <eval_llm>
    """
    if len(sys.argv) < 3:
        raise Exception("Usage: test <n_iterations> <eval_llm_model_name>")

    csv_path = os.getenv("HPATH_RESULTS_CSV", "results.csv")
    raw_summary_json = _build_raw_summary_json(csv_path)

    try:
        crew = HyperparamPathologyCrew()
        crew.crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs={"raw_summary_json": raw_summary_json},
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with a trigger payload.
    Expect a JSON string as the first CLI argument, optionally containing
    'csv_path'. If not provided, falls back to env/ default.
    """
    import json as _json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = _json.loads(sys.argv[1])
    except _json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    csv_path = trigger_payload.get("csv_path") or os.getenv("HPATH_RESULTS_CSV", "results.csv")
    raw_summary_json = _build_raw_summary_json(csv_path)

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "raw_summary_json": raw_summary_json,
        "current_year": str(datetime.now().year),
    }

    try:
        crew = HyperparamPathologyCrew()
        result = crew.crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
