from pathlib import Path
import sys

from mcp.server.fastmcp import FastMCP

from hyperparam_pathology.main import _build_raw_summary_json
from hyperparam_pathology.crew import HyperparamPathologyCrew

mcp = FastMCP("hyperparam-pathology-inspector")


@mcp.tool()
def analyze_hparam_csv(csv_path: str) -> str:
    """
    Analyze a hyperparameter sweep CSV and return a markdown pathology report.

    csv_path: Absolute or project-relative path to a CSV file.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    raw_summary_json = _build_raw_summary_json(str(path))

    crew = HyperparamPathologyCrew()
    result = crew.crew().kickoff(inputs={"raw_summary_json": raw_summary_json})

    # Convert CrewOutput to string and extract raw output
    result_str = result.raw if hasattr(result, 'raw') else str(result)

    phrase = "I now can give a great answer"
    if phrase in result_str:
        result_str = result_str.replace(phrase, "").lstrip()

    return result_str


if __name__ == "__main__":
    # Debug mode: `python -m hyperparam_pathology.mcp_server debug`
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        report = analyze_hparam_csv("examples/sample_results.csv")
        print(report)
    else:
        # Normal MCP server mode (used by ChatGPT / Claude / etc.)
        mcp.run()
