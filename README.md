# Hyperparameter Pathology Inspector 🧪

> A small but opinionated agentic tool that inspects hyperparameter sweeps, finds “sick” trials, and explains **why** your experiments misbehaved.

This project takes a **CSV of hyperparameter sweep results**, runs a set of **detectors** to find pathologies (NaNs, divergence, overfitting, short runs, etc.), and then uses a **CrewAI agent team** to:

- summarize what went wrong,
- identify patterns like “high LR → NaNs” or “no weight decay → overfitting”,
- generate a clean **markdown report** with **actionable recommendations**.

It’s a compact, systems-flavoured project that shows you think about **ML experiments like an engineer**, not just a model user.

---

## Features

### Core analysis (pure Python)

- Detects trial-level issues:
  - NaN / Inf metrics
  - Failed / crashed runs (via `status`)
  - Overfitting suspects (`val_loss / train_loss` ratio)
  - Suspiciously short runs (epochs / runtime quantiles)
- Computes **hyperparameter–issue correlations**:
  - For each hyperparameter, estimates issue rates by:
    - quantile buckets (numeric),
    - category values (categorical).

### Agentic layer (CrewAI + LLM)

- **Pattern Analyst Agent**

  Reads the structured summary and turns correlations into human-readable patterns, e.g.:

  > “High learning rates (> 0.01) consistently lead to NaN loss.”

- **Experiment Review Writer Agent**

  Produces a markdown report:

  - Overview of the sweep health  
  - Issue breakdown by type  
  - Hyperparameter pathologies (with evidence)  
  - Concrete recommendations (what to try next)

### Output

- A single file: `hparam_report.md`, e.g.:

  ```markdown
  # Hyperparameter Sweep Pathology Report
  ## Overview
  - This sweep had a total of N issues, with M trials experiencing at least one issue.
  ...
  ```

---

## Project Structure

```text
hyperparam_pathology/
├─ README.md
├─ pyproject.toml
├─ .gitignore
├─ .env                  # not committed; holds HF_TOKEN, etc.
├─ examples/
│  └─ sample_results.csv
└─ src/
   └─ hyperparam_pathology/
      ├─ main.py                # Entry point for running the crew
      ├─ crew.py                # CrewAI crew definition (agents + tasks)
      ├─ mcp_server.py          # MCP server exposing analyze_hparam_csv tool
      ├─ core/
      │  ├─ detectors.py        # Pure Python detectors (no LLM)
      │  └─ summarizer.py       # Build structured JSON summary
      └─ config/
         ├─ agents.yaml         # Agent definitions (roles, goals, backstories)
         └─ tasks.yaml          # Task definitions & workflow
```

---

## Requirements

- Python **3.10+** (tested with 3.12)
- [CrewAI](https://docs.crewai.com/)
- [uv](https://github.com/astral-sh/uv) (CrewAI uses it under the hood)
- A Hugging Face account + Inference API token
- (Optional) [MCP Python SDK](https://pypi.org/project/mcp/) for MCP server support

Main Python dependencies (managed via `pyproject.toml`):

- `crewai`
- `litellm`
- `pandas`
- `numpy`
- `mcp` (for the MCP server)

---

## Setup

From the project root:

```bash
cd hyperparam_pathology

# Install dependencies via uv (recommended)
uv sync
```

### Configure the LLM (Hugging Face Inference)

1. Go to Hugging Face → **Settings → Access Tokens**
2. Create a token with `read`/inference permissions.
3. In `.env`, set:

   ```env
   HF_TOKEN=hf_your_token_here

   # LLM model served via Hugging Face router
   MODEL=meta-llama/Llama-3.1-8B-Instruct
   ```

CrewAI + LiteLLM will use `HF_TOKEN` and `MODEL` to talk to the HF router at  
`https://router.huggingface.co/v1/chat/completions`.

---

## Input: CSV Format

The tool expects a CSV with **one row per trial**.

Recommended columns (you can extend this later):

```csv
trial_id,status,lr,batch_size,weight_decay,train_loss,val_loss,train_acc,val_acc,epochs,runtime_sec
1,completed,0.001,64,0.0001,0.35,0.42,0.89,0.86,20,320
2,failed,0.01,64,0.0001,NaN,NaN,NaN,NaN,3,45
3,completed,0.0001,16,0.0,0.15,0.80,0.99,0.60,40,600
4,completed,0.02,32,0.0001,0.50,inf,0.80,0.70,10,200
5,completed,0.005,8,0.0001,0.20,0.25,0.95,0.93,5,100
6,completed,0.0005,128,0.001,0.30,0.32,0.90,0.88,25,350
```

Special columns:

- `trial_id` – unique trial identifier
- `status` – e.g. `completed`, `failed`, etc.
- `train_loss`, `val_loss` – used for:
  - NaN / Inf checks
  - overfitting detection (`val_loss / train_loss` ratio)
- `epochs` or `runtime_sec` – used to detect short runs

All other columns (e.g. `lr`, `batch_size`, `weight_decay`, etc.) are treated as **hyperparameters**.

If you follow this schema, you can just drop your sweep CSV into `examples/` or project root.

---

## Running the Crew (CLI)

### 1. Point to your CSV

Either:

- Rename your file to `results.csv` in the project root, **or**
- Set an env variable pointing to your file:

```bash
export HPATH_RESULTS_CSV=examples/sample_results.csv
```

### 2. Run the agent

From the project root:

```bash
crewai run
```

What happens:

1. `main.py`:
   - loads the CSV via `pandas`,
   - runs detectors in `core/detectors.py`,
   - builds a structured summary via `core/summarizer.py`.
2. `HyperparamPathologyCrew` kicks off:
   - `analyze_pathologies_task` → Pattern Analyst agent
   - `write_report_task` → Experiment Review Writer agent
3. A cleaned markdown report is written to:

```text
hparam_report.md
```

You should see logs like:

```text
Crew: crew
├── Task: analyze_pathologies_task ... ✅ Completed
└── Task: write_report_task ... ✅ Completed
[INFO] Cleaned report written to hparam_report.md
```

---

## MCP Server (optional)

This project also exposes an **MCP server** so MCP-aware tools (ChatGPT desktop, Claude Desktop, Cursor, etc.) can call it as a tool.

The server is implemented in:

```text
src/hyperparam_pathology/mcp_server.py
```

It provides one tool:

- `analyze_hparam_csv(csv_path: str) -> str`  
  → runs the same detector + CrewAI pipeline and returns the markdown report.

### Install MCP dependency

If you haven’t already:

```bash
uv add "mcp[cli]"
# or
pip install "mcp[cli]"
```

### Local debug mode (no MCP client)

You can test the tool locally without any MCP client:

```bash
cd hyperparam_pathology
source .venv/bin/activate

python -m hyperparam_pathology.mcp_server debug
```

This will analyze `examples/sample_results.csv` and print the markdown report.

### Running as an MCP server

To run in real MCP mode (for use by an MCP client):

```bash
python -m hyperparam_pathology.mcp_server
```

The process will then speak MCP JSON-RPC over stdio. A client like Cursor, Claude Desktop, or a VS Code MCP extension can connect to it by configuring:

- **Command:**

  ```text
  /path/to/your/project/.venv/bin/python
  ```

- **Args:**

  ```text
  -m
  hyperparam_pathology.mcp_server
  ```

- **Working directory:**

  ```text
  /path/to/your/project
  ```

Then the client can call:

```json
{
  "tool": "analyze_hparam_csv",
  "arguments": {
    "csv_path": "/absolute/or/relative/path/to/your_sweep.csv"
  }
}
```

and receive the markdown report as the tool result.

---

## What the Report Contains

`hparam_report.md` (or the MCP tool result) includes:

- **Overview**
  - total issues, how many trials affected, severity distribution
- **Issue Breakdown**
  - table of issue types → counts → example trial_ids
- **Hyperparameter Pathologies**
  - each key pattern with:
    - title, explanation, evidence (buckets/values + trial_ids)
- **Recommendations**
  - concrete next steps like:
    - “Avoid lr > 0.01”
    - “Use non-zero weight decay”
    - “Increase batch size above 16”
    - “Monitor NaN loss and add gradient clipping”

---

## Extending the Tool

You can easily evolve this into a richer experiment-intelligence tool:

- **Detectors**
  - Add:
    - accuracy-based overfitting rules,
    - unstable metric detectors (high variance),
    - calibration / class-imbalance checks.

- **Hyperparameters**
  - Treat any new columns as hyperparams; `detect_param_correlations` dynamically adapts.

- **Agents**
  - Add a “Next Experiment Designer” agent:
    - suggest concrete new sweeps based on detected pathologies.

- **UI**
  - Wrap it in a small Streamlit/Gradio app:
    - upload CSV → see report & plots.

---

## Why this project is interesting

- It touches **real ML systems concerns**:
  - debugging experiments,
  - triaging failures,
  - reasoning about hyperparameter ranges.
- It combines:
  - **plain Python stats + heuristics** (detectors) with  
  - **LLM-based reasoning & explanation** (CrewAI agents).
- The core analysis (`core/`) is framework-agnostic and reusable in:
  - scripts,
  - notebooks,
  - other agent frameworks (e.g. LangGraph),
  - MCP servers.

---

## License

MIT (or any license you prefer).

---

## Roadmap

- [ ] Add visualization of hyperparameter–issue heatmaps
- [ ] Support multiple metric pairs (loss, accuracy, custom metrics)
- [ ] Export JSON summary alongside markdown
- [ ] Optional: integrate with Weights & Biases / MLflow logs
- [ ] Expose more tools through MCP (e.g., next-experiment suggestions)
