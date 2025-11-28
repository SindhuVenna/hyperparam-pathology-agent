# src/hyperparam_pathology/crew.py

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class HyperparamPathologyCrew:
    """Hyperparameter Pathology Inspector crew"""

    # Populated automatically by @CrewBase using config/agents.yaml & tasks.yaml
    agents: List[BaseAgent]
    tasks: List[Task]

    # ---------- Agents ----------

    @agent
    def pathology_analyzer(self) -> Agent:
        """Agent that reasons about raw pathology summary JSON."""
        return Agent(
            config=self.agents_config["pathology_analyzer"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def pattern_analyst(self) -> Agent:
        """Agent that turns correlations into readable patterns."""
        return Agent(
            config=self.agents_config["pattern_analyst"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def report_writer(self) -> Agent:
        """Agent that writes the final markdown report."""
        return Agent(
            config=self.agents_config["report_writer"],  # type: ignore[index]
            verbose=True,
        )

    # ---------- Tasks ----------

    @task
    def analyze_pathologies_task(self) -> Task:
        """LLM task: read raw_summary_json and emit patterns JSON."""
        return Task(
            config=self.tasks_config["analyze_pathologies_task"],  # type: ignore[index]
        )

    @task
    def write_report_task(self) -> Task:
        """LLM task: consume raw_summary_json + pattern_json and write markdown."""
        return Task(
            config=self.tasks_config["write_report_task"],  # type: ignore[index]
        )

    # ---------- Crew ----------

    @crew
    def crew(self) -> Crew:
        """Creates the HyperparamPathology crew."""
        return Crew(
            agents=self.agents,  # filled from @agent methods
            tasks=self.tasks,    # filled from @task methods
            process=Process.sequential,
            verbose=True,
        )
