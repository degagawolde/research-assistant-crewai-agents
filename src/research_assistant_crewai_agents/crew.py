# src/latest_ai_development/crew.py
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import requests

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

class CustomSerperDevTool(SerperDevTool):
    def _make_api_request(self, search_query, search_type="web"):
        # Change this URL to your desired endpoint
        search_url = "https://serpapi.com/search"

        # Parameters for GET request
        params = {
            "q": search_query,
            "api_key": os.getenv("SERPER_API_KEY"),
            "engine": "google"  # example: google engine
        }

        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

@CrewBase
class LatestAiDevelopmentCrew():
    """LatestAiDevelopment crew"""     
    agents: List[BaseAgent]
    tasks: List[Task]

    def _build_llm(self, agent_name: str) -> LLM:
        """Build LLM for a specific agent from YAML + env API key"""
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        llm_conf = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "base_url": "https://api.openai.com/v1",
            "api_key": api_key  # make sure key name is a string
        }
        return LLM(**llm_conf)
    
    @agent
    def researcher(self) -> Agent:
        # Override api_key with env variable

        llm = self._build_llm("researcher")
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            llm=llm,
            tools=[SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        # Override api_key with env variable

        llm = self._build_llm("researcher")
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True,
            llm=llm,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='output/report.md' # This is the file that will be contain the final report.
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )