#date: 2024-09-17T16:55:05Z
#url: https://api.github.com/gists/18888e23fee5a7788305f1aa35a1df3b
#owner: https://api.github.com/users/zitterbewegung

#!/usr/bin/env python3

import subprocess
import os
import git
import requests
import re
import json
from langgraph import LangGraph, RAGCodeAssistant, Tool
from pygdbmi.gdbcontroller import GdbController  # GDB Python interface
from ghidra_bridge import GhidraBridge  # Ghidra Python bridge
from litellm import LiteLLMClient  # Import LiteLLM client
import vowpalwabbit
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

# Retrieve paths and tokens from environment variables
GDB_PATH = os.getenv('GDB_PATH')
GHIDRA_PATH = os.getenv('GHIDRA_PATH')
DTRACE_PATH = os.getenv('DTRACE_PATH')
LITELLM_API_KEY = os.getenv('LITELLM_API_KEY')
ADVISORY_DB_API_TOKEN = "**********"
ADVISORY_DB_URL = "https://api.github.com/repos/github/advisory-database/contents/advisories"

# Initialize LiteLLM client with the API key from the .env file
llm_client = LiteLLMClient(api_key=LITELLM_API_KEY)

class LiteLLMAgent:
    """Agent to interact with LiteLLM for generating code explanations and other prompts."""

    def __init__(self, model="gpt-3.5"):
        self.model = model
        self.client = llm_client

    def generate_response(self, prompt):
        """Generate response using LiteLLM."""
        response = self.client.complete(
            prompt=prompt,
            model=self.model,
            max_tokens= "**********"
        )
        return response.text

class CodeExplanationTool(Tool):
    """A tool that uses LiteLLM to explain the functionality of C/C++ source code."""

    def __init__(self):
        super().__init__(name="code_explanation", description="Generates explanations for source code using LiteLLM.")
        self.llm_agent = LiteLLMAgent()

    def explain_code(self, code):
        """Generate a description of what the code does using LiteLLM."""
        explanation_prompt = f"Analyze the following C/C++ code and provide a detailed explanation of what it does:\n\n{code}\n\nExplanation:"
        response = self.llm_agent.generate_response(explanation_prompt)
        return response

class AdvisoryScanTool(Tool):
    """A tool that scans a GitHub repository for known vulnerabilities based on the GitHub Advisory Database."""

    def __init__(self):
        super().__init__(name="advisory_scan", description="Scans dependencies for known vulnerabilities.")
        self.advisory_data = self.load_advisory_database()

    def load_advisory_database(self):
        """Loads advisories from the GitHub Advisory Database."""
        headers = {'Authorization': "**********"
        response = requests.get(ADVISORY_DB_URL, headers=headers)
        if response.status_code == 200:
            advisories = response.json()
            print("Advisory Database Loaded Successfully")
            return advisories
        else:
            print("Failed to load Advisory Database")
            return []

    def parse_advisory_entries(self, advisory):
        """Parse advisories to extract relevant information."""
        details = {
            "package_name": advisory.get("package_name", ""),
            "vulnerable_versions": advisory.get("vulnerable_versions", ""),
            "description": advisory.get("description", ""),
            "severity": advisory.get("severity", ""),
            "identifiers": advisory.get("identifiers", []),
        }
        return details

    def scan_repository(self, repo_path):
        """Scans the given GitHub repository path for dependencies and checks against known advisories."""
        repo = git.Repo(repo_path)
        dependencies = self.extract_dependencies(repo)
        vulnerabilities = self.match_advisories(dependencies)
        return vulnerabilities

    def extract_dependencies(self, repo):
        """Extracts dependencies from the repository (example for C/C++ projects)."""
        dependencies = []
        files = repo.git.ls_files('*.txt', '*.json', '*.yaml', '*.lock').splitlines()
        for file_path in files:
            with open(os.path.join(repo.working_dir, file_path), 'r') as file:
                content = file.read()
                dependencies.extend(self.parse_dependencies_from_file(content))
        return dependencies

    def parse_dependencies_from_file(self, content):
        """Parses dependencies from a given file content."""
        dependencies = []
        for line in content.splitlines():
            if "==" in line:
                package = line.split("==")[0].strip()
                dependencies.append(package)
        return dependencies

    def match_advisories(self, dependencies):
        """Matches dependencies against advisories from the GitHub Advisory Database."""
        matched_vulnerabilities = []
        for advisory in self.advisory_data:
            advisory_details = self.parse_advisory_entries(advisory)
            for dependency in dependencies:
                if advisory_details["package_name"].lower() == dependency.lower():
                    matched_vulnerabilities.append(advisory_details)
                    print(f"Vulnerability found for {dependency}: {advisory_details}")
        return matched_vulnerabilities

# Main LangGraph application setup
class CodeAnalysisLangGraph:
    def __init__(self):
        self.langgraph = LangGraph()
        self.vulnerability_tool = VulnerabilityDetectionTool()
        self.advisory_scan_tool = AdvisoryScanTool()
        self.code_explanation_tool = CodeExplanationTool()

    def analyze_repository(self, repo_path):
        """Main function to analyze the repository and program behavior."""
        # Initialize GitPython repository
        repo = git.Repo(repo_path)

        # Scan and explain each C/C++ source file
        source_files = repo.git.ls_files('*.c', '*.cpp').splitlines()
        for file_path in source_files:
            full_path = os.path.join(repo_path, file_path)
            with open(full_path, 'r') as file:
                source_code = file.read()
                
                # Vulnerability Detection
                vulnerabilities = self.vulnerability_tool.scan_code(source_code)
                if vulnerabilities:
                    print(f"Vulnerabilities in {file_path}:")
                    for vulnerability in vulnerabilities:
                        print("-", vulnerability)

                # Code Explanation
                explanation = self.code_explanation_tool.explain_code(source_code)
                print(f"Explanation for {file_path}:\n{explanation}\n")

        # Advisory Scan Tool
        vulnerabilities = self.advisory_scan_tool.scan_repository(repo_path)
        if vulnerabilities:
            print("Dependency Vulnerabilities Detected:")
            for vulnerability in vulnerabilities:
                print(json.dumps(vulnerability, indent=4))

def main():
    parser = argparse.ArgumentParser(description="Analyze a GitHub repository for vulnerabilities and code functionality.")
    parser.add_argument('--repo', type=str, help='Path to the GitHub repository', default=os.getenv('REPO_PATH'))
    args = parser.parse_args()

    # Run the analysis with the provided or default repository path
    analyzer = CodeAnalysisLangGraph()
    analyzer.analyze_repository(args.repo)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
