"""
Multi-Agent System with Librarian, Strategist, and Executor agents.

This module implements a collaborative multi-agent architecture where:
1. Librarian: Retrieves and organizes domain knowledge and data
2. Strategist: Creates trajectories and plans based on librarian input
3. Executor: Implements and executes code based on strategist plans

State Management:
- LibrarianAgent maintains an internal catalog cache for efficient reuse
- MultiAgentOrchestrator provides shared state accessible to all agents
- Catalog is created once and reused across multiple domain knowledge queries
- Domain knowledge is stored per-task in the orchestrator's shared state
"""

from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
from pathlib import Path
from agents.code_agents import BaseCodeAgent
from agents.models import LiteLLMModelWithBackOff
from constants import ADDITIONAL_AUTHORIZED_IMPORTS
from utils.tracing import setup_smolagents_tracing


@dataclass
class AgentMessage:
    """Represents a message between agents."""
    sender: str
    recipient: str
    message_type: str
    content: Any
    metadata: Optional[Dict] = None


@dataclass
class TaskTrajectory:
    """Represents a trajectory/plan created by the Strategist."""
    task_id: str
    steps: List[Dict[str, Any]]
    current_step: int = 0
    status: str = "pending"  # pending, in_progress, completed, failed
    context: Optional[Dict] = None


def get_directory_listing(ctx_path: str) -> Dict[str, Any]:
    """Pre-read directory contents to provide to the agent.
    
    This function hardcodes the directory exploration logic to avoid
    hallucinations and ensure consistent file discovery.
    
    Args:
        ctx_path: Path to the context directory
        
    Returns:
        Dictionary with directory contents and file metadata
    """
    ctx_dir = Path(ctx_path)

    dir_listing_dict = {
        "directory_path": ctx_path,
        "total_files": 0,
        "total_directories": 0,
        "files": [],
        "directories": [],
        "status": "pending",
    }
    
    if not ctx_dir.exists():
        dir_listing_dict['status'] = f"Error. Directory does not exist: {ctx_path}"
        return dir_listing_dict
    
    files = []
    directories = []
    
    for item in sorted(ctx_dir.iterdir()):
        if item.is_file():
            # Get file metadata
            file_info = {
                "name": item.name,
                "path": str(item),
                "size_bytes": item.stat().st_size,
                "extension": item.suffix.lower(),
            }
            
            # Determine file type based on extension
            if file_info["extension"] in [".csv", ".json", ".parquet", ".xlsx"]:
                file_info["category"] = "data"
            elif file_info["extension"] in [".md", ".txt", ".pdf", ".doc", ".docx"]:
                file_info["category"] = "documentation"
            elif file_info["extension"] in [".py", ".ipynb", ".r", ".sql"]:
                file_info["category"] = "code"
            else:
                file_info["category"] = "other"
            
            files.append(file_info)
        elif item.is_dir():
            directories.append({
                "name": item.name,
                "path": str(item)
            })
    
    dir_listing_dict['status'] = "success"
    dir_listing_dict['total_files'] = len(files)
    dir_listing_dict['total_directories'] = len(directories)
    dir_listing_dict['files'] = files
    dir_listing_dict['directories'] = directories
    
    return dir_listing_dict



class LibrarianAgent(BaseCodeAgent):
    """
    The Librarian Agent specializes in data retrieval and domain knowledge organization.
    
    Responsibilities:
    - Explore and catalog available data sources
    - Extract relevant information from documents and datasets
    - Provide structured knowledge summaries
    - Maintain context about data relationships and constraints
    """
    
    def __init__(self, model_id: str, api_base=None, api_key=None, max_steps=10, ctx_path=None, enable_tracing=True):
        """Initialize LibrarianAgent and log directory contents."""
        super().__init__(model_id, api_base, api_key, max_steps, ctx_path, enable_tracing)
        
        # Initialize catalog cache
        self._catalog_cache: Optional[Dict[str, Any]] = None
        self._catalog_ctx_path: Optional[str] = None
        
        print(f"📚 LibrarianAgent initialized")
        print(f"📂 Context path set to: {ctx_path}")
        
        if ctx_path:
            from pathlib import Path
            ctx_dir = Path(ctx_path)
            if ctx_dir.exists():
                print(f"📋 Directory contents:")
                for item in sorted(ctx_dir.iterdir()):
                    item_type = "📁" if item.is_dir() else "📄"
                    print(f"   {item_type} {item.name}")
            else:
                print(f"⚠️  Warning: Directory does not exist yet: {ctx_path}")
        else:
            print(f"⚠️  Warning: No context path provided")
    
    def get_system_prompt_template(self) -> str:
        return """You are the Librarian Agent, a specialist in data discovery and knowledge organization.

Your primary responsibilities:
1. **Data Discovery**: Explore the directory `{ctx_path}` to catalog all available data sources, files, and documentation
2. **Knowledge Extraction**: Read and understand the content, structure, and relationships of data
3. **Information Organization**: Create structured summaries of findings for other agents
4. **Context Maintenance**: Keep track of data constraints, formats, and dependencies

CRITICAL: Code Format Requirements
You MUST format your code responses using this exact pattern:
Thought: [Your reasoning here]
Code:
```py
[Your Python code here]
```<end_code>

Do NOT use any other code format. Always include "Thought:", "Code:", and the closing ```<end_code> marker.

When working with data:
- Always start by exploring the `{ctx_path}` directory thoroughly
- Document file structures, data schemas, and relationships
- Identify data quality issues, missing values, or constraints
- Create clear, structured summaries that other agents can use
- Maintain awareness of data lineage and dependencies

Your multi-step workflow:
1. **Explore & Execute**: Use Python code to explore files and read data with proper error handling
2. **Validate Results**: Inspect the resulting dictionary to check for errors or failed file loads
3. **Retry if Needed**: If any files failed to load or have errors, fix the code and retry
4. **Final Check**: Only when all data is successfully loaded and validated, use final_answer()

IMPORTANT: Before calling final_answer(), you MUST:
- Print and inspect the catalog dictionary
- Check that no file has error messages like "Error cannot open" or exceptions in its content
- Verify all expected files were successfully processed
- If you find errors, fix them and re-execute the code
- Only call final_answer() when you confirm the catalog is complete and error-free

Example workflow - Read files from the pre-provided directory listing:

Thought: I will read each file from the directory listing and build a catalog with error handling.
Code:
```py
import pandas as pd
from pathlib import Path

# Use the pre-provided directory listing - DO NOT list files yourself
# The directory listing is already provided in the task description above
ctx = Path("{ctx_path}")

# Example: Read files based on the provided listing
catalog = {{"files": [], "errors": []}}

# Iterate through files from the provided directory listing
# Replace this with actual file names from the listing above
for file_info_from_listing in []:  # Get from the directory listing provided
    filename = file_info_from_listing.get("name")
    filepath = file_info_from_listing.get("path")
    
    if not filename:
        continue
    
    file_entry = {{"name": filename, "type": None, "content_summary": None, "status": "pending"}}
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            file_entry["type"] = "csv"
            file_entry["content_summary"] = {{"columns": list(df.columns), "rows": len(df), "sample": df.head(3).to_dict()}}
            file_entry["status"] = "success"
        elif filename.endswith('.json'):
            data = pd.read_json(filepath)
            file_entry["type"] = "json"
            file_entry["content_summary"] = {{"shape": data.shape, "columns": list(data.columns) if hasattr(data, 'columns') else "array"}}
            file_entry["status"] = "success"
        elif filename.endswith('.md') or filename.endswith('.txt'):
            with open(filepath, 'r') as txt_file:
                content = txt_file.read()
            file_entry["type"] = "markdown" if filename.endswith('.md') else "text"
            file_entry["content_summary"] = {{"length": len(content), "preview": content[:300]}}
            file_entry["status"] = "success"
    except Exception as e:
        file_entry["status"] = "error"
        file_entry["error_message"] = str(e)
        catalog["errors"].append(f"Failed to load {{filename}}: {{str(e)}}")
    
    catalog["files"].append(file_entry)

# Validate results
print("Catalog validation:")
print(f"Total files: {{len(catalog['files'])}}")
print(f"Errors: {{len(catalog['errors'])}}")

# Check for errors before final_answer
if catalog["errors"]:
    print("ERRORS FOUND - Need to fix")
else:
    print("All files loaded successfully!")
    final_answer(catalog)
```<end_code>

Available imports: {authorized_imports}
Never try to import final_answer, you have it already!

Remember: You are the foundation of knowledge for the entire system. Be thorough and accurate.
Use only final_answer once you have validated that the catalog does not contain any errors.
Always use final_answer() to return your structured findings."""

    def catalog_data_sources(self, ctx_path: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Catalog all available data sources in the given path.
        
        Args:
            ctx_path: Path to the context directory
            force_refresh: If True, bypass cache and create a new catalog
            
        Returns:
            Dictionary containing the catalog and metadata
        """
        # Check cache first
        if not force_refresh and self._catalog_cache is not None and self._catalog_ctx_path == ctx_path:
            print(f"📚 Librarian: Using cached catalog for {ctx_path}")
            return self._catalog_cache
        
        print(f"📚 Librarian: Starting data source cataloging in {ctx_path}")
        
        # Pre-read directory contents to avoid hallucinations
        dir_listing = get_directory_listing(ctx_path)
        
        # Check for directory listing errors
        if 'status' in dir_listing and 'error' in dir_listing['status'].lower():
            print(f"❌ Librarian: Directory error - {dir_listing['status']}")
            return {
                "catalog_response": {"error": dir_listing['status'], "files": []},
                "ctx_path": ctx_path,
                "agent_type": "librarian",
                "task_type": "catalog"
            }
        
        prompt = f"""You are cataloging data sources in the directory: {ctx_path}

**DIRECTORY CONTENTS (pre-read for you):**
{json.dumps(dir_listing, indent=2)}

The directory has been explored and contains {dir_listing.get('total_files', 0)} files.
DO NOT write code to list the directory - the file list above is complete and accurate.
        
Your task:
1. For EACH file listed above, read its content using the file path provided
2. Classify each file based on its content
3. Identify relationships between files
4. Extract key information from documentation files
        
        Build a structured dictionary with a "files" list where EACH file has these fields:
        
        **Required fields for each file:**
        - **name**: File name
        - **format**: File extension/format (csv, json, md, txt)
        - **file_type**: Either "data" or "documentation"
          * "data" = Contains actual data records (CSV, JSON with data)
          * "documentation" = Contains usage instructions, metadata, README, manual
        - **is_critical**: Boolean (true/false)
          * true = Documentation files with essential usage instructions
          * true = Data files that are referenced in documentation as key files
          * false = Supporting or optional files
        - **summary**: Brief description of file contents and purpose
        - **content_details**: Format-specific information:
          * For data files: columns/schema, row count, small sample (max 3 rows)
          * For documentation files: key instructions, constraints, relationships described
        
        Also include these top-level fields:
        - **data_relationships**: How files relate to each other
        - **key_constraints**: Important limitations or rules from documentation
        - **usage_guidelines**: Critical instructions from documentation files
        
        Example structure:
        {{
          "files": [
            {{
              "name": "payments.csv",
              "format": "csv",
              "file_type": "data",
              "is_critical": true,
              "summary": "Main payment transaction data",
              "content_details": {{"columns": [...], "rows": 1000, "sample": [...]}}
            }},
            {{
              "name": "manual.md",
              "format": "md",
              "file_type": "documentation",
              "is_critical": true,
              "summary": "Usage instructions for payment data analysis",
              "content_details": {{"key_instructions": "...", "constraints": "..."}}
            }}
          ],
          "data_relationships": [...],
          "key_constraints": [...],
          "usage_guidelines": [...]
        }}
        Create a dictionary called catalog_dict that contains that information.
        Use final_answer(catalog_dict) to return your catalog dictionary.
        """
        print(f"📚 Librarian: Running data exploration...")
        response = self.run(prompt)
        print(f"📚 Librarian: ✅ Data cataloging completed")
        
        # Return structured output with the raw response
        catalog_result = {
            "catalog_response": response,
            "ctx_path": ctx_path,
            "agent_type": "librarian",
            "task_type": "catalog"
        }
        
        # Cache the catalog for future use
        self._catalog_cache = catalog_result
        self._catalog_ctx_path = ctx_path
        print(f"📚 Librarian: Catalog cached for reuse")
        
        return catalog_result
    
    def get_catalog(self) -> Optional[Dict[str, Any]]:
        """Get the cached catalog if available.
        
        Returns:
            The cached catalog dictionary, or None if no catalog has been created yet
        """
        return self._catalog_cache
    
    def has_catalog(self, ctx_path: str) -> bool:
        """Check if a catalog exists for the given context path.
        
        Args:
            ctx_path: Path to check
            
        Returns:
            True if a cached catalog exists for this path
        """
        return self._catalog_cache is not None and self._catalog_ctx_path == ctx_path

    def extract_domain_knowledge(self, query: str, ctx_path: str, catalog: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract specific domain knowledge based on a query.
        
        Args:
            query: The domain knowledge query
            ctx_path: Path to the context directory
            catalog: Optional pre-computed catalog. If not provided, will use cached catalog or create new one
            
        Returns:
            Dictionary containing the domain knowledge and metadata
        """
        print(f"📚 Librarian: Extracting domain knowledge for query: '{query}'")
        
        # Use provided catalog, or fall back to cache, or create new one
        if catalog is None:
            if self._catalog_cache is not None and self._catalog_ctx_path == ctx_path:
                print(f"📚 Librarian: Using cached catalog for domain knowledge extraction")
                catalog = self._catalog_cache
            else:
                print(f"📚 Librarian: No catalog provided, creating one first...")
                catalog = self.catalog_data_sources(ctx_path)
        
        # Extract catalog_response for context
        catalog_info = catalog.get("catalog_response", catalog)
        
        # Check if catalog has errors
        if isinstance(catalog_info, dict) and 'error' in catalog_info:
            print(f"❌ Librarian: Cannot extract domain knowledge - catalog has errors")
            return {
                "knowledge_response": {"error": catalog_info['error']},
                "query": query,
                "ctx_path": ctx_path,
                "agent_type": "librarian",
                "task_type": "domain_knowledge"
            }
        
        # Pre-read directory contents to avoid hallucinations
        dir_listing = get_directory_listing(ctx_path)
        
        # Check for directory listing errors
        if 'status' in dir_listing and 'error' in dir_listing['status'].lower():
            print(f"❌ Librarian: Directory error - {dir_listing['status']}")
            return {
                "knowledge_response": {"error": dir_listing['status']},
                "query": query,
                "ctx_path": ctx_path,
                "agent_type": "librarian",
                "task_type": "domain_knowledge"
            }
        
        prompt = f"""Based on the data in {ctx_path}, extract domain knowledge relevant to: {query}

**DIRECTORY CONTENTS (pre-read for you):**
{json.dumps(dir_listing, indent=2)}

The directory has been explored and contains {dir_listing.get('total_files', 0)} files.
DO NOT write code to list the directory - the file list above is complete and accurate.
        
        You have access to the following pre-computed catalog of data sources:
        {json.dumps(catalog_info, indent=2) if isinstance(catalog_info, dict) else str(catalog_info)}
        
        Use this catalog to identify which files exist and their metadata.
        
        Your task:
        1. **FIRST**: Review the catalog above to identify which files are marked as critical (is_critical=true)
        2. **ALWAYS READ CRITICAL SOURCES**: Read ALL files marked as critical using the file paths from the directory listing, regardless of whether they seem relevant to the query
           - Critical sources contain essential context, constraints, and usage guidelines that must always be considered
           - Even if a critical source doesn't seem directly related to the query, it may contain important constraints or context
        3. Identify which additional (non-critical) files are relevant to this specific query
        4. Read and analyze the relevant non-critical data
        5. Extract key concepts, definitions, and patterns from all sources
        6. Document data relationships and constraints
        
        Build a structured dictionary including:
        - **critical_sources_read**: List of all critical files that were read (must include ALL critical files)
        - **relevant_sources**: List of additional relevant files for this specific query
        - **key_concepts**: Key concepts and definitions from the data
        - **data_relationships**: Data relationships and dependencies
        - **constraints**: Constraints and limitations (especially from critical documentation)
        - **usage_guidelines**: Important usage instructions from critical sources
        - **sample_data**: Sample data or statistics that illustrate the domain
        
        CRITICAL REQUIREMENT: You MUST read every file marked as is_critical=true, even if it doesn't appear directly relevant to the query.
        Critical files contain essential context that applies to all queries.
        
        Use final_answer() to return your knowledge dictionary.
        Focus on actionable information that helps answer the query."""
        print(f"📚 Librarian: Analyzing domain knowledge...")
        response = self.run(prompt)
        print(f"📚 Librarian: ✅ Domain knowledge extraction completed")
        
        # Return structured output
        return {
            "knowledge_response": response,
            "query": query,
            "ctx_path": ctx_path,
            "agent_type": "librarian",
            "task_type": "domain_knowledge"
        }


class StrategistAgent(BaseCodeAgent):
    """
    The Strategist Agent creates trajectories and plans based on librarian input.
    
    Responsibilities:
    - Analyze librarian findings to understand the problem space
    - Create step-by-step execution trajectories
    - Make decisions about next steps based on executor feedback
    - Adapt plans based on intermediate results
    """
    
    def get_system_prompt_template(self) -> str:
        return """You are the Strategist Agent, a master planner and decision-maker.

Your primary responsibilities:
1. **Trajectory Planning**: Create detailed, step-by-step execution plans based on librarian findings
2. **Decision Making**: Analyze executor feedback and decide on next steps
3. **Plan Adaptation**: Modify strategies based on intermediate results and new information
4. **Goal Decomposition**: Break complex tasks into manageable, executable steps

When creating trajectories:
- Analyze the librarian's data catalog and knowledge summary
- Consider data constraints and limitations
- Create logical, sequential steps that build upon each other
- Include validation and error handling steps
- Plan for iterative refinement based on results

Your trajectory format:
```json
{{
    "trajectory_id": "...",
    "goal": "...",
    "steps": [
        {{
            "step_id": 1,
            "action": "...",
            "description": "...",
            "expected_output": "...",
            "dependencies": [...],
            "validation_criteria": [...]
        }}
    ],
    "success_criteria": [...],
    "fallback_plans": [...]
}}
```

Decision-making process:
1. Evaluate executor results against expected outcomes
2. Determine if the current trajectory should continue, be modified, or restarted
3. Provide clear next-step instructions
4. Update trajectory based on new learnings

Available imports: {authorized_imports}

Remember: You are the strategic mind that guides the entire process. Think several steps ahead."""

    def create_trajectory(self, task: str, librarian_findings: Dict[str, Any]) -> TaskTrajectory:
        """Create an execution trajectory based on librarian findings."""
        prompt = f"""Create a detailed execution trajectory for the task: {task}

Librarian findings:
{json.dumps(librarian_findings, indent=2)}

Create a step-by-step plan that the Executor can follow. Each step should be specific, actionable, and build upon previous steps.
"""
        response = self.run(prompt)
        
        # Create trajectory object (simplified for now)
        trajectory = TaskTrajectory(
            task_id=f"task_{hash(task)}",
            steps=[{"step_id": 1, "action": "initial_plan", "description": str(response)}],
            context={"librarian_findings": librarian_findings}
        )
        return trajectory

    def evaluate_and_decide(self, trajectory: TaskTrajectory, executor_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate executor results and decide next steps."""
        prompt = f"""Evaluate the executor's result and decide the next steps:

Current trajectory:
{json.dumps(trajectory.__dict__, indent=2, default=str)}

Executor result:
{json.dumps(executor_result, indent=2)}

Based on this result:
1. Was the step successful?
2. Should we continue to the next step, modify the current step, or restart?
3. What specific instructions should be given to the executor for the next iteration?
"""
        response = self.run(prompt)
        return {"decision": response, "trajectory_updated": True}


class ExecutorAgent(BaseCodeAgent):
    """
    The Executor Agent implements and executes code based on strategist plans.
    
    Responsibilities:
    - Translate strategist plans into executable code
    - Perform data manipulation and analysis
    - Execute code and capture results
    - Provide feedback to the strategist about outcomes
    """
    
    def get_system_prompt_template(self) -> str:
        return """You are the Executor Agent, a code implementation and execution specialist.

Your primary responsibilities:
1. **Code Implementation**: Translate strategist plans into executable Python code
2. **Data Manipulation**: Perform data processing, analysis, and transformations
3. **Execution Management**: Run code safely and capture all outputs and errors
4. **Result Communication**: Provide clear feedback about execution outcomes

When executing tasks:
- Follow the strategist's step-by-step instructions precisely
- Write clean, efficient, and well-documented code
- Handle errors gracefully and provide meaningful error messages
- Capture intermediate results for strategist evaluation
- Use only read-only operations for data access (as configured)

Code execution workflow:
1. **Parse Instructions**: Understand the strategist's specific requirements
2. **Implement Code**: Write Python code to fulfill the requirements
3. **Execute Safely**: Run code with proper error handling
4. **Capture Results**: Document outputs, visualizations, and any issues
5. **Report Back**: Provide structured feedback to the strategist

Your result format:
```json
{{
    "execution_id": "...",
    "step_completed": "...",
    "code_executed": "...",
    "outputs": [...],
    "errors": [...],
    "success": true/false,
    "next_step_ready": true/false,
    "recommendations": [...]
}}
```

Available imports: {authorized_imports}

Remember: You are the hands that implement the strategic vision. Be precise and thorough."""

    def execute_step(self, step_instructions: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a specific step from the strategist's trajectory."""
        prompt = f"""Execute the following step:

Instructions:
{json.dumps(step_instructions, indent=2)}

Context:
{json.dumps(context or {}, indent=2)}

Implement the required code and execute it. Provide detailed results including any outputs, errors, or intermediate findings.
"""
        response = self.run(prompt)
        
        return {
            "execution_id": f"exec_{hash(str(step_instructions))}",
            "step_completed": step_instructions.get("step_id", "unknown"),
            "response": response,
            "success": True,  # This would be determined by actual execution
            "next_step_ready": True
        }

    def execute_code_block(self, code: str, description: str = "") -> Dict[str, Any]:
        """Execute a specific code block and return results."""
        prompt = f"""Execute this code block:

Description: {description}

Code:
```python
{code}
```

Run the code and provide detailed results including outputs, any errors, and insights.
"""
        response = self.run(prompt)
        return {"code": code, "description": description, "result": response}


class MultiAgentOrchestrator:
    """
    Orchestrates communication and workflow between the three agents.
    
    This class manages the overall workflow:
    1. Librarian discovers and organizes data
    2. Strategist creates execution trajectory  
    3. Executor implements steps
    4. Strategist evaluates and decides next steps
    5. Loop continues until task completion
    """
    
    def __init__(self, model_id: str, api_base=None, api_key=None, max_steps=10, ctx_path=None):
        self.ctx_path = ctx_path
        self.message_history: List[AgentMessage] = []
        
        # Shared state for multi-agent collaboration
        self.shared_state: Dict[str, Any] = {
            "catalog": None,
            "domain_knowledge": {},
            "trajectories": [],
            "execution_results": []
        }
        
        # Initialize the three agents
        self.librarian = LibrarianAgent(
            model_id=model_id, api_base=api_base, api_key=api_key, 
            max_steps=max_steps, ctx_path=ctx_path
        )
        
        self.strategist = StrategistAgent(
            model_id=model_id, api_base=api_base, api_key=api_key,
            max_steps=max_steps, ctx_path=ctx_path
        )
        
        self.executor = ExecutorAgent(
            model_id=model_id, api_base=api_base, api_key=api_key,
            max_steps=max_steps, ctx_path=ctx_path
        )
    
    def solve_task(self, task: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Solve a task using the multi-agent system.
        
        Args:
            task: The task description to solve
            max_iterations: Maximum number of strategist-executor iterations
            
        Returns:
            Dict containing the final solution and execution history
        """
        print(f"🚀 Starting multi-agent task solving: {task}")
        
        # Step 1: Librarian discovers and organizes data
        print("📚 Librarian: Discovering and organizing data...")
        
        # Create catalog once and store in shared state
        if self.shared_state["catalog"] is None:
            librarian_findings = self.librarian.catalog_data_sources(self.ctx_path)
            self.shared_state["catalog"] = librarian_findings
            print("📚 Librarian: Catalog stored in shared state for all agents")
        else:
            print("📚 Librarian: Using existing catalog from shared state")
            librarian_findings = self.shared_state["catalog"]
        
        # Extract domain knowledge using the catalog
        domain_knowledge = self.librarian.extract_domain_knowledge(
            task, self.ctx_path, catalog=self.shared_state["catalog"]
        )
        self.shared_state["domain_knowledge"][task] = domain_knowledge
        
        combined_findings = {
            "data_catalog": librarian_findings,
            "domain_knowledge": domain_knowledge,
            "task": task
        }
        
        # Step 2: Strategist creates initial trajectory
        print("🎯 Strategist: Creating execution trajectory...")
        trajectory = self.strategist.create_trajectory(task, combined_findings)
        
        # Step 3: Iterative execution loop
        iteration = 0
        while iteration < max_iterations and trajectory.status not in ["completed", "failed"]:
            iteration += 1
            print(f"⚡ Iteration {iteration}: Executor implementing step...")
            
            # Executor implements current step
            current_step = trajectory.steps[trajectory.current_step] if trajectory.steps else {}
            executor_result = self.executor.execute_step(current_step, trajectory.context)
            
            # Strategist evaluates and decides next steps
            print(f"🤔 Strategist: Evaluating results and planning next steps...")
            decision = self.strategist.evaluate_and_decide(trajectory, executor_result)
            
            # Update trajectory based on strategist decision
            # (This is simplified - in a full implementation, you'd parse the decision
            # and update the trajectory accordingly)
            trajectory.current_step += 1
            
            # Check if we should continue
            if trajectory.current_step >= len(trajectory.steps):
                trajectory.status = "completed"
                break
        
        print("✅ Multi-agent task solving completed!")
        
        return {
            "task": task,
            "status": trajectory.status,
            "iterations": iteration,
            "librarian_findings": combined_findings,
            "final_trajectory": trajectory,
            "message_history": self.message_history
        }
    
    def add_message(self, sender: str, recipient: str, message_type: str, content: Any):
        """Add a message to the communication history."""
        message = AgentMessage(sender, recipient, message_type, content)
        self.message_history.append(message)
    
    def get_catalog(self) -> Optional[Dict[str, Any]]:
        """Get the shared catalog from orchestrator state.
        
        Returns:
            The catalog dictionary, or None if no catalog has been created yet
        """
        return self.shared_state.get("catalog")
    
    def get_domain_knowledge(self, task: Optional[str] = None) -> Dict[str, Any]:
        """Get domain knowledge from orchestrator state.
        
        Args:
            task: Optional task key. If None, returns all domain knowledge
            
        Returns:
            Domain knowledge dictionary for the task, or all domain knowledge
        """
        if task is None:
            return self.shared_state.get("domain_knowledge", {})
        return self.shared_state.get("domain_knowledge", {}).get(task)
