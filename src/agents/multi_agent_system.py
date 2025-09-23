"""
Multi-Agent System with Librarian, Strategist, and Executor agents.

This module implements a collaborative multi-agent architecture where:
1. Librarian: Retrieves and organizes domain knowledge and data
2. Strategist: Creates trajectories and plans based on librarian input
3. Executor: Implements and executes code based on strategist plans
"""

from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
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


class LibrarianAgent(BaseCodeAgent):
    """
    The Librarian Agent specializes in data retrieval and domain knowledge organization.
    
    Responsibilities:
    - Explore and catalog available data sources
    - Extract relevant information from documents and datasets
    - Provide structured knowledge summaries
    - Maintain context about data relationships and constraints
    """
    
    def get_system_prompt_template(self) -> str:
        return """You are the Librarian Agent, a specialist in data discovery and knowledge organization.

Your primary responsibilities:
1. **Data Discovery**: Explore the directory `{ctx_path}` to catalog all available data sources, files, and documentation
2. **Knowledge Extraction**: Read and understand the content, structure, and relationships of data
3. **Information Organization**: Create structured summaries of findings for other agents
4. **Context Maintenance**: Keep track of data constraints, formats, and dependencies

When working with data:
- Always start by exploring the `{ctx_path}` directory thoroughly
- Document file structures, data schemas, and relationships
- Identify data quality issues, missing values, or constraints
- Create clear, structured summaries that other agents can use
- Maintain awareness of data lineage and dependencies

Your output should be structured as:
```json
{{
    "data_catalog": {{
        "files": [...],
        "schemas": {{...}},
        "relationships": [...]
    }},
    "knowledge_summary": "...",
    "constraints": [...],
    "recommendations": [...]
}}
```

Available imports: {authorized_imports}

Remember: You are the foundation of knowledge for the entire system. Be thorough and accurate."""

    def catalog_data_sources(self, ctx_path: str) -> Dict[str, Any]:
        """Catalog all available data sources in the given path."""
        print(f"📚 Librarian: Starting data source cataloging in {ctx_path}")
        prompt = f"Explore and catalog all data sources in {ctx_path}. Provide a comprehensive overview of available files, their formats, and contents."
        print(f"📚 Librarian: Running data exploration...")
        response = self.run(prompt)
        print(f"📚 Librarian: ✅ Data cataloging completed")
        return {"catalog_response": response}

    def extract_domain_knowledge(self, query: str, ctx_path: str) -> Dict[str, Any]:
        """Extract specific domain knowledge based on a query."""
        print(f"📚 Librarian: Extracting domain knowledge for query: '{query}'")
        prompt = f"""Based on the data in {ctx_path}, extract domain knowledge relevant to: {query}
        
        Provide structured information including:
        - Relevant data sources
        - Key concepts and definitions
        - Data relationships
        - Constraints and limitations
        """
        print(f"📚 Librarian: Analyzing domain knowledge...")
        response = self.run(prompt)
        print(f"📚 Librarian: ✅ Domain knowledge extraction completed")
        return {"knowledge_response": response, "query": query}


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
        librarian_findings = self.librarian.catalog_data_sources(self.ctx_path)
        domain_knowledge = self.librarian.extract_domain_knowledge(task, self.ctx_path)
        
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
