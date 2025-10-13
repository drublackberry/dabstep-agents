#!/usr/bin/env python3
"""
Example demonstrating catalog persistence in the multi-agent system.

This example shows how:
1. The catalog is created once and cached in LibrarianAgent
2. The catalog is reused for multiple domain knowledge queries
3. The orchestrator maintains shared state across all agents
"""

import os
from pathlib import Path
from utils.execution import get_env
from utils.dabstep_utils import download_context
from agents.multi_agent_system import LibrarianAgent, MultiAgentOrchestrator


def example_librarian_catalog_persistence():
    """Demonstrate catalog persistence at the LibrarianAgent level."""
    
    print("="*80)
    print("EXAMPLE 1: LibrarianAgent Catalog Persistence")
    print("="*80)
    
    # Setup
    env_config = get_env()
    base_dir = str(Path().resolve())
    ctx_path = download_context(base_dir, env_config.get("HF_TOKEN"))
    
    # Initialize LibrarianAgent
    librarian = LibrarianAgent(
        model_id=f"{os.getenv('LLM_GATEWAY')}/{env_config.get('MODEL')}",
        api_base=env_config.get("BASE_URL"),
        api_key=env_config.get("API_KEY"),
        max_steps=10,
        ctx_path=ctx_path
    )
    
    # First call: Creates catalog
    print("\n--- First Call: Creating catalog ---")
    catalog = librarian.catalog_data_sources(ctx_path)
    print(f"Catalog created: {catalog.get('task_type')}")
    
    # Check if catalog is cached
    print(f"\nCatalog cached: {librarian.has_catalog(ctx_path)}")
    
    # Second call: Uses cached catalog
    print("\n--- Second Call: Should use cache ---")
    catalog2 = librarian.catalog_data_sources(ctx_path)
    print(f"Same catalog instance: {catalog is catalog2}")
    
    # Domain knowledge queries automatically use cached catalog
    print("\n--- Domain Knowledge Query 1 ---")
    query1 = "How do I compute transaction fees?"
    knowledge1 = librarian.extract_domain_knowledge(query1, ctx_path)
    print(f"Query 1 completed: {knowledge1.get('query')}")
    
    print("\n--- Domain Knowledge Query 2 (uses same catalog) ---")
    query2 = "What are the payment data constraints?"
    knowledge2 = librarian.extract_domain_knowledge(query2, ctx_path)
    print(f"Query 2 completed: {knowledge2.get('query')}")
    
    # Force refresh if needed
    print("\n--- Force Refresh: Creating new catalog ---")
    catalog3 = librarian.catalog_data_sources(ctx_path, force_refresh=True)
    print(f"New catalog created: {catalog3.get('task_type')}")
    
    print("\n✅ LibrarianAgent catalog persistence example completed!")


def example_orchestrator_shared_state():
    """Demonstrate shared state management at the orchestrator level."""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Orchestrator Shared State")
    print("="*80)
    
    # Setup
    env_config = get_env()
    base_dir = str(Path().resolve())
    ctx_path = download_context(base_dir, env_config.get("HF_TOKEN"))
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(
        model_id=f"{os.getenv('LLM_GATEWAY')}/{env_config.get('MODEL')}",
        api_base=env_config.get("BASE_URL"),
        api_key=env_config.get("API_KEY"),
        max_steps=10,
        ctx_path=ctx_path
    )
    
    # Check initial state
    print("\n--- Initial State ---")
    print(f"Catalog exists: {orchestrator.get_catalog() is not None}")
    print(f"Domain knowledge: {orchestrator.get_domain_knowledge()}")
    
    # Solve first task - catalog will be created and stored
    print("\n--- Task 1: Catalog will be created ---")
    task1 = "Analyze payment transaction patterns"
    result1 = orchestrator.solve_task(task1, max_iterations=2)
    
    print(f"\nTask 1 status: {result1['status']}")
    print(f"Catalog now exists: {orchestrator.get_catalog() is not None}")
    
    # Solve second task - catalog will be reused
    print("\n--- Task 2: Catalog will be reused ---")
    task2 = "Calculate average transaction fees"
    result2 = orchestrator.solve_task(task2, max_iterations=2)
    
    print(f"\nTask 2 status: {result2['status']}")
    
    # Access shared state
    print("\n--- Shared State Summary ---")
    catalog = orchestrator.get_catalog()
    print(f"Catalog: {catalog.get('task_type') if catalog else 'None'}")
    
    all_knowledge = orchestrator.get_domain_knowledge()
    print(f"Domain knowledge for {len(all_knowledge)} tasks stored")
    
    # Access specific task knowledge
    task1_knowledge = orchestrator.get_domain_knowledge(task1)
    print(f"Task 1 knowledge available: {task1_knowledge is not None}")
    
    print("\n✅ Orchestrator shared state example completed!")


def example_manual_catalog_passing():
    """Demonstrate manually passing catalog between operations."""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Manual Catalog Passing")
    print("="*80)
    
    # Setup
    env_config = get_env()
    base_dir = str(Path().resolve())
    ctx_path = download_context(base_dir, env_config.get("HF_TOKEN"))
    
    # Initialize LibrarianAgent
    librarian = LibrarianAgent(
        model_id=f"{os.getenv('LLM_GATEWAY')}/{env_config.get('MODEL')}",
        api_base=env_config.get("BASE_URL"),
        api_key=env_config.get("API_KEY"),
        max_steps=10,
        ctx_path=ctx_path
    )
    
    # Create catalog once
    print("\n--- Creating catalog ---")
    catalog = librarian.catalog_data_sources(ctx_path)
    
    # Manually pass catalog to domain knowledge extraction
    print("\n--- Using catalog for multiple queries ---")
    
    queries = [
        "What payment methods are supported?",
        "How are fees calculated?",
        "What are the data quality constraints?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        knowledge = librarian.extract_domain_knowledge(
            query, ctx_path, catalog=catalog
        )
        print(f"  ✓ Completed with catalog")
    
    print("\n✅ Manual catalog passing example completed!")


if __name__ == "__main__":
    print("Catalog Persistence Examples")
    print("="*80)
    
    # Run examples
    example_librarian_catalog_persistence()
    example_orchestrator_shared_state()
    example_manual_catalog_passing()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
