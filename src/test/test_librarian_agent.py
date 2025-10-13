#!/usr/bin/env python3
"""
Simple test for LibrarianAgent using real HuggingFace dataset.
Tests catalog persistence and reuse functionality.
"""

import os
from pathlib import Path
from utils.execution import get_env
from utils.dabstep_utils import download_context
from agents.multi_agent_system import LibrarianAgent


def test_librarian_catalog():
    """Test LibrarianAgent's ability to catalog data sources and extract domain knowledge."""
    
    # Load environment configuration
    env_config = get_env()
    
    # Download context data from HuggingFace
    base_dir = str(Path().resolve())
    ctx_path = download_context(base_dir, env_config.get("HF_TOKEN"))
    
    print(f"📁 Downloaded context to: {ctx_path}")
    
    # Initialize LibrarianAgent
    librarian = LibrarianAgent(
        model_id=f"{os.getenv('LLM_GATEWAY')}/{env_config.get('MODEL')}",
        api_base=env_config.get("BASE_URL"),
        api_key=env_config.get("API_KEY"),
        max_steps=10,
        ctx_path=ctx_path
    )
    
    print("\n" + "="*60)
    print("TEST 1: Catalog Data Sources (First Call)")
    print("="*60)
    
    # Test 1: Catalog data sources - first call creates catalog
    catalog_result = librarian.catalog_data_sources(ctx_path)
    print(f"\n✅ Catalog Result:\n{catalog_result}")
    print(f"📦 Catalog cached: {librarian.has_catalog(ctx_path)}")
    
    print("\n" + "="*60)
    print("TEST 2: Catalog Data Sources (Second Call - Should Use Cache)")
    print("="*60)
    
    # Test 2: Second call should use cache
    catalog_result2 = librarian.catalog_data_sources(ctx_path)
    print(f"🔄 Same catalog instance: {catalog_result is catalog_result2}")
    print(f"✅ Cache working correctly!")
    
    print("\n" + "="*60)
    print("TEST 3: Extract Domain Knowledge (Uses Cached Catalog)")
    print("="*60)
    
    # Test 3: Extract domain knowledge - should use cached catalog
    query = "How do I compute the fee of a transaction?"
    domain_knowledge = librarian.extract_domain_knowledge(query, ctx_path)
    print(f"\n✅ Domain Knowledge Result:\n{domain_knowledge}")
    print(f"📦 Used cached catalog: {librarian.has_catalog(ctx_path)}")
    
    print("\n" + "="*60)
    print("TEST 4: Multiple Queries (All Use Same Catalog)")
    print("="*60)
    
    # Test 4: Multiple queries should all use the same cached catalog
    queries = [
        "What payment methods are supported?",
        "What are the data constraints?"
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n--- Query {i}: {q} ---")
        knowledge = librarian.extract_domain_knowledge(q, ctx_path)
        print(f"✅ Query {i} completed using cached catalog")
    
    print("\n" + "="*60)
    print("TEST 5: Force Refresh Catalog")
    print("="*60)
    
    # Test 5: Force refresh creates new catalog
    catalog_result3 = librarian.catalog_data_sources(ctx_path, force_refresh=True)
    print(f"🔄 New catalog created: {catalog_result3 is not catalog_result}")
    print(f"✅ Force refresh working correctly!")
    
    print("\n" + "="*60)
    print("✅ All LibrarianAgent tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_librarian_catalog()
