#!/usr/bin/env python3
"""
Test script for ChatCodeAgent data analysis capabilities.
Tests that the chat agent can perform comprehensive data analysis tasks.

Usage:
    python test_data_analysis_chat.py
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Add src to path so we can import our modules
sys.path.insert(0, 'src')

from agents.code_agents import ChatCodeAgent
from utils.execution import get_env
from utils.tracing import setup_smolagents_tracing
from test_data_analysis_shared import (
    create_sales_data,
    run_analysis_tests,
    print_test_summary,
    setup_test_environment
)

# Initialize tracing for the test with chat-specific project
setup_smolagents_tracing(resource_name="chat-analysis", force_reinit=True)


def main():
    """Main test function for ChatCodeAgent."""
    print("📊 Testing ChatCodeAgent Data Analysis Capabilities")
    print("=" * 65)
    
    # Set up test environment
    config, api_key, model_id, api_base = setup_test_environment()
    
    # Create temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create test data
        data_dir = create_sales_data(temp_dir)
        
        # Initialize ChatCodeAgent
        print("\n🤖 Initializing ChatCodeAgent...")
        try:
            agent = ChatCodeAgent(
                model_id=model_id,
                api_base=api_base,
                api_key=api_key,
                max_steps=10,
                ctx_path=data_dir
            )
            print("✅ ChatCodeAgent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize ChatCodeAgent: {e}")
            return
        
        # Run analysis tests
        exploration_passed, analysis_passed, insights_passed = run_analysis_tests(agent, data_dir)
        
        # Print summary
        all_passed = print_test_summary(exploration_passed, analysis_passed, insights_passed, "ChatCodeAgent")
        
        if all_passed:
            print("\n🎯 ChatCodeAgent demonstrates strong conversational data analysis capabilities!")
        else:
            print("\n🔧 ChatCodeAgent may need tuning for optimal data analysis performance.")


if __name__ == "__main__":
    main()
