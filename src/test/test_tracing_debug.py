#!/usr/bin/env python3
"""
Debug script to test tracing connectivity and configuration.
"""

import time
from utils.tracing import setup_smolagents_tracing, is_tracing_enabled
from agents.code_agents import ChatCodeAgent
from utils.execution import get_env

def test_tracing_setup():
    """Test if tracing setup works correctly."""
    print("🔍 Testing tracing setup...")
    
    # Test with debug endpoint
    success = setup_smolagents_tracing(
        endpoint="http://127.0.0.1:6006/v1/traces",
        enable_tracing=True,
        resource_name="debug-test"
    )
    
    print(f"✅ Tracing setup successful: {success}")
    print(f"✅ Tracing enabled: {is_tracing_enabled()}")
    
    return success

def test_agent_with_tracing():
    """Test agent creation and execution with tracing."""
    print("\n🤖 Testing agent with tracing...")
    
    try:
        # Get environment config
        env_config = get_env()
        
        # Create agent with tracing enabled
        agent = ChatCodeAgent(
            model_id=f"{env_config['LLM_GATEWAY']}/{env_config.get("MODEL")}",
            api_base=env_config.get("BASE_URL"),
            api_key=env_config.get("API_KEY"),
            
            max_steps=3,
            enable_tracing=True
        )
        
        print("✅ Agent created successfully")
        
        # Run a simple test
        result = agent.run("What is 2 + 2? Just return the number.")
        print(f"✅ Agent execution result: {result}")
        
        # Wait a moment for traces to be exported
        print("⏳ Waiting for traces to be exported...")
        time.sleep(2)
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def main():
    """Main debug function."""
    print("🚀 Starting tracing debug session...")
    
    # Test tracing setup
    tracing_ok = test_tracing_setup()
    
    if not tracing_ok:
        print("❌ Tracing setup failed - check Phoenix server and configuration")
        return
    
    # Test agent with tracing
    agent_ok = test_agent_with_tracing()
    
    if agent_ok:
        print("\n🎯 Debug complete! Check Phoenix UI at http://127.0.0.1:6006 for traces")
        print("📊 Look for project: 'debug-test'")
    else:
        print("\n🔧 Agent execution failed - check logs above")

if __name__ == "__main__":
    main()
