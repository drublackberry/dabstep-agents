#!/usr/bin/env python3
"""
Test script for ReasoningCodeAgent read-only security.
Tests that the agent properly enforces read-only file access restrictions.

Usage:
    python test_read_only_security.py
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

from agents.code_agents import ReasoningCodeAgent
from utils.execution import get_env
from utils.tracing import setup_smolagents_tracing

# Initialize tracing for the test
setup_smolagents_tracing()


def create_test_data(data_dir):
    """Create some test data files."""
    data_path = Path(data_dir)
    
    # Create a simple CSV file
    csv_file = data_path / "test_data.csv"
    csv_content = """name,value
test1,100
test2,200
test3,300"""
    csv_file.write_text(csv_content)
    
    # Create a text file
    txt_file = data_path / "readme.txt"
    txt_file.write_text("This is a test file for read-only security testing.")
    
    print(f"✅ Created test data in: {data_dir}")
    return data_dir


def test_write_file_blocked(agent, data_dir):
    """Test that file creation is blocked."""
    print("\n🔒 Testing file creation blocking...")
    
    write_task = f"""
    Try to create a new file in {data_dir}:
    
    Code:
    ```py
    file_path = "{data_dir}/malicious.txt"
    with open(file_path, 'w') as f:
        f.write('This should not work')
    print("File created successfully")
    ```<end_code>
    """
    
    try:
        result = agent.run(write_task)
        malicious_file = Path(data_dir) / "malicious.txt"
        if malicious_file.exists():
            print("❌ SECURITY ISSUE: File was created despite read-only mode!")
            return False
        else:
            if "PermissionError" in str(result) or "read mode" in str(result):
                print("✅ Write operation was properly blocked by read-only restriction")
                return True
            else:
                print("✅ Write operation was blocked (no file created)")
                return True
    except Exception as e:
        if "PermissionError" in str(e) or "read mode" in str(e):
            print(f"✅ Write operation blocked by read-only restriction: {e}")
            return True
        else:
            print(f"✅ Write operation blocked with error: {e}")
            return True


def test_append_file_blocked(agent, data_dir):
    """Test that file appending is blocked."""
    print("\n🔒 Testing file append blocking...")
    
    append_task = f"""
    Try to append to an existing file in {data_dir}:
    
    Code:
    ```py
    file_path = "{data_dir}/readme.txt"
    with open(file_path, 'a') as f:
        f.write('\\nThis append should not work')
    print("File appended successfully")
    ```<end_code>
    """
    
    # Store original content
    original_content = (Path(data_dir) / "readme.txt").read_text()
    
    try:
        result = agent.run(append_task)
        current_content = (Path(data_dir) / "readme.txt").read_text()
        
        if current_content != original_content:
            print("❌ SECURITY ISSUE: File was modified despite read-only mode!")
            return False
        else:
            if "PermissionError" in str(result) or "read mode" in str(result):
                print("✅ Append operation was properly blocked by read-only restriction")
                return True
            else:
                print("✅ Append operation was blocked (file unchanged)")
                return True
    except Exception as e:
        if "PermissionError" in str(e) or "read mode" in str(e):
            print(f"✅ Append operation blocked by read-only restriction: {e}")
            return True
        else:
            print(f"✅ Append operation blocked with error: {e}")
            return True


def test_read_file_allowed(agent, data_dir):
    """Test that file reading is still allowed."""
    print("\n📖 Testing file reading (should work)...")
    
    read_task = f"""
    Read the contents of a file in {data_dir}:
    
    Code:
    ```py
    file_path = "{data_dir}/test_data.csv"
    with open(file_path, 'r') as f:
        content = f.read()
    print(f"File content: {{content}}")
    ```<end_code>
    """
    
    try:
        result = agent.run(read_task)
        if "test1,100" in str(result):
            print("✅ Read operation works correctly")
            return True
        else:
            print("❌ Read operation failed or returned unexpected content")
            return False
    except Exception as e:
        print(f"❌ Read operation failed: {e}")
        return False


def main():
    """Main test function."""
    print("🔒 Testing ReasoningCodeAgent Read-Only Security")
    print("=" * 60)
    
    # Get LLM configuration
    config = get_env()
    api_key = config["API_KEY"]
    model_id = f"{config['LLM_GATEWAY']}/{config['MODEL']}"
    api_base = config["BASE_URL"]
    
    print(f"🔧 Using model: {model_id}")
    if api_base:
        print(f"🔧 Using API base: {api_base}")
    
    # Show tracing endpoint
    tracing_endpoint = os.getenv("OTLP_ENDPOINT", "http://0.0.0.0:6006/v1/traces")
    phoenix_ui = tracing_endpoint.replace("/v1/traces", "").replace("0.0.0.0", "localhost")
    print(f"📊 Phoenix UI available at: {phoenix_ui}")
    print(f"🔍 Traces endpoint: {tracing_endpoint}")
    
    # Create temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create test data
        data_dir = create_test_data(temp_dir)
        
        # Initialize agent
        print("\n🤖 Initializing ReasoningCodeAgent...")
        try:
            agent = ReasoningCodeAgent(
                model_id=model_id,
                api_base=api_base,
                api_key=api_key,
                max_steps=10,
                ctx_path=data_dir
            )
            print("✅ Agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            return
        
        # Run security tests
        write_blocked = test_write_file_blocked(agent, data_dir)
        append_blocked = test_append_file_blocked(agent, data_dir)
        read_allowed = test_read_file_allowed(agent, data_dir)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 READ-ONLY SECURITY TEST SUMMARY")
        print(f"🚫 Write blocking: {'✅ PASSED' if write_blocked else '❌ FAILED'}")
        print(f"🚫 Append blocking: {'✅ PASSED' if append_blocked else '❌ FAILED'}")
        print(f"📖 Read access: {'✅ PASSED' if read_allowed else '❌ FAILED'}")
        
        all_passed = write_blocked and append_blocked and read_allowed
        if all_passed:
            print("\n🎉 All read-only security tests passed!")
        else:
            print("\n⚠️  Some security tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
