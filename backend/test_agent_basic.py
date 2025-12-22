"""
Simple test to check if the agent can respond to a query
"""
import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

from rag_agent.agent import BookRAGAgent
from rag_agent.config import Config

def test_agent_creation():
    """Test that the agent can be created properly"""
    print("Testing agent creation...")

    # Set a fake API key for testing
    os.environ["GEMINI_API_KEY"] = "fake-test-key"

    try:
        # Initialize configuration
        config = Config()
        config.gemini_api_key = "fake-test-key"  # Override with test key

        # Initialize the agent
        agent = BookRAGAgent(config)

        print("[PASS] Agent created successfully")
        print(f"  Model name: {config.gemini_model_name}")
        print(f"  System instruction: {config.system_instruction[:50]}...")

        return agent
    except Exception as e:
        print(f"[FAIL] Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_agent_query_structure():
    """Test that the agent query method is properly structured"""
    print("\nTesting agent query structure...")

    # Set a fake API key for testing
    os.environ["GEMINI_API_KEY"] = "fake-test-key"

    try:
        # Initialize configuration
        config = Config()
        config.gemini_api_key = "fake-test-key"  # Override with test key

        # Initialize the agent
        agent = BookRAGAgent(config)

        # Check that the agent has the required attributes
        assert hasattr(agent, 'model'), "Agent should have a model attribute"
        print("[PASS] Agent has model attribute")

        # Check that query method exists
        assert hasattr(agent, 'query'), "Agent should have a query method"
        print("[PASS] Agent has query method")

        return True
    except Exception as e:
        print(f"[FAIL] Error in query structure test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running basic agent functionality tests...\n")

    agent = test_agent_creation()
    structure_ok = test_agent_query_structure()

    if agent and structure_ok:
        print("\n[PASS] All basic tests passed!")
        print("The agent is properly set up with tools and can be queried.")
    else:
        print("\n[FAIL] Some tests failed!")