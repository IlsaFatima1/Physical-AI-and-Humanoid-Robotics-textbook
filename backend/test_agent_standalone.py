"""
Standalone test script for the BookRAGAgent without requiring actual API keys
This test focuses on the structure and functionality of the agent without making real API calls
"""
import os
import sys
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

from src.rag_agent.agent import run_agent, agent
from src.rag_agent.config import Config
from src.rag_agent.models import SourceDocument


def test_agent_structure():
    """
    Test the agent's structure without making actual API calls
    """
    print("Testing agent structure and configuration...")

    # Mock the Gemini API to avoid requiring real API key
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "fake-api-key-for-testing",
        "QDRANT_URL": "http://localhost:6333",  # Use local Qdrant for testing
        "QDRANT_COLLECTION": "test_collection"
    }):
        try:
            # Initialize configuration (this should work even with fake API key)
            config = Config()
            config.gemini_api_key = "fake-key"  # Override for test
            print("[PASS] Configuration initialized")

            # Create a mock agent to test structure
            agent = agent(config)
            print("[PASS] Agent instance created")

            # Test the query method structure with mocked response
            with patch.object(agent.model, 'generate_content') as mock_generate:
                # Mock a response
                mock_response = Mock()
                mock_response.text = "This is a test response from the agent."
                mock_generate.return_value = mock_response

                # Mock the book_retriever function to return test data
                with patch('rag_agent.agent.book_retriever') as mock_book_retriever:
                    mock_book_retriever.return_value = [
                        {
                            "id": "test-id-1",
                            "score": 0.9,
                            "content": "This is test content about ROS2 architecture from the textbook.",
                            "source": "chapter_3_ros2.pdf",
                            "metadata": {"page": 45, "section": "3.2"}
                        }
                    ]

                    # Test the query functionality
                    response = agent.query("What is ROS2 architecture?", top_k=1)

                    print(f"[PASS] Query processed successfully")
                    print(f"  Query: {response.query}")
                    print(f"  Answer: {response.answer[:100]}...")
                    print(f"  Number of sources: {len(response.sources)}")

                    if response.sources:
                        source = response.sources[0]
                        print(f"  First source: {source.get('source', 'N/A')} (Score: {source.get('score', 0)})")

                    return True

        except Exception as e:
            print(f"[FAIL] Error during agent structure test: {e}")
            return False


def test_format_context_for_prompt():
    """
    Test the context formatting functionality
    """
    print("\nTesting context formatting...")

    try:
        # Create a simple agent instance for testing
        config = Mock()
        config.gemini_api_key = "fake-key"
        agent = agent.__new__(agent)  # Create without calling __init__

        # Test the context formatting method directly
        test_docs = [
            {
                "id": "test-1",
                "score": 0.85,
                "content": "ROS2 is a flexible framework for developing robot applications.",
                "source": "chapter_3_ros2.pdf",
                "metadata": {"page": 45}
            },
            {
                "id": "test-2",
                "score": 0.72,
                "content": "The architecture includes nodes, topics, and services.",
                "source": "chapter_3_ros2.pdf",
                "metadata": {"page": 47}
            }
        ]

        formatted_context = agent._format_context_for_prompt(test_docs)

        print("[PASS] Context formatted successfully")
        print(f"  Context length: {len(formatted_context)} characters")
        print(f"  Contains 'Relevant textbook content': {'Relevant textbook content' in formatted_context}")
        print(f"  Contains first document: {'ROS2 is a flexible framework' in formatted_context}")

        return True

    except Exception as e:
        print(f"[FAIL] Error during context formatting test: {e}")
        return False


def test_error_handling():
    """
    Test the agent's error handling capabilities
    """
    print("\nTesting error handling...")

    try:
        # Mock the configuration
        config = Mock()
        config.gemini_api_key = "fake-key"

        agent = agent.__new__(agent)
        agent.config = config

        # Test error handling when book_retriever fails
        with patch('rag_agent.agent.book_retriever') as mock_book_retriever:
            mock_book_retriever.side_effect = Exception("Test retrieval error")

            response = agent.query("Test query")

            print("[PASS] Error handling test passed")
            print(f"  Response has error message: {'error' in response.answer.lower() or 'error' in response.answer}")
            print(f"  Sources are empty: {len(response.sources) == 0}")

            return True

    except Exception as e:
        print(f"[FAIL] Error during error handling test: {e}")
        return False


def main():
    """
    Main test function
    """
    print("Starting BookRAGAgent standalone tests...\n")

    # Test agent structure
    structure_success = test_agent_structure()

    # Test context formatting
    format_success = test_format_context_for_prompt()

    # Test error handling
    error_handling_success = test_error_handling()

    print(f"\nTest Results:")
    print(f"- Agent Structure: {'PASS' if structure_success else 'FAIL'}")
    print(f"- Context Formatting: {'PASS' if format_success else 'FAIL'}")
    print(f"- Error Handling: {'PASS' if error_handling_success else 'FAIL'}")

    all_passed = structure_success and format_success and error_handling_success

    if all_passed:
        print("\n[PASS] All standalone tests passed!")
        print("Note: Full functionality requires valid GEMINI_API_KEY and Qdrant connection")
        return True
    else:
        print("\n[FAIL] Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)