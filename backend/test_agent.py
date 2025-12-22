"""
Test script for the BookRAGAgent
"""
import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

from .src.rag_agent.agent import agent
from .src.rag_agent.config import Config

def test_ros2_architecture_query():
    """
    Test agent response to 'ROS2 architecture' query
    """
    print("Testing agent response to 'ROS2 architecture' query...")

    try:
        # Initialize configuration
        config = Config()

        # Initialize the agent
        agent = agent(config)

        # Test query
        query = "ROS2 architecture"
        response = agent.query(query, top_k=3)

        print(f"Query: {response.query}")
        print(f"Answer: {response.answer[:500]}...")  # First 500 chars
        print(f"Number of sources used: {len(response.sources)}")

        for i, source in enumerate(response.sources[:2], 1):  # Show first 2 sources
            print(f"Source {i}: {source['source']} (Score: {source['score']:.3f})")
            print(f"  Content: {source['content_snippet']}")

        print("✓ ROS2 architecture query test completed")
        return True

    except Exception as e:
        print(f"✗ Error during ROS2 architecture query test: {e}")
        return False


def test_gazebo_simulation_query():
    """
    Test agent response to 'Gazebo simulation' query
    """
    print("\nTesting agent response to 'Gazebo simulation' query...")

    try:
        # Initialize configuration
        config = Config()

        # Initialize the agent
        agent = agent(config)

        # Test query
        query = "Gazebo simulation"
        response = agent.query(query, top_k=3)

        print(f"Query: {response.query}")
        print(f"Answer: {response.answer[:500]}...")  # First 500 chars
        print(f"Number of sources used: {len(response.sources)}")

        for i, source in enumerate(response.sources[:2], 1):  # Show first 2 sources
            print(f"Source {i}: {source['source']} (Score: {source['score']:.3f})")
            print(f"  Content: {source['content_snippet']}")

        print("✓ Gazebo simulation query test completed")
        return True

    except Exception as e:
        print(f"✗ Error during Gazebo simulation query test: {e}")
        return False


def test_humanoid_robotics_query():
    """
    Test agent response to 'humanoid robotics' query (T022)
    """
    print("\nTesting agent response to 'humanoid robotics' query...")

    try:
        # Initialize configuration
        config = Config()

        # Initialize the agent
        agent = agent(config)

        # Test query
        query = "humanoid robotics"
        response = agent.query(query, top_k=3)

        print(f"Query: {response.query}")
        print(f"Answer: {response.answer[:500]}...")  # First 500 chars
        print(f"Number of sources used: {len(response.sources)}")

        for i, source in enumerate(response.sources[:2], 1):  # Show first 2 sources
            print(f"Source {i}: {source['source']} (Score: {source['score']:.3f})")
            print(f"  Content: {source['content_snippet']}")

        print("✓ Humanoid robotics query test completed")
        return True

    except Exception as e:
        print(f"✗ Error during humanoid robotics query test: {e}")
        return False


def main():
    """
    Main test function
    """
    print("Starting BookRAGAgent tests...\n")

    # Test T014: ROS2 architecture query
    ros2_success = test_ros2_architecture_query()

    # Test T015: Gazebo simulation query
    gazebo_success = test_gazebo_simulation_query()

    # Test T022: Humanoid robotics query
    humanoid_success = test_humanoid_robotics_query()

    print(f"\nTest Results:")
    print(f"- ROS2 Architecture Query: {'PASS' if ros2_success else 'FAIL'}")
    print(f"- Gazebo Simulation Query: {'PASS' if gazebo_success else 'FAIL'}")
    print(f"- Humanoid Robotics Query: {'PASS' if humanoid_success else 'FAIL'}")

    if ros2_success and gazebo_success and humanoid_success:
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)