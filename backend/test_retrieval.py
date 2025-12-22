import os
from backend.src.rag_agent.retrieval import RetrievalSystem
from dotenv import load_dotenv

load_dotenv()

# Example usage of the retrieval system
def test_retrieval():
    # Get API keys from environment variables
    cohere_api_key = os.getenv("COHERE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not cohere_api_key:
        print("Please set COHERE_API_KEY environment variable")
        return

    # Initialize the retrieval system
    try:
        retrieval_system = RetrievalSystem(
            cohere_api_key=cohere_api_key,
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            qdrant_api_key=qdrant_api_key
        )

        print("RetrievalSystem initialized successfully")

        # Test embedding a query
        query = "What is ROS2 architecture?"
        embedding = retrieval_system.embed_query(query)
        print(f"Embedded query: '{query}'")
        print(f"Embedding length: {len(embedding)}")

        # Test retrieval (this will fail if Qdrant collection doesn't exist)
        try:
            results = retrieval_system.retrieve(query, top_k=2)
            print(f"Retrieved {len(results)} results")
            for i, result in enumerate(results):
                # Safely handle content with potential encoding issues
                content = result['content'][:500] if result['content'] else 'No content field'
                # Remove or replace problematic characters
                safe_content = content.encode('utf-8', errors='ignore').decode('utf-8')

                print(f"Result {i+1}:")
                print(f"  ID: {result['id']}")
                print(f"  Content: {safe_content}")
                print(f"  Relevance Score: {result['relevance_score']}")
                print(f"  Position: {result['position']}")
                print(f"  Metadata keys: {list(result['metadata'].keys())}")
                print(f"  Source URL: {result['metadata'].get('source_url', 'N/A')}")
                print()
        except Exception as e:
            print(f"Retrieval failed (expected if Qdrant collection doesn't exist): {e}")

    except Exception as e:
        print(f"Failed to initialize RetrievalSystem: {e}")

if __name__ == "__main__":
    test_retrieval()