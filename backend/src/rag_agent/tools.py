from agents import function_tool
from .retrieval import RetrievalSystem
import os
from dotenv import load_dotenv


load_dotenv()

# Initialize ONCE (important)
_retrieval_system = None


def get_retrieval_system() -> RetrievalSystem:
    global _retrieval_system
    if _retrieval_system is None:
        # Check for required environment variables
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            print("ERROR: COHERE_API_KEY not set. Retrieval system will not work properly.")
            return None

        _retrieval_system = RetrievalSystem(
            cohere_api_key=cohere_api_key,
            qdrant_url=os.getenv("QDRANT_URL", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", 6333)),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            collection_name=os.getenv("QDRANT_COLLECTION", "rag_embedding"),
        )
    return _retrieval_system


@function_tool
def book_retriever(query: str) -> str:
    """
    Retrieve relevant book content for a user query.
    """
    print(f"DEBUG: book_retriever called with query: {query}")

    # Check environment variables for retrieval system
    cohere_api_key = os.getenv("COHERE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    print(f"DEBUG: COHERE_API_KEY set: {bool(cohere_api_key)}")
    print(f"DEBUG: QDRANT_URL: {qdrant_url}")
    print(f"DEBUG: QDRANT_PORT: {qdrant_port}")

    if not cohere_api_key:
        print("DEBUG: COHERE_API_KEY is not set!")
        return "Error: Cohere API key not configured. Please set the environment variable."

    try:
        retrieval_system = get_retrieval_system()
        if retrieval_system is None:
            return "Error: Retrieval system not properly configured. Please check your API keys."

        print("DEBUG: Retrieval system initialized")
        print("DEBUG: Starting retrieval...")
        results = retrieval_system.retrieve(query, top_k=6)
        print(f"DEBUG: Retrieval completed, found {len(results) if results else 0} items")

        if not results:
            print("DEBUG: No relevant content found")
            return "No relevant content found in the book."

        # Keep output SMALL & clean (important for agents)
        chunks = []
        for i, r in enumerate(results):
            print(f"DEBUG: Result {i+1}: {len(r['content'])} chars")
            chunks.append(r["content"])

        result = "\n\n".join(chunks)
        print(f"DEBUG: Final retrieval result: {len(result)} chars")
        return result
    except Exception as e:
        print(f"DEBUG: Retrieval error: {e}")
        import traceback
        print(f"DEBUG: Retrieval error traceback: {traceback.format_exc()}")

        # Check for common specific errors
        error_msg = str(e).lower()
        if "api" in error_msg or "key" in error_msg or "auth" in error_msg:
            return "Error: API authentication failed. Please check your Cohere API key."
        elif "connection" in error_msg or "network" in error_msg:
            return "Error: Network connection failed. Please check your internet connection."
        elif "collection" in error_msg or "not found" in error_msg:
            return "Error: Vector database collection not found. Please ensure the embedding pipeline has been run."
        else:
            # Return a more user-friendly error message
            return f"Retrieval error occurred. Please try again. (Error: {str(e)[:100]}...)"
