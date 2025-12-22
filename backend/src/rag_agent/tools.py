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
        _retrieval_system = RetrievalSystem(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
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

    retrieval_system = get_retrieval_system()
    print("DEBUG: Retrieval system initialized")

    try:
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
        raise
