from typing import List, Optional
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv


load_dotenv()
class RetrievalSystem:
    """
    A simple retrieval system that combines embedding generation and vector search.
    """

    def __init__(self, cohere_api_key: str, qdrant_url: str = "localhost",
                 qdrant_port: int = 6333, qdrant_api_key: Optional[str] = None,
                 collection_name: str = "rag_embedding"):
        print(f"DEBUG: Initializing RetrievalSystem")
        print(f"DEBUG: Cohere API key set: {bool(cohere_api_key)}")
        print(f"DEBUG: Qdrant URL: {qdrant_url}")
        print(f"DEBUG: Qdrant Port: {qdrant_port}")
        print(f"DEBUG: Qdrant API key set: {bool(qdrant_api_key)}")
        print(f"DEBUG: Collection name: {collection_name}")

        # Initialize Cohere client
        print("DEBUG: Initializing Cohere client...")
        try:
            self.cohere_client = cohere.Client(cohere_api_key)
            print("DEBUG: Cohere client initialized successfully")
        except Exception as e:
            print(f"DEBUG: Error initializing Cohere client: {e}")
            raise

        # Initialize Qdrant client
        print("DEBUG: Initializing Qdrant client...")
        try:
            if qdrant_api_key:
                # Check if qdrant_url is a full URL (contains 'http')
                if qdrant_url.startswith('http'):
                    # For cloud instances with full URL
                    print(f"DEBUG: Connecting to Qdrant cloud instance: {qdrant_url}")
                    self.qdrant_client = QdrantClient(
                        url=qdrant_url,
                        api_key=qdrant_api_key
                    )
                else:
                    # For local instances
                    print(f"DEBUG: Connecting to Qdrant local instance: {qdrant_url}:{qdrant_port}")
                    self.qdrant_client = QdrantClient(
                        url=qdrant_url,
                        port=qdrant_port,
                        api_key=qdrant_api_key
                    )
            else:
                # For local instances without API key
                print(f"DEBUG: Connecting to Qdrant local instance without API key: {qdrant_url}:{qdrant_port}")
                self.qdrant_client = QdrantClient(
                    host=qdrant_url,
                    port=qdrant_port
                )
            print("DEBUG: Qdrant client initialized successfully")
        except Exception as e:
            print(f"DEBUG: Error initializing Qdrant client: {e}")
            raise

        self.collection_name = collection_name
        print(f"DEBUG: RetrievalSystem initialization completed")

    def embed_query(self, query_text: str, model: str = "embed-english-v3.0",
                    input_type: str = "search_query") -> List[float]:
        """
        Generate embedding for a query text using Cohere.

        Args:
            query_text: The query text to embed
            model: The embedding model to use
            input_type: The type of input (search_query, search_document, etc.)

        Returns:
            List of floats representing the embedding vector
        """
        print(f"DEBUG: embed_query called with text: {query_text[:50]}...")
        print(f"DEBUG: Using model: {model}, input_type: {input_type}")

        try:
            response = self.cohere_client.embed(
                texts=[query_text],
                model=model,
                input_type=input_type
            )
            print(f"DEBUG: Embedding generated successfully, length: {len(response.embeddings[0])}")
            return response.embeddings[0]
        except Exception as e:
            print(f"DEBUG: Error generating embedding: {e}")
            import traceback
            print(f"DEBUG: Embedding error traceback: {traceback.format_exc()}")
            raise

    def retrieve(self, query_text: str, top_k: int = 3,
                 filters: Optional[dict] = None) -> List[dict]:
        """
        Retrieve relevant chunks for a query from Qdrant.

        Args:
            query_text: The query text
            top_k: Number of top results to retrieve
            filters: Optional filters to apply to the search

        Returns:
            List of dictionaries containing retrieved chunks with metadata and scores
        """
        print(f"DEBUG: retrieve called with query: {query_text[:50]}...")
        print(f"DEBUG: top_k: {top_k}, filters: {filters}")

        # Generate embedding for the query
        query_embedding = self.embed_query(query_text)

        # Prepare filters if provided
        search_filter = None
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            if filter_conditions:
                search_filter = models.Filter(must=filter_conditions)
            print(f"DEBUG: Filters prepared: {search_filter}")

        print(f"DEBUG: Starting Qdrant search in collection: {self.collection_name}")
        # Perform the search using the search API compatible with qdrant-client 1.8.0
        try:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            print(f"DEBUG: Qdrant search completed, found {len(search_results)} results")
        except Exception as e:
            print(f"DEBUG: Qdrant search failed: {e}")
            import traceback
            print(f"DEBUG: Qdrant search error traceback: {traceback.format_exc()}")
            raise

        # Format results
        results = []
        for i, result in enumerate(search_results):
            print(f"DEBUG: Processing result {i+1}")
            # Extract content - the embedding pipeline stores text under "text" field
            content = ""
            if result.payload:
                # First try to get "text" field (as stored by the embedding pipeline)
                content = result.payload.get("text", "")
                # Fallback to "content" if "text" doesn't exist
                if not content:
                    content = result.payload.get("content", "")

            # Extract metadata - handle the nested metadata structure if present
            metadata = result.payload or {}
            if metadata and "metadata" in metadata:  # Handle nested metadata structure
                # Merge the outer metadata with the inner metadata
                outer_metadata = {k: v for k, v in metadata.items() if k != "metadata"}
                inner_metadata = metadata["metadata"]
                final_metadata = {**outer_metadata, **inner_metadata}
            else:
                final_metadata = metadata

            result_dict = {
                "id": str(result.id),
                "content": content,
                "metadata": final_metadata,
                "relevance_score": float(result.score),
                "position": final_metadata.get("chunk_index", 0)
            }
            results.append(result_dict)
            print(f"DEBUG: Result {i+1} - Content length: {len(content)}, Score: {result.score}")

        print(f"DEBUG: retrieve completed, returning {len(results)} results")
        return results


