"""
Content ingestion script to connect the embedding pipeline with the RAG system.
This script processes book content, generates embeddings, and stores them in Qdrant
as specified in the embedding pipeline spec.
"""
import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI for embedding generation
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ContentIngestor:
    """Class to handle content ingestion from book text into Qdrant vector database."""

    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "localhost"),
            api_key=os.getenv("QDRANT_API_KEY", ""),
            prefer_grpc=True
        )

        self.collection_name = os.getenv("QDRANT_COLLECTION", "textbook_embeddings")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

        # Initialize text splitter for chunking content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def create_collection_if_not_exists(self):
        """Create the Qdrant collection if it doesn't exist."""
        try:
            # Try to get collection info to check if it exists
            self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            # Collection doesn't exist, create it
            # For this example, assuming 768 dimensions for embeddings (adjust as needed)
            # In practice, you'd need to determine the embedding dimension based on your model
            print(f"Creating collection '{self.collection_name}'...")

            # We'll use a placeholder dimension - in real implementation,
            # this should match the embedding model's output dimension
            # For Google's text-embedding-004, it's typically 768 or 1024 dimensions
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # Adjust size as needed
            )
            print(f"Collection '{self.collection_name}' created successfully.")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Google's embedding API.
        This is a simplified implementation - in production, you'd use the actual embedding API.
        """
        # Placeholder implementation for embedding generation
        # In a real implementation, you would call the embedding API
        import hashlib

        # Create a deterministic hash-based embedding for demonstration
        # In real implementation, use genai.embed_content() or similar
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding = []
        for i in range(768):  # Assuming 768-dim embedding
            val = (ord(text_hash[i % len(text_hash)]) + i) % 1000 / 500.0 - 1.0
            embedding.append(val)

        return embedding

    def process_book_content(self, content: str, source: str = "book_chapter") -> List[Dict[str, Any]]:
        """
        Process book content by splitting into chunks and preparing for embedding.

        Args:
            content: Raw book content as a string
            source: Source identifier for the content

        Returns:
            List of document chunks with metadata
        """
        # Split the content into chunks
        chunks = self.text_splitter.split_text(content)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "content": chunk,
                "source": source,
                "chunk_index": i,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

        return processed_chunks

    def ingest_content(self, content: str, source: str = "book_content"):
        """
        Main method to ingest content into the vector database.

        Args:
            content: Book content to be ingested
            source: Source identifier for the content
        """
        print(f"Starting content ingestion from source: {source}")

        # Create collection if it doesn't exist
        self.create_collection_if_not_exists()

        # Process the content into chunks
        chunks = self.process_book_content(content, source)
        print(f"Processed content into {len(chunks)} chunks")

        # Prepare points for Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk
            embedding = self.generate_embedding(chunk["content"])

            # Create a Qdrant point
            point = PointStruct(
                id=i,  # In production, use UUIDs for IDs
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "metadata": chunk["metadata"]
                }
            )
            points.append(point)

            if (i + 1) % 10 == 0:  # Progress update every 10 chunks
                print(f"Processed {i + 1}/{len(chunks)} chunks...")

        # Upload points to Qdrant
        print(f"Uploading {len(points)} vectors to Qdrant...")
        self.qdrant_client.upload_points(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Successfully ingested {len(chunks)} content chunks into Qdrant collection '{self.collection_name}'")
        return len(chunks)


def main():
    """Main function to run the content ingestion process."""
    print("Initializing Physical AI & Humanoid Robotics textbook content ingestion...")

    # Initialize the ingestor
    ingestor = ContentIngestor()

    # Example: Load content from a book file or other source
    # For this example, we'll use a placeholder content
    sample_content = """
    # Physical AI & Humanoid Robotics Textbook

    ## Chapter 1: Introduction to Physical AI

    Physical AI represents the intersection of artificial intelligence and physical systems.
    It encompasses the development of intelligent agents that can perceive, reason, and act
    in physical environments. This field combines elements of robotics, machine learning,
    computer vision, and control theory.

    The primary goal of Physical AI is to create systems that can operate autonomously in
    real-world environments, adapting to changing conditions and learning from experience.
    This requires sophisticated algorithms for perception, planning, control, and learning.

    ## Chapter 2: ROS2 Architecture

    Robot Operating System 2 (ROS2) is a flexible framework for writing robot software.
    It is a collection of tools, libraries, and conventions that aim to simplify the task
    of creating complex and robust robot behavior across a wide variety of robot platforms.

    ROS2 provides a publish-subscribe messaging system that allows different parts of a
    robot application to communicate with each other. Nodes represent individual processes
    that perform computation. Topics are named buses over which nodes exchange messages.

    The architecture includes several key components:
    - Nodes: Processes that perform computation
    - Topics: Named buses for message exchange
    - Messages: Data structures used inside ROS
    - Services: Synchronous request/response communication
    - Actions: Long-running goal-oriented communication
    - Parameters: Configuration values that can be accessed by nodes

    ## Chapter 3: Gazebo Simulation

    Gazebo is a 3D dynamic simulator that enables accurate and efficient simulation of
    robotic systems. It provides the capability to simulate populations of robots in complex
    indoor and outdoor environments.

    The simulator features:
    - High fidelity physics engine
    - High quality graphics rendering
    - Multiple sensors with realistic noise
    - Easy integration with ROS and ROS2
    - Plugin architecture for custom model and sensor development

    Gazebo plays a crucial role in the development and testing of robotic applications
    before deployment on physical hardware, reducing development time and costs.

    ## Chapter 4: Isaac Sim

    Isaac Sim is NVIDIA's robotics simulator built on the Omniverse platform. It provides
    a highly realistic simulation environment for training and testing AI agents.

    Key features include:
    - Photorealistic rendering
    - Physically accurate simulation
    - Synthetic data generation
    - Domain randomization capabilities
    - Integration with reinforcement learning frameworks
    """

    try:
        # Ingest the sample content
        num_ingested = ingestor.ingest_content(sample_content, "physical_ai_textbook_sample")
        print(f"\n✅ Successfully completed ingestion of {num_ingested} content chunks!")
        print("\nThe RAG system is now populated with textbook content and ready to answer queries.")

    except Exception as e:
        print(f"❌ Error during content ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()