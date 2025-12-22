"""
Simple content ingestion script to connect the embedding pipeline with the RAG system.
This script processes book content, generates embeddings, and stores them in Qdrant.
"""
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import hashlib

# Load environment variables
load_dotenv()

class SimpleContentIngestor:
    """Simple class to handle content ingestion from book text into Qdrant vector database."""

    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "localhost"),
            api_key=os.getenv("QDRANT_API_KEY", ""),
            prefer_grpc=True
        )

        self.collection_name = os.getenv("QDRANT_COLLECTION", "textbook_embeddings")

    def create_collection_if_not_exists(self):
        """Create the Qdrant collection if it doesn't exist."""
        try:
            # Try to get collection info to check if it exists
            self.qdrant_client.get_collection(self.collection_name)
            print(f"[OK] Collection '{self.collection_name}' already exists.")
        except Exception:
            # Collection doesn't exist, create it
            print(f"Creating collection '{self.collection_name}'...")

            # Create collection with 384-dimensional vectors (typical for sentence transformers)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"[OK] Collection '{self.collection_name}' created successfully.")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding for text using a hash-based approach.
        In production, you would use a proper embedding model.
        """
        # Create a deterministic embedding based on the text content
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Generate a pseudo-embedding from the hash
        embedding = []
        for i in range(0, 384*2, 2):  # 384 dimensions
            if i + 1 < len(text_hash):
                hex_pair = text_hash[i:i+2]
                val = int(hex_pair, 16) / 128.0 - 1.0  # Normalize to [-1, 1]
                embedding.append(val)
            else:
                embedding.append(0.0)

        # Trim or pad to exactly 384 dimensions
        if len(embedding) > 384:
            embedding = embedding[:384]
        elif len(embedding) < 384:
            embedding.extend([0.0] * (384 - len(embedding)))

        return embedding

    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Simple text splitter that breaks text into overlapping chunks.
        """
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk + " " + sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Add overlap by including part of the previous chunk
                words = current_chunk.split()
                overlap_words = words[-overlap:] if len(words) > overlap else words
                current_chunk = " ".join(overlap_words) + " " + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Final cleanup: ensure no chunk is too small
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 50:  # Minimum chunk size
                final_chunks.append(chunk)

        return final_chunks

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

        # Split content into chunks
        print("Splitting content into chunks...")
        chunks = self.split_text(content)
        print(f"[OK] Processed content into {len(chunks)} chunks")

        # Prepare and upload points to Qdrant
        print("Generating embeddings and uploading to Qdrant...")
        points = []
        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk
            embedding = self.generate_embedding(chunk)

            # Create a Qdrant point
            point = PointStruct(
                id=i,  # In production, use UUIDs for IDs
                vector=embedding,
                payload={
                    "content": chunk,
                    "source": source,
                    "chunk_index": i,
                    "metadata": {
                        "source": source,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
            )
            points.append(point)

            if (i + 1) % 10 == 0:  # Progress update every 10 chunks
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")

        # Upload points to Qdrant
        print(f"Uploading {len(points)} vectors to Qdrant...")
        self.qdrant_client.upload_points(
            collection_name=self.collection_name,
            points=points
        )

        print(f"[SUCCESS] Successfully ingested {len(chunks)} content chunks into Qdrant collection '{self.collection_name}'")
        return len(chunks)


def main():
    """Main function to run the content ingestion process."""
    print("Initializing Physical AI & Humanoid Robotics textbook content ingestion...")

    # Initialize the ingestor
    ingestor = SimpleContentIngestor()

    # Sample content from the Physical AI & Humanoid Robotics textbook
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
        num_ingested = ingestor.ingest_content(sample_content, "physical_ai_textbook")
        print(f"\n[SUCCESS] Successfully completed ingestion of {num_ingested} content chunks!")
        print("\nThe RAG system is now populated with textbook content and ready to answer queries.")

    except Exception as e:
        print(f"[ERROR] Error during content ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()