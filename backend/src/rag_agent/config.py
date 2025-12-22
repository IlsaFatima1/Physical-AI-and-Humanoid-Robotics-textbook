from pydantic_settings import BaseSettings
from typing import Optional
import os


class Config(BaseSettings):
    """Configuration settings for the BookRAGAgent"""

    # Gemini API Configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")  # Using gemini-pro as default

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "localhost")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_embedding")

    # Application Configuration
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    # System instruction for the agent
    system_instruction: str = (
        "You are an expert assistant for the Physical AI & Humanoid Robotics textbook. "
        "Your role is to answer questions about physical AI, humanoid robotics, ROS2, "
        "Gazebo simulation, Isaac Gym, and related topics based on the provided textbook content. "
        "Always ground your responses in the provided context and cite sources when possible. "
        "If the context doesn't contain enough information, state that clearly rather than "
        "making up information."
    )

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Validate required settings (skip in test environments)
        import os
        if not self.gemini_api_key and not os.getenv("TESTING", "false").lower() == "true":
            raise ValueError("GEMINI_API_KEY must be set in environment variables")