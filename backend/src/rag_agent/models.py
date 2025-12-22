from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class QueryRequest(BaseModel):
    """
    Request model for querying the RAG agent
    """
    query: str = Field(
        ...,
        description="The natural language question to ask about the textbook content",
        example="What is the ROS2 architecture?"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant documents to retrieve from the vector store (1-20)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Controls randomness in response generation (0.0-1.0)"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional ID to maintain conversation context across requests"
    )


class SourceDocument(BaseModel):
    """
    Model representing a source document used in the response
    """
    id: str = Field(..., description="Unique identifier of the source document")
    score: float = Field(..., description="Similarity score of the document to the query")
    content: str = Field(..., description="Content of the source document")
    source: str = Field(..., description="Source reference of the document")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the source document"
    )


class QueryResponse(BaseModel):
    """
    Response model from the RAG agent
    """
    query: str = Field(..., description="The original query that was processed")
    answer: str = Field(..., description="The AI-generated answer to the query")
    sources: List[SourceDocument] = Field(
        default=[],
        description="List of source documents used to generate the response"
    )
    context_used: List[str] = Field(
        default=[],
        description="List of context snippets used in generating the response"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="ID of the conversation if context was maintained"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when the response was generated"
    )


class HealthStatus(str, Enum):
    """
    Enum for health check status
    """
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint
    """
    status: HealthStatus = Field(..., description="Current health status of the service")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of the health check"
    )
    version: str = Field(default="1.0.0", description="Version of the API")
    services: Dict[str, HealthStatus] = Field(
        default={},
        description="Status of dependent services (e.g., Qdrant, Gemini API)"
    )


class ErrorResponse(BaseModel):
    """
    Model for error responses
    """
    error: str = Field(..., description="Error message describing what went wrong")
    error_code: Optional[str] = Field(
        default=None,
        description="Specific error code for programmatic handling"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when the error occurred"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional details about the error"
    )


# API Response models for FastAPI endpoints
class ChatResponse(BaseModel):
    """
    Response model for the chat endpoint
    """
    id: str = Field(..., description="Unique identifier for this response")
    query: str = Field(..., description="The original query that was processed")
    answer: str = Field(..., description="The AI-generated answer to the query")
    sources: List[SourceDocument] = Field(
        default=[],
        description="List of source documents used to generate the response"
    )
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Unix timestamp when the response was created"
    )


class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint
    """
    message: str = Field(
        ...,
        description="The user's message/question to the chatbot",
        example="Explain how Gazebo simulation works with ROS2"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant documents to retrieve from the vector store (1-20)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Controls randomness in response generation (0.0-1.0)"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional ID to maintain conversation context across requests"
    )