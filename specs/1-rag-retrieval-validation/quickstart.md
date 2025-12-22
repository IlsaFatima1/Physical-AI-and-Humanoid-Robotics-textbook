# Quickstart: RAG Retrieval & Vector Validation

## Overview
This guide provides a quick start for setting up and using the RAG retrieval validation system for the Physical AI & Humanoid Robotics textbook.

## Prerequisites
- Python 3.11+
- Qdrant vector database running
- Cohere API key for embeddings
- Git

## Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd physical-ai-textbook
```

### 2. Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file with the following:
```env
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_url
QDRANT_PORT=6333
QDRANT_COLLECTION=textbook_embeddings
```

## Running Validation Tests

### 1. Execute a simple validation test
```bash
python -m backend.src.rag_validation validate --query "What is ROS2 architecture?"
```

### 2. Run comprehensive validation suite
```bash
python -m backend.src.rag_validation run-all-tests
```

### 3. Validate with specific query types
```bash
# Factual query
python -m backend.src.rag_validation validate --query "What is a URDF file?" --type factual

# Conceptual query
python -m backend.src.rag_validation validate --query "Explain robot perception systems" --type conceptual

# Procedural query
python -m backend.src.rag_validation validate --query "How to configure a ROS2 launch file?" --type procedural
```

## API Usage

### Using the validation API
```python
from backend.src.rag_validation.services.validation_service import ValidationService

# Initialize the service
validator = ValidationService()

# Validate a query
result = validator.validate_query(
    query_text="What is Gazebo simulation?",
    query_type="factual",
    top_k=3
)

print(f"Validation passed: {result.validation_passed}")
print(f"Relevance score: {result.relevance_metrics.top_k_accuracy}")
```

## Expected Output
The validation will return:
- Whether the validation passed or failed
- Relevance metrics including top-k accuracy
- Metadata validation results
- List of retrieved chunks with their relevance scores

## Troubleshooting

### Common Issues
1. **Connection to Qdrant fails**: Check that Qdrant is running and the URL/port are correct
2. **API key errors**: Verify your Cohere API key is valid and properly configured
3. **No results returned**: Ensure the Qdrant collection contains the textbook embeddings

### Getting Help
- Check the logs in `logs/rag_validation.log`
- Verify the vector database has been populated with textbook content
- Ensure all required environment variables are set