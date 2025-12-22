# Quickstart: RAG Agent & API Service

## Overview
This guide provides a quick start for setting up and using the RAG agent service that answers questions about the Physical AI & Humanoid Robotics textbook using AI and vector retrieval.

## Prerequisites
- Python 3.11+
- Qdrant vector database with textbook embeddings
- Gemini API key
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
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=textbook_embeddings
```

## Running the Service

### 1. Start the API server
```bash
cd backend
uvicorn src.main:app --reload --port 8000
```

### 2. Query the RAG agent
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is ROS2 architecture?",
    "temperature": 0.7
  }'
```

## API Usage

### Using the chat endpoint
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "query": "Explain Gazebo simulation",
    "temperature": 0.5
})

data = response.json()
print(f"Answer: {data['response']}")
print(f"Sources: {data['sources']}")
```

## Expected Output
The API will return:
- A contextual answer based on the textbook content
- Retrieved context that informed the response
- Source references for fact-checking

## Troubleshooting

### Common Issues
1. **Connection to Qdrant fails**: Check that Qdrant is running and the URL/API key are correct
2. **API key errors**: Verify your Gemini API key is valid and properly configured
3. **No results returned**: Ensure the Qdrant collection contains the textbook embeddings

### Getting Help
- Check the logs in the terminal where the service is running
- Verify the vector database has been populated with textbook content
- Ensure all required environment variables are set