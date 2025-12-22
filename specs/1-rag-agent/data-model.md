# Data Model: RAG Agent & API Service

## Entities

### QueryRequest
- **query**: str - The user's natural language question about the textbook content
- **history**: List[Dict] - Optional conversation history for context (default: empty list)
- **temperature**: float - Optional parameter to control response randomness (default: 0.7)

### QueryResponse
- **response**: str - The agent's answer to the user's query
- **context**: List[str] - The retrieved text chunks that informed the response
- **sources**: List[str] - Source identifiers for the retrieved content
- **timestamp**: datetime - When the response was generated

### RetrievalResult
- **content**: str - The retrieved text chunk from the textbook
- **source**: str - Reference to the original location in the textbook
- **score**: float - Relevance score from the vector search (0.0-1.0)
- **metadata**: Dict - Additional metadata about the retrieved chunk

### AgentConfig
- **model**: str - The name of the LLM model to use
- **temperature**: float - Parameter controlling response randomness
- **max_tokens**: int - Maximum tokens in the response
- **retrieval_kwargs**: Dict - Parameters for the retrieval tool

## Relationships

- QueryRequest → (1) : (1) → QueryResponse (Each query request generates one response)
- QueryResponse → (1) : (0..n) → RetrievalResult (Each response may include multiple retrieved chunks)
- AgentConfig → (1) : (1..n) → QueryRequest (Configuration applies to all queries)

## Validation Rules

### QueryRequest Validation
- Query must not be empty
- Query length must be less than 1000 characters
- Temperature must be between 0.0 and 1.0

### QueryResponse Validation
- Response must not be empty
- Context must contain at least one retrieval result when available
- Timestamp must be in the past or present

### RetrievalResult Validation
- Content must not be empty
- Score must be between 0.0 and 1.0
- Source must be a valid reference

## State Transitions

### Query Processing States
- RECEIVED → RETRIEVING → GENERATING → COMPLETED (Query lifecycle through the RAG agent)

## API Contract

### Request Schema
```json
{
  "query": "string (required) - The user's question about the textbook",
  "history": "array[object] (optional) - Previous conversation turns",
  "temperature": "number (optional) - Response randomness (0.0-1.0)"
}
```

### Response Schema
```json
{
  "response": "string - The agent's answer",
  "context": "array[string] - Retrieved text chunks",
  "sources": "array[string] - Source identifiers",
  "timestamp": "string - ISO 8601 datetime"
}
```