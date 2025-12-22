# Data Model: RAG Retrieval & Vector Validation

## Entities

### Query
- **id**: str - Unique identifier for the query
- **text**: str - The natural language query text
- **embedding**: List[float] - Vector representation of the query
- **query_type**: str - Category of query (factual, conceptual, procedural)
- **created_at**: datetime - Timestamp of query creation

### RetrievedChunk
- **id**: str - Unique identifier for the chunk
- **content**: str - The text content of the retrieved chunk
- **metadata**: ChunkMetadata - Associated metadata object
- **relevance_score**: float - Semantic similarity score (0.0-1.0)
- **position**: int - Position of chunk in original document

### ChunkMetadata
- **source_url**: str - URL or path to the original source
- **chunk_index**: int - Index position of this chunk in the document sequence
- **creation_timestamp**: datetime - When the chunk was created/processed
- **document_title**: str - Title of the source document
- **section**: str - Section or chapter name in the textbook
- **page_reference**: str - Page number or location in original document (optional)

### ValidationResult
- **query_id**: str - Reference to the original query
- **retrieved_chunks**: List[RetrievedChunk] - List of chunks returned for the query
- **validation_passed**: bool - Whether validation criteria were met
- **relevance_metrics**: RelevanceMetrics - Detailed relevance scoring
- **metadata_validation**: MetadataValidation - Results of metadata validation
- **validation_timestamp**: datetime - When validation was performed

### RelevanceMetrics
- **top_k_accuracy**: float - Percentage of top-k results that are relevant
- **mean_reciprocal_rank**: float - MRR of relevant results
- **precision_at_k**: Dict[int, float] - Precision at different k values
- **semantic_similarity_scores**: List[float] - Individual similarity scores

### MetadataValidation
- **all_metadata_present**: bool - Whether all required metadata fields are present
- **metadata_accuracy**: Dict[str, bool] - Validation result for each metadata field
- **missing_fields**: List[str] - List of any missing metadata fields
- **errors**: List[str] - Any validation errors encountered

## Relationships

- Query → (1) : (0..k) → RetrievedChunk (Query can retrieve multiple chunks)
- RetrievedChunk → (1) : (1) → ChunkMetadata (Each chunk has one metadata object)
- Query → (1) : (1) → ValidationResult (Each query has one validation result)
- ValidationResult → (1) : (1) → RelevanceMetrics (Validation has relevance metrics)
- ValidationResult → (1) : (1) → MetadataValidation (Validation has metadata validation)

## Validation Rules

### Query Validation
- Text must not be empty
- Query type must be one of: 'factual', 'conceptual', 'procedural'
- Embedding must have the correct dimensionality for the selected model

### RetrievedChunk Validation
- Content must not be empty
- Relevance score must be between 0.0 and 1.0
- Position must be a non-negative integer

### ChunkMetadata Validation
- Source URL must be a valid URL or file path
- Chunk index must be a non-negative integer
- Creation timestamp must be in the past or present
- Document title must not be empty

### ValidationResult Validation
- Query ID must reference an existing query
- Retrieved chunks list must not be empty
- Validation timestamp must be in the past or present

## State Transitions

### Query States
- DRAFT → EMBEDDED → SEARCHED → VALIDATED (Query lifecycle through the RAG pipeline)

### ValidationResult States
- PENDING → IN_PROGRESS → COMPLETED → (PASSED|FAILED) (Validation process states)