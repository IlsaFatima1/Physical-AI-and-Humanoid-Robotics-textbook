# RAG Chatbot Backend

This backend implements the embedding pipeline for the Physical AI & Humanoid Robotics Textbook RAG chatbot.

## Features

- Fetches all URLs from the deployed Docusaurus documentation site
- Extracts clean text content from each page
- Chunks content into manageable pieces
- Generates embeddings using Cohere
- Stores embeddings with metadata in Qdrant vector database

## Architecture

The system implements the following functions in `main.py`:

1. `get_all_urls()` - Fetches all URLs from the deployed Docusaurus site
2. `extract_text_from_url()` - Extracts clean text from a given URL
3. `chunk_text()` - Splits text into chunks of specified maximum length
4. `embed()` - Generates embeddings for texts using Cohere
5. `create_collection()` - Creates or reuses a Qdrant collection named 'rag_embedding'
6. `save_chunk_to_qdrant()` - Saves text chunks with embeddings and metadata to Qdrant
7. `main()` - Executes the complete RAG embedding pipeline

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up environment variables:
   ```bash
   export COHERE_API_KEY="your-cohere-api-key"
   export QDRANT_URL="your-qdrant-url"  # default: localhost
   export QDRANT_API_KEY="your-qdrant-api-key"  # default: ""
   export QDRANT_PORT="6333"  # default: 6333
   ```

3. Run the pipeline:
   ```bash
   uv run main.py
   ```

## Environment Variables

- `COHERE_API_KEY` (required): Your Cohere API key for generating embeddings
- `QDRANT_URL` (optional): URL for the Qdrant vector database (default: localhost)
- `QDRANT_API_KEY` (optional): API key for Qdrant (default: "")
- `QDRANT_PORT` (optional): Port for Qdrant (default: 6333)

## Usage

The main pipeline will:

1. Fetch all URLs from https://physical-ai-textbook-eight.vercel.app/
2. Extract text content from each URL
3. Chunk the content into 512-character segments
4. Generate embeddings using Cohere
5. Store the embeddings in a Qdrant collection named 'rag_embedding'
6. Include metadata such as source URL and chunk index

For testing purposes, the pipeline currently limits to the first 10 URLs. To process all URLs, modify the slice in the main function from `urls[:10]` to `urls`.

## Dependencies

- requests: For HTTP requests
- beautifulsoup4: For HTML parsing
- cohere: For generating embeddings
- qdrant-client: For vector database operations
- aiohttp: For async operations
- lxml: For XML parsing (needed for sitemap)