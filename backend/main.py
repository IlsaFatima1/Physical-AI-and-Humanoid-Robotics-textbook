"""
RAG Chatbot Backend - Embedding Pipeline

This module implements the complete pipeline for:
1. Fetching all URLs from the deployed Docusaurus site
2. Extracting clean text from each URL
3. Chunking content and generating embeddings using Cohere
4. Storing vectors with metadata in Qdrant collection named 'rag_embedding'
"""

import asyncio
import os
import re
from typing import List, Dict, Tuple
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
# Initialize clients (these will be configured with environment variables)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None
qdrant_client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=30.0)


def get_all_urls(base_url: str = "https://physical-ai-textbook-eight.vercel.app/") -> List[str]:
    """
    Fetch all URLs from the deployed Docusaurus site.

    Args:
        base_url: The base URL of the Docusaurus site

    Returns:
        List of all URLs found on the site
    """
    logger.info(f"Fetching URLs from {base_url}")

    # First, try to get URLs from sitemap.xml
    sitemap_url = urljoin(base_url, "sitemap.xml")
    urls = []

    try:
        response = requests.get(sitemap_url)
        if response.status_code == 200:
            # Use html parser instead of xml to avoid requiring lxml
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for <loc> tags in the sitemap (they may appear as regular tags in html parser)
            loc_tags = soup.find_all('loc')
            urls = [loc.text.strip() for loc in loc_tags]

            # Fix placeholder domain in sitemap URLs
            actual_domain = base_url.replace("https://", "").split("/")[0]  # Extract domain from base_url
            urls = [url.replace("your-vercel-domain.vercel.app", actual_domain) for url in urls]

            logger.info(f"Found {len(urls)} URLs from sitemap")
        else:
            logger.warning(f"Sitemap not found at {sitemap_url}, attempting to crawl site")

            # Fallback: crawl the site starting from the base URL
            urls = _crawl_site(base_url)
    except Exception as e:
        logger.error(f"Error fetching sitemap: {e}")
        # Fallback: crawl the site starting from the base URL
        urls = _crawl_site(base_url)

    # Filter out any non-documentation URLs if needed
    filtered_urls = [url for url in urls if _is_valid_content_url(url)]

    logger.info(f"Total valid URLs found: {len(filtered_urls)}")
    return filtered_urls


def _crawl_site(base_url: str) -> List[str]:
    """
    Crawl the site to find all URLs if sitemap is not available.

    Args:
        base_url: The base URL to start crawling from

    Returns:
        List of URLs found by crawling
    """
    visited_urls = set()
    urls_to_visit = [base_url]

    while urls_to_visit:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)

        try:
            response = requests.get(current_url)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']

                # Convert relative URLs to absolute URLs
                absolute_url = urljoin(current_url, href)

                # Only add URLs from the same domain and that haven't been visited
                if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                    if absolute_url not in visited_urls and absolute_url not in urls_to_visit:
                        urls_to_visit.append(absolute_url)

        except Exception as e:
            logger.error(f"Error crawling {current_url}: {e}")

    return list(visited_urls)


def _is_valid_content_url(url: str) -> bool:
    """
    Check if a URL is a valid content page (not an external link, asset, etc.).

    Args:
        url: The URL to check

    Returns:
        True if the URL is a valid content page, False otherwise
    """
    parsed = urlparse(url)

    # Exclude non-http(s) URLs
    if parsed.scheme not in ['http', 'https']:
        return False

    # Exclude common non-content extensions
    excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', '.svg']
    if any(url.lower().endswith(ext) for ext in excluded_extensions):
        return False

    # Include only documentation-related paths
    included_paths = ['/ch', '/appendices/', '/reference/', '/getting-started/']
    if any(path in url.lower() for path in included_paths):
        return True

    # Include the home page
    if url == 'https://physical-ai-textbook-eight.vercel.app/' or url == 'https://physical-ai-textbook-eight.vercel.app':
        return True

    return False


def extract_text_from_url(url: str) -> str:
    """
    Extract clean text from a given URL.

    Args:
        url: The URL to extract text from

    Returns:
        Clean text content from the URL
    """
    logger.info(f"Extracting text from {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try to find the main content area in Docusaurus
        # Docusaurus typically has content in divs with specific classes
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find(class_='main-wrapper') or
            soup.find(class_='container') or
            soup.find(id='main') or
            soup
        )

        # Extract text from the main content
        text = main_content.get_text(separator='\n')

        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        logger.info(f"Extracted {len(text)} characters from {url}")
        return text

    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return ""


def chunk_text(text: str, max_length: int = 512) -> List[str]:
    """
    Split text into chunks of specified maximum length.

    Args:
        text: The text to chunk
        max_length: Maximum length of each chunk (in characters)

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Split text into sentences to avoid breaking sentences across chunks
    sentences = re.split(r'[.!?]+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Clean up the sentence
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence would exceed the max length
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            # Add the current chunk to the list
            chunks.append(current_chunk.strip())
            # Start a new chunk with the current sentence
            current_chunk = sentence
        else:
            # Add the sentence to the current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"Text chunked into {len(chunks)} chunks")
    return chunks


def embed(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Cohere.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    if not co:
        raise ValueError("Cohere client not initialized. Please set COHERE_API_KEY environment variable.")

    if not texts:
        return []

    logger.info(f"Generating embeddings for {len(texts)} texts")

    try:
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",  # Using a suitable Cohere embedding model
            input_type="search_document"  # Specify this is for search documents
        )

        embeddings = response.embeddings
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [[] for _ in texts]  # Return empty embeddings for each text


def create_collection(collection_name: str = "rag_embedding"):
    """
    Create or reuse a Qdrant collection for storing embeddings.

    Args:
        collection_name: Name of the collection to create or reuse
    """
    logger.info(f"Creating/reusing Qdrant collection: {collection_name}")

    try:
        # Check if collection already exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists, reusing it")
            return

        # Create new collection
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),  # Cohere v3 embeddings are 1024-dim
        )

        logger.info(f"Collection '{collection_name}' created successfully")

    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise


def save_chunk_to_qdrant(text_chunk: str, embedding: List[float], metadata: Dict, collection_name: str = "rag_embedding"):
    """
    Save a text chunk with its embedding and metadata to Qdrant.

    Args:
        text_chunk: The original text chunk
        embedding: The embedding vector for the text chunk
        metadata: Metadata dictionary containing source URL, etc.
        collection_name: Name of the collection to save to
    """
    logger.info(f"Saving chunk to Qdrant collection: {collection_name}")

    try:
        import time
        # Generate a unique ID using timestamp and index to avoid needing to count existing points
        import uuid
        point_id = str(uuid.uuid4())

        # Prepare point for insertion
        points = [
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": text_chunk,
                    "metadata": metadata
                }
            )
        ]

        # Insert the point into the collection
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info(f"Chunk saved successfully to collection '{collection_name}'")

    except Exception as e:
        logger.error(f"Error saving chunk to Qdrant: {e}")
        raise


async def main():
    """
    Main function to execute the complete RAG embedding pipeline.
    """
    logger.info("Starting RAG embedding pipeline")

    try:
        # Step 1: Get all URLs from the deployed site
        urls = get_all_urls()

        # Step 2: Create or reuse the Qdrant collection
        create_collection("rag_embedding")

        # Process each URL
        for i, url in enumerate(urls[:10]):  # Limit to first 10 URLs for initial testing
            logger.info(f"Processing URL {i+1}/{len(urls[:10])}: {url}")

            # Extract text from the URL
            text = extract_text_from_url(url)
            if not text:
                logger.warning(f"No text extracted from {url}, skipping")
                continue

            # Chunk the text
            text_chunks = chunk_text(text, max_length=512)

            # Process each chunk
            for j, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {j+1}/{len(text_chunks)} for URL: {url}")

                # Generate embedding for the chunk
                embeddings = embed([chunk])

                if embeddings and len(embeddings[0]) > 0:
                    # Prepare metadata
                    metadata = {
                        "source_url": url,
                        "chunk_index": j,
                        "total_chunks": len(text_chunks),
                        "created_at": str(asyncio.get_event_loop().time())
                    }

                    # Save to Qdrant
                    save_chunk_to_qdrant(
                        text_chunk=chunk,
                        embedding=embeddings[0],
                        metadata=metadata,
                        collection_name="rag_embedding"
                    )
                else:
                    logger.warning(f"Failed to generate embedding for chunk {j+1} of {url}")

        logger.info("RAG embedding pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise


if __name__ == "__main__":
    # Check if required environment variables are set
    if not os.getenv("COHERE_API_KEY"):
        logger.warning("COHERE_API_KEY environment variable not set. Please set it before running.")

    # Run the main function
    asyncio.run(main())
