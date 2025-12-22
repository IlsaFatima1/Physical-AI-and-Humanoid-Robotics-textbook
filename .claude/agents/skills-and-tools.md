# Skills and Tools for AI Agents

This document provides comprehensive information about creating and using skills/tools with AI agents, including patterns, best practices, and examples.

## Table of Contents
- [Introduction to Skills and Tools](#introduction-to-skills-and-tools)
- [Creating Function Tools](#creating-function-tools)
- [Tool Registration and Management](#tool-registration-and-management)
- [Advanced Tool Patterns](#advanced-tool-patterns)
- [Tool Security Considerations](#tool-security-considerations)
- [Examples](#examples)

## Introduction to Skills and Tools

Skills and tools are functions that agents can call to perform specific tasks. They extend the capabilities of LLMs by allowing them to interact with external systems, databases, APIs, and perform complex operations that pure text generation cannot handle.

### Key Concepts

- **Tools**: Functions that an agent can call to perform specific tasks
- **Skills**: Higher-level capabilities that may combine multiple tools
- **Function Calling**: The mechanism by which agents call tools
- **Tool Schema**: The definition of what parameters a tool accepts and what it returns

## Creating Function Tools

### Basic Tool Creation

The most common way to create tools is using the `@function_tool` decorator:

```python
from agents import function_tool

@function_tool
def search_tool(query: str) -> str:
    """
    Search for information based on the query.

    Args:
        query: The search query string

    Returns:
        A string containing the search results
    """
    # Implementation here
    return f"Search results for: {query}"
```

### Tool with Multiple Parameters

```python
@function_tool
def calculator_tool(expression: str, precision: int = 2) -> str:
    """
    Calculate the result of a mathematical expression.

    Args:
        expression: The mathematical expression to evaluate
        precision: Number of decimal places for the result (default: 2)

    Returns:
        A string containing the calculated result
    """
    try:
        # Note: In production, use a safe evaluation method
        result = eval(expression)
        return str(round(result, precision))
    except Exception:
        return "Error: Invalid expression"
```

### Tool with Complex Types

```python
from typing import Dict, List, Any
from agents import function_tool

@function_tool
def user_data_tool(user_id: str) -> Dict[str, Any]:
    """
    Get user data based on user ID.

    Args:
        user_id: The unique identifier for the user

    Returns:
        A dictionary containing user information
    """
    # Implementation would fetch user data from a database
    return {
        "id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "created_at": "2023-01-01"
    }
```

## Tool Registration and Management

### Registering Tools with an Agent

```python
from agents import Agent, Runner, AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel
import os

# Define tools
@function_tool
def weather_tool(city: str) -> str:
    """Get weather information for a city."""
    return f"Sunny in {city} with 22°C"

@function_tool
def news_tool(topic: str) -> str:
    """Get latest news on a topic."""
    return f"Latest news about {topic}: Interesting developments"

# Create agent with tools
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = OpenAIChatCompletionsModel(
    model="gpt-4o",
    openai_client=client,
)
config = RunConfig(model=model, model_provider=client, tracing_disabled=True)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    tools=[weather_tool, news_tool],  # Register tools here
)
```

### Dynamic Tool Registration

```python
# You can also register tools dynamically
def create_agent_with_tools(tool_list):
    return Agent(
        name="DynamicAgent",
        instructions="You are a dynamic assistant.",
        tools=tool_list,
    )

# Example usage
tools = [weather_tool, news_tool, search_tool]
dynamic_agent = create_agent_with_tools(tools)
```

## Advanced Tool Patterns

### Tool with Side Effects

```python
from agents import function_tool
import requests

@function_tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """
    Send an email to the specified recipient.

    Args:
        to: Email address of the recipient
        subject: Subject of the email
        body: Content of the email

    Returns:
        Confirmation message
    """
    try:
        # Implementation would send an actual email
        # This is a mock implementation
        response = {"status": "sent", "to": to}
        return f"Email sent to {to}: {response['status']}"
    except Exception as e:
        return f"Error sending email: {str(e)}"
```

### Tool with File Handling

```python
import os
from agents import function_tool
from typing import Optional

@function_tool
def file_reader_tool(file_path: str) -> str:
    """
    Read and return the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        Contents of the file as a string
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found - {file_path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Limit content size to prevent overwhelming the agent
            return content[:1000] + "..." if len(content) > 1000 else content
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

### Tool with API Integration

```python
import requests
from agents import function_tool
from typing import Dict, Any

@function_tool
def api_call_tool(url: str, method: str = "GET", data: str = "") -> str:
    """
    Make an API call to the specified URL.

    Args:
        url: The API endpoint to call
        method: HTTP method (GET, POST, PUT, DELETE)
        data: JSON data to send with POST/PUT requests

    Returns:
        API response as a string
    """
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data if data else {})
        elif method.upper() == "PUT":
            response = requests.put(url, json=data if data else {})
        elif method.upper() == "DELETE":
            response = requests.delete(url)
        else:
            return f"Error: Unsupported method {method}"

        return f"Status: {response.status_code}, Response: {response.text[:500]}"
    except Exception as e:
        return f"API call failed: {str(e)}"
```

## Tool Security Considerations

### Input Validation

Always validate inputs to prevent injection attacks:

```python
import re
from agents import function_tool

@function_tool
def safe_command_tool(command: str) -> str:
    """
    Execute a safe system command.

    Args:
        command: The command to execute (with validation)

    Returns:
        Command output or error message
    """
    # Only allow safe commands
    allowed_commands = ["ls", "pwd", "date", "echo"]

    # Validate command format
    if not re.match(r"^[a-z0-9\s-]+$", command):
        return "Error: Invalid command format"

    # Check if command is in allowed list
    cmd_parts = command.split()
    if cmd_parts and cmd_parts[0] not in allowed_commands:
        return f"Error: Command not allowed: {cmd_parts[0]}"

    # Implementation would execute the safe command
    return f"Would execute safe command: {command}"
```

### Rate Limiting

```python
import time
from agents import function_tool

# Simple rate limiting
_last_call_time = {}

@function_tool
def rate_limited_tool(user_id: str, data: str) -> str:
    """
    A tool with rate limiting to prevent abuse.
    """
    current_time = time.time()

    if user_id in _last_call_time:
        time_since_last = current_time - _last_call_time[user_id]
        if time_since_last < 1:  # 1 second minimum between calls
            return "Error: Rate limit exceeded. Please wait before trying again."

    _last_call_time[user_id] = current_time

    # Implementation here
    return f"Processed for user {user_id}: {data[:50]}..."
```

### Authentication and Authorization

```python
from agents import function_tool
import os

@function_tool
def authenticated_tool(user_token: str, resource: str) -> str:
    """
    Access a resource with authentication.

    Args:
        user_token: Authentication token
        resource: The resource to access

    Returns:
        Resource content or error message
    """
    # Validate token against stored tokens
    valid_token = os.getenv("VALID_API_TOKEN")
    if user_token != valid_token:
        return "Error: Invalid authentication token"

    # Access the resource
    return f"Accessing resource: {resource}"
```

## Examples

### RAG Tool (from our textbook project)

```python
from agents import function_tool
from .retrieval import RetrievalSystem
import os

# Global retrieval system instance
_retrieval_system = None

def get_retrieval_system() -> RetrievalSystem:
    global _retrieval_system
    if _retrieval_system is None:
        _retrieval_system = RetrievalSystem(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", 6333)),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            collection_name=os.getenv("QDRANT_COLLECTION", "rag_embedding"),
        )
    return _retrieval_system

@function_tool
def book_retriever(query: str) -> str:
    """
    Retrieve relevant book content for a user query.

    Args:
        query: The search query for book content

    Returns:
        Relevant book content as a string
    """
    retrieval_system = get_retrieval_system()

    try:
        results = retrieval_system.retrieve(query, top_k=6)

        if not results:
            return "No relevant content found in the book."

        # Combine the retrieved chunks
        chunks = [r["content"] for r in results]
        result = "\n\n".join(chunks)
        return result
    except Exception as e:
        print(f"Retrieval error: {e}")
        import traceback
        print(f"Retrieval error traceback: {traceback.format_exc()}")
        raise
```

### Database Query Tool

```python
from agents import function_tool
import sqlite3
from typing import List, Dict, Any

@function_tool
def database_query_tool(query: str) -> List[Dict[str, Any]]:
    """
    Execute a SELECT query on the database.

    Args:
        query: The SQL SELECT query to execute

    Returns:
        Query results as a list of dictionaries
    """
    # Only allow SELECT queries for safety
    if not query.strip().upper().startswith("SELECT"):
        return [{"error": "Only SELECT queries are allowed"}]

    try:
        conn = sqlite3.connect("example.db")
        cursor = conn.cursor()
        cursor.execute(query)

        # Get column names
        columns = [description[0] for description in cursor.description]

        # Get results
        rows = cursor.fetchall()

        # Convert to list of dictionaries
        results = []
        for row in rows:
            results.append({columns[i]: row[i] for i in range(len(columns))})

        conn.close()
        return results
    except Exception as e:
        return [{"error": f"Database error: {str(e)}"}]
```

### Web Scraping Tool

```python
from agents import function_tool
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any

@function_tool
def web_scraper_tool(url: str, element_selector: str = "body") -> str:
    """
    Scrape content from a web page.

    Args:
        url: The URL to scrape
        element_selector: CSS selector for the element to extract (default: body)

    Returns:
        Scraped content as a string
    """
    try:
        # Validate URL format
        if not url.startswith(("http://", "https://")):
            return "Error: Invalid URL format"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Agent/1.0)"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Find the specified element
        element = soup.select_one(element_selector)
        if element:
            # Extract text content, limit length
            content = element.get_text(strip=True)
            return content[:2000] + "..." if len(content) > 2000 else content
        else:
            return f"Element '{element_selector}' not found"
    except Exception as e:
        return f"Scraping error: {str(e)}"
```

### Time Series Analysis Tool

```python
from agents import function_tool
import pandas as pd
from typing import Dict, Any

@function_tool
def time_series_analyzer_tool(data_str: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """
    Analyze time series data.

    Args:
        data_str: Time series data in CSV format with date,value columns
        analysis_type: Type of analysis ('summary', 'trend', 'forecast')

    Returns:
        Analysis results as a dictionary
    """
    try:
        # Parse the CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(data_str))

        # Ensure required columns exist
        if 'date' not in df.columns or 'value' not in df.columns:
            return {"error": "Data must contain 'date' and 'value' columns"}

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        if analysis_type == "summary":
            return {
                "count": len(df),
                "date_range": f"{df['date'].min()} to {df['date'].max()}",
                "avg_value": float(df['value'].mean()),
                "min_value": float(df['value'].min()),
                "max_value": float(df['value'].max())
            }
        elif analysis_type == "trend":
            # Simple linear trend analysis
            df['date_ordinal'] = pd.to_datetime(df['date']).map(pd.Timestamp.toordinal)
            slope = (df['value'].corr(df['date_ordinal']) *
                    (df['value'].std() / df['date_ordinal'].std()))
            return {
                "trend_slope": float(slope),
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            }
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    except Exception as e:
        return {"error": f"Analysis error: {str(e)}"}
```

## Best Practices

### 1. Clear Documentation
Always provide clear docstrings explaining what the tool does, its parameters, and return values.

### 2. Error Handling
Implement proper error handling and return meaningful error messages.

### 3. Input Validation
Validate all inputs to prevent injection attacks and unexpected behavior.

### 4. Performance Considerations
Cache results when appropriate, and be mindful of execution time.

### 5. Security First
Never execute arbitrary code or allow unrestricted system access through tools.

### 6. Testing
Test tools thoroughly with various inputs, including edge cases and invalid inputs.

### 7. Logging
Include appropriate logging for debugging and monitoring purposes.

## Tool Composition

You can create higher-level skills by composing multiple tools:

```python
@function_tool
def research_skill(topic: str) -> str:
    """
    A skill that combines multiple tools for research.
    """
    # Use search tool to find information
    search_results = await search_tool(topic)

    # Use analysis tool to process results
    analysis = await analysis_tool(search_results)

    # Use summarization tool to create final output
    summary = await summarization_tool(analysis)

    return summary
```

This approach allows you to create sophisticated capabilities by combining simpler, focused tools.