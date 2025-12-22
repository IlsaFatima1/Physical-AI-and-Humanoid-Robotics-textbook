# OpenAI Agent SDK Guide

This document provides comprehensive information about creating and using agents with the OpenAI Agent SDK, including patterns, best practices, and examples.

## Table of Contents
- [Introduction](#introduction)
- [Agent Creation Patterns](#agent-creation-patterns)
- [Core Components](#core-components)
- [Configuration Options](#configuration-options)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Introduction

The OpenAI Agent SDK provides a framework for creating intelligent agents that can interact with various tools, maintain conversation context, and perform complex tasks. Agents are built around the concept of using Large Language Models (LLMs) with function calling capabilities.

## Agent Creation Patterns

### Basic Agent Structure

```python
from openai import OpenAI
from agents import Agent, Runner, AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel
import os

# Initialize the OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # or your specific API key
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),  # Custom base URL if needed
)

# Create model configuration
model = OpenAIChatCompletionsModel(
    model="gpt-4o",  # or "gpt-4", "gpt-3.5-turbo", etc.
    openai_client=client,
)

# Create run configuration
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True,  # Set to False for debugging
)

# Create the agent
agent = Agent(
    name="MyAgent",
    instructions="You are a helpful assistant...",
    tools=[tool1, tool2],  # Optional: list of tools the agent can use
)
```

### Async Agent Pattern

For async applications, use the async pattern:

```python
import asyncio
from agents import Agent, Runner

async def run_agent(query: str) -> str:
    response = await Runner.run(agent, query, run_config=config)
    return response.final_output

# Usage
async def main():
    result = await run_agent("Hello, what can you do?")
    print(result)
```

## Core Components

### 1. Client Configuration
The client handles communication with the LLM provider:

```python
client = AsyncOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.openai.com/v1",  # or custom endpoint
)
```

### 2. Model Configuration
The model configuration specifies which LLM to use:

```python
model = OpenAIChatCompletionsModel(
    model="gpt-4o",
    openai_client=client,
)
```

### 3. Run Configuration
The run configuration defines how the agent executes:

```python
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True,
    # Additional configuration options...
)
```

### 4. Agent Definition
The agent combines all components:

```python
agent = Agent(
    name="AgentName",  # Unique identifier for the agent
    instructions="System prompt for the agent",  # Behavior instructions
    tools=[tool1, tool2],  # Tools the agent can use
)
```

## Configuration Options

### RunConfig Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Model object | The LLM model to use |
| `model_provider` | Client | The API client |
| `tracing_disabled` | bool | Enable/disable tracing |
| `max_steps` | int | Maximum steps for the agent |
| `temperature` | float | Creativity parameter (0.0-2.0) |

### Agent Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Unique name for the agent |
| `instructions` | str | System prompt/behavior instructions |
| `tools` | list | List of tools the agent can use |
| `description` | str | Optional description |

## Best Practices

### 1. Environment Variables
Always use environment variables for sensitive information:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
```

### 2. Error Handling
Include proper error handling in your agent functions:

```python
async def run_agent(query: str) -> str:
    try:
        response = await Runner.run(agent, query, run_config=config)
        return response.final_output
    except Exception as e:
        print(f"Agent error: {e}")
        return "Sorry, I encountered an error processing your request."
```

### 3. Tool Functions
Create well-defined tool functions:

```python
from agents import function_tool

@function_tool
def search_tool(query: str) -> str:
    """
    Search for information based on the query.
    """
    # Implementation here
    return results
```

### 4. Resource Management
Properly manage resources and connections:

```python
# Initialize tools and agents once, not on every request
# Use connection pooling where appropriate
# Consider caching for expensive operations
```

## Examples

### Simple Q&A Agent

```python
from agents import Agent, Runner, AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel
import os

# Initialize client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create model configuration
model = OpenAIChatCompletionsModel(
    model="gpt-4o",
    openai_client=client,
)

# Create run configuration
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True,
)

# Create agent
simple_agent = Agent(
    name="SimpleQA",
    instructions="You are a helpful assistant that answers questions concisely.",
)

# Run the agent
async def ask_question(question: str) -> str:
    response = await Runner.run(simple_agent, question, run_config=config)
    return response.final_output
```

### Agent with Tools

```python
from agents import Agent, Runner, AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel
from agents import function_tool
import os

# Define a tool
@function_tool
def calculator_tool(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.
    """
    try:
        # Safe evaluation (in real applications, use more secure methods)
        result = eval(expression)
        return str(result)
    except Exception:
        return "Error: Invalid expression"

# Initialize client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create model configuration
model = OpenAIChatCompletionsModel(
    model="gpt-4o",
    openai_client=client,
)

# Create run configuration
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True,
)

# Create agent with tools
math_agent = Agent(
    name="MathAssistant",
    instructions="You are a math assistant. Use the calculator tool for complex calculations.",
    tools=[calculator_tool],
)
```

### Multi-Agent System

```python
# You can create multiple specialized agents
research_agent = Agent(
    name="Researcher",
    instructions="You are a research assistant that finds and summarizes information.",
    tools=[search_tool, web_scraping_tool]
)

writer_agent = Agent(
    name="Writer",
    instructions="You are a writing assistant that creates well-structured content.",
)

analyst_agent = Agent(
    name="Analyst",
    instructions="You are an analysis assistant that interprets data and provides insights.",
    tools=[data_analysis_tool]
)
```

## Advanced Topics

### Custom Models
You can use non-OpenAI models through providers like LiteLLM:

```python
# Using a different provider
client = AsyncOpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://api.anthropic.com/v1",  # Example
)
```

### Streaming Responses
For real-time applications, you can stream responses:

```python
async def stream_agent_response(query: str):
    async for chunk in Runner.stream(agent, query, run_config=config):
        yield chunk
```

### State Management
Agents can maintain state between conversations:

```python
# For stateful agents, you may need to implement custom state management
# depending on your specific SDK implementation
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API key is properly set in environment variables
2. **Model Not Found**: Verify the model name is correct and available
3. **Tool Not Working**: Check that tool functions are properly decorated with `@function_tool`
4. **Rate Limits**: Implement proper retry logic and rate limiting

### Debugging Tips

- Enable tracing temporarily to see detailed execution logs
- Check environment variables are properly loaded
- Verify network connectivity to the API endpoint
- Test tools individually before integrating with agents