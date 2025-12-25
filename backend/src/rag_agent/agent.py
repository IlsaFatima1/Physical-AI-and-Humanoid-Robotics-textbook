import os
from agents import Agent, Runner, AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel
from .tools import book_retriever
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Global configuration to avoid repeated initialization
_agent_instance = None
_config_instance = None

def _initialize_agent():
    """Initialize the agent and configuration only once."""
    global _agent_instance, _config_instance

    if _agent_instance is not None and _config_instance is not None:
        return _agent_instance, _config_instance

    # Create OpenAI client configuration for Gemini API
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("WARNING: GEMINI_API_KEY not set. Agent will not work properly.")
        # Create a mock client for testing purposes
        client = None
    else:
        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize Gemini client: {e}")
            client = None

    # Create model configuration using OpenAIChatCompletionsModel
    if client:
        try:
            model = OpenAIChatCompletionsModel(
                model="gemini-2.5-flash",
                openai_client=client,
            )

            config = RunConfig(
                model=model,  # Use the model object instead of string
                model_provider=client,
                tracing_disabled=True  # Disable tracing to avoid API key warnings
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize model/config: {e}")
            config = None
    else:
        config = None

    agent = Agent(
        name="BookRAGAgent",
        instructions=(
            "You are a RAG assistant for a technical book. "
            "Use the retrieval tool to find relevant book content when answering. "
            "If a specific context is provided, focus on that context. "
            "If no specific context is provided, retrieve and use general book content to answer. "
            "If the answer is not found in the book, say you don't know."
        ),
        tools=[book_retriever],
    )

    _agent_instance = agent
    _config_instance = config

    return agent, config

# ✅ ASYNC ONLY — NO asyncio.run, NO threads
async def run_agent(query: str) -> str:
    print(f"DEBUG: run_agent called with query: {query}")

    # Check for required environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not gemini_api_key:
        print("DEBUG: GEMINI_API_KEY is not set!")
        return "Error: GEMINI_API_KEY not configured. Please set the environment variable."

    if not cohere_api_key:
        print("DEBUG: COHERE_API_KEY is not set!")
        return "Error: COHERE_API_KEY not configured. Please set the environment variable."

    try:
        # Initialize agent and config
        agent, config = _initialize_agent()

        if config is None:
            print("DEBUG: Configuration is not properly initialized")
            return "Error: Agent configuration failed. Check your API keys and network connection."

        print("DEBUG: Running agent with query...")
        # Run the agent with the query
        response = await Runner.run(agent, query, run_config=config)
        print(f"DEBUG: Agent response received")

        if hasattr(response, 'final_output'):
            print(f"DEBUG: Final output length: {len(response.final_output)} chars")
            return response.final_output
        else:
            print(f"DEBUG: Response type: {type(response)}, value: {str(response)[:100]}...")
            return str(response)
    except asyncio.TimeoutError:
        print("DEBUG: Agent call timed out")
        return "Error: Request timed out. Please try again later."
    except Exception as e:
        print(f"DEBUG: Agent error: {e}")
        import traceback
        print(f"DEBUG: Agent error traceback: {traceback.format_exc()}")

        # Check for common specific errors
        error_msg = str(e).lower()
        if "api" in error_msg or "key" in error_msg or "auth" in error_msg:
            return "Error: API authentication failed. Please check your API keys."
        elif "connection" in error_msg or "network" in error_msg:
            return "Error: Network connection failed. Please check your internet connection."
        elif "rate" in error_msg or "limit" in error_msg:
            return "Error: Rate limit exceeded. Please try again later."
        else:
            # Return a more user-friendly error message
            return f"Agent error occurred. Please try again. (Error: {str(e)[:100]}...)"
