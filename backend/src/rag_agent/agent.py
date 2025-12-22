import os
from agents import Agent, Runner, AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel
from .tools import book_retriever
from dotenv import load_dotenv

load_dotenv()

# Create OpenAI client configuration for Gemini API
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Create model configuration using OpenAIChatCompletionsModel
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client,
)

config = RunConfig(
    model=model,  # Use the model object instead of string
    model_provider=client,
    tracing_disabled=True  # Disable tracing to avoid API key warnings
)

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

# ✅ ASYNC ONLY — NO asyncio.run, NO threads
async def run_agent(query: str) -> str:
    print(f"DEBUG: run_agent called with query: {query}")
    try:
        # Test if the API key is properly loaded
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        print(f"DEBUG: GEMINI_API_KEY loaded: {bool(api_key)}")
        print(f"DEBUG: GEMINI_API_KEY value: {'Set' if api_key else 'Not set'}")

        if not api_key:
            print("DEBUG: GEMINI_API_KEY is not set!")
            return "Error: API key not configured"

        # Check if we can reach the Gemini API endpoint
        print("DEBUG: Attempting to connect to Gemini API...")
        import aiohttp
        import time
        start_time = time.time()

        try:
            # Test the API key by making a simple request
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash?key={api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    print(f"DEBUG: Gemini API test request - Status: {response.status}")
                    if response.status == 200:
                        print("DEBUG: Gemini API connection successful")
                    else:
                        print(f"DEBUG: Gemini API connection failed - Status: {response.status}")
        except Exception as conn_error:
            print(f"DEBUG: Gemini API connection test failed: {conn_error}")

        connection_time = time.time() - start_time
        print(f"DEBUG: API connection test took: {connection_time:.2f}s")

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
    except Exception as e:
        print(f"DEBUG: Agent error: {e}")
        import traceback
        print(f"DEBUG: Agent error traceback: {traceback.format_exc()}")
        # Return a more descriptive error
        return f"Agent error: {str(e)}"
