
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import os
from datetime import datetime

from rag_agent.agent import run_agent

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="RAG Chat API",
    version="1.0.0"
)

# --------------------------------------------------
# CORS (frontend ke liye zaroori)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Root
# --------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Backend is working"
    }

# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# --------------------------------------------------
# CHAT ENDPOINT (🔥 MAIN FIX)
# --------------------------------------------------
@app.post("/chat")
async def chat_endpoint(request: Request):
    print("DEBUG: Chat endpoint called")
    try:
        print("DEBUG: Reading request body...")
        body = await request.json()
        print(f"DEBUG: Request body received: {body}")

        message = body.get("message")
        print(f"DEBUG: Message extracted: {message}")

        if not message:
            print("DEBUG: No message provided")
            return JSONResponse(
                status_code=200,
                content={"answer": "No message provided"}
            )

        # Check environment variables before calling agent
        print("DEBUG: Checking environment variables...")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")
        qdrant_url = os.getenv("QDRANT_URL", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        print(f"DEBUG: GEMINI_API_KEY set: {bool(gemini_api_key)}")
        print(f"DEBUG: COHERE_API_KEY set: {bool(cohere_api_key)}")
        print(f"DEBUG: QDRANT_URL: {qdrant_url}")
        print(f"DEBUG: QDRANT_PORT: {qdrant_port}")

        print("DEBUG: Calling agent...")
        start = time.time()

        # Call agent directly
        answer = await run_agent(message)
        print(f"DEBUG: Agent response received: {answer[:100]}..." if len(str(answer)) > 100 else f"DEBUG: Agent response received: {answer}")

        response_time = time.time() - start
        print(f"DEBUG: Agent processing time: {response_time:.2f}s")

        response = {
            "answer": str(answer),
            "created": int(time.time())
        }
        print(f"DEBUG: Returning response")
        return response

    except Exception as e:
        print(f"DEBUG: Chat endpoint error: {e}")
        import traceback
        print(f"DEBUG: Error traceback: {traceback.format_exc()}")

        logger.exception("Chat error")
        return JSONResponse(
            status_code=200,   # frontend-safe
            content={
                "answer": "Sorry, something went wrong. Please try again.",
                "error": str(e) if os.getenv("DEBUG") == "true" else None
            }
        )

# --------------------------------------------------
# QUERY ENDPOINT (OPTIONAL)
# --------------------------------------------------
@app.post("/query")
async def query_endpoint(request: Request):
    try:
        body = await request.json()
        query = body.get("query")

        if not query:
            return JSONResponse(
                status_code=200,
                content={"answer": "No query provided"}
            )

        answer = await run_agent(query)

        return {
            "query": query,
            "answer": str(answer),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.exception("Query error")
        return JSONResponse(
            status_code=200,
            content={
                "answer": "Sorry, something went wrong.",
                "error": str(e) if os.getenv("DEBUG") == "true" else None
            }
        )

# --------------------------------------------------
# Run locally
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
