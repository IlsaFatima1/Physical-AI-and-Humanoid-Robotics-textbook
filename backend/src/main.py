from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import os
from datetime import datetime
from rag_agent.agent import run_agent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="RAG Chat API",
    version="1.0.0"
)

# --------------------------------------------------
# CORS
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
    return {"status": "running"}

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
# CHAT ENDPOINT (FINAL FIX)
# --------------------------------------------------
@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        question = payload.question or payload.message

        if not question or not question.strip():
            return JSONResponse(
                status_code=400,
                content={"answer": "No question provided"}
            )

        start_time = time.time()
        logger.info(f"User question: {question}")

        result = await run_agent(question)

        response_time = round(time.time() - start_time, 2)

        return {
            "answer": str(result),
            "response_time": response_time,
            "timestamp": int(time.time())
        }

    except Exception as e:
        logger.exception("Chat endpoint error")
        return JSONResponse(
            status_code=500,
            content={
                "answer": "Internal server error",
                "error": str(e) if os.getenv("DEBUG") == "true" else None
            }
        )

# --------------------------------------------------
# QUERY ENDPOINT (OPTIONAL / SAME LOGIC)
# --------------------------------------------------
@app.post("/query")
async def query_endpoint(request: Request):
    body = await request.json()
    question = body.get("query")

    if not question:
        return {"answer": "No query provided"}

    result = await run_agent(question)
    return {
        "answer": str(result),
        "timestamp": datetime.now().isoformat()
    }

# --------------------------------------------------
# Run locally
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
