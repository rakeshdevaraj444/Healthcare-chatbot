"""
FastAPI Main Application for Medical Assistant Chatbot
Now includes GUI Chatbot Interface (Frontend + Backend API)
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
from datetime import datetime

# Import the medical assistant logic
from assistant import (
    create_chat_session,
    get_medical_response,
    get_chat_history,
    end_chat_session,
    medical_assistant
)

# Initialize FastAPI app
app = FastAPI(
    title="MediCare AI - Medical Assistant API",
    description="AI-powered medical assistant for early disease prediction and health queries",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML templates setup
templates = Jinja2Templates(directory="templates")

# ----------------------------
# Pydantic Models
# ----------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message/query", min_length=1)
    session_id: Optional[str] = Field(None, description="Optional session ID for continuing conversation")


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_response: str
    is_emergency: bool
    timestamp: str
    message_count: int


class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str


class HistoryResponse(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    history: List[dict]


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    message: str


# ----------------------------
# ROUTES
# ----------------------------

# ✅ Web Interface (Frontend)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the chatbot HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})


# ✅ API Health Check
@app.get("/api", response_model=HealthCheck)
async def api_health():
    """API health status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "MediCare AI Medical Assistant API is running"
    }


# ✅ Chat Endpoint (Used by Frontend)
@app.post("/ask")
async def ask_question(data: ChatRequest):
    """Receive user question and return assistant reply"""
    session_id = data.session_id or str(uuid.uuid4())
    try:
        response = get_medical_response(session_id, data.message)
        if "error" in response:
            raise HTTPException(status_code=500, detail=response.get("message", "Internal server error"))
        return {"answer": response["assistant_response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Original Backend Endpoints (kept same)
@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Service is operational"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    try:
        response = get_medical_response(session_id, request.message)
        if "error" in response:
            raise HTTPException(status_code=500, detail=response.get("message", "Internal server error"))
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/create", response_model=SessionResponse)
async def create_session():
    session_id = str(uuid.uuid4())
    try:
        result = create_chat_session(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    try:
        history = get_chat_history(session_id)
        if "error" in history:
            raise HTTPException(status_code=404, detail="Session not found")
        return history
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    try:
        result = end_chat_session(session_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail="Session not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    return {
        "active_sessions": len(medical_assistant.sessions),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/admin/clear-sessions")
async def clear_all_sessions():
    result = medical_assistant.clear_all_sessions()
    return result


@app.get("/docs/examples")
async def get_examples():
    return {
        "examples": [
            {"title": "Create a new session", "method": "POST", "endpoint": "/session/create"},
            {"title": "Send a message", "method": "POST", "endpoint": "/chat"},
            {"title": "Get history", "method": "GET", "endpoint": "/session/{session_id}/history"},
            {"title": "End a session", "method": "DELETE", "endpoint": "/session/{session_id}"}
        ]
    }


# ----------------------------
# Run the server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port=8000)