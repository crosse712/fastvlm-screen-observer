"""
Lightweight version of the API for deployment without FastVLM model.
This version provides mock responses for demo purposes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import json
import random
from typing import Optional

app = FastAPI(title="FastVLM Screen Observer API (Lightweight)")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    capture_screen: bool = True
    include_thumbnail: bool = False
    prompt: Optional[str] = None

@app.get("/")
async def root():
    return {
        "status": "FastVLM Screen Observer API is running (Lightweight Mode)",
        "model": {
            "is_loaded": False,
            "model_type": "mock",
            "model_name": "Mock Model (for demo)",
            "device": "cpu",
            "error": None,
            "note": "This is a lightweight version without the actual FastVLM model",
            "timestamp": datetime.now().isoformat()
        }
    }

@app.post("/analyze")
async def analyze_screen(request: AnalyzeRequest):
    """Mock analysis endpoint for demo purposes"""
    
    # Generate mock analysis result
    mock_result = {
        "timestamp": datetime.now().isoformat(),
        "summary": "Mock analysis: Screen captured successfully",
        "ui_elements": [
            {"type": "button", "text": "Submit", "location": "bottom-right"},
            {"type": "link", "text": "Home", "location": "top-left"},
            {"type": "input", "text": "Search...", "location": "top-center"}
        ],
        "text_snippets": [
            "Welcome to the application",
            "Click here to continue",
            f"Current time: {datetime.now().strftime('%H:%M:%S')}"
        ],
        "risk_flags": [],
        "frame_id": f"frame_{random.randint(1000, 9999)}",
        "processing_time": round(random.uniform(0.1, 0.5), 3),
        "model_used": "mock",
        "include_thumbnail": request.include_thumbnail
    }
    
    if request.include_thumbnail:
        mock_result["thumbnail"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    return JSONResponse(content=mock_result)

@app.post("/demo")
async def run_demo():
    """Mock demo endpoint"""
    return {
        "status": "success",
        "message": "Demo completed (mock mode)",
        "actions": [
            "Opened browser",
            "Navigated to example.com",
            "Captured screenshot",
            "Analyzed content"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/export")
async def export_logs():
    """Mock export endpoint"""
    return {
        "status": "success",
        "message": "Export feature available in full version",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/logs/stream")
async def stream_logs():
    """Mock SSE endpoint for logs"""
    def generate():
        for i in range(5):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": f"Mock log entry {i+1}",
                "type": "analysis"
            }
            yield f"data: {json.dumps(log_entry)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)