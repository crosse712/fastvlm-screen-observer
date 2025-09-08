from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
import time
import os
import sys
import io
import zipfile
from datetime import datetime
import base64
from pathlib import Path
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fastvlm_model import FastVLMModel
from utils.screen_capture import ScreenCapture
from utils.automation import BrowserAutomation
from utils.logger import NDJSONLogger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = FastVLMModel()
screen_capture = ScreenCapture()
automation = BrowserAutomation()
logger = NDJSONLogger()

class AnalysisRequest(BaseModel):
    capture_screen: bool = True
    include_thumbnail: bool = False
    image_data: Optional[str] = None  # Base64 encoded image from browser
    width: Optional[int] = None
    height: Optional[int] = None

class AnalysisResponse(BaseModel):
    summary: str
    ui_elements: List[Dict[str, Any]]
    text_snippets: List[str]
    risk_flags: List[str]
    timestamp: str
    frame_id: Optional[str] = None

class DemoRequest(BaseModel):
    url: str = "https://example.com"
    text_to_type: str = "test"

@app.on_event("startup")
async def startup_event():
    print("Loading FastVLM-7B model...")
    await model.initialize(model_type="fastvlm")  # Load FastVLM-7B with quantization
    status = model.get_status()
    if status["is_loaded"]:
        print(f"Model loaded successfully: {status['model_name']} on {status['device']}")
    else:
        print(f"Model loading failed: {status['error']}")
        print("Running in mock mode for development")

@app.get("/")
async def root():
    model_status = model.get_status()
    return {
        "status": "FastVLM Screen Observer API is running",
        "model": model_status
    }

@app.get("/model/status")
async def get_model_status():
    """Get detailed model status"""
    return model.get_status()

@app.post("/model/reload")
async def reload_model(model_type: str = "auto"):
    """Reload the model with specified type"""
    try:
        status = await model.reload_model(model_type)
        return {
            "success": status["is_loaded"],
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/test")
async def test_model():
    """Test model with a sample image"""
    try:
        # Create a test image
        test_image = PILImage.new('RGB', (640, 480), color='white')
        draw = ImageDraw.Draw(test_image)
        
        # Add some text and shapes to test
        draw.rectangle([50, 50, 200, 150], fill='blue', outline='black')
        draw.text((100, 100), "Test Button", fill='white')
        draw.rectangle([250, 50, 400, 150], fill='green', outline='black')
        draw.text((300, 100), "Submit", fill='white')
        draw.text((50, 200), "Sample text for testing", fill='black')
        draw.text((50, 250), "Another line of text", fill='black')
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Analyze the test image
        result = await model.analyze_image(img_byte_arr.getvalue())
        
        return {
            "test_image_size": "640x480",
            "analysis_result": result,
            "model_status": model.get_status()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_screen(request: AnalysisRequest):
    try:
        timestamp = datetime.now().isoformat()
        frame_id = f"frame_{int(time.time() * 1000)}"
        
        # Check if image data was provided from browser
        if request.image_data:
            # Process base64 image from browser
            try:
                # Remove data URL prefix if present
                if request.image_data.startswith('data:image'):
                    image_data = request.image_data.split(',')[1]
                else:
                    image_data = request.image_data
                
                # Decode base64 to bytes
                import base64 as b64
                screenshot = b64.b64decode(image_data)
                
                if request.include_thumbnail:
                    thumbnail = screen_capture.create_thumbnail(screenshot)
                    logger.log_frame(frame_id, thumbnail, timestamp)
                else:
                    logger.log_frame(frame_id, None, timestamp)
                
                analysis = await model.analyze_image(screenshot)
                
                # Include model info in response if available
                summary = analysis.get("summary", "Browser screen captured and analyzed")
                if analysis.get("mock_mode"):
                    summary = f"[MOCK MODE] {summary}"
                
                response = AnalysisResponse(
                    summary=summary,
                    ui_elements=analysis.get("ui_elements", []),
                    text_snippets=analysis.get("text_snippets", []),
                    risk_flags=analysis.get("risk_flags", []),
                    timestamp=timestamp,
                    frame_id=frame_id
                )
                
                logger.log_analysis(response.dict())
                return response
                
            except Exception as e:
                print(f"Error processing browser image: {e}")
                return AnalysisResponse(
                    summary=f"Error processing browser screenshot: {str(e)}",
                    ui_elements=[],
                    text_snippets=[],
                    risk_flags=['PROCESSING_ERROR'],
                    timestamp=timestamp
                )
        
        elif request.capture_screen:
            # Fallback to server-side capture
            screenshot = screen_capture.capture()
            
            if request.include_thumbnail:
                thumbnail = screen_capture.create_thumbnail(screenshot)
                logger.log_frame(frame_id, thumbnail, timestamp)
            else:
                logger.log_frame(frame_id, None, timestamp)
            
            analysis = await model.analyze_image(screenshot)
            
            response = AnalysisResponse(
                summary=analysis.get("summary", ""),
                ui_elements=analysis.get("ui_elements", []),
                text_snippets=analysis.get("text_snippets", []),
                risk_flags=analysis.get("risk_flags", []),
                timestamp=timestamp,
                frame_id=frame_id
            )
            
            logger.log_analysis(response.dict())
            return response
            
        else:
            return AnalysisResponse(
                summary="No screen captured",
                ui_elements=[],
                text_snippets=[],
                risk_flags=[],
                timestamp=timestamp
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/demo")
async def run_demo(request: DemoRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            automation.run_demo,
            request.url,
            request.text_to_type
        )
        
        return {
            "status": "Demo started",
            "url": request.url,
            "text": request.text_to_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export")
async def export_logs():
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            logs_path = Path("logs/logs.ndjson")
            if logs_path.exists():
                zipf.write(logs_path, "logs.ndjson")
            
            frames_dir = Path("logs/frames")
            if frames_dir.exists():
                for frame_file in frames_dir.glob("*.png"):
                    zipf.write(frame_file, f"frames/{frame_file.name}")
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=screen_observer_export_{int(time.time())}.zip"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/stream")
async def stream_logs():
    async def log_generator():
        last_position = 0
        log_file = Path("logs/logs.ndjson")
        
        while True:
            if log_file.exists():
                with open(log_file, "r") as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()
                    
                    for line in new_lines:
                        yield f"data: {line}\n\n"
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)