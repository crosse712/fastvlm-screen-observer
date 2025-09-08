# FastVLM Screen Observer - Comprehensive Guide

A production-ready screen monitoring and analysis system powered by vision-language models. This application captures screen content, analyzes it using state-of-the-art AI models, and provides detailed insights about UI elements, text content, and security risks.

## 🌟 Key Features

- **Browser-Based Screen Capture**: WebRTC `getDisplayMedia` API with comprehensive error handling
- **Multiple VLM Support**: Automatic fallback between FastVLM, LLaVA, and BLIP models
- **Real-Time Analysis**: Instant detection of UI elements, text, and potential risks
- **Production Ready**: Proper error handling, model verification, and status monitoring
- **Structured Logging**: NDJSON format with frame captures and detailed analysis
- **Modern Web Interface**: React + Vite with real-time updates via SSE
- **Export Functionality**: Download analysis data and captured frames as ZIP

## 🚀 Quick Start

```bash
# Clone and start everything with one command
git clone https://github.com/yourusername/fastvlm-screen-observer.git
cd fastvlm-screen-observer
./start.sh
```

Access the application at:
- Frontend: http://localhost:5174
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📖 Detailed Setup Instructions

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.10+ |
| Node.js | 16+ | 18+ |
| RAM | 4GB | 8GB+ |
| Storage | 2GB | 10GB+ |
| GPU | Optional | NVIDIA/Apple Silicon |

### Backend Installation

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # NVIDIA
# Apple Silicon MPS support is automatic

# Start backend
python app/main.py
```

### Frontend Installation

```bash
cd frontend

# Install dependencies
npm install

# Development mode
npm run dev

# Production build
npm run build
npm run preview
```

## 🤖 Model Configuration

### Current Status

The system currently loads **BLIP model** successfully on Apple Silicon (MPS):
- Model: Salesforce/blip-image-captioning-large
- Size: 470MB
- Parameters: 470M
- Device: MPS (Metal Performance Shaders)

### Available Models

| Model | Status | Size | Use Case |
|-------|--------|------|----------|
| **BLIP** | ✅ Working | 470MB | Image captioning, basic analysis |
| **LLaVA** | ⚠️ Config issue | 7GB | Detailed UI analysis |
| **FastVLM** | ❌ Tokenizer missing | 7GB | Advanced analysis |
| **Mock** | ✅ Fallback | 0MB | Development/testing |

### Loading Different Models

```python
# Via API
curl -X POST "http://localhost:8000/model/reload?model_type=blip"

# Check status
curl http://localhost:8000/model/status | python3 -m json.tool
```

## 🎮 Usage Guide

### Web Interface Features

1. **Screen Capture**
   - Click "Capture Screen" to start
   - Browser will prompt for permission
   - Select entire screen, window, or tab
   - Click "Take Screenshot" to capture

2. **Auto Capture Mode**
   - Enable checkbox for automatic capture
   - Set interval (minimum 1000ms)
   - Useful for monitoring changes

3. **Analysis Results**
   - Summary: AI-generated description
   - UI Elements: Detected buttons, links, forms
   - Text Snippets: Extracted text content
   - Risk Flags: Security/privacy concerns

4. **Export Data**
   - Downloads as ZIP file
   - Contains NDJSON logs
   - Includes captured thumbnails

### API Usage Examples

```python
import requests
import base64
from PIL import Image
import io

# 1. Check API and model status
response = requests.get("http://localhost:8000/")
status = response.json()
print(f"Model: {status['model']['model_name']}")
print(f"Device: {status['model']['device']}")

# 2. Capture and analyze screen
def analyze_screenshot(image_path):
    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    # Send to API
    response = requests.post(
        "http://localhost:8000/analyze",
        json={
            "image_data": f"data:image/png;base64,{image_base64}",
            "include_thumbnail": True
        }
    )
    
    return response.json()

# 3. Test model with synthetic image
response = requests.post("http://localhost:8000/model/test")
result = response.json()
print(f"Test result: {result['analysis_result']['summary']}")

# 4. Export logs
response = requests.get("http://localhost:8000/export")
with open("export.zip", "wb") as f:
    f.write(response.content)
```

## 📊 Sample Logs Generation

### Generate Test Logs

```bash
# Run test script to generate sample logs
cd /Users/kmh/fastvlm-screen-observer
python3 generate_sample_logs.py
```

### Sample NDJSON Format

```json
{"timestamp": "2025-09-04T10:30:00.123Z", "type": "frame_capture", "frame_id": "frame_1756947707", "has_thumbnail": true}
{"timestamp": "2025-09-04T10:30:00.456Z", "type": "analysis", "frame_id": "frame_1756947707", "summary": "a close up of a computer screen with code editor", "ui_elements": [{"type": "button", "text": "Save", "position": {"x": 100, "y": 50}}], "text_snippets": ["def main():", "return True"], "risk_flags": []}
{"timestamp": "2025-09-04T10:30:05.789Z", "type": "automation", "action": "click", "target": "button#submit", "success": true}
```

## 🎥 Demo Video Instructions

### Recording Setup

1. **Preparation**
```bash
# Clean environment
rm -rf logs/*.ndjson logs/frames/*
./start.sh
```

2. **Recording Tools**
   - **macOS**: QuickTime Player (Cmd+Shift+5) or OBS Studio
   - **Windows**: OBS Studio or Windows Game Bar (Win+G)
   - **Linux**: OBS Studio or SimpleScreenRecorder

3. **Demo Script** (2-3 minutes)

```
[0:00-0:15] Introduction
- Show terminal with ./start.sh
- Explain FastVLM Screen Observer purpose

[0:15-0:30] Interface Overview
- Navigate to http://localhost:5174
- Show control panel, analysis panel, logs

[0:30-1:00] Screen Capture Demo
- Click "Capture Screen"
- Show permission dialog
- Select screen to share
- Take screenshot
- Review AI analysis results

[1:00-1:30] Advanced Features
- Enable auto-capture (5s interval)
- Show multiple captures
- Point out UI element detection
- Highlight text extraction

[1:30-2:00] Model & Export
- Open http://localhost:8000/docs
- Show /model/status endpoint
- Export logs as ZIP
- Open and review contents

[2:00-2:30] Error Handling
- Deny permission to show error message
- Click "Try Again" to recover
- Show browser compatibility info
```

### Recording Tips
- Use 1920x1080 resolution
- Include audio narration
- Show actual screen content
- Demonstrate error recovery
- Keep under 3 minutes

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

| Issue | Error Message | Solution |
|-------|--------------|----------|
| Model won't load | `Tokenizer class Qwen2Tokenizer does not exist` | System auto-fallbacks to BLIP |
| Permission denied | `NotAllowedError: Permission denied` | Click "Allow" in browser prompt |
| Out of memory | `CUDA out of memory` | Use CPU or load smaller model |
| Port in use | `Port 5173 is already in use` | Kill process: `lsof -ti:5173 \| xargs kill -9` |
| API timeout | `Connection timeout` | Check backend is running |

### Debug Commands

```bash
# Check if services are running
curl http://localhost:8000/model/status
curl http://localhost:5174

# View backend logs
cd backend && tail -f logs/logs.ndjson

# Check Python dependencies
pip list | grep torch

# Monitor system resources
# macOS
top -o cpu
# Linux
htop
```

## 📁 Complete Project Structure

```
fastvlm-screen-observer/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI with model endpoints
│   │   └── __init__.py
│   ├── models/
│   │   ├── fastvlm_model.py     # Multi-model VLM wrapper
│   │   └── __init__.py
│   ├── utils/
│   │   ├── screen_capture.py    # MSS screen capture
│   │   ├── automation.py        # Selenium automation
│   │   ├── logger.py            # NDJSON logger
│   │   └── __init__.py
│   ├── requirements.txt         # Python dependencies
│   └── venv/                    # Virtual environment
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   ├── ScreenCapture.jsx    # WebRTC capture component
│   │   ├── App.css              # Main styles
│   │   ├── ScreenCapture.css    # Capture component styles
│   │   └── main.jsx             # Entry point
│   ├── public/                  # Static assets
│   ├── node_modules/            # Node dependencies
│   ├── package.json             # Node configuration
│   └── vite.config.js           # Vite configuration
├── logs/
│   ├── logs.ndjson              # Analysis logs
│   └── frames/                  # Captured thumbnails
│       └── *.png
├── docs/                        # Documentation
├── start.sh                     # Startup script
├── test_model_verification.py   # Model testing
├── test_api.py                  # API testing
├── generate_sample_logs.py      # Log generation
├── README.md                    # Basic readme
└── README_COMPREHENSIVE.md      # This file
```

## 🔒 Security Considerations

- **Screen Content**: May contain sensitive information
- **Permissions**: Always requires explicit user consent
- **Local Processing**: All ML inference runs locally
- **Data Storage**: Logs stored locally only
- **HTTPS**: Required for production WebRTC

## 📄 Complete API Reference

### Endpoints

```yaml
GET /:
  description: API status with model info
  response:
    status: string
    model: ModelStatus object

GET /model/status:
  description: Detailed model information
  response:
    is_loaded: boolean
    model_type: string
    model_name: string
    device: string
    parameters_count: number
    loading_time: float

POST /model/reload:
  parameters:
    model_type: string (auto|fastvlm|llava|blip|mock)
  response:
    success: boolean
    status: ModelStatus object

POST /model/test:
  description: Test model with synthetic image
  response:
    test_image_size: string
    analysis_result: AnalysisResult
    model_status: ModelStatus

POST /analyze:
  body:
    image_data: string (base64)
    include_thumbnail: boolean
    capture_screen: boolean
  response:
    summary: string
    ui_elements: array
    text_snippets: array
    risk_flags: array
    timestamp: string
    frame_id: string

GET /export:
  description: Export logs as ZIP
  response: Binary ZIP file

GET /logs/stream:
  description: Server-sent events stream
  response: SSE stream
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Salesforce for BLIP model (currently working)
- Apple for FastVLM concept
- Microsoft for LLaVA architecture
- HuggingFace for model hosting
- Open source community

---

**Current Status**: ✅ Fully functional with BLIP model on Apple Silicon MPS