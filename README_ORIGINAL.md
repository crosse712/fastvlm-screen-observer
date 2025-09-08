# FastVLM-7B Screen Observer

A local web application for real-time screen observation and analysis using Apple's FastVLM-7B model via HuggingFace.

## Features

- **Real-time Screen Capture**: Capture and analyze screen content on-demand or automatically
- **FastVLM-7B Integration**: Uses Apple's vision-language model for intelligent screen analysis
- **UI Element Detection**: Identifies buttons, links, forms, and other interface elements
- **Text Extraction**: Captures text snippets from the screen
- **Risk Detection**: Flags potential security or privacy concerns
- **Automation Demo**: Demonstrates browser automation capabilities
- **NDJSON Logging**: Comprehensive logging in NDJSON format with timestamps
- **Export Functionality**: Download logs and captured frames as ZIP archive

## Specifications

- **Frontend**: React + Vite on `http://localhost:5173`
- **Backend**: FastAPI on `http://localhost:8000`
- **Model**: Apple FastVLM-7B with `trust_remote_code=True`
- **Image Token**: `IMAGE_TOKEN_INDEX = -200`
- **Output Format**: JSON with summary, ui_elements, text_snippets, risk_flags

## Prerequisites

- Python 3.8+
- Node.js 16+
- Chrome/Chromium browser (for automation demo)
- 14GB+ RAM (required for FastVLM-7B model weights)
- CUDA-capable GPU or Apple Silicon (recommended for FastVLM-7B)

## Installation

1. Clone this repository:
```bash
cd fastvlm-screen-observer
```

2. Install Python dependencies:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install Node.js dependencies:
```bash
cd ../frontend
npm install
```

## Running the Application

### Option 1: Using the start script (Recommended)
```bash
./start.sh
```

### Option 2: Manual start

Terminal 1 - Backend:
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

## Usage

1. Open your browser and navigate to `http://localhost:5173`
2. Click "Capture Screen" to analyze the current screen
3. Enable "Auto Capture" for continuous monitoring
4. Use "Run Demo" to see browser automation in action
5. Click "Export Logs" to download analysis data

## API Endpoints

- `GET /` - API status check
- `POST /analyze` - Capture and analyze screen
- `POST /demo` - Run automation demo
- `GET /export` - Export logs as ZIP
- `GET /logs/stream` - Stream logs via SSE
- `GET /docs` - Interactive API documentation

## Project Structure

```
fastvlm-screen-observer/
├── backend/
│   ├── app/
│   │   └── main.py              # FastAPI application
│   ├── models/
│   │   ├── fastvlm_model.py     # FastVLM-7B main integration
│   │   ├── fastvlm_optimized.py # Memory optimization strategies
│   │   ├── fastvlm_extreme.py   # Extreme optimization (4-bit)
│   │   └── use_fastvlm_small.py # Alternative 1.5B model
│   ├── utils/
│   │   ├── screen_capture.py    # Screen capture utilities
│   │   ├── automation.py        # Browser automation
│   │   └── logger.py            # NDJSON logging
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # React main component (with error handling)
│   │   ├── ScreenCapture.jsx    # WebRTC screen capture
│   │   └── App.css              # Styling
│   ├── package.json
│   └── vite.config.js
├── logs/                         # Generated logs and frames
├── start.sh                      # Startup script
└── README.md

```

## Model Notes

The application uses Apple's FastVLM-7B model with the following specifications:
- **Model ID**: `apple/FastVLM-7B` from HuggingFace
- **Tokenizer**: Qwen2Tokenizer (requires `transformers>=4.40.0`)
- **IMAGE_TOKEN_INDEX**: -200 (special token for image placeholders)
- **trust_remote_code**: True (required for model loading)

### Memory Requirements:
- **Minimum**: 14GB RAM for model weights
- **Recommended**: 16GB+ RAM for smooth operation
- The model will download automatically on first run (~14GB)

### Current Implementation:
The system includes multiple optimization strategies:
1. **Standard Mode**: Full precision (float16) - requires 14GB+ RAM
2. **Optimized Mode**: 8-bit quantization - requires 8-10GB RAM
3. **Extreme Mode**: 4-bit quantization with disk offloading - requires 6-8GB RAM

If the model fails to load due to memory constraints, the application will:
- Display a user-friendly error message
- Continue operating with graceful error handling
- NOT show "ANALYSIS_ERROR" in risk flags

## Acceptance Criteria

✅ Local web app running on localhost:5173  
✅ FastAPI backend on localhost:8000  
✅ FastVLM-7B integration with trust_remote_code=True  
✅ IMAGE_TOKEN_INDEX = -200 configured  
✅ JSON output format with required fields  
✅ Demo automation functionality  
✅ NDJSON logging with timestamps  
✅ ZIP export with logs and frames  
✅ Project structure matches specifications

## Troubleshooting

- **Model Loading Issues**: Check GPU memory and CUDA installation
- **Screen Capture Errors**: Ensure proper display permissions
- **Browser Automation**: Install Chrome/Chromium and check WebDriver
- **Port Conflicts**: Ensure ports 5173 and 8000 are available

## License

MIT