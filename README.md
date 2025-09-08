---
title: FastVLM Screen Observer
emoji: 🖥️👁️
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "3.9"
app_port: 7860
pinned: false
license: mit
models:
  - apple/FastVLM-7B
suggested_hardware: t4-small
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
---

# FastVLM Screen Observer 🖥️👁️

Real-time screen observation and analysis using Apple's FastVLM-7B model, optimized for low-RAM systems (3-8GB).

## Features
- 🎯 Real-time screen capture and analysis
- 🤖 FastVLM-7B vision-language model integration
- 🔍 UI element detection
- 📝 Text extraction from screenshots
- ⚠️ Risk detection for security concerns
- 🎮 Browser automation demo
- 💾 Export logs and captured frames
- 🚀 Optimized for 3-8GB RAM with 4-bit quantization

## How to Use
1. Click "Capture Screen" to analyze your current screen
2. Enable "Auto Capture" for continuous monitoring
3. Use "Run Demo" to see browser automation
4. Export logs as ZIP archive

## Model Information
- **Model**: Apple FastVLM-7B
- **Optimization**: Extreme memory optimization with 4-bit quantization
- **Memory**: Runs on 3-8GB RAM systems
- **Device**: Supports CPU, CUDA, and MPS (Apple Silicon)

## API Endpoints
- `GET /api/` - Status check
- `POST /api/analyze` - Screen analysis
- `POST /api/demo` - Automation demo
- `GET /api/export` - Export logs
- `GET /api/logs/stream` - Stream logs via SSE

## GitHub Repository
https://github.com/crosse712/fastvlm-screen-observer

---
Built with ❤️ using FastAPI, React, and FastVLM-7B