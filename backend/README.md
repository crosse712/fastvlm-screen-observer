---
title: FastVLM Screen Observer
emoji: 🖥️
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: 3.9
app_file: app/main.py
pinned: false
models:
  - apple/FastVLM-7B
---

# FastVLM Screen Observer Backend

Real-time screen observation and analysis using Apple's FastVLM-7B model.

## Requirements
- 14GB+ RAM for model weights
- GPU (CUDA or MPS) recommended
- Python 3.9+

## API Endpoints
- `GET /` - Status check
- `POST /analyze` - Screen analysis
- `POST /demo` - Automation demo
- `GET /export` - Export logs
- `GET /logs/stream` - Stream logs via SSE