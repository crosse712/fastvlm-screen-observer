# Hugging Face Spaces Dockerfile - Frontend + Backend
FROM node:18-slim as frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Python backend stage
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy frontend build from builder stage
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

# Create nginx configuration
RUN rm -f /etc/nginx/sites-enabled/default && \
    echo 'server { \
    listen 7860; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    \
    location /api/ { \
        proxy_pass http://127.0.0.1:8000/; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        proxy_buffering off; \
        proxy_read_timeout 86400; \
    } \
}' > /etc/nginx/sites-enabled/default

# Create startup script to run both services
RUN mkdir -p /var/log

# Create startup script to run both nginx and backend
RUN echo '#!/bin/bash \
set -e \
echo "Starting FastVLM Screen Observer..." \
echo "Starting nginx..." \
service nginx start \
echo "Starting backend API..." \
exec python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose port
EXPOSE 7860

# Environment variables
ENV USE_EXTREME_OPTIMIZATION=true
ENV MAX_MEMORY_GB=3
ENV PYTHONUNBUFFERED=1

# Start the application
CMD ["/app/start.sh"]