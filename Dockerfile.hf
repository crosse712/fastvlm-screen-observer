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
    supervisor \
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

# Configure nginx to serve frontend and proxy to backend
RUN echo 'server { \n\
    listen 7860; \n\
    root /usr/share/nginx/html; \n\
    index index.html; \n\
    \n\
    location / { \n\
        try_files $uri $uri/ /index.html; \n\
    } \n\
    \n\
    location /api/ { \n\
        proxy_pass http://127.0.0.1:8000/; \n\
        proxy_http_version 1.1; \n\
        proxy_set_header Upgrade $http_upgrade; \n\
        proxy_set_header Connection "upgrade"; \n\
        proxy_set_header Host $host; \n\
        proxy_set_header X-Real-IP $remote_addr; \n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \n\
        proxy_set_header X-Forwarded-Proto $scheme; \n\
        proxy_buffering off; \n\
    } \n\
}' > /etc/nginx/sites-available/default

# Create supervisor config
RUN echo '[supervisord] \n\
nodaemon=true \n\
\n\
[program:nginx] \n\
command=nginx -g "daemon off;" \n\
autostart=true \n\
autorestart=true \n\
stderr_logfile=/var/log/nginx.err.log \n\
stdout_logfile=/var/log/nginx.out.log \n\
\n\
[program:backend] \n\
command=python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 \n\
directory=/app \n\
autostart=true \n\
autorestart=true \n\
stderr_logfile=/var/log/backend.err.log \n\
stdout_logfile=/var/log/backend.out.log \n\
environment=USE_EXTREME_OPTIMIZATION="true",MAX_MEMORY_GB="3"' > /etc/supervisor/conf.d/supervisord.conf

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]