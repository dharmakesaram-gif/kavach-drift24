# Stage 1: Build React Frontend
FROM node:20-slim as frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Python Backend
FROM python:3.12-slim as backend

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Copy built React files from the frontend stage
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

# Expose port for FastAPI (8000)
EXPOSE 8000

# Default command: launch FastAPI server on Railway's dynamic $PORT
CMD uvicorn src.integrations.api:app --host 0.0.0.0 --port ${PORT:-8000}
