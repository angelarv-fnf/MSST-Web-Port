# MSST Web Port – Hugging Face Spaces / Docker
FROM python:3.10-slim

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# HF Spaces expects the app to listen on port 7860
ENV PORT=7860
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Create runtime dirs
RUN mkdir -p uploads outputs checkpoints

CMD ["python", "app.py"]
