# Use a lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set work directory
WORKDIR /app

# Install system dependencies (for PyMuPDF and Torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data directories with permissions
RUN mkdir -p data/pdfs data/chroma_db && chmod -R 777 data

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the application
CMD uvicorn app.main:app --host 0.0.0.0 --port 7860
