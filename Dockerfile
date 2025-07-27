FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY requirements.txt setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create .env file with default settings
RUN echo "ENDPOINT=https://models.github.ai/inference\n\
MODEL_NAME=deepseek/DeepSeek-V3-0324\n\
TOP_K=3\n\
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2\n\
INDEX_PATH=/app/store/faiss_index\n\
REQUEST_TIMEOUT=30" > .env

# Copy application code
COPY src/ ./src
COPY data/ ./data

# Install the package
RUN pip install -e .

# Create directory for FAISS index and build it
RUN mkdir -p /app/store/faiss_index
RUN python -m src.index_documents --docs-path data/ --index-path /app/store/faiss_index

# Expose port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 