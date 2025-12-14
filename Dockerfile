# Legal Text Understandability Classifier
# Docker image for training, evaluation, inference, and serving

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and notebooks
COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh

# Create directories
RUN mkdir -p /app/data /app/models /app/log
RUN chmod +x /app/run.sh || true

# Expose port for API server
EXPOSE 8000

# Default: run the full pipeline
# Override with: docker run ... python src/api.py  (to start API server)
CMD ["bash", "/app/run.sh"]
