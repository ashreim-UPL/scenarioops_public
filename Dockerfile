FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if any needed for reportlab or others)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY pyproject.toml .
COPY README.md .
COPY src src
COPY config config
COPY data data
COPY docs docs
COPY prompts prompts

# Install the package itself
RUN pip install --no-cache-dir .

# Create storage directory for run artifacts
RUN mkdir -p storage/runs storage/logs storage/vectordb

# Set python path
ENV PYTHONPATH=/app/src

# Port defaults to 8080 for Cloud Run
ENV PORT=8080

# Run the application
# We use the factory pattern or direct instance. The api.py has 'app' instance.
CMD uvicorn scenarioops.app.api:app --host 0.0.0.0 --port ${PORT}
