FROM --platform=linux/amd64 python:3.12-slim

# Add curl for health checks and system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory including all subdirectories
COPY . .

# Create necessary directories and ensure they exist
RUN mkdir -p /usr/src/app/ai-analyzer/prompt_template /usr/src/app/ai-analyzer/data

# Move prompt_template directory to the correct location if it exists at the root
RUN if [ -d "/usr/src/app/prompt_template" ]; then \
    mv /usr/src/app/prompt_template/* /usr/src/app/ai-analyzer/prompt_template/ && \
    rm -rf /usr/src/app/prompt_template; \
    fi

ENV PYTHONPATH=/usr/src/app/ai-analyzer

# Print directory structure for debugging
RUN ls -la /usr/src/app && \
    ls -la /usr/src/app/ai-analyzer && \
    ls -la /usr/src/app/ai-analyzer/prompt_template || true

# Command to run the FastAPI app
CMD ["uvicorn", "ai_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
