FROM python:3.12-slim

# Add curl for health checks
RUN apt-get update && apt-get install -y curl


WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

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