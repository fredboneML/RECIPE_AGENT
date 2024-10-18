# Use a base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the requirements.txt file if you are using it
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure uvicorn is available
RUN pip install uvicorn

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "ai_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]
