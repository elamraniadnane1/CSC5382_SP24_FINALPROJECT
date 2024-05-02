# Use an official Python runtime as a parent image
FROM python:3.9-slim as builder

# Set the working directory in the container
WORKDIR /usr/src/app

# Use a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in .venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Setup a lighter container
FROM python:3.9-slim

WORKDIR /usr/src/app

# Copy virtual env from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the scripts directory from your local machine to the container
COPY C:/Users/LENOVO/Desktop/CSC5356_SP24/scripts /usr/src/app/scripts

# Copy the remaining current directory contents into the container at /usr/src/app
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable for model path
ENV MODEL_PATH="/usr/src/app/models/bert-election2020-twitter-stance-biden"

# Healthcheck to ensure the service is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
