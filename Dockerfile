FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN useradd -m appuser

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set correct permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port Streamlit runs on
EXPOSE 8502

# Command to run the application
CMD ["streamlit", "run", "gum-disease-analysis.py", "--server.port=8502", "--server.address=0.0.0.0"]
