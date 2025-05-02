FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

# Create a non-root user to run the application
RUN useradd -m appuser

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Create a virtual environment using uv
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies using uv in the virtual environment
RUN uv pip install -r requirements.txt

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
