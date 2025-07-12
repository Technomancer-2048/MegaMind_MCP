FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    default-libmysqlclient-dev \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code and database initialization files
COPY megamind_database_server.py .
COPY init_database.py .
COPY init_schema.sql .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV MEGAMIND_MCP_SERVER_PORT=8002

# Expose port
EXPOSE 8002

# Disable health check for now
HEALTHCHECK NONE

# Run the server with initialization
CMD ["./entrypoint.sh"]